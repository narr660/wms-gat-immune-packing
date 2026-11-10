from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

from src.immune.dca_ainet import DCAAiNet, RELATIONS
from src.immune.risk_signals import RiskSignals


def clip01(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("유한한 실수가 필요합니다.")
    return max(0.0, min(1.0, value))


@dataclass
class PlacementCandidate:
    """PCT가 제안한 3D 배치 후보를 표현하는 중간 형식."""
    candidate_id: str
    box_id: str
    position: Tuple[float, float, float]
    rotation_size: Tuple[float, float, float]
    pct_score: float = 0.0


@dataclass
class ExistingItem:
    item_uid: str
    box_id: str
    position: Tuple[float, float, float]
    rotation_size: Tuple[float, float, float]


@dataclass
class CandidateEvaluation:
    candidate: PlacementCandidate
    feasible: bool
    hard_forbidden: bool
    policy_blocked: bool
    repulsion: float
    relation_repulsion: Dict[str, float]
    adjusted_score: float
    pair_decisions: Dict[str, object] = field(
        default_factory=dict
    )


class ImmunePCTAdapter:
    """Bridge between a PCT candidate generator and DCA–aiNet.

    This does not replace the original PCT policy.
    It evaluates candidate placements supplied by that policy.
    """

    def __init__(
        self,
        immune: DCAAiNet,
        risk_provider,
        risk_penalty=1.0,
        block_high_risk=False,
    ):
        self.immune = immune
        self.risk_provider = risk_provider
        self.risk_penalty = float(risk_penalty)
        self.block_high_risk = bool(block_high_risk)

    @staticmethod
    def _validate_geometry(candidate):
        values = (
            list(candidate.position)
            + list(candidate.rotation_size)
        )
        if len(candidate.position) != 3:
            raise ValueError("position은 3차원이어야 합니다.")
        if len(candidate.rotation_size) != 3:
            raise ValueError("rotation_size는 3차원이어야 합니다.")
        if any(
            not math.isfinite(float(v)) for v in values
        ):
            raise ValueError("잘못된 3D 좌표 또는 크기")
        if any(
            float(v) <= 0
            for v in candidate.rotation_size
        ):
            raise ValueError("박스 크기는 양수여야 합니다.")

    def evaluate_candidate(
        self,
        incoming_uid,
        candidate: PlacementCandidate,
        existing_items: List[ExistingItem],
    ):
        self._validate_geometry(candidate)

        same_box = [
            item for item in existing_items
            if item.box_id == candidate.box_id
        ]

        relation_repulsion = {
            relation: 0.0 for relation in RELATIONS
        }
        pair_decisions = {}
        hard_forbidden = False
        policy_blocked = False

        for existing in same_box:
            signals = self.risk_provider(
                incoming_uid,
                existing.item_uid,
                candidate,
                existing,
            )

            if not isinstance(signals, RiskSignals):
                raise TypeError(
                    "risk_provider는 RiskSignals를 반환해야 합니다."
                )

            decision = self.immune.evaluate(
                incoming_uid,
                existing.item_uid,
                signals,
            )

            pair_decisions[existing.item_uid] = decision

            hard_forbidden = (
                hard_forbidden
                or decision.hard_forbidden
            )

            policy_blocked = (
                policy_blocked
                or decision.policy_blocked
            )

            for relation in RELATIONS:
                relation_repulsion[relation] = max(
                    relation_repulsion[relation],
                    decision.relations[relation].repulsion,
                )

        repulsion = max(relation_repulsion.values())

        feasible = (
            not hard_forbidden
            and not (
                self.block_high_risk
                and policy_blocked
            )
        )

        adjusted_score = (
            float(candidate.pct_score)
            - self.risk_penalty * repulsion
        )

        return CandidateEvaluation(
            candidate=candidate,
            feasible=feasible,
            hard_forbidden=hard_forbidden,
            policy_blocked=policy_blocked,
            repulsion=repulsion,
            relation_repulsion=relation_repulsion,
            adjusted_score=adjusted_score,
            pair_decisions=pair_decisions,
        )

    def rank_candidates(
        self,
        incoming_uid,
        candidates,
        existing_items,
    ):
        results = [
            self.evaluate_candidate(
                incoming_uid,
                candidate,
                existing_items,
            )
            for candidate in candidates
        ]

        return sorted(
            results,
            key=lambda result: (
                not result.feasible,
                -result.adjusted_score,
                result.candidate.candidate_id,
            ),
        )

    def commit_placement(
        self,
        incoming_uid,
        evaluation: CandidateEvaluation,
        observed_exposures: Dict[str, Dict[str, float]],
    ):
        """Call only after a placement is actually accepted.

        observed_exposures:
            {existing_item_uid: {relation: exposure}}
        """
        if not evaluation.feasible:
            raise ValueError(
                "차단된 후보는 배치 확정할 수 없습니다."
            )

        for existing_uid, decision in (
            evaluation.pair_decisions.items()
        ):
            if existing_uid not in observed_exposures:
                raise ValueError(
                    f"{existing_uid}의 노출 관측값이 없습니다."
                )

            self.immune.feedback(
                incoming_uid,
                existing_uid,
                decision,
                observed_exposures[existing_uid],
            )

        return {
            "item_uid": incoming_uid,
            "box_id": evaluation.candidate.box_id,
            "position": evaluation.candidate.position,
            "rotation_size": (
                evaluation.candidate.rotation_size
            ),
            "relation_repulsion": dict(
                evaluation.relation_repulsion
            ),
            "total_repulsion": evaluation.repulsion,
            "feedback_pairs": len(
                evaluation.pair_decisions
            ),
        }

    def gat_feedback(self, incoming_uid, existing_uid):
        """Return pairwise feedback for a future GAT context.

        This is a feedback signal, not a trained GAT update.
        """
        return {
            relation: {
                "affinity": self.immune.memory.affinities.get(
                    self.immune.pair_key(
                        incoming_uid,
                        existing_uid,
                        relation,
                    ),
                    0.0,
                ),
                "exposure": self.immune.memory.exposures.get(
                    self.immune.pair_key(
                        incoming_uid,
                        existing_uid,
                        relation,
                    ),
                    0.0,
                ),
            }
            for relation in RELATIONS
        }
