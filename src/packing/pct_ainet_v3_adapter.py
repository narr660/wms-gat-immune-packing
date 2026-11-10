from dataclasses import dataclass
from typing import Callable, List, Dict, Any
import numpy as np


@dataclass
class PackedItem:
    item_uid: str
    position: tuple
    size: tuple


@dataclass
class LeafEvaluation:
    leaf_index: int
    valid: bool
    hard_forbidden: bool
    policy_blocked: bool
    risk_penalty: float
    pair_decisions: list


class PCTAiNetV3Adapter:
    """
    External PCT leaf-node safety evaluator.

    Does not modify the original PCT policy or environment.
    """

    def __init__(
        self,
        bridge,
        risk_provider: Callable,
        risk_penalty_weight=1.0,
        block_policy_risk=False,
    ):
        self.bridge = bridge
        self.risk_provider = risk_provider
        self.risk_penalty_weight = float(
            risk_penalty_weight
        )
        self.block_policy_risk = bool(block_policy_risk)

    @staticmethod
    def decode_leaf(leaf, incoming_size):
        leaf = np.asarray(leaf, dtype=float)

        if leaf.shape != (9,):
            raise ValueError("PCT leaf node는 길이 9여야 합니다.")

        if np.sum(leaf[:6]) == 0 or leaf[8] <= 0:
            return None

        x0, y0, z0, x1, y1, _ = leaf[:6]
        sx, sy = x1 - x0, y1 - y0

        remaining = list(map(float, incoming_size))

        def pop_match(value):
            for i, candidate in enumerate(remaining):
                if abs(candidate - value) < 1e-6:
                    return remaining.pop(i)
            raise ValueError("leaf 회전 크기가 상품 크기와 불일치")

        pop_match(sx)
        pop_match(sy)

        return {
            "position": (float(x0), float(y0), float(z0)),
            "size": (float(sx), float(sy), remaining[0]),
        }

    def evaluate_leaf(
        self,
        leaf_index,
        leaf,
        incoming_uid,
        incoming_size,
        packed_items: List[PackedItem],
    ):
        decoded = self.decode_leaf(leaf, incoming_size)

        if decoded is None:
            return LeafEvaluation(
                leaf_index, False, False, False, 0.0, []
            )

        pair_decisions = []
        hard_forbidden = False
        policy_blocked = False
        pair_risks = []

        for packed in packed_items:
            m3_output = self.risk_provider(
                incoming_uid,
                packed.item_uid,
                decoded,
                packed,
            )

            decision = self.bridge.evaluate(
                incoming_uid,
                packed.item_uid,
                m3_output,
            )

            pair_decisions.append(
                (packed.item_uid, decision)
            )

            hard_forbidden |= decision.hard_forbidden
            policy_blocked |= decision.policy_blocked
            pair_risks.append(decision.total_repulsion)

        risk_penalty = (
            max(pair_risks, default=0.0)
            * self.risk_penalty_weight
        )

        valid = (
            not hard_forbidden
            and (
                not self.block_policy_risk
                or not policy_blocked
            )
        )

        return LeafEvaluation(
            leaf_index=leaf_index,
            valid=valid,
            hard_forbidden=hard_forbidden,
            policy_blocked=policy_blocked,
            risk_penalty=risk_penalty,
            pair_decisions=pair_decisions,
        )

    def commit_success(
        self,
        incoming_uid,
        evaluation: LeafEvaluation,
        observed_exposure: Dict[str, Dict[str, float]],
    ):
        """
        Call only after PCT environment confirms placement.
        observed_exposure is keyed by existing item_uid.
        """
        if not evaluation.valid:
            raise ValueError(
                "금지된 후보의 면역 기억은 갱신할 수 없습니다."
            )

        for packed_uid, decision in evaluation.pair_decisions:
            exposure = observed_exposure.get(
                packed_uid, {}
            )

            self.bridge.feedback(
                incoming_uid,
                packed_uid,
                decision,
                exposure,
            )
