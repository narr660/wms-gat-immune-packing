from dataclasses import dataclass
from typing import Dict
import math

from src.immune.dca_ainet import (
    DCAAiNet,
    RELATIONS,
    RelationDecision,
)
from src.immune.dca_risk_calibrator import (
    DCARiskCalibrator,
)
from src.immune.risk_signals import RiskSignals


def clip01(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("비유한 값은 허용되지 않습니다.")
    return max(0.0, min(1.0, value))


@dataclass
class PipelineRelation:
    original_probabilities: list
    calibrated_probabilities: list
    original_grade: int
    calibrated_grade: int
    calibrated_expected_risk: float
    danger: float
    safe: float
    mature: float
    repulsion: float


@dataclass
class PipelineDecision:
    relations: Dict[str, PipelineRelation]
    total_repulsion: float
    dominant_relation: str
    hard_forbidden: bool
    constraint_status: str
    policy_blocked: bool


class DCAAiNetPipeline:
    """M3 -> DCA calibration -> aiNet repulsion.

    DCA is executed once. aiNet uses calibrated probabilities
    and existing immune memory without invoking DCA again.
    """

    def __init__(
        self,
        immune: DCAAiNet,
        calibration_strength=0.0,
    ):
        self.immune = immune
        self.calibrator = DCARiskCalibrator(
            immune=immune,
            strength=calibration_strength,
        )

    def evaluate(
        self,
        item_a,
        item_b,
        signals: RiskSignals,
    ):
        calibrated = self.calibrator.calibrate(
            item_a,
            item_b,
            signals,
        )

        relations = {}

        for relation in RELATIONS:
            result = calibrated.relations[relation]

            # aiNet consumes the DCA-calibrated distribution.
            risk = result.calibrated_expected_risk

            key = self.immune.pair_key(
                item_a,
                item_b,
                relation,
            )

            affinity = self.immune.memory.affinities.get(
                key, 0.0
            )

            clone = clip01(
                risk
                + self.immune.memory_gain * affinity
            )

            if clone > self.immune.suppression_threshold:
                clone = (
                    self.immune.suppression_threshold
                    + 0.25 * (
                        clone - self.immune.suppression_threshold
                    )
                )

            repulsion = clip01(
                0.65 * risk + 0.35 * clone
            )

            relations[relation] = PipelineRelation(
                original_probabilities=(
                    result.original_probabilities
                ),
                calibrated_probabilities=(
                    result.calibrated_probabilities
                ),
                original_grade=result.original_grade,
                calibrated_grade=result.calibrated_grade,
                calibrated_expected_risk=risk,
                danger=result.danger,
                safe=result.safe,
                mature=result.mature,
                repulsion=repulsion,
            )

        dominant = max(
            RELATIONS,
            key=lambda r: relations[r].repulsion
        )

        total = max(
            relations[r].repulsion
            for r in RELATIONS
        )

        return PipelineDecision(
            relations=relations,
            total_repulsion=total,
            dominant_relation=dominant,
            hard_forbidden=calibrated.hard_forbidden,
            constraint_status=calibrated.constraint_status,
            policy_blocked=(
                total >= self.immune.policy_threshold
            ),
        )

    def feedback(
        self,
        item_a,
        item_b,
        decision: PipelineDecision,
        observed_exposure: Dict[str, float],
    ):
        """Update aiNet memory from the committed placement.

        Does not re-run DCA or update GAT weights.
        """
        for relation in RELATIONS:
            exposure = clip01(
                observed_exposure.get(relation, 0.0)
            )

            key = self.immune.pair_key(
                item_a,
                item_b,
                relation,
            )

            old = self.immune.memory.affinities.get(
                key, 0.0
            )

            antigen = (
                decision.relations[relation]
                .calibrated_expected_risk
                * exposure
            )

            selected = max(old, antigen)

            updated = clip01(
                old
                + self.immune.learning_rate
                * (selected - old)
            )

            self.immune.memory.affinities[key] = updated
            self.immune.memory.exposures[key] = exposure

    def gat_feedback(self, item_a, item_b):
        """Feedback features for the next GAT inference.

        Returning features does not itself update GAT.
        """
        return {
            relation: {
                "affinity": self.immune.memory.affinities.get(
                    self.immune.pair_key(
                        item_a, item_b, relation
                    ),
                    0.0,
                ),
                "exposure": self.immune.memory.exposures.get(
                    self.immune.pair_key(
                        item_a, item_b, relation
                    ),
                    0.0,
                ),
            }
            for relation in RELATIONS
        }
