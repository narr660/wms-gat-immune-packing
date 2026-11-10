from dataclasses import dataclass
from typing import Dict, List
import math

from src.immune.dca_ainet import DCAAiNet, RELATIONS
from src.immune.risk_signals import RiskSignals


@dataclass
class RelationCalibration:
    original_probabilities: List[float]
    calibrated_probabilities: List[float]
    original_grade: int
    calibrated_grade: int
    original_expected_risk: float
    calibrated_expected_risk: float
    danger: float
    safe: float
    mature: float


@dataclass
class CalibrationResult:
    relations: Dict[str, RelationCalibration]
    hard_forbidden: bool
    constraint_status: str


class DCARiskCalibrator:
    """
    Relation-specific DCA probability calibration.

    This is a testable calibration mechanism, not proof
    of improved classification accuracy.

    strength=0 reproduces original M3 probabilities.
    """

    def __init__(
        self,
        immune: DCAAiNet,
        strength=0.0,
    ):
        self.immune = immune
        self.strength = float(strength)

        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(
                "strength는 0~1 범위여야 합니다."
            )

    @staticmethod
    def normalize(probabilities):
        if len(probabilities) != 5:
            raise ValueError(
                "위험 등급 0~4의 확률 5개가 필요합니다."
            )

        values = [float(x) for x in probabilities]

        if any(
            not math.isfinite(x) or x < 0.0
            for x in values
        ):
            raise ValueError("잘못된 확률")

        total = sum(values)

        if total <= 0.0:
            raise ValueError("확률 합이 0입니다.")

        return [x / total for x in values]

    @staticmethod
    def expected_risk(probabilities):
        return sum(
            grade * probability
            for grade, probability
            in enumerate(probabilities)
        ) / 4.0

    @staticmethod
    def grade(probabilities):
        return max(
            range(5),
            key=lambda index: probabilities[index],
        )

    @staticmethod
    def tilted_distribution(probabilities, shift):
        """
        Exponential tilt of the original ordinal distribution.

        Positive shift favors higher grades.
        Negative shift favors lower grades.
        """
        logits = [
            math.log(max(p, 1e-12))
            + shift * (grade - 2)
            for grade, p in enumerate(probabilities)
        ]

        peak = max(logits)

        weights = [
            math.exp(x - peak) for x in logits
        ]

        total = sum(weights)

        return [x / total for x in weights]

    def calibrate(
        self,
        item_a,
        item_b,
        signals: RiskSignals,
    ):
        decision = self.immune.evaluate(
            item_a,
            item_b,
            signals,
        )

        results = {}

        for relation in RELATIONS:
            original = self.normalize(
                signals.relation_probabilities[relation]
            )

            dca = decision.relations[relation]

            # Independent DCA danger/safe evidence.
            # Zero strength is an identity transformation.
            shift = (
                self.strength
                * 2.0
                * (dca.danger - dca.safe)
            )

            calibrated = self.tilted_distribution(
                original, shift
            )

            results[relation] = RelationCalibration(
                original_probabilities=original,
                calibrated_probabilities=calibrated,
                original_grade=self.grade(original),
                calibrated_grade=self.grade(calibrated),
                original_expected_risk=(
                    self.expected_risk(original)
                ),
                calibrated_expected_risk=(
                    self.expected_risk(calibrated)
                ),
                danger=dca.danger,
                safe=dca.safe,
                mature=dca.mature,
            )

        return CalibrationResult(
            relations=results,
            hard_forbidden=decision.hard_forbidden,
            constraint_status=decision.constraint_status,
        )
