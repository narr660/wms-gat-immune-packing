from dataclasses import dataclass
from typing import Dict, List, Optional
import math


RELATIONS = ("odor", "contamination", "physical")


@dataclass
class M3RelationOutput:
    probabilities: List[float]
    embedding: List[float]
    danger: Optional[float] = None
    safe: Optional[float] = None
    mature: Optional[float] = None


@dataclass
class M3PairOutput:
    relations: Dict[str, M3RelationOutput]
    hard_forbidden: bool = False
    constraint_status: str = "unknown"


class M3AiNetBridge:
    """
    M3 output interface.

    This class does not run M3, DCA, or aiNet.
    It only validates M3 probabilities and computes
    the expected ordinal risk for downstream aiNet.
    """

    @staticmethod
    def expected_risk(probabilities):
        if len(probabilities) != 5:
            raise ValueError(
                "M3 관계별 위험 확률은 0~4등급, 총 5개여야 합니다."
            )

        values = [float(x) for x in probabilities]

        if any(
            not math.isfinite(x) or x < 0
            for x in values
        ):
            raise ValueError("M3 확률값이 잘못되었습니다.")

        total = sum(values)

        if total <= 0:
            raise ValueError("M3 확률의 합이 0입니다.")

        return sum(
            grade * probability
            for grade, probability in enumerate(values)
        ) / (4.0 * total)
