
from typing import Dict

from src.immune.risk_signals import RiskSignals


def to_immune_input(signals: RiskSignals) -> Dict:
    """
    DCA/aiNet 추론 입력 생성.

    Oracle 정답(relation_risks, final_risk)은
    절대 전달하지 않는다.
    """
    if not signals.relation_probabilities:
        raise ValueError("GAT 예측 확률이 없습니다.")

    if not signals.relation_embeddings:
        raise ValueError("GAT 관계 임베딩이 없습니다.")

    if signals.constraint_status not in (
        "forbidden", "unknown", "no_prohibition_found"
    ):
        raise ValueError("알 수 없는 제약 판정 상태입니다.")

    return {
        "relation_probabilities": {
            key: list(value)
            for key, value in signals.relation_probabilities.items()
        },
        "relation_embeddings": {
            key: list(value)
            for key, value in signals.relation_embeddings.items()
        },
        "uncertainty": signals.uncertainty,
        "context_delta": dict(signals.context_delta),
        "hard_forbidden": signals.hard_forbidden,
        "constraint_status": signals.constraint_status,
    }
