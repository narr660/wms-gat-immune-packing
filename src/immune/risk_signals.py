
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RiskSignals:
    # 기존 Oracle 기반 위험도: 학습 전에는 참조용
    relation_risks: Dict[str, int]
    final_risk: int

    # 독립적인 규칙 판정 결과
    hard_forbidden: bool
    constraint_status: str

    # 향후 GAT가 예측한 출력
    relation_probabilities: Dict[str, List[float]] = field(
        default_factory=dict
    )
    relation_embeddings: Dict[str, List[float]] = field(
        default_factory=dict
    )
    uncertainty: Optional[float] = None

    # 온라인 박스 상태 변화
    context_delta: Dict[str, float] = field(
        default_factory=dict
    )


def build_risk_signals(
    assessment: Dict,
    relation_probabilities=None,
    relation_embeddings=None,
    uncertainty=None,
    context_delta=None,
) -> RiskSignals:

    return RiskSignals(
        relation_risks=dict(assessment["relation_risks"]),
        final_risk=int(assessment["final_risk"]),
        hard_forbidden=bool(assessment["hard_forbidden"]),
        constraint_status=str(assessment["constraint_status"]),
        relation_probabilities=relation_probabilities or {},
        relation_embeddings=relation_embeddings or {},
        uncertainty=uncertainty,
        context_delta=context_delta or {},
    )
