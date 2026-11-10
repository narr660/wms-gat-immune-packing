from dataclasses import dataclass
from typing import Dict

import torch


RELATION_NAMES = (
    "odor",
    "contamination",
    "physical",
)

RELATION_NUM_CLASSES = {
    "odor": 5,
    "contamination": 5,
    "physical": 5,
}


@dataclass
class RelationRepulsionOutput:
    """
    관계별 방향성 위험 예측 출력.

    logits[r]      : [E, num_classes]
    repulsion[r]   : [E, 1], 범위 [0, 1]
    embeddings[r]  : [N, hidden_dim]
    """

    logits: Dict[str, torch.Tensor]
    repulsion: Dict[str, torch.Tensor]
    embeddings: Dict[str, torch.Tensor]

    def validate(self):
        expected = set(RELATION_NAMES)

        for name, values in (
            ("logits", self.logits),
            ("repulsion", self.repulsion),
            ("embeddings", self.embeddings),
        ):
            if set(values) != expected:
                raise ValueError(
                    f"{name} 관계 구성이 다릅니다"
                )

        edge_count = None

        for relation in RELATION_NAMES:
            logits = self.logits[relation]
            repulsion = self.repulsion[relation]
            embeddings = self.embeddings[relation]

            if logits.ndim != 2:
                raise ValueError("logits는 [E,C]이어야 합니다")

            if logits.shape[1] != RELATION_NUM_CLASSES[relation]:
                raise ValueError("logits 클래스 수 불일치")

            if repulsion.ndim != 2 or repulsion.shape[1] != 1:
                raise ValueError("repulsion은 [E,1]이어야 합니다")

            if embeddings.ndim != 2:
                raise ValueError("embeddings는 [N,H]이어야 합니다")

            if logits.shape[0] != repulsion.shape[0]:
                raise ValueError("logits와 repulsion 엣지 수 불일치")

            for tensor in (logits, repulsion, embeddings):
                if not torch.isfinite(tensor).all():
                    raise ValueError("비유한값이 있습니다")

            if ((repulsion < 0) | (repulsion > 1)).any():
                raise ValueError("repulsion 범위가 [0,1]을 벗어났습니다")

            if edge_count is None:
                edge_count = logits.shape[0]
            elif edge_count != logits.shape[0]:
                raise ValueError("관계별 엣지 수가 다릅니다")

        return True


def apply_immune_feedback(
    repulsion,
    immune_signal,
    beta,
    eps=1e-6,
):
    """
    방향성 반발 강도에 면역 신호를 반영한다.

    repulsion: [E,1]
    immune_signal: [E,1]
    beta: 스칼라 또는 학습 가능한 파라미터

    beta=0이면 원래 repulsion을 그대로 반환한다.
    """

    if repulsion.shape != immune_signal.shape:
        raise ValueError("repulsion과 immune_signal 크기 불일치")

    if not torch.isfinite(repulsion).all():
        raise ValueError("repulsion에 비유한값이 있습니다")

    if not torch.isfinite(immune_signal).all():
        raise ValueError("immune_signal에 비유한값이 있습니다")

    if ((repulsion < 0) | (repulsion > 1)).any():
        raise ValueError("repulsion은 [0,1] 범위여야 합니다")

    safe_repulsion = repulsion.clamp(eps, 1 - eps)

    adjusted = torch.sigmoid(
        torch.logit(safe_repulsion)
        + beta * immune_signal
    )

    if not isinstance(beta, torch.Tensor):
        if beta == 0:
            return repulsion
    elif beta.numel() == 1 and bool((beta.detach() == 0).item()):
        return repulsion

    return adjusted
