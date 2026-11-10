import torch
from torch import nn

from src.models.m2_repulsion_gat import M2RepulsionGAT
from src.models.relation_repulsion import (
    RELATION_NAMES,
    RelationRepulsionOutput,
    apply_immune_feedback,
)


class DCASignalGenerator(nn.Module):
    """
    학습 가능한 DCA-inspired 위험/안전 신호 생성기.

    입력: 방향성 상품 쌍의 임베딩과 원본 특성
    출력:
        danger [E,1]
        safe   [E,1]
        immune [E,1] = danger - safe

    실제 DCA의 세포 모집단/성숙 과정은 아직 미구현.
    """

    def __init__(self, input_dim=26, hidden_dim=32):
        super().__init__()

        pair_dim = 2 * (input_dim + hidden_dim)

        self.shared = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
        )
        self.danger_head = nn.Linear(hidden_dim, 1)
        self.safe_head = nn.Linear(hidden_dim, 1)

    def forward(self, pair_features):
        hidden = self.shared(pair_features)

        danger = torch.sigmoid(
            self.danger_head(hidden)
        )
        safe = torch.sigmoid(
            self.safe_head(hidden)
        )
        immune = danger - safe

        return danger, safe, immune


class M3ImmuneFeedbackGAT(M2RepulsionGAT):
    """
    M3 = M2 + 관계별 DCA-inspired 면역 피드백.

    M2와 동일한:
        - 1차 GAT
        - 방향성 반발 헤드
        - Attention-반발 메시지 패싱
        - 위험 분류 헤드

    M3 추가:
        - 관계별 위험/안전 신호 생성
        - 관계별 면역 피드백 강도 beta
        - 조정된 반발 강도의 2차 메시지 패싱
    """

    def __init__(
        self,
        input_dim=26,
        hidden_dim=32,
        heads=4,
        initial_beta=0.5,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            heads=heads,
        )

        self.dca_modules = nn.ModuleDict({
            relation: DCASignalGenerator(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
            )
            for relation in RELATION_NAMES
        })

        self.feedback_beta = nn.ParameterDict({
            relation: nn.Parameter(
                torch.tensor(float(initial_beta))
            )
            for relation in RELATION_NAMES
        })

    def forward(self, x, edge_index):
        source, target = edge_index

        logits = {}
        repulsion = {}
        embeddings = {}

        # 부가 분석용 신호: 모델의 공통 출력과 별도로 보관
        immune_signals = {}
        base_repulsions = {}
        danger_signals = {}
        safe_signals = {}

        for relation in RELATION_NAMES:
            channel = self.channels[relation]

            # M2와 동일한 1차 인코더
            h1 = torch.relu(
                channel.gat1(x, edge_index)
            )

            pair_features = torch.cat(
                [
                    h1[source],
                    h1[target],
                    x[source],
                    x[target],
                ],
                dim=-1,
            )

            # M2의 기본 방향성 반발 강도
            base_rho = channel.repulsion_head(
                pair_features
            )

            # 관계별 DCA-inspired 신호
            danger, safe, immune = self.dca_modules[
                relation
            ](pair_features)

            # M3의 면역 피드백
            adjusted_rho = apply_immune_feedback(
                base_rho,
                immune,
                self.feedback_beta[relation],
            )

            # 면역 피드백을 실제 메시지 패싱에 반영
            h2 = torch.relu(
                channel.gat2(
                    h1,
                    edge_index,
                    adjusted_rho,
                )
            )

            edge_features = torch.cat(
                [
                    h2[source],
                    h2[target],
                    x[source],
                    x[target],
                ],
                dim=-1,
            )

            logits[relation] = self.edge_heads[
                relation
            ](edge_features)

            repulsion[relation] = adjusted_rho
            embeddings[relation] = h2

            immune_signals[relation] = immune
            base_repulsions[relation] = base_rho
            danger_signals[relation] = danger
            safe_signals[relation] = safe

        output = RelationRepulsionOutput(
            logits=logits,
            repulsion=repulsion,
            embeddings=embeddings,
        )
        output.validate()

        # 기존 공통 인터페이스를 유지하면서
        # M3 분석용 텐서를 추가로 반환
        self.last_immune_signals = immune_signals
        self.last_base_repulsions = base_repulsions
        self.last_danger_signals = danger_signals
        self.last_safe_signals = safe_signals

        return output
