
import torch
from torch import nn
from torch_geometric.nn import GATConv

from src.models.relation_repulsion import (
    RELATION_NAMES,
    RELATION_NUM_CLASSES,
    RelationRepulsionOutput,
)


class BasicGAT(nn.Module):
    """
    Baseline: Shared Graph Attention Network.

    - 하나의 공유 GAT 인코더
    - 관계별 독립 분류 헤드
    - 방향성 상품 쌍 분류
    - 관계별 반발력 및 면역 피드백 없음

    repulsion 출력은 공통 인터페이스 호환을 위한
    0 텐서이며 실제 반발력을 의미하지 않는다.
    """

    def __init__(
        self,
        input_dim=26,
        hidden_dim=32,
        heads=4,
    ):
        super().__init__()

        if hidden_dim % heads != 0:
            raise ValueError(
                "hidden_dim은 heads의 배수여야 합니다."
            )

        self.gat1 = GATConv(
            input_dim,
            hidden_dim // heads,
            heads=heads,
            concat=True,
            add_self_loops=False,
        )

        self.gat2 = GATConv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            concat=True,
            add_self_loops=False,
        )

        pair_dim = hidden_dim * 2 + input_dim * 2

        self.edge_heads = nn.ModuleDict({
            relation: nn.Sequential(
                nn.Linear(pair_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(
                    hidden_dim,
                    RELATION_NUM_CLASSES[relation],
                ),
            )
            for relation in RELATION_NAMES
        })

    def forward(self, x, edge_index):
        source, target = edge_index

        h1 = torch.relu(self.gat1(x, edge_index))
        h2 = torch.relu(self.gat2(h1, edge_index))

        pair_features = torch.cat(
            [
                h2[source],
                h2[target],
                x[source],
                x[target],
            ],
            dim=-1,
        )

        logits = {
            relation: self.edge_heads[relation](pair_features)
            for relation in RELATION_NAMES
        }

        edge_count = source.numel()

        # 공통 출력 구조 호환용 중립값
        # 학습 가능한 반발력은 아니다.
        repulsion = {
            relation: x.new_zeros((edge_count, 1))
            for relation in RELATION_NAMES
        }

        embeddings = {
            relation: h2
            for relation in RELATION_NAMES
        }

        output = RelationRepulsionOutput(
            logits=logits,
            repulsion=repulsion,
            embeddings=embeddings,
        )

        output.validate()
        return output
