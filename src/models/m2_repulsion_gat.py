import torch
from torch import nn
from torch_geometric.nn import GATConv

from src.models.relation_repulsion import (
    RELATION_NAMES,
    RELATION_NUM_CLASSES,
    RelationRepulsionOutput,
)



class AttentionRepulsionPropagation(nn.Module):
    """
    목적지별 softmax Attention과 방향성 반발 게이트를
    동일한 메시지 전달 단계에서 결합한다.

    attention: [E, heads]
    repulsion: [E, 1]
    output: [N, hidden_dim]
    """

    def __init__(self, hidden_dim, heads=4, lambda_repulsion=1.0):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim은 heads의 배수여야 합니다")
        if lambda_repulsion < 0:
            raise ValueError("lambda_repulsion은 음수가 될 수 없습니다")

        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.lambda_repulsion = lambda_repulsion

        self.projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.att_source = nn.Parameter(torch.empty(heads, self.head_dim))
        self.att_target = nn.Parameter(torch.empty(heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.att_source)
        nn.init.xavier_uniform_(self.att_target)
        nn.init.zeros_(self.bias)

    def forward(self, h, edge_index, repulsion):
        from torch_geometric.utils import softmax

        source, target = edge_index
        edge_count = source.numel()

        if repulsion.shape != (edge_count, 1):
            raise ValueError("repulsion은 [E,1]이어야 합니다")
        if torch.any(repulsion < 0) or torch.any(repulsion > 1):
            raise ValueError("repulsion은 [0,1] 범위여야 합니다")

        node_count = h.shape[0]
        projected = self.projection(h).reshape(
            node_count, self.heads, self.head_dim
        )

        source_h = projected[source]
        target_h = projected[target]

        scores = (
            (source_h * self.att_source).sum(dim=-1)
            + (target_h * self.att_target).sum(dim=-1)
        )
        scores = torch.nn.functional.leaky_relu(
            scores, negative_slope=0.2
        )

        # 각 목적지 노드의 수신 엣지끼리 Attention 정규화
        attention = softmax(
            scores, target, num_nodes=node_count
        )

        # 방향성 반발 강도는 Attention과 별도로 증폭
        gate = 1.0 + self.lambda_repulsion * repulsion
        messages = (
            attention.unsqueeze(-1)
            * gate.unsqueeze(-1)
            * source_h
        )

        output = projected.new_zeros(
            (node_count, self.heads, self.head_dim)
        )
        output.index_add_(0, target, messages)

        return output.reshape(node_count, -1) + self.bias



class DirectionalRepulsionChannel(nn.Module):
    """
    관계별 독립 인코더 + 방향성 반발 게이팅.

    GAT 1차: 기본 노드 임베딩
    Repulsion: 방향성 엣지 강도
    GAT 2차: 반발 강도 기반 메시지 전달
    Classifier: 방향성 위험 등급
    """

    def __init__(self, input_dim, hidden_dim, heads):
        super().__init__()

        if hidden_dim % heads != 0:
            raise ValueError(
                "hidden_dim은 heads의 배수여야 합니다"
            )

        self.gat1 = GATConv(
            input_dim,
            hidden_dim // heads,
            heads=heads,
            concat=True,
            add_self_loops=False,
        )

        self.repulsion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.gat2 = AttentionRepulsionPropagation(
            hidden_dim=hidden_dim,
            heads=heads,
            lambda_repulsion=1.0,
        )

    def forward(self, x, edge_index):
        source, target = edge_index

        h1 = torch.relu(
            self.gat1(x, edge_index)
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

        repulsion = self.repulsion_head(
            pair_features
        )

        # Attention과 방향성 반발 강도를 동일 단계에서 결합
        h2 = torch.relu(
            self.gat2(h1, edge_index, repulsion)
        )

        return h2, repulsion



class M2RepulsionGAT(nn.Module):
    """
    M2: 관계 분리형 방향성 반발 GAT.

    세 관계별 인코더/반발 헤드/분류 헤드 독립.
    """

    def __init__(
        self,
        input_dim=26,
        hidden_dim=32,
        heads=4,
    ):
        super().__init__()

        self.channels = nn.ModuleDict({
            relation: DirectionalRepulsionChannel(
                input_dim,
                hidden_dim,
                heads,
            )
            for relation in RELATION_NAMES
        })

        self.edge_heads = nn.ModuleDict({
            relation: nn.Sequential(
                nn.Linear(
                    hidden_dim * 2 + input_dim * 2,
                    hidden_dim,
                ),
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

        logits = {}
        repulsion = {}
        embeddings = {}

        for relation in RELATION_NAMES:
            h, rho = self.channels[relation](
                x,
                edge_index,
            )

            edge_features = torch.cat(
                [
                    h[source],
                    h[target],
                    x[source],
                    x[target],
                ],
                dim=-1,
            )

            logits[relation] = self.edge_heads[
                relation
            ](edge_features)

            repulsion[relation] = rho
            embeddings[relation] = h

        result = RelationRepulsionOutput(
            logits=logits,
            repulsion=repulsion,
            embeddings=embeddings,
        )

        result.validate()
        return result
