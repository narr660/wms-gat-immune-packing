
import torch

from src.models.m3_immune_feedback_gat import M3ImmuneFeedbackGAT
from src.models.relation_repulsion import (
    RELATION_NAMES,
    RelationRepulsionOutput,
    apply_immune_feedback,
)


class M4BidirectionalImmuneGAT(M3ImmuneFeedbackGAT):
    """
    관계별·간선별 aiNet 기억을 GAT 내부 반발력에 반영한다.

    immune_memory:
      {
        "odor": tensor[E] 또는 tensor[E, 1],
        "contamination": tensor[E],
        "physical": tensor[E],
      }

    기존 관계별 스칼라 기억 입력도 호환한다.
    실제 적재 환경에서는 간선별 입력을 사용해야 한다.
    """

    def __init__(
        self,
        input_dim=26,
        hidden_dim=32,
        heads=4,
        initial_beta=0.5,
        memory_gain=0.2,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            initial_beta=initial_beta,
        )
        self.memory_gain = float(memory_gain)
        self.last_memory_adjustments = {}

    @staticmethod
    def edge_memory_from_ainet(
        ainet, items, edge_index, device=None, dtype=torch.float32
    ):
        """
        items[node_id]는 item_uid를 가진 실제 상품 dict.
        aiNet의 pair_key와 동일한 상품 식별자를 사용한다.
        """
        relations = tuple(RELATION_NAMES)
        edge_cpu = edge_index.detach().cpu()
        E = edge_cpu.shape[1]

        result = {}
        for relation in relations:
            values = []
            for e in range(E):
                src = int(edge_cpu[0, e])
                dst = int(edge_cpu[1, e])
                a, b = items[src], items[dst]

                key = ainet.pair_key(a, b, relation)
                clones = ainet.memory.get(key, [])

                # 현재 aiNet의 기억 affinity를 사용
                affinity = (
                    max(float(c.affinity) for c in clones)
                    if clones else 0.0
                )
                values.append(affinity)

            result[relation] = torch.tensor(
                values, device=device, dtype=dtype
            ).view(E, 1)

        return result

    def forward(self, x, edge_index, immune_memory=None, immune_direction=None, immune_confidence=None):
        immune_memory = immune_memory or {}
        immune_direction = immune_direction or {}
        immune_confidence = immune_confidence or {}
        source, target = edge_index
        E = edge_index.shape[1]

        logits, repulsion, embeddings = {}, {}, {}
        immune_signals = {}
        base_repulsions = {}
        danger_signals = {}
        safe_signals = {}
        memory_adjustments = {}

        for relation in RELATION_NAMES:
            channel = self.channels[relation]

            h1 = torch.relu(channel.gat1(x, edge_index))

            pair_features = torch.cat(
                [
                    h1[source],
                    h1[target],
                    x[source],
                    x[target],
                ],
                dim=-1,
            )

            base_rho = channel.repulsion_head(pair_features)

            danger, safe, immune = self.dca_modules[
                relation
            ](pair_features)

            dca_rho = apply_immune_feedback(
                base_rho,
                immune,
                self.feedback_beta[relation],
            )

            value = immune_memory.get(relation, 0.0)

            if isinstance(value, dict):
                value = value.get("memory_affinity", 0.0)

            memory = torch.as_tensor(
                value, device=x.device, dtype=dca_rho.dtype
            )

            if memory.ndim == 0:
                memory = memory.expand(E).reshape(E, 1)
            elif memory.shape == (E,):
                memory = memory.reshape(E, 1)
            elif memory.shape != (E, 1):
                raise ValueError(
                    f"{relation} memory shape 오류: "
                    f"{tuple(memory.shape)}, 기대값 {(E, 1)}"
                )

            memory = memory.clamp(0.0, 1.0)

            # 부호 있는 기억 보정:
            # memory_strength: 기억의 강도 [0,1]
            # memory_direction: 강화(+)/완화(-) 방향 [-1,1]
            direction_value = immune_direction.get(relation, 0.0)
            direction = torch.as_tensor(
                direction_value,
                device=x.device,
                dtype=dca_rho.dtype,
            )

            if direction.ndim == 0:
                direction = direction.expand(E).reshape(E, 1)
            elif direction.shape == (E,):
                direction = direction.reshape(E, 1)
            elif direction.shape != (E, 1):
                raise ValueError(
                    f"{relation} direction shape 오류: "
                    f"{tuple(direction.shape)}"
                )

            if not torch.isfinite(direction).all():
                raise ValueError("비정상적인 면역 보정 방향")

            direction = direction.clamp(-1.0, 1.0)

            # 강화: aiNet 위험 기억 M
            # 완화: 관측 횟수 기반 신뢰도 C
            confidence_value = immune_confidence.get(relation, 0.0)
            confidence = torch.as_tensor(
                confidence_value,
                device=x.device,
                dtype=dca_rho.dtype,
            )

            if confidence.ndim == 0:
                confidence = confidence.expand(E).reshape(E, 1)
            elif confidence.shape == (E,):
                confidence = confidence.reshape(E, 1)
            elif confidence.shape != (E, 1):
                raise ValueError(
                    f"{relation} confidence shape 오류: "
                    f"{tuple(confidence.shape)}"
                )

            confidence = confidence.clamp(0.0, 1.0)

            positive = direction.clamp(min=0.0)
            negative = (-direction).clamp(min=0.0)

            adjusted_rho = (
                dca_rho
                + self.memory_gain * (
                    memory * positive
                    - confidence * negative
                )
            ).clamp(0.0, 1.0)

            memory_adjustments[relation] = (
                adjusted_rho - dca_rho
            )

            h2 = torch.relu(
                channel.gat2(
                    h1, edge_index, adjusted_rho
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

        self.last_immune_signals = immune_signals
        self.last_base_repulsions = base_repulsions
        self.last_danger_signals = danger_signals
        self.last_safe_signals = safe_signals
        self.last_memory_adjustments = memory_adjustments

        return output

    def predict_probabilities(
        self, x, edge_index,
        immune_memory=None, immune_direction=None,
        immune_confidence=None
    ):
        output = self.forward(
            x, edge_index,
            immune_memory, immune_direction, immune_confidence
        )
        return {
            r: torch.softmax(output.logits[r], dim=-1)
            for r in RELATION_NAMES
        }
