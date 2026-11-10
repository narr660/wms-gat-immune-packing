
import torch

from src.immune.ainet_v3 import RelationAiNetV3

RELATIONS = ("odor", "contamination", "physical")


class M4FeedbackManager:
    """
    aiNet: 상품 쌍별 면역 기억 학습
    direction_memory: 상품 쌍별 강화/완화 방향 저장

    관측값은 [0,1] 범위의 노출 강도로 해석한다.
    """

    def __init__(self, ainet=None, temperature=0.25):
        self.ainet = (
            ainet if ainet is not None
            else RelationAiNetV3(seed=42)
        )
        self.temperature = float(temperature)

        if self.temperature <= 0:
            raise ValueError("temperature는 양수여야 합니다.")

        self.direction_memory = {}
        self.observation_counts = {}

    @staticmethod
    def expected_risk(probabilities):
        """
        0~4 위험등급 확률을 [0,1] 기대 위험으로 변환.
        """
        p = torch.as_tensor(
            probabilities, dtype=torch.float32
        ).flatten()

        if p.numel() != 5:
            raise ValueError("위험 확률은 5개여야 합니다.")

        p = p / p.sum().clamp_min(1e-12)
        levels = torch.arange(5, dtype=p.dtype)
        return float((p * levels).sum().item() / 4.0)

    def observe(
        self, a, b, probabilities, exposure,
        dca_signals=None
    ):
        """
        probabilities:
          {relation: [p0,p1,p2,p3,p4]}

        exposure:
          {relation: observed_exposure}

        aiNet 기억 갱신과 방향 저장을 함께 수행.
        """
        dca_signals = dca_signals or {}

        risks = {}
        directions = {}

        for relation in RELATIONS:
            risks[relation] = self.expected_risk(
                probabilities[relation]
            )

            observed = float(exposure.get(relation, 0.0))
            observed = max(0.0, min(1.0, observed))

            error = observed - risks[relation]

            direction = max(
                -1.0,
                min(1.0, error / self.temperature)
            )

            key = self.ainet.pair_key(a, b, relation)

            self.direction_memory[key] = direction
            self.observation_counts[key] = (
                self.observation_counts.get(key, 0) + 1
            )
            directions[relation] = direction

        # 기존 aiNet V3의 실제 면역 기억 갱신
        self.ainet.feedback(
            a, b, risks, exposure, dca_signals
        )

        return {
            "risks": risks,
            "directions": directions,
        }

    def edge_confidence(
        self, items, edge_index, kappa=3.0,
        device=None, dtype=torch.float32
    ):
        if kappa <= 0:
            raise ValueError("kappa는 양수여야 합니다.")

        edges = edge_index.detach().cpu()
        E = edges.shape[1]
        result = {}

        for relation in RELATIONS:
            values = []

            for e in range(E):
                a = items[int(edges[0, e])]
                b = items[int(edges[1, e])]

                key = self.ainet.pair_key(a, b, relation)
                n = self.observation_counts.get(key, 0)

                values.append(n / (n + kappa))

            result[relation] = torch.tensor(
                values,
                device=device,
                dtype=dtype,
            ).reshape(E, 1)

        return result

    def edge_feedback(
        self, items, edge_index, device=None,
        dtype=torch.float32
    ):
        """
        상품 쌍별 aiNet 기억과 방향을
        그래프의 정확한 간선 위치로 변환.
        """
        edges = edge_index.detach().cpu()
        E = edges.shape[1]

        memory = {}
        direction = {}

        for relation in RELATIONS:
            affinities = []
            directions = []

            for e in range(E):
                a = items[int(edges[0, e])]
                b = items[int(edges[1, e])]

                key = self.ainet.pair_key(
                    a, b, relation
                )

                clones = self.ainet.memory.get(key, [])

                affinity = (
                    max(float(c.affinity) for c in clones)
                    if clones else 0.0
                )

                # aiNet affinity는 위험 기억으로 유지한다.
                # 보정 방향이 기록된 경우에는 관측 경험이 존재하므로
                # 최소 보정 강도를 확보한다.
                affinities.append(affinity)
                directions.append(
                    float(self.direction_memory.get(key, 0.0))
                )

            memory[relation] = torch.tensor(
                affinities, device=device, dtype=dtype
            ).reshape(E, 1)

            direction[relation] = torch.tensor(
                directions, device=device, dtype=dtype
            ).reshape(E, 1)

        return memory, direction
