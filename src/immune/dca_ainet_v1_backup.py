from dataclasses import dataclass, field
from typing import Dict
import math
import random

from src.immune.risk_signals import RiskSignals
from src.immune.immune_input import to_immune_input

RELATIONS = ("odor", "contamination", "physical")


def clip01(x):
    return max(0.0, min(1.0, float(x)))


@dataclass
class RelationDecision:
    danger: float
    safe: float
    mature: float
    repulsion: float


@dataclass
class ImmuneDecision:
    relations: Dict[str, RelationDecision]
    total_repulsion: float
    dominant_relation: str
    hard_forbidden: bool
    constraint_status: str
    policy_blocked: bool


@dataclass
class ImmuneMemory:
    affinities: Dict[str, float] = field(default_factory=dict)
    exposures: Dict[str, float] = field(default_factory=dict)


class DCAAiNet:
    def __init__(
        self,
        memory_gain=0.35,
        learning_rate=0.2,
        mutation_rate=0.02,
        suppression_threshold=0.95,
        policy_threshold=0.85,
        seed=42,
    ):
        self.memory_gain = float(memory_gain)
        self.learning_rate = float(learning_rate)
        self.mutation_rate = float(mutation_rate)
        self.suppression_threshold = float(
            suppression_threshold
        )
        self.policy_threshold = float(policy_threshold)
        self.memory = ImmuneMemory()
        self.rng = random.Random(seed)

    @staticmethod
    def pair_key(item_a, item_b, relation):
        a, b = sorted((str(item_a), str(item_b)))
        return f"{a}|{b}|{relation}"

    @staticmethod
    def expected_risk(probabilities):
        """Convert five ordinal class probabilities into [0,1]."""
        if len(probabilities) != 5:
            raise ValueError(
                "관계별 확률은 위험 등급 0~4의 5개 값이어야 합니다."
            )

        p = [float(x) for x in probabilities]

        if any(not math.isfinite(x) or x < 0 for x in p):
            raise ValueError("잘못된 예측 확률입니다.")

        total = sum(p)

        if total <= 0:
            raise ValueError("예측 확률의 합이 0입니다.")

        p = [x / total for x in p]

        return sum(
            i * probability
            for i, probability in enumerate(p)
        ) / 4.0

    def evaluate(self, item_a, item_b, signals: RiskSignals):
        data = to_immune_input(signals)
        relations = {}

        uncertainty = clip01(
            data["uncertainty"]
            if data["uncertainty"] is not None
            else 0.0
        )

        context = data["context_delta"]

        for relation in RELATIONS:
            if relation not in data["relation_probabilities"]:
                raise ValueError(
                    f"{relation} 관계의 GAT 예측이 없습니다."
                )

            if relation not in data["relation_embeddings"]:
                raise ValueError(
                    f"{relation} 관계의 GAT 임베딩이 없습니다."
                )

            predicted = self.expected_risk(
                data["relation_probabilities"][relation]
            )

            context_risk = clip01(
                context.get(relation, 0.0)
            )

            # DCA: danger and safe are separately computed.
            danger = clip01(
                0.75 * predicted
                + 0.15 * context_risk
                + 0.10 * uncertainty
            )

            safe = clip01(
                (1.0 - predicted)
                * (1.0 - context_risk)
                * (1.0 - 0.5 * uncertainty)
            )

            mature = clip01(
                danger / (danger + safe + 1e-9)
            )

            key = self.pair_key(
                item_a, item_b, relation
            )

            affinity = self.memory.affinities.get(
                key, 0.0
            )

            clone = clip01(
                mature + self.memory_gain * affinity
            )

            # Suppression of excessive clone activation
            if clone > self.suppression_threshold:
                clone = (
                    self.suppression_threshold
                    + 0.25 * (
                        clone - self.suppression_threshold
                    )
                )

            repulsion = clip01(
                0.65 * mature + 0.35 * clone
            )

            relations[relation] = RelationDecision(
                danger=danger,
                safe=safe,
                mature=mature,
                repulsion=repulsion,
            )

        dominant = max(
            RELATIONS,
            key=lambda r: relations[r].repulsion
        )

        total = max(
            relations[r].repulsion for r in RELATIONS
        )

        return ImmuneDecision(
            relations=relations,
            total_repulsion=total,
            dominant_relation=dominant,
            hard_forbidden=data["hard_forbidden"],
            constraint_status=data["constraint_status"],
            policy_blocked=(
                total >= self.policy_threshold
            ),
        )

    def feedback(
        self,
        item_a,
        item_b,
        decision: ImmuneDecision,
        observed_exposure: Dict[str, float],
    ):
        """Update immune memory after a PPO placement."""
        for relation in RELATIONS:
            exposure = clip01(
                observed_exposure.get(relation, 0.0)
            )

            key = self.pair_key(
                item_a, item_b, relation
            )

            old = self.memory.affinities.get(
                key, 0.0
            )

            antigen = (
                decision.relations[relation].mature
                * exposure
            )

            selected = max(old, antigen)

            mutation = self.rng.uniform(
                -self.mutation_rate,
                self.mutation_rate
            ) * exposure

            updated = clip01(
                old
                + self.learning_rate
                * (selected - old)
                + mutation
            )

            self.memory.affinities[key] = updated
            self.memory.exposures[key] = exposure

    def reset_memory(self):
        self.memory = ImmuneMemory()
