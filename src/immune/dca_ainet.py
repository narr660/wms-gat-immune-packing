from dataclasses import dataclass, field
from typing import Dict, Optional
import math

from src.immune.risk_signals import RiskSignals
from src.immune.immune_input import to_immune_input

RELATIONS = ("odor", "contamination", "physical")


def clip01(x):
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("비유한 실수 입력")
    return max(0.0, min(1.0, x))


@dataclass
class RelationDecision:
    danger: float
    safe: float
    mature: float
    repulsion: float
    embedding_signal: float


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
        suppression_threshold=0.95,
        policy_threshold=0.85,
        embedding_weights=None,
    ):
        self.memory_gain = float(memory_gain)
        self.learning_rate = float(learning_rate)
        self.suppression_threshold = float(
            suppression_threshold
        )
        self.policy_threshold = float(policy_threshold)
        self.embedding_weights = (
            embedding_weights or {}
        )
        self.memory = ImmuneMemory()

    @staticmethod
    def pair_key(item_a, item_b, relation):
        # Preserve A -> B direction.
        return f"{item_a}|{item_b}|{relation}"

    @staticmethod
    def expected_risk(probabilities):
        if len(probabilities) != 5:
            raise ValueError("위험 등급 확률은 5개여야 합니다.")

        p = [float(x) for x in probabilities]

        if any(
            not math.isfinite(x) or x < 0
            for x in p
        ):
            raise ValueError("잘못된 확률")

        total = sum(p)
        if total <= 0:
            raise ValueError("확률 합이 0")

        return sum(
            i * x / total for i, x in enumerate(p)
        ) / 4.0

    def embedding_signal(self, relation, embedding):
        """Relation-specific linear projection.

        Zero weights preserve the pretrained GAT baseline.
        Weights must be fitted on training data, never test labels.
        """
        vector = [float(x) for x in embedding]

        if not vector or any(
            not math.isfinite(x) for x in vector
        ):
            raise ValueError("잘못된 관계 임베딩")

        weights = self.embedding_weights.get(relation)

        if weights is None:
            return 0.0

        if len(weights) != len(vector):
            raise ValueError("임베딩 차원 불일치")

        return math.tanh(
            sum(float(w) * v for w, v in zip(weights, vector))
        )

    def evaluate(
        self,
        item_a,
        item_b,
        signals: RiskSignals,
    ):
        data = to_immune_input(signals)

        uncertainty = clip01(
            data["uncertainty"]
            if data["uncertainty"] is not None
            else 0.0
        )

        relations = {}

        for relation in RELATIONS:
            probs = data["relation_probabilities"][relation]
            embedding = data["relation_embeddings"][relation]

            predicted = self.expected_risk(probs)

            projected = self.embedding_signal(
                relation, embedding
            )

            # Projection is an additive calibration term.
            calibrated = clip01(
                predicted + 0.15 * projected
            )

            context_risk = clip01(
                data["context_delta"].get(relation, 0.0)
            )

            danger = clip01(
                0.75 * calibrated
                + 0.15 * context_risk
                + 0.10 * uncertainty
            )

            safe = clip01(
                (1.0 - calibrated)
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
                embedding_signal=projected,
            )

        dominant = max(
            RELATIONS,
            key=lambda r: relations[r].repulsion
        )

        total = max(
            x.repulsion for x in relations.values()
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

            # Deterministic clonal affinity update.
            updated = clip01(
                old + self.learning_rate
                * (selected - old)
            )

            self.memory.affinities[key] = updated
            self.memory.exposures[key] = exposure

    def reset_memory(self):
        self.memory = ImmuneMemory()
