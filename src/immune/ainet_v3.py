from dataclasses import dataclass
from typing import Dict, List
import random
import math

RELATIONS = ("odor", "contamination", "physical")

def clip01(x):
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("유한한 숫자가 필요합니다.")
    return max(0.0, min(1.0, x))

@dataclass
class Clone:
    affinity: float
    exposure: float

@dataclass
class Decision:
    relation_repulsion: Dict[str, float]
    total_repulsion: float
    dominant_relation: str

class RelationAiNetV3:
    def __init__(self, seed=42, clone_count=5,
                 mutation_rate=0.04, learning_rate=0.25,
                 suppression_distance=0.04, memory_gain=0.35):
        self.rng = random.Random(seed)
        self.clone_count = clone_count
        self.mutation_rate = mutation_rate
        self.learning_rate = learning_rate
        self.suppression_distance = suppression_distance
        self.memory_gain = memory_gain
        self.memory: Dict[str, List[Clone]] = {}

    @staticmethod
    def pair_key(a, b, relation):
        return f"{a}|{b}|{relation}"

    @staticmethod
    def activation(signals):
        if not signals or any(
            signals.get(k) is None
            for k in ("danger", "safe", "mature")
        ):
            return 1.0
        return clip01(
            0.5 * clip01(signals["danger"])
            + 0.3 * clip01(signals["mature"])
            + 0.2 * (1 - clip01(signals["safe"]))
        )

    def evaluate(self, a, b, risks, dca_signals=None):
        dca_signals = dca_signals or {}
        repulsions = {}
        for relation in RELATIONS:
            risk = clip01(risks[relation])
            key = self.pair_key(a, b, relation)
            clones = self.memory.get(key, [])
            affinity = (
                min(clones, key=lambda c: abs(c.affinity-risk)).affinity
                if clones else 0.0
            )
            activation = self.activation(dca_signals.get(relation))
            repulsions[relation] = clip01(
                risk + (1-risk) * self.memory_gain
                * affinity * activation
            )
        dominant = max(repulsions, key=repulsions.get)
        return Decision(
            relation_repulsion=repulsions,
            total_repulsion=repulsions[dominant],
            dominant_relation=dominant,
        )

    def feedback(self, a, b, risks, exposure, dca_signals=None):
        dca_signals = dca_signals or {}
        for relation in RELATIONS:
            risk = clip01(risks[relation])
            observed = clip01(exposure.get(relation, 0))
            activation = self.activation(dca_signals.get(relation))
            antigen = clip01(risk * observed * activation)
            key = self.pair_key(a, b, relation)
            clones = list(self.memory.get(key, []))
            if not clones:
                clones = [Clone(antigen, observed)]
            selected = min(
                clones, key=lambda c: abs(c.affinity-antigen)
            )
            candidates = list(clones)
            for _ in range(self.clone_count):
                mutation = self.rng.uniform(
                    -self.mutation_rate, self.mutation_rate
                ) * observed * activation
                affinity = clip01(
                    selected.affinity
                    + self.learning_rate*(antigen-selected.affinity)
                    + mutation
                )
                candidates.append(Clone(affinity, observed))
            candidates.sort(key=lambda c: abs(c.affinity-antigen))
            kept = []
            for candidate in candidates:
                if all(
                    abs(candidate.affinity-c.affinity)
                    >= self.suppression_distance for c in kept
                ):
                    kept.append(candidate)
                if len(kept) >= self.clone_count:
                    break
            self.memory[key] = kept

    def feedback_context(self, a, b):
        result = {}
        for relation in RELATIONS:
            clones = self.memory.get(self.pair_key(a,b,relation), [])
            result[relation] = {
                "memory_affinity": max(
                    (c.affinity for c in clones), default=0.0
                ),
                "memory_exposure": max(
                    (c.exposure for c in clones), default=0.0
                ),
                "clone_count": len(clones),
            }
        return result
