from dataclasses import dataclass, field
from typing import Dict, List
import math
import random

RELATIONS = ("odor", "contamination", "physical")


def clip01(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("유한한 숫자만 허용합니다.")
    return max(0.0, min(1.0, value))


@dataclass
class ImmuneClone:
    affinity: float
    exposure: float
    age: int = 0


@dataclass
class AiNetV2Decision:
    relation_repulsion: Dict[str, float]
    total_repulsion: float
    dominant_relation: str


class RelationAiNetV2:
    """Relation-specific adaptive immune memory.

    Input risks must come from the existing M3 model.
    No DCA signal is recalculated here.
    """

    def __init__(
        self,
        clone_count=5,
        mutation_rate=0.08,
        learning_rate=0.25,
        suppression_distance=0.04,
        memory_gain=0.35,
        seed=42,
    ):
        self.clone_count = int(clone_count)
        self.mutation_rate = float(mutation_rate)
        self.learning_rate = float(learning_rate)
        self.suppression_distance = float(
            suppression_distance
        )
        self.memory_gain = float(memory_gain)
        self.rng = random.Random(seed)
        self.memory: Dict[str, List[ImmuneClone]] = {}

        if self.clone_count < 1:
            raise ValueError("clone_count는 1 이상이어야 합니다.")

    @staticmethod
    def pair_key(item_a, item_b, relation):
        return f"{item_a}|{item_b}|{relation}"

    def evaluate(self, item_a, item_b, relation_risks):
        relation_repulsion = {}

        for relation in RELATIONS:
            if relation not in relation_risks:
                raise ValueError(
                    f"{relation} 위험도가 없습니다."
                )

            risk = clip01(relation_risks[relation])
            key = self.pair_key(
                item_a, item_b, relation
            )

            clones = self.memory.get(key, [])

            if clones:
                nearest = min(
                    clones,
                    key=lambda c: abs(
                        c.affinity - risk
                    ),
                )
                affinity = nearest.affinity
            else:
                affinity = 0.0

            relation_repulsion[relation] = clip01(
                risk + self.memory_gain * affinity
            )

        dominant = max(
            RELATIONS,
            key=lambda r: relation_repulsion[r]
        )

        return AiNetV2Decision(
            relation_repulsion=relation_repulsion,
            total_repulsion=max(
                relation_repulsion.values()
            ),
            dominant_relation=dominant,
        )

    def feedback(
        self,
        item_a,
        item_b,
        relation_risks,
        observed_exposure,
    ):
        """Update clones only after a committed placement."""

        for relation in RELATIONS:
            risk = clip01(relation_risks[relation])
            exposure = clip01(
                observed_exposure.get(relation, 0.0)
            )

            key = self.pair_key(
                item_a, item_b, relation
            )

            clones = list(self.memory.get(key, []))

            antigen = clip01(risk * exposure)

            if not clones:
                clones = [
                    ImmuneClone(
                        affinity=antigen,
                        exposure=exposure,
                    )
                ]

            # Clonal selection: closest affinity to antigen.
            selected = min(
                clones,
                key=lambda c: abs(
                    c.affinity - antigen
                ),
            )

            new_clones = []

            for _ in range(self.clone_count):
                mutation = self.rng.uniform(
                    -self.mutation_rate,
                    self.mutation_rate,
                ) * exposure

                affinity = clip01(
                    selected.affinity
                    + self.learning_rate
                    * (antigen - selected.affinity)
                    + mutation
                )

                new_clones.append(
                    ImmuneClone(
                        affinity=affinity,
                        exposure=exposure,
                    )
                )

            # Suppression: remove near-duplicate affinities.
            combined = clones + new_clones

            combined.sort(
                key=lambda c: abs(
                    c.affinity - antigen
                )
            )

            suppressed = []

            for clone in combined:
                if all(
                    abs(clone.affinity - existing.affinity)
                    >= self.suppression_distance
                    for existing in suppressed
                ):
                    suppressed.append(clone)

            self.memory[key] = suppressed[
                :self.clone_count
            ]

    def reset_memory(self):
        self.memory.clear()
