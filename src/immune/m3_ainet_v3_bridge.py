from dataclasses import dataclass
from typing import Dict
from src.immune.ainet_v3 import RelationAiNetV3, RELATIONS
from src.immune.m3_ainet_bridge import M3AiNetBridge

@dataclass
class M3AiNetV3Decision:
    relation_risks: Dict[str, float]
    dca_signals: Dict[str, dict]
    relation_repulsion: Dict[str, float]
    total_repulsion: float
    dominant_relation: str
    hard_forbidden: bool
    constraint_status: str
    policy_blocked: bool

class M3AiNetV3Bridge:
    def __init__(self, ainet: RelationAiNetV3,
                 policy_threshold=0.85):
        self.ainet = ainet
        self.policy_threshold = policy_threshold

    def evaluate(self, a, b, m3):
        if m3.constraint_status not in (
            "forbidden", "unknown", "no_prohibition_found"
        ):
            raise ValueError("잘못된 규제 판정 상태")
        if m3.constraint_status == "forbidden" and not m3.hard_forbidden:
            raise ValueError("금지 판정과 hard_forbidden 불일치")

        risks, signals = {}, {}
        for relation in RELATIONS:
            output = m3.relations[relation]
            risks[relation] = M3AiNetBridge.expected_risk(
                output.probabilities
            )
            signals[relation] = {
                "danger": output.danger,
                "safe": output.safe,
                "mature": output.mature,
            }

        decision = self.ainet.evaluate(a, b, risks, signals)
        return M3AiNetV3Decision(
            relation_risks=risks,
            dca_signals=signals,
            relation_repulsion=decision.relation_repulsion,
            total_repulsion=decision.total_repulsion,
            dominant_relation=decision.dominant_relation,
            hard_forbidden=m3.hard_forbidden,
            constraint_status=m3.constraint_status,
            policy_blocked=(
                decision.total_repulsion >= self.policy_threshold
            ),
        )

    def feedback(self, a, b, decision, observed_exposure):
        self.ainet.feedback(
            a, b, decision.relation_risks,
            observed_exposure, decision.dca_signals
        )

    def feedback_context(self, a, b):
        return self.ainet.feedback_context(a, b)
