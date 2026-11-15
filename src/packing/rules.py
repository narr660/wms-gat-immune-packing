"""PART 3 심화 — 파라미터화된 위험-비용 변환 규칙.

기존 탐욕 비용(policies.choose)은 세 관계의 보정 반발강도를 '최댓값 하나'로 합쳐
고정 가중치로 쓴다. 이 모듈은 그 변환을 규칙 벡터로 파라미터화한다:

    risk_term = max(0, w_odor*r_o + w_contam*r_c + w_physical*r_p - tau)
    cost      = risk_term + w_space * z/BOX + w_newbox * [새 박스]

- r_o, r_c : 박스 단위 관계(냄새·오염) — 같은 박스 상품들과의 보정 반발강도 최대(양방향)
- r_p      : 쌓기 단위 관계(물리) — 이 자리에서 아래에 깔리는 상품들과의 반발강도 최대(위->아래)
- tau      : 연속 임계(soft threshold). 금지 규칙·고위험 마스킹(하드 제약)은 그대로 유지한다.

규칙 벡터는 (w_odor, w_contam, w_physical, w_space, w_newbox, tau) 6차원이며,
seed 규칙 (1, 1, 1, 0.15, 1.0, 0)은 균형점 가중치의 가중합 버전이다
(기존 규칙은 관계 간 max 집계라 가중합과 동일하지는 않다 — 보고서에 둘 다 기준으로 싣는다).
"""
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.packing.environment import BOX, PackingEnv
from src.packing.safety import BOX_LEVEL, STACK_LEVEL, SafetyFilter

#                 w_odor  w_contam  w_phys  w_space  w_newbox   tau
RULE_LO = np.array([0.0,    0.0,     0.0,    0.0,     0.10,     0.0], dtype=np.float64)
RULE_HI = np.array([3.0,    3.0,     3.0,    0.50,    3.00,     1.5], dtype=np.float64)
SEED_RULE = np.array([1.0,  1.0,     1.0,    0.15,    1.00,     0.0], dtype=np.float64)
RULE_NAMES = ["w_odor", "w_contam", "w_physical", "w_space", "w_newbox", "tau"]


def clip_rule(vec: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(vec, dtype=np.float64), RULE_LO, RULE_HI)


@dataclass
class RuleParams:
    w_odor: float = 1.0
    w_contam: float = 1.0
    w_physical: float = 1.0
    w_space: float = 0.15
    w_newbox: float = 1.0
    tau: float = 0.0

    @classmethod
    def from_vec(cls, vec) -> "RuleParams":
        v = clip_rule(vec)
        return cls(*[float(x) for x in v])

    def to_vec(self) -> np.ndarray:
        return np.array([self.w_odor, self.w_contam, self.w_physical,
                         self.w_space, self.w_newbox, self.tau], dtype=np.float64)


def _relation_risks(safety: SafetyFilter, new_index: int, box, spot, dims):
    """이 후보 자리의 관계별 위험 (r_odor, r_contam, r_physical)."""
    risk = safety.risk
    if risk is None:
        return 0.0, 0.0, 0.0
    idx = box.item_indices()
    out: Dict[str, float] = {r: 0.0 for r in BOX_LEVEL}
    for j in idx:
        for r in BOX_LEVEL:
            v = max(float(risk.rho[r][new_index, j]), float(risk.rho[r][j, new_index]))
            if v > out[r]:
                out[r] = v
    r_phys = 0.0
    for j in safety.under_items(box, spot, dims):
        r_phys = max(r_phys, float(risk.rho[STACK_LEVEL][new_index, j]))
    return out["odor"], out["contamination"], r_phys


def rule_choose(env: PackingEnv, safety: SafetyFilter, item_index: int, p: RuleParams):
    """choose()와 같은 후보·하드 제약에서, 비용만 규칙 벡터로 계산한다."""
    blocked_by_box = [safety.blocked_columns(item_index, box) for box in env.boxes]
    best, best_cost = None, float("inf")
    for box_id, spot, dims in env.feasible_actions(item_index, blocked_by_box):
        is_new = box_id == len(env.boxes)
        if not is_new and not safety.box_allowed(item_index, env.boxes[box_id].item_indices()):
            continue
        if is_new:
            risk_term = 0.0
        else:
            r_o, r_c, r_p = _relation_risks(safety, item_index, env.boxes[box_id], spot, dims)
            risk_term = max(0.0, p.w_odor * r_o + p.w_contam * r_c
                            + p.w_physical * r_p - p.tau)
        cost = risk_term + p.w_space * spot[2] / BOX + p.w_newbox * float(is_new)
        if cost < best_cost:
            best, best_cost = (box_id, spot, dims), cost
    return best


def pack_sequence_rule(items: List[Dict], safety: SafetyFilter, p: RuleParams,
                       allow_rotation: bool = True) -> PackingEnv:
    env = PackingEnv(items, allow_rotation=allow_rotation)
    for i in range(len(items)):
        action = rule_choose(env, safety, i, p)
        if action is None:
            env.step(i, 0, (0, 0, 0))
            continue
        env.step(i, *action)
    return env
