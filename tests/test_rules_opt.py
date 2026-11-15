"""비용 규칙 파라미터화·탐색기(랜덤/GA/CSA/aiNet) 검증 — 외부 데이터 불필요."""
import numpy as np
import pytest

from src.packing.environment import PackingEnv
from src.packing.optimizers import OPTIMIZERS, _norm, ainet, csa, ga, random_search
from src.packing.rules import (RULE_HI, RULE_LO, SEED_RULE, RuleParams,
                               pack_sequence_rule, rule_choose)
from src.packing.safety import PairwiseRisk, SafetyFilter


# ---------- 탐색기: 장난감 목적함수 (정규화 공간의 2차 비용) ----------

TARGET = RULE_LO + 0.37 * (RULE_HI - RULE_LO)


def toy_eval(vec, gen):
    return float(np.sum((_norm(np.asarray(vec)) - _norm(TARGET)) ** 2))


@pytest.mark.parametrize("name,factor", [("random", 1.0), ("ga", 0.5),
                                         ("csa", 0.5), ("ainet", 0.5)])
def test_optimizers_improve_on_seed(name, factor):
    rng = np.random.default_rng(0)
    seed_j = toy_eval(SEED_RULE, 0)
    r = OPTIMIZERS[name](toy_eval, rng, pop=10, gens=8, seed_vec=SEED_RULE)
    assert r["best_j"] <= seed_j * factor + 1e-9, \
        f"{name}: 개선 부족 ({r['best_j']:.3f} vs seed {seed_j:.3f})"
    budget = 10 * 8
    assert budget * 0.9 <= r["evals"] <= budget * 1.6
    assert len(r["history"]) >= 2 and r["history"][-1] <= r["history"][0]


def test_history_monotone_nonincreasing():
    rng = np.random.default_rng(1)
    for name in OPTIMIZERS:
        r = OPTIMIZERS[name](toy_eval, np.random.default_rng(1), pop=8, gens=6,
                             seed_vec=SEED_RULE)
        h = r["history"]
        assert all(h[i + 1] <= h[i] + 1e-12 for i in range(len(h) - 1)), name


def test_ainet_suppression_keeps_diversity():
    rng = np.random.default_rng(2)
    r = ainet(toy_eval, rng, pop=10, gens=8, seed_vec=SEED_RULE, eps=0.12)
    assert "survivors" in r and len(r["survivors"]) == 10
    assert r["min_pairwise_dist"] is not None
    assert r["min_pairwise_dist"] >= 0.12 * 0.5     # 억제가 근접 중복을 걸러냄
    # CSA(억제 없음) 대비 생존 집단이 더 흩어져 있어야 한다
    rc = csa(toy_eval, np.random.default_rng(2), pop=10, gens=8, seed_vec=SEED_RULE)
    def spread(pop):
        v = np.stack([_norm(np.asarray(p)) for p in pop])
        d = [np.linalg.norm(a - b) for i, a in enumerate(v) for b in v[i + 1:]]
        return float(np.mean(d))
    assert spread(r["population"]) > spread(rc["population"])


# ---------- 규칙 적재: 작은 합성 시퀀스 ----------

def _items(n):
    return [dict(size=(2.0, 2.0, 2.0), category="일반", hazard_type="none",
                 hazard_level=0.0) for _ in range(n)]


def _risk(n, odor=0.0):
    src, dst = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    keep = src != dst
    ei = np.stack([src[keep], dst[keep]])
    m = ei.shape[1]
    rho = {r: np.zeros(m, dtype=np.float32) for r in ("odor", "contamination", "physical")}
    rho["odor"][:] = odor
    p_high = {r: np.zeros(m, dtype=np.float32) for r in rho}
    return PairwiseRisk(n, rho, p_high, ei)


def test_rule_weights_change_decision():
    """냄새 반발강도 0.9인 쌍: w_odor가 크면 분리(박스 2), 0이면 혼적(박스 1)."""
    thresholds = {"odor": 0.99, "contamination": 0.99, "physical": 0.99}
    items = _items(2)
    for w_odor, expect_boxes in [(0.0, 1), (3.0, 2)]:
        safety = SafetyFilter(items, _risk(2, odor=0.9), thresholds)
        p = RuleParams(w_odor=w_odor, w_contam=0, w_physical=0,
                       w_space=0.15, w_newbox=1.0, tau=0.0)
        env = pack_sequence_rule(items, safety, p)
        assert len(env.boxes) == expect_boxes, f"w_odor={w_odor}"


def test_tau_soft_threshold_restores_mixing():
    """tau가 위험 가중합을 전부 흡수하면 혼적으로 돌아간다."""
    thresholds = {"odor": 0.99, "contamination": 0.99, "physical": 0.99}
    items = _items(2)
    safety = SafetyFilter(items, _risk(2, odor=0.9), thresholds)
    p = RuleParams(w_odor=1.0, w_contam=0, w_physical=0,
                   w_space=0.15, w_newbox=1.0, tau=1.2)   # 0.9 - 1.2 -> 위험항 0
    env = pack_sequence_rule(items, safety, p)
    assert len(env.boxes) == 1


def test_rule_choose_respects_hard_filter():
    """마스킹(P_high >= theta)은 규칙과 무관하게 차단된다."""
    thresholds = {"odor": 0.5, "contamination": 0.99, "physical": 0.99}
    items = _items(2)
    risk = _risk(2, odor=0.9)
    for r in ("odor",):
        risk.p_high[r][:] = 0.9                      # 임계 초과 -> 박스 단위 차단
    safety = SafetyFilter(items, risk, thresholds)
    p = RuleParams(w_odor=0.0, w_contam=0, w_physical=0,
                   w_space=0.15, w_newbox=1.0, tau=1.5)  # 비용상 혼적 유리해도
    env = pack_sequence_rule(items, safety, p)
    assert len(env.boxes) == 2
