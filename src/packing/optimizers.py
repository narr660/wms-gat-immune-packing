"""비용 규칙 탐색기 4종 — 랜덤 탐색 · GA · CSA · aiNet (동일 평가 예산 비교).

모든 탐색기는 같은 서명을 가진다:

    result = optimize(eval_fn, rng, pop=16, gens=12, seed_vec=SEED_RULE)

- eval_fn(vec, gen) -> float : 규칙 벡터의 비용 J (낮을수록 좋음). gen 은 세대 번호로,
  호출 측이 세대별 평가 시퀀스 부분집합을 스케줄링하는 데 쓴다 (모든 탐색기가 같은
  세대에 같은 부분집합을 보게 되어 공정하다).
- 총 평가 횟수 = pop * gens 로 통일한다 (evals 로 보고).

계보 관계: aiNet = CSA(클론·친화도 비례 변이·기억) + 네트워크 억제(유사 항체 제거).
GA 는 교차 + 고정 변이율의 표준 구현. 따라서
  GA  vs CSA  = 교차 없음 + 친화도 비례 변이의 효과
  CSA vs aiNet = 억제(다양성 유지)의 효과
가 동일 예산에서 분리된다.
"""
from typing import Callable, Dict, List

import numpy as np

from src.packing.rules import RULE_HI, RULE_LO, clip_rule

RANGE = RULE_HI - RULE_LO


def _norm(v: np.ndarray) -> np.ndarray:
    return (v - RULE_LO) / RANGE


def random_rule(rng: np.random.Generator) -> np.ndarray:
    return RULE_LO + rng.random(len(RULE_LO)) * RANGE


def _init_pop(rng, pop, seed_vec):
    """seed 규칙 1 + 그 주변 변이 일부 + 나머지 무작위 — '최소한 seed 만큼'을 보장."""
    out = [clip_rule(seed_vec)]
    for _ in range(max(0, pop // 4 - 1)):
        out.append(clip_rule(seed_vec + rng.normal(0, 0.1, len(seed_vec)) * RANGE))
    while len(out) < pop:
        out.append(random_rule(rng))
    return out[:pop]


def _result(best_vec, best_j, history, evals, population, extras=None):
    r = dict(best=np.asarray(best_vec), best_j=float(best_j), history=history,
             evals=int(evals), population=[np.asarray(p) for p in population])
    if extras:
        r.update(extras)
    return r


def random_search(eval_fn: Callable, rng: np.random.Generator,
                  pop: int = 16, gens: int = 12, seed_vec=None) -> Dict:
    best_vec, best_j, history, evals = None, float("inf"), [], 0
    last = []
    for g in range(gens):
        last = [clip_rule(seed_vec)] if (g == 0 and seed_vec is not None) else []
        while len(last) < pop:
            last.append(random_rule(rng))
        for v in last:
            j = eval_fn(v, g); evals += 1
            if j < best_j:
                best_vec, best_j = v, j
        history.append(best_j)
    return _result(best_vec, best_j, history, evals, last)


def ga(eval_fn: Callable, rng: np.random.Generator,
       pop: int = 16, gens: int = 12, seed_vec=None,
       sigma: float = 0.15, p_mut: float = 0.3, elite: int = 2) -> Dict:
    """표준 GA: 토너먼트 선택(k=3) + 균등 교차 + 고정 가우시안 변이 + 엘리트 보존."""
    P = _init_pop(rng, pop, seed_vec if seed_vec is not None else random_rule(rng))
    J = [eval_fn(v, 0) for v in P]; evals = len(P)
    history = [float(min(J))]
    for g in range(1, gens):
        order = np.argsort(J)
        nxt = [P[i] for i in order[:elite]]
        while len(nxt) < pop:
            def pick():
                c = rng.integers(0, pop, 3)
                return P[min(c, key=lambda i: J[i])]
            a, b = pick(), pick()
            mask = rng.random(len(a)) < 0.5
            child = np.where(mask, a, b)
            mut = rng.random(len(child)) < p_mut
            child = child + mut * rng.normal(0, sigma, len(child)) * RANGE
            nxt.append(clip_rule(child))
        P = nxt
        J = [eval_fn(v, g) for v in P]; evals += len(P)
        history.append(float(min(min(J), history[-1])))
    i = int(np.argmin(J))
    return _result(P[i], J[i], history, evals, P)


def _clone_mutate_select(eval_fn, rng, P, J, g, pop, beta, s_min, s_max, n_random):
    """CSA 한 세대: 친화도 비례 클론 -> 친화도 역비례 변이 -> 선택 (+ 무작위 주입)."""
    order = np.argsort(J)
    cand_v, cand_j = [P[i] for i in order], [J[i] for i in order]
    n_evals = 0
    for rank, i in enumerate(order):
        n_clones = max(1, int(round(beta * (pop - rank) / pop)))
        sigma = s_min + (s_max - s_min) * rank / max(1, pop - 1)   # 좋은 항체일수록 작게
        for _ in range(n_clones):
            v = clip_rule(P[i] + rng.normal(0, sigma, len(P[i])) * RANGE)
            cand_v.append(v); cand_j.append(eval_fn(v, g)); n_evals += 1
    for _ in range(n_random):                                      # 다양성 주입
        v = random_rule(rng)
        cand_v.append(v); cand_j.append(eval_fn(v, g)); n_evals += 1
    return cand_v, cand_j, n_evals


def csa(eval_fn: Callable, rng: np.random.Generator,
        pop: int = 16, gens: int = 12, seed_vec=None,
        beta: float = 0.6, s_min: float = 0.03, s_max: float = 0.30,
        n_random: int = 2) -> Dict:
    """CLONALG 계열: 클론 선택 + 친화도 비례 변이 + 기억 (억제 없음)."""
    P = _init_pop(rng, pop, seed_vec if seed_vec is not None else random_rule(rng))
    J = [eval_fn(v, 0) for v in P]; evals = len(P)
    budget = pop * gens
    history = [float(min(J))]
    g = 1
    while evals < budget:
        cand_v, cand_j, n = _clone_mutate_select(
            eval_fn, rng, P, J, g, pop, beta, s_min, s_max, n_random)
        evals += n
        order = np.argsort(cand_j)[:pop]
        P = [cand_v[i] for i in order]
        J = [cand_j[i] for i in order]
        history.append(float(min(min(J), history[-1])))
        g += 1
    i = int(np.argmin(J))
    return _result(P[i], J[i], history, evals, P)


def ainet(eval_fn: Callable, rng: np.random.Generator,
          pop: int = 16, gens: int = 12, seed_vec=None,
          beta: float = 0.6, s_min: float = 0.03, s_max: float = 0.30,
          n_random: int = 2, eps: float = 0.12) -> Dict:
    """aiNet = CSA + 네트워크 억제: 정규화 공간 거리 eps 미만의 유사 항체 중 열등한
    쪽을 제거해 다양성을 유지한다. 산출물은 최적 규칙 하나가 아니라 생존 항체 집합
    (서로 다른 안전-효율 균형점들)이다."""
    P = _init_pop(rng, pop, seed_vec if seed_vec is not None else random_rule(rng))
    J = [eval_fn(v, 0) for v in P]; evals = len(P)
    budget = pop * gens
    history = [float(min(J))]
    g = 1
    while evals < budget:
        cand_v, cand_j, n = _clone_mutate_select(
            eval_fn, rng, P, J, g, pop, beta, s_min, s_max, n_random)
        evals += n
        order = list(np.argsort(cand_j))
        keep_v, keep_j = [], []
        for i in order:                                     # 억제: 좋은 것부터, 근접 중복 제거
            v = cand_v[i]
            if any(np.linalg.norm(_norm(v) - _norm(k)) < eps for k in keep_v):
                continue
            keep_v.append(v); keep_j.append(cand_j[i])
            if len(keep_v) >= pop:
                break
        while len(keep_v) < pop:                            # 빈자리는 무작위로 (eps 존중 시도)
            v = random_rule(rng)
            for _ in range(20):
                if not any(np.linalg.norm(_norm(v) - _norm(k)) < eps for k in keep_v):
                    break
                v = random_rule(rng)
            keep_v.append(v); keep_j.append(eval_fn(v, g)); evals += 1
        P, J = keep_v, keep_j
        history.append(float(min(min(J), history[-1])))
        g += 1
    i = int(np.argmin(J))
    sup = [dict(vec=np.asarray(v), j=float(j)) for v, j in zip(P, J)]
    dists = [float(np.linalg.norm(_norm(a) - _norm(b)))
             for x, a in enumerate(P) for b in P[x + 1:]]
    extras = dict(survivors=sup,
                  min_pairwise_dist=float(min(dists)) if dists else None)
    return _result(P[i], J[i], history, evals, P, extras)


OPTIMIZERS = {"random": random_search, "ga": ga, "csa": csa, "ainet": ainet}
