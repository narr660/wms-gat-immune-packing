"""PART 3 심화 — 위험-비용 변환 규칙의 면역 기반 최적화 (RQ3 보강 실험).

같은 평가 예산에서 네 탐색기(랜덤·GA·CSA·aiNet)로 6차원 비용 규칙을 탐색하고,
그리드 균형점(현행 규칙)과 비교한다. 검증 체계는 본 실험과 동일:
- 탐색·선택은 전부 검증셋에서 (세대별 부분집합은 모든 탐색기에 공통)
- 테스트셋은 각 탐색기의 최종 규칙 하나당 1회만 평가 (--test)

    python -m scripts.optimize_rules --config configs/main.yaml --seed 42 --test
    python -m scripts.optimize_rules --config configs/main.yaml --seed 42 \
        --pop 8 --gens 4 --subset 12 --val-seqs 40            # 빠른 연결 확인

비용 J = 동적재(부분합)/기준 동적재 + 박스 평균/기준 박스  (기준 = seed 규칙, 같은 부분집합).
안전과 효율을 같은 비중으로 묶은 단일 목적이며, aiNet 생존 집단은 서로 다른
균형점(프런티어 후보)으로 별도 보고한다.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.run_packing import evaluate_boxes, sequence_risk
from src.config import load_config
from src.data.catalog import SkuCatalog
from src.data.graphs import _ordered_items, load_sequences, load_split
from src.immune.ainet import AiNetMemory
from src.models.factory import build_model
from src.packing.environment import PackingEnv
from src.packing.optimizers import OPTIMIZERS
from src.packing.policies import PolicyConfig, choose
from src.packing.rules import (RULE_NAMES, SEED_RULE, RuleParams,
                               pack_sequence_rule)
from src.packing.safety import SafetyFilter
from src.pipeline import device_auto, prepare_data, run_dir, save_json
from src.schema import RELATIONS


def build_tables(cfg, model, device, memory, ids, graphs, raw, label):
    out = {}
    t0 = time.time()
    with torch.no_grad():
        for k, sid in enumerate(ids):
            items = _ordered_items(raw[sid], cfg.data.max_nodes)
            risk, _ = sequence_risk(model, graphs[sid], device, memory)
            out[sid] = (items, risk)
            if (k + 1) % 100 == 0:
                print(f"  {label} 위험 테이블 {k + 1}/{len(ids)} ({time.time() - t0:.0f}s)")
    print(f"{label} 위험 테이블 {len(ids)}개 준비 ({time.time() - t0:.0f}s)")
    return out


def eval_rule(vec, tables, ids, thresholds):
    """규칙 벡터를 ids 시퀀스에서 적재 평가 — (동적재 합, 박스 평균, 활용률 평균)."""
    p = RuleParams.from_vec(vec)
    high, boxes, utils = 0, [], []
    for sid in ids:
        items, risk = tables[sid]
        safety = SafetyFilter(items, risk, thresholds)
        env = pack_sequence_rule(items, safety, p)
        _, h, _ = evaluate_boxes(env, items)
        high += h
        boxes.append(len(env.boxes))
        utils.append(env.utilization())
    return dict(high=int(high), boxes=float(np.mean(boxes)), util=float(np.mean(utils)))


def eval_grid_rule(tables, ids, thresholds):
    """현행 그리드 균형점 규칙 (관계 max 집계, policies.choose) — 참조 기준."""
    cfgp = PolicyConfig()
    high, boxes, utils = 0, [], []
    for sid in ids:
        items, risk = tables[sid]
        safety = SafetyFilter(items, risk, thresholds)
        env = PackingEnv(items)
        for i in range(len(items)):
            a = choose(env, safety, i, cfgp)
            env.step(i, *a) if a else env.step(i, 0, (0, 0, 0))
        _, h, _ = evaluate_boxes(env, items)
        high += h
        boxes.append(len(env.boxes))
        utils.append(env.utilization())
    return dict(high=int(high), boxes=float(np.mean(boxes)), util=float(np.mean(utils)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", default="random,ga,csa,ainet")
    parser.add_argument("--pop", type=int, default=16)
    parser.add_argument("--gens", type=int, default=12)
    parser.add_argument("--subset", type=int, default=30, help="세대별 평가 시퀀스 수")
    parser.add_argument("--val-seqs", type=int, default=315, help="검증 시퀀스 수 (최종 선택용)")
    parser.add_argument("--test", action="store_true", help="최종 규칙을 테스트셋에서 1회 평가")
    parser.add_argument("--limit-test", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_auto()
    model_dir = run_dir(cfg, "immune", args.seed)
    model = build_model("immune", cfg.model).to(device)
    model.load_state_dict(torch.load(model_dir / "best.pt", map_location=device))
    model.eval()
    thresholds = json.loads((model_dir / "thresholds.json").read_text())["thresholds"]
    mem_path = Path("outputs") / cfg.experiment / f"immune_memory_seed{args.seed}" / "ainet_memory.pkl"
    memory = AiNetMemory.load(mem_path) if mem_path.exists() else None
    print(f"aiNet 기억(frozen): {'불러옴' if memory else '없음'} | 임계값: {thresholds}")

    datasets, _, _ = prepare_data(cfg)
    graphs = {g.sequence_id: g for ds in datasets.values() for g in ds}
    raw = load_sequences(cfg.data.dataset)
    if cfg.data.sku_catalog:
        raw = SkuCatalog(raw).apply(raw)
    split = load_split(cfg.data.split)
    val_ids = list(split["val"])[: args.val_seqs]
    val_tab = build_tables(cfg, model, device, memory, val_ids, graphs, raw, "검증")

    # 세대별 공통 부분집합 + seed 규칙 기준치 (모든 탐색기가 같은 것을 본다)
    sched_rng = np.random.default_rng(args.seed * 7919)
    subsets = [list(sched_rng.choice(val_ids, size=min(args.subset, len(val_ids)),
                                     replace=False)) for _ in range(args.gens + 4)]
    base_cache = {}

    def eval_fn(vec, gen):
        ids = subsets[min(gen, len(subsets) - 1)]
        if gen not in base_cache:
            base_cache[gen] = eval_rule(SEED_RULE, val_tab, ids, thresholds)
        base = base_cache[gen]
        m = eval_rule(vec, val_tab, ids, thresholds)
        return (m["high"] / max(1, base["high"])) + (m["boxes"] / base["boxes"])

    # 전체 검증셋 기준치·참조 규칙
    print("기준 규칙 평가 (검증 전체)...")
    base_full = eval_rule(SEED_RULE, val_tab, val_ids, thresholds)
    grid_full = eval_grid_rule(val_tab, val_ids, thresholds)

    def j_full(m):
        return m["high"] / max(1, base_full["high"]) + m["boxes"] / base_full["boxes"]

    results = {"_baselines": {
        "grid(현행 max-규칙)": dict(val=grid_full, j_val=j_full(grid_full)),
        "seed(가중합 1,1,1)": dict(val=base_full, j_val=j_full(base_full)),
    }}
    for name in args.methods.split(","):
        opt = OPTIMIZERS[name.strip()]
        rng = np.random.default_rng(args.seed * 1000 + hash(name) % 1000)
        t0 = time.time()
        r = opt(eval_fn, rng, pop=args.pop, gens=args.gens, seed_vec=SEED_RULE)
        # 최종 선택: 그 탐색기의 best 를 검증 전체로 재평가 (선택도 검증셋 안에서)
        m_val = eval_rule(r["best"], val_tab, val_ids, thresholds)
        entry = dict(best_vec={n: float(v) for n, v in zip(RULE_NAMES, r["best"])},
                     evals=r["evals"], history=r["history"],
                     val=m_val, j_val=j_full(m_val), seconds=round(time.time() - t0, 1))
        if name.strip() == "ainet":
            entry["min_pairwise_dist"] = r.get("min_pairwise_dist")
            entry["survivors"] = [
                dict(vec={n: float(x) for n, x in zip(RULE_NAMES, s["vec"])},
                     j_subset=s["j"],
                     val=eval_rule(s["vec"], val_tab, val_ids, thresholds))
                for s in sorted(r.get("survivors", []), key=lambda s: s["j"])[:8]]
        results[name.strip()] = entry
        print(f"[{name.strip():>6}] evals {r['evals']} | 검증 동적재 {m_val['high']} "
              f"박스 {m_val['boxes']:.2f} | J {entry['j_val']:.3f} ({entry['seconds']}s)")

    if args.test:
        test_ids = list(split["test"])[: args.limit_test or None]
        test_tab = build_tables(cfg, model, device, memory, test_ids, graphs, raw, "테스트")
        results["_baselines"]["grid(현행 max-규칙)"]["test"] = eval_grid_rule(test_tab, test_ids, thresholds)
        results["_baselines"]["seed(가중합 1,1,1)"]["test"] = eval_rule(SEED_RULE, test_tab, test_ids, thresholds)
        for name in args.methods.split(","):
            n = name.strip()
            vec = np.array([results[n]["best_vec"][k] for k in RULE_NAMES])
            results[n]["test"] = eval_rule(vec, test_tab, test_ids, thresholds)
            print(f"[{n:>6}] 테스트 동적재 {results[n]['test']['high']} "
                  f"박스 {results[n]['test']['boxes']:.2f} 활용률 {results[n]['test']['util']:.1%}")

    out_dir = Path("reports") / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# 비용 규칙 최적화 — 시드 {args.seed} · pop {args.pop} × gens {args.gens} "
             f"(예산 {args.pop * args.gens}평가) · 세대 부분집합 {args.subset}", "",
             "| 방법 | 검증 동적재 | 검증 박스 | 검증 활용률 | J(검증) |"
             + (" 테스트 동적재 | 테스트 박스 | 테스트 활용률 |" if args.test else ""),
             "|---|---|---|---|---|" + ("---|---|---|" if args.test else "")]
    rows = list(results["_baselines"].items()) + [
        (k, v) for k, v in results.items() if k != "_baselines"]
    for name, e in rows:
        v = e["val"]
        line = (f"| {name} | {v['high']:,} | {v['boxes']:.2f} | {v['util']:.1%} "
                f"| {e['j_val']:.3f} |")
        if args.test and "test" in e:
            t = e["test"]
            line += f" {t['high']:,} | {t['boxes']:.2f} | {t['util']:.1%} |"
        lines.append(line)
    if "ainet" in results and results["ainet"].get("survivors"):
        lines += ["", "## aiNet 생존 항체 (억제가 유지한 서로 다른 균형점들)", "",
                  "| # | 검증 동적재 | 검증 박스 | w_odor | w_contam | w_phys | w_space | w_newbox | tau |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for i, s in enumerate(results["ainet"]["survivors"], 1):
            v, w = s["val"], s["vec"]
            lines.append(f"| {i} | {v['high']:,} | {v['boxes']:.2f} | "
                         + " | ".join(f"{w[k]:.2f}" for k in RULE_NAMES) + " |")
    save_json(out_dir / f"rules_opt_seed{args.seed}.json", results)
    (out_dir / f"rules_opt_seed{args.seed}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"저장: {out_dir}/rules_opt_seed{args.seed}.md")


if __name__ == "__main__":
    main()
