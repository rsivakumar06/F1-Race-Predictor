"""
Reconciliation test: option 3 (one coherent Monte-Carlo model) vs option 2
(separate delta-tree for order + grid/tree blend for probabilities).

The incoherence we're fixing: today the finishing ORDER (delta tree) and the
WIN/PODIUM probabilities (grid+tree blend) come from different models and can
contradict (a driver predicted P2 with 1% win). Option 3 derives BOTH from one
simulation, so they can't contradict — the question is whether that costs
accuracy or calibration vs the separate best-in-class models.

Monte-Carlo model (option 3):
  predicted position = grid + tree-delta
  per-driver DNF risk = 1 - rolling completion rate
  position noise ~ Normal(0, sigma), sigma = empirical spread of finish-grid
  simulate N races by ranking noisy outcomes; derive win/podium/expected-order.

Reports, on the same walk-forward races:
  ORDER        : MAE / Spearman / Winner / Top3   — tree (opt 2) vs mc (opt 3)
  WIN PROB     : Brier / LogLoss / ECE            — blend (opt 2) vs mc (opt 3)
  PODIUM PROB  : Brier / LogLoss / ECE            — blend (opt 2) vs mc (opt 3)
  COHERENCE    : how often the headline P1 == the highest-win-prob driver

Run from the project root:
    python reconcile_test.py [--start-year 2024] [--stride 1] [--sims 3000]
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
import config
from data.feature_engineering import FeatureEngineer
from models.gradient_boost import F1GradientBoostModel

COMPLETION_COL = "Driver_Rolllong_CompletionRate"


# ── metrics ──
def brier(p, y): return float(np.mean((p - y) ** 2))


def logloss(p, y):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); n = len(p); e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < bins - 1 else (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0: continue
        e += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(e)


def spearman(a, p):
    if len(a) < 3: return np.nan
    ar = pd.Series(a).rank().values; pr = pd.Series(p).rank().values
    if np.std(ar) == 0 or np.std(pr) == 0: return np.nan
    return float(np.corrcoef(ar, pr)[0, 1])


def _clf():
    params = config.XGBOOST_PARAMS.copy()
    params.pop("eval_metric", None)
    params["objective"] = "binary:logistic"; params["eval_metric"] = "logloss"
    return xgb.XGBClassifier(**params)


def simulate(pred_pos, dnf_prob, sigma, n_sims=3000, seed=42):
    """Rank noisy predicted positions over many sims -> win/podium/expected order."""
    rng = np.random.default_rng(seed)
    n = len(pred_pos)
    latent = pred_pos[None, :] + rng.normal(0, max(sigma, 0.5), size=(n_sims, n))
    dnf = rng.random((n_sims, n)) < dnf_prob[None, :]
    latent = latent + dnf * (1000.0 + rng.random((n_sims, n)))  # DNFs ranked last
    order = np.argsort(latent, axis=1)
    ranks = np.empty((n_sims, n), dtype=float)
    rows = np.arange(n_sims)[:, None]
    ranks[rows, order] = np.arange(1, n + 1)[None, :]
    return (ranks == 1).mean(axis=0), (ranks <= 3).mean(axis=0), ranks.mean(axis=0)


def _fit_delta_tree(prefix, feats):
    m = F1GradientBoostModel(target_mode="delta")
    m.feature_columns = feats
    clean = prefix.dropna(subset=["Target_RaceDelta", m.grid_col])
    X = clean[feats].copy(); m._feature_medians = X.median()
    m._fit_model(X.fillna(m._feature_medians).fillna(0), clean["Target_RaceDelta"].to_numpy())
    return m


def run(start_year=2024, stride=1, n_sims=3000):
    engineer = FeatureEngineer()
    df = pd.read_parquet(config.DATA_DIR / "features.parquet").sort_values(
        ["Year", "RoundNumber"]).reset_index(drop=True)
    keys = list(df[df["Year"] >= start_year][["Year", "RoundNumber"]]
                .drop_duplicates().itertuples(index=False, name=None))[::stride]
    print(f"  Walking {len(keys)} races...")

    recs = []
    coh_opt2 = coh_opt3 = coh_recon = n_races = 0
    for i, (yr, rnd) in enumerate(keys, 1):
        before = (df["Year"] < yr) | ((df["Year"] == yr) & (df["RoundNumber"] < rnd))
        prefix = df[before]
        test = df[(df["Year"] == yr) & (df["RoundNumber"] == rnd)].copy()
        test = test.dropna(subset=["Target_FinishPosition"])
        if prefix.empty or len(test) < 3:
            continue
        feats = engineer.get_race_feature_columns(prefix)

        # delta tree -> predicted positions (order, option 2)
        tree = _fit_delta_tree(prefix, feats)
        tp = tree.predict(test).set_index("Abbreviation")["PredictedPosition"]
        test["tree_pos"] = test["Abbreviation"].map(tp).to_numpy()

        # blend probs (option 2): grid prior + classifiers
        med = prefix[feats].median()
        Xtr = prefix[feats].fillna(med).fillna(0); Xte = test[feats].fillna(med).fillna(0)
        gp = prefix[prefix["GridPosition"] > 0]
        wbg = gp.groupby("GridPosition")["Target_IsWinner"].mean()
        pbg = gp.groupby("GridPosition")["Target_IsPodium"].mean()
        gw, gpd = gp["Target_IsWinner"].mean(), gp["Target_IsPodium"].mean()
        cw, cp = _clf(), _clf()
        cw.fit(Xtr, prefix["Target_IsWinner"].astype(int))
        cp.fit(Xtr, prefix["Target_IsPodium"].astype(int))
        gwin = test["GridPosition"].map(wbg).fillna(gw).to_numpy()
        gpod = test["GridPosition"].map(pbg).fillna(gpd).to_numpy()
        test["blend_win"] = 0.5 * gwin + 0.5 * cw.predict_proba(Xte)[:, 1]
        test["blend_pod"] = 0.5 * gpod + 0.5 * cp.predict_proba(Xte)[:, 1]

        # Monte-Carlo (option 3): one model -> order + probs
        pred_pos = test["tree_pos"].to_numpy(dtype=float)
        if COMPLETION_COL in test.columns:
            dnf = (1.0 - test[COMPLETION_COL]).to_numpy(dtype=float)
        else:
            dnf = np.full(len(test), np.nan)
        global_dnf = float(1.0 - prefix["IsFinished"].mean()) if "IsFinished" in prefix.columns else 0.12
        dnf = np.where(np.isnan(dnf), global_dnf, dnf)
        dnf = np.clip(dnf, 0.02, 0.5)
        finisher_delta = prefix["Target_RaceDelta"].dropna().clip(-10, 10)
        sigma = float(finisher_delta.std()) if len(finisher_delta) > 5 else 3.0
        mc_win, mc_pod, mc_pos = simulate(pred_pos, dnf, sigma, n_sims=n_sims)
        test["mc_win"], test["mc_pod"], test["mc_pos"] = mc_win, mc_pod, mc_pos

        # ── Option 2 reconciled order: probs set the top, tree sets the rest ──
        abbr = test["Abbreviation"].to_numpy()
        win_a = test["blend_win"].to_numpy(); pod_a = test["blend_pod"].to_numpy()
        tpos_a = test["tree_pos"].to_numpy()
        p1 = int(np.argmax(win_a))                       # P1 = win favourite
        pod_rest = [i for i in np.argsort(-pod_a) if i != p1][:2]  # P2,P3 = podium favs
        top = [p1] + pod_rest
        rest = [i for i in np.argsort(tpos_a) if i not in top]      # rest by tree
        recon_idx = top + rest
        recon_pos = np.empty(len(abbr))
        recon_pos[recon_idx] = np.arange(1, len(abbr) + 1)
        test["recon_pos"] = recon_pos

        # coherence: does the headline P1 match the highest-win-prob driver?
        p1_order_opt2 = abbr[np.argmin(test["tree_pos"].to_numpy())]
        p1_win_opt2 = abbr[np.argmax(win_a)]
        coh_opt2 += int(p1_order_opt2 == p1_win_opt2)
        coh_opt3 += int(abbr[np.argmin(mc_pos)] == abbr[np.argmax(mc_win)])
        coh_recon += int(abbr[np.argmin(recon_pos)] == abbr[np.argmax(win_a)])
        n_races += 1

        test["win"] = test["Target_IsWinner"].astype(float)
        test["pod"] = test["Target_IsPodium"].astype(float)
        test["actual"] = test["Target_FinishPosition"].astype(float)
        recs.append(test[["Year", "RoundNumber", "Abbreviation", "actual", "win", "pod",
                          "tree_pos", "mc_pos", "recon_pos",
                          "blend_win", "mc_win", "blend_pod", "mc_pod"]])
        if i % 10 == 0:
            print(f"    ...{i}/{len(keys)} races")

    rec = pd.concat(recs, ignore_index=True)
    return rec, coh_opt2 / max(n_races, 1), coh_opt3 / max(n_races, 1), coh_recon / max(n_races, 1), n_races


def _order_metrics(rec, col):
    d = rec.dropna(subset=[col, "actual"])
    mae = float(np.mean(np.abs(d[col] - d["actual"])))
    win_hit = top3 = sp = nr = 0
    for _, g in d.groupby(["Year", "RoundNumber"]):
        if len(g) < 3: continue
        a = g["actual"].to_numpy(); p = g[col].to_numpy(); ab = g["Abbreviation"].to_numpy()
        win_hit += int(ab[np.argmin(a)] == ab[np.argmin(p)])
        top3 += len(set(ab[np.argsort(a)[:3]]) & set(ab[np.argsort(p)[:3]])) / 3
        s = spearman(a, p); sp += 0 if np.isnan(s) else s; nr += 1
    nr = max(nr, 1)
    return mae, sp / nr, win_hit / nr, top3 / nr


def _prob_metrics(rec, col, outcome):
    d = rec.dropna(subset=[col, outcome])
    p, y = d[col].to_numpy(), d[outcome].to_numpy()
    return brier(p, y), logloss(p, y), ece(p, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2024)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--sims", type=int, default=3000)
    args = ap.parse_args()

    print("=" * 64 + "\n  RECONCILIATION TEST — option 3 (MC) vs option 2 (separate)\n" + "=" * 64)
    rec, coh2, coh3, cohr, nr = run(args.start_year, args.stride, args.sims)
    n_rows = len(rec)
    print(f"\n  {nr} races, {n_rows} driver-rows\n" + "=" * 64)

    print("\nORDER (predicting finishing position):")
    print(f"  {'model':14s} {'MAE':>7s} {'Spearman':>9s} {'Winner':>7s} {'Top3':>7s}")
    for name, col in [("opt2 tree", "tree_pos"), ("opt2 recon", "recon_pos"), ("opt3 mc", "mc_pos")]:
        mae, sp, wn, t3 = _order_metrics(rec, col)
        print(f"  {name:14s} {mae:7.3f} {sp:9.3f} {wn*100:6.0f}% {t3:7.3f}")

    for kind, oc in [("WIN", "win"), ("PODIUM", "pod")]:
        print(f"\n{kind} probability:")
        print(f"  {'model':14s} {'Brier':>8s} {'LogLoss':>8s} {'ECE':>7s}")
        suffix = "win" if kind == "WIN" else "pod"
        for name, col in [("opt2 blend", f"blend_{suffix}"), ("opt3 mc", f"mc_{suffix}")]:
            b, ll, e = _prob_metrics(rec, col, oc)
            print(f"  {name:14s} {b:8.4f} {ll:8.4f} {e:7.4f}")

    print(f"\nCOHERENCE (headline P1 == highest win-prob driver):")
    print(f"  opt2 tree: {coh2*100:5.1f}%   opt2 recon: {cohr*100:5.1f}%   opt3 mc: {coh3*100:5.1f}%")
    print("\n  Decision: 'opt2 recon' should hit ~100% coherence. Keep it if its order "
          "\n  metrics match or beat 'opt2 tree' (esp. Winner/Top3) — then it's coherent "
          "\n  AND no worse. Probabilities themselves are unchanged (the blend stays).")


if __name__ == "__main__":
    main()
