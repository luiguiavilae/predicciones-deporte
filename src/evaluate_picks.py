import argparse
import pandas as pd
from pathlib import Path


# -----------------------------
# Normalización
# -----------------------------

def normalize_market(market: str) -> str:
    m = str(market).strip().upper()
    if m in ["1X2", "1X", "FULLTIME_1X2"]:
        return "1X2"
    if m in ["OU25", "OVER_UNDER", "O/U", "OU", "O/U25"]:
        return "OU25"
    if m in ["BTTS", "BOTH_TEAMS_TO_SCORE"]:
        return "BTTS"
    return m


def normalize_pick(market: str, pick: str) -> str:
    p = str(pick).strip().upper()

    if market == "1X2":
        if p in ["H", "HOME", "1"]:
            return "1"
        if p in ["D", "DRAW", "X"]:
            return "X"
        if p in ["A", "AWAY", "2"]:
            return "2"
        return p

    if market == "OU25":
        if p in ["OVER", "O", "MAS", "MÁS"]:
            return "OVER"
        if p in ["UNDER", "U", "MENOS"]:
            return "UNDER"
        return p

    if market == "BTTS":
        if p in ["YES", "Y", "SI", "SÍ"]:
            return "YES"
        if p in ["NO", "N"]:
            return "NO"
        return p

    return p


# -----------------------------
# Confidence Score (0..100)
# -----------------------------

def top2(values):
    probs = sorted([float(x) for x in values if x is not None], reverse=True)
    best = probs[0] if len(probs) >= 1 else 0.0
    second = probs[1] if len(probs) >= 2 else 0.0
    return best, second


def coherence_1x2(pick, lam_h, lam_a):
    diff = lam_h - lam_a
    if pick == "1":
        if diff >= 0.20: return 100.0
        if diff >= 0.05: return 50.0
        return 0.0
    if pick == "2":
        if diff <= -0.20: return 100.0
        if diff <= -0.05: return 50.0
        return 0.0
    # X
    if abs(diff) <= 0.10: return 100.0
    if abs(diff) <= 0.20: return 50.0
    return 0.0


def coherence_ou25(pick, lam_total):
    if pick == "OVER":
        if lam_total >= 2.7: return 100.0
        if lam_total >= 2.5: return 50.0
        return 0.0
    if pick == "UNDER":
        if lam_total <= 2.3: return 100.0
        if lam_total <= 2.5: return 50.0
        return 0.0
    return 0.0


def coherence_btts(pick, lam_h, lam_a):
    if pick == "YES":
        if lam_h >= 0.95 and lam_a >= 0.95: return 100.0
        if lam_h >= 0.75 and lam_a >= 0.75: return 50.0
        return 0.0
    if pick == "NO":
        if lam_h <= 0.70 or lam_a <= 0.70: return 100.0
        if lam_h <= 0.85 or lam_a <= 0.85: return 50.0
        return 0.0
    return 0.0


def compute_confidence(market, pick, row):
    """
    Score 0..100

    PESOS (ajustados para Poisson):
      - prob_max: 50%
      - diferencial: 20%
      - coherencia con lambdas: 30%
    """
    lam_h = float(row.get("lambda_home", 0.0) or 0.0)
    lam_a = float(row.get("lambda_away", 0.0) or 0.0)
    lam_total = lam_h + lam_a

    if market == "1X2":
        ph = float(row.get("p_home", 0.0) or 0.0)
        pd_ = float(row.get("p_draw", 0.0) or 0.0)
        pa = float(row.get("p_away", 0.0) or 0.0)
        best, second = top2([ph, pd_, pa])
        coh = coherence_1x2(pick, lam_h, lam_a)

    elif market == "OU25":
        pov = float(row.get("p_over", 0.0) or 0.0)
        pun = float(row.get("p_under", 0.0) or 0.0)
        best, second = top2([pov, pun])
        coh = coherence_ou25(pick, lam_total)

    elif market == "BTTS":
        pys = float(row.get("p_btts_yes", 0.0) or 0.0)
        pno = float(row.get("p_btts_no", 0.0) or 0.0)
        best, second = top2([pys, pno])
        coh = coherence_btts(pick, lam_h, lam_a)

    else:
        best, second, coh = 0.0, 0.0, 0.0

    prob_max = best * 100.0
    diff = (best - second) * 100.0

    # NUEVOS PESOS
    score = (0.50 * prob_max) + (0.20 * diff) + (0.30 * coh)

    if score < 0: score = 0.0
    if score > 100: score = 100.0
    return float(score)


# -----------------------------
# Resultado y ROI
# -----------------------------

def infer_result(market, pick, hg, ag):
    if market == "1X2":
        actual = "1" if hg > ag else ("2" if hg < ag else "X")
        return "WIN" if pick == actual else "LOSS"

    if market == "OU25":
        total = hg + ag
        actual = "OVER" if total >= 3 else "UNDER"
        return "WIN" if pick == actual else "LOSS"

    if market == "BTTS":
        actual = "YES" if (hg > 0 and ag > 0) else "NO"
        return "WIN" if pick == actual else "LOSS"

    raise ValueError(f"Market desconocido: {market}")


def profit(odds, stake, result):
    if result == "WIN":
        return stake * (odds - 1.0)
    if result == "LOSS":
        return -stake
    return 0.0


def evaluate_ledger(df, threshold, stake):
    df = df.copy()

    # Normaliza market/pick
    df["market"] = df["market"].apply(normalize_market)
    df["pick"] = df.apply(lambda r: normalize_pick(r["market"], r["pick"]), axis=1)

    # Probabilidades (si faltan, se crean)
    prob_cols = ["p_home","p_draw","p_away","p_over","p_under","p_btts_yes","p_btts_no"]
    for c in prob_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Lambdas
    if "lambda_home" not in df.columns: df["lambda_home"] = 0.0
    if "lambda_away" not in df.columns: df["lambda_away"] = 0.0
    df["lambda_home"] = pd.to_numeric(df["lambda_home"], errors="coerce").fillna(0.0)
    df["lambda_away"] = pd.to_numeric(df["lambda_away"], errors="coerce").fillna(0.0)

    # Odds
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")

    # Confidence (autocalculado)
    df["confidence"] = df.apply(lambda r: compute_confidence(r["market"], r["pick"], r), axis=1)

    # Result / Profit
    results = []
    profits = []
    for _, row in df.iterrows():
        if pd.isna(row.get("actual_home_goals")) or pd.isna(row.get("actual_away_goals")):
            results.append("PENDING")
            profits.append(0.0)
            continue

        hg = int(row["actual_home_goals"])
        ag = int(row["actual_away_goals"])

        mkt = row["market"]
        if mkt not in ["1X2", "OU25", "BTTS"]:
            raise ValueError(f"Market desconocido: {mkt}. Usa 1X2, OU25 o BTTS.")

        res = infer_result(mkt, row["pick"], hg, ag)
        results.append(res)
        profits.append(profit(float(row["odds"]), stake, res))

    df["result"] = results
    df["profit"] = profits

    # Debug (clave para no quedar ciego)
    pending = int((df["result"] == "PENDING").sum())
    max_conf = float(df["confidence"].max()) if len(df) else 0.0
    passed = int(((df["confidence"] >= threshold) & (df["result"] != "PENDING")).sum())

    print("\n=== DEBUG ===")
    print(f"Rows total: {len(df)} | PENDING: {pending} | Max confidence: {max_conf:.2f} | Passed(threshold & resolved): {passed}")

    # Apuestas válidas
    bet_df = df[(df["confidence"] >= threshold) & (df["result"] != "PENDING")].copy()
    if bet_df.empty:
        summary = pd.DataFrame([{
            "bets": 0, "wins": 0, "hit_rate": 0.0, "profit": 0.0, "roi": 0.0,
            "threshold": threshold, "stake": stake
        }])
        return df, summary, pd.DataFrame(), pd.DataFrame()

    bet_df["win"] = bet_df["result"] == "WIN"

    total_bets = int(len(bet_df))
    wins = int(bet_df["win"].sum())
    total_profit = float(bet_df["profit"].sum())
    total_staked = float(stake * total_bets)
    roi = (total_profit / total_staked) if total_staked else 0.0

    summary = pd.DataFrame([{
        "bets": total_bets,
        "wins": wins,
        "hit_rate": round(wins / total_bets, 3),
        "profit": round(total_profit, 3),
        "roi": round(roi, 3),
        "threshold": threshold,
        "stake": stake
    }])

    by_day = (
        bet_df.groupby("date", dropna=False)
        .agg(bets=("result","count"), wins=("win","sum"), profit=("profit","sum"))
        .reset_index()
    )
    by_day["hit_rate"] = (by_day["wins"] / by_day["bets"]).round(3)
    by_day["roi"] = (by_day["profit"] / (stake * by_day["bets"])).round(3)

    by_league = (
        bet_df.groupby("league", dropna=False)
        .agg(bets=("result","count"), wins=("win","sum"), profit=("profit","sum"))
        .reset_index()
    )
    by_league["hit_rate"] = (by_league["wins"] / by_league["bets"]).round(3)
    by_league["roi"] = (by_league["profit"] / (stake * by_league["bets"])).round(3)

    return df, summary, by_day, by_league


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate picks (confidence + ROI)")
    parser.add_argument("ledger", type=Path, help="CSV con picks (ej: data/ledger.csv)")
    parser.add_argument("--threshold", type=float, default=65.0, help="Confidence mínima para apostar")
    parser.add_argument("--stake", type=float, default=1.0, help="Stake por pick")

    args = parser.parse_args()

    df = pd.read_csv(args.ledger)

    required = ["date", "league", "home", "away", "market", "pick", "odds", "actual_home_goals", "actual_away_goals"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Faltan columnas en ledger.csv: {missing}")

    full_df, summary, by_day, by_league = evaluate_ledger(df, threshold=args.threshold, stake=args.stake)

    print("\n=== RESUMEN GLOBAL (confidence autocalculado) ===")
    print(summary.to_string(index=False))

    if not by_day.empty:
        print("\n=== POR DÍA ===")
        print(by_day.to_string(index=False))

    if not by_league.empty:
        print("\n=== POR LIGA ===")
        print(by_league.to_string(index=False))

    bet_df = full_df[(full_df["confidence"] >= args.threshold) & (full_df["result"] != "PENDING")].copy()
    if not bet_df.empty:
        cols = ["date","league","home","away","market","pick","odds","confidence","result","profit"]
        print("\n=== PICKS APOSTADOS ===")
        print(bet_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
