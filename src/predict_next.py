import os
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from math import exp, factorial

PROCESSED_DIR = "data/processed"
META_DIR = "data/meta"

def poisson_pmf(k, lam):
    return (lam ** k) * exp(-lam) / factorial(k)

def outcome_probs(lam_h, lam_a, max_goals=10):
    ph = np.zeros((max_goals + 1, max_goals + 1))
    pH = [poisson_pmf(i, lam_h) for i in range(max_goals + 1)]
    pA = [poisson_pmf(j, lam_a) for j in range(max_goals + 1)]
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            ph[i, j] = pH[i] * pA[j]

    p_home = float(np.sum(np.tril(ph, -1)))
    p_draw = float(np.sum(np.diag(ph)))
    p_away = float(np.sum(np.triu(ph, 1)))

    p_over25 = float(np.sum([ph[i, j] for i in range(max_goals + 1) for j in range(max_goals + 1) if (i + j) >= 3]))
    p_under25 = 1.0 - p_over25
    p_btts = float(np.sum([ph[i, j] for i in range(1, max_goals + 1) for j in range(1, max_goals + 1)]))

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_over25": p_over25,
        "p_under25": p_under25,
        "p_btts": p_btts,
    }

def load_model(league: str):
    path = os.path.join(META_DIR, f"model_{league}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe {path}. Ejecuta: python -m src.train_model")
    return json.loads(open(path, "r", encoding="utf-8").read())

def predict_row(model, league, season, match_id, match_date, home_team, away_team):
    home_adv = model["home_adv"]
    atk = model["attack"]
    dfn = model["defense"]

    a_h = float(atk.get(home_team, 0.0))
    d_h = float(dfn.get(home_team, 0.0))
    a_a = float(atk.get(away_team, 0.0))
    d_a = float(dfn.get(away_team, 0.0))

    log_lam_h = home_adv + a_h - d_a
    log_lam_a = a_a - d_h

    lam_h = float(np.exp(log_lam_h))
    lam_a = float(np.exp(log_lam_a))

    probs = outcome_probs(lam_h, lam_a, max_goals=10)

    return {
        "match_id": match_id,
        "league": league,
        "season": season,
        "match_date": match_date,
        "home_team": home_team,
        "away_team": away_team,
        "lambda_home": lam_h,
        "lambda_away": lam_a,
        **probs
    }

def main():
    in_path = os.path.join(PROCESSED_DIR, "match_level.csv")
    if not os.path.exists(in_path):
        raise SystemExit("No existe data/processed/match_level.csv. Ejecuta: python -m src.update_data")

    df = pd.read_csv(in_path)
    df["fthg"] = pd.to_numeric(df["fthg"], errors="coerce")
    df["ftag"] = pd.to_numeric(df["ftag"], errors="coerce")

    pending = df[df["fthg"].isna() | df["ftag"].isna()].copy()
    if pending.empty:
        print("No hay partidos pendientes en el dataset. (Nada que predecir)")
        return

    snapshot = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    preds = []
    for league, g in pending.groupby("league"):
        model = load_model(league)
        for _, r in g.iterrows():
            preds.append(
                predict_row(
                    model=model,
                    league=str(r["league"]),
                    season=str(r["season"]),
                    match_id=str(r["match_id"]),
                    match_date=str(r["match_date"]),
                    home_team=str(r["home_team"]),
                    away_team=str(r["away_team"]),
                )
            )

    out = pd.DataFrame(preds)
    out["data_snapshot_utc"] = snapshot

    out_path = os.path.join(PROCESSED_DIR, "predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"OK: wrote {out_path} with {out.shape[0]} predictions. snapshot={snapshot}")

if __name__ == "__main__":
    main()
