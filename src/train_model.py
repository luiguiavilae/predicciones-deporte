import os
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

PROCESSED_DIR = "data/processed"
META_DIR = "data/meta"

def poisson_nll(y, log_lambda):
    lam = np.exp(log_lambda)
    return lam - y * log_lambda + gammaln(y + 1)

def fit_poisson_attack_defense(matches: pd.DataFrame):
    teams = sorted(pd.unique(pd.concat([matches["home_team"], matches["away_team"]]).dropna()))
    if len(teams) < 2:
        raise ValueError("No hay suficientes equipos para entrenar.")

    team_to_idx = {t: i for i, t in enumerate(teams)}
    ref_team = teams[-1]

    n = len(teams)
    p = 1 + (n - 1) + (n - 1)

    h_idx = matches["home_team"].map(team_to_idx).to_numpy()
    a_idx = matches["away_team"].map(team_to_idx).to_numpy()
    y_h = matches["fthg"].to_numpy(dtype=int)
    y_a = matches["ftag"].to_numpy(dtype=int)

    def unpack(theta):
        home_adv = theta[0]
        atk = np.zeros(n)
        dfn = np.zeros(n)
        atk[: n - 1] = theta[1 : 1 + (n - 1)]
        dfn[: n - 1] = theta[1 + (n - 1) : ]
        return home_adv, atk, dfn

    def nll(theta):
        home_adv, atk, dfn = unpack(theta)
        log_lam_h = home_adv + atk[h_idx] - dfn[a_idx]
        log_lam_a = atk[a_idx] - dfn[h_idx]
        return float(np.sum(poisson_nll(y_h, log_lam_h)) + np.sum(poisson_nll(y_a, log_lam_a)))

    def nll_reg(theta, lam=0.01):
        return nll(theta) + lam * float(np.sum(theta[1:] ** 2))

    x0 = np.zeros(p)
    x0[0] = 0.2

    res = minimize(fun=nll_reg, x0=x0, method="L-BFGS-B", options={"maxiter": 500})
    if not res.success:
        raise RuntimeError(f"Optimización falló: {res.message}")

    home_adv, atk, dfn = unpack(res.x)

    model = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ref_team": ref_team,
        "home_adv": float(home_adv),
        "teams": teams,
        "attack": {t: float(atk[team_to_idx[t]]) for t in teams},
        "defense": {t: float(dfn[team_to_idx[t]]) for t in teams},
        "n_matches": int(matches.shape[0]),
    }
    return model

def main():
    os.makedirs(META_DIR, exist_ok=True)
    in_path = os.path.join(PROCESSED_DIR, "match_level.csv")
    if not os.path.exists(in_path):
        raise SystemExit("No existe data/processed/match_level.csv. Ejecuta: python -m src.update_data")

    df = pd.read_csv(in_path)
    df["fthg"] = pd.to_numeric(df["fthg"], errors="coerce")
    df["ftag"] = pd.to_numeric(df["ftag"], errors="coerce")
    played = df.dropna(subset=["fthg", "ftag"]).copy()
    played["fthg"] = played["fthg"].astype(int)
    played["ftag"] = played["ftag"].astype(int)

    for league, g in played.groupby("league"):
        g = g[["home_team", "away_team", "fthg", "ftag"]].dropna()
        model = fit_poisson_attack_defense(g)
        out_path = os.path.join(META_DIR, f"model_{league}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(model, f, ensure_ascii=False, indent=2)
        print(f"OK: trained model for {league} with {model['n_matches']} matches -> {out_path}")

if __name__ == "__main__":
    main()
