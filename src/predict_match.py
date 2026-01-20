import os
import json
import sys
import numpy as np
from math import exp, factorial

META_DIR = "data/meta"

def poisson_pmf(k, lam):
    return (lam ** k) * exp(-lam) / factorial(k)

def outcome_probs(lam_h, lam_a, max_goals=10):
    pH = [poisson_pmf(i, lam_h) for i in range(max_goals + 1)]
    pA = [poisson_pmf(j, lam_a) for j in range(max_goals + 1)]
    p_home = p_draw = p_away = 0.0
    p_over25 = 0.0
    p_btts = 0.0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = pH[i] * pA[j]
            if i > j: p_home += p
            elif i == j: p_draw += p
            else: p_away += p
            if i + j >= 3: p_over25 += p
            if i >= 1 and j >= 1: p_btts += p

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_over25": p_over25,
        "p_under25": 1.0 - p_over25,
        "p_btts": p_btts,
    }

def main():
    if len(sys.argv) < 4:
        print('Uso: python -m src.predict_match <LEAGUE> "<HOME_TEAM>" "<AWAY_TEAM>"')
        print('Ej:  python -m src.predict_match EPL "Arsenal" "Chelsea"')
        sys.exit(1)

    league = sys.argv[1].strip()
    home_team = sys.argv[2].strip()
    away_team = sys.argv[3].strip()

    model_path = os.path.join(META_DIR, f"model_{league}.json")
    if not os.path.exists(model_path):
        raise SystemExit(f"No existe {model_path}. Ejecuta: python -m src.train_model")

    model = json.loads(open(model_path, "r", encoding="utf-8").read())

    home_adv = float(model["home_adv"])
    atk = model["attack"]
    dfn = model["defense"]

    a_h = float(atk.get(home_team, 0.0))
    d_h = float(dfn.get(home_team, 0.0))
    a_a = float(atk.get(away_team, 0.0))
    d_a = float(dfn.get(away_team, 0.0))

    lam_h = float(np.exp(home_adv + a_h - d_a))
    lam_a = float(np.exp(a_a - d_h))
    probs = outcome_probs(lam_h, lam_a, max_goals=10)

    def pct(x): return f"{100*x:.1f}%"
    print(f"{league}: {home_team} vs {away_team}")
    print(f"λ goles esperados: local={lam_h:.2f} | visita={lam_a:.2f}")
    print(f"1X2: Local {pct(probs['p_home'])} | Empate {pct(probs['p_draw'])} | Visita {pct(probs['p_away'])}")
    print(f"Over/Under 2.5: Over {pct(probs['p_over25'])} | Under {pct(probs['p_under25'])}")
    print(f"BTTS: Sí {pct(probs['p_btts'])} | No {pct(1-probs['p_btts'])}")
def predict_match(league: str, home: str, away: str):
    # aquí llama a la misma lógica que usa el CLI
    return run_prediction(league, home, away)  # o como se llame tu función interna
if __name__ == "__main__":
    main()