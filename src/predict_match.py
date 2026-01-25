import os
import json
import numpy as np
from math import exp, factorial
import argparse

from src.ledger import build_pick_rows, append_rows, LEDGER_PATH_DEFAULT

META_DIR = "data/meta"


def poisson_pmf(k: int, lam: float) -> float:
    return (lam ** k) * exp(-lam) / factorial(k)


def outcome_probs(lam_h: float, lam_a: float, max_goals: int = 10):
    pH = [poisson_pmf(i, lam_h) for i in range(max_goals + 1)]
    pA = [poisson_pmf(j, lam_a) for j in range(max_goals + 1)]

    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    p_over25 = 0.0
    p_btts = 0.0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = pH[i] * pA[j]
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p

            if i + j >= 3:
                p_over25 += p

            if i >= 1 and j >= 1:
                p_btts += p

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_over25": p_over25,
        "p_under25": 1.0 - p_over25,
        "p_btts_yes": p_btts,
        "p_btts_no": 1.0 - p_btts,
    }


def run_prediction(league: str, home_team: str, away_team: str):
    """
    Devuelve lambdas + probabilidades. Esta función es la "verdad única".
    """
    model_path = os.path.join(META_DIR, f"model_{league}.json")
    if not os.path.exists(model_path):
        raise SystemExit(f"No existe {model_path}. Ejecuta: python -m src.train_model")

    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    home_adv = float(model["home_adv"])
    atk = model["attack"]
    dfn = model["defense"]

    # Si un equipo no existe en el modelo, cae a 0.0 (neutral) para no reventar
    a_h = float(atk.get(home_team, 0.0))
    d_h = float(dfn.get(home_team, 0.0))
    a_a = float(atk.get(away_team, 0.0))
    d_a = float(dfn.get(away_team, 0.0))

    lam_h = float(np.exp(home_adv + a_h - d_a))
    lam_a = float(np.exp(a_a - d_h))

    probs = outcome_probs(lam_h, lam_a, max_goals=10)

    return {
        "league": league,
        "home": home_team,
        "away": away_team,
        "lambda_home": lam_h,
        "lambda_away": lam_a,
        "p_home": probs["p_home"],
        "p_draw": probs["p_draw"],
        "p_away": probs["p_away"],
        "p_over": probs["p_over25"],
        "p_under": probs["p_under25"],
        "p_btts_yes": probs["p_btts_yes"],
        "p_btts_no": probs["p_btts_no"],
    }


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main():
    parser = argparse.ArgumentParser(
        description="Predicción Poisson (attack/defense) + opcional: escribir picks al ledger"
    )

    parser.add_argument("league", help="Ej: EPL, LL, SA, UCL")
    parser.add_argument("home_team", help='Ej: "Arsenal"')
    parser.add_argument("away_team", help='Ej: "Chelsea"')

    parser.add_argument(
        "--write-ledger",
        action="store_true",
        help="Guarda 3 picks (1X2/OU25/BTTS) en data/ledger.csv"
    )
    parser.add_argument(
        "--ledger-path",
        default=LEDGER_PATH_DEFAULT,
        help="Ruta del ledger CSV (default: data/ledger.csv)"
    )
    parser.add_argument(
        "--odds-default",
        type=float,
        default=1.90,
        help="Odds por defecto para simular (hasta conectar odds reales)"
    )
    parser.add_argument(
        "--date",
        default="",
        help="Fecha del partido YYYY-MM-DD (default: hoy)"
    )

    args = parser.parse_args()

    out = run_prediction(args.league.strip(), args.home_team.strip(), args.away_team.strip())

    league = out["league"]
    home = out["home"]
    away = out["away"]
    lam_h = out["lambda_home"]
    lam_a = out["lambda_away"]

    print(f"{league}: {home} vs {away}")
    print(f"λ goles esperados: local={lam_h:.2f} | visita={lam_a:.2f}")
    print(f"1X2: Local {pct(out['p_home'])} | Empate {pct(out['p_draw'])} | Visita {pct(out['p_away'])}")
    print(f"Over/Under 2.5: Over {pct(out['p_over'])} | Under {pct(out['p_under'])}")
    print(f"BTTS: Sí {pct(out['p_btts_yes'])} | No {pct(out['p_btts_no'])}")

    if args.write_ledger:
        rows = build_pick_rows(
            league=league,
            home=home,
            away=away,
            lambda_home=lam_h,
            lambda_away=lam_a,
            p_home=out["p_home"],
            p_draw=out["p_draw"],
            p_away=out["p_away"],
            p_over=out["p_over"],
            p_under=out["p_under"],
            p_btts_yes=out["p_btts_yes"],
            p_btts_no=out["p_btts_no"],
            odds_default=args.odds_default,
            match_date=args.date,
        )
        append_rows(args.ledger_path, rows)
        print(f"\nOK: ledger actualizado -> {args.ledger_path} (+3 filas)")


# Para uso futuro desde otros scripts
def predict_match(league: str, home: str, away: str):
    return run_prediction(league, home, away)


if __name__ == "__main__":
    main()
