"""
Predice una jornada completa.

Uso:
  python -m src.predict_jornada LL "Getafe vs Valencia, Atletico vs Alaves, Celta vs Rayo"
  python -m src.predict_jornada EPL --file fixtures.txt

fixtures.txt (un partido por línea):
  Arsenal vs Chelsea
  Liverpool vs Man City
"""
import argparse
import sys

from src.predict_match import run_prediction, pct


def parse_fixtures(text: str):
    """Parsea 'Home vs Away, Home2 vs Away2' o líneas separadas."""
    fixtures = []
    for part in text.replace("\n", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if " vs " in part.lower():
            idx = part.lower().index(" vs ")
            home = part[:idx].strip()
            away = part[idx + 4:].strip()
            fixtures.append((home, away))
        elif " - " in part:
            home, away = part.split(" - ", 1)
            fixtures.append((home.strip(), away.strip()))
        else:
            print(f"WARN: no se pudo parsear '{part}'. Usa formato 'Home vs Away'.", file=sys.stderr)
    return fixtures


def main():
    parser = argparse.ArgumentParser(description="Predecir una jornada completa")
    parser.add_argument("league", help="Código de liga: EPL, LL, SA, UCL")
    parser.add_argument("fixtures", nargs="?", default="",
                        help='Partidos: "Home1 vs Away1, Home2 vs Away2"')
    parser.add_argument("--file", type=str, default=None,
                        help="Archivo con un partido por línea (Home vs Away)")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.fixtures:
        text = args.fixtures
    else:
        print("Error: proporciona fixtures como argumento o con --file", file=sys.stderr)
        sys.exit(1)

    fixtures = parse_fixtures(text)
    if not fixtures:
        print("No se encontraron partidos válidos.", file=sys.stderr)
        sys.exit(1)

    league = args.league.strip().upper()
    print(f"\n=== Predicciones · {league} ({len(fixtures)} partidos) ===\n")

    for home, away in fixtures:
        print(f"--- {home} vs {away} ---")
        try:
            out = run_prediction(league, home, away)
            print(f"  λ: local={out['lambda_home']:.2f} | visita={out['lambda_away']:.2f}")
            print(f"  1X2: Local {pct(out['p_home'])} | Empate {pct(out['p_draw'])} | Visita {pct(out['p_away'])}")
            print(f"  O/U 2.5: Over {pct(out['p_over'])} | Under {pct(out['p_under'])}")
            print(f"  BTTS: Sí {pct(out['p_btts_yes'])} | No {pct(out['p_btts_no'])}")
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
        print()


if __name__ == "__main__":
    main()
