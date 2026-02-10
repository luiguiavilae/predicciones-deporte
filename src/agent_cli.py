"""
CLI conversacional para el agente de predicción de fútbol.

Uso interactivo:
  python -m src.agent_cli

Uso directo:
  python -m src.agent_cli "predice Barcelona vs Real Madrid en LaLiga"
  python -m src.agent_cli "equipos de la Premier"
  python -m src.agent_cli "status"
"""
from __future__ import annotations

import os
import json
import sys
import re
from difflib import get_close_matches

import pandas as pd

from src.predict_match import run_prediction, pct
from src.config import LEAGUES

PROCESSED_DIR = "data/processed"
META_DIR = "data/meta"

# ─── Alias de ligas: lenguaje natural → código interno ───────────────────────

LEAGUE_ALIASES = {
    # LaLiga
    "ll": "LL", "laliga": "LL", "la liga": "LL", "liga": "LL",
    "liga española": "LL", "liga espanola": "LL", "primera": "LL",
    "primera division": "LL", "spain": "LL", "españa": "LL", "espana": "LL",
    # Premier League
    "epl": "EPL", "premier": "EPL", "premier league": "EPL",
    "premiership": "EPL", "england": "EPL", "inglaterra": "EPL",
    # Serie A
    "sa": "SA", "serie a": "SA", "seriea": "SA", "calcio": "SA",
    "italy": "SA", "italia": "SA",
    # Champions League
    "ucl": "UCL", "champions": "UCL", "champions league": "UCL",
    "champions liga": "UCL", "cl": "UCL",
}

# Alias de equipos: nombre coloquial → nombre en football-data.co.uk
TEAM_ALIASES = {
    # LaLiga
    "barça": "Barcelona", "barca": "Barcelona", "fc barcelona": "Barcelona",
    "real": "Real Madrid", "madrid": "Real Madrid", "real madrid cf": "Real Madrid",
    "atletico": "Ath Madrid", "atleti": "Ath Madrid", "atletico madrid": "Ath Madrid",
    "atletico de madrid": "Ath Madrid", "at madrid": "Ath Madrid", "atm": "Ath Madrid",
    "athletic": "Ath Bilbao", "athletic bilbao": "Ath Bilbao",
    "athletic club": "Ath Bilbao", "athletic de bilbao": "Ath Bilbao",
    "real sociedad": "Sociedad", "la real": "Sociedad",
    "betis": "Betis", "real betis": "Betis",
    "rayo": "Vallecano", "rayo vallecano": "Vallecano",
    "espanyol": "Espanol", "rcd espanyol": "Espanol",
    "las palmas": "Las Palmas", "ud las palmas": "Las Palmas",
    "real valladolid": "Valladolid",
    "real oviedo": "Oviedo",
    "cd leganes": "Leganes", "leganés": "Leganes",
    # Premier League
    "city": "Man City", "manchester city": "Man City",
    "united": "Man United", "manchester united": "Man United", "man utd": "Man United",
    "liverpool fc": "Liverpool",
    "chelsea fc": "Chelsea",
    "arsenal fc": "Arsenal",
    "spurs": "Tottenham", "tottenham hotspur": "Tottenham",
    "villa": "Aston Villa",
    "palace": "Crystal Palace",
    "forest": "Nott'm Forest", "nottingham forest": "Nott'm Forest",
    "nottingham": "Nott'm Forest",
    "wolves": "Wolves", "wolverhampton": "Wolves",
    "newcastle united": "Newcastle",
    "brighton and hove": "Brighton", "brighton & hove": "Brighton",
    "west ham": "West Ham", "west ham united": "West Ham",
    # Serie A
    "juve": "Juventus", "inter": "Inter", "inter milan": "Inter",
    "ac milan": "Milan", "milan": "Milan",
    "napoli": "Napoli", "ssc napoli": "Napoli",
    "roma": "Roma", "as roma": "Roma",
    "lazio": "Lazio", "ss lazio": "Lazio",
    "atalanta": "Atalanta",
    "fiorentina": "Fiorentina",
}


# ─── Utilidades ──────────────────────────────────────────────────────────────

def load_meta():
    path = os.path.join(META_DIR, "update_meta.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_model_teams(league: str) -> list[str]:
    path = os.path.join(META_DIR, f"model_{league}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)
    return model.get("teams", [])


def all_league_codes() -> list[str]:
    return [lg.code for lg in LEAGUES]


def resolve_league(text: str) -> str | None:
    """Intenta resolver un string a un código de liga."""
    t = text.strip().lower()
    # Directo
    if t.upper() in all_league_codes():
        return t.upper()
    # Alias
    if t in LEAGUE_ALIASES:
        return LEAGUE_ALIASES[t]
    return None


def resolve_team(name: str, league: str) -> str:
    """Resuelve nombre coloquial → nombre del modelo. Fuzzy si no hay match exacto."""
    stripped = name.strip()

    # Alias directo
    low = stripped.lower()
    if low in TEAM_ALIASES:
        return TEAM_ALIASES[low]

    known = load_model_teams(league)
    if not known:
        return stripped

    # Match exacto (case-insensitive)
    for t in known:
        if t.lower() == low:
            return t

    # Fuzzy match
    matches = get_close_matches(stripped, known, n=1, cutoff=0.5)
    if matches:
        return matches[0]

    return stripped


# ─── Parsers de intents ─────────────────────────────────────────────────────

# Patrones para detectar "equipo1 vs equipo2"
VS_PATTERN = re.compile(
    r"(.+?)\s+(?:vs\.?|versus|contra|v)\s+(.+)",
    re.IGNORECASE,
)

# Patrones para detectar liga en la frase
LEAGUE_IN_PHRASE = re.compile(
    r"(?:en\s+(?:la\s+)?|de\s+(?:la\s+)?|liga\s*:?\s*)"
    r"(laliga|la\s+liga|premier(?:\s+league)?|serie\s+a|champions(?:\s+league)?|"
    r"epl|ll|sa|ucl|liga\s+española|liga\s+espanola|calcio|primera(?:\s+division)?|"
    r"inglaterra|españa|espana|italy|italia)",
    re.IGNORECASE,
)


def detect_predict_intent(text: str):
    """
    Intenta extraer (liga, home, away) de una frase en lenguaje natural.
    Retorna (league, home, away) o None.
    """
    t = text.strip()

    # Buscar liga en la frase
    league_code = None
    league_match = LEAGUE_IN_PHRASE.search(t)
    if league_match:
        league_code = resolve_league(league_match.group(1))
        # Eliminar la parte de liga del texto para parsear equipos
        t = t[:league_match.start()] + t[league_match.end():]

    # Limpiar prefijos comunes
    t = re.sub(
        r"^(predice|predecir|predicción|prediccion|pronostica|pronóstico|pronostico|"
        r"analiza|análisis|analisis|cómo ves|como ves|qué opinas de|que opinas de|"
        r"dame|dime|muéstrame|muestrame|quiero)\s+",
        "", t, flags=re.IGNORECASE,
    ).strip()

    # Limpiar sufijos tipo "este fin de semana", "hoy", "mañana"
    t = re.sub(
        r"\s+(este|esta|el|la|del|de)\s+(fin de semana|finde|sábado|sabado|domingo|"
        r"lunes|martes|miércoles|miercoles|jueves|viernes|semana|jornada\s*\d*)\s*$",
        "", t, flags=re.IGNORECASE,
    ).strip()
    t = re.sub(r"\s+(hoy|mañana|manana)\s*[?]?\s*$", "", t, flags=re.IGNORECASE).strip()
    t = t.rstrip("?").strip()

    # Buscar patrón "Home vs Away"
    vs = VS_PATTERN.search(t)
    if not vs:
        return None

    raw_home = vs.group(1).strip()
    raw_away = vs.group(2).strip()

    # Limpiar artículos y preposiciones sueltas
    _art = re.compile(r"^(el|al|del|la|los|las|partido\s+de[l]?\s*)\s+", re.IGNORECASE)
    raw_home = _art.sub("", raw_home).strip()
    raw_away = _art.sub("", raw_away).strip()
    # Limpiar sufijo "en ..." residual del away
    raw_away = re.sub(r"\s*(en\s+.*)$", "", raw_away, flags=re.IGNORECASE).strip()

    # Si no detectamos liga, intentar inferirla probando cada modelo
    if not league_code:
        league_code = infer_league(raw_home, raw_away)

    if not league_code:
        return None

    home = resolve_team(raw_home, league_code)
    away = resolve_team(raw_away, league_code)

    return league_code, home, away


def infer_league(home: str, away: str) -> str | None:
    """Intenta adivinar la liga buscando los equipos en todos los modelos."""
    low_h = home.lower()
    low_a = away.lower()

    # Primero ver si los alias mapean a una liga
    resolved_h = TEAM_ALIASES.get(low_h)
    resolved_a = TEAM_ALIASES.get(low_a)

    for league_code in all_league_codes():
        teams = load_model_teams(league_code)
        teams_low = {t.lower(): t for t in teams}

        h_found = (low_h in teams_low
                    or (resolved_h and resolved_h.lower() in teams_low)
                    or bool(get_close_matches(home, teams, n=1, cutoff=0.6)))
        a_found = (low_a in teams_low
                    or (resolved_a and resolved_a.lower() in teams_low)
                    or bool(get_close_matches(away, teams, n=1, cutoff=0.6)))

        if h_found and a_found:
            return league_code

    # Si solo uno coincide, es suficiente
    for league_code in all_league_codes():
        teams = load_model_teams(league_code)
        teams_low = {t.lower(): t for t in teams}
        h_found = low_h in teams_low or (resolved_h and resolved_h.lower() in teams_low)
        a_found = low_a in teams_low or (resolved_a and resolved_a.lower() in teams_low)
        if h_found or a_found:
            return league_code

    return None


def detect_teams_intent(text: str) -> str | None:
    """Detecta si el usuario pide listar equipos de una liga."""
    t = text.strip().lower()
    patterns = [
        r"equipos?\s+(?:de\s+(?:la\s+)?)?(.+)",
        r"(?:listar?|muestra|dame|dime|ver)\s+equipos?\s+(?:de\s+(?:la\s+)?)?(.+)",
        r"(?:que|qué|cuales|cuáles)\s+equipos?\s+(?:tiene|hay)\s+(?:en\s+(?:la\s+)?)?(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            return resolve_league(m.group(1).strip())
    return None


def detect_status_intent(text: str) -> bool:
    t = text.strip().lower()
    return any(kw in t for kw in ["status", "estado", "info", "información", "informacion"])


def detect_help_intent(text: str) -> bool:
    t = text.strip().lower()
    return any(kw in t for kw in ["ayuda", "help", "como funciona", "cómo funciona", "comandos", "que puedes hacer", "qué puedes hacer"])


def detect_leagues_intent(text: str) -> bool:
    t = text.strip().lower()
    return any(kw in t for kw in ["ligas", "leagues", "que ligas", "qué ligas"])


# ─── Acciones ────────────────────────────────────────────────────────────────

def action_predict(league: str, home: str, away: str):
    try:
        out = run_prediction(league, home, away)
    except SystemExit as e:
        print(f"Error: {e}")
        return

    lam_h = out["lambda_home"]
    lam_a = out["lambda_away"]

    print(f"\n{'='*50}")
    print(f"  {out['league']}: {out['home']} vs {out['away']}")
    print(f"{'='*50}")
    print(f"  Goles esperados: {out['home']} {lam_h:.2f} - {lam_a:.2f} {out['away']}")
    print()
    print(f"  1X2:    Local {pct(out['p_home'])}  |  Empate {pct(out['p_draw'])}  |  Visita {pct(out['p_away'])}")
    print(f"  O/U 2.5: Over {pct(out['p_over'])}  |  Under {pct(out['p_under'])}")
    print(f"  BTTS:    Sí {pct(out['p_btts_yes'])}  |  No {pct(out['p_btts_no'])}")

    # Pick recomendado
    picks = []
    probs_1x2 = {"Local": out["p_home"], "Empate": out["p_draw"], "Visita": out["p_away"]}
    best_1x2 = max(probs_1x2, key=probs_1x2.get)
    picks.append(f"1X2 -> {best_1x2} ({pct(probs_1x2[best_1x2])})")

    if out["p_over"] >= out["p_under"]:
        picks.append(f"O/U -> Over 2.5 ({pct(out['p_over'])})")
    else:
        picks.append(f"O/U -> Under 2.5 ({pct(out['p_under'])})")

    if out["p_btts_yes"] >= out["p_btts_no"]:
        picks.append(f"BTTS -> Sí ({pct(out['p_btts_yes'])})")
    else:
        picks.append(f"BTTS -> No ({pct(out['p_btts_no'])})")

    print(f"\n  Picks recomendados:")
    for p in picks:
        print(f"    -> {p}")
    print()


def action_status():
    match_path = os.path.join(PROCESSED_DIR, "match_level.csv")
    if not os.path.exists(match_path):
        print("No hay datos. Ejecuta: python -m src.update_data")
        return

    df = pd.read_csv(match_path)
    meta = load_meta()
    last_update = meta.get("last_update_utc", "desconocido")

    print("\n=== ESTADO DEL AGENTE ===")
    print(f"Última actualización: {last_update}")

    df["season"] = df["season"].astype(str)
    df["league"] = df["league"].astype(str)

    print("\nLigas y temporadas:")
    for lg, ss in df.groupby("league")["season"].unique().items():
        print(f"  {lg}: {', '.join(sorted(ss))}")

    df["fthg"] = pd.to_numeric(df.get("fthg"), errors="coerce")
    df["ftag"] = pd.to_numeric(df.get("ftag"), errors="coerce")
    played = int(df["fthg"].notna().sum())
    pending = int(df["fthg"].isna().sum())
    print(f"\nPartidos jugados: {played} | Pendientes: {pending}")

    # Modelos disponibles
    models = []
    for lg in all_league_codes():
        mp = os.path.join(META_DIR, f"model_{lg}.json")
        if os.path.exists(mp):
            models.append(lg)
    print(f"Modelos entrenados: {', '.join(models) if models else 'ninguno'}")
    print()


def action_teams(league: str):
    teams = load_model_teams(league)
    if not teams:
        print(f"No hay modelo entrenado para {league}. Ejecuta: python -m src.train_model")
        return
    print(f"\nEquipos en {league} ({len(teams)}):")
    for t in sorted(teams):
        print(f"  - {t}")
    print()


def action_leagues():
    print("\nLigas disponibles:")
    for lg in LEAGUES:
        teams = load_model_teams(lg.code)
        status = f"{len(teams)} equipos" if teams else "sin modelo"
        print(f"  {lg.code:5s} ({status})")
    print()


def action_help():
    print("""
=== Agente de Predicción de Fútbol ===

Puedes escribir en lenguaje natural. Ejemplos:

  Predicciones:
    "predice Barcelona vs Real Madrid"
    "cómo ves Arsenal contra Chelsea en la Premier"
    "pronóstico Napoli vs Inter en Serie A"
    "Barcelona vs Atletico en LaLiga"

  Información:
    "equipos de la Premier"
    "ligas"
    "status"

  Otros:
    "ayuda"
    "salir"

Nota: los nombres de equipo se resuelven automáticamente
      (ej: "Barça" → Barcelona, "Atleti" → Ath Madrid)
""")


# ─── Loop principal ──────────────────────────────────────────────────────────

def process_input(text: str) -> bool:
    """Procesa un input. Retorna False si hay que salir."""
    t = text.strip()
    if not t:
        return True

    if t.lower() in ("salir", "exit", "quit", "q"):
        print("Hasta luego.")
        return False

    # Help
    if detect_help_intent(t):
        action_help()
        return True

    # Status
    if detect_status_intent(t):
        action_status()
        return True

    # Ligas
    if detect_leagues_intent(t):
        action_leagues()
        return True

    # Equipos de liga
    league_for_teams = detect_teams_intent(t)
    if league_for_teams:
        action_teams(league_for_teams)
        return True

    # Predicción
    pred = detect_predict_intent(t)
    if pred:
        league, home, away = pred
        action_predict(league, home, away)
        return True

    # No entendido
    print(
        "No entendí tu petición. Prueba algo como:\n"
        '  "predice Barcelona vs Real Madrid en LaLiga"\n'
        '  "equipos de la Premier"\n'
        '  "ayuda"'
    )
    return True


def main():
    # Modo directo: python -m src.agent_cli "predice ..."
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        process_input(text)
        return

    # Modo interactivo
    print("=== Agente de Predicción de Fútbol ===")
    print('Escribe "ayuda" para ver comandos, "salir" para terminar.\n')

    while True:
        try:
            text = input(">> ")
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if not process_input(text):
            break


if __name__ == "__main__":
    main()
