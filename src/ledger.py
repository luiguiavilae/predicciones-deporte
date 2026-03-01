import os
import csv
from datetime import date
from typing import Dict, Any, List


LEDGER_PATH_DEFAULT = "data/ledger.csv"


LEDGER_COLUMNS = [
    "date",
    "league",
    "home",
    "away",
    "market",
    "pick",
    "odds",
    "lambda_home",
    "lambda_away",
    "p_home",
    "p_draw",
    "p_away",
    "p_over",
    "p_under",
    "p_btts_yes",
    "p_btts_no",
    "actual_home_goals",
    "actual_away_goals",
]


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def today_iso() -> str:
    return date.today().isoformat()


def append_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    """
    Agrega filas al CSV. Si no existe, crea el archivo con header.
    """
    ensure_parent_dir(path)
    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        if not file_exists:
            writer.writeheader()

        for r in rows:
            clean = {k: r.get(k, "") for k in LEDGER_COLUMNS}
            writer.writerow(clean)


def build_pick_rows(
    *,
    league: str,
    home: str,
    away: str,
    lambda_home: float,
    lambda_away: float,
    p_home: float,
    p_draw: float,
    p_away: float,
    p_over: float,
    p_under: float,
    p_btts_yes: float,
    p_btts_no: float,
    odds_default: float = 1.90,
    match_date: str = "",
) -> List[Dict[str, Any]]:
    """
    Crea 3 filas (1X2, OU25, BTTS). Cada fila tiene SOLO el set de probabilidades relevante.
    """
    if not match_date:
        match_date = today_iso()

    # picks: el de mayor probabilidad por mercado
    pick_1x2 = max(
        [("1", p_home), ("X", p_draw), ("2", p_away)],
        key=lambda x: x[1],
    )[0]

    pick_ou25 = "OVER" if p_over >= p_under else "UNDER"
    pick_btts = "YES" if p_btts_yes >= p_btts_no else "NO"

    base = {
        "date": match_date,
        "league": league,
        "home": home,
        "away": away,
        "odds": float(odds_default),
        "lambda_home": float(lambda_home),
        "lambda_away": float(lambda_away),
        "actual_home_goals": "",
        "actual_away_goals": "",
    }

    row_1x2 = dict(base)
    row_1x2.update({
        "market": "1X2",
        "pick": pick_1x2,
        "p_home": float(p_home),
        "p_draw": float(p_draw),
        "p_away": float(p_away),
        "p_over": "",
        "p_under": "",
        "p_btts_yes": "",
        "p_btts_no": "",
    })

    row_ou = dict(base)
    row_ou.update({
        "market": "OU25",
        "pick": pick_ou25,
        "p_home": "",
        "p_draw": "",
        "p_away": "",
        "p_over": float(p_over),
        "p_under": float(p_under),
        "p_btts_yes": "",
        "p_btts_no": "",
    })

    row_btts = dict(base)
    row_btts.update({
        "market": "BTTS",
        "pick": pick_btts,
        "p_home": "",
        "p_draw": "",
        "p_away": "",
        "p_over": "",
        "p_under": "",
        "p_btts_yes": float(p_btts_yes),
        "p_btts_no": float(p_btts_no),
    })

    return [row_1x2, row_ou, row_btts]
