import os
import json
from datetime import datetime, timezone
import pandas as pd
import requests
from dateutil import parser

from .config import LEAGUES, FOOTBALL_DATA_BASE

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
META_DIR = "data/meta"

def ensure_dirs():
    for d in [RAW_DIR, PROCESSED_DIR, META_DIR]:
        os.makedirs(d, exist_ok=True)

def season_label(season_code: str) -> str:
    y1 = int("20" + season_code[:2])
    y2 = int("20" + season_code[2:])
    return f"{y1}-{y2}"

def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(r.text))

def normalize_date(x) -> str:
    if pd.isna(x):
        return ""
    dt = parser.parse(str(x), dayfirst=True)
    return dt.date().isoformat()

def build_match_id(league: str, season: str, match_date: str, home: str, away: str) -> str:
    def slug(s: str) -> str:
        return "".join([c for c in s.upper().replace(" ", "_") if c.isalnum() or c == "_"])
    return f"{league}_{season.replace('-', '')}_{match_date}_{slug(home)[:20]}_{slug(away)[:20]}"

def main():
    ensure_dirs()
    snapshot = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    all_rows = []
    failures = []

    for lg in LEAGUES:
        print(f"\n=== League {lg.code} (fd_code={lg.fd_code}) ===")
        for sc in lg.seasons:
            url = f"{FOOTBALL_DATA_BASE}/{sc}/{lg.fd_code}.csv"
            print(f"Fetching: {url}")

            try:
                df = fetch_csv(url)
            except Exception as e:
                failures.append({"league": lg.code, "season_code": sc, "url": url, "error": str(e)})
                print(f"  !! FAIL: {lg.code} {sc} -> {e}")
                continue

            print(f"  OK: rows={df.shape[0]} cols={df.shape[1]}")

            raw_path = os.path.join(RAW_DIR, f"{lg.code}_{sc}.csv")
            df.to_csv(raw_path, index=False)

            wanted = {
                "Date": "match_date",
                "Time": "kickoff_time",
                "HomeTeam": "home_team",
                "AwayTeam": "away_team",
                "FTHG": "fthg",
                "FTAG": "ftag",
                "FTR": "ftr",
                "HTHG": "hthg",
                "HTAG": "htag",
                "HTR": "htr",
                "Attendance": "attendance",
                "Referee": "referee",
                "HS": "hs",
                "AS": "as",
                "HST": "hst",
                "AST": "ast",
                "HF": "hf",
                "AF": "af",
                "HC": "hc",
                "AC": "ac",
                "HY": "hy",
                "AY": "ay",
                "HR": "hr",
                "AR": "ar",
                "AvgH": "avg_h",
                "AvgD": "avg_d",
                "AvgA": "avg_a",
                "MaxH": "max_h",
                "MaxD": "max_d",
                "MaxA": "max_a",
                "Avg>2.5": "avg_over25",
                "Avg<2.5": "avg_under25",
                "Max>2.5": "max_over25",
                "Max<2.5": "max_under25",
                "AvgCH": "closing_avg_h",
                "AvgCD": "closing_avg_d",
                "AvgCA": "closing_avg_a",
            }

            out = pd.DataFrame()
            for src, dst in wanted.items():
                out[dst] = df[src] if src in df.columns else pd.NA

            out["league"] = lg.code
            out["season"] = season_label(sc)
            out["data_snapshot_utc"] = snapshot

            out["match_date"] = out["match_date"].apply(normalize_date)

            out["match_id"] = out.apply(
                lambda r: build_match_id(
                    r["league"], r["season"], r["match_date"], str(r["home_team"]), str(r["away_team"])
                ),
                axis=1
            )

            cols = ["match_id","league","season","match_date","kickoff_time","home_team","away_team",
                    "fthg","ftag","ftr","hthg","htag","htr","attendance","referee",
                    "hs","as","hst","ast","hf","af","hc","ac","hy","ay","hr","ar",
                    "avg_h","avg_d","avg_a","max_h","max_d","max_a",
                    "avg_over25","avg_under25","max_over25","max_under25",
                    "closing_avg_h","closing_avg_d","closing_avg_a",
                    "data_snapshot_utc"]
            out = out[cols]

            all_rows.append(out)

    if not all_rows:
        print("ERROR: No se pudo descargar ninguna liga/temporada. Revisa failures.")
        if failures:
            print("Failures:")
            for f in failures:
                print(f)
        raise SystemExit(1)

    match_level = pd.concat(all_rows, ignore_index=True)
    match_level = match_level.drop_duplicates(subset=["match_id"], keep="last")

    out_path = os.path.join(PROCESSED_DIR, "match_level.csv")
    match_level.to_csv(out_path, index=False)

    meta = {"last_update_utc": snapshot, "rows": int(match_level.shape[0]), "failures": failures}
    with open(os.path.join(META_DIR, "update_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nOK: wrote {out_path} with {match_level.shape[0]} rows. snapshot={snapshot}")
    if failures:
        print(f"WARNING: hubo {len(failures)} fallos. Revisa data/meta/update_meta.json")

if __name__ == "__main__":
    main()
