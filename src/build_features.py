import os
import json
from datetime import datetime, timezone
import pandas as pd

PROCESSED_DIR = "data/processed"
META_DIR = "data/meta"

K_ELO = 20
HOME_ADV = 60

def expected_score(r_home: float, r_away: float) -> float:
    return 1 / (1 + 10 ** ((r_away - r_home) / 400))

def result_to_score(ftr: str) -> float:
    if ftr == "H":
        return 1.0
    if ftr == "D":
        return 0.5
    return 0.0

def main():
    in_path = os.path.join(PROCESSED_DIR, "match_level.csv")
    if not os.path.exists(in_path):
        raise SystemExit("No existe data/processed/match_level.csv. Ejecuta update_data.py primero.")

    df = pd.read_csv(in_path)
    df = df.sort_values(["league","season","match_date"]).reset_index(drop=True)

    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df["fthg"] = pd.to_numeric(df["fthg"], errors="coerce")
    df["ftag"] = pd.to_numeric(df["ftag"], errors="coerce")
    df["ftr"] = df["ftr"].astype(str)

    features_rows = []
    snapshot = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    for (league, season), g in df.groupby(["league","season"], sort=False):
        g = g.sort_values("match_date").copy()

        teams = pd.unique(pd.concat([g["home_team"], g["away_team"]]).dropna())
        elo = {t: 1500.0 for t in teams}

        history = {t: [] for t in teams}
        goals_for = {t: [] for t in teams}
        goals_against = {t: [] for t in teams}
        last_date = {t: None for t in teams}

        for _, r in g.iterrows():
            home = r["home_team"]
            away = r["away_team"]
            date = r["match_date"]

            def avg_last(vals, n=5):
                vals = vals[-n:]
                return sum(vals) / len(vals) if len(vals) else None

            def sum_last(vals, n=5):
                return sum(vals[-n:]) if len(vals) else 0

            home_form_5 = sum_last(history[home], 5)
            away_form_5 = sum_last(history[away], 5)

            home_goals_avg_5 = avg_last(goals_for[home], 5)
            away_goals_avg_5 = avg_last(goals_for[away], 5)

            home_conc_avg_5 = avg_last(goals_against[home], 5)
            away_conc_avg_5 = avg_last(goals_against[away], 5)

            def rest_days(team):
                if last_date[team] is None or pd.isna(date):
                    return None
                return int((date - last_date[team]).days)

            rest_h = rest_days(home)
            rest_a = rest_days(away)

            elo_h = elo[home]
            elo_a = elo[away]

            features_rows.append({
                "match_id": r["match_id"],
                "league": league,
                "season": season,
                "match_date": date.date().isoformat() if pd.notna(date) else "",
                "home_team": home,
                "away_team": away,
                "home_form_5": home_form_5,
                "away_form_5": away_form_5,
                "home_goals_avg_5": home_goals_avg_5,
                "away_goals_avg_5": away_goals_avg_5,
                "home_conceded_avg_5": home_conc_avg_5,
                "away_conceded_avg_5": away_conc_avg_5,
                "home_elo": int(round(elo_h)),
                "away_elo": int(round(elo_a)),
                "rest_days_home": rest_h,
                "rest_days_away": rest_a,
                "is_derby": 0,
                "data_snapshot_utc": snapshot
            })

            if pd.notna(r["fthg"]) and pd.notna(r["ftag"]) and r["ftr"] in ["H","D","A"]:
                r_home = elo_h + HOME_ADV
                r_away = elo_a
                e_home = expected_score(r_home, r_away)
                s_home = result_to_score(r["ftr"])

                elo[home] = elo_h + K_ELO * (s_home - e_home)
                elo[away] = elo_a + K_ELO * ((1 - s_home) - (1 - e_home))

                if r["ftr"] == "H":
                    history[home].append(3); history[away].append(0)
                elif r["ftr"] == "D":
                    history[home].append(1); history[away].append(1)
                else:
                    history[home].append(0); history[away].append(3)

                goals_for[home].append(int(r["fthg"]))
                goals_against[home].append(int(r["ftag"]))
                goals_for[away].append(int(r["ftag"]))
                goals_against[away].append(int(r["fthg"]))

                last_date[home] = date
                last_date[away] = date

    out = pd.DataFrame(features_rows)
    out_path = os.path.join(PROCESSED_DIR, "match_features.csv")
    out.to_csv(out_path, index=False)

    meta = {"last_features_build_utc": snapshot, "rows": int(out.shape[0])}
    with open(os.path.join(META_DIR, "features_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"OK: wrote {out_path} with {out.shape[0]} rows. snapshot={snapshot}")

if __name__ == "__main__":
    main()
