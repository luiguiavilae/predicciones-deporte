import os
import pickle
import pandas as pd
import yaml

from src.elo import EloConfig, EloRatings

def load_config():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    processed_path = os.path.join(cfg["data"]["processed_dir"], "atp_matches_all.csv")
    df = pd.read_csv(processed_path)

    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date", "winner_name", "loser_name"]).sort_values("tourney_date")

    elo_cfg = EloConfig(
        k_factor=float(cfg["elo"]["k_factor"]),
        initial_rating=float(cfg["elo"]["initial_rating"]),
        surface_weight=bool(cfg["elo"]["surface_weight"]),
    )
    ratings = EloRatings(elo_cfg)

    for _, row in df.iterrows():
        surface = row.get("surface", "") or ""
        ratings.update_match(row["winner_name"], row["loser_name"], surface)

    os.makedirs("models", exist_ok=True)
    out_path = cfg["train"]["model_out"]
    with open(out_path, "wb") as f:
        pickle.dump(ratings, f)

    print(f"Saved model: {out_path}")

if __name__ == "__main__":
    main()
