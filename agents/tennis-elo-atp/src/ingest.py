import os
import pandas as pd
import yaml
from urllib.request import urlretrieve

def load_config():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs(cfg):
    os.makedirs(cfg["data"]["raw_dir"], exist_ok=True)
    os.makedirs(cfg["data"]["processed_dir"], exist_ok=True)

def download_season(base_url: str, season: int, out_dir: str) -> str:
    filename = f"atp_matches_{season}.csv"
    url = base_url + filename
    out_path = os.path.join(out_dir, filename)
    if not os.path.exists(out_path):
        print(f"Downloading {url}")
        urlretrieve(url, out_path)
    return out_path

def main():
    cfg = load_config()
    ensure_dirs(cfg)

    base_url = cfg["data"]["atp_matches_base_url"]
    raw_dir = cfg["data"]["raw_dir"]
    processed_dir = cfg["data"]["processed_dir"]
    seasons = cfg["data"]["seasons"]

    paths = [download_season(base_url, s, raw_dir) for s in seasons]
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    keep = ["tourney_date", "tourney_name", "surface", "round", "winner_name", "loser_name", "score", "best_of"]
    df = df[[c for c in keep if c in df.columns]].copy()

    out_path = os.path.join(processed_dir, "atp_matches_all.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    main()
