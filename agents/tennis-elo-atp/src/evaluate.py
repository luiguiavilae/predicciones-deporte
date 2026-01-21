import os
import pickle
import pandas as pd
import yaml
from sklearn.metrics import log_loss, brier_score_loss

def load_config():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    processed_path = os.path.join(cfg["data"]["processed_dir"], "atp_matches_all.csv")
    df = pd.read_csv(processed_path)

    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["tourney_date", "winner_name", "loser_name"]).sort_values("tourney_date")

    with open(cfg["train"]["model_out"], "rb") as f:
        model = pickle.load(f)

    y_true, y_pred = [], []
    for _, row in df.tail(2000).iterrows():
        surface = row.get("surface", "") or ""
        p_win = model.predict_proba(row["winner_name"], row["loser_name"], surface)
        y_true.append(1)
        y_pred.append(max(1e-6, min(1 - 1e-6, p_win)))

    ll = log_loss(y_true, y_pred)
    bs = brier_score_loss(y_true, y_pred)

    os.makedirs("reports", exist_ok=True)
    out = f"""# Evaluation (naive)
Matches: {len(y_true)}

- LogLoss: {ll:.4f}
- Brier: {bs:.4f}

> Esto es un sanity check, no walk-forward.
"""
    with open("reports/evaluation.md", "w", encoding="utf-8") as f:
        f.write(out)

    print(out)

if __name__ == "__main__":
    main()
