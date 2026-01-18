import os
import json
import sys
import re
import pandas as pd

PROCESSED_DIR = "data/processed"
META_DIR = "data/meta"

def load_meta():
    path = os.path.join(META_DIR, "update_meta.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def normalize(s: str) -> str:
    return str(s).strip().lower()

def main():
    if len(sys.argv) < 2:
        print('Uso: python -m src.agent_cli "status"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    q = normalize(question)

    match_path = os.path.join(PROCESSED_DIR, "match_level.csv")
    if not os.path.exists(match_path):
        print("No existe data/processed/match_level.csv")
        sys.exit(1)

    df = pd.read_csv(match_path)
    meta = load_meta()
    last_update = meta.get("last_update_utc", "desconocido")

    if "status" in q:
        print("=== ESTADO DEL AGENTE ===")
        print(f"Última actualización: {last_update}")
       df["season"] = df["season"].astype(str)
        df["league"] = df["league"].astype(str)

        print("\nTemporadas por liga:")
        for lg, ss in df.groupby("league")["season"].unique().items():
            print(f"- {lg}: {', '.join(sorted(ss))}")

        print("\nPartidos por liga y temporada:")
        for (lg, se), n in df.groupby(["league","season"]).size().items():
            print(f"- {lg} {se}: {n}")

        df["fthg"] = pd.to_numeric(df.get("fthg"), errors="coerce")
        df["ftag"] = pd.to_numeric(df.get("ftag"), errors="coerce")
        pending = int(((df["fthg"].isna()) | (df["ftag"].isna())).sum())
        print(f"\nPartidos pendientes: {pending}")
        return

    print("Comando no reconocido")

if __name__ == "__main__":
    main()
