import argparse
import pickle
import yaml

def load_config():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser(description="ATP ELO predictor")
    parser.add_argument("player_a", type=str, help="Jugador A (nombre exacto del dataset)")
    parser.add_argument("player_b", type=str, help="Jugador B (nombre exacto del dataset)")
    parser.add_argument("--surface", type=str, default=None, help="Hard | Clay | Grass (opcional)")
    args = parser.parse_args()

    cfg = load_config()
    model = load_model(cfg["train"]["model_out"])
    surface = args.surface or cfg["predict"]["default_surface"]

    p_a = model.predict_proba(args.player_a, args.player_b, surface)

    print(f"{args.player_a} vs {args.player_b} ({surface})")
    print(f"P({args.player_a}) = {p_a:.3f}")
    print(f"P({args.player_b}) = {1-p_a:.3f}")

if __name__ == "__main__":
    main()
