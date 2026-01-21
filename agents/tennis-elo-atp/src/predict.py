import pickle
import yaml

def load_config():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    cfg = load_config()
    model = load_model(cfg["train"]["model_out"])

    player_a = "Carlos Alcaraz"
    player_b = "Jannik Sinner"
    surface = cfg["predict"]["default_surface"]

    p_a = model.predict_proba(player_a, player_b, surface)
    print(f"{player_a} vs {player_b} ({surface})")
    print(f"P({player_a}) = {p_a:.3f}")
    print(f"P({player_b}) = {1-p_a:.3f}")

if __name__ == "__main__":
    main()
