import subprocess
import sys

FIXTURES = [
    ("LL", "Getafe", "Valencia"),
    ("LL", "Atletico", "Alaves"),
    ("LL", "Celta", "Rayo"),
    ("LL", "Sociedad", "Barcelona"),
]

print("\n=== Predicciones Jornada 20 · LaLiga ===\n")

for league, home, away in FIXTURES:
    print(f"\n--- {home} vs {away} ---")
    subprocess.run([sys.executable, "-m", "src.predict_match", league, home, away], check=False)