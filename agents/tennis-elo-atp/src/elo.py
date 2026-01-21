from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class EloConfig:
    k_factor: float = 32.0
    initial_rating: float = 1500.0
    surface_weight: bool = True

def expected_score(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

class EloRatings:
    def __init__(self, cfg: EloConfig):
        self.cfg = cfg
        self.global_ratings: Dict[str, float] = {}
        self.surface_ratings: Dict[Tuple[str, str], float] = {}  # (player, surface) -> rating

    def _get_global(self, player: str) -> float:
        return self.global_ratings.get(player, self.cfg.initial_rating)

    def _get_surface(self, player: str, surface: str) -> float:
        return self.surface_ratings.get((player, surface), self.cfg.initial_rating)

    def _set_global(self, player: str, rating: float):
        self.global_ratings[player] = rating

    def _set_surface(self, player: str, surface: str, rating: float):
        self.surface_ratings[(player, surface)] = rating

    def get_rating(self, player: str, surface: str) -> float:
        if self.cfg.surface_weight and surface:
            return 0.5 * self._get_global(player) + 0.5 * self._get_surface(player, surface)
        return self._get_global(player)

    def update_match(self, winner: str, loser: str, surface: str):
        r_w = self.get_rating(winner, surface)
        r_l = self.get_rating(loser, surface)
        exp_w = expected_score(r_w, r_l)

        k = self.cfg.k_factor

        # update
        new_r_w = r_w + k * (1.0 - exp_w)
        new_r_l = r_l + k * (0.0 - (1.0 - exp_w))

        delta_w = new_r_w - r_w
        delta_l = new_r_l - r_l

        # global
        self._set_global(winner, self._get_global(winner) + delta_w)
        self._set_global(loser, self._get_global(loser) + delta_l)

        # surface
        if self.cfg.surface_weight and surface:
            self._set_surface(winner, surface, self._get_surface(winner, surface) + delta_w)
            self._set_surface(loser, surface, self._get_surface(loser, surface) + delta_l)

    def predict_proba(self, player_a: str, player_b: str, surface: str) -> float:
        r_a = self.get_rating(player_a, surface)
        r_b = self.get_rating(player_b, surface)
        return expected_score(r_a, r_b)
