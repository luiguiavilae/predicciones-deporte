from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class EloConfig:
    k_factor: float = 24.0
    initial_rating: float = 1500.0
    surface_weight: bool = True

    # Pesos (si no existen en config, se usan estos defaults)
    global_weight: float = 0.5
    surface_rating_weight: float = 0.5

    # Escala para suavizar probabilidades (tenis necesita > 400)
    scale: float = 600.0


def expected_score(r_a: float, r_b: float, scale: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / scale))


class EloRatings:
    def __init__(self, cfg: EloConfig):
        self.cfg = cfg
        self.global_ratings: Dict[str, float] = {}
        self.surface_ratings: Dict[Tuple[str, str], float] = {}

    def _get_global(self, player: str) -> float:
        return self.global_ratings.get(player, self.cfg.initial_rating)

    def _get_surface(self, player: str, surface: str) -> float:
        return self.surface_ratings.get((player, surface), self.cfg.initial_rating)

    def _set_global(self, player: str, rating: float):
        self.global_ratings[player] = rating

    def _set_surface(self, player: str, surface: str, rating: float):
        self.surface_ratings[(player, surface)] = rating

    def get_rating(self, player: str, surface: Optional[str]) -> float:
        """
        Rating efectivo SOLO para inferencia (expected score).
        NO se usa directamente para actualizar global/superficie.
        """
        g = self._get_global(player)

        if self.cfg.surface_weight and surface:
            s = self._get_surface(player, surface)

            wg = float(getattr(self.cfg, "global_weight", 0.5))
            ws = float(getattr(self.cfg, "surface_rating_weight", 0.5))
            total = wg + ws

            if total <= 0:
                wg, ws, total = 0.5, 0.5, 1.0

            wg /= total
            ws /= total

            return wg * g + ws * s

        return g

    def update_match(self, winner: str, loser: str, surface: Optional[str]):
        """
        Actualización estable:
        - Expected score con rating efectivo (mezclado)
        - Updates aplicados por separado a global y superficie
        """
        r_w_eff = self.get_rating(winner, surface)
        r_l_eff = self.get_rating(loser, surface)

        scale = float(getattr(self.cfg, "scale", 400.0))
        exp_w = expected_score(r_w_eff, r_l_eff, scale=scale)
        exp_l = 1.0 - exp_w

        k = float(self.cfg.k_factor)

        # Resultado observado
        s_w, s_l = 1.0, 0.0

        delta_w = k * (s_w - exp_w)
        delta_l = k * (s_l - exp_l)

        # Global (siempre)
        self._set_global(winner, self._get_global(winner) + delta_w)
        self._set_global(loser, self._get_global(loser) + delta_l)

        # Superficie (si aplica)
        if self.cfg.surface_weight and surface:
            self._set_surface(winner, surface, self._get_surface(winner, surface) + delta_w)
            self._set_surface(loser, surface, self._get_surface(loser, surface) + delta_l)

    def predict_proba(self, player_a: str, player_b: str, surface: Optional[str]) -> float:
        r_a = self.get_rating(player_a, surface)
        r_b = self.get_rating(player_b, surface)
        scale = float(getattr(self.cfg, "scale", 400.0))
        return expected_score(r_a, r_b, scale=scale)
