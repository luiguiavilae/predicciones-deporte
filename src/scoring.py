from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional


def _top2(values: Dict[str, float]) -> Tuple[float, float]:
    """Return (best, second_best) from a dict of probabilities."""
    probs = sorted(values.values(), reverse=True)
    best = probs[0] if probs else 0.0
    second = probs[1] if len(probs) > 1 else 0.0
    return best, second


def _coherence_1x2(pick: str, lam_h: float, lam_a: float) -> float:
    """
    Coherencia básica para 1X2 usando lambdas.
    Devuelve 0..100.
    """
    diff = lam_h - lam_a

    # Buffers para evitar "coherencia" por diferencias mínimas
    if pick == "H":
        if diff >= 0.20:
            return 100.0
        if diff >= 0.05:
            return 50.0
        return 0.0

    if pick == "A":
        if diff <= -0.20:
            return 100.0
        if diff <= -0.05:
            return 50.0
        return 0.0

    # Draw (D)
    if abs(diff) <= 0.10:
        return 100.0
    if abs(diff) <= 0.20:
        return 50.0
    return 0.0


def _coherence_ou25(pick: str, lam_total: float) -> float:
    """
    Coherencia para Over/Under 2.5 con lambda total.
    Devuelve 0..100.
    """
    # Buffers
    if pick.lower() == "over":
        if lam_total >= 2.7:
            return 100.0
        if lam_total >= 2.5:
            return 50.0
        return 0.0

    if pick.lower() == "under":
        if lam_total <= 2.3:
            return 100.0
        if lam_total <= 2.5:
            return 50.0
        return 0.0

    return 0.0


def _coherence_btts(pick: str, lam_h: float, lam_a: float) -> float:
    """
    Coherencia para BTTS usando lambdas.
    Devuelve 0..100.
    """
    if pick.lower() == "yes":
        # Para que ambos anoten, esperamos lambdas razonables en ambos
        if lam_h >= 0.95 and lam_a >= 0.95:
            return 100.0
        if lam_h >= 0.75 and lam_a >= 0.75:
            return 50.0
        return 0.0

    if pick.lower() == "no":
        # Para que NO ambos anoten, basta que al menos uno tenga lambda baja
        if lam_h <= 0.70 or lam_a <= 0.70:
            return 100.0
        if lam_h <= 0.85 or lam_a <= 0.85:
            return 50.0
        return 0.0

    return 0.0


def confidence_score(
    market: str,
    pick: str,
    dist: Dict[str, float],
    lambda_home: float,
    lambda_away: float,
) -> float:
    """
    Score 0..100 basado en:
      - probabilidad max del mercado (50%)
      - diferencial vs 2do (30%)
      - coherencia con lambdas (20%)

    market: '1X2' | 'OU25' | 'BTTS'
    pick:   para 1X2: 'H'|'D'|'A'
            para OU25: 'Over'|'Under'
            para BTTS: 'Yes'|'No'
    dist:   dict con probabilidades del mercado (0..1)
    """
    best, second = _top2(dist)
    prob_max = best * 100.0
    diff = (best - second) * 100.0

    lam_total = float(lambda_home) + float(lambda_away)

    if market == "1X2":
        coh = _coherence_1x2(pick, float(lambda_home), float(lambda_away))
    elif market == "OU25":
        coh = _coherence_ou25(pick, lam_total)
    elif market == "BTTS":
        coh = _coherence_btts(pick, float(lambda_home), float(lambda_away))
    else:
        coh = 0.0

    score = (0.5 * prob_max) + (0.3 * diff) + (0.2 * coh)
    # clamp
    if score < 0:
        score = 0.0
    if score > 100:
        score = 100.0
    return float(score)
