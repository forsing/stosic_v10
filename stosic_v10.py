from __future__ import annotations

"""
https://github.com/gajaka/luces-pvs-theories
"""

"""
stosic_v10.py — 7-node krug (K=7 / prilagodjenje 7/39) — McCann displacement interpolation (7/39)

Izvor (Stosić / LUCES):
  luces-pvs-theories-main/displacement_interpolation.pvs
  — μ_t = ((1-t)Id + tT)_# μ₀  (jedinstvena W₂ geodezija)
  — ax_endpoints, ax_constant_speed; thm_midpoint_equidistant (t=1/2)

Mapiranje na 7/39:
  x_k = ko-pojavljivanje (ceo CSV); c(i,j)=||x_i-x_j||²
  za svaki uzastopni A→B: min-cost matching T
  McCann u feature prostoru: m = (1-t)·x_i + t·x_j, t=SEED/(SEED+SEED)=1/2
  skor[k] += 1 / (ε + ||m - x_k||²)   (push-forward ka diskretnim atomima)
  next = top 7; bez randoma; stop ako uzastopni ili AP (konstantan korak)
"""

from typing import List

import numpy as np

from stosic_v1 import EPS, MAX_NUM, N_PICK, SEED, load_draws
from stosic_v2 import top7_from_freq
from stosic_v8 import cooccurrence_features, cost_matrix, is_degenerate_consecutive
from stosic_v9 import optimal_matching_support


def is_degenerate_ap(combo: List[int]) -> bool:
    s = sorted(combo)
    if len(s) != N_PICK:
        return False
    d = s[1] - s[0]
    return d >= 1 and all(s[i + 1] - s[i] == d for i in range(N_PICK - 1))


def is_degenerate(combo: List[int]) -> bool:
    return is_degenerate_consecutive(combo) or is_degenerate_ap(combo)


def accumulate_mccann_mid(
    draws: np.ndarray, feats: np.ndarray, C: np.ndarray, t: float
) -> tuple[np.ndarray, np.ndarray]:
    skor = np.zeros(MAX_NUM, dtype=np.float64)
    sq = np.sum(feats * feats, axis=1)

    def add_mid(i: int, j: int) -> None:
        m = (1.0 - t) * feats[i] + t * feats[j]
        d2 = sq + np.dot(m, m) - 2.0 * (feats @ m)
        skor[:] += 1.0 / (EPS + np.maximum(d2, 0.0))

    for k in range(len(draws) - 1):
        src = [int(n) - 1 for n in draws[k]]
        tgt = [int(n) - 1 for n in draws[k + 1]]
        for i, j in optimal_matching_support(src, tgt, C):
            add_mid(i, j)
    nu = np.zeros(MAX_NUM, dtype=np.float64)
    for d in draws:
        for n in d:
            nu[int(n) - 1] += 1.0
    mu1 = [int(x) - 1 for x in top7_from_freq(nu)]
    src = [int(n) - 1 for n in draws[-1]]
    for i, j in optimal_matching_support(src, mu1, C):
        add_mid(i, j)
    return skor, nu


def predict_next(draws: np.ndarray) -> List[int]:
    feats = cooccurrence_features(draws)
    C = cost_matrix(feats)
    t = float(SEED) / float(SEED + SEED)  # 1/2
    skor, nu = accumulate_mccann_mid(draws, feats, C, t)
    nu = nu / max(nu.sum(), EPS)
    combo = top7_from_freq(skor + EPS * nu)
    if is_degenerate(combo):
        combo = top7_from_freq(nu)
    return combo


def main():
    draws = load_draws()
    next_combo = predict_next(draws)
    if is_degenerate(next_combo):
        raise SystemExit("degenerisan next (uzastopni/AP) — zaustavljen pre ispisa")
    print(next_combo)


if __name__ == "__main__":
    main()



"""
[3, 5, 8, 23, 25, 34, 35]
"""



"""
displacement_interpolation → v10.

ako je skoro degenerisano (AP korak 2) — odbacujem; McCann u feature prostoru.

v10: displacement_interpolation — McCann mid u feature prostoru (t=1/2).
"""



"""
21 teorija

fisher_voronoi → v1, v2
dual_observability → v3
v4 se pozivao na W₂/stabilnost — slabo / nije strogo
entropy_along_geodesic → v5
velocity_asymmetry (+ delom lie_generator_structure) → v6
brenier_uniqueness (+ delom rank_orientation) → v7

kantorovich_duality
cyclical_monotonicity
displacement_interpolation
displacement_concavity
wasserstein_metric (strogo)
transport_structure
transport_structure_v2
transport_stability
stability_of_maps
monge_kantorovich_equivalence
lie_generator_structure (pun T10)
fisher_boundary
hybrid_observability
tangent_bundle
global_optimality
"""



"""
Kratko, o repou:

21 PVS teorija — sve su prošle kroz v1–v22 (neke ranije labavo: naročito v3/v4; rank_orientation je ušao uz Brenier u v7).
Repo je o spektralnom OT / LUCES (ESP32), ne o lotou — 7/39 je naša mapa, ne Stosićev domen.
Najčistije jezgro oko Fisher–Voronoi, Brenier/CM, W₂, T10 (lie_generator_structure). global_optimality je samo aksiomi + lema (bez teorema).
Empirija u PVS-u (bootovi, κ, Monge fraction) ne prenosi se automatski na CSV — samo struktura ideja.
"""
