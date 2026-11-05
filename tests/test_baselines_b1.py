"""B1 re-implementations: IMOEA/D-AWN (Jin et al. 2023) and IMEA-IED (Cui et al. 2024)."""
import numpy as np
import pytest

from gat_imoea_dqn import baselines_b1 as b1


def test_confidence_level():
    assert b1.confidence_level(0, 1, 2, 3) == pytest.approx(1.0)          # A entirely below B
    assert b1.confidence_level(2, 3, 0, 1) == pytest.approx(0.0)
    assert b1.confidence_level(0, 1, 0, 1) == pytest.approx(0.5)          # identical intervals
    assert b1.confidence_level(0, 2, 1, 3) + b1.confidence_level(1, 3, 0, 2) == pytest.approx(1.0)


def test_interval_distance():
    assert b1.interval_distance(0.0, 0.0, 3.0, 3.0) == pytest.approx(3.0)  # points: |difference|
    assert b1.interval_distance(0.0, 2.0, 0.0, 2.0) == pytest.approx(0.0)  # identical: w^2/3 + w^2/3 - 2 w^2/3


def test_awn_ranking_prefers_sparse_and_precise():
    # three clustered solutions and one isolated: the isolated one ranks best on sparsity
    lo = np.array([[0.0, 1.0], [0.01, 0.99], [0.02, 0.98], [1.0, 0.0]])
    hi = lo + 0.01
    R = b1.awn_ranking(lo, hi, alpha=1.0)
    assert np.argmin(R) == 3


def test_ied_matches_paper_example():
    # Cui et al. 2024, Fig. 3: a = [0.18, 0.68], b = [0.1, 0.7]; group A -> a dominates b,
    # group B -> b dominates a (gamma = 0.5)
    groups = {"A": [[0.18, 0.68], [0.1, 0.7], [0.02, 0.9], [0.3, 0.6]],
              "B": [[0.18, 0.68], [0.1, 0.7], [0.2, 0.8], [0.25, 0.85]]}
    P = {k: b1.ied_probabilities(np.array(g)[:, :1], np.array(g)[:, 1:])[0, 1, 0] for k, g in groups.items()}
    assert P["A"] > 0.5 > P["B"]


def test_ied_probabilities_complementary():
    rng = np.random.default_rng(0)
    lo = rng.random((6, 3))
    hi = lo + rng.random((6, 3))
    P = b1.ied_probabilities(lo, hi)
    np.testing.assert_allclose(P + P.transpose(1, 0, 2), 1.0)


def test_ied_niche_select_returns_n_distinct():
    rng = np.random.default_rng(1)
    lo = rng.random((20, 3))
    hi = lo + 0.05
    from gat_imoea_dqn.intervals import nondominated_fronts
    from gat_imoea_dqn.baselines import _weights
    Z = _weights(10)
    fronts = nondominated_fronts(b1.ied_dominance_matrix(lo, hi))
    keep, C = b1.ied_niche_select(lo, hi, fronts, 10, Z, rng, np.zeros(len(Z)))
    assert len(keep) == 10 and len(set(keep.tolist())) == 10 and C.shape == (len(Z),)
