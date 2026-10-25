import numpy as np
import pytest

from gat_imoea_dqn.metrics import hv_jin, hypervolume_3d, igd, imprecision, nondominated, pooled_reference


def test_jin_metrics():
    F = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)    # row 2 dominated, row 3 a duplicate
    assert nondominated(F).tolist() == [[0, 1, 1], [1, 0, 1]]
    ref = pooled_reference([F[:2], F[2:]])
    assert len(ref) == 2
    assert igd(ref, ref) == 0.0
    assert igd(np.array([[1, 1, 1]], float), ref) == pytest.approx(1.0)
    lo = np.array([[0.2, 0.2, 0.2]])
    hi = np.array([[0.6, 0.6, 0.6]])
    assert hv_jin(lo, hi, [1, 1, 1]) == pytest.approx((0.8 ** 3 + 0.4 ** 3) / 2)
    assert imprecision(np.vstack([lo, lo]), np.vstack([hi, hi])) == pytest.approx(2 * 3 * 0.4)


def test_hypervolume_by_hand():
    assert hypervolume_3d([[0, 0, 0]], [1, 1, 1]) == pytest.approx(1.0)
    # two boxes of 0.5 overlapping in 0.25
    assert hypervolume_3d([[0, 0.5, 0], [0.5, 0, 0]], [1, 1, 1]) == pytest.approx(0.75)
    assert hypervolume_3d([[2, 0, 0]], [1, 1, 1]) == 0.0                    # outside the reference box
    assert hypervolume_3d([[0, 0, 0], [0.5, 0.5, 0.5]], [1, 1, 1]) == pytest.approx(1.0)   # dominated point


def test_hypervolume_matches_monte_carlo():
    rng = np.random.default_rng(0)
    F = rng.random((25, 3))
    exact = hypervolume_3d(F, [1, 1, 1])
    S = rng.random((200_000, 3))
    mc = np.mean(np.any(np.all(F[None, :, :] <= S[:, None, :], axis=2), axis=1))
    assert exact == pytest.approx(mc, abs=0.005)
