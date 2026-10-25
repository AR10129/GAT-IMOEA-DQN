import numpy as np
import pytest

from gat_imoea_dqn.metrics import hypervolume_3d


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
