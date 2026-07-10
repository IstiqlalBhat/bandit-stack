import numpy as np
import pytest

from bandit_core import LinTS


def _train(policy: LinTS, x: np.ndarray, y: np.ndarray) -> None:
    for xi, yi in zip(x, y):
        # update() reads features for the chosen arm; wrap each row as arm 0
        policy.update(0, float(yi), [xi, np.zeros_like(xi)])


def test_posterior_mean_matches_ridge_regression():
    rng = np.random.default_rng(0)
    dim, n, lam = 5, 200, 2.0
    theta_true = rng.normal(size=dim)
    x = rng.normal(size=(n, dim))
    y = x @ theta_true + rng.normal(0, 0.1, size=n)

    policy = LinTS(dim=dim, lambda_reg=lam, seed=0)
    _train(policy, x, y)

    ridge = np.linalg.solve(lam * np.eye(dim) + x.T @ x, x.T @ y)
    np.testing.assert_allclose(policy.a_inv @ policy.b, ridge, atol=1e-8)


def test_sherman_morrison_matches_direct_inverse():
    rng = np.random.default_rng(1)
    dim, n, lam = 4, 50, 1.0
    x = rng.normal(size=(n, dim))

    policy = LinTS(dim=dim, lambda_reg=lam, seed=0)
    _train(policy, x, np.zeros(n))

    direct = np.linalg.inv(lam * np.eye(dim) + x.T @ x)
    np.testing.assert_allclose(policy.a_inv, direct, atol=1e-8)


def test_learns_to_pick_best_arm():
    rng = np.random.default_rng(2)
    dim = 4
    theta_true = np.array([1.0, -0.5, 0.3, 0.8])
    policy = LinTS(dim=dim, propensity_samples=50, seed=0)

    for _ in range(400):
        feats = rng.normal(size=(3, dim))
        feats /= np.linalg.norm(feats, axis=1, keepdims=True)
        d = policy.choose(feats)
        reward = float(feats[d.arm] @ theta_true + rng.normal(0, 0.1))
        policy.update(d.arm, reward, feats)

    correct = 0
    trials = 200
    for _ in range(trials):
        feats = rng.normal(size=(3, dim))
        feats /= np.linalg.norm(feats, axis=1, keepdims=True)
        d = policy.choose(feats)
        correct += d.arm == int(np.argmax(feats @ theta_true))
    assert correct / trials > 0.85


def test_requires_features():
    policy = LinTS(dim=3, seed=0)
    with pytest.raises(ValueError):
        policy.choose()
    with pytest.raises(ValueError):
        policy.update(0, 1.0)


def test_rejects_wrong_feature_shape():
    policy = LinTS(dim=3, seed=0)
    with pytest.raises(ValueError):
        policy.choose([np.zeros(2), np.zeros(2)])
