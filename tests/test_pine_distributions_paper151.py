"""Economic invariance tests for the corrected Pine dividend reconstruction."""

import numpy as np


def reconstruct(c, adj):
    c = np.asarray(c, float)
    factor = c / np.asarray(adj, float)
    distribution = np.r_[0, c[:-1] * (1 - factor[1:] / factor[:-1])]
    distribution[np.abs(distribution) < 1e-6] = 0
    return distribution, np.r_[0, (c[1:] + distribution[1:]) / c[:-1] - 1]


def test_ex_dividend_drop_without_economic_loss_on_split_adjusted_basis():
    c = np.array([50, 49.5, 50.5])
    adjustment = np.array([0.9, 0.9 / 0.99, 0.9 / 0.99])
    dividends, returns = reconstruct(c, c * adjustment)
    np.testing.assert_allclose(dividends, [0, 0.5, 0], atol=1e-12)
    assert abs(returns[1]) < 1e-12
    assert returns[2] == 50.5 / 49.5 - 1
    # Adding the pre-split $1 payment to a post-split $50 quote would create false profit.
    assert (49.5 + 1) / 50 - 1 > 0


def test_later_common_adjustment_cannot_change_earlier_returns():
    c = np.array([50, 49.5, 50.5])
    adj = c * np.array([0.9, 0.9 / 0.99, 0.9 / 0.99])
    a = reconstruct(c, adj)
    b = reconstruct(c, adj * 0.73)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_another_share_split_rescales_cash_not_total_returns():
    c = np.array([50, 49.5, 50.5])
    adj = c * np.array([0.9, 0.9 / 0.99, 0.9 / 0.99])
    d, r = reconstruct(c, adj)
    d2, r2 = reconstruct(c / 2, adj / 2)
    np.testing.assert_allclose(d2, d / 2, atol=1e-12)
    np.testing.assert_allclose(r2, r, atol=1e-12)
