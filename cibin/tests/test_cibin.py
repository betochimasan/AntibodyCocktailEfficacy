"""Various tests for methods from cibin.py file."""


import sys
import pytest
import numpy as np
from ..cibin import *


def test_tau_twoside_liding_1():
    """
    Test that tau_twoside gives the confidence intervals.

    (Li & Ding paper).
    """
    N = 16
    res1 = tau_twoside(1, 1, 1, 13, 0.05, 103)
    expected1 = [-1/N, 14/N]
    np.testing.assert_approx_equal(res1['tau_lower'],
                                   expected1[0], significant=1)
    np.testing.assert_approx_equal(res1['tau_upper'],
                                   expected1[1], significant=1)


def test_tau_twoside_liding_2():
    """
    Test that tau_twoside gives the confidence intervals.

    (Li & Ding paper).
    """
    N = 16
    res2 = tau_twoside(2, 6, 8, 0, 0.05, 113)
    expected2 = [-14/N, -5/N]
    np.testing.assert_approx_equal(res2['tau_lower'],
                                   expected2[0], significant=1)
    np.testing.assert_approx_equal(res2['tau_upper'],
                                   expected2[1], significant=1)


def test_tau_twoside_liding_3():
    """
    Test that tau_twoside gives the confidence intervals.

    (Li & Ding paper).
    """
    N = 20
    res3 = tau_twoside(6, 0, 11, 3, 0.05, 283)
    expected3 = [-4/N, 8/N]
    np.testing.assert_approx_equal(res3['tau_lower'],
                                   expected3[0], significant=1)
    np.testing.assert_approx_equal(res3['tau_upper'],
                                   expected3[1], significant=1)


def test_tau_twoside_liding_4():
    """
    Test that tau_twoside gives the confidence intervals.

    (Li & Ding paper).
    """
    N = 20
    res4 = tau_twoside(6, 4, 4, 6, 0.05, 308)
    expected4 = [-4/N, 10/N]
    np.testing.assert_approx_equal(res4['tau_lower'],
                                   expected4[0], significant=1)
    np.testing.assert_approx_equal(res4['tau_upper'],
                                   expected4[1], significant=1)


def test_tau_twoside_liding_5():
    """
    Test that tau_twoside gives the confidence intervals.

    (Li & Ding paper).
    """
    N = 24
    res5 = tau_twoside(1, 1, 3, 19, 0.05, 251)
    expected5 = [-3/N, 20/N]
    np.testing.assert_approx_equal(res5['tau_lower'],
                                   expected5[0], significant=1)
    np.testing.assert_approx_equal(res5['tau_upper'],
                                   expected5[1], significant=1)


def test_tau_twoside_liding_6():
    """
    Test that tau_twoside gives the confidence intervals.

    (Li & Ding paper).
    """
    N = 24
    res6 = tau_twoside(8, 4, 5, 7, 0.05, 421)
    expected6 = [-3/N, 13/N]
    np.testing.assert_approx_equal(res6['tau_lower'],
                                   expected6[0], significant=1)
    np.testing.assert_approx_equal(res6['tau_upper'],
                                   expected6[1], significant=1)


def test_nchoosem():
    """Test nchoosem."""
    n1 = 6
    m1 = 2
    Z1 = np.array([[1., 1., 0., 0., 0., 0.],
                   [1., 0., 1., 0., 0., 0.],
                   [1., 0., 0., 1., 0., 0.],
                   [1., 0., 0., 0., 1., 0.],
                   [1., 0., 0., 0., 0., 1.],
                   [0., 1., 1., 0., 0., 0.],
                   [0., 1., 0., 1., 0., 0.],
                   [0., 1., 0., 0., 1., 0.],
                   [0., 1., 0., 0., 0., 1.],
                   [0., 0., 1., 1., 0., 0.],
                   [0., 0., 1., 0., 1., 0.],
                   [0., 0., 1., 0., 0., 1.],
                   [0., 0., 0., 1., 1., 0.],
                   [0., 0., 0., 1., 0., 1.],
                   [0., 0., 0., 0., 1., 1.]])
    assert((nchoosem(n1, m1) == Z1).all())

    n2 = 4
    m2 = 3
    Z2 = np.array([[1., 1., 1., 0.],
                   [1., 1., 0., 1.],
                   [1., 0., 1., 1.],
                   [0., 1., 1., 1.]])
    assert((nchoosem(n2, m2) == Z2).all())

    n3 = 5
    m3 = 1
    Z3 = np.array([[1., 0., 0., 0., 0.],
                   [0., 1., 0., 0., 0.],
                   [0., 0., 1., 0., 0.],
                   [0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 1.]])
    assert((nchoosem(n3, m3) == Z3).all())


def test_pval_two():
    """Test if pval_two outputs the expect p-value."""
    n = 4
    m = 2
    N = [1, 1, 1, 1]
    tau_obs = 0
    Z_all = nchoosem(n, m)
    expected = 1
    assert(pval_two(n, m, N, Z_all, tau_obs) == expected)

    n2 = 4
    m2 = 2
    N2 = [1, 1, 1, 1]
    tau_obs2 = 0.5
    Z_all2 = nchoosem(n2, m2)
    expected2 = 2/3
    assert(pval_two(n2, m2, N2, Z_all2, tau_obs2) == expected2)

    n3 = 5
    m3 = 1
    N3 = [1, 1, 1, 2]
    tau_obs3 = 0
    Z_all3 = nchoosem(n3, m3)
    expected3 = 1
    assert(pval_two(n3, m3, N3, Z_all3, tau_obs3) == expected3)

    n4 = 5
    m4 = 2
    N4 = [1, 1, 1, 2]
    tau_obs4 = 0.5
    Z_all4 = nchoosem(n4, m4)
    expected4 = 0.3
    assert(pval_two(n4, m4, N4, Z_all4, tau_obs4) == expected4)


def test_check_compatible():
    """Test if check_compatible performs as expected."""
    n11 = 5
    n10 = 5
    n01 = 5
    n00 = 5
    N11 = [1, 5]
    N10 = [12, 13, 42]
    N01 = [5, 5, 8, 9, 5]
    compat = np.array([False, False, False, False, False])
    assert((check_compatible(n11, n10, n01, n00,
                             N11, N10, N01) == compat).all())

    n11_2 = 1
    n10_2 = 1
    n01_2 = 1
    n00_2 = 13
    N11_2 = [1, 1, 1, 1]
    N10_2 = [1, 1, 1, 1]
    N01_2 = [1, 1, 1, 1]
    compat_2 = np.array([True, True, True, True])
    assert((check_compatible(n11_2, n10_2, n01_2, n00_2,
                             N11_2, N10_2, N01_2) == compat_2).all())
