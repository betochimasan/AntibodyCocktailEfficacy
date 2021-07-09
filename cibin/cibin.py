"""Implementation of CI functions."""

import numpy as np
import math
from scipy.special import comb
from itertools import combinations


def nchoosem(n, m):
    """
    Exact re-randomization matrix for small n choose m.

    Matrix is computed by first calculating all the
    possible combinations for n choose m and then
    listing the combinations in an outputted matrix.

    Parameters
    ----------
    n : int
        Total number of items.
    m : int
        Number of items in each combination.

    Returns
    -------
    Z : np.array
        A matrix where each row represents one
        possible combination for given n and m.
    """
    assert m <= n, "Number of items within each \
                    combination cannot be greater than \
                    the total number of items."

    c = math.factorial(n) // (math.factorial(n - m) * math.factorial(m))

    lst = [i for i in range(n)]
    arr = []
    combination = combinations(lst, m)
    for i in combination:
        arr.append([i])
    trt = np.array(arr)

    Z = np.array([[math.nan for i in range(n)] for i in range(c)])

    for i in range(c):
        index = np.array([lst[j] in trt[i] for j in range(n)])
        Z[i][index] = 1
        Z[i][~index] = 0

    return Z


def comb(n, m, nperm):
    """
    Sample from re-randomization matrix.

    Sample of size nperm from all the possible combinations
    given n choose m, outputted in the form of a matrix
    where each row is one sample.

    Parameters
    ----------
    n : int
        Total number of items.
    m : int
        Number of items in each combination.
    nperm : int
        The number of items in the desired sample.

    Returns
    -------
    Z : np.array
        A matrix where each row represents one sample.
    """
    lst = np.array([0 for i in range(n-m)])
    lst = np.append(lst, np.array([1 for i in range(m)]))
    np.random.shuffle(lst)

    sampled_mat = np.copy(lst)
    for k in range(nperm-1):
        np.random.shuffle(lst)
        sampled_mat = np.vstack([sampled_mat, lst])

    return sampled_mat


def pval_two(n, m, N, Z_all, tau_obs):
    """
    Calculate the p-value.

    Calculate the p-value for a given observed difference.

    Parameters
    ----------
    n : int
        Total number of items.
    m : int
        Number of items in each combination.
    N : list
        List of values corresponding to the distribution of
        True Positives, False Positives, True Negatives, False
        Negatives
    Z_all : np.array
        A matrix where each row represents one
        possible combination for given n and m.
    tau_obs : float
        The observed difference.

    Returns
    -------
    pd : float
        The calculated p-value.
    """
    n_Z_all = Z_all.shape[0]
    dat = np.array([[math.nan for i in range(2)] for j in range(n)])

    if N[0] > 0:
        for i in range(N[0]):
            dat[i] = 1
    if N[1] > 0:
        for i in range(N[0], N[0]+N[1]):
            dat[i][0] = 1
            dat[i][1] = 0
    if N[2] > 0:
        for i in range(N[0]+N[1], N[0]+N[1]+N[2]):
            dat[i][0] = 0
            dat[i][1] = 1
    if N[3] > 0:
        for i in range(N[0]+N[1]+N[2], N[0]+N[1]+N[2]+N[3]):
            dat[i] = 0

    tau_hat = np.matmul(Z_all, dat[:, 0]/m) - np.matmul((1 - Z_all),
                                                        dat[:, 1]/(n-m))
    tau_N = (N[1]-N[2])/n
    pd_n = np.round(abs(tau_hat-tau_N), 15) >= np.round(abs(tau_obs-tau_N), 15)
    pd = sum(pd_n) / n_Z_all
    return pd


def check_compatible(n11, n10, n01, n00, N11, N10, N01):
    """
    Checked if the inputs are compatible.

    Checks if the inputs are compatible by comparing
    the values of the inputs.

    Parameters
    ----------
    n11 : int
        The number of True Positives.
    n10 : int
        The number of False Negatives.
    n01 : int
        The number of False Positives.
    n00 : int
        The number of True Negatives.
    N11 : list
        Represents N11.
    N10 : list
        Represents N10.
    N01 : list
        Represents N01.

    Returns
    -------
    compat : np.array
        Array of booleans representing the compatibility.
    """
    n = n11 + n10 + n01 + n00
    n_t = len(N10)

    rep_left = np.array([0 for i in range(n_t)])
    a = np.array([n11 - i for i in N10])
    b = np.array([i - n01 for i in N11])
    c = np.array([N11[i % len(N11)] + N01[i % len(N01)] - n10 - n01
                  for i in range(max(len(N11), len(N01)))])

    max_len = max(len(a), len(b), len(c), len(rep_left))
    rep_left_bcast = np.array([rep_left[i % len(rep_left)]
                               for i in range(max_len)])
    a_broadcast = np.array([a[i % len(a)] for i in range(max_len)])
    b_broadcast = np.array([b[i % len(b)] for i in range(max_len)])
    c_broadcast = np.array([c[i % len(c)] for i in range(max_len)])

    bl = (np.array([rep_left_bcast, a_broadcast, b_broadcast, c_broadcast]).T)
    left = np.array([max(bl[i]) for i in range(max_len)])

    w = np.array(N11)
    x = np.array([n11 for i in range(n_t)])
    y = np.array([N11[i % len(N11)] + N01[i % len(N01)] - n01
                  for i in range(max(len(N11), len(N01)))])
    z = np.array([n - i - n01 - n10 for i in N10])

    max_len_right = max(len(w), len(x), len(y), len(z))
    w_broadcast = np.array([w[i % len(w)] for i in range(max_len)])
    x_broadcast = np.array([x[i % len(x)] for i in range(max_len)])
    y_broadcast = np.array([y[i % len(y)] for i in range(max_len)])
    z_broadcast = np.array([z[i % len(z)] for i in range(max_len)])

    br = (np.array([w_broadcast, x_broadcast, y_broadcast, z_broadcast]).T)
    right = np.array([min(br[i]) for i in range(max_len)])

    compat = (left <= right)
    return compat


def tau_lower_N11_twoside(n11, n10, n01, n00, N11, Z_all, alpha):
    """
    Calculate tau_min and N_accept for method I.

    Given a set of observed values n and N11, computes different p-values and
    calculates two-sided confidence interval for tau, trying out different
    values of N10, N01, and N00 compatible with N11.

    Parameters
    ----------
    n11 : int
        number of subjects assigned to treatment 1 who had outcome 1
    n10 : int
        number of subjects assigned to treatment 1 who had outcome 0
    n01 : int
        number of subjects assigned to treatment 0 who had outcome 1
    n00 : int
        number of subjects assigned to treatment 0 who had outcome 0
    N11 : int
        chosen fixed value to begin calculating p-value between the differences
    Z_all : np.array
        A matrix of all possible combinations between
        n and m.
    alpha : float
        1 - alpha confidence interval.

    Returns
    -------
    lower_N11_twoside : dict
        Dictionary that contains the lower and upper bounds for tau, as well
        as the allocation of N11, N10, N01, N00 that would give the bounds
        and number of tables considered.
    """
    n = n11 + n10 + n01 + n00
    m = n11 + n10
    tau_obs = n11/m - n01/(n-m)
    ntau_obs = n*n11/m - n*n01/(n-m)

    N10 = 0
    N01_vec0 = np.arange(n-N11+1)[np.arange(n-N11+1) >= -ntau_obs]

    N01 = min(N01_vec0)
    M = np.repeat(np.nan, len(N01_vec0))

    rand_test_num = 0
    while (N10 <= (n-N11-N01)) & (N01 <= (n-N11)):
        if (N10 <= (N01+ntau_obs)):
            pl = pval_two(n, m, np.array([N11, N10, N01, n-(N11+N10+N01)]),
                          Z_all, tau_obs)
            rand_test_num = rand_test_num + 1
            if (pl >= alpha):
                M[N01_vec0 == N01] = N10
                N01 = N01 + 1
            else:
                N10 = N10 + 1
        else:
            M[N01_vec0 == N01] = N10
            N01 = N01 + 1
    if (N01 <= (n-N11)):
        M[N01_vec0 >= N01] = np.floor(N01_vec0[N01_vec0 >= N01]+ntau_obs) + 1

    N11_vec0 = np.repeat(N11, len(N01_vec0))
    N10_vec0 = M

    N11_vec = np.array([])
    N10_vec = np.array([])
    N01_vec = np.array([])

    for i in range(len(N11_vec0)):
        N10_upper = min((n-N11_vec0[i]-N01_vec0[i]),
                        np.floor(N01_vec0[i]+ntau_obs))
        if (N10_vec0[i] <= N10_upper):
            N10_vec = np.append(N10_vec, np.arange(N10_vec0[i],
                                                   N10_upper+1))
            N11_vec = np.append(N11_vec, np.repeat(N11_vec0[i],
                                                   (N10_upper-N10_vec0[i]+1)))
            N01_vec = np.append(N01_vec, np.repeat(N01_vec0[i],
                                                   (N10_upper-N10_vec0[i]+1)))

    compat = check_compatible(n11, n10, n01, n00, N11_vec, N10_vec, N01_vec)

    if (np.sum(compat) > 0):
        N10 = N10_vec[compat]
        N01 = N01_vec[compat]

        tau_min = min(N10 - N01)/n
        acc_pos = np.where(N10-N01 == n*tau_min)[0]
        N_accept_min = np.array([N11, N10[acc_pos][0],
                                 N01[acc_pos][0],
                                 n-(N11+N10[acc_pos][0] + N01[acc_pos][0])])
        tau_max = max(N10 - N01)/n
        acc_pos = np.where(N10-N01 == n*tau_max)[0]
        N_accept_max = np.array([N11, N10[acc_pos][0],
                                 N01[acc_pos][0],
                                 n-(N11+N10[acc_pos][0] + N01[acc_pos][0])])
    else:
        tau_min = np.inf
        N_accept_min = None
        tau_max = -np.inf
        N_accept_max = None
    lower_N11_twoside = {'tau_min': tau_min, 'tau_max': tau_max,
                         'N_accept_min': N_accept_min,
                         'N_accept_max': N_accept_max,
                         'rand_test_num': rand_test_num}
    return lower_N11_twoside


def tau_twoside_lower(n11, n10, n01, n00, alpha, Z_all):
    """
    Calculate taus and N_accepts for method 3.

    Calculate two-sided confidence interval for tau given observed
    values. Function where tau bounds are continuously updated as
    different values for N11 are compared.

    Parameters
    ----------
    n11 : int
        number of subjects assigned to treatment 1 who had outcome 1
    n10 : int
        number of subjects assigned to treatment 1 who had outcome 0
    n01 : int
        number of subjects assigned to treatment 0 who had outcome 1
    n00 : int
        number of subjects assigned to treatment 0 who had outcome 0
    alpha : float
        1 - alpha confidence interval
    nperm : int
        the upper bound of number of randomizations to perform for
        each hypothesis test
    Z_all : np.array
        A matrix of all possible combinations between
        n and m.

    Returns
    -------
    XXX : dict
        Dictionary that contains the lower and upper bounds for tau,
        as well as the allocation of N11, N10, N01, N00 that would give
        the bounds and number of tables considered
    """
    n = n11+n10+n01+n00
    m = n11+n10

    tau_obs = n11/m - n01/(n-m)
    ntau_obs = n*n11/m - n*n01/(n-m)
    tau_min = np.inf
    tau_max = -np.inf
    N_accept_min = None
    N_accept_max = None
    rand_test_total = 0

    for N11 in np.arange(min((n11+n01), n+ntau_obs)):
        N11 = int(N11)
        tau_N11 = tau_lower_N11_twoside(n11, n10, n01, n00, N11, Z_all, alpha)
        rand_test_total = rand_test_total + tau_N11['rand_test_num']
        if (tau_N11['tau_min'] < tau_min):
            N_accept_min = tau_N11['N_accept_min']
        if (tau_N11['tau_max'] > tau_max):
            N_accept_max = tau_N11['N_accept_max']
        tau_min = min(tau_min, tau_N11['tau_min'])
        tau_max = max(tau_max, tau_N11['tau_max'])
    tau_lower = tau_min
    tau_upper = tau_max
    N_accept_lower = N_accept_min
    N_accept_upper = N_accept_max
    twoside_lower = {'tau_lower': tau_lower,
                     'N_accept_lower': N_accept_lower,
                     'tau_upper': tau_upper,
                     'N_accept_upper': N_accept_upper,
                     'rand_test_total': rand_test_total}
    return twoside_lower


def tau_twoside_less_treated(n11, n10, n01, n00, alpha, nperm):
    """
    Calculate two-sided confidence interval for tau.

    Given observedvalues. Function where the decision to use all or some of the
    potential tables is decided based on nperm.
    Parameters
    ----------
    n11 : int
        number of subjects assigned to treatment 1 who had outcome 1
    n10 : int
        number of subjects assigned to treatment 1 who had outcome 0
    n01 : int
        number of subjects assigned to treatment 0 who had outcome 1
    n00 : int
        number of subjects assigned to treatment 0 who had outcome 0
    alpha : float
        1 - alpha confidence interval
    nperm : int
        the upper bound of number of randomizations to perform for
        each hypothesis test
    Returns
    -------
    XXX : dict
        Dictionary that contains the lower and upper bounds for tau,
        as well as the allocation of N11, N10, N01, N00 that would give
        the bounds and number of tables considered
    """
    n = n11+n10+n01+n00
    m = n11+n10
    c = math.factorial(n) // (math.factorial(n - m) * math.factorial(m))
    if (c <= nperm):
        Z_all = nchoosem(n, m)
    else:
        Z_all = comb(n, m, nperm)
    ci_lower = tau_twoside_lower(n11, n10, n01, n00, alpha, Z_all)
    ci_upper = tau_twoside_lower(n10, n11, n00, n01, alpha, Z_all)
    rand_test_total = ci_lower['rand_test_total'] + ci_upper['rand_test_total']

    tau_lower = min(ci_lower['tau_lower'], -1*ci_upper['tau_upper'])
    tau_upper = max(ci_lower['tau_upper'], -1*ci_upper['tau_lower'])

    if (tau_lower == ci_lower['tau_lower']):
        N_accept_lower = ci_lower['N_accept_lower']
    else:
        N_accept_lower = list(np.array(ci_upper['N_accept_upper']).take(
            [3, 2, 1, 0]))

    if (tau_upper == -1*ci_upper['tau_lower']):
        N_accept_upper = list(np.array(ci_upper['N_accept_lower']).take(
            [3, 2, 1, 0]))
    else:
        N_accept_upper = ci_lower['N_accept_upper']

    return {'tau_lower': tau_lower, 'tau_upper': tau_upper,
            'N_accept_lower': N_accept_lower,
            'N_accept_upper': N_accept_upper,
            'rand_test_total': rand_test_total}


def tau_twoside(n11, n10, n01, n00, alpha, nperm):
    """
    Calculate two-sided confidence interval for tau given observed values.

    Parameters
    ----------
    n11 : int
        number of subjects assigned to treatment 1 who had outcome 1
    n10 : int
        number of subjects assigned to treatment 1 who had outcome 0
    n01 : int
        number of subjects assigned to treatment 0 who had outcome 1
    n00 : int
        number of subjects assigned to treatment 0 who had outcome 0
    alpha : float
        1 - alpha confidence interval
    nperm : int
        the upper bound of number of randomizations to perform for each
        hypothesis test
    Returns
    -------
    XXX : dict
        Dictionary that contains the lower and upper bounds for tau,
        as well as the allocation of N11, N10, N01, N00 that would
        give the bounds and number of tables considered
    """
    n = n11+n10+n01+n00
    m = n11+n10
    if (m > (n/2)):
        ci = tau_twoside_less_treated(n01, n00, n11, n10, alpha, nperm)
        tau_lower = -ci['tau_upper']
        tau_upper = -ci['tau_lower']
        N_accept_lower = list(np.array(ci['N_accept_lower']).take(
            [0, 2, 1, 3]))
        N_accept_upper = list(np.array(ci['N_accept_upper']).take(
            [0, 2, 1, 3]))
        rand_test_total = ci['rand_test_total']
    else:
        ci = tau_twoside_less_treated(n11, n10, n01, n00, alpha, nperm)
        tau_lower = ci['tau_lower']
        tau_upper = ci['tau_upper']
        N_accept_lower = list(ci['N_accept_lower'])
        N_accept_upper = list(ci['N_accept_upper'])
        rand_test_total = ci['rand_test_total']
    return {'tau_lower': tau_lower, 'tau_upper': tau_upper,
            'N_accept_lower': N_accept_lower,
            'N_accept_upper': N_accept_upper,
            'rand_test_total': rand_test_total}


def N_generator(N, n00, n01, n10, n11):
    """
    Generate tables algebraically consistent with data from an experiment.

    Parameters
    ----------
    N : int
        number of subjects
    n00 : int
        number of subjects assigned to treatment 0 who had outcome 0
    n01 : int
        number of subjects assigned to treatment 0 who had outcome 0
    n10 : int
        number of subjects assigned to treatment 1 who had outcome 0
    n11 : int
        number of subjects assigned to treatment 1 who had outcome 1
    Returns
    -------
    Nt : list of 4 ints
        N00, subjects with potential outcome 0 under treatments 0 and 1
        N01, subjects with potential outcome 0 under treatment 0
        and 1 under treatment 1
        N10, subjects with potential outcome 1 under treatment 0
        and 0 under treatment 1
        N11, subjects with potential outcome 1 under treatments 0 and 1
    """
    for i in range(min(N-n00, N-n10)+1):
        # allocate space for the observed 0 outcomes, n00 and n10
        N11 = i
        for j in range(max(0, n01-N11), N-n00-N11):
            # N11+N10 >= n01; N11+N10+n00 <= N
            N10 = j
            for k in range(max(0, n11-N11), min(N-n10-N11, N-N11-N10)):
                # N11+N01 >= n11; N11+N01+n10 <= N; no more than N subjects
                N01 = k
                N00 = N-N11-N10-N01
                if filterTable([N00, N01, N10, N11], n00, n01, n10, n11):
                    yield [N00, N01, N10, N11]
                else:
                    pass


def filterTable(Nt, n00, n01, n10, n11):
    """
    Check whether summary table Nt of binary outcomes is consistent.

    Implements the test in Theorem 1 of Li and Ding (2016)
    Parameters:
    ----------
    Nt : list of four ints
        the table of counts of subjects with each combination of
        potential outcomes, in the order N_00, N_01, N_10, N_11
    n01 : int
        number of subjects assigned to control whose observed response was 1
    n11 : int
        number of subjects assigned to treatment whose observed response was 1
    Returns:
    --------
    ok : boolean
        True if table is consistent with the data
    """
    N = np.sum(Nt)   # total subjects
    left_max = max(0, n11-Nt[2], Nt[3]-n01, Nt[2]+Nt[3]-n10-n01)
    right_min = min(Nt[3], n11, Nt[2]+Nt[3]-n01, N-Nt[2]-n01-n10)
    return left_max <= right_min


def potential_outcomes(Nt):
    """
    Make a 2xN table of potential outcomes.

    From the 2x2 summary table Nt

    Parameters
    ----------
    Nt : list of 4 ints
        N00, N01, N10, N11
    Returns
    -------
    po : Nx2 table of potential outcomes consistent with Nt
    """
    po = np.array([0, 0]*Nt[0]+[0, 1]*Nt[1]+[1, 0]*Nt[2]+[1, 1]*Nt[3])
    return np.reshape(po, [-1, 2])


def tau_sampling(N, n, tau_N, outcome_table, Z_all):
    """
    Find or simulate the sampling distribution of |tau_hat - tau_N|.

    Parameters
    ----------
    N : list of 4 ints
        N00, N01, N10, N11
    n : int
        number of subjects under treatment (n11 + n10)
    tau_N : float
        tau value calculated from the current table considered N
    outcome_table: Nx2 table
        table of potential outcomes consistent with N
    Z_all : np.array
        exact or sample from re-randomization matrix
    Returns
    -------
    tau_statistic : array of exact or approximate sampling distribution
    """
    tau_statistic = np.array([])
    for Z in Z_all:
        X0, X1 = 0, 0
        i = 0
        for zi in Z:
            z_val = int(zi)
            if outcome_table[i][z_val] == 1:
                if z_val == 0:
                    # Increment X0
                    X0 += 1
                else:
                    # Increment X1
                    X1 += 1
            i += 1
        t_hat = (X1 / n) - (X0 / (N - n))
        tau_statistic = np.append(tau_statistic,
                                  np.array([abs(t_hat - tau_N)]))
    return tau_statistic


def tau_twosided_ci(n11, n10, n01, n00, alpha, exact=True,
                    max_combinations=10**4, reps=10**3):
    """
    Calculate 1-alpha conf. interval for tau given observed values nxx.

    Parameters
    ----------
    n11 : int
        number of subjects assigned to treatment 1 who had outcome 1
    n10 : int
        number of subjects assigned to treatment 1 who had outcome 0
    n01 : int
        number of subjects assigned to treatment 0 who had outcome 1
    n00 : int
        number of subjects assigned to treatment 0 who had outcome 0
    alpha : float
        1 - alpha confidence interval
    exact : bool
        whether to use all possible treatment assignments for hypo. test
    max_combination: int
        maximum number of algebraically consistent tables to consider
    reps :
        if not exact, how many treatment assignments to use for each table
    Returns
    -------
    list of shape 3x2, where the first list is the lower/upper bound of
    the confidence interval, the second list are the corresponding
    allocations of N that give each, and the third list is the number of
    tables examined and the total reps made across the simulation
    """
    # under treatment
    n = n11 + n10
    # under no treatment
    m = n01 + n00
    N = n + m
    tau_obs = n11/n - n01/(N-n)

    if exact:
        c = math.factorial(N) // (math.factorial(N - m) * math.factorial(m))
        if c > max_combinations:
            raise ValueError
    N_gen = N_generator(N, n00, n01, n10, n11)
    confidence_set = []
    ci_to_N = {}
    limit = max_combinations
    i = 0
    try:
        while i < limit:
            table = next(N_gen)
            N00 = table[3]
            N11 = table[0]
            # Calculate t = |tau* - tau(N)|
            N01 = table[2]
            N10 = table[1]
            tau_N = (N01 - N10) / N
            t = abs(tau_obs - tau_N)
            # Create a full table of potential outcomes consistent
            # with the summary table N
            potential_table = potential_outcomes(table)
            tau_dist = np.array([])
            if exact:
                Z_all = nchoosem(N, m)
            else:
                Z_all = comb(N, m, reps)
            sampling_dist = tau_sampling(N, n, tau_N, potential_table, Z_all)
            if t <= np.percentile(sampling_dist, (1-alpha)*100):
                confidence_set += [tau_N]
                ci_to_N[tau_N] = [N11, N10, N01, N00]
            i += 1
        pass
    except StopIteration:
        pass

    if (m > (N/2)):
        tau_lower = -max(confidence_set)
        tau_upper = -min(confidence_set)
    else:
        tau_lower = min(confidence_set)
        tau_upper = max(confidence_set)
    N_accept_min = ci_to_N[tau_lower]
    N_accept_max = ci_to_N[tau_upper]
    return [[tau_lower, tau_upper], [N_accept_min, N_accept_max],
            [i, i * reps]]
