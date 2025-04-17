import numpy as np
import numpy as np
from scipy.special import erf
from numpy import sqrt, exp, log, pi






def price_parisian_up_in_call_mc(S_paths, K, H, r, T, D, num_steps):
    """
    Prices a Parisian Up-and-In Call option using Monte Carlo simulation.

    Args:
        S_paths (np.ndarray): Simulated paths (num_simulations x num_steps+1)
        K (float): Strike price
        H (float): Upper barrier
        r (float): Risk-free rate
        T (float): Time to maturity
        D (float): Required continuous time above the barrier (in years)
        num_steps (int): Number of time steps in each path

    Returns:
        float: Estimated PUIC option price
    """
    dt = T / num_steps
    min_consec_steps = int(D / dt)
    num_simulations = S_paths.shape[0]
    payoffs = np.zeros(num_simulations)

    for i in range(num_simulations):
        path = S_paths[i, :]
        above_barrier = path > H

        consec = 0
        activated = False
        for flag in above_barrier:
            if flag:
                consec += 1
                if consec >= min_consec_steps:
                    activated = True
                    break
            else:
                consec = 0

        if activated:
            ST = path[-1]
            payoffs[i] = max(ST - K, 0)

    discounted_payoffs = np.exp(-r * T) * payoffs
    return np.mean(discounted_payoffs)



#  Laplace transform for PUIC★ using the formula from the image (K <= L)

import numpy as np
from scipy.special import erf
from numpy import sqrt, exp, log, pi


# Complex-safe standard normal CDF
def N_complex(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


# ψ(z) function as defined in the paper
def psi(z):
    return 1 + z * sqrt(2 * pi) * exp(z ** 2 / 2) * N_complex(z)


def laplace_transform_puic_star_K_leq_L(x, K, L, r, delta, sigma, D, lam):
    m = (r - delta - 0.5 * sigma**2) / sigma
    theta = sqrt(2 * lam)
    k = (1 / sigma) * log(K / x)
    b = (1 / sigma) * log(L / x)

    # psi evaluations
    psi_theta = psi(theta * sqrt(D))
    psi_neg_theta = psi(-theta * sqrt(D))
    psi_m = psi(m * sqrt(D))
    psi_m_sigma = psi((m + sigma) * sqrt(D))

    # First term
    term1_num = 2 * exp((m - theta) * b)
    term1_den = psi_theta
    A = (K / (m**2 - theta**2)) * psi_m
    B = (L / ((m + sigma)**2 - theta**2)) * psi_m_sigma
    term1 = (term1_num / term1_den) * (A - B)

    # Second term
    term2_num = exp(-2 * b * theta) * psi_neg_theta
    term2_den = theta * psi_theta
    factor = K * exp((m + theta) * k)
    bracket = (1 / (m + theta)) - (1 / (m + theta + sigma))
    term2 = (term2_num / term2_den) * factor * bracket

    return term1 + term2


# Rerun inverse Laplace transform for PUIC★ with smaller alpha
def inverse_laplace_transform_puic_K_leq_H(t, x, K, H, r, delta, sigma, D, alpha=2.0, N=15):
    h = pi / t
    summation = 0.5 * laplace_transform_puic_star_K_leq_L(x, K, H, r, delta, sigma, D, alpha)
    for k in range(1, N + 1):
        lam = alpha + 1j * k * h
        term = laplace_transform_puic_star_K_leq_L(x, K, H, r, delta, sigma, D, lam)
        summation += (-1) ** k * term.real
    f_t = (exp(alpha * t) / t) * summation
    return f_t.real
