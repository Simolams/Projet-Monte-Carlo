import numpy as np
import numpy as np
from scipy.special import erf
from numpy import sqrt, exp, log, pi



def simulate_black_scholes_paths(S0, r, q, sigma, T, num_steps, num_simulations, seed=None):
    """
    Simulates asset paths using the Black-Scholes model under risk-neutral measure.

    Returns:
        A NumPy array of shape (num_simulations, num_steps + 1)
    """
    if seed is not None:
        np.random.seed(seed)
        
    dt = T / num_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Generate random normal increments
    Z = np.random.randn(num_simulations, num_steps)
    
    # Compute the log returns and cumulative sum
    log_returns = drift + diffusion * Z
    log_paths = np.cumsum(log_returns, axis=1)
    log_paths = np.hstack([np.zeros((num_simulations, 1)), log_paths])
    
    S_paths = S0 * np.exp(log_paths)
    
    return S_paths


def price_parisian_down_in_call_mc(S_paths, K, L, r, T, D, num_steps):
    """
    Prices a Parisian Down-and-In Call using Monte Carlo simulation.
    
    Args:
        S_paths (np.ndarray): Simulated asset paths (num_simulations x (num_steps + 1))
        K (float): Strike price
        L (float): Barrier level
        r (float): Risk-free rate
        T (float): Time to maturity
        D (float): Minimum excursion time below the barrier (in years)
        num_steps (int): Number of time steps in the simulation

    Returns:
        PDIC price estimate (float)
    """
    dt = T / num_steps
    min_consec_steps = int(D / dt)

    num_simulations = S_paths.shape[0]
    payoffs = np.zeros(num_simulations)

    for i in range(num_simulations):
        path = S_paths[i, :]
        below_barrier = path < L

        # Check for a contiguous sequence of True values of length ≥ min_consec_steps
        consec = 0
        triggered = False
        for flag in below_barrier:
            if flag:
                consec += 1
                if consec >= min_consec_steps:
                    triggered = True
                    break
            else:
                consec = 0

        if triggered:
            ST = path[-1]
            payoffs[i] = max(ST - K, 0)

    discounted_payoffs = np.exp(-r * T) * payoffs
    return np.mean(discounted_payoffs)


def price_parisian_down_out_call_mc(S_paths, K, L, r, T, D, num_steps):
    """
    Prices a Parisian Down-and-Out Call using Monte Carlo simulation.
    
    Args:
        S_paths (np.ndarray): Simulated asset paths (num_simulations x (num_steps + 1))
        K (float): Strike price
        L (float): Barrier level
        r (float): Risk-free rate
        T (float): Time to maturity
        D (float): Minimum excursion time below the barrier (in years)
        num_steps (int): Number of time steps in the simulation

    Returns:
        PDOC price estimate (float)
    """
    dt = T / num_steps
    min_consec_steps = int(D / dt)

    num_simulations = S_paths.shape[0]
    payoffs = np.zeros(num_simulations)

    for i in range(num_simulations):
        path = S_paths[i, :]
        below_barrier = path < L

        consec = 0
        knocked_out = False
        for flag in below_barrier:
            if flag:
                consec += 1
                if consec >= min_consec_steps:
                    knocked_out = True
                    break
            else:
                consec = 0

        if not knocked_out:
            ST = path[-1]
            payoffs[i] = max(ST - K, 0)

    discounted_payoffs = np.exp(-r * T) * payoffs
    return np.mean(discounted_payoffs)

def price_parisian_down_in_call_mc_with_ci(S_paths, K, L, r, T, D, num_steps, confidence=0.95):
    dt = T / num_steps
    min_consec_steps = int(D / dt)
    num_simulations = S_paths.shape[0]
    payoffs = np.zeros(num_simulations)

    for i in range(num_simulations):
        path = S_paths[i, :]
        below_barrier = path < L
        consec = 0
        triggered = False
        for flag in below_barrier:
            if flag:
                consec += 1
                if consec >= min_consec_steps:
                    triggered = True
                    break
            else:
                consec = 0

        if triggered:
            ST = path[-1]
            payoffs[i] = max(ST - K, 0)

    discounted = np.exp(-r * T) * payoffs
    mean = np.mean(discounted)
    std_err = np.std(discounted, ddof=1) / np.sqrt(num_simulations)
    z = norm.ppf(0.5 + confidence / 2)
    ci = (mean - z * std_err, mean + z * std_err)
    return mean, ci


def price_parisian_down_out_call_mc_with_ci(S_paths, K, L, r, T, D, num_steps, confidence=0.95):
    dt = T / num_steps
    min_consec_steps = int(D / dt)
    num_simulations = S_paths.shape[0]
    payoffs = np.zeros(num_simulations)

    for i in range(num_simulations):
        path = S_paths[i, :]
        below_barrier = path < L
        consec = 0
        knocked_out = False
        for flag in below_barrier:
            if flag:
                consec += 1
                if consec >= min_consec_steps:
                    knocked_out = True
                    break
            else:
                consec = 0

        if not knocked_out:
            ST = path[-1]
            payoffs[i] = max(ST - K, 0)

    discounted = np.exp(-r * T) * payoffs
    mean = np.mean(discounted)
    std_err = np.std(discounted, ddof=1) / np.sqrt(num_simulations)
    z = norm.ppf(0.5 + confidence / 2)
    ci = (mean - z * std_err, mean + z * std_err)
    return mean, ci


from scipy.stats import norm

def black_scholes_call_price(S0, K, T, r, q, sigma):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price



#############################    Laplace ####################################################### 

import numpy as np
from scipy.special import erf
from numpy import sqrt, exp, log, pi


# Complex-safe standard normal CDF
def N_complex(z):
    return 0.5 * (1 + erf(z / sqrt(2)))


# ψ(z) function as defined in the paper
def psi(z):
    return 1 + z * sqrt(2 * pi) * exp(z ** 2 / 2) * N_complex(z)


# Laplace transform of PDIC⋆ when K > L
def laplace_transform_pdic_star_K_gt_L(x, K, L, r, delta, sigma, D, lam):
    m = (r - delta - 0.5 * sigma ** 2) / sigma
    theta = sqrt(2 * lam)
    k = (1 / sigma) * log(K / x)
    b = (1 / sigma) * log(L / x)

    numerator = psi(-theta * sqrt(D)) * exp(2 * b * theta)
    denominator = theta * psi(theta * sqrt(D))
    front = K * exp((m - theta) * k)
    bracket = 1 / (m - theta) - 1 / (m + sigma - theta)

    return numerator / denominator * front * bracket


# Inverse Laplace using Euler acceleration (from paper Section 7.3)
def inverse_laplace_transform(f_laplace, t, x, K, L, r, delta, sigma, D, alpha=10, N=15):
    """
    Approximate the inverse Laplace transform using Euler summation as in the paper.
    
    f_laplace: function taking lambda and returning Laplace transform at lambda
    t: maturity time (T)
    alpha: real part for inversion contour
    N: number of terms in Euler summation
    """
    h = pi / t
    summation = 0.5 * f_laplace(x, K, L, r, delta, sigma, D, alpha)
    for k in range(1, N + 1):
        lam = alpha + 1j * k * h
        term = f_laplace(x, K, L, r, delta, sigma, D, lam)
        summation += (-1) ** k * term.real

    f_t = (exp(alpha * t) / t) * summation
    return f_t.real



