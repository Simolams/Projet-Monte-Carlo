import numpy as np


def simulate_paths_bs(S0, r, q, sigma, T, steps, paths, seed=None, sub=5):
    """
    Sublattice Brownien : chaque pas "jour" est découpé en `sub` sous-pas.
    Retourne un tableau (paths, steps*sub + 1).
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / (steps * sub)
    drift = (r - q - 0.5*sigma*sigma) * dt
    diff  = sigma * np.sqrt(dt)

    Z = np.random.randn(paths, steps*sub)
    log_inc   = drift + diff * Z
    log_paths = np.cumsum(log_inc, axis=1)
    log_paths = np.hstack([np.zeros((paths,1)), log_paths])

    return S0 * np.exp(log_paths)         


def has_continuous_excursion(path, L, U, D, dt):
    consec = 0
    min_steps = int(np.ceil(D/dt))
    for S in path:
        if (S < L) or (S > U):
            consec += 1
            if consec >= min_steps:
                return True
        else:
            consec = 0
    return False


def price_parisian_double_in_call_mc(
        S_paths, K, L, U, r, T, D, num_steps):
    """
    Knock-In : payoff payé si excursion ≥ D hors [L,U] avant T.
    num_steps = pas réels de la grille (S_paths.shape[1]-1)
    """
    dt  = T / num_steps
    pay = []
    for p in S_paths:
        if has_continuous_excursion(p, L, U, D, dt):
            pay.append(max(p[-1] - K, 0.0))
        else:
            pay.append(0.0)
    return np.exp(-r*T) * np.mean(pay)
