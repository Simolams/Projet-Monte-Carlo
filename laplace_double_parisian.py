

import numpy as np
from math import log, pi, exp
from scipy.special import erf, ndtr       

from PDIC_pricing import laplace_transform_pdic_star_K_gt_L
from PUIC_pricing import laplace_transform_puic_star_K_leq_L

from scipy.stats import norm

def black_scholes_call_star(x, K, T, r, q, sigma):
    # prix "étoilé" du call vanille
    from scipy.stats import norm
    if T == 0:                       # garde-fou
        return max(x - K, 0.0)
    d1 = (np.log(x/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = (x*np.exp(-q*T)*norm.cdf(d1)
            - K*np.exp(-r*T)*norm.cdf(d2))
    m = (r - q - 0.5*sigma**2)/sigma
    return np.exp((r+0.5*m*m)*T) * call



def psi(z: complex):
    return 1 + z * np.sqrt(2 * pi) * np.exp(z*z / 2) * ndtr(z)


def H_closed(z, theta, D):

    s = np.sqrt(D)
    t1 = (D/theta) * np.exp(-theta*z + 0.5*theta*theta*D) * (1 + ndtr(z/s - theta*s))
    t2 = (D/theta) * np.exp( theta*z + 0.5*theta*theta*D) * (1 - ndtr(z/s + theta*s))
    return t1 + t2



def laplace_transform_double_in_star(x, K, L, U,
                                     r, q, sigma, D,
                                     lam,
                                     y_pad=15.0):

    pdic_star = laplace_transform_pdic_star_K_gt_L(x, K, L, r, q, sigma, D, lam)
    puic_star = laplace_transform_puic_star_K_leq_L(x, K, U, r, q, sigma, D, lam)


    m  = (r - q - 0.5*sigma*sigma) / sigma
    θ  = np.sqrt(2 * lam)          
    k  = log(K / x) / sigma
    b1 = log(L / x) / sigma
    b2 = log(U / x) / sigma

    absθ  = np.abs(θ)
    y_max = k + max(y_pad, 8*np.sqrt(D)*absθ.real)
    Ny    = int(max(400, 40*absθ.real))
    y     = np.linspace(k, y_max, Ny)

    Hy = H_closed(y - b2, θ, D)

    A1 = np.trapz(np.exp(m*y) * (x*np.exp(sigma*y) - K) * Hy, y)

    factor = (np.exp((2*b1 - b2)*θ)
              * psi(θ*np.sqrt(D))
              / (θ * psi(-θ*np.sqrt(D))))

    return pdic_star + puic_star - factor*A1


def inverse_laplace_transform(f_hat, t, *args, alpha=10.0, N=30):

    h  = pi / t
    S  = 0.5 * f_hat(*args, lam=alpha)
    for k in range(1, N+1):
        lam = alpha + 1j * k * h
        S  += (-1)**k * f_hat(*args, lam=lam).real
    return exp(alpha*t) * S / t



def price_parisian_double_in_call_lt(x, K, L, U,
                                     r, q, sigma,
                                     T, D,
                                     alpha=10.0, N=30):

    star = inverse_laplace_transform(
        laplace_transform_double_in_star, T,
        x, K, L, U, r, q, sigma, D,
        alpha=alpha, N=N
    )
    m    = (r - q - 0.5*sigma*sigma) / sigma
    disc = np.exp(-(r + 0.5*m*m) * T)
    return disc * star

def price_parisian_double_out_call_lt(x, K, L, U,
                                      r, q, sigma,
                                      T, D, alpha=10.0, N=30):
    # prix Knock-Out = call vanille - knock-in
    C_vanille = black_scholes_call_star(x, K, T, r, q, sigma)
    C_in_star = inverse_laplace_transform(
                    laplace_transform_double_in_star, T,
                    x, K, L, U, r, q, sigma, D,
                    alpha=alpha, N=N)
    m = (r - q - 0.5*sigma*sigma)/sigma
    disc = np.exp(-(r + 0.5*m*m)*T)
    return disc * (C_vanille - C_in_star)

