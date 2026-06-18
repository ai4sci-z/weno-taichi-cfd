"""
1D inviscid Burgers equation with WENO5 + TVD-RK3 and convergence analysis.

  u_t + (u^2/2)_x = 0,  x in [0, 2*pi],  periodic BC

Usage:
    python burgers1d.py
"""

import numpy as np
import matplotlib.pyplot as plt
from weno1d import weno5_reconstruct


def burgers_flux(u):
    return 0.5 * u**2


def lf_flux_burgers(u_left, u_right):
    alpha = np.max(np.abs(np.concatenate([u_left, u_right])))
    f_l = burgers_flux(u_left)
    f_r = burgers_flux(u_right)
    return 0.5 * (f_l + f_r - alpha * (u_right - u_left))


def rhs_burgers(u, dx):
    N = len(u)
    ug = np.concatenate([u[-3:], u, u[:3]])

    u_right = weno5_reconstruct(ug)
    u_left  = weno5_reconstruct(ug[::-1])[::-1]

    f = lf_flux_burgers(u_right, u_left)
    return -(f[1:] - f[:-1]) / dx


def rk3_burgers(u, dt, dx):
    L = lambda v: rhs_burgers(v, dx)
    u1 = u + dt * L(u)
    u2 = 0.75*u + 0.25*(u1 + dt * L(u1))
    return (1/3)*u + (2/3)*(u2 + dt * L(u2))


def solve(N, T=0.5, CFL=0.4):
    """Solve up to time T (before shock forms at T~1)."""
    x  = np.linspace(0, 2*np.pi, N, endpoint=False)
    dx = x[1] - x[0]
    u  = np.sin(x)
    t  = 0.0
    while t < T:
        alpha = np.max(np.abs(u)) + 1e-10
        dt = min(CFL * dx / alpha, T - t)
        u  = rk3_burgers(u, dt, dx)
        t += dt
    return x, u


def exact_burgers(x, T):
    """Implicit exact solution via Newton iteration (valid before shock)."""
    u = np.sin(x).copy()
    for _ in range(50):
        f  = u - np.sin(x - u * T)
        fp = 1 + T * np.cos(x - u * T)
        u -= f / fp
    return u


def convergence_study():
    T  = 0.5
    Ns = [50, 100, 200, 400, 800]
    errors = []
    for N in Ns:
        x, u_num = solve(N, T)
        u_ex  = exact_burgers(x, T)
        dx    = x[1] - x[0]
        l2    = np.sqrt(dx * np.sum((u_num - u_ex)**2))
        errors.append(l2)
        print(f"N={N:4d}  L2={l2:.3e}")

    # convergence rates
    for i in range(1, len(Ns)):
        rate = np.log(errors[i-1]/errors[i]) / np.log(Ns[i]/Ns[i-1])
        print(f"  N={Ns[i-1]}->{Ns[i]}: order = {rate:.2f}")

    plt.figure(figsize=(6, 4))
    plt.loglog(Ns, errors, 'o-', label='WENO5 L2 error')
    ref = errors[0] * (np.array(Ns) / Ns[0])**(-5)
    plt.loglog(Ns, ref, 'k--', label='5th-order ref')
    plt.xlabel('N'); plt.ylabel('L2 error')
    plt.title('Convergence — 1D Burgers WENO5')
    plt.legend(); plt.tight_layout()
    plt.savefig('burgers_convergence.png', dpi=150)

    plt.figure(figsize=(8, 4))
    x, u_num = solve(400, T)
    u_ex = exact_burgers(x, T)
    plt.plot(x, u_ex,  'k-',  lw=1.5, label='Exact')
    plt.plot(x, u_num, 'r--', lw=1.5, label='WENO5 N=400')
    plt.xlabel('x'); plt.ylabel('u')
    plt.title(f'1D Burgers  T={T}')
    plt.legend(); plt.tight_layout()
    plt.savefig('burgers_solution.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    convergence_study()
