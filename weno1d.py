"""
1D linear advection equation solved with WENO5 + TVD-RK3.

  u_t + a * u_x = 0,  x in [0, 2*pi],  periodic BC

Usage:
    python weno1d.py
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# WENO5 reconstruction (Jiang & Shu 1996)
# ---------------------------------------------------------------------------

def weno5_reconstruct(v):
    """Return WENO5 left-biased reconstruction at cell right interfaces.

    v : array of shape (N,) with periodic ghost cells already included.
    Returns array of shape (N-4,) — one value per interior interface.
    """
    eps = 1e-6

    # stencil values
    vm2 = v[:-4]; vm1 = v[1:-3]; v0 = v[2:-2]; vp1 = v[3:-1]; vp2 = v[4:]

    # candidate reconstructions (three 3-point stencils)
    q0 = ( 1/3)*vm2 - (7/6)*vm1 + (11/6)*v0
    q1 = (-1/6)*vm1 + (5/6)*v0  + ( 1/3)*vp1
    q2 = ( 1/3)*v0  + (5/6)*vp1 - ( 1/6)*vp2

    # smoothness indicators
    b0 = (13/12)*(vm2 - 2*vm1 + v0)**2 + (1/4)*(vm2 - 4*vm1 + 3*v0)**2
    b1 = (13/12)*(vm1 - 2*v0  + vp1)**2 + (1/4)*(vm1 - vp1)**2
    b2 = (13/12)*(v0  - 2*vp1 + vp2)**2 + (1/4)*(3*v0 - 4*vp1 + vp2)**2

    # ideal weights
    d0, d1, d2 = 0.1, 0.6, 0.3

    # nonlinear weights
    a0 = d0 / (eps + b0)**2
    a1 = d1 / (eps + b1)**2
    a2 = d2 / (eps + b2)**2
    a_sum = a0 + a1 + a2

    w0 = a0 / a_sum
    w1 = a1 / a_sum
    w2 = a2 / a_sum

    return w0*q0 + w1*q1 + w2*q2


def lf_flux(u_left, u_right, a):
    """Lax-Friedrichs numerical flux."""
    return 0.5 * (a*u_left + a*u_right - abs(a)*(u_right - u_left))


def rhs(u, dx, a):
    """Spatial residual  -a * du/dx  using WENO5 + Lax-Friedrichs."""
    N = len(u)
    # periodic extension: 3 ghost cells on each side
    ug = np.concatenate([u[-3:], u, u[:3]])

    u_right = weno5_reconstruct(ug)          # left-biased: u_{i+1/2}^-
    # right-biased reconstruction = mirror WENO5 on reversed stencil
    u_left  = weno5_reconstruct(ug[::-1])[::-1]  # u_{i+1/2}^+

    f = lf_flux(u_right, u_left, a)         # f_{i+1/2}
    return -(f[1:] - f[:-1]) / dx           # -(f_{i+1/2} - f_{i-1/2}) / dx


def rk3(u, dt, dx, a):
    """3rd-order TVD Runge-Kutta (Shu & Osher)."""
    L = lambda v: rhs(v, dx, a)
    u1 = u + dt * L(u)
    u2 = 0.75*u + 0.25*(u1 + dt * L(u1))
    return (1/3)*u + (2/3)*(u2 + dt * L(u2))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    N  = 200
    a  = 1.0
    T  = 2 * np.pi          # one full period
    CFL = 0.5

    x  = np.linspace(0, 2*np.pi, N, endpoint=False)
    dx = x[1] - x[0]
    dt = CFL * dx / abs(a)

    u  = np.sin(x)
    u0 = u.copy()

    t = 0.0
    while t < T:
        dt = min(dt, T - t)
        u  = rk3(u, dt, dx, a)
        t += dt

    u_exact = np.sin(x - a*T)
    l2_err  = np.sqrt(dx * np.sum((u - u_exact)**2))
    print(f"N={N:4d}  L2 error = {l2_err:.3e}")

    plt.figure(figsize=(8, 4))
    plt.plot(x, u_exact, 'k-',  lw=1.5, label='Exact')
    plt.plot(x, u,       'r--', lw=1.5, label=f'WENO5  (N={N})')
    plt.xlabel('x'); plt.ylabel('u')
    plt.title('1D Linear Advection — WENO5 + RK3')
    plt.legend(); plt.tight_layout()
    plt.savefig('weno1d_result.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
