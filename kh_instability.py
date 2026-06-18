"""
2D Kelvin-Helmholtz instability — incompressible vorticity-streamfunction.

  omega_t + u * omega_x + v * omega_y = 0
  -laplacian(psi) = omega
  u = psi_y,  v = -psi_x

Solved with pseudo-spectral method (FFT) for the Poisson step
and finite difference + RK3 for advection.

Usage:
    python kh_instability.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def solve_kh(N=256, T=10.0, dt=0.005):
    L  = 2 * np.pi
    dx = L / N
    x  = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing='ij')

    # wavenumbers for spectral Poisson solver
    k  = np.fft.fftfreq(N, d=1.0/N)
    KX, KY = np.meshgrid(k, k, indexing='ij')
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0   # avoid division by zero

    # initial vorticity: two shear layers + small perturbation
    delta = 0.05
    omega = (np.tanh((Y - np.pi/2) / delta) -
             np.tanh((Y - 3*np.pi/2) / delta) - 1.0)
    omega += 0.01 * np.sin(X) * np.exp(-((Y - np.pi/2)**2 +
                                          (Y - 3*np.pi/2)**2) / (2*delta**2))

    def poisson(om):
        """Solve -laplacian(psi) = om spectrally."""
        return np.real(np.fft.ifft2(np.fft.fft2(om) / K2))

    def velocity(om):
        psi = poisson(om)
        u =  np.gradient(psi, dx, axis=1)
        v = -np.gradient(psi, dx, axis=0)
        return u, v

    def L_adv(om):
        u, v = velocity(om)
        domx = np.gradient(om, dx, axis=0)
        domy = np.gradient(om, dx, axis=1)
        return -(u * domx + v * domy)

    def rk3_step(om, dt):
        k1 = L_adv(om)
        k2 = L_adv(om + dt * k1)
        k3 = L_adv(om + 0.25*dt*(k1 + k2))
        return om + dt/6 * (k1 + k2 + 4*k3)

    frames = []
    save_every = int(0.5 / dt)
    t = 0.0
    step = 0
    print(f"KH instability  N={N}  T={T}")
    while t < T:
        omega = rk3_step(omega, dt)
        t += dt; step += 1
        if step % save_every == 0:
            frames.append((t, omega.copy()))
            print(f"  t={t:.2f}")

    return frames, x


def main():
    frames, x = solve_kh(N=256, T=8.0, dt=0.005)

    # static plot grid
    n = min(6, len(frames))
    idx = np.linspace(0, len(frames)-1, n, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    for ax, i in zip(axes, idx):
        t, om = frames[i]
        ax.imshow(om.T, origin='lower', cmap='RdBu_r',
                  extent=[0, 2*np.pi, 0, 2*np.pi])
        ax.set_title(f't = {t:.1f}')
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle('Kelvin-Helmholtz Instability — Vorticity', fontsize=13)
    plt.tight_layout()
    plt.savefig('kh_instability.png', dpi=150)
    plt.show()
    print("Saved kh_instability.png")


if __name__ == '__main__':
    main()
