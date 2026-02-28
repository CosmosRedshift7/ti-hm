import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from scipy.optimize import root, root_scalar
from scipy.interpolate import interp1d  # <-- NEW
from icecream import ic
from typing import Tuple, Dict

# 1 eV / ħ in rad/s
_EV_TO_RAD_S = 1.519267447e15  # rad/s per eV

# Rakić (1998) Drude parameters (Drude-only piece):
_METAL_RAKIC: Dict[str, Tuple[float, float, float]] = {
    "Ag": (1.0, 9.01 * _EV_TO_RAD_S, 0.048 * _EV_TO_RAD_S),
    "Au": (1.0, 9.03 * _EV_TO_RAD_S, 0.053 * _EV_TO_RAD_S),
    "Al": (1.0, 14.98 * _EV_TO_RAD_S, 0.047 * _EV_TO_RAD_S),
    "Cu": (1.0, 10.83 * _EV_TO_RAD_S, 0.030 * _EV_TO_RAD_S),
    "Ni": (1.0, 15.92 * _EV_TO_RAD_S, 0.048 * _EV_TO_RAD_S),
    "Ti": (1.0, 7.29 * _EV_TO_RAD_S, 0.082 * _EV_TO_RAD_S),
}


def set_metal_rakic(name: str) -> Tuple[float, float, float]:
    """
    Set global (eps_inf, omega_p, gamma0) from Rakić (1998) Drude parameters.
    Returns the tuple so you can also do: eps_inf, omega_p, gamma0 = set_metal_rakic("Ag")
    """
    key = name.strip()
    lut = {k.lower(): k for k in _METAL_RAKIC}
    k = lut.get(key.lower())
    if k is None:
        opts = ", ".join(_METAL_RAKIC.keys())
        raise ValueError(f"Unknown metal '{name}'. Choose one of: {opts}")

    eps_inf, omega_p, gamma = _METAL_RAKIC[k]
    return eps_inf, omega_p, gamma


plt.style.use(["science"])
fig_one_panel = (7.0, 4.2)
fig_two_panel = (7.0, 7.5)
fig_three_panel = (7.0, 10.5)

# =============================
# Global constants
# =============================
c = 299_792_458.0
alpha_fs = 1 / 137.035999084

# EMT and Drude parameters (from manuscript)
eps_d = 2.1
f_m = 0.4

# eps_inf = 5.0
# omega_p = 1.38e16  # rad/s
# gamma_lossy = 5.07e13

eps_inf, omega_p, gamma_lossy = set_metal_rakic("Ag")


# TI permittivity (kept real like manuscript) for lossless figs (2–6)
eps2 = 2.25 + 0j

fmin = 1e-3  # THz (avoid omega=0 singularity)


# =============================
# TI eps2(THz) data -> interpolator (complex)
# =============================
def load_eps2_interpolator(
    csv_path: str,
    kind: str = "linear",
    allow_extrapolate: bool = False,
):
    """
    CSV format:
      freq_thz,eps_real,eps_imag
      90.84...,26.96...,1.43...
      ...

    Returns:
      eps2_of_freq(freq_thz) -> complex array
      eps2_of_omega(omega_rad_s) -> complex array
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    f = np.asarray(data["freq_thz"], dtype=float)
    er = np.asarray(data["eps_real"], dtype=float)
    ei = np.asarray(data["eps_imag"], dtype=float)

    # sort just in case
    idx = np.argsort(f)
    f, er, ei = f[idx], er[idx], ei[idx]

    bounds_error = not allow_extrapolate
    fill = "extrapolate" if allow_extrapolate else np.nan

    er_i = interp1d(f, er, kind=kind, bounds_error=bounds_error, fill_value=fill)
    ei_i = interp1d(f, ei, kind=kind, bounds_error=bounds_error, fill_value=fill)

    def eps2_of_freq(freq_thz):
        freq_thz = np.asarray(freq_thz, dtype=float)
        return er_i(freq_thz) + 1j * ei_i(freq_thz)
        # return er_i(freq_thz) + 0j
        # return eps2

    def eps2_of_omega(omega_rad_s):
        omega_rad_s = np.asarray(omega_rad_s, dtype=float)
        freq_thz = omega_rad_s / (2 * np.pi) / 1e12
        return eps2_of_freq(freq_thz)

    return eps2_of_freq, eps2_of_omega


# =============================
# Material model (parametrized by gamma)
# =============================
def eps_metal(omega: float, gamma: float) -> complex:
    return eps_inf - (omega_p**2) / (omega**2 + 1j * gamma * omega)


def eps_metal_rakic(omega: float, gamma: float) -> complex:
    return eps_inf - (omega_p**2) / (omega**2 + 1j * gamma * omega)


def eps_o_e(omega: float, gamma: float) -> tuple[complex, complex]:
    # em = eps_metal(omega, gamma)
    em = eps_metal_rakic(omega, gamma)
    eps_o = f_m * em + (1.0 - f_m) * eps_d
    eps_e = (eps_d * em) / (f_m * eps_d + (1.0 - f_m) * em)
    return eps_o, eps_e


# =============================
# Helpers (plot styling)
# =============================
def style_ax(ax, xlabel="THz", ylabel=None, xlim=None, ylim=None):
    ax.set_xlabel(xlabel, fontsize=18)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(labelsize=16)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)


# =============================
# Dissipative (complex neff): Fig7–Fig8
# =============================
def csqrt_decay(z: complex) -> complex:
    """
    Choose sqrt branch consistent with decay away from interface:
    prefer Re(sqrt(z)) > 0; if Re==0, prefer Im(sqrt(z)) > 0.
    """
    w = np.sqrt(z)
    if w.real < 0:
        w = -w
    if abs(w.real) < 1e-14 and w.imag < 0:
        w = -w
    return w


def F_disp_complex(
    neff: complex,
    omega: float,
    case: str,
    gamma: float,
    kappa: float,
    eps2_fn_omega=None,  # <-- NEW
) -> complex:
    """
    (q+p1)*(q/eps2 + p2/eps_o) + kappa^2*(p2*q)/(eps_o*eps2) = 0

    case:
      "ty" : l = t_y
      "n"  : l = n

    eps2_fn_omega:
      callable eps2(omega_rad_s) -> complex
      if None, uses global eps2 (constant).
    """
    k0 = omega / c
    eps_o, eps_e = eps_o_e(omega, gamma)

    eps2_loc = eps2 if eps2_fn_omega is None else eps2_fn_omega(omega)

    # If interpolation returns NaN (out of range), fail gracefully
    if not np.isfinite(eps2_loc.real) or not np.isfinite(eps2_loc.imag):
        return np.nan + 1j * np.nan

    q = k0 * csqrt_decay(neff**2 - eps2_loc)

    if case == "ty":
        p1 = k0 * csqrt_decay(neff**2 - eps_e)
        p2 = k0 * csqrt_decay(neff**2 - eps_o)
    elif case == "n":
        p1 = k0 * csqrt_decay(neff**2 - eps_o)
        p2 = k0 * csqrt_decay((eps_o / eps_e) * (neff**2 - eps_e))
    else:
        raise ValueError("case must be 'ty' or 'n'")

    return (q + p1) * (q / eps2_loc + p2 / eps_o) + (kappa**2) * (p2 * q) / (
        eps_o * eps2_loc
    )


def solve_curve_complex(
    freq_thz: np.ndarray,
    case: str,
    neff0: complex,
    gamma: float,
    kappa: float,
    eps2_fn_omega=None,  # <-- NEW
) -> np.ndarray:
    """
    Solve Re(F)=0 and Im(F)=0 for neff(ω) via continuation.
    """
    omega = 2 * np.pi * (freq_thz * 1e12)
    out = np.full(freq_thz.shape, np.nan + 1j * np.nan, dtype=np.complex128)

    neff_guess = neff0

    for i, om in enumerate(omega):

        def fun_xy(xy):
            ne = xy[0] + 1j * xy[1]
            F = F_disp_complex(
                ne,
                om,
                case,
                gamma=gamma,
                kappa=kappa,
                eps2_fn_omega=eps2_fn_omega,
            )
            if not (np.isfinite(F.real) and np.isfinite(F.imag)):
                return np.array([1e30, 1e30], dtype=float)
            return np.array([F.real, F.imag], dtype=float)

        x0 = np.array([neff_guess.real, neff_guess.imag], dtype=float)
        sol = root(fun_xy, x0, method="hybr", tol=1e-12)

        # small rescue if needed
        if not sol.success:
            ok = False
            for k in range(12):
                x_try = x0 + np.array([0.05 * (k + 1), 0.005 * (k + 1)], dtype=float)
                sol2 = root(fun_xy, x_try, method="hybr", tol=1e-12)
                if sol2.success:
                    sol = sol2
                    ok = True
                    break
            if not ok:
                continue

        ne = sol.x[0] + 1j * sol.x[1]

        # enforce attenuation along +z: Im(beta)>0 <=> Im(neff)>0
        if ne.imag < 0:
            ne = ne.real - 1j * ne.imag

        out[i] = ne
        neff_guess = ne

    return out


def plot_fig_reim(freq_thz, neff, fname, xlim=None, ylim=None, title=None):
    # keep full x; break lines at NaNs
    y_re = neff.real.copy()
    y_im = neff.imag.copy()

    # optional: also break if values are non-finite (covers inf too)
    y_re[~np.isfinite(y_re)] = np.nan
    y_im[~np.isfinite(y_im)] = np.nan

    plt.figure(figsize=fig_one_panel)
    if title is not None:
        plt.title(title, fontsize=18)

    plt.plot(
        freq_thz,
        y_re,
        "k-",
        linewidth=3.0,
        label=r"Re($n_{\rm eff}(\nu)$)",
    )
    plt.plot(
        freq_thz,
        y_im,
        "k--",
        linewidth=3.0,
        label=r"Im($n_{\rm eff}(\nu)$)",
    )

    plt.xlabel("THz", fontsize=18)
    plt.ylabel(r"$n_{\rm eff}$", fontsize=18)
    plt.tick_params(labelsize=16)
    plt.legend(loc="best", frameon=True, fontsize=18)

    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print("Saved:", fname)


# =============================
# Lossless (real neff): Fig2–Fig6
# (unchanged; still uses constant eps2)
# =============================
def F_pq_real(
    neff: float, omega: float, kappa: float, case: str, gamma: float
) -> float:
    k0 = omega / c
    eps_o_c, eps_e_c = eps_o_e(omega, gamma)

    eps_o = float(np.real(eps_o_c))
    eps_e = float(np.real(eps_e_c))
    eps2_r = float(np.real(eps2))

    q2 = k0**2 * (neff**2 - eps2_r)
    if q2 <= 0:
        return np.nan
    q = np.sqrt(q2)

    if case == "ty":
        p1_2 = k0**2 * (neff**2 - eps_e)
        p2_2 = k0**2 * (neff**2 - eps_o)
        if p1_2 <= 0 or p2_2 <= 0:
            return np.nan
        p1 = np.sqrt(p1_2)
        p2 = np.sqrt(p2_2)

    elif case == "n":
        p1_2 = k0**2 * (neff**2 - eps_o)
        p2_2 = k0**2 * (eps_o / eps_e) * (neff**2 - eps_e)
        if p1_2 <= 0 or p2_2 <= 0:
            return np.nan
        p1 = np.sqrt(p1_2)
        p2 = np.sqrt(p2_2)

    else:
        raise ValueError("case must be 'ty' or 'n'")

    return (q + p1) * (q / eps2_r + p2 / eps_o) + (kappa**2) * (p2 * q) / (
        eps_o * eps2_r
    )


def solve_neff_curve_real(
    freq_thz: np.ndarray, kappa: float, case: str, neff0: float, gamma: float
) -> np.ndarray:
    omega = 2 * np.pi * (freq_thz * 1e12)
    neff = np.full_like(freq_thz, np.nan, dtype=float)
    guess = neff0
    eps2_r = float(np.real(eps2))

    for i, om in enumerate(omega):
        eps_o_c, eps_e_c = eps_o_e(om, gamma)
        eps_o = float(np.real(eps_o_c))
        eps_e = float(np.real(eps_e_c))

        if case == "ty":
            low2 = max(eps2_r, eps_e, eps_o) + 1e-6
            low = np.sqrt(low2) if low2 > 0 else 1e-3
            high = 40.0
        elif case == "n":
            if eps_e <= eps2_r + 1e-6:
                continue
            low = np.sqrt(eps2_r + 1e-6)
            high = np.sqrt(eps_e - 1e-6)
        else:
            raise ValueError

        def F(x):
            return F_pq_real(x, om, kappa, case, gamma)

        a = max(low, guess * 0.8)
        b = min(high, guess * 1.2)
        if b <= a:
            a, b = low, high

        fa, fb = F(a), F(b)

        if not (np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0):
            xs = np.linspace(low, high, 350)
            Fs = np.array([F(x) for x in xs])
            good = np.isfinite(Fs)
            xs, Fs = xs[good], Fs[good]
            if len(xs) < 5:
                continue

            idx = None
            for j in range(len(xs) - 1):
                if Fs[j] == 0 or Fs[j] * Fs[j + 1] < 0:
                    idx = j
                    break
            if idx is None:
                continue
            a, b = xs[idx], xs[idx + 1]

        try:
            sol = root_scalar(
                F, bracket=(a, b), method="brentq", xtol=1e-12, rtol=1e-12, maxiter=300
            )
            if sol.converged:
                neff[i] = sol.root
                guess = sol.root
        except Exception:
            continue

    return neff


def plot_fig2(gamma: float):
    freq = np.linspace(fmin, 700.0, 600)
    omega = 2 * np.pi * (freq * 1e12)
    eps_o = np.array([np.real(eps_o_e(om, gamma)[0]) for om in omega])
    eps_e = np.array([np.real(eps_o_e(om, gamma)[1]) for om in omega])

    plt.figure(figsize=fig_one_panel)
    plt.plot(freq, eps_o, "k-", linewidth=3.0, label=r"$\varepsilon_o(\nu)$")
    plt.plot(freq, eps_e, "k--", linewidth=3.0, label=r"$\varepsilon_e(\nu)$")
    style_ax(
        plt.gca(), xlabel="THz", ylabel=r"$\varepsilon$", xlim=(0, 700), ylim=(-50, 30)
    )
    plt.legend(loc="best", frameon=True, fontsize=18)
    plt.tight_layout()
    plt.savefig("fig2.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: fig2.pdf")


def csqrt_pos(z: complex) -> complex:
    """sqrt branch with positive real part (or positive imag if purely imag)."""
    w = np.sqrt(z + 0j)
    if w.real < 0:
        w = -w
    if abs(w.real) < 1e-14 and w.imag < 0:
        w = -w
    return w


def plot_fig3(gamma: float):
    freq = np.linspace(fmin, 620.0, 450)
    omega = 2 * np.pi * (freq * 1e12)

    neff = np.full_like(freq, np.nan, dtype=float)
    guess = np.sqrt(np.real(eps2)) + 1e-3  # continuation seed

    for i, om in enumerate(omega):
        k0 = om / c
        eps_o, _ = eps_o_e(om, gamma)

        # For kappa = 0 branch: q/eps2 + p2/eps_o = 0
        def G(x: float) -> float:
            q = k0 * csqrt_pos(x**2 - eps2)
            p2 = k0 * csqrt_pos(x**2 - eps_o)
            val = q / eps2 + p2 / eps_o
            return float(np.real(val))  # should be real on this branch

        low = np.sqrt(np.real(eps2)) + 1e-6
        high = 40.0

        a = max(low, guess * 0.9)
        b = min(high, guess * 1.1)
        if b <= a:
            a, b = low, high

        fa, fb = G(a), G(b)

        if not (np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0):
            xs = np.linspace(low, high, 800)
            Fs = np.array([G(x) for x in xs])
            good = np.isfinite(Fs)
            xs, Fs = xs[good], Fs[good]
            idx = None
            for j in range(len(xs) - 1):
                if Fs[j] * Fs[j + 1] < 0:
                    idx = j
                    break
            if idx is None:
                continue
            a, b = xs[idx], xs[idx + 1]

        sol = root_scalar(
            G, bracket=(a, b), method="brentq", xtol=1e-12, rtol=1e-12, maxiter=300
        )
        if sol.converged:
            neff[i] = sol.root
            guess = sol.root

    plt.figure(figsize=fig_one_panel)
    plt.plot(freq, neff**2, "k-", linewidth=3.0, label=r"$n_{\rm eff}^2(\nu)$")
    style_ax(
        plt.gca(), xlabel="THz", ylabel=r"$n_{\rm eff}^2$", xlim=(0, 620), ylim=(0, 70)
    )
    plt.legend(loc="upper left", frameon=True, fontsize=18)
    plt.tight_layout()
    plt.savefig("fig3.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: fig3.pdf")


def plot_fig4(gamma: float):
    freq = np.linspace(fmin, 700.0, 520)
    omega = 2 * np.pi * (freq * 1e12)
    eps_e = np.array([np.real(eps_o_e(om, gamma)[1]) for om in omega])

    kappa_a = alpha_fs * (2 * 10 + 1)
    kappa_b = alpha_fs * (2 * 100 + 1)

    neff_a = solve_neff_curve_real(
        freq, kappa=kappa_a, case="ty", neff0=3.0, gamma=gamma
    )
    neff_b = solve_neff_curve_real(
        freq, kappa=kappa_b, case="ty", neff0=3.0, gamma=gamma
    )

    fig, axs = plt.subplots(2, 1, figsize=fig_two_panel, sharex=True)

    axs[0].plot(freq, neff_a**2, "k-", linewidth=3.0, label=r"$n_{\rm eff}^2(\nu)$")
    axs[0].plot(freq, eps_e, "k--", linewidth=3.0, label=r"$\varepsilon_e(\nu)$")
    axs[0].set_title(r"(a) $\kappa=(2\times 10+1)\alpha$", fontsize=16)
    style_ax(
        axs[0], xlabel="THz", ylabel=r"$n_{\rm eff}^2$", xlim=(0, 700), ylim=(0, 70)
    )
    axs[0].legend(loc="upper left", frameon=True, fontsize=14)

    axs[1].plot(freq, neff_b**2, "k-", linewidth=3.0, label=r"$n_{\rm eff}^2(\nu)$")
    axs[1].plot(freq, eps_e, "k--", linewidth=3.0, label=r"$\varepsilon_e(\nu)$")
    axs[1].set_title(r"(b) $\kappa=(2\times 100+1)\alpha$", fontsize=16)
    style_ax(
        axs[1], xlabel="THz", ylabel=r"$n_{\rm eff}^2$", xlim=(0, 700), ylim=(0, 70)
    )
    axs[1].legend(loc="upper left", frameon=True, fontsize=14)

    plt.tight_layout()
    plt.savefig("fig4.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: fig4.pdf")


def plot_fig5(gamma: float):
    freq = np.linspace(fmin, 700.0, 520)
    omega = 2 * np.pi * (freq * 1e12)
    eps_e = np.array([np.real(eps_o_e(om, gamma)[1]) for om in omega])
    eps2_line = np.real(eps2) * np.ones_like(freq)

    kappa0 = 0.0
    kappa_b = alpha_fs * (2 * 100 + 1)

    neff0 = solve_neff_curve_real(
        freq, kappa=kappa0, case="n", neff0=np.sqrt(np.real(eps2)) + 1e-3, gamma=gamma
    )
    neffb = solve_neff_curve_real(
        freq, kappa=kappa_b, case="n", neff0=np.sqrt(np.real(eps2)) + 1e-3, gamma=gamma
    )

    fig, axs = plt.subplots(2, 1, figsize=fig_two_panel, sharex=True)

    axs[0].plot(freq, neff0**2, "k-", linewidth=3.0, label=r"$n_{\rm eff}^2(\nu)$")
    axs[0].plot(freq, eps_e, "k--", linewidth=3.0, label=r"$\varepsilon_e(\nu)$")
    axs[0].plot(freq, eps2_line, "k-.", linewidth=3.0, label=r"$\varepsilon_2$")
    axs[0].set_title(r"(a) $\kappa=0$", fontsize=16)
    style_ax(
        axs[0], xlabel="THz", ylabel=r"$n_{\rm eff}^2$", xlim=(0, 700), ylim=(0, 22)
    )
    axs[0].legend(loc="upper left", frameon=True, fontsize=14)

    axs[1].plot(freq, neffb**2, "k-", linewidth=3.0, label=r"$n_{\rm eff}^2(\nu)$")
    axs[1].plot(freq, eps_e, "k--", linewidth=3.0, label=r"$\varepsilon_e(\nu)$")
    axs[1].plot(freq, eps2_line, "k-.", linewidth=3.0, label=r"$\varepsilon_2$")
    axs[1].set_title(r"(b) $\kappa=(2\times 100+1)\alpha$", fontsize=16)
    style_ax(
        axs[1], xlabel="THz", ylabel=r"$n_{\rm eff}^2$", xlim=(0, 700), ylim=(0, 22)
    )
    axs[1].legend(loc="upper left", frameon=True, fontsize=14)

    plt.tight_layout()
    plt.savefig("fig5.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: fig5.pdf")


def plot_fig6(gamma: float):
    freq = np.linspace(fmin, 700.0, 520)
    omega = 2 * np.pi * (freq * 1e12)
    eps_e = np.array([np.real(eps_o_e(om, gamma)[1]) for om in omega])
    eps2_line = np.real(eps2) * np.ones_like(freq)

    kappas = [5.0, 10.0, 20.0]
    titles = [r"(a) $\kappa=5$", r"(b) $\kappa=10$", r"(c) $\kappa=20$"]

    neffs = [
        solve_neff_curve_real(
            freq, kappa=k, case="n", neff0=np.sqrt(np.real(eps2)) + 1e-3, gamma=gamma
        )
        for k in kappas
    ]

    fig, axs = plt.subplots(3, 1, figsize=fig_three_panel, sharex=True)

    for ax, neff, ttl in zip(axs, neffs, titles):
        ax.plot(freq, neff**2, "k-", linewidth=3.0, label=r"$n_{\rm eff}^2(\nu)$")
        ax.plot(freq, eps_e, "k--", linewidth=3.0, label=r"$\varepsilon_e(\nu)$")
        ax.plot(freq, eps2_line, "k-.", linewidth=3.0, label=r"$\varepsilon_2$")
        ax.set_title(ttl, fontsize=16)
        style_ax(
            ax, xlabel="THz", ylabel=r"$n_{\rm eff}^2$", xlim=(0, 700), ylim=(0, 22)
        )
        ax.legend(loc="upper left", frameon=True, fontsize=14)

    plt.tight_layout()
    plt.savefig("fig6.pdf", bbox_inches="tight")
    plt.close()
    print("Saved: fig6.pdf")


def plot_fig_atten_length(freq_thz, neff, fname, xlim=None, ylim=None, title=None):
    """
    Attenuation length along +z:
      beta = k0 * n_eff,  k0 = omega/c
      power decays as exp(-2 Im(beta) z)
      => L_att (1/e power) = 1 / (2 Im(beta))
    """
    omega = 2 * np.pi * (freq_thz * 1e12)
    k0 = omega / c
    beta_im = k0 * np.imag(neff)

    # 1/e power attenuation length (meters)
    L = np.full_like(freq_thz, np.nan, dtype=float)
    good = np.isfinite(beta_im) & (beta_im > 0)
    L[good] = 1.0 / (2.0 * beta_im[good])

    # break lines at gaps automatically
    plt.figure(figsize=fig_one_panel)
    if title is not None:
        plt.title(title, fontsize=18)

    plt.plot(
        freq_thz,
        L,
        "k-",
        linewidth=3.0,
        label=r"$L_{\rm att} = 1/(2\,\mathrm{Im}\,\beta)$",
    )

    plt.xlabel("THz", fontsize=18)
    plt.ylabel(r"$L_{\rm att}$ (m)", fontsize=18)
    plt.tick_params(labelsize=16)
    plt.yscale("log")
    plt.legend(loc="best", frameon=True, fontsize=16)

    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    print("Saved:", fname)


def compute_penetration_depths(
    freq_thz: np.ndarray,
    neff: np.ndarray,
    case: str,
    gamma: float,
    eps2_fn_omega=None,
):
    """
    Compute penetration depths into:
      - TI: q
      - HM: p1 and p2 (as defined by the chosen case)

    Depth definition (lossy case): delta = 1 / Re(decay_const), only if Re>0.
    Returns depths in meters (float arrays), NaNs where undefined.
    """
    omega = 2 * np.pi * (freq_thz * 1e12)
    k0 = omega / c

    d_ti = np.full(freq_thz.shape, np.nan, dtype=float)
    d_p1 = np.full(freq_thz.shape, np.nan, dtype=float)
    d_p2 = np.full(freq_thz.shape, np.nan, dtype=float)

    for i, om in enumerate(omega):
        ne = neff[i]
        if not (np.isfinite(ne.real) and np.isfinite(ne.imag)):
            continue

        eps_o, eps_e = eps_o_e(om, gamma)
        eps2_loc = eps2 if eps2_fn_omega is None else eps2_fn_omega(om)

        if not (np.isfinite(eps2_loc.real) and np.isfinite(eps2_loc.imag)):
            continue

        q = k0[i] * csqrt_decay(ne**2 - eps2_loc)

        if case == "ty":
            p1 = k0[i] * csqrt_decay(ne**2 - eps_e)
            p2 = k0[i] * csqrt_decay(ne**2 - eps_o)
        elif case == "n":
            p1 = k0[i] * csqrt_decay(ne**2 - eps_o)
            p2 = k0[i] * csqrt_decay((eps_o / eps_e) * (ne**2 - eps_e))
        else:
            raise ValueError("case must be 'ty' or 'n'")

        rq = float(np.real(q))
        rp1 = float(np.real(p1))
        rp2 = float(np.real(p2))

        if rq > 0 and np.isfinite(rq):
            d_ti[i] = 1.0 / rq
        if rp1 > 0 and np.isfinite(rp1):
            d_p1[i] = 1.0 / rp1
        if rp2 > 0 and np.isfinite(rp2):
            d_p2[i] = 1.0 / rp2

    return d_ti, d_p1, d_p2


def plot_fig10(
    freq7: np.ndarray,
    neff7: np.ndarray,
    freq8: np.ndarray,
    neff8: np.ndarray,
    gamma_lossy: float,
    eps2_of_omega=None,
    fname: str = "fig10.pdf",
):
    """
    Fig10: 3x2 grid.
      Left col:  l = t_y (case='ty')
      Right col: l = n   (case='n')
    """
    # compute (meters)
    dti7, dp17, dp27 = compute_penetration_depths(
        freq7, neff7, case="ty", gamma=gamma_lossy, eps2_fn_omega=eps2_of_omega
    )
    dti8, dp18, dp28 = compute_penetration_depths(
        freq8, neff8, case="n", gamma=gamma_lossy, eps2_fn_omega=eps2_of_omega
    )

    # convert to nm
    nm = 1e9
    dti7 *= nm
    dp17 *= nm
    dp27 *= nm
    dti8 *= nm
    dp18 *= nm
    dp28 *= nm

    fig, axs = plt.subplots(3, 2, figsize=(10.5, 9.0), sharex=False)

    # --- Row 1: TI ---
    axs[0, 0].plot(freq7, dti7, linewidth=3.0)
    axs[0, 0].set_title(
        r"$\mathbf{l}=\mathbf{t}_y$: $\delta_{\rm TI}=1/\Re(q)$", fontsize=14
    )

    axs[0, 1].plot(freq8, dti8, linewidth=3.0)
    axs[0, 1].set_title(
        r"$\mathbf{l}=\mathbf{n}$: $\delta_{\rm TI}=1/\Re(q)$", fontsize=14
    )

    # --- Row 2: HM p1 ---
    axs[1, 0].plot(freq7, dp17, linewidth=3.0)
    axs[1, 0].set_title(
        r"$\mathbf{l}=\mathbf{t}_y$: $\delta_{1}=1/\Re(p_1)$", fontsize=14
    )

    axs[1, 1].plot(freq8, dp18, linewidth=3.0)
    axs[1, 1].set_title(
        r"$\mathbf{l}=\mathbf{n}$: $\delta_{1}=1/\Re(p_1)$", fontsize=14
    )

    # --- Row 3: HM p2 ---
    axs[2, 0].plot(freq7, dp27, linewidth=3.0)
    axs[2, 0].set_title(
        r"$\mathbf{l}=\mathbf{t}_y$: $\delta_{2}=1/\Re(p_2)$", fontsize=14
    )

    axs[2, 1].plot(freq8, dp28, linewidth=3.0)
    axs[2, 1].set_title(
        r"$\mathbf{l}=\mathbf{n}$: $\delta_{2}=1/\Re(p_2)$", fontsize=14
    )

    # axis labels + formatting
    for r in range(3):
        for ccol in range(2):
            style_ax(axs[r, ccol], xlabel="THz", ylabel="depth (nm)")

    # ---- FIX: force identical freq ranges within each column ----
    xlim_left = (float(freq7[0]), float(freq7[-1]))
    xlim_right = (float(freq8[0]), float(freq8[-1]))

    for r in range(3):
        axs[r, 0].set_xlim(*xlim_left)
        axs[r, 1].set_xlim(*xlim_right)

    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("Saved:", fname)


# =============================
# Main: generate all figures
# =============================
def main():
    # --------- fig1 is your schematic (not generated here) ---------

    # --------- Lossless (fig2–fig6) ---------
    # gamma_lossless = 0.0
    # plot_fig2(gamma_lossless)  # fig2.pdf
    # plot_fig3(gamma_lossless)  # fig3.pdf
    # plot_fig4(gamma_lossless)  # fig4.pdf
    # plot_fig5(gamma_lossless)  # fig5.pdf
    # plot_fig6(gamma_lossless)  # fig6.pdf

    # --------- Dissipative (fig7–fig8) ---------

    # Load TI eps2(freq) data for dissipative runs only
    EPS2_CSV = "eps_vs_freq.csv"  # <-- set your filename here
    eps2_of_freq, eps2_of_omega = load_eps2_interpolator(
        EPS2_CSV,
        kind="linear",  # "cubic" is ok too if the data is smooth/dense
        allow_extrapolate=False,  # safer: outside range => NaNs => solver skips
    )

    # Manuscript convention: theta = 2n+1 (NO pi), kappa = alpha*theta
    n_theta = 100
    theta = 2 * n_theta + 1
    kappa = alpha_fs * theta  # ~ 201/137 ≈ 1.47

    # fig7
    # freq7 = np.linspace(412.0, 488.5, 220)
    freq7 = np.linspace(100.0, 700.0, 200)
    neff7 = solve_curve_complex(
        freq7,
        case="ty",
        neff0=1.0 + 1.0j,
        gamma=gamma_lossy,
        kappa=kappa,
        eps2_fn_omega=eps2_of_omega,  # <-- NEW
    )
    plot_fig_reim(
        freq7,
        neff7,
        fname="fig7.pdf",
        xlim=(freq7[0], freq7[-1]),
        title=r"HM--TI surface wave with dissipation ($\mathbf{l}=\mathbf{t}_y$)",
    )

    # fig8
    # freq8 = np.linspace(50.0, 650.0, 320)
    freq8 = np.linspace(100.0, 700.0, 200)  # or slightly above min
    neff8 = solve_curve_complex(
        freq8,
        case="n",
        neff0=1.0 + 1.0j,
        gamma=gamma_lossy,
        kappa=kappa,
        eps2_fn_omega=eps2_of_omega,  # <-- NEW
    )
    plot_fig_reim(
        freq8,
        neff8,
        fname="fig8.pdf",
        xlim=(freq8[0], freq8[-1]),
        title=r"HM--TI surface wave with dissipation ($\mathbf{l}=\mathbf{n}$)",
    )

    good7 = np.isfinite(neff7.real) & np.isfinite(neff7.imag)
    good8 = np.isfinite(neff8.real) & np.isfinite(neff8.imag)
    print("Converged points: fig7 =", good7.sum(), "/", len(freq7))
    print("Converged points: fig8 =", good8.sum(), "/", len(freq8))

    plot_fig_atten_length(
        freq7,
        neff7,
        fname="fig9.pdf",
        xlim=(freq7[0], freq7[-1]),
        title=r"Attenuation length ($\mathbf{l}=\mathbf{t}_y$)",
    )

    plot_fig10(
        freq7=freq7,
        neff7=neff7,
        freq8=freq8,
        neff8=neff8,
        gamma_lossy=gamma_lossy,
        eps2_of_omega=eps2_of_omega,
        fname="fig10.pdf",
    )


if __name__ == "__main__":
    main()
