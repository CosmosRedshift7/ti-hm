import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

from scipy.optimize import root, root_scalar
from scipy.interpolate import interp1d
from typing import Tuple, Dict, Callable, Optional

EV2RADS = 1.519267447e15  # rad/s per eV

# Rakic (1998) Drude parameters:
_METAL_RAKIC: Dict[str, Tuple[float, float, float]] = {
    "Ag": (1.0, 9.01 * EV2RADS, 0.048 * EV2RADS),
    "Au": (1.0, 9.03 * EV2RADS, 0.053 * EV2RADS),
    "Al": (1.0, 14.98 * EV2RADS, 0.047 * EV2RADS),
    "Cu": (1.0, 10.83 * EV2RADS, 0.030 * EV2RADS),
    "Ni": (1.0, 15.92 * EV2RADS, 0.048 * EV2RADS),
    "Ti": (1.0, 7.29 * EV2RADS, 0.082 * EV2RADS),
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

hm_metal = "Ti"
hm_dielectric = "eps_vs_freq_si.csv"
ti_material = "eps_vs_freq_bi2se3.csv"

c = 299_792_458.0
eps0 = 8.854_187_8128e-12
alpha_fs = 1 / 137.035999084

eps_d = 4.6
f_m = 0.4

eps_inf, omega_p, gamma_lossy = set_metal_rakic(hm_metal)

eps2 = 2.25 + 0j

fmin = 1e-3  # THz (avoid omega=0 singularity)


def load_eps_interpolator(
    csv_path: str,
    kind: str = "linear",
    allow_extrapolate: bool = False,
):
    """
    CSV format (comma-separated, with header):
      freq_thz,eps_real,eps_imag
      59.89,1.79,0.012
      ...

    Returns:
      eps_of_freq(freq_thz) -> complex array
      eps_of_omega(omega_rad_s) -> complex array
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    f = np.asarray(data["freq_thz"], dtype=float)
    er = np.asarray(data["eps_real"], dtype=float)
    ei = np.asarray(data["eps_imag"], dtype=float)

    idx = np.argsort(f)
    f, er, ei = f[idx], er[idx], ei[idx]

    bounds_error = not allow_extrapolate
    fill = "extrapolate" if allow_extrapolate else np.nan

    er_i = interp1d(f, er, kind=kind, bounds_error=bounds_error, fill_value=fill)
    ei_i = interp1d(f, ei, kind=kind, bounds_error=bounds_error, fill_value=fill)

    def eps_of_freq(freq_thz):
        freq_thz = np.asarray(freq_thz, dtype=float)
        return er_i(freq_thz) + 1j * ei_i(freq_thz)

    def eps_of_omega(omega_rad_s):
        omega_rad_s = np.asarray(omega_rad_s, dtype=float)
        freq_thz = omega_rad_s / (2 * np.pi) / 1e12
        return eps_of_freq(freq_thz)

    return eps_of_freq, eps_of_omega


def eps_metal(omega: float, gamma: float) -> complex:
    return eps_inf - (omega_p**2) / (omega**2 + 1j * gamma * omega)


def eps_o_e(
    omega: float,
    gamma: float,
    eps_d_fn_omega: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> tuple[complex, complex]:
    em = eps_metal(omega, gamma)

    if eps_d_fn_omega is None:
        eps_d_loc = eps_d
    else:
        eps_d_loc = eps_d_fn_omega(omega)
        if not (np.isfinite(np.real(eps_d_loc)) and np.isfinite(np.imag(eps_d_loc))):
            return np.nan + 1j * np.nan, np.nan + 1j * np.nan

    eps_o = f_m * em + (1.0 - f_m) * eps_d_loc
    eps_e = (eps_d_loc * em) / (f_m * eps_d_loc + (1.0 - f_m) * em)
    return eps_o, eps_e


def style_ax(ax, xlabel="THz", ylabel=None, xlim=None, ylim=None):
    ax.set_xlabel(xlabel, fontsize=18)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(labelsize=16)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)


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
    eps2_fn_omega=None,
    eps_d_fn_omega=None,
) -> complex:
    """
    (q+p1)*(q/eps2 + p2/eps_o) + kappa^2*(p2*q)/(eps_o*eps2) = 0
    """
    k0 = omega / c
    eps_o, eps_e = eps_o_e(omega, gamma, eps_d_fn_omega=eps_d_fn_omega)

    if not (np.isfinite(eps_o.real) and np.isfinite(eps_o.imag)):
        return np.nan + 1j * np.nan
    if not (np.isfinite(eps_e.real) and np.isfinite(eps_e.imag)):
        return np.nan + 1j * np.nan

    eps2_loc = eps2 if eps2_fn_omega is None else eps2_fn_omega(omega)

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
    eps2_fn_omega=None,
    eps_d_fn_omega=None,
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
                eps_d_fn_omega=eps_d_fn_omega,
            )
            if not (np.isfinite(F.real) and np.isfinite(F.imag)):
                return np.array([1e30, 1e30], dtype=float)
            return np.array([F.real, F.imag], dtype=float)

        x0 = np.array([neff_guess.real, neff_guess.imag], dtype=float)
        sol = root(fun_xy, x0, method="hybr", tol=1e-12)

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

        if ne.imag < 0:
            ne = ne.real - 1j * ne.imag

        out[i] = ne
        neff_guess = ne

    return out


def plot_fig_reim(freq_thz, neff, fname, xlim=None, ylim=None, title=None):
    y_re = neff.real.copy()
    y_im = neff.imag.copy()

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
    guess = np.sqrt(np.real(eps2)) + 1e-3

    for i, om in enumerate(omega):
        k0 = om / c
        eps_o, _ = eps_o_e(om, gamma)

        def G(x: float) -> float:
            q = k0 * csqrt_pos(x**2 - eps2)
            p2 = k0 * csqrt_pos(x**2 - eps_o)
            val = q / eps2 + p2 / eps_o
            return float(np.real(val))

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


def plot_fig_atten_length(
    freq7_thz,
    neff7,
    freq8_thz,
    neff8,
    fname,
    xlim_left=None,
    xlim_right=None,
    ylim=None,
    title_left=None,
    title_right=None,
):
    """
    Fig9: 1x2 subplots: attenuation length along +z for two cases.

    Attenuation length along +z:
      beta = k0 * n_eff,  k0 = omega/c
      power decays as exp(-2 Im(beta) z)
      => L_att (1/e power) = 1 / (2 Im(beta))
    """

    def _atten_len(freq_thz, neff):
        omega = 2 * np.pi * (freq_thz * 1e12)
        k0 = omega / c
        beta_im = k0 * np.imag(neff)

        L = np.full_like(freq_thz, np.nan, dtype=float)
        good = np.isfinite(beta_im) & (beta_im > 0)
        L[good] = 1.0 / (2.0 * beta_im[good])
        return L

    L7 = _atten_len(freq7_thz, neff7)  # l = t_y
    L8 = _atten_len(freq8_thz, neff8)  # l = n

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)

    # ---- left: l = t_y ----
    axs[0].plot(
        freq7_thz, L7, linewidth=3.0, label=r"$L_{\rm att} = 1/(2\,\mathrm{Im}\,\beta)$"
    )
    if title_left is not None:
        axs[0].set_title(title_left, fontsize=18)
    axs[0].set_xlabel("THz", fontsize=18)
    axs[0].set_ylabel(r"$L_{\rm att}$ (m)", fontsize=18)
    axs[0].tick_params(labelsize=16)
    axs[0].set_yscale("log")
    axs[0].legend(loc="best", frameon=True, fontsize=14)
    if xlim_left:
        axs[0].set_xlim(*xlim_left)

    # ---- right: l = n ----
    axs[1].plot(
        freq8_thz, L8, linewidth=3.0, label=r"$L_{\rm att} = 1/(2\,\mathrm{Im}\,\beta)$"
    )
    if title_right is not None:
        axs[1].set_title(title_right, fontsize=18)
    axs[1].set_xlabel("THz", fontsize=18)
    axs[1].tick_params(labelsize=16)
    axs[1].set_yscale("log")
    axs[1].legend(loc="best", frameon=True, fontsize=14)
    if xlim_right:
        axs[1].set_xlim(*xlim_right)

    if ylim:
        axs[0].set_ylim(*ylim)
        axs[1].set_ylim(*ylim)

    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("Saved:", fname)


def compute_penetration_depths(
    freq_thz: np.ndarray,
    neff: np.ndarray,
    case: str,
    gamma: float,
    eps2_fn_omega=None,
    return_decays: bool = False,
):
    """
    Compute penetration depths into:
      - TI: q
      - HM: p1 and p2 (as defined by the chosen case)

    Depth definition (lossy case): delta = 1 / Re(decay_const), only if Re>0.
    Returns depths in meters (float arrays), NaNs where undefined.

    If return_decays=True, also returns complex arrays (q_arr, p1_arr, p2_arr).
    """
    omega = 2 * np.pi * (freq_thz * 1e12)
    k0 = omega / c

    d_ti = np.full(freq_thz.shape, np.nan, dtype=float)
    d_p1 = np.full(freq_thz.shape, np.nan, dtype=float)
    d_p2 = np.full(freq_thz.shape, np.nan, dtype=float)

    q_arr = np.full(freq_thz.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    p1_arr = np.full(freq_thz.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    p2_arr = np.full(freq_thz.shape, np.nan + 1j * np.nan, dtype=np.complex128)

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

        q_arr[i] = q
        p1_arr[i] = p1
        p2_arr[i] = p2

        rq = float(np.real(q))
        rp1 = float(np.real(p1))
        rp2 = float(np.real(p2))

        if rq > 0 and np.isfinite(rq):
            d_ti[i] = 1.0 / rq
        if rp1 > 0 and np.isfinite(rp1):
            d_p1[i] = 1.0 / rp1
        if rp2 > 0 and np.isfinite(rp2):
            d_p2[i] = 1.0 / rp2

    if return_decays:
        return d_ti, d_p1, d_p2, q_arr, p1_arr, p2_arr

    return d_ti, d_p1, d_p2


def compute_te_tm_mixing_ratio(
    freq_thz: np.ndarray,
    neff: np.ndarray,
    case: str,
    gamma: float,
    kappa: float,
    eps2_fn_omega=None,
    eps_d_fn_omega=None,
) -> np.ndarray:
    omega = 2 * np.pi * (freq_thz * 1e12)
    eta = np.full(freq_thz.shape, np.nan, dtype=float)

    for i, om in enumerate(omega):
        ne = neff[i]
        if not (np.isfinite(ne.real) and np.isfinite(ne.imag)):
            continue

        k0 = om / c
        beta = k0 * ne
        if abs(beta) == 0:
            continue

        eps_o, eps_e = eps_o_e(om, gamma, eps_d_fn_omega=eps_d_fn_omega)
        eps2_loc = eps2 if eps2_fn_omega is None else eps2_fn_omega(om)

        if not (
            np.isfinite(eps_o.real)
            and np.isfinite(eps_o.imag)
            and np.isfinite(eps_e.real)
            and np.isfinite(eps_e.imag)
            and np.isfinite(eps2_loc.real)
            and np.isfinite(eps2_loc.imag)
        ):
            continue

        q = k0 * csqrt_decay(ne**2 - eps2_loc)
        if case == "ty":
            p1 = k0 * csqrt_decay(ne**2 - eps_e)
            p2 = k0 * csqrt_decay(ne**2 - eps_o)
        elif case == "n":
            p1 = k0 * csqrt_decay(ne**2 - eps_o)
            p2 = k0 * csqrt_decay((eps_o / eps_e) * (ne**2 - eps_e))
        else:
            raise ValueError("case must be 'ty' or 'n'")

        if np.real(q) <= 0 or np.real(p1) <= 0 or np.real(p2) <= 0:
            continue

        te_over_tm = -kappa * p2 / (eps_o * (q + p1))

        i_te = (1.0 / (2.0 * np.real(q))) + (1.0 / (2.0 * np.real(p1)))

        tm_ti = (1.0 + abs(q / beta) ** 2) / (2.0 * np.real(q))
        tm_hm = (1.0 + abs(p2 / beta) ** 2) / (2.0 * np.real(p2))
        i_tm = tm_ti + tm_hm

        if i_tm <= 0 or not np.isfinite(i_tm):
            continue

        eta_val = abs(te_over_tm) ** 2 * i_te / i_tm
        if np.isfinite(eta_val):
            eta[i] = float(np.real(eta_val))

    return eta


def plot_fig_te_tm_mixing(
    freq7: np.ndarray,
    eta7: np.ndarray,
    freq8: np.ndarray,
    eta8: np.ndarray,
    fname: str = "fig11.pdf",
):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)

    axs[0].plot(freq7, eta7, "k-", linewidth=3.0)
    axs[0].set_title(r"$\mathbf{l}=\mathbf{t}_y$", fontsize=16)
    style_ax(axs[0], xlabel="THz", ylabel=r"$\eta(\nu)$")
    axs[0].set_yscale("log")
    axs[0].set_xlim(float(freq7[0]), float(freq7[-1]))

    axs[1].plot(freq8, eta8, "k-", linewidth=3.0)
    axs[1].set_title(r"$\mathbf{l}=\mathbf{n}$", fontsize=16)
    style_ax(axs[1], xlabel="THz", ylabel=None)
    axs[1].set_yscale("log")
    axs[1].set_xlim(float(freq8[0]), float(freq8[-1]))

    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("Saved:", fname)


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

    Primary y-axis (left): penetration depth (nm)
    Secondary y-axis (right): Re/Im of the corresponding decay constant:
      row 1 -> q
      row 2 -> p1
      row 3 -> p2
    """
    (dti7, dp17, dp27, q7, p17, p27) = compute_penetration_depths(
        freq7,
        neff7,
        case="ty",
        gamma=gamma_lossy,
        eps2_fn_omega=eps2_of_omega,
        return_decays=True,
    )

    (dti8, dp18, dp28, q8, p18, p28) = compute_penetration_depths(
        freq8,
        neff8,
        case="n",
        gamma=gamma_lossy,
        eps2_fn_omega=eps2_of_omega,
        return_decays=True,
    )

    # convert depths to nm
    nm = 1e9
    dti7 *= nm
    dp17 *= nm
    dp27 *= nm
    dti8 *= nm
    dp18 *= nm
    dp28 *= nm

    fig, axs = plt.subplots(3, 2, figsize=(12.5, 10.5), sharex=False)
    axr = np.empty_like(axs, dtype=object)

    for r in range(3):
        for c in range(2):
            axr[r, c] = axs[r, c].twinx()

    def _clean_real(y):
        y = np.asarray(np.real(y), dtype=float).copy()
        y[~np.isfinite(y)] = np.nan
        return y

    def _clean_imag(y):
        y = np.asarray(np.imag(y), dtype=float).copy()
        y[~np.isfinite(y)] = np.nan
        return y

    def _clean_float(y):
        y = np.asarray(y, dtype=float).copy()
        y[~np.isfinite(y)] = np.nan
        return y

    def _plot_depth(ax, freq, depth_nm, title):
        ax.plot(freq, _clean_float(depth_nm), color="k", linewidth=3.0, label="depth")
        ax.set_title(title, fontsize=14)
        ax.legend(loc="upper left", frameon=True, fontsize=10)

    def _plot_decay(ax, freq, decay_arr, symbol_label):
        ax.plot(
            freq,
            _clean_real(decay_arr),
            color="tab:blue",
            linewidth=2.0,
            label=rf"$\Re({symbol_label})$",
        )
        ax.plot(
            freq,
            _clean_imag(decay_arr),
            color="tab:orange",
            linewidth=2.0,
            linestyle="--",
            label=rf"$\Im({symbol_label})$",
        )
        ax.set_ylabel(r"decay const. (1/m)", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    _plot_depth(
        axs[0, 0], freq7, dti7, r"$\mathbf{l}=\mathbf{t}_y$: $\delta_{\rm TI}=1/\Re(q)$"
    )
    _plot_decay(axr[0, 0], freq7, q7, "q")

    _plot_depth(
        axs[0, 1], freq8, dti8, r"$\mathbf{l}=\mathbf{n}$: $\delta_{\rm TI}=1/\Re(q)$"
    )
    _plot_decay(axr[0, 1], freq8, q8, "q")

    _plot_depth(
        axs[1, 0], freq7, dp17, r"$\mathbf{l}=\mathbf{t}_y$: $\delta_{1}=1/\Re(p_1)$"
    )
    _plot_decay(axr[1, 0], freq7, p17, "p_1")

    _plot_depth(
        axs[1, 1], freq8, dp18, r"$\mathbf{l}=\mathbf{n}$: $\delta_{1}=1/\Re(p_1)$"
    )
    _plot_decay(axr[1, 1], freq8, p18, "p_1")

    _plot_depth(
        axs[2, 0], freq7, dp27, r"$\mathbf{l}=\mathbf{t}_y$: $\delta_{2}=1/\Re(p_2)$"
    )
    _plot_decay(axr[2, 0], freq7, p27, "p_2")

    _plot_depth(
        axs[2, 1], freq8, dp28, r"$\mathbf{l}=\mathbf{n}$: $\delta_{2}=1/\Re(p_2)$"
    )
    _plot_decay(axr[2, 1], freq8, p28, "p_2")

    for r in range(3):
        for c in range(2):
            style_ax(axs[r, c], xlabel="THz", ylabel="depth (nm)")

    xlim_left = (float(freq7[0]), float(freq7[-1]))
    xlim_right = (float(freq8[0]), float(freq8[-1]))
    for r in range(3):
        axs[r, 0].set_xlim(*xlim_left)
        axs[r, 1].set_xlim(*xlim_right)
        axr[r, 0].set_xlim(*xlim_left)
        axr[r, 1].set_xlim(*xlim_right)

    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("Saved:", fname)


def compute_poynting_profile(
    freq_thz: np.ndarray,
    neff: np.ndarray,
    case: str,
    gamma: float,
    eps2_fn_omega=None,
    eps_d_fn_omega=None,
    nfreq_samples: int = 3,
    x_nm_max: float = 350.0,
    nx_each_side: int = 220,
):
    """
    Build S_z(x) profiles using a TM-like estimate:
      S_z = 0.5 * Re(E_x H_y*)
      E_x ≈ beta/(omega*eps0*eps) * H_y
    with H_y decaying away from the interface via q (TI) and p2 (HM).

    Returns
      x_nm, list[(freq_thz, Sz_norm)]
    where x_nm spans HM (x<0) and TI (x>0).
    """
    valid = np.where(np.isfinite(neff.real) & np.isfinite(neff.imag))[0]
    if len(valid) == 0:
        return np.array([]), []

    if len(valid) <= nfreq_samples:
        sample_ids = valid
    else:
        t = np.linspace(0, len(valid) - 1, nfreq_samples)
        sample_ids = valid[np.round(t).astype(int)]

    x_ti = np.linspace(0.0, x_nm_max * 1e-9, nx_each_side)
    x_hm = np.linspace(-x_nm_max * 1e-9, 0.0, nx_each_side, endpoint=False)
    x_m = np.concatenate([x_hm, x_ti])
    x_nm = x_m * 1e9

    curves = []
    for idx in sample_ids:
        om = 2 * np.pi * freq_thz[idx] * 1e12
        k0 = om / c
        beta = k0 * neff[idx]
        eps_o, eps_e = eps_o_e(om, gamma, eps_d_fn_omega=eps_d_fn_omega)
        eps2_loc = eps2 if eps2_fn_omega is None else eps2_fn_omega(om)

        if case == "ty":
            p2 = k0 * csqrt_decay(neff[idx] ** 2 - eps_o)
        elif case == "n":
            p2 = k0 * csqrt_decay((eps_o / eps_e) * (neff[idx] ** 2 - eps_e))
        else:
            raise ValueError("case must be 'ty' or 'n'")
        q = k0 * csqrt_decay(neff[idx] ** 2 - eps2_loc)

        if np.real(q) <= 0 or np.real(p2) <= 0:
            continue

        hy_ti = np.exp(-q * x_ti)
        hy_hm = np.exp(p2 * x_hm)

        ex_ti = (beta / (om * eps0 * eps2_loc)) * hy_ti
        ex_hm = (beta / (om * eps0 * eps_o)) * hy_hm

        sz_ti = 0.5 * np.real(ex_ti * np.conj(hy_ti))
        sz_hm = 0.5 * np.real(ex_hm * np.conj(hy_hm))
        sz = np.concatenate([sz_hm, sz_ti])

        if np.all(~np.isfinite(sz)):
            continue
        smax = np.nanmax(np.abs(sz))
        if smax <= 0 or not np.isfinite(smax):
            continue

        curves.append((float(freq_thz[idx]), sz / smax))

    return x_nm, curves


def plot_fig12_poynting(
    freq7: np.ndarray,
    neff7: np.ndarray,
    freq8: np.ndarray,
    neff8: np.ndarray,
    gamma_lossy: float,
    eps2_of_omega=None,
    epsd_of_omega=None,
    fname: str = "fig12.pdf",
):
    x7, curves7 = compute_poynting_profile(
        freq7,
        neff7,
        case="ty",
        gamma=gamma_lossy,
        eps2_fn_omega=eps2_of_omega,
        eps_d_fn_omega=epsd_of_omega,
    )
    x8, curves8 = compute_poynting_profile(
        freq8,
        neff8,
        case="n",
        gamma=gamma_lossy,
        eps2_fn_omega=eps2_of_omega,
        eps_d_fn_omega=epsd_of_omega,
    )

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for f_thz, sz in curves7:
        axs[0].plot(x7, sz, linewidth=2.5, label=rf"{f_thz:.0f} THz")
    for f_thz, sz in curves8:
        axs[1].plot(x8, sz, linewidth=2.5, label=rf"{f_thz:.0f} THz")

    axs[0].axvline(0.0, color="k", linestyle="--", linewidth=1.2)
    axs[1].axvline(0.0, color="k", linestyle="--", linewidth=1.2)

    axs[0].set_title(r"$\mathbf{l}=\mathbf{t}_y$", fontsize=16)
    axs[1].set_title(r"$\mathbf{l}=\mathbf{n}$", fontsize=16)
    style_ax(axs[0], xlabel=r"$x$ (nm)", ylabel=r"$S_z(x)/\max|S_z|$")
    style_ax(axs[1], xlabel=r"$x$ (nm)", ylabel=None)
    axs[0].legend(loc="best", frameon=True, fontsize=11)
    axs[1].legend(loc="best", frameon=True, fontsize=11)

    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("Saved:", fname)


def plot_fig13_group_velocity(
    freq7: np.ndarray,
    neff7: np.ndarray,
    freq8: np.ndarray,
    neff8: np.ndarray,
    fname: str = "fig13.pdf",
):
    def _vg_and_ng(freq_thz, neff):
        omega = 2 * np.pi * freq_thz * 1e12
        beta = (omega / c) * np.real(neff)
        dbeta_domega = np.gradient(beta, omega)

        vg = np.full_like(beta, np.nan, dtype=float)
        good = np.isfinite(dbeta_domega) & (np.abs(dbeta_domega) > 1e-30)
        vg[good] = 1.0 / dbeta_domega[good]

        ng = np.full_like(beta, np.nan, dtype=float)
        ng[good] = c * dbeta_domega[good]
        return vg, ng

    vg7, ng7 = _vg_and_ng(freq7, neff7)
    vg8, ng8 = _vg_and_ng(freq8, neff8)

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)

    axs[0].plot(freq7, vg7 / c, "k-", linewidth=3.0, label=r"$v_g/c$")
    axs[0].plot(freq7, ng7, "k--", linewidth=2.5, label=r"$n_g=c/v_g$")
    axs[0].set_title(r"$\mathbf{l}=\mathbf{t}_y$", fontsize=16)
    style_ax(axs[0], xlabel="THz", ylabel=r"$v_g/c$ and $n_g$")
    axs[0].set_xlim(float(freq7[0]), float(freq7[-1]))
    axs[0].legend(loc="best", frameon=True, fontsize=12)

    axs[1].plot(freq8, vg8 / c, "k-", linewidth=3.0, label=r"$v_g/c$")
    axs[1].plot(freq8, ng8, "k--", linewidth=2.5, label=r"$n_g=c/v_g$")
    axs[1].set_title(r"$\mathbf{l}=\mathbf{n}$", fontsize=16)
    style_ax(axs[1], xlabel="THz", ylabel=None)
    axs[1].set_xlim(float(freq8[0]), float(freq8[-1]))
    axs[1].legend(loc="best", frameon=True, fontsize=12)

    plt.tight_layout()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("Saved:", fname)


def main():
    gamma_lossless = 0.0
    plot_fig2(gamma_lossless)
    plot_fig3(gamma_lossless)
    plot_fig4(gamma_lossless)
    plot_fig5(gamma_lossless)
    plot_fig6(gamma_lossless)

    EPS2_CSV = ti_material
    _, eps2_of_omega = load_eps_interpolator(
        EPS2_CSV,
        kind="linear",
        allow_extrapolate=False,
    )

    EPSD_CSV = hm_dielectric
    _, epsd_of_omega = load_eps_interpolator(
        EPSD_CSV,
        kind="linear",
        allow_extrapolate=False,
    )

    n_theta = 100
    theta = 2 * n_theta + 1
    kappa = alpha_fs * theta

    freq7 = np.linspace(100.0, 700.0, 200)
    neff7 = solve_curve_complex(
        freq7,
        case="ty",
        neff0=1.0 + 1.0j,
        gamma=gamma_lossy,
        kappa=kappa,
        eps2_fn_omega=eps2_of_omega,
        eps_d_fn_omega=epsd_of_omega,
    )
    plot_fig_reim(
        freq7,
        neff7,
        fname="fig7.pdf",
        xlim=(freq7[0], freq7[-1]),
        title=r"HM--TI surface wave with dissipation ($\mathbf{l}=\mathbf{t}_y$)",
    )

    freq8 = np.linspace(100.0, 700.0, 200)
    neff8 = solve_curve_complex(
        freq8,
        case="n",
        neff0=1.0 + 1.0j,
        gamma=gamma_lossy,
        kappa=kappa,
        eps2_fn_omega=eps2_of_omega,
        eps_d_fn_omega=epsd_of_omega,
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
        freq7_thz=freq7,
        neff7=neff7,
        freq8_thz=freq8,
        neff8=neff8,
        fname="fig9.pdf",
        xlim_left=(freq7[0], freq7[-1]),
        xlim_right=(freq8[0], freq8[-1]),
        title_left=r"Attenuation length ($\mathbf{l}=\mathbf{t}_y$)",
        title_right=r"Attenuation length ($\mathbf{l}=\mathbf{n}$)",
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

    eta7 = compute_te_tm_mixing_ratio(
        freq7,
        neff7,
        case="ty",
        gamma=gamma_lossy,
        kappa=kappa,
        eps2_fn_omega=eps2_of_omega,
        eps_d_fn_omega=epsd_of_omega,
    )
    eta8 = compute_te_tm_mixing_ratio(
        freq8,
        neff8,
        case="n",
        gamma=gamma_lossy,
        kappa=kappa,
        eps2_fn_omega=eps2_of_omega,
        eps_d_fn_omega=epsd_of_omega,
    )
    plot_fig_te_tm_mixing(
        freq7=freq7,
        eta7=eta7,
        freq8=freq8,
        eta8=eta8,
        fname="fig11.pdf",
    )

    plot_fig12_poynting(
        freq7=freq7,
        neff7=neff7,
        freq8=freq8,
        neff8=neff8,
        gamma_lossy=gamma_lossy,
        eps2_of_omega=eps2_of_omega,
        epsd_of_omega=epsd_of_omega,
        fname="fig12.pdf",
    )

    plot_fig13_group_velocity(
        freq7=freq7,
        neff7=neff7,
        freq8=freq8,
        neff8=neff8,
        fname="fig13.pdf",
    )


if __name__ == "__main__":
    main()
