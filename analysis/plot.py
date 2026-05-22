from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configuration ─────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")

L_VALUES = [6, 8, 10]
H_VALUES = [0.5, 2.5, 6.5, 10.5]
H_FIG2 = np.arange(0.5, 12.1, 1.0)

MIDDLE_FRACTION = (1 / 3, 2 / 3)

L_COLORS = {6: "tab:blue", 8: "tab:orange", 10: "tab:green", 12: "tab:red"}
L_MARKERS = {6: "v", 8: "o", 10: "^", 12: "s"}

# ── File helpers ───────────────────────────────────────────────────────────────


def w_tag(W: float) -> int:
    return int(round(1000 * W))


def eigvec_path(L: int, W: float, dup: int) -> Path:
    return RESULTS_DIR / f"eigenvectors_P{L}_W{w_tag(W)}_dupli{dup}.txt"


def load_eigvecs(L: int, W: float, dup: int) -> np.ndarray | None:
    p = eigvec_path(L, W, dup)
    if not p.exists():
        return None
    return np.loadtxt(p)  # shape (dim, dim), column k = k-th eigenvector


def available_duplicates(L: int, W: float) -> list[int]:
    tag = w_tag(W)
    pattern = re.compile(rf"eigenvectors_P{L}_W{tag}_dupli(\d+)\.txt")
    dups = []
    if RESULTS_DIR.exists():
        for f in RESULTS_DIR.iterdir():
            m = pattern.match(f.name)
            if m:
                dups.append(int(m.group(1)))
    return sorted(dups)


def middle_indices(n: int) -> np.ndarray:
    lo = int(MIDDLE_FRACTION[0] * n)
    hi = int(MIDDLE_FRACTION[1] * n)
    return np.arange(lo, hi)


# ── Basis ─────────────────────────────────────────────────────────────────────


def build_basis(L: int, S: int = 0) -> np.ndarray:
    """Sz=S sector basis states as integers."""
    states = []
    for i in range(1 << L):
        up = bin(i).count("1")
        if up - (L - up) == S:
            states.append(i)
    return np.array(states, dtype=np.int64)


def state_to_spins(state: int, L: int) -> np.ndarray:
    """Integer state → ±0.5 spin array, MSB = site 0."""
    return np.array([0.5 if (state >> (L - 1 - i)) & 1 else -0.5 for i in range(L)])


# ── Operators in sector basis ──────────────────────────────────────────────────


def build_Sz_diag(basis: np.ndarray, L: int) -> np.ndarray:
    """
    Diagonal of Sz_i in the sector basis for all sites.
    Returns array of shape (dim, L).
    """
    return np.array([state_to_spins(s, L) for s in basis])  # (dim, L)


def build_M1_matrix(basis: np.ndarray, L: int) -> np.ndarray:
    """
    Full matrix of M̂_1 = Σ_j Sz_j exp(i2πj/L) in the sector basis.

    M̂_1 is NOT diagonal in the sector basis — Sz_j is diagonal, but the
    sum with complex phases still produces a diagonal matrix because Sz_j
    is diagonal in the computational basis and the sector basis consists
    of computational basis states.  So M1 IS diagonal here; the off-diagonal
    elements of M̂†M̂ come from M̂_1† M̂_1 = (Σ_j Sz_j e^{-i2πj/L})(Σ_k Sz_k e^{i2πk/L}).
    """
    dim = len(basis)
    phases = np.exp(1j * 2 * np.pi * np.arange(L) / L)  # (L,)
    Sz = build_Sz_diag(basis, L)  # (dim, L)
    # M1_alpha = Σ_j Sz_j(alpha) * phase_j  — diagonal in sector basis
    M1_diag = Sz @ phases  # (dim,) complex
    return M1_diag


def build_M1sq_matrix(basis: np.ndarray, L: int) -> np.ndarray:
    """
    Full matrix of M̂†_1 M̂_1 in the sector basis.

    M̂†_1 M̂_1 = Σ_{j,k} Sz_j Sz_k exp(i2π(k-j)/L)

    This is NOT diagonal — Sz_j Sz_k connects states that differ by
    spin flips at sites j and k.  We build it explicitly.
    Returns (dim, dim) complex matrix.
    """
    dim = len(basis)
    phases = np.exp(1j * 2 * np.pi * np.arange(L) / L)  # (L,)
    Sz = build_Sz_diag(basis, L)  # (dim, L)

    # M̂†_1 M̂_1 = (Σ_j Sz_j e^{-i2πj/L})(Σ_k Sz_k e^{i2πk/L})
    # In computational basis both Sz_j are diagonal, so the matrix element
    # <alpha | M̂†M̂ | beta> = Σ_{j,k} <alpha|Sz_j|alpha> <beta|Sz_k|beta> * e^{i2π(k-j)/L} * delta_{alpha,beta}
    # Wait — Sz_j is diagonal so M̂†M̂ is also diagonal in the sector basis.
    # <alpha|M̂†M̂|alpha> = |M1_alpha|^2 = |Σ_j Sz_j(alpha) e^{i2πj/L}|^2
    M1_diag = Sz @ phases  # (dim,)
    M1sq_diag = np.abs(M1_diag) ** 2  # (dim,) real
    return M1_diag, M1sq_diag


# ── Figure 1: mean |Δm| between adjacent eigenstates ─────────────────────────


def compute_delta_m(L: int, W: float) -> tuple[float, float]:
    dups = available_duplicates(L, W)
    if not dups:
        return np.nan, np.nan

    basis = build_basis(L)
    dim = len(basis)
    mid_idx = middle_indices(dim)
    Sz = build_Sz_diag(basis, L)  # (dim, L)

    per_sample = []
    for dup in dups:
        vecs = load_eigvecs(L, W, dup)
        if vecs is None:
            continue
        vecs_mid = vecs[:, mid_idx]  # (dim, n_mid)
        # ⟨n|Sz_i|n⟩ = Σ_alpha |c_alpha|^2 Sz_i(alpha)
        mags = (vecs_mid ** 2).T @ Sz  # (n_mid, L)
        diffs = np.abs(mags[1:] - mags[:-1])  # (n_mid-1, L)
        per_sample.append(np.mean(diffs))

    if not per_sample:
        return np.nan, np.nan

    mean = np.mean(per_sample)
    stderr = np.std(per_sample) / np.sqrt(len(per_sample))
    return mean, stderr


# ── Figure 2: dynamic fraction f ─────────────────────────────────────────────


def compute_f(L: int, W: float) -> tuple[float, float]:
    """
    f^(n) = 1 - |<n|M1|n>|^2 / <n|M1†M1|n>

    Since M̂_1 = Σ_j Sz_j e^{i2πj/L} and Sz_j is diagonal in the
    computational (sector) basis:

      <n|M1|n>    = Σ_alpha |c_alpha|^2  M1_alpha        (diagonal expectation)
      <n|M1†M1|n> = Σ_alpha |c_alpha|^2 |M1_alpha|^2    (also diagonal)

    Both are correct because M̂_1 is diagonal in the sector basis —
    the sector basis states are eigenstates of each Sz_j.
    The reason f→1 in the ergodic phase is NOT from off-diagonal elements
    of M̂_1 itself, but because in a thermal state the phases of c_alpha
    are essentially random, making <n|M1|n> ≈ 0 while <n|M1†M1|n> stays finite.
    """
    dups = available_duplicates(L, W)
    if not dups:
        return np.nan, np.nan

    basis = build_basis(L)
    dim = len(basis)
    mid_idx = middle_indices(dim)

    phases = np.exp(1j * 2 * np.pi * np.arange(L) / L)  # (L,)
    Sz = build_Sz_diag(basis, L)  # (dim, L)
    M1_diag = Sz @ phases  # (dim,) complex
    M1sq = np.abs(M1_diag) ** 2  # (dim,) real

    per_sample_f = []
    for dup in dups:
        vecs = load_eigvecs(L, W, dup)
        if vecs is None:
            continue
        vecs_mid = vecs[:, mid_idx]  # (dim, n_mid)  — real eigenvectors

        # <n|M1†M1|n> = Σ_alpha |c_alpha|^2 |M1_alpha|^2
        denom = (vecs_mid ** 2).T @ M1sq  # (n_mid,) real

        # <n|M1|n> = Σ_alpha c_alpha^2 * M1_alpha
        # Note: eigenvectors from dsyev are real, so c_alpha are real.
        # In the ergodic phase c_alpha are pseudorandom → cancellation → ≈0
        numer_static = np.abs((vecs_mid ** 2).T @ M1_diag) ** 2  # (n_mid,) real

        f_n = 1.0 - numer_static / np.where(denom > 1e-15, denom, 1e-15)
        per_sample_f.append(np.mean(np.clip(f_n, 0.0, 1.0)))

    if not per_sample_f:
        return np.nan, np.nan

    mean = np.mean(per_sample_f)
    stderr = np.std(per_sample_f) / np.sqrt(len(per_sample_f))
    return mean, stderr


# ── Plotting ──────────────────────────────────────────────────────────────────


def setup_style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
            "lines.linewidth": 1.4,
            "lines.markersize": 5,
        }
    )


def plot_figure1(h_values: list[float], L_values: list[int]):
    print("Computing Figure 1 …")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    h_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(h_values)))

    for h, color in zip(h_values, h_colors):
        Ls, means, errs = [], [], []
        for L in L_values:
            m, e = compute_delta_m(L, h)
            if np.isnan(m):
                print(f"  No data: L={L}, h={h:.1f}")
                continue
            Ls.append(L)
            means.append(m)
            errs.append(e)
            print(f"  L={L:2d}  h={h:.1f}  log|Δm|={np.log(m):.3f}")

        if not Ls:
            continue

        log_means = np.log(np.array(means))
        log_errs = np.array(errs) / np.array(means)
        ax.errorbar(
            Ls,
            log_means,
            yerr=log_errs,
            label=f"h={h}",
            color=color,
            marker="o",
            capsize=3,
            linewidth=1.2,
        )

    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$\log\,[\,|\,m^{(n)}_{i\alpha} - m^{(n+1)}_{i\alpha}\,|\,]$")
    ax.legend(ncol=2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    fig.tight_layout()
    return fig


def plot_figure2(h_values: np.ndarray, L_values: list[int]):
    print("\nComputing Figure 2 …")
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for L in L_values:
        hs, fs, errs = [], [], []
        for h in h_values:
            mean_f, err_f = compute_f(L, h)
            if np.isnan(mean_f):
                print(f"  No data: L={L}, h={h:.2f}")
                continue
            hs.append(h)
            fs.append(mean_f)
            errs.append(err_f)
            print(f"  L={L:2d}  h={h:.2f}  [f]={mean_f:.4f}")

        if not hs:
            continue

        ax.errorbar(
            hs,
            fs,
            yerr=errs,
            label=f"L={L}",
            color=L_COLORS.get(L),
            marker=L_MARKERS.get(L, "o"),
            capsize=3,
            linewidth=1.2,
        )

    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$[\,f^{(n)}_\alpha\,]$")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    setup_style()

    fig1 = plot_figure1(H_VALUES, L_VALUES)
    fig1.savefig("figures/figure1_delta_m.svg", dpi=200, bbox_inches="tight")

    fig2 = plot_figure2(H_FIG2, L_VALUES)
    fig2.savefig("figures/figure2_dynamic_f.svg", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
