# Student Name: Oliver Wuttke
# Student FAN: wutt0019
# File: visualise.py
# Date: 16-06-2026
# Description: Generates the three key HMM figures for the A2 report.
# Usage: python visualise.py   (run AFTER train_test_val.py has produced hmm_params.npz)

"""
This file generates 3 figures to help interpret the HMM.
Figures are saved as PNGs into ../Misc/ so they can be embedded in the report
and regenerated deterministically.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from matplotlib.colors import LinearSegmentedColormap

# Reuse the exact assembly + scaling from training so figures match the model
from Source.train_test_val import (
    assemble_feature_matrix, add_direction_label, apply_scaler,
    OBS_COLS, DIRECTIONS, N_STATES, TRAIN_RANGE,
)


# Output directory for figures
FIG_DIR = "../Misc"

# Preferred color scheme
INK = "#1b1b1f"
GRID = "#d9d6cf"
CALM = "#a9b7c0"
TRANSITION = "#e0a458"
TURBULENT = "#c1463d"
STATE_COLORS = [CALM, TRANSITION, TURBULENT]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


"""
Reload params, rebuild decoded states for the full series.
We reconstruct a GaussianHMM from the saved arrays.
"""
def rebuild_model_and_decode():
    p = np.load("hmm_params.npz", allow_pickle=True)

    model = GaussianHMM(n_components=int(p["n_states"]), covariance_type="full")
    model.startprob_ = p["startprob"]
    model.transmat_ = p["transmat"]
    model.means_ = p["means"]

    # Regularize covariances before assigning. A state fit from very few
    # observations can have a degenerate, non-positive-definite covariance that
    # hmmlearn's setter rejects on reload. Adding a small ridge to the diagonal
    # makes every component valid without materially changing populated states.
    covars = np.array(p["covars"], dtype=float)
    n_features = covars.shape[-1]
    ridge = 1e-6 * np.eye(n_features)
    covars = covars + ridge
    model.covars_ = covars

    scaler = (p["scaler_mean"], p["scaler_std"])
    state_to_dir = p["state_to_dir"]
    winsor = (p["winsor_lo"], p["winsor_hi"])

    full = add_direction_label(assemble_feature_matrix())
    raw = full[OBS_COLS].to_numpy(float)

    # Apply the SAME winsor clip + standardization used at training,
    # so decoded states are consistent
    raw = np.clip(raw, winsor[0], winsor[1])
    obs = apply_scaler(raw, scaler)
    decoded = model.predict(obs)
    full = full.copy()
    full["state"] = decoded
    return full, model, state_to_dir


"""
Map raw state ids -> rank by d_Oil spread.
Ranking by oil-return spread makes the colour scheme meaningful 
(calm -> turbulent) and stable across runs.
"""
def order_states_by_volatility(full):
    spread = full.groupby("state")["d_Oil"].std()
    order = spread.sort_values().index.tolist()
    return {raw: rank for rank, raw in enumerate(order)}

"""
XEJ close with the background shaded by decoded HMM state. Shows the
high-volatility state aligning with the 2020 oil shock, the visual
evidence that latent states are volatility regimes.
"""
def fig_regime_timeline(full, rank_map):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    dates = full["date"].to_numpy()
    close = full["Close"].to_numpy()

    # Shade background by ranked state
    ranked = full["state"].map(rank_map).to_numpy()
    n = len(full)
    min_span = np.timedelta64(4, "D")
    start = 0
    for i in range(1, n + 1):
        if i == n or ranked[i] != ranked[start]:
            x0 = dates[start]
            x1 = dates[min(i, n - 1)]
            if np.datetime64(x1) - np.datetime64(x0) < min_span:
                x1 = np.datetime64(x0) + min_span
            ax.axvspan(x0, x1, color=STATE_COLORS[ranked[start]],
                       alpha=0.32, linewidth=0)
            start = i

    ax.plot(dates, close, color=INK, linewidth=1.1)
    ax.set_title("XEJ log-close shaded by decoded HMM regime",
                 fontsize=13, fontweight="bold", loc="left")
    ax.set_ylabel("log(1 + close)")
    ax.margins(x=0.01)

    # Legend mapping rank -> meaning
    labels = ["Calm regime", "Transitional regime", "High-volatility regime"]
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATE_COLORS[i], alpha=0.5)
               for i in range(N_STATES)]
    ax.legend(handles, labels[:N_STATES], loc="upper left",
              frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_regime_timeline.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")

"""
Heatmap of the learned transition matrix. Shows regime persistence,
strong diagonal self-loops.
"""
def fig_transition_matrix(model, rank_map):
    # Reorder the matrix by volatility rank for a heatmap.
    order = sorted(range(N_STATES), key=lambda s: rank_map[s])
    A = model.transmat_[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    cmap = LinearSegmentedColormap.from_list("ink", ["white", INK])
    im = ax.imshow(A, cmap=cmap, vmin=0, vmax=1)

    ticks = ["Calm", "Transit.", "High-vol"][:N_STATES]
    ax.set_xticks(range(N_STATES)); ax.set_xticklabels(ticks, fontsize=9)
    ax.set_yticks(range(N_STATES)); ax.set_yticklabels(ticks, fontsize=9)
    ax.set_xlabel("to state"); ax.set_ylabel("from state")
    ax.set_title("Transition matrix (regime persistence)",
                 fontsize=12, fontweight="bold", loc="left")

    for i in range(N_STATES):
        for j in range(N_STATES):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                    color="white" if A[i, j] > 0.5 else INK, fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_transition_matrix.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")

"""
Next-day direction proportions within each state. Near-identical bars
across states = the regime carries little directional information
(the central finding).
"""
def fig_direction_mix(full, rank_map):
    # Next-day direction proportions within each ranked state, TRAIN-set only.
    start, end = TRAIN_RANGE
    train = full[(full["date"] >= start) & (full["date"] <= end)]

    ranks = sorted(set(rank_map.values()))
    dir_colors = {"down": TURBULENT, "flat": "#9a9a9a", "up": "#4f7a5b"}

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    width = 0.25
    x = np.arange(len(ranks))
    inv = {v: k for k, v in rank_map.items()}

    for di, dname in enumerate(DIRECTIONS):
        props = []
        for r in ranks:
            raw_state = inv[r]
            sub = train[train["state"] == raw_state]["direction"]
            props.append((sub == di).mean() if len(sub) else 0)
        ax.bar(x + (di - 1) * width, props, width,
               label=dname, color=dir_colors[dname])

    ax.axhline(1 / 3, color=INK, linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Calm", "Transitional", "High-vol"][:len(ranks)])
    ax.set_ylabel("next-day direction proportion")
    ax.set_title("Next-day open direction is ~flat across regimes",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    ax.set_ylim(0, 0.75)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig3_direction_mix.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)  # ensure ../Misc/ exists
    full, model, _ = rebuild_model_and_decode()
    rank_map = order_states_by_volatility(full)
    fig_regime_timeline(full, rank_map)
    fig_transition_matrix(model, rank_map)
    fig_direction_mix(full, rank_map)
    print("\nAll figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()