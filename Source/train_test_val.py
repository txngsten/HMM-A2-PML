# Student Name: Oliver Wuttke
# Student FAN: wutt0019
# File: train_test_val.py
# Date: 16-06-2026
# Description: 3-state GaussianHMM (hmmlearn) on DIFFERENCED observations for XEJ next-day open regime.
# Usage: python train_test_val.py

"""
Unsupervised 3-state GaussianHMM for XEJ next-day open direction, on
DIFFERENCED (stationary) observations.

Why differenced?
A first attempt used observation LEVELS (GPR_log, oil price, prev-close). Those
series are non-stationary and trend over years, so the HMM discovered slow
"era" regimes (near-absorbing self-loops in the transition matrix) that had no
relationship to next-day open direction - every state collapsed to the majority
class. See report for that result; it is kept as a baseline.

Here the observations are daily CHANGES instead:
    d_GPR   : month-over-month change in GPR_log (broadcast to daily)
    d_Oil   : daily log-return of oil close
    d_Close : daily log-return of XEJ close (prev-day)
These are roughly stationary and zero-centred, giving the HMM a chance to find
behaviour-based regimes rather than epoch.

The model is still unsupervised (Baum-Welch); latent states are mapped to
up/flat/down post-hoc using TRAIN next-day directions only, then frozen.

Two evaluation routes:
1. evaluate() - SMOOTHED decode: Viterbi over the whole split (uses past AND
   future obs to infer each state), mapped to direction. Useful as an upper
   bound / descriptive fit, but NOT a forecast.
2. evaluate_forecast() - FILTERED one-step-ahead prediction: at each day t,
   uses only obs up to t, projects the filtered state belief one step through
   the transition matrix, maps to a next-day direction distribution, and scores
   it. This is the honest predictive metric and is the one to quote for
   "next-day open" performance. Reports accuracy, confusion matrix, log-loss,
   Brier score, plus the model log-likelihood per split.

Split (chronological):
    train = 2018-01-01 .. 2022-12-31
    test  = 2023-01-01 .. 2023-12-31
    val   = 2024-01-01 .. 2024-12-31
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from Source.etl_helpers import build_datasets, load_gpr

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TRAIN_RANGE = ("2018-01-01", "2022-12-31")
TEST_RANGE = ("2023-01-01", "2023-12-31")
VAL_RANGE = ("2024-01-01", "2024-12-31")

N_STATES = 3
DIRECTIONS = ["down", "flat", "up"]
FLAT_THRESHOLD = 0.001

# Winsorising bound (percentile) applied to observations before fitting. The
# April-2020 negative-oil event produces a d_Oil of roughly -6.6 outlier so extreme it
# captures an entire HMM state by itself. Clipping each observation column to its
# [1, 99] percentile keeps the volatility signal while stopping a single point
# from owning a state. The clip bounds are learned on TRAIN and reused.
WINSOR_PCT = (1.0, 99.0)

# Differenced observation columns = data contract shared with inference file.
OBS_COLS = ["d_GPR", "d_Oil", "d_Close"]


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

"""
Build daily frame with DIFFERENCED observations + next-day open label.
build_datasets() already returns log-transformed Open/High/Low/Close,
Close_prev and Oil_close on the XEJ trading calendar. We:
    - broadcast monthly GPR_log onto days, then take its month-over-month diff
    - take daily log-returns of oil and XEJ close (these are diffs of logs)
    - compute next-day open direction for labelling/eval only
"""
def assemble_feature_matrix():
    daily = build_datasets().copy()
    daily["date"] = pd.to_datetime(daily["date"])

    # Monthly GPR -> daily by year-month, forward-filled.
    gpr = load_gpr().copy()
    gpr["ym"] = gpr["month"].dt.to_period("M")
    gpr_map = gpr.set_index("ym")["GPR_log"]
    daily["ym"] = daily["date"].dt.to_period("M")
    daily["GPR_log"] = daily["ym"].map(gpr_map).ffill()

    # Differenced observations.
    # GPR: month-over-month change. GPR_log is constant within a month
    # (forward-filled), so we diff the per-month series and broadcast. Every
    # day in a month carries that month's change vs the previous month.
    monthly_gpr = gpr.set_index("ym")["GPR_log"].sort_index()
    monthly_change = monthly_gpr.diff()
    daily["d_GPR"] = daily["ym"].map(monthly_change).fillna(0.0)
    daily = daily.drop(columns=["ym"])

    # Oil and XEJ close are already log transformed
    daily["d_Oil"] = daily["Oil_close"].diff()
    daily["d_Close"] = daily["Close"].diff()

    # Next-day open + current open are log transformed -> their diff is the next-day
    # open log-return used to label direction.
    daily["Open_next"] = daily["Open"].shift(-1)

    # First row has NaN diffs; last row has NaN Open_next.
    daily = daily.dropna(subset=OBS_COLS + ["Open_next"]).reset_index(drop=True)
    return daily


def add_direction_label(df):
    df = df.copy()
    log_ret = df["Open_next"] - df["Open"]
    df["direction"] = np.where(
        log_ret > FLAT_THRESHOLD, 2,
        np.where(log_ret < -FLAT_THRESHOLD, 0, 1),
    ).astype(int)
    return df


def split_by_date(df, date_range):
    start, end = date_range
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Standardisation (train only)
# ---------------------------------------------------------------------------

"""
Returns mean and standard deviation for obs.
"""
def fit_scaler(obs):
    mean = obs.mean(axis=0)
    std = obs.std(axis=0)
    std[std == 0] = 1.0
    return mean, std

"""
Applies scaling.
"""
def apply_scaler(obs, scaler):
    mean, std = scaler
    return (obs - mean) / std

"""
Learn per-column clip bounds from TRAIN at the WINSOR_PCT percentiles.
"""
def fit_winsor_bounds(obs):
    lo = np.percentile(obs, WINSOR_PCT[0], axis=0)
    hi = np.percentile(obs, WINSOR_PCT[1], axis=0)
    return lo, hi

"""
Clip observations to learned bounds (reused for test/val/inference).
"""
def apply_winsor(obs, bounds):
    lo, hi = bounds
    return np.clip(obs, lo, hi)

"""
Raw OBS_COLS -> winsorised -> standardised array, this is the model's input.
"""
def prepare_obs(df, scaler, winsor):
    raw = apply_winsor(df[OBS_COLS].to_numpy(float), winsor)
    return apply_scaler(raw, scaler)


# ---------------------------------------------------------------------------
# State -> direction mapping (train set only)
# ---------------------------------------------------------------------------

"""
Creates a python dict that maps latent states to directions.
"""
def learn_state_to_direction(decoded_states, directions):
    mapping = {}
    for s in range(N_STATES):
        idx = decoded_states == s
        mapping[s] = (
            int(np.bincount(directions[idx], minlength=3).argmax())
            if idx.sum() > 0 else 1
        )
    return mapping


def states_to_directions(decoded_states, mapping):
    return np.array([mapping[s] for s in decoded_states])


# ---------------------------------------------------------------------------
# Evaluation - view 1: smoothed decode (descriptive, peeks ahead)
# ---------------------------------------------------------------------------

"""
Smoothed-decode evaluation (not a forecast).
Viterbi-decodes the whole split (uses past AND future observations),
maps each decoded state to a direction, and reports overall accuracy plus
per-class recall. The per-class recall exposes any collapse to a single
direction that overall accuracy alone would hide. Because it peeks at future
data it is an upper bound on the mapping's performance, not a prediction.
"""
def evaluate(name, model, scaler, winsor, mapping, df):
    obs = prepare_obs(df, scaler, winsor)
    decoded = model.predict(obs)
    pred_dir = states_to_directions(decoded, mapping)
    true_dir = df["direction"].to_numpy()

    acc = float(np.mean(pred_dir == true_dir))
    print(f"\n[{name}] (smoothed decode) n={len(df)}  direction accuracy={acc:.3f}")
    for d, label in enumerate(DIRECTIONS):
        idx = true_dir == d
        if idx.sum() == 0:
            print(f"  {label:>4}: no samples")
            continue
        recall = float(np.mean(pred_dir[idx] == d))
        print(f"  {label:>4}: support={idx.sum():4d}  recall={recall:.3f}")
    return acc


# ---------------------------------------------------------------------------
# Evaluation - view 2: honest one-step-ahead forecast (filtered, no lookahead)
# ---------------------------------------------------------------------------

"""
Filtered state beliefs P(state_t | obs_1..t) for every t, no lookahead.
The last row of predict_proba on an expanding prefix obs[:t+1] is
the filtered (forward-only) posterior at t, smoothing over a
prefix that ends at t cannot use information after t.
"""
def filtered_posteriors(model, obs):
    n = len(obs)
    filt = np.zeros((n, model.n_components))
    for t in range(n):
        filt[t] = model.predict_proba(obs[: t + 1])[-1]
    return filt


"""
Map filtered beliefs to a next-day DIRECTION probability distribution.
next-day state prob = filtered @ A  (one-step projection). Each state's
probability is then added to the direction it maps to, giving a (n, 3)
matrix of P(next-day direction) that sums to 1 per row.
"""
def direction_distribution(filtered, transmat, state_to_dir):
    next_state_prob = filtered @ transmat       # (n, n_states)
    n = next_state_prob.shape[0]
    dir_prob = np.zeros((n, len(DIRECTIONS)))
    for s in range(len(state_to_dir)):
        dir_prob[:, state_to_dir[s]] += next_state_prob[:, s]
    # Guard against tiny numerical drift
    dir_prob /= dir_prob.sum(axis=1, keepdims=True)
    return dir_prob


"""
3x3 counts, rows = true direction, cols = predicted direction.
"""
def confusion_matrix(true_dir, pred_dir):
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(true_dir, pred_dir):
        cm[t, p] += 1
    return cm

"""
Multiclass cross-entropy (nats). Lower is better.
"""
def log_loss_manual(true_dir, dir_prob):
    eps = 1e-12
    p = np.clip(dir_prob, eps, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(true_dir)), true_dir])))

"""
Multiclass Brier score: mean squared error vs one-hot truth.
"""
def brier_multiclass(true_dir, dir_prob):
    onehot = np.zeros_like(dir_prob)
    onehot[np.arange(len(true_dir)), true_dir] = 1.0
    return float(np.mean(np.sum((dir_prob - onehot) ** 2, axis=1)))

"""
Honest next-day-open forecast metrics for one split.
Uses filtered (forward-only) beliefs -> one-step projection -> direction
distribution. Reports accuracy vs majority baseline, confusion matrix,
log-loss, Brier score, and model log-likelihood.
"""
def evaluate_forecast(name, model, scaler, winsor, mapping, train_df, df):
    obs = prepare_obs(df, scaler, winsor)
    state_to_dir = np.array([mapping[s] for s in range(N_STATES)])

    filt = filtered_posteriors(model, obs)
    dir_prob = direction_distribution(filt, model.transmat_, state_to_dir)
    pred_dir = dir_prob.argmax(axis=1)
    true_dir = df["direction"].to_numpy()

    acc = float(np.mean(pred_dir == true_dir))
    maj = int(np.bincount(train_df["direction"].to_numpy(), minlength=3).argmax())
    base_acc = float(np.mean(true_dir == maj))
    ll = log_loss_manual(true_dir, dir_prob)
    brier = brier_multiclass(true_dir, dir_prob)
    model_loglik = float(model.score(obs))

    print(f"\n[{name}] (one-step-ahead FORECAST) n={len(df)}")
    print(f"  accuracy        = {acc:.3f}   (majority baseline = {base_acc:.3f}, "
          f"'{DIRECTIONS[maj]}')")
    print(f"  log-loss (nats) = {ll:.3f}   (lower better)")
    print(f"  Brier score     = {brier:.3f}   (lower better)")
    print(f"  model log-lik   = {model_loglik:.1f}")
    cm = confusion_matrix(true_dir, pred_dir)
    print(f"  confusion (rows=true {DIRECTIONS}, cols=pred):")
    for d, row in enumerate(cm):
        print(f"    {DIRECTIONS[d]:>4}: {row}")
    return {"accuracy": acc, "baseline": base_acc, "log_loss": ll,
            "brier": brier, "model_loglik": model_loglik}

"""
Accuracy of always predicting train's most common direction.
"""
def majority_baseline(train_df, eval_df):
    maj = int(np.bincount(train_df["direction"].to_numpy(), minlength=3).argmax())
    return float(np.mean(eval_df["direction"].to_numpy() == maj)), DIRECTIONS[maj]

"""
Describe what each latent state IS, in observation units.
Reports each state's mean observation vector and the
spread of the assigned training observations, evidence for the volatility-regime 
interpretation plus the next-day direction mix per state.
"""
def characterise_regimes(model, scaler, train_decoded, train_df):
    mean, std = scaler
    raw = train_df[OBS_COLS].to_numpy(float)
    dirs = train_df["direction"].to_numpy()

    print("\n--- Regime characterisation (observation units) ---")
    for s in range(N_STATES):
        idx = train_decoded == s
        if idx.sum() == 0:
            print(f"  state {s}: empty")
            continue
        state_mean_raw = model.means_[s] * std + mean
        state_spread = raw[idx].std(axis=0)
        mean_str = ", ".join(f"{c}={m:+.4f}" for c, m in zip(OBS_COLS, state_mean_raw))
        spread_str = ", ".join(f"{c}={v:.4f}" for c, v in zip(OBS_COLS, state_spread))
        counts = np.bincount(dirs[idx], minlength=3)
        mix = counts / counts.sum()
        mix_str = ", ".join(f"{DIRECTIONS[d]}={mix[d]:.2f}" for d in range(3))
        print(f"  state {s} (n={idx.sum()}):")
        print(f"      mean   : {mean_str}")
        print(f"      spread : {spread_str}")
        print(f"      nextdir: {mix_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Assembling differenced feature matrix...")
    full = add_direction_label(assemble_feature_matrix())

    train_df = split_by_date(full, TRAIN_RANGE)
    test_df = split_by_date(full, TEST_RANGE)
    val_df = split_by_date(full, VAL_RANGE)
    print(f"Rows -> train={len(train_df)} test={len(test_df)} val={len(val_df)}")

    if len(train_df) < N_STATES:
        raise RuntimeError("Training split too small to fit a 3-state HMM.")

    train_obs_raw = train_df[OBS_COLS].to_numpy(float)

    # Winsorise (learn bounds on train) BEFORE scaling, then standardize.
    winsor = fit_winsor_bounds(train_obs_raw)
    train_obs_w = apply_winsor(train_obs_raw, winsor)
    scaler = fit_scaler(train_obs_w)
    train_obs = apply_scaler(train_obs_w, scaler)

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=200,
        random_state=RANDOM_SEED,
        min_covar=1e-3,  # covariance floor
    )
    model.fit(train_obs)
    if not model.monitor_.converged:
        print("WARNING: EM did not converge - try more n_iter or fewer states.")

    print("\nLearned transition matrix (rows=from, cols=to):")
    print(np.round(model.transmat_, 3))

    train_decoded = model.predict(train_obs)
    mapping = learn_state_to_direction(train_decoded, train_df["direction"].to_numpy())
    print("\nState -> direction mapping (learned on train):")
    for s in range(N_STATES):
        n = int((train_decoded == s).sum())
        print(f"  state {s} -> {DIRECTIONS[mapping[s]]:>4}  (train rows: {n})")

    # Smoothed decode (descriptive upper bound)
    print("\n=== SMOOTHED decode (descriptive - uses full sequence) ===")
    evaluate("TRAIN", model, scaler, winsor, mapping, train_df)
    evaluate("TEST", model, scaler, winsor, mapping, test_df)
    evaluate("VAL", model, scaler, winsor, mapping, val_df)

    # Honest one-step-ahead forecast
    print("\n=== ONE-STEP-AHEAD FORECAST (filtered, no lookahead) ===")
    evaluate_forecast("TRAIN", model, scaler, winsor, mapping, train_df, train_df)
    evaluate_forecast("TEST", model, scaler, winsor, mapping, train_df, test_df)
    evaluate_forecast("VAL", model, scaler, winsor, mapping, train_df, val_df)

    # Regime characterization
    characterise_regimes(model, scaler, train_decoded, train_df)

    # Per-state next-day direction mix.
    # Saved so one-off inference can use honest probabilities without touchin any historical data.
    train_dirs = train_df["direction"].to_numpy()
    state_mix = np.zeros((N_STATES, 3))
    for s in range(N_STATES):
        idx = train_decoded == s
        state_mix[s] = (np.bincount(train_dirs[idx], minlength=3) / idx.sum()
                        if idx.sum() > 0 else np.array([1 / 3, 1 / 3, 1 / 3]))

    # Save the params to a numpy array so other files can use
    np.savez(
        "hmm_params.npz",
        transmat=model.transmat_,
        startprob=model.startprob_,
        means=model.means_,
        covars=model.covars_,
        scaler_mean=scaler[0],
        scaler_std=scaler[1],
        winsor_lo=winsor[0],
        winsor_hi=winsor[1],
        state_to_dir=np.array([mapping[s] for s in range(N_STATES)]),
        state_mix=state_mix,
        obs_cols=np.array(OBS_COLS),
        directions=np.array(DIRECTIONS),
        n_states=N_STATES,
    )
    print("\nSaved parameters -> hmm_params.npz")


if __name__ == "__main__":
    main()