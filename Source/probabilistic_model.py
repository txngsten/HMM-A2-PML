# Student Name: Oliver Wuttke
# Student FAN: wutt0019
# File: probabilistic_model.py
# Date: 16-06-2026
# Description: One-off next-trading-day XEJ open prediction from a saved HMM (hmm_params.npz).
# Usage: python predict_next_day.py (run AFTER train_test_val.py has produced hmm_params.npz)

"""
One-off inference: predict the NEXT TRADING DAY's XEJ open direction.

Inputs:
Frozen model parameters from hmm_params.npz (transition matrix, emissions,
scaler, winsor bounds, state->direction mapping) produced by train_test_val.py.
The single most recent observation only: current GPR, last oil close, last
XEJ close.

Trading calendar:
The data getters operate on the XEJ trading calendar (yfinance returns only
trading days), so the most recent day is the most recent TRADING day and
next day is the next TRADING day automatically. If the latest data is a
Friday, the prediction is for Monday's open; if Monday is a public holiday, the
next available row is Tuesday.

Method and its limitation:
Observations are DIFFERENCES (d_GPR, d_Oil, d_Close), so forming today's
observation needs today's value AND the prior trading day's value. We therefore
read the last two rows to compute the diff, but the prediction itself uses only
the most recent day's observation vector.

Output:
Both the hard label (argmax direction) and the calibrated probability
distribution over {down, flat, up}, so the report can show the point prediction
alongside honestly-quantified uncertainty.
"""

import datetime as dt

import numpy as np
import pandas as pd

from Source.etl_helpers import (
    download_yf_data, load_gpr, signed_log1p,
    XEJ_TICKER, OIL_TICKER,
)

PARAMS_PATH = "hmm_params.npz"
OBS_COLS = ["d_GPR", "d_Oil", "d_Close"]
DIRECTIONS = ["down", "flat", "up"]

# How many recent calendar days to pull. 25 calendar days comfortably covers
# the 10 trading rows needed for clean diffs and a month-boundary GPR change,
# while staying far smaller than the full 2018+ history.
TAIL_DAYS = 25

"""
Load frozen HMM parameters.
"""
def load_params(path=PARAMS_PATH):
    p = np.load(path, allow_pickle=True)
    params = {
        "transmat": p["transmat"],
        "startprob": p["startprob"],
        "means": p["means"],
        "covars": p["covars"],
        "scaler_mean": p["scaler_mean"],
        "scaler_std": p["scaler_std"],
        "winsor_lo": p["winsor_lo"],
        "winsor_hi": p["winsor_hi"],
        "state_to_dir": p["state_to_dir"],
        "n_states": int(p["n_states"]),
    }
    # Calibrated soft mapping, if training saved it
    params["state_mix"] = p["state_mix"] if "state_mix" in p.files else None
    return params

"""
Build the most recent day's differenced observation from a SHORT tail.
Fetches only the last TAIL_DAYS of XEJ and oil (one small yfinance call
each) rather than the full training history, the frozen model needs no
history, only today's differenced observation. A short tail (not just two
rows) is fetched to guarantee clean diffs across ASX/oil holiday mismatches
and to capture the month-over-month GPR change.

Returns (obs_vector, latest_date) where obs_vector = [d_GPR, d_Oil, d_Close]
for the most recent trading day.
"""
def latest_observation():
    today = dt.date.today()
    start = (today - dt.timedelta(days=TAIL_DAYS)).isoformat()
    end = (today + dt.timedelta(days=1)).isoformat()

    # XEJ tail (close only needed for d_Close). Positive prices -> log1p.
    xej = download_yf_data(XEJ_TICKER, start, end)[["Close"]].copy()
    xej["Close"] = np.log1p(xej["Close"])
    xej = xej.rename(columns={"Close": "xej_close"}).reset_index()
    xej = xej.rename(columns={"Date": "date"})

    # Oil tail. signed_log1p mirrors training (handles any negative values).
    oil = download_yf_data(OIL_TICKER, start, end)[["Close"]].copy()
    oil["Close"] = signed_log1p(oil["Close"])
    oil = oil.rename(columns={"Close": "oil_close"}).reset_index()
    oil = oil.rename(columns={"Date": "date"})

    # Align oil onto the XEJ trading calendar, forward-fill holiday gaps.
    df = xej.merge(oil, on="date", how="left")
    df["oil_close"] = df["oil_close"].ffill().bfill()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < 2:
        raise RuntimeError(
            f"Only {len(df)} trading rows in the last {TAIL_DAYS} days - "
            "increase TAIL_DAYS or check the data source."
        )

    # Daily log-returns
    df["d_Oil"] = df["oil_close"].diff()
    df["d_Close"] = df["xej_close"].diff()

    # GPR month-over-month change for the latest day's month
    gpr = load_gpr().copy()
    gpr["ym"] = gpr["month"].dt.to_period("M")
    monthly_change = gpr.set_index("ym")["GPR_log"].sort_index().diff()
    latest_ym = df["date"].iloc[-1].to_period("M")
    d_gpr = monthly_change.get(latest_ym, np.nan)
    if pd.isna(d_gpr):
        # GPR for the current month may not be published yet, fall back to the
        # most recent available month-change so the feature stays defined.
        d_gpr = float(monthly_change.dropna().iloc[-1])

    last = df.iloc[-1]
    obs = np.array([d_gpr, last["d_Oil"], last["d_Close"]], dtype=float)

    if np.any(np.isnan(obs)):
        raise RuntimeError(f"NaN in latest observation {obs} - check the tail fetch.")

    return obs, last["date"]

"""
Log N(x | mean, cov) for one observation vector.
"""
def gaussian_log_likelihood(x, mean, cov):
    d = x.shape[0]
    diff = x - mean
    sign, logdet = np.linalg.slogdet(cov)
    inv = np.linalg.inv(cov)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + diff @ inv @ diff)

"""
Predict next-trading-day direction from a single observation.
Steps:
    1. winsorise with saved bounds
    2. standardise with saved scaler
    3. emission likelihood per state -> belief = startprob x emission, norm.
    4. project one step: next_state = belief @ transmat
    5. map next-state probs to a direction distribution
"""
def predict(params, obs_raw, calibrated_mix=None):
    n_states = params["n_states"]

    # Winsorise then standardise.
    clipped = np.clip(obs_raw, params["winsor_lo"], params["winsor_hi"])
    x = (clipped - params["scaler_mean"]) / params["scaler_std"]

    # Emission log-likelihood per state, combine with start prior.
    log_em = np.array([
        gaussian_log_likelihood(x, params["means"][s], params["covars"][s])
        for s in range(n_states)
    ])
    log_belief = np.log(params["startprob"] + 1e-12) + log_em
    log_belief -= log_belief.max()
    belief = np.exp(log_belief)
    belief /= belief.sum()

    # One-step projection through the transition matrix.
    next_state = belief @ params["transmat"]

    # State -> direction distribution.
    dir_prob = np.zeros(3)
    if calibrated_mix is not None:
        # Soft: each state contributes its empirical direction mix.
        for s in range(n_states):
            dir_prob += next_state[s] * calibrated_mix[s]
    else:
        # Hard: all of a state's mass goes to its mapped direction.
        for s in range(n_states):
            dir_prob[int(params["state_to_dir"][s])] += next_state[s]
    dir_prob /= dir_prob.sum()

    return belief, next_state, dir_prob

"""
Per-state empirical next-day direction mix (the calibrated soft mapping).
NOTE ON HISTORY: this re-decodes the TRAIN period to recover each state's
up/flat/down proportions, so it DOES read historical data, unlike the
prediction itself, which needs only the latest observation. The mix is a
fixed property of the trained model, so the clean design is to save it into
hmm_params.npz during training and load it here instead of recomputing. 
Until then this recomputes it.

If you only need the hard label, skip this entirely (pass calibrated_mix
=None to predict()), and no history is touched.
"""
def empirical_state_mix(params):
    if "state_mix" in params and params["state_mix"] is not None:
        return params["state_mix"]

    from Source.train_test_val import (
        assemble_feature_matrix, add_direction_label, split_by_date,
        apply_winsor, apply_scaler, TRAIN_RANGE,
    )
    from hmmlearn.hmm import GaussianHMM

    full = add_direction_label(assemble_feature_matrix())
    train = split_by_date(full, TRAIN_RANGE)

    model = GaussianHMM(n_components=params["n_states"], covariance_type="full")
    model.startprob_ = params["startprob"]
    model.transmat_ = params["transmat"]
    model.means_ = params["means"]
    model.covars_ = np.array(params["covars"], float) + 1e-6 * np.eye(len(OBS_COLS))

    raw = apply_winsor(train[OBS_COLS].to_numpy(float),
                       (params["winsor_lo"], params["winsor_hi"]))
    obs = apply_scaler(raw, (params["scaler_mean"], params["scaler_std"]))
    decoded = model.predict(obs)
    dirs = train["direction"].to_numpy()

    mix = np.zeros((params["n_states"], 3))
    for s in range(params["n_states"]):
        idx = decoded == s
        if idx.sum() == 0:
            mix[s] = np.array([1 / 3, 1 / 3, 1 / 3])
        else:
            mix[s] = np.bincount(dirs[idx], minlength=3) / idx.sum()
    return mix


def main():
    params = load_params()
    obs, latest_date = latest_observation()

    # Hard mapping (point label) and calibrated soft mapping (honest probs).
    mix = empirical_state_mix(params)
    belief, next_state, hard_prob = predict(params, obs)
    _, _, soft_prob = predict(params, obs, calibrated_mix=mix)

    label = DIRECTIONS[int(hard_prob.argmax())]

    print("=" * 60)
    print("XEJ NEXT-TRADING-DAY OPEN PREDICTION")
    print("=" * 60)
    print(f"Latest trading day in data : {latest_date.date()}")
    print(f"Predicting open for         : next trading day after {latest_date.date()}")
    print(f"  (weekends/holidays handled by the trading calendar)")
    print(f"\nObservation (d_GPR, d_Oil, d_Close): "
          f"{', '.join(f'{v:+.4f}' for v in obs)}")
    print(f"State belief today          : "
          f"{', '.join(f'{b:.2f}' for b in belief)}")

    print(f"\n--- Prediction ---")
    print(f"Hard label (point estimate) : {label.upper()}")
    print(f"Calibrated probabilities    :")
    for d, name in enumerate(DIRECTIONS):
        bar = "#" * int(round(soft_prob[d] * 30))
        print(f"    {name:>4}: {soft_prob[d]:.3f}  {bar}")

    top = soft_prob.max()
    if top < 0.45:
        print("\nNote: probabilities are close to even, the model expresses low "
              "confidence, consistent with next-day direction being near-random "
              "for this target.")


if __name__ == "__main__":
    main()