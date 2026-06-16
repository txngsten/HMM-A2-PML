
import pandas as pd
import numpy as np
import yfinance as yf

RANDOM_SEED = 42
START_DATE = "2018-01-01"
END_DATE = "2025-01-01"

XEJ_TICKER = "^AXEJ"
OIL_TICKER = "CL=F"
GPR_INDEX_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"


def signed_log1p(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


"""
Gets the GPR Index data from GPR_INDEX_URL read into a pandas dataframe.
Gets only the month and GPR values from the dataframe.
Then log(x + 1) transforms the data as done in the A1 ETL and EDA process.
"""
def load_gpr():
    df = pd.read_excel(GPR_INDEX_URL, usecols=["month", "GPR"])
    df["month"] = pd.to_datetime(df["month"], errors="coerce")

    # Sorts by month
    df = df.dropna(subset=["month", "GPR"]).sort_values("month").reset_index(drop=True)

    # log(x + 1) transforms the GPR value column
    df["GPR_log"] = np.log1p(df["GPR"])
    return df


"""
Filters the GPR Index data by the data contract period for train/test/val process.
"""
def gpr_train():
    df = load_gpr()
    mask = (df["month"] >= START_DATE) & (df["month"] <= END_DATE)
    return df.loc[mask].reset_index(drop=True)

"""
Gets the latest GPR Index value and month.
"""
def gpr_latest():
    df = load_gpr()
    return df.tail(1).reset_index(drop=True)

"""
Gets data from yfinance based on a ticker, start, and end date period.
"""
def download_yf_data(ticker, start, end):
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        # Flatten if multi-ticker shape
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")


"""
Gets the XEJ (ASX 200 Energy Index) daily Open, High, Low, Close.
The target for the HMM is next-day Open (predict row t's Open using information available up to row t-1)
"""
def load_xej():
    df = download_yf_data(XEJ_TICKER, START_DATE, END_DATE)
    df = df[["Open", "High", "Low", "Close"]].copy()

    # log(x + 1) transform on all price columns (matches A1 approach)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = np.log1p(df[col])

    df["Close_prev"] = df["Close"].shift(1)
    df = df.dropna(subset=["Close_prev"]).reset_index()
    df = df.rename(columns={"Date": "date"})

    return df

"""
Gets WTI oil daily close, log(x + 1) transformed.
Returned on the OIL market's own trading calendar.
"""
def load_oil():
    df = download_yf_data(OIL_TICKER, START_DATE, END_DATE)
    df = df[["Close"]].copy()
    df = df.rename(columns={"Close": "Oil_close"})

    # log(x + 1) transform on all price columns (matches A1 approach)
    df["Oil_close"] = signed_log1p(df["Oil_close"])
    df = df.dropna(subset=["Oil_close"]).reset_index()
    df = df.rename(columns={"Date": "date"})

    return df

"""
Joins XEJ and oil onto the XEJ trading calendar for the HMM.
XEJ defines the trading days (target market). Oil is left-joined because
US/NYMEX holidays differ from the ASX, some XEJ trading days have no oil print. 
Those gaps are forward-filled (last known oil price carried forward) rather than dropping the XEJ row.
"""
def build_datasets():
    xej = load_xej()
    oil = load_oil()

    merged = xej.merge(oil, on="date", how="left").sort_values("date")

    # Forward-fill oil into XEJ-only trading days (different holiday calendars)
    merged["Oil_close"] = merged["Oil_close"].ffill()

    # Any leading NaN (oil missing before its first print), back-fill once
    merged["Oil_close"] = merged["Oil_close"].bfill()

    return merged.reset_index(drop=True)


"""
Returns the single most recent trading day's feature row (XEJ).
"""
def xej_latest():
    return build_datasets().tail(1).reset_index(drop=True)


