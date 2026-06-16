
import pandas as pd
import numpy as np

RANDOM_SEED = 42
START_DATE = "2018-01-01"
END_DATE = "2025-01-01"

GPR_INDEX_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"

"""
Gets the GPR Index data from GPR_INDEX_URL read into a pandas dataframe.
Gets only the month and GPR values from the dataframe.
Then log(x + 1) transforms the data as done in the A1 ETL and EDA process.
"""
def load_gpr():
    df = pd.read_excel(GPR_INDEX_URL, usecols=["month", "GPR"])
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.dropna(subset=["month", "GPR"]).sort_values("month").reset_index(drop=True)
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




