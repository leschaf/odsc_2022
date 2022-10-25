import pandas as pd
import tsfresh as ts

import accessor
from ml_timeseries import build_time_series, generate_tsfresh_features

HISTORY_STEPS = 6
FORECAST_STEPS = 3

# read sample data
data = pd.read_csv("example_data_input.csv")

# pre-process into format for direct forecasting approach
args = {"history_steps": HISTORY_STEPS, "forecast_steps": FORECAST_STEPS}
data_processed = data.parallelize.groupby_apply(
    func=build_time_series, groupby_ls=["item"], **args
)

# automated feature engineering with tsfresh
settings = ts.feature_extraction.settings.MinimalFCParameters()
for col in ["m-" + str(x) for x in range(1, HISTORY_STEPS)]:
    extracted_features = generate_tsfresh_features(
        data_processed, col, settings
    ).reset_index()

    extracted_features = extracted_features.reset_index().rename(
        {"index": "item"}, axis=1
    )

    data_processed = data_processed.merge(extracted_features, how="left", on="item")

data_processed.to_csv("example_data_output.csv")
