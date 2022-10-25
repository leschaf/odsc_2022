import pandas as pd
import tsfresh as ts


def build_time_series(
    name: str, df_grouped: pd.DataFrame, history_steps: int, forecast_steps: int
) -> pd.DataFrame:
    """Builds time series for a forecast item @name:
       - m-1 ... m-@history_steps for historical values
       - target+0 ... target+@forecast steps for future values (training targets)

    Args:
        name (str): forecasted item
        df_grouped (pd.DataFrame): data for @name
        history_steps (int): length of history
        forecast_steps (int): maximum predicted time step
    Returns:
        pd.DataFrame: dataframe with one row for each date at wich @name is forecasted
    """
    df = df_grouped.groupby("date").agg({"value": "sum"}).reset_index()

    # create a new column for each point of relative history
    # shift data backwards
    for x in range(1, history_steps):
        df["m-" + str(x)] = df["value"].shift(x)
    # create a new column for reach forecast target
    # shift data forwards
    for x in range(forecast_steps):
        df["target+" + str(x)] = df["value"].shift(-x)

    df["item"] = name

    return df


def generate_tsfresh_features(
    data: pd.DataFrame,
    column: str,
    settings: ts.feature_extraction.settings.MinimalFCParameters,
) -> pd.DataFrame:
    """Generates tsfresh features for a given column,
       grouped by date & forecasted item

    Args:
        data (pd.DataFrame): data input
        column (str): column to calculate features on
        settings (ts.feature_extraction.settings.MinimalFCParameters):
                 features to generate

    Returns:
        pd.DataFrame: _description_
    """

    extracted_features = ts.extract_features(
        data[["date", "item"] + [column]].dropna(axis=0),
        default_fc_parameters=settings,
        column_id="item",
        column_sort="date",
    )
    return extracted_features
