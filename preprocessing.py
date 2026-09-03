import pandas as pd


def merge_env_pmv(df_env, df_pmv, tolerance_minutes=30):
    """Merge PMV records with their nearest environmental reading.

    Sorts both datasets by time and matches each PMV record to the
    nearest environmental sensor reading within a given time tolerance,
    using a nearest-timestamp (as-of) merge. The PMV dataframe's original
    'date' and 'time' text columns are dropped first, since they are no
    longer needed once 'datetime' has been created, and because keeping
    them would collide with column names already present in df_env.

    Args:
        df_env (pandas.DataFrame): Environmental data as returned by
            load_env_data, with a 'time' datetime64 column.
        df_pmv (pandas.DataFrame): PMV data as returned by load_pmv_data,
            with a 'datetime' datetime64 column.
        tolerance_minutes (int): Maximum allowed gap, in minutes, between
            a PMV record's timestamp and its nearest environmental
            reading for them to be matched. Defaults to 30.

    Returns:
        pandas.DataFrame: One row per PMV record, with the nearest
            matching environmental reading's columns attached. PMV rows
            with no environmental reading within the tolerance window
            will have missing (NaN) values in the environmental columns.
    """
    df_pmv_sorted = df_pmv.drop(columns=["date", "time"]).sort_values("datetime")
    df_env_sorted = df_env.sort_values("time")
    merged = pd.merge_asof(
        df_pmv_sorted, df_env_sorted,
        left_on="datetime", right_on="time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )
    return merged

def filter_overlap(merged, df_env):
  """Uses merged dataframe to restrict it to rows within ENV data range

  Args:
      merged (pandas.DataFrame): Merged PMV and ENV data.
      df_env (pandas.DataFrame): Environmental data as returned by
          load_env_data
    
  Returns:
        (pandas.DataFrame): Subset of merged rows which
          datetime column is within ENV data range
  """
  overlap_start = df_env["time"].min()
  overlap_end = df_env["time"].max()
  merged_overlap = merged[(merged["datetime"] >= overlap_start) & (merged["datetime"] <= overlap_end)]
  return merged_overlap

def clean_model_data(merged_overlap):
    """Drops rows missing key environmental features or target PMV values.

    Filters the merged dataset to keep only complete observations required for
    model training and evaluation.

    Args:
        merged_overlap (pandas.DataFrame): Dataframe containing merged
            environmental metrics and PMV targets.

    Returns:
        pandas.DataFrame: Cleaned dataframe with non-null values for
            Pressure, Temperature, Humidity, CO2, and col7 (PMV).
    """
    required_cols = [
        "Pressure (mb)",
        "Temperature (°C)",
        "Humidity (RH %)",
        "CO2 (ppm)",
        "col7",
    ]
    model_data = merged_overlap.dropna(subset=required_cols)
    return model_data

