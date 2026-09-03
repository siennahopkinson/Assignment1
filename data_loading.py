import pandas as pd


def load_env_data(filepath):
    """Load and parse the environmental sensor log.

    Reads the environmental sensor CSV, which has a proper header row,
    and converts its timestamp column to a pandas datetime type so it
    can later be merged against the PMV records.

    Args:
        filepath (str): Path to the Env_data_MSI_Lab CSV file.

    Returns:
        pandas.DataFrame: The environmental data with columns
            'time', 'Pressure (mb)', 'Temperature (°C)',
            'Humidity (RH %)', 'CO2 (ppm)', 'LVOC (ppb)', and
            'Formaldehyde (µg/m3)', where 'time' is a datetime64 column.
    """
    df_env = pd.read_csv(filepath)
    df_env["time"] = pd.to_datetime(df_env["time"])
    return df_env

def load_pmv_data(filepath):
  """Load and parse the PMV data.

  Reads the pmv sensor CSV, which has no proper header row,
  and takes the two existing time and date columns and combines
  them into one string, converts that into an actual datetime64 value
  so it can later be merged against the ENV records.

  Args:
      filepath (str): Path to the PMV_07_25 - 04_26.csv CSV file.

  Returns:
      pandas.DataFrame: The PMV data with its original columns
          ('date', 'time', 'col3'-'col9', all as read from the file),
          plus a new 'datetime' column containing the combined date
          and time parsed as a single datetime64 value.
  """
  df_pmv = pd.read_csv(filepath, header=None,
      names=["date","time","col3","col4","col5","col6","col7","col8","col9"])
  df_pmv["datetime"] = pd.to_datetime(df_pmv["date"] + " " + df_pmv["time"],
      format="%d/%m/%Y %H:%M:%S", errors="coerce")
  return df_pmv
