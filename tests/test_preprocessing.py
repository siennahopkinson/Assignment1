import pandas as pd
from preprocessing import clean_model_data, filter_overlap


def test_clean_model_data_drops_missing_required_columns():
    """clean_model_data should drop rows missing any required column."""
    data = pd.DataFrame({
        "Pressure (mb)": [1000, None, 1002],
        "Temperature (°C)": [22, 23, 24],
        "Humidity (RH %)": [50, 51, 52],
        "CO2 (ppm)": [400, 410, 420],
        "col7": [0.1, 0.2, None],
    })
    result = clean_model_data(data)
    assert len(result) == 1
    assert result.iloc[0]["Pressure (mb)"] == 1000


def test_filter_overlap():
    """Tests that filter_overlap restricts data to the time range of reference data.

    Creates a target dataframe and a reference dataframe with distinct date
    ranges, asserting that rows outside the reference start and end bounds are
    dropped.
    """
    env_df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2026-01-05", "2026-01-10", "2026-01-15", "2026-01-20"]
            )
        }
    )
    merged_df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-05",
                    "2026-01-12",
                    "2026-01-20",
                    "2026-01-25",
                ]
            ),
            "value": [1, 2, 3, 4, 5],
        }
    )
    filtered = filter_overlap(merged_df, env_df)
    assert len(filtered) == 3
    assert filtered["datetime"].min() == pd.Timestamp("2026-01-05")
    assert filtered["datetime"].max() == pd.Timestamp("2026-01-20")
