import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def minmax_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Min-Max scaling [0, 1] to feature columns (between timestamp and class label).
    Returns DataFrame with same structure (timestamp + scaled features + class).
    """
    cols = df.columns
    timestamp_col = cols[0]
    class_col = cols[-2]
    feature_cols = cols[1:-2]

    feature_data = df[feature_cols].values
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(feature_data)

    df_scaled = pd.concat(
        [
            df[[timestamp_col]].reset_index(drop=True),
            pd.DataFrame(scaled_features, columns=feature_cols),
            df[[class_col]].reset_index(drop=True),
        ],
        axis=1,
    )
    return df_scaled


def zscore_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Z-score standardization (mean=0, std=1) to feature columns (between timestamp and class label).
    Returns DataFrame with same structure (timestamp + scaled features + class).
    """
    cols = df.columns
    timestamp_col = cols[0]
    class_col = cols[-2]
    feature_cols = cols[1:-2]

    feature_data = df[feature_cols].values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_data)

    df_scaled = pd.concat(
        [
            df[[timestamp_col]].reset_index(drop=True),
            pd.DataFrame(scaled_features, columns=feature_cols),
            df[[class_col]].reset_index(drop=True),
        ],
        axis=1,
    )
    return df_scaled
