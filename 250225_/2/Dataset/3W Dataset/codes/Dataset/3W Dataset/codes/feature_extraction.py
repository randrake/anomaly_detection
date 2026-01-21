import numpy as np

def extract_features(dataframes, window_size=180, stride=30, class_col="class"):
    """
    Extracts sliding window features from a list of cleaned & normalized DataFrames.

    For each window:
    - Computes mean, std, min, max, median, var for each feature column.
    - Uses the class label at the last time point of the window.

    Args:
        dataframes (list of pd.DataFrame): list of cleaned DataFrames.
        window_size (int): number of rows per window.
        stride (int): step size between windows.
        class_col (str): name of the class/label column.

    Returns:
        tuple: X (2D numpy array of features), y (1D numpy array of labels).
    """

    features_list = []
    labels_list = []

    for df_idx, df in enumerate(dataframes, 1):
        if len(df) < window_size:
            continue  # skip short series

        # Feature columns: skip timestamp & class
        feature_cols = df.columns[1:-1]

        # Slide over the dataframe
        for start_idx in range(0, len(df) - window_size + 1, stride):
            window = df.iloc[start_idx : start_idx + window_size]

            features = []
            for col in feature_cols:
                series = window[col]
                features.extend([
                    series.mean(),
                    series.std(),
                    series.min(),
                    series.max(),
                    series.median(),
                    series.var()
                ])

            # Use label at last row of window
            window_label = window[class_col].iloc[-1]

            features_list.append(features)
            labels_list.append(window_label)

        #print(f" Processed DataFrame {df_idx}/{len(dataframes)} with {len(df)} rows.")

    X = np.array(features_list)
    y = np.array(labels_list)
    print(f"\n Extracted features from {len(features_list)} windows.")
    return X, y
