def split_X_y(dataframes):
    """
    Splits each DataFrame into X (features) and y (class label of the last row).
    Features are columns between timestamp and class column.

    Args:
        dataframes (list of pd.DataFrame): cleaned DataFrames.

    Returns:
        tuple: (X_list, y_list), where:
            - X_list is a list of NumPy arrays (each shape: n_rows x n_features)
            - y_list is a list of scalar class labels (one per file)
    """

    X_list, y_list = [], []

    for idx, df in enumerate(dataframes, 1):
        cols = df.columns
        timestamp_col = cols[0]
        class_col = "class"
        feature_cols = cols[1:-1]  # drop timestamp and class

        if len(feature_cols) == 0:
            print(f" Skipping DataFrame {idx}: no feature columns.")
            continue

        X = df[feature_cols].values  # shape: (n_rows, n_features)
        y = df[class_col].iloc[-1]   # scalar: class of the last row

        X_list.append(X)
        y_list.append(y)

        print(f" DataFrame {idx}/{len(dataframes)}: X shape={X.shape}, y={y}")

    return X_list, y_list
