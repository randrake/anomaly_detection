import pandas as pd
import numpy as np

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans feature columns in a DataFrame:
    - Only processes columns between the first (timestamp) and second-to-last (class).
    - Drops columns that are entirely NaN or frozen (constant value).
    - Returns the cleaned DataFrame with timestamp, cleaned features, and class columns.
    """

    # Identify columns
    cols = df.columns
    timestamp_col = cols[0]
    class_col = cols[-2]             # second-to-last column is class
    feature_cols = cols[1:-2]        # features: columns after timestamp up to (not including) class

    df_clean = df.copy()
    feature_df = df_clean[feature_cols]

    # Drop fully NaN columns
    feature_df = feature_df.dropna(axis=1, how='all')

    # Drop frozen columns (constant value)
    non_frozen_cols = feature_df.loc[:, feature_df.nunique(dropna=False) > 1]

    # Reconstruct DataFrame with timestamp + cleaned features + class
    cleaned_df = pd.concat(
        [df_clean[[timestamp_col]].reset_index(drop=True),
         non_frozen_cols.reset_index(drop=True),
         df_clean[[class_col]].reset_index(drop=True)],
        axis=1
    )

    return cleaned_df


#############################################################################################################################################
import pandas as pd
import numpy as np
import random
from pathlib import Path
from collections import defaultdict

def clean_with_peer_imputation(csv_dir: str, n_peers=10, missing_threshold=3):
    """
    Cleans all CSVs in csv_dir by:
    1. Dropping files with > missing_threshold frozen or NaN target columns.
    2. Imputing missing/frozen columns using row-wise average of peer files
       (same well type and final class).

    Returns:
        list of cleaned DataFrames
    """
    csv_dir = Path(csv_dir)
    files = list(csv_dir.glob("*.csv"))
    target_cols = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'P-JUS-CKGL', 'T-JUS-CKGL', 'QGL']
    cleaned = []

    # Build peer dictionary: (well_type, class) -> list of files
    peer_dict = defaultdict(list)

    for file in files:
        try:
            df = pd.read_csv(file)
            final_class = df['class'].dropna().iloc[-1]
            if not (0 <= final_class <= 8):
                continue

            fname = file.name.lower()
            if fname.startswith("well"):
                well_type = "well"
            elif fname.startswith("drawn"):
                well_type = "drawn"
            elif fname.startswith("simulated"):
                well_type = "simulated"
            else:
                continue

            peer_dict[(well_type, int(final_class))].append(file)

        except Exception:
            continue

    # Clean each file
    for file in files:
        try:
            df = pd.read_csv(file)

            if not all(col in df.columns for col in target_cols):
                continue

            final_class = df['class'].dropna().iloc[-1]
            if not (0 <= final_class <= 8):
                continue

            fname = file.name.lower()
            if fname.startswith("well"):
                well_type = "well"
            elif fname.startswith("drawn"):
                well_type = "drawn"
            elif fname.startswith("simulated"):
                well_type = "simulated"
            else:
                continue

            feature_df = df[target_cols].copy()
            missing_or_frozen = 0
            frozen_mask = []

            for col in target_cols:
                series = feature_df[col]
                is_missing = series.isna().all()
                is_frozen = (series.nunique(dropna=True) <= 1)
                frozen_mask.append(is_missing or is_frozen)
                if is_missing or is_frozen:
                    missing_or_frozen += 1

            # Drop file if too many missing/frozen
            if missing_or_frozen > missing_threshold:
                continue

            # Impute missing/frozen columns
            for i, col in enumerate(target_cols):
                if not frozen_mask[i]:
                    continue  # column is OK

                # Get peers
                peers = [f for f in peer_dict[(well_type, int(final_class))] if f != file]
                random.shuffle(peers)
                peer_dfs = []
                for peer_file in peers[:n_peers]:
                    try:
                        peer_df = pd.read_csv(peer_file)
                        if col in peer_df.columns:
                            peer_series = peer_df[col]
                            if not (peer_series.isna().all() or peer_series.nunique(dropna=True) <= 1):
                                peer_dfs.append(peer_series)
                    except:
                        continue

                if peer_dfs:
                    max_len = len(df)
                    padded = []
                    for ps in peer_dfs:
                        temp = ps.copy()
                        if len(temp) < max_len:
                            pad_len = max_len - len(temp)
                            temp = pd.concat([temp, pd.Series([np.nan]*pad_len)], ignore_index=True)
                        padded.append(temp.values[:max_len])
                    avg = np.nanmean(np.stack(padded), axis=0)
                    df[col] = avg
                else:
                    df[col] = 0  # fallback if no good peer

            cleaned.append(df)

        except Exception as e:
            #print(f"⚠️ Skipped {file.name}: {e}")
            continue

    print(f"\n Cleaned {len(cleaned)} out of {len(files)} files.")
    return cleaned

###############################################################################################################
#PADDING
import numpy as np

def padding(data, max_rows, pad_value=0):
    """
    Pads each 2D array in X_train to have max_rows rows by adding rows of pad_value.

    Args:
        X_train (list of np.ndarray): List of 2D arrays, each with shape (num_rows_i, num_features).
        max_rows (int): Target number of rows to pad/truncate each array to.
        pad_value (numeric, optional): Value to pad with. Default is 0.

    Returns:
        list of np.ndarray: List of padded arrays, each shape (max_rows, num_features).
    """
    padded_list = []
    for arr in data:
        n_rows, n_cols = arr.shape
        if n_rows >= max_rows:
            # truncate if longer than max_rows
            padded_arr = arr[:max_rows, :]
        else:
            # pad with pad_value rows at the bottom
            pad_rows = max_rows - n_rows
            padding = np.full((pad_rows, n_cols), pad_value)
            padded_arr = np.vstack([arr, padding])
        padded_list.append(padded_arr)
    return padded_list
