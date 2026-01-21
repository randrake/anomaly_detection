import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path





def vis_plot(
    data_dir: str,
    n_samples: int = 20,
    output_dir: str = "random_sample_figures",
    seed: int = 42,
    nrows: int = 3,
    ncols: int = 3,
):
    """
    Randomly selects n_samples CSV files from data_dir,
    plots multiple numeric columns vs. timestamp in nrows x ncols grids, and saves the figures.

    Args:
        data_dir (str): Path to directory containing CSVs.
        n_samples (int): Number of random files to select.
        output_dir (str): Directory where plots will be saved.
        seed (int): Random seed for reproducibility.
        nrows (int): Number of rows in grid.
        ncols (int): Number of columns in grid.
    """

    random.seed(seed)

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")
    
    csv_files = list(data_path.glob("*.csv"))
    if len(csv_files) < n_samples:
        raise ValueError(f"Found only {len(csv_files)} CSVs, but requested {n_samples} samples.")

    random_files = random.sample(csv_files, n_samples)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, csv_file in enumerate(random_files, 1):
        df = pd.read_csv(csv_file)

        # Assume first column is timestamp
        timestamp_col = df.columns[0]
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        df = df.dropna(subset=[timestamp_col]).sort_values(by=timestamp_col)

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if timestamp_col in numeric_cols:
            numeric_cols.remove(timestamp_col)

        num_plots = min(len(numeric_cols), nrows * ncols)
        if num_plots == 0:
            print(f"⚠️ Skipping {csv_file.name}: no numeric columns.")
            continue

        colors = plt.cm.tab10(np.linspace(0, 1, num_plots))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 10))
        axes = axes.flatten()

        fig.suptitle(f" #{idx}: {csv_file.name}", fontsize=14)

        for i, (col, color) in enumerate(zip(numeric_cols[:num_plots], colors)):
            ax = axes[i]
            ax.plot(df[timestamp_col], df[col], color=color)
            ax.set_title(col, fontsize=10)
            ax.set_xlabel("Time")
            ax.set_ylabel(col)
            ax.tick_params(axis='x', rotation=30)
            ax.grid(True)

        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        save_path = output_path / f"{idx:02d}_{csv_file.stem}_grid.png"
        plt.savefig(save_path, dpi=150)
        plt.close()

        #print(f" Saved figure: {save_path}")

    print(f"\n All {n_samples} random grid plots saved to '{output_path}'.")
    #######################################################################################################################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
from collections import defaultdict

def variation(csv_files, 
                                       feature_exclude=['timestamp', 'class', 'instance_label'], 
                                       class_col='class',
                                       n_samples=1061):
    """
    For a random sample of files, count how often each feature is present per class.
    Plots a grouped bar chart with:
        X-axis: class (0–8)
        Y-axis: count of presence
        Bars: different features in color
    """
    random.seed(42)
    sampled_files = random.sample(csv_files, n_samples)

    # Map: class → feature → presence count
    presence_counts = defaultdict(lambda: defaultdict(int))

    for file in sampled_files:
        try:
            df = pd.read_csv(file)

            if class_col not in df.columns:
                continue

            final_class = df[class_col].dropna().iloc[-1]
            if not (0 <= final_class <= 8):
                continue

            final_class = int(final_class)
            feature_cols = [col for col in df.columns if col not in feature_exclude and pd.api.types.is_numeric_dtype(df[col])]

            for col in feature_cols:
                series = df[col]
                if not (series.isna().all() or np.all(series == 0) or series.nunique(dropna=True) <= 1):
                    presence_counts[final_class][col] += 1  # feature is present

        except Exception as e:
            print(f"⚠️ Skipped {file.name}: {e}")
            continue

    # Convert to DataFrame for plotting
    all_classes = list(range(9))
    all_features = sorted({col for d in presence_counts.values() for col in d.keys()})
    data = []

    for cls in all_classes:
        row = {'class': cls}
        for feat in all_features:
            row[feat] = presence_counts[cls].get(feat, 0)
        data.append(row)

    plot_df = pd.DataFrame(data)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.1
    positions = np.arange(len(plot_df['class']))

    for i, feat in enumerate(all_features):
        ax.bar(positions + i * bar_width, plot_df[feat], width=bar_width, label=feat)

    ax.set_xticks(positions + (len(all_features)-1) * bar_width / 2)
    ax.set_xticklabels(plot_df['class'])
    ax.set_xlabel("Class")
    ax.set_ylabel("Presence Count in Sampled Files")
    ax.set_title("Feature Presence per Class (out of sampled files)")
    ax.legend(title="Feature", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return plot_df


#######################################################################################################################################################################################################

def corr_plot(
    data_dir: str,
    output_dir: str = "csvs_grid_plots",
    nrows: int = 4,
    ncols: int = 4,
):
    """
    Loops through all CSV files in data_dir, plots them in batches of nrows*ncols per figure,
    each CSV in one subplot, plotting feature columns (columns between timestamp and class column)
    vs. timestamp. Saves each figure to output_dir.
    """

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")
    
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_path}.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    batch_size = nrows * ncols
    total_batches = (len(csv_files) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(csv_files))
        batch_files = csv_files[start:end]

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 12))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i >= len(batch_files):
                ax.axis('off')
                continue

            csv_file = batch_files[i]
            df = pd.read_csv(csv_file)

            timestamp_col = df.columns[0]
            class_col = df.columns[-2]
            feature_cols = df.columns[1:-1]  # precisely: columns between timestamp and class label

            if len(feature_cols) == 0:
                ax.set_title(f"{csv_file.stem} - No features", fontsize=8)
                ax.axis('off')
                continue

            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
            df = df.dropna(subset=[timestamp_col]).sort_values(by=timestamp_col)

            num_features = len(feature_cols)
            colors = plt.cm.tab10(np.linspace(0, 1, num_features))

            for col, color in zip(feature_cols, colors):
                ax.plot(df[timestamp_col], df[col], color=color, linewidth=0.8, label=col)

            ax.set_title(f"{start + i + 1}. {csv_file.stem}", fontsize=8)
            ax.tick_params(axis='x', labelsize=6, rotation=30)
            ax.tick_params(axis='y', labelsize=6)
            ax.grid(True)

            # Add legend with small font size
            ax.legend(loc='upper right', fontsize=6, frameon=False)

        plt.tight_layout()
        save_name = output_path / f"csvs_grid_batch_{batch_idx+1:03d}.png"
        plt.savefig(save_name, dpi=150)
        plt.close()

        # print(f" Saved batch {batch_idx+1}/{total_batches}: {save_name}")

    print(f"\n Plots saved in '{output_path}'.")


#######################################################################################################################################################################################################

def corr_plot_clean(
    data_dir: str,
    output_dir: str = "csvs_grid_plots",
    nrows: int = 4,
    ncols: int = 4,
):
    """
    Loops through all CSV files in data_dir, plots them in batches of nrows*ncols per figure,
    each CSV in one subplot, plotting feature columns (columns between timestamp and class column)
    vs. timestamp. Saves each figure to output_dir.
    """

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_path}")
    
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_path}.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    batch_size = nrows * ncols
    total_batches = (len(csv_files) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(csv_files))
        batch_files = csv_files[start:end]

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 12))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i >= len(batch_files):
                ax.axis('off')
                continue

            csv_file = batch_files[i]
            df = pd.read_csv(csv_file)

            timestamp_col = df.columns[0]
            class_col = df.columns[-1]
            feature_cols = df.columns[1:]  # precisely: columns after timestamp and including class label

            if len(feature_cols) == 0:
                ax.set_title(f"{csv_file.stem} - No features", fontsize=8)
                ax.axis('off')
                continue

            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
            df = df.dropna(subset=[timestamp_col]).sort_values(by=timestamp_col)

            num_features = len(feature_cols)
            colors = plt.cm.tab10(np.linspace(0, 1, num_features))

            for col, color in zip(feature_cols, colors):
                ax.plot(df[timestamp_col], df[col], color=color, linewidth=0.8, label=col)

            ax.set_title(f"{start + i + 1}. {csv_file.stem}", fontsize=8)
            ax.tick_params(axis='x', labelsize=6, rotation=30)
            ax.tick_params(axis='y', labelsize=6)
            ax.grid(True)

            # Add legend with small font size
            ax.legend(loc='upper right', fontsize=6, frameon=False)

        plt.tight_layout()
        save_name = output_path / f"csvs_grid_batch_{batch_idx+1:03d}.png"
        plt.savefig(save_name, dpi=150)
        plt.close()

        # print(f"Saved batch {batch_idx+1}/{total_batches}: {save_name}")

    print(f"\n Plots saved in '{output_path}'.")
