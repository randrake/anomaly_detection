import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def anomally_division(train_path, save_dir, seed=42):
    """
    - Selects 50 random CSV files from train_path
    - Plots each CSV in a 3x3 grid of numeric columns vs timestamp
    - Highlights normal, transient, and steady anomaly sections
    - Saves each figure to save_dir
    - Uses random seed for reproducibility
    """

    random.seed(seed)
    train_path = Path(train_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect all CSVs
    csv_files = list(train_path.glob("*.csv"))
    if len(csv_files) < 50:
        raise ValueError(f"Found only {len(csv_files)} files, but need at least 50.")

    # Sample 50 files
    selected_files = random.sample(csv_files, 50)

    for idx, csv_file in enumerate(selected_files, 1):
        df = pd.read_csv(csv_file)

        # DEBUG: print columns to verify structure
        # print(f"\n[{idx}/50] Processing {csv_file.name}: columns = {list(df.columns)}")

        # Skip files with too few columns
        if df.shape[1] < 3:  # timestamp + at least 1 feature + class
            print(f"Skipping {csv_file.name}: not enough columns ({df.shape[1]} columns).")
            continue

        # Identify timestamp and class columns
        timestamp = df.columns[0]
        class_col = 'class'  # second-to-last column as class

        if class_col not in df.columns:
            print(f"Skipping {csv_file.name}: class column '{class_col}' not found.")
            continue


        # Convert timestamp to datetime
        df[timestamp] = pd.to_datetime(df[timestamp], errors='coerce')
        df = df.dropna(subset=[timestamp]).sort_values(by=timestamp)

        # Feature columns between timestamp and class
        feature_cols = df.columns[1:-1]
        if not len(feature_cols):
            print(f"Skipping {csv_file.name}: no feature columns found.")
            continue

        # Determine grid size
        num_plots = len(feature_cols)
        rows, cols = 3, 3
        total_subplots = rows * cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 8), sharex=False)
        axes = axes.flatten()

       
        # Class sections for coloring
        normal_condition = df[class_col] == 0
        steady_condition = df[class_col].between(1, 8)
        transient_condition = ~(normal_condition | steady_condition)

        class_sections = {
            "Normal": {
                "condition": normal_condition,
                "color": "green",
                "bg": "lightgreen",
                "label": "Normal"
            },
            "Transient Anomaly": {
                "condition": transient_condition,
                "color": "orange",
                "bg": "mistyrose",
                "label": "Transient Anomaly"
            },
            "Steady Anomaly": {
                "condition": steady_condition,
                "color": "red",
                "bg": "lightcoral",
                "label": "Steady Anomaly"
            }
        }


        for i, col in enumerate(feature_cols[:total_subplots]):
            ax = axes[i]

            for key, section in class_sections.items():
                mask = section["condition"]
                ax.plot(df[timestamp][mask], df[col][mask], label=section["label"], color=section["color"], linewidth=0.8)
                ax.fill_between(df[timestamp], df[col].min(), df[col].max(), where=mask, color=section["bg"], alpha=0.2)

            ax.set_title(col, fontsize=10)
            ax.legend(loc='upper right', fontsize=6, frameon=True, framealpha=0.8, edgecolor="black", fancybox=True)
            ax.set_xlabel("Time")
            ax.tick_params(axis="x", rotation=30)
            ax.grid(False)

        for i in range(num_plots, total_subplots):
            fig.delaxes(axes[i])

        fig.suptitle(f"{csv_file.stem}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the figure
        save_path = save_dir / f"{idx:02d}_{csv_file.stem}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        #print(f" Saved: {save_path}")
    print ("Anomally division visualisation...")