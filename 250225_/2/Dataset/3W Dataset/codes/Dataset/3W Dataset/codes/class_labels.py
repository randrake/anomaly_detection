# Example class label mapping
class_labels = {
    0: "Normal",
    1: "Abrupt increase in BSW (Basic sediment and water)",
    2: "DHSV Spurious closure",
    3: "Severe intermittence",
    4: "Flow instability",
    5: "Fast productivity loss",
    6: "Fast restriction in CKP",
    7: "Incrustation in CKP",
    8: "Hydrate in production line",
}


def anomally_classs(df, class_col, class_labels):
    """
    Gets the class of the last row and returns a dict with its label.

    Args:
        df (pd.DataFrame): DataFrame with class column.
        class_col (str): Name of the class column in df.
        class_labels (dict): Mapping from class numbers to labels.

    Returns:
        dict: {class_number: class_label} for the last row only.
    """

    # Get the class value from the last row
    last_class = df[class_col].iloc[-1]

    # Look up its label
    label = class_labels.get(last_class, "Unknown")

    print(f"Last class {last_class}: {label}")

    return {last_class: label}

