import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(
    "/diabetes_prediction_app/Data/diabetes.csv"
)

df = df.drop(["BloodPressure", "SkinThickness"], axis=1)
columns = df.columns.to_list()


def feature_max(column: str):
    """
    Returns the maximum value of the specified column in the DataFrame.

    Args:
        column (str): The name of the column.

    Returns:
        float: The maximum value of the specified column.
    """

    if column not in columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

    return df[column].max()


def feature_min(column: str):
    """
    Returns the miniumum value of the specified column in the DataFrame.

    Args:
        column (str): The name of the column.

    Returns:
        float: The miniumum value of the specified column.
    """

    if column not in columns:
        raise ValueError(f"Column {column} does not exist in the DataFrame")

    return df[column].min()


def line_chart(data, x, y, hue):
    fig, ax = plt.subplots()
    ax.set_title(f"{x} vs {y}")
    sns.lineplot(data=data, x=x, y=y, hue=hue)
    return fig


def heat_map(data):
    fig, ax = plt.subplots()
    sns.heatmap(data=data, annot=True)
    return fig
