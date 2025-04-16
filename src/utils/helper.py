import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Data/diabetes.csv")

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
    """
    Returns a line chart of the specified data.

    Args:
        data (pd.DataFrame): The dataset containing the data to be plotted.
        x (str): The column name to be used for the x-axis.
        y (str): The column name to be used for the y-axis.
        hue (str): The column name to be used for grouping and color coding.

    Returns:
        matplotlib.figure.Figure: The generated line chart figure.
    """
    fig, ax = plt.subplots()
    ax.set_title(f"{x} vs {y}")
    sns.lineplot(data=data, x=x, y=y, hue=hue)
    return fig


def box_plot(data: pd.DataFrame):
    """
    Returns a box plot of the specified data.

    Args:
        data (pd.DataFrame): The dataset to be visualized as a box plot.

    Returns:
        matplotlib.figure.Figure: The generated box plot figure.
    """
    fig, ax = plt.subplots()
    sns.boxplot(data=data, ax=ax)
    ax.tick_params(axis="x", rotation=45)  # Rotate x-axis labels by 45 degrees
    return fig


def heat_map(data):
    """
    Returns a heatmap of the specified data.

    Args:
        data (pd.DataFrame): The dataset to be visualized as a heatmap.

    Returns:
        matplotlib.figure.Figure: The generated heatmap figure.
    """
    fig, ax = plt.subplots()
    sns.heatmap(data=data, annot=True, cmap="coolwarm")
    return fig
