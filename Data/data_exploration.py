import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path: str) -> pd.DataFrame:
    """Loads the dataset from the given file path."""
    return pd.read_csv(file_path)


def basic_df_info(df: pd.DataFrame):
    """Prints basic information about the dataset."""
    print("Basic information about the dataset")
    print("First 5 rows of the dataset")
    print(df.head())
    print("\nDataset Description:")
    print(df.describe())
    print("\nDataset Info")
    print(df.info())


def heat_map_features(df: pd.DataFrame):
    """Plots a heatmap of feature correlations."""
    correlation = df.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(data=correlation.drop("Outcome", axis=1), annot=True, cmap="coolwarm")
    plt.title("Correlation between Features")
    plt.show()


def heat_map_outcome(df: pd.DataFrame):
    """Plots a heatmap of correlations with the Outcome variable."""
    correlation_outcome = df.corr()[["Outcome"]].drop("Outcome")
    plt.figure(figsize=(12, 8))
    sns.heatmap(data=correlation_outcome, annot=True, cmap="coolwarm", cbar=True)
    plt.title("Correlation with Outcome")
    plt.show()


def plot_glucose_vs_outcome(df: pd.DataFrame):
    """Plots Glucose vs Outcome."""
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x="Glucose", y="Outcome")
    plt.title("Glucose vs Outcome")
    plt.xlabel("Glucose")
    plt.ylabel("Outcome")
    plt.show()


def plot_bmi_vs_outcome(df: pd.DataFrame):
    """Plots BMI vs Outcome."""
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x="BMI", y="Outcome")
    plt.title("BMI vs Outcome")
    plt.xlabel("BMI")
    plt.ylabel("Outcome")
    plt.show()


def plot_age_vs_outcome(df: pd.DataFrame):
    """Plots Age vs Outcome."""
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x="Age", y="Outcome")
    plt.title("Age vs Outcome")
    plt.xlabel("Age")
    plt.ylabel("Outcome")
    plt.show()


def plot_bmi_vs_glucose(df: pd.DataFrame):
    """Plots BMI vs Glucose levels."""
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x="BMI", y="Glucose")
    plt.title("BMI vs Glucose")
    plt.xlabel("BMI")
    plt.ylabel("Glucose")
    plt.show()


def plot_age_vs_glucose(df: pd.DataFrame):
    """Plots Age vs Glucose levels with Outcome as hue."""
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x="Age", y="Glucose", hue="Outcome")
    plt.title("Age vs Glucose (Colored by Outcome)")
    plt.xlabel("Age")
    plt.ylabel("Glucose")
    plt.show()


# File path to the dataset
data_file_path = "Data/diabetes.csv"

# Load the dataset
df = load_data(file_path=data_file_path)

# Display basic dataset information
basic_df_info(df)

# Data visualization
heat_map_features(df)
heat_map_outcome(df)
plot_age_vs_outcome(df)
plot_glucose_vs_outcome(df)
plot_bmi_vs_outcome(df)
plot_bmi_vs_glucose(df)
plot_age_vs_glucose(df)
