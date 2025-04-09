import pandas as pd

df = pd.read_csv(
    "Data/diabetes.csv"
)

df = df.drop(["BloodPressure", "SkinThickness"], axis=1)


