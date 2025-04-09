import pandas as pd

df = pd.read_csv(
    "/Users/deanjupic/Documents/Python/Projects/diabetes_prediction_app/Data/diabetes.csv"
)

df = df.drop(["BloodPressure", "SkinThickness"], axis=1)


