import numpy as np
import streamlit as st

from models.logistic_model import df
from models.predictions import diabetes_prediction

# colums = df.columns
df.info()


# @st.cache
def prediction_page():
    st.write(
        "Adjust the below selections as they pertain to you. If unknown, leave as is (the preselected mean)."
    )

    pregnancies = st.selectbox(
        "Number of Pregnancies", sorted(df["Pregnancies"].unique())
    )
    glucose = st.slider(
        "Glucose Level",
        min_value=float(0),
        max_value=float(df["Glucose"].max()),
        value=float(df["Glucose"].mean()),
    )
    insulin = st.slider(
        "Insulin Level",
        min_value=float(0),
        max_value=float(df["Insulin"].max()),
        value=float(df["Insulin"].mean()),
    )
    bmi = st.slider(
        "BMI Level",
        min_value=float(0),
        max_value=float(df["BMI"].max()),
        value=float(df["BMI"].mean()),
    )
    diabetes_pedigree_function = st.slider(
        "Diabetes Pedigree Function",
        min_value=float(0),
        max_value=float(df["DiabetesPedigreeFunction"].max()),
        value=float(df["DiabetesPedigreeFunction"].mean()),
    )
    age = st.slider(
        "Age",
        min_value=int(df["Age"].min()),
        max_value=int(df["Age"].max()),
        value=int(df["Age"].mean()),
    )

    if st.button("Click to Calculate Probability"):
        X = np.array(
            [pregnancies, glucose, insulin, bmi, diabetes_pedigree_function, age]
        ).reshape(1, -1)

        # Get the probability and binary prediction from the prediction function
        probability, prediction = diabetes_prediction(X)

        # Display the probability
        st.write(f"The probability of having diabetes is: **{probability[0]:.3f}**")

        # Display the binary prediction
        st.write(f"The binary prediction (0 = No, 1 = Yes): **{int(prediction[0])}**")

        # About Prediction
        st.write(
            "Note: The prediction is based on a threshold of **0.45**, chosen to prioritize recall in a medical context, minimizing the risk of false negatives."
        )
        st.write("See **About Page**")
