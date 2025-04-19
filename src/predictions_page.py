import numpy as np
import streamlit as st
import pandas as pd

from models.logistic_model import df
from models.predictions import diabetes_prediction


def prediction_page():
    tab1, tab2 = st.tabs(["Predictions via Selections", "Batch Predictions"])

    with tab1:
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
            st.write(
                f"The binary prediction (0 = No, 1 = Yes): **{int(prediction[0])}**"
            )

            # About Prediction
            st.write(
                "Note: The prediction is based on a threshold of **0.45**, chosen to prioritize recall in a medical context, minimizing the risk of false negatives."
            )
            st.write("See **About Page**")

    with tab2:
        with st.expander("Acceptable Data Frame Features"):
            st.write(list(df.drop("Outcome", axis=1).columns))

        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

        if uploaded_file is not None:
            uploaded_df = pd.read_csv(uploaded_file)
            X_uploaded = uploaded_df.values

            probabilities, predictions = diabetes_prediction(X_uploaded)

            uploaded_df["probability"] = probabilities
            uploaded_df["prediction"] = predictions

            st.write("Prediction Results Sample:")
            st.dataframe(uploaded_df.sample(5))

            csv = uploaded_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                data=csv,
                label="Download Prediction Results as CSV",
                file_name="model_results.csv",
                mime="text/csv",
            )

            # results_file_path = "model_results.csv"
            # with open(results_file_path, "wt") as file:
            #     file.write(uploaded_df.to_csv(index=False).encode("utf-8"))
        else:
            st.info("Please upload a CSV file to make predictions")
