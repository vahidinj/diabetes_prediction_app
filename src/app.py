import streamlit as st
from models.logistic_model import df
from predictions_page import prediction_page
from data_exploration_page import data_exploration
from model_evaluation_page import model_evaluation

# st.set_page_config(layout='wide')


def main():
    page = st.sidebar.selectbox(
        "PAGES", ("Prediction Page", "Data Exploration Page", "Model Evaluation Page")
    )

    if page == "Prediction Page":
        st.title("Diabetes Prediction Page")
        prediction_page()
    elif page == "Data Exploration Page":
        st.title("Data Exploration Page")
        st.subheader("Data and Data Description Statistics")
        df
        data_exploration()
    else:
        st.title("Model Evaluation Page")
        model_evaluation()


if __name__ == "__main__":
    main()
