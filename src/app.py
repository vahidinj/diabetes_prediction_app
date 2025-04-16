import streamlit as st
from predictions_page import prediction_page
from data_exploration_page import data_exploration
from model_evaluation_page import model_evaluation
from about_page import about_page

# st.set_page_config(layout='wide')


def main():
    # Define a dictionary to map page names to their corresponding functions
    pages = {
        "About Page": about_page,
        "Prediction Page": prediction_page,
        "Data Exploration Page": data_exploration,
        "Model Evaluation Page": model_evaluation,
    }
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.selectbox("Select a Page", list(pages.keys()))

    # Call the selected page's function
    st.title(selected_page)
    pages[selected_page]()  # Dynamically call the function


if __name__ == "__main__":
    main()
