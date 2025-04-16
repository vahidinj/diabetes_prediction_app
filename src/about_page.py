import streamlit as st


def about_page():
    st.title("About the Diabetes Prediction App")
    st.write("""
        This app is designed to predict the likelihood of diabetes based on diagnostic measurements.
        It provides tools for data exploration, model evaluation, and predictions.
        
        ### Data Info:
        This dataset is originally from the National Institute of Diabetes and Digestive and Kidney
        Diseases. The objective of the dataset is to diagnostically predict whether a patient has diabetes,
        based on certain diagnostic measurements included in the dataset. Several constraints were placed
        on the selection of these instances from a larger database. In particular, all patients here are females
        at least 21 years old of Pima Indian heritage.
        From the data set in the (.csv) File we can find several variables, some of them are independent
        (several medical predictor variables) and only one target dependent variable (Outcome).
        
        ### Features:
        - **Prediction Page**: Input diagnostic data to get predictions.
        - **Data Exploration Page**: Visualize and analyze the dataset.
        - **Model Evaluation Page**: Review the performance of the machine learning model.
        
        ### Threshold Selection:
        In this app, a threshold of **0.45** was chosen for the diabetes prediction model. This decision was
        made considering the medical context of the application. In healthcare, it is critical to minimize
        false negatives (i.e., cases where the model fails to identify someone who has diabetes). A lower
        threshold increases the recall (sensitivity), ensuring that more potential diabetes cases are identified,
        even if it slightly increases false positives. This trade-off is essential in medical applications where
        missing a diagnosis can have severe consequences for the patient.

        Built using Python and Streamlit.
    """)