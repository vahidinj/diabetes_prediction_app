import streamlit as st
from models.logistic_model import roc_curve_graph, cm, class_report_df, score, loss


@st.fragment()
def model_evaluation():
    st.write("## Model Performance Metrics")
    col1, col2, col3 = st.columns([3, 1, 1])

    # Display the classification report in its own row
    with st.container():
        st.subheader("Classification Report")
        st.data_editor(class_report_df)

    # Display the confusion matrix and accuracy/loss side by side
    col2, col3 = st.columns([1, 1])

    with col2:
        st.subheader("Confusion Matrix")
        st.data_editor(cm)

    with col3:
        st.subheader("Accuracy / Loss")
        metrics_data = {
            "Metric": ["Accuracy", "Loss"],
            "Value": [f"{score:.2f}", f"{loss:.4f}"],
        }
        st.data_editor(metrics_data)

        # Display ROC Curve
    # Initialize session state for the button
    st.session_state.setdefault("show_roc", False)

    # Toggle the state when the button is pressed
    if st.button("Show/Hide ROC Curve"):
        st.session_state.show_roc = True if not st.session_state.show_roc else False

    # Display the ROC curve based on the state
    if st.session_state.show_roc:
        fig = roc_curve_graph()
        st.pyplot(fig)
