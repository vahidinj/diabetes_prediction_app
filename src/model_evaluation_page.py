import streamlit as st
from models.logistic_model import roc_curve_graph, compute_metrics
from utils.helper import select_threshold


@st.fragment()
def model_evaluation():
    # Select threshold dynamically
    selected_threshold = select_threshold()

    # Recompute metrics based on the selected threshold
    cm, class_report_df, score, loss = compute_metrics(selected_threshold)

    # Display the classification report
    st.subheader("Classification Report")
    st.data_editor(class_report_df)

    # Display the confusion matrix and accuracy/loss side by side
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Confusion Matrix")
        st.data_editor(cm)

    with col2:
        st.subheader("Accuracy / Loss")
        metrics_data = {
            "Metric": ["Accuracy", "Loss"],
            "Value": [f"{score:.2f}", f"{loss:.4f}"],
        }
        st.data_editor(metrics_data)

    # Display ROC Curve
    st.session_state.setdefault("show_roc", False)

    if st.button("Show/Hide ROC Curve"):
        st.session_state.show_roc = not st.session_state.show_roc

    if st.session_state.show_roc:
        fig = roc_curve_graph()
        st.pyplot(fig)
