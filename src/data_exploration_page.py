import streamlit as st

from models.logistic_model import df
from utils.helper import line_chart, heat_map


def data_exploration():
    st.session_state.setdefault("show_describe", False)

    if st.button("Show/Hide Description"):
        st.session_state.show_describe = (
            True if not st.session_state.show_describe else False
        )
    if st.session_state.show_describe:
        st.table(df.describe())

    st.subheader("Data Visualization")
    st.write("#### Line Charts")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        x = st.selectbox("Select X Value", df.columns)
    with col2:
        y = st.selectbox("Select Y Value", df.columns, index=5)
    with col3:
        hue = st.selectbox("Select hue", df.columns, index=6)

    st.pyplot(line_chart(df, x=x, y=y, hue=str(hue)))

    st.write("#### Heat Maps")
    st.write("##### Heat Map showing the correlation between features")
    st.pyplot(heat_map(data=df.corr()))
    st.write(
        "##### Heat Map showing the correlation of features with the Outcome Variable"
    )
    st.pyplot(heat_map(data=df.corr()[["Outcome"]].drop("Outcome")))
