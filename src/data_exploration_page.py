import streamlit as st

from models.logistic_model import df
from utils.helper import line_chart, heat_map, box_plot


@st.fragment()
def data_exploration():
    st.data_editor(df)
    st.session_state.setdefault("show_describe", False)
    if st.button("Show/Hide Description"):
        st.session_state.show_describe = (
            True if not st.session_state.show_describe else False
        )
    if st.session_state.show_describe:
        st.data_editor(df.describe())
    st.divider()
    st.subheader("Data Visualization")

    tab1, tab2, tab3 = st.tabs(["Line Charts", "Box Plots", "Heat Maps"])

    with tab1:
        st.write("#### Line Charts")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            x = st.selectbox("Select X Value", df.columns)
        with col2:
            y = st.selectbox("Select Y Value", df.columns, index=5)
        with col3:
            hue = st.selectbox("Select hue", df.columns, index=6)

        st.pyplot(line_chart(df, x=x, y=y, hue=str(hue)))

    with tab2:
        st.write("#### Box Plots")
        st.write("##### Box Plot of all Features")
        st.pyplot(box_plot(df.drop("Outcome", axis=1)))

        st.write("##### Feature Box Plot ")
        box_feature = st.selectbox("Select Box Plot Feature", df.columns)
        st.pyplot(box_plot(df[box_feature]))

    with tab3:
        st.write("#### Heat Maps")
        st.write("##### Heat Map showing the correlation between features")
        st.pyplot(heat_map(data=df.corr()))
        st.write(
            "##### Heat Map showing the correlation of features with the Outcome Variable"
        )
        st.pyplot(heat_map(data=df.corr()[["Outcome"]].drop("Outcome")))
