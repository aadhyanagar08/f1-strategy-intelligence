import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, ".")

import streamlit as st  # noqa: E402
from graph import run_pipeline  # noqa: E402

st.set_page_config(page_title="F1 Strategy Intelligence", page_icon="🏎️", layout="centered")

st.title("🏎️ F1 Strategy Intelligence")
st.caption("2026 Canadian GP — Montreal")
st.divider()

query = st.text_input(
    "Strategy question",
    value="What is the optimal tyre strategy for Montreal given the 2026 regulations?",
)

if st.button("Analyse Strategy", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a strategy question.")
    else:
        with st.spinner("Analysing strategy..."):
            result = run_pipeline(query)

        if result is None:
            st.error("Pipeline returned no result. Check your API key and Ollama/OpenAI connectivity.")
        else:
            st.success("Analysis complete")

            # Recommendation
            st.subheader("Recommendation")
            st.info(result.recommendation)

            # Supporting evidence
            st.subheader("Supporting Evidence")
            for point in result.supporting_evidence:
                st.markdown(f"- {point}")

            # Confidence score
            st.subheader("Confidence")
            st.progress(
                result.confidence_score,
                text=f"{result.confidence_score:.0%} confidence",
            )

            # Reasoning
            st.subheader("Reasoning")
            st.markdown(f"*{result.reasoning}*")

            # Alternative strategies
            with st.expander("Alternative Strategies"):
                if result.alternative_strategies:
                    for alt in result.alternative_strategies:
                        st.markdown(f"- {alt}")
                else:
                    st.write("No alternatives generated.")

            # Citations
            with st.expander("Citations", expanded=False):
                if result.citations:
                    for cite in result.citations:
                        st.markdown(f"<small>{cite}</small>", unsafe_allow_html=True)
                else:
                    st.write("No citations available.")
