import pandas as pd
import streamlit as st

from src.placeholder_models import (
    GenAIClassifierService,
    NewsClassifierService,
    NewsSummarizerService,
    PlaceholderAPIClient,
)

st.set_page_config(
    page_title="News Intelligence Studio",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%); }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] { background: white; padding: 0.8rem 1rem; border-radius: 14px; border: 1px solid #dbeafe; box-shadow: 0 6px 16px rgba(15,23,42,0.05); }
    div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; box-shadow: 0 6px 16px rgba(15,23,42,0.05); }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div { border-radius: 10px; }
    section[data-testid="stSidebar"] { background-color: #f8fafc; }
    </style>
    """,
    unsafe_allow_html=True,
)

classifier = NewsClassifierService()
genai_classifier = GenAIClassifierService()
summarizer = NewsSummarizerService()
api_client = PlaceholderAPIClient()

st.sidebar.title("📰 News Intelligence Studio")
st.sidebar.caption("Capstone UI for article classification and GenAI summarization")
st.sidebar.markdown("### Navigation")

nav = st.sidebar.radio(
    "Choose a view",
    [
        "Overview",
        "ML Classification",
        "GenAI Classification",
        "GenAI Summarization",
        "Business Insights",
        "Deployment",
    ],
    index=0,
)

st.sidebar.markdown("### Project scope")
st.sidebar.write("- Baseline ML model")
st.sidebar.write("- Advanced ML model")
st.sidebar.write("- LLM-based classifier")
st.sidebar.write("- GenAI summary generation")
st.sidebar.write("- Business insights")


def get_model_metrics(model_name: str) -> dict:
    if model_name == "Baseline model":
        return {
            "model": "Baseline model",
            "accuracy": 0.81,
            "precision": 0.78,
            "recall": 0.79,
            "f1_score": 0.79,
            "approach": "TF-IDF + Logistic Regression",
        }
    return {
        "model": "Advanced model",
        "accuracy": 0.88,
        "precision": 0.86,
        "recall": 0.85,
        "f1_score": 0.86,
        "approach": "TF-IDF + SVM / Ensemble",
    }


def build_comparison_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": "Baseline model",
                "Accuracy": 0.81,
                "Precision": 0.78,
                "Recall": 0.79,
                "F1 Score": 0.79,
            },
            {
                "Model": "Advanced model",
                "Accuracy": 0.88,
                "Precision": 0.86,
                "Recall": 0.85,
                "F1 Score": 0.86,
            },
            {
                "Model": "GenAI classifier",
                "Accuracy": 0.90,
                "Precision": 0.89,
                "Recall": 0.87,
                "F1 Score": 0.88,
            },
        ]
    )


if nav == "Overview":
    st.title("AI-powered news categorization and summarization")
    st.write(
        "A professional, presentation-ready interface for Milestone 2 that can later connect to your trained ML and GenAI services."
    )

    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Classification", "2 model paths", "Baseline + Advanced")
        with col2:
            st.metric("GenAI", "LLM-based classifier", "Ready")
        with col3:
            st.metric("Summaries", "Headline-style", "60 words")
        with col4:
            st.metric("Insights", "Business view", "Ready")

    st.subheader("Project overview")
    st.write(
        "This app is structured to show how Meridian Media Group can classify news articles, compare conventional and GenAI-based classifiers, generate concise summaries, and frame business insights for stakeholders."
    )

    status = api_client.get_status()
    with st.expander("System readiness", expanded=True):
        st.write(f"API gateway: {status['api_status']}")
        st.write(f"Model pipeline: {status['pipeline_status']}")
        st.write(f"Summary service: {status['summary_status']}")

elif nav == "ML Classification":
    st.header("Model building and evaluation")
    st.write("Review the baseline and advanced NLP models and compare their performance.")

    selected_model = st.selectbox(
        "Model selection",
        ["Baseline model", "Advanced model"],
        index=0,
    )

    model_metrics = get_model_metrics(selected_model)
    metric_df = pd.DataFrame(
        [
            {"Metric": "Accuracy", "Value": model_metrics["accuracy"]},
            {"Metric": "Precision", "Value": model_metrics["precision"]},
            {"Metric": "Recall", "Value": model_metrics["recall"]},
            {"Metric": "F1 Score", "Value": model_metrics["f1_score"]},
        ]
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Selected model", model_metrics["model"])
        st.metric("Approach", model_metrics["approach"])
    with col_b:
        st.dataframe(metric_df, use_container_width=True, hide_index=True)

    st.subheader("Comparative evaluation")
    st.dataframe(build_comparison_table(), use_container_width=True, hide_index=True)

    st.subheader("Try the selected classifier")
    with st.form("classification_form"):
        headline = st.text_input(
            "Headline",
            value="OpenAI launches a new enterprise automation suite for media teams",
        )
        description = st.text_area(
            "Description",
            value="The platform promises faster content generation, enhanced workflow automation, and stronger collaboration across editorial teams.",
            height=90,
        )
        content = st.text_area(
            "Content",
            value=(
                "The rollout targets organizations that need dependable content production at scale. "
                "It is expected to streamline publishing workflows and reduce repetitive tasks across digital channels."
            ),
            height=180,
        )
        submitted = st.form_submit_button("Classify article")

    if submitted:
        result = classifier.predict_category(headline, description, content, model_name=selected_model)
        st.success("Classification completed")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted category", result["category"])
        with col2:
            st.metric("Confidence", f"{result['confidence']:.0%}")
        with col3:
            st.metric("Model used", result["model_name"])
        st.write(result["explanation"])

elif nav == "GenAI Classification":
    st.header("GenAI-based classifier")
    st.write("Compare the conventional NLP models with a placeholder LLM-powered classifier.")

    st.subheader("Performance snapshot")
    st.dataframe(build_comparison_table(), use_container_width=True, hide_index=True)

    st.subheader("Try the GenAI classifier")
    with st.form("genai_classifier_form"):
        headline = st.text_input(
            "Headline",
            value="Global media groups invest in AI-led content automation",
            key="genai_headline",
        )
        description = st.text_area(
            "Description",
            value="The move highlights a growing trend toward AI-assisted storytelling and newsroom productivity.",
            height=90,
            key="genai_description",
        )
        content = st.text_area(
            "Content",
            value=(
                "Editorial teams are looking to modern tools that can improve speed, consistency, and publishing workflows. "
                "The technology is expected to support both workflow automation and content personalization."
            ),
            height=180,
            key="genai_content",
        )
        submitted = st.form_submit_button("Run GenAI classifier")

    if submitted:
        conventional_result = classifier.predict_category(headline, description, content, model_name="Baseline model")
        advanced_result = classifier.predict_category(headline, description, content, model_name="Advanced model")
        genai_result = genai_classifier.predict_category(headline, description, content)

        st.success("GenAI classifier completed")

        comparison_df = pd.DataFrame(
            [
                {
                    "Approach": "Baseline model",
                    "Category": conventional_result["category"],
                    "Confidence": conventional_result["confidence"],
                },
                {
                    "Approach": "Advanced model",
                    "Category": advanced_result["category"],
                    "Confidence": advanced_result["confidence"],
                },
                {
                    "Approach": "GenAI classifier",
                    "Category": genai_result["category"],
                    "Confidence": genai_result["confidence"],
                },
            ]
        )
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Baseline")
            st.metric("Category", conventional_result["category"])
            st.write(conventional_result["explanation"])
        with col2:
            st.subheader("Advanced")
            st.metric("Category", advanced_result["category"])
            st.write(advanced_result["explanation"])
        with col3:
            st.subheader("GenAI")
            st.metric("Category", genai_result["category"])
            st.write(genai_result["explanation"])

elif nav == "GenAI Summarization":
    st.header("GenAI summarization")
    st.write("Generate a concise summary that can later be powered by your GenAI workflow.")

    with st.form("summary_form"):
        headline = st.text_input(
            "Headline",
            value="Global markets react to new AI-driven productivity tools",
            key="summary_headline",
        )
        description = st.text_area(
            "Description",
            value="Investors are watching the space as companies race to integrate AI into business operations.",
            height=90,
            key="summary_description",
        )
        content = st.text_area(
            "Content",
            value=(
                "Analysts say the recent wave of AI announcements is reshaping how firms plan, communicate, and automate work. "
                "The market response reflects stronger interest in tools that increase productivity and reduce manual effort."
            ),
            height=180,
            key="summary_content",
        )
        submitted = st.form_submit_button("Generate summary")

    if submitted:
        summary = summarizer.summarize(headline, description, content, max_words=60)
        st.success("Summary prepared")
        st.subheader("Generated summary")
        st.write(summary)

        with st.expander("Evaluation placeholders", expanded=True):
            st.metric("Summary length", "~60 words")
            st.metric("Relevance score", "High")
            st.metric("Clarity score", "High")

elif nav == "Business Insights":
    st.header("Final summary and business insights")
    st.write("Capture the executive summary, insights, and recommendations for the final presentation.")

    executive_summary = st.text_area(
        "Executive summary",
        value="The proposed solution helps Meridian Media Group classify news articles with high accuracy, reduce manual tagging effort, and improve reader experience by enabling better content organization and discovery.",
        height=120,
    )
    actionable_insights = st.text_area(
        "Actionable insights",
        value="- Improve content tagging consistency across departments.\n- Use automated summaries to accelerate newsroom workflows.\n- Personalize article recommendations for better engagement.",
        height=140,
    )
    recommendation = st.text_area(
        "Recommendation",
        value="Deploy the best-performing model in a Streamlit application and use GenAI summaries as an editorial assistive feature for faster publishing.",
        height=120,
    )

    if st.button("Save summary view"):
        st.success("Business insights prepared for presentation")

    st.subheader("Preview")
    st.write(executive_summary)
    st.subheader("Recommended actions")
    st.write(actionable_insights)
    st.subheader("Final recommendation")
    st.write(recommendation)

elif nav == "Deployment":
    st.header("Deployment readiness")
    st.write("A compact view suitable for stakeholder presentations.")

    st.checkbox("Model integration points connected", value=True)
    st.checkbox("GenAI summarization service connected", value=True)
    st.checkbox("UI styling finalized for presentation", value=True)
    st.progress(100)

    st.subheader("Replace these placeholders later")
    st.code(
        "classifier.predict_category(...)\nsummarizer.summarize(...)\napi_client.get_status()",
        language="python",
    )

st.markdown("---")
st.caption("Capstone Project By Team-2, Cohort VII")
st.caption("Participants: Shouvik Guha, Inbarajan P, Promod Reddy Serikar, Manoj Thomas, Mayank Tyagi")
