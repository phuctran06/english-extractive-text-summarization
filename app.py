import sys
from pathlib import Path

import streamlit as st


#Add src folder to Python path
sys.path.append(
    str(Path(__file__).resolve().parent.parent / "src")
)

from summarizer import summarize_article
from preprocessing import (
    preprocess_article,
    calculate_summary_length
)


#Page configuration
st.set_page_config(
    page_title="AI Text Summarizer",
    page_icon="📝",
    layout="wide"
)


#Custom CSS
st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    .summary-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        line-height: 1.7;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


#Header
st.markdown(
    '<div class="main-title">📝 AI Text Summarizer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Extractive text summarization powered by TextRank + MMR'
    '</div>',
    unsafe_allow_html=True
)


#Article input
st.markdown(
    '<div class="section-title">Article</div>',
    unsafe_allow_html=True
)

article = st.text_area(
    "Paste your English article below",
    height=350,
    placeholder="Paste an English article here...",
    label_visibility="collapsed"
)


#Summary settings
st.markdown(
    '<div class="section-title">Summary Length</div>',
    unsafe_allow_html=True
)

ratio = st.slider(
    "Choose the percentage of the article to keep",
    min_value=5,
    max_value=30,
    value=15,
    step=5,
    format="%d%%"
)


#Calculate estimated length
if article.strip():
    english_stopwords = set(
        __import__("nltk").corpus.stopwords.words("english")
    )

    sentences, processed_sentences = preprocess_article(
        article,
        english_stopwords
    )

    estimated_sentences = calculate_summary_length(
        len(sentences),
        ratio / 100
    )

    st.caption(
        f"Article: {len(sentences)} sentences  •  "
        f"Estimated summary: {estimated_sentences} sentences"
    )


#Generate summary
if st.button(
    "✨ Generate Summary",
    use_container_width=True
):
    if not article.strip():
        st.warning(
            "Please enter an English article first."
        )
    else:
        with st.spinner("Generating summary..."):
            summary = summarize_article(
                article,
                ratio=ratio / 100,
                threshold=0.06,
                lambda_param=0.7
            )

        english_stopwords = set(
            __import__("nltk").corpus.stopwords.words("english")
        )

        sentences, processed_sentences = preprocess_article(
            article,
            english_stopwords
        )

        summary_sentences, _ = preprocess_article(
            summary,
            english_stopwords
        )

        input_sentence_count = len(sentences)
        summary_sentence_count = len(summary_sentences)

        compression_ratio = (
            1 - summary_sentence_count / input_sentence_count
        ) * 100

        #Summary result
        st.markdown(
            '<div class="section-title">Generated Summary</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="summary-box">{summary}</div>',
            unsafe_allow_html=True
        )

        #Statistics
        st.markdown(
            '<div class="section-title">Summary Statistics</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Input Sentences",
                input_sentence_count
            )

        with col2:
            st.metric(
                "Summary Sentences",
                summary_sentence_count
            )

        with col3:
            st.metric(
                "Compression",
                f"{compression_ratio:.1f}%"
            )

        #Method information
        st.markdown(
            '<div class="section-title">Method</div>',
            unsafe_allow_html=True
        )

        st.info(
            "TextRank ranks sentences based on their importance "
            "and similarity. MMR then reduces redundancy while "
            "preserving important information."
        )