"""
CISB5123 Text Analytics Project Part 2
Interactive Streamlit Dashboard

How to run:
1. Install required libraries:
   pip install streamlit pandas matplotlib scikit-learn nltk wordcloud

2. Run the dashboard:
   streamlit run Part2_GroupLeaderID_dashboard_perfect.py

3. Upload your processed review CSV file in the app.
"""

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Optional dependencies. The dashboard still opens even if one optional package is missing.
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Jabal Omar Review Text Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    """Load uploaded CSV safely."""
    return pd.read_csv(file)


def clean_text(text: object) -> str:
    """Clean review text for topic modeling and word frequency analysis."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_column(df: pd.DataFrame, candidates: Iterable[str], fallback: Optional[str] = None) -> Optional[str]:
    """Return the first matching column from a candidate list, ignoring case."""
    lower_to_original = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return fallback


def vader_label(compound_score: float) -> str:
    """Convert VADER compound score into a sentiment label."""
    if compound_score >= 0.05:
        return "Positive"
    if compound_score <= -0.05:
        return "Negative"
    return "Neutral"


@st.cache_data(show_spinner=False)
def create_vader_sentiment(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Create VADER compound scores and sentiment labels from the original review text."""
    analyzer = SentimentIntensityAnalyzer()
    result = df.copy()
    result["compound_score"] = result[text_col].fillna("").astype(str).apply(
        lambda review: analyzer.polarity_scores(review)["compound"]
    )
    result["sentiment"] = result["compound_score"].apply(vader_label)
    return result


@st.cache_data(show_spinner=False)
def create_lda_topics(
    df: pd.DataFrame,
    text_col: str,
    n_topics: int,
    max_features: int,
    min_df: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[float]]:
    """Create LDA topics using scikit-learn and return data, keywords, and perplexity."""
    result = df.copy()
    result["clean_text_dashboard"] = result[text_col].fillna("").astype(str).apply(clean_text)

    non_empty_text = result["clean_text_dashboard"].replace("", pd.NA).dropna()
    if len(non_empty_text) < 2:
        result["topic"] = "Not enough text"
        topic_keyword_df = pd.DataFrame({"Topic": ["Not enough text"], "Top Keywords": ["Not enough usable text"]})
        return result, topic_keyword_df, None

    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=max_features,
        min_df=min_df,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )

    try:
        doc_term_matrix = vectorizer.fit_transform(result["clean_text_dashboard"])
    except ValueError:
        # If min_df is too strict for a small dataset, try again with min_df=1.
        vectorizer = CountVectorizer(
            stop_words="english",
            max_features=max_features,
            min_df=1,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        )
        doc_term_matrix = vectorizer.fit_transform(result["clean_text_dashboard"])

    feature_names = vectorizer.get_feature_names_out()
    safe_topics = max(2, min(n_topics, doc_term_matrix.shape[0], len(feature_names)))

    lda = LatentDirichletAllocation(
        n_components=safe_topics,
        random_state=42,
        learning_method="batch",
        max_iter=20,
    )
    topic_matrix = lda.fit_transform(doc_term_matrix)
    result["topic"] = topic_matrix.argmax(axis=1)
    result["topic_confidence"] = topic_matrix.max(axis=1).round(3)

    topic_rows = []
    for topic_idx, topic_weights in enumerate(lda.components_):
        top_indices = topic_weights.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topic_rows.append(
            {
                "Topic": topic_idx,
                "Suggested Topic Name": f"Topic {topic_idx}: {', '.join(top_words[:3]).title()}",
                "Top Keywords": ", ".join(top_words),
            }
        )

    topic_keyword_df = pd.DataFrame(topic_rows)
    perplexity = float(lda.perplexity(doc_term_matrix))
    return result, topic_keyword_df, perplexity


def make_percentage_table(series: pd.Series, label_name: str) -> pd.DataFrame:
    """Create count and percentage table from a series."""
    counts = series.value_counts(dropna=False).reset_index()
    counts.columns = [label_name, "Count"]
    counts["Percentage"] = (counts["Count"] / counts["Count"].sum() * 100).round(2)
    return counts


def create_word_frequency(text_values: pd.Series, extra_stopwords: Optional[List[str]] = None, top_n: int = 20) -> pd.DataFrame:
    """Return most frequent meaningful words."""
    built_in_stopwords = {
        "the", "and", "for", "with", "this", "that", "was", "were", "are", "you", "your", "our", "not",
        "but", "all", "can", "had", "have", "has", "from", "they", "their", "very", "there", "here", "will",
        "hotel", "review", "stay", "stayed",
    }
    if extra_stopwords:
        built_in_stopwords.update([word.strip().lower() for word in extra_stopwords if word.strip()])

    words = []
    for text in text_values.dropna().astype(str):
        words.extend([word for word in clean_text(text).split() if word not in built_in_stopwords and len(word) > 2])

    most_common = Counter(words).most_common(top_n)
    return pd.DataFrame(most_common, columns=["Word", "Frequency"])


def fig_bar(data: pd.Series, title: str, xlabel: str, ylabel: str):
    """Create a simple bar chart for Streamlit."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def fig_horizontal_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    """Create a horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df[y_col], df[x_col])
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def fig_stacked_bar(crosstab: pd.DataFrame, title: str, xlabel: str, ylabel: str):
    """Create a stacked bar chart."""
    fig, ax = plt.subplots(figsize=(9, 5))
    crosstab.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def generate_dynamic_insights(filtered_df: pd.DataFrame, text_col: str) -> List[str]:
    """Generate simple data-driven insights for the dashboard."""
    insights = []
    total = len(filtered_df)
    if total == 0:
        return ["No records match the selected filters."]

    sentiment_percentages = filtered_df["sentiment"].value_counts(normalize=True).mul(100).round(1)
    dominant_sentiment = sentiment_percentages.index[0]
    insights.append(f"The dominant sentiment is {dominant_sentiment}, representing {sentiment_percentages.iloc[0]}% of the filtered reviews.")

    if "topic" in filtered_df.columns and filtered_df["topic"].nunique() > 0:
        dominant_topic = filtered_df["topic"].value_counts().index[0]
        topic_count = int(filtered_df["topic"].value_counts().iloc[0])
        insights.append(f"The most frequent topic is Topic {dominant_topic}, appearing in {topic_count} filtered reviews.")

    if "Negative" in filtered_df["sentiment"].values:
        negative_df = filtered_df[filtered_df["sentiment"] == "Negative"]
        if len(negative_df) > 0 and "topic" in negative_df.columns:
            main_negative_topic = negative_df["topic"].value_counts().index[0]
            insights.append(f"Negative feedback is most concentrated around Topic {main_negative_topic}, so this topic should be checked carefully in the sample reviews.")

    common_words = create_word_frequency(filtered_df[text_col], top_n=5)
    if not common_words.empty:
        insights.append("The most repeated keywords include: " + ", ".join(common_words["Word"].tolist()) + ".")

    return insights


# -----------------------------------------------------------------------------
# Dashboard header
# -----------------------------------------------------------------------------
st.title("📊 Jabal Omar Review Text Analytics Dashboard")
st.caption("CISB5123 Project Part 2 — VADER Sentiment Analysis, LDA Topic Modeling, Visualization, and Insights")

with st.expander("Project objective", expanded=False):
    st.write(
        "This dashboard helps analyze customer review text by identifying overall sentiment, main review topics, "
        "topic-specific sentiment patterns, frequent keywords, and example reviews. It is designed to support the "
        "Project Part 2 requirements for modeling, evaluation, visualization, insights, and dashboard presentation."
    )

uploaded_file = st.file_uploader("Upload your processed review CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload your CSV file to begin. The dashboard will automatically detect review, sentiment, topic, rating, and date columns where possible.")
    st.stop()

# -----------------------------------------------------------------------------
# Data loading and sidebar setup
# -----------------------------------------------------------------------------
df_original = load_csv(uploaded_file)
df_original.columns = [str(col).strip() for col in df_original.columns]

if df_original.empty:
    st.error("The uploaded file is empty. Please upload a CSV file with review text.")
    st.stop()

TEXT_COLUMN_CANDIDATES = [
    "review_content", "Review Content", "review_text", "Review Text", "review", "Review",
    "text", "Text", "content", "Content", "comment", "Comment", "summary", "Summary",
    "clean_text", "cleaned_text", "processed_text",
]
DATE_COLUMN_CANDIDATES = ["review_date", "Review Date", "date", "Date", "created_at", "Created At"]
RATING_COLUMN_CANDIDATES = ["rating", "Rating", "stars", "Stars", "score", "Score"]
SENTIMENT_COLUMN_CANDIDATES = ["sentiment", "Sentiment", "vader_sentiment", "VADER_Sentiment"]
TOPIC_COLUMN_CANDIDATES = ["topic", "Topic", "lda_topic", "dominant_topic", "Dominant_Topic"]
REVIEWER_COLUMN_CANDIDATES = ["reviewer_name", "Reviewer Name", "name", "Name", "user", "User"]

text_default = detect_column(df_original, TEXT_COLUMN_CANDIDATES, fallback=df_original.columns[0])
date_default = detect_column(df_original, DATE_COLUMN_CANDIDATES)
rating_default = detect_column(df_original, RATING_COLUMN_CANDIDATES)
sentiment_existing = detect_column(df_original, SENTIMENT_COLUMN_CANDIDATES)
topic_existing = detect_column(df_original, TOPIC_COLUMN_CANDIDATES)
reviewer_default = detect_column(df_original, REVIEWER_COLUMN_CANDIDATES)

st.sidebar.header("⚙️ Dashboard Controls")
text_col = st.sidebar.selectbox(
    "Text / review column",
    options=df_original.columns,
    index=list(df_original.columns).index(text_default),
)

date_col = st.sidebar.selectbox(
    "Date column, if available",
    options=["None"] + list(df_original.columns),
    index=(list(df_original.columns).index(date_default) + 1) if date_default else 0,
)

rating_col = st.sidebar.selectbox(
    "Rating column, if available",
    options=["None"] + list(df_original.columns),
    index=(list(df_original.columns).index(rating_default) + 1) if rating_default else 0,
)

reviewer_col = st.sidebar.selectbox(
    "Reviewer column, if available",
    options=["None"] + list(df_original.columns),
    index=(list(df_original.columns).index(reviewer_default) + 1) if reviewer_default else 0,
)

st.sidebar.markdown("---")

if sentiment_existing:
    sentiment_options = ["Use existing column", "Create with VADER"]
else:
    sentiment_options = ["Create with VADER"]

sentiment_mode = st.sidebar.radio(
    "Sentiment source",
    options=sentiment_options,
    index=0,
)

if topic_existing:
    topic_options = ["Use existing column", "Create with LDA"]
else:
    topic_options = ["Create with LDA"]

topic_mode = st.sidebar.radio("Topic source", options=topic_options, index=0)

n_topics = st.sidebar.slider("Number of LDA topics", min_value=2, max_value=8, value=4)
max_features = st.sidebar.slider("Maximum topic-model words", min_value=300, max_value=3000, value=1000, step=100)
recommended_min_df = 1 if len(df_original) < 20 else 2
min_df = st.sidebar.slider("Minimum word document frequency", min_value=1, max_value=5, value=recommended_min_df)

extra_stopwords_input = st.sidebar.text_input(
    "Extra stopwords, comma-separated",
    value="jabal, omar, makkah, hotel",
)
extra_stopwords = [word.strip() for word in extra_stopwords_input.split(",") if word.strip()]

# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------
df = df_original.copy()

if sentiment_mode == "Use existing column" and sentiment_existing:
    df["sentiment"] = df[sentiment_existing].astype(str).str.title()
else:
    if not VADER_AVAILABLE:
        st.error("VADER is not available. Please install NLTK: pip install nltk")
        st.stop()
    df = create_vader_sentiment(df, text_col)

lda_perplexity = None
topic_keyword_df = pd.DataFrame()

if topic_mode == "Use existing column" and topic_existing:
    df["topic"] = df[topic_existing]
else:
    with st.spinner("Creating LDA topics..."):
        df, topic_keyword_df, lda_perplexity = create_lda_topics(df, text_col, n_topics, max_features, min_df)

# Date parsing, if available.
if date_col != "None":
    df["dashboard_date"] = pd.to_datetime(df[date_col], errors="coerce")

# Rating parsing, if available.
if rating_col != "None":
    df["dashboard_rating"] = pd.to_numeric(df[rating_col], errors="coerce")

# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filters")

keyword_filter = st.sidebar.text_input("Search inside reviews")

all_sentiments = sorted(df["sentiment"].dropna().astype(str).unique())
selected_sentiments = st.sidebar.multiselect(
    "Filter by sentiment",
    options=all_sentiments,
    default=all_sentiments,
)

all_topics = sorted(df["topic"].dropna().unique(), key=lambda value: str(value))
selected_topics = st.sidebar.multiselect(
    "Filter by topic",
    options=all_topics,
    default=all_topics,
)

filtered_df = df.copy()
filtered_df = filtered_df[
    filtered_df["sentiment"].astype(str).isin([str(item) for item in selected_sentiments])
    & filtered_df["topic"].astype(str).isin([str(item) for item in selected_topics])
]

if keyword_filter.strip():
    filtered_df = filtered_df[
        filtered_df[text_col].fillna("").astype(str).str.contains(keyword_filter.strip(), case=False, na=False)
    ]

if date_col != "None" and df["dashboard_date"].notna().any():
    min_date = df["dashboard_date"].min().date()
    max_date = df["dashboard_date"].max().date()
    selected_date_range = st.sidebar.date_input("Filter by date range", value=(min_date, max_date))
    if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
        start_date, end_date = selected_date_range
        filtered_df = filtered_df[
            (filtered_df["dashboard_date"].dt.date >= start_date)
            & (filtered_df["dashboard_date"].dt.date <= end_date)
        ]

if rating_col != "None" and df["dashboard_rating"].notna().any():
    min_rating = float(df["dashboard_rating"].min())
    max_rating = float(df["dashboard_rating"].max())
    selected_rating_range = st.sidebar.slider(
        "Filter by rating",
        min_value=min_rating,
        max_value=max_rating,
        value=(min_rating, max_rating),
    )
    filtered_df = filtered_df[
        filtered_df["dashboard_rating"].between(selected_rating_range[0], selected_rating_range[1], inclusive="both")
    ]

# -----------------------------------------------------------------------------
# Overview metrics
# -----------------------------------------------------------------------------
st.markdown("---")
metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
metric_col1.metric("Total Reviews", f"{len(df):,}")
metric_col2.metric("Filtered Reviews", f"{len(filtered_df):,}")
metric_col3.metric("Sentiment Classes", df["sentiment"].nunique())
metric_col4.metric("Topics", df["topic"].nunique())

if "compound_score" in df.columns and len(filtered_df) > 0:
    metric_col5.metric("Avg. VADER Score", round(float(filtered_df["compound_score"].mean()), 3))
elif "dashboard_rating" in df.columns and len(filtered_df) > 0:
    metric_col5.metric("Avg. Rating", round(float(filtered_df["dashboard_rating"].mean()), 2))
else:
    metric_col5.metric("Text Column", text_col)

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
overview_tab, sentiment_tab, topic_tab, review_tab, method_tab = st.tabs(
    ["Overview", "Sentiment Analysis", "Topic Modeling", "Review Explorer", "Method & Findings"]
)

with overview_tab:
    st.subheader("Dataset Preview")
    st.dataframe(df_original.head(10), use_container_width=True)

    st.subheader("Automatic Insights")
    for insight in generate_dynamic_insights(filtered_df, text_col):
        st.write(f"• {insight}")

    if len(filtered_df) == 0:
        st.warning("No records match the current filters. Adjust the sidebar filters to show visualizations.")
    else:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            sentiment_counts = filtered_df["sentiment"].value_counts()
            st.pyplot(fig_bar(sentiment_counts, "Sentiment Distribution", "Sentiment", "Number of Reviews"))
        with chart_col2:
            topic_counts = filtered_df["topic"].value_counts().sort_index()
            st.pyplot(fig_bar(topic_counts, "Topic Distribution", "Topic", "Number of Reviews"))

        st.subheader("Sentiment by Topic")
        topic_sentiment = pd.crosstab(filtered_df["topic"], filtered_df["sentiment"])
        st.pyplot(fig_stacked_bar(topic_sentiment, "Sentiment by Topic", "Topic", "Number of Reviews"))

with sentiment_tab:
    st.subheader("Sentiment Summary")
    if len(filtered_df) == 0:
        st.warning("No sentiment data to display for the selected filters.")
    else:
        sentiment_summary = make_percentage_table(filtered_df["sentiment"], "Sentiment")
        st.dataframe(sentiment_summary, use_container_width=True)

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.pyplot(fig_bar(filtered_df["sentiment"].value_counts(), "Sentiment Count", "Sentiment", "Count"))
        with chart_col2:
            if "compound_score" in filtered_df.columns:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.hist(filtered_df["compound_score"].dropna(), bins=20)
                ax.set_title("VADER Compound Score Distribution")
                ax.set_xlabel("Compound Score")
                ax.set_ylabel("Number of Reviews")
                fig.tight_layout()
                st.pyplot(fig)
            elif "dashboard_rating" in filtered_df.columns:
                rating_counts = filtered_df["dashboard_rating"].value_counts().sort_index()
                st.pyplot(fig_bar(rating_counts, "Rating Distribution", "Rating", "Count"))
            else:
                st.info("No compound score or rating column is available for a second sentiment chart.")

        if "dashboard_rating" in filtered_df.columns:
            st.subheader("Average Rating by Sentiment")
            avg_rating = filtered_df.groupby("sentiment", as_index=False)["dashboard_rating"].mean().round(2)
            avg_rating = avg_rating.rename(columns={"dashboard_rating": "Average Rating"})
            st.dataframe(avg_rating, use_container_width=True)

with topic_tab:
    st.subheader("Topic Model Summary")

    if lda_perplexity is not None:
        st.write(f"LDA model perplexity: **{lda_perplexity:.2f}**. Lower perplexity generally indicates a better statistical fit for the topic model.")

    if not topic_keyword_df.empty:
        st.write("Top words help interpret what each LDA topic represents.")
        st.dataframe(topic_keyword_df, use_container_width=True)

    if len(filtered_df) == 0:
        st.warning("No topic data to display for the selected filters.")
    else:
        topic_summary = make_percentage_table(filtered_df["topic"], "Topic")
        st.dataframe(topic_summary, use_container_width=True)
        st.pyplot(fig_bar(filtered_df["topic"].value_counts().sort_index(), "Topic Distribution", "Topic", "Count"))

        st.subheader("Top Words in Filtered Reviews")
        word_freq = create_word_frequency(filtered_df[text_col], extra_stopwords=extra_stopwords, top_n=20)
        if word_freq.empty:
            st.info("Not enough words are available after filtering and cleaning.")
        else:
            st.pyplot(fig_horizontal_bar(word_freq, "Frequency", "Word", "Most Frequent Words"))
            st.dataframe(word_freq, use_container_width=True)

        st.subheader("Word Cloud")
        if WORDCLOUD_AVAILABLE:
            cloud_text = " ".join(filtered_df[text_col].dropna().astype(str).apply(clean_text))
            if cloud_text.strip():
                wordcloud = WordCloud(
                    width=1000,
                    height=450,
                    background_color="white",
                    stopwords=set(extra_stopwords),
                    collocations=False,
                ).generate(cloud_text)
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                fig.tight_layout()
                st.pyplot(fig)
            else:
                st.info("No usable text is available for the word cloud.")
        else:
            st.warning("WordCloud is not installed. Install it with: pip install wordcloud")

with review_tab:
    st.subheader("Review Explorer")
    st.write("Use this section to show real review examples behind each sentiment and topic.")

    if len(filtered_df) == 0:
        st.warning("No reviews match the selected filters.")
    else:
        display_columns = []
        if reviewer_col != "None":
            display_columns.append(reviewer_col)
        if date_col != "None":
            display_columns.append(date_col)
        if rating_col != "None":
            display_columns.append(rating_col)
        display_columns.extend([text_col, "sentiment", "topic"])
        if "compound_score" in filtered_df.columns:
            display_columns.append("compound_score")
        if "topic_confidence" in filtered_df.columns:
            display_columns.append("topic_confidence")

        display_columns = [col for col in display_columns if col in filtered_df.columns]
        st.dataframe(filtered_df[display_columns], use_container_width=True)

        st.download_button(
            label="Download filtered dashboard data as CSV",
            data=filtered_df[display_columns].to_csv(index=False).encode("utf-8"),
            file_name="filtered_text_analytics_dashboard_data.csv",
            mime="text/csv",
        )

        st.subheader("Sample Reviews by Topic")
        for topic_value in sorted(filtered_df["topic"].dropna().unique(), key=lambda value: str(value)):
            topic_sample = filtered_df[filtered_df["topic"] == topic_value].head(3)

            with st.expander(f"Topic {topic_value} — {len(filtered_df[filtered_df['topic'] == topic_value])} review(s)"):
                for _, row in topic_sample.iterrows():
                    if reviewer_col != "None" and reviewer_col in row.index:
                        st.write(f"**Reviewer:** {row[reviewer_col]}")

                    if date_col != "None" and date_col in row.index:
                        st.write(f"**Date:** {row[date_col]}")

                    if rating_col != "None" and rating_col in row.index:
                        st.write(f"**Rating:** {row[rating_col]}")

                    st.write(f"**Sentiment:** {row['sentiment']}")

                    if "compound_score" in row.index:
                        st.write(f"**VADER score:** {row['compound_score']}")

                    if "topic_confidence" in row.index:
                        st.write(f"**Topic confidence:** {row['topic_confidence']}")

                    st.write(f"**Review:** {str(row[text_col])}")
                    st.markdown("---")

with method_tab:
    st.subheader("Methods Used")
    st.markdown(
        """
        **1. Sentiment Analysis — VADER**  
        VADER was used to classify reviews into Positive, Neutral, and Negative sentiment. It is suitable for short customer-review text because it uses a rule-based sentiment lexicon and produces a compound score.

        **2. Topic Modeling — LDA**  
        Latent Dirichlet Allocation was used to discover recurring themes in the review text. The dashboard shows topic keywords, topic distribution, and sentiment by topic.

        **3. Visualization and Dashboard**  
        The dashboard includes summary metrics, sentiment distribution, topic distribution, sentiment-by-topic comparison, word frequency, word cloud, and review examples.
        """
    )

    st.subheader("Key Findings Template")
    for insight in generate_dynamic_insights(filtered_df, text_col):
        st.write(f"• {insight}")

    st.subheader("Challenges and Improvements")
    st.markdown(
        """
        **Challenges**
        - VADER may misclassify reviews with sarcasm, mixed opinions, or context-specific words.
        - LDA topic quality depends strongly on preprocessing, stopword removal, and the selected number of topics.
        - Some reviews may discuss multiple issues, but LDA assigns one dominant topic for dashboard simplicity.

        **Future Improvements**
        - Compare several topic numbers using coherence score in the notebook.
        - Use transformer-based sentiment models such as BERT or RoBERTa for deeper context understanding.
        - Use BERTopic for more interpretable topic clusters.
        - Add more review sources to improve coverage and reliability.
        """
    )

    st.subheader("Dashboard Export")
    export_columns = [text_col, "sentiment", "topic"]
    if "compound_score" in df.columns:
        export_columns.append("compound_score")
    if "topic_confidence" in df.columns:
        export_columns.append("topic_confidence")
    if date_col != "None" and date_col in df.columns:
        export_columns.insert(0, date_col)
    if rating_col != "None" and rating_col in df.columns:
        export_columns.insert(0, rating_col)
    if reviewer_col != "None" and reviewer_col in df.columns:
        export_columns.insert(0, reviewer_col)

    export_columns = [col for col in export_columns if col in df.columns]
    st.download_button(
        label="Download full modeled data as CSV",
        data=df[export_columns].to_csv(index=False).encode("utf-8"),
        file_name="Part2_dashboard_modeled_data.csv",
        mime="text/csv",
    )
