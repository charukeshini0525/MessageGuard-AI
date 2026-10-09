
import html
import io
import ipaddress
import re
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import requests
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MessageGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fb;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(120deg, #10254a, #2457a6);
        padding: 30px;
        border-radius: 20px;
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 8px;
    }

    .hero p {
        color: #e0eaff;
        font-size: 1.05rem;
    }

    .section-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 14px;
    }

    .small-muted {
        color: #64748b;
        font-size: 0.9rem;
    }

    .risk-high {
        color: #b91c1c;
        font-weight: 700;
    }

    .risk-medium {
        color: #92400e;
        font-weight: 700;
    }

    .risk-low {
        color: #166534;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding-top: 24px;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATASET AND MACHINE-LEARNING MODEL
# ============================================================

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/"
    "228/sms+spam+collection.zip"
)

DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "SMSSpamCollection"


@st.cache_data(show_spinner=False)
def load_dataset():
    """Load the UCI SMS Spam Collection dataset."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            members = [
                name for name in archive.namelist()
                if name.rstrip("/").split("/")[-1]
                == "SMSSpamCollection"
            ]

            if not members:
                raise ValueError(
                    "The SMS dataset file was not found in the download."
                )

            DATA_FILE.write_bytes(archive.read(members[0]))

    df = pd.read_csv(
        DATA_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
        on_bad_lines="skip",
    )

    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["message"] = df["message"].astype(str).str.strip()

    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"] != ""]
    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)

    if df["label"].nunique() != 2:
        raise ValueError("The dataset must contain both ham and spam messages.")

    return df


@st.cache_resource(show_spinner="Training the message detection model...")
def train_model():
    """Train and evaluate the text classification model."""

    df = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=10000,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", MultinomialNB(alpha=0.5)),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=["ham", "spam"],
        target_names=["Likely genuine", "Spam"],
        output_dict=True,
        zero_division=0,
    )

    return model, accuracy, len(df), report


# ============================================================
# LINK EXTRACTION
# ============================================================

URL_PATTERN = re.compile(
    r"""(?i)\b(?:https?://|www\.)[^\s<>"']+"""
    r"""|\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s<>"']*)?"""
)

SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "is.gd",
    "cutt.ly",
    "ow.ly",
    "rb.gy",
}


def extract_urls(message):
    """Extract URLs from text without visiting them."""

    results = []

    for match in URL_PATTERN.findall(message):
        url = match.rstrip(".,!?;:)]}")

        if url and url not in results:
            results.append(url)

    return results


# ============================================================
# HEURISTIC URL RISK ANALYSIS
# ============================================================

def analyse_url(url):
    """
    Identify possible URL warning signs.
    This function does not verify a URL against a live threat database.
    """

    candidate = (
        url
        if url.lower().startswith(("http://", "https://"))
        else "http://" + url
    )

    try:
        parts = urlsplit(candidate)
        host = (parts.hostname or "").lower().rstrip(".")

        if not host:
            return {
                "url": url,
                "level": "NEEDS REVIEW",
                "reasons": ["The domain could not be identified."],
            }

        reasons = []

        # URL credentials and @ symbols can make destinations misleading.
        if parts.username is not None or "@" in parts.netloc:
            reasons.append(
                "Contains an @ symbol or user information in the URL."
            )

        # An IP address is less recognisable than a domain.
        try:
            ipaddress.ip_address(host)
            reasons.append(
                "Uses an IP address instead of a familiar domain name."
            )
        except ValueError:
            pass

        # Punycode domains can be used in misleading lookalike domains.
        if any(label.startswith("xn--") for label in host.split(".")):
            reasons.append(
                "Contains an encoded internationalized domain name."
            )

        if host in SHORTENERS:
            reasons.append(
                "Uses a link-shortening service that hides the final destination."
            )

        if len(host.split(".")) >= 5:
            reasons.append(
                "Contains several subdomains; check the actual registered domain."
            )

        if parts.scheme.lower() == "http":
            reasons.append(
                "Uses HTTP rather than HTTPS. This alone does not prove maliciousness."
            )

        suspicious_terms = (
            "login",
            "verify",
            "password",
            "claim",
            "wallet",
            "otp",
            "urgent",
            "gift",
            "account",
            "signin",
        )

        path_and_query = (parts.path + "?" + parts.query).lower()

        if any(term in path_and_query for term in suspicious_terms):
            reasons.append(
                "The URL path or query contains a sensitive-action keyword."
            )

        if len(reasons) >= 2:
            level = "HIGHER RISK"
        elif len(reasons) == 1:
            level = "CAUTION"
        else:
            level = "NO OBVIOUS WARNING SIGNS"

        return {
            "url": url,
            "level": level,
            "reasons": reasons,
        }

    except (ValueError, UnicodeError):
        return {
            "url": url,
            "level": "NEEDS REVIEW",
            "reasons": ["The URL could not be parsed reliably."],
        }


# ============================================================
# EXPLAINABLE MESSAGE ANALYSIS
# ============================================================

def highlight_words(message, model):
    """Highlight words associated with spam in the trained model."""

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    vocabulary = vectorizer.vocabulary_

    classes = list(classifier.classes_)

    if "spam" not in classes or "ham" not in classes:
        return html.escape(message)

    spam_index = classes.index("spam")
    ham_index = classes.index("ham")

    spam_log_probs = classifier.feature_log_prob_[spam_index]
    ham_log_probs = classifier.feature_log_prob_[ham_index]

    def replace_token(match):
        token = match.group(0).lower()
        escaped = html.escape(match.group(0))

        index = vocabulary.get(token)

        if index is not None:
            difference = (
                spam_log_probs[index] - ham_log_probs[index]
            )

            if difference > 0.5:
                return (
                    '<span style="background-color:#fee2e2;'
                    'color:#991b1b;padding:2px 4px;'
                    'border-radius:4px;">'
                    + escaped
                    + "</span>"
                )

        return escaped

    return re.sub(r"\b[\w'-]+\b", replace_token, message)


# ============================================================
# CALLBACKS
# ============================================================

def clear_message():
    """Clear the input before Streamlit redraws the widget."""
    st.session_state["message_input"] = ""
    st.session_state.pop("last_result", None)


def use_sample():
    """Load the chosen demonstration message."""
    st.session_state["message_input"] = st.session_state["sample_choice"]
    st.session_state.pop("last_result", None)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("🛡️ MessageGuard AI")
    st.caption("Message and link risk screening")

    st.divider()

    st.subheader("Detection modules")

    st.markdown(
        """
        - **Message AI:** TF-IDF + Naive Bayes
        - **URL extraction:** Finds links in text
        - **URL risk checks:** Flags selected warning signs
        - **Explainability:** Highlights some spam-associated words
        """
    )

    st.divider()

    st.subheader("How to use")

    st.markdown(
        """
        1. Paste a message.
        2. Select **Analyse message**.
        3. Review the message classification.
        4. Check each extracted URL.
        5. Verify important claims independently.
        """
    )

    st.divider()

    st.caption(
        "Privacy note: this prototype processes the message in the app. "
        "The initial dataset download requires an internet connection. "
        "No URLs are opened by the URL analyser."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🛡️ MessageGuard AI</h1>
        <p>
            AI-powered message classification and suspicious-link screening.
            Review warning signs before trusting a forwarded message.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Security note: a warning does not prove a link is malicious, "
    "and the absence of warnings does not prove it is safe. "
    "Do not enter passwords, OTPs, or payment details on unverified sites."
)


# ============================================================
# LOAD MODEL
# ============================================================

try:
    with st.spinner("Preparing the machine-learning model..."):
        model, accuracy, dataset_size, report = train_model()

except Exception as error:
    st.error(
        "The model or dataset could not be loaded. "
        "Check your internet connection and try again."
    )

    with st.expander("Technical error details"):
        st.code(str(error))

    st.stop()


# ============================================================
# OVERVIEW METRICS
# ============================================================

metric1, metric2, metric3 = st.columns(3)

metric1.metric("Training dataset", f"{dataset_size:,} messages")
metric2.metric("Test accuracy", f"{accuracy:.1%}")
metric3.metric("Detection modules", "2")


st.divider()


# ============================================================
# MESSAGE INPUT
# ============================================================

st.subheader("🔎 Analyse a message")

st.write(
    "Paste an SMS, WhatsApp forward, email text, or another message "
    "to screen its content and extract URLs."
)

samples = {
    "Choose an example": "",
    "Likely genuine message": (
        "Hey, are we meeting at college tomorrow? Let me know."
    ),
    "Possible spam message": (
        "CONGRATULATIONS! You have won a FREE prize! "
        "Claim your reward now by clicking this link!"
    ),
    "Suspicious forward with a link": (
        "URGENT! Verify your account now at "
        "http://192.0.2.1/verify-account to claim your gift."
    ),
    "Forwarded announcement": (
        "Forwarded message: A major announcement will happen tomorrow. "
        "Please share this information with everyone."
    ),
}

st.selectbox(
    "Try a sample message",
    options=list(samples.keys()),
    key="sample_choice",
)

sample_col, empty_col = st.columns([1, 3])

with sample_col:
    st.button(
        "Use selected example",
        on_click=use_sample,
        use_container_width=True,
    )

message = st.text_area(
    "Message text",
    key="message_input",
    height=180,
    max_chars=10000,
    placeholder=(
        "Paste your message here...\n\n"
        "Example: Your account needs verification. Visit https://example.com"
    ),
)

st.caption(f"Characters: {len(message)} / 10,000")

action_col1, action_col2, empty_col = st.columns([1, 1, 3])

with action_col1:
    analyse_clicked = st.button(
        "🔍 Analyse message",
        type="primary",
        use_container_width=True,
    )

with action_col2:
    st.button(
        "Clear",
        on_click=clear_message,
        use_container_width=True,
    )


# ============================================================
# RUN ANALYSIS
# ============================================================

if analyse_clicked:
    if not message.strip():
        st.warning("Please enter a message before analysing it.")

    else:
        try:
            predicted_class = model.predict([message])[0]
            probabilities = model.predict_proba([message])[0]
            classes = list(model.named_steps["classifier"].classes_)

            probability_map = {
                label: float(probabilities[index])
                for index, label in enumerate(classes)
            }

            max_probability = max(probability_map.values())

            if max_probability < 0.70:
                message_label = "SUSPICIOUS / UNCERTAIN"
            elif predicted_class == "spam":
                message_label = "LIKELY SPAM"
            else:
                message_label = "LIKELY GENUINE"

            urls = extract_urls(message)
            url_results = [analyse_url(url) for url in urls]

            st.session_state["last_result"] = {
                "message": message,
                "label": message_label,
                "predicted_class": predicted_class,
                "confidence": max_probability,
                "probabilities": probability_map,
                "urls": url_results,
            }

        except Exception as error:
            st.error("Analysis failed. Please try again.")

            with st.expander("Technical error details"):
                st.code(str(error))


# ============================================================
# DISPLAY SAVED RESULTS
# ============================================================

result = st.session_state.get("last_result")

if result:
    st.divider()
    st.subheader("📊 Analysis results")

    label = result["label"]

    if label == "LIKELY SPAM":
        st.error("🚨 Message classification: LIKELY SPAM")
    elif label == "SUSPICIOUS / UNCERTAIN":
        st.warning(
            "⚠️ Message classification: SUSPICIOUS / UNCERTAIN"
        )
    else:
        st.success("Message classification: LIKELY GENUINE")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Model confidence",
            f"{result['confidence']:.1%}",
        )

    with result_col2:
        st.metric(
            "Links detected",
            len(result["urls"]),
        )

    st.caption(
        "Model confidence is not the same as the probability that "
        "a real-world message is safe."
    )

    st.markdown("#### Model scores")

    probability_map = result["probabilities"]

    st.progress(
        probability_map.get("ham", 0.0),
        text=(
            "Likely genuine score: "
            f"{probability_map.get('ham', 0.0):.1%}"
        ),
    )

    st.progress(
        probability_map.get("spam", 0.0),
        text=(
            "Spam score: "
            f"{probability_map.get('spam', 0.0):.1%}"
        ),
    )

    st.markdown("#### Message explanation")

    highlighted = highlight_words(result["message"], model)

    st.markdown(
        '<div class="section-card">' + highlighted + "</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "Highlighted words are associated with spam in the trained model. "
        "They are clues, not proof of malicious intent."
    )

    # --------------------------------------------------------
    # LINK RESULTS
    # --------------------------------------------------------

    st.divider()
    st.subheader("🔗 Link security analysis")

    url_results = result["urls"]

    if not url_results:
        st.info(
            "No URLs were detected in this message. "
            "This does not establish that the message itself is trustworthy."
        )

    else:
        for index, item in enumerate(url_results, start=1):
            st.markdown(f"**Link {index}:** `{item['url']}`")

            level = item["level"]

            if level == "HIGHER RISK":
                st.error(f"Risk screening: {level}")

            elif level == "CAUTION":
                st.warning(f"Risk screening: {level}")

            else:
                st.info(f"Risk screening: {level}")

            if item["reasons"]:
                for reason in item["reasons"]:
                    st.markdown(f"- {reason}")
            else:
                st.markdown(
                    "- No warning signs were identified by these basic checks."
                )

            st.divider()

        st.caption(
            "These checks use URL structure and selected keywords only. "
            "They do not query a live malicious-URL database, inspect "
            "website content, or follow redirects. Treat all unverified "
            "links with caution."
        )

    # --------------------------------------------------------
    # SAFETY RECOMMENDATIONS
    # --------------------------------------------------------

    st.subheader("🧭 Recommended next steps")

    recommendations = [
        "Do not open unexpected links or download unknown attachments.",
        "Never share passwords, OTPs, payment details, or recovery codes.",
        "Verify urgent requests through a separate, trusted contact method.",
        "For account issues, navigate to the official website yourself.",
        "Check news and forwarded claims with reliable independent sources.",
    ]

    for recommendation in recommendations:
        st.markdown(f"- {recommendation}")


# ============================================================
# MODEL EVALUATION
# ============================================================

with st.expander("🧪 Model evaluation and technical details"):
    st.write(
        "The model uses TF-IDF text features and Multinomial Naive Bayes. "
        "Evaluation uses a stratified 80/20 train/test split."
    )

    st.metric("Held-out test accuracy", f"{accuracy:.1%}")

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(
        report_df.round(3),
        use_container_width=True,
    )

    st.warning(
        "Accuracy alone does not measure every type of scam or phishing "
        "message. The training dataset is an SMS spam dataset, so "
        "performance may be lower on modern WhatsApp forwards, other "
        "languages, and sophisticated phishing messages."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        MessageGuard AI · Machine Learning + URL Risk Screening<br>
        Educational prototype · Always verify suspicious messages independently
    </div>
    """,
    unsafe_allow_html=True,
)
