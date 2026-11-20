
import os

import requests
import streamlit as st


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Multimodal Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_API_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)

ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]
REQUEST_TIMEOUT = 180


# ============================================================
# API Helpers
# ============================================================

def get_api_url():
    return st.session_state.api_url.rstrip("/")


def send_request(method, endpoint, **kwargs):
    """Send an HTTP request to the FastAPI backend."""

    url = f"{get_api_url()}{endpoint}"

    try:
        response = requests.request(
            method=method,
            url=url,
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.Timeout:
        st.error(
            "The request timed out. The model may need more time "
            "to process your query."
        )

    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to the API. "
            "Make sure the FastAPI server is running."
        )

    except requests.exceptions.HTTPError:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text

        st.error(f"API Error ({response.status_code}): {detail}")

    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {exc}")

    return None


def parse_excluded_isbns(value):
    """Convert comma-separated ISBNs into a list."""

    return [
        isbn.strip()
        for isbn in value.split(",")
        if isbn.strip()
    ]


def build_form_data(
    top_k,
    min_score,
    max_per_author,
    exclude_isbns,
):
    """Build multipart form fields for image endpoints."""

    data = [
        ("top_k", str(top_k)),
        ("min_score", str(min_score)),
    ]

    if max_per_author is not None:
        data.append(("max_per_author", str(max_per_author)))

    for isbn in exclude_isbns:
        data.append(("exclude_isbns", isbn))

    return data


# ============================================================
# Recommendation Display
# ============================================================

def display_recommendations(result):
    """Display recommendation results returned by the API."""

    if not result:
        return

    recommendations = result.get("recommendations", [])

    st.divider()

    st.subheader("📖 Recommendation Results")

    col1, col2 = st.columns(2)

    col1.metric(
        "Recommendations Found",
        result.get("total", len(recommendations)),
    )

    col2.metric(
        "Search Type",
        result.get("query_type", "Unknown").title(),
    )

    if not recommendations:
        st.warning(
            "No books found. Try changing your query "
            "or lowering the minimum similarity score."
        )
        return

    st.caption(
        "Results are ordered by cosine similarity. "
        "Scores are not probabilities."
    )

    for book in recommendations:
        with st.container(border=True):

            col1, col2 = st.columns([5, 1])

            with col1:
                st.markdown(
                    f"### {book.get('rank', '')}. "
                    f"{book.get('title') or 'Unknown Title'}"
                )

                st.write(
                    f"**Author:** {book.get('author') or 'Unknown'}"
                )

                st.write(
                    f"**Publisher:** "
                    f"{book.get('publisher') or 'Unknown'}"
                )

                st.write(
                    f"**Year:** {book.get('year') or 'Unknown'}"
                )

                st.caption(
                    f"ISBN: {book.get('isbn') or 'N/A'}"
                )

                if book.get("image_available"):
                    st.caption("🖼️ Book cover available in the dataset")
                else:
                    st.caption("Cover image not available")

            with col2:
                st.metric(
                    "Similarity",
                    f"{book.get('score', 0):.4f}",
                )


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("⚙️ Settings")

if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL

st.sidebar.text_input(
    "FastAPI Base URL",
    key="api_url",
)

if st.sidebar.button("Check API Connection", use_container_width=True):
    health = send_request("GET", "/health")

    if health and health.get("status") == "ok":
        st.sidebar.success("API is running!")
    elif health:
        st.sidebar.warning("Unexpected API health response.")

st.sidebar.divider()

st.sidebar.subheader("Recommendation Settings")

top_k = st.sidebar.slider(
    "Number of recommendations",
    min_value=1,
    max_value=30,
    value=10,
)

min_score = st.sidebar.slider(
    "Minimum similarity score",
    min_value=-1.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
)

max_per_author = st.sidebar.number_input(
    "Maximum books per author (0 = unlimited)",
    min_value=0,
    max_value=30,
    value=0,
)

exclude_input = st.sidebar.text_input(
    "Exclude ISBNs (comma-separated)",
    placeholder="9780140328721, 9780061120084",
)

exclude_isbns = parse_excluded_isbns(exclude_input)

max_per_author_value = (
    max_per_author if max_per_author > 0 else None
)


# ============================================================
# Main Header
# ============================================================

st.title("📚 Multimodal Book Recommender")

st.markdown(
    """
    Discover books using **text descriptions**, **book cover images**,
    or a combination of both.

    Powered by CLIP embeddings and Qdrant vector search.
    """
)


# ============================================================
# Search Tabs
# ============================================================

text_tab, image_tab, multimodal_tab = st.tabs(
    [
        "🔤 Text Search",
        "🖼️ Image Search",
        "✨ Multimodal Search",
    ]
)


# ============================================================
# 1. Text Recommendation
# ============================================================

with text_tab:

    st.subheader("Search using a text description")

    with st.form("text_search_form"):

        query = st.text_area(
            "Describe the book you are looking for",
            placeholder=(
                "Example: A science fiction novel about "
                "space exploration and artificial intelligence."
            ),
            height=120,
        )

        submitted = st.form_submit_button(
            "Find Books",
            use_container_width=True,
        )

    if submitted:

        if not query.strip():
            st.warning("Please enter a search description.")

        else:
            with st.spinner("Finding similar books..."):

                payload = {
                    "query": query.strip(),
                    "top_k": top_k,
                    "min_score": min_score,
                    "exclude_isbns": exclude_isbns,
                    "max_per_author": max_per_author_value,
                }

                result = send_request(
                    "POST",
                    "/recommend/text",
                    json=payload,
                )

            if result:
                display_recommendations(result)


# ============================================================
# 2. Image Recommendation
# ============================================================

with image_tab:

    st.subheader("Search using a book cover image")

    with st.form("image_search_form"):

        uploaded_image = st.file_uploader(
            "Upload a book cover",
            type=ALLOWED_IMAGE_TYPES,
        )

        submitted = st.form_submit_button(
            "Find Similar Books",
            use_container_width=True,
        )

    if uploaded_image:
        st.image(
            uploaded_image,
            caption="Uploaded image",
            width=250,
        )

    if submitted:

        if uploaded_image is None:
            st.warning("Please upload an image.")

        else:
            with st.spinner("Searching for similar book covers..."):

                files = {
                    "file": (
                        uploaded_image.name,
                        uploaded_image.getvalue(),
                        uploaded_image.type,
                    )
                }

                data = build_form_data(
                    top_k,
                    min_score,
                    max_per_author_value,
                    exclude_isbns,
                )

                result = send_request(
                    "POST",
                    "/recommend/image",
                    files=files,
                    data=data,
                )

            if result:
                display_recommendations(result)


# ============================================================
# 3. Multimodal Recommendation
# ============================================================

with multimodal_tab:

    st.subheader("Search using text and an image")

    st.write(
        "Combine a text description with a reference image "
        "to find more relevant books."
    )

    with st.form("multimodal_search_form"):

        query = st.text_area(
            "Text description",
            placeholder=(
                "Example: A fantasy novel with a mysterious "
                "and magical atmosphere."
            ),
            height=100,
        )

        uploaded_image = st.file_uploader(
            "Upload a reference image",
            type=ALLOWED_IMAGE_TYPES,
            key="multimodal_image",
        )

        st.markdown("#### Modality Weights")

        text_weight = st.slider(
            "Text importance",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )

        image_weight = round(1.0 - text_weight, 2)

        st.caption(
            f"Text weight: {text_weight:.2f} | "
            f"Image weight: {image_weight:.2f}"
        )

        submitted = st.form_submit_button(
            "Find Books",
            use_container_width=True,
        )

    if uploaded_image:
        st.image(
            uploaded_image,
            caption="Reference image",
            width=250,
        )

    if submitted:

        if not query.strip():
            st.warning("Please enter a text description.")

        elif uploaded_image is None:
            st.warning("Please upload a reference image.")

        else:
            with st.spinner("Generating multimodal recommendations..."):

                files = {
                    "file": (
                        uploaded_image.name,
                        uploaded_image.getvalue(),
                        uploaded_image.type,
                    )
                }

                data = build_form_data(
                    top_k,
                    min_score,
                    max_per_author_value,
                    exclude_isbns,
                )

                data.extend([
                    ("text", query.strip()),
                    ("text_weight", str(text_weight)),
                    ("image_weight", str(image_weight)),
                ])

                result = send_request(
                    "POST",
                    "/recommend/multimodal",
                    files=files,
                    data=data,
                )

            if result:
                display_recommendations(result)


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Multimodal Book Recommender | "
    "Built with FastAPI, Streamlit, CLIP, and Qdrant."
)
