import pathlib
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="DemenciaScan — Neural Acoustic v2.4",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Full-bleed dark workstation styling so the Google Stitch UI fills 100% of the viewport
st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background-color: #111318 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
        }
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"],
        footer,
        #MainMenu {
            display: none !important;
        }
        .block-container, [data-testid="stMainBlockContainer"] {
            padding: 0 !important;
            margin: 0 !important;
            max-width: 100vw !important;
            width: 100vw !important;
            height: 100vh !important;
        }
        iframe {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            border: none !important;
            z-index: 9999 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

html_path = pathlib.Path(__file__).parent / "stitch_ui.html"
if not html_path.exists():
    html_path = pathlib.Path(__file__).parent.parent / "index.html"

stitch_html = html_path.read_text(encoding="utf-8")
components.html(stitch_html, height=1080, scrolling=True)
