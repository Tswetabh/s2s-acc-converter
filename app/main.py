"""Unified Open-Source TTS Evaluation Platform — Main Streamlit Application."""

import streamlit as st
import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Page configuration
st.set_page_config(
    page_title="Unified Open-Source TTS Evaluation Platform",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import UI views
from app.ui.dashboard import render_dashboard_page
from app.ui.models import render_models_page
from app.ui.generate import render_generate_page
from app.ui.benchmark import render_benchmark_page
from app.ui.comparison import render_comparison_page
from app.ui.evaluation import render_evaluation_page

def main():
    # Sidebar navigation
    st.sidebar.title("🎙️ Unified TTS POC")
    st.sidebar.caption("RTX 3050 Laptop GPU (6GB VRAM) — Windows 11")
    st.sidebar.divider()

    pages = {
        "🖥️ System Diagnostics": render_dashboard_page,
        "📦 Models Catalog": render_models_page,
        "🎙️ Generate Audio": render_generate_page,
        "⚡ Benchmark Suite": render_benchmark_page,
        "📊 Comparison & Telemetry": render_comparison_page,
        "🎧 Human Evaluation (Blind)": render_evaluation_page,
    }

    selected_page = st.sidebar.radio("Navigation:", list(pages.keys()))

    st.sidebar.divider()
    st.sidebar.markdown(
        "**Core Hardware Rule:**\n"
        "• Max Safe Local VRAM: **4.8 GB**\n"
        "• Single-Model Residency Enforced\n"
        "• 0 bytes cached on Drive C\n"
        "• Storage: `D:\\tts-poc`"
    )

    # Render selected view
    pages[selected_page]()

if __name__ == "__main__":
    main()
