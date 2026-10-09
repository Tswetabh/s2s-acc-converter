"""Page 4 — Standardized Benchmark Runner UI."""

import streamlit as st
import pandas as pd
from app.registry import get_all_models, get_model
from app.core.benchmark import BenchmarkEngine
from app.core.result_store import ResultStore
from app.config import HW_CLOUD_RECOMMENDED, RESULTS_CSV_PATH

def render_benchmark_page():
    st.title("⚡ Standardized Benchmark Suite")
    st.markdown(
        "Execute automated, evidence-based performance benchmarks against standardized test cases. "
        "Strict single-model memory unloading guarantees that models do not leak VRAM."
    )

    models = get_all_models()
    model_names = [m.get_metadata()["name"] for m in models]
    model_id_map = {m.get_metadata()["name"]: m.get_metadata()["model_id"] for m in models}

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_model_names = st.multiselect(
            "Select Models to Benchmark:",
            options=model_names,
            default=["Kokoro v1.0 — 82M", "Piper"]
        )
    with col2:
        test_mode = st.radio("Test Scope:", ["Quick Run (normal_01)", "Full Test Suite (12 Categories)"])

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    run_selected = col_btn1.button("▶️ Run Selected Models", type="primary", use_container_width=True)
    run_local = col_btn2.button("💻 Run All Local Models", use_container_width=True)
    run_full = col_btn3.button("🌐 Run All 14 Models", use_container_width=True)

    test_case_filter = ["normal_01"] if "Quick" in test_mode else None

    target_models = []
    skip_cloud = False

    if run_selected:
        target_models = [get_model(model_id_map[n]) for n in selected_model_names if model_id_map.get(n)]
    elif run_local:
        target_models = models
        skip_cloud = True
    elif run_full:
        target_models = models
        skip_cloud = False

    if target_models:
        progress_bar = st.progress(0, text="Initializing benchmark runs...")
        status_placeholder = st.empty()

        total = len(target_models)
        for idx, model_adapter in enumerate(target_models):
            meta = model_adapter.get_metadata()
            status_placeholder.info(f"Running benchmark [{idx+1}/{total}]: **{meta['name']}**...")

            BenchmarkEngine.benchmark_model(
                model_adapter=model_adapter,
                test_case_ids=test_case_filter,
                skip_cloud_recommended=skip_cloud,
            )
            progress_bar.progress((idx + 1) / total)

        status_placeholder.success("🎉 Benchmark suite finished! Results appended to CSV storage.")

    st.divider()
    st.subheader("Historical Benchmark Records (`results.csv`)")

    df = ResultStore.load_benchmark_results()
    if not df.empty:
        st.dataframe(df.sort_index(ascending=False), use_container_width=True)
        if RESULTS_CSV_PATH.exists():
            with open(RESULTS_CSV_PATH, "rb") as f:
                st.download_button("⬇️ Download Full results.csv", data=f, file_name="benchmark_results.csv", mime="text/csv")
    else:
        st.info("No benchmark records found yet. Run an automated benchmark above.")
