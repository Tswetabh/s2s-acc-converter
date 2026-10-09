"""Page 5 — Cross-Model Comparison & Metrics Visualization UI."""

import streamlit as st
import pandas as pd
import numpy as np
from app.registry import get_all_models
from app.core.result_store import ResultStore
from app.core.scoring import ScoringEngine

def render_comparison_page():
    st.title("📊 Model Comparison & Performance Telemetry")
    st.markdown(
        "Comparative analytics across speed (RTF), memory footprint (Peak VRAM), "
        "and human evaluation scores under configurable weighting."
    )

    bench_df = ResultStore.load_benchmark_results()
    human_df = ResultStore.load_human_scores()

    models = get_all_models()
    meta_map = {m.get_metadata()["model_id"]: m.get_metadata() for m in models}

    # Weight Customization Expander
    with st.expander("⚙️ Customize Weighted Scoring Formula", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        w_nat = c1.slider("Naturalness", 0, 50, 30, 5) / 100.0
        w_pros = c2.slider("Prosody", 0, 30, 15, 5) / 100.0
        w_pron = c3.slider("Pronunciation", 0, 30, 15, 5) / 100.0
        w_expr = c4.slider("Expressiveness", 0, 30, 10, 5) / 100.0

        c5, c6, c7, c8 = st.columns(4)
        w_clone = c5.slider("Voice Cloning", 0, 30, 10, 5) / 100.0
        w_multi = c6.slider("Multilingual", 0, 30, 10, 5) / 100.0
        w_speed = c7.slider("Speed / RTF", 0, 30, 5, 5) / 100.0
        w_vram = c8.slider("VRAM Efficiency", 0, 30, 5, 5) / 100.0

    custom_weights = {
        "naturalness": w_nat,
        "prosody": w_pros,
        "pronunciation": w_pron,
        "expressiveness": w_expr,
        "voice_cloning": w_clone,
        "multilingual": w_multi,
        "speed": w_speed,
        "vram_efficiency": w_vram,
    }

    scorer = ScoringEngine(custom_weights)
    composite_df = scorer.compute_composite_scores(bench_df, human_df, meta_map)
    rankings = scorer.derive_rankings(composite_df)

    st.subheader("🏆 Evidence-Based Use-Case Rankings")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("🥇 Best Overall Naturalness", rankings.get("best_naturalness", "N/A"))
    r2.metric("💻 Best Local (RTX 3050)", rankings.get("best_local_3050", "N/A"))
    r3.metric("⚡ Best Low-Latency", rankings.get("best_low_latency", "N/A"))
    r4.metric("👥 Best Voice Cloning", rankings.get("best_voice_cloning", "N/A"))

    r5, r6, r7, r8 = st.columns(4)
    r5.metric("🌍 Best Multilingual", rankings.get("best_multilingual", "N/A"))
    r6.metric("🎭 Best Expressive", rankings.get("best_expressive", "N/A"))
    r7.metric("💬 Best Dialogue", rankings.get("best_dialogue", "N/A"))
    r8.metric("⚖️ Best Quality/VRAM Ratio", rankings.get("best_quality_to_vram", "N/A"))

    st.divider()

    st.subheader("Composite Score Leaderboard")
    st.dataframe(
        composite_df[[
            "model_name", "composite_score", "naturalness", "prosody",
            "voice_cloning", "speed_score", "vram_score", "hardware_classification"
        ]],
        use_container_width=True,
        hide_index=True
    )

    # Real-Time Factor (RTF) Bar Chart
    if not bench_df.empty:
        st.subheader("Measured Real-Time Factor (RTF) Comparison")
        st.caption("Lower is faster (RTF < 1.0 is faster than real time).")
        valid_bench = bench_df[bench_df["status"] == "SUCCESS"].copy()
        if not valid_bench.empty:
            avg_rtf_by_model = valid_bench.groupby("model")["rtf"].mean().reset_index()
            st.bar_chart(avg_rtf_by_model.set_index("model")["rtf"])

            st.subheader("Measured Peak VRAM (GB) Comparison")
            st.caption("Peak VRAM reached on RTX 3050 6GB during synthesis.")
            avg_vram = valid_bench.groupby("model")["vram_peak_gb"].max().reset_index()
            st.bar_chart(avg_vram.set_index("model")["vram_peak_gb"])
