"""Page 6 — Human Quality Evaluation & Blind A/B/C Testing UI."""

import streamlit as st
import random
import os
from pathlib import Path
from app.config import AUDIO_OUTPUT_DIR, DEFAULT_REFERENCE_AUDIO
from app.core.result_store import ResultStore
from app.core.benchmark import BenchmarkEngine

def render_evaluation_page():
    st.title("🎧 Human Quality Evaluation & Blind Testing")
    st.markdown(
        "Subjective perceptual evaluation page supporting **Blind A/B/C testing**. "
        "Audio samples are presented anonymously (Sample A, Sample B) to eliminate cognitive bias."
    )

    evaluator_name = st.sidebar.text_input("Evaluator Name / ID:", value="Evaluator_1")
    blind_mode = st.sidebar.checkbox("Enable Blind Comparison Mode", value=True)

    # Find available generated audio samples
    audio_files = list(AUDIO_OUTPUT_DIR.glob("**/*.wav"))
    if not audio_files:
        st.warning("No generated audio samples found. Run generation or benchmarks first!")
        return

    # Group audio by test case
    test_cases = BenchmarkEngine.load_evaluation_texts()
    test_options = [t["id"] for t in test_cases]
    selected_test = st.selectbox("Select Test Utterance to Evaluate:", test_options, index=0)

    # Target sentence text
    matching_text = next((t["text"] for t in test_cases if t["id"] == selected_test), "")
    st.info(f"**Target Sentence:** *\"{matching_text}\"*")

    if DEFAULT_REFERENCE_AUDIO.exists():
        with st.expander("Reference Voice Audio (Ground Truth Timbre)"):
            st.audio(str(DEFAULT_REFERENCE_AUDIO))

    # Matching samples for this test case
    matching_audios = [f for f in audio_files if selected_test in f.name]
    if not matching_audios:
        st.info(f"No audio files found for test case `{selected_test}` yet. Synthesize them in the Generate or Benchmark pages.")
        return

    st.subheader(f"Audio Samples for '{selected_test}'")

    # Map samples to blind identifiers
    if "blind_mapping" not in st.session_state or st.session_state.get("current_test") != selected_test:
        random.seed(42)
        shuffled = list(matching_audios)
        random.shuffle(shuffled)
        mapping = {}
        for idx, f in enumerate(shuffled):
            label = f"Sample {chr(65 + idx)}" if blind_mode else f.parent.name.upper()
            mapping[label] = f
        st.session_state["blind_mapping"] = mapping
        st.session_state["current_test"] = selected_test

    sample_mapping = st.session_state["blind_mapping"]

    for label, audio_path in sample_mapping.items():
        st.markdown(f"#### {label}")
        st.audio(str(audio_path))

        with st.form(key=f"eval_form_{label}_{selected_test}"):
            c1, c2, c3 = st.columns(3)
            with c1:
                score_nat = st.slider("Naturalness (1-5)", 1, 5, 4, key=f"nat_{label}")
                score_pron = st.slider("Pronunciation (1-5)", 1, 5, 4, key=f"pron_{label}")
            with c2:
                score_pros = st.slider("Prosody & Flow (1-5)", 1, 5, 4, key=f"pros_{label}")
                score_expr = st.slider("Expressiveness (1-5)", 1, 5, 4, key=f"expr_{label}")
            with c3:
                score_clone = st.slider("Speaker Similarity (1-5)", 1, 5, 3, key=f"clone_{label}")
                score_overall = st.slider("Overall Quality (1-5)", 1, 5, 4, key=f"over_{label}")

            notes = st.text_input("Qualitative Notes (Artifacts, Robotic tone, etc.):", key=f"notes_{label}")
            submitted = st.form_submit_button(f"Submit Score for {label}")

            if submitted:
                model_real_name = audio_path.parent.name
                ResultStore.append_human_eval({
                    "evaluator": evaluator_name,
                    "sample_id": audio_path.name,
                    "model": model_real_name,
                    "test_case": selected_test,
                    "naturalness": score_nat,
                    "pronunciation": score_pron,
                    "prosody": score_pros,
                    "expressiveness": score_expr,
                    "speaker_similarity": score_clone,
                    "overall_quality": score_overall,
                    "notes": notes,
                })
                st.success(f"Score recorded for {label}!")

    st.divider()
    st.subheader("Submitted Human Evaluation Scores (`human_scores.csv`)")
    scores_df = ResultStore.load_human_scores()
    if not scores_df.empty:
        st.dataframe(scores_df.sort_index(ascending=False), use_container_width=True)
    else:
        st.caption("No human evaluation scores submitted yet.")
