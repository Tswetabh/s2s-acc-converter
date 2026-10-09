"""Page 3 — Interactive Audio Generation Playground."""

import streamlit as st
import os
import time
from datetime import datetime
from pathlib import Path

from app.registry import get_model, list_available_models
from app.core.inference import InferenceEngine
from app.config import AUDIO_OUTPUT_DIR, DEFAULT_REFERENCE_AUDIO

def render_generate_page():
    st.title("🎙️ Interactive Audio Synthesis")
    st.markdown("Test single-sentence or custom text inference with model-specific controls.")

    model_list = list_available_models()
    col_m, col_d = st.columns([3, 1])
    with col_m:
        selected_model_id = st.selectbox("Choose TTS Model:", model_list, index=0)
    with col_d:
        device_choice = st.selectbox("Device:", ["auto", "cuda", "cpu"])

    model_adapter = get_model(selected_model_id)
    if not model_adapter:
        st.error("Selected model adapter not found.")
        return

    meta = model_adapter.get_metadata()
    health = model_adapter.health_check()

    # Model status alert
    if "READY" in health["status"]:
        st.success(f"Status: {health['status']} | Hardware Profile: {meta['hardware_classification']}")
    elif "LIMITED" in health["status"]:
        st.warning(f"Status: {health['status']} | Hardware Profile: {meta['hardware_classification']}")
    elif "DEPENDENCY_MISSING" in health["status"]:
        st.warning(f"⚠️ {meta['name']} is not installed yet: {health['reason']}")
        if st.button(f"📥 Install {meta['name']} Dependencies Now", type="secondary"):
            with st.spinner(f"Installing packages for {meta['name']} on Drive D..."):
                import subprocess, sys
                cmd = [sys.executable, "scripts/install_model.py", "--model", selected_model_id]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0:
                    st.success("Installed successfully! Reloading...")
                    st.rerun()
                else:
                    st.error(f"Installation failed: {res.stderr[:300]}")
    else:
        st.info(f"Status: {health['status']} — {health['reason']}")

    text_input = st.text_area(
        "Text to Synthesize:",
        value="Artificial intelligence is changing the way people learn, work, and communicate across the globe.",
        height=100
    )

    KOKORO_VOICES = {
        "af_bella": "🇺🇸 American Female — Bella (Default)",
        "af_sarah": "🇺🇸 American Female — Sarah",
        "af_heart": "🇺🇸 American Female — Heart (High Quality)",
        "af_nicole": "🇺🇸 American Female — Nicole",
        "af_sky": "🇺🇸 American Female — Sky",
        "af_alloy": "🇺🇸 American Female — Alloy",
        "am_adam": "🇺🇸 American Male — Adam",
        "am_michael": "🇺🇸 American Male — Michael",
        "am_echo": "🇺🇸 American Male — Echo",
        "am_eric": "🇺🇸 American Male — Eric",
        "am_fenrir": "🇺🇸 American Male — Fenrir",
        "am_liam": "🇺🇸 American Male — Liam",
        "am_onyx": "🇺🇸 American Male — Onyx",
        "am_puck": "🇺🇸 American Male — Puck",
        "bf_emma": "🇬🇧 British Female — Emma",
        "bf_isabella": "🇬🇧 British Female — Isabella",
        "bf_alice": "🇬🇧 British Female — Alice",
        "bm_george": "🇬🇧 British Male — George",
        "bm_lewis": "🇬🇧 British Male — Lewis",
        "bm_daniel": "🇬🇧 British Male — Daniel",
        "hf_alpha": "🇮🇳 Hindi Female — Alpha",
        "hf_beta": "🇮🇳 Hindi Female — Beta",
        "hm_omega": "🇮🇳 Hindi Male — Omega",
        "hm_psi": "🇮🇳 Hindi Male — Psi",
        "jf_alpha": "🇯🇵 Japanese Female — Alpha",
        "ef_dora": "🇪🇸 Spanish Female — Dora",
        "em_alex": "🇪🇸 Spanish Male — Alex",
        "ff_siwis": "🇫🇷 French Female — Siwis",
        "if_sara": "🇮🇹 Italian Female — Sara",
    }

    c_opt1, c_opt2, c_opt3 = st.columns(3)
    with c_opt1:
        lang_choice = st.selectbox("Language:", meta["languages"], index=0)
    with c_opt2:
        if selected_model_id == "kokoro":
            voice_keys = list(KOKORO_VOICES.keys())
            chosen_key = st.selectbox(
                "Voice Style / Speaker:",
                options=voice_keys,
                format_func=lambda k: KOKORO_VOICES[k],
                index=0
            )
            speaker_input = chosen_key
        else:
            speaker_input = st.text_input("Speaker / Voice ID (Optional):", value="")
    with c_opt3:
        speed_slider = st.slider("Speech Speed:", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

    ref_audio_path = None
    if meta["supports_voice_cloning"]:
        st.subheader("Zero-Shot Voice Cloning Conditioning")
        st.caption("Upload a clean reference speech sample (3 to 10 seconds recommended).")
        uploaded_ref = st.file_uploader(
            "Upload Reference Audio (WAV, MP3, FLAC):",
            type=["wav", "mp3", "flac", "ogg"],
            help="Zero-shot models (like F5-TTS, XTTS-v2) condition on this audio clip to mimic vocal timbre."
        )
        if uploaded_ref:
            from app.core.audio_utils import prepare_reference_audio
            temp_ref = Path("D:/tts-poc/outputs/audio/temp_ref.wav")
            try:
                prep_info = prepare_reference_audio(
                    input_source=uploaded_ref,
                    output_path=temp_ref,
                    max_duration_sec=10.0,
                    target_sample_rate=24000
                )
                ref_audio_path = prep_info["output_path"]
                if prep_info["was_trimmed"]:
                    st.info(
                        f"✂️ Clip was **{prep_info['orig_duration_sec']}s** ({uploaded_ref.size / (1024*1024):.1f} MB). "
                        f"Safely downmixed to mono and trimmed to **{prep_info['final_duration_sec']}s** to prevent GPU VRAM overload on RTX 3050."
                    )
                else:
                    st.success(f"Reference voice ready ({prep_info['final_duration_sec']}s, 24kHz mono).")
                st.audio(ref_audio_path)
            except Exception as e:
                st.error(f"Error processing audio file: {e}")
        else:
            preset_voices = {
                "🇮🇳 Hindi Female Voice (Tyagi, 5.2s)": "data/reference_voices/hindi_f_tyagi.wav",
                "👩 English Female — Studio (af_heart, 8.5s)": "data/reference_voices/reference_01.wav",
                "👨 English Male — Studio (am_adam, 8.3s)": "data/reference_voices/reference_02.wav",
            }
            chosen_preset = st.radio("Or select a clean preset reference voice:", list(preset_voices.keys()), horizontal=True)
            preset_file = Path(preset_voices[chosen_preset])
            if preset_file.exists():
                ref_audio_path = str(preset_file)
                st.audio(ref_audio_path)

    if st.button("🚀 Generate Audio", type="primary", use_container_width=True):
        out_dir = AUDIO_OUTPUT_DIR / selected_model_id
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = str(out_dir / f"{selected_model_id}_{timestamp}.wav")

        with st.spinner(f"Synthesizing audio via {meta['name']}..."):
            try:
                metrics = InferenceEngine.run_generation(
                    model_adapter=model_adapter,
                    text=text_input,
                    output_path=out_file,
                    device=device_choice,
                    language=lang_choice,
                    speaker=speaker_input or None,
                    reference_audio=ref_audio_path,
                    speed=speed_slider,
                )

                st.success("Generation completed successfully!")
                st.audio(out_file)

                # Metric Cards
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Gen Time", f"{metrics['generation_time_sec']} s")
                m2.metric("Duration", f"{metrics['audio_duration_sec']} s")
                m3.metric("RTF", f"{metrics['rtf']}")
                m4.metric("Peak VRAM", f"{metrics['vram_peak_gb']} GB")
                m5.metric("TTFA", f"{metrics['ttfa_sec']} s" if metrics['ttfa_sec'] != "N/A" else "N/A")

                with open(out_file, "rb") as f:
                    st.download_button(
                        label="⬇️ Download WAV Audio",
                        data=f,
                        file_name=f"{selected_model_id}_{timestamp}.wav",
                        mime="audio/wav"
                    )

            except Exception as e:
                st.error(f"Inference Error: {str(e)}")
            finally:
                InferenceEngine.unload_active_model()
