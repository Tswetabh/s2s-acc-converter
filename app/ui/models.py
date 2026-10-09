"""Page 2 — Models Catalog & Diagnostic Health Check Cards."""

import streamlit as st
import pandas as pd
from app.registry import get_all_models

def render_models_page():
    st.title("📦 Evaluated TTS Models Catalog")
    st.markdown(
        "Complete technical cards and health diagnostics for all 14 evaluated Text-to-Speech models. "
        "Each adapter exposes standardized capabilities and licensing metadata."
    )

    models = get_all_models()

    # Filter by classification
    categories = ["All Models", "LOCAL — COMFORTABLE", "LOCAL — MEMORY LIMITED", "CLOUD — RECOMMENDED"]
    selected_cat = st.selectbox("Filter by Hardware Profile:", categories)

    # Summary table
    table_rows = []
    for m in models:
        meta = m.get_metadata()
        health = m.health_check()
        if selected_cat != "All Models" and selected_cat not in meta["hardware_classification"]:
            continue

        table_rows.append({
            "Model": meta["name"],
            "Parameters": meta["parameter_count"],
            "Languages": ", ".join(meta["languages"][:4]) + ("..." if len(meta["languages"]) > 4 else ""),
            "Cloning Type": meta["cloning_type"].replace("_", " ").title(),
            "Streaming": meta["streaming_type"].title(),
            "Hardware Profile": meta["hardware_classification"],
            "Health Status": health["status"],
            "Weights License": meta["license_weights"],
        })

    if table_rows:
        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No models match this filter criteria.")

    st.divider()
    st.subheader("Individual Model Diagnostic Cards")

    for m in models:
        meta = m.get_metadata()
        if selected_cat != "All Models" and selected_cat not in meta["hardware_classification"]:
            continue

        health = m.health_check()
        status_color = "🟢" if "READY" in health["status"] else ("🟡" if "LIMITED" in health["status"] else "🔵")

        with st.expander(f"{status_color} {meta['name']} ({meta['parameter_count']}) — {meta['hardware_classification']}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Model ID:** `{meta['model_id']}`")
                st.markdown(f"**Version / Revision:** {meta['version']}")
                st.markdown(f"**Supported Languages:** {', '.join(meta['languages'])}")
                st.markdown(f"**Voice Cloning:** {'Yes (' + meta['cloning_type'] + ')' if meta['supports_voice_cloning'] else 'No (Predefined voices)'}")
                st.markdown(f"**Streaming:** {meta['streaming_type'].title()}")
                st.markdown(f"**Sample Rate:** {meta['sample_rate']} Hz")
            with c2:
                st.markdown(f"**Code License:** `{meta['license_code']}`")
                st.markdown(f"**Weights License:** `{meta['license_weights']}`")
                st.markdown(f"**Expected VRAM:** ~{meta['expected_vram_gb']} GB")
                st.markdown(f"**Health Check:** `{health['status']}` — *{health['reason']}*")
                if meta["repo_url"]:
                    st.markdown(f"🔗 [Official Repository]({meta['repo_url']})")

                if "DEPENDENCY_MISSING" in health["status"]:
                    if st.button(f"📥 Install {meta['name']} Dependencies", key=f"inst_{meta['model_id']}"):
                        with st.spinner(f"Installing dependencies for {meta['name']}..."):
                            import subprocess, sys
                            cmd = [sys.executable, "scripts/install_model.py", "--model", meta["model_id"]]
                            res = subprocess.run(cmd, capture_output=True, text=True)
                            if res.returncode == 0:
                                st.success("Installed successfully!")
                                st.rerun()
                            else:
                                st.error(f"Installation failed: {res.stderr[:300]}")
