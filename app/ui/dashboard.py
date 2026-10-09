"""Page 1 — System Diagnostics & Hardware Monitoring UI."""

import streamlit as st
from app.system_info import get_system_environment
from app.config import VRAM_SAFE_CEILING_GB, VRAM_MAX_TOTAL_GB

def render_dashboard_page():
    st.title("🖥️ System Diagnostics & Runtime Environment")
    st.markdown(
        "Real-time hardware inspection for the **RTX 3050 6GB Laptop** development workstation. "
        "Strict memory monitoring prevents CUDA Out-Of-Memory exceptions."
    )

    env = get_system_environment()

    # Metric Cards Top Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("GPU Device", env["gpu_name"] if env["gpu_available"] else "CPU Only")
    with col2:
        st.metric("Total VRAM", f"{env['vram_total_gb']} GB")
    with col3:
        st.metric("Free VRAM", f"{env['vram_free_gb']} GB")
    with col4:
        st.metric("System RAM", f"{env['ram_total_gb']} GB")

    st.divider()

    # Detailed Hardware Breakdown
    c_left, c_right = st.columns(2)

    with c_left:
        st.subheader("Hardware Specifications")
        st.table({
            "Component": ["Processor (CPU)", "Graphics Card", "Total VRAM", "System RAM", "Operating System"],
            "Specification": [
                env["cpu"],
                env["gpu_name"],
                f"{env['vram_total_gb']} GB",
                f"{env['ram_total_gb']} GB (Available: {env['ram_available_gb']} GB)",
                f"{env['os']} (Build {env['os_version']})",
            ]
        })

    with c_right:
        st.subheader("Software & Acceleration Stack")
        st.table({
            "Runtime": ["Python Version", "PyTorch Version", "CUDA Runtime", "NVIDIA Driver"],
            "Status": [
                env["python_version"],
                env["torch_version"],
                env["cuda_version"] if env["gpu_available"] else "N/A",
                env["driver_version"] if env["gpu_available"] else "N/A",
            ]
        })

    st.subheader("VRAM Headroom & 6 GB Safety Thresholds")
    if env["vram_total_gb"] > 0:
        used_ratio = (env["vram_total_gb"] - env["vram_free_gb"]) / env["vram_total_gb"]
        st.progress(min(1.0, max(0.0, used_ratio)), text=f"Active VRAM Utilization: {round(used_ratio * 100, 1)}%")

    st.info(
        f"**Hardware Policy Notice:** Working under a 6.0 GB VRAM constraint. "
        f"Safe local inference ceiling is set to **{VRAM_SAFE_CEILING_GB} GB** to preserve OS display buffers. "
        "Models exceeding this budget are classified as `☁️ CLOUD — RECOMMENDED`."
    )
