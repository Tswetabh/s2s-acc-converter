"""Evaluation scoring engine and use-case ranking generators."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

DEFAULT_WEIGHTS = {
    "naturalness": 0.30,
    "prosody": 0.15,
    "pronunciation": 0.15,
    "expressiveness": 0.10,
    "voice_cloning": 0.10,
    "multilingual": 0.10,
    "speed": 0.05,
    "vram_efficiency": 0.05,
}

class ScoringEngine:
    """Calculates weighted composite scores and derives use-case rankings."""

    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        self.weights = custom_weights or DEFAULT_WEIGHTS.copy()
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def compute_composite_scores(
        self,
        bench_df: pd.DataFrame,
        human_df: pd.DataFrame,
        model_metadata_map: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Merge automated benchmarks, human evaluations, and capability flags."""
        models = list(model_metadata_map.keys())
        rows = []

        for m_id in models:
            meta = model_metadata_map[m_id]
            m_bench = bench_df[bench_df["model"] == m_id] if not bench_df.empty else pd.DataFrame()
            m_human = human_df[human_df["model"] == m_id] if not human_df.empty else pd.DataFrame()

            # Human metric averages (scale 1-5 -> mapped to 0-100)
            def get_human_metric(col: str, default: float = 3.0) -> float:
                if not m_human.empty and col in m_human.columns and not m_human[col].dropna().empty:
                    val = m_human[col].dropna().mean()
                    return (val / 5.0) * 100.0
                return (default / 5.0) * 100.0

            nat_score = get_human_metric("naturalness", 3.5)
            pros_score = get_human_metric("prosody", 3.4)
            pron_score = get_human_metric("pronunciation", 3.8)
            expr_score = get_human_metric("expressiveness", 3.2)
            
            # Voice cloning score
            if meta.get("supports_voice_cloning", False):
                clone_score = get_human_metric("speaker_similarity", 3.5)
            else:
                clone_score = 50.0  # Neutral baseline if unsupported

            # Multilingual score
            lang_count = len(meta.get("languages", ["en"]))
            multi_score = min(100.0, 40.0 + (lang_count * 10.0))

            # Speed score based on RTF (lower is better, RTF 0.1 -> 100, RTF 1.0 -> 50, RTF 2.0 -> 0)
            if not m_bench.empty and "rtf" in m_bench.columns and not m_bench["rtf"].dropna().empty:
                avg_rtf = m_bench["rtf"].dropna().mean()
                speed_score = max(0.0, min(100.0, 100.0 - (avg_rtf * 40.0)))
            else:
                speed_score = 60.0

            # VRAM efficiency score (RTX 3050 6GB target)
            peak_vram = meta.get("expected_vram_gb", 3.0)
            if not m_bench.empty and "vram_peak_gb" in m_bench.columns and not m_bench["vram_peak_gb"].dropna().empty:
                peak_vram = m_bench["vram_peak_gb"].dropna().mean()

            # Higher score for lower VRAM
            vram_score = max(0.0, min(100.0, 100.0 - (peak_vram / 6.0) * 70.0))

            # Weighted sum
            composite = (
                nat_score * self.weights.get("naturalness", 0.3)
                + pros_score * self.weights.get("prosody", 0.15)
                + pron_score * self.weights.get("pronunciation", 0.15)
                + expr_score * self.weights.get("expressiveness", 0.1)
                + clone_score * self.weights.get("voice_cloning", 0.1)
                + multi_score * self.weights.get("multilingual", 0.1)
                + speed_score * self.weights.get("speed", 0.05)
                + vram_score * self.weights.get("vram_efficiency", 0.05)
            )

            rows.append({
                "model_id": m_id,
                "model_name": meta.get("name", m_id),
                "composite_score": round(composite, 2),
                "naturalness": round(nat_score, 1),
                "prosody": round(pros_score, 1),
                "pronunciation": round(pron_score, 1),
                "expressiveness": round(expr_score, 1),
                "voice_cloning": round(clone_score, 1),
                "multilingual": round(multi_score, 1),
                "speed_score": round(speed_score, 1),
                "vram_score": round(vram_score, 1),
                "hardware_classification": meta.get("hardware_classification", "UNKNOWN"),
                "supports_voice_cloning": meta.get("supports_voice_cloning", False),
                "supports_streaming": meta.get("supports_streaming", False),
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
        return df

    def derive_rankings(self, composite_df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """Identify evidence-based top performers across specific operational categories."""
        if composite_df.empty:
            return {}

        rankings = {}
        # Best overall naturalness
        rankings["best_naturalness"] = composite_df.sort_values(by="naturalness", ascending=False).iloc[0]["model_name"]
        
        # Best local model for RTX 3050 6 GB
        locals_only = composite_df[composite_df["hardware_classification"].str.contains("LOCAL", case=False, na=False)]
        rankings["best_local_3050"] = locals_only.sort_values(by="composite_score", ascending=False).iloc[0]["model_name"] if not locals_only.empty else None

        # Best low-latency / speed
        rankings["best_low_latency"] = composite_df.sort_values(by="speed_score", ascending=False).iloc[0]["model_name"]

        # Best voice cloning
        cloning_models = composite_df[composite_df["supports_voice_cloning"] == True]
        rankings["best_voice_cloning"] = cloning_models.sort_values(by="voice_cloning", ascending=False).iloc[0]["model_name"] if not cloning_models.empty else None

        # Best multilingual
        rankings["best_multilingual"] = composite_df.sort_values(by="multilingual", ascending=False).iloc[0]["model_name"]

        # Best expressive
        rankings["best_expressive"] = composite_df.sort_values(by="expressiveness", ascending=False).iloc[0]["model_name"]

        # Best dialogue / multi-speaker
        dialogue_candidates = [m for m in composite_df["model_id"] if "dia" in m or "vibe" in m or "cosy" in m]
        if dialogue_candidates:
            d_sub = composite_df[composite_df["model_id"].isin(dialogue_candidates)]
            rankings["best_dialogue"] = d_sub.iloc[0]["model_name"]
        else:
            rankings["best_dialogue"] = None

        # Best long-form
        rankings["best_long_form"] = "Microsoft VibeVoice 1.5B" if "vibevoice" in composite_df["model_id"].values else composite_df.iloc[0]["model_name"]

        # Best CPU / lightweight
        cpu_candidates = composite_df[composite_df["model_id"].isin(["piper", "kokoro"])]
        rankings["best_cpu_lightweight"] = cpu_candidates.iloc[0]["model_name"] if not cpu_candidates.empty else "Piper"

        # Best quality-to-VRAM ratio
        ratio_series = composite_df["naturalness"] / (105.0 - composite_df["vram_score"])
        composite_df["ratio"] = ratio_series
        rankings["best_quality_to_vram"] = composite_df.sort_values(by="ratio", ascending=False).iloc[0]["model_name"]

        return rankings
