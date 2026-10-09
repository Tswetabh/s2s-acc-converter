import copy
import logging
import os
import queue
import statistics
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import safetensors
import torch
from torch import nn
from torch.nn import functional as F
from typing_extensions import Self

from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.default_parameters import (
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_TEMPERATURE,
    DEFAULT_VARIANT,
)
from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules import mimi_transformer
from pocket_tts.modules.dummy_quantizer import DummyQuantizer
from pocket_tts.modules.seanet import SEANetDecoder, SEANetEncoder
from pocket_tts.modules.stateful_module import increment_steps, init_states
from pocket_tts.preprocessing.text_normalizer import build_tts_prompt
from pocket_tts.utils.config import Config, load_config
from pocket_tts.utils.utils import (
    PREDEFINED_VOICES,
    display_execution_time,
    download_if_necessary,
    load_predefined_voice,
    size_of_dict,
)
from pocket_tts.utils.weights_loading import get_flow_lm_state_dict, get_mimi_state_dict

torch.set_num_threads(1)
logger = logging.getLogger(__name__)


def _clone_flow_state_for_batch(
    model_state: dict[str, dict[str, torch.Tensor]], batch_size: int
) -> dict[str, dict[str, torch.Tensor]]:
    """Duplicate batch-one voice-conditioned FlowLM state across items."""
    result = {}
    for module_name, module_state in model_state.items():
        result[module_name] = {}
        for key, value in module_state.items():
            if key == "cache":
                repeats = [1, batch_size] + [1] * (value.ndim - 2)
                result[module_name][key] = value.repeat(*repeats)
            elif isinstance(value, torch.Tensor):
                result[module_name][key] = value.clone()
            else:
                result[module_name][key] = value
    return result


class TTSModel(nn.Module):
    """Class representing a Text-to-Speech (TTS) model using a flow language model for generation.
    Initialization parameters include the flow language model, temperature for sampling, number of decoding steps, optional noise clamp, end-of-sequence threshold, and configuration settings.
    """
    def __init__(
        self,
        flow_lm: FlowLMModel,
        temp: float,
        lsd_decode_steps: int,
        noise_clamp: float | None,
        eos_threshold,
        config: Config,
    ):
        """Initialize a new instance of the class.
        Args:
        flow_lm (FlowLMModel): The language model to use.
        temp (float): Temperature for sampling.
        lsd_decode_steps (int): Number of steps for LSD decoding.
        noise_clamp (float | None): Noise clamp value, or None if not used.
        eos_threshold: Threshold for end-of-sequence detection.
        config (Config): Configuration object.
        """
        super().__init__()
        self.flow_lm = flow_lm
        self.temp = temp
        self.lsd_decode_steps = lsd_decode_steps
        self.noise_clamp = noise_clamp
        self.eos_threshold = eos_threshold
        self.config = config
        self.has_voice_cloning = True
        self.max_tokens_per_chunk = 50
        self.last_generation_timing = {}
        self.last_batch_step_timing: dict = {}
        # Opt-in CUDA graphs for AR steps (T1/v2). Env POCKET_TTS_CUDA_GRAPHS=1 also enables.
        # Registry keys: (batch_size, lsd_decode_steps) → capture once, prepare+replay per unit.
        self.cuda_graphs_enabled = False
        self._flow_lm_step_graphs: dict[tuple[int, int], Any] = {}
        self._mimi_decode_graphs: dict[int, Any] = {}
        self.cuda_graph_stats = self._new_cuda_graph_stats()

    @property
    def device(self) -> str:
        """Returns the device type of the next parameter.
        Args:
        None
        Returns:
        str: The device type.
        ---
        Provides access to the sample rate configuration.
        Args:
        None
        Returns:
        int: The sample rate.
        ---
        Creates an instance from a Pydantic config.
        Args:
        config (Config): The Pydantic config object.
        temp (float): Temperature for sampling.
        lsd_decode_steps (int): Number of steps for LSD decoding.
        noise_clamp (float | None): Noise clamp value, or None if not applicable.
        eos_threshold: Threshold for end-of-sentence.
        Returns:
        Self: A new instance of the class.
        """
        return next(self.parameters()).device.type

    @property
    def sample_rate(self) -> int:
        """Returns the sample rate from the configuration.
        Args:
        - self: The instance of the class.
        - Returns: An integer representing the sample rate.
        ---
        Initializes a TTS model from Pydantic config and additional parameters.
        Args:
        - cls: The class itself.
        - config: The Pydantic config object.
        - temp: Temperature value for decoding.
        - lsd_decode_steps: Steps for LSD decoding.
        - noise_clamp: Optional float to clamp noise.
        - eos_threshold: Threshold for end-of-sentence detection.
        - Returns: A new instance of the TTS model.
        ---
        Initializes a TTS model from Pydantic config with weights.
        """
        return self.config.mimi.sample_rate

    @classmethod
    def _from_pydantic_config(
        cls, config: Config, temp, lsd_decode_steps, noise_clamp: float | None, eos_threshold
    ) -> Self:
        """Constructs a TTSModel instance from a Pydantic configuration.
        Args:
        config (Config): The Pydantic configuration.
        temp (float): Temperature value for generation.
        lsd_decode_steps (int): Number of steps for LSD decoding.
        noise_clamp (float | None): Clamping value for noise, or None if not applicable.
        eos_threshold: Threshold for end-of-sentence.
        Returns:
        Self: The constructed TTSModel instance.
        """
        flow_lm = FlowLMModel.from_pydantic_config(
            config.flow_lm, latent_dim=config.mimi.quantizer.dimension
        )
        tts_model = cls(flow_lm, temp, lsd_decode_steps, noise_clamp, eos_threshold, config)
        return tts_model

    @classmethod
    def _from_pydantic_config_with_weights(
        cls, config: Config, temp, lsd_decode_steps, noise_clamp: float | None, eos_threshold
    ) -> Self:
        """Constructs a TTS model instance from Pydantic configuration with specified weights.
        Args:
        config (Config): Configuration object containing necessary parameters.
        temp (float): Temperature value for decoding.
        lsd_decode_steps (int): Number of steps for LSD decoding.
        noise_clamp (float | None): Noise clamp value, optional.
        eos_threshold: End-of-sentence threshold.
        Returns:
        Self: An instance of the TTS model with initialized weights.
        """
        tts_model = cls._from_pydantic_config(
            config, temp, lsd_decode_steps, noise_clamp, eos_threshold
        )
        tts_model.flow_lm.speaker_proj_weight = torch.nn.Parameter(
            torch.zeros((1024, 512), dtype=torch.float32)
        )
        if config.flow_lm.weights_path is not None:
            if config.mimi.weights_path is None:
                raise ValueError(
                    "If you specify flow_lm.weights_path you should specify mimi.weights_path"
                )
            logger.info(f"Loading FlowLM weights from {config.flow_lm.weights_path}")
            state_dict_flowlm = get_flow_lm_state_dict(
                download_if_necessary(config.flow_lm.weights_path)
            )
            tts_model.flow_lm.load_state_dict(state_dict_flowlm, strict=True)

        # safetensors.torch.save_file(tts_model.state_dict(), "7442637a.safetensors")
        # Create mimi config directly from the provided config using model_dump
        mimi_config = config.mimi.model_dump()

        # Build mimi model from config
        encoder = SEANetEncoder(**mimi_config["seanet"])
        decoder = SEANetDecoder(**mimi_config["seanet"])

        encoder_transformer = mimi_transformer.ProjectedTransformer(**mimi_config["transformer"])
        decoder_transformer = mimi_transformer.ProjectedTransformer(**mimi_config["transformer"])
        quantizer = DummyQuantizer(**mimi_config["quantizer"])

        tts_model.mimi = MimiModel(
            encoder,
            decoder,
            quantizer,
            channels=mimi_config["channels"],
            sample_rate=mimi_config["sample_rate"],
            frame_rate=mimi_config["frame_rate"],
            encoder_frame_rate=mimi_config["sample_rate"] / encoder.hop_length,
            encoder_transformer=encoder_transformer,
            decoder_transformer=decoder_transformer,
        )

        # Load mimi weights from the config safetensors file with complete mapping for strict loading

        if config.mimi.weights_path is not None:
            if config.flow_lm.weights_path is None:
                raise ValueError(
                    "If you specify mimi.weights_path you should specify flow_lm.weights_path"
                )
            logger.info(f"Loading Mimi weights from {config.mimi.weights_path}")
            mimi_state = get_mimi_state_dict(download_if_necessary(config.mimi.weights_path))
            tts_model.mimi.load_state_dict(mimi_state, strict=True)

        tts_model.mimi.eval()
        # tts_model.to(dtype=torch.float32)

        # uncomment to save the weights
        # tts_model = tts_model.to(dtype=torch.bfloat16)
        # safetensors.torch.save_file(tts_model.state_dict(), "tts_b6369a24.safetensors")
        if config.weights_path is not None:
            logger.info(f"Loading TTSModel weights from {config.weights_path}")
            try:
                weights_file = download_if_necessary(config.weights_path)
            except Exception as e:
                # Log the real cause before falling back - auth failure,
                # network error, and a moved/missing revision all land here
                # and look identical to the user otherwise.
                logger.warning(
                    f"Could not download voice-cloning weights ({config.weights_path}): "
                    f"{type(e).__name__}: {e}. Falling back to the no-voice-cloning model."
                )
                tts_model.has_voice_cloning = False
                weights_file = download_if_necessary(config.weights_path_without_voice_cloning)

            state_dict = safetensors.torch.load_file(weights_file)
            tts_model.load_state_dict(state_dict, strict=True)

        if config.flow_lm.weights_path is None and config.weights_path is None:
            logger.warning(
                "No weights_path specified for FlowLM or TTSModel, model is uninitialized!"
            )
        size_in_mb = size_of_dict(tts_model.state_dict()) // 1e6
        logging.info(f"TTS Model loaded successfully. Its size is {size_in_mb} MB")

        return tts_model

    def load_model(
        variant: str = DEFAULT_VARIANT,
        temp: float | int = DEFAULT_TEMPERATURE,
        lsd_decode_steps: int = DEFAULT_LSD_DECODE_STEPS,
        noise_clamp: float | int | None = DEFAULT_NOISE_CLAMP,
        eos_threshold: float = DEFAULT_EOS_THRESHOLD,
        device: str = "auto",
    ) -> Self:
        """Load a pre-trained TTS model with specified configuration.

        This class method loads a complete TTS model including the flow language model
        and Mimi compression model from pre-trained weights. The model is initialized
        with the specified generation parameters and ready for inference.

        Args:
            variant: Model variant identifier corresponding to a config file name
                (e.g., '610b0b2c'). Must match a YAML file in the config directory.
            temp: Sampling temperature for generation. Higher values produce more
                diverse but potentially lower quality output.
            lsd_decode_steps: Number of steps for Lagrangian Self Distillation
                decoding. More steps can improve quality but increase computation.
            noise_clamp: Maximum value for noise sampling. If None, no clamping
                is applied. Helps prevent extreme values in generation.
            eos_threshold: Threshold for end-of-sequence detection. Higher values
                make the model more likely to continue generating.
            device: Device to load the model onto. "auto" detects CUDA, "cuda"
                forces GPU, "cpu" forces CPU.

        Returns:
            TTSModel: Fully initialized model with loaded weights, ready for
                text-to-speech generation.

        Raises:
            FileNotFoundError: If the specified config file or model weights
                are not found.
            ValueError: If the configuration is invalid or incompatible.
        """
        config = load_config(Path(__file__).parents[1] / f"config/{variant}.yaml")
        tts_model = TTSModel._from_pydantic_config_with_weights(
            config, temp, lsd_decode_steps, noise_clamp, eos_threshold
        )

        # Resolve device
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = device

        if resolved_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            resolved_device = "cpu"

        # Move model to target device
        tts_model = tts_model.to(device=resolved_device)
        logger.info(f"TTS model loaded on {resolved_device}")

        # Set CUDA-specific optimizations
        if resolved_device == "cuda":
            os.environ["NO_CUDA_GRAPH"] = "1"
            tts_model.max_tokens_per_chunk = 80  # Larger chunks for GPU

            logger.info(f"CUDA optimizations enabled. GPU: {torch.cuda.get_device_name(0)}")

        return tts_model

    def _run_flow_lm_and_increment_step(
        self,
        model_state: dict,
        text_tokens: torch.Tensor | None = None,
        backbone_input_latents: torch.Tensor | None = None,
        audio_conditioning: torch.Tensor | None = None,
        temperature: float | torch.Tensor | None = None,
        eos_threshold: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """First one is the backbone output, second one is the audio decoding output."""
        if text_tokens is None:
            text_tokens = torch.zeros((1, 0), dtype=torch.int64, device=self.flow_lm.device)
        if backbone_input_latents is None:
            backbone_input_latents = torch.empty(
                (1, 0, self.flow_lm.ldim), dtype=self.flow_lm.dtype, device=self.flow_lm.device
            )
        if audio_conditioning is None:
            audio_conditioning = torch.empty(
                (1, 0, self.flow_lm.dim), dtype=self.flow_lm.dtype, device=self.flow_lm.device
            )

        output = self._run_flow_lm(
            text_tokens=text_tokens,
            backbone_input_latents=backbone_input_latents,
            model_state=model_state,
            audio_conditioning=audio_conditioning,
            temperature=temperature,
            eos_threshold=eos_threshold,
        )
        increment_by = (
            text_tokens.shape[1] + backbone_input_latents.shape[1] + audio_conditioning.shape[1]
        )
        increment_steps(self.flow_lm, model_state, increment=increment_by)
        return output

    def _run_flow_lm(
        self,
        model_state: dict,
        text_tokens: torch.Tensor,
        backbone_input_latents: torch.Tensor,
        audio_conditioning: torch.Tensor,
        temperature: float | torch.Tensor | None = None,
        eos_threshold: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs a language model flow to generate embeddings for input text and audio conditioning.
        Args:
        model_state (dict): Current state of the model.
        text_tokens (torch.Tensor): Tokens representing the input text.
        backbone_input_latents (torch.Tensor): Latent inputs from the backbone network.
        audio_conditioning (torch.Tensor): Conditioning information derived from audio data.
        Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the output embeddings and a boolean indicating if the end of sequence is reached.
        """
        text_embeddings = self.flow_lm.conditioner(TokenizedText(text_tokens))
        text_embeddings = torch.cat([text_embeddings, audio_conditioning.to(text_embeddings.device)], dim=1)

        output_embeddings, is_eos = self.flow_lm._sample_next_latent(
            backbone_input_latents,
            text_embeddings,
            model_state=model_state,
            lsd_decode_steps=self.lsd_decode_steps,
            temp=self.temp if temperature is None else temperature,
            noise_clamp=self.noise_clamp,
            eos_threshold=self.eos_threshold if eos_threshold is None else eos_threshold,
        )
        return output_embeddings[:, None, :], is_eos

    def _encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encodes an audio tensor to a latent representation and applies linear transformation.
        Args:
        audio (torch.Tensor): Input audio tensor.
        Returns:
        torch.Tensor: Transposed and transformed latent tensor.
        """
        target_device = next(iter(self.parameters())).device
        target_dtype = next(iter(self.parameters())).dtype
        encoded = self.mimi.encode_to_latent(audio.to(device=target_device, dtype=target_dtype))
        latents = encoded.transpose(-1, -2)
        conditioning = F.linear(latents, self.flow_lm.speaker_proj_weight.to(target_device))
        return conditioning

    @torch.no_grad
    def _decode_audio_worker(self, latents_queue: queue.Queue, result_queue: queue.Queue):
        """Worker thread function for decoding audio latents from queue with immediate streaming."""
        try:
            audio_chunks = []
            decoder_time = 0.0
            mimi_state = init_states(self.mimi, batch_size=1, sequence_length=1000, device=self.device)
            while True:
                latent = latents_queue.get()
                if latent is None:
                    break
                mimi_decoding_input = latent * self.flow_lm.emb_std + self.flow_lm.emb_mean
                transposed = mimi_decoding_input.transpose(-1, -2)
                quantized = self.mimi.quantizer(transposed)

                t = time.monotonic()
                audio_frame = self.mimi.decode_from_latent(quantized, mimi_state)
                decoder_time += time.monotonic() - t
                increment_steps(self.mimi, mimi_state, increment=16)
                audio_frame_duration = audio_frame.shape[2] / self.config.mimi.sample_rate
                # We could log the timings here.
                logger.debug(
                    " " * 30 + "Decoded %d ms of audio with mimi in %d ms",
                    int(audio_frame_duration * 1000),
                    int((time.monotonic() - t) * 1000),
                )
                audio_chunks.append(audio_frame)

                result_queue.put(("chunk", audio_frame))

                latents_queue.task_done()

            # Signal completion
            if hasattr(self, "_active_generation_timing"):
                self._active_generation_timing["decoder_ms"] = decoder_time * 1000
            result_queue.put(("done", None))

        except Exception as e:
            # Put error in result queue
            result_queue.put(("error", e))

    @torch.no_grad
    def generate_audio(
        self,
        model_state: dict,
        text_to_generate: str,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
    ) -> torch.Tensor:
        """Generate complete audio tensor from text input.

        This method generates the full audio output for the given text prompt
        and returns it as a single tensor. It internally uses the streaming
        generation method but collects all chunks before returning.

        This method is NOT thread-safe; separate model instances should be used
        for concurrent generation.

        Args:
            model_state: Model state dictionary containing hidden states and
                positional information. Can be obtained from get_state_for_audio_prompt()
                or init_states(). The state may be modified during generation.
            text_to_generate: Input text to convert to speech. The text will be
                automatically formatted (capitalization, punctuation) for optimal
                generation quality.
            frames_after_eos: Number of additional frames to generate after
                detecting end-of-sequence. If None, automatically determined
                based on text length (1-3 frames).
            copy_state: Whether to create a deep copy of the model state before
                generation. If True, preserves the original state for reuse.
                If False, modifies the input state in-place. Defaults to True.

        Returns:
            torch.Tensor: Generated audio tensor with shape [channels, samples]
                at the model's sample rate (typically 24kHz). The audio is
                normalized and ready for playback or saving.
                You can get the sample rate from the `sample_rate` attribute.

        Raises:
            ValueError: If text_to_generate is empty or invalid.
            RuntimeError: If generation fails due to model errors.
        """
        self._active_generation_timing = {
            "tokenization_ms": 0.0,
            "prompt_ms": 0.0,
            "autoregressive_ms": 0.0,
            "decoder_ms": 0.0,
        }
        audio_chunks = []
        try:
            for chunk in self.generate_audio_stream(
                model_state=model_state,
                text_to_generate=text_to_generate,
                frames_after_eos=frames_after_eos,
                copy_state=copy_state,
            ):
                audio_chunks.append(chunk)
        finally:
            self.last_generation_timing = dict(self._active_generation_timing)
            self._active_generation_timing = None
        if not audio_chunks:
            return torch.zeros(0, dtype=torch.float32)
        target_device = audio_chunks[0].device
        audio_chunks = [c.to(target_device) if c.device != target_device else c for c in audio_chunks]
        return torch.cat(audio_chunks, dim=0)

    @torch.no_grad
    def generate_audio_batch(
        self,
        model_state: dict,
        requests: list[dict[str, Any]],
    ) -> list[torch.Tensor]:
        """Generate exact-token-length requests together with per-item settings.

        Args:
            model_state: Batch-one voice-conditioned FlowLM state.
            requests: Requests containing text, temperature, EOS threshold,
                frames_after_eos, and lsd_decode_steps values.

        Returns:
            One trimmed mono audio tensor per request, in request order.

        Raises:
            ValueError: If requests have incompatible token lengths or settings.
        """
        if not requests:
            return []

        # Preserve caller-owned request text for chunk sidecars and ASR references.
        requests = [{**request, "text": build_tts_prompt(request["text"])} for request in requests]
        batch_size = len(requests)
        prepared_tokens = [
            self.flow_lm.conditioner.prepare(request["text"]).tokens.squeeze(0)
            for request in requests
        ]
        token_lengths = {tokens.shape[-1] for tokens in prepared_tokens}
        if len(token_lengths) != 1:
            raise ValueError("Batched requests require equal token lengths")

        shared_keys = ("frames_after_eos", "eos_threshold", "lsd_decode_steps")
        for key in shared_keys:
            if len({request[key] for request in requests}) != 1:
                raise ValueError(f"Batched requests require shared {key}")

        device = self.flow_lm.device
        dtype = self.flow_lm.dtype
        flow_state = _clone_flow_state_for_batch(model_state, batch_size)
        mimi_state = init_states(self.mimi, batch_size=batch_size, sequence_length=1000, device=device)
        text_tokens = torch.stack(prepared_tokens).to(device=device)
        empty_tokens = torch.empty((batch_size, 0), dtype=torch.int64, device=device)
        empty_conditioning = torch.empty(
            (batch_size, 0, self.flow_lm.dim), dtype=dtype, device=device
        )
        temperatures = torch.tensor(
            [float(request["temperature"]) for request in requests],
            dtype=dtype,
            device=device,
        )
        eos_thresholds = torch.tensor(
            [float(request["eos_threshold"]) for request in requests],
            dtype=dtype,
            device=device,
        )
        frames_after_eos = int(requests[0]["frames_after_eos"])
        max_gen_len = max(
            int((len(request["text"].split()) + 2.0) * 12.5)
            for request in requests
        )

        # Shared LSD for this batch (validated above)
        shared_lsd = int(requests[0]["lsd_decode_steps"])
        prev_lsd = self.lsd_decode_steps
        self.lsd_decode_steps = shared_lsd

        self._run_flow_lm_and_increment_step(
            model_state=flow_state,
            text_tokens=text_tokens,
            backbone_input_latents=torch.empty(
                (batch_size, 0, self.flow_lm.ldim), dtype=dtype, device=device
            ),
            audio_conditioning=empty_conditioning,
            temperature=temperatures,
            eos_threshold=eos_thresholds,
        )

        backbone_input = torch.full(
            (batch_size, 1, self.flow_lm.ldim),
            float("nan"),
            dtype=dtype,
            device=device,
        )
        # v2: optional CUDA graph for batched AR steps (reuse registry by B,lsd)
        batch_graph = None
        if self._should_use_cuda_graphs():
            batch_graph = self._get_or_create_ar_step_graph(
                flow_state,
                backbone_input,
                temperature=temperatures,
                eos_threshold=eos_thresholds,
                lsd_decode_steps=shared_lsd,
            )
            if batch_graph is not None:
                # AR mutates owned static state; rebind so eager fallback mid-loop stays consistent
                flow_state = batch_graph.static_state

        eos_steps = [-1] * batch_size
        audio_frames = []
        # Optional step split: FlowLM vs Mimi (POCKET_TTS_BATCH_TIMING=1)
        time_steps = os.environ.get("POCKET_TTS_BATCH_TIMING", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        # Dual-stream Mimi||Flow pipeline: measured ~0% on 4060 Ti (SM full); default off.
        pipeline_env = os.environ.get("POCKET_TTS_MIMI_PIPELINE", "0").strip().lower()
        use_mimi_pipeline = (
            pipeline_env in ("1", "true", "yes", "on")
            and not time_steps
            and str(device).startswith("cuda")
            and torch.cuda.is_available()
        )
        # Mimi CUDA graph (default on when FlowLM graphs enabled, or POCKET_TTS_MIMI_GRAPH=1)
        mimi_graph_env = os.environ.get("POCKET_TTS_MIMI_GRAPH", "").strip().lower()
        if mimi_graph_env in ("0", "false", "no", "off"):
            want_mimi_graph = False
        elif mimi_graph_env in ("1", "true", "yes", "on"):
            want_mimi_graph = True
        else:
            want_mimi_graph = self._should_use_cuda_graphs()
        mimi_graph = None
        if want_mimi_graph and str(device).startswith("cuda") and torch.cuda.is_available():
            mimi_graph = self._get_or_create_mimi_decode_graph(
                mimi_state,
                torch.empty((batch_size, 1, self.flow_lm.ldim), device=device, dtype=dtype),
            )
            if mimi_graph is not None:
                mimi_state = mimi_graph.static_state
        flow_ms_total = 0.0
        mimi_ms_total = 0.0
        n_timed_steps = 0
        use_cuda_sync = time_steps and str(device).startswith("cuda") and torch.cuda.is_available()

        def _one_flow_step(bb_in: torch.Tensor):
            """One FlowLM AR step (graph or eager); updates cuda_graph_stats."""
            nonlocal batch_graph
            if batch_graph is not None and batch_graph.available:
                try:
                    lat, eos = batch_graph.step(
                        bb_in,
                        temperature=temperatures,
                        eos_threshold=eos_thresholds,
                    )
                    self.cuda_graph_stats["steps_graph"] = (
                        int(self.cuda_graph_stats.get("steps_graph", 0)) + 1
                    )
                    return lat, eos
                except Exception as exc:
                    logger.warning(
                        "Batch CUDA graph step failed; falling back to eager: %s", exc
                    )
                    batch_graph = None
            lat, eos = self._run_flow_lm_and_increment_step(
                model_state=flow_state,
                text_tokens=empty_tokens,
                backbone_input_latents=bb_in,
                audio_conditioning=empty_conditioning,
                temperature=temperatures,
                eos_threshold=eos_thresholds,
            )
            self.cuda_graph_stats["steps_eager"] = (
                int(self.cuda_graph_stats.get("steps_eager", 0)) + 1
            )
            return lat, eos

        def _one_mimi_step(latent: torch.Tensor) -> torch.Tensor:
            """Quantize + decode one AR latent frame through Mimi; advance mimi_state."""
            if mimi_graph is not None and mimi_graph.available:
                try:
                    frame = mimi_graph.step(latent)
                    self.cuda_graph_stats["mimi_steps_graph"] = (
                        int(self.cuda_graph_stats.get("mimi_steps_graph", 0)) + 1
                    )
                    return frame
                except Exception as exc:
                    logger.warning("Mimi CUDA graph step failed; eager decode: %s", exc)
            mimi_input = latent * self.flow_lm.emb_std + self.flow_lm.emb_mean
            quantized = self.mimi.quantizer(mimi_input.transpose(-1, -2))
            frame = self.mimi.decode_from_latent(quantized, mimi_state)
            increment_steps(self.mimi, mimi_state, increment=16)
            self.cuda_graph_stats["mimi_steps_eager"] = (
                int(self.cuda_graph_stats.get("mimi_steps_eager", 0)) + 1
            )
            return frame

        def _record_eos(is_eos: torch.Tensor, generation_step: int) -> None:
            """Mark first EOS step per batch item."""
            for index, value in enumerate(is_eos.reshape(-1).bool().tolist()):
                if value and eos_steps[index] < 0:
                    eos_steps[index] = generation_step

        def _should_stop(generation_step: int) -> bool:
            """True when all items have finished frames_after_eos after their EOS."""
            if not all(step >= 0 for step in eos_steps):
                return False
            return generation_step >= max(eos_steps) + frames_after_eos - 1

        try:
            if use_mimi_pipeline:
                # Dual-stream: Flow(i) on default stream, then Mimi(i) on side stream
                # while Flow(i+1) runs. Same mimi stream orders Mimi steps vs each other.
                mimi_stream = torch.cuda.Stream(device=device)
                default_stream = torch.cuda.current_stream(device=device)
                steps_done = 0
                for generation_step in range(max_gen_len):
                    # Flow(i) — can overlap prior Mimi(i-1) still running on mimi_stream.
                    next_latent, is_eos = _one_flow_step(backbone_input)
                    _record_eos(is_eos, generation_step)
                    # Mimi(i): wait for this step's latent, then decode (ordered after Mimi(i-1)).
                    with torch.cuda.stream(mimi_stream):
                        mimi_stream.wait_stream(default_stream)
                        audio_frames.append(_one_mimi_step(next_latent))
                    backbone_input = next_latent
                    steps_done = generation_step + 1
                    if _should_stop(generation_step):
                        break
                # Drain mimi before reading frames / concat
                default_stream.wait_stream(mimi_stream)
                torch.cuda.synchronize(device=device)
                self.last_batch_step_timing = {
                    "batch_size": batch_size,
                    "pipeline": True,
                    "used_graph": batch_graph is not None and batch_graph.available,
                    "steps": steps_done,
                }
            else:
                for generation_step in range(max_gen_len):
                    if time_steps:
                        if use_cuda_sync:
                            torch.cuda.synchronize()
                        t_flow0 = time.perf_counter()
                    next_latent, is_eos = _one_flow_step(backbone_input)
                    if time_steps:
                        if use_cuda_sync:
                            torch.cuda.synchronize()
                        flow_ms_total += (time.perf_counter() - t_flow0) * 1000.0
                        t_mimi0 = time.perf_counter()

                    _record_eos(is_eos, generation_step)
                    audio_frames.append(_one_mimi_step(next_latent))
                    backbone_input = next_latent

                    if time_steps:
                        if use_cuda_sync:
                            torch.cuda.synchronize()
                        mimi_ms_total += (time.perf_counter() - t_mimi0) * 1000.0
                        n_timed_steps += 1

                    if _should_stop(generation_step):
                        break
        finally:
            self.lsd_decode_steps = prev_lsd
            if time_steps and n_timed_steps > 0:
                total_ms = flow_ms_total + mimi_ms_total
                self.last_batch_step_timing = {
                    "batch_size": batch_size,
                    "steps": n_timed_steps,
                    "flow_ms": flow_ms_total,
                    "mimi_ms": mimi_ms_total,
                    "total_ms": total_ms,
                    "flow_pct": 100.0 * flow_ms_total / total_ms if total_ms > 0 else 0.0,
                    "mimi_pct": 100.0 * mimi_ms_total / total_ms if total_ms > 0 else 0.0,
                    "flow_ms_per_step": flow_ms_total / n_timed_steps,
                    "mimi_ms_per_step": mimi_ms_total / n_timed_steps,
                    "used_graph": batch_graph is not None and batch_graph.available,
                    "pipeline": False,
                }
                logger.info(
                    "Batch AR step split B=%s steps=%s flow=%.1fms (%.0f%%) mimi=%.1fms (%.0f%%) "
                    "per_step flow=%.2f mimi=%.2f graph=%s",
                    batch_size,
                    n_timed_steps,
                    flow_ms_total,
                    self.last_batch_step_timing["flow_pct"],
                    mimi_ms_total,
                    self.last_batch_step_timing["mimi_pct"],
                    self.last_batch_step_timing["flow_ms_per_step"],
                    self.last_batch_step_timing["mimi_ms_per_step"],
                    self.last_batch_step_timing["used_graph"],
                )
            elif not use_mimi_pipeline:
                self.last_batch_step_timing = {}
        if not audio_frames:
            return [torch.zeros(0, device=device) for _ in requests]

        audio_batch = torch.cat(audio_frames, dim=-1)
        frame_samples = audio_batch.shape[-1] // len(audio_frames)
        outputs = []
        for index, eos_step in enumerate(eos_steps):
            frame_count = (
                eos_step + frames_after_eos
                if eos_step >= 0
                else len(audio_frames)
            )
            outputs.append(audio_batch[index, 0, : frame_count * frame_samples])
        return outputs

    @torch.no_grad
    def generate_audio_stream(
        self,
        model_state: dict,
        text_to_generate: str,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
    ):
        """Generate audio streaming chunks from text input.

        This method generates audio from text and yields chunks as they become
        available, enabling real-time playback or processing. It uses multithreading
        to parallelize generation and decoding for optimal performance.
        This method is NOT thread-safe; separate model instances should be used
        for concurrent generation.

        Args:
            model_state: Model state dictionary containing hidden states and
                positional information. Can be obtained from get_state_for_audio_prompt()
                or init_states(). The state may be modified during generation.
            text_to_generate: Input text to convert to speech. The text will be
                automatically formatted (capitalization, punctuation) for optimal
                generation quality.
            frames_after_eos: Number of additional frames to generate after
                detecting end-of-sequence. If None, automatically determined
                based on text length (1-3 frames). Defaults to None.
            copy_state: Whether to create a deep copy of the model state before
                generation. If True, preserves the original state for reuse.
                If False, modifies the input state in-place. Defaults to True.

        Yields:
            torch.Tensor: Audio chunks with shape [samples] at the model's
                sample rate (typically 24kHz). Chunks are yielded as soon as
                they are decoded, enabling real-time streaming.

        Raises:
            ValueError: If text_to_generate is empty or invalid.
            RuntimeError: If generation fails due to model errors or threading issues.

        Note:
            This method uses multithreading to parallelize latent generation
            and audio decoding. Generation performance is logged including
            real-time factor (RTF) metrics.
        """

        # This is a very simplistic way of handling long texts. We could do much better
        # by using teacher forcing, but it would be a bit slower.
        # TODO: add the teacher forcing method for long texts where we use the audio of one chunk
        # as conditioning for the next chunk.
        tokenization_start = time.monotonic()
        chunks = split_into_best_sentences(self.flow_lm.conditioner.tokenizer, text_to_generate, self.max_tokens_per_chunk)
        if hasattr(self, "_active_generation_timing") and self._active_generation_timing is not None:
            self._active_generation_timing["tokenization_ms"] += (
                time.monotonic() - tokenization_start
            ) * 1000

        for chunk in chunks:
            prompt_text, frames_after_eos_guess = prepare_text_prompt(chunk)
            frames_after_eos_guess += 2
            yield from self._generate_audio_stream_short_text(
                model_state=model_state,
                text_to_generate=prompt_text,
                frames_after_eos=frames_after_eos_guess,
                copy_state=copy_state,
            )

    @torch.no_grad
    def _generate_audio_stream_short_text(
        self, model_state: dict, text_to_generate: str, frames_after_eos: int, copy_state: bool
    ):
        """Generates an audio stream for short text using a model state.
        Args:
        model_state (dict): The current state of the model.
        text_to_generate (str): The text to generate audio for.
        frames_after_eos (int): Number of frames to process after end-of-sentence.
        copy_state (bool): Whether to copy the model state before modifying it.
        Returns:
        None
        """
        if copy_state:
            model_state = copy.deepcopy(model_state)

        # Set up multithreaded generation and decoding
        latents_queue = queue.Queue()
        result_queue = queue.Queue()

        # Start decoder worker thread
        decoder_thread = threading.Thread(
            target=self._decode_audio_worker, args=(latents_queue, result_queue), daemon=True
        )
        logger.info("starting timer now!")
        t_generating = time.monotonic()
        decoder_thread.start()

        # Generate latents and add them to queue (decoder processes them in parallel)
        self._generate(
            model_state=model_state,
            text_to_generate=text_to_generate,
            frames_after_eos=frames_after_eos,
            latents_queue=latents_queue,
            result_queue=result_queue,
        )

        # Stream audio chunks as they become available
        total_generated_samples = 0
        while True:
            result = result_queue.get()
            if result[0] == "chunk":
                # Audio chunk available immediately for streaming/playback
                audio_chunk = result[1]
                total_generated_samples += audio_chunk.shape[-1]
                yield audio_chunk[0, 0]  # Remove batch, channel
            elif result[0] == "done":
                # Generation complete
                break
            elif result[0] == "error":
                # Wait for decoder thread to finish cleanly before propagating error
                with display_execution_time("Waiting for mimi decoder to finish"):
                    decoder_thread.join()
                # Propagate error
                raise result[1]

        # Wait for decoder thread to finish cleanly
        with display_execution_time("Waiting for mimi decoder to finish"):
            decoder_thread.join()

        # Print timing information
        duration_generated_audio = int(
            total_generated_samples * 1000 / self.config.mimi.sample_rate
        )
        generation_time = int((time.monotonic() - t_generating) * 1000)
        real_time_factor = duration_generated_audio / generation_time

        logger.info(
            "Generated: %d ms of audio in %d ms so %.2fx faster than real-time",
            duration_generated_audio,
            generation_time,
            real_time_factor,
        )

    @torch.no_grad
    def _generate(
        self,
        model_state: dict,
        text_to_generate: str,
        frames_after_eos: int,
        latents_queue: queue.Queue,
        result_queue: queue.Queue,
    ):
        """Generates a response based on the given text and state.
        Args:
        model_state (dict): The current state of the model.
        text_to_generate (str): The text to generate a response for.
        frames_after_eos (int): Number of frames after end-of-sequence.
        latents_queue (queue.Queue): Queue for latent data.
        result_queue (queue.Queue): Queue for results.
        Returns:
        None
        """
        gen_len_sec = len(text_to_generate.split()) * 1 + 2.0
        max_gen_len = int(gen_len_sec * 12.5)
        tokenization_start = time.monotonic()
        prepared = self.flow_lm.conditioner.prepare(text_to_generate)
        if hasattr(self, "_active_generation_timing") and self._active_generation_timing is not None:
            self._active_generation_timing["tokenization_ms"] += (
                time.monotonic() - tokenization_start
            ) * 1000

        prompt_start = time.monotonic()
        with display_execution_time("Prompting text"):
            self._run_flow_lm_and_increment_step(
                model_state=model_state, text_tokens=prepared.tokens
            )
        if hasattr(self, "_active_generation_timing") and self._active_generation_timing is not None:
            self._active_generation_timing["prompt_ms"] += (
                time.monotonic() - prompt_start
            ) * 1000

        def run_generation():
            """Runs autoregressive generation in a separate thread, handling exceptions by logging and signaling the decoder to stop. Returns None.
            Args:
            model_state: Current state of the model.
            max_gen_len: Maximum length of the generated sequence.
            frames_after_eos: Frames to generate after the end of sequence token.
            latents_queue: Queue for latent codes.
            """
            try:
                self._autoregressive_generation(
                    model_state, max_gen_len, frames_after_eos, latents_queue
                )
            except Exception as e:
                logger.error(f"Error in autoregressive generation: {e}")
                # Signal decoder to stop by putting None (completion sentinel)
                if latents_queue is not None:
                    latents_queue.put(None)
                # Report error to main thread
                if result_queue is not None:
                    result_queue.put(("error", e))

        generation_thread = threading.Thread(target=run_generation, daemon=True)
        generation_thread.start()

    @staticmethod
    def _new_cuda_graph_stats() -> dict:
        """Empty CUDA graph telemetry counters for one model instance."""
        return {
            "enabled": False,
            "captures_ok": 0,
            "captures_fail": 0,
            "prepares": 0,
            "reuses": 0,
            "steps_graph": 0,
            "steps_eager": 0,
            "last_capture_ok": None,
            "last_capture_error": None,
            "last_batch_size": None,
            "registry_size": 0,
            "mimi_captures_ok": 0,
            "mimi_captures_fail": 0,
            "mimi_steps_graph": 0,
            "mimi_steps_eager": 0,
        }

    def reset_cuda_graph_stats(self) -> None:
        """Clear CUDA graph counters (e.g. start of a worker session)."""
        self.cuda_graph_stats = self._new_cuda_graph_stats()

    def get_cuda_graph_stats(self) -> dict:
        """Return a copy of CUDA graph capture/step counters (FlowLM + Mimi)."""
        stats = dict(getattr(self, "cuda_graph_stats", self._new_cuda_graph_stats()))
        stats["enabled"] = self._should_use_cuda_graphs()
        graphs = getattr(self, "_flow_lm_step_graphs", {}) or {}
        stats["registry_size"] = len(graphs)
        mimi_graphs = getattr(self, "_mimi_decode_graphs", {}) or {}
        stats["mimi_registry_size"] = len(mimi_graphs)
        total = int(stats.get("steps_graph", 0)) + int(stats.get("steps_eager", 0))
        stats["steps_total"] = total
        if total > 0:
            stats["steps_graph_pct"] = 100.0 * float(stats["steps_graph"]) / float(total)
        else:
            stats["steps_graph_pct"] = 0.0
        mimi_total = int(stats.get("mimi_steps_graph", 0)) + int(
            stats.get("mimi_steps_eager", 0)
        )
        stats["mimi_steps_total"] = mimi_total
        if mimi_total > 0:
            stats["mimi_steps_graph_pct"] = (
                100.0 * float(stats.get("mimi_steps_graph", 0)) / float(mimi_total)
            )
        else:
            stats["mimi_steps_graph_pct"] = 0.0
        return stats

    def _should_use_cuda_graphs(self) -> bool:
        """Return True when CUDA graph AR steps are enabled and device is CUDA."""
        env_on = os.environ.get("POCKET_TTS_CUDA_GRAPHS", "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        enabled = bool(getattr(self, "cuda_graphs_enabled", False) or env_on)
        if not enabled:
            return False
        try:
            return str(self.device).startswith("cuda") and torch.cuda.is_available()
        except Exception:
            return False

    def _record_cuda_graph_capture(
        self, ok: bool, batch_size: int, error: str | None = None
    ) -> None:
        """Update capture telemetry after a graph capture attempt."""
        stats = self.cuda_graph_stats
        stats["enabled"] = True
        stats["last_batch_size"] = int(batch_size)
        stats["last_capture_ok"] = bool(ok)
        stats["last_capture_error"] = error
        if ok:
            stats["captures_ok"] = int(stats.get("captures_ok", 0)) + 1
            logger.info(
                "CUDA graph capture OK (batch_size=%s, captures_ok=%s)",
                batch_size,
                stats["captures_ok"],
            )
        else:
            stats["captures_fail"] = int(stats.get("captures_fail", 0)) + 1
            logger.warning(
                "CUDA graph capture FAILED (batch_size=%s): %s",
                batch_size,
                error or "unknown",
            )

    def _graph_registry_key(
        self, batch_size: int, lsd_decode_steps: int | None = None
    ) -> tuple[int, int]:
        """Registry key for a reusable AR step graph."""
        lsd = int(
            lsd_decode_steps if lsd_decode_steps is not None else self.lsd_decode_steps
        )
        return (int(batch_size), lsd)

    def _get_or_create_ar_step_graph(
        self,
        model_state: dict,
        backbone_input: torch.Tensor,
        temperature: float | torch.Tensor | None = None,
        eos_threshold: float | torch.Tensor | None = None,
        lsd_decode_steps: int | None = None,
    ):
        """Return a prepared FlowLM AR graph for this unit, capturing only if needed.

        Reuses a prior capture for the same (batch_size, lsd_decode_steps) by
        copying post-prompt state into the graph's static buffers (prepare).

        Args:
            model_state: Live post-prompt FlowLM state for this generation unit.
            backbone_input: Template/current backbone [B, 1, ldim].
            temperature: Optional temp for capture defaults / first static fill.
            eos_threshold: Optional eos threshold for capture defaults.
            lsd_decode_steps: LSD steps baked into the graph (defaults to model).

        Returns:
            Prepared FlowLMStepGraph, or None if capture/prepare failed.
        """
        from pocket_tts.models.cuda_graph_flow_lm import FlowLMStepGraph

        batch_size = int(backbone_input.shape[0]) if backbone_input is not None else 1
        lsd = int(
            lsd_decode_steps if lsd_decode_steps is not None else self.lsd_decode_steps
        )
        key = self._graph_registry_key(batch_size, lsd)
        if not hasattr(self, "_flow_lm_step_graphs") or self._flow_lm_step_graphs is None:
            self._flow_lm_step_graphs = {}

        runner = self._flow_lm_step_graphs.get(key)
        if runner is not None and runner.available:
            try:
                runner.prepare(model_state)
                self.cuda_graph_stats["prepares"] = (
                    int(self.cuda_graph_stats.get("prepares", 0)) + 1
                )
                self.cuda_graph_stats["reuses"] = (
                    int(self.cuda_graph_stats.get("reuses", 0)) + 1
                )
                self.cuda_graph_stats["last_batch_size"] = batch_size
                logger.debug(
                    "CUDA graph reuse (B=%s, lsd=%s, reuses=%s)",
                    batch_size,
                    lsd,
                    self.cuda_graph_stats["reuses"],
                )
                return runner
            except Exception as exc:
                logger.warning(
                    "CUDA graph prepare failed (B=%s, lsd=%s); re-capturing: %s",
                    batch_size,
                    lsd,
                    exc,
                )
                self._flow_lm_step_graphs.pop(key, None)

        runner = FlowLMStepGraph(self, batch_size=batch_size)
        if runner.capture(
            model_state,
            backbone_input,
            temperature=temperature,
            eos_threshold=eos_threshold,
            lsd_decode_steps=lsd,
        ):
            # Capture leaves static at post-prompt of this unit — ready for AR.
            # Count as prepare=0 (first use after capture); next unit will prepare.
            self._flow_lm_step_graphs[key] = runner
            self._record_cuda_graph_capture(True, batch_size)
            return runner
        self._record_cuda_graph_capture(
            False, batch_size, getattr(runner, "_capture_error", None)
        )
        return None

    def _ensure_ar_step_graph(self, model_state: dict, backbone_input: torch.Tensor):
        """Get or capture a FlowLM AR step graph for scalar generation (B from backbone)."""
        return self._get_or_create_ar_step_graph(model_state, backbone_input)

    def _get_or_create_mimi_decode_graph(
        self,
        mimi_state: dict,
        latent_template: torch.Tensor,
    ):
        """Return prepared Mimi decode graph for this batch size, capturing if needed.

        Args:
            mimi_state: Live Mimi state at generation start (usually fresh init).
            latent_template: Shape [B, 1, ldim] for static buffer allocation.

        Returns:
            Prepared MimiDecodeGraph, or None on failure.
        """
        from pocket_tts.models.cuda_graph_mimi import MimiDecodeGraph

        batch_size = int(latent_template.shape[0])
        if not hasattr(self, "_mimi_decode_graphs") or self._mimi_decode_graphs is None:
            self._mimi_decode_graphs = {}

        runner = self._mimi_decode_graphs.get(batch_size)
        if runner is not None and runner.available:
            try:
                runner.prepare(mimi_state)
                return runner
            except Exception as exc:
                logger.warning(
                    "Mimi graph prepare failed (B=%s); re-capturing: %s", batch_size, exc
                )
                self._mimi_decode_graphs.pop(batch_size, None)

        runner = MimiDecodeGraph(self, batch_size=batch_size)
        if runner.capture(mimi_state, latent_template):
            self._mimi_decode_graphs[batch_size] = runner
            self.cuda_graph_stats["mimi_captures_ok"] = (
                int(self.cuda_graph_stats.get("mimi_captures_ok", 0)) + 1
            )
            logger.info(
                "Mimi graph capture OK (batch_size=%s, captures_ok=%s)",
                batch_size,
                self.cuda_graph_stats["mimi_captures_ok"],
            )
            return runner
        self.cuda_graph_stats["mimi_captures_fail"] = (
            int(self.cuda_graph_stats.get("mimi_captures_fail", 0)) + 1
        )
        return None

    def _ar_step(
        self,
        model_state: dict,
        backbone_input: torch.Tensor,
        step_graph,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One AR step via CUDA graph when available, else eager _run_flow_lm_and_increment_step."""
        if step_graph is not None and step_graph.available:
            try:
                out = step_graph.step(backbone_input)
                self.cuda_graph_stats["steps_graph"] = (
                    int(self.cuda_graph_stats.get("steps_graph", 0)) + 1
                )
                return out
            except Exception as exc:
                logger.warning("CUDA graph AR step failed; falling back to eager: %s", exc)
        self.cuda_graph_stats["steps_eager"] = (
            int(self.cuda_graph_stats.get("steps_eager", 0)) + 1
        )
        return self._run_flow_lm_and_increment_step(
            model_state=model_state, backbone_input_latents=backbone_input
        )

    @torch.no_grad
    def _autoregressive_generation(
        self, model_state: dict, max_gen_len: int, frames_after_eos: int, latents_queue: queue.Queue
    ):
        """Generate latents using an autoregressive process.
        Args:
        model_state (dict): Current state of the model.
        max_gen_len (int): Maximum length of generation.
        frames_after_eos (int): Number of frames after EOS to generate.
        latents_queue (queue.Queue): Queue for storing generated latents.
        Returns:
        None
        """
        backbone_input = torch.full(
            (1, 1, self.flow_lm.ldim),
            fill_value=float("NaN"),
            device=next(iter(self.flow_lm.parameters())).device,
            dtype=self.flow_lm.dtype,
        )
        steps_times = []
        eos_step = None
        generation_start = time.monotonic()
        # Default: plain monotonic timing (Item #2 cut). Set POCKET_TTS_AR_TIMER=context
        # to restore per-step display_execution_time for multi-worker A/B tests.
        use_context_timer = os.environ.get("POCKET_TTS_AR_TIMER", "monotonic").lower() == "context"
        use_graphs = self._should_use_cuda_graphs()
        step_graph = None
        if use_graphs:
            step_graph = self._ensure_ar_step_graph(model_state, backbone_input)
            if step_graph is not None:
                # Graph owns static buffers; rebind so mid-loop eager fallback stays consistent
                model_state = step_graph.static_state

        for generation_step in range(max_gen_len):
            if use_context_timer:
                with display_execution_time("Generating latent", print_output=False) as timer:
                    next_latent, is_eos = self._ar_step(
                        model_state, backbone_input, step_graph
                    )
                    if is_eos.item() and eos_step is None:
                        eos_step = generation_step
                    if eos_step is not None and generation_step >= eos_step + frames_after_eos:
                        break
                    latents_queue.put(next_latent)
                    backbone_input = next_latent
                steps_times.append(timer.elapsed_time_ms)
            else:
                step_start = time.monotonic()
                next_latent, is_eos = self._ar_step(
                    model_state, backbone_input, step_graph
                )
                if is_eos.item() and eos_step is None:
                    eos_step = generation_step
                if eos_step is not None and generation_step >= eos_step + frames_after_eos:
                    break
                latents_queue.put(next_latent)
                backbone_input = next_latent
                steps_times.append(int((time.monotonic() - step_start) * 1000))
        else:
            self.last_generation_reached_limit = True
            if os.environ.get("KPOCKET_TTS_ERROR_WITHOUT_EOS", "0") == "1":
                raise RuntimeError("Generation reached maximum length without EOS!")
            logger.warning(
                "Maximum generation length reached without EOS, this very often indicates an error."
            )

        self.last_generation_eos_step = eos_step
        self.last_generation_max_len = max_gen_len
        if eos_step is not None:
            self.last_generation_reached_limit = False

        # Add sentinel value to signal end of generation
        latents_queue.put(None)
        if hasattr(self, "_active_generation_timing") and self._active_generation_timing is not None:
            self._active_generation_timing["autoregressive_ms"] += (
                time.monotonic() - generation_start
            ) * 1000
        logger.info("Average generation step time: %d ms", int(statistics.mean(steps_times)))

    @lru_cache(maxsize=2)
    def _cached_get_state_for_audio_prompt(
        self, audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
    ) -> dict:
        """Create model state conditioned on audio prompt for continuation.
        Args:
        audio_conditioning (Path | str | torch.Tensor): Audio input used to condition the model state.
        truncate (bool, optional): If True, truncate the audio input before processing. Default is False.
        Returns:
        dict: Dictionary containing the model state with acoustic characteristics of the audio prompt.
        """
        return self.get_state_for_audio_prompt(audio_conditioning, truncate)

    @torch.no_grad
    def get_state_for_audio_prompt(
        self, audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
    ) -> dict:
        """Create model state conditioned on audio prompt for continuation.

        This method processes an audio prompt and creates a model state that
        captures the acoustic characteristics (speaker voice, style, prosody)
        for use in subsequent text-to-speech generation. The resulting state
        enables voice cloning and audio continuation with speaker consistency.

        Args:
            audio_conditioning: Audio prompt to condition on. Can be:
                - Path: Local file path to audio file
                - str: URL to download audio file from
                - torch.Tensor: Pre-loaded audio tensor with shape [channels, samples]
            truncate: Whether to truncate long audio prompts to 30 seconds.
                Helps prevent memory issues with very long inputs. Defaults to False.

        Returns:
            dict: Model state dictionary containing hidden states and positional
                information conditioned on the audio prompt. This state can be
                passed to `generate_audio()` or `generate_audio_stream()` for
                voice-consistent generation.

        Raises:
            FileNotFoundError: If audio file path doesn't exist.
            ValueError: If audio tensor is invalid or empty.
            RuntimeError: If audio processing or encoding fails.

        Note:
            - Audio is automatically resampled to the model's sample rate (24kHz)
            - The audio is encoded using the Mimi compression model and projected
              to the flow model's latent space
            - Processing time is logged for performance monitoring
            - The state preserves speaker characteristics for voice cloning
        """
        if isinstance(audio_conditioning, str) and audio_conditioning in PREDEFINED_VOICES:
            # We get the audio conditioning directly from the safetensors file.
            prompt = load_predefined_voice(audio_conditioning)
        else:
            if not self.has_voice_cloning and isinstance(audio_conditioning, (str, Path)):
                raise ValueError(
                    f"We could not download the weights for the model with voice cloning, "
                    f"but you're trying to use voice cloning. "
                    f"Without voice cloning, you can use our catalog of voices {list(PREDEFINED_VOICES)}. "
                    f"If you want access to the model with voice cloning, go to "
                    f"https://huggingface.co/kyutai/pocket-tts and accept the terms, "
                    f"then make sure you're logged in locally with `uvx hf auth login`."
                )
            if isinstance(audio_conditioning, str):
                audio_conditioning = download_if_necessary(audio_conditioning)

            if isinstance(audio_conditioning, Path):
                audio, conditioning_sample_rate = audio_read(audio_conditioning)

                if truncate:
                    max_samples = int(30 * conditioning_sample_rate)  # 30 seconds of audio
                    if audio.shape[-1] > max_samples:
                        audio = audio[..., :max_samples]
                        logger.info(f"Audio truncated to first 30 seconds ({max_samples} samples)")

                audio_conditioning = convert_audio(
                    audio, conditioning_sample_rate, self.config.mimi.sample_rate, 1
                )

            with display_execution_time("Encoding audio prompt"):
                prompt = self._encode_audio(audio_conditioning.unsqueeze(0).to(self.device))
                # import safetensors.torch
                # safetensors.torch.save_file(
                #     {"audio_prompt": prompt},
                #     "/projects/huggingface/pocket-tts/embeddings/cosette.safetensors"
                # )

        model_state = init_states(self.flow_lm, batch_size=1, sequence_length=1000, device=self.flow_lm.device)

        with display_execution_time("Prompting audio"):
            self._run_flow_lm_and_increment_step(model_state=model_state, audio_conditioning=prompt)

        return model_state


def prepare_text_prompt(text: str) -> tuple[str, int]:
    """Build the shared model prompt and select an EOS-frame estimate.

    Args:
        text: Authored text or a pause-delimited speech segment.

    Returns:
        The model-only prompt and its frames-after-EOS estimate.
    """
    text = build_tts_prompt(text)
    if text == "":
        raise ValueError("Text prompt cannot be empty")
    number_of_words = len(text.split())
    if number_of_words <= 4:
        frames_after_eos_guess = 3
    else:
        frames_after_eos_guess = 1
    return text, frames_after_eos_guess


def split_into_best_sentences(tokenizer, text_to_generate: str, max_nb_tokens_in_a_chunk: int = 50) -> list[str]:
    """Splits input text into a list of sentences using a tokenizer.
    Args:
    tokenizer: A tokenizer instance used to tokenize the text.
    text_to_generate: The text to be split into sentences.
    max_nb_tokens_in_a_chunk: Maximum number of tokens per chunk (increase for GPU).
    Returns:
    A list of sentences extracted from the input text.
    """
    text_to_generate, _ = prepare_text_prompt(text_to_generate)
    text_to_generate = text_to_generate.strip()
    tokens = tokenizer(text_to_generate)
    list_of_tokens = tokens.tokens[0].tolist()

    _, *end_of_sentence_tokens = tokenizer(".!...?").tokens[0].tolist()

    end_of_sentences_indices = [0]
    previous_was_end_of_sentence_token = False

    for token_idx, token in enumerate(list_of_tokens):
        if token in end_of_sentence_tokens:
            previous_was_end_of_sentence_token = True
        else:
            if previous_was_end_of_sentence_token:
                end_of_sentences_indices.append(token_idx)
            previous_was_end_of_sentence_token = False
    end_of_sentences_indices.append(len(list_of_tokens))

    nb_tokens_and_sentences = []
    for i in range(len(end_of_sentences_indices) - 1):
        # let's print
        start = end_of_sentences_indices[i]
        end = end_of_sentences_indices[i + 1]
        text = tokenizer.sp.decode(list_of_tokens[start:end])
        nb_tokens_and_sentences.append((end - start, text))

    max_nb_tokens_in_a_chunk = max_nb_tokens_in_a_chunk
    chunks = []
    current_chunk = ""
    current_nb_of_tokens_in_chunk = 0
    for nb_tokens, sentence in nb_tokens_and_sentences:
        if current_chunk == "":
            current_chunk = sentence
            current_nb_of_tokens_in_chunk = nb_tokens
            continue

        if current_nb_of_tokens_in_chunk + nb_tokens > max_nb_tokens_in_a_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_nb_of_tokens_in_chunk = nb_tokens
        else:
            current_chunk += " " + sentence
            current_nb_of_tokens_in_chunk += nb_tokens

    if current_chunk != "":
        chunks.append(current_chunk.strip())

    return chunks
