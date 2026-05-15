import os
import re
import tempfile
import gc
from dataclasses import dataclass
from pathlib import Path

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torch
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    sway_sampling_coef,
    target_rms,
    tempfile_kwargs,
)


INFER_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET_AUDIO = INFER_DIR / "examples/wavs/14_208_000042_000000.wav"
DEFAULT_SOURCE_AUDIO = INFER_DIR / "examples/wavs/237_134493_000015_000004.wav"
DEFAULT_CONFIG = INFER_DIR.parent / "configs/F5TTS_Base_EZ-VC.yaml"
DEFAULT_CKPT = "hf://SPRINGLab/EZ-VC/model_2700000.safetensors"
DEFAULT_VOCAB = "hf://SPRINGLab/EZ-VC/vocab.txt"
DEFAULT_OUTPUT = Path(tempfile.gettempdir()) / "ezvc_gradio_output.wav"
DEFAULT_NFE_STEPS = 32
DEFAULT_SPEED = 1.0
DEFAULT_CFG_STRENGTH = 2.0
DEFAULT_SWAY = sway_sampling_coef
DEFAULT_CROSS_FADE = 0.0
DEFAULT_TARGET_RMS = target_rms
DEFAULT_SEED = 0
DEFAULT_OUTPUT_DURATION = 0.0
DEFAULT_MATCH_SOURCE_DURATION = True


@dataclass
class EZVCState:
    device: str
    xeus_model: object
    apply_kmeans: object
    vocoder: object
    ema_model: object


_state: EZVCState | None = None


def _release_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def unload_models() -> str:
    global _state
    _state = None
    _release_torch_memory()
    return "Models unloaded."


def _select_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def _resolve_cached_path(path: str) -> str:
    if path.startswith("hf://"):
        return str(cached_path(path))
    return path


def _load_state(device_choice: str) -> EZVCState:
    global _state
    device = _select_device(device_choice)
    if _state is not None and _state.device == device:
        return _state

    previous_cwd = Path.cwd()
    os.chdir(INFER_DIR)
    try:
        from f5_tts.infer.utils_xeus import ApplyKmeans, load_xeus_model

        xeus_model = load_xeus_model(device).eval()
        apply_kmeans = ApplyKmeans(device)
        vocoder = load_vocoder(vocoder_name="bigvgan", device=device)

        model_cfg = OmegaConf.load(DEFAULT_CONFIG)
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        ema_model = load_model(
            model_cls,
            model_cfg.model.arch,
            _resolve_cached_path(DEFAULT_CKPT),
            mel_spec_type="bigvgan",
            vocab_file=_resolve_cached_path(DEFAULT_VOCAB),
            device=device,
        )
    finally:
        os.chdir(previous_cwd)

    _state = EZVCState(
        device=device,
        xeus_model=xeus_model,
        apply_kmeans=apply_kmeans,
        vocoder=vocoder,
        ema_model=ema_model,
    )
    return _state


def _extract_units(audio_path: str, state: EZVCState) -> str:
    previous_cwd = Path.cwd()
    os.chdir(INFER_DIR)
    try:
        from f5_tts.infer.utils_xeus import extract_units

        return extract_units(audio_path, state.xeus_model, state.apply_kmeans, state.device)
    finally:
        os.chdir(previous_cwd)


def convert_voice(
    target_audio: str | None,
    source_audio: str | None,
    device_choice: str,
    keep_models_loaded: bool,
    nfe_steps: int,
    speed_value: float,
    cfg_value: float,
    sway_value: float,
    cross_fade_value: float,
    target_rms_value: float,
    output_duration_value: float,
    match_source_duration: bool,
    seed_value: int,
    progress: gr.Progress = gr.Progress(),
):
    if not target_audio:
        raise gr.Error("Target speaker audio is required.")
    if not source_audio:
        raise gr.Error("Source speech audio is required.")

    try:
        state = _load_state(device_choice)
        progress(0.2, desc="Extracting target units")
        ref_text = _extract_units(target_audio, state)
        progress(0.4, desc="Extracting source units")
        source_units = _extract_units(source_audio, state)

        generated_segments = []
        chunks = [
            re.sub(r"\[(\w+)\]", "", text).strip()
            for text in re.split(r"(?=\[\w+\])", source_units)
        ]
        chunks = [text for text in chunks if text]
        if not chunks:
            raise gr.Error("No source units were extracted.")

        seed = None if seed_value is None or seed_value < 0 else int(seed_value)
        target_duration = sf.info(target_audio).duration
        source_duration = sf.info(source_audio).duration
        requested_output_duration = (
            float(output_duration_value) if output_duration_value is not None and output_duration_value > 0 else None
        )
        fixed_output_duration = requested_output_duration
        if fixed_output_duration is None and match_source_duration:
            fixed_output_duration = source_duration

        chunk_byte_lengths = [max(len(text.encode("utf-8")), 1) for text in chunks]
        total_chunk_bytes = sum(chunk_byte_lengths)

        progress(0.55, desc="Converting speech")
        for chunk_index, gen_text in enumerate(chunks):
            fixed_duration = None
            if fixed_output_duration is not None:
                chunk_output_duration = fixed_output_duration * chunk_byte_lengths[chunk_index] / total_chunk_bytes
                fixed_duration = target_duration + chunk_output_duration

            audio_segment, sample_rate, _ = infer_process(
                target_audio,
                ref_text,
                gen_text,
                state.ema_model,
                state.vocoder,
                mel_spec_type="bigvgan",
                target_rms=target_rms_value,
                cross_fade_duration=cross_fade_value,
                nfe_step=int(nfe_steps),
                cfg_strength=cfg_value,
                sway_sampling_coef=sway_value,
                speed=speed_value,
                fix_duration=fixed_duration,
                seed=None if seed is None else seed + chunk_index * 1000,
                device=state.device,
                progress=None,
                show_info=lambda _: None,
            )
            generated_segments.append(audio_segment)

        if not generated_segments:
            raise gr.Error("No audio was generated.")

        output_wave = np.concatenate(generated_segments)
        with tempfile.NamedTemporaryFile(suffix=".wav", **tempfile_kwargs) as output_file:
            output_path = output_file.name
        sf.write(output_path, output_wave, sample_rate)
        progress(1.0, desc="Done")
        return output_path, f"{len(output_wave) / sample_rate:.2f}s"
    finally:
        if not keep_models_loaded:
            unload_models()


with gr.Blocks(title="EZ-VC") as app:
    gr.Markdown("# EZ-VC")
    with gr.Row():
        target_audio_input = gr.Audio(
            label="Target Speaker",
            type="filepath",
            value=str(DEFAULT_TARGET_AUDIO),
        )
        source_audio_input = gr.Audio(
            label="Source Speech",
            type="filepath",
            value=str(DEFAULT_SOURCE_AUDIO),
        )
    with gr.Row():
        convert_button = gr.Button("Convert", variant="primary")
        unload_button = gr.Button("Unload Models")
        device_input = gr.Dropdown(["auto", "mps", "cpu", "cuda"], value="auto", label="Device")
    keep_models_input = gr.Checkbox(label="Keep models loaded between conversions", value=False)
    with gr.Row():
        nfe_input = gr.Slider(4, 32, value=DEFAULT_NFE_STEPS, step=2, label="NFE Steps")
        speed_input = gr.Slider(0.5, 1.5, value=DEFAULT_SPEED, step=0.05, label="Speed")
    with gr.Row():
        cfg_input = gr.Slider(0.5, 4.0, value=DEFAULT_CFG_STRENGTH, step=0.1, label="CFG")
        sway_input = gr.Slider(-2.0, 2.0, value=DEFAULT_SWAY, step=0.1, label="Sway")
        cross_fade_input = gr.Slider(0.0, 1.0, value=DEFAULT_CROSS_FADE, step=0.05, label="Crossfade")
    with gr.Row():
        target_rms_input = gr.Slider(0.02, 0.30, value=DEFAULT_TARGET_RMS, step=0.01, label="Target RMS")
        output_duration_input = gr.Number(
            value=DEFAULT_OUTPUT_DURATION,
            label="Output Duration (seconds, 0 = automatic)",
            precision=2,
        )
        seed_input = gr.Number(value=DEFAULT_SEED, label="Seed (-1 = random)", precision=0)
    match_source_duration_input = gr.Checkbox(
        label="Match source duration when output duration is 0",
        value=DEFAULT_MATCH_SOURCE_DURATION,
    )
    with gr.Row():
        converted_audio_output = gr.Audio(label="Converted Audio", type="filepath")
        duration_output = gr.Textbox(label="Duration")

    convert_button.click(
        convert_voice,
        inputs=[
            target_audio_input,
            source_audio_input,
            device_input,
            keep_models_input,
            nfe_input,
            speed_input,
            cfg_input,
            sway_input,
            cross_fade_input,
            target_rms_input,
            output_duration_input,
            match_source_duration_input,
            seed_input,
        ],
        outputs=[converted_audio_output, duration_output],
    )
    unload_button.click(unload_models, outputs=[duration_output])


@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default="127.0.0.1", help="Host to run the app on")
@click.option("--share", "-s", default=False, is_flag=True, help="Share the app via Gradio")
@click.option("--inbrowser", "-i", default=False, is_flag=True, help="Open in browser")
def main(port, host, share, inbrowser):
    app.queue(default_concurrency_limit=1).launch(
        server_name=host,
        server_port=port,
        share=share,
        inbrowser=inbrowser,
    )


if __name__ == "__main__":
    main()
