from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import gradio as gr
import soundfile as sf
from voxcpm import VoxCPM


APP_TITLE = "VoxCPM Local Web UI"
DEFAULT_MODEL = "openbmb/VoxCPM2"
DEFAULT_TEXT = "VoxCPM2 is a multilingual speech model with strong voice cloning and voice design."

_MODEL: VoxCPM | None = None
_MODEL_ID: str | None = None


def get_model(model_id: str, optimize: bool) -> VoxCPM:
    global _MODEL, _MODEL_ID
    if _MODEL is not None and _MODEL_ID == model_id:
        return _MODEL
    _MODEL = VoxCPM.from_pretrained(model_id, optimize=optimize)
    _MODEL_ID = model_id
    return _MODEL


def maybe_write_prompt_upload(upload: Optional[str | bytes], tmp_dir: Path) -> Optional[str]:
    if not upload:
        return None
    if isinstance(upload, str):
        return upload
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_path = tmp_dir / "uploaded_reference.wav"
    out_path.write_bytes(upload)
    return str(out_path)


def synthesize(
    *,
    model_id: str,
    optimize: bool,
    text: str,
    control_instruction: str,
    reference_audio: Optional[str],
    prompt_text: str,
    cfg_value: float,
    inference_timesteps: int,
    normalize: bool,
    denoise_reference: bool,
):
    text = (text or "").strip()
    control_instruction = (control_instruction or "").strip()
    prompt_text = (prompt_text or "").strip()

    if not text:
        raise gr.Error("Enter target text.")

    final_text = f"({control_instruction}){text}" if control_instruction else text
    model = get_model(model_id, optimize=optimize)

    kwargs = {
        "text": final_text,
        "cfg_value": float(cfg_value),
        "inference_timesteps": int(inference_timesteps),
        "normalize": bool(normalize),
        "denoise": bool(denoise_reference),
    }

    if reference_audio:
        kwargs["reference_wav_path"] = reference_audio
    if reference_audio and prompt_text:
        kwargs["prompt_wav_path"] = reference_audio
        kwargs["prompt_text"] = prompt_text

    wav = model.generate(**kwargs)
    return model.tts_model.sample_rate, wav


def save_last_audio(audio_data, output_dir: Path) -> str:
    if audio_data is None:
        raise gr.Error("No generated audio to save.")
    sr, wav = audio_data
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "last_voxcpm_output.wav"
    sf.write(str(out_path), wav, sr)
    return f"Saved to {out_path}"


def build_demo(model_id: str, optimize: bool, output_dir: Path) -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n"
            "Modes:\n"
            "- Voice design: text only, optionally add a control instruction\n"
            "- Controllable cloning: add reference audio\n"
            "- Ultimate cloning: add reference audio and exact prompt transcript"
        )

        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(label="Target Text", lines=4, value=DEFAULT_TEXT)
                control_instruction = gr.Textbox(
                    label="Control Instruction",
                    lines=3,
                    placeholder="Example: A young woman, soft voice, slightly excited, speaking a bit faster.",
                )
                reference_audio = gr.Audio(
                    label="Reference Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                prompt_text = gr.Textbox(
                    label="Prompt Transcript for Ultimate Cloning",
                    lines=3,
                    placeholder="Optional. Use only if the reference audio transcript is known exactly.",
                )

                with gr.Accordion("Advanced", open=False):
                    cfg_value = gr.Slider(0.1, 4.0, value=2.0, step=0.1, label="CFG Value")
                    inference_timesteps = gr.Slider(1, 20, value=10, step=1, label="Inference Timesteps")
                    normalize = gr.Checkbox(value=True, label="Normalize Text")
                    denoise_reference = gr.Checkbox(value=False, label="Denoise Reference Audio")

                generate = gr.Button("Generate", variant="primary")
                save = gr.Button("Save Last Output")

            with gr.Column(scale=2):
                model_box = gr.Textbox(label="Model", value=model_id, interactive=False)
                optimize_box = gr.Checkbox(label="Optimize", value=optimize, interactive=False)
                audio_out = gr.Audio(label="Generated Audio")
                status = gr.Markdown("Idle.")

        generate.click(
            fn=lambda text, control_instruction, reference_audio, prompt_text, cfg_value, inference_timesteps, normalize, denoise_reference: (
                synthesize(
                    model_id=model_id,
                    optimize=optimize,
                    text=text,
                    control_instruction=control_instruction,
                    reference_audio=reference_audio,
                    prompt_text=prompt_text,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                    normalize=normalize,
                    denoise_reference=denoise_reference,
                ),
                "Generation complete.",
            ),
            inputs=[
                text,
                control_instruction,
                reference_audio,
                prompt_text,
                cfg_value,
                inference_timesteps,
                normalize,
                denoise_reference,
            ],
            outputs=[audio_out, status],
        )

        save.click(fn=lambda audio: save_last_audio(audio, output_dir), inputs=[audio_out], outputs=[status])

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8808)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "voxcpm_web")
    args = parser.parse_args()

    demo = build_demo(model_id=args.model, optimize=not args.no_optimize, output_dir=args.output_dir)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
