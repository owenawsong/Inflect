# Changelog

## 3.0.0 - 2026-11-11

### Released

- Inflect Mini v3 (34.31M), Micro v3 (11.79M), and Atom v3 (157.19K learned synthesis-core parameters).
- Six English preset voices, self-contained model packages and native 24 kHz output.
- Mini's optional compact 48 kHz enhancement, reference-based Voice Tuner and vocal-event component.
- A public model-comparison playground, v3 installation guides and separately scoped quality, WER and runtime evidence.
- Apache 2.0 Inflect-owned weights and inference code, with third-party terms retained.

### Compatibility

- V3 inference uses the self-contained Hugging Face packages, not the legacy v1/v2 root library interface.
- Optional Mini features are not implied for Micro or Atom. Check the package feature guide for exact support.
- V1/v2 material is preserved for existing users. Their measurements are not v3 results.


## 2.0.0 - 2026-07-24

### Added

- Inflect-Micro-v2, a complete 9.36M-parameter English text-to-waveform model.
- Inflect-Nano-v2, a complete 3.96M-parameter English text-to-waveform model.
- Shared self-contained Python API and CLI for CPU and CUDA.
- Integrated 24 kHz waveform generation with no external vocoder.
- Deterministic seeds, speaking speed, and delivery-variation controls.
- Punctuation-aware long-text chunking.
- Modern400 multi-ASR evaluation, UTMOS22 reports, blind preference evidence,
  managed CPU measurements, and integrity manifests.
- GitHub download-and-synthesize and matched-comparison examples.

### Release boundary

- One fixed English male voice per model.
- PyTorch FP32 is the validated format.
- Voice cloning, selectable speakers, multilingual synthesis, acoustic
  streaming, and quantized/mobile exports are not claimed.

## 1.x

Inflect Nano v1 remains available as a legacy release for reproducibility.
Inflect v2 is recommended for new integrations.
