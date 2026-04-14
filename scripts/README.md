# Scripts

This folder contains the operational entrypoints for the current Inflect workspace.

## Dataset generation

- `generate_voxcpm_dataset.py`
  - generate VoxCPM synthetic data through the public Hugging Face Space
- `generate_voxcpm_dataset_local.py`
  - generate the same type of data locally on a rental GPU
- `build_voxcpm_text_corpus.py`
  - older corpus builder
- `build_voxcpm_text_corpus_v2.py`
  - current higher-quality English `20k` text pool builder
- `build_voxcpm_generation_plan.py`
  - balanced `60k` plan builder over the `20k` text pool
- `merge_voxcpm_metadata_shards.py`
  - combine sharded local-generation metadata files
- `prepare_public_voxcpm_dataset.py`
  - freeze and relabel a dataset snapshot for public Hugging Face release
- `upload_voxcpm_dataset.py`
  - upload an `AudioFolder` dataset directory to Hugging Face

## Runtime benchmarks

- `test_base_zipvoice.py`
  - main ZipVoice benchmark driver
- `run_phase2_sweeps.py`
  - Lux-inspired ablation sweeps
- `run_inflect_base_bench.py`
  - focused Inflect-base benchmark
- `run_lux_vs_inflect_bench.py`
  - direct Lux vs Inflect comparison
- `test_true_lux.py`
  - real LuxTTS local test path
- `test_chatterbox_turbo.py`
  - side benchmark

## Comparison servers

- `serve_zipvoice_compare.py`
- `serve_zipvoice_tracka_compare.py`
- `serve_inflect_base_compare.py`
- `serve_lux_vs_inflect_compare.py`

These expose local listening pages over HTTP for A/B work.

## Fine-tuning

- `prepare_inflect_finetune_data.py`
  - split VoxCPM dataset into train/dev TSVs
- `prepare_tokens_serial.py`
  - Windows-safe token preparation
- `run_inflect_teacher_finetune.py`
  - main launcher for ZipVoice teacher fine-tuning

## Reference voice utilities

- `organize_voices.py`
  - organize local reference voice previews
- `preprocess_reference_voices.py`
  - clean and normalize reference voice audio
- `transcribe_reference_voices.py`
  - produce prompt transcripts for cloning references

## Miscellaneous

- `generate_enhancer_pairs.py`
- `generate_inflect_enhance_pairs.py`
  - pair generation for the enhancement pipeline
- `restore_cremad_sample.py`
  - small utility for CREMA-D sample restoration/testing
- `voxcpm_local_web.py`
  - local VoxCPM web utility
- `run_voxcpm_web.ps1`
- `setup_voxcpm_env.ps1`
  - PowerShell helpers for the VoxCPM environment
