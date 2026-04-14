# Inflect — Todo List

Last updated: 2026-04-13

## Highest Priority

- [ ] finish the current VoxCPM generation push toward `60k`
- [ ] finish a short teacher fine-tune demo and listen to results
- [ ] clean up the public GitHub repo structure and docs
- [ ] freeze a publishable public dataset snapshot
- [ ] upload the public dataset to `owensong` on Hugging Face

## Runtime / Backbone

- [x] choose a working runtime default
- [x] reject Lux `speed * 1.3`
- [x] reject prompt cap as a default
- [x] choose `inflect_base + lux solver`
- [ ] optional: benchmark `k2`-enabled runtime once
- [ ] optional: re-audit reference clips for branded leakage

## VoxCPM Dataset

- [x] build `voxcpm_texts_20000_v2.csv`
- [x] build `voxcpm_generation_plan_v2_60k.csv`
- [x] keep the voice pool at `56`
- [ ] let the active HF Space run keep climbing toward `60k`
- [ ] sample-listen random clips from the active run
- [ ] decide whether the final public release should be:
  - a snapshot of the active run
  - or a fresh clean full `v2` run

## Teacher Fine-Tuning

- [x] fix the non-finite gradient crash
- [x] preserve the ZipVoice local fix as a patch file
- [x] improve trainer logging with progress and ETA
- [ ] complete a short demo run
- [ ] listen against the baseline
- [ ] if promising, run the first longer VoxCPM foundation fine-tune
- [ ] decide whether a LibriTTS-R anchor mix is needed later
- [ ] stage expressive data only after the foundation run is judged

## Expressive Data

- [ ] prepare an Expresso integration plan
- [ ] avoid mixing Expresso into the first foundation demo
- [ ] define the later expressive fine-tune objective clearly

## Publishing

- [ ] rewrite the top-level README around the actual repo
- [ ] add subfolder READMEs for scripts / inflect / voice-encoder
- [ ] add a proper repo license file
- [ ] strip tracked `.pyc` files from Git
- [ ] restore a clean `voice-encoder/.gitignore`
- [ ] prepare a public VoxCPM dataset card with honest licensing
- [ ] upload the dataset once an HF token is available
- [ ] push the cleaned branch and update the draft PR

## Later

- [ ] add a proper evaluation score sheet
- [ ] decide what counts as "V1 done"
- [ ] only then reopen paralinguistics and richer product features
