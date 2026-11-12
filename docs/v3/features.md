# Inflect Mini v3 features

[Back to Inflect](../../README.md) | [Installation](getting-started.md) | [Release scope](release.md)

All commands below run from the downloaded Mini package in its documented
environment. Start with [ordinary speech](getting-started.md#mini-native-24-khz).

## Reference-voice tuning

Voice Tuner creates a reusable profile from a recording. It adapts voice
conditioning toward the reference; it is not a guarantee of an exact clone,
speaker identity or accent transfer.

Use an authorized 5-15 second recording with clear speech. The qualified
enrollment environment is **Linux CPU with Python 3.12**. Enrollment encoders
are separate downloads, with their own licenses and resource requirements.

```sh
python -m pip install uv
python -m uv pip install --python python --find-links core/baseline/vendor -c extras/tuner/constraints-cpu.txt -r extras/tuner/requirements-tuner-enrollment-cpu-20261110.txt
python tuner.py enroll --reference ../my-reference.wav --profile ../my-voice.json --work-dir ../tuner-work --device cpu --consent-confirmed --download-public-assets --accept-third-party-licenses
python tuner.py say --profile ../my-voice.json --work-dir ../tuner-work --device cpu --text "A new chapter starts here." --output ../my-voice.wav
```

Use new profile/output paths in existing directories outside the package. Keep
the profile JSON, its companion `.pt` file and the work directory for reuse.
The recording is used for enrollment, not resubmitted for every utterance.

[Full tuner guide](https://huggingface.co/owensong/Inflect-Mini-v3/blob/main/extras/tuner/README.md)
| [Component licenses](https://huggingface.co/owensong/Inflect-Mini-v3/blob/main/extras/tuner/LICENSE-MAP.json)

## Vocal-event tags

The optional event branch generates learned vocal performances and assembles
them with speech. It does not make the speech before or after the tag angry,
happy, whispered or otherwise emotionally conditioned. Available performances
have limited variation.

**Eight tags are available across all six voices:** Cassidy, Hope, Sophia,
James, Jasper and Mark. The local event path uses **CPU, native 24 kHz**.

`[laugh]` · `[sigh]` · `[gasp]` · `[cough]` · `[yawn]` · `[inhale]` · `[exhale]` · `[clear_throat]`

`[throat_clear]` is an alias for `[clear_throat]`. Events use voice-specific
learned performances and calibrated playback levels. Where multiple takes
are available, a seed selects among them. Cassidy's original laugh, sigh,
gasp and cough remain fixed performances. This is a small bank, not unlimited new
performances or emotion conditioning.

**Already downloaded Mini?** Refresh the complete package before using the
new tags; older snapshots contain only the original four Cassidy events.
Run this refresh block from the directory containing `inflect-mini-v3`,
not from inside the package.

```sh
hf download owensong/Inflect-Mini-v3 --local-dir inflect-mini-v3
cd inflect-mini-v3
python verify_release.py
```

Then install the event dependencies and generate:

```sh
python -m pip install -r extras/events/component/requirements-cpu.txt
python run.py events --voice cassidy --profile base --text "You really surprised me. [laugh] That was a good one." --output ../mini-laugh.wav
```

Change `--voice cassidy` to any of the other five voice IDs to use their
event performances. For example, use `[yawn]` with `--voice hope`.

The event command does not apply the speech enhancer. Event component parameters
are additional to Mini's core count.

[Published event guide](https://huggingface.co/owensong/Inflect-Mini-v3/blob/main/extras/events/README.md)

## 48 kHz enhancement

Native speech is 24 kHz by default. The optional enhancer adds **2,846,722
parameters** and writes 48 kHz output without replacing the speech model.

```sh
python run.py speech --profile enhanced --voice hope --text "The next chapter starts here." --subtype PCM_16 --output ../mini-48k.wav
```

The enhancer is a different output presentation, not a universal quality
upgrade. Native output can be preferable on revealing headphones; enhanced
output can sound brighter on some speakers. Some material can expose a faint
buzzy texture. Compare the same utterance before choosing a default for your
application.

[Native sample](https://huggingface.co/owensong/Inflect-Mini-v3/resolve/main/media/mini-cassidy.wav)
| [The same sample enhanced](https://huggingface.co/owensong/Inflect-Mini-v3/resolve/main/media/mini-cassidy-48k.wav)

## Long text

Preset speech is split into manageable chunks. This bounds the work performed
per chunk; it is not a claim of acoustic streaming or a measured first-audio
latency. Check passage joins, pronunciation and numbers for your application.
