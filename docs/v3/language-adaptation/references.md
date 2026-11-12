# References and Evidence Boundaries

Primary references were inspected through BrowserOS Neo on 2026-11-12. These
are sources for specific engineering or linguistic facts, not evidence that an
Inflect adaptation was trained, sounded good, or passed human review. Research
was read-only; no account settings or public content were changed.

## Project Implementation

- [V3 model bridge](../../../finetune_v3/model.py): local native33 Mini
  factories, fixed-size symbols/conditioning, experimental unused-ID migration,
  parameter groups, batch-one aligned forward, token/frame bounds, and
  predicted-duration inference. See [tested scope](implementation-status.md).
- [V3 checkpoint/export](../../../finetune_v3/checkpoint.py): model and
  optional optimizer identity/state, explicit absence of controller resume, and
  model-core export with strict public-loader tensor readback. An export is not
  an assembled raw-text release or quality approval.
- [V3 training controller](../../../finetune_v3/training.py): strict aligned
  duration/waveform rows, `TrainOptions`/`train`, supervised duration and
  multi-resolution spectral loss, reconstruction validation, and paired
  controller resume receipts. Source inspection only, not language-quality
  qualification. See the [current workflow](workflow.md).
- [V3 data preparation](../../../finetune_v3/data.py) and
  [language frontend](../../../finetune_v3/frontend.py): explicit aligned-label
  provenance, connected split groups, native clocks and hashes, declared
  language/vocabulary, and refusal of unknown phones or English fallback.
  [CLI wiring](../../../finetune_v3/cli.py) supports the eight verbs listed in
  [current commands](commands.md); parser checks and mechanical smoke evidence
  are summarized with installation and quality boundaries in
  [tested scope](implementation-status.md).
- [Mechanical fixture generator](../../../finetune_v3/smoke.py): six English
  self-generated examples with their original synthesis clock, preparation,
  and an explicit scope receipt. It is not external-recording alignment or
  language-quality evidence.
- [Model API verification receipt](../../../finetune_v3/MODEL_API_VERIFICATION.json):
  parent-side report of 41 CPU bridge/checkpoint/migration/export checks. Its
  stated scope excludes CUDA and language quality. Separate CUDA2/resume4 and
  continuous4 receipts, parent export/phoneme-inference passes, and the parent's
  final lightweight result (138 passed, 1 skipped) and separate v2 regression
  result (39 passed) are attributed with their limits
  in [implementation status](implementation-status.md).
- [Tested requirements snapshot](../../../finetune_v3/requirements-tested.txt):
  versions in existing verification environments, with Torch chosen separately
  for CPU/CUDA. It is not a complete freshly downloaded environment lock.
- [Notice retention](../../../finetune_v3/notices.py): full-source-package
  requirement, original license/notice preservation, and retained-file hashes
  included by CLI export. Historical source notices are not rewritten as
  adapted-weight claims.
- Current CLI source additionally verifies controller/checkpoint/run-contract
  binding for export and requires the saved eSpeak engine version for raw-text
  inference. The fixed-English `smoke-dataset --frontend espeak` chain now has
  separate parent mechanical pass evidence, including raw-text output with
  matching engine `(1, 52, 0)`. It is not folded into earlier phoneme-input
  receipts or described as foreign-language quality.
- [V2 adaptation contract](../../../finetune/CONTRACT.md) and
  [language documentation](../../../finetune/docs/LANGUAGES.md): separate
  fixed-voice, single-language warm start, not a private release recipe or
  multilingual runtime selector.
- [V2 implementation notes](implementation-status.md#existing-v2-toolkit-useful-precedent-not-a-v3-loader):
  direct source review of preparation, conversion, frontend, audit, model
  loading, training, checkpointing, and CLI. Code behavior takes precedence over
  broad wording in an older guide.

The existing packaged Mini deployment and its phoneme-input/English frontend
were also inspected locally as distributed source. This guide does not reproduce
private training data, teacher-generation instructions, internal curricula,
storage locations, or credentials.

## External Primary Sources

| Source | What was inspected and used | What it does not prove |
| --- | --- | --- |
| [Phonemizer documentation](https://bootphon.github.io/phonemizer/) | Backend capability differences for phones, stress, punctuation, and boundaries | Acoustic-language coverage in Inflect |
| [Phonemizer API](https://bootphon.github.io/phonemizer/api_reference.html) | Separators, stress/tie options, language-switch flags and word-count mismatch behavior | That removing flags corrects a foreign-word pronunciation |
| [Phonemizer common issues](https://bootphon.github.io/phonemizer/common_issues.html) | Shared-library/data discovery and warning visibility | Any particular local environment is qualified |
| [eSpeak NG language list](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md) | Polish appears under `pl` | That installed or bundled resources include it |
| [eSpeak NG Polish definition](https://github.com/espeak-ng/espeak-ng/blob/master/espeak-ng-data/lang/zlw/pl) | Upstream named Polish language definition | Inflect runtime language selection |
| [eSpeak NG Polish rules](https://github.com/espeak-ng/espeak-ng/blob/master/dictsource/pl_rules) | Context-dependent rules in the upstream implementation, including distinct Polish spellings | A complete reviewed phonetic transcription of the proposed corpus |
| [Jassem, Polish, JIPA 33(1), 103-107 (2003)](https://www.cambridge.org/core/journals/journal-of-the-international-phonetic-association/article/polish/8CDD2C699493A69A40A5E7D923629574) | Publisher's article identity and extract describing consonant richness and clusters | That the authored example panel is a complete phonological inventory |
| [Unicode Standard Annex #15](https://unicode.org/reports/tr15/) | Canonical versus compatibility normalization, NFC/NFKC distinctions | Linguistically correct normalization or identical model IDs |
| [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) | Effective-batch accumulation and consistent scaling/update boundaries | AMP/accumulation support or quality in a new v3 controller |
| [PyTorch CUDA memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management) | Allocated versus reserved memory and limits of releasing unused cache | A measured Inflect training VRAM requirement |

The Jassem landing-page extract was readable; the browser PDF view did not
yield an extracted full article during this task. No claims here rely on a
full-PDF reading. The Polish test categories and newly authored prompts are
proposed review work, not quotations or approved pronunciation labels.

Live upstream pages can change. Pin actual tool/backend versions and resource
hashes for an experiment; a link to current documentation is not an environment
lock. The memory arithmetic in the guide is explicit planning arithmetic, not
a benchmark. No new-language quality metrics or tested hyperparameters were
established by these mechanical tests.
