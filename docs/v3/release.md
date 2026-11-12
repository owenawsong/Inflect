# Inflect v3 release scope

[Back to Inflect](../../README.md) | [Installation](getting-started.md)

## Packages

| Package | Learned synthesis-core parameters | Native output |
| :--- | ---: | :--- |
| [Mini](https://huggingface.co/owensong/Inflect-Mini-v3) | 34,310,790 | 24 kHz |
| [Micro](https://huggingface.co/owensong/Inflect-Micro-v3) | 11,787,634 | 24 kHz |
| [Atom](https://huggingface.co/owensong/Inflect-Atom-v3) | 157,190 | 24 kHz |

All three offer six fixed English presets. Their complete inference packages,
weights, pronunciation assets and integrity manifests are hosted together on
Hugging Face. The GitHub repository is the project entry point and guide; its
original `inflect/` library and old examples remain v1/v2 interfaces.

Core parameter counts are not total runtime parameters, installation sizes or
memory budgets. Atom's default Misaki pronunciation frontend reports another
1,570,514 active frontend parameters. Fixed conditioning and dependencies are
additional. Mini's optional enhancer has 2,846,722 parameters; event components
and reference enrollment have their own footprints.

## Optional components

Mini's reference-based Voice Tuner, event branch and enhancer are separate
components. Their presence does not imply the same features in Micro or Atom.
The event package includes eight vocal-event tags for all six preset voices,
with CPU/native-24-kHz playback. See [features and supported commands](features.md).

The published language is English. General emotion/style tags, exact voice
cloning, guaranteed accent transfer, acoustic streaming, browser/WASM exports,
mobile runtimes and integer-quantized deployment are not claimed here.

## License and attribution

Inflect-owned code and weights are Apache License 2.0. Third-party code, models,
dependencies, pronunciation assets and separately downloaded encoders retain
their own terms. Review the notices and component license maps in the package
you download, not just the headline license.

- [Repository license](../../LICENSE)
- [Repository notice](../../NOTICE)
- [Mini notices](https://huggingface.co/owensong/Inflect-Mini-v3/blob/main/THIRD_PARTY_NOTICES.md)
- [Core component scope](https://huggingface.co/owensong/Inflect-Mini-v3/blob/main/core/LICENSE_SCOPE.json)
- [Voice Tuner license map](https://huggingface.co/owensong/Inflect-Mini-v3/blob/main/extras/tuner/LICENSE-MAP.json)

The release includes open weights and runnable inference. The full training
pipeline, corpus-generation systems, training data, private reference recordings
and internal experiment controllers are not part of the public release.

## Reproducibility and safe use

Record the exact Hugging Face revision, model, voice, output mode and command
when comparing results. Use package verification before inference. Store outputs
and voice profiles outside package folders, and retain only recordings you have
permission to use. Do not impersonate speakers or present synthesized speech as
an authentic recording of someone else.

## Earlier releases

V2 and v1 remain available for reproducibility and existing integrations.
[V2 setup and measurements](../v2/README.md) retain their original scope and
must not be read as v3 features or results.
