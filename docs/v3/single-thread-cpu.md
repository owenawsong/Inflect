# Mini v3: Single-Thread CPU Measurement

This additional measurement answers a specific deployment question: how fast is
the released native Mini path with **one Torch compute thread and one interop
thread**, including the text frontend? It is not a streaming latency claim or a
new model checkpoint. The published weights were unchanged.

| Measurement | Observed result |
| --- | ---: |
| Host | AMD Ryzen 9 3900X, Windows 11 |
| Python / PyTorch | 3.12.11 / 2.11.0+cpu |
| Warm aggregate RTF, lower is faster | **1.11035** |
| Warm throughput | **0.9006x real time** |
| Audio / timed computation | 86.4 s / 95.934 s |
| Median whole-utterance latency | 4.096 s |
| 95th percentile whole-utterance latency | 4.430 s |
| First complete utterance after loading | 4.755 s for 4.05 s of audio |

**At this setting, this machine was slightly slower than real time.** The
previous two-thread and optimized CUDA measurements use different settings and
must not be substituted for this result. Runtime varies with hardware, text,
background load, caching and the chosen inference path.

## Protocol

Six authored short prompts, Cassidy and Hope, two timed repetitions: 24 cases.
Six warmup calls were excluded. Each timer includes raw-text normalization,
Misaki G2P with its required bundled eSpeak fallback, native 24 kHz synthesis
and in-memory PCM16 WAV serialization. Loading, integrity checks and disk WAV
writes are outside the warm timer. Threads were asserted around every call,
including after the inference loader initially selected two threads.

The setup before the first utterance took 60.458 s in this fresh child process;
setup plus that utterance took 65.213 s. Those sums exclude process startup and
unmeasured gaps. OS disk caches were not controlled, and no repeated cold-start
distribution or isolated CPU-affinity test was performed.

**This release returns complete chunks, not acoustic streaming.** The first
complete-utterance measurement is not time to a first streaming audio packet.

## Frontend And Footprint

The learned synthesis core is 34,310,790 parameters. The active spaCy frontend
pipeline inspected here contains an additional 1,570,514 learned values. That
does not count lexicons, eSpeak language data or every installed dependency.
The complete installed environment size was not measured.

The frontend used Misaki 0.9.4, spaCy 3.8.16, en_core_web_sm 3.8.0,
phonemizer-fork 3.3.2 and espeakng-loader 0.2.4. No reference adaptation,
event tags, delivery profile or output enhancer was enabled.

## Evidence

The [machine-readable report](evidence/singlethread-20261112.json) includes all
24 timed cases, prompts, warmup scope, version/source hashes and aggregation.
RTF is total measured computation divided by total generated audio; throughput
is its reciprocal, not an unweighted average of per-clip ratios.

This is a local self-run measurement, not an independent comparison or an
Open TTS Leaderboard submission. No new quality or competitor ranking follows
from this CPU timing probe.
