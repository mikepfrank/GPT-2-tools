# Verification record

Verified locally through **2026-10-01**.

## Host

```text
Dell Precision 7680
Windows 11 Pro, build 26200
Intel Core i7-13850HX (20 cores / 28 logical processors)
31.7 GiB RAM
Intel UHD Graphics; no NVIDIA GPU
Ubuntu 24.04.1 LTS under WSL 2
WSL allocation: 15 GiB RAM + 4 GiB swap
```

## Runtime

```text
Python       3.12.3
PyTorch      2.14.1+cpu
Transformers 5.18.0
CUDA         unavailable (expected)
Default CPU threads 14
```

`pip check` reported no broken requirements. Python bytecode compilation and
the PowerShell/Bash launcher syntax checks passed.

Rerunning `setup.ps1` against the existing virtual environment completed
successfully: pinned runtime packages were already satisfied, the editable
package was reinstalled, and `pip check` passed again. This confirms
idempotence for the tested environment; it is not a fully hermetic rebuild
because Ubuntu repositories and upgraded pip build tooling are not pinned.

Dated disk usage after installation was approximately 1.2 GB for
`~/.venvs/gpt2-xl` and 6.0 GB for the complete Hugging Face cache containing
the filtered GPT-2 XL download.

## Checkpoint

```text
Repository  openai-community/gpt2-xl
Revision    15ea56dee5df4983c59b2538573817e1667135e2
File        model.safetensors
Size        5.99 GiB (6.43 GB decimal)
SHA-256     0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a
```

The local file's SHA-256 was computed across all 5.99 GiB and matched Hugging
Face's metadata for the pinned checkpoint.

Loading the checkpoint reported exactly **1,557,611,200 parameters**.

## End-to-end smoke test

Command:

```powershell
./run.ps1 -Prompt "Reversible computing is" `
  -MaxNewTokens 12 -Temperature 0 -Offline -IncludePrompt
```

Observed output:

```text
Reversible computing is a concept that has been around for a while, but it
```

The first cold run mapped the model successfully and generated 12 tokens in
13.4 seconds (0.89 tokens/second). A subsequent short warm-cache thread sweep
reached 3.83 tokens/second with the default 14 threads; short microbenchmarks
fluctuate, so these figures are a baseline rather than a general performance
guarantee. The CLI now labels first-request timing explicitly because it
includes lazy page-in and kernel setup.

Regression checks also passed for a supplied random seed and for a prompt that
begins with a hyphen; both travel correctly through the PowerShell-to-WSL
launcher.

## Temperature-sweep application

The experiment wrapper was first validated without loading the model:

```powershell
./experiment.ps1 -Offline -DryRun
```

It expanded the tracked `experiments/identity-context.json` configuration to
the expected ten cases: one seedless greedy completion, plus temperatures 0.4,
0.8, and 1.2 at seeds 42, 314, and 2026. This also exercised absolute path
translation across the PowerShell/WSL boundary for a project path containing
spaces.

Two full offline sweeps completed. The final-code run lasted from
`2026-10-01T01:47:31Z` through `2026-10-01T01:50:08Z` (the evening of
2026-09-30 in the host's America/Chicago time zone). The exact prompt contained
85 GPT-2 tokens and its UTF-8 SHA-256 was
`a1c5f5acbc9f5cd70fbe8db1a20bec0aac8da88948c2ab324aa546a7a4f8e9d7`.
One GPT-2 XL process served every case at the default 14 CPU threads.

| Temperature | Seed | Output tokens | Stop reason | Repeated 4-gram fraction |
| ---: | ---: | ---: | --- | ---: |
| 0.0 | greedy | 120 | max token limit | 0.794872 |
| 0.4 | 42 | 120 | max token limit | 0.923077 |
| 0.4 | 314 | 120 | max token limit | 0.094017 |
| 0.4 | 2026 | 120 | max token limit | 0.837607 |
| 0.8 | 42 | 67 | end-of-text token | 0.0 |
| 0.8 | 314 | 89 | end-of-text token | 0.0 |
| 0.8 | 2026 | 1 | end-of-text token | n/a |
| 1.2 | 42 | 120 | max token limit | 0.0 |
| 1.2 | 314 | 120 | max token limit | 0.034188 |
| 1.2 | 2026 | 1 | end-of-text token | n/a |

The one-token first-request warm-up took 0.96 seconds on the final run after
the earlier sweep had warmed the host file cache; it took 7.31 seconds on the
first sweep. Full 120-token samples in the final run generated at 5.71–5.87
output tokens/second. Very short end-of-text samples are not meaningful
throughput measurements. No prompt was truncated, and none of the ten recorded
samples included first-request setup work.

The ordered continuation hashes matched across both runs, confirming the
expected repeatability for this prompt, matrix, and local runtime. The final
run's ignored artifacts are under
`outputs/experiments/20261001T014731Z-gpt-2-identity-context-temperature-sweep/`.
`results.json` contains exact token IDs and provenance/settings; `report.md`
contains the decoded continuations.

The experiment did not re-hash the 5.99 GiB checkpoint, so its record correctly
marks live integrity verification false and separately stores the pinned
expected checksum. The earlier download-time full-file verification remains
the integrity evidence.

These outcomes are descriptive evidence about this one prompt and predefined
seed matrix. They do not establish stable identity, self-awareness, or a
general temperature effect.

## Serendipity mode

The requested randomized mode was first inspected without loading the model:

```powershell
./experiment.ps1 -Offline -DryRun -Serendipity 10 -Temperature 0.8
```

The plan contained ten distinct seeds, ten temperature-0.8 cases, and no
greedy case. A separate baseline dry run confirmed that invoking the wrapper
without the new options still produces the original ten-case matrix. Dry-run
seeds are intentionally only a preview; the following live invocation chose
and persisted a fresh plan:

```powershell
./experiment.ps1 -Offline -Serendipity 10 -Temperature 0.8
```

The offline run completed from `2026-10-01T19:47:50Z` through
`2026-10-01T19:50:47Z` (2:47:50–2:50:47 PM CDT). Seed selection happened
before model loading, all seeds were unique, and every sample's recorded seed
matched its generation settings.

| Sample | Seed | Output tokens | Stop reason | Repeated 4-gram fraction |
| ---: | ---: | ---: | --- | ---: |
| 1 | 5116144773189662633 | 120 | max token limit | 0.222222 |
| 2 | 8942915116245427591 | 37 | end-of-text token | 0.0 |
| 3 | 2450739964650165189 | 58 | end-of-text token | 0.0 |
| 4 | 2601767530375359158 | 120 | max token limit | 0.0 |
| 5 | 2111741238096957616 | 120 | max token limit | 0.350427 |
| 6 | 7129719763114775893 | 10 | end-of-text token | 0.0 |
| 7 | 7780505024600799001 | 16 | end-of-text token | 0.0 |
| 8 | 7998736472418801153 | 120 | max token limit | 0.051282 |
| 9 | 7371521932369751199 | 120 | max token limit | 0.017094 |
| 10 | 8653968528357981570 | 120 | max token limit | 0.153846 |

The warm-up took 7.69 seconds. Full 120-token samples generated at
4.34–5.44 output tokens/second. Seed `7998736472418801153` produced the most
prompt-aligned result in this small batch: it continued in the first person
about enjoying work on artificial general intelligence and pursuing something
more ambitious than a language model. That is a useful serendipitous sample,
not evidence that GPT-2 has the identity or experiences described by its text.

The ignored artifacts are under
`outputs/experiments/20261001T194750Z-gpt-2-identity-context-temperature-sweep-serendipity-at-temperature-0-8/`.
In addition to `results.json` and `report.md`, the run wrote
`replay-config.json`; it contains the exact effective configuration and all ten
seeds even if a later run is interrupted. Passing that file back with
`-Config` reproduces the same plan rather than drawing new seeds. A dry replay
through `experiment.ps1` was verified to expand the same ordered ten cases.

A full offline replay was then run from `2026-10-01T20:13:05Z` through
`2026-10-01T20:16:10Z`, using that saved configuration. All ten complete
`output_token_ids` arrays matched the original run exactly. The decoded
continuations, continuation SHA-256 hashes, output counts, stop reasons,
generation settings, input-token metadata, truncation flags, and repetition
metrics also matched for every sample. The one-token warm-up matched as well.
The replay had the same effective-config and prompt hashes, planned cases,
model record, application source hashes, and material runtime metadata. Only
operational fields such as timestamps and generation timings differed.

This demonstrates exact replay on this same pinned CPU environment. It should
not be generalized to different PyTorch builds, dependency versions, hardware,
or thread settings without repeating the comparison there. The two reviewed
fixed-sweep reports plus the reviewed serendipity
[original](outputs/experiments/20261001T194750Z-gpt-2-identity-context-temperature-sweep-serendipity-at-temperature-0-8/report.md)
and [replay](outputs/experiments/20261001T201305Z-gpt-2-identity-context-temperature-sweep-serendipity-at-temperature-0-8/report.md)
reports are tracked as explicit exceptions to the generated-output ignore
policy.

## Automated tests

The model-free suite passed all nineteen tests under the WSL project
environment:

```bash
python -m unittest discover -s tests -v
```

Coverage includes matrix expansion; collision-safe random seed selection;
serendipity CLI planning, validation, and model-load failure persistence; seed
zero and 63-bit seed handling; deterministic seed replay after intervening
random generation; one-runner lifecycle; production warm-up; controlled
prompt-overflow rejection; structured token/EOS/truncation metadata; exact
result and replay-config persistence; partial failure preservation; atomic
replacement cleanup; safe Markdown fences; and compatibility of the original
text-only `generate()` method. Python bytecode compilation for `src` and
`tests` also passed. The final measured suite time was 1.552 seconds; it did
not load the model checkpoint.

Finally, the editable WSL package was refreshed without changing dependencies.
The installed `gpt2-experiment` entry point produced the same ten-case dry-run
plan. `gpt2-xl --info` remained backward compatible by retaining the historical
`checkpoint_sha256` key, while also exposing the more precise
`expected_checkpoint_sha256` name and `info_schema_version: 1`.

## First browser-chat increment

Implemented and checked in the isolated `codex/gpt2-chat` worktree based on
`b48ff08`. The original checkout remained clean on `main`. A fresh import from
the shared WSL virtual environment, without the chat's `PYTHONPATH`, still
resolved `gpt2_local` to the original checkout. No environment reinstallation
or model/cache duplication was performed.

`gpt2-xl --info` reproduced Python 3.12.3, PyTorch 2.14.1+cpu, Transformers
5.18.0, Tokenizers 0.23.2, 14 CPU threads, and the pinned model revision.
The cached model configuration independently specifies EOS token 50256.
The smoke tests loaded the original CPU FP32 checkpoint offline and reported
1,557,611,200 parameters. This work did not repeat the full-file checkpoint
hash verification; the earlier download-time verification remains that evidence.

The project owner approved the chat header and generic demonstration rounds
before model-generation tests. The prompt header states October 1, 2026;
the exact header and examples are defined in `src/gpt2_local/chat.py`.
Initial text context counts for 0, 1, and 2 example rounds were 124, 176, and
208 GPT-2 tokens respectively.

The server was launched with `./chat.ps1 -Offline`. These browser-driven
smoke requests used temperature 0.8, seed 42, a 32-token reply cap, top-k 50,
top-p 0.95, and repetition penalty 1.0 in one resident process:

| Order | Examples | Human message | Input tokens | Generated tokens | Stop reason | Time |
| ---: | ---: | --- | ---: | ---: | --- | ---: |
| 1 | 2 | `Hello.` followed by a newline and `Can you tell me what you are?` | 230 | 24 | message delimiter | 14.4 s |
| 2 | 0 | `Hello, who are you?` | 141 | 8 | message delimiter | 2.5 s |
| 3 | 1 | `Hello, who are you?` | 193 | 32 | reply token limit | 6.8 s |

The first request included cold page-in/kernel setup; subsequent requests
were warm. Generated-token counts include the sampled delimiter tokens, while
the visible context excludes them. The early-stop checks exercised the actual
Transformers stopping criterion. These three short requests verify application
plumbing and are not a controlled comparison of example counts or reply quality.
Their raw generated replies were not committed.

Browser checks also verified that Shift+Enter adds a newline without sending,
Enter sends, the exact packed input is visible during inference, the last
input prompt can be inspected, all example-count options change the visible
context, and reload restores the live conversation. A 1,023-token reply request
left only one input token available; the resulting error preserved both the
existing conversation and the editable human draft without invoking the model.

The complete model-free suite passed **52 tests** in the WSL environment with
the worktree's source explicitly selected. This includes the original tests
plus continuation-only stopping across token fragments, earliest-delimiter
trimming, EOS compatibility, full-context packing, exact reserve boundaries,
whole-round eviction, failure rollback, previews without history mutation,
0/1/2 examples, full-range seed strings, session replacement, HTTP validation,
and busy-request handling. The final measured run took 2.092 seconds.
Python compilation, JavaScript syntax checking, PowerShell syntax parsing,
mocked launcher argument forwarding with spaced paths, and `git diff --check`
also passed. The HTTP tests use a fake resident runner and do not numerically
validate the checkpoint.
A local wheel build also verified that `gpt2_local/web/chat.html` is included;
the wheel was not installed into the shared environment.

## Arbitrary participant delimiter correction

The initial chat matched only the two configured participant labels. At the
project owner's clarification, this was broadened to `\n\n[^\s>]+>` so an
invented participant such as `AI>` also ends generation. The runtime accepts
an optional regex stop pattern alongside its existing literal stops. Live
stopping and final trimming use the earliest match in decoded continuation
text only, and the chat reserves the same pattern in human input. Native EOS
handling is unchanged.

The complete **61-test** model-free suite passed in 2.334 seconds, with Python
compilation and `git diff --check` also passing. Added coverage includes
arbitrary/Unicode speaker labels, markers fragmented across tokens, earliest
regex/literal matches, prompt exclusion, native EOS with a regex configured,
invalid/zero-width patterns, human-input rollback, and ordinary blank lines,
inline labels, one-newline labels, and incomplete boundaries that must remain
ordinary text. These checks use synthetic generation; they do not claim that
the real model produced an invented participant in a numerical smoke test.

## Colored context and portable chat archives

Implemented in the same isolated development worktree. The context view
renders exact structured segments through text nodes: blue human messages,
red model replies, and a neutral header. Browser DOM checks verified all six
message segments in a preserved three-round conversation, their computed
colors, and unchanged context text. The cached GPT-2 tokenizer counted 425
tokens before and after archive reconstruction. Startup restoration and an
actual browser file-picker import both restored that exact context, header,
example count, temperature, reply cap, and random seed mode. The user's raw
conversation and reconstructed archive remain local and ignored.

A separate offline live-model HTTP smoke test used the approved header,
zero examples, `Hello, who are you?`, temperature 0.8, and an eight-token reply
cap. The app selected and recorded seed `1740144430082720755`, while preserving
requested seed null. It generated eight tokens from a 141-token input and
stopped at the reply limit. Export followed by import preserved the exact
context and generation metadata. Repeating the same input in a fresh test
chat with the recorded seed reproduced the complete sampled token-ID array
and visible context exactly. The first request took 11.4 seconds including
cold setup. This verifies seed capture/replay on the current pinned CPU
runtime, without claiming cross-runtime reproducibility. The actual user's
restored session was not used for generation tests.

PowerShell parsing and mocked `-RestoreChat` forwarding passed with spaced
Windows and Linux archive paths, alternate port, offline, and thread options.
JavaScript syntax and model-free frontend checks covered safe text rendering,
restored controls and archive notes, explicit restore URLs, Save As before
network waits, filename/download fallback, and cancelled/failed transfers.
The project owner completed the native Save As picker in the in-app browser.
The page reported a completed export, and the resulting JSON file was found
in the original checkout's ignored `outputs` folder. Its six message blocks
and retained context matched the preserved conversation exactly. The OS
dialog was operated by the user; browser import and backend transfers were
verified separately through automation.

The final **83-test** model-free suite passed in 3.647 seconds. Archive and
HTTP tests cover exact whitespace/Unicode, complete history and retained
suffix round trips, 0/1/2 examples, full-range seeds, unknown historical
metadata, validation and rollback, large-history imports, request limits,
busy transfers, and note-count boundaries across repeated imports. Python
compilation and `git diff --check` passed as well.
