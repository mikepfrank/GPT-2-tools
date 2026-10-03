# Verification record

Verified through **2026-10-03**, including the local tools, public-service
preparation, isolated Amazon Linux runtime installation, and short real-model
throughput trials, private backend integration, HTTPS public deployment, and
real-model browser/API checks. Native browser file export and an actual reboot
remain unverified; the cgroup budget has not been tested with a cold host cache.

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

## Seed chosen at chat initialization

At the project owner's correction, a blank seed now selects one random
63-bit value when New chat starts. It is stored, displayed, and exported
before any model reply. Blank reply/export controls reuse the stored value;
explicit changes remain supported. The runtime reseeds before each reply
with that active seed, so a turn can be replayed independently. This replaces
the preceding increment's per-reply seed selection.

Application imports of older files with active seed null select a seed for
future replies and add an explanatory note. The archive parser remains
lossless, and unknown historical generation metadata stays null. Restoring
the user's chat preserved its exact 425-token context and all three rounds;
the newly chosen future seed appeared in the browser control. A separate
live HTTP check verified seed recording before generation, blank-export
reuse, and import continuity without generating any additional model text.
The prior exported conversation was preserved, and a separate ignored local
backup includes the active future seed.

All **86 model-free tests** passed in 3.786 seconds. Added coverage checks
selection once at initialization, reuse over multiple replies/exports,
separate chat initialization, legacy import without fabricated provenance,
explicit overrides, and failure rollback. Python compilation and diff checks
passed. JavaScript syntax and mocked frontend checks verified seed display
on creation/import/reply/export, preserving other pending controls, and
leaving typed controls unchanged on cancellation or failure.

## Last-reply regeneration (2026-10-02)

Implemented in the existing `codex/gpt2-chat` worktree. Regeneration increments
the active/requested seed, samples the exact recorded input prompt, and
replaces only the most recent reply after successful inference. Earlier
candidates retain their text and generation records in version-2 exports.
Version-1 imports remain supported.

The complete **111-test** model-free suite passed in 4.775 seconds under WSL
with the development source selected through `PYTHONPATH`. New checks cover
explicit/blank seeds and maximum-seed wraparound; original prompt reuse after
eviction and repeated retries; unchanged state during previews; deep-copied
candidate history; greedy/overflow/eligibility rejection; unrelated saved
prompt rejection; sampling and token-accounting rollback; generic delimiter
trimming; HTTP validation and busy requests; and version-1 migration/version-2
candidate restoration. Python compilation and `git diff --check` passed.
JavaScript syntax checking and five mocked frontend scenarios passed,
including full-range seed handling, greedy rejection, rollback, and busy state.

The owner saved their ongoing chat before the server restart. Startup import
of that version-1 file restored all eight rounds, its exact 856-token context,
saved header/date, and active seed. No generation test modified that session.

A separate offline live-model chat used the approved header with the current
October 2 date, zero examples, `Hello, who are you?`, temperature 0.8, and an
eight-token reply cap. The first reply used seed 42 and a 141-token input,
sampling eight tokens in 10.6 seconds including cold setup. Clicking the real
browser's Regenerate button used seed 43 with that identical input, sampling
eight different token IDs in 2.3 seconds. The transcript still contained one
human/model round, with the original candidate and all its metadata retained
as a previous response. The seed control updated to 43, and an unsent draft
remained unchanged.

Version-2 export/import preserved the exact current context, both candidates'
metadata, active seed, and regeneration eligibility. Repeating the same input
in two fresh test sessions with seeds 42 and 43 reproduced each respective
complete sampled token-ID array and displayed context exactly, in 2.4 and
2.5 seconds. This verifies replay on the same pinned CPU runtime; it does not
establish equivalence across runtimes. The raw test archives remain ignored.

An additional browser check set the reply cap to 1,023 before regeneration.
The app rejected it because the original 141-token prompt leaves at most 883
reply tokens. The current candidate, seed 43, typed controls, and unsent draft
were preserved without inference. The owner's restored conversation was left
open with the new button enabled for their hands-on test.

## Context omission hint (2026-10-02)

The first successful context eviction now appends literal `\n\n...\n\n`
to the header once. The marker's token cost participates in packing before
inference; it can require another whole-round eviction. Preview proposes the
marked header without committing it, and failed generation preserves the
original header/history. The existing archive header field stores the marker,
so no schema change was required. Legacy archives remain exact on import and
regeneration and receive the hint on the next new message.

All **125 model-free tests** passed in 4.730 seconds, with Python compilation
and diff checks passing. Fourteen new tests cover no-eviction behavior,
matching preview blocks, one-time marker persistence, its exact reserve
boundary and extra eviction, overflow and inference/accounting rollback,
example eviction, legacy version-1/version-2 import and original-prompt
regeneration, marked archive round trips, and New chat reset. Existing
eviction boundary tests were updated to reserve the marker's token cost.

The owner exported their latest chat before restart. Startup restoration
preserved all nine rounds, the exact 893-token retained context with one
omitted round, and all candidate/generation metadata. A live HTTP preview of
a short next message added the marker, packed 807 input tokens beside a
120-token reply reserve, and left the stored session unchanged. The browser
showed the restored French exchange; no test sampled from that live session.

A separate synthetic archive held one explicitly authored placeholder round
in a 1,016-token context. Using the real cached tokenizer, the short approved
`Hello, who are you?` input with an eight-token reply reserve evicted that round
and included the omission hint. The offline model generated eight tokens from
the exact 145-token preview input in 11.7 seconds including cold setup, using
temperature 0.8 and seed 42. The hint appeared once in the visible header,
the counter showed one removed round, and the full transcript retained both
rounds. Export/import preserved the exact marked context and generation
prompt; regeneration preview reproduced that saved prompt exactly. Raw test
archives and the owner's conversation remain local and ignored.

## Ellipsis spacing correction (2026-10-02)

The header suffix is now `\n\n...`, leaving the next Human delimiter to
provide exactly one blank line: `...\n\nHuman>`. A new-message preparation
also compacts the older `\n\n...\n\n` suffix. Saved contexts and historical
generation prompts remain unchanged on import and regeneration; only a
successful new reply commits the compact header.

The existing **125-test** suite passed in 4.477 seconds, with compilation and
diff checks passing. Existing omission checks now assert the exact boundary
and cover version-1/version-2 imports both with an unmarked header and with
the older spaced marker, preserving original regeneration prompts before
compacting the next new input.

The owner's latest eleven-round archive restored its exact 926-token context
and all generation/candidate metadata. Live previews verified the compact
boundary in an 826-token next input and unchanged original regeneration input.
A separate test chat imported an older marked archive and generated one token
from a compact 165-token input in 1.6 seconds, using temperature 0.8 and seed
42. Browser inspection and a screenshot confirmed one blank line after the
ellipsis. No generation test altered the owner's restored chat.

## Public anonymous service preparation (2026-10-02)

Work is isolated on `codex/gpt2-public`, starting at chat commit `3657a81`.
The existing local server, its source, and the shared editable WSL installation
were not used for the public-service smoke test or reinstalled.

The complete **157-test Python suite** passed in **8.664 seconds** using the
existing pinned WSL environment with this worktree's `src` selected explicitly
through `PYTHONPATH`. This includes the original 125 tests and 32 new
fake-runner/HTTP tests for public archive restoration, independent browser
chats, FIFO single-worker execution, previews/transfers during generation,
queue/job/rate/connection limits, expiry, transactional failure and regeneration,
origin/proxy validation, tokenizer synchronization, cooperative decoding
deadlines, and host/cgroup memory guards before runner construction.

All **16 frontend lifecycle tests** passed under bundled Node **24.19.0**.
They cover durable transcript/draft/settings recovery, server restarts, offline
export, 63-bit seed handling and export seed updates, duplicate tabs, queue
errors with explicit rejection acknowledgements, pending job recovery,
transient polling failures, uncertain POST acknowledgements,
import/regeneration rollback, unavailable/full storage, and near-limit Unicode
exports that remain within the import byte limit without losing history.
Python compilation, Bash setup syntax, and `git diff --check` passed.

A separate localhost server on port 8899 used only a synthetic runner and
synthetic token counting. A real in-app browser completed a message, then
reloaded during regeneration and recovered that same queued job. The active
seed advanced from 42 to 43, the transcript still contained one generated
human/model round after its two authored examples, and the unsent draft
survived. The screenshot and temporary helper remain in ignored
`outputs/chats/`. Their displayed TPS is synthetic and provides no GPT-2
performance evidence.

The benchmark's dry run saved a plan without importing Torch or loading the
model. An intentionally unreachable 1,000 GiB available-memory floor caused
the non-dry benchmark to decline with exit code 2 before model loading,
using the system Python that has no Torch installation. These checks verify
planning and the guard, not inference or numeric performance.

The October 2 read-only SSH inspection found Amazon Linux 2023.9, x86_64, two vCPUs
(Xeon Platinum 8175M, reporting one core/two threads), 7.6 GiB total RAM,
about 4.8 GiB available RAM, no swap, and about 100 GiB free disk. Existing
Astra, Arcturus, Caddy, and other user processes share the host. System Python
is 3.9.25; namespaced Python 3.11.14 with ensurepip is installed. These are
dated observations of the original host. At that inspection, no package
installation, model download/load, service change, proxy edit, or throughput
trial had been performed. The original available RAM failed the unchanged
FP32 model's capacity requirement; the owner subsequently upgraded the host.

## Amazon Linux installation and short inference trials (2026-10-03)

After the owner's upgrade and authorization, the host reported Amazon Linux
2023.9 on x86_64, kernel `6.1.158-180.294.amzn2023.x86_64`, and an Intel Xeon
Platinum 8175M at 2.50 GHz with four vCPUs: two cores and two SMT threads per
core. Total RAM was **15.42 GiB**, initial available RAM **14.57 GiB**, swap
absent, and approximately 260 GiB free on the 320 GiB disk. These remain
dated observations, not a guarantee of later spare capacity.

Source commit `663c413865ab2264cf56d8743f33e8abb1edec7e` was transferred by
Git bundle and cloned as `gpt-2` into `/home/gpt-2/GPT-2-tools`; its origin
was set to the project's GitHub repository without pushing. Only namespaced
Python 3.12 prerequisites were installed with DNF, without an OS upgrade.
System Python remained **3.9.25**. The isolated environment and cache were:

```text
Environment  /home/gpt-2/.venvs/gpt2-public
HF_HOME      /home/gpt-2/.cache/gpt2-public/huggingface
Python       3.12.12
PyTorch      2.14.1+cpu
Transformers 5.18.0
Tokenizers   0.23.2
Model        openai-community/gpt2-xl, CPU FP32, 1,557,611,200 parameters
Revision     15ea56dee5df4983c59b2538573817e1667135e2
```

`scripts/setup-amazon-linux.sh --download-model` completed. `pip check`
reported no broken requirements, and the entire **5.99 GiB** safetensors file
was hashed and matched
`0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a`.
The **157-test Python suite** passed on Amazon Linux in **8.508 seconds**.
Those automated tests use synthetic runners; the measurements below provide
separate real-model integration evidence.

Two sequential offline benchmark processes used one resident model each:

```text
Thread trial   --threads 1 2 4 --samples 1 --max-new-tokens 32
Context trial  --threads 2 4 --samples 1 --max-new-tokens 32 --near-full-context
Common         temperature 0, seed 42, top-k 50, top-p 0.95, repetition penalty 1.0
```

The short-chat prompt had 39 tokens; the near-full-context prompt had exactly
992, reserving 32 of GPT-2's 1,024 tokens for the continuation. Every sample
generated all 32 requested tokens, stopped at `max_new_tokens`, and reported
no input truncation. Each thread count received a separate one-token warm-up.
The five short-prompt samples had identical greedy output token IDs and text
hashes; the two near-full samples also matched across two/four threads. This
is evidence for these recorded samples, not a general claim of equivalence
across hardware, precision, or arbitrary prompts.

| Trial | Input tokens | Threads | Output tokens | Generation seconds | Output tokens/second | Process CPU, % of four-vCPU host | Process-lifetime peak RSS, GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Thread | 39 | 1 | 32 | 20.44807 | 1.56494 | 24.95 | 6.240 |
| Thread | 39 | 2 | 32 | 12.10384 | 2.64379 | 49.85 | 6.240 |
| Thread | 39 | 4 | 32 | 11.54576 | 2.77158 | 98.25 | 6.240 |
| Context | 39 | 2 | 32 | 12.46027 | 2.56816 | 49.75 | 6.240 |
| Context | 992 | 2 | 32 | 30.23434 | 1.05840 | 49.51 | 7.928 |
| Context | 39 | 4 | 32 | 12.60798 | 2.53808 | 95.78 | 7.928 |
| Context | 992 | 4 | 32 | 29.64731 | 1.07936 | 98.29 | 7.928 |

Generation TPS includes prompt prefill, decoding, and stopping; it is neither
time to first token nor isolated decode speed. Model construction wall time
was **6.33134** and **6.08605 seconds**, including initial runtime imports.
The first one-token warm-ups separately paid lazy page-in/kernel setup costs
of **2.41501** and **1.91170 seconds**. Pre-load available RAM was 14.39 and
14.36 GiB, respectively; both runs passed the unchanged 8 GiB floor without
override. RSS is a process-lifetime high-water mark, not per-request memory;
the second trial's four-thread short sample inherited the earlier near-full
peak. CPU percentages derive from process user/system time normalized across
four logical CPUs, not a CloudWatch whole-instance observation.

The runs generated **224 measured + 5 warm-up = 229 tokens**: 99 in the first
process and 130 in the second. Monitoring used one-second `vmstat` samples.
Excluding each command's initial since-boot average row, the first run's 56
intervals reported zero steal; the second run's 95 intervals reported up to
3% steal (0.084% mean). Swap-in/out were zero throughout. These brief trials do not establish steady
performance after burst-capacity depletion or under concurrent traffic.
**Two Torch threads are the recommended initial service setting**: four
threads occupied nearly all available vCPUs for little or no gain, while two
threads averaged roughly half the host. The service still uses one inference
worker; request serialization and memory guards remain in place.

The before/after preservation checks found `astra-chess`, `or-chess`, and
`astra-caddy` still active with unchanged MainPIDs **4149 / 4313 / 4341** and
unchanged configuration/unit-file hashes. No proxy edit, persistent GPT-2
web-service installation, or public endpoint was added. The isolated runtime
and inference trials preceded the temporary private API test and subsequent
public deployment/browser checks recorded below.

Exact prompts/token IDs, generated token IDs/text hashes, dependency and source
provenance, resource snapshots, and UTC timing remain in the ignored files
`outputs/benchmarks/lightsail-threads-20261003.json` and
`outputs/benchmarks/lightsail-context-20261003.json`. The installation and test
logs remain ignored under `outputs/deploy/`. The benchmark records the expected
checkpoint digest; the actual whole-file verification came from installation.
Raw JSON, logs, model weights, and caches have not been added to Git.

The copied benchmark JSON files matched their server copies with these SHA-256
digests:

```text
lightsail-threads-20261003.json  56704377229882c707f5609eebf6de8ddcefe2d857f3bcfff8e994b226a10b89
lightsail-context-20261003.json  8ad1805ce9c84dda33aca6e30331e1d687a27c818c7d6fb72e140c04e5e152d6
```

## Private real-model public-backend integration (2026-10-03)

A temporary systemd unit, `gpt2-public-smoke-20261003`, ran the installed
public backend as `gpt-2`, with niceness 10, `ProtectSystem=strict`,
`ProtectHome=read-only`, `MemoryHigh=8704M`, `MemoryMax=9G`,
`MemorySwapMax=0`, and `RuntimeMaxSec=180`. The service listened only on
`127.0.0.1:8899`, allowed origin `http://localhost:8899`, and used offline
inference, two Torch threads, a 32-token reply cap, and the unchanged 8 GiB
available-memory pre-load floor. No persistent unit was installed or enabled.

The private HTTP client observed ready/idle health, started a zero-example
chat, obtained its exact preview, submitted **one** generation, and polled
that job through completion. Export/import preserved the archive exactly
except for its export timestamp; the source archive was unchanged, a separate
new chat stayed empty, and final health was ready/idle. The real generation
used **144 input tokens** and generated **13 output tokens** before the chat
boundary stop (`stop_sequence`), in **7.106534691 seconds** (**1.82930 generated
tokens/second**). Raw token metadata includes stopping tokens; the delimiter
was omitted from the visible reply/context. Settings were temperature 0,
seed 42, and at most 32 reply tokens. The full HTTP client flow took
**8.05260 seconds**, including polling and transfers; it is not time to first
token or service startup time.

This was the first generation in that process and recorded
`cold_start_included=true`; the OS file cache was already warmed by the prior
trials, so it is not a cold-machine benchmark. The unit later stopped at its
configured 180-second runtime limit, producing the expected systemd timeout
result after the API job had already passed. No GPT-2 process or port-8899
listener remained. Pre-job cgroup memory events were zero; the cgroup was
already collected at the post-job inspection. This short test therefore does
**not** validate near-full-context memory, the soft/hard memory limits under
load, or sustained service operation.

Final preservation checks again found the three existing services active with
the same MainPIDs. The Astra and Arcturus application roots returned HTTP 200
when queried with their configured Host names, `astraplayschess.com` and
`arcturuschess.com`. Their configuration/unit hashes remained unchanged.

The exact API requests/responses, prompt, sampled token IDs, reply hash,
generation metadata, and checks remain in ignored
`outputs/deploy/public-api-smoke-20261003.json` (SHA-256
`7035942230611429e5479447382e8a5a02d1b1006a55460acc81c16d28e15016`).
This verified private backend integration with the real model before the
public deployment and browser checks recorded below.

## Public HTTPS deployment and browser/API checks (2026-10-03)

The service was deployed from source
`e627fd3c5be59d3687e154fd228ab8345949aea6` and is active/enabled as `gpt-2`
(observed PID **39135**), with two Torch threads, `MemoryHigh=8704M` (8.5 GiB),
`MemoryMax=9G`, and exact origin `https://gpt2-chat.arcturuschess.com`.
The backend remains on `127.0.0.1:8899`. Autostart was verified with systemd;
no reboot test was performed.

The canonical name `gpt2-chat.arcturuschess.com` and alias
`gpt-2.arcturuschess.com` both resolve by CNAME through `arcturuschess.com`
to **16.145.194.176**. Authoritative DNS verified CNAME/A; public Cloudflare
DNS-over-HTTPS verified A and absence of AAAA.
The instance retained a cached NXDOMAIN, so preflight used public DNS-over-HTTPS
and certificate/route checks used explicit loopback resolution without changing
its resolver. Let's Encrypt issued certificates for both names at
**22:47:26 UTC**. HTTP redirects 308 to HTTPS for each name; the HTTPS alias
then redirects 308 to the canonical host, preserving path/query. A public
request with a disallowed Origin returned 403 and `accepted:false`.

Caddy **2.11.4** retained PID **4341**, `admin off`, and its file-based
`caddy run` startup (no resume/watch/API workflow). The complete candidate was
validated as `astra` with the same HOME/XDG environment under shared lock
`/run/lock/astra-caddy-config.lock`. A `cp -a` backup was saved at
`/home/astra/.config/astra-chess/Caddyfile.before-gpt2-20261003T224724Z`.
The live file's inode was retained by an in-place write/fsync; host and
process-mount-namespace hashes matched before SIGUSR1. The successful reload
log, valid HTTPS API, redirects, and existing sites were checked. The original
Caddyfile bytes remained its prefix, and the three existing unit-file hashes
and chess PIDs **4149 / 4313** were unchanged. Caddyfile SHA-256 changed from
`4d7ca60383b6ad7a27c97c5ee3c36327a5d8489894a9edfd2962e56fe362b529`
to `fa0162212b9b205b9b186c16969a8201bf35b1d3a846d4adf6ab7f98dc681d03`.

A real in-app HTTPS browser generated a reply from **227 input / 9 generated
output tokens** with greedy decoding, seed 42, and a 16-token cap, in about
**7.8 seconds (1.16 tokens/second)**, including first-request page-in/setup.
Reload during inference recovered the same pending job without resubmission;
reload after completion restored the validated chat and controls. Navigating
through the alias redirected to the canonical URL and restored that same chat.
The screenshot remains ignored at
`outputs/deploy/public-browser-live-20261003.jpg`. Native Export/Save As
automation did not complete its download wait; it did not establish a saved
browser file. API export/import verification below is separate.

A second HTTPS test, **22:49:08.344340–22:49:44.731901 UTC**, used a
synthetically constructed near-full message/prompt with the **real GPT-2 model**:
992 input tokens, 32 generated output tokens, greedy decoding, seed 42,
`max_new_tokens` stop, and no truncation. One generation POST completed in
**30.956060646 seconds (1.033723 tokens/second)**, with
`cold_start_included=false`. Exact preview, one canonical round, source-archive
preservation, export/import except timestamp, independent empty chat, and idle
health before/after all passed. Full token IDs/metadata remain in ignored
`outputs/deploy/public-near-context-smoke-20261003.json`, SHA-256
`1c3412c158ffbf5cc001901c944b4b94382f3c6b513d22a453ef7e1b94493d05`.

After that request, process VmHWM was **8,178,144 KiB (7.799 GiB)** and RSS
**7,756,736 KiB (7.397 GiB)**; cgroup peak was **2,052,915,200 bytes
(1.912 GiB)** and all `memory.events` counters were zero. The lower cgroup
figure is consistent with previously cached/shared weight pages remaining
charged elsewhere; page ownership was not directly inspected. This follows
the kernel's [memory-ownership rules](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory-ownership).
Available host RAM was **13,730,713,600 bytes (12.79 GiB)** of
16,558,186,496 total, with no swap. The host cache was not flushed: this proves
the observed warm-cache request passed, not a 9 GiB cold-cache cgroup budget or
sustained public capacity.
