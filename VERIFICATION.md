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

## Dedicated-domain transition (2026-10-03 CDT)

At **2026-10-04 00:03:42–00:03:49 UTC** (October 3 in the owner's CDT
timezone), the authorized migration made **https://gpt2-chat.com** canonical.
Public DNS-over-HTTPS verified A **16.145.194.176** and no AAAA. Both temporary
subdomains now redirect HTTPS 308 to the new domain, preserving path/query.
The owner confirmed the only existing chat was exported before changing the
browser-storage origin. Earlier measurements above remain tied to their old
origin and source revisions.

The idle GPT-2 service alone restarted (queue 0/running false), changing PID
**39135 → 43867** and exact allowed origin to `https://gpt2-chat.com`.
The unit stayed unchanged/enabled; CPU FP32, two threads, the 8 GiB startup
floor, and 8.5/9 GiB soft/hard limits were unchanged. The transition used
checkout `5c6de2a968e1e2474260bfd6bf65704a6bc44fa9`. All four unit hashes,
including GPT-2, were unchanged; only its environment and Caddyfile changed.

Caddy retained PID **4341**, and chess PIDs **4149 / 4313** and the original
chess-configuration prefix were preserved. Under the shared lock, the complete
candidate passed validation in Caddy's own environment; backup, inode-preserving
write/fsync, namespace verification, SIGUSR1 success, TLS/health, and existing
chess HTTP 200 checks passed. Backups are
`/home/astra/.config/astra-chess/Caddyfile.before-domain-20261004T000342Z` and
`/etc/gpt2-public.env.before-domain-20261004T000342Z`. SHA-256 digests at that transition:

```text
Caddyfile        a1f03617831839dcb493d7311d513ad5259a5d66fc6d47e5c181ba45cfef2311
gpt2-public.env  b1a380290d49e4abd19d8ba61eebfff4d81271ff05eeb5a23eeea057ab59fdba
```

Let's Encrypt issued the new certificate at **00:03:45.368892 UTC**. External
laptop HTTPS returned 200, HTTP redirected 308 to HTTPS, and a disallowed-Origin
POST `/api/new` returned 403 with `accepted:false`.

A real in-app browser initialized the new origin and completed one greedy reply
with **227 input / 9 generated output tokens**, seed 42 and a 16-token cap,
in about **7.6 seconds (1.18 tokens/second)**, including first-request page-in.
The displayed reply matched the earlier browser check; token-ID equality was
not tested. The screenshot remains ignored at
`outputs/deploy/gpt2-chat-com-live-20261003.jpg`. This domain-only migration
did not repeat the full-context exercise or test reboot behavior.

## Ignore public request query parameters (2026-10-03 CDT)

At **20:04:33–20:04:35 CDT** (2026-10-04 01:04:33–01:04:35 UTC), the
canonical GPT-2 Caddy site gained `rewrite * ?`, discarding all query parameters
before proxying. The [Caddy 2.11.4 source](https://github.com/caddyserver/caddy/blob/v2.11.4/modules/caddyhttp/rewrite/rewrite.go#L54-L68)
confirms that this empty, query-only rewrite preserves the request path and method. No Python code,
runtime, model, service environment, or unit changed.

The complete candidate passed validation/adaptation in Caddy's own environment.
Under the shared maintenance lock, an inode-preserving write/fsync, matching
mount-namespace bytes, and a successful SIGUSR1 reload log after a fresh journal
cursor passed. The backup is
`/home/astra/.config/astra-chess/Caddyfile.before-query-20261004T010433Z`.
Caddyfile SHA-256 changed from
`a1f03617831839dcb493d7311d513ad5259a5d66fc6d47e5c181ba45cfef2311` to
`700beceabe7ed95cf8a08d2e20456abbae03a0fd4569aedf6cad92989438e5d9`.
The GPT-2 environment hash remained
`b1a380290d49e4abd19d8ba61eebfff4d81271ff05eeb5a23eeea057ab59fdba`,
with its exact bytes verified against the pre-change snapshot. All four
unit-file bytes were unchanged.

Before the change, a Facebook-style root URL returned 404. Afterwards, local
TLS checks passed root/health requests with `fbclid`, repeated keys, empty/bare
values, encoded delimiters, malformed query escapes, and an empty query. The
served page bytes matched the plain root. A query-bearing POST `/api/new`
returned a valid version-2 archive, while a disallowed Origin still returned
403. Unknown and escaped paths remained 404. Both chess sites remained HTTP 200.

Twenty independent laptop requests through public DNS/TLS then passed canonical
HTTP/HTTPS, both aliases over HTTP/HTTPS, page-byte equality, health, new-chat
POST, Origin rejection, and unknown/escaped paths. Header-only requests with
`Content-Length: 65537` returned 413 on `/api/new` both with and without a query.
An initial full oversized upload was reset while the backend rejected it early;
the header-only checks verified the established limit without that upload race.
No inference or existing conversation was accessed. All PIDs remained unchanged:
GPT-2 **43867**, Caddy **4341**, Astra **4149**, and Arcturus **4313**.
Evidence/scripts remain ignored under `outputs/deploy`, including
`query-params-smoke-20261003.json`. This verifies URL handling, not streaming or
new model-performance behavior.

## Local streaming preview (2026-10-03 CDT)

Implementation and checks ran only in `.worktrees/gpt2-streaming`, branch
`codex/gpt2-streaming`, based on `e5971db` plus design `aeff42e` and the public
query-handling cherry-pick `4b5cf86`. No production service, Caddy setting,
shared editable installation, or other worktree was changed for streaming.
Streaming is opt-in through `--enable-streaming`; the previous polling path
remains the default. A separate optional observer reports raw sampled tokens,
without changing sampling arguments or RNG use. Safe cumulative display text
withholds incomplete Unicode and possible next-speaker delimiters. The final
validated result alone commits the chat archive.

The full WSL suite passed **197 Python tests** in 13.525 seconds, including
three cached-tokenizer checks without model weights. The public HTTP subset
also passed **41 tests** on Windows. The final frontend suite passed **30
JavaScript tests**. Synthetic coverage includes token/prompt accounting,
transactional failures, Unicode and split SSE framing, heartbeat delivery,
bounded subscriptions, stalled writers, expiry/shutdown, reconnect/reload,
polling fallback, export isolation, regeneration, seed wrapping, browser
back/forward restoration, and result-body timeouts. A deterministic local-chat
regression also verifies that response transmission no longer retains the
generation lock, which could previously reject an immediate follow-up request.
PowerShell parsing and mocked WSL argument forwarding passed for the new
`streaming-preview.ps1` launcher; that foreground launcher was not used to
load another model while the durable preview was resident.

At **2026-10-04 02:19:45–02:20:34 UTC** (October 3 CDT), a private harness
bound `127.0.0.1:8891` before loading exactly one offline GPT-2 XL runner.
The runtime was Python **3.12.3**, PyTorch **2.14.1+cpu**, Transformers
**5.18.0**, Tokenizers **0.23.2**, and **two Torch threads** on WSL Ubuntu.
All **1,557,611,200 parameters** were CPU `torch.float32`, with the pinned
revision `15ea56dee5df4983c59b2538573817e1667135e2`. The expected checkpoint
digest was recorded; the earlier full-file digest verification was not repeated.
Pre-load available RAM exceeded the 8 GiB guard. The process was started in
a separate session with redirected logs and remains the preview server.

Four sequential trials used the exact message **"Tell me one interesting fact
about stars."**, zero examples, fixed date **2026-10-03**, seed **42**, and a
**32-token allowance**. Greedy temperature **0** and sampled temperature
**0.8** each ran once with streaming off and once with streaming on, using the
same runner. Each pair matched the complete input prompt, settings, input IDs,
raw output IDs, final text, stop reason, counts, and canonical archive after
excluding only `exported_at`, `elapsed_seconds`, and `cold_start_included`.
Observer IDs/counts exactly matched raw results; every safe display snapshot
was a monotonic prefix of final text. Export/import round trips also matched
except their timestamps.

| Trial | Input / generated tokens | Generation seconds | First safe text seconds |
| --- | ---: | ---: | ---: |
| Greedy, stream off, first process request | 143 / 9 | 11.248 | Not observed |
| Greedy, stream on | 143 / 9 | 4.883 | 3.144 |
| Sampled, stream off | 143 / 11 | 4.172 | Not observed |
| Sampled, stream on | 143 / 11 | 4.212 | 2.153 |
| Near context, stream on | 992 / 7 | 18.012 | No safe text |

These are short operational observations. The first request includes page-in
and kernel initialization; OS cache state was not controlled. First-text
times start at `model.generate()` and exclude queue/browser/network latency.
The near-context message began "Consider reversible computing, physical
information, and logical reversibility." and repeated " physics" until the
exact prompt contained 992 tokens. It stopped on a new-speaker delimiter
after seven tokens, all withheld from provisional display. It verifies this
context/stop path, not a 32-token near-context throughput benchmark. During
that trial, lifetime peak RSS reached **7,627,063,296 bytes (7.103 GiB)**;
the measured process CPU time was about two logical CPUs. Swap remained
unused. These laptop measurements do not establish Lightsail streaming latency
or public capacity.

After validation, the harness attached the **same runner** to one inference
worker, with streaming enabled and a 256-token service cap. The live in-app
browser initialized **http://localhost:8891/** and completed the same sampled
star prompt. Regeneration advanced seed 42 to 43. A second regeneration with
seed 44 visibly displayed the first sampled token, **" Tell"**; reloading
while it was pending resumed at **" Tell me how"** and completed at 15 raw
tokens with the final seed still 44. Controls unlocked and no browser warning
or error was logged. This confirms real provisional output and pending-page
recovery, in addition to the synthetic no-resubmission tests.

Private evidence remains ignored under `outputs/deploy`, including
`streaming-preview-parity-20261004T021945Z-341.json` (full prompts, settings,
token arrays, observer snapshots, archives, resources and source-file hashes),
`streaming-preview.log`, and `streaming-preview-active-20261003.jpg`.
The harness's WSL Git lookup returned null HEAD/status because this is a
Windows-managed worktree; its explicit source hashes provide tested-source
identity, supplemented by Windows Git metadata. The preview launched before
the implementation commit, so source-file hashes are the loaded-code evidence.
No source change or model restart was needed to commit the verified files.
The owner has not yet accepted this candidate for production.

An independent HTTP exercise also passed, with exactly **two generation
POSTs**: a reply at seed 42 and regeneration at seed 43, both sampled at
temperature 0.8 with zero examples and a 16-token allowance. The message was
"What are some interesting ideas about reversible computing?" Both requests
used the same 144-token input prompt and generated all 16 allowed tokens.
Model generation took **5.425 / 5.260 seconds**, with first-token callbacks at
**2.292 / 2.140 seconds**. Health stayed ready and reported running during both
generations. Stable streamed text prefixes, canonical raw IDs/settings, final
GET retrieval, export/import equality except `exported_at`, unchanged source
archives, and the exact retained seed-42 regeneration candidate all passed.

The helper disconnected its first SSE after token one. A Windows `localhost`
dual-stack connection fallback added roughly two seconds per connection,
causing its running-GET assertion to race completion. The helper then used
IPv4 with the same localhost Host/Origin, recovered the existing completed
job with `Last-Event-ID: 3`, and received its terminal SSE without resubmitting.
This verifies retained-result reconnect, while the separate browser trial
above verifies recovery during active generation. The complete evidence is
`outputs/deploy/streaming-http-smoke-20261003.json`, SHA-256
`dfe78c9a21128de8ff0863191bcd060bd28155a0b425d3a0415f59688fb0e472`.
The browser was finally left on a fresh chat with the normal 120-token limit
and two examples; the resident preview remained healthy and idle.

## Lightsail streaming preview staging (2026-10-03 CDT)

After the owner accepted the laptop preview and requested a hosted test, a
Git bundle containing application commit
`5389a30f5833c8bcaaf0314f1c0e1c63960f309e` was transferred into the separate
`/home/gpt-2/GPT-2-streaming-preview` checkout owned by `gpt-2`. Bundle SHA-256:
`ad5ed53635117853dbb76d3b80181a8708694c6e75caac932836eb3fee8758e4`.
Production remained at `e5971db` in `/home/gpt-2/GPT-2-tools`; neither editable
installation nor checkout was changed. The full **197-test Python suite**
passed on Amazon Linux in **13.327 seconds**, including cached-tokenizer
checks, without loading model weights. `pip check` found no broken requirements.
The synthetic generation timings printed by these tests are not real TPS.

Live inspection reported **16,558,186,496 bytes (15.42 GiB)** total RAM,
**13,778,664 KiB (13.14 GiB)** available, **4,788,268 KiB (4.57 GiB)** free,
and no swap. Production GPT-2 RSS was **7,174,892 KiB (6.84 GiB)**. Reclaimable
file cache accounts for much of the available-memory figure; no safe combined
budget or shared physical-page behavior was established for two resident
FP32 models. A one-model test window remains the conservative plan.

The preview unit/environment were installed and validated with
`systemd-analyze verify`; it remained **inactive/static**, with a startup guard
requiring inactive production, MainPID 0, and an empty production cgroup,
loopback **8898**, two threads, FP32,
streaming enabled, a 256-token cap, 8 GiB startup floor, and 8.5/9 GiB soft/hard
memory limits. Only systemd's unit definitions were re-read; no service restarted.
An unrelated pre-existing `acpid.socket` legacy `/var/run` warning did not
prevent unit validation. Production and chess/Caddy PIDs remained
**43867 / 4149 / 4313 / 4341**, respectively.

The owner created `test.gpt2-chat.com`; the host resolved it to
**16.145.194.176**. A complete parked Caddy candidate containing its separate
site passed validation in `astra`'s proxy environment. Candidate SHA-256:
`71a372868d75cb175b8e4bca649ebe9743b95d36412a705c9f29fa50762ff570`.
The active Caddyfile remained
`700beceabe7ed95cf8a08d2e20456abbae03a0fd4569aedf6cad92989438e5d9`, and
production's environment remained
`b1a380290d49e4abd19d8ba61eebfff4d81271ff05eeb5a23eeea057ab59fdba`.
No preview model, live route, test-domain TLS certificate, or real hosted
streaming request was exercised during staging. The owner then explicitly
approved a temporary GPT-2-only production pause for this hosted preview.

The lifecycle helper passed **11 focused tests** on Windows and on Amazon
Linux (**0.031 seconds** there). These model-free checks cover admission gates,
drain refusal, full result retention, model exclusivity, startup memory guards,
immutable candidate validation, partial-write/reload rollback, and preserving
the original Caddy bytes/inode. A separate read-only review found no remaining
blocking defect in the approved helper path. The standalone preflight checks
the production cgroup root; activation additionally checks descendants.

At **2026-10-04 03:25:27 UTC**, `--prepare` saved five validated, root-owned,
read-only full Caddy candidates and a private manifest under
`/home/astra/.config/astra-chess/gpt2-preview-stage-20261004`. The original
Caddyfile and the service/environment definitions were unchanged. The
installed helper is `/home/ec2-user/gpt2-preview-tooling/scripts/lightsail-preview.py`,
run with `/usr/bin/python3.12`; it is separate from the frozen application
checkout. The model-exclusivity guard refused startup while production was
active, without loading preview weights.

After approval, a durable `gpt2-preview-transition-20261004-0330` transient
systemd unit began activation. At **03:29:10.206842 UTC**, the gate was verified,
accepted production work was drained, and the full **600-second** result
retention window began. Production remained PID 43867 during that window;
health GETs returned 200, generation POSTs returned a readable 503 with
`accepted: false`, and both chess HTTPS sites returned 200 with unchanged PIDs.
The tested activation path retains results before stopping only GPT-2,
requires the old cgroup to be empty before loading the new model, verifies
backend/TLS readiness, and opens test admission last. These staging and
transition checks alone do not prove hosted streaming delivery.

## Lightsail streaming preview activation (2026-10-03 CDT)

The production result-retention window ended at **2026-10-04
03:39:10.486915 UTC**. Production stopped cleanly, and the guarded preview
started as `gpt-2`, **PID 54021**, on loopback 8898. The startup log reported
**1,557,611,200 parameters**, CPU FP32 and **two Torch threads**; the checkpoint
was mapped in 0.9 seconds. Mapping is not a warmed first inference. At
**03:39:20.541803 UTC**, the helper emitted `preview_active` and exited
successfully. No second Lightsail model was loaded and no chess process was
stopped. Production application checkout remains `e5971db`; preview
application checkout remains `5389a30`.

An independent read-only verifier at **03:39:59 UTC** passed all assertions:
production inactive/MainPID 0/empty cgroup, preview active under `gpt-2` with
one known model process, exact preserved unit/drop-in/environment/guard hashes,
unchanged Caddy/chess PIDs **4341 / 4149 / 4313**, preserved host/bind-mounted
Caddy inode and attributes, and retained original configuration bytes. The
active Caddyfile SHA-256 is
`2a4b9815e1bead3e084ed84156c23e7da14d51a12a4d3c6fb96592007ce0088f`.
Direct backend and normal public DNS/TLS health returned ready with
`streaming: true`; production HTTPS returned maintenance 503 with
`accepted: false`; both chess HTTPS roots returned 200. The test HTTP root
redirected to HTTPS with 308; both older GPT-2 aliases retained their 308
redirects to production. Certificate validation was never bypassed.

A separate in-app browser tab completed three short synthetic star-prompt
jobs, each with a 143-token input, zero examples, temperature 0.8, a 32-token
allowance and the fixed October 3 context date. The first reply used seed 42;
two regenerations used 43 and 44. Browser-reported model durations were
**6.8 / 7.1 / 7.8 seconds**, with **11 / 13 / 15 raw generated tokens** and a
new-message delimiter stop. These browser-rounded observations are not
sustained benchmarks or end-to-end latency measurements. The first request
after mapping included page-in/kernel initialization; OS cache state was not
controlled.

Seed 43 visibly streamed `" How"` at token one before reload, then recovered
its completed reply at the same seed. A batched seed-44 trial captured
`" Tell"` at token one, reloaded during generation, and showed
`" Tell me how"` at token three while still pending with seed 44. It completed
at 15 raw tokens, controls unlocked, and warning/error console logs were empty.
Two earlier provisional-status observations missed the short in-progress
window; the batched trial supplies the direct active-reload evidence.
Private screenshots are under `outputs/deploy/streaming-hosted-*.jpg`.
This verifies incremental browser delivery through the shared Caddy proxy
and recovery of the same pending regeneration before final commit.

An independent HTTPS client then completed exactly **two generation POSTs**
with the reversible-computing prompt used in the laptop HTTP checks, fixed
October 3 date, zero examples, temperature 0.8 and a 32-token allowance.
Both input prompts contained **144 tokens**. The seed-42 reply generated
**32 tokens** in **13.8907 seconds**, ending at the allowance. Seed-43
regeneration generated **28 tokens** in **12.3936 seconds**, ending at a
speaker stop sequence. Server first-token callbacks occurred at
**2.7973 / 2.8507 seconds**, first safe-text callbacks at
**2.7973 / 2.8507 seconds**, and decoding rates excluding first-token prefill
were **2.7948 / 2.8299 tokens/second**. These are two warmed operational
observations after the browser trials; cache state was not controlled.

The client ran as `gpt-2` with the existing **Python 3.12.12 / OpenSSL 3.2.2**
on Lightsail, using normal public DNS and verified TLS. Client first-progress
latencies were **0.0297 / 0.0431 seconds**; these initial snapshots contained
zero sampled tokens. First sampled-token/safe-text receipt was
**2.8123 / 2.8674 seconds** from generation POST start, including acknowledgement,
queue, TLS/SSE setup, proxy delivery and parser observation. Same-host client
timings do not measure a remote user's network or browser paint latency.
External Windows curl separately passed TLS validation and health 200, and
the browser above supplies external incremental-delivery proof. Bundled
Windows Python/OpenSSL health attempts failed before any generation POST
(socket permission/reset); their zero-job records are retained as client
limitations, not hosted application or certificate failures.

The first SSE connection closed after sampled token one and reconnected to
the same job while its initial resumed snapshot still reported running.
Exactly two SSE connections were used for that job. Both jobs passed stable
safe prefixes, progress/final raw-count equality, health during inference,
exact input/output IDs and settings, terminal GET retrieval, export/import
equality except `exported_at`, unchanged source archives, and retained original
seed-42 regeneration history. Root/API query stripping and wrong-origin
rejection also passed. Private evidence is
`outputs/deploy/streaming-hosted-smoke-lightsail-20261003.json`, **294,513 bytes**,
SHA-256 `cb415e9fe0ea700dfff88dff84bde40b5b2098c05dc1b7a1b7a9cde4b42c3726`.

At **03:48:21 UTC**, the final independent verifier passed all invariants
again, with the preview idle and both chess sites healthy. Process RSS was
**6,630,032 KiB (6.323 GiB)** and lifetime peak RSS **6,654,096 KiB (6.346 GiB)**.
Host available RAM was **14,327,108 KiB (13.663 GiB)**, with no swap. Preview
cgroup memory-current/peak were only **446.2 / 470.4 MiB**, much lower than
process RSS; shared mapped file-cache pages can retain charges outside this
cgroup after the warm-cache service transition. Those counters do not prove
that the full model needs under 500 MiB or that a cold-cache start fits the
9 GiB limit. High/max/OOM event counters and CPU throttling counters remained
zero throughout these short tests.

Between the two resource samples, preview CPU accounting increased by
**95.400 seconds** over **502.835 wall seconds**, averaging **0.190 busy CPUs**
or **4.743% of four host CPUs**. This interval includes browser automation and
idle waits; it is the preview cgroup's usage, not whole-instance utilization,
active-generation utilization, an AWS burst-credit reading or sustained-load
proof. The preview retains two Torch threads and the existing low scheduling
weight. Source/model/unit identity and private resource/transition evidence
are retained under `outputs/deploy`; the lifecycle helper's local and installed
SHA-256 matched
`77c7175dd9093d955e8d8475b2a5f028f7c1c03c1c8675a2032707d4d431df99`.

The hosted tab was left on a fresh chat with the normal 120-token allowance
and two examples. **https://test.gpt2-chat.com/** remains active for owner
testing; production GPT-2 remains paused at its original revision. The local
laptop preview remains separate. Streaming has not been rotated into
production and owner acceptance of the hosted preview remains pending.

## Streaming production rollout preparation (2026-10-03 CDT)

The owner accepted the hosted incremental display and explicitly authorized
production rotation. The tested application remains pinned to
`5389a30f5833c8bcaaf0314f1c0e1c63960f309e`; its source, dependencies and model
were not changed for promotion. The local development branch is
`codex/gpt2-streaming`. The preserved server production checkout remains
`e5971db3b79110cb62e92575a2054a87ba5b98c4`, even though the separate local
`codex/gpt2-public` branch tip is `a1ee558` (the later proxy query fix).
Rollback therefore selects the exact saved deployment, not a moving branch tip.

The production override uses `/usr/bin/env PYTHONPATH=...` to select the
accepted checkout explicitly, sets its working directory, adds streaming to
the existing loopback-8899 command, and checks preview inactivity before
startup. It retains the existing base unit, environment, user/cache/runtime,
two Torch threads, 256-token cap and 8/8.5/9 GiB startup/soft/hard memory
settings. The generic opposing-unit guard passed **11 model-free tests** on
Amazon Linux system Python 3.9 in **0.009 seconds**. Running it as `gpt-2`
against the active preview correctly refused startup with exit 1, without
loading a model. Installed guard SHA-256:
`e351ed35469d5ffc4d708f364edb206af845cc7be2db56fe2963e1c700579895`.

The separate promotion/rollback helper passed **14 model-free tests** on
Amazon Linux Python 3.12 in **0.092 seconds**. They cover full retention,
exclusive handoff, capability checks before admission, exact preparation
properties/file sets, source/guard/candidate drift, failed reload/partial-write
cleanup, PID-zero restart cancellation, accepted-work drain refusal, and
rollback of only the owned override. Independent review found no remaining
source blocker. These tests simulate failures; a live production rollback
was not exercised.

At **2026-10-04 04:47:57.130018 UTC**, `--prepare` validated seven immutable full
Caddy candidates, the assembled production unit, and private original-file
snapshots in `/home/astra/.config/astra-chess/gpt2-production-stage-20261004`.
Preparation did not change the active routes or units. Tooling is installed
separately in `/home/ec2-user/gpt2-promotion-tooling`; promotion helper SHA-256
`59fa0430ef4fadfbe3f5fa0db31b064787c92d5327fad4d4835977a07841ace0`
and reused preview helper SHA-256
`77c7175dd9093d955e8d8475b2a5f028f7c1c03c1c8675a2032707d4d431df99`
matched local reviewed bytes. The final proxy candidate preserves all
original public/chess blocks, including the canonical query rewrite and gzip,
and appends only a temporary test-domain 307 redirect with `Cache-Control:
no-store`. This keeps the test origin available for future preview reuse.

Durable unit `gpt2-production-promote-20261004-0448` began the verified preview
generation gate and **600-second** completed-result retention at
**04:48:13.198694 UTC**. New test-site generation returned 503 with
`accepted: false`; both chess HTTPS roots returned 200. During that window,
preview PID **54021** remained the sole resident model, production remained
inactive, and Caddy/Astra/Arcturus PIDs stayed **4341 / 4149 / 4313**.
Only after accepted work and retained results drain may the helper stop the
preview, install its owned `90-gpt2-streaming.conf`, load production under a
POST gate, verify exact model/streaming capability and TLS, then open admission.

### Production streaming activation and canonical HTTPS checks

The helper confirmed preview drain/result expiry at **2026-10-04
04:58:13.414328 UTC**, stopped that unit, installed only the owned override,
and started production under its generation gate. After direct and canonical
TLS readiness, it emitted **`production_streaming_active` at
04:58:27.430940 UTC** with production PID **58714** and exited successfully.
`gpt2-public.service` is active and enabled as `gpt-2`; preview is inactive,
PID zero, and its recursive cgroup is empty. The sole hosted application
remains frozen at `5389a30f5833c8bcaaf0314f1c0e1c63960f309e`. The older clean
`e5971db3b79110cb62e92575a2054a87ba5b98c4` checkout, base production unit and
environment remain byte-identical. Installed override SHA-256:
`524068351cc10be7557e08344c5e2ee5cc6e92a07dc51f720eac9fc76fed106a`.
Final active Caddy SHA-256:
`003c9b072341c7ccd565d7eb1d94a9f86c60e90923e3d1b700c82cbba944c98f`.

An independent read-only verifier passed before and after generation at
**04:59:00.504526 / 05:01:30.491948 UTC**, with no warnings. It checked exact
original and owned files, loaded launch/source path and resource settings,
protected guards/backups, one production model, clean frozen source trees,
preserved Caddy bind-mounted inode/attributes and complete original blocks,
canonical HTTPS streaming health, the test redirect, and chess health.
Caddy/Astra/Arcturus PIDs remained **4341 / 4149 / 4313**, their unit/config
hashes were unchanged, and both chess HTTPS roots returned 200. Canonical
health also passed certificate-verified `curl.exe` from the Windows laptop
with a synthetic `fbclid` query; the query was ignored.

Exactly **two** short real-model jobs ran through normal public DNS and
certificate-verified `https://gpt2-chat.com` between **05:00:17.492853 and
05:00:44.826132 UTC** (27.333301 seconds), with a same-host Lightsail client.
Both prepared prompts had **144 tokens**, zero seeded examples, temperature
**0.8**, and a **32-token** reply cap. The second job regenerated seed 42's
reply at seed 43. Measurements:

| Seed | Raw generated tokens | Generation seconds | Server first token seconds | Client first safe text seconds | Decode tokens/second |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 32 | 14.255855 | 3.196352 | 3.212617 | 2.803459 |
| 43 | 28 | 12.708865 | 2.840064 | 2.857443 | 2.736567 |

The active first job reconnected after its initial token with the same job
identity/cursor and no new POST. Cumulative stream prefixes stayed stable;
the final GET owned the canonical archive, token IDs and settings. Export
and import preserved the archive except its export timestamp, and seed 43
preserved the seed-42 candidate. Full raw token IDs and accepted reply text
exactly matched the corresponding earlier hosted-preview jobs. Exact-origin
rejection, ignored GET/POST queries, live health during generation, and idle
health before/after passed. The test domain returned temporary **307** with
`Cache-Control: no-store`; old aliases and canonical HTTP returned **308**.
Redirects preserved encoded/repeated/empty path/query test cases, and the
destination health endpoint ignored query parameters. This verifies actual
canonical SSE delivery through the preserved gzip/proxy configuration.

Post-run production RSS was **6.316 GiB**, process-lifetime high-water RSS
**6.346 GiB**. Its warm cgroup charge was **438.3 MiB**, peak **469.5 MiB**;
that accounting excludes already parent-charged mapped checkpoint pages and
is not the model's total physical footprint. No memory-high/max/OOM events,
swap, or cgroup CPU throttling were observed. Over the **149.985-second**
before/after interval, production used **53.590 CPU seconds**, averaging
**8.933%** of this four-CPU host. This interval includes verification and
idle time; neither it nor same-host first-text timing establishes sustained
load capacity, external-browser latency, or Lightsail burst charges.

Private byte-identical evidence remains ignored under `outputs/deploy`:

| Evidence | Bytes / SHA-256 |
| --- | --- |
| `streaming-production-smoke-lightsail-20261003.json` | 308,157 / `864a356629ac9c4a7e55d2c10e84dcb42d0dd56f6696708724959ea901bc367c` |
| `streaming-production-verify-before.json` | `1f901b817811b1062d8f8459cbe643cfb67ba7bd9a5d26815d0e42c19452038b` |
| `streaming-production-verify-after.json` | `b68532d139cc38ebf6165666ef744fad8044f87b462489f7292e949333f30430` |

The owner already accepted browser streaming on the identical hosted build;
this production check used synthetic HTTP jobs and did not inspect or modify
the owner's browser archives. No application source/dependency/model change,
reboot test, live rollback exercise, additional model copy, chess restart or
Git push was performed. The local laptop preview remains separate. The old
preview helper's `--restore` is obsolete after promotion: exact durable
`--rollback` instructions in
[`docs/PRODUCTION-STREAMING.md`](docs/PRODUCTION-STREAMING.md) drain and retain
accepted work before removing only the owned override and restoring the
original polling production build. Export/import transfers a test-origin chat
to canonical browser storage.

## Automatic seed advancement: local preview (2026-10-04)

The owner requested a successor seed after every generation, with local testing
before public deployment. Shared chat generation now uses the selected/stored
seed unchanged and commits its successor modulo `2**63` only after successful
inference and validation. This applies to both new messages and regeneration;
regeneration no longer pre-increments. Session/control/archive settings mean
**next seed**; historical actual/requested seeds, input prompts, output IDs and
candidate records remain exact. Manual seed entry is honored, blank uses the
stored next seed, and greedy completions advance the field although their
sampling ignores seeds. Imports preserve older saved seeds as the next seed
without rewriting history or changing schema version 2.

The complete model-free Python suite passed **242 tests in 16.083 seconds**
under the pinned WSL Python runtime. All **37 JavaScript browser tests** passed
with Node, and both inline scripts compile. Coverage exercises consecutive
message/regeneration transactions, explicit zero and values above JavaScript's
safe-integer range, maximum-seed wrap, legacy archives and omission prompts,
preview/rejection/failure preservation, queued independent browser clones,
final archive-validation failure, export/import/offline reload, lost-response
recovery, streaming reconnect and repeated terminal GETs. Actual seed and next
seed assertions remain separate; client code performs no seed arithmetic.

The owned parity-preview PID **341** was command-verified and idle before
SIGTERM. Only after it exited and no other known GPT-2 process remained, the
local service started as WSL PID **1355** on **127.0.0.1:8891**, using this
worktree's `PYTHONPATH`, the existing `~/.venvs/gpt2-xl` runtime, offline pinned
CPU FP32 GPT-2 XL, two Torch threads, streaming, the 256-token service cap and
unchanged 8 GiB startup floor. Ready/idle HTTP health passed. No dependency
reinstall, shared editable-package change, browser-storage modification or
second model copy occurred. The active private log/PID are
`outputs/deploy/seed-preview-20261004.log` and `seed-preview.pid`.

The first synthetic trial acknowledged one seed-42 generation, but its
50-millisecond GET polling exceeded the 240-request/minute local limit and
received HTTP 429 before retaining the result. Its job ID was not persisted,
so the canonical result was not recovered or claimed verified. The accepted
12-token job finished naturally (14.7 seconds per the private server log);
idle health was checked. That POST was not resubmitted. The aborted original
client/evidence remain private as `seed-policy-local-smoke-too-fast.py`
(SHA-256 `fc9b551dc23935094ae95be5e7d44cca3ddb235fc333d6e6127719cab9fdcbf7`)
and `seed-policy-local-smoke-20261004.json`
(`a48b3952a27d268f74fe42e651875e3f1bf4e550091b4bbdf734b1eb69f062ff`).
The corrected client saves an acknowledged job ID immediately and polls at
750-millisecond intervals. It ran a fresh sequence starting at seed 100.

The corrected trial passed **three** deliberately short real-model jobs:

| Action | Actual seed | Saved next seed | Input / output tokens | Generation seconds |
| --- | ---: | ---: | ---: | ---: |
| New message | 100 | 101 | 144 / 12 | 4.08 |
| New message | 101 | 102 | 174 / 12 | 4.27 |
| Regenerate latest reply | 102 | 103 | 174 / 12 | 4.23 |

Each canonical state/archive stored the successor, each generation retained
its actual/requested seed, and repeated result GETs returned identical state.
Import preserved the next seed and complete transcript. Regeneration reused
the exact preceding input prompt, kept two human rounds, and preserved the
seed-101 candidate. Input archives remained unmodified, all jobs stayed within
the 12-token cap, and final health was ready/idle. Four generations were
accepted in total across the aborted and corrected trials; no statistical
claim about repeated-response frequency or model-quality improvement follows
from this short functional check.

Full prompts, raw output IDs, controls, canonical archives, imports and exact
application/client source SHA-256s are retained in ignored private evidence
`outputs/deploy/seed-policy-local-smoke-20261004-v2.json`, SHA-256
`286b8b63582c399e3dfb0fa370edd323fbd9682a39c87b2741956b64c0dee769`.
Browser lifecycle checks were model-free; this live check used synthetic HTTP
jobs and did not inspect the owner's browser chats. The owner later accepted
the local test and authorized the production update recorded below. The local
implementation work itself did not change the public Lightsail service,
frozen `5389a30` source, Caddy, chess services or rollback configuration.

## Automatic seed advancement: production update (2026-10-04)

The owner accepted the streaming laptop test and explicitly authorized public
deployment. A subsequent status report confirmed the app had closed before
activation; production still had its unchanged `5389a30` checkout, PID 58714,
original streaming override and the earlier fixed-seed behavior. No partial
service switch had occurred. The resumed rollout uses a durable server unit
so a client or app disconnect does not interrupt retention or activation.

The exact accepted application commit
`78ebbdc254249562187c06db8ec9c3cb7dd0e87b` was cloned from the published branch
into `/home/gpt-2/GPT-2-releases/seed-78ebbdc`, then frozen root-owned and
read-only. All **60 tracked files** have staged hashes. The chat backend and
both HTML files match the locally accepted source hashes:

| File | SHA-256 |
| --- | --- |
| `src/gpt2_local/chat.py` | `9a96a673a8db31ce41846770b45fc0fde75aabd561e3a417a3ef2ff7d473fe5d` |
| `src/gpt2_local/web/public.html` | `bc73d2205be3de546c4f977a5dac78d34dbf290e318564df4f29598331621f72` |
| `src/gpt2_local/web/chat.html` | `02e83dc9fb9e046500b34f281258393f778dc9bd72c1514a438e7a56dca8c713` |

The pinned Amazon Linux runtime passed its complete application suite:
**239 tests, OK with one skip, in 13.959 seconds**. The first nonroot invocation
hit ownership assertions in root-only deployment fixtures; rerunning as root
passed. That fixture mismatch was not an application inference failure. The
suite performs no real model generation. No dependencies or shared editable
installation changed. The previously passed 37 frontend tests apply to the
identical accepted HTML source.

The final frozen update/promotion/exclusivity tooling passed **40 model-free
Linux tests in 0.327 seconds** before preparation. Coverage includes full
drain/600-second retention for update and rollback, single-model starts,
PID-zero retry cancellation, short-write and failed daemon-reload recovery,
atomic manifest failure, refusing source/override/loaded-setting drift,
retention-window drift leaving the existing model gated and running, unchanged
chess/preview control, and restoration of the original polling rollback
contract after returning to the previous streaming release. Failure and rollback
paths were exercised with synthetic service fixtures, not live failure injection
or a live production rollback.

| Frozen tooling | SHA-256 |
| --- | --- |
| `scripts/lightsail-update.py` | `4292bf7688cdd826b802836245e78e815119266f1a306ee4b084fb359a24f96d` |
| `tests/test_lightsail_update.py` | `6dc870170a8538a3b523ce03d310135e41deab1711a0648c8edb24e8530791b2` |
| Preserved promotion helper | `59fa0430ef4fadfbe3f5fa0db31b064787c92d5327fad4d4835977a07841ace0` |
| Preserved preview helper | `77c7175dd9093d955e8d8475b2a5f028f7c1c03c1c8675a2032707d4d431df99` |

Preparation completed without live changes at **19:24:41.381777 UTC**. The
durable unit `gpt2-seed-update-20261004-1925` started at **19:24:56 UTC**, applied
the POST gate at **19:25:02.155623 UTC**, and began the complete 600-second
retention wait at **19:25:10.321480 UTC**. The new override selects only the new
release path; its hash is
`a64825106023ba5603872dd73eaf3a1f029b2332ea4dbe3cf1f1fb776f6452a4`.
The old override (`524068351cc10be7557e08344c5e2ee5cc6e92a07dc51f720eac9fc76fed106a`)
and original promotion manifest
(`f49cad63c1269ce6a186e1c2c1af8bb8a077ebcde01bfdfc6ee8cff7de403f39`)
remain preserved for the controlled streaming rollback. Private baseline and
retention-stage copies have SHA-256s
`80b2f0d5677c789cb1f4e0ac78d883c28a894fe829002ecb56211f78014bc9fc` and
`76c78b0f22318d354c095e329ecc0c6deb2cbed51bbf13d2b46bdc14a2367479`.

The full retention period ended at **19:35:10.562603 UTC**. The durable helper
restored the exact active Caddy bytes at **19:35:21.991938 UTC**, then emitted
`production_version_updated` at **19:35:22.787953 UTC / 2:35 p.m. CDT**, exited
successfully, and left production active/enabled as PID **91398** selecting
the new frozen release. Preview stayed inactive. Direct/TLS streaming-ready
health, pinned CPU FP32 model identity and limits passed before admission opened.

An independent read-only verification at **19:36:08 UTC** passed the completed
schema-3 stage, all 60 release hashes/protection checks, exact new override and
loaded argv/PYTHONPATH/account, preserved base units/env/drop-ins/guards, previous
streaming/polling sources and stage snapshots, sole known model/cgroup, original
Caddy bytes/inode/attributes, test-domain redirect, and both chess HTTPS sites.
It checked the unchanged Caddy/chess PIDs **4341 / 4149 / 4313**. Three short
certificate-verified canonical HTTPS generations then passed between
**19:36:09 and 19:36:39 UTC**:

| Action | Actual seed | Saved next seed | Input / output tokens | Generation seconds | Server first visible seconds | Decode tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| New message | 100 | 101 | 144 / 16 | 8.627 | 3.135 | 2.731 |
| New message | 101 | 102 | 178 / 16 | 8.831 | 3.466 | 2.798 |
| Regenerate latest reply | 102 | 103 | 178 / 16 | 8.799 | 3.472 | 2.817 |

The client issued exactly three generation POSTs without retries, persisted
acknowledged IDs before subscription, disconnected/reconnected during the first
active generation, and verified that provisional updates never changed the
committed source seed/archive. Canonical and repeated terminal GETs preserved
the same actual/requested seed and successor controls, exact raw IDs and safe
provisional prefixes. Export/import preserved next seed and historical metadata.
Regeneration reused the preceding exact prompt, kept two human rounds, and
preserved the seed-101 candidate. Query stripping, exact-Origin rejection,
existing 308 aliases and test-origin 307/no-store path/query preservation passed.
The client ran on Lightsail over normal public DNS and verified TLS, so its
3.150/3.483/3.496-second first-visible observations do not measure an external
browser's network path or painting. Server timings use their own clock.

Independent after-verification at **19:37:53 UTC** passed with no warnings, the
same production/chess/Caddy PIDs, ready/idle health, and no new memory high/max,
OOM or CPU-throttling events. Production CPU usage rose **52.153 seconds** over
the **104.519-second** before/after interval: about **12.47%** average of the
four-vCPU host, including the intervening idle time. Process RSS was
**6,627,524 KiB**, process-lifetime peak RSS **6,686,316 KiB**, and host available
memory **14,315,596 KiB**. Cgroup current/peak were 464,830,464 / 525,828,096
bytes; those charge counters include shared-cache accounting and are not the
model's physical memory footprint. The existing 8.5/9 GiB limits and swap-free
8 GiB startup floor remain unchanged. Three capped jobs do not establish
sustained throughput, burst billing, repetition-frequency improvement, or
cold-start/near-full-context memory capacity. A separate certificate-verified
health request from the Windows laptop's external network also returned
ready/idle streaming CPU FP32 identity and unchanged limits.

Byte-identical private evidence was copied locally under ignored
`outputs/deploy/`; no owner browser archive was inspected or modified:

| Evidence/client | SHA-256 |
| --- | --- |
| `seed-production-before-20261004.json` | `4b7cf27602dab06559ab185c5fc4dea072a20fb74272a8828b3e9422ba92e3b4` |
| `seed-production-after-20261004.json` | `0c4bdfd3082a0a1b11167468990ef9df702d9e1468639c9619a0e58206eac2cb` |
| `seed-production-smoke-next-lightsail-20261004.json` | `f9835bce4a2329476da60146471c27039aba854016f5b1e0974c3a6680de2b7b` |
| `seed-update-completed-manifest-20261004.json` | `3e296d4f89fe9c370e32364cf18b35f5aa7505c38091f5ea2b1a0759155226b8` |
| `seed-production-smoke.py` | `a62a2cd301930363bc6924e20fc4c88299a4a2d4aea3c9c25037cd5617c7b2a0` |
| `verify-seed-production.py` | `7b15e0aa569194064664732444922d47feb48cb8aebce74d7717ff63bee115f6` |

Production is ready for the owner's fresh browser test after refreshing
`https://gpt2-chat.com/`. Operator instructions and two-step polling rollback
are in `docs/SEED-UPDATE.md`. Reboot/autostart behavior and live rollback were
not retested; existing enabled state and guarded fixture paths were verified.
