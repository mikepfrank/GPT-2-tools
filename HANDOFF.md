# Project handoff

Last updated: **2026-10-02**

This is the living continuation brief for developers and agentic coding
sessions working on `GPT-2-tools`. Update it when project direction, verified
capabilities, constraints, or likely next work materially change.

## Start here

The project is a working, CPU-first local deployment of OpenAI's original
GPT-2 XL learned checkpoint. The baseline runner and the first reproducible
temperature-sweep application are complete and verified.

The first browser-chat increment is developed separately on `codex/gpt2-chat`,
starting from `b48ff08`. Its local worktree is `.worktrees/gpt2-chat`; the
original `main` checkout and shared editable WSL installation remain intact.
The project owner has tested basic chat, speaker colors, seed recording,
import/export, and last-reply regeneration. Context omission hints are the
latest increment; streaming, participant renaming, and in-context commands
remain later work.

- Public repository: <https://github.com/mikepfrank/GPT-2-tools>
- Primary branch: `main`
- Installation and user-facing commands: [`README.md`](README.md)
- Reproduced environment and test evidence:
  [`VERIFICATION.md`](VERIFICATION.md)

Keep this file focused on durable decisions and continuation context. Put
user instructions in `README.md` and exact reproduced measurements in
`VERIFICATION.md` rather than copying them here.

## Current state

- The setup downloads and verifies the full 1.5B-class GPT-2 XL checkpoint;
  the runner then loads it and generates text offline under Ubuntu in WSL 2.
- A PowerShell launcher supports one-shot generation and an interactive REPL.
- A localhost browser chat supports a resident model, the complete visible
  context with speaker colors, Enter/Shift+Enter input, selectable 0/1/2
  example rounds, portable JSON transcripts with import/export, and last-reply
  regeneration with seed increments and preserved candidate history.
- A config-driven experiment application keeps the model resident while it
  sweeps temperatures and seeds, then saves canonical JSON and Markdown.
- The same application has a serendipity mode that selects distinct random
  seeds for repeated sampling at one temperature and saves an exact replay
  configuration before model loading.
- The verified backend is CPU-only PyTorch in FP32. No NVIDIA GPU is present.
- The Hugging Face model cache and Python virtual environment live in WSL,
  outside the Git repository.
- Python application dependencies and CPU PyTorch are pinned. The setup is not
  fully hermetic because Ubuntu packages and pip build tooling can change.
- The local `main` branch tracks `origin/main`. The initial project snapshot
  was commit `2a8bc6d` (`Initial GPT-2 XL local runner`).
- This public repository does not contain model weights, generated output,
  credentials, virtual environments, or machine-local caches.

## Core decisions and invariants

### Model identity

Use the exact historical GPT-2 XL model unless a future task explicitly adds
another model:

```text
Hugging Face repository  openai-community/gpt2-xl
Revision                 15ea56dee5df4983c59b2538573817e1667135e2
Parameters               1,557,611,200
Context window            1,024 tokens
Weight format             FP32 safetensors
model.safetensors SHA-256 0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a
```

These constants are enforced in
[`src/gpt2_local/runtime.py`](src/gpt2_local/runtime.py). The Hugging Face
artifact is a converted/repacked representation of the historical learned
checkpoint, not a byte-identical copy of OpenAI's TensorFlow checkpoint and
not a retrained derivative.

The 2019 OpenAI runner was deliberately not used: it expects TensorFlow 1.12
and Python 3.6-era dependencies. The current PyTorch/Transformers path retains
the historical model while remaining practical on Ubuntu 24.04.

### Download and integrity policy

- Download only the safetensors checkpoint plus required config/tokenizer
  files. Do not clone the complete Hugging Face repository; it contains about
  32 GB of duplicate framework-specific weights.
- Keep the checkpoint pinned by revision.
- Keep the explicit SHA-256 verification for `model.safetensors`.
- Do not commit the checkpoint or Hugging Face cache to Git.

### Runtime baseline

- Preserve CPU FP32 as the fidelity and comparison baseline.
- Quantized or alternate backends must be named and measured separately;
  they can change generated output even when using the same learned model.
- Do not silently substitute a smaller GPT-2 checkpoint.
- Keep model provenance and verification status explicit in documentation.

### Licensing

The upstream GPT-2 repository/model uses OpenAI's **Modified MIT License**;
see the references in `README.md`. The newly written runner currently has no
top-level `LICENSE`, so normal default copyright applies to it. Choosing a
license for this repository remains an explicit project-owner decision.

## Environment assumptions

The verified machine and versions are recorded in `VERIFICATION.md`. Important
operational assumptions are:

- Windows PowerShell launches a WSL 2 distribution named `Ubuntu`.
- The WSL virtual environment is fixed at `~/.venvs/gpt2-xl`.
- The launchers resolve the current WSL user's home dynamically; do not
  hard-code an absolute Linux home path.
- The model normally resides under `~/.cache/huggingface`, outside the repo.
- WSL currently has about 15 GiB RAM and 4 GiB swap. GPT-2 XL fits and has
  run successfully without changing `.wslconfig`.
- Closing Chrome or other memory-heavy Windows applications can reduce paging,
  but no WSL memory change is presently required.
- The Intel UHD integrated GPU is not a verified inference backend. Treat the
  machine as CPU-only unless a future change proves otherwise.

The dated disk-usage measurements are recorded in `VERIFICATION.md`.

## Repository map

- [`setup.ps1`](setup.ps1) translates the Windows project path and invokes the
  WSL setup script. `-DownloadModel` also downloads and verifies the model.
- [`scripts/setup-wsl.sh`](scripts/setup-wsl.sh) installs Ubuntu venv support,
  creates/reuses `~/.venvs/gpt2-xl`, installs exact CPU-only PyTorch and locked
  dependencies, installs the package editable, and runs `pip check`.
- [`run.ps1`](run.ps1) safely passes a prompt and selected controls from
  PowerShell into the WSL Python runner.
- [`experiment.ps1`](experiment.ps1) launches a batch experiment through WSL
  while safely translating config and output paths.
- [`chat.ps1`](chat.ps1) launches the browser chat with this checkout's `src`
  explicitly selected through `PYTHONPATH`, avoiding reinstallation into the
  shared WSL environment when working in a separate worktree.
- [`src/gpt2_local/chat.py`](src/gpt2_local/chat.py) owns the approved chat
  header/examples, whole-round context packing, in-memory sessions, exact
  input previews, and a loopback-only standard-library HTTP server.
- [`src/gpt2_local/web/chat.html`](src/gpt2_local/web/chat.html) implements the
  browser interface without frontend dependencies.
- [`experiments/identity-context.json`](experiments/identity-context.json)
  defines the first fixed-prompt, ten-sample temperature sweep.
- [`src/gpt2_local/cli.py`](src/gpt2_local/cli.py) implements download, info,
  one-shot, and interactive modes.
- [`src/gpt2_local/experiment.py`](src/gpt2_local/experiment.py) validates and
  expands experiment configs, runs one resident model, checkpoints results,
  and renders reports.
- [`src/gpt2_local/runtime.py`](src/gpt2_local/runtime.py) owns model identity,
  downloading, checksum verification, FP32 loading, prompt truncation,
  structured generation results, validation, and timing.
- [`tests/`](tests) contains model-free `unittest` coverage for the experiment
  and the original text-only generation API.
- [`pyproject.toml`](pyproject.toml) defines the package and `gpt2-xl` entry
  point.
- [`requirements.lock.txt`](requirements.lock.txt) records the exact verified
  dependency resolution. CPU PyTorch is installed separately from its CPU
  wheel index by the setup script.
- [`.gitattributes`](.gitattributes) protects LF line endings for Bash/Python
  sources and Windows line endings for PowerShell entry points.

## Operating the current runner

The shortest Windows workflow is:

```powershell
./setup.ps1 -DownloadModel
./run.ps1 -Offline
```

The interactive interface is a simple text-completion REPL. Each prompt is
independent; keeping the process open retains the loaded model but does not
accumulate conversation history. Exit with `/q`, `/quit`, `/exit`, or an input
interrupt.

Default generation settings are:

```text
max_new_tokens       80
temperature          0.8
top_k                50
top_p                0.95
repetition_penalty   1.0
seed                 unset
```

`temperature=0` selects greedy decoding. A supplied seed makes sampled output
repeatable for the same prompt and settings on the same runtime.

Prompt plus continuation must fit the 1,024-token context. When needed, the
runner preserves the requested continuation space by truncating the prompt
from the left and retaining its ending.

The PowerShell wrapper presently exposes `MaxNewTokens`, `Temperature`,
`Seed`, `Offline`, and `IncludePrompt`. `Seed` accepts the nonnegative signed
64-bit range used by experiment replays. Top-k, top-p, repetition penalty, and
thread count are available through the WSL `gpt2-xl` CLI but not yet through
`run.ps1`.

## Operating the experiment runner

The tracked `identity-context.json` pilot uses the project owner's exact
`Context:`-prefixed prompt, one greedy completion, and temperatures 0.4, 0.8,
and 1.2 at seeds 42, 314, and 2026. All other generation controls remain fixed.

```powershell
./experiment.ps1 -Offline -DryRun
./experiment.ps1 -Offline
./experiment.ps1 -Offline -Serendipity 10 -Temperature 0.8
```

The runner performs a recorded one-token greedy warm-up, then executes all ten
cases in one process. It stores exact token IDs, decoded text and hash, stop
reason, settings, runtime/model provenance, simple repetition metrics, and
timing. Controlled experiment configs are rejected if the prompt would require
the baseline runtime's left truncation.

In serendipity mode, `-Serendipity N` accepts 1–100 samples and replaces the
configured matrix with one positive, finite temperature (0.8 by default).
Distinct seeds are drawn from `[0, 2^63)` using Python's OS-backed `secrets`
source. Seed selection occurs before model construction and the plan is
checkpointed immediately, so model-load failures do not lose it. A dry run
previews a newly randomized plan; a subsequent live invocation chooses a fresh
one.

`results.json` is canonical; `report.md` is derived from the same in-memory
record, and `replay-config.json` freezes the effective settings and seeds.
All three are atomically replaced after every sample and retain `failed` or
`interrupted` partial results. Replay with `-Config <run>/replay-config.json`.
Generated runs remain under ignored `outputs/experiments/`. Do not commit raw
continuations to this public repository without reviewing them and receiving
the project owner's approval. The project owner explicitly approved all four
current `report.md` files—the two fixed sweeps, the serendipity run, and its
same-seed replay—as tracked exceptions. This does not change the default ignore
policy for raw JSON, replay configs, or future generated artifacts.

## Operating the browser chat

Run `./chat.ps1 -Offline` from the development checkout and open
`http://localhost:8765/` after the ready message. The model remains resident;
Ctrl+C stops the server. The browser supplies its local date when starting a
chat. The project owner reviewed and approved the concise identity/history
header and two generic demonstration rounds before generation tests.

The header is preserved while the oldest complete human/model rounds leave
the 1,024-token window, including the examples first. A 120-token reply
allowance is the default; oversized current messages are rejected without
changing history. `/api/preview` shows the exact next input before inference,
and the main scrolling widget then shows the retained input plus visible reply.
At the first eviction, `prepare` proposes appending literal `\n\n...\n\n`
to the header, then repacks with its token overhead included. A successful
reply commits that header; preview and failed inference leave it unchanged.
The suffix is appended only once, including when seeded examples are evicted,
and persists through export/import in the existing `prompt_header` field.
Legacy archives with dropped rounds and an unmarked header remain byte-exact
on import and regeneration; the next new message adds the hint. Existing
message delimiters still contribute their own blank-line prefix after the
literal header suffix. New chat starts again without the hint.

The runtime's optional literal `stop_sequences` and regex `stop_pattern`
support match decoded continuation text across token boundaries. The chat
uses `\n\n[^\s>]+>` so invented participants such as `AI>` also end the
reply. It excludes the entire boundary and everything following it from the
displayed reply and subsequent context, and reserves that same pattern in
human input. Native `<|endoftext|>` (50256) also stops generation; ordinary
blank lines do not. Raw generated-token metadata still includes sampled
stopping tokens. Default completion/experiment behavior is unchanged when no
custom stops are supplied.

The server uses in-memory sessions and serializes inference because the model
and Torch RNG are shared. Reloading the browser restores a live session; New
chat releases its previous session. The context widget colors exact human
segments blue and model segments red without changing prompt text.

`chat_archive.py` exports version-2 JSON and imports versions 1 and 2. Archives retain the exact
header/date, complete transcript including evicted rounds, active suffix
position, current controls, example flags, checkpoint/protocol provenance,
and available per-reply generation metadata. Imports validate the entire file
before replacing a session, recreate the saved active context, and restore
controls. Seed values remain decimal strings across the browser boundary.
Blank seeds select and record one OS-random 63-bit seed when New chat starts.
The returned controls display that concrete seed; replies reuse it unless the
user supplies a different one. A blank message/export seed falls back to the
stored active seed. The runtime reseeds before each reply, allowing individual
turn replay. Legacy imports with active seed null choose a seed for future
replies in `ChatApplication.import_session`, with an explanatory note. The
pure archive parser preserves null seeds and historical missing metadata;
it must not invent provenance for old replies.

Regenerate replaces only the most recent non-example reply, using the exact
recorded input prompt. The current seed control (or stored seed if blank) is
incremented by one modulo 2**63 before sampling; current temperature and reply
limit apply. Temperature zero is rejected because changing the seed cannot
change greedy decoding. Increasing the cap beyond the original prompt's
remaining token budget is rejected rather than repacking that prompt. Preview
and failed inference do not commit changes. The previous candidate's exact
text and metadata are appended to `previous_responses`; only the replacement
enters active context. Version-1 files import with empty candidate histories.
The archive supports up to 1,000 previous candidates per model block; unknown
historical provenance stays unknown. Do not infer exact replay from the seed
alone: retain the recorded prompt, settings, token IDs, and pinned runtime.

The bottom Import/Export buttons use file pickers, with a filename/download
fallback where Save As is unavailable. Import accepts up to 4 MiB; failed
imports preserve the current chat and draft. `chat.ps1 -RestoreChat PATH`
restores an archive at startup and prints a session URL that takes precedence
over the browser's previous session. Native Save As must be invoked before
network waits to retain the click's user activation. Keep a live conversation
exported before restarting its server. Raw user transcripts and smoke-test
artifacts remain under ignored `outputs/chats/`; do not publish them as part
of routine code pushes. Streaming and participant-name commands are still
later increments.

## Performance notes

- Safetensors maps the model quickly, but the first generation also pays for
  lazy page-in and kernel initialization. Do not interpret map time alone as
  full cold-start latency.
- The runner labels first-request timing explicitly. Interactive or resident
  applications avoid paying that cost for every prompt.
- PyTorch selects 14 CPU threads by default on this laptop. This performed
  much better than forcing all 28 logical CPUs; hybrid-core oversubscription
  can cause a severe slowdown.
- Treat the numbers in `VERIFICATION.md` as baselines rather than general
  guarantees. Reproduce measurements after any runtime, backend, precision,
  or generation change.

## Behavioral context from the first exploratory run

The project owner used interactive mode with 120 requested output tokens and
provided a self-referential identity prompt describing GPT-2 from a 2026
perspective. With the default sampling settings and no seed, the model adopted
a human first-person role and entered a repeating `GPT-2:`/declaration pattern.
The displayed run achieved 4.79 output tokens/second.

This was considered characteristic base-model behavior:

- GPT-2 completes the apparent document or transcript; it does not follow a
  chat protocol or consult a stable self-model.
- Prompt facts, evaluative claims, and role framing are all conditioning text;
  the model does not independently verify them.
- Once a repeated local structure enters the context, the default
  `repetition_penalty=1.0` does not discourage it and the pattern can become a
  sampling attractor.
- Human-role adoption and self-referential prose are evidence of flexible role
  simulation, not evidence of self-awareness.
- Because no seed was supplied, that exact sample is not reproducible.

The temporary screenshot and full output were intentionally not committed to
the public repository.

The controlled temperature sweep is summarized in `VERIFICATION.md`. It showed
large seed-to-seed variation: greedy and two of three temperature-0.4 samples
looped heavily, while other seeds produced less repetitive text or an early
end-of-text token. This small, single-prompt pilot is descriptive only; it does
not support broad temperature or model-identity conclusions.

## Verification status

The following have been reproduced locally:

- filtered checkpoint download and full-file SHA-256 verification;
- offline model load in FP32;
- exact parameter count;
- deterministic and sampled generation paths;
- PowerShell, Bash, and Python syntax/bytecode checks;
- `pip check` with no broken requirements;
- prompts beginning with a hyphen crossing the PowerShell/WSL boundary;
- supplied seed handling;
- rerunning setup against the existing virtual environment using the dependency
  lock;
- the ten-case experiment dry run and two full offline executions;
- the ten-sample temperature-0.8 serendipity run, recorded unique seeds, saved
  replay plan, and exact token-ID equality across a full same-machine replay;
- the original nineteen model-free automated tests covering experiment planning,
  serendipity selection and CLI parsing, seed replay, structured results,
  warm-up, overflow rejection, failure, and persistence paths.
- the expanded 61-test model-free suite, including arbitrary-speaker regex
  stopping and the earlier decoded literal delimiter stopping,
  whole-round rolling context, exact input previews, transactional errors,
  63-bit seed handling, and localhost HTTP requests;
- live offline browser chats with 0, 1, and 2 example rounds, early delimiter
  stopping and reply-limit stopping, multiline keyboard input, overlong-input
  preservation, and browser reload recovery.

Preserve the distinction between reproduced verification and untested
expectations. The tests deliberately do not load or numerically validate the
6 GB model; the full offline sweeps are the integration evidence for that path.

## Known limitations and caution points

- GPT-2 is a base text completer, not an instruction-following assistant.
- The original completion REPL has no automatic history; browser chat does.
- `--include-prompt` affects one-shot generation; interactive mode currently
  prints only the continuation.
- The PowerShell launchers assume the distro is named `Ubuntu` and the venv is
  at `~/.venvs/gpt2-xl`.
- A one-shot process reloads/maps the model each time. Prefer interactive mode
  or a persistent service for repeated use.
- The experiment schema currently supports one fixed prompt per run. Fixed
  matrices execute in declared order; serendipity mode randomizes seeds but not
  case order. Treat timings as operational observations, not a controlled
  benchmark.
- The model can produce false, biased, offensive, repetitive, or incoherent
  text. Generated content should not be treated as factual.
- Do not report quantized output as numerically equivalent to the FP32
  baseline.
- Do not add secrets, GitHub tokens, local caches, model weights, virtual
  environments, or temporary screenshots to the public repository.

## Preliminary optimization research—not yet reproduced

OpenVINO was researched as a plausible later optimization but has not been
installed, exported, or benchmarked in this project. Treat these as leads to
revalidate against the exact future package versions, not verified behavior:

- CPU OpenVINO is the sensible target; the integrated Intel UHD GPU was not
  exposed as a verified WSL compute device.
- Export FP32 explicitly when establishing an OpenVINO comparison, and confirm
  the chosen Optimum Intel version's default compression behavior first.
- Benchmark warmed steady-state generation and cold-start behavior separately.
- An explicit INT8 experiment may improve speed and memory use, but it is a
  distinct numerical model and output may diverge.
- Prefer a separate environment/artifact for experiments so the verified
  PyTorch baseline remains intact.

## Likely next work

Good candidates, roughly in priority order:

1. Extend model-free coverage to PowerShell argument forwarding and download
   metadata.
2. Add multi-prompt experiment configs or an analysis command that aggregates
   repetition/coherence diagnostics across more than one prompt.
3. Expose top-k, top-p, repetition penalty, and CPU threads through `run.ps1`,
   or add interactive commands for inspecting/changing settings.
4. Have the project owner test last-reply regeneration, then consider streaming
   output and participant-name commands as separate later increments.
5. Run a separate controlled comparison of repetition penalties around `1.05`,
   `1.10`, and `1.20`; do not mix that variable into the temperature pilot.
6. Benchmark an isolated OpenVINO FP32 backend against the existing baseline,
   then optionally evaluate an explicitly labeled INT8 variant.
7. Choose and add a license for this repository's newly written code.

## Guidance for future agentic sessions

1. Read this file, `README.md`, and `VERIFICATION.md` before changing the
   runtime or setup.
2. Start with `git status --short --branch`, inspect recent commits, and confirm
   the expected `origin` before editing or pushing.
3. Verify the live WSL/runtime state instead of assuming an older snapshot is
   still current. `gpt2-xl --info` is a cheap first check.
4. Preserve unrelated user changes and keep the model/cache outside Git.
5. For behavior or performance claims, record exact prompts, decoding settings,
   seed, runtime versions, and whether the model was cold or warm.
6. Update `VERIFICATION.md` only with reproduced evidence. Update this handoff
   when decisions, constraints, current status, or next priorities change.
7. Keep `README.md` oriented toward users; do not turn it into an agent log.
8. Commit cohesive changes and push `main` only when authorized.

The repository-initialization session encountered Git's “dubious ownership”
protection because the Codex sandbox identity differed from the folder owner.
Other sessions may not reproduce it. If it occurs, prefer a per-command
`safe.directory` configuration for this repository rather than modifying the
user's global Git configuration. A separate sandbox warning about an unreadable
global Git ignore file was non-blocking; the repository's own `.gitignore`
remains authoritative.
