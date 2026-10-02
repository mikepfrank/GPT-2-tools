# Local GPT-2 XL

This project runs OpenAI's original **1.5-billion-parameter GPT-2 XL** learned
checkpoint locally. It uses Hugging Face's modern PyTorch/safetensors
conversion of that historical checkpoint, rather than the archived 2019
TensorFlow 1.12 runner. It is the same trained model, repacked into a current
file format; it is not a byte-identical copy of OpenAI's TensorFlow checkpoint.

The target machine is CPU-only: a 13th-generation Intel Core i7 with 32 GB of
RAM, Ubuntu 24.04 under WSL 2, and Intel UHD graphics. The full FP32 checkpoint
fits in memory, but generation will be much slower than on a modern discrete
GPU.

## Install in WSL 2

From PowerShell in this directory:

```powershell
./setup.ps1
```

The script installs Ubuntu's Python venv support, creates
`~/.venvs/gpt2-xl`, installs CPU-only PyTorch, and installs this project.

Then download only the files needed for GPT-2 XL's safetensors checkpoint
(about 6.43 GB), and verify the checkpoint's SHA-256:

```powershell
./setup.ps1 -DownloadModel
```

The model stays in WSL's normal Hugging Face cache, usually
`~/.cache/huggingface/hub`, rather than being copied into this project.

## Run it

Start an interactive session from PowerShell:

```powershell
./run.ps1
```

Or generate one continuation:

```powershell
./run.ps1 "Reversible computing is" -MaxNewTokens 60 -Seed 42
```

The first model load can take a while and needs several additional gigabytes
of memory. Close memory-heavy applications first. After the model is cached,
use `-Offline` to prohibit network access:

```powershell
./run.ps1 "In the future," -Offline
```

Inside Ubuntu, the equivalent commands are:

```bash
source ~/.venvs/gpt2-xl/bin/activate
gpt2-xl --info
gpt2-xl "Reversible computing is" --max-new-tokens 60 --seed 42
gpt2-xl                         # interactive mode
```

## Chat in a local browser

Start the chat server from PowerShell in this checkout:

```powershell
./chat.ps1 -Offline
```

After the terminal prints **GPT-2 chat ready**, open
<http://localhost:8765/> in a browser. Keep that terminal open; Ctrl+C stops
the server. Use `-Port 8899` if another application uses the default port.
The first reply can take longer while the checkpoint pages into memory.

The main scrolling view shows the complete current text context: the identity
and date header, retained examples, retained conversation, and latest reply.
Human messages are blue and GPT-2 replies are red; the header stays neutral.
Enter sends a message; Shift+Enter inserts a newline. Choose 0, 1, or 2 example
rounds and click **New chat** to begin with that many generic demonstrations.
Examples are authored prompt material, not measured model outputs. The date
comes from the browser's local calendar when starting the chat.

Temperature, optional seed, and maximum reply length apply to each next
message. Leaving the seed blank before **New chat** chooses a random seed once
and shows it in the field. Replies reuse that seed unless you change it;
clearing the field during a chat continues to use its stored seed. The default reply
allowance is 120 tokens, leaving up to 904 input
tokens within GPT-2's 1,024-token window. The fixed header is preserved while
the oldest complete conversation rounds roll out, starting with the examples.
When the first round rolls out, the app appends `\n\n...\n\n` to the header
once as a hint that earlier conversation was omitted. This hint remains for
the rest of the chat and counts toward the token budget. Saved chats retain
it; an older archive that already omitted rounds gains it on its next new
message, while regeneration continues to replay its original recorded prompt.
The page shows how many rounds have rolled out and lets you inspect the exact
input prompt for the last reply. If a message cannot fit beside the header and
reply allowance, shorten it or reduce the reply length; the app preserves your
unsent input and existing history.

Click **Regenerate last reply** to replace the latest model reply using its
exact original input prompt. The app increments the seed shown in the control
by one first; a blank control uses the stored active seed. The new seed appears
after a successful reply and becomes the active seed for subsequent messages.
The largest supported seed, `9223372036854775807`, wraps to zero. Regeneration
uses the current temperature and reply limit, and requires temperature above
zero because greedy decoding ignores the seed. A different seed can still
produce the same text. If an increased reply allowance cannot fit alongside
the original prompt, reduce it; regeneration preserves that prompt unchanged.
An error preserves the existing reply, stored seed, and unsent draft.

GPT-2's native end-of-text token, `<|endoftext|>` (50256), ends generation.
The chat also stops at the first new-message boundary matching
`\n\n[^\s>]+>`: a blank line, a nonempty speaker label containing no whitespace
or `>`, then `>`. This includes `Human>`, `GPT-2>`, and invented participants
such as `AI>` or `Assistant>`. It detects boundaries across generated-token
fragments and removes the entire delimiter and everything after it from the
displayed reply and future context. Ordinary blank lines and inline mentions
of speaker labels remain allowed. The boundary pattern and the literal
special-token spelling are reserved in human input.
The generated-token count includes any sampled stopping tokens; the visible
context excludes them.

Use **Export chat** at the bottom to save a JSON transcript, and **Import chat**
to select a saved file and resume it. Exports contain the exact prompt header,
all conversation blocks (including examples and rounds that left the context
window), the retained-context position, and the current controls. Each new
model reply also records its actual seed, requested seed, temperature, reply
allowance, input prompt, sampled token IDs, stop reason, and timing. Seeds are
decimal strings so full 63-bit values survive browser JSON handling. Missing
metadata from older conversations remains explicitly unknown. When an older
archive has no active seed, import chooses one for future replies; it does not
recover or change the historical seeds.

Regenerated replies retain earlier candidates in the model block's
`previous_responses` list, each with its text and generation metadata. Only
the current candidate enters the model context. The app can regenerate a
reply only when its original input prompt was recorded; authored examples
and older replies with missing metadata do not qualify.

Import reconstructs the original context and restores the controls, including
the example count and the saved active seed. It preserves the saved header's
date and accepts this app's version-1 and version-2 JSON files up to 4 MiB.
New exports use version 2, including the previous-response lists. A failed import
preserves the existing chat and unsent draft. Import opens a file picker;
export uses Save As where supported, with a filename prompt and browser
download fallback when the picker is unavailable.

The server binds only to the local loopback interface, uses one resident CPU
FP32 model, and returns completed replies. Live history stays in server memory;
export it before stopping the server to keep a copy. A browser reload restores
the current chat while its server remains running. To start with a saved chat,
run `./chat.ps1 -Offline -RestoreChat ./outputs/chats/example.json` and open the
printed **Restored chat** URL. Participant renaming, `/set` commands, and
streaming are left for later increments.

The launcher uses this checkout's source directly through `PYTHONPATH`, without
changing the shared WSL environment's editable installation. This matters when
trying a development worktree alongside the original completion tools. There
are no new application dependencies. Inside WSL the equivalent invocation is:

```bash
PYTHONPATH="$PWD/src" ~/.venvs/gpt2-xl/bin/python -m gpt2_local.chat --offline
```

## Run the temperature experiment

The repository includes a reproducible experiment built around the supplied
GPT-2 identity-context prompt. It loads the model once, performs a one-token
warm-up, and records ten 120-token-or-shorter completions:

- greedy decoding at temperature 0;
- temperatures 0.4, 0.8, and 1.2 at seeds 42, 314, and 2026.

Inspect the complete plan without loading the model:

```powershell
./experiment.ps1 -Offline -DryRun
```

Then run it:

```powershell
./experiment.ps1 -Offline
```

For a "serendipity pump" at one selected temperature, request any number from
1 through 100. This example creates ten independent samples at temperature
0.8, choosing a different random seed for each sample:

```powershell
./experiment.ps1 -Offline -Serendipity 10 -Temperature 0.8
```

`-Temperature` defaults to 0.8 when `-Serendipity` is present and is rejected
when used by itself. The seeds are selected and recorded before the model is
loaded, so even a model-load failure leaves behind the exact planned seeds.
Use `-DryRun` to preview a newly randomized plan without loading the model;
the later real invocation will intentionally choose a fresh plan.

Inside WSL, the equivalent entry point is:

```bash
gpt2-experiment experiments/identity-context.json --offline
gpt2-experiment experiments/identity-context.json --offline \
  --serendipity 10 --temperature 0.8
```

Each run creates a timestamped directory under `outputs/experiments/` with a
canonical `results.json`, a readable `report.md`, and the exact effective
`replay-config.json`. The files are updated atomically after each completion,
so an interrupted run retains its completed samples. A serendipity report
shows every selected seed beside its sample. Replay the same plan with:

```powershell
./experiment.ps1 -Offline -Config ./outputs/experiments/<run>/replay-config.json
```

The output directory is intentionally ignored by Git; generated text can be
false, biased, offensive, or memorized. All four reports produced during the
verified experiment work are reviewed and tracked as explicit records:

- [first fixed sweep](outputs/experiments/20261001T013927Z-gpt-2-identity-context-temperature-sweep/report.md);
- [repeated fixed sweep](outputs/experiments/20261001T014731Z-gpt-2-identity-context-temperature-sweep/report.md);
- [serendipity run](outputs/experiments/20261001T194750Z-gpt-2-identity-context-temperature-sweep-serendipity-at-temperature-0-8/report.md);
- [same-seed replay](outputs/experiments/20261001T201305Z-gpt-2-identity-context-temperature-sweep-serendipity-at-temperature-0-8/report.md).

Other generated artifacts remain local unless deliberately reviewed and added.

Edit or copy [`experiments/identity-context.json`](experiments/identity-context.json)
to define another fixed-prompt temperature/seed matrix. Temperature 0 is run
only once because seeds do not affect greedy decoding. The report's repetition
metrics are descriptive diagnostics, not measures of awareness or understanding.

## What this is—and is not

- The checkpoint is `openai-community/gpt2-xl`, pinned to revision
  `15ea56dee5df4983c59b2538573817e1667135e2`: the original GPT-2 XL model
  with 1,558 million parameters, a 1,024-token context window, and FP32
  weights.
- It is a base text-completion model, not an instruction-following assistant.
  Prompt it with the beginning of the kind of text you want it to continue.
- The runner downloads only the safetensors checkpoint and tokenizer/config
  files. Cloning the entire model repository would download duplicate
  TensorFlow, PyTorch, Flax, and Rust weight formats totaling roughly 32 GB.
- The original OpenAI GitHub implementation remains available as historical
  source, but it expects TensorFlow 1.12 and Python 3.6-era dependencies. A
  current runtime is considerably easier to reproduce on Ubuntu 24.04.
- The original model/repository uses OpenAI's **Modified MIT License**, which
  adds responsible-use and clear-labeling requests to MIT-like permission and
  warranty terms. The code in this local runner is separate from the model.

## Useful controls

```text
--temperature 0       deterministic greedy continuation
--temperature 0.8     sampled continuation (default)
--top-k 50            keep the 50 likeliest next tokens
--top-p 0.95          nucleus-sampling threshold
--max-new-tokens 80   maximum continuation length
--seed 42             repeatable sampling
--threads N           override PyTorch's CPU thread count
--offline             forbid downloads and use the local cache only
```

On this laptop, PyTorch's default of 14 threads performed much better than 28
threads. The processor's 28 logical CPUs are not a recommended thread setting;
if tuning, benchmark values in roughly the 4–20 range. Also expect the first
request after loading to be slower because it pages in mapped weights and
initializes kernels.

GPT-2 reflects biases and factual errors in its training data. Treat its text
as generated material, not as a reliable factual source.

## Run the model-free tests

From the project directory inside WSL with the project environment active:

```bash
python -m unittest discover -s tests -v
```

These tests exercise planning, validation, atomic result persistence, partial
failure recovery, Markdown rendering, and compatibility with the original
one-shot runner without loading the 6 GB checkpoint.

## Primary references

- [OpenAI's November 2019 GPT-2 1.5B release](https://openai.com/index/gpt-2-1-5b-release/)
- [Pinned GPT-2 XL checkpoint](https://huggingface.co/openai-community/gpt2-xl/tree/15ea56dee5df4983c59b2538573817e1667135e2)
- [Pinned safetensors file metadata](https://huggingface.co/openai-community/gpt2-xl/blob/15ea56dee5df4983c59b2538573817e1667135e2/model.safetensors)
- [Original OpenAI GPT-2 repository](https://github.com/openai/gpt-2)
- [OpenAI's Modified MIT License](https://github.com/openai/gpt-2/blob/master/LICENSE)
- [Transformers GPT-2 documentation](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [Microsoft's WSL filesystem guidance](https://learn.microsoft.com/windows/wsl/filesystems)
