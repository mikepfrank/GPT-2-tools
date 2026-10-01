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
false, biased, offensive, or memorized. The reviewed
[original](outputs/experiments/20261001T194750Z-gpt-2-identity-context-temperature-sweep-serendipity-at-temperature-0-8/report.md)
and [replay](outputs/experiments/20261001T201305Z-gpt-2-identity-context-temperature-sweep-serendipity-at-temperature-0-8/report.md)
reports are tracked as an explicit reproducibility record; other generated
runs remain local unless deliberately reviewed and added.

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
