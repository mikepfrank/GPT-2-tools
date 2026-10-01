# Verification record

Verified locally on **2026-09-30**.

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
