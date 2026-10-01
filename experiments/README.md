# Experiments

Experiment definitions are versioned JSON files. Generated results are written
under `outputs/experiments/`, which is intentionally ignored by Git because raw
GPT-2 continuations can be large, false, biased, offensive, or memorized.

`identity-context.json` is the first pilot. It keeps the prompt and every
decoding control fixed except temperature, and uses the same three predefined
seeds at each sampled temperature. Temperature 0 is greedy and therefore runs
only once. The one-token warm-up keeps cold page-in and kernel setup out of the
ten recorded completion timings.

From PowerShell:

```powershell
./experiment.ps1 -Offline -DryRun
./experiment.ps1 -Offline
```

The same application also has a randomized serendipity mode. It replaces the
configured temperature/seed matrix with the requested number of samples at one
temperature, selects distinct 63-bit seeds using the operating system's random
source, and records the selected plan before model loading:

```powershell
./experiment.ps1 -Offline -Serendipity 10 -Temperature 0.8
```

The sample count must be 1–100 and the temperature must be finite and greater
than zero. If `-Temperature` is omitted in serendipity mode, it defaults to
0.8. A serendipity `-DryRun` is a preview whose seeds are discarded; an actual
run chooses fresh seeds and persists them for replay.

Each completed or partial run contains:

- `results.json`, the machine-readable source of truth with model provenance,
  runtime versions, exact settings, token IDs, outputs, and timings;
- `report.md`, a readable report derived from that JSON, including the seed of
  every randomized sample;
- `replay-config.json`, the complete effective fixed-seed configuration needed
  to reproduce the plan.

The files are updated atomically after every completion, so an interrupted
multi-minute CPU run keeps all samples completed up to that point. Replay a
saved plan with:

```powershell
./experiment.ps1 -Offline -Config ./outputs/experiments/<run>/replay-config.json
```
