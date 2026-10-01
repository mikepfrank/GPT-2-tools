from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import secrets
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Protocol

from . import runtime as runtime_module
from .runtime import (
    MODEL_CONTEXT_TOKENS,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    GenerationResult,
    GenerationSettings,
    Gpt2Runner,
    runtime_summary,
)

SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path("experiments/identity-context.json")
DEFAULT_OUTPUT_ROOT = Path("outputs/experiments")
DEFAULT_SERENDIPITY_TEMPERATURE = 0.8
MAX_SERENDIPITY_SAMPLES = 100
SEED_UPPER_BOUND = 2**63


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    prompt: str
    temperatures: tuple[float, ...]
    seeds: tuple[int, ...]
    max_new_tokens: int
    top_k: int
    top_p: float
    repetition_penalty: float
    warmup_tokens: int = 1
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ExperimentConfig":
        allowed = {
            "schema_version",
            "name",
            "description",
            "prompt",
            "temperatures",
            "seeds",
            "max_new_tokens",
            "top_k",
            "top_p",
            "repetition_penalty",
            "warmup_tokens",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown configuration field(s): {', '.join(unknown)}")

        required = allowed - {"description", "warmup_tokens"}
        missing = sorted(required - set(values))
        if missing:
            raise ValueError(f"missing configuration field(s): {', '.join(missing)}")

        raw_temperatures = values["temperatures"]
        if not isinstance(raw_temperatures, list):
            raise ValueError("temperatures must be a JSON array")
        temperatures = tuple(
            _require_number(value, f"temperatures[{index}]")
            for index, value in enumerate(raw_temperatures)
        )

        raw_seeds = values["seeds"]
        if not isinstance(raw_seeds, list):
            raise ValueError("seeds must be a JSON array")
        seeds = tuple(
            _require_int(value, f"seeds[{index}]")
            for index, value in enumerate(raw_seeds)
        )

        config = cls(
            schema_version=_require_int(values["schema_version"], "schema_version"),
            name=_require_string(values["name"], "name"),
            description=_require_string(values.get("description", ""), "description"),
            prompt=_require_string(values["prompt"], "prompt"),
            temperatures=temperatures,
            seeds=seeds,
            max_new_tokens=_require_int(values["max_new_tokens"], "max_new_tokens"),
            top_k=_require_int(values["top_k"], "top_k"),
            top_p=_require_number(values["top_p"], "top_p"),
            repetition_penalty=_require_number(
                values["repetition_penalty"], "repetition_penalty"
            ),
            warmup_tokens=_require_int(values.get("warmup_tokens", 1), "warmup_tokens"),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SCHEMA_VERSION}, received {self.schema_version}"
            )
        if not self.name.strip():
            raise ValueError("name cannot be empty")
        if not self.prompt.strip():
            raise ValueError("prompt cannot be empty")
        if not self.temperatures:
            raise ValueError("temperatures cannot be empty")
        if len(set(self.temperatures)) != len(self.temperatures):
            raise ValueError("temperatures cannot contain duplicates")
        if any(temperature < 0 for temperature in self.temperatures):
            raise ValueError("temperatures cannot be negative")
        if any(temperature > 0 for temperature in self.temperatures) and not self.seeds:
            raise ValueError("at least one seed is required for sampled temperatures")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds cannot contain duplicates")
        if any(seed < 0 or seed > 2**63 - 1 for seed in self.seeds):
            raise ValueError("seeds must be between 0 and 2^63 - 1")
        if not 0 <= self.warmup_tokens < MODEL_CONTEXT_TOKENS:
            raise ValueError(
                f"warmup_tokens must be between 0 and {MODEL_CONTEXT_TOKENS - 1}"
            )

        # Reuse the runtime's authoritative validation for decoding controls.
        for temperature in self.temperatures:
            GenerationSettings(
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
            ).validate()

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["temperatures"] = list(self.temperatures)
        values["seeds"] = list(self.seeds)
        return values


@dataclass(frozen=True)
class ExperimentCase:
    index: int
    temperature: float
    seed: int | None


class ExperimentRunner(Protocol):
    def count_prompt_tokens(self, prompt: str) -> int: ...

    def generate_result(
        self,
        prompt: str,
        settings: GenerationSettings,
        *,
        include_prompt: bool = False,
    ) -> GenerationResult: ...


def load_config(path: Path) -> ExperimentConfig:
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(values, dict):
        raise ValueError("experiment configuration must be a JSON object")
    return ExperimentConfig.from_mapping(values)


def build_plan(config: ExperimentConfig) -> list[ExperimentCase]:
    """Expand the temperature/seed matrix, running greedy decoding only once."""
    cases: list[ExperimentCase] = []
    for temperature in config.temperatures:
        if temperature == 0:
            cases.append(
                ExperimentCase(
                    index=len(cases) + 1,
                    temperature=temperature,
                    seed=None,
                )
            )
            continue
        for seed in config.seeds:
            cases.append(
                ExperimentCase(
                    index=len(cases) + 1,
                    temperature=temperature,
                    seed=seed,
                )
            )
    return cases


def select_unique_random_seeds(
    count: int,
    *,
    randbelow: Callable[[int], int] | None = None,
) -> tuple[int, ...]:
    """Choose distinct portable PyTorch seeds using OS entropy by default."""
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 1 <= count <= MAX_SERENDIPITY_SAMPLES
    ):
        raise ValueError(
            f"serendipity sample count must be between 1 and "
            f"{MAX_SERENDIPITY_SAMPLES}"
        )
    choose = randbelow or secrets.randbelow
    selected: list[int] = []
    seen: set[int] = set()
    while len(selected) < count:
        seed = choose(SEED_UPPER_BOUND)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("random seed source returned a non-integer")
        if not 0 <= seed < SEED_UPPER_BOUND:
            raise ValueError("random seed source returned an out-of-range value")
        if seed not in seen:
            seen.add(seed)
            selected.append(seed)
    return tuple(selected)


def make_serendipity_config(
    base: ExperimentConfig,
    *,
    count: int,
    temperature: float,
    randbelow: Callable[[int], int] | None = None,
) -> ExperimentConfig:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("serendipity temperature must be finite and greater than 0")
    seeds = select_unique_random_seeds(count, randbelow=randbelow)
    config = replace(
        base,
        name=f"{base.name} - serendipity at temperature {temperature:g}",
        description=(
            f"Serendipity mode: {count} independently seeded samples at "
            f"temperature {temperature:g}. Random seeds were selected before model "
            "loading and recorded for replay."
        ),
        temperatures=(temperature,),
        seeds=seeds,
    )
    config.validate()
    return config


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "experiment"


def _new_run_directory(output_root: Path, experiment_name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = output_root / f"{stamp}-{_slug(experiment_name)}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = Path(f"{base}-{suffix}")
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _settings_for(config: ExperimentConfig, case: ExperimentCase) -> GenerationSettings:
    return GenerationSettings(
        max_new_tokens=config.max_new_tokens,
        temperature=case.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        seed=case.seed,
    )


def _result_fields(result: GenerationResult) -> dict[str, object]:
    bigrams = list(zip(result.output_token_ids, result.output_token_ids[1:]))
    fourgrams = list(
        zip(
            result.output_token_ids,
            result.output_token_ids[1:],
            result.output_token_ids[2:],
            result.output_token_ids[3:],
        )
    )
    return {
        "continuation": result.text,
        "continuation_sha256": hashlib.sha256(
            result.text.encode("utf-8")
        ).hexdigest(),
        "original_input_tokens": result.original_input_tokens,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "output_token_ids": list(result.output_token_ids),
        "stop_reason": result.stop_reason,
        "distinct_2": round(len(set(bigrams)) / len(bigrams), 6) if bigrams else None,
        "repeated_4gram_fraction": (
            round((len(fourgrams) - len(set(fourgrams))) / len(fourgrams), 6)
            if fourgrams
            else None
        ),
        "elapsed_seconds": round(result.elapsed_seconds, 6),
        "tokens_per_second": round(result.tokens_per_second, 6),
        "prompt_truncated": result.prompt_truncated,
        "cold_start_included": result.cold_start_included,
    }


def _package_version() -> str:
    try:
        return importlib.metadata.version("local-gpt2-xl")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _initial_record(
    config: ExperimentConfig,
    config_path: Path,
    plan: list[ExperimentCase],
    *,
    mode: str,
    seed_source: str,
) -> dict[str, object]:
    effective_config = config.to_dict()
    canonical_config = json.dumps(
        effective_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "result_schema_version": SCHEMA_VERSION,
        "status": "starting",
        "experiment": {
            "name": config.name,
            "description": config.description,
            "mode": mode,
            "seed_source": seed_source,
            "source_config": str(config_path),
            "effective_config_sha256": hashlib.sha256(canonical_config).hexdigest(),
            "prompt_utf8_sha256": hashlib.sha256(
                config.prompt.encode("utf-8")
            ).hexdigest(),
            "config": effective_config,
            "planned_samples": len(plan),
            "planned_cases": [asdict(case) for case in plan],
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "checkpoint_filename": MODEL_FILENAME,
            "expected_checkpoint_sha256": MODEL_SHA256,
            "integrity_verified_this_run": False,
            "context_tokens": MODEL_CONTEXT_TOKENS,
        },
        "application": {
            "package": "local-gpt2-xl",
            "version": _package_version(),
            "experiment_module_sha256": _file_sha256(Path(__file__)),
            "runtime_module_sha256": _file_sha256(Path(runtime_module.__file__)),
        },
        "execution": None,
        "started_at": _utc_now(),
        "completed_at": None,
        "runtime": None,
        "warmup": None,
        "samples": [],
        "error": None,
    }


def _markdown_block(text: str) -> str:
    longest = max((len(match) for match in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def _table_text(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(record: Mapping[str, object]) -> str:
    experiment = record["experiment"]
    if not isinstance(experiment, dict):
        raise ValueError("record has no experiment metadata")
    config = experiment["config"]
    if not isinstance(config, dict):
        raise ValueError("record has no effective configuration")
    samples = record.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("record samples must be a list")

    lines = [
        f"# {_table_text(experiment['name'])}",
        "",
        str(experiment.get("description", "")),
        "",
        f"- Status: **{record['status']}**",
        f"- Started (UTC): `{record['started_at']}`",
        f"- Completed (UTC): `{record.get('completed_at') or 'not yet'}`",
        f"- Completed samples: **{len(samples)} / {experiment['planned_samples']}**",
        "",
        "This is a text-completion experiment. The outputs show how the fixed prompt",
        "conditions GPT-2 XL under different decoding settings; they are not evidence",
        "that the model understands the prompt's claims or has a persistent identity.",
        "",
        (
            f"## Fixed prompt ({experiment['prompt_tokens']} GPT-2 tokens)"
            if "prompt_tokens" in experiment
            else "## Fixed prompt"
        ),
        "",
        _markdown_block(str(config["prompt"])),
        "",
        "## Experiment controls",
        "",
        "| Control | Value |",
        "| --- | --- |",
        f"| Mode | `{_table_text(experiment.get('mode', 'configured_matrix'))}` |",
        f"| Seed source | `{_table_text(experiment.get('seed_source', 'configuration'))}` |",
        f"| Temperatures | `{_table_text(config['temperatures'])}` |",
        f"| Sampling seeds | `{_table_text(config['seeds'])}` |",
        f"| Maximum new tokens | `{config['max_new_tokens']}` |",
        f"| Top-k | `{config['top_k']}` |",
        f"| Top-p | `{config['top_p']}` |",
        f"| Repetition penalty | `{config['repetition_penalty']}` |",
        f"| Warm-up tokens | `{config['warmup_tokens']}` |",
        "",
    ]

    if experiment.get("mode") == "serendipity":
        lines.extend(
            [
                "",
                "The serendipity seeds were selected before model loading and are",
                "shown above and beside every sample below. Use `replay-config.json`",
                "from this run to reproduce the same seed set and settings.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Temperature 0 uses greedy decoding once, because a random seed has",
                "no effect on that path. Every positive temperature is run once for",
                "each listed seed.",
            ]
        )

    lines.extend(
        [
            "",
            "## Model provenance",
            "",
            "| Field | Value |",
            "| --- | --- |",
        ]
    )

    model = record.get("model")
    if isinstance(model, dict):
        for key, value in model.items():
            lines.append(f"| {_table_text(key)} | `{_table_text(value)}` |")
    else:
        lines.append("| unavailable | `true` |")

    lines.extend(
        [
            "",
            "The checkpoint checksum above is the pinned expected value. This experiment",
            "does not re-hash the 5.99 GiB file. The `integrity_verified_this_run` field",
            "is authoritative; consult the project's verification record for any separate",
            "download-time integrity check.",
            "",
            "## Runtime",
            "",
        ]
    )

    runtime = record.get("runtime")
    if isinstance(runtime, dict):
        lines.extend(["| Field | Value |", "| --- | --- |"])
        for key, value in runtime.items():
            lines.append(f"| {_table_text(key)} | `{_table_text(value)}` |")
    else:
        lines.append("Runtime metadata is not available yet.")

    lines.extend(
        [
            "",
            "## Results summary",
            "",
            "| # | Temperature | Seed | Output tokens | Stop | Distinct-2 | Repeated 4-grams | Seconds | Tokens/s |",
            "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        seed = "greedy" if sample["seed"] is None else sample["seed"]
        lines.append(
            f"| {sample['index']} | {sample['temperature']} | {seed} | "
            f"{sample['output_tokens']} | {sample['stop_reason']} | "
            f"{sample['distinct_2'] if sample['distinct_2'] is not None else 'n/a'} | "
            f"{sample['repeated_4gram_fraction'] if sample['repeated_4gram_fraction'] is not None else 'n/a'} | "
            f"{sample['elapsed_seconds']:.2f} | "
            f"{sample['tokens_per_second']:.2f} |"
        )

    if not samples:
        lines.append("| — | — | — | — | — | — | — | — | — |")

    lines.extend(["", "Timings are operational observations, not a controlled benchmark."])

    for sample in samples:
        if not isinstance(sample, dict):
            continue
        seed = "greedy" if sample["seed"] is None else sample["seed"]
        lines.extend(
            [
                "",
                f"## Sample {sample['index']}: temperature {sample['temperature']}, seed {seed}",
                "",
                _markdown_block(str(sample["continuation"])),
            ]
        )

    error = record.get("error")
    if error:
        lines.extend(["", "## Error", "", _markdown_block(str(error))])

    return "\n".join(lines).rstrip() + "\n"


def _write_artifacts(run_directory: Path, record: Mapping[str, object]) -> None:
    _atomic_write(
        run_directory / "results.json",
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write(run_directory / "report.md", render_report(record))
    experiment = record.get("experiment")
    if isinstance(experiment, dict) and isinstance(experiment.get("config"), dict):
        _atomic_write(
            run_directory / "replay-config.json",
            json.dumps(experiment["config"], ensure_ascii=False, indent=2) + "\n",
        )


def run_experiment(
    config: ExperimentConfig,
    *,
    config_path: Path,
    output_root: Path,
    offline: bool = False,
    threads: int | None = None,
    mode: str = "configured_matrix",
    seed_source: str = "configuration",
    runner_factory: Callable[..., ExperimentRunner] = Gpt2Runner,
    runtime_summary_factory: Callable[[], dict[str, object]] = runtime_summary,
) -> Path:
    config.validate()
    if threads is not None and threads < 1:
        raise ValueError("threads must be at least 1")
    plan = build_plan(config)
    run_directory = _new_run_directory(output_root, config.name)
    record = _initial_record(
        config,
        config_path,
        plan,
        mode=mode,
        seed_source=seed_source,
    )
    record["execution"] = {
        "offline": offline,
        "requested_cpu_threads": threads,
        "case_order": "temperature order from config, then seed order",
        "warmup_before_recorded_samples": config.warmup_tokens > 0,
        "artifact_update_policy": "atomic replacement after every completion",
        "selection": {
            "mode": mode,
            "requested_samples": len(plan),
            "temperature_override": (
                config.temperatures[0] if mode == "serendipity" else None
            ),
            "seed_method": seed_source,
            "unique_seeds": len(config.seeds) == len(set(config.seeds)),
            "selected_seeds": list(config.seeds),
        },
    }
    _write_artifacts(run_directory, record)

    print(f"Experiment output: {run_directory}", file=sys.stderr, flush=True)
    print(f"Planned completions: {len(plan)}", file=sys.stderr, flush=True)

    try:
        runner = runner_factory(offline=offline, threads=threads)
        runtime = runtime_summary_factory()
        # The legacy runtime-summary key is retained for external compatibility but
        # omitted here because the experiment has an explicit expected/live distinction.
        runtime.pop("checkpoint_sha256", None)
        parameter_count = getattr(runner, "parameter_count", None)
        if parameter_count is not None:
            runtime["parameter_count"] = parameter_count
        runtime.setdefault("backend", "PyTorch")
        runtime.setdefault("device", "CPU")
        runtime.setdefault("dtype", "FP32")
        record["runtime"] = runtime
        execution = record["execution"]
        if isinstance(execution, dict):
            execution["actual_cpu_threads"] = runtime.get("cpu_threads")
        prompt_tokens = runner.count_prompt_tokens(config.prompt)
        experiment = record["experiment"]
        if isinstance(experiment, dict):
            experiment["prompt_tokens"] = prompt_tokens
            experiment["maximum_prompt_tokens"] = (
                MODEL_CONTEXT_TOKENS - config.max_new_tokens
            )
        if prompt_tokens + config.max_new_tokens > MODEL_CONTEXT_TOKENS:
            raise ValueError(
                f"experiment prompt has {prompt_tokens} tokens but at most "
                f"{MODEL_CONTEXT_TOKENS - config.max_new_tokens} fit with "
                f"max_new_tokens={config.max_new_tokens}; controlled experiments "
                "do not silently truncate prompts"
            )
        record["status"] = "running"
        _write_artifacts(run_directory, record)

        if config.warmup_tokens:
            print(
                f"Warm-up: {config.warmup_tokens} greedy token(s)...",
                file=sys.stderr,
                flush=True,
            )
            warmup_started = _utc_now()
            warmup = runner.generate_result(
                config.prompt,
                GenerationSettings(
                    max_new_tokens=config.warmup_tokens,
                    temperature=0,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                ),
            )
            record["warmup"] = {
                "started_at": warmup_started,
                "completed_at": _utc_now(),
                **_result_fields(warmup),
            }
            _write_artifacts(run_directory, record)

        for case in plan:
            seed_label = "greedy" if case.seed is None else str(case.seed)
            print(
                f"[{case.index}/{len(plan)}] temperature={case.temperature:g}, "
                f"seed={seed_label}",
                file=sys.stderr,
                flush=True,
            )
            settings = _settings_for(config, case)
            sample_started = _utc_now()
            result = runner.generate_result(config.prompt, settings)
            sample = {
                "index": case.index,
                "temperature": case.temperature,
                "seed": case.seed,
                "settings": asdict(settings),
                "started_at": sample_started,
                "completed_at": _utc_now(),
                **_result_fields(result),
            }
            samples = record["samples"]
            if not isinstance(samples, list):
                raise RuntimeError("internal error: samples record is not a list")
            samples.append(sample)
            _write_artifacts(run_directory, record)

        record["status"] = "complete"
        record["completed_at"] = _utc_now()
        _write_artifacts(run_directory, record)
        print(f"Experiment complete: {run_directory}", file=sys.stderr, flush=True)
        return run_directory
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        record["completed_at"] = _utc_now()
        record["error"] = "Interrupted by user"
        _write_artifacts(run_directory, record)
        raise
    except Exception as exc:
        record["status"] = "failed"
        record["completed_at"] = _utc_now()
        record["error"] = f"{type(exc).__name__}: {exc}"
        _write_artifacts(run_directory, record)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpt2-experiment",
        description=(
            "Run a reproducible temperature/seed matrix with local GPT-2 XL and "
            "save JSON plus Markdown results."
        ),
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"experiment JSON (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"parent directory for timestamped runs (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use only files already present in the local model cache",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="override PyTorch's CPU thread count",
    )
    parser.add_argument(
        "--serendipity",
        type=int,
        metavar="N",
        help=(
            "run N samples at one temperature using unique OS-random seeds "
            f"(maximum: {MAX_SERENDIPITY_SAMPLES})"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help=(
            "sampling temperature for --serendipity "
            f"(default: {DEFAULT_SERENDIPITY_TEMPERATURE})"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the config and print the expanded plan without loading GPT-2",
    )
    return parser


def _plan_document(
    config: ExperimentConfig,
    plan: list[ExperimentCase],
    *,
    mode: str,
    seed_source: str,
) -> dict[str, object]:
    return {
        "mode": mode,
        "seed_source": seed_source,
        "config": config.to_dict(),
        "sample_count": len(plan),
        "cases": [asdict(case) for case in plan],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config_path = args.config.resolve()
        config = load_config(config_path)
        mode = "configured_matrix"
        seed_source = "configuration"
        if args.serendipity is not None:
            temperature = (
                args.temperature
                if args.temperature is not None
                else DEFAULT_SERENDIPITY_TEMPERATURE
            )
            config = make_serendipity_config(
                config,
                count=args.serendipity,
                temperature=temperature,
            )
            mode = "serendipity"
            seed_source = "OS entropy via Python secrets.randbelow"
        elif args.temperature is not None:
            raise ValueError("--temperature requires --serendipity")

        plan = build_plan(config)
        if args.dry_run:
            print(
                json.dumps(
                    _plan_document(
                        config,
                        plan,
                        mode=mode,
                        seed_source=seed_source,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        run_experiment(
            config,
            config_path=config_path,
            output_root=args.output_root.resolve(),
            offline=args.offline,
            threads=args.threads,
            mode=mode,
            seed_source=seed_source,
        )
        return 0
    except KeyboardInterrupt:
        print("Experiment interrupted; partial results were preserved.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
