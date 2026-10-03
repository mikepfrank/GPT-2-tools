#!/usr/bin/env python3
"""Short Linux CPU-FP32 measurements; never download or load in --dry-run."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from gpt2_local.runtime import (  # noqa: E402
    GenerationSettings,
    Gpt2Runner,
    MODEL_CONTEXT_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    runtime_summary,
)

DEFAULT_PROMPT = (
    "Context: This is a conversation with GPT-2, a language model trained in 2019.\n\n"
    "Human> What are some interesting ideas about reversible computing?\n\nGPT-2>"
)
NEAR_FULL_PARAGRAPH = (
    "Reversible computing investigates how information can be preserved during "
    "computation. A reversible transformation permits each input state to be "
    "recovered from the resulting output state. Physical implementations involve "
    "questions about energy, noise, control, and the practical cost of operations. "
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def memory_snapshot() -> dict[str, int | None]:
    """Host MemAvailable and the process's cgroup-v1/v2 headroom, if limited."""
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        fields[key] = int(value.strip().split()[0]) * 1024
    available = fields["MemAvailable"]
    root = Path("/sys/fs/cgroup")
    candidates = {(root, "memory.max", "memory.current")}
    membership = Path("/proc/self/cgroup")
    if membership.exists():
        for line in membership.read_text().splitlines():
            parts = line.split(":", 2)
            if len(parts) != 3 or not parts[2].startswith("/") or ".." in parts[2].split("/"):
                continue
            if parts[0] == "0" and not parts[1]:
                group_root, limit_name, current_name = root, "memory.max", "memory.current"
            elif "memory" in parts[1].split(","):
                group_root, limit_name, current_name = root / "memory", "memory.limit_in_bytes", "memory.usage_in_bytes"
            else:
                continue
            group = group_root.joinpath(*parts[2].strip("/").split("/"))
            while True:
                candidates.add((group, limit_name, current_name))
                if group == group_root:
                    break
                group = group.parent
    limits = []
    for group, limit_name, current_name in candidates:
        limit_path = group / limit_name
        if not limit_path.exists():
            continue
        raw_limit = limit_path.read_text().strip()
        if raw_limit == "max":
            continue
        limit = int(raw_limit)
        if limit >= 2**60:  # cgroup v1's effectively-unlimited sentinel
            continue
        current = int((group / current_name).read_text())
        if limit < 0 or current < 0:
            raise ValueError("invalid cgroup memory capacity")
        limits.append(max(0, limit - current))
    cgroup_headroom = min(limits) if limits else None
    return {
        "mem_total_bytes": fields["MemTotal"],
        "mem_available_bytes": available,
        "swap_total_bytes": fields.get("SwapTotal", 0),
        "swap_free_bytes": fields.get("SwapFree", 0),
        "cgroup_headroom_bytes": cgroup_headroom,
        "effective_available_bytes": min(available, cgroup_headroom) if cgroup_headroom is not None else available,
    }


def resource_snapshot() -> dict[str, float]:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "user_seconds": usage.ru_utime,
        "system_seconds": usage.ru_stime,
        # Linux ru_maxrss is KiB, not bytes. It is a process-lifetime peak.
        "process_peak_rss_mib": usage.ru_maxrss / 1024,
    }


def source_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={PROJECT_DIR}", "rev-parse", "HEAD"],
            cwd=PROJECT_DIR, stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resource_delta(before: dict, after: dict, wall: float) -> dict:
    user = after["user_seconds"] - before["user_seconds"]
    system = after["system_seconds"] - before["system_seconds"]
    percent = 100 * (user + system) / wall if wall > 0 else None
    cpu_count = os.cpu_count() or 1
    return {
        "wall_seconds": wall,
        "cpu_user_seconds": user,
        "cpu_system_seconds": system,
        "cpu_percent_one_logical_cpu": percent,
        "cpu_percent_all_host_logical_cpus": percent / cpu_count if percent is not None else None,
        "process_peak_rss_mib": after["process_peak_rss_mib"],
    }


def save_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--threads", type=int, nargs="+", default=[1], help="short trials at each thread count; start at 1")
    result.add_argument("--samples", type=int, default=2, help="warm samples per prompt/thread count (1-4; default 2)")
    result.add_argument("--max-new-tokens", type=int, default=32, help="output cap from 1 through 64 (default 32)")
    result.add_argument("--temperature", type=float, choices=[0.0, 0.8], default=0.0)
    result.add_argument("--near-full-context", action="store_true", help="also measure a prompt filling the remaining 1024-token window")
    result.add_argument("--minimum-available-memory-gib", type=float, default=8.0, help="pre-load host/cgroup available RAM floor; swap excluded (default 8)")
    result.add_argument("--allow-low-memory", action="store_true", help="deliberately bypass the pre-load guard; may OOM this and unrelated services")
    result.add_argument("--dry-run", action="store_true", help="record the plan and memory check; no torch import or model load")
    result.add_argument("--output", type=Path, help="new JSON path; default ignored outputs/benchmarks/<UTC>-<pid>.json")
    return result


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if platform.system() != "Linux":
        argument_parser.error("this resource benchmark targets Linux")
    if not args.threads or any(thread < 1 or thread > 32 for thread in args.threads):
        argument_parser.error("thread counts must be between 1 and 32")
    if len(args.threads) != len(set(args.threads)):
        argument_parser.error("thread counts must be distinct")
    if not 1 <= args.samples <= 4 or not 1 <= args.max_new_tokens <= 64:
        argument_parser.error("samples must be 1-4 and max-new-tokens must be 1-64")
    if not math.isfinite(args.minimum_available_memory_gib) or args.minimum_available_memory_gib <= 0:
        argument_parser.error("minimum-available-memory-gib must be finite and positive")
    output = args.output or PROJECT_DIR / "outputs" / "benchmarks" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}.json"
    )
    if output.exists():
        argument_parser.error("output already exists; choose a new file to preserve prior evidence")
    try:
        memory = memory_snapshot()
    except (OSError, ValueError, KeyError) as exc:
        argument_parser.error(f"cannot verify available memory: {exc}")
    floor = args.minimum_available_memory_gib * 1024**3
    guard_passed = memory["effective_available_bytes"] >= floor
    record = {
        "benchmark_schema_version": 1,
        "status": "planned",
        "started_at": utc_now(),
        "command": sys.argv if argv is None else [str(Path(__file__)), *argv],
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "expected_checkpoint_sha256": MODEL_SHA256, "precision": "FP32", "device": "CPU"},
        "plan": {"threads": args.threads, "samples_per_prompt_per_threads": args.samples, "max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "seed": 42, "warmup_tokens_per_thread_count": 1, "near_full_context": args.near_full_context, "prompt": DEFAULT_PROMPT, "offline": True},
        "host": {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version, "logical_cpus": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None, "cpu_model_names": sorted({line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")})},
        "source": {"git_revision": source_revision(), "requirements_lock_sha256": hashlib.sha256((PROJECT_DIR / "requirements.lock.txt").read_bytes()).hexdigest(), "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "preload_memory": memory,
        "memory_guard": {"minimum_available_gib": args.minimum_available_memory_gib, "passed": guard_passed, "deliberate_override": args.allow_low_memory},
        "model_load": None,
        "warmups": [],
        "samples": [],
        "timing_scope": "model-load construction wall time includes runtime imports; first warm-up includes lazy page-in/kernel setup; generation TPS includes prompt prefill, decoding and stopping; no time-to-first-token measurement",
        "resource_scope": "process CPU times; RSS is a lifetime peak, not a per-sample allocation; host-normalized CPU is not a CloudWatch observation",
    }
    save_record(output, record)
    print(f"Memory guard: available {memory['effective_available_bytes'] / 1024**3:.2f} GiB; required {args.minimum_available_memory_gib:.2f} GiB.", flush=True)
    if args.dry_run:
        record["status"] = "dry_run"
        record["finished_at"] = utc_now()
        save_record(output, record)
        print(f"Plan recorded; model was not loaded: {output}")
        return 0
    if not guard_passed and not args.allow_low_memory:
        record["status"] = "declined_insufficient_memory"
        record["finished_at"] = utc_now()
        save_record(output, record)
        print("Refusing to load GPT-2 XL: insufficient available RAM. Resolve the host memory plan first.", file=sys.stderr)
        return 2
    try:
        record["status"] = "running"
        before = resource_snapshot()
        started = time.perf_counter()
        runner = Gpt2Runner(offline=True, threads=args.threads[0])
        record["model_load"] = resource_delta(before, resource_snapshot(), time.perf_counter() - started)
        record["runtime"] = runtime_summary()
        record["runtime"]["installed_dependencies"] = {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        }
        record["model"]["parameter_count"] = runner.parameter_count
        prompts = [("short_chat", DEFAULT_PROMPT)]
        if args.near_full_context:
            target = MODEL_CONTEXT_TOKENS - args.max_new_tokens
            tokens = runner.tokenizer.encode(NEAR_FULL_PARAGRAPH * 100, add_special_tokens=False)[:target]
            prompt = runner.tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
            if runner.tokenizer.encode(prompt, add_special_tokens=False) != tokens:
                raise RuntimeError("near-full prompt did not preserve exact tokenizer round trip")
            prompts.append(("near_full_context", prompt))
        record["prompts"] = {
            name: {"text": prompt, "utf8_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "input_token_ids": runner.tokenizer.encode(prompt, add_special_tokens=False), "input_tokens": runner.count_prompt_tokens(prompt)}
            for name, prompt in prompts
        }
        save_record(output, record)

        def measure(name: str, prompt: str, settings: GenerationSettings, threads: int, sample: int | None) -> dict:
            before = resource_snapshot()
            started = time.perf_counter()
            generated = runner.generate_result(prompt, settings)
            wall = time.perf_counter() - started
            return {
                "prompt_name": name,
                "threads": threads,
                "sample": sample,
                "settings": asdict(settings),
                "generation": asdict(generated),
                "text_utf8_sha256": hashlib.sha256(generated.text.encode()).hexdigest(),
                "resources": resource_delta(before, resource_snapshot(), wall),
                "memory_after": memory_snapshot(),
                "finished_at": utc_now(),
            }

        for threads in args.threads:
            runner._torch.set_num_threads(threads)
            warmup = measure("short_chat", DEFAULT_PROMPT, GenerationSettings(max_new_tokens=1, temperature=0, seed=42), threads, None)
            record["warmups"].append(warmup)
            save_record(output, record)
            for name, prompt in prompts:
                for sample in range(1, args.samples + 1):
                    measured = measure(name, prompt, GenerationSettings(max_new_tokens=args.max_new_tokens, temperature=args.temperature, seed=42), threads, sample)
                    record["samples"].append(measured)
                    save_record(output, record)
                    result = measured["generation"]
                    print(f"threads={threads} prompt={name} sample={sample}: {result['output_tokens']} tokens, {result['elapsed_seconds']:.2f}s, {result['tokens_per_second']:.3f} tokens/s", flush=True)
        record["status"] = "complete"
        return_code = 0
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        return_code = 130
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(record["error"], file=sys.stderr)
        return_code = 1
    record["finished_at"] = utc_now()
    save_record(output, record)
    print(f"Evidence saved: {output}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
