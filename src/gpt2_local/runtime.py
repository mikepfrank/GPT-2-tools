from __future__ import annotations

import hashlib
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

MODEL_ID = "openai-community/gpt2-xl"
MODEL_REVISION = "15ea56dee5df4983c59b2538573817e1667135e2"
MODEL_FILENAME = "model.safetensors"
MODEL_SHA256 = "0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a"
MODEL_CONTEXT_TOKENS = 1024


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 80
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    seed: int | None = None

    def validate(self) -> None:
        if not 1 <= self.max_new_tokens < MODEL_CONTEXT_TOKENS:
            raise ValueError(
                f"max_new_tokens must be between 1 and {MODEL_CONTEXT_TOKENS - 1}"
            )
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and cannot be negative")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")
        if (
            not math.isfinite(self.repetition_penalty)
            or self.repetition_penalty <= 0
        ):
            raise ValueError("repetition_penalty must be finite and positive")
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 2**63 - 1
        ):
            raise ValueError("seed must be an integer between 0 and 2^63 - 1")


@dataclass(frozen=True)
class GenerationResult:
    """A continuation plus the measurements needed to reproduce a run."""

    text: str
    original_input_tokens: int
    input_tokens: int
    output_tokens: int
    output_token_ids: tuple[int, ...]
    stop_reason: str
    elapsed_seconds: float
    tokens_per_second: float
    prompt_truncated: bool
    cold_start_included: bool


def _format_gib(byte_count: int) -> str:
    return f"{byte_count / (1024**3):.2f} GiB"


def _sha256(path: Path, progress: Callable[[str], None] | None = None) -> str:
    total = path.stat().st_size
    consumed = 0
    next_report = 0.1
    digest = hashlib.sha256()

    with path.open("rb") as model_file:
        while chunk := model_file.read(8 * 1024 * 1024):
            digest.update(chunk)
            consumed += len(chunk)
            fraction = consumed / total if total else 1
            if progress is not None and fraction >= next_report:
                progress(f"  verified {fraction:.0%} ({_format_gib(consumed)})")
                next_report += 0.1

    return digest.hexdigest()


def download_model(*, verify: bool = True) -> Path:
    """Download only the files needed for the PyTorch/safetensors checkpoint."""
    from huggingface_hub import hf_hub_download

    required_files = (
        "config.json",
        "generation_config.json",
        "merges.txt",
        MODEL_FILENAME,
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )

    print(f"Downloading the required {MODEL_ID} files into the Hugging Face cache...")
    paths: dict[str, Path] = {}
    for filename in required_files:
        downloaded = hf_hub_download(
            repo_id=MODEL_ID,
            filename=filename,
            revision=MODEL_REVISION,
        )
        paths[filename] = Path(downloaded)

    model_path = paths[MODEL_FILENAME]
    print(f"Checkpoint: {model_path} ({_format_gib(model_path.stat().st_size)})")

    if verify:
        print("Verifying the checkpoint SHA-256...")
        actual = _sha256(model_path, progress=print)
        if actual != MODEL_SHA256:
            raise RuntimeError(
                "Checkpoint integrity check failed. "
                f"Expected {MODEL_SHA256}, received {actual}."
            )
        print(f"SHA-256 verified: {actual}")

    return model_path


class Gpt2Runner:
    def __init__(
        self,
        *,
        offline: bool = False,
        threads: int | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The runtime is not installed. Run scripts/setup-wsl.sh first."
            ) from exc

        self._torch = torch
        self.model_id = MODEL_ID
        self._generation_count = 0

        if threads is not None:
            if threads < 1:
                raise ValueError("threads must be at least 1")
            torch.set_num_threads(threads)

        print(f"Loading {MODEL_ID} on CPU in FP32...", file=sys.stderr, flush=True)
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=offline,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=offline,
        )
        self.model.eval()
        self.parameter_count = self.model.num_parameters()
        elapsed = time.perf_counter() - started
        print(
            f"Model mapped in {elapsed:.1f}s: {self.parameter_count:,} parameters, "
            f"{torch.get_num_threads()} CPU threads.",
            file=sys.stderr,
            flush=True,
        )

    def count_prompt_tokens(self, prompt: str) -> int:
        """Count GPT-2 tokens without adding model-specific special tokens."""
        if not prompt:
            raise ValueError("prompt cannot be empty")
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))

    def generate(
        self,
        prompt: str,
        settings: GenerationSettings,
        *,
        include_prompt: bool = False,
    ) -> str:
        """Generate text and return only the decoded text for CLI compatibility."""
        return self.generate_result(
            prompt,
            settings,
            include_prompt=include_prompt,
        ).text

    def generate_result(
        self,
        prompt: str,
        settings: GenerationSettings,
        *,
        include_prompt: bool = False,
    ) -> GenerationResult:
        """Generate text and return structured timing and token-count metadata."""
        settings.validate()
        if not prompt:
            raise ValueError("prompt cannot be empty")

        torch = self._torch
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        original_input_length = inputs["input_ids"].shape[-1]
        input_length = original_input_length
        maximum_prompt_length = MODEL_CONTEXT_TOKENS - settings.max_new_tokens
        prompt_truncated = input_length > maximum_prompt_length
        if input_length > maximum_prompt_length:
            print(
                f"Prompt has {input_length} tokens; keeping its final "
                f"{maximum_prompt_length} tokens to fit GPT-2's context window.",
                file=sys.stderr,
            )
            inputs = {
                name: tensor[:, -maximum_prompt_length:]
                for name, tensor in inputs.items()
            }
            input_length = maximum_prompt_length

        if settings.seed is not None:
            torch.manual_seed(settings.seed)

        sampling = settings.temperature > 0
        generation_arguments: dict[str, object] = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": sampling,
            "repetition_penalty": settings.repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if sampling:
            generation_arguments.update(
                temperature=settings.temperature,
                top_k=settings.top_k,
                top_p=settings.top_p,
            )

        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation_arguments)
        elapsed = time.perf_counter() - started

        generated_tokens = output.shape[-1] - input_length
        rate = generated_tokens / elapsed if elapsed else float("inf")
        cold_start_included = self._generation_count == 0
        first_request_note = (
            " First request includes cold page-in and kernel setup."
            if cold_start_included
            else ""
        )
        self._generation_count += 1
        print(
            f"End-to-end: generated {generated_tokens} output tokens in "
            f"{elapsed:.1f}s ({rate:.2f} tokens/s).{first_request_note}",
            file=sys.stderr,
        )

        generated = output[0, input_length:]
        generated_token_ids = tuple(int(token_id) for token_id in generated.tolist())
        stopped_on_eos = bool(
            generated_token_ids
            and generated_token_ids[-1] == self.tokenizer.eos_token_id
        )
        selected = output[0] if include_prompt else generated
        return GenerationResult(
            text=self.tokenizer.decode(selected, skip_special_tokens=True),
            original_input_tokens=original_input_length,
            input_tokens=input_length,
            output_tokens=generated_tokens,
            output_token_ids=generated_token_ids,
            stop_reason="eos_token" if stopped_on_eos else "max_new_tokens",
            elapsed_seconds=elapsed,
            tokens_per_second=rate,
            prompt_truncated=prompt_truncated,
            cold_start_included=cold_start_included,
        )


def runtime_summary() -> dict[str, object]:
    try:
        import tokenizers
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "The runtime is not installed. Run scripts/setup-wsl.sh first."
        ) from exc

    return {
        "info_schema_version": 1,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        # Compatibility alias: this is the expected pinned digest, not a live hash.
        "checkpoint_sha256": MODEL_SHA256,
        "expected_checkpoint_sha256": MODEL_SHA256,
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_threads": torch.get_num_threads(),
        "logical_cpus": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
    }
