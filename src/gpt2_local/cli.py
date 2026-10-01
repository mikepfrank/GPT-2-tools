from __future__ import annotations

import argparse
import json
import sys

from .runtime import (
    Gpt2Runner,
    GenerationSettings,
    download_model,
    runtime_summary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpt2-xl",
        description=(
            "Run OpenAI's original 1.5B-parameter GPT-2 XL checkpoint locally. "
            "With no prompt, starts an interactive session."
        ),
    )
    parser.add_argument("prompt", nargs="?", help="text for GPT-2 to continue")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="download and verify the checkpoint without loading it",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="skip the 6.43 GB checkpoint SHA-256 check",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="show installed runtime information and exit",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use only files already present in the local cache",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="maximum continuation length (default: 80)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="sampling temperature; 0 selects greedy decoding (default: 0.8)",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--threads",
        type=int,
        help="override PyTorch's CPU thread count",
    )
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="print the prompt together with its continuation",
    )
    return parser


def _settings(args: argparse.Namespace) -> GenerationSettings:
    return GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )


def _interactive(runner: Gpt2Runner, settings: GenerationSettings) -> None:
    print(
        "GPT-2 XL interactive mode. Enter a prompt, or use /quit to leave.\n"
        "Each prompt is independent; GPT-2 is a text completer, not a chat model."
    )
    while True:
        try:
            prompt = input("\ngpt2-xl> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if prompt.strip().lower() in {"/q", "/quit", "/exit"}:
            return
        if not prompt.strip():
            continue

        print(runner.generate(prompt, settings))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        if args.download_only:
            download_model(verify=not args.skip_verification)
            return 0

        if args.info:
            print(json.dumps(runtime_summary(), indent=2))
            return 0

        settings = _settings(args)
        settings.validate()
        runner = Gpt2Runner(
            offline=args.offline,
            threads=args.threads,
        )

        if args.prompt is None:
            _interactive(runner, settings)
        else:
            print(
                runner.generate(
                    args.prompt,
                    settings,
                    include_prompt=args.include_prompt,
                )
            )
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
