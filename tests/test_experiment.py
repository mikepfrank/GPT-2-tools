from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from gpt2_local.experiment import (
    ExperimentConfig,
    MAX_SERENDIPITY_SAMPLES,
    _atomic_write,
    build_plan,
    main,
    make_serendipity_config,
    render_report,
    run_experiment,
    select_unique_random_seeds,
)
from gpt2_local.runtime import GenerationResult


def _config(**overrides: object) -> ExperimentConfig:
    values: dict[str, object] = {
        "name": "test sweep",
        "description": "model-free test",
        "prompt": "Context: Test prompt",
        "temperatures": (0.0, 0.8),
        "seeds": (0, 1),
        "max_new_tokens": 4,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "warmup_tokens": 0,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def _result(text: str = " continuation") -> GenerationResult:
    return GenerationResult(
        text=text,
        original_input_tokens=5,
        input_tokens=5,
        output_tokens=4,
        output_token_ids=(10, 11, 10, 11),
        stop_reason="max_new_tokens",
        elapsed_seconds=2.0,
        tokens_per_second=2.0,
        prompt_truncated=False,
        cold_start_included=False,
    )


class CapturingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def count_prompt_tokens(self, prompt: str) -> int:
        return 4

    def generate_result(self, prompt: str, settings: object, **_: object) -> GenerationResult:
        self.calls.append((prompt, settings))
        return _result()


class ExperimentPlanTests(unittest.TestCase):
    def test_greedy_is_collapsed_and_sampled_seeds_include_zero(self) -> None:
        plan = build_plan(_config())
        self.assertEqual(
            [(case.temperature, case.seed) for case in plan],
            [(0.0, None), (0.8, 0), (0.8, 1)],
        )

    def test_sampled_temperature_requires_a_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one seed"):
            _config(temperatures=(0.8,), seeds=()).validate()

    def test_non_finite_temperature_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            _config(temperatures=(float("nan"),)).validate()

    def test_serendipity_seeds_are_unique_ordered_and_collision_safe(self) -> None:
        candidates = iter((7, 7, 0, 2**63 - 1))
        bounds: list[int] = []

        def choose(bound: int) -> int:
            bounds.append(bound)
            return next(candidates)

        self.assertEqual(
            select_unique_random_seeds(3, randbelow=choose),
            (7, 0, 2**63 - 1),
        )
        self.assertEqual(bounds, [2**63] * 4)

    def test_serendipity_config_replaces_matrix_and_rejects_bad_inputs(self) -> None:
        candidates = iter((101, 202, 303))
        config = make_serendipity_config(
            _config(),
            count=3,
            temperature=0.8,
            randbelow=lambda _: next(candidates),
        )
        self.assertEqual(config.temperatures, (0.8,))
        self.assertEqual(config.seeds, (101, 202, 303))
        self.assertEqual(
            [(case.temperature, case.seed) for case in build_plan(config)],
            [(0.8, 101), (0.8, 202), (0.8, 303)],
        )

        with self.assertRaisesRegex(ValueError, "greater than 0"):
            make_serendipity_config(_config(), count=1, temperature=0)
        with self.assertRaisesRegex(ValueError, "between 1"):
            make_serendipity_config(_config(), count=0, temperature=0.8)
        with self.assertRaisesRegex(ValueError, "between 1"):
            make_serendipity_config(
                _config(),
                count=MAX_SERENDIPITY_SAMPLES + 1,
                temperature=0.8,
            )

    def test_serendipity_cli_dry_run_records_selected_seeds_without_a_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(_config().to_dict()),
                encoding="utf-8",
            )
            output = StringIO()
            with (
                mock.patch(
                    "gpt2_local.experiment.select_unique_random_seeds",
                    return_value=(11, 22, 33),
                ),
                mock.patch("gpt2_local.experiment.run_experiment") as run,
                redirect_stdout(output),
            ):
                status = main(
                    [
                        str(config_path),
                        "--serendipity",
                        "3",
                        "--temperature",
                        "0.8",
                        "--dry-run",
                    ]
                )

        document = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        run.assert_not_called()
        self.assertEqual(document["mode"], "serendipity")
        self.assertEqual(document["config"]["temperatures"], [0.8])
        self.assertEqual(document["config"]["seeds"], [11, 22, 33])
        self.assertEqual(
            [case["seed"] for case in document["cases"]],
            [11, 22, 33],
        )

    def test_temperature_override_requires_serendipity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(_config().to_dict()),
                encoding="utf-8",
            )
            errors = StringIO()
            with (
                mock.patch("gpt2_local.experiment.run_experiment") as run,
                redirect_stderr(errors),
            ):
                status = main(
                    [str(config_path), "--temperature", "0.8", "--dry-run"]
                )

        self.assertEqual(status, 1)
        run.assert_not_called()
        self.assertIn("--temperature requires --serendipity", errors.getvalue())


class ExperimentExecutionTests(unittest.TestCase):
    def test_success_uses_one_runner_and_persists_canonical_results(self) -> None:
        runner = CapturingRunner()
        factory_calls: list[dict[str, object]] = []

        def factory(**kwargs: object) -> CapturingRunner:
            factory_calls.append(kwargs)
            return runner

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            run_directory = run_experiment(
                _config(),
                config_path=Path("test.json"),
                output_root=output_root,
                offline=True,
                threads=3,
                runner_factory=factory,
                runtime_summary_factory=lambda: {"pytorch": "test", "cpu_threads": 3},
            )
            record = json.loads((run_directory / "results.json").read_text("utf-8"))
            report = (run_directory / "report.md").read_text("utf-8")
            replay = json.loads(
                (run_directory / "replay-config.json").read_text("utf-8")
            )

        self.assertEqual(factory_calls, [{"offline": True, "threads": 3}])
        self.assertEqual(len(runner.calls), 3)
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["experiment"]["planned_samples"], 3)
        self.assertEqual(record["samples"][0]["output_token_ids"], [10, 11, 10, 11])
        self.assertEqual(record["samples"][0]["seed"], None)
        self.assertEqual(record["samples"][1]["seed"], 0)
        self.assertEqual(
            record["samples"][1]["seed"],
            record["samples"][1]["settings"]["seed"],
        )
        self.assertIn("## Sample 3", report)
        self.assertIn("continuation_sha256", record["samples"][0])
        self.assertEqual(replay, record["experiment"]["config"])

    def test_serendipity_execution_labels_and_reports_every_seed(self) -> None:
        config = _config(temperatures=(0.8,), seeds=(101, 202))
        with tempfile.TemporaryDirectory() as directory:
            run_directory = run_experiment(
                config,
                config_path=Path("test.json"),
                output_root=Path(directory),
                mode="serendipity",
                seed_source="test entropy",
                runner_factory=lambda **_: CapturingRunner(),
                runtime_summary_factory=lambda: {},
            )
            record = json.loads((run_directory / "results.json").read_text("utf-8"))
            report = (run_directory / "report.md").read_text("utf-8")

        self.assertEqual(record["experiment"]["mode"], "serendipity")
        self.assertEqual(record["experiment"]["seed_source"], "test entropy")
        self.assertEqual([sample["seed"] for sample in record["samples"]], [101, 202])
        self.assertIn("replay-config.json", report)
        self.assertIn("101", report)
        self.assertIn("202", report)

    def test_failure_preserves_completed_samples(self) -> None:
        class FailingRunner(CapturingRunner):
            def generate_result(
                self, prompt: str, settings: object, **kwargs: object
            ) -> GenerationResult:
                if self.calls:
                    raise RuntimeError("deliberate failure")
                return super().generate_result(prompt, settings, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "deliberate failure"):
                run_experiment(
                    _config(),
                    config_path=Path("test.json"),
                    output_root=output_root,
                    runner_factory=lambda **_: FailingRunner(),
                    runtime_summary_factory=lambda: {},
                )
            run_directory = next(output_root.iterdir())
            record = json.loads((run_directory / "results.json").read_text("utf-8"))

        self.assertEqual(record["status"], "failed")
        self.assertEqual(len(record["samples"]), 1)
        self.assertIn("deliberate failure", record["error"])

    def test_selected_seeds_survive_model_load_failure(self) -> None:
        config = _config(temperatures=(0.8,), seeds=(101, 202, 303))

        def fail_to_load(**_: object) -> CapturingRunner:
            raise RuntimeError("model load failed")

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "model load failed"):
                run_experiment(
                    config,
                    config_path=Path("test.json"),
                    output_root=output_root,
                    mode="serendipity",
                    seed_source="test entropy",
                    runner_factory=fail_to_load,
                    runtime_summary_factory=lambda: {},
                )
            run_directory = next(output_root.iterdir())
            record = json.loads((run_directory / "results.json").read_text("utf-8"))
            replay = json.loads(
                (run_directory / "replay-config.json").read_text("utf-8")
            )

        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["experiment"]["config"]["seeds"], [101, 202, 303])
        self.assertEqual(
            [case["seed"] for case in record["experiment"]["planned_cases"]],
            [101, 202, 303],
        )
        self.assertEqual(replay["seeds"], [101, 202, 303])

    def test_default_warmup_is_recorded_but_not_counted_as_a_sample(self) -> None:
        runner = CapturingRunner()
        with tempfile.TemporaryDirectory() as directory:
            run_directory = run_experiment(
                _config(temperatures=(0.0,), seeds=(), warmup_tokens=1),
                config_path=Path("test.json"),
                output_root=Path(directory),
                runner_factory=lambda **_: runner,
                runtime_summary_factory=lambda: {"cpu_threads": 2},
            )
            record = json.loads((run_directory / "results.json").read_text("utf-8"))

        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(runner.calls[0][1].max_new_tokens, 1)
        self.assertEqual(record["experiment"]["planned_samples"], 1)
        self.assertEqual(len(record["samples"]), 1)
        self.assertIsNotNone(record["warmup"])

    def test_overlong_prompt_fails_instead_of_silently_truncating(self) -> None:
        class LongPromptRunner(CapturingRunner):
            def count_prompt_tokens(self, prompt: str) -> int:
                return 1021

        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            with self.assertRaisesRegex(ValueError, "do not silently truncate"):
                run_experiment(
                    _config(max_new_tokens=4),
                    config_path=Path("test.json"),
                    output_root=output_root,
                    runner_factory=lambda **_: LongPromptRunner(),
                    runtime_summary_factory=lambda: {},
                )
            record = json.loads(
                (next(output_root.iterdir()) / "results.json").read_text("utf-8")
            )

        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["experiment"]["prompt_tokens"], 1021)
        self.assertEqual(record["samples"], [])

    def test_atomic_write_keeps_old_target_and_cleans_temp_on_replace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "results.json"
            target.write_text("old", encoding="utf-8")
            with mock.patch(
                "gpt2_local.experiment.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    _atomic_write(target, "new")
            self.assertEqual(target.read_text("utf-8"), "old")
            self.assertFalse((target.parent / ".results.json.tmp").exists())

    def test_report_uses_a_safe_fence_for_generated_backticks(self) -> None:
        config = _config().to_dict()
        record = {
            "status": "complete",
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T00:01:00Z",
            "experiment": {
                "name": "fence test",
                "description": "",
                "planned_samples": 1,
                "config": config,
            },
            "runtime": {},
            "samples": [
                {
                    "index": 1,
                    "temperature": 0.0,
                    "seed": None,
                    "output_tokens": 4,
                    "stop_reason": "max_new_tokens",
                    "distinct_2": 1.0,
                    "repeated_4gram_fraction": 0.0,
                    "elapsed_seconds": 1.0,
                    "tokens_per_second": 4.0,
                    "continuation": "contains ``` a fence",
                }
            ],
            "error": None,
        }
        report = render_report(record)
        self.assertIn("````text\ncontains ``` a fence\n````", report)


if __name__ == "__main__":
    unittest.main()
