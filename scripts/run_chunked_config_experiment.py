from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from call_assistant.common.config import AppConfig
from call_assistant.orchestrator.worker import run_manual_step
from call_assistant.reprocess import reset_call_for_retranscription

BENCHMARK_CALL_ID = "20260404_164304_1b9540"
EXPECTED_PATH = Path("data/eval/post_russian_recovery/examples/call_20260404_164304_1b9540/expected_case.txt")

DEFAULT_COMBINATIONS = [
    {  # gradual increase
        "chunked_unknown_ratio_penalty": 0.35,
        "chunked_language_switch_penalty": 0.35,
        "chunked_repetitive_chunk_penalty": 0.3,
        "chunked_unsupported_penalty": 0.35,
    },
    {  # aggressive all penalties
        "chunked_unknown_ratio_penalty": 0.45,
        "chunked_language_switch_penalty": 0.45,
        "chunked_repetitive_chunk_penalty": 0.4,
        "chunked_unsupported_penalty": 0.45,
    },
    {  # zero unknown penalty (worst-case)
        "chunked_unknown_ratio_penalty": 0.0,
        "chunked_language_switch_penalty": 0.35,
        "chunked_repetitive_chunk_penalty": 0.3,
        "chunked_unsupported_penalty": 0.3,
    },
    {  # zero switch penalty
        "chunked_unknown_ratio_penalty": 0.35,
        "chunked_language_switch_penalty": 0.0,
        "chunked_repetitive_chunk_penalty": 0.3,
        "chunked_unsupported_penalty": 0.3,
    },
    {  # zero repetitive penalty
        "chunked_unknown_ratio_penalty": 0.35,
        "chunked_language_switch_penalty": 0.35,
        "chunked_repetitive_chunk_penalty": 0.0,
        "chunked_unsupported_penalty": 0.3,
    },
    {  # zero unsupported penalty
        "chunked_unknown_ratio_penalty": 0.35,
        "chunked_language_switch_penalty": 0.35,
        "chunked_repetitive_chunk_penalty": 0.3,
        "chunked_unsupported_penalty": 0.0,
    },
]


def tokenize_text(text: str) -> list[str]:
    return [
        token.strip(",:.;?!.\"⭐'" )
        for token in text.lower().split()
        if token.strip(",:.;?!.\"⭐'" )
    ]


def load_expected_tokens() -> list[str]:
    lines = EXPECTED_PATH.read_text(encoding="utf-8").splitlines()
    tokens: list[str] = []
    for line in lines:
        if ":" not in line:
            continue
        _, value = line.split(":", 1)
        tokens.extend(tokenize_text(value))
    return tokens


def compare_tokens(actual: str, reference_tokens: list[str]) -> float:
    actual_tokens = tokenize_text(actual)
    if not reference_tokens or not actual_tokens:
        return 0.0
    matches = sum(1 for token in actual_tokens if token in reference_tokens)
    return matches / max(len(reference_tokens), len(actual_tokens))


def update_transcription_settings(config: AppConfig, overrides: dict[str, Any]) -> None:
    section = config.section("transcription")
    section.setdefault("chunked_auto_fallback_to_legacy", False)
    section.setdefault("chunked_plausibility_min_score", 0.45)
    section.update(overrides)
    config.section("segment_detection")["enabled"] = False


def ensure_call_dir(config: AppConfig) -> Path:
    call_dir = config.archive_root / "2026/04/04/call_20260404_164304_1b9540"
    call_dir.mkdir(parents=True, exist_ok=True)
    return call_dir


def run_combo(base_config_path: Path, combo: dict[str, Any]) -> dict[str, Any]:
    config = AppConfig.load(base_config_path)
    config.ensure_directories()
    update_transcription_settings(config, combo)
    call_dir = ensure_call_dir(config)
    try:
        reset_call_for_retranscription(config, BENCHMARK_CALL_ID, call_dir, "local:large-v3-turbo", "he")
        run_manual_step(config, scan_first=False)
        transcript_path = call_dir / "transcript_raw.json"
        transcript_text = transcript_path.read_text(encoding="utf-8")
        metadata = json.loads((call_dir / "metadata.json").read_text(encoding="utf-8"))
        tokens_overlap = compare_tokens(json.loads(transcript_text).get("text", ""), load_expected_tokens())
        return {
            "combo": combo,
            "tokens_overlap": tokens_overlap,
            "chunked_score": metadata.get("transcription_chunked_plausibility_score"),
            "flags": metadata.get("transcription_chunked_plausibility_flags", []),
            "text": json.loads(transcript_text).get("text", ""),
            "error": None,
        }
    except Exception as exc:
        return {
            "combo": combo,
            "tokens_overlap": None,
            "chunked_score": None,
            "flags": [],
            "text": "",
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep chunked heuristics for the Hebrew benchmark call")
    parser.add_argument("--config", default="config.yaml", help="Base config file")
    parser.add_argument(
        "--output",
        default="chunked_experiment_results.json",
        help="Output file that will contain the serialized experiment data",
    )
    args = parser.parse_args()

    results = []
    for combo in DEFAULT_COMBINATIONS:
        print(f"Running combo: {combo}")
        result = run_combo(Path(args.config), combo)
        print(f"	-> tokens_overlap={result['tokens_overlap']} score={result['chunked_score']} flags={result['flags']} error={result['error']}")
        results.append(result)

    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote experiment report to {args.output}")


if __name__ == "__main__":
    main()
