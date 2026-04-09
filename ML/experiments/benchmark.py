from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ML.evaluation.metrics import character_error_rate, word_error_rate
from ML.inference.predict import predict_page


logger = logging.getLogger("ml.benchmark")


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _resolve_image_path(manifest_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Image not found: {candidate}")

    resolved = (manifest_path.parent / candidate).resolve()
    if resolved.exists():
        return resolved

    # Handle archive layout where files are extracted under TeluguSeg/TeluguSeg/...
    parts = list(candidate.parts)
    if "TeluguSeg" in parts:
        idx = parts.index("TeluguSeg")
        nested_parts = parts[: idx + 1] + ["TeluguSeg"] + parts[idx + 1 :]
        nested_candidate = Path(*nested_parts)
        nested_resolved = (manifest_path.parent / nested_candidate).resolve()
        if nested_resolved.exists():
            return nested_resolved

    raise FileNotFoundError(f"Image not found: {resolved}")


def run_benchmark(
    manifest_path: Path,
    output_path: Path,
    limit: int | None = None,
    log_every: int = 200,
) -> dict[str, Any]:
    logger.info("Loading manifest: %s", manifest_path)
    samples = _load_manifest(manifest_path)
    if not samples:
        raise ValueError("Manifest is empty.")

    if limit is not None and limit > 0:
        samples = samples[:limit]

    total = len(samples)
    logger.info("Loaded %s samples", total)

    item_reports: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    failures = 0

    for idx, sample in enumerate(samples, start=1):
        try:
            image_path = _resolve_image_path(manifest_path, str(sample["image_path"]))
            reference = str(sample.get("text", ""))

            with Image.open(image_path) as image:
                start = time.perf_counter()
                prediction = predict_page(image)
                latency = (time.perf_counter() - start) * 1000.0

            hypothesis = str(prediction.get("telugu", ""))
            cer = character_error_rate(reference, hypothesis)
            wer = word_error_rate(reference, hypothesis)

            latencies_ms.append(latency)
            item_reports.append(
                {
                    "image_path": str(image_path),
                    "reference": reference,
                    "prediction": hypothesis,
                    "cer": cer,
                    "wer": wer,
                    "latency_ms": round(latency, 2),
                }
            )

            if idx == 1 or idx % max(1, log_every) == 0 or idx == total:
                logger.info("Progress %s/%s | last_latency_ms=%.2f", idx, total, latency)
        except Exception as exc:
            failures += 1
            logger.exception("Failed sample at index %s: %s", idx, exc)

    if not item_reports:
        raise RuntimeError("No successful samples were processed.")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "num_samples": len(item_reports),
        "failed_samples": failures,
        "cer_mean": round(statistics.mean([item["cer"] for item in item_reports]), 4),
        "wer_mean": round(statistics.mean([item["wer"] for item in item_reports]), 4),
        "latency_ms_mean": round(statistics.mean(latencies_ms), 2),
        "latency_ms_p95": round(sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))], 2),
    }

    report = {"summary": summary, "items": item_reports}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PaddleOCR baseline benchmark from JSONL manifest")
    parser.add_argument("--manifest", required=True, help="Path to JSONL manifest with image_path,text")
    parser.add_argument("--output", required=False, help="Optional output report path")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of samples to evaluate")
    parser.add_argument("--log-every", type=int, default=200, help="Progress print interval in samples")
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    manifest_path = Path(args.manifest).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    default_output = PROJECT_ROOT / "ML" / "evaluation" / "reports" / f"baseline_{timestamp}.json"
    output_path = Path(args.output).resolve() if args.output else default_output

    report = run_benchmark(
        manifest_path=manifest_path,
        output_path=output_path,
        limit=args.limit,
        log_every=args.log_every,
    )
    logger.info("Benchmark complete | samples=%s failures=%s", report["summary"]["num_samples"], report["summary"]["failed_samples"])
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
