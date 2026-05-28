"""
Run v0.7.2 progressive evaluation (1k/2k/3k/4k) on n=800 test set
using Unsloth-accelerated inference backend.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


MODELS = {
    "1k": "models/llava_v07_qlora_1k_img336",
    "2k": "models/llava_v07_qlora_2k",
    "3k": "models/llava_v07_qlora_3k",
    "4k": "models/llava_v07_qlora_4k",
}


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\n$", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def build_report(evaluation_files: dict[str, Path], report_md: Path, summary_csv: Path) -> None:
    rows = []
    for stage in ["1k", "2k", "3k", "4k"]:
        if stage not in evaluation_files:
            continue
        data = json.loads(evaluation_files[stage].read_text(encoding="utf-8"))
        cs = data["overall_metrics"]["cosine_similarity"]
        rows.append(
            {
                "stage": stage,
                "mean": cs["mean"],
                "std": cs["std"],
                "median": cs["median"],
                "q25": cs["q25"],
                "q75": cs["q75"],
                "min": cs["min"],
                "max": cs["max"],
            }
        )

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        f.write("stage,mean,std,median,q25,q75,min,max\n")
        for r in rows:
            f.write(
                f"{r['stage']},{r['mean']:.6f},{r['std']:.6f},{r['median']:.6f},"
                f"{r['q25']:.6f},{r['q75']:.6f},{r['min']:.6f},{r['max']:.6f}\n"
            )

    lines = []
    lines.append("# v0.7.2 Unsloth Inference Evaluation (n=800)\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("\n## Cosine Similarity Comparison\n")
    lines.append("| Scale | Mean ± Std | Median | Q25 | Q75 | Min | Max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['stage']} | {r['mean']:.4f} ± {r['std']:.4f} | {r['median']:.4f} | "
            f"{r['q25']:.4f} | {r['q75']:.4f} | {r['min']:.4f} | {r['max']:.4f} |"
        )

    if rows:
        base = rows[0]
        lines.append("\n## Improvement vs 1k\n")
        for r in rows[1:]:
            delta = r["mean"] - base["mean"]
            lines.append(f"- {r['stage']}: {r['mean']:.4f} ({delta:+.4f} vs 1k)")

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:
    parser = argparse.ArgumentParser(description="v0.7.2 Unsloth progressive evaluation")
    parser.add_argument("--base-dir", type=str, default=".")
    parser.add_argument("--test-csv", type=str, default="data/v07_fine_tuning/test_set_n800.csv")
    parser.add_argument("--image-dir", type=str, default="data/inspect_images_336")
    parser.add_argument("--output-dir", type=str, default="data/v07_fine_tuning/evaluations_unsloth")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stages", nargs="*", default=["1k", "2k", "3k", "4k"])
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    test_csv = base_dir / args.test_csv
    image_dir = base_dir / args.image_dir
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    evaluation_files: dict[str, Path] = {}

    for stage in args.stages:
        if stage not in MODELS:
            raise ValueError(f"Unknown stage: {stage}")

        model_dir = base_dir / MODELS[stage]
        if not model_dir.exists():
            raise FileNotFoundError(f"Model dir not found: {model_dir}")

        inference_csv = output_dir / f"inference_results_v07_{stage}.csv"
        eval_json = output_dir / f"evaluation_v07_{stage}.json"

        if args.skip_existing and inference_csv.exists() and eval_json.exists():
            print(f"[skip] {stage}: existing outputs found")
            evaluation_files[stage] = eval_json
            continue

        infer_cmd = [
            sys.executable,
            "inference_v051_qlora.py",
            "--use-unsloth",
            "--model-dir",
            str(model_dir),
            "--test-csv",
            str(test_csv),
            "--output-csv",
            str(inference_csv),
            "--image-dir",
            str(image_dir),
            "--batch-size",
            str(args.batch_size),
            "--max-tokens",
            str(args.max_tokens),
        ]
        if args.limit is not None:
            infer_cmd.extend(["--limit", str(args.limit)])

        run_cmd(infer_cmd, base_dir)

        eval_cmd = [
            sys.executable,
            "evaluate_cosine_hf.py",
            "--csv",
            str(inference_csv),
            "--gt-col",
            "ground_truth",
            "--pred-col",
            "prediction",
            "--id-col",
            "image_id",
            "--output",
            str(eval_json),
            "--model",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ]
        run_cmd(eval_cmd, base_dir)
        evaluation_files[stage] = eval_json

    report_md = output_dir / "v072_unsloth_progressive_report.md"
    summary_csv = output_dir / "v072_unsloth_cosine_summary.csv"
    build_report(evaluation_files, report_md, summary_csv)

    print("\nDone")
    print(f"- Report: {report_md}")
    print(f"- Summary: {summary_csv}")


if __name__ == "__main__":
    main()
