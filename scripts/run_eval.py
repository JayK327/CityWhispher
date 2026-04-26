"""
scripts/run_eval.py
Runs the full evaluation suite against the golden set and logs
all metrics to MLflow for experiment tracking.

Usage:
  python scripts/run_eval.py

After running, view results:
  mlflow ui               # open http://localhost:5000
  mlflow runs list        # in terminal

What you'll see in MLflow:
  - Per-dimension scores (factual_accuracy, length_compliance, driving_safety)
  - Pass rate across the golden set
  - Run tagged with current prompt template git hash
  - Compare across prompt versions visually in the MLflow UI
"""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import mlflow.tracking

from app.config import settings
from app.models.poi import POIRecord, ContentCategory, Tone
from app.services.module2_llm.generator import generate_narration
from app.services.module5_eval.scorer import score_script

GOLDEN_SET = Path(__file__).parent.parent / "tests" / "golden_set.json"


def get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def dict_to_poi(sample: dict) -> POIRecord:
    facts = sample["source_facts"]
    return POIRecord(
        poi_id=sample["id"],
        name=facts["name"],
        lat=sample["lat"],
        lon=sample["lon"],
        category=ContentCategory(sample["category"]),
        description=facts.get("description", ""),
        source_url=facts.get("source_url", ""),
        address=facts.get("address", ""),
        opening_hours=facts.get("opening_hours", ""),
        source_count=2,
        confidence_score=0.85,
    )


async def run_evaluation():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("citywhisper_prompt_eval")

    with open(GOLDEN_SET) as f:
        samples = json.load(f)

    git_hash = get_git_hash()
    print(f"\nRunning eval on {len(samples)} golden samples (git: {git_hash})\n")

    all_scores = []

    with mlflow.start_run(run_name=f"eval_{git_hash}") as run:
        # Tag the run with metadata
        mlflow.set_tags({
            "git_commit":      git_hash,
            "model":           settings.llm_model,
            "word_target":     f"{settings.target_word_min}-{settings.target_word_max}",
            "prompt_template": "narrator.j2",
        })

        for i, sample in enumerate(samples):
            poi = dict_to_poi(sample)
            print(f"  [{i+1:2d}/{len(samples)}] {poi.name:<35}", end="", flush=True)

            try:
                script_result, prompt_tokens = await generate_narration(poi, Tone.informative)
                score = await score_script(script_result.script, sample["source_facts"])

                all_scores.append(score)

                status = "PASS" if score.passed else "FAIL"
                print(
                    f"  {status}  "
                    f"factual={score.factual_accuracy:.1f}  "
                    f"length={score.length_compliance:.1f}  "
                    f"safety={score.driving_safety:.1f}  "
                    f"overall={score.overall:.2f}  "
                    f"wc={score.word_count}"
                )

                # Log per-sample metrics
                mlflow.log_metrics({
                    f"sample_{sample['id']}_factual":  score.factual_accuracy,
                    f"sample_{sample['id']}_length":   score.length_compliance,
                    f"sample_{sample['id']}_safety":   score.driving_safety,
                    f"sample_{sample['id']}_overall":  score.overall,
                }, step=i)

            except Exception as e:
                print(f"  ERROR: {e}")

        # ── Aggregate metrics ─────────────────────────────────────────────────
        if all_scores:
            n = len(all_scores)
            agg = {
                "avg_factual_accuracy":  sum(s.factual_accuracy  for s in all_scores) / n,
                "avg_length_compliance": sum(s.length_compliance for s in all_scores) / n,
                "avg_driving_safety":    sum(s.driving_safety    for s in all_scores) / n,
                "avg_overall_score":     sum(s.overall           for s in all_scores) / n,
                "pass_rate":             sum(1 for s in all_scores if s.passed) / n,
                "avg_word_count":        sum(s.word_count         for s in all_scores) / n,
                "sample_count":          float(n),
            }

            mlflow.log_metrics(agg)

            print(f"\n{'─'*60}")
            print(f"  Results ({n} samples):")
            print(f"  Factual accuracy:   {agg['avg_factual_accuracy']:.3f}")
            print(f"  Length compliance:  {agg['avg_length_compliance']:.3f}")
            print(f"  Driving safety:     {agg['avg_driving_safety']:.3f}")
            print(f"  Overall score:      {agg['avg_overall_score']:.3f}")
            print(f"  Pass rate:          {agg['pass_rate']*100:.1f}%")
            print(f"  Avg word count:     {agg['avg_word_count']:.1f}")
            print(f"  Run ID:             {run.info.run_id}")
            print(f"  MLflow UI:          mlflow ui --port 5000")
            print(f"{'─'*60}\n")

            # Exit with failure code if pass rate is below threshold (for CI)
            if agg["pass_rate"] < 0.80:
                print("EVAL FAILED: pass rate below 80% threshold")
                sys.exit(1)
        else:
            print("No scores recorded — check for errors above.")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_evaluation())
