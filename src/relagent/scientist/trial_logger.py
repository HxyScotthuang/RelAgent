"""JSONL trial logger for scientist runs."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .validation import ProgramSpec, ValidationResult


class TrialLogger:
    """Logs trial results as JSONL + writes summary artifacts."""

    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._trials_path = self.artifact_dir / "trials.jsonl"
        self._trace_path = self.artifact_dir / "conversation_trace.jsonl"
        self._trials: List[Dict[str, Any]] = []

    def log_trace_event(self, event: Dict[str, Any]) -> None:
        """Append one interaction/trace event to conversation_trace.jsonl."""
        payload = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            **event,
        }
        with self._trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")

    def log_trial(
        self,
        program: ProgramSpec,
        result: ValidationResult,
        is_best: bool = False,
        split: str = "val",
    ) -> None:
        """Append one trial record to trials.jsonl."""
        record = {
            "trial_id": result.trial_id,
            "split": split,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "feature_queries": program.feature_queries,
            "model_choice": program.model_choice,
            "model_config": program.model_config,
            "metrics": result.metrics,
            "primary_score": result.score,
            "is_best": is_best,
            "n_predictions": result.n_predictions,
            "shap_importance_top30": result.shap_importance[:30],
            "shap_note": result.shap_note,
            "worst_predictions": result.worst_predictions[:5],  # truncate for JSONL
            "best_predictions": result.best_predictions[:5],
            "error": result.error,
        }
        self._trials.append(record)
        with self._trials_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def save_best_program(self, program: ProgramSpec) -> None:
        """Write best program to a separate JSON file."""
        path = self.artifact_dir / "best_program.json"
        path.write_text(
            json.dumps(program.to_dict(), indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def save_test_results(self, result: ValidationResult, program: ProgramSpec) -> None:
        """Write test set evaluation results."""
        path = self.artifact_dir / "test_results.json"
        payload = {
            "trial_id": result.trial_id,
            "split": "test",
            "metrics": result.metrics,
            "primary_metric": result.primary_metric_name,
            "primary_score": result.score,
            "n_predictions": result.n_predictions,
            "shap_importance_top30": result.shap_importance[:30],
            "shap_note": result.shap_note,
            "error": result.error,
            "program": program.to_dict(),
        }
        path.write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def save_report(self, best_score: float, best_trial_id: int, n_trials: int) -> None:
        """Write a human-readable markdown summary."""
        path = self.artifact_dir / "summary.md"
        lines = [
            "# Scientist Run Summary",
            "",
            f"- Total trials: {n_trials}",
            f"- Best trial: #{best_trial_id}",
            f"- Best score: {best_score:.6f}",
            "",
            "## Trial History",
            "",
            "| Trial | Score | Best? | Error |",
            "|-------|-------|-------|-------|",
        ]
        for t in self._trials:
            best_marker = "**yes**" if t["is_best"] else ""
            err = t["error"][:50] if t["error"] else ""
            lines.append(
                f"| {t['trial_id']} | {t['primary_score']:.4f} | {best_marker} | {err} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @property
    def trials(self) -> List[Dict[str, Any]]:
        return self._trials

    def get_history_summary(self) -> str:
        """Format trial history for the agent to see."""
        if not self._trials:
            return "No trials yet."

        lines = ["Trial History:"]
        best_score = float("-inf")
        for t in self._trials:
            if t["split"] != "val":
                continue
            score = t["primary_score"]
            improved = score > best_score
            if improved:
                best_score = score
            status = "OK" if t["error"] is None else f"FAILED: {t['error'][:80]}"
            marker = " (NEW BEST)" if improved and t["error"] is None else ""
            # Summarize approach from query names
            query_names = [q["name"] for q in t["feature_queries"]]
            lines.append(
                f"  Trial #{t['trial_id']}: score={score:.4f}{marker} | "
                f"queries=[{', '.join(query_names)}] | {status}"
            )
        lines.append(f"\nBest score so far: {best_score:.4f}")
        return "\n".join(lines)
