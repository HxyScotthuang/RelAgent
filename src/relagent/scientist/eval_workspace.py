"""Durable evaluation workspace: trials + eval_predictions DuckDB tables."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd


def _hash_queries(feature_queries: List[Dict[str, str]]) -> str:
    """Short stable hash of the feature query SQL."""
    canonical = json.dumps(
        sorted((q["name"], q["sql"]) for q in feature_queries),
        sort_keys=True,
    )
    return hashlib.md5(canonical.encode()).hexdigest()[:12]


class EvalWorkspace:
    """Writable DuckDB workspace for trial metadata and prediction results.

    Maintains two tables:
      - trials: one row per evaluation run
      - eval_predictions: row-level predictions for every evaluated example

    The workspace is stored in a file-backed DuckDB database so results
    persist after the run and can be queried offline.

    These tables are analysis-only. They must NOT become predictive feature
    sources for training new models.
    """

    _SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS trials (
        trial_id              TEXT PRIMARY KEY,
        trial_name            TEXT NOT NULL,
        parent_trial_id       TEXT,
        created_at            TIMESTAMPTZ NOT NULL,
        split                 TEXT NOT NULL,
        model_choice          TEXT,
        resolved_model_config TEXT,
        feature_query_hash    TEXT,
        feature_block_names   TEXT,
        primary_metric        TEXT,
        primary_score         DOUBLE,
        metrics_json          TEXT,
        notes                 TEXT
    );

    CREATE TABLE IF NOT EXISTS eval_predictions (
        trial_id        TEXT    NOT NULL,
        row_id          INTEGER NOT NULL,
        entity_id       TEXT,
        label           TEXT,
        score           DOUBLE,
        predicted_class TEXT,
        candidate_id    TEXT,
        rank            INTEGER,
        is_true_positive BOOL,
        split           TEXT    NOT NULL,
        eval_cutoff     TIMESTAMPTZ
    );
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self._db_path))
        self._conn.execute(self._SCHEMA_SQL)

    @property
    def db_path(self) -> Path:
        return self._db_path

    def insert_trial(
        self,
        trial_id: str,
        trial_name: str,
        split: str,
        metrics: Dict[str, float],
        primary_metric: str,
        primary_score: float,
        feature_queries: List[Dict[str, str]],
        model_choice: Optional[str] = None,
        resolved_model_config: Optional[Dict[str, Any]] = None,
        parent_trial_id: Optional[str] = None,
        notes: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Insert one trial record.

        Silently replaces an existing row with the same trial_id.
        """
        feature_block_names = ",".join(q["name"] for q in feature_queries)
        feature_query_hash = _hash_queries(feature_queries)
        metrics_json = json.dumps(metrics, default=str)
        resolved_cfg_json = (
            json.dumps(resolved_model_config, default=str)
            if resolved_model_config is not None
            else None
        )
        created_at = dt.datetime.now(dt.timezone.utc).isoformat()
        if error:
            notes = (notes or "") + f" [error: {error[:200]}]"

        # DELETE + INSERT to handle re-runs that collide on trial_id
        self._conn.execute("DELETE FROM trials WHERE trial_id = ?", [trial_id])
        self._conn.execute(
            """
            INSERT INTO trials (
                trial_id, trial_name, parent_trial_id, created_at, split,
                model_choice, resolved_model_config, feature_query_hash,
                feature_block_names, primary_metric, primary_score,
                metrics_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                trial_id, trial_name, parent_trial_id, created_at, split,
                model_choice, resolved_cfg_json, feature_query_hash,
                feature_block_names, primary_metric, primary_score,
                metrics_json, notes,
            ],
        )

    def insert_eval_predictions(self, predictions_df: pd.DataFrame) -> int:
        """Bulk-insert evaluation predictions. Returns number of rows written."""
        if predictions_df is None or predictions_df.empty:
            return 0
        n = len(predictions_df)
        try:
            self._conn.register("_ep_tmp", predictions_df)
            self._conn.execute(
                """
                INSERT INTO eval_predictions (
                    trial_id, row_id, entity_id, label,
                    score, predicted_class, split, eval_cutoff
                )
                SELECT trial_id, row_id, entity_id, label,
                       score, predicted_class, split, eval_cutoff
                FROM _ep_tmp
                """
            )
        finally:
            try:
                self._conn.unregister("_ep_tmp")
            except Exception:
                pass
        return n

    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute a read query and return results as a DataFrame."""
        return self._conn.execute(sql).df()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
