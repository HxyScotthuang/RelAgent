"""FunctionTool definitions for the scientist agent."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

from camel.toolkits import FunctionTool

from .eval_workspace import EvalWorkspace
from .validation import (
    ProgramSpec,
    ValidationResult,
    execute_and_validate,
    VALID_SEVEN_MODEL_CHOICES,
    _resolve_model_config,
)
from .trial_logger import TrialLogger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workspace trial ID helpers
# ---------------------------------------------------------------------------

def _workspace_trial_id(result_trial_id: int, split: str) -> str:
    if result_trial_id == -2:
        return "final_val"
    if result_trial_id == -1:
        return "test"
    return f"{split}_{result_trial_id:04d}"


def _workspace_trial_name(
    trial_counter: int,
    model_choice: Optional[str],
    feature_queries: List[Dict[str, str]],
) -> str:
    query_names = "+".join(q["name"] for q in feature_queries[:3])
    if len(feature_queries) > 3:
        query_names += f"+{len(feature_queries) - 3}more"
    model_part = model_choice or "custom"
    return f"trial_{trial_counter}_{model_part}_{query_names}"[:100]


def _write_trial_to_workspace(
    workspace: EvalWorkspace,
    program: ProgramSpec,
    result: ValidationResult,
    split: str,
    entity_col: str = "",
) -> Dict[str, Any]:
    """Write one trial (metadata + predictions) to the workspace."""
    ws_trial_id = _workspace_trial_id(result.trial_id, split)
    ws_trial_name = _workspace_trial_name(
        result.trial_id, program.model_choice, program.feature_queries
    )

    resolved_cfg: Optional[Dict[str, Any]] = None
    if result.wrapped_diagnostics:
        resolved_cfg = result.wrapped_diagnostics.get("resolved_model_config")

    pname = result.primary_metric_name or ""
    primary_score = result.metrics.get(pname, result.score) if result.metrics else result.score

    workspace.insert_trial(
        trial_id=ws_trial_id,
        trial_name=ws_trial_name,
        split=split,
        metrics=result.metrics,
        primary_metric=pname,
        primary_score=primary_score,
        feature_queries=program.feature_queries,
        model_choice=program.model_choice,
        resolved_model_config=resolved_cfg,
        error=result.error,
    )

    n_written = 0
    if result.eval_predictions_df is not None:
        preds_df = result.eval_predictions_df.copy()
        preds_df["trial_id"] = ws_trial_id
        n_written = workspace.insert_eval_predictions(preds_df)
        result.eval_predictions_df = None

    return {
        "trial_id": ws_trial_id,
        "trial_name": ws_trial_name,
        "parent_trial_id": None,
        "n_eval_preds_written": n_written,
    }


# ---------------------------------------------------------------------------
# Tool factories
# ---------------------------------------------------------------------------

def make_validate_program_wrapped_tool(
    conn: duckdb.DuckDBPyConnection,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    evaluator: Any,  # RelBenchEvaluator
    task_type: str,
    entity_col: str,
    target_col: str,
    trial_logger: TrialLogger,
    tool_name: str = "validate_program",
    split: str = "val",
    show_coverage_feedback: bool = False,
    require_asof_for_multi_entity: bool = False,
    asof_col: Optional[str] = None,
    extra_entity_cols: Optional[List[str]] = None,
    regression_primary_metric: str = "mae",
    sql_timeout_seconds: float = 300.0,
    eval_workspace: Optional[EvalWorkspace] = None,
    max_train_entities: Optional[int] = None,
    seven_models: bool = False,
    fixed_model_choice: Optional[str] = None,
) -> FunctionTool:
    """Create the validate_program FunctionTool for wrapped-model mode."""
    trial_counter = [0]
    best_score = [float("-inf")]
    if fixed_model_choice is not None:
        _effective_choices = (fixed_model_choice,)
    elif seven_models:
        _effective_choices = VALID_SEVEN_MODEL_CHOICES
    else:
        from .validation import VALID_MODEL_CHOICES
        _effective_choices = VALID_MODEL_CHOICES

    def validate_program(
        feature_queries_json: str,
        model_choice: str,
        model_config_json: str = "{}",
    ) -> str:
        """Test a predictive program using a wrapped model on a data split.

        Args:
            feature_queries_json: JSON string of feature queries.
                Format: [{"name": "query_name", "sql": "SELECT ..."}]
                Each query must include the entity column and return feature columns.
                `eval_table` is available in SQL and contains the split rows currently being scored.
                `train_table` is available in SQL and contains the labeled training rows.
            model_choice: One of "gbdt", "rf", "dart", "goss", "xgboost", "xgb_dart", "catboost".
                - gbdt     : Standard Gradient Boosted Trees (histogram, fast, strong default)
                - rf       : Random Forest (bagging of trees; less sensitive to learning rate)
                - dart     : DART Boosting (dropout regularization; can reduce overfitting)
                - goss     : GOSS (gradient-based subsampling; fast on large datasets)
                - xgboost  : XGBoost (second-order gradients; different regularization profile)
                - xgb_dart : XGBoost + DART dropout
                - catboost : CatBoost (ordered boosting; robust on heterogeneous features)
            model_config_json: JSON dict of hyperparameters (optional, strong defaults apply).
                Shared keys (gbdt/rf/dart/goss/xgboost/xgb_dart):
                  n_estimators (50-500), learning_rate (0.01-0.3), max_depth (2-10),
                  subsample (0.5-1.0), colsample_bytree (0.5-1.0)
                gbdt/rf/dart/goss only: min_child_samples (1-100)
                gbdt/rf/dart/goss regularization: lambda_l1 (0.0-10.0), lambda_l2 (0.0-10.0)
                xgboost/xgb_dart only: min_child_weight (1-100)
                xgboost/xgb_dart regularization: reg_alpha (0.0-10.0), reg_lambda (0.0-10.0)
                catboost only: l2_leaf_reg (0.1-10.0)
                Regression objective (gbdt/rf/dart/goss): "regression_l1" (MAE), "regression_l2" (MSE), "huber"
                Regression objective (xgboost/xgb_dart): "reg:absoluteerror", "reg:squarederror", "reg:pseudohubererror"
                Native categorical features: add "categorical_features": ["<query_name>__<column>", ...]
                Out-of-bounds numeric values are clamped. Unknown keys are ignored.

        Returns:
            Formatted string with metrics, diagnostics, and trial history.
        """
        try:
            feature_queries = json.loads(feature_queries_json)
            if not isinstance(feature_queries, list):
                return f"ERROR: feature_queries_json must be a JSON array, got {type(feature_queries).__name__}"
        except json.JSONDecodeError as e:
            return f"ERROR: Invalid JSON in feature_queries_json: {e}"

        for i, q in enumerate(feature_queries):
            if not isinstance(q, dict) or "name" not in q or "sql" not in q:
                return f'ERROR: Query {i} must be a dict with "name" and "sql" keys, got: {q}'

        if fixed_model_choice is not None:
            model_choice = fixed_model_choice

        if model_choice not in _effective_choices:
            return (
                f"ERROR: Invalid model_choice '{model_choice}'. "
                f"Must be one of: {', '.join(_effective_choices)}"
            )

        try:
            model_config = json.loads(model_config_json) if model_config_json else {}
            if not isinstance(model_config, dict):
                return f"ERROR: model_config_json must be a JSON object, got {type(model_config).__name__}"
        except json.JSONDecodeError as e:
            return f"ERROR: Invalid JSON in model_config_json: {e}"

        trial_counter[0] += 1
        trial_id = trial_counter[0]

        program = ProgramSpec(
            feature_queries=feature_queries,
            model_choice=model_choice,
            model_config=model_config,
        )

        result = execute_and_validate(
            program=program,
            conn=conn,
            val_df=val_df,
            evaluator=evaluator,
            task_type=task_type,
            entity_col=entity_col,
            target_col=target_col,
            trial_id=trial_id,
            split=split,
            require_asof_for_multi_entity=require_asof_for_multi_entity,
            asof_col=asof_col,
            extra_entity_cols=extra_entity_cols,
            regression_primary_metric=regression_primary_metric,
            train_df=train_df,
            sql_timeout_seconds=sql_timeout_seconds,
            store_eval_preds=(eval_workspace is not None),
            max_train_entities=max_train_entities,
            seven_models=seven_models,
        )

        is_best = result.error is None and result.score > best_score[0]
        if is_best:
            best_score[0] = result.score

        trial_logger.log_trial(program, result, is_best=is_best, split=split)

        workspace_info: Optional[Dict[str, Any]] = None
        if eval_workspace is not None:
            try:
                workspace_info = _write_trial_to_workspace(
                    eval_workspace, program, result, split, entity_col=entity_col
                )
            except Exception as _ws_err:
                logger.warning(f"Workspace write failed for trial {trial_id}: {_ws_err}")

        return _format_validation_result(
            result,
            best_score[0],
            is_best,
            split=split,
            show_coverage_feedback=show_coverage_feedback,
            workspace_info=workspace_info,
        )

    validate_program.__name__ = tool_name
    return FunctionTool(validate_program)


def make_trial_history_tool(trial_logger: TrialLogger) -> FunctionTool:
    """Create the get_trial_history FunctionTool."""

    def get_trial_history() -> str:
        """Get a summary of all previous validation trials.

        Returns:
            Formatted string showing trial history with scores and approach summaries.
        """
        return trial_logger.get_history_summary()

    return FunctionTool(get_trial_history)


def make_query_eval_workspace_tool(workspace: EvalWorkspace) -> FunctionTool:
    """Create the query_eval_workspace FunctionTool."""

    def query_eval_workspace(sql: str) -> str:
        """Query the evaluation workspace to analyse trial results.

        The workspace is a DuckDB database. Tables:

          trials
            trial_id TEXT, trial_name TEXT, parent_trial_id TEXT,
            created_at TIMESTAMPTZ, split TEXT, model_choice TEXT,
            resolved_model_config TEXT, feature_query_hash TEXT,
            feature_block_names TEXT, primary_metric TEXT,
            primary_score DOUBLE, metrics_json TEXT, notes TEXT

          eval_predictions
            trial_id TEXT, row_id INTEGER, entity_id TEXT, label TEXT,
            score DOUBLE, predicted_class TEXT, split TEXT,
            eval_cutoff TIMESTAMPTZ

        row_id in eval_predictions is stable across trials for the same split,
        so you can join two trials on row_id to compare predictions.

        IMPORTANT: These tables are analysis artifacts only. Do NOT use them
        as feature sources in your SQL feature queries.

        Args:
            sql: A SELECT query to run against the workspace.

        Returns:
            Query results (up to 500 rows) or an error message.
        """
        sql_lower = sql.lower().strip()
        for banned in ("drop ", "delete ", "update ", "insert ", "alter ", "create "):
            if banned in sql_lower:
                return (
                    f"ERROR: Only SELECT queries are allowed in query_eval_workspace. "
                    f"Got banned keyword: '{banned.strip()}'"
                )
        try:
            df = workspace.execute_query(sql)
        except Exception as e:
            return f"ERROR: {e}"

        if df.empty:
            return "(no rows)"

        truncated = ""
        if len(df) > 500:
            df = df.head(500)
            truncated = "\n... (truncated to 500 rows)"

        return df.to_string(index=False) + truncated

    return FunctionTool(query_eval_workspace)


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def _format_validation_result(
    result: ValidationResult,
    best_score: float,
    is_new_best: bool,
    split: str = "val",
    show_coverage_feedback: bool = False,
    workspace_info: Optional[Dict[str, Any]] = None,
) -> str:
    lines = []

    if result.error:
        lines.append(f"Trial #{result.trial_id} FAILED:")
        lines.append(f"  Error: {result.error}")
        if workspace_info:
            lines.append(f"\n  Workspace trial_id: {workspace_info['trial_id']}")
        lines.append(f"\nBest score so far: {best_score:.4f}")
        return "\n".join(lines)

    lines.append(f"Trial #{result.trial_id} Results ({split}):")
    for metric, value in result.metrics.items():
        lines.append(f"  {metric}: {value:.4f}")

    if is_new_best:
        lines.append(f"\n  *** NEW BEST SCORE: {result.score:.4f} ***")
    else:
        lines.append(f"\n  Best score so far: {best_score:.4f} (this trial: {result.score:.4f})")

    lines.append(f"  Predictions made: {result.n_predictions}")
    if show_coverage_feedback:
        lines.append(
            "  Coverage: "
            f"{result.total_entities - result.missing_predictions}/{result.total_entities} "
            f"({result.coverage_rate:.2%})"
        )
        if result.missing_predictions > 0:
            lines.append(f"  Missing predictions filled with defaults: {result.missing_predictions}")

    diag = result.wrapped_diagnostics
    if diag:
        lines.append(f"\n  Model: {diag.get('resolved_model_choice', '?')}")
        lines.append(f"  Config: {json.dumps(diag.get('resolved_model_config', {}))}")

        for w in diag.get("config_warnings", []):
            lines.append(f"  WARNING: {w}")

        lines.append(f"  Training rows: {diag.get('n_train_rows', '?')}")
        lines.append(f"  Val rows: {diag.get('n_val_rows', diag.get('n_val_entities', '?'))}")
        lines.append(f"  Feature count: {diag.get('n_features', '?')}")

        train_diag = diag.get("train", {})
        merge_counts = train_diag.get("merge_row_counts", {})
        if merge_counts:
            lines.append("  Merge row counts (train):")
            for qname, count in merge_counts.items():
                lines.append(f"    After '{qname}': {count} rows")

        missingness = train_diag.get("missingness", {})
        if missingness:
            lines.append("  Missingness (train):")
            for qname, rate in missingness.items():
                lines.append(f"    '{qname}': {rate:.1%}")

        for w in train_diag.get("warnings", []):
            lines.append(f"  WARNING: {w}")

        if "candidate_col" in diag:
            lines.append(f"  Candidate column: {diag['candidate_col']}")
            lines.append(f"  Positive pairs: {diag.get('n_positive_pairs', '?')}")
            lines.append(f"  Negative pairs: {diag.get('n_negative_pairs', '?')}")
            lines.append(f"  Candidates scored: {diag.get('n_candidates_scored', '?')}")

    resid_corr_df = (result.wrapped_diagnostics or {}).get("residual_corr_df")

    if workspace_info:
        _append_workspace_diagnostics(lines, result, workspace_info)
    else:
        if resid_corr_df is not None and not resid_corr_df.empty:
            lines.append(_format_residual_correlations(resid_corr_df))
        else:
            if result.best_predictions:
                lines.append("\nTop-10 BEST predictions:")
                lines.append(_format_prediction_table(result.best_predictions))
            if result.worst_predictions:
                lines.append("\nTop-10 WORST predictions:")
                lines.append(_format_prediction_table(result.worst_predictions))

    return "\n".join(lines)


def _append_workspace_diagnostics(
    lines: List[str],
    result: ValidationResult,
    workspace_info: Dict[str, Any],
) -> None:
    ws_id = workspace_info["trial_id"]
    ws_name = workspace_info["trial_name"]
    n_written = workspace_info.get("n_eval_preds_written", 0)

    lines.append("\nFull evaluation results persisted to workspace:")
    lines.append(f"  trial_id   : {ws_id}")
    lines.append(f"  trial_name : {ws_name}")
    if workspace_info.get("parent_trial_id"):
        lines.append(f"  parent     : {workspace_info['parent_trial_id']}")
    lines.append(f"  primary_metric : {result.primary_metric_name or '?'}")
    lines.append(f"  metrics    : {json.dumps(result.metrics, default=str)}")
    if result.wrapped_diagnostics:
        lines.append(f"  model      : {result.wrapped_diagnostics.get('resolved_model_choice', '?')}")
        lines.append(f"  config     : {json.dumps(result.wrapped_diagnostics.get('resolved_model_config', {}))}")
    lines.append(f"  rows written to eval_predictions: {n_written}")
    lines.append("")
    lines.append("  Use query_eval_workspace() for error analysis. Example:")
    lines.append(f"    SELECT entity_id, label, score,")
    lines.append(f"           ABS(CAST(label AS DOUBLE) - score) AS abs_err")
    lines.append(f"    FROM eval_predictions WHERE trial_id = '{ws_id}'")
    lines.append(f"    ORDER BY abs_err DESC LIMIT 20;")


def _format_prediction_table(predictions: List[Dict[str, Any]], max_cols: int = 8) -> str:
    if not predictions:
        return "  (none)"

    all_cols = list(predictions[0].keys())
    cols = all_cols[:max_cols]
    if len(all_cols) > max_cols:
        cols.append("...")

    header = "  | " + " | ".join(str(c)[:20] for c in cols) + " |"
    sep = "  |" + "|".join("-" * 22 for _ in cols) + "|"

    rows = [header, sep]
    for pred in predictions[:10]:
        vals = []
        for c in cols[:max_cols]:
            v = pred.get(c, "")
            s = str(v)
            if len(s) > 20:
                s = s[:17] + "..."
            vals.append(s)
        if len(all_cols) > max_cols:
            vals.append("...")
        rows.append("  | " + " | ".join(v.ljust(20) for v in vals) + " |")

    return "\n".join(rows)


def _format_residual_correlations(resid_corr_df: "pd.DataFrame", top_n: int = 15) -> str:
    if resid_corr_df is None or resid_corr_df.empty:
        return ""

    lines = [
        "\nResidual correlations — features correlated with prediction error (y − ŷ):",
        "  Positive r: model UNDER-predicts when feature is high → boost signal.",
        "  Negative r: model OVER-predicts when feature is high → dampen signal.",
    ]

    top = resid_corr_df.head(top_n)
    for _, row in top.iterrows():
        rank = int(row.get("rank", 0))
        fname = str(row.get("feature_name", "?"))
        extractor = str(row.get("extractor_name", ""))
        corr = row.get("correlation", float("nan"))
        sign = "+" if corr >= 0 else "-"
        abs_c = abs(corr)
        extractor_tag = f"  [{extractor}]" if extractor and extractor not in fname else ""
        lines.append(f"  #{rank:>2}  {fname:<50}  r={sign}{abs_c:.4f}{extractor_tag}")

    n_total = len(resid_corr_df)
    if n_total > top_n:
        lines.append(f"  ... {n_total - top_n} more features not shown.")
    return "\n".join(lines)
