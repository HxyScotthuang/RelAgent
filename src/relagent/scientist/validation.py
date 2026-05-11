"""Program execution engine: run SQL queries + wrapped model, score via RelBenchEvaluator."""

from __future__ import annotations

import gc
import logging
import math
import re
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _coerce_ml_feature_column(series: pd.Series) -> pd.Series:
    """Convert a feature column to float64 for ML (DuckDB often yields nullable Int64).

    Prevents pandas errors when SQL returns fractional values (e.g. 25.5) in a column
    previously inferred as Int64.
    """
    return pd.to_numeric(series, errors="coerce").astype(np.float64)


@dataclass
class ProgramSpec:
    """A scientist program = SQL feature queries + wrapped model."""

    feature_queries: List[Dict[str, str]]
    # Each dict has {"name": str, "sql": str}
    model_choice: Optional[str] = None
    # "logreg"|"lightgbm"|"xgboost"|"ridge" (standard); "gbdt"|"rf"|"dart"|"goss"|"xgb_dart"|"catboost" (seven_models)
    model_config: Optional[Dict[str, Any]] = None
    # Bounded hyperparameters for the chosen model family

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProgramSpec":
        return cls(
            feature_queries=data["feature_queries"],
            model_choice=data.get("model_choice"),
            model_config=data.get("model_config"),
        )


@dataclass
class ValidationResult:
    """Result of running a program against a validation set."""

    trial_id: int
    score: float  # Primary metric (higher = better)
    metrics: Dict[str, float]
    worst_predictions: List[Dict[str, Any]]
    best_predictions: List[Dict[str, Any]]
    n_predictions: int
    missing_predictions: int = 0
    total_entities: int = 0
    coverage_rate: float = 1.0
    shap_importance: List[Dict[str, Any]] = field(default_factory=list)
    shap_note: Optional[str] = None
    error: Optional[str] = None
    wrapped_diagnostics: Optional[Dict[str, Any]] = None
    # Name of the primary metric (e.g. "roc_auc", "mae", "map@k")
    primary_metric_name: Optional[str] = None
    # Full prediction DataFrame for persisting to EvalWorkspace.
    # Populated only when store_eval_preds=True is passed to execute_and_validate.
    # Columns: row_id, entity_id, label, score, predicted_class, split, eval_cutoff.
    # Note: trial_id column is NOT included here; the caller adds it before writing.
    eval_predictions_df: Optional[Any] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("eval_predictions_df", None)  # not serialisable / not needed
        return d


def _determine_primary_metric_name(
    task_type: str,
    metrics: Dict[str, float],
    regression_primary_metric: str = "mae",
) -> str:
    """Return the human-readable name of the primary metric for this result."""
    if task_type == "entity_regression":
        return (regression_primary_metric or "mae").lower()
    # Classification: prefer threshold-free ranking metrics; mrr first for multi-class
    return next(
        (k for k in ("mrr", "roc_auc", "auroc", "average_precision", "macro_f1", "micro_f1", "f1", "accuracy") if k in metrics),
        (list(metrics.keys())[0] if metrics else "score"),
    )


# Row advisory for SQL feature queries: log when result is large (memory can spike on merge).
MAX_SQL_ROWS_SOFT = 500_000

# Hard cap for node-level (classification/regression): queries aggregate per entity → 500 rows/entity is generous.
NODE_ROWS_PER_ENTITY = 500

# Subsample target for large non-LP train sets (used when max_train_entities is set).
# rel-amazon user-ltv has ~1.5M customers × 3 timestamps = 4.7M rows which OOMs
# at 512G.  500K entities (~1.5M rows) keeps peak feature DataFrame memory manageable.
MAX_REGRESSION_TRAIN_ENTITIES = 500_000

# Fallback hard cap when entity count is unknown.
MAX_SQL_ROWS_HARD_FALLBACK = 10_000_000

# Log at most this many characters of SQL at INFO (full text at DEBUG, capped).
FEATURE_SQL_PREVIEW_CHARS = 800

# Heuristic: many JOIN tokens often correlates with cartesian blowups / high memory.
FEATURE_SQL_JOIN_WARN_THRESHOLD = 8
FEATURE_SQL_CROSS_JOIN_WARN_THRESHOLD = 1


def _sql_join_heuristics(sql: str) -> Dict[str, int]:
    """Cheap parse-free heuristics for logging (not a full SQL parser)."""
    s = sql.lower()
    # " join " avoids matching column names like "joint_id" in most cases.
    join_spaced = s.count(" join ")
    join_keywords = len(re.findall(r"(?i)\bjoin\b", sql))
    cross_joins = len(re.findall(r"\bcross\s+join\b", s))
    comma_from = s.count(",")  # comma joins / wide SELECT lists — weak signal only
    return {
        "join_spaced_count": join_spaced,
        "join_keyword_hits": join_keywords,
        "cross_join_count": cross_joins,
        "comma_count": comma_from,
        "sql_chars": len(sql),
    }


def _feature_sql_preview(sql: str, max_chars: int = FEATURE_SQL_PREVIEW_CHARS) -> str:
    one_line = " ".join(sql.split())
    if len(one_line) <= max_chars:
        return one_line
    return one_line[: max_chars - 3] + "..."

# Default wall-clock limit per feature-query SQL (DuckDB execute); interrupt if exceeded.
DEFAULT_SQL_QUERY_TIMEOUT_SECONDS = 300.0

SQL_TIMEOUT_MESSAGE = (
    "SQL took too long (time limit exceeded). "
    "Reduce intermediate size: tighter filters, earlier aggregation, smaller joins or time windows, "
    "or pre-aggregate before joining to eval_table."
)


def _execute_sql_with_timeout(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    *,
    query_name: str,
    timeout_seconds: float,
) -> pd.DataFrame:
    """Run `conn.execute(sql).df()` with a wall-clock cap via DuckDB interrupt."""
    if timeout_seconds is None or timeout_seconds <= 0:
        return conn.execute(sql).df()

    done = threading.Event()

    def _interrupt_after() -> None:
        if done.wait(timeout=timeout_seconds):
            return
        try:
            conn.interrupt()
        except Exception:
            pass

    intr = threading.Thread(target=_interrupt_after, daemon=True)
    intr.start()
    try:
        return conn.execute(sql).df()
    except duckdb.InterruptException as e:
        raise ValueError(
            f"Query '{query_name}': {SQL_TIMEOUT_MESSAGE} (limit {timeout_seconds:.0f}s)"
        ) from e
    finally:
        done.set()


def _aggressive_collect() -> None:
    """Force a local GC cycle to reduce RSS growth across trials."""
    try:
        gc.collect()
    except Exception:
        pass


def _run_feature_queries(
    conn: duckdb.DuckDBPyConnection,
    feature_queries: List[Dict[str, str]],
    *,
    entity_col: Optional[str] = None,
    allowed_entities: Optional[pd.Series] = None,
    sql_timeout_seconds: float = DEFAULT_SQL_QUERY_TIMEOUT_SECONDS,
    n_entities: Optional[int] = None,
    n_candidates: Optional[int] = None,  # unused, kept for call-site compatibility
    eval_df: Optional[pd.DataFrame] = None,  # unused, kept for call-site compatibility
) -> Dict[str, pd.DataFrame]:
    """Execute SQL feature queries and return a dict of DataFrames.

    Cap logic (scales with entity count):
      - Node mode: hard cap = n_entities × NODE_ROWS_PER_ENTITY (500 rows/entity).
        Node queries should never legitimately exceed this; excess → raise.
      - Fallback (no entity count): MAX_SQL_ROWS_HARD_FALLBACK.
    """
    results: Dict[str, pd.DataFrame] = {}

    allowed_set = None
    if entity_col and allowed_entities is not None:
        # Convert to a Python set for fast membership tests via pandas.isin()
        # (works for ints/strings; drops nulls defensively).
        allowed_set = set(allowed_entities.dropna().tolist())

    # Hard cap scales with entity count.
    if n_entities is not None and n_entities > 0:
        row_hard_cap = n_entities * NODE_ROWS_PER_ENTITY
    else:
        row_hard_cap = MAX_SQL_ROWS_HARD_FALLBACK

    for q in feature_queries:
        name = q["name"]
        sql = q["sql"]
        jh = _sql_join_heuristics(sql)
        logger.debug(
            "Feature SQL [%s] full (truncated to 50k chars):\n%s",
            name,
            sql[:50_000] + ("…[truncated]" if len(sql) > 50_000 else ""),
        )
        logger.info(
            "Feature SQL [%s] start: sql_chars=%d join_spaced=%d join_keywords=%d cross_joins=%d preview=%r",
            name,
            jh["sql_chars"],
            jh["join_spaced_count"],
            jh["join_keyword_hits"],
            jh["cross_join_count"],
            _feature_sql_preview(sql),
        )
        join_warn_level = max(jh["join_spaced_count"], jh["join_keyword_hits"])
        if join_warn_level >= FEATURE_SQL_JOIN_WARN_THRESHOLD:
            logger.warning(
                "Feature SQL [%s]: high join count (spaced=%d, keywords=%d; warn≥%d) — risk of large "
                "intermediates / OOM; prefer aggregating before joining to eval_table.",
                name,
                jh["join_spaced_count"],
                jh["join_keyword_hits"],
                FEATURE_SQL_JOIN_WARN_THRESHOLD,
            )
        if jh["cross_join_count"] >= FEATURE_SQL_CROSS_JOIN_WARN_THRESHOLD:
            logger.warning(
                "Feature SQL [%s]: contains CROSS JOIN (%d) — often explodes row count; check intent.",
                name,
                jh["cross_join_count"],
            )

        # Basic safety: block write operations
        sql_lower = sql.lower()
        for banned in ["drop ", "delete ", "update ", "insert ", "alter ", "create ", "attach "]:
            if banned in sql_lower:
                raise ValueError(f"Query '{name}' contains blocked SQL operation: {banned.strip()}")

        # Temporal leakage guard: reject queries that compare event timestamps to
        # eval-table timestamps in ways that include future events.
        # Pattern 1: event_col >= e.<time_col>  (events at or after entity creation)
        # Pattern 2: event_col < e.<time_col> + INTERVAL  (future window via positive offset)
        # Both allow events that occurred AFTER the entity's creation time, leaking the target.
        _sql_no_comments = re.sub(r"--[^\n]*", "", sql)  # strip line comments before scanning
        _future_ge = re.findall(
            r"\b\w+\s*>=\s*e\s*\.\s*\w+",
            _sql_no_comments,
            re.IGNORECASE,
        )
        # Filter out join-key equalities that also have >= (false positives rare, but check):
        # A genuine false positive would be e.g. `ON t.x >= e.x` for range joins — flag anyway.
        if _future_ge:
            raise ValueError(
                f"Query '{name}' uses a forward-looking temporal filter: {_future_ge!r}. "
                "Conditions like `event_col >= e.time_col` include events that occurred AFTER the "
                "entity's creation time, causing temporal leakage. Use `event_col < e.time_col` instead."
            )
        _future_interval = re.findall(
            r"\w+\s*<\s*e\s*\.\s*\w+\s*\+\s*INTERVAL",
            _sql_no_comments,
            re.IGNORECASE,
        )
        if _future_interval:
            raise ValueError(
                f"Query '{name}' uses a forward-looking temporal window: {_future_interval!r}. "
                "Conditions like `event_col < e.time_col + INTERVAL 'X'` include events up to X "
                "time after entity creation, causing temporal leakage. Use `event_col < e.time_col` "
                "or `event_col < e.time_col - INTERVAL 'X'` for a past-only window instead."
            )

        # Preflight COUNT: check row count before materializing into pandas to prevent OOM.
        # Wrap in try/except — if the COUNT itself fails or times out, proceed anyway.
        preflight_rows: Optional[int] = None
        try:
            count_sql = f"SELECT COUNT(*) AS __n__ FROM ({sql}) __count_subq__"
            count_df = _execute_sql_with_timeout(
                conn, count_sql, query_name=f"{name}__count", timeout_seconds=sql_timeout_seconds
            )
            preflight_rows = int(count_df["__n__"].iloc[0])
            cap_mode = (
                f"{n_entities:,}e × {NODE_ROWS_PER_ENTITY}/e = {row_hard_cap:,}" if n_entities is not None
                else f"fallback={row_hard_cap:,}"
            )
            logger.info("Feature SQL [%s] preflight count: %d rows (hard cap %s)", name, preflight_rows, cap_mode)
            if preflight_rows > row_hard_cap:
                raise ValueError(
                    f"Query '{name}' would return {preflight_rows:,} rows which exceeds the hard cap "
                    f"({cap_mode}). For node-level tasks, queries must aggregate per entity before returning results. "
                    f"Simplify the query: add tighter time-window filters, aggregate earlier, "
                    f"or reduce join fanout."
                )
        except ValueError:
            raise
        except Exception as preflight_err:
            logger.warning("Feature SQL [%s] preflight count failed (%s); running query anyway.", name, preflight_err)

        t0 = time.perf_counter()
        try:
            df = _execute_sql_with_timeout(
                conn, sql, query_name=name, timeout_seconds=sql_timeout_seconds
            )
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Query '{name}' failed: {e}") from e
        elapsed = time.perf_counter() - t0

        raw_rows, raw_cols = len(df), len(df.columns)
        try:
            deep_mb = float(df.memory_usage(deep=True).sum()) / (1024 * 1024)
        except Exception:
            deep_mb = float("nan")

        logger.info(
            "Feature SQL [%s] done: exec_s=%.3f raw_shape=(%d,%d) approx_df_MiB=%.2f",
            name,
            elapsed,
            raw_rows,
            raw_cols,
            deep_mb,
        )

        # If the query returns a table keyed by the task entity, restrict it to the
        # current eval entity universe (val/train/test), so we don't reject
        # otherwise-reasonable feature tables that were computed for "all entities".
        if allowed_set is not None and entity_col in df.columns:
            before = len(df)
            # Build mask once and release it to reduce temporary allocations.
            mask = df[entity_col].isin(allowed_set)
            df = df.loc[mask]
            del mask
            after = len(df)
            if before != after:
                logger.info(
                    f"Filtered feature query '{name}' from {before}→{after} rows "
                    f"by restricting {entity_col} to the eval entity set ({len(allowed_set)} entities)."
                )
                try:
                    deep_mb_f = float(df.memory_usage(deep=True).sum()) / (1024 * 1024)
                except Exception:
                    deep_mb_f = float("nan")
                logger.info(
                    "Feature SQL [%s] after entity filter: rows=%d cols=%d approx_df_MiB=%.2f "
                    "(raw was %d rows — large raw + small filtered suggests join explosion before filter)",
                    name,
                    after,
                    len(df.columns),
                    deep_mb_f,
                    before,
                )
                if before > max(50_000, after * 50):
                    logger.warning(
                        "Feature SQL [%s]: filtered away %d of %d rows (%.1f%%) — query likely produced "
                        "far more rows than eval entities; tighten joins or aggregate earlier.",
                        name,
                        before - after,
                        before,
                        100.0 * (before - after) / max(before, 1),
                    )

        # Post-load safety net: catches cases where preflight COUNT was skipped/failed.
        if len(df) > row_hard_cap:
            cap_desc = (
                f"{n_entities:,}e × {NODE_ROWS_PER_ENTITY}/e = {row_hard_cap:,}" if n_entities is not None
                else f"fallback={row_hard_cap:,}"
            )
            raise ValueError(
                f"Query '{name}' returned {len(df):,} rows which exceeds the hard cap "
                f"({cap_desc}). Simplify the query: add tighter time-window filters, "
                f"aggregate earlier, or reduce join fanout. The query produced a join explosion."
            )
        if len(df) > MAX_SQL_ROWS_SOFT:
            logger.warning(
                f"Large feature query '{name}' returned {len(df)} rows "
                f"(soft max {MAX_SQL_ROWS_SOFT}); continuing."
            )
        if df.empty:
            logger.warning(f"Query '{name}' returned empty result")

        results[name] = df
        # Release short-lived temporaries from this query iteration.
        _aggressive_collect()
    return results


# ---------------------------------------------------------------------------
# Wrapped-model support: config validation, feature matrix, model fitting
# ---------------------------------------------------------------------------

VALID_MODEL_CHOICES = ("logreg", "lightgbm", "xgboost", "ridge")

# Seven fully-specified learners exposed by --seven_models (mutually exclusive with --other_tree_models)
VALID_SEVEN_MODEL_CHOICES = ("gbdt", "rf", "dart", "goss", "xgboost", "xgb_dart", "catboost")

# Multi-model evaluation mode: agent proposes features only; environment evaluates all models
MULTI_MODEL_CLS_CHOICES = ("lightgbm", "xgboost", "catboost", "logreg")
MULTI_MODEL_REG_CHOICES = ("lightgbm", "xgboost", "catboost")

_LGBM_BASE_DEFAULTS = {
    "n_estimators": 200, "learning_rate": 0.05, "max_depth": 6,
    "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
    "objective": "regression_l1",
}
_XGB_BASE_DEFAULTS = {
    "n_estimators": 200, "learning_rate": 0.05, "max_depth": 6,
    "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
}

_MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "logreg": {"C": 1.0, "class_weight": "balanced", "max_iter": 1000},
    "lightgbm": {
        "n_estimators": 200, "learning_rate": 0.05, "max_depth": 6,
        "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
        "objective": "regression_l1",
    },
    "xgboost": {
        "n_estimators": 200, "learning_rate": 0.05, "max_depth": 6,
        "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
    },
    # Ridge regression for regression tasks (features are z-score normalised before fitting)
    "ridge": {"alpha": 1.0},
    # Seven-models choices
    "gbdt":     {**_LGBM_BASE_DEFAULTS},
    "rf":       {**_LGBM_BASE_DEFAULTS},
    "dart":     {**_LGBM_BASE_DEFAULTS},
    "goss":     {**_LGBM_BASE_DEFAULTS},
    "xgb_dart": {**_XGB_BASE_DEFAULTS},
    "catboost": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "l2_leaf_reg": 3.0},
}
# Share the xgboost entry between the legacy "xgboost" and the seven-models "xgboost"
# (already present above as "xgboost")

_LGBM_ALLOWED = {"n_estimators", "learning_rate", "max_depth", "min_child_samples", "subsample", "colsample_bytree", "objective", "lambda_l1", "lambda_l2"}
_XGB_ALLOWED = {"n_estimators", "learning_rate", "max_depth", "min_child_weight", "subsample", "colsample_bytree", "objective", "reg_alpha", "reg_lambda"}

# Seven-models choices additionally accept a "categorical_features" key: a list of
# fully-qualified feature names ("{query_name}__{col}") to be handled natively
# by the learner rather than factorized to float. See _build_feature_matrix and
# _fit_and_predict_wrapped for the implementation details.
_SEVEN_MODELS_LGBM_ALLOWED = _LGBM_ALLOWED | {"categorical_features"}
_SEVEN_MODELS_XGB_ALLOWED = _XGB_ALLOWED | {"categorical_features"}
_SEVEN_MODELS_CATBOOST_ALLOWED = {"n_estimators", "learning_rate", "max_depth", "l2_leaf_reg", "categorical_features"}

_ALLOWED_KEYS: Dict[str, set] = {
    "logreg": {"C", "class_weight", "max_iter"},
    "lightgbm": _LGBM_ALLOWED,
    "xgboost":  _XGB_ALLOWED,
    "ridge": {"alpha"},
    # Seven-models choices (allow native categorical features)
    "gbdt":     _SEVEN_MODELS_LGBM_ALLOWED,
    "rf":       _SEVEN_MODELS_LGBM_ALLOWED,
    "dart":     _SEVEN_MODELS_LGBM_ALLOWED,
    "goss":     _SEVEN_MODELS_LGBM_ALLOWED,
    "xgb_dart": _SEVEN_MODELS_XGB_ALLOWED,
    "catboost": _SEVEN_MODELS_CATBOOST_ALLOWED,
}

# Seven-models choices that allow a native "categorical_features" key in model_config.
# Note: "xgboost" in seven-models mode also accepts it, but the choice name is shared
# with the legacy wrapped-model xgboost choice which does NOT. _resolve_model_config
# uses the seven_models flag to disambiguate.
_SEVEN_MODELS_CAT_CAPABLE = frozenset(VALID_SEVEN_MODEL_CHOICES)

_BOUNDS: Dict[str, tuple] = {
    "n_estimators": (50, 500),
    "learning_rate": (0.01, 0.3),
    "max_depth": (2, 10),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "C": (0.01, 100),
    "max_iter": (100, 2000),
    "min_child_weight": (1, 100),
    "min_child_samples": (1, 100),
    "num_layers": (1, 5),
    "hidden_dim": (16, 512),
    "alpha": (0.0001, 1000.0),
    "lambda_l1": (0.0, 10.0),
    "lambda_l2": (0.0, 10.0),
    "reg_alpha": (0.0, 10.0),
    "reg_lambda": (0.0, 10.0),
    "l2_leaf_reg": (0.1, 10.0),
}


_VALID_LIGHTGBM_BOOSTING_TYPES = ("gbdt", "rf", "dart", "goss")
_VALID_XGBOOST_BOOSTERS = ("gbtree", "dart")


def _resolve_model_config(
    model_choice: str,
    model_config: Optional[Dict[str, Any]],
    seven_models: bool = False,
) -> tuple:
    """Validate model_choice, clamp config to bounds, fill defaults.

    Returns (resolved_config, warnings).
    """
    warnings: List[str] = []
    if seven_models:
        if model_choice not in VALID_SEVEN_MODEL_CHOICES:
            raise ValueError(
                f"Invalid model_choice '{model_choice}'. With --seven_models must be one of {VALID_SEVEN_MODEL_CHOICES}."
            )
    elif model_choice not in VALID_MODEL_CHOICES:
        raise ValueError(
            f"Invalid model_choice '{model_choice}'. Must be one of {VALID_MODEL_CHOICES}."
        )

    defaults = _MODEL_DEFAULTS[model_choice].copy()
    allowed = set(_ALLOWED_KEYS[model_choice])
    # In --seven_models mode, all seven learners (including the shared "xgboost"
    # choice) support a native "categorical_features" list passed through the
    # model_config_json. _ALLOWED_KEYS already has this for gbdt/rf/dart/goss/
    # xgb_dart/catboost; we add it for the shared "xgboost" entry too.
    if seven_models and model_choice in _SEVEN_MODELS_CAT_CAPABLE:
        allowed = allowed | {"categorical_features"}
    user_cfg = dict(model_config) if model_config else {}

    # "categorical_features" is a list[str], not a hyperparameter — validate the
    # shape here but skip the numeric _BOUNDS clamp below (it keys off _BOUNDS,
    # which does not contain this key, so it is naturally skipped).
    if "categorical_features" in user_cfg:
        cat_val = user_cfg["categorical_features"]
        if cat_val is None:
            del user_cfg["categorical_features"]
        elif not isinstance(cat_val, list) or not all(isinstance(x, str) for x in cat_val):
            warnings.append(
                f"Ignored 'categorical_features' (must be a list of strings, got {type(cat_val).__name__})."
            )
            del user_cfg["categorical_features"]
        else:
            # De-duplicate while preserving order.
            seen: set = set()
            deduped: List[str] = []
            for name in cat_val:
                if name not in seen:
                    seen.add(name)
                    deduped.append(name)
            user_cfg["categorical_features"] = deduped

    for k in list(user_cfg.keys()):
        if k not in allowed:
            warnings.append(f"Ignored unknown config key '{k}' for {model_choice}.")
            del user_cfg[k]

    if model_choice == "logreg" and "class_weight" in user_cfg:
        cw = user_cfg["class_weight"]
        if cw not in ("balanced", None, "null"):
            warnings.append(f"class_weight must be 'balanced' or null; got '{cw}', using 'balanced'.")
            user_cfg["class_weight"] = "balanced"
        if cw == "null":
            user_cfg["class_weight"] = None

    if model_choice == "lightgbm" and "boosting_type" in user_cfg:
        bt = user_cfg["boosting_type"]
        if bt not in _VALID_LIGHTGBM_BOOSTING_TYPES:
            warnings.append(
                f"Invalid boosting_type '{bt}'; must be one of {_VALID_LIGHTGBM_BOOSTING_TYPES}. Using 'gbdt'."
            )
            user_cfg["boosting_type"] = "gbdt"

    if model_choice == "xgboost" and "booster" in user_cfg:
        b = user_cfg["booster"]
        if b not in _VALID_XGBOOST_BOOSTERS:
            warnings.append(
                f"Invalid booster '{b}'; must be one of {_VALID_XGBOOST_BOOSTERS}. Using 'gbtree'."
            )
            user_cfg["booster"] = "gbtree"

    resolved = defaults.copy()
    resolved.update(user_cfg)
    for key, (lo, hi) in _BOUNDS.items():
        if key in resolved and resolved[key] is not None and key != "class_weight":
            try:
                v = float(resolved[key])
            except (ValueError, TypeError):
                warnings.append(f"Non-numeric value for '{key}', using default.")
                resolved[key] = defaults.get(key, lo)
                continue
            if v < lo or v > hi:
                clamped = max(lo, min(hi, v))
                warnings.append(f"Clamped '{key}' from {v} to {clamped} (bounds [{lo}, {hi}]).")
                v = clamped
            if key in ("n_estimators", "max_depth", "max_iter", "min_child_weight", "min_child_samples",
                       "num_layers", "hidden_dim"):
                resolved[key] = int(v)
            else:
                resolved[key] = v

    return resolved, warnings


def _build_feature_matrix(
    features: Dict[str, pd.DataFrame],
    target_df: pd.DataFrame,
    entity_col: str,
    asof_col: Optional[str] = None,
    extra_entity_cols: Optional[List[str]] = None,
    return_raw: bool = False,
    categorical_features: Optional[List[str]] = None,
    train_categories: Optional[Dict[str, "pd.CategoricalDtype"]] = None,
    return_categories: bool = False,
) -> tuple:
    """Build a feature matrix by joining feature query results with target_df.

    Returns (X, diagnostics) normally, or (X, X_raw, diagnostics) when
    return_raw=True. X_raw is the feature DataFrame *before* NaN-fill, which
    is useful for computing null_fraction and other diagnostics faithfully.

    Categorical support (seven-models mode):
        categorical_features: list of fully-qualified column names
            ("{query_name}__{col}") to keep as `pd.Categorical` rather than
            factorize-to-float. Only columns that exist after the merges are
            honored; missing names are recorded as a warning.
        train_categories: optional pre-computed mapping column name ->
            CategoricalDtype. When provided (val/test call), the same
            CategoricalDtype is applied so columns share train categories;
            unseen values become NaN. When omitted (train call), dtypes are
            learned from the data.
        return_categories: when True, append the realized
            `{col: CategoricalDtype}` dict as the final tuple element.

    Categorical columns are left as pandas Categorical dtype in the returned X
    (no numeric coercion, no median fill) so downstream learners can consume
    them natively (LightGBM / XGBoost enable_categorical / CatBoost cat_features).
    """
    join_cols = [entity_col]
    if asof_col and asof_col in target_df.columns:
        join_cols.append(asof_col)
    for _ec in (extra_entity_cols or []):
        if _ec not in join_cols and _ec in target_df.columns:
            join_cols.append(_ec)

    base = target_df[join_cols].copy().drop_duplicates().reset_index(drop=True)
    diagnostics: Dict[str, Any] = {
        "base_rows": len(base),
        "merge_row_counts": {},
        "missingness": {},
        "warnings": [],
    }

    for qname, fdf in features.items():
        if fdf.empty:
            diagnostics["warnings"].append(f"Feature block '{qname}' is empty.")
            continue

        if entity_col not in fdf.columns:
            diagnostics["warnings"].append(
                f"Feature block '{qname}' missing entity column '{entity_col}', skipping."
            )
            continue
        feature_join_cols = [c for c in join_cols if c in fdf.columns]

        fdf_dedup = fdf.drop_duplicates(subset=feature_join_cols, keep="first")
        feat_cols = [c for c in fdf_dedup.columns if c not in feature_join_cols]
        if not feat_cols:
            continue
        renamed = fdf_dedup[feature_join_cols + feat_cols].rename(
            columns={c: f"{qname}__{c}" for c in feat_cols}
        )

        base = base.merge(renamed, on=feature_join_cols, how="left")
        diagnostics["merge_row_counts"][qname] = len(base)

        block_cols = [f"{qname}__{c}" for c in feat_cols if f"{qname}__{c}" in base.columns]
        if block_cols:
            miss_rate = float(base[block_cols].isna().any(axis=1).mean())
            diagnostics["missingness"][qname] = miss_rate
            if miss_rate > 0.5:
                diagnostics["warnings"].append(
                    f"Feature block '{qname}' has {miss_rate:.1%} missingness."
                )

    feat_columns = [c for c in base.columns if c not in join_cols]
    if not feat_columns:
        raise ValueError("No feature columns available after merging all feature blocks.")

    # Resolve the set of categorical columns that actually exist in the merged matrix.
    # Fall back to resolving unqualified names (e.g. "country") by looking for a
    # unique match among feat_columns of the form "{any_query}__country".
    categorical_cols: List[str] = []
    if categorical_features:
        feat_set = set(feat_columns)
        for name in categorical_features:
            if name in feat_set:
                categorical_cols.append(name)
                continue
            # Try unqualified match: any column ending with "__{name}"
            suffix = f"__{name}"
            candidates = [c for c in feat_columns if c.endswith(suffix)]
            if len(candidates) == 1:
                categorical_cols.append(candidates[0])
            elif len(candidates) > 1:
                diagnostics["warnings"].append(
                    f"Ambiguous categorical feature '{name}' matches {candidates}; "
                    f"use a fully-qualified name like '{{query_name}}__{{col}}'."
                )
            else:
                diagnostics["warnings"].append(
                    f"Categorical feature '{name}' not found in feature matrix; ignoring."
                )
    cat_cols_set = set(categorical_cols)

    suspicious_patterns = ["target", "label", "churn", "ground_truth", "y_true"]
    for c in feat_columns:
        c_lower = c.lower()
        for pat in suspicious_patterns:
            if pat in c_lower:
                diagnostics["warnings"].append(
                    f"Suspicious feature column '{c}' (name contains '{pat}') — possible leakage."
                )
                break

    realized_categories: Dict[str, "pd.CategoricalDtype"] = {}
    for col in feat_columns:
        if col in cat_cols_set:
            # Preserve categorical information natively. Cast using the train-side
            # CategoricalDtype when provided so val/test share the same encoding.
            if train_categories is not None and col in train_categories:
                base[col] = base[col].astype(train_categories[col])
            else:
                # Learn categories from the data. Drop datetime noise first.
                s = base[col]
                if pd.api.types.is_datetime64_any_dtype(s):
                    s = pd.to_datetime(s, errors="coerce").astype("int64")
                base[col] = s.astype("category")
            realized_categories[col] = base[col].dtype  # CategoricalDtype
            continue

        if pd.api.types.is_datetime64_any_dtype(base[col]):
            base[col] = pd.to_datetime(base[col], errors="coerce").astype("int64")
        elif not pd.api.types.is_numeric_dtype(base[col]):
            base[col] = pd.factorize(base[col], sort=True)[0]
        base[col] = _coerce_ml_feature_column(base[col])

    for col in feat_columns:
        if col in base.columns and base[col].nunique(dropna=True) <= 1:
            diagnostics["warnings"].append(f"Constant column '{col}' (nunique<=1).")

    X = base[feat_columns].copy()
    X_raw = X.copy()  # preserve pre-fillna values for diagnostics

    # Median-fill numeric columns only; leave categoricals with NaN so native
    # learners (LGBM / XGBoost with enable_categorical) can handle missing as
    # a first-class category. CatBoost cannot accept NaN in cat features; the
    # caller (_fit_and_predict_wrapped) substitutes a "__NA__" category right
    # before fit/predict when model_choice == "catboost".
    if cat_cols_set:
        numeric_cols = [c for c in feat_columns if c not in cat_cols_set]
        if numeric_cols:
            X[numeric_cols] = (
                X[numeric_cols].fillna(X[numeric_cols].median(numeric_only=True)).fillna(0.0)
            )
    else:
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    if return_raw and return_categories:
        return X, X_raw, diagnostics, realized_categories
    if return_raw:
        return X, X_raw, diagnostics
    if return_categories:
        return X, diagnostics, realized_categories
    return X, diagnostics


def _instantiate_model(
    model_choice: str,
    resolved_config: Dict[str, Any],
    task_type: str,
    categorical_feature_indices: Optional[List[int]] = None,
):
    """Create a sklearn-compatible model instance.

    categorical_feature_indices: positional indices of categorical columns in the
        fitted feature matrix (used only by XGBoost and CatBoost at construction
        time). LightGBM receives categorical information via .fit(..., categorical_feature=)
        at the caller site, so this argument is ignored for the LGBM family.
    """
    seed = 42
    is_classification = task_type == "entity_classification"

    # "categorical_features" is a meta-key that rides inside resolved_config from
    # _resolve_model_config; strip it so it never reaches a sklearn constructor.
    resolved_config = {k: v for k, v in resolved_config.items() if k != "categorical_features"}
    has_cat = bool(categorical_feature_indices)

    if model_choice == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=seed, solver="lbfgs", **resolved_config)

    if model_choice == "lightgbm":
        import lightgbm as lgb
        cfg = {**resolved_config, "random_state": seed, "verbosity": -1}
        # RF boosting mode requires subsample_freq > 0
        if cfg.get("boosting_type") == "rf":
            cfg.setdefault("subsample_freq", 1)
        return lgb.LGBMClassifier(**cfg) if is_classification else lgb.LGBMRegressor(**cfg)

    if model_choice in ("xgboost", "xgb_dart"):
        import xgboost as xgb
        cfg = {**resolved_config, "random_state": seed, "verbosity": 0, "tree_method": "hist"}
        if model_choice == "xgb_dart":
            cfg["booster"] = "dart"
        if has_cat:
            # XGBoost >=1.5 supports native categorical splits when the input
            # DataFrame carries pandas Categorical columns and enable_categorical=True.
            cfg["enable_categorical"] = True
        if is_classification:
            return xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", **cfg)
        return xgb.XGBRegressor(**cfg)


    if model_choice == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=float(resolved_config.get("alpha", 1.0)))

    # Seven-models: lgbm variants with baked-in boosting_type
    if model_choice in ("gbdt", "rf", "dart", "goss"):
        import lightgbm as lgb
        cfg = {**resolved_config, "random_state": seed, "verbosity": -1,
               "boosting_type": model_choice}
        if model_choice == "rf":
            cfg.setdefault("subsample_freq", 1)
        return lgb.LGBMClassifier(**cfg) if is_classification else lgb.LGBMRegressor(**cfg)

    # Seven-models: catboost
    if model_choice == "catboost":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as e:
            raise ImportError("catboost package not installed. Install with: pip install catboost") from e
        cfg = dict(resolved_config)
        iterations = int(cfg.pop("n_estimators", 200))
        depth = int(cfg.pop("max_depth", 6))
        l2_leaf_reg = float(cfg.pop("l2_leaf_reg", 3.0))
        learning_rate = float(cfg.pop("learning_rate", 0.05))
        cls_cb = CatBoostClassifier if is_classification else CatBoostRegressor
        ctor_kwargs: Dict[str, Any] = dict(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            random_seed=seed,
            verbose=0,
        )
        if has_cat:
            ctor_kwargs["cat_features"] = list(categorical_feature_indices)
        return cls_cb(**ctor_kwargs)

    raise ValueError(f"Unknown model_choice: {model_choice}")


# Sentinel category used when CatBoost encounters missing categorical values —
# CatBoost does not accept NaN in categorical columns, unlike LightGBM and XGBoost.
_CATBOOST_NA_SENTINEL = "__NA__"


def _prepare_catboost_frame(X: pd.DataFrame, categorical_col_names: List[str]) -> pd.DataFrame:
    """Return a copy of X with categorical columns cast to string and NaN replaced
    with a sentinel value, so CatBoost's cat_features path accepts them."""
    if not categorical_col_names:
        return X
    out = X.copy()
    for c in categorical_col_names:
        if c in out.columns:
            out[c] = out[c].astype(object).where(out[c].notna(), _CATBOOST_NA_SENTINEL).astype(str)
    return out


# Objectives that target the conditional median (L1-equivalent) — no bias correction needed.
_LIGHTGBM_L1_OBJECTIVES: frozenset = frozenset({"regression_l1", "mean_absolute_error", "mae"})
_XGBOOST_L1_OBJECTIVES: frozenset = frozenset({"reg:absoluteerror"})


def _is_l2_objective(model_choice: str, resolved_config: Dict[str, Any]) -> bool:
    """Return True if the model optimises an L2/MSE loss.

    When fitting on log1p(y) with an L2 objective, expm1(pred) is a biased
    estimate of E[Y] because E[exp(Z)] != exp(E[Z]).  The correction
    exp(sigma^2 / 2) removes this bias (exact for log-normal residuals,
    a good approximation otherwise).

    L1 objectives predict the conditional median; by monotonicity of log,
    expm1(median_log_pred) == median_original, which is the MAE-optimal
    predictor — no correction needed.
    """
    if model_choice == "ridge":
        return True
    if model_choice in ("lightgbm", "gbdt", "rf", "dart", "goss"):
        return str(resolved_config.get("objective", "")).lower() not in _LIGHTGBM_L1_OBJECTIVES
    if model_choice in ("xgboost", "xgb_dart"):
        return str(resolved_config.get("objective", "")).lower() not in _XGBOOST_L1_OBJECTIVES
    if model_choice == "catboost":
        # CatBoost uses RMSE by default (L2); MAE objective is not easily specified here
        return True
    return False  # logreg / classification: not applicable


def _extract_importances(model, feature_names: List[str]) -> List[Dict[str, Any]]:
    """Extract feature importances or coefficients from a fitted model."""
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef).mean(axis=0) if coef.ndim == 2 else np.abs(coef)
    elif hasattr(model, "coefs_"):
        # MLP: L2 norm of first-layer weights per input feature as proxy importance
        importances = np.linalg.norm(model.coefs_[0], axis=1)

    if importances is None:
        return []

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances.tolist(),
    }).sort_values("importance", ascending=False)

    return [
        {"rank": int(i + 1), "feature": row["feature"], "importance": float(row["importance"])}
        for i, row in enumerate(imp_df.head(30).to_dict("records"))
    ]


def _compute_full_importances_df(
    model,
    feature_names: List[str],
    model_choice: str,
    trial_id: str,
    split: str,
    X_val: Optional[pd.DataFrame] = None,
    shap_max_rows: int = 2000,
) -> pd.DataFrame:
    """Return a DataFrame with ALL feature importances for workspace-plus storage.

    For logreg:
      - One row per feature, importance_type="coefficient" (raw signed value).

    For lightgbm / xgboost:
      - One row per feature, importance_type="gain" (total loss reduction).
      - If X_val is provided: one additional row per feature,
        importance_type="shap_mean_abs" (mean |SHAP| on val data, sign from
        mean SHAP).  Native tree-SHAP is used — no surrogate model.

    rank and normalized_importance are computed independently within each
    importance_type group so queries like
      WHERE importance_type='shap_mean_abs' ORDER BY rank
    work correctly.

    Returns an empty DataFrame with the correct schema when the model does not
    expose importances.
    """
    _empty = pd.DataFrame(columns=[
        "trial_id", "split", "model_type", "feature_name", "extractor_name",
        "importance_type", "importance_value", "abs_importance",
        "sign", "rank", "normalized_importance",
    ])

    all_groups: List[List[Dict[str, Any]]] = []

    # ------------------------------------------------------------------
    # Logistic Regression / Ridge: signed coefficients
    # ------------------------------------------------------------------
    if model_choice in ("logreg", "ridge") and hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim == 2:
            coef = coef[0]
        rows: List[Dict[str, Any]] = []
        for fname, c in zip(feature_names, coef):
            extractor = fname.split("__")[0] if "__" in fname else fname
            rows.append({
                "trial_id": trial_id, "split": split, "model_type": model_choice,
                "feature_name": fname, "extractor_name": extractor,
                "importance_type": "coefficient",
                "importance_value": float(c),
                "abs_importance": float(abs(c)),
                "sign": int(np.sign(c)) if c != 0 else 0,
                "n_shap_rows": None,
            })
        all_groups.append(rows)

    # ------------------------------------------------------------------
    # LightGBM / XGBoost: gain-based importance + native SHAP on X_val
    # ------------------------------------------------------------------
    elif model_choice in ("lightgbm", "gbdt", "rf", "dart", "goss"):
        # --- Gain ---
        try:
            gains = model.booster_.feature_importance(importance_type="gain")
        except Exception:
            gains = np.zeros(len(feature_names))

        gain_rows: List[Dict[str, Any]] = []
        for fname, g in zip(feature_names, gains):
            extractor = fname.split("__")[0] if "__" in fname else fname
            gain_rows.append({
                "trial_id": trial_id, "split": split, "model_type": "lightgbm",
                "feature_name": fname, "extractor_name": extractor,
                "importance_type": "gain",
                "importance_value": float(g),
                "abs_importance": float(g),  # gain is always >= 0
                "sign": None,
                "n_shap_rows": None,
            })
        all_groups.append(gain_rows)

        # --- Native SHAP on X_val ---
        if X_val is not None and len(X_val) > 0:
            try:
                # Preserve DataFrame when categorical columns are present so LGBM
                # can use the same categorical encoding it saw at fit time.
                has_cat_cols = any(
                    isinstance(X_val[c].dtype, pd.CategoricalDtype) for c in X_val.columns
                )
                if has_cat_cols:
                    if len(X_val) > shap_max_rows:
                        idx = np.linspace(0, len(X_val) - 1, shap_max_rows, dtype=int)
                        X_shap = X_val.iloc[idx].reset_index(drop=True)
                    else:
                        X_shap = X_val
                    n_shap_used = len(X_shap)
                else:
                    X_shap = X_val.values
                    if len(X_shap) > shap_max_rows:
                        idx = np.linspace(0, len(X_shap) - 1, shap_max_rows, dtype=int)
                        X_shap = X_shap[idx]
                    n_shap_used = len(X_shap)
                sv = model.predict(X_shap, pred_contrib=True)
                # Binary/regression: shape (n, n_features + 1); last col = bias
                # Multiclass: shape (n, n_classes * (n_features + 1)) — skip
                if sv.ndim == 2 and sv.shape[1] == len(feature_names) + 1:
                    sv = sv[:, :-1]
                    mean_abs = np.abs(sv).mean(axis=0)
                    mean_dir = sv.mean(axis=0)
                    shap_rows: List[Dict[str, Any]] = []
                    for fname, ma, md in zip(feature_names, mean_abs, mean_dir):
                        extractor = fname.split("__")[0] if "__" in fname else fname
                        shap_rows.append({
                            "trial_id": trial_id, "split": split, "model_type": "lightgbm",
                            "feature_name": fname, "extractor_name": extractor,
                            "importance_type": "shap_mean_abs",
                            "importance_value": float(ma),
                            "abs_importance": float(ma),
                            "sign": int(np.sign(md)) if md != 0 else 0,
                            "n_shap_rows": n_shap_used,
                        })
                    all_groups.append(shap_rows)
            except Exception as _shap_err:
                logger.debug(f"LightGBM native SHAP failed: {_shap_err}")

    elif model_choice in ("xgboost", "xgb_dart"):
        # --- Gain ---
        try:
            import xgboost as xgb
            booster = model.get_booster()
            fscore = booster.get_score(importance_type="total_gain")
            # When fit on numpy arrays, booster uses 'f0', 'f1', ... names
            gains = np.zeros(len(feature_names))
            for k, v in fscore.items():
                if k.startswith("f") and k[1:].isdigit():
                    idx = int(k[1:])
                    if idx < len(gains):
                        gains[idx] = v
        except Exception:
            gains = np.zeros(len(feature_names))

        gain_rows = []
        for fname, g in zip(feature_names, gains):
            extractor = fname.split("__")[0] if "__" in fname else fname
            gain_rows.append({
                "trial_id": trial_id, "split": split, "model_type": "xgboost",
                "feature_name": fname, "extractor_name": extractor,
                "importance_type": "gain",
                "importance_value": float(g),
                "abs_importance": float(g),
                "sign": None,
                "n_shap_rows": None,
            })
        all_groups.append(gain_rows)

        # --- Native SHAP on X_val ---
        if X_val is not None and len(X_val) > 0:
            try:
                import xgboost as xgb
                has_cat_cols = any(
                    isinstance(X_val[c].dtype, pd.CategoricalDtype) for c in X_val.columns
                )
                if has_cat_cols:
                    if len(X_val) > shap_max_rows:
                        idx = np.linspace(0, len(X_val) - 1, shap_max_rows, dtype=int)
                        X_shap = X_val.iloc[idx].reset_index(drop=True)
                    else:
                        X_shap = X_val
                    n_shap_used = len(X_shap)
                    dmat = xgb.DMatrix(X_shap, enable_categorical=True)
                else:
                    X_shap = X_val.values
                    if len(X_shap) > shap_max_rows:
                        idx = np.linspace(0, len(X_shap) - 1, shap_max_rows, dtype=int)
                        X_shap = X_shap[idx]
                    n_shap_used = len(X_shap)
                    dmat = xgb.DMatrix(X_shap)
                sv = model.get_booster().predict(dmat, pred_contribs=True)
                # shape (n, n_features + 1); last col = bias
                if sv.ndim == 2 and sv.shape[1] == len(feature_names) + 1:
                    sv = sv[:, :-1]
                    mean_abs = np.abs(sv).mean(axis=0)
                    mean_dir = sv.mean(axis=0)
                    shap_rows = []
                    for fname, ma, md in zip(feature_names, mean_abs, mean_dir):
                        extractor = fname.split("__")[0] if "__" in fname else fname
                        shap_rows.append({
                            "trial_id": trial_id, "split": split, "model_type": "xgboost",
                            "feature_name": fname, "extractor_name": extractor,
                            "importance_type": "shap_mean_abs",
                            "importance_value": float(ma),
                            "abs_importance": float(ma),
                            "sign": int(np.sign(md)) if md != 0 else 0,
                            "n_shap_rows": n_shap_used,
                        })
                    all_groups.append(shap_rows)
            except Exception as _shap_err:
                logger.debug(f"XGBoost native SHAP failed: {_shap_err}")

    # ------------------------------------------------------------------
    # CatBoost: feature importance (PredictionValuesChange)
    # ------------------------------------------------------------------
    elif model_choice == "catboost":
        try:
            fi = model.get_feature_importance()
            rows = []
            for fname, v in zip(feature_names, fi):
                extractor = fname.split("__")[0] if "__" in fname else fname
                rows.append({
                    "trial_id": trial_id, "split": split, "model_type": "catboost",
                    "feature_name": fname, "extractor_name": extractor,
                    "importance_type": "gain",
                    "importance_value": float(v),
                    "abs_importance": float(v),
                    "sign": None,
                    "n_shap_rows": None,
                })
            all_groups.append(rows)
        except Exception as _cb_err:
            logger.debug(f"CatBoost importance failed: {_cb_err}")

    if not all_groups:
        return _empty

    # Assign rank + normalized_importance within each importance_type group
    result_dfs: List[pd.DataFrame] = []
    for group_rows in all_groups:
        if not group_rows:
            continue
        gdf = pd.DataFrame(group_rows)
        gdf = gdf.sort_values("abs_importance", ascending=False).reset_index(drop=True)
        gdf["rank"] = range(1, len(gdf) + 1)
        max_abs = gdf["abs_importance"].max()
        gdf["normalized_importance"] = (
            gdf["abs_importance"] / max_abs if max_abs > 0 else 0.0
        )
        result_dfs.append(gdf)

    return pd.concat(result_dfs, ignore_index=True)


def _compute_feature_diagnostics_df(
    X_raw: pd.DataFrame,
    trial_id: str,
    split: str,
) -> pd.DataFrame:
    """Compute per-feature summary statistics from the raw (pre-fillna) feature matrix.

    Args:
        X_raw: Feature DataFrame *before* NaN fill.  Numeric columns only
               (non-numeric features are already factorized by _build_feature_matrix,
               so all columns here are numeric or integer-encoded).
        trial_id: Workspace trial ID string.
        split: Dataset split label (e.g. "val", "train").

    Returns:
        DataFrame with one row per feature and diagnostic columns.
    """
    n_rows = len(X_raw)
    rows: List[Dict[str, Any]] = []

    for col in X_raw.columns:
        s = X_raw[col]
        null_frac = float(s.isna().mean())
        non_null = s.dropna()
        n_nonnull = len(non_null)
        unique_count = int(s.nunique(dropna=True))
        is_constant = unique_count <= 1

        mean = std = min_v = max_v = p01 = p50 = p99 = None
        zero_frac = neg_frac = None

        if n_nonnull > 0 and pd.api.types.is_numeric_dtype(s):
            mean = float(non_null.mean())
            std = float(non_null.std()) if n_nonnull > 1 else 0.0
            min_v = float(non_null.min())
            max_v = float(non_null.max())
            p01 = float(non_null.quantile(0.01))
            p50 = float(non_null.quantile(0.50))
            p99 = float(non_null.quantile(0.99))
            zero_frac = float((non_null == 0).mean())
            neg_frac = float((non_null < 0).mean())

        # Simple rule-based suspicious flags
        flags: List[str] = []
        if is_constant:
            flags.append("constant feature")
        if null_frac > 0.8:
            flags.append("mostly null")
        elif zero_frac is not None and zero_frac > 0.95:
            flags.append("mostly zero")
        if std is not None and std < 1e-8 and not is_constant:
            flags.append("very low variance")
        if max_v is not None and min_v is not None:
            if max_v > 1e9 or min_v < -1e9:
                flags.append("extreme scale")

        rows.append({
            "trial_id": trial_id,
            "split": split,
            "feature_name": col,
            "n_rows": n_rows,
            "null_fraction": null_frac,
            "non_null_fraction": 1.0 - null_frac,
            "unique_count": unique_count,
            "is_constant": is_constant,
            "mean": mean,
            "std": std,
            "min": min_v,
            "max": max_v,
            "p01": p01,
            "p50": p50,
            "p99": p99,
            "zero_fraction": zero_frac,
            "negative_fraction": neg_frac,
            "suspicious_flag": len(flags) > 0,
            "suspicious_reason": "; ".join(flags) if flags else None,
        })

    return pd.DataFrame(rows)


def _fit_and_predict_wrapped(
    features: Dict[str, pd.DataFrame],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    entity_col: str,
    target_col: str,
    model_choice: str,
    resolved_config: Dict[str, Any],
    task_type: str,
    asof_col: Optional[str] = None,
    extra_entity_cols: Optional[List[str]] = None,
    train_features: Optional[Dict[str, pd.DataFrame]] = None,
    plus_split: str = "val",
    is_final: bool = False,
    model_save_path: Optional[str] = None,
    log_transform_target: bool = False,
    store_eval_preds: bool = False,
    _prebuilt_matrices: Optional[Dict[str, Any]] = None,
) -> tuple:
    """Build feature matrices, train model, predict on val set.

    Args:
        features: Feature query results for val split (used for val prediction).
        train_features: Feature query results for train split (used for model fitting).
            If None, uses `features` for both (legacy behavior).
        plus_split: Split label used for plus data rows.
        _prebuilt_matrices: Optional dict with pre-built matrices (X_train, y_train,
            X_val, train_diag, val_diag, train_categories). When provided, skips
            _build_feature_matrix calls entirely. Used by multi-model mode to avoid
            rebuilding the same matrices for each model.

    Returns (pred_df, diagnostics).
    """
    train_feats = train_features if train_features is not None else features
    val_feats = features

    # --- Classification / Regression path ---

    # Extract the (already-validated) categorical feature list carried by
    # _resolve_model_config inside resolved_config. It is a list of fully-qualified
    # column names ("{query_name}__{col}"). Only seven-models choices support
    # this key; for other choices it will be absent.
    categorical_features_list: List[str] = list(resolved_config.get("categorical_features") or [])

    if _prebuilt_matrices is not None:
        X_train = _prebuilt_matrices["X_train"]
        y_train = _prebuilt_matrices["y_train"]
        train_diag = _prebuilt_matrices["train_diag"]
        train_categories = _prebuilt_matrices["train_categories"]
        X_train_raw = None
    else:
        X_train, train_diag, train_categories = _build_feature_matrix(
            train_feats, train_df, entity_col, asof_col,
            extra_entity_cols=extra_entity_cols,
            categorical_features=categorical_features_list or None,
            return_categories=True,
        )
        X_train_raw = None
        join_cols = [entity_col]
        if asof_col and asof_col in train_df.columns:
            join_cols.append(asof_col)
        for _ec in (extra_entity_cols or []):
            if _ec not in join_cols and _ec in train_df.columns:
                join_cols.append(_ec)
        train_keys = train_df[join_cols].copy().drop_duplicates().reset_index(drop=True)
        y_aligned = train_keys.merge(train_df[join_cols + [target_col]], on=join_cols, how="left")[target_col]
        y_train = pd.to_numeric(y_aligned, errors="coerce").fillna(0).values

    # Cap training set to avoid multi-hour C-thread hangs when step_timeout fires.
    # On very large datasets (e.g. SALT item-plant: 1.6M rows) CatBoost/LGBM training
    # blocks in C++ beyond step_timeout; the Python thread cannot be killed, leaving
    # the process stuck forever. 200K rows is sufficient for good generalisation and
    # keeps training well under 5 minutes on a 16-CPU node.
    _MAX_TRAIN_ROWS = 200_000
    if len(X_train) > _MAX_TRAIN_ROWS:
        _orig_train_size = len(X_train)
        _cap_seed = 42
        rng = np.random.default_rng(_cap_seed)
        if task_type == "entity_classification":
            # Stratified sample: preserve class distribution; fall back to random
            # if any class has < 2 samples (train_test_split requirement).
            from sklearn.model_selection import train_test_split
            try:
                _, _keep_idx = train_test_split(
                    np.arange(_orig_train_size),
                    test_size=_MAX_TRAIN_ROWS,
                    stratify=y_train,
                    random_state=_cap_seed,
                )
            except ValueError:
                _keep_idx = rng.choice(_orig_train_size, size=_MAX_TRAIN_ROWS, replace=False)
        else:
            _keep_idx = rng.choice(_orig_train_size, size=_MAX_TRAIN_ROWS, replace=False)
        _keep_idx = np.sort(_keep_idx)
        X_train = X_train.iloc[_keep_idx].reset_index(drop=True)
        y_train = y_train[_keep_idx]
        logger.info(f"Training set capped: {_MAX_TRAIN_ROWS}/{_orig_train_size} rows sampled")

    # Categorical columns actually realized in X_train (as pandas Categorical dtype).
    cat_col_names: List[str] = [
        c for c in X_train.columns
        if c in train_categories and isinstance(X_train[c].dtype, pd.CategoricalDtype)
    ]
    cat_indices: List[int] = [X_train.columns.get_loc(c) for c in cat_col_names]
    has_cat = len(cat_indices) > 0
    # Only the seven-models tree learners support native categoricals; any
    # categorical_features list on other choices has already been filtered out
    # by _resolve_model_config, so has_cat can only be True for those.

    # Z-score normalization for ridge: fit scaler on train split, apply to both splits.
    # r = (v - μ_col) / σ_col, using training-split column mean and std.
    # Columns with σ=0 (constant) are left as-is (StandardScaler default).
    # Tree models and logreg are used without normalization (trees are scale-invariant;
    # logreg's lbfgs is robust enough for typical tabular feature scales).
    # Ridge regression is a linear model and is sensitive to feature scales, so we normalise.
    scaler = None
    if has_cat:
        # Keep DataFrame so Categorical dtypes survive to the fit/predict calls.
        X_train_fit: Any = X_train
    else:
        X_train_fit = X_train.values
    if model_choice == "ridge":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_fit = scaler.fit_transform(X_train.values.astype(float))

    # Log-transform target for regression (e.g. skewed counts / monetary values).
    # Applied as log1p(y) before fitting; predictions are inverse-transformed with expm1.
    # Metrics (MAE, R2) are always reported in the original scale after inverse transform.
    # Targets are clipped to [0, ∞) before log1p to handle any negative noise.
    do_log = log_transform_target and task_type == "entity_regression"
    if do_log:
        y_train = np.log1p(np.clip(y_train, 0, None))

    # For multi-class tasks: strip any non-multiclass objective so LightGBM auto-selects softmax.
    _n_unique_y = len(np.unique(y_train[np.isfinite(y_train.astype(float, copy=False))]))
    _is_multiclass_clf = task_type == "entity_classification" and _n_unique_y > 2
    _LGBM_MULTICLASS_OBJECTIVES = frozenset({"multiclass", "softmax", "multiclassova", "multiclass_ova", "ovr"})
    if _is_multiclass_clf and model_choice in ("lightgbm", "gbdt", "rf", "dart", "goss"):
        obj = resolved_config.get("objective")
        if obj is not None and str(obj).lower() not in _LGBM_MULTICLASS_OBJECTIVES:
            resolved_config = {k: v for k, v in resolved_config.items() if k != "objective"}
            logger.warning("Stripped objective='%s': y_train has %d classes (multiclass).", obj, _n_unique_y)

    model = _instantiate_model(
        model_choice,
        resolved_config,
        task_type,
        categorical_feature_indices=cat_indices if has_cat else None,
    )

    # CatBoost cannot handle NaN in categorical columns — substitute a sentinel.
    if has_cat and model_choice == "catboost":
        X_train_fit = _prepare_catboost_frame(X_train, cat_col_names)

    if has_cat and model_choice in ("gbdt", "rf", "dart", "goss"):
        # LightGBM accepts a DataFrame and takes categorical_feature at fit time.
        model.fit(X_train_fit, y_train, categorical_feature=cat_indices)
    else:
        model.fit(X_train_fit, y_train)

    # Bias correction for log-transformed targets with L2 loss.
    # Fitting on log1p(y) with an L2 objective produces predictions of
    # E[log1p(Y)].  Applying expm1 directly gives exp(E[log1p(Y)]) which
    # underestimates E[Y] by roughly exp(-sigma^2/2).  Adding sigma^2/2
    # to the log-space prediction before expm1 removes this bias.
    # L1 objectives predict the conditional median, so no correction is needed.
    log_bias_correction = 0.0
    if do_log and _is_l2_objective(model_choice, resolved_config):
        log_train_preds = model.predict(X_train_fit)
        sigma2 = float(np.mean((y_train - log_train_preds) ** 2))
        log_bias_correction = sigma2 / 2.0
        logger.info(
            f"Log-transform bias correction (L2 objective, σ²/2 = {log_bias_correction:.6f})"
        )

    # Save fitted model (and scaler if normalised) for final evaluations.
    if model_save_path is not None:
        try:
            import joblib
            import os as _os
            _os.makedirs(_os.path.dirname(model_save_path), exist_ok=True)
            joblib.dump({
                "model": model,
                "scaler": scaler,
                "feature_names": list(X_train.columns),
                "log_transform_target": do_log,
                "log_bias_correction": log_bias_correction,
                "categorical_feature_names": cat_col_names,
            }, model_save_path)
            logger.info(f"Saved fitted model to {model_save_path}")
        except Exception as _save_err:
            logger.warning(f"Model save failed: {_save_err}")

    feature_names = list(X_train.columns)
    if _prebuilt_matrices is not None:
        X_val = _prebuilt_matrices["X_val"]
        val_diag = _prebuilt_matrices["val_diag"]
    else:
        X_val, val_diag = _build_feature_matrix(
            val_feats, val_df, entity_col, asof_col,
            extra_entity_cols=extra_entity_cols,
            categorical_features=categorical_features_list or None,
            train_categories=train_categories,
        )
        # Align val columns to match train columns (same order, fill missing with 0)
        for col in feature_names:
            if col not in X_val.columns:
                X_val[col] = 0.0
        X_val = X_val[feature_names]

    if has_cat:
        X_val_fit: Any = X_val
    else:
        X_val_fit = X_val.values
    if scaler is not None:
        X_val_fit = scaler.transform(X_val.values.astype(float))
    elif has_cat and model_choice == "catboost":
        X_val_fit = _prepare_catboost_frame(X_val, cat_col_names)

    is_clf = task_type == "entity_classification"
    if is_clf and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_val_fit)
        if proba.ndim == 2 and proba.shape[1] == 2:
            # Binary: positive class probability
            preds = proba[:, 1]
        elif proba.ndim == 2 and proba.shape[1] > 2:
            # Multi-class: map model.classes_ back to correct indices and pad to
            # num_classes_total so relbench's mrr() never gets an index out of range.
            _model_classes = getattr(model, "classes_", None)
            _nc_total = int(np.max(y_train)) + 1  # classes seen in training
            if _model_classes is not None:
                _nc_total = max(_nc_total, int(np.max(_model_classes)) + 1)
            # Also cover any class labels present in the val set
            if target_col in val_df.columns:
                try:
                    _val_max = int(pd.to_numeric(val_df[target_col], errors="coerce").dropna().max())
                    _nc_total = max(_nc_total, _val_max + 1)
                except Exception:
                    pass
            preds = np.empty(proba.shape[0], dtype=object)
            for _mc_i in range(proba.shape[0]):
                _vec = [0.0] * _nc_total
                if _model_classes is not None:
                    for _ci, _cls in enumerate(_model_classes):
                        _vec[int(_cls)] = float(proba[_mc_i, _ci])
                else:
                    for _ci in range(min(proba.shape[1], _nc_total)):
                        _vec[_ci] = float(proba[_mc_i, _ci])
                preds[_mc_i] = _vec
        else:
            preds = proba
    else:
        preds = model.predict(X_val_fit)
        if do_log:
            preds = np.expm1(preds + log_bias_correction)

    val_join_cols = [entity_col]
    if asof_col and asof_col in val_df.columns:
        val_join_cols.append(asof_col)
    for _ec in (extra_entity_cols or []):
        if _ec not in val_join_cols and _ec in val_df.columns:
            val_join_cols.append(_ec)
    val_keys = val_df[val_join_cols].copy().drop_duplicates().reset_index(drop=True)
    pred_df = val_keys.copy()
    pred_df["prediction"] = preds

    diagnostics = {
        "train": train_diag, "val": val_diag,
        "feature_importances": _extract_importances(model, feature_names),
        "n_train_rows": len(X_train), "n_val_rows": len(X_val),
        "n_features": len(feature_names),
    }

    return pred_df, diagnostics


def _find_best_worst(
    val_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    entity_col: str,
    target_col: str,
    task_type: str,
    features: Dict[str, pd.DataFrame],
    asof_col: Optional[str] = None,
    n: int = 10,
) -> tuple:
    """Find top-N best and worst predictions with feature values."""
    # Some splits (e.g. masked test tables) may not include the target column.
    # In that case we can still score via RelBench (which can use unmasked tables),
    # but we can't compute per-row best/worst examples locally.
    if target_col not in val_df.columns:
        logger.info(
            f"Skipping best/worst examples because target_col='{target_col}' "
            "is not present in the provided eval DataFrame."
        )
        return [], []

    join_cols = [entity_col]
    if asof_col and asof_col in val_df.columns and asof_col in pred_df.columns:
        join_cols.append(asof_col)
    merged = val_df[join_cols + [target_col]].merge(
        pred_df[join_cols + ["prediction"]], on=join_cols, how="inner"
    )

    # Compute error magnitude
    try:
        _sample_pred = merged["prediction"].iloc[0] if len(merged) > 0 else None
        if isinstance(_sample_pred, (list, np.ndarray)):
            # Multi-class: predictions are probability vectors; error = wrong argmax
            merged["_actual"] = pd.to_numeric(merged[target_col], errors="coerce")
            merged["_pred"] = merged["prediction"].apply(
                lambda p: int(np.argmax(p)) if isinstance(p, (list, np.ndarray)) else int(float(p))
            )
            merged["_error"] = (merged["_actual"] != merged["_pred"]).astype(float)
        else:
            merged["_actual"] = merged[target_col].astype(float)
            merged["_pred"] = merged["prediction"].astype(float)
            merged["_error"] = (merged["_actual"] - merged["_pred"]).abs()
    except (ValueError, TypeError):
        # For non-numeric (e.g., classification with string labels)
        merged["_error"] = (merged[target_col] != merged["prediction"]).astype(float)

    # Join feature values for context (same dedupe + join keys as _build_feature_matrix).
    for qname, fdf in features.items():
        if entity_col not in fdf.columns:
            continue
        feature_join_cols = [entity_col]
        if len(join_cols) == 2 and asof_col and asof_col in fdf.columns:
            feature_join_cols = [entity_col, asof_col]
        fdf_dedup = fdf.drop_duplicates(subset=feature_join_cols, keep="first")
        feat_cols = [c for c in fdf_dedup.columns if c not in feature_join_cols]
        if not feat_cols:
            continue
        renamed = fdf_dedup[feature_join_cols + feat_cols].rename(
            columns={c: f"{qname}__{c}" for c in feat_cols}
        )
        merged = merged.merge(renamed, on=feature_join_cols, how="left")

    sorted_df = merged.sort_values("_error", ascending=True)
    best_rows = sorted_df.head(n).drop(columns=["_actual", "_pred", "_error"], errors="ignore")
    worst_rows = sorted_df.tail(n).drop(columns=["_actual", "_pred", "_error"], errors="ignore")

    return best_rows.to_dict("records"), worst_rows.to_dict("records")


def _compute_residual_correlations(
    val_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    features: Dict[str, pd.DataFrame],
    entity_col: str,
    target_col: str,
    asof_col: Optional[str] = None,
    min_valid_rows: int = 50,
) -> Optional[pd.DataFrame]:
    """Compute Pearson correlation of each feature with prediction residuals.

    Returns a DataFrame with columns: feature_name, extractor_name, correlation,
    abs_correlation, rank — sorted descending by abs_correlation.
    Returns None if target is unavailable or fewer than min_valid_rows.
    Only meaningful for regression tasks; caller should gate on task_type.
    """
    if target_col not in val_df.columns:
        return None

    join_cols = [entity_col]
    if asof_col and asof_col in val_df.columns and asof_col in pred_df.columns:
        join_cols.append(asof_col)

    try:
        merged = val_df[join_cols + [target_col]].merge(
            pred_df[join_cols + ["prediction"]], on=join_cols, how="inner"
        )
        merged["_actual"] = pd.to_numeric(merged[target_col], errors="coerce")
        merged["_pred"] = pd.to_numeric(merged["prediction"], errors="coerce")
        merged["_residual"] = merged["_actual"] - merged["_pred"]
    except Exception:
        return None

    valid_mask = merged["_residual"].notna()
    if valid_mask.sum() < min_valid_rows:
        return None

    rows = []
    for qname, fdf in features.items():
        if entity_col not in fdf.columns:
            continue
        feat_join_cols = [entity_col]
        if asof_col and asof_col in fdf.columns and len(join_cols) == 2:
            feat_join_cols = [entity_col, asof_col]
        fdf_dedup = fdf.drop_duplicates(subset=feat_join_cols, keep="first")
        feat_cols = [c for c in fdf_dedup.columns if c not in feat_join_cols]
        if not feat_cols:
            continue
        try:
            tmp = merged[feat_join_cols + ["_residual"]].merge(
                fdf_dedup[feat_join_cols + feat_cols], on=feat_join_cols, how="left"
            )
        except Exception:
            continue
        for col in feat_cols:
            feat_vals = pd.to_numeric(tmp[col], errors="coerce")
            resid_vals = tmp["_residual"]
            valid = feat_vals.notna() & resid_vals.notna()
            if valid.sum() < min_valid_rows:
                continue
            try:
                corr = float(feat_vals[valid].corr(resid_vals[valid]))
                if pd.notna(corr):
                    rows.append({
                        "feature_name": f"{qname}__{col}",
                        "extractor_name": qname,
                        "correlation": corr,
                        "abs_correlation": abs(corr),
                    })
            except Exception:
                pass

    if not rows:
        return None

    df = (
        pd.DataFrame(rows)
        .sort_values("abs_correlation", ascending=False)
        .reset_index(drop=True)
    )
    df["rank"] = range(1, len(df) + 1)
    return df


def _compute_shap_importance(
    *,
    features: Dict[str, pd.DataFrame],
    val_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    entity_col: str,
    require_asof_for_multi_entity: bool,
    asof_col: Optional[str],
    max_features: int = 30,
    max_rows: int = 2000,
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """Compute top feature importances using SHAP on a surrogate regressor.

    We explain the model's prediction output by fitting a small tree-based surrogate
    from engineered features -> prediction, then running Tree SHAP on that surrogate.
    """
    # SHAP does not apply to list-valued predictions.
    if pred_df["prediction"].apply(lambda x: isinstance(x, list)).any():
        return [], "SHAP skipped: list-valued predictions are not supported."

    join_cols = [entity_col]
    if require_asof_for_multi_entity and asof_col and asof_col in val_df.columns and asof_col in pred_df.columns:
        join_cols.append(asof_col)

    base = val_df[join_cols].copy().drop_duplicates().reset_index(drop=True)
    pred_keyed = (
        pred_df[join_cols + ["prediction"]]
        .drop_duplicates(subset=join_cols, keep="first")
        .copy()
    )
    base = base.merge(pred_keyed, on=join_cols, how="left")
    if base["prediction"].isna().all():
        return [], "SHAP skipped: no aligned predictions."
    base["prediction"] = pd.to_numeric(base["prediction"], errors="coerce").fillna(0.0)

    # Build a row-level feature matrix by joining each feature query output.
    for qname, fdf in features.items():
        if entity_col not in fdf.columns:
            continue
        feature_join_cols = [entity_col]
        if len(join_cols) == 2 and asof_col and asof_col in fdf.columns:
            feature_join_cols = [entity_col, asof_col]
        feat_cols = [c for c in fdf.columns if c not in feature_join_cols]
        if not feat_cols:
            continue
        renamed = fdf[feature_join_cols + feat_cols].rename(
            columns={c: f"{qname}__{c}" for c in feat_cols}
        )
        renamed = renamed.drop_duplicates(subset=feature_join_cols, keep="first")
        base = base.merge(renamed, on=feature_join_cols, how="left")

    candidate_cols = [c for c in base.columns if c not in join_cols + ["prediction"]]
    if not candidate_cols:
        return [], "SHAP skipped: no feature columns available after joins."

    X = base[candidate_cols].copy()
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = pd.to_datetime(X[col], errors="coerce").astype("int64")
        elif not pd.api.types.is_numeric_dtype(X[col]):
            # Keep deterministic and compact: factorize non-numeric columns.
            X[col] = pd.factorize(X[col], sort=True)[0]
        X[col] = _coerce_ml_feature_column(X[col])

    # SHAP TreeExplainer validates float32; inf survives pd.fillna and breaks explainer.fit.
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    if X.shape[0] == 0 or X.shape[1] == 0:
        return [], "SHAP skipped: empty feature matrix."

    y = pd.to_numeric(base["prediction"], errors="coerce").fillna(0.0).to_numpy()
    if len(y) > max_rows:
        # Deterministic subsample for speed/stability.
        idx = np.linspace(0, len(y) - 1, num=max_rows, dtype=int)
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]

    if np.allclose(y, y[0]):
        return [], "SHAP skipped: predictions are constant."

    try:
        import shap
    except Exception:
        return [], "SHAP skipped: `shap` package is not installed."

    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception:
        return [], "SHAP skipped: `scikit-learn` package is not installed."

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if getattr(shap_values, "ndim", 1) == 1:
        shap_values = np.expand_dims(np.asarray(shap_values), axis=1)
    else:
        shap_values = np.asarray(shap_values)

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame(
        {
            "feature": X.columns.tolist(),
            "mean_abs_shap": mean_abs.tolist(),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    top = importance.head(max_features)
    rows = [
        {"rank": int(i + 1), "feature": row["feature"], "mean_abs_shap": float(row["mean_abs_shap"])}
        for i, row in enumerate(top.to_dict("records"))
    ]
    note = (
        "SHAP computed on a tree-based surrogate model over engineered features. "
        "Top features are ranked by mean absolute SHAP value."
    )
    week_cols = [c for c in X.columns if "week_of_year" in c.lower()]
    if week_cols:
        max_week_nunique = max(int(X[c].nunique(dropna=True)) for c in week_cols)
        if max_week_nunique <= 3:
            note += (
                " week_of_year feature shows little variability because the validation/test "
                "rows are temporally concentrated in a few weeks."
            )
    return rows, note


def execute_and_validate(
    program: ProgramSpec,
    conn: duckdb.DuckDBPyConnection,
    val_df: pd.DataFrame,
    evaluator: Any,  # RelBenchEvaluator
    task_type: str,
    entity_col: str,
    target_col: str,
    trial_id: int = 0,
    split: str = "val",
    require_asof_for_multi_entity: bool = False,
    asof_col: Optional[str] = None,
    extra_entity_cols: Optional[List[str]] = None,
    regression_primary_metric: str = "mae",
    compute_shap: bool = False,
    train_df: Optional[pd.DataFrame] = None,
    sql_timeout_seconds: float = DEFAULT_SQL_QUERY_TIMEOUT_SECONDS,
    store_eval_preds: bool = False,
    is_final: bool = False,
    wrapped_model_save_path: Optional[str] = None,
    max_train_entities: Optional[int] = None,
    seven_models: bool = False,
) -> ValidationResult:
    """
    Execute a scientist program and validate against the ground truth.

    is_final=True: used for final val / test evaluation — all entities are scored.
    is_final=False (default): agent optimization loop.

    1. Run SQL feature queries against DuckDB
    2. Run SQL feature queries and wrapped model
    3. Join predictions with val_df
    4. Feed into RelBenchEvaluator for official RelBench metrics
    5. Find top-10 best and worst predictions
    """
    wrapped_diagnostics: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, pd.DataFrame]] = None
    pred_df: Optional[pd.DataFrame] = None
    aligned: Optional[pd.DataFrame] = None
    val_reset: Optional[pd.DataFrame] = None
    pred_keyed: Optional[pd.DataFrame] = None

    try:
        _aggressive_collect()
        # Always expose the current evaluation rows in SQL as `eval_table`.
        # IMPORTANT: keep this label-free to avoid leaking target values into SQL.
        # This lets feature queries anchor on split keys without exposing answers.
        eval_sql_df = val_df.copy()
        if target_col in eval_sql_df.columns:
            eval_sql_df = eval_sql_df.drop(columns=[target_col])
        try:
            try:
                conn.unregister("eval_table")
            except Exception:
                pass
            conn.register("eval_table", eval_sql_df)
        except Exception as e:
            raise ValueError(f"Failed to register eval_table for split='{split}': {e}") from e


        # Also register train rows as train_table
        if train_df is not None:
            try:
                conn.unregister("train_table")
            except Exception:
                pass
            conn.register("train_table", train_df)

        # Step 1: Run SQL queries
        _n_val_entities = val_df[entity_col].nunique() if (entity_col and entity_col in val_df.columns) else None
        features = _run_feature_queries(
            conn,
            program.feature_queries,
            entity_col=entity_col,
            allowed_entities=None,
            sql_timeout_seconds=sql_timeout_seconds,
            n_entities=_n_val_entities,
            n_candidates=None,
            eval_df=None,
        )

        # --- Wrapped model path ---
        if train_df is None:
            raise ValueError("Wrapped model mode requires train_df to be provided.")

        # Extract log_transform_target before resolving model config (it is not a
        # model hyperparameter and must not be forwarded to sklearn).
        raw_model_config = dict(program.model_config) if program.model_config else {}
        log_transform_target = bool(raw_model_config.pop("log_transform_target", False))
        if log_transform_target and task_type != "entity_regression":
            logger.warning("log_transform_target=True ignored for non-regression task.")
            log_transform_target = False

        resolved_config, config_warnings = _resolve_model_config(
            program.model_choice, raw_model_config,
            seven_models=seven_models,
        )
        logger.info(
            f"Wrapped model: {program.model_choice}, resolved config: {resolved_config}"
            + (", log_transform_target=True" if log_transform_target else "")
        )
        for w in config_warnings:
            logger.warning(f"Config warning: {w}")

        # Run feature queries a second time with train rows as eval_table
        # so the LLM's queries anchored on eval_table produce train features too.
        train_df_for_features = train_df
        if (
            max_train_entities is not None
            and entity_col
            and entity_col in train_df.columns
        ):
            # Caller-specified entity cap for large train sets.
            # Sample by entity (not row) to preserve temporal structure —
            # all timestamps for a sampled entity are retained.
            train_entity_ids = train_df[entity_col].unique()
            if len(train_entity_ids) > max_train_entities:
                rng = np.random.default_rng(42)
                sampled_ids = rng.choice(
                    train_entity_ids, max_train_entities, replace=False
                )
                sampled_set = set(sampled_ids.tolist())
                train_df_for_features = train_df[
                    train_df[entity_col].isin(sampled_set)
                ].reset_index(drop=True)
                logger.info(
                    f"Train entity subsample: {len(train_entity_ids):,} → "
                    f"{max_train_entities:,} unique entities "
                    f"({len(train_df_for_features):,} rows) for feature queries."
                )

        train_eval_df = train_df_for_features.copy()
        if target_col in train_eval_df.columns:
            train_eval_df = train_eval_df.drop(columns=[target_col])
        try:
            conn.unregister("eval_table")
        except Exception:
            pass
        conn.register("eval_table", train_eval_df)

        _n_train_entities = train_df_for_features[entity_col].nunique() if (entity_col and entity_col in train_df_for_features.columns) else None
        train_features = _run_feature_queries(
            conn, program.feature_queries,
            entity_col=entity_col, allowed_entities=None,
            sql_timeout_seconds=sql_timeout_seconds,
            n_entities=_n_train_entities,
            n_candidates=None,
            eval_df=None,
        )

        # Re-register val rows as eval_table and re-run for val features
        try:
            conn.unregister("eval_table")
        except Exception:
            pass
        conn.register("eval_table", eval_sql_df)

        val_features = _run_feature_queries(
            conn, program.feature_queries,
            entity_col=entity_col, allowed_entities=None,
            sql_timeout_seconds=sql_timeout_seconds,
            n_entities=_n_val_entities,
            n_candidates=None,
            eval_df=None,
        )

        pred_df, wrapped_diagnostics = _fit_and_predict_wrapped(
            features=val_features,
            train_features=train_features,
            train_df=train_df_for_features,
            val_df=val_df,
            entity_col=entity_col,
            target_col=target_col,
            model_choice=program.model_choice,
            resolved_config=resolved_config,
            task_type=task_type,
            asof_col=asof_col if require_asof_for_multi_entity else None,
            extra_entity_cols=extra_entity_cols,
            plus_split=split,
            is_final=is_final,
            model_save_path=wrapped_model_save_path,
            log_transform_target=log_transform_target,
            store_eval_preds=store_eval_preds,
        )
        wrapped_diagnostics["resolved_model_choice"] = program.model_choice
        wrapped_diagnostics["resolved_model_config"] = resolved_config
        if log_transform_target:
            wrapped_diagnostics["log_transform_target"] = True
        wrapped_diagnostics["config_warnings"] = config_warnings
        if pred_df.empty:
            logger.warning("Wrapped model returned empty predictions")

        val_unique_entities = (
            val_df[entity_col].nunique(dropna=True) if entity_col in val_df.columns else len(val_df)
        )
        if require_asof_for_multi_entity:
            if asof_col is None:
                raise ValueError("Strict as-of alignment enabled but no asof_col provided.")
            if asof_col not in pred_df.columns:
                raise ValueError(
                    f"Strict as-of alignment requires combiner output column '{asof_col}'."
                )
            pred_unique_keys = len(pred_df[[entity_col, asof_col]].drop_duplicates())
            pred_duplicate_rows = len(pred_df) - pred_unique_keys
            logger.info(
                "Prediction coverage before alignment: "
                f"val_rows={len(val_df)}, val_unique_{entity_col}={val_unique_entities}, "
                f"pred_rows={len(pred_df)}, pred_unique_({entity_col},{asof_col})={pred_unique_keys}, "
                f"pred_duplicate_({entity_col},{asof_col})_rows={pred_duplicate_rows}"
            )
            if pred_duplicate_rows > 0:
                logger.warning(
                    f"Combiner returned duplicate ({entity_col}, {asof_col}) rows; "
                    f"deduplicating by keeping first prediction per key."
                )
                pred_df = pred_df.drop_duplicates(
                    subset=[entity_col, asof_col], keep="first", ignore_index=True
                )
        else:
            pred_unique_entities = pred_df[entity_col].nunique(dropna=True)
            pred_duplicate_rows = len(pred_df) - pred_unique_entities
            logger.info(
                "Prediction coverage before alignment: "
                f"val_rows={len(val_df)}, val_unique_{entity_col}={val_unique_entities}, "
                f"pred_rows={len(pred_df)}, pred_unique_{entity_col}={pred_unique_entities}, "
                f"pred_duplicate_{entity_col}_rows={pred_duplicate_rows}"
            )
            if pred_duplicate_rows > 0:
                logger.warning(
                    f"Combiner returned duplicate {entity_col} rows; "
                    f"deduplicating by keeping first prediction per entity."
                )
                pred_df = pred_df.drop_duplicates(subset=[entity_col], keep="first", ignore_index=True)

        # Step 3: Align predictions with val_df.
        # For tasks with repeated entities across timestamps, optionally require
        # explicit as-of key alignment to prevent entity-only collapse.
        val_reset = val_df.reset_index(drop=True)
        if require_asof_for_multi_entity:
            if asof_col not in val_reset.columns:
                raise ValueError(
                    f"Strict as-of alignment enabled, but val_df lacks '{asof_col}'."
                )

            pred_keyed = pred_df[[entity_col, asof_col, "prediction"]].drop_duplicates(
                subset=[entity_col, asof_col], keep="first", ignore_index=True
            )
            val_keys = val_reset[[entity_col, asof_col]].copy()
            aligned = val_keys.merge(pred_keyed, on=[entity_col, asof_col], how="left")
        else:
            # Default entity-only alignment.
            val_entities = val_reset[[entity_col]].copy()
            pred_map = pred_df.drop_duplicates(subset=[entity_col], keep="first").set_index(entity_col)["prediction"]
            aligned = val_entities.copy()
            aligned["prediction"] = aligned[entity_col].map(pred_map)

        n_missing = aligned["prediction"].isna().sum()
        if require_asof_for_multi_entity and n_missing == len(aligned) and len(aligned) > 0:
            raise ValueError(
                f"0% prediction coverage after strict ({entity_col}, {asof_col}) alignment. "
                "Your SQL is likely keyed to the wrong row set (e.g., train_table timestamps). "
                "Anchor feature queries on eval_table and return predictions keyed by eval_table rows."
            )
        if n_missing > 0:
            missing_ids = (
                aligned.loc[aligned["prediction"].isna(), entity_col]
                .dropna()
                .drop_duplicates()
                .head(10)
                .tolist()
            )
            logger.warning(f"Missing predictions for {n_missing}/{len(aligned)} entities, filling with defaults")
            logger.info(f"Sample missing {entity_col} values after alignment (up to 10): {missing_ids}")
            if task_type == "entity_classification":
                aligned["prediction"] = aligned["prediction"].fillna(0)
            else:
                aligned["prediction"] = aligned["prediction"].fillna(0.0)

        # Step 4: Feed into RelBenchEvaluator
        # Reset evaluator and add predictions row by row
        evaluator.predictions.clear()
        evaluator.results.clear()

        # Iterate by row position to keep prediction index aligned with val_df row order.
        aligned = aligned.reset_index(drop=True)
        for pos, row in aligned.iterrows():
            pred_val = row["prediction"]
            gt_val = val_reset.iloc[pos][target_col] if target_col in val_reset.columns else None
            evaluator.add_prediction(
                pos,
                {"predictions": [pred_val] if not isinstance(pred_val, list) else pred_val},
                gt_val,
            )

        metrics = evaluator.evaluate_all(split=split)

        # Determine primary score (higher = better)
        if task_type == "entity_regression":
            metric_choice = (regression_primary_metric or "mae").lower()
            if metric_choice == "r2":
                score = metrics.get("r2", metrics.get("r_squared", float("-inf")))
                if score == float("-inf"):
                    logger.warning(
                        "Regression objective set to R2 but evaluator did not return R2; "
                        "falling back to negative MAE for scoring."
                    )
                    score = -metrics.get("mae", float("inf"))
            else:
                # MAE is minimized; negate so higher = better for search.
                score = -metrics.get("mae", float("inf"))
        else:
            # Classification: prefer threshold-free ranking metrics when available.
            # mrr for multi-class; roc_auc for binary.
            score = metrics.get(
                "mrr",
                metrics.get(
                    "roc_auc",
                    metrics.get(
                        "auroc",
                        metrics.get(
                            "average_precision",
                            metrics.get("macro_f1", metrics.get("f1", metrics.get("accuracy", 0.0))),
                        ),
                    ),
                ),
            )

        # Step 5: Find best/worst predictions
        best_preds, worst_preds = _find_best_worst(
            val_df,
            pred_df,
            entity_col,
            target_col,
            task_type,
            features,
            asof_col=asof_col if require_asof_for_multi_entity else None,
        )

        shap_importance: List[Dict[str, Any]] = []
        shap_note: Optional[str] = None
        if compute_shap:
            try:
                shap_importance, shap_note = _compute_shap_importance(
                    features=features,
                    val_df=val_df,
                    pred_df=pred_df,
                    entity_col=entity_col,
                    require_asof_for_multi_entity=require_asof_for_multi_entity,
                    asof_col=asof_col,
                )
            except Exception as e:
                shap_importance = []
                shap_note = f"SHAP failed: {e}"

        primary_metric_name = _determine_primary_metric_name(
            task_type, metrics, regression_primary_metric
        )

        # Build full eval predictions DataFrame for EvalWorkspace persistence.
        # This is a copy so the finally block can safely clear `aligned` / `val_reset`.
        eval_preds_df: Optional[pd.DataFrame] = None
        if store_eval_preds:
            try:
                if aligned is not None:
                    n_rows = len(aligned)
                    eid_series = (
                        aligned[entity_col].astype(str)
                        if entity_col in aligned.columns
                        else pd.Series([""] * n_rows, dtype=str)
                    )
                    if target_col in val_reset.columns:
                        lbl_series = val_reset.reset_index(drop=True)[target_col].astype(str)
                    else:
                        lbl_series = pd.Series([None] * n_rows, dtype=object)

                    _sample_p = aligned["prediction"].iloc[0] if n_rows > 0 else None
                    _is_mc = isinstance(_sample_p, (list, np.ndarray))
                    if _is_mc:
                        # Multi-class: score = max proba; pred_class = argmax
                        score_series = aligned["prediction"].apply(
                            lambda p: float(np.max(p)) if isinstance(p, (list, np.ndarray)) else 0.0
                        )
                        pred_class_series = aligned["prediction"].apply(
                            lambda p: str(int(np.argmax(p))) if isinstance(p, (list, np.ndarray)) else "0"
                        )
                    else:
                        raw_preds = pd.to_numeric(aligned["prediction"], errors="coerce")
                        score_series = raw_preds.astype(float)
                        if task_type == "entity_classification":
                            pred_class_series = (
                                (raw_preds > 0.5).astype(int).astype(str)
                            )
                        else:
                            pred_class_series = pd.Series([None] * n_rows, dtype=object)

                    if require_asof_for_multi_entity and asof_col and asof_col in aligned.columns:
                        cutoff_series = aligned[asof_col].values
                    else:
                        cutoff_series = [None] * n_rows

                    eval_preds_df = pd.DataFrame({
                        "row_id": range(n_rows),
                        "entity_id": eid_series.values,
                        "label": lbl_series.values,
                        "score": score_series.values,
                        "predicted_class": pred_class_series.values,
                        "split": split,
                        "eval_cutoff": cutoff_series,
                    })
            except Exception as _ep_err:
                logger.warning(f"Failed to build eval_predictions_df for trial {trial_id}: {_ep_err}")
                eval_preds_df = None

        # Compute residual correlations for regression tasks (no eval_workspace).
        # When store_eval_preds=True (--eval_workspace), skip: workspace has no
        # residual_correlations_table and validate_program omits this inline block.
        if task_type == "entity_regression" and features is not None and not store_eval_preds:
            try:
                resid_corr_df = _compute_residual_correlations(
                    val_df=val_df,
                    pred_df=pred_df,
                    features=features,
                    entity_col=entity_col,
                    target_col=target_col,
                    asof_col=asof_col if require_asof_for_multi_entity else None,
                )
                if resid_corr_df is not None and not resid_corr_df.empty:
                    if wrapped_diagnostics is None:
                        wrapped_diagnostics = {}
                    wrapped_diagnostics["residual_corr_df"] = resid_corr_df
            except Exception as _rc_err:
                logger.warning(f"Residual correlation computation failed: {_rc_err}")

        return ValidationResult(
            trial_id=trial_id,
            score=score,
            metrics=metrics,
            worst_predictions=worst_preds,
            best_predictions=best_preds,
            n_predictions=len(aligned),
            missing_predictions=int(n_missing),
            total_entities=int(len(aligned)),
            coverage_rate=float((len(aligned) - n_missing) / max(len(aligned), 1)),
            shap_importance=shap_importance,
            shap_note=shap_note,
            wrapped_diagnostics=wrapped_diagnostics,
            primary_metric_name=primary_metric_name,
            eval_predictions_df=eval_preds_df,
        )

    except Exception as e:
        logger.error(f"Trial {trial_id} failed: {e}\n{traceback.format_exc()}")
        return ValidationResult(
            trial_id=trial_id,
            score=float("-inf"),
            metrics={},
            worst_predictions=[],
            best_predictions=[],
            n_predictions=0,
            error=str(e),
        )
    finally:
        # Aggressively release local dataframes between trials.
        try:
            if isinstance(features, dict):
                features.clear()
            for obj_name in ("pred_df", "aligned", "val_reset", "pred_keyed"):
                obj = locals().get(obj_name)
                if isinstance(obj, pd.DataFrame):
                    obj.drop(obj.index, inplace=True)
        except Exception:
            pass
        _aggressive_collect()


def _evaluate_pred_df_for_model(
    pred_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: Optional[Dict[str, pd.DataFrame]],
    wrapped_diagnostics: Optional[Dict[str, Any]],
    evaluator: Any,
    task_type: str,
    entity_col: str,
    target_col: str,
    trial_id: int,
    split: str,
    require_asof_for_multi_entity: bool,
    asof_col: Optional[str],
    regression_primary_metric: str,
    store_eval_preds: bool,
) -> ValidationResult:
    """Align pred_df with val_df, evaluate, and return a ValidationResult.

    Extracted to allow multi-model evaluation to run SQL queries once and reuse
    features/train_features across multiple model fits.
    """
    try:
        # Deduplicate predictions
        if require_asof_for_multi_entity and asof_col and asof_col in pred_df.columns:
            pred_df = pred_df.drop_duplicates(
                subset=[entity_col, asof_col], keep="first", ignore_index=True
            )
        else:
            if entity_col in pred_df.columns:
                pred_df = pred_df.drop_duplicates(subset=[entity_col], keep="first", ignore_index=True)

        # Align predictions with val_df
        val_reset = val_df.reset_index(drop=True)
        if require_asof_for_multi_entity and asof_col:
            pred_keyed = pred_df[[entity_col, asof_col, "prediction"]].drop_duplicates(
                subset=[entity_col, asof_col], keep="first", ignore_index=True
            )
            val_keys = val_reset[[entity_col, asof_col]].copy()
            aligned = val_keys.merge(pred_keyed, on=[entity_col, asof_col], how="left")
        else:
            val_entities = val_reset[[entity_col]].copy()
            pred_map = pred_df.drop_duplicates(subset=[entity_col], keep="first").set_index(entity_col)["prediction"]
            aligned = val_entities.copy()
            aligned["prediction"] = aligned[entity_col].map(pred_map)

        n_missing = aligned["prediction"].isna().sum()
        if n_missing > 0:
            if task_type == "entity_classification":
                aligned["prediction"] = aligned["prediction"].fillna(0)
            else:
                aligned["prediction"] = aligned["prediction"].fillna(0.0)

        # Evaluate
        evaluator.predictions.clear()
        evaluator.results.clear()
        aligned = aligned.reset_index(drop=True)
        for pos, row in aligned.iterrows():
            pred_val = row["prediction"]
            gt_val = val_reset.iloc[pos][target_col] if target_col in val_reset.columns else None
            evaluator.add_prediction(
                pos,
                {"predictions": [pred_val] if not isinstance(pred_val, list) else pred_val},
                gt_val,
            )
        metrics = evaluator.evaluate_all(split=split)

        # Primary score
        if task_type == "entity_regression":
            metric_choice = (regression_primary_metric or "mae").lower()
            if metric_choice == "r2":
                score = metrics.get("r2", metrics.get("r_squared", float("-inf")))
                if score == float("-inf"):
                    score = -metrics.get("mae", float("inf"))
            else:
                score = -metrics.get("mae", float("inf"))
        else:
            score = metrics.get(
                "mrr",
                metrics.get(
                    "roc_auc",
                    metrics.get("auroc", metrics.get("average_precision", metrics.get("macro_f1", metrics.get("f1", metrics.get("accuracy", 0.0))))),
                ),
            )

        best_preds, worst_preds = _find_best_worst(
            val_df, pred_df, entity_col, target_col, task_type, features,
            asof_col=asof_col if require_asof_for_multi_entity else None,
        )

        primary_metric_name = _determine_primary_metric_name(task_type, metrics, regression_primary_metric)

        eval_preds_df: Optional[pd.DataFrame] = None
        if store_eval_preds and aligned is not None:
            try:
                n_rows = len(aligned)
                eid_series = (
                    aligned[entity_col].astype(str)
                    if entity_col in aligned.columns
                    else pd.Series([""] * n_rows, dtype=str)
                )
                if target_col in val_reset.columns:
                    lbl_series = val_reset.reset_index(drop=True)[target_col].astype(str)
                else:
                    lbl_series = pd.Series([None] * n_rows, dtype=object)
                _sample_p2 = aligned["prediction"].iloc[0] if n_rows > 0 else None
                _is_mc2 = isinstance(_sample_p2, (list, np.ndarray))
                if _is_mc2:
                    score_series = aligned["prediction"].apply(
                        lambda p: float(np.max(p)) if isinstance(p, (list, np.ndarray)) else 0.0
                    )
                    pred_class_series = aligned["prediction"].apply(
                        lambda p: str(int(np.argmax(p))) if isinstance(p, (list, np.ndarray)) else "0"
                    )
                else:
                    raw_preds = pd.to_numeric(aligned["prediction"], errors="coerce")
                    score_series = raw_preds.astype(float)
                    pred_class_series = (
                        (raw_preds > 0.5).astype(int).astype(str)
                        if task_type == "entity_classification"
                        else pd.Series([None] * n_rows, dtype=object)
                    )
                if require_asof_for_multi_entity and asof_col and asof_col in aligned.columns:
                    cutoff_series = aligned[asof_col].values
                else:
                    cutoff_series = [None] * n_rows
                eval_preds_df = pd.DataFrame({
                    "row_id": range(n_rows),
                    "entity_id": eid_series.values,
                    "label": lbl_series.values,
                    "score": score_series.values,
                    "predicted_class": pred_class_series.values,
                    "split": split,
                    "eval_cutoff": cutoff_series,
                })
            except Exception as _ep_err:
                logger.warning(f"Failed to build eval_predictions_df for trial {trial_id}: {_ep_err}")

        return ValidationResult(
            trial_id=trial_id,
            score=score,
            metrics=metrics,
            worst_predictions=worst_preds,
            best_predictions=best_preds,
            n_predictions=len(aligned),
            missing_predictions=int(n_missing),
            total_entities=int(len(aligned)),
            coverage_rate=float((len(aligned) - n_missing) / max(len(aligned), 1)),
            wrapped_diagnostics=wrapped_diagnostics,
            primary_metric_name=primary_metric_name,
            eval_predictions_df=eval_preds_df,
        )
    except Exception as e:
        logger.error(f"Trial {trial_id} evaluation failed: {e}\n{traceback.format_exc()}")
        return ValidationResult(
            trial_id=trial_id,
            score=float("-inf"),
            metrics={},
            worst_predictions=[],
            best_predictions=[],
            n_predictions=0,
            error=str(e),
        )


def execute_and_validate_multi_model(
    feature_queries: List[Dict[str, str]],
    conn: duckdb.DuckDBPyConnection,
    val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    evaluator: Any,
    task_type: str,
    entity_col: str,
    target_col: str,
    trial_id_start: int,
    model_choices: tuple,
    split: str = "val",
    require_asof_for_multi_entity: bool = False,
    asof_col: Optional[str] = None,
    extra_entity_cols: Optional[List[str]] = None,
    regression_primary_metric: str = "mae",
    sql_timeout_seconds: float = DEFAULT_SQL_QUERY_TIMEOUT_SECONDS,
    store_eval_preds: bool = False,
    is_final: bool = False,
    max_train_entities: Optional[int] = None,
) -> "List[tuple]":
    """Run SQL feature queries ONCE, then evaluate each model in model_choices with default HPs.

    Designed for the multi-model mode where the agent proposes only SQL features and
    the environment automatically benchmarks all standard models.

    Returns list of (model_choice, ValidationResult) in the order of model_choices.
    On SQL failure, returns error ValidationResults for all models.
    """
    results: list = []
    train_features: Optional[Dict[str, pd.DataFrame]] = None
    val_features: Optional[Dict[str, pd.DataFrame]] = None

    # --- Train entity subsampling ---
    train_df_for_features = train_df
    if max_train_entities is not None and entity_col and entity_col in train_df.columns:
        train_entity_ids = train_df[entity_col].unique()
        if len(train_entity_ids) > max_train_entities:
            rng = np.random.default_rng(42)
            sampled_ids = rng.choice(train_entity_ids, max_train_entities, replace=False)
            sampled_set = set(sampled_ids.tolist())
            train_df_for_features = train_df[train_df[entity_col].isin(sampled_set)].reset_index(drop=True)

    # --- SQL setup ---
    eval_sql_df = val_df.copy()
    if target_col in eval_sql_df.columns:
        eval_sql_df = eval_sql_df.drop(columns=[target_col])
    try:
        conn.unregister("eval_table")
    except Exception:
        pass
    conn.register("eval_table", eval_sql_df)
    try:
        conn.unregister("train_table")
    except Exception:
        pass
    conn.register("train_table", train_df)

    _n_val_entities = val_df[entity_col].nunique() if (entity_col and entity_col in val_df.columns) else None

    sql_error: Optional[str] = None
    try:
        # Run train feature queries
        train_eval_df = train_df_for_features.copy()
        if target_col in train_eval_df.columns:
            train_eval_df = train_eval_df.drop(columns=[target_col])
        try:
            conn.unregister("eval_table")
        except Exception:
            pass
        conn.register("eval_table", train_eval_df)

        _n_train_entities = (
            train_df_for_features[entity_col].nunique()
            if (entity_col and entity_col in train_df_for_features.columns)
            else None
        )
        train_features = _run_feature_queries(
            conn, feature_queries,
            entity_col=entity_col, allowed_entities=None,
            sql_timeout_seconds=sql_timeout_seconds,
            n_entities=_n_train_entities,
            n_candidates=None,
            eval_df=None,
        )

        # Re-register val as eval_table and run val feature queries
        try:
            conn.unregister("eval_table")
        except Exception:
            pass
        conn.register("eval_table", eval_sql_df)

        val_features = _run_feature_queries(
            conn, feature_queries,
            entity_col=entity_col, allowed_entities=None,
            sql_timeout_seconds=sql_timeout_seconds,
            n_entities=_n_val_entities,
            n_candidates=None,
            eval_df=None,
        )
    except Exception as sql_err:
        sql_error = str(sql_err)
        logger.error(f"Multi-model SQL query failed: {sql_err}\n{traceback.format_exc()}")

    if sql_error is not None:
        for i, mc in enumerate(model_choices):
            results.append((mc, ValidationResult(
                trial_id=trial_id_start + i,
                score=float("-inf"),
                metrics={},
                worst_predictions=[],
                best_predictions=[],
                n_predictions=0,
                error=sql_error,
            )))
        return results

    # --- Precompute feature matrices once (shared across all models) ---
    _mm_asof = asof_col if require_asof_for_multi_entity else None
    _X_train_pre, _train_diag_pre, _train_cats_pre = _build_feature_matrix(
        train_features, train_df_for_features, entity_col, _mm_asof,
        extra_entity_cols=extra_entity_cols,
        return_categories=True,
    )
    _join_cols_pre = [entity_col]
    if _mm_asof and _mm_asof in train_df_for_features.columns:
        _join_cols_pre.append(_mm_asof)
    for _ec in (extra_entity_cols or []):
        if _ec not in _join_cols_pre and _ec in train_df_for_features.columns:
            _join_cols_pre.append(_ec)
    _train_keys_pre = train_df_for_features[_join_cols_pre].copy().drop_duplicates().reset_index(drop=True)
    _y_aligned_pre = _train_keys_pre.merge(
        train_df_for_features[_join_cols_pre + [target_col]], on=_join_cols_pre, how="left"
    )[target_col]
    _y_train_pre = pd.to_numeric(_y_aligned_pre, errors="coerce").fillna(0).values

    _X_val_pre, _val_diag_pre = _build_feature_matrix(
        val_features, val_df, entity_col, _mm_asof,
        extra_entity_cols=extra_entity_cols,
        train_categories=_train_cats_pre,
    )
    _feat_names_pre = list(_X_train_pre.columns)
    for _col in _feat_names_pre:
        if _col not in _X_val_pre.columns:
            _X_val_pre[_col] = 0.0
    _X_val_pre = _X_val_pre[_feat_names_pre]

    _prebuilt = {
        "X_train": _X_train_pre,
        "y_train": _y_train_pre,
        "X_val": _X_val_pre,
        "train_diag": _train_diag_pre,
        "val_diag": _val_diag_pre,
        "train_categories": _train_cats_pre,
    }
    del train_features  # free raw SQL DataFrames before fitting models

    # --- Fit each model with default HPs ---
    for i, model_choice in enumerate(model_choices):
        trial_id = trial_id_start + i
        try:
            resolved_config, config_warnings = _resolve_model_config(
                model_choice, {},
                seven_models=(model_choice in _SEVEN_MODELS_CAT_CAPABLE),
            )
            logger.info(f"Multi-model trial {trial_id}: fitting {model_choice} with defaults")

            pred_df, wrapped_diagnostics = _fit_and_predict_wrapped(
                features=val_features,
                train_features=None,
                train_df=train_df_for_features,
                val_df=val_df,
                entity_col=entity_col,
                target_col=target_col,
                model_choice=model_choice,
                resolved_config=resolved_config,
                task_type=task_type,
                asof_col=_mm_asof,
                extra_entity_cols=extra_entity_cols,
                is_final=is_final,
                store_eval_preds=store_eval_preds,
                _prebuilt_matrices=_prebuilt,
            )
            wrapped_diagnostics["resolved_model_choice"] = model_choice
            wrapped_diagnostics["resolved_model_config"] = resolved_config
            wrapped_diagnostics["config_warnings"] = config_warnings

            result = _evaluate_pred_df_for_model(
                pred_df=pred_df,
                val_df=val_df,
                features=val_features,
                wrapped_diagnostics=wrapped_diagnostics,
                evaluator=evaluator,
                task_type=task_type,
                entity_col=entity_col,
                target_col=target_col,
                trial_id=trial_id,
                split=split,
                require_asof_for_multi_entity=require_asof_for_multi_entity,
                asof_col=asof_col,
                regression_primary_metric=regression_primary_metric,
                store_eval_preds=store_eval_preds,
            )
            results.append((model_choice, result))
        except Exception as model_err:
            logger.error(f"Multi-model trial {trial_id} ({model_choice}) failed: {model_err}\n{traceback.format_exc()}")
            results.append((model_choice, ValidationResult(
                trial_id=trial_id,
                score=float("-inf"),
                metrics={},
                worst_predictions=[],
                best_predictions=[],
                n_predictions=0,
                error=str(model_err),
            )))
        _aggressive_collect()

    return results
