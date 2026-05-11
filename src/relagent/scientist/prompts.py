"""Scientist agent prompts — system and execution prompts for each task type."""
from typing import Optional

# ---------------------------------------------------------------------------
# Common rules
# ---------------------------------------------------------------------------

_COMMON_RULES_WRAPPED = r"""
Rules:
- Use SQL tools to explore the database. Do NOT guess table/column names.
- Start by running SHOW TABLES and PRAGMA table_info('table') to understand the schema.
- train_table contains labeled training examples. Use it to learn patterns.
- eval_table contains the current rows being scored (validation/test input keys).
- Always anchor feature SQL on eval_table (not train_table) — eval_table is swapped with the correct split at each call.
- Write SQL queries that extract features PER ENTITY — each query must include {entity_col}.
- Call validate_program() with your SQL feature blocks, model_choice, and optional model_config_json to run on the validation set.
- Iterate based on the metrics, diagnostics, and error examples returned.
"""


def _format_tool_inventory(
    *,
    use_eval_workspace: bool,
) -> str:
    """Opening tool list for system prompts."""
    validate_line = (
        "2. validate_program(feature_queries_json, model_choice, model_config_json) "
        "— train and evaluate your feature pipeline on the validation split"
    )
    lines = [
        "You have the following tools:",
        "1. SQL tools (execute_query, get_table_info, etc.) — explore and query the database",
        validate_line,
        "3. get_trial_history() — see what you've already tried and their scores",
    ]
    if use_eval_workspace:
        lines.append(
            "4. query_eval_workspace(sql) — analyze the evaluation workspace (trials, eval_predictions)"
        )
    return "\n".join(lines) + "\n\n"


def _execute_prompt_capability_footer(*, use_eval_workspace: bool) -> str:
    if not use_eval_workspace:
        return ""
    return (
        "\n\nYou have query_eval_workspace(sql): after each validate_program(), analyze trials "
        "and eval_predictions as in the system prompt."
    )


_TIME_AWARE_RULES = r"""
Time-aware prediction requirements:
- This task is row-level over time. The prediction key is ({entity_col}, {time_col}), not {entity_col} alone.
- Build features and predictions per row timestamp; do NOT collapse to one row per {entity_col}.
- For time-aware tasks, anchor feature SQL on eval_table (current split rows), not train_table.
- Keep both {entity_col} and {time_col} in feature query outputs whenever possible.
- combine() output must include columns: [{entity_col}, {time_col}, 'prediction'].
- IMPORTANT — explore BOTH sides of every relationship: the target entity's own properties are just a starting point. For every foreign key in the schema, also build aggregated features from the entity on the other side of that key, using all its historical records filtered to before {time_col}. Multi-hop features (two or more joins deep) are often the strongest predictors.
- Every aggregation over event/interaction tables MUST include a temporal filter (e.g. WHERE event_time < {time_col}) to avoid using future data.
- The accumulated count of relevant events BEFORE {time_col} is often the single strongest predictor. Always use WHERE event_time < {time_col} — never aggregate events that occur after {time_col}, as those are future data. Always use {time_col} as the temporal cutoff for ALL aggregations — never substitute the target entity's own creation timestamp or any other entity date column as the boundary.
- For any table with a categorical or type column, enumerate all possible values before writing aggregations — run SELECT DISTINCT type_col FROM related_table LIMIT 200 first, then build one aggregated feature per value rather than exploring values one at a time.
"""

# ---------------------------------------------------------------------------
# System prompts (wrapped model mode)
# ---------------------------------------------------------------------------

SCIENTIST_SYSTEM_PROMPT_CLASSIFICATION_WRAPPED = r"""
You are a data scientist building a predictive pipeline for a {benchmark_name} CLASSIFICATION task.

GOAL: Find a set of SQL feature queries + a model choice that accurately predict
the class label ({target_col}) for ALL entities in the validation set.

{tool_inventory}""" + _COMMON_RULES_WRAPPED + r"""
- For BINARY classification, the model outputs probability scores (0.0 to 1.0).
- For MULTICLASS classification, the model outputs class labels.
- Think about features like: entity frequency, recency, aggregates from related tables.
- The validation tool returns official {benchmark_name} metrics (AUROC, F1, etc.) + diagnostics.
"""

SCIENTIST_SYSTEM_PROMPT_REGRESSION_WRAPPED = r"""
You are a data scientist building a predictive pipeline for a {benchmark_name} REGRESSION task.

GOAL: Find a set of SQL feature queries + a model choice that accurately predict
the numerical value ({target_col}) for ALL entities in the validation set.

{tool_inventory}""" + _COMMON_RULES_WRAPPED + r"""
- The model outputs a numeric value for each entity.
- The validation tool returns official {benchmark_name} metrics (MAE) + diagnostics.
- Think about: historical averages, trends, aggregates from related tables.
- Primary objective for this run: {regression_objective}.
"""

# ---------------------------------------------------------------------------
# Model rules — appended to system prompts
# ---------------------------------------------------------------------------

_WRAPPED_MODEL_RULES_CLS = r"""

=== WRAPPED MODEL MODE ===

Your validate_program() tool accepts:
  - feature_queries_json: SQL feature queries (same as before)
  - model_choice: one of the 7 learners below
  - model_config_json: optional JSON dict of hyperparameters (strong defaults apply)

Available models:
  1. "gbdt"     — Standard Gradient Boosted Trees. Fast, strong default.
                  Config: n_estimators (50-500), learning_rate (0.01-0.3), max_depth (2-10),
                          min_child_samples (1-100), subsample (0.5-1.0), colsample_bytree (0.5-1.0)
                  Regularization: lambda_l1 (0.0-10.0), lambda_l2 (0.0-10.0)
  2. "rf"       — Random Forest (bagging of trees; less sensitive to learning rate).
                  Config: same keys as gbdt
  3. "dart"     — DART Boosting (dropout regularization; can reduce overfitting).
                  Config: same keys as gbdt
  4. "goss"     — GOSS (gradient-based subsampling; fast on large datasets).
                  Config: same keys as gbdt
  5. "xgboost"  — XGBoost (second-order gradients; different regularization profile).
                  Config: n_estimators (50-500), learning_rate (0.01-0.3), max_depth (2-10),
                          min_child_weight (1-100), subsample (0.5-1.0), colsample_bytree (0.5-1.0)
                  Regularization: reg_alpha (0.0-10.0), reg_lambda (0.0-10.0)
  6. "xgb_dart" — XGBoost + DART dropout.
                  Config: same keys as xgboost
  7. "catboost" — CatBoost (ordered boosting; robust on heterogeneous tabular features).
                  Config: n_estimators (50-500), learning_rate (0.01-0.3), max_depth (2-10),
                          l2_leaf_reg (0.1-10.0)

Categorical features (all 7 learners):
  Add "categorical_features" inside model_config_json to have a column treated
  natively as categorical rather than silently integer-factorized to float.
  Format: a list of fully-qualified feature names "<query_name>__<col>" —
  the harness renames every column from a feature query "q" to "q__col" before
  handing it to the model. Example:
    {"n_estimators": 300, "categorical_features": ["user_demo__country", "user_demo__state"]}
  Per-family behaviour:
    - gbdt/rf/dart/goss: LightGBM's native categorical_feature at fit time.
    - xgboost/xgb_dart : XGBoost with enable_categorical=True.
    - catboost         : CatBoost cat_features (NaN replaced by a sentinel).
  Train-set categories are learned once and reused on val/test; unseen values
  become missing. High-cardinality columns (>~4000 levels) should be bucketed
  in SQL before being listed. Omit the key to keep the legacy float behaviour.

Constraints:
  - Do NOT write free-form training code, custom objectives, or ensembles.
  - Do NOT perform hyperparameter search loops — the environment validates one config per call.
  - Out-of-bounds config values are clamped automatically.
  - Omitted config fields use strong defaults.
  - The environment handles train/val splitting, fitting, and evaluation.

The tool returns: metrics, resolved model config, row counts after merges,
missingness rates, warnings, and residual correlations or best/worst prediction examples.
Use this feedback to iterate on your SQL features.
"""

_WRAPPED_MODEL_RULES_REG = r"""

=== WRAPPED MODEL MODE ===

Your validate_program() tool accepts:
  - feature_queries_json: SQL feature queries (same as before)
  - model_choice: one of the 7 learners below
  - model_config_json: optional JSON dict of hyperparameters (strong defaults apply)

Available models:
  1. "gbdt"     — Standard Gradient Boosted Trees. Fast, strong default.
                  Config: n_estimators (50-500), learning_rate (0.01-0.3), max_depth (2-10),
                          min_child_samples (1-100), subsample (0.5-1.0), colsample_bytree (0.5-1.0)
                  objective: "regression_l1" (MAE), "regression_l2" (MSE), "huber"
                  Default: "regression_l1" — directly minimises the evaluation metric.
  2. "rf"       — Random Forest (bagging of trees; less sensitive to learning rate).
                  Config: same keys as gbdt (including objective)
  3. "dart"     — DART Boosting (dropout regularization; can reduce overfitting).
                  Config: same keys as gbdt (including objective)
  4. "goss"     — GOSS (gradient-based subsampling; fast on large datasets).
                  Config: same keys as gbdt (including objective)
  5. "xgboost"  — XGBoost (second-order gradients; different regularization profile).
                  Config: n_estimators (50-500), learning_rate (0.01-0.3), max_depth (2-10),
                          min_child_weight (1-100), subsample (0.5-1.0), colsample_bytree (0.5-1.0)
                  objective: "reg:absoluteerror" (MAE), "reg:squarederror" (MSE), "reg:pseudohubererror" (Huber)
  6. "xgb_dart" — XGBoost + DART dropout.
                  Config: same keys as xgboost (including objective)
  7. "catboost" — CatBoost (ordered boosting; robust on heterogeneous tabular features).
                  Config: n_estimators (50-500), learning_rate (0.01-0.3), max_depth (2-10),
                          l2_leaf_reg (0.1-10.0)

  For skewed targets (counts, monetary values, long-tailed distributions): add
  "log_transform_target": true to model_config_json. The harness will fit on
  log1p(y) and report MAE back in the original scale. Compare with/without to decide.

Categorical features (all 7 learners):
  Add "categorical_features" inside model_config_json to have a column treated
  natively as categorical rather than silently integer-factorized to float.
  Format: a list of fully-qualified feature names "<query_name>__<col>" —
  the harness renames every column from a feature query "q" to "q__col" before
  handing it to the model. Example:
    {"objective": "regression_l1",
     "categorical_features": ["user_demo__country", "product__category"]}
  Per-family behaviour:
    - gbdt/rf/dart/goss: LightGBM's native categorical_feature at fit time.
    - xgboost/xgb_dart : XGBoost with enable_categorical=True.
    - catboost         : CatBoost cat_features (NaN replaced by a sentinel).
  Train-set categories are learned once and reused on val/test; unseen values
  become missing. High-cardinality columns (>~4000 levels) should be bucketed
  in SQL before being listed. Omit the key to keep the legacy float behaviour.

Constraints:
  - Do NOT write free-form training code, custom objectives, or ensembles.
  - Do NOT perform hyperparameter search loops — the environment validates one config per call.
  - Out-of-bounds config values are clamped automatically.
  - Omitted config fields use strong defaults.
  - The environment handles train/val splitting, fitting, and evaluation.

The tool returns: metrics, resolved model config, row counts after merges,
missingness rates, warnings, and residual correlations or best/worst prediction examples.
Use this feedback to iterate on your SQL features.
"""

# ---------------------------------------------------------------------------
# Execution prompts — sent as the first user message
# ---------------------------------------------------------------------------

SCIENTIST_EXECUTE_PROMPT_WRAPPED = r"""
Task: {task_type}
Task Description: {task_description}
Dataset: {dataset_name}

Entity column: {entity_col}
Target column: {target_col}
Validation set size: {n_val} entities

WRAPPED MODEL MODE: You propose SQL features + a model choice. The environment trains and evaluates.

Steps:
1. Run SHOW TABLES to see available tables
2. Run PRAGMA table_info('train_table') and inspect other relevant tables
3. Explore data distributions (SELECT COUNT(*), sample rows, etc.)
4. Design SQL feature queries that extract per-entity predictive signals
5. Call validate_program() with your features, model_choice, and optional config

Example call:
  validate_program(
    feature_queries_json='[{{"name": "basic_stats", "sql": "SELECT ..."}}]',
    model_choice="gbdt",
    model_config_json='{{}}'
  )

Iterate by improving SQL features based on diagnostics.
"""

SCIENTIST_EXECUTE_PROMPT_TIME_AWARE_WRAPPED = r"""
Task: {task_type}
Task Description: {task_description}
Dataset: {dataset_name}

Entity column: {entity_col}
Time column: {time_col}
Target column: {target_col}
Validation set size: {n_val} rows

WRAPPED MODEL MODE: You propose SQL features + a model choice. The environment trains and evaluates.
This is a time-aware row-level task. Build features per ({entity_col}, {time_col}) row.

Steps:
1. Run SHOW TABLES to see available tables
2. Run PRAGMA table_info('train_table') and inspect other relevant tables
3. Explore data distributions (SELECT COUNT(*), sample rows, etc.)
4. Design SQL features keyed by both {entity_col} and {time_col}
5. Call validate_program() with your features, model_choice, and optional config

Example call:
  validate_program(
    feature_queries_json='[{{"name": "basic_stats", "sql": "SELECT ..."}}]',
    model_choice="gbdt",
    model_config_json='{{}}'
  )

Iterate by improving SQL features based on diagnostics.
"""

# ---------------------------------------------------------------------------
# Eval workspace guidance
# ---------------------------------------------------------------------------

_EVAL_WORKSPACE_GUIDANCE = r"""

=== EVALUATION WORKSPACE ===
After each validate_program() call, the full evaluation output is persisted to
a queryable workspace. Use query_eval_workspace(sql) to analyse results.

Workspace tables:

  trials
    trial_id TEXT, trial_name TEXT, parent_trial_id TEXT,
    created_at TIMESTAMPTZ, split TEXT, model_choice TEXT,
    resolved_model_config TEXT, feature_query_hash TEXT,
    feature_block_names TEXT, primary_metric TEXT,
    primary_score DOUBLE, metrics_json TEXT, notes TEXT

  eval_predictions
    trial_id TEXT, row_id INTEGER, entity_id TEXT, label TEXT,
    score DOUBLE, predicted_class TEXT, split TEXT, eval_cutoff TIMESTAMPTZ

row_id is positionally stable across trials for the same split, so you can
join two trials on row_id to compare predictions on the same examples.

"""

# ---------------------------------------------------------------------------
# DuckDB cheatsheet
# ---------------------------------------------------------------------------

_DUCKDB_CHEATSHEET = r"""
=== DuckDB SQL Quick Reference ===

Date/time arithmetic (use these — NOT SQLite/MySQL equivalents):
  ts - INTERVAL '7 days'              -- subtract 7 days from timestamp
  ts + INTERVAL '30 days'             -- add 30 days
  DATE_DIFF('day', t1, t2)            -- integer days between t1 and t2
  DATE_TRUNC('month', ts)             -- truncate to month start
  EXTRACT(epoch FROM ts)              -- Unix epoch seconds (float)
  CURRENT_TIMESTAMP                   -- current time

WRONG — these do NOT exist in DuckDB:
  ts - 7  or  ts - 7.0               -- use INTERVAL instead
  julianday(ts)                       -- use DATE_DIFF or EXTRACT(epoch …)
  DATEADD(day, 7, ts)                 -- use ts + INTERVAL '7 days'
  DATE_SUB(ts, INTERVAL 7 DAY)        -- use ts - INTERVAL '7 days'
  DATEDIFF(day, t1, t2)               -- use DATE_DIFF('day', t1, t2)
  TIMESTAMP_NS - INTEGER              -- always use INTERVAL, never bare numbers

Type conversions:
  CAST(ts AS DATE)                    -- extract date part
  epoch_ms(ms_int)                    -- milliseconds integer → TIMESTAMP
  TO_TIMESTAMP(epoch_secs)            -- seconds integer → TIMESTAMP

Useful aggregation patterns:
  COUNT(*) FILTER (WHERE cond)        -- conditional count (DuckDB native)
  COALESCE(expr, 0)                   -- replace NULL with 0

IMPORTANT: Always run PRAGMA table_info('tablename') to verify exact column
names before referencing them. Column names differ across datasets — never
assume a column like 'id', 'type', 'date', or 'count' exists without checking.
"""

# ---------------------------------------------------------------------------
# Follow-up prompts — sent between turns
# ---------------------------------------------------------------------------

SCIENTIST_FOLLOWUP_PROMPT_WORKSPACE = r"""
Continue improving your pipeline.
- If you haven't validated yet, call validate_program() now.
- Full evaluation results are available in the workspace after each trial.
  Use query_eval_workspace() for error analysis
- Call get_trial_history() to see all past attempts.
- Use what you find in the workspace to guide your next SQL feature improvements.
"""

SCIENTIST_WRAPUP_PROMPT = r"""
This is your last turn. If you haven't submitted a validation yet, do so now.
Call validate_program() with your best SQL queries and model choice.
If you already have results, call get_trial_history() to confirm your best score.
"""


def get_scientist_system_prompt(
    task_type: str,
    entity_col: str = "entity_id",
    target_col: str = "target",
    time_col: Optional[str] = None,
    regression_objective: str = "mae",
    use_eval_workspace: bool = True,
    use_duckdb_cheatsheet: bool = False,
    database_path: str = "",
    benchmark_name: str = "RelBench",
) -> str:
    """Return the system prompt for the given task type."""
    if task_type == "entity_regression":
        template = SCIENTIST_SYSTEM_PROMPT_REGRESSION_WRAPPED
        model_rules = _WRAPPED_MODEL_RULES_REG
    else:
        template = SCIENTIST_SYSTEM_PROMPT_CLASSIFICATION_WRAPPED
        model_rules = _WRAPPED_MODEL_RULES_CLS

    tool_inventory = _format_tool_inventory(use_eval_workspace=use_eval_workspace)
    prompt = template.format(
        entity_col=entity_col,
        target_col=target_col,
        regression_objective=regression_objective.upper(),
        tool_inventory=tool_inventory,
        benchmark_name=benchmark_name,
    )

    if use_eval_workspace:
        prompt = prompt.replace(" + top-10 best/worst predictions.", ".")

    if time_col:
        prompt += "\n" + _TIME_AWARE_RULES.format(entity_col=entity_col, time_col=time_col)

    prompt += model_rules

    if use_eval_workspace:
        prompt += _EVAL_WORKSPACE_GUIDANCE

    if use_duckdb_cheatsheet:
        prompt += _DUCKDB_CHEATSHEET

    return prompt


def get_scientist_execute_prompt(
    task_type: str,
    task_description: str,
    dataset_name: str,
    entity_col: str,
    target_col: str,
    n_val: int,
    time_col: Optional[str] = None,
    use_eval_workspace: bool = True,
) -> str:
    """Return the execution prompt with task-specific details filled in."""
    footer = _execute_prompt_capability_footer(use_eval_workspace=use_eval_workspace)

    if time_col:
        body = SCIENTIST_EXECUTE_PROMPT_TIME_AWARE_WRAPPED.format(
            task_type=task_type,
            task_description=task_description,
            dataset_name=dataset_name,
            entity_col=entity_col,
            time_col=time_col,
            target_col=target_col,
            n_val=n_val,
        )
    else:
        body = SCIENTIST_EXECUTE_PROMPT_WRAPPED.format(
            task_type=task_type,
            task_description=task_description,
            dataset_name=dataset_name,
            entity_col=entity_col,
            target_col=target_col,
            n_val=n_val,
        )

    return body + footer
