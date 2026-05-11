"""CLI entry point for the RelAgent scientist agent."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import shlex

if __package__ in (None, "") and __name__ == "__main__":
    _src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)
    __package__ = "relagent"

from pathlib import Path as _Path

from .relbench_loader import RelBenchLoader
from .fourdbinfer_loader import FourDBInferLoader, fourdbinfer_cli_pair_registered
from .scientist import ScientistAgent
from .utils import setup_database_from_relbench, setup_logging

_DEFAULT_ARTIFACT_DIR = "artifacts/scientist_runs"


def _scratch_prefixes() -> tuple:
    """Return scratch path prefixes to watch for auto-deletion.

    Reads SCRATCH_DIR env var if set; always includes /scratch/ as a fallback.
    """
    prefixes = ["/scratch/"]
    custom = os.environ.get("SCRATCH_DIR", "").strip()
    if custom and custom not in prefixes:
        prefixes.append(custom.rstrip("/") + "/")
    return tuple(prefixes)


def _delete_scratch_db(db_path: str, logger=None) -> None:
    if not db_path:
        return
    p = _Path(db_path)
    if not any(str(p).startswith(pfx) for pfx in _scratch_prefixes()):
        return
    for candidate in [p, p.with_suffix(".duckdb.wal"), _Path(str(p) + ".wal")]:
        if candidate.exists():
            try:
                candidate.unlink()
                if logger:
                    logger.info(f"Deleted scratch DB file: {candidate}")
            except Exception as exc:
                if logger:
                    logger.warning(f"Could not delete {candidate}: {exc}")


def _infer_task_type(loader: RelBenchLoader) -> str:
    try:
        from relbench.tasks import TaskType
        tt = loader.task.task_type
        if tt == TaskType.BINARY_CLASSIFICATION or tt == TaskType.MULTICLASS_CLASSIFICATION:
            return "entity_classification"
        if tt == TaskType.REGRESSION:
            return "entity_regression"
        raise ValueError(f"Unsupported task type: {tt}")
    except ImportError:
        return "entity_classification"


def _resolve_entity_col(loader: RelBenchLoader) -> str:
    task = loader.task
    for attr in ["entity_col", "src_entity_col"]:
        col = getattr(task, attr, None)
        if col:
            return col
    return task.get_table("train").df.columns[0]


def _resolve_target_col(loader: RelBenchLoader) -> str:
    task = loader.task
    for attr in ["target_col", "dst_entity_col"]:
        col = getattr(task, attr, None)
        if col:
            return col
    return "target"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RelAgent scientist for a RelBench task")
    p.add_argument("--dataset", default="rel-amazon", help="RelBench dataset name (e.g. rel-amazon)")
    p.add_argument("--task", default="user-churn", help="RelBench task name (e.g. user-churn)")
    p.add_argument("--model", default="gpt-5.2", help="LiteLLM model string")
    p.add_argument("--max_turns", type=int, default=60, help="Max agent turns (default 60)")
    p.add_argument("--artifact_dir", default=_DEFAULT_ARTIFACT_DIR,
                   help=f"Output directory for run artifacts (default: {_DEFAULT_ARTIFACT_DIR})")
    p.add_argument("--temperature", type=float, default=1.0, help="LLM sampling temperature")
    p.add_argument("--max_tokens", type=int, default=8000, help="Max LLM output tokens")
    p.add_argument("--step_timeout", type=float, default=900.0,
                   help="Per-turn timeout in seconds (default 900)")
    p.add_argument("--sql_query_timeout", type=float, default=300.0,
                   help="Max seconds per DuckDB feature query (default 300)")
    p.add_argument("--api_base", default=None,
                   help="LiteLLM proxy base URL (or set LITELLM_API_BASE env var)")
    p.add_argument("--api_key", default=None,
                   help="LiteLLM API key (or set LITELLM_API_KEY env var)")
    p.add_argument("--eval_sample", type=int, default=None,
                   help="Sample N entities from val/test splits")
    p.add_argument("--fourdbinfer_data_dir", default=None,
                   help="Local cache directory for 4DBInfer datasets (default: ~/.dgl/)")
    p.add_argument("--port", type=int, default=8000,
                   help="vLLM server port (for locally-hosted models)")
    p.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING"], default="INFO",
                   help="Log level for the run file (console stays INFO)")
    return p.parse_args()


def _redact_secret(value: str) -> str:
    return "***REDACTED***" if value else value


def _build_redacted_cli(argv: list[str]) -> str:
    redacted: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--api_key="):
            redacted.append("--api_key=***REDACTED***")
            i += 1
            continue
        redacted.append(tok)
        if tok == "--api_key" and i + 1 < len(argv):
            redacted.append("***REDACTED***")
            i += 2
            continue
        i += 1
    return " ".join(shlex.quote(x) for x in redacted)


def main() -> int:
    args = parse_args()

    log_level = getattr(logging, args.log_level)
    logger = setup_logging(model_name=args.model, log_level=log_level)

    args_for_log = vars(args).copy()
    args_for_log["api_key"] = _redact_secret(args_for_log.get("api_key") or "")
    logger.info(f"CLI command: {_build_redacted_cli(sys.argv)}")
    logger.info(f"CLI args: {json.dumps(args_for_log, sort_keys=True, default=str)}")
    logger.info(f"Starting: dataset={args.dataset}, task={args.task}, model={args.model}")

    is_4dbinfer = fourdbinfer_cli_pair_registered(args.dataset, args.task)

    if is_4dbinfer:
        loader = FourDBInferLoader(
            dataset_name=args.dataset,
            task_name=args.task,
            data_dir=args.fourdbinfer_data_dir,
            logger=logger,
        )
        benchmark_name = "4DBInfer"
    else:
        loader = RelBenchLoader(dataset_name=args.dataset, task_name=args.task, logger=logger)
        benchmark_name = "RelBench"

    task_type = _infer_task_type(loader)
    entity_col = _resolve_entity_col(loader)
    target_col = _resolve_target_col(loader)

    if is_4dbinfer:
        try:
            from .task_descriptions import get_4dbinfer_task_description
            task_description = get_4dbinfer_task_description(args.dataset, args.task)
        except Exception:
            task_description = f"Binary classification task on {args.dataset}/{args.task}."
    else:
        try:
            from .task_descriptions import get_task_description
            task_description = get_task_description(args.dataset, args.task)
        except Exception:
            task_description = f"{task_type} task on {args.dataset}/{args.task}"

    extra_entity_cols = getattr(loader.task, "extra_entity_cols", [])
    time_col = getattr(loader.task, "time_col", None)
    require_asof_for_multi_entity = bool(time_col)
    asof_col = time_col if require_asof_for_multi_entity else None
    if require_asof_for_multi_entity:
        task_description += (
            " IMPORTANT: this task is time-aware. "
            "Return predictions keyed by both entity and as-of time."
        )

    logger.info(
        f"Task: type={task_type}, entity_col={entity_col}, target_col={target_col}, "
        f"benchmark={benchmark_name}"
    )

    db_path = setup_database_from_relbench(
        loader,
        reuse_existing=True,
        logger=logger,
        upto_timestamp=loader.get_val_timestamp(),
    )

    scientist = ScientistAgent(
        model_name=args.model,
        loader=loader,
        database_path=db_path,
        task_type=task_type,
        entity_col=entity_col,
        target_col=target_col,
        task_description=task_description,
        dataset_name=args.dataset,
        benchmark_name=benchmark_name,
        max_turns=args.max_turns,
        artifact_dir=args.artifact_dir,
        port=args.port,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        eval_sample=args.eval_sample,
        api_base=args.api_base,
        api_key=args.api_key,
        step_timeout=args.step_timeout,
        sql_query_timeout=args.sql_query_timeout,
        require_asof_for_multi_entity=require_asof_for_multi_entity,
        asof_col=asof_col,
        extra_entity_cols=extra_entity_cols,
        time_col=time_col,
        regression_primary_metric="mae",
        # --- hardcoded paper settings ---
        use_eval_workspace=True,
        seven_models=True,
    )

    try:
        result = scientist.run()
    finally:
        scientist.close()
        _delete_scratch_db(db_path, logger)

    print("\n" + "=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str))
    print(f"\nArtifacts saved to: {result['artifact_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
