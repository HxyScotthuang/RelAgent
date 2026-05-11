"""ScientistAgent — CAMEL ChatAgent with SQL + validation tools."""

from __future__ import annotations

import datetime as dt
import logging
import re
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.toolkits import FunctionTool, SQLToolkit
from camel.types import ModelPlatformType

from ..evaluator import RelBenchEvaluator
from ..models import ModelName
from ..utils import setup_database_from_relbench

from .eval_workspace import EvalWorkspace
from .prompts import (
    SCIENTIST_FOLLOWUP_PROMPT_WORKSPACE,
    SCIENTIST_WRAPUP_PROMPT,
    get_scientist_execute_prompt,
    get_scientist_system_prompt,
)
from .tools import (
    make_query_eval_workspace_tool,
    make_trial_history_tool,
    make_validate_program_wrapped_tool,
    _write_trial_to_workspace,
)
from .trial_logger import TrialLogger
from .validation import ProgramSpec, execute_and_validate

logger = logging.getLogger(__name__)


def _delete_scratch_db(db_path: str, logger=None) -> None:
    if not db_path:
        return
    p = Path(db_path)
    scratch_prefixes = tuple(
        {"/scratch/"} | (
            {os.environ["SCRATCH_DIR"].rstrip("/") + "/"}
            if os.environ.get("SCRATCH_DIR") else set()
        )
    )
    if not any(str(p).startswith(pfx) for pfx in scratch_prefixes):
        return
    for candidate in [p, p.with_suffix(".duckdb.wal"), Path(str(p) + ".wal")]:
        if candidate.exists():
            try:
                candidate.unlink()
                if logger:
                    logger.info(f"Deleted scratch DB file: {candidate}")
            except Exception as exc:
                if logger:
                    logger.warning(f"Could not delete {candidate}: {exc}")


class ScientistAgent:
    """LLM agent that discovers SQL features for RelBench tasks.

    Uses CAMEL ChatAgent with:
    - SQLToolkit for database exploration
    - validate_program tool for testing pipelines
    - get_trial_history tool for tracking experiments
    """

    MAX_TURNS = 30
    MAX_SQL_ROWS = 500
    DEFAULT_TOOL_OUTPUT_LIMIT = 5000

    def __init__(
        self,
        model_name: str,
        loader: Any,  # RelBenchLoader
        database_path: str,
        task_type: str,
        entity_col: str,
        target_col: str,
        task_description: str = "",
        dataset_name: str = "",
        max_turns: int = MAX_TURNS,
        artifact_dir: str = "artifacts/scientist_runs",
        port: int = 8000,
        temperature: float = 0.7,
        max_tokens: int = 8000,
        tool_output_limit: int = DEFAULT_TOOL_OUTPUT_LIMIT,
        eval_sample: Optional[int] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        step_timeout: float = 300.0,
        sql_query_timeout: float = 300.0,
        show_coverage_feedback: bool = False,
        require_asof_for_multi_entity: bool = False,
        asof_col: Optional[str] = None,
        extra_entity_cols: Optional[List[str]] = None,
        time_col: Optional[str] = None,
        regression_primary_metric: str = "mae",
        use_eval_workspace: bool = True,
        use_duckdb_cheatsheet: bool = False,
        seven_models: bool = True,
        benchmark_name: str = "RelBench",
    ):
        self.loader = loader
        self.task_type = task_type
        self.entity_col = entity_col
        self.target_col = target_col
        self.task_description = task_description
        self.dataset_name = dataset_name
        # Per-task train entity cap for runs with very large train sets.
        # Without a cap, LightGBM training can take 10-17 min per trial (too slow for vLLM-hosted
        # models with short step timeouts). Cap targets ~500K training rows per trial.
        _TRAIN_ENTITY_CAPS = {
            ("rel-amazon", "user-ltv"): 50_000,
            ("rel-hm", "item-sales"): 10_000,
        }
        self.max_train_entities: Optional[int] = _TRAIN_ENTITY_CAPS.get(
            (dataset_name, loader.task_name)
        )
        self.max_turns = max_turns
        self.eval_sample = eval_sample
        self.tool_output_limit = tool_output_limit
        self.step_timeout = float(step_timeout)
        self.sql_query_timeout = float(sql_query_timeout)
        self.show_coverage_feedback = bool(show_coverage_feedback)
        self.require_asof_for_multi_entity = bool(require_asof_for_multi_entity)
        self.asof_col = asof_col
        self.extra_entity_cols: List[str] = extra_entity_cols or []
        self.time_col = time_col
        self.regression_primary_metric = regression_primary_metric
        self.use_eval_workspace = bool(use_eval_workspace)
        self.use_duckdb_cheatsheet = bool(use_duckdb_cheatsheet)
        self.seven_models = bool(seven_models)
        self.benchmark_name = benchmark_name
        self._database_path = database_path
        self._model_name_str = model_name
        self._port = port
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_base = api_base
        self._api_key = api_key

        # Set up artifact directory
        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        run_name = f"{dataset_name}_{loader.task_name}_{ts}"
        self.run_dir = Path(artifact_dir) / run_name
        self.trial_logger = TrialLogger(self.run_dir)
        self.eval_workspace: Optional[EvalWorkspace] = None
        if self.use_eval_workspace:
            self.eval_workspace = EvalWorkspace(self.run_dir / "eval_workspace.db")

        # Load validation data
        val_table = loader.get_table("val", eval_sample=eval_sample)
        self.val_df = val_table.df.copy()
        train_table = loader.get_table("train", eval_sample=eval_sample)
        self.train_df = train_table.df.copy()

        # Set up DuckDB connection
        self.conn = duckdb.connect(database_path, read_only=True)

        # Set up evaluator
        self.evaluator = RelBenchEvaluator(
            loader=loader,
            logger=logger,
            eval_sample=eval_sample,
            task_type=task_type,
        )

        # Build tools
        tools = self._build_tools(database_path)

        # Create model
        model = self._create_model(model_name, port, temperature, max_tokens, api_base, api_key)
        self._model_backend = model

        # Create CAMEL ChatAgent
        system_prompt = get_scientist_system_prompt(
            task_type=task_type,
            entity_col=entity_col,
            target_col=target_col,
            time_col=time_col,
            regression_objective=regression_primary_metric,
            use_eval_workspace=self.use_eval_workspace,
            use_duckdb_cheatsheet=self.use_duckdb_cheatsheet,
            database_path=database_path,
            benchmark_name=self.benchmark_name,
        )

        self.agent = ChatAgent(
            model=model,
            system_message=system_prompt,
            tools=tools,
            summarize_threshold=None,
            mask_tool_output=False,
            prune_tool_calls_from_memory=False,
            agent_id=str(uuid.uuid4()),
        )
        self.agent.step_timeout = self.step_timeout
        logger.info(f"Scientist agent step timeout set to {self.step_timeout:.1f}s")
        logger.info(f"Per SQL feature-query time limit: {self.sql_query_timeout:.1f}s")

        # Track state
        self._best_score = float("-inf")
        self._best_program: Optional[ProgramSpec] = None
        self._trace_last_count = 0

    def _safe_trace_obj(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
            try:
                return obj.to_dict()
            except Exception:
                pass
        if isinstance(obj, dict):
            return {str(k): self._safe_trace_obj(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._safe_trace_obj(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    def _trace_new_memory_events(self, turn: int) -> None:
        try:
            if not hasattr(self.agent, "memory") or self.agent.memory is None:
                return
            records = self.agent.memory.retrieve()
        except Exception:
            return

        total = len(records)
        if total <= self._trace_last_count:
            return

        for idx in range(self._trace_last_count, total):
            try:
                rec = records[idx]
                msg = rec.memory_record.message
                md = msg.to_dict() if hasattr(msg, "to_dict") else {}
                role = md.get("role_name") or md.get("role_type") or "unknown"
                tool_name = md.get("func_name")
                tool_call_id = md.get("tool_call_id")
                content = md.get("content")
                args = md.get("args")
                result = md.get("result")
                if tool_name:
                    if args is not None:
                        event = {
                            "event_type": "tool_call",
                            "turn": turn,
                            "memory_index": idx,
                            "role": role,
                            "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                            "args": self._safe_trace_obj(args),
                        }
                    else:
                        event = {
                            "event_type": "tool_result",
                            "turn": turn,
                            "memory_index": idx,
                            "role": role,
                            "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                            "result": self._safe_trace_obj(
                                result if result is not None else content
                            ),
                        }
                else:
                    event = {
                        "event_type": "message",
                        "turn": turn,
                        "memory_index": idx,
                        "role": role,
                        "content": self._safe_trace_obj(content),
                    }
                self.trial_logger.log_trace_event(event)
            except Exception as e:
                self.trial_logger.log_trace_event(
                    {
                        "event_type": "trace_error",
                        "turn": turn,
                        "memory_index": idx,
                        "error": str(e),
                    }
                )

        self._trace_last_count = total

    def _create_model(
        self,
        model_name: str,
        port: int,
        temperature: float,
        max_tokens: int,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Any:
        import os

        try:
            model_enum = ModelName.from_string(model_name)
            if model_enum.is_vllm_model():
                vllm_name = model_enum.get_vllm_model_name()
                model = ModelFactory.create(
                    model_platform=ModelPlatformType.VLLM,
                    model_type=vllm_name,
                    url=f"http://localhost:{port}/v1",
                    model_config_dict={"temperature": temperature, "max_tokens": max_tokens},
                )
                logger.info(f"Initialized vLLM model via CAMEL ModelFactory: {model_enum.value} -> {vllm_name} on port {port}")
                return model
        except ValueError:
            pass

        if api_key is None:
            api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_base is None:
            api_base = os.environ.get("LITELLM_API_BASE") or os.environ.get("LITELLM_BASE_URL")

        try:
            import litellm  # type: ignore

            litellm.num_retries = 2
            _req_timeout = int(os.environ.get("LITELLM_REQUEST_TIMEOUT", "600"))
            litellm.request_timeout = _req_timeout
            _debug_env = (
                os.environ.get("LITELLM_LOG", "").upper() == "DEBUG"
                or os.environ.get("LITELLM_DEBUG", "") in ("1", "true", "TRUE")
            )
            if _debug_env:
                try:
                    litellm._turn_on_debug()
                except AttributeError:
                    litellm.set_verbose = True
                logger.info("LiteLLM debug logging ENABLED via env (LITELLM_LOG/LITELLM_DEBUG)")
            logger.info(f"LiteLLM globals: num_retries={litellm.num_retries}, request_timeout={litellm.request_timeout}s, debug={_debug_env}")
        except ImportError:
            pass

        try:
            create_kwargs = {
                "model_platform": ModelPlatformType.LITELLM,
                "model_type": model_name,
                "model_config_dict": {"temperature": temperature, "max_tokens": max_tokens},
            }
            if api_base:
                create_kwargs["url"] = api_base
            if api_key:
                create_kwargs["api_key"] = api_key
            model = ModelFactory.create(**create_kwargs)
            logger.info(f"Initialized model via LiteLLM: {model_name} (base={api_base})")
            return model
        except Exception as e:
            logger.info(f"LiteLLM failed for '{model_name}': {e}. Trying ModelFactory fallback.")

        try:
            model_enum = ModelName.from_string(model_name)
        except ValueError:
            model = ModelFactory.create(
                model_platform=ModelPlatformType.VLLM,
                model_type=model_name,
                url=f"http://localhost:{port}/v1",
                model_config_dict={"temperature": temperature, "max_tokens": max_tokens},
            )
            logger.info(f"Initialized custom vLLM model: {model_name} on port {port}")
            return model

        if model_enum.is_openai_model():
            platform, camel_type, _ = model_enum.get_camel_model_type()
            model = ModelFactory.create(
                model_platform=platform,
                model_type=camel_type,
                model_config_dict={"temperature": temperature, "max_tokens": max_tokens},
            )
            logger.info(f"Initialized OpenAI model: {model_enum.value}")
        else:
            raise ValueError(f"Cannot initialize model: {model_name}")

        return model

    def _build_tools(self, database_path: str) -> List[Any]:
        tools: List[Any] = []

        sql_toolkit = SQLToolkit(
            database_path=database_path,
            database_type="duckdb",
            read_only=True,
        )
        for tool in sql_toolkit.get_tools():
            tools.append(self._wrap_sql_tool(tool))

        validate_tool = make_validate_program_wrapped_tool(
            conn=self.conn,
            train_df=self.train_df,
            val_df=self.val_df,
            evaluator=self.evaluator,
            task_type=self.task_type,
            entity_col=self.entity_col,
            target_col=self.target_col,
            trial_logger=self.trial_logger,
            tool_name="validate_program",
            split="val",
            show_coverage_feedback=self.show_coverage_feedback,
            require_asof_for_multi_entity=self.require_asof_for_multi_entity,
            asof_col=self.asof_col,
            extra_entity_cols=self.extra_entity_cols,
            regression_primary_metric=self.regression_primary_metric,
            sql_timeout_seconds=self.sql_query_timeout,
            eval_workspace=self.eval_workspace,
            max_train_entities=self.max_train_entities,
            seven_models=self.seven_models,
        )
        tools.append(validate_tool)

        history_tool = make_trial_history_tool(self.trial_logger)
        tools.append(history_tool)

        if self.eval_workspace is not None:
            tools.append(make_query_eval_workspace_tool(self.eval_workspace))

        return tools

    def _wrap_sql_tool(self, tool: FunctionTool) -> FunctionTool:
        original_func = tool.func
        tool_name = tool.get_function_name() if hasattr(tool, "get_function_name") else ""
        is_execute_query = tool_name == "execute_query"
        max_rows = self.MAX_SQL_ROWS
        size_limit = self.tool_output_limit

        def wrapped_func(*args, **kwargs):
            call_args = args

            if is_execute_query:
                query = call_args[0] if call_args else kwargs.get("query", "")
                if isinstance(query, str) and query.strip():
                    q_stripped = query.strip()
                    q_upper = q_stripped.upper()
                    is_select = q_upper.startswith("SELECT") or q_upper.startswith("WITH")
                    has_limit = bool(re.search(r"\bLIMIT\s+\d+", q_stripped, re.IGNORECASE))
                    if is_select and not has_limit:
                        modified = q_stripped.rstrip(";") + f" LIMIT {max_rows}"
                        if call_args:
                            call_args = (modified,) + call_args[1:]
                        else:
                            kwargs["query"] = modified

            result = original_func(*call_args, **kwargs)

            try:
                if isinstance(result, pd.DataFrame):
                    result = result.to_dict("records")
                elif isinstance(result, pd.Series):
                    result = result.tolist()
            except Exception:
                pass

            if result is not None:
                result_str = str(result)
                if len(result_str) > size_limit:
                    result = result_str[:size_limit] + "..."

            return result

        tool.func = wrapped_func
        return tool

    def _update_best_program(self) -> None:
        for trial in self.trial_logger.trials:
            if trial["error"] is None and trial["primary_score"] > self._best_score:
                self._best_score = trial["primary_score"]
                self._best_program = ProgramSpec(
                    feature_queries=trial["feature_queries"],
                    model_choice=trial.get("model_choice"),
                    model_config=trial.get("model_config"),
                )
                try:
                    queries = "\n\n".join(
                        [f"-- {q['name']}\n{q['sql'].strip()}" for q in self._best_program.feature_queries]
                    )
                except Exception:
                    queries = "(unavailable)"
                logger.info(
                    "NEW BEST PROGRAM\n"
                    "================\n"
                    f"score={self._best_score:.6f}\n\n"
                    "FEATURE QUERIES:\n"
                    f"{queries}\n"
                )

    def _agent_step_content(self, prompt: str, turn: int, max_attempts: int = 2) -> str:
        last_error: Optional[str] = None

        for attempt in range(1, max_attempts + 1):
            agent_response = self.agent.step(
                BaseMessage.make_user_message(content=prompt, role_name="User")
            )
            self._trace_new_memory_events(turn=turn)

            msgs = getattr(agent_response, "msgs", None) or []
            if msgs:
                if len(msgs) > 1:
                    logger.warning(
                        "Turn %s attempt %s returned %s messages; using the first.",
                        turn,
                        attempt,
                        len(msgs),
                    )
                    try:
                        self.agent.record_message(msgs[0])
                    except Exception as exc:
                        logger.warning("Could not record selected CAMEL message: %s", exc)
                return msgs[0].content or ""

            info = self._safe_trace_obj(getattr(agent_response, "info", {}))
            terminated = bool(getattr(agent_response, "terminated", False))
            last_error = (
                f"CAMEL returned no messages on turn {turn} attempt {attempt}; "
                f"terminated={terminated}; info={info}"
            )
            logger.warning(last_error)
            self.trial_logger.log_trace_event(
                {
                    "event_type": "empty_agent_response",
                    "turn": turn,
                    "attempt": attempt,
                    "terminated": terminated,
                    "info": info,
                }
            )
            if terminated:
                break

        raise RuntimeError(last_error or f"CAMEL returned no messages on turn {turn}")

    def _run_agent_loop(self, execute_prompt: str, max_turns: int) -> None:
        response = self._agent_step_content(execute_prompt, turn=0)
        logger.info(f"Turn 0 response: {response[:500] if response else '(empty)'}")
        self.trial_logger.log_trace_event(
            {"event_type": "turn_response", "turn": 0, "content": response}
        )

        for turn in range(1, max_turns):
            n_trials = len(self.trial_logger.trials)
            self._update_best_program()

            followup = SCIENTIST_WRAPUP_PROMPT if turn >= max_turns - 1 else SCIENTIST_FOLLOWUP_PROMPT_WORKSPACE

            try:
                response = self._agent_step_content(followup, turn=turn)
            except Exception as e:
                logger.error(f"Turn {turn} failed: {e}")
                self.trial_logger.log_trace_event(
                    {"event_type": "turn_error", "turn": turn, "error": str(e)}
                )
                continue

            logger.info(f"Turn {turn} response: {response[:500] if response else '(empty)'}")
            self.trial_logger.log_trace_event(
                {"event_type": "turn_response", "turn": turn, "content": response}
            )

            if response and n_trials > 0:
                done_signals = ["i'm done", "final answer", "best program found", "no further improvement"]
                if any(s in response.lower() for s in done_signals):
                    logger.info(f"Agent signaled completion at turn {turn}")
                    break

    def run(self) -> Dict[str, Any]:
        logger.info(
            f"Starting scientist run: task={self.task_type}, "
            f"entity_col={self.entity_col}, target_col={self.target_col}, "
            f"max_turns={self.max_turns}, val_size={len(self.val_df)}"
        )

        execute_prompt = get_scientist_execute_prompt(
            task_type=self.task_type,
            task_description=self.task_description,
            dataset_name=self.dataset_name,
            entity_col=self.entity_col,
            target_col=self.target_col,
            n_val=len(self.val_df),
            time_col=self.time_col,
            use_eval_workspace=self.use_eval_workspace,
        )

        loop_error: Optional[BaseException] = None
        try:
            self._run_agent_loop(execute_prompt, self.max_turns)
            self._update_best_program()
        except Exception as _loop_exc:
            loop_error = _loop_exc
            logger.error(
                "Agent loop raised %s: %s. Attempting best-effort save of "
                "partial results before re-raising.",
                type(_loop_exc).__name__,
                _loop_exc,
            )
            try:
                self.trial_logger.log_trace_event(
                    {
                        "event_type": "agent_loop_error",
                        "error_type": type(_loop_exc).__name__,
                        "error": str(_loop_exc),
                    }
                )
            except Exception:
                pass
            try:
                self._update_best_program()
            except Exception as _upd_exc:
                logger.warning(
                    "_update_best_program during crash cleanup failed: %s", _upd_exc
                )

        if self._best_program:
            try:
                self.trial_logger.save_best_program(self._best_program)
            except Exception as _exc:
                logger.warning("save_best_program failed: %s", _exc)

        n_total_trials = len(self.trial_logger.trials)
        best_trial_id = 0
        try:
            for t in self.trial_logger.trials:
                if t["error"] is None and abs(t["primary_score"] - self._best_score) < 1e-9:
                    best_trial_id = t["trial_id"]
                    break
        except Exception as _exc:
            logger.warning("best_trial_id lookup failed: %s", _exc)

        try:
            self.trial_logger.save_report(
                best_score=self._best_score,
                best_trial_id=best_trial_id,
                n_trials=n_total_trials,
            )
        except Exception as _exc:
            logger.warning("save_report failed: %s", _exc)

        result: Dict[str, Any] = {
            "best_score": self._best_score,
            "best_trial_id": best_trial_id,
            "n_trials": n_total_trials,
            "artifact_dir": str(self.run_dir),
        }
        if loop_error is not None:
            result["crashed"] = True
            result["crash_error"] = f"{type(loop_error).__name__}: {loop_error}"

        try:
            final_val_results = self._evaluate_final_validation()
            if final_val_results is not None:
                result["final_validation_results"] = final_val_results
        except Exception as _exc:
            logger.warning("_evaluate_final_validation failed: %s", _exc)

        try:
            test_results = self._evaluate_test_set()
            if test_results is not None:
                result["test_results"] = test_results
        except Exception as _exc:
            logger.warning("_evaluate_test_set failed: %s", _exc)

        if loop_error is not None:
            raise loop_error

        return result

    def _log_shap_summary(self, split: str, shap_rows: List[Dict[str, Any]], note: Optional[str]) -> None:
        if note:
            logger.info(f"{split} SHAP note: {note}")
        if not shap_rows:
            logger.info(f"{split} SHAP top-30: unavailable")
            return
        logger.info(f"{split} SHAP top-{min(30, len(shap_rows))} features (mean |SHAP|):")
        for row in shap_rows[:30]:
            logger.info(
                f"  #{row.get('rank', '?'):>2}  {row.get('feature', '')}: "
                f"{float(row.get('mean_abs_shap', 0.0)):.6f}"
            )

    def _evaluate_final_validation(self) -> Optional[Dict[str, Any]]:
        if self._best_program is None:
            logger.warning("No successful trials — skipping final validation rerun")
            return None

        logger.info("Running final validation rerun with best program...")
        try:
            final_val_evaluator = RelBenchEvaluator(
                loader=self.loader,
                logger=logger,
                eval_sample=self.eval_sample,
                task_type=self.task_type,
            )
            result = execute_and_validate(
                program=self._best_program,
                conn=self.conn,
                val_df=self.val_df.copy(),
                evaluator=final_val_evaluator,
                task_type=self.task_type,
                entity_col=self.entity_col,
                target_col=self.target_col,
                trial_id=-2,
                split="val",
                require_asof_for_multi_entity=self.require_asof_for_multi_entity,
                asof_col=self.asof_col,
                extra_entity_cols=self.extra_entity_cols,
                regression_primary_metric=self.regression_primary_metric,
                compute_shap=True,
                train_df=self.train_df,
                sql_timeout_seconds=self.sql_query_timeout,
                store_eval_preds=(self.eval_workspace is not None),
                is_final=True,
                max_train_entities=self.max_train_entities,
                seven_models=self.seven_models,
            )
            self.trial_logger.log_trial(self._best_program, result, is_best=True, split="final_val")
            if self.eval_workspace is not None:
                try:
                    _write_trial_to_workspace(
                        self.eval_workspace, self._best_program, result, "final_val",
                        entity_col=self.entity_col,
                    )
                except Exception as _ws_err:
                    logger.warning(f"Workspace write failed for final_val: {_ws_err}")
            self._log_shap_summary("Final validation", result.shap_importance, result.shap_note)
            return {
                "metrics": result.metrics,
                "primary_score": result.score,
                "shap_importance_top30": result.shap_importance[:30],
                "shap_note": result.shap_note,
            }
        except Exception as e:
            logger.error(f"Final validation rerun failed: {e}")
            return None

    def _evaluate_test_set(self) -> Optional[Dict[str, Any]]:
        if self._best_program is None:
            logger.warning("No successful trials — skipping test evaluation")
            return None

        logger.info("Evaluating best program on test set...")

        test_db_path: Optional[str] = None
        test_conn = None
        try:
            # Use an uncensored DB for test evaluation so test-period rows have
            # same-day events available via per-row SQL temporal filters,
            # matching RDBLearn's per-row cutoff behavior.
            test_db_path = setup_database_from_relbench(
                self.loader,
                reuse_existing=False,
                logger=logger,
                upto_timestamp=None,
            )
            test_conn = duckdb.connect(test_db_path, read_only=True)

            test_table = self.loader.get_table("test", eval_sample=self.eval_sample)
            test_df = test_table.df.copy()

            test_evaluator = RelBenchEvaluator(
                loader=self.loader,
                logger=logger,
                eval_sample=self.eval_sample,
                task_type=self.task_type,
            )

            result = execute_and_validate(
                program=self._best_program,
                conn=test_conn,
                val_df=test_df,
                evaluator=test_evaluator,
                task_type=self.task_type,
                entity_col=self.entity_col,
                target_col=self.target_col,
                trial_id=-1,
                split="test",
                require_asof_for_multi_entity=self.require_asof_for_multi_entity,
                asof_col=self.asof_col,
                extra_entity_cols=self.extra_entity_cols,
                regression_primary_metric=self.regression_primary_metric,
                compute_shap=True,
                train_df=self.train_df,
                sql_timeout_seconds=self.sql_query_timeout,
                store_eval_preds=(self.eval_workspace is not None),
                is_final=True,
                max_train_entities=self.max_train_entities,
                seven_models=self.seven_models,
            )

            self.trial_logger.log_trial(
                self._best_program, result, is_best=True, split="test"
            )
            self.trial_logger.save_test_results(result, self._best_program)
            if self.eval_workspace is not None:
                try:
                    _write_trial_to_workspace(
                        self.eval_workspace, self._best_program, result, "test",
                        entity_col=self.entity_col,
                    )
                except Exception as _ws_err:
                    logger.warning(f"Workspace write failed for test: {_ws_err}")

            logger.info(f"Test set results: {result.metrics}")
            self._log_shap_summary("Test", result.shap_importance, result.shap_note)
            return {
                "metrics": result.metrics,
                "primary_score": result.score,
                "shap_importance_top30": result.shap_importance[:30],
                "shap_note": result.shap_note,
            }

        except Exception as e:
            logger.error(f"Test set evaluation failed: {e}")
            return None
        finally:
            if test_conn is not None:
                try:
                    test_conn.close()
                except Exception:
                    pass
            if test_db_path is not None:
                _delete_scratch_db(test_db_path, logger)

    def close(self):
        if hasattr(self, "conn") and self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
        if hasattr(self, "eval_workspace") and self.eval_workspace is not None:
            try:
                self.eval_workspace.close()
            except Exception:
                pass
