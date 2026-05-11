"""
Utility functions for RelAgent.

This module provides utility functions for configuration loading, logging setup,
database initialization, vLLM compatibility fixes, and other helper functions
used throughout RelAgent.
"""
import os
import time
import uuid
import json
import threading
import numpy as np
import pandas as pd
try:
    from colorama import Fore, Style, Back
except ImportError:
    class Fore:
        GREEN = RED = BLACK = RESET = ""
    class Style:
        RESET_ALL = ""
    class Back:
        WHITE = ""
import logging
import datetime
import yaml
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict
import sys


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Optional path to config.yaml. If None, looks in current directory.
        
    Returns:
        dict: Configuration dictionary with normalized paths.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        # Return default values if config file doesn't exist
        return {
            "db_path": "artifacts/rel_amazon.duckdb",
            "model_name": "gpt-5.2",
            "llm_backend": "openai_camel",
            "max_limit": 200,
            "batch_size": 32,
            "artifact_dir": "artifacts",
            "run_log_dir": "artifacts/logs"
        }

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Path normalization and directory creation (from old config.py)
    artifact_dir = Path(config.get('artifact_dir', 'artifacts'))
    run_log_dir = Path(config.get('run_log_dir', 'artifacts/logs'))
    
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir.mkdir(parents=True, exist_ok=True)
    
    config['artifact_dir'] = str(artifact_dir.resolve())
    config['run_log_dir'] = str(run_log_dir.resolve())
    config['db_path'] = str(Path(config.get('db_path', 'artifacts/rel_amazon.duckdb')).resolve())
    
    return config


def setup_logging(model_name, log_file_path=None, log_level=logging.INFO):
    """
    Set up logging configuration with a file handler based on model name or custom log path.
    
    Args:
        model_name: Name of the model to use in the log directory
        log_file_path: Optional custom path for the log file. If provided, this will be used
                       instead of generating a path based on model_name.
        log_level: Logging level (default: logging.INFO). Use logging.DEBUG for detailed logs.
    
    Returns:
        logger: Configured logger instance
    """
    # Create a filter to suppress harmless vLLM warnings about 'strict' field
    class SuppressStrictFieldWarning(logging.Filter):
        """Filter to suppress harmless warnings about 'strict' field in vLLM requests."""
        def filter(self, record):
            message = record.getMessage()
            # Suppress warnings about 'strict' field being ignored (compatibility issue)
            if "The following fields were present in the request but ignored: {'strict'}" in message:
                return False
            return True
    
    # Create a logger
    logger = logging.getLogger("relagent")
    logger.setLevel(log_level)
    # Avoid duplicate logs if setup_logging is called multiple times in the same process.
    logger.propagate = False
    
    # Close and clear existing handlers to prevent handler buildup and memory leaks
    # This is critical when setup_logging is called multiple times across experiment runs
    if logger.handlers:
        for handler in logger.handlers[:]:  # Copy list to avoid modification during iteration
            try:
                handler.close()  # Properly close file handlers to release resources
            except Exception:
                pass  # Ignore errors when closing handlers
        logger.handlers.clear()
    
    # Create formatter - use more detailed format for DEBUG level
    if log_level == logging.DEBUG:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
    else:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create filter instance
    strict_filter = SuppressStrictFieldWarning()
    
    # Create console handler - always INFO for readability
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(strict_filter)
    logger.addHandler(console_handler)
    
    # Determine log file path
    if log_file_path:
        # Use the provided custom log file path
        # Ensure the directory exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_file = log_file_path
    elif model_name:
        # Create a log file based on model name
        log_dir = os.path.join("logs", model_name)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a timestamp for the log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"run_{timestamp}.log")
    else:
        # No logging to file if neither is provided
        return logger
    
    # Create file handler - use the specified log level
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(strict_filter)
    logger.addHandler(file_handler)
    
    level_name = logging.getLevelName(log_level)
    logger.info(f"Logging to file: {log_file} (level: {level_name})")
    
    return logger

def get_git_commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"




class SampledTable:
    """
    Wrapper for a sampled RelBench Table that preserves the Table interface.
    
    This class wraps a sampled dataframe and original table to maintain compatibility
    with RelBench's evaluation API while working with a subset of rows.
    
    Implements the same interface as ``relbench.base.Table`` so that
    ``task.evaluate(pred, target_table)`` works transparently.
    """
    def __init__(self, df, original_table, sampled_indices, original_size):
        """
        Initialize a sampled table wrapper.
        
        Args:
            df: Sampled pandas DataFrame
            original_table: Original RelBench Table object
            sampled_indices: List of original indices that were sampled
            original_size: Original size of the table before sampling
        """
        self.df = df
        self._original_table = original_table
        self._original_indices = sampled_indices
        self._is_sampled = True
        self._original_size = original_size
        # Copy Table-interface attributes required by RelBench
        for attr in ['fkey_col_to_pkey_table', 'pkey_col', 'time_col',
                      'name', 'entity_col', 'target_col',
                      'src_entity_col', 'dst_entity_col']:
            if hasattr(original_table, attr):
                setattr(self, attr, getattr(original_table, attr))

    def __len__(self) -> int:
        """Return the number of rows in the sampled table."""
        return len(self.df)

    def __repr__(self) -> str:
        return (
            f"SampledTable(rows={len(self.df)}, "
            f"original_size={self._original_size}, "
            f"sampled={self._is_sampled})"
        )


class StatisticsTracker:
    """Track and report task-type-aware evaluation statistics.

    Reports ALL available metrics per task type:
      - entity_classification: AUROC, Average Precision, F1, Accuracy
      - entity_regression: MAE, RMSE, R²
    """

    def __init__(self, logger: Optional[logging.Logger] = None, task_type: Optional[str] = None):
        self.correct = 0
        self.incorrect = 0
        self.skipped = 0
        self.total = 0
        self.task_type = task_type
        self.logger = logger or logging.getLogger(__name__)
        # Store raw predictions and ground truths for metric computation
        self._predictions: list = []
        self._ground_truths: list = []
    
    def record_result(self, is_correct: bool, skipped: bool = False,
                      prediction=None, ground_truth=None):
        """Record a single problem result with raw values for metric computation."""
        if skipped:
            self.skipped += 1
        else:
            self.total += 1
            if is_correct:
                self.correct += 1
            else:
                self.incorrect += 1
            # Store raw values for running metric computation
            if prediction is not None and ground_truth is not None:
                self._predictions.append(prediction)
                self._ground_truths.append(ground_truth)
    
    @property
    def accuracy(self) -> float:
        """Calculate current accuracy percentage (for classification)."""
        total = self.correct + self.incorrect
        return (self.correct / total * 100) if total > 0 else 0.0
    
    # ---------- Classification metrics ----------
    def _compute_classification_metrics(self) -> Dict:
        """Compute binary classification metrics: AUROC, AP, F1, Accuracy."""
        import numpy as np
        metrics = {}
        if len(self._predictions) < 2 or len(self._ground_truths) < 2:
            return metrics

        y_true_list, y_pred_list = [], []
        raw_correct = 0
        for pred, gt in zip(self._predictions, self._ground_truths):
            try:
                gt_mapped = float(gt)
            except (ValueError, TypeError):
                continue
            try:
                pred_mapped = float(pred)
            except (ValueError, TypeError):
                pred_mapped = 0.0
            y_true_list.append(gt_mapped)
            y_pred_list.append(pred_mapped)
            if abs(gt_mapped - pred_mapped) < 1e-9:
                raw_correct += 1

        if not y_true_list:
            return metrics
        metrics['accuracy'] = raw_correct / len(y_true_list)
        if len(y_true_list) < 2:
            return metrics

        y_true_f = np.array(y_true_list)
        y_pred_f = np.array(y_pred_list)
        unique_classes = np.unique(y_true_f)
        if len(unique_classes) >= 2:
            from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true_f, y_pred_f))
            except Exception as e:
                self.logger.debug(f"Error computing AUROC: {e}")
            try:
                metrics['average_precision'] = float(average_precision_score(y_true_f, y_pred_f))
            except Exception as e:
                self.logger.debug(f"Error computing Average Precision: {e}")
            try:
                metrics['f1'] = float(f1_score(y_true_f, y_pred_f, zero_division=0))
            except Exception as e:
                self.logger.debug(f"Error computing F1: {e}")

        return metrics
    
    # ---------- Regression metrics ----------
    def _compute_regression_metrics(self) -> Dict:
        """Compute all regression metrics: MAE, RMSE, R²."""
        import numpy as np
        metrics = {}
        if len(self._predictions) == 0:
            return metrics
        try:
            y_true = np.array(self._ground_truths, dtype=float)
            y_pred = np.array(self._predictions, dtype=float)
            
            residuals = y_true - y_pred
            metrics['mae'] = float(np.mean(np.abs(residuals)))
            metrics['rmse'] = float(np.sqrt(np.mean(residuals ** 2)))
            
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics['r2'] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        except Exception:
            pass
        return metrics
    
    def compute_all_metrics(self) -> Dict:
        """Compute all available running metrics for the current task type."""
        if self.task_type == "entity_classification":
            return self._compute_classification_metrics()
        elif self.task_type == "entity_regression":
            return self._compute_regression_metrics()
        else:
            return {}
    
    @property
    def primary_metric_name(self) -> str:
        """Return the name of the primary metric for this task type."""
        if self.task_type == "entity_regression":
            return "MAE"
        return "AUROC"
    
    def _format_metric(self) -> str:
        """Format a concise metric summary for progress bar display."""
        metrics = self.compute_all_metrics()
        if self.task_type == "entity_classification":
            parts = []
            if 'roc_auc' in metrics:
                parts.append(f"AUROC={metrics['roc_auc']:.4f}")
            if 'average_precision' in metrics:
                parts.append(f"AP={metrics['average_precision']:.4f}")
            if 'f1' in metrics:
                parts.append(f"F1={metrics['f1']:.4f}")
            if 'accuracy' in metrics:
                parts.append(f"Acc={metrics['accuracy']:.1%}")
            return " | ".join(parts) if parts else f"Acc={self.accuracy:.1f}%"
        elif self.task_type == "entity_regression":
            parts = []
            if 'mae' in metrics:
                parts.append(f"MAE={metrics['mae']:.4f}")
            if 'rmse' in metrics:
                parts.append(f"RMSE={metrics['rmse']:.4f}")
            return " | ".join(parts) if parts else "MAE=N/A"
        else:
            return f"Acc={self.accuracy:.1f}%"
    
    def log_running_stats(self):
        """Log current running statistics with ALL task-appropriate metrics."""
        self.logger.info(f"\n{Back.WHITE}{Fore.BLACK}Running Statistics:{Style.RESET_ALL}")
        self.logger.info(f"Solved: {self.total} | Skipped: {self.skipped}")
        
        metrics = self.compute_all_metrics()
        
        if self.task_type == "entity_classification":
            self.logger.info(f"{Fore.GREEN}Correct: {self.correct}{Style.RESET_ALL} | {Fore.RED}Incorrect: {self.incorrect}{Style.RESET_ALL}")
            for key, label in [('roc_auc', 'AUROC'), ('average_precision', 'Average Precision'),
                               ('f1', 'F1'), ('macro_f1', 'Macro F1'), ('micro_f1', 'Micro F1'),
                               ('accuracy', 'Accuracy')]:
                if key in metrics:
                    fmt = f"{metrics[key]:.1%}" if key == 'accuracy' else f"{metrics[key]:.4f}"
                    self.logger.info(f"{label}: {fmt}")
            if not metrics and self.correct + self.incorrect > 0:
                self.logger.info(f"Accuracy: {self.accuracy:.1f}%")
            if not metrics:
                self.logger.info("(Metrics require at least 2 samples)")
        elif self.task_type == "entity_regression":
            if 'mae' in metrics:
                self.logger.info(f"MAE: {metrics['mae']:.4f}")
            if 'rmse' in metrics:
                self.logger.info(f"RMSE: {metrics['rmse']:.4f}")
            if 'r2' in metrics:
                self.logger.info(f"R²: {metrics['r2']:.4f}")
            if not metrics:
                self.logger.info("(No valid numeric predictions yet)")
    
    def log_final_stats(self):
        """Log final evaluation statistics with ALL task-appropriate metrics."""
        self.logger.info(f"\n{Back.WHITE}{Fore.BLACK}Final Running Statistics:{Style.RESET_ALL}")
        self.logger.info(f"Total solved: {self.total} | Skipped: {self.skipped}")
        
        metrics = self.compute_all_metrics()
        
        if self.task_type == "entity_classification":
            self.logger.info(f"{Fore.GREEN}Correct: {self.correct}{Style.RESET_ALL} | {Fore.RED}Incorrect: {self.incorrect}{Style.RESET_ALL}")
            for key, label in [('roc_auc', 'AUROC'), ('average_precision', 'Average Precision'),
                               ('f1', 'F1'), ('macro_f1', 'Macro F1'), ('micro_f1', 'Micro F1'),
                               ('accuracy', 'Accuracy')]:
                if key in metrics:
                    fmt = f"{metrics[key]:.1%}" if key == 'accuracy' else f"{metrics[key]:.4f}"
                    self.logger.info(f"{label}: {fmt}")
        elif self.task_type == "entity_regression":
            if 'mae' in metrics:
                self.logger.info(f"MAE: {metrics['mae']:.4f}")
            if 'rmse' in metrics:
                self.logger.info(f"RMSE: {metrics['rmse']:.4f}")
            if 'r2' in metrics:
                self.logger.info(f"R²: {metrics['r2']:.4f}")

        self.logger.info("(Note: official metrics from RelBench evaluator are reported separately)")
    
    def to_dict(self) -> dict:
        """Return statistics as a dictionary with ALL task-appropriate metrics."""
        result = {
            'correct': self.correct,
            'incorrect': self.incorrect,
            'skipped': self.skipped,
            'total': self.total,
            'accuracy': self.accuracy,
            'task_type': self.task_type,
        }
        # Add all task-specific metrics
        metrics = self.compute_all_metrics()
        result.update(metrics)
        return result




# Cache for database paths to avoid reloading.
# Keyed by (dataset, task, cutoff_tag) so val/test DBs are isolated.
_database_cache: Dict[Tuple[str, str, str], str] = {}

def setup_database_from_relbench(
    loader,
    db_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    reuse_existing: bool = True,
    upto_timestamp: Optional[Any] = None,
) -> str:
    """
    Create a DuckDB database from the task-modified RelBench Database object.
    
    Args:
        loader: RelBenchLoader instance
        db_path: Optional path for the DuckDB file. If None, creates in artifacts/.
        logger: Optional logger instance
        reuse_existing: If True, reuse existing database for same dataset/task (default: True)
        upto_timestamp: Optional temporal cutoff; if set, materialize db.upto(upto_timestamp)
                        before exposing SQL tables.
        
    Returns:
        str: Path to the created DuckDB file
    """
    try:
        import duckdb
    except ImportError:
        raise ImportError("duckdb is required. Install with: pip install duckdb")
    
    log = logger or logging.getLogger(__name__)
    
    try:
        # IMPORTANT: when a task is loaded, use the task's dataset.get_db()
        # so AutoCompleteTask target/anti-leakage removals are applied.
        if hasattr(loader, "task") and loader.task is not None:
            db = loader.task.dataset.get_db()
        else:
            db = loader.get_database()
        dataset_name = loader.dataset_name
        if upto_timestamp is not None:
            db = db.upto(upto_timestamp)
    except Exception as e:
        log.error(f"Failed to get RelBench database path: {e}")
        raise
    
    cutoff_tag = str(upto_timestamp) if upto_timestamp is not None else "none"

    # Check cache first if reuse_existing is True
    if db_path is None and reuse_existing:
        cache_key = (
            dataset_name,
            loader.task_name if hasattr(loader, "task_name") else "unknown",
            cutoff_tag,
        )
        if cache_key in _database_cache:
            cached_path = _database_cache[cache_key]
            if Path(cached_path).exists():
                log.debug(f"Reusing cached database: {cached_path}")
                return cached_path
    
    if db_path is None:
        scratch_root = os.environ.get("SCRATCH_DIR", "/scratch")
        artifacts_dir = Path(scratch_root) / os.environ.get("USER", "user") / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        task_name = loader.task_name if hasattr(loader, 'task_name') and loader.task_name else "unknown"
        # Use consistent name for same dataset/task (instead of unique ID)
        if reuse_existing:
            suffix = "full" if upto_timestamp is None else cutoff_tag.replace(" ", "_").replace(":", "-")
            db_path = str(artifacts_dir / f"{dataset_name}_{task_name}_{suffix}.duckdb")
        else:
            unique_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            db_path = str(artifacts_dir / f"{dataset_name}_{task_name}_{unique_id}.duckdb")
    else:
        db_path = str(Path(db_path))
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    def _validate_leakage_columns(conn, db_obj) -> None:
        """Fail fast if task-protected columns are still SQL-visible."""
        if not hasattr(loader, "task") or loader.task is None:
            return

        blocked_pairs = []
        entity_table = getattr(loader.task, "entity_table", None)
        target_col = getattr(loader.task, "target_col", None)
        if entity_table and target_col:
            blocked_pairs.append((entity_table, target_col))

        for pair in getattr(loader.task, "remove_columns", []):
            if isinstance(pair, tuple) and len(pair) == 2:
                blocked_pairs.append(pair)

        # Deduplicate while preserving order
        seen = set()
        unique_pairs = []
        for table_name, col_name in blocked_pairs:
            key = (table_name, col_name)
            if key not in seen:
                seen.add(key)
                unique_pairs.append(key)

        violations = []
        for table_name, col_name in unique_pairs:
            if table_name not in db_obj.table_dict:
                continue
            try:
                sql = f"SELECT {col_name} FROM {table_name} LIMIT 1"
                conn.execute(sql).fetchall()
                violations.append(f"{table_name}.{col_name}")
            except Exception:
                # Expected path: binder error because the column is absent.
                continue

        if violations:
            raise RuntimeError(
                "Leakage guard failed: protected columns are still visible in SQL DB: "
                + ", ".join(violations)
            )

    def _drop_relation_if_exists(conn, relation_name: str) -> None:
        """Drop an existing table/view by checking actual relation type first."""
        rows = conn.execute(
            """
            SELECT table_type
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?
            """,
            [relation_name],
        ).fetchall()
        if not rows:
            return
        table_type = rows[0][0]
        if table_type == "VIEW":
            conn.execute(f"DROP VIEW {relation_name};")
        else:
            conn.execute(f"DROP TABLE {relation_name};")

    def _connect_with_lock_fallback(path: str):
        """Connect to DuckDB, falling back to a unique path if lock is held."""
        try:
            return duckdb.connect(path), path
        except Exception as e:
            msg = str(e).lower()
            if "conflicting lock" in msg or "could not set lock on file" in msg:
                unique_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
                alt_path = str(Path(path).with_name(f"{Path(path).stem}_{unique_id}.duckdb"))
                if log:
                    log.warning(
                        f"DuckDB path locked ({path}); retrying with unique DB path: {alt_path}"
                    )
                return duckdb.connect(alt_path), alt_path
            raise

    # Create DuckDB connection and materialize task-safe tables
    con, db_path = _connect_with_lock_fallback(db_path)
    with con:
        try:
            # Get table names from RelBench database
            table_names = list(db.table_dict.keys())
            
            # Materialize task-safe tables directly from loader.get_database().
            # This preserves AutoCompleteTask anti-leakage column removals.
            created_tables = []
            for name in table_names:
                table_obj = db.table_dict[name]
                if not hasattr(table_obj, "df"):
                    log.warning(f"Skipping table {name}: missing DataFrame")
                    continue
                source_name = f"_src_{name}"
                con.register(source_name, table_obj.df)
                # Ensure we can switch between previous table/view definitions.
                _drop_relation_if_exists(con, name)
                con.execute(f"CREATE TABLE {name} AS SELECT * FROM {source_name};")
                con.unregister(source_name)
                created_tables.append(name)
            
            # Add train table as a separate view for in-context learning
            # This table contains labeled examples that can help the model understand patterns
            if loader.task is not None:
                try:
                    train_table = loader.get_train_table()
                    if train_table is not None and hasattr(train_table, 'df'):
                        train_source_name = "_src_train_table"
                        con.register(train_source_name, train_table.df)
                        _drop_relation_if_exists(con, "train_table")
                        con.execute("CREATE TABLE train_table AS SELECT * FROM _src_train_table;")
                        con.unregister(train_source_name)
                        created_tables.append("train_table")
                        
                        if log:
                            log.info(f"Added train_table with {len(train_table.df)} labeled examples for in-context learning")
                except Exception as e:
                    if log:
                        log.warning(f"Failed to add train_table: {e}")
            
            if log:
                log.info(f"Created DuckDB database at {db_path} with {len(created_tables)} tables")

            # Final guard: ensure target/anti-leakage columns are truly inaccessible.
            _validate_leakage_columns(con, db)
            if log:
                log.info("Leakage guard passed: protected columns are not SQL-visible.")
            
            # Cache the database path
            if reuse_existing and db_path is not None:
                cache_key = (
                    dataset_name,
                    loader.task_name if hasattr(loader, "task_name") else "unknown",
                    cutoff_tag,
                )
                _database_cache[cache_key] = db_path
            
            return db_path
        except Exception as e:
            log.error(f"Failed to create DuckDB views: {e}")
            raise


def solve_problem_with_retries(
    solver,
    problem,
    max_retries: int = 3,
    logger: Optional[logging.Logger] = None,
) -> Optional[Any]:
    """
    Solve a single problem with retry logic.
    
    Args:
        solver: SQLSolver instance
        problem: RelBenchProblem to solve
        max_retries: Maximum number of retry attempts
        logger: Optional logger instance
        
    Returns:
        Solver answer or None if all retries failed
    """
    log = logger or logging.getLogger(__name__)
    
    for attempt in range(max_retries):
        try:
            answer = solver.solve_sql_problem(problem.problem)
            return answer
        except Exception as e:
            if attempt + 1 >= max_retries:
                log.error(
                    f"Error solving problem {problem.id}: {e}. "
                    f"Maximum retries ({max_retries}) reached, skipping problem."
                )
                return None
            log.warning(
                f"Error solving problem {problem.id}: {e}. "
                f"Retry {attempt + 1}/{max_retries}..."
            )
    
    return None


# ============================================================================
# Parallel Evaluation Utilities
# ============================================================================

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types and other non-serializable objects."""
    
    def default(self, obj):
        # Handle numpy types
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                           np.int16, np.int32, np.int64, np.uint8, np.uint16,
                           np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Handle other non-serializable types
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return super().default(obj)


def truncate_output(output: Any, max_length: int = 5000) -> Any:
    """
    Truncate very long outputs to prevent huge JSON files.
    
    Args:
        output: The output to truncate (string, dict, list, etc.)
        max_length: Maximum length for string outputs
        
    Returns:
        Truncated output with indication if truncated
    """
    if isinstance(output, str):
        if len(output) > max_length:
            return output[:max_length] + f"\n... [TRUNCATED: {len(output)} chars -> {max_length} chars]"
        return output
    elif isinstance(output, dict):
        truncated = {}
        for key, value in output.items():
            truncated[key] = truncate_output(value, max_length)
        return truncated
    elif isinstance(output, list):
        return [truncate_output(item, max_length) for item in output]
    else:
        return output


def extract_conversation_history(solver) -> List[Dict[str, Any]]:
    """
    Extract conversation history in simple format from solver.
    
    Args:
        solver: SQLSolver instance
        
    Returns:
        List of messages in format: [{"role": "...", "content": "...", "turn": 0}]
    """
    conversation = []
    
    if not hasattr(solver, "agent") or not hasattr(solver.agent, "memory"):
        return conversation
    
    try:
        records = solver.agent.memory.retrieve()
        for turn, rec in enumerate(records):
            msg = rec.memory_record.message
            
            if hasattr(msg, "to_dict"):
                d = msg.to_dict()
                role = d.get("role_name") or d.get("role_type") or "unknown"
                content = d.get("content", "")
                func_name = d.get("func_name")
                
                # Handle tool calls
                if func_name:
                    args = solver._extract_args_from_message(msg, d)
                    conversation.append({
                        "role": role,
                        "type": "tool_call",
                        "tool_name": func_name,
                        "args": truncate_output(args, max_length=2000),  # Truncate args
                        "tool_call_id": d.get("tool_call_id", ""),
                        "turn": turn,
                    })
                elif content:
                    conversation.append({
                        "role": role,
                        "type": "message",
                        "content": truncate_output(content, max_length=5000),
                        "turn": turn,
                    })
            else:
                # Fallback for messages without to_dict
                conversation.append({
                    "role": "unknown",
                    "type": "message",
                    "content": truncate_output(str(msg), max_length=5000),
                    "turn": turn,
                })
    except Exception:
        # Silently return empty if extraction fails
        pass
    
    return conversation


def extract_tool_calls(solver) -> List[Dict[str, Any]]:
    """
    Extract structured tool calls from solver.
    
    Args:
        solver: SQLSolver instance
        
    Returns:
        List of tool calls with outputs
    """
    tool_calls = []
    
    try:
        tool_ios = solver._extract_tool_calls_from_memory()
        for tool_io in tool_ios:
            tool_calls.append({
                "tool_name": tool_io.tool_name,
                "args": truncate_output(tool_io.args, max_length=2000),
                "output": truncate_output(tool_io.output, max_length=5000),
                "tool_call_id": tool_io.tool_call_id or "",
            })
    except Exception:
        # Silently return empty if extraction fails
        pass
    
    return tool_calls


def extract_reasoning_output(solver, include_formatted_log: bool = True) -> Dict[str, Any]:
    """
    Extract all reasoning output from solver.
    
    Args:
        solver: SQLSolver instance
        include_formatted_log: Whether to include formatted log (can be large)
        
    Returns:
        Dictionary with all reasoning data
    """
    conversation_history = extract_conversation_history(solver)
    tool_calls = extract_tool_calls(solver)
    tool_usage = solver.get_tool_usage()
    
    reasoning = {
        "conversation_history": conversation_history,
        "tool_calls": tool_calls,
        "tool_usage": tool_usage,
        "num_turns": len(conversation_history),
    }
    
    # Add formatted log if requested
    if include_formatted_log:
        try:
            formatted_log = solver.get_solver_log()
            # Store full formatted log without truncation (user requested full reasoning output)
            # Only truncate if it's extremely large (>100KB) to prevent JSON serialization issues
            if len(formatted_log) > 100000:
                reasoning["formatted_log"] = truncate_output(formatted_log, max_length=100000)
            else:
                reasoning["formatted_log"] = formatted_log
        except Exception:
            reasoning["formatted_log"] = None
    
    return reasoning


class ParallelResultCollector:
    """Thread-safe result collector that writes to a single JSONL file."""
    
    def __init__(self, output_dir: Path, concurrency: int):
        """
        Initialize parallel result collector.
        
        Args:
            output_dir: Directory to write result file
            concurrency: Maximum concurrency (used for worker_id assignment)
        """
        self.output_dir = Path(output_dir)
        self.concurrency = concurrency
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create single result file for all results
        result_file = self.output_dir / "results.jsonl"
        self.result_file = open(result_file, 'w', encoding='utf-8')
        
        self.lock = threading.Lock()
    
    def add_result(
        self,
        worker_id: int,
        index: int,
        problem_id: str,
        problem_text: str,
        prediction: Any,
        ground_truth: Any,
        is_correct: bool,
        elapsed: float,
        solver: Optional[Any] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Write result to single JSONL file (thread-safe).
        
        Args:
            worker_id: Worker thread ID (for tracking purposes)
            index: Table row index
            problem_id: Problem ID
            problem_text: Problem text
            prediction: Prediction value
            ground_truth: Ground truth value
            is_correct: Whether prediction is correct
            elapsed: Time elapsed in seconds
            solver: Optional SQLSolver instance for extracting reasoning
            metadata: Optional additional metadata
        """
        result = {
            "index": index,
            "problem_id": problem_id,
            "problem_text": problem_text,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "worker_id": worker_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "elapsed": elapsed,
            **(metadata or {}),
        }
        
        # Add reasoning output if solver provided OR if already extracted in metadata
        # Extract reasoning and immediately serialize to avoid keeping references
        if solver is not None:
            try:
                reasoning = extract_reasoning_output(solver, include_formatted_log=True)
                result["reasoning"] = reasoning
                # Explicitly delete the reasoning dict after adding to result
                # This helps garbage collection, though Python will handle it anyway
                del reasoning
            except Exception as e:
                # If extraction fails, include error in metadata
                result["reasoning_error"] = str(e)
        elif metadata and "reasoning" in metadata:
            # Reasoning already extracted and passed in metadata
            result["reasoning"] = metadata["reasoning"]
        
        # Write to single file (thread-safe with lock)
        with self.lock:
            json_str = json.dumps(result, ensure_ascii=False, cls=CustomJSONEncoder)
            self.result_file.write(json_str + '\n')
            self.result_file.flush()  # Immediate write for crash safety
    
    def close_all(self):
        """Close the result file."""
        with self.lock:
            if self.result_file:
                self.result_file.close()
                self.result_file = None


class PerWorkerLogger:
    """Setup per-worker logger with separate log file."""
    
    # Thread lock to prevent race conditions when multiple threads call setup_worker_logger
    _lock = threading.Lock()
    
    @staticmethod
    def setup_worker_logger(output_dir: Path, worker_id: int, 
                           log_level: int = logging.INFO) -> logging.Logger:
        """
        Create a logger for a specific worker.
        
        Args:
            output_dir: Directory to write log file
            worker_id: Worker thread ID
            log_level: Logging level
            
        Returns:
            Logger instance for the worker
        """
        # Thread-safe: use lock to prevent race conditions when multiple threads call this simultaneously
        with PerWorkerLogger._lock:
            logger = logging.getLogger(f"worker_{worker_id}")
            logger.setLevel(log_level)
            
            # Close and clear existing handlers to prevent handler buildup and memory leaks
            # This is critical when workers are reused or setup_worker_logger is called multiple times
            # IMPORTANT: Always clear handlers first, even if list appears empty, to prevent accumulation
            existing_handlers = list(logger.handlers)  # Copy list before iteration
            for handler in existing_handlers:
                try:
                    handler.close()  # Properly close file handlers to release resources
                except Exception:
                    pass  # Ignore errors when closing handlers
            logger.handlers.clear()
            
            # Always create a fresh handler (handlers were just cleared above)
            # File handler for worker-specific log
            log_file = output_dir / f"worker_{worker_id}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            
            # Format: include worker_id in log messages
            formatter = logging.Formatter(
                '%(asctime)s - worker_%(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Don't propagate to root logger (avoid duplicate logs)
            logger.propagate = False
            
            return logger


def apply_vllm_transformers_compat():
    """
    Apply transformers compatibility fixes for vLLM 0.13.0 with transformers 5.1.0.
    Ensures ALLOWED_LAYER_TYPES exists before vLLM imports it.
    
    Returns:
        bool: True if compatibility fix was applied or already exists, False otherwise
    """
    try:
        import transformers.configuration_utils as config_utils
        
        # Check if ALLOWED_LAYER_TYPES already exists (transformers >= 5.1.0 may have it)
        if hasattr(config_utils, 'ALLOWED_LAYER_TYPES'):
            # Already exists, no fix needed
            return True
        
        # For transformers 5.0+, combine the new constants
        if hasattr(config_utils, 'ALLOWED_ATTENTION_LAYER_TYPES') and hasattr(config_utils, 'ALLOWED_MLP_LAYER_TYPES'):
            # Create a combined tuple for compatibility
            config_utils.ALLOWED_LAYER_TYPES = tuple(
                list(config_utils.ALLOWED_ATTENTION_LAYER_TYPES) + 
                list(config_utils.ALLOWED_MLP_LAYER_TYPES)
            )
            print(f"Applied compatibility fix: ALLOWED_LAYER_TYPES ({len(config_utils.ALLOWED_LAYER_TYPES)} types)", file=sys.stderr)
            return True
        else:
            print("Warning: Could not find ALLOWED_ATTENTION_LAYER_TYPES or ALLOWED_MLP_LAYER_TYPES", file=sys.stderr)
            return False
    except ImportError as e:
        print(f"Error importing transformers: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error applying compatibility fix: {e}", file=sys.stderr)
        return False