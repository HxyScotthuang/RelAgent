"""Adapter that wraps 4DBInfer (dbinfer_bench) datasets to match the RelBenchLoader interface.

Usage:
    from relagent.fourdbinfer_loader import FourDBInferLoader
    loader = FourDBInferLoader("amazon-4db", "user-churn", logger=logger)

The loader exposes the same duck-typed interface expected by ScientistAgent,
RelBenchEvaluator, and setup_database_from_relbench. The raw relational tables from
4DBInfer are loaded via dbb.load_rdb_data(base_name) (no variant suffix = raw tables).
"""

from __future__ import annotations

import logging
import random
import sys
import types
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


def _patch_torchdata_stubs() -> None:
    """Pre-stub modules that dbinfer_bench.graph_dataset would import via DGL.

    dbinfer_bench/__init__.py does `from .graph_dataset import *`.
    graph_dataset imports dgl.graphbolt which needs removed torchdata sub-modules.
    We only use dbb.load_rdb_data() (relational tables), not graph functionality,
    so pre-stubbing dbinfer_bench.graph_dataset as empty prevents DGL from loading.
    """
    # Pre-stub dbinfer_bench.graph_dataset so the wildcard import in
    # dbinfer_bench/__init__.py is a no-op (Python uses sys.modules cache).
    if "dbinfer_bench.graph_dataset" not in sys.modules:
        sys.modules["dbinfer_bench.graph_dataset"] = types.ModuleType(
            "dbinfer_bench.graph_dataset"
        )

# ---------------------------------------------------------------------------
# Dataset / task name maps (CLI names → dbinfer_bench names)
# ---------------------------------------------------------------------------

_DATASET_NAME_MAP: Dict[str, str] = {
    "amazon-4db": "amazon",
    "outbrain-4db": "outbrain-small",
    "retailrocket-4db": "retailrocket",
    "stackexchange-4db": "stackexchange",
}

_TASK_NAME_MAP: Dict[str, str] = {
    "user-churn": "churn",
    "ad-ctr": "ctr",
    "item-cvr": "cvr",
    "post-upvote": "upvote",
}

def fourdbinfer_cli_pair_registered(dataset_name: str, task_name: str) -> bool:
    """Return True if (dataset_name, task_name) is a supported 4DBInfer CLI pair (no I/O)."""
    return dataset_name in _DATASET_NAME_MAP and task_name in _TASK_NAME_MAP
    
# Fallback label column name (dbinfer_bench uses the target_column from metadata instead).
_LABEL_COL_FALLBACK = "label"

# Columns in stackexchange raw tables that encode accumulated final state rather than
# state at prediction time. Including them gives the agent near-oracle signals
# (e.g. final Post.Score predicts whether a post ever got upvoted). Stripping them
# forces the agent to rely only on features available at the prediction timestamp.
_TEMPORAL_LEAKY_COLS: Dict[str, Dict[str, set]] = {
    "stackexchange": {
        "Posts": {
            "Score", "ViewCount", "AnswerCount", "CommentCount", "FavoriteCount",
            "LastActivityDate", "LastEditDate", "ClosedDate", "CommunityOwnedDate",
            "AcceptedAnswerId",
        },
        "Users": {
            "Reputation", "Views", "UpVotes", "DownVotes", "LastAccessDate",
        },
    },
}


# ---------------------------------------------------------------------------
# Light wrappers so setup_database_from_relbench can materialise tables
# ---------------------------------------------------------------------------

class FourDBInferTableObj:
    """Minimal Table-like object (duck-types relbench.base.Table) with a .df attribute."""

    def __init__(self, df: pd.DataFrame, entity_col: str = "", target_col: str = ""):
        self.df = df
        self.entity_col = entity_col
        self.target_col = target_col
        self.time_col = None
        self.fkey_col_to_pkey_table: Dict[str, str] = {}
        self.pkey_col: Optional[str] = None
        self.name: Optional[str] = None
        # Attributes expected by SampledTable copy loop
        self.src_entity_col: Optional[str] = None
        self.dst_entity_col: Optional[str] = None
        # Sampling metadata (set when sampling is applied)
        self._is_sampled: bool = False
        self._original_size: Optional[int] = None
        self._sampled_indices: Optional[List[int]] = None

    def __len__(self) -> int:
        return len(self.df)


class FourDBInferDB:
    """Duck-types a RelBench Database: has .table_dict used by setup_database_from_relbench."""

    def __init__(
        self,
        table_dict: Dict[str, FourDBInferTableObj],
        time_col_map: Optional[Dict[str, str]] = None,
        target_table: Optional[str] = None,
        label_tables: Optional[set] = None,
    ):
        self.table_dict = table_dict
        # Maps table name → the name of its primary datetime column (from dataset metadata).
        self._time_col_map: Dict[str, str] = time_col_map or {}
        # Name of the task's target entity table (from task.metadata.target_table).
        self._target_table: Optional[str] = target_table
        # Tables from which the label column was actually stripped (row presence may encode
        # the label, so these must still be row-filtered in upto()).
        self._label_tables: set = label_tables or set()

    def upto(self, timestamp: Any) -> "FourDBInferDB":
        """Return a copy of this DB with each table's rows censored to <= timestamp.

        Event tables and tables that directly encode the label (Click for outbrain) are
        filtered to <= timestamp. The target entity table is skipped when the label is
        NOT stored there — val/test entities (e.g. stackexchange Posts) need to exist in
        the DB for the agent's feature JOINs to work.

        Static lookup tables without a datetime column are kept whole.
        """
        if timestamp is None:
            return self
        ts = pd.Timestamp(timestamp)
        new_table_dict: Dict[str, FourDBInferTableObj] = {}
        for tbl_name, tbl_obj in self.table_dict.items():
            time_col = self._time_col_map.get(tbl_name)
            # Skip row-filtering the target entity table when the label is not stored there.
            # Rationale: val/test entities ARE rows in that table (e.g. Posts for upvote,
            # Users for churn). Filtering to <= val_cutoff removes the very rows we need
            # to predict for, causing all-NULL features and constant 0.5 AUROC.
            # Exception: if the label WAS in that table (e.g. outbrain Click.clicked),
            # row presence encodes the label and we must still filter.
            is_label_free_entity_table = (
                tbl_name == self._target_table
                and tbl_name not in self._label_tables
            )
            if time_col and time_col in tbl_obj.df.columns and not is_label_free_entity_table:
                mask = pd.to_datetime(tbl_obj.df[time_col]) <= ts
                filtered_df = tbl_obj.df[mask].reset_index(drop=True)
                new_obj = FourDBInferTableObj(
                    filtered_df,
                    entity_col=tbl_obj.entity_col,
                    target_col=tbl_obj.target_col,
                )
                new_obj.time_col = tbl_obj.time_col
                new_obj.fkey_col_to_pkey_table = tbl_obj.fkey_col_to_pkey_table
                new_obj.pkey_col = tbl_obj.pkey_col
                new_obj.name = tbl_obj.name
                new_obj.src_entity_col = tbl_obj.src_entity_col
                new_obj.dst_entity_col = tbl_obj.dst_entity_col
                new_table_dict[tbl_name] = new_obj
            else:
                new_table_dict[tbl_name] = tbl_obj
        return FourDBInferDB(new_table_dict, self._time_col_map, self._target_table, self._label_tables)


class FourDBInferDatasetAdapter:
    """Duck-types a RelBench Dataset (provides .get_db(), .val_timestamp, .test_timestamp)."""

    def __init__(self, db: FourDBInferDB):
        self._db = db
        self.val_timestamp = None
        self.test_timestamp = None

    def get_db(self) -> FourDBInferDB:
        return self._db


# ---------------------------------------------------------------------------
# Task adapter
# ---------------------------------------------------------------------------

class FourDBInferTaskAdapter:
    """Duck-types a RelBench EntityTask so the rest of the pipeline works unchanged."""

    def __init__(
        self,
        dbb_dataset: Any,
        dbb_task: Any,
        dataset_adapter: FourDBInferDatasetAdapter,
        entity_col: str,
        target_col: str,
        extra_entity_cols: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self._dbb_dataset = dbb_dataset
        self._dbb_task = dbb_task
        self.dataset = dataset_adapter
        self.entity_col = entity_col
        self.target_col = target_col
        self.extra_entity_cols: List[str] = extra_entity_cols or []
        self._logger = logger or logging.getLogger(__name__)

        # Mirror RelBench task fields expected by the pipeline
        self.time_col: Optional[str] = getattr(dbb_task.metadata, "time_column", None)
        self.eval_k: int = 10              # not used for classification, but expected
        self.remove_columns: List = []     # disables leakage-guard column checks
        self.entity_table: Optional[str] = None  # disables entity-table leakage guard

        # Set task_type to match RelBench's TaskType.BINARY_CLASSIFICATION so
        # RelBenchEvaluator's binary-label detection logic fires correctly.
        try:
            from relbench.tasks import TaskType
            self.task_type = TaskType.BINARY_CLASSIFICATION
        except Exception:
            self.task_type = "binary_classification"

    def _split_to_array(self, split: str) -> Dict[str, np.ndarray]:
        if split == "train":
            return self._dbb_task.train_set
        if split == "val":
            return self._dbb_task.validation_set
        if split == "test":
            return self._dbb_task.test_set
        raise ValueError(f"Unknown split: {split!r}. Expected 'train', 'val', or 'test'.")

    def get_table(self, split: str, mask_input_cols: bool = False) -> FourDBInferTableObj:
        data = self._split_to_array(split)
        entity_vals = data[self.entity_col]
        # Labels stored under target_col name in the split dict (e.g. "churn", "clicked")
        raw_labels = data.get(self.target_col, data.get(_LABEL_COL_FALLBACK, np.zeros(len(entity_vals))))
        label_vals = np.asarray(raw_labels, dtype=float).astype(int)
        row: Dict[str, Any] = {self.entity_col: entity_vals}
        # Include timestamp when entity_col has duplicates (temporal windows).
        # The asof_col join in validation.py uses (entity_col, time_col) as composite
        # key, which makes every row unique and prevents shape mismatch.
        if self.time_col and self.time_col in data:
            row[self.time_col] = data[self.time_col]
        # Include secondary entity columns (e.g. visitorid for retailrocket/cvr) so
        # the agent can write visitor-level feature queries from the eval_table.
        for ec in self.extra_entity_cols:
            if ec in data:
                row[ec] = data[ec]
        row[self.target_col] = label_vals
        df = pd.DataFrame(row)

        # Ensure the composite key is unique per row.  Include extra entity cols so
        # that (itemid, visitorid, timestamp) triples are preserved rather than
        # collapsed to (itemid, timestamp).
        key_cols = [self.entity_col]
        if self.time_col and self.time_col in df.columns:
            key_cols.append(self.time_col)
        for ec in self.extra_entity_cols:
            if ec in df.columns and ec not in key_cols:
                key_cols.append(ec)
        n_before = len(df)
        df = df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            self._logger.warning(
                f"get_table({split!r}): dropped {n_dropped}/{n_before} rows with duplicate "
                f"({', '.join(key_cols)}) keys to ensure unique composite join key."
            )

        return FourDBInferTableObj(df, entity_col=self.entity_col, target_col=self.target_col)

    def evaluate(self, preds_array: np.ndarray, table: FourDBInferTableObj) -> Dict[str, float]:
        from sklearn.metrics import roc_auc_score, average_precision_score

        y_true = table.df[self.target_col].to_numpy().astype(float)
        y_score = np.asarray(preds_array, dtype=float)

        if len(y_true) != len(y_score):
            raise ValueError(
                f"evaluate(): preds length ({len(y_score)}) != table length ({len(y_true)})"
            )

        # Guard against constant label splits (e.g. a tiny eval_sample with all-zeros)
        unique = np.unique(y_true)
        if len(unique) < 2:
            self._logger.warning(
                f"evaluate(): only {len(unique)} unique label(s) in split; "
                "AUROC/AP are undefined — returning 0.5 / 0.0 as fallback."
            )
            return {"roc_auc": 0.5, "average_precision": float(unique[0]) if len(unique) else 0.0}

        roc_auc = float(roc_auc_score(y_true, y_score))
        avg_prec = float(average_precision_score(y_true, y_score))
        return {"roc_auc": roc_auc, "average_precision": avg_prec}


# ---------------------------------------------------------------------------
# Top-level loader (mirrors RelBenchLoader)
# ---------------------------------------------------------------------------

class FourDBInferLoader:
    """
    Adapter that wraps a 4DBInfer dataset+task and exposes the RelBenchLoader interface.

    Parameters
    ----------
    dataset_name : str
        CLI dataset name, e.g. "amazon-4db". Mapped internally to the dbinfer_bench name.
    task_name : str
        CLI task name, e.g. "user-churn". Mapped internally to the dbinfer_bench task name.
    data_dir : str, optional
        Path to the local 4DBInfer cache directory (passed to dbb.load_rdb_data as ``root``).
        If None, dbinfer_bench uses its default cache (~/.dgl/).
    logger : logging.Logger, optional
    """

    def __init__(
        self,
        dataset_name: str,
        task_name: str,
        data_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        try:
            _patch_torchdata_stubs()
            import dbinfer_bench as dbb
        except ImportError as exc:
            raise ImportError(
                "dbinfer-bench is not installed. "
                "Install with: pip install dbinfer-bench"
            ) from exc

        self._logger = logger or logging.getLogger(__name__)
        self.dataset_name = dataset_name
        self.task_name = task_name

        # Map CLI names → dbinfer names
        dbb_dataset_name = _DATASET_NAME_MAP.get(dataset_name, dataset_name)
        dbb_task_name = _TASK_NAME_MAP.get(task_name, task_name)

        self._logger.info(
            f"Loading 4DBInfer dataset: {dbb_dataset_name} / task: {dbb_task_name}"
        )

        # Load raw relational tables (base name = no variant suffix)
        load_kwargs: Dict[str, Any] = {}
        if data_dir is not None:
            load_kwargs["root"] = data_dir
        self._dbb_dataset = dbb.load_rdb_data(dbb_dataset_name, **load_kwargs)
        self._dbb_task = self._dbb_dataset.get_task(dbb_task_name)

        # Resolve entity column from task split keys
        entity_col = self._resolve_entity_col()
        extra_entity_cols = self._resolve_extra_entity_cols(entity_col)
        target_col_meta = getattr(self._dbb_task.metadata, "target_column", _LABEL_COL_FALLBACK)

        # Build DB adapter from raw tables (strips target_col to prevent label leakage)
        raw_table_dict, label_tables = self._build_table_dict(target_col_meta)
        time_col_map = self._build_time_col_map()
        target_table = getattr(self._dbb_task.metadata, "target_table", None)
        db = FourDBInferDB(
            raw_table_dict,
            time_col_map=time_col_map,
            target_table=target_table,
            label_tables=label_tables,
        )
        dataset_adapter = FourDBInferDatasetAdapter(db)

        self._task_adapter = FourDBInferTaskAdapter(
            dbb_dataset=self._dbb_dataset,
            dbb_task=self._dbb_task,
            dataset_adapter=dataset_adapter,
            entity_col=entity_col,
            extra_entity_cols=extra_entity_cols,
            target_col=target_col_meta,
            logger=self._logger,
        )

        # Cache for sampled tables (split_evalSample → FourDBInferTableObj)
        self._sampled_tables: Dict[str, FourDBInferTableObj] = {}

        self._logger.info(
            f"4DBInfer loader ready: entity_col={entity_col!r}, "
            f"target_col={target_col_meta!r}, "
            f"tables={list(raw_table_dict.keys())}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_entity_col(self) -> str:
        """Return the entity/node ID column name from the task split dicts.

        Skips the target column and time column (both from task metadata), then:
        - If one candidate remains → return it.
        - If multiple remain → pick the first one in the target table's schema column order.
        """
        meta = self._dbb_task.metadata
        label_col = getattr(meta, "target_column", None)
        time_col = getattr(meta, "time_column", None)
        skip = {c for c in (label_col, time_col) if c}

        train_keys = list(self._dbb_task.train_set.keys())
        candidates = [k for k in train_keys if k not in skip]

        if len(candidates) == 1:
            return candidates[0]

        if not candidates:
            raise ValueError(
                f"Cannot determine entity column from train_set keys {train_keys}. "
                f"After skipping {skip!r}, no candidates remain."
            )

        # Multiple candidates: pick the first in target table schema column order.
        target_table_name = getattr(meta, "target_table", None)
        if target_table_name:
            for tbl_schema in self._dbb_dataset.metadata.tables:
                if getattr(tbl_schema, "name", None) == target_table_name:
                    for col in getattr(tbl_schema, "columns", []):
                        col_name = getattr(col, "name", None)
                        if col_name and col_name in candidates:
                            return col_name

        return candidates[0]

    def _resolve_extra_entity_cols(self, primary_entity_col: str) -> List[str]:
        """Return any additional entity columns beyond the primary entity_col.

        For tasks like retailrocket/cvr where the prediction unit is a
        (visitorid, itemid, timestamp) triple, this returns ['visitorid'] so that
        the eval_table exposes visitorid and the agent can write visitor-level
        feature queries.
        """
        meta = self._dbb_task.metadata
        label_col = getattr(meta, "target_column", None)
        time_col = getattr(meta, "time_column", None)
        skip = {c for c in (label_col, time_col, primary_entity_col) if c}

        train_keys = list(self._dbb_task.train_set.keys())
        extras = [k for k in train_keys if k not in skip]
        if extras:
            self._logger.info(f"4DBInfer extra entity cols: {extras}")
        return extras

    def _build_time_col_map(self) -> Dict[str, str]:
        """Return a mapping of table name → its primary datetime column.

        Uses the dbinfer dataset metadata column dtype field ('datetime') to identify
        timestamp columns in each raw table. Only the first datetime column per table is
        used (tables rarely have more than one meaningful timestamp).
        """
        time_col_map: Dict[str, str] = {}
        for tbl_schema in getattr(self._dbb_dataset.metadata, "tables", []):
            tbl_name = getattr(tbl_schema, "name", None)
            if not tbl_name:
                continue
            for col in getattr(tbl_schema, "columns", []):
                col_name = getattr(col, "name", None)
                col_dtype = getattr(col, "dtype", None)
                if col_name and col_dtype == "datetime":
                    time_col_map[tbl_name] = col_name
                    break
        if time_col_map:
            self._logger.info(f"4DBInfer time column map: {time_col_map}")
        return time_col_map

    def _build_table_dict(
        self, target_col: str
    ) -> tuple:  # (Dict[str, FourDBInferTableObj], set)
        """Convert dbinfer_bench tables (Dict[str, Dict[str, np.ndarray]]) to table objects.

        Two categories of leakage are prevented here:

        1. Direct label leakage: strip target_col from every raw table (e.g. outbrain's
           Click.clicked IS the prediction target; leaving it inflates AUROC to ~0.9985).

        2. Temporal snapshot leakage: some tables store accumulated aggregate values as
           of data-collection time (e.g. stackexchange Posts.Score is the 2017 final
           accumulated vote count, not the value at the 2013-2015 prediction timestamps).
           Row-level timestamp filtering cannot fix this because the column itself encodes
           future state regardless of which rows are retained. Stripping these columns
           forces the agent to compute them from event tables (Vote, Comments, etc.) which
           have row-level timestamps and naturally support temporally-correct aggregation.

        Returns
        -------
        (result, label_tables) where label_tables is the set of table names from which
        target_col was actually stripped. These tables must still be row-filtered in
        upto() because row presence may directly encode the label (e.g. outbrain Click).
        """
        # Resolve dataset-level temporal leaky cols (keyed by dbinfer base name)
        dbb_base_name = _DATASET_NAME_MAP.get(self.dataset_name, self.dataset_name)
        temporal_leaky = _TEMPORAL_LEAKY_COLS.get(dbb_base_name, {})

        result: Dict[str, FourDBInferTableObj] = {}
        label_tables: set = set()
        for tbl_name, col_dict in self._dbb_dataset.tables.items():
            df = pd.DataFrame({k: v for k, v in col_dict.items()})
            if target_col in df.columns:
                df = df.drop(columns=[target_col])
                label_tables.add(tbl_name)
                self._logger.info(
                    f"Table '{tbl_name}': stripped target column '{target_col}' to prevent label leakage."
                )
            tbl_leaky = temporal_leaky.get(tbl_name, set())
            cols_to_drop = [c for c in tbl_leaky if c in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                self._logger.info(
                    f"Table '{tbl_name}': stripped temporal snapshot columns "
                    f"{sorted(cols_to_drop)} to prevent temporal leakage."
                )
            result[tbl_name] = FourDBInferTableObj(df)
        return result, label_tables

    # ------------------------------------------------------------------
    # Public interface (mirrors RelBenchLoader)
    # ------------------------------------------------------------------

    @property
    def task(self) -> FourDBInferTaskAdapter:
        return self._task_adapter

    def get_table(
        self, split: str = "train", eval_sample: Optional[int] = None
    ) -> FourDBInferTableObj:
        cache_key = f"{split}_{eval_sample}"
        if cache_key in self._sampled_tables:
            return self._sampled_tables[cache_key]

        table = self._task_adapter.get_table(split)
        original_size = len(table.df)

        if eval_sample is not None and original_size > eval_sample:
            sampled_indices = sorted(random.sample(range(original_size), eval_sample))
            sampled_df = table.df.iloc[sampled_indices].reset_index(drop=True).copy()
            obj = FourDBInferTableObj(
                sampled_df, entity_col=table.entity_col, target_col=table.target_col
            )
            obj._is_sampled = True
            obj._original_size = original_size
            obj._sampled_indices = sampled_indices
            self._sampled_tables[cache_key] = obj
            self._logger.info(
                f"Sampled 4DBInfer {split} table: {eval_sample} / {original_size} rows"
            )
            return obj

        return table

    def get_train_table(self) -> FourDBInferTableObj:
        return self.get_table("train")

    def get_val_table(self) -> FourDBInferTableObj:
        return self.get_table("val")

    def get_test_table(self) -> FourDBInferTableObj:
        return self.get_table("test")

    def get_val_timestamp(self) -> Optional[pd.Timestamp]:
        """Return the earliest prediction timestamp in the validation split.

        This serves as the DB cutoff for agent evaluation: the SQL database is
        censored to rows <= this timestamp so no validation-period events are visible.
        Returns None for tasks without a time dimension (e.g. amazon, outbrain).
        """
        time_col = self._task_adapter.time_col
        if time_col and time_col in self._dbb_task.validation_set:
            return pd.Timestamp(self._dbb_task.validation_set[time_col].min())
        return None

    def get_test_timestamp(self) -> Optional[pd.Timestamp]:
        """Return the earliest prediction timestamp in the test split.

        The test-time DB is censored to rows <= this timestamp, mirroring RelBench's
        two-DB design (agent DB at val cutoff, test-eval DB at test cutoff).
        Returns None for tasks without a time dimension.
        """
        time_col = self._task_adapter.time_col
        if time_col and time_col in self._dbb_task.test_set:
            return pd.Timestamp(self._dbb_task.test_set[time_col].min())
        return None

    def evaluate(
        self,
        predictions: np.ndarray,
        split: str = "test",
        table: Optional[FourDBInferTableObj] = None,
    ) -> Dict[str, float]:
        if table is None:
            table = self.get_table(split)
        return self._task_adapter.evaluate(predictions, table)
