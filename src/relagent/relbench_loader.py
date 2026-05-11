from typing import Optional, List, Dict, TYPE_CHECKING
import logging
import numpy as np
import pandas as pd
import random

try:
    from relbench.base import Table, Database, Dataset, EntityTask
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task
    RELBENCH_AVAILABLE = True
except ImportError:
    RELBENCH_AVAILABLE = False
    # Create dummy types for type hints when RelBench is not available
    if TYPE_CHECKING:
        from typing import Any as Database, Any as Dataset, Any as EntityTask, Any as Table
    else:
        Database = None
        Dataset = None
        EntityTask = None
        Table = None
    print("Warning: RelBench not available. Please install relbench package.")

try:
    from .utils import SampledTable
except ImportError:
    from utils import SampledTable


class RelBenchProblem:
    """Wrapper for a RelBench task problem (entity to predict on)."""
    
    def __init__(
        self,
        entity_id: str,
        problem_text: str,
        ground_truth: Optional[float] = None, 
        split: str = "train",
        table_row_index: Optional[int] = None
    ):
        """
        Initialize a RelBench problem.
        
        Args:
            entity_id: Unique identifier for the entity
            problem_text: The problem/question text (e.g., SQL query description)
            ground_truth: The ground truth value for evaluation (if available)
            split: The split this problem belongs to ("train", "val", or "test")
            table_row_index: The index of this entity in the task table
        """
        self.id = entity_id
        self.problem = problem_text
        self.solution = str(ground_truth) if ground_truth is not None else None
        self.ground_truth = ground_truth
        self.split = split
        self.table_row_index = table_row_index
    
    def __repr__(self):
        return f"RelBenchProblem(id={self.id}, split={self.split})"


class RelBenchLoader:
    """
    A wrapper class for RelBench that provides database and task loading functionality.
    
    This class wraps RelBench's Dataset, Database, and EntityTask classes to provide
    a unified interface for loading datasets, databases, tasks, and their associated
    train/val/test tables. It handles temporal splitting and prevents data leakage.
    
    Attributes:
        dataset (Dataset): The RelBench dataset
        database (Database): The RelBench database
        task (EntityTask): The current task
        dataset_name (str): Name of the loaded dataset
        task_name (str): Name of the loaded task
        logger: Logger instance for logging
    """
    
    def __init__(self, dataset_name: str, task_name: Optional[str] = None, 
                 download: bool = True, logger: Optional[logging.Logger] = None):
        """
        Initialize the RelBench loader.
        
        Args:
            dataset_name: Name of the RelBench dataset (e.g., "rel-amazon")
            task_name: Name of the task (e.g., "user-churn"). If None, only dataset is loaded.
            download: Whether to download the dataset if not already cached (default: True)
            logger: Logger instance for logging
        
        Raises:
            ImportError: If RelBench is not installed
            ValueError: If dataset or task cannot be loaded
        """
        if not RELBENCH_AVAILABLE:
            raise ImportError("RelBench is not available. Please install it with: pip install relbench")
        
        self.logger = logger
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.dataset: Optional[Dataset] = None
        self.database: Optional[Database] = None
        self.task: Optional[EntityTask] = None
        # Cache for sampled tables to ensure consistency
        self._sampled_tables: Dict[str, Table] = {}  # split -> sampled table
        
        # Load dataset
        if self.logger:
            self.logger.info(f"Loading RelBench dataset: {dataset_name}")
        
        try:
            self.dataset = get_dataset(dataset_name, download=download)
            if self.logger:
                self.logger.info(f"Successfully loaded dataset: {dataset_name}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise ValueError(f"Failed to load dataset {dataset_name}: {e}")
        
        # Get database from dataset
        self.database = self.dataset.get_db()
        if self.logger:
            self.logger.info(f"Database loaded. Temporal splits: val={self.dataset.val_timestamp}, test={self.dataset.test_timestamp}")
        
        # Load task if specified
        if task_name:
            self.load_task(task_name, download=download)
    
    def load_task(self, task_name: str, download: bool = True):
        """
        Load a specific task for the current dataset.
        
        Args:
            task_name: Name of the task to load (e.g., "user-churn")
                      This will be normalized to the actual RelBench task name if needed.
            download: Whether to download the task if not already cached (default: True)
        
        Raises:
            ValueError: If task cannot be loaded
        """
        if self.dataset is None:
            raise ValueError("Dataset must be loaded before loading a task")
        
        if self.logger:
            self.logger.info(f"Loading task: {task_name} for dataset: {self.dataset_name}")
        
        try:
            self.task = get_task(self.dataset_name, task_name, download=download)
            self.task_name = task_name
            if self.logger:
                self.logger.info(f"Successfully loaded task: {task_name}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load task {task_name}: {e}")
            raise ValueError(f"Failed to load task {task_name}: {e}")
    
    def get_database(self) -> Database:
        """
        Get the RelBench database object.
        
        Returns:
            Database: The RelBench database object
        """
        if self.database is None:
            raise ValueError("Database not loaded. Ensure dataset was loaded successfully.")
        return self.database
    
    def get_task(self) -> EntityTask:
        """
        Get the current RelBench task object.
        
        Returns:
            EntityTask: The RelBench task object
        
        Raises:
            ValueError: If no task is loaded
        """
        if self.task is None:
            raise ValueError("No task loaded. Call load_task() first.")
        return self.task
    
    def get_table(self, split: str = "train", eval_sample: Optional[int] = None) -> Table:
        """
        Get a task table for the specified split, optionally sampling if table is large.
        Sampled tables are cached to ensure consistency across multiple calls.
        
        Args:
            split: The split to get ("train", "val", or "test")
            eval_sample: If provided and table size exceeds this, randomly sample this many rows
        
        Returns:
            Table: The RelBench table for the specified split (possibly sampled)
        
        Raises:
            ValueError: If task is not loaded or split is invalid
        """
        if self.task is None:
            raise ValueError("No task loaded. Call load_task() first.")
        
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")
        
        # Check cache for sampled table
        cache_key = f"{split}_{eval_sample}"
        if cache_key in self._sampled_tables:
            if self.logger:
                self.logger.debug(f"Using cached sampled table for {split}")
            return self._sampled_tables[cache_key]
        
        # Track calls to task.get_table() to detect if RelBench creates new copies
        if not hasattr(self, '_task_get_table_calls'):
            self._task_get_table_calls = {}
        call_key = f"{split}"
        self._task_get_table_calls[call_key] = self._task_get_table_calls.get(call_key, 0) + 1
        
        table = self.task.get_table(split)
        
        # Track the Table object ID to see if RelBench returns the same object or creates new ones
        if not hasattr(self, '_table_object_ids'):
            self._table_object_ids = {}
        table_id = id(table)
        table_df_id = id(table.df) if hasattr(table, 'df') else None
        
        if call_key not in self._table_object_ids:
            self._table_object_ids[call_key] = []
        self._table_object_ids[call_key].append({
            'table_id': hex(table_id),
            'df_id': hex(table_df_id) if table_df_id else None,
            'call_count': self._task_get_table_calls[call_key]
        })
        
        if self.logger and self._task_get_table_calls[call_key] > 1:
            prev_ids = self._table_object_ids[call_key][-2]
            if prev_ids['table_id'] != hex(table_id) or (table_df_id and prev_ids['df_id'] != hex(table_df_id)):
                self.logger.warning(
                    f"RelBench task.get_table('{split}') returned DIFFERENT objects on call #{self._task_get_table_calls[call_key]}: "
                    f"Table ID changed: {prev_ids['table_id']} -> {hex(table_id)}, "
                    f"DF ID changed: {prev_ids['df_id']} -> {hex(table_df_id) if table_df_id else 'N/A'}"
                )
        original_size = len(table.df)
        
        # Handle sampling: if eval_sample is provided and table is larger, sample it
        should_sample = eval_sample is not None and original_size > eval_sample
        
        if should_sample:
            if self.logger:
                self.logger.info(f"Table size ({original_size}) exceeds eval_sample ({eval_sample}), sampling {eval_sample} rows")
            
            # Randomly sample indices
            sampled_indices = sorted(random.sample(range(original_size), eval_sample))
            
            # Create sampled dataframe
            sampled_df = table.df.iloc[sampled_indices].reset_index(drop=True).copy()
            
            # Create sampled table wrapper
            sampled_table = SampledTable(sampled_df, table, sampled_indices, original_size)
            
            if self.logger:
                self.logger.info(f"Sampled table: {len(sampled_df)} rows (from {original_size})")
            
            # Cache the sampled table
            self._sampled_tables[cache_key] = sampled_table
            return sampled_table
        else:
            # No sampling needed - return original table
            # Only log at DEBUG level to avoid spam when called millions of times
            # The first call will be logged at INFO level by checking if this is the first call
            if self.logger:
                call_key = f"{split}_{eval_sample}"
                if not hasattr(self, '_table_logged'):
                    self._table_logged = set()
                if call_key not in self._table_logged:
                    # First time getting this table - log at INFO level
                    if eval_sample is None:
                        self.logger.info(f"Retrieved {split} table with {original_size} rows (no sampling)")
                    else:
                        self.logger.info(f"Retrieved {split} table with {original_size} rows (table size <= eval_sample, no sampling needed)")
                    self._table_logged.add(call_key)
                else:
                    # Subsequent calls - only log at DEBUG level
                    self.logger.debug(f"Retrieved {split} table with {original_size} rows (cached)")
            return table
    
    def get_train_table(self) -> Table:
        """Get the training table."""
        return self.get_table("train")
    
    def get_val_table(self) -> Table:
        """Get the validation table."""
        return self.get_table("val")
    
    def get_test_table(self) -> Table:
        """Get the test table."""
        return self.get_table("test")
    
    def get_val_timestamp(self) -> Optional[float]:
        """
        Get the validation timestamp for temporal splitting.
        
        Returns:
            Optional[float]: The validation timestamp, or None if not available
        """
        if self.dataset is None:
            return None
        return self.dataset.val_timestamp
    
    def get_test_timestamp(self) -> Optional[float]:
        """
        Get the test timestamp for temporal splitting.
        
        Returns:
            Optional[float]: The test timestamp, or None if not available
        """
        if self.dataset is None:
            return None
        return self.dataset.test_timestamp
    
    def get_problems(self, split: str = "train", num: Optional[int] = None, 
                     start_idx: int = 0, generate_problem_text: bool = True,
                     eval_sample: Optional[int] = None, 
                     load_unmasked_gt: bool = False,
                     task_type: Optional[str] = None) -> List[RelBenchProblem]:
        """
        Get problems (entities) from a specific split as RelBenchProblem objects.
        
        Args:
            split: The split to get problems from ("train", "val", or "test")
            num: Number of problems to return. If None, returns all problems (default: None)
            start_idx: Starting index for problems (default: 0)
            generate_problem_text: Whether to generate problem text from task metadata (default: True)
            eval_sample: If provided and table size exceeds this, randomly sample this many rows first
            load_unmasked_gt: If True, load unmasked ground truth labels separately for test split.
                             This allows computing metrics during inference without data leakage.
                             The masked table is still used for problem text (no labels visible to model).
        
        Returns:
            List[RelBenchProblem]: List of RelBenchProblem objects
        """
        # Load masked table (for model - no labels visible for test split)
        table = self.get_table(split, eval_sample=eval_sample)
        
        # Optionally load unmasked ground truth for test split (for evaluation only)
        # Note: Val and train splits already have labels visible (not masked), so they don't need special handling
        # Only test split is masked by RelBench to prevent data leakage during inference
        unmasked_gt_map = None
        if load_unmasked_gt and split == "test":
            try:
                # Get unmasked test table for ground truth labels
                # This is safe because we only use it to populate ground_truth in problems,
                # not exposed to the model during inference
                unmasked_table = self.task.get_table("test", mask_input_cols=False)
                target_col = self.task.target_col if hasattr(self.task, 'target_col') else None
                
                if target_col and target_col in unmasked_table.df.columns:
                    # Create mapping from row index to ground truth
                    # Note: We use row index, not product_id, because same product_id can appear
                    # multiple times with different timestamps (temporal splits)
                    
                    # Handle sampling: if eval_sample was used, we need to map accordingly
                    if eval_sample is not None and len(unmasked_table.df) > eval_sample:
                        # The masked table was sampled, so we need to sample unmasked table the same way
                        # Since sampling happens in get_table, we need to redo the sampling
                        # For now, we'll use the full unmasked table and match by original indices
                        # The table_row_index in problems will correspond to the sampled indices
                        unmasked_gt_map = {}
                        # Get the sampled indices from the table if available
                        if hasattr(table, '_sampled_indices'):
                            sampled_indices = table._sampled_indices
                            for i, orig_idx in enumerate(sampled_indices):
                                if orig_idx < len(unmasked_table.df):
                                    unmasked_gt_map[i] = unmasked_table.df.iloc[orig_idx][target_col]
                        else:
                            # If not sampled, direct index mapping
                            for idx in range(len(unmasked_table.df)):
                                unmasked_gt_map[idx] = unmasked_table.df.iloc[idx][target_col]
                    else:
                        # No sampling, direct index mapping
                        unmasked_gt_map = {}
                        for idx in range(len(unmasked_table.df)):
                            unmasked_gt_map[idx] = unmasked_table.df.iloc[idx][target_col]
                    
                    if self.logger:
                        self.logger.info(f"Loaded unmasked ground truth labels for {len(unmasked_gt_map)} test problems (for evaluation only, not exposed to model)")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to load unmasked ground truth: {e}")
                unmasked_gt_map = None
        
        target_col = self.task.target_col

        # Extract problems from table
        problems = []
        df = table.df

        end_idx = start_idx + num if num is not None else len(df)
        end_idx = min(end_idx, len(df))

        if hasattr(self.task, 'entity_col'):
            entity_col = self.task.entity_col
        elif hasattr(self.task, 'entity_column'):
            entity_col = self.task.entity_column
        else:
            entity_col = df.index.name if df.index.name else 'entity'

        for idx in range(start_idx, end_idx):
            row = df.iloc[idx]

            if entity_col and entity_col in df.columns:
                entity_id = str(row[entity_col])
            elif df.index.name:
                entity_id = str(df.index[idx])
            elif hasattr(df.index, '__getitem__'):
                entity_id = str(df.index[idx])
            else:
                entity_id = str(idx)

            if generate_problem_text:
                problem_text = self._generate_problem_text(split, entity_id, idx, task_type=task_type)
            else:
                problem_text = f"Predict {target_col} for entity {entity_id}"

            # For test split, ground truth is not in masked table - use unmasked_gt_map if available
            if split == "test" and unmasked_gt_map is not None:
                ground_truth = unmasked_gt_map.get(idx, None)
            else:
                ground_truth = row[target_col] if target_col and target_col in row else None
            
            problem = RelBenchProblem(
                entity_id=entity_id,
                problem_text=problem_text,
                ground_truth=ground_truth,
                split=split,
                table_row_index=idx
            )
            problems.append(problem)
        
        if self.logger:
            self.logger.info(f"Generated {len(problems)} problems from {split} split")
        
        return problems
    
    def _generate_problem_text(self, split: str, entity_id: str, idx: int, task_type: Optional[str] = None) -> str:
        """
        Generate problem text for a given entity.
        
        Args:
            split: The split name
            entity_id: The entity ID
            idx: The row index
            task_type: Optional task type ("entity_classification", "entity_regression")
        
        Returns:
            str: Generated problem text
        """
        if task_type is None:
            task_class_name = self.task.__class__.__name__.lower()
            if "regression" in task_class_name:
                task_type = "entity_regression"
            else:
                task_type = "entity_classification"

        entity_col = self.task.entity_col if hasattr(self.task, 'entity_col') else "entity"
        target_col = self.task.target_col if hasattr(self.task, 'target_col') else "target"

        if task_type == "entity_regression":
            return f"Predict {target_col} (numerical value) for {entity_col} {entity_id} at prediction time"
        else:
            return f"Predict {target_col} (class label) for {entity_col} {entity_id} at prediction time"
    
    def evaluate(self, predictions: np.ndarray, split: str = "test", table: Optional[Table] = None) -> dict:
        """
        Evaluate predictions using the task's evaluation method.
        
        Args:
            predictions: NumPy array of predictions following the order of the split table
            split: The split to evaluate on ("train", "val", or "test") (default: "test")
            table: Optional table to use for evaluation. If None, gets table for split.
        
        Returns:
            dict: Dictionary of evaluation metrics
        
        Raises:
            ValueError: If task is not loaded or predictions length doesn't match table length
        """
        if self.task is None:
            raise ValueError("No task loaded. Call load_task() first.")
        
        # Use provided table or get it
        if table is None:
            table = self.get_table(split)
        
        if len(predictions) != len(table.df):
            raise ValueError(
                f"Predictions length ({len(predictions)}) doesn't match table length ({len(table.df)})"
            )
        
        if self.logger:
            is_sampled = getattr(table, '_is_sampled', False)
            if is_sampled:
                self.logger.info(f"Evaluating predictions on {split} split (sampled table: {len(table.df)} rows)")
            else:
                self.logger.info(f"Evaluating predictions on {split} split")
        
        # Use the task's evaluate method
        # For sampled tables, we need to pass the table directly
        # RelBench's task.evaluate expects predictions aligned with table rows
        metrics = self.task.evaluate(predictions, table)
        
        if self.logger:
            self.logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def get_available_tasks(self) -> List[str]:
        """
        Get list of available tasks for the current dataset.
        
        Returns:
            List[str]: List of available task names
        
        Note:
            This requires inspecting the dataset object or using RelBench's task registry.
            Implementation may vary based on RelBench version.
        """
        if self.dataset is None:
            return []
        
        # This would depend on RelBench's API - placeholder implementation
        # In practice, you might need to check RelBench's task registry or dataset metadata
        if hasattr(self.dataset, 'available_tasks'):
            return self.dataset.available_tasks
        elif hasattr(self.dataset, 'task_names'):
            return self.dataset.task_names
        
        # Fallback: return empty list if not available
        if self.logger:
            self.logger.warning("Cannot determine available tasks from dataset")
        return []
    
    def get_schema_info(self) -> dict:
        """
        Get schema information about the database.
        
        Returns:
            dict: Dictionary containing schema information
        """
        if self.database is None:
            raise ValueError("Database not loaded")
        
        schema_info = {
            "tables": list(self.database.table_dict.keys()) if hasattr(self.database, 'table_dict') else [],
            "val_timestamp": self.dataset.val_timestamp if self.dataset else None,
            "test_timestamp": self.dataset.test_timestamp if self.dataset else None,
        }
        
        return schema_info


if __name__ == "__main__":
    # Example usage
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    try:
        # Initialize loader
        loader = RelBenchLoader("rel-amazon", task_name="user-churn", download=True, logger=logger)
        
        # Get database
        db = loader.get_database()
        logger.info(f"Database loaded: {len(loader.get_schema_info()['tables'])} tables")
        
        # Get task tables
        train_table = loader.get_train_table()
        val_table = loader.get_val_table()
        test_table = loader.get_test_table()
        
        logger.info(f"Train table: {len(train_table.df)} rows")
        logger.info(f"Val table: {len(val_table.df)} rows")
        logger.info(f"Test table: {len(test_table.df)} rows")
        
        # Get problems
        train_problems = loader.get_problems("train", num=5)
        logger.info(f"\nSample train problems:")
        for prob in train_problems[:3]:
            logger.info(f"  - {prob}")
        
        # Example evaluation (with dummy predictions)
        # test_predictions = np.random.rand(len(test_table.df))
        # metrics = loader.evaluate(test_predictions, split="test")
        # logger.info(f"Evaluation metrics: {metrics}")
        
    except ImportError as e:
        logger.error(f"RelBench not available: {e}")
    except Exception as e:
        logger.error(f"Error: {e}")

