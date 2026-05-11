"""
Evaluator for RelBench tasks.

This module provides the RelBenchEvaluator class for collecting predictions,
evaluating them against ground truth, and generating evaluation metrics using
the RelBench framework.
"""
from colorama import Fore, Style, Back
import logging
import numpy as np
from typing import Dict, List, Any, Optional

class RelBenchEvaluator:
    """
    Evaluator for RelBench tasks.
    Groups predictions, calls RelBench for evaluation, and generates feedback.
    """
    
    def __init__(self, loader: Optional[Any] = None, logger: Optional[logging.Logger] = None, 
                 eval_sample: Optional[int] = None, task_type: Optional[str] = None):
        """
        Initialize the RelBenchEvaluator with an optional RelBenchLoader and logger.
        
        Args:
            loader: Optional RelBenchLoader instance for calling relbench.evaluate()
            logger: Optional logger instance for logging evaluation results
            eval_sample: Maximum number of rows to sample from test/val tables (default: None)
            task_type: Task type ("entity_classification", "entity_regression")
        """
        self.loader = loader
        self.logger = logger or logging.getLogger("relbench_evaluator")
        self.eval_sample = eval_sample
        self.task_type = task_type
        self.predictions: Dict[int, Any] = {}  # table_row_index -> prediction
        self.results: List[Dict[str, Any]] = []      # list of individual results

    def add_prediction(self, index: int, prediction: Any, ground_truth: Any = None):
        """
        Add a prediction for a specific row index.
        
        Args:
            index: The row index in the task table
            prediction: The predicted value (could be a string, float, dict, or list)
            ground_truth: The ground truth value (optional)
        """
        # If prediction is a dict with 'predictions' key (common from SQLSolver), extract the value
        if isinstance(prediction, dict) and 'predictions' in prediction:
            preds = prediction['predictions']
            if isinstance(preds, list) and len(preds) > 0:
                actual_prediction = preds[0]
            else:
                actual_prediction = preds
        else:
            actual_prediction = prediction
            
        self.predictions[index] = actual_prediction
        
        # Store individual result
        result = {
            "index": index,
            "prediction": actual_prediction,
            "ground_truth": ground_truth,
        }
        
        self.results.append(result)
        
        # Avoid log spam: only log individual predictions at DEBUG level.
        if self.logger and self.logger.isEnabledFor(logging.DEBUG):
            gt_str = f" (GT: {ground_truth})" if ground_truth is not None else ""
            self.logger.debug(
                f"Prediction for index {index}: {Fore.CYAN}{actual_prediction}{Style.RESET_ALL}{gt_str}"
            )

    def evaluate_all(self, split: str = "val", eval_sample: Optional[int] = None) -> Dict[str, float]:
        """
        Group all collected predictions in the form RelBench desires and call evaluation.
        
        Args:
            split: The split to evaluate on ("train", "val", or "test")
            eval_sample: Maximum number of rows to sample (if None, uses self.eval_sample)
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics from RelBench
        """
        if self.loader is None:
            self.logger.error("No RelBenchLoader provided to evaluator. Cannot call RelBench evaluation.")
            return {}

        # Use provided eval_sample or fall back to instance variable
        sample_size = eval_sample if eval_sample is not None else self.eval_sample

        try:
            # For test split, use unmasked table for evaluation (ground truth labels visible)
            # This is safe because evaluation happens AFTER predictions are made
            if split == "test":
                # Get unmasked test table for evaluation
                unmasked_table = self.loader.task.get_table("test", mask_input_cols=False)
                
                # Apply sampling if needed (matching the sampling used when loading problems)
                if sample_size is not None and len(unmasked_table.df) > sample_size:
                    # Need to replicate the same sampling that was used when loading problems
                    # For now, get the masked sampled table to see which indices were sampled
                    masked_sampled_table = self.loader.get_table("test", eval_sample=sample_size)
                    if hasattr(masked_sampled_table, '_sampled_indices'):
                        # Use the same sampled indices for unmasked table
                        sampled_indices = masked_sampled_table._sampled_indices
                        sampled_df = unmasked_table.df.iloc[sampled_indices].reset_index(drop=True).copy()
                        from .utils import SampledTable
                        table = SampledTable(sampled_df, unmasked_table, sampled_indices, len(unmasked_table.df))
                        if self.logger:
                            self.logger.info(f"Using unmasked test table for evaluation (sampled: {sample_size} rows)")
                    else:
                        # Fallback: use full unmasked table
                        table = unmasked_table
                        if self.logger:
                            self.logger.warning("Could not replicate sampling, using full unmasked test table")
                else:
                    table = unmasked_table
                    if self.logger:
                        self.logger.info("Using unmasked test table for evaluation (ground truth labels visible)")
            else:
                # For train/val splits, labels are already visible (these splits are not masked by RelBench)
                # Only test split is masked to prevent data leakage
                table = self.loader.get_table(split, eval_sample=sample_size)
        except Exception as e:
            self.logger.error(f"Failed to get table for split {split}: {e}")
            return {}
            
        num_rows = len(table.df)
        
        if not self.predictions:
            self.logger.warning("No predictions collected. Nothing to evaluate.")
            return {}

        # Check if table was sampled
        is_sampled = getattr(table, '_is_sampled', False)
        if is_sampled:
            original_size = getattr(table, '_original_size', num_rows)
            if self.logger:
                self.logger.info(f"Evaluating on sampled table: {num_rows} rows (from {original_size} total)")

        # Group predictions in order of the table rows as required by RelBench
        all_preds = []
        missing_count = 0
        is_classification = (self.task_type == "entity_classification")
        is_regression = (self.task_type == "entity_regression")
        is_binary = is_classification
        neg_label, pos_label = 0, 1

        def _match_binary_label(pred_val, label):
            """Check if a prediction matches a binary label (handles str, bool, numeric).
            
            Safely handles non-scalar inputs (empty lists, numpy arrays, None)
            that would otherwise cause 'ambiguous truth value' errors with numpy.
            """
            # Guard: non-scalar predictions can never match a label
            if pred_val is None:
                return False
            if isinstance(pred_val, (list, tuple)):
                return False
            if isinstance(pred_val, np.ndarray):
                return False
            # Exact Python equality (handles strings like 't'/'f')
            try:
                if pred_val == label:
                    return True
            except (ValueError, TypeError):
                pass  # numpy ambiguous truth value — fall through
            # String-insensitive match
            if isinstance(pred_val, str) and isinstance(label, str):
                return pred_val.strip().lower() == label.strip().lower()
            # Numeric comparison (handles int/float/bool/numpy scalars)
            try:
                return abs(float(pred_val) - float(label)) < 1e-9
            except (ValueError, TypeError):
                return False

        for i in range(num_rows):
            if i in self.predictions:
                pred = self.predictions[i]
                if is_binary:
                    # Binary classification: accept either:
                    # - hard labels matching {neg_label,pos_label} (mapped to {0.0,1.0})
                    # - numeric scores/probabilities in [0,1] (passed through)
                    # This lets AUROC / average_precision use ranking signal from probabilities.
                    if _match_binary_label(pred, pos_label):
                        all_preds.append(1.0)
                    elif _match_binary_label(pred, neg_label):
                        all_preds.append(0.0)
                    else:
                        # Try interpret as a probability score.
                        try:
                            v = float(pred)
                            if np.isnan(v) or np.isinf(v):
                                raise ValueError("non-finite")
                            # Keep scores bounded; if a model emits logits, clipping is safer
                            # than treating as a hard negative.
                            if v < 0.0:
                                v = 0.0
                            elif v > 1.0:
                                v = 1.0
                            all_preds.append(v)
                        except (ValueError, TypeError):
                            all_preds.append(0.0)
                elif is_regression:
                    try:
                        val = float(pred)
                    except (ValueError, TypeError):
                        val = 0.0
                    all_preds.append(val)
                else:
                    all_preds.append(pred)
            else:
                all_preds.append(0)
                missing_count += 1
        
        if missing_count > 0:
            self.logger.warning(f"Missing predictions for {missing_count}/{num_rows} rows. Used placeholder values.")

        # Convert to numpy array as RelBench expects
        preds_array = np.array(all_preds)

        try:
            # Call RelBench evaluation with the (possibly sampled) table
            metrics = self.loader.evaluate(preds_array, split=split, table=table)
            
            # Log feedback results
            self.logger.info(f"\n{Back.BLUE}{Fore.WHITE} RelBench Evaluation Results ({split}) {Style.RESET_ALL}")
            if is_sampled:
                self.logger.info(f"{Fore.YELLOW}Note: Evaluated on sampled subset ({num_rows} rows){Style.RESET_ALL}")
            for metric, value in metrics.items():
                self.logger.info(f"{Fore.GREEN}{metric}: {value:.4f}{Style.RESET_ALL}")
                
            return metrics
        except Exception as e:
            self.logger.error(f"Error during RelBench evaluation: {e}")
            if self.logger:
                import traceback
                self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return {}

    def get_individual_results(self) -> List[Dict[str, Any]]:
        """
        Generate and return individual results.
        
        Returns:
            List[Dict]: List of dictionaries containing index, prediction, ground_truth, and is_correct
        """
        return self.results

    def print_individual_summary(self):
        """
        Print a summary of individual results.
        """
        if not self.results:
            self.logger.info("No results to summarize.")
            return

        total = len(self.results)
        with_gt = [r for r in self.results if r.get("ground_truth") is not None]
        correct = [r for r in with_gt if r.get("is_correct") is True]
        
        self.logger.info(f"\n{Back.WHITE}{Fore.BLACK} Individual Prediction Summary {Style.RESET_ALL}")
        self.logger.info(f"Total predictions: {total}")
        
        if with_gt:
            accuracy = (len(correct) / len(with_gt)) * 100
            self.logger.info(f"Predictions with Ground Truth: {len(with_gt)}")
            self.logger.info(f"{Fore.GREEN}Exact Matches: {len(correct)}{Style.RESET_ALL}")
            self.logger.info(f"{Fore.CYAN}Exact Match Accuracy: {accuracy:.1f}%{Style.RESET_ALL}")
        else:
            self.logger.info("No ground truth available for individual comparison.")
