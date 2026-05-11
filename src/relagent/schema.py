from pydantic import BaseModel, Field, field_validator
from typing import List, Union, Optional


class BinaryClassificationResult(BaseModel):
    """
    Schema for binary classification task results in RelBench (integer format).
    
    For binary classification tasks with integer ground truth (0/1), the output MUST be integers 0 or 1.
    This schema enforces strict format to prevent descriptive string outputs.
    
    Attributes:
        predictions (List[int]): List of predicted binary class labels (0 or 1) for each entity.
            Each element must be exactly 0 or 1, corresponding to one row in the test table, in order.
    """
    predictions: List[int] = Field(
        description="List of predicted binary class labels for each entity in the test set. "
                    "Each prediction MUST be an integer: 0 or 1. "
                    "Do NOT use descriptive strings like 'yes', 'no', 'will have transactions', etc. "
                    "Do NOT use booleans True/False. Use integers 0 or 1 only. "
                    "Each prediction should correspond to one row in the test table, maintaining the same order."
    )


class BooleanBinaryClassificationResult(BaseModel):
    """
    Schema for binary classification task results in RelBench (boolean format).
    
    For binary classification tasks with boolean ground truth (True/False), the output MUST be booleans True or False.
    This schema enforces strict format to prevent descriptive string outputs or integer outputs.
    
    Attributes:
        predictions (List[bool]): List of predicted binary class labels (True or False) for each entity.
            Each element must be exactly True or False, corresponding to one row in the test table, in order.
    """
    predictions: List[bool] = Field(
        description="List of predicted binary class labels for each entity in the test set. "
                    "Each prediction MUST be a boolean: True or False. "
                    "Do NOT use integers 0/1. Do NOT use descriptive strings like 'yes', 'no', 'will have transactions', etc. "
                    "Use booleans True or False only. "
                    "Each prediction should correspond to one row in the test table, maintaining the same order."
    )


class EntityClassificationResult(BaseModel):
    """
    Schema for entity classification task results in RelBench.
    
    For entity classification tasks, the output should be class labels for each entity.
    The predictions should be in the same order as the test table rows.
    
    Attributes:
        predictions (List[Union[str, int]]): List of predicted class labels for each entity.
            Each element corresponds to one row in the test table, in order.
    """
    predictions: List[Union[str, int]] = Field(
        description="List of predicted class labels for each entity in the test set. "
                    "Each prediction should be a class label (string or integer) corresponding "
                    "to one row in the test table, maintaining the same order."
    )


class EntityRegressionResult(BaseModel):
    """
    Schema for entity regression task results in RelBench.
    
    For entity regression tasks, the output should be numerical predictions for each entity.
    The predictions should be in the same order as the test table rows.
    
    Attributes:
        predictions (List[float]): List of predicted numerical values for each entity.
            Each element corresponds to one row in the test table, in order.
    """
    predictions: List[float] = Field(
        description="List of predicted numerical values for each entity in the test set. "
                    "Each prediction should be a float value corresponding to one row in the "
                    "test table, maintaining the same order."
    )


class RelBenchTaskResult(BaseModel):
    """
    Generic schema for RelBench task results that can handle different task types.

    Attributes:
        task_type (str): Type of RelBench task - one of: "entity_classification", "entity_regression"
        predictions (List[Union[str, int, float]]): List of predictions matching the task type.
    """
    task_type: str = Field(
        description="Type of RelBench task. Must be one of: 'entity_classification', 'entity_regression'"
    )
    predictions: List[Union[str, int, float]] = Field(
        description="List of predictions for the test set. Format depends on task_type: "
                    "- entity_classification: class labels (str or int) "
                    "- entity_regression: numerical values (float)"
    )

    @field_validator('task_type')
    @classmethod
    def validate_task_type(cls, v):
        allowed_types = ['entity_classification', 'entity_regression']
        if v not in allowed_types:
            raise ValueError(f'task_type must be one of {allowed_types}')
        return v
