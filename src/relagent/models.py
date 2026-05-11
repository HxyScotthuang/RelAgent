"""
Model name enums and utilities following CAMEL-AI naming conventions.

This module provides a consistent way to name and reference all supported models,
using enums similar to CAMEL-AI's ModelType enum.
"""
from enum import Enum
from typing import Optional, Tuple
from camel.types import ModelPlatformType, ModelType


class ModelName(str, Enum):
    """
    Enumeration of all supported model names.
    Follows CAMEL-AI naming conventions with consistent UPPER_SNAKE_CASE.
    
    Values are the canonical string names used in command-line and config.
    """
    # OpenAI Models
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_2 = "gpt-5.2"
    
    # vLLM Models - Qwen series
    QWEN3_4B_INSTRUCT = "qwen3-4b-instruct"
    QWEN3_32B = "qwen3-32b"

    # vLLM Models - Llama series
    LLAMA_3_2_3B_INSTRUCT = "llama-3.2-3b-instruct"
    
    @classmethod
    def from_string(cls, model_str: str) -> 'ModelName':
        """
        Convert a string model name to ModelName enum.
        
        Args:
            model_str: String model name (case-insensitive, must match exact enum value)
            
        Returns:
            ModelName enum value if matched
            
        Raises:
            ValueError: If model_str doesn't match any enum value, with list of valid options
        """
        model_lower = model_str.lower().strip()
        
        # Exact match only (case-insensitive)
        for model_enum in cls:
            if model_enum.value.lower() == model_lower:
                return model_enum
        
        # No match found - raise error with helpful message
        valid_models = ", ".join([m.value for m in cls])
        raise ValueError(
            f"Invalid model name '{model_str}'. "
            f"Valid model names are: {valid_models}. "
            f"Use the exact enum value (case-insensitive)."
        )
    
    def get_camel_model_type(self) -> Tuple[ModelPlatformType, ModelType, Optional[str]]:
        """
        Get CAMEL-AI ModelPlatformType and ModelType for OpenAI models.
        
        Returns:
            Tuple of (ModelPlatformType, ModelType, None)
            
        Raises:
            ValueError: If model is not an OpenAI model
        """
        if self == ModelName.GPT_4O_MINI:
            return (ModelPlatformType.OPENAI, ModelType.GPT_4O_MINI, None)
        elif self == ModelName.GPT_4O:
            return (ModelPlatformType.OPENAI, ModelType.GPT_4O, None)
        elif self == ModelName.GPT_5_NANO:
            return (ModelPlatformType.OPENAI, ModelType.GPT_5_NANO, None)
        elif self == ModelName.GPT_5_2:
            # CAMEL's ModelType may not include this newer name yet; LiteLLM path is recommended.
            return (ModelPlatformType.OPENAI, ModelType.GPT_5_NANO, "gpt-5.2")
        else:
            raise ValueError(f"{self} is not an OpenAI model. Use get_vllm_model_name() for vLLM models.")
    
    def get_vllm_model_name(self) -> str:
        """
        Get the HuggingFace model name/identifier for vLLM models.
        
        Returns:
            HuggingFace model identifier string
            
        Raises:
            ValueError: If model is not a vLLM model
        """
        vllm_model_mapping = {
            # Qwen3-4B-Instruct maps to the actual HuggingFace identifier
            ModelName.QWEN3_4B_INSTRUCT: "Qwen/Qwen3-4B-Instruct-2507",
            ModelName.QWEN3_32B: "Qwen/Qwen3-32B",
            ModelName.LLAMA_3_2_3B_INSTRUCT: "meta-llama/Llama-3.2-3B-Instruct",
        }
        
        if self not in vllm_model_mapping:
            raise ValueError(f"{self} is not a vLLM model. Use get_camel_model_type() for OpenAI models.")
        
        return vllm_model_mapping[self]
    
    def is_openai_model(self) -> bool:
        """Check if this is an OpenAI model."""
        return self in [ModelName.GPT_4O_MINI, ModelName.GPT_4O, ModelName.GPT_5_NANO, ModelName.GPT_5_2]
    
    def is_vllm_model(self) -> bool:
        """Check if this is a vLLM model."""
        return self in [ModelName.QWEN3_4B_INSTRUCT, ModelName.QWEN3_32B, ModelName.LLAMA_3_2_3B_INSTRUCT]
    
    def __str__(self) -> str:
        """Return the canonical string value."""
        return self.value

