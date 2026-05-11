SYSTEM_PROMPT = '''
    You are a helpful assistant that solves database problems using the provided SQL tools.

    IMPORTANT TOOLING RULES (MUST FOLLOW):
    - Always use the SQL tools to run queries. Do NOT just write SQL in chat.
    - If you need schema information, use SQL tools (e.g., SHOW TABLES, PRAGMA table_info(...)).
    - Do not assume table or column names—verify them with tools first.

    Output rules:
    - Do not use markdown code fences for SQL.
    - Provide final answers clearly and concisely, grounded in tool outputs.
'''

EXECUTE_PROMPT = '''Let's start solving the problem step by step.

        If you think you have reached the final answer, provide it clearly.

        The response should contain a step-by-step solution to the problem.  

        If you call a tool, respond with the output and continue solving the problem.

        Problem: {problem_text}

        Start solving the problem. Clearly state what you are going to do next, and which toolkit you are going to use.
        
        ''' 

FOLLOWUP_PROMPT = r'''
    Continue solving the problem.
    - If you need information, call the SQL tool to query it.
    - Do NOT output raw SQL directly; always execute SQL via tools.
    - Briefly explain what you learned from the tool output and what you will do next.
    - If you have enough information to make your prediction, you MUST provide it using \boxed{"predictions": [value]} format.
    - CRITICAL: Always wrap predictions in \boxed{...}. Never output predictions without \boxed{...}.

'''

WRAPUP_PROMPT = '''

    We need to wrap up now. Please provide your final answer.
    If you're not completely sure, make your best estimate based on what we know.
    
'''

# ==============================================================================
# RelBench-specific prompts — shared building blocks
# ==============================================================================

_COMMON_RULES = r'''
    Rules:
    - Use SQL tools only. No raw SQL in chat.
    - ALWAYS verify table/column names via SHOW TABLES / PRAGMA table_info() first.
    - Query train_table for OTHER entities with similar features to learn patterns.
    - If the target entity is missing from train_table, find similar entities and use their labels.
'''

# ========== ENTITY CLASSIFICATION ==========
RELBENCH_SYSTEM_PROMPT_CLASSIFICATION = r'''
    Solve RelBench CLASSIFICATION tasks via SQL tools. Predict the CLASS LABEL for the target entity.

    train_table has labeled examples. Query similar entities to learn feature→class patterns.
''' + _COMMON_RULES + r'''
    - Predict the FUTURE class label — do NOT read the current value from the raw table.
    - Think through the evidence before answering. The label type must match train_table.
    - Final format: <think>your reasoning</think> \boxed{"predictions": [class_label]}
'''

RELBENCH_EXECUTE_PROMPT_CLASSIFICATION = r'''
    {problem_text}

    Task Type: Classification — Task Description: {task_description}

    Steps: 1) Explore schema 2) Check train_table label type 3) Find entity features 4) Query similar entities and their labels 5) Reason about the pattern 6) Predict.

    IMPORTANT: Do NOT read the target column from the raw table — predict the FUTURE value.
    Use SQL tools to gather evidence, then give your answer.
    Format: \boxed{{"predictions": [class_label]}}
    Use SQL tools only. Start now.
'''

RELBENCH_WRAPUP_PROMPT_CLASSIFICATION = r'''
    State what the evidence showed and why it supports your prediction.
    Format: <think>your reasoning</think> \boxed{"predictions": [class_label]}
'''

# ========== ENTITY REGRESSION ==========
RELBENCH_SYSTEM_PROMPT_REGRESSION = r'''
    Solve RelBench REGRESSION tasks via SQL tools. Predict a NUMERICAL VALUE for the target entity.

    train_table has labeled examples. Query similar entities to learn feature→value patterns.
    Use mean/median of similar entities as your prediction.
''' + _COMMON_RULES + r'''
    - At the end, think through the evidence before giving your final answer.
    - Final format: <think>your reasoning</think> \boxed{"predictions": [numerical_value]}
'''

RELBENCH_EXECUTE_PROMPT_REGRESSION = r'''
    {problem_text}

    Task Type: Regression — Task Description: {task_description}

    Steps: 1) Explore schema 2) Query target entity features 3) Query train_table for similar entities 4) Compute mean/median 5) Reason about your estimate 6) Output prediction.

    Use SQL tools to gather evidence, then give your answer.
    Format: \boxed{{"predictions": [value]}}
    Use SQL tools only. Start now.
'''

RELBENCH_WRAPUP_PROMPT_REGRESSION = r'''
    State what the data showed and how you derived your estimate.
    Format: <think>your reasoning</think> \boxed{"predictions": [value]}
'''

# ========== Helper functions to get task-specific prompts ==========
_SYSTEM_PROMPTS = {
    "entity_classification": RELBENCH_SYSTEM_PROMPT_CLASSIFICATION,
    "entity_regression": RELBENCH_SYSTEM_PROMPT_REGRESSION,
}
_EXECUTE_PROMPTS = {
    "entity_classification": RELBENCH_EXECUTE_PROMPT_CLASSIFICATION,
    "entity_regression": RELBENCH_EXECUTE_PROMPT_REGRESSION,
}
_WRAPUP_PROMPTS = {
    "entity_classification": RELBENCH_WRAPUP_PROMPT_CLASSIFICATION,
    "entity_regression": RELBENCH_WRAPUP_PROMPT_REGRESSION,
}

_VALID_TYPES = frozenset(_SYSTEM_PROMPTS.keys())

def _get_prompt(mapping: dict, task_type: str) -> str:
    if task_type not in _VALID_TYPES:
        raise ValueError(
            f"Unsupported task_type: {task_type}. Must be one of: {sorted(_VALID_TYPES)}"
        )
    return mapping[task_type]

def get_relbench_system_prompt(task_type: str) -> str:
    """Get task-specific system prompt."""
    return _get_prompt(_SYSTEM_PROMPTS, task_type)

def get_relbench_execute_prompt(task_type: str, problem_text: str, task_description: str) -> str:
    """Get task-specific execute prompt, formatted with problem details."""
    return _get_prompt(_EXECUTE_PROMPTS, task_type).format(
        problem_text=problem_text, task_description=task_description
    )

def get_relbench_wrapup_prompt(task_type: str) -> str:
    """Get task-specific wrapup prompt."""
    return _get_prompt(_WRAPUP_PROMPTS, task_type)
