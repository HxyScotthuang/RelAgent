# RelAgent: LLM Agents as Data Scientists for Relational Learning

An LLM-based scientist agent for solving predictve tasks on database via iterative SQL feature engineering. The agent proposes SQL feature queries, selects a tree-based learner from a menu of seven options, and iterates based on validation feedback — all without writing any training code.

## Overview

RelAgent targets RelBench and 4DBInfer entity classification and regression tasks. At each turn the agent:

1. Explores the DuckDB schema with SQL tools
2. Proposes SQL feature queries and a model choice (one of seven learners: LightGBM GBDT/RF/DART/GOSS, XGBoost/XGBoost-DART, CatBoost)
3. Receives validation metrics + workspace diagnostics, then refines features

## Installation

**Python 3.10+** is required.

```bash
git clone <repository-url>
cd relagent

# (Recommended) create a virtual environment
python -m venv .venv && source .venv/bin/activate   # or: conda create -n relagent python=3.11

pip install -r requirements.txt

# Tree-based learners (install separately — not on PyPI as a bundle)
pip install lightgbm xgboost catboost
```

The key runtime dependencies are:

| Package | Purpose |
|---------|---------|
| `camel-ai` | Agent loop and tool-use framework |
| `litellm` | Unified LLM API (OpenAI, vLLM, Anthropic, …) |
| `relbench` | Benchmark datasets and task loaders |
| `duckdb` | In-process SQL engine for feature queries |
| `lightgbm`, `xgboost`, `catboost` | Tree-based learners |
| `shap` | Feature importance |

## Usage

### API-hosted models (e.g. GPT-5.2 via LiteLLM proxy)

```bash
export LITELLM_API_KEY=your_key_here
export LITELLM_API_BASE=https://your-litellm-proxy/

python src/relagent/main.py \
  --dataset rel-amazon \
  --task user-churn \
  --model gpt-5.2 \
  --max_turns 60 \
  --artifact_dir artifacts/my_run
```

### Locally-hosted models (vLLM)

Start a vLLM server, then point the agent at it:

```bash
# Start vLLM server (separate terminal)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B-Instruct \
  --port 8000 \
  --tensor-parallel-size 2

# Run agent
python src/relagent/main.py \
  --dataset rel-amazon \
  --task user-churn \
  --model Qwen/Qwen3-32B-Instruct \
  --port 8000 \
  --max_turns 60
```

### CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `rel-amazon` | RelBench dataset name |
| `--task` | `user-churn` | RelBench task name |
| `--model` | `gpt-5.2` | LiteLLM / vLLM model string |
| `--max_turns` | 60 | Max agent turns |
| `--artifact_dir` | `artifacts/scientist_runs` | Output directory |
| `--temperature` | 1.0 | LLM sampling temperature |
| `--max_tokens` | 8000 | Max LLM output tokens |
| `--step_timeout` | 900 | Per-turn timeout (seconds) |
| `--sql_query_timeout` | 300 | Per SQL feature query timeout (seconds) |
| `--api_base` | — | LiteLLM proxy base URL (or `LITELLM_API_BASE` env) |
| `--api_key` | — | LiteLLM API key (or `LITELLM_API_KEY` env) |
| `--port` | 8000 | vLLM server port |
| `--eval_sample` | — | Sample N entities from val/test splits |
| `--log_level` | `INFO` | Log verbosity |


## Outputs

Each run saves artifacts to `artifact_dir/<dataset>_<task>_<timestamp>/`:

| File | Contents |
|------|----------|
| `trials.jsonl` | Per-trial metrics, SQL queries, model config |
| `best_program.json` | Best SQL + model choice found |
| `test_results.json` | Final evaluation on the held-out test split |
| `conversation_trace.jsonl` | Full agent conversation trace |
| `eval_workspace.db` | DuckDB workspace with all trial predictions |
| `summary.md` | Human-readable run summary |

