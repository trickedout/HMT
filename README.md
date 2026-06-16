# HMT

Hierarchical Memory Tree (HMT) for web agents.

HMT stores successful web trajectories as a three-level memory:

```text
Intent -> Stage -> Action pattern
```

At inference time, HMT retrieves a related intent, selects the current stage, retrieves action patterns under that stage, and grounds the selected pattern onto current-page candidate elements. GPT-4 is used for structured decisions when enabled. Qwen embedding and reranking models are loaded locally through `transformers`.

## Setup

```bash
conda create -n hmt python=3.10 -y
conda activate hmt
pip install -e .
cp .env.example .env
```

Set `OPENAI_API_KEY` before using GPT-4 calls. Qwen models are loaded from the local Hugging Face cache or downloaded by `transformers` according to your environment.

## Files

```text
hmt/                    core HMT code
prompts/                prompts for construction and inference
schemas/                structured-output constraints for GPT calls
configs/                default and benchmark configs
scripts/                command-line entry points
mind2web/pipeline.py    Mind2Web wrapper
webarena/pipeline.py    WebArena wrapper
examples/               HMT memory examples
```

## Mind2Web

```bash
python scripts/prepare_mind2web.py \
  --source /path/to/Mind2Web/data \
  --output-dir data/mind2web_jsonl

python scripts/run_mind2web.py \
  --config configs/real_mind2web.yaml \
  --split train \
  --mode build-memory \
  --train-path /path/to/Mind2Web/data/train \
  --memory outputs/mind2web/T_mem_train.jsonl

python scripts/run_mind2web.py \
  --config configs/real_mind2web.yaml \
  --split cross_website \
  --mode evaluate \
  --data-path /path/to/Mind2Web/data/test_website \
  --memory outputs/mind2web/T_mem_train.jsonl \
  --output-dir outputs/mind2web/cross_website
```

## WebArena

```bash
python scripts/materialize_webarena_order.py \
  --task-config-dir /path/to/webarena/config_files \
  --output data/webarena_task_order.jsonl

python scripts/run_webarena.py \
  --config configs/real_webarena.yaml \
  --domain reddit \
  --task-order data/webarena_task_order.jsonl \
  --task-config-dir /path/to/webarena/config_files \
  --backend browsergym \
  --output-dir outputs/webarena/reddit
```

By default the WebArena runner resets memory per domain and inserts only successful episodes online.

## Citation

```bibtex
@misc{tan2026hmt,
  title = {Enhancing Web Agents with a Hierarchical Memory Tree},
  author = {Tan, Yunteng and Gao, Zhi and Wu, Xinxiao},
  year = {2026},
  eprint = {2603.07024},
  archivePrefix = {arXiv},
  doi = {10.48550/arXiv.2603.07024},
  url = {https://arxiv.org/abs/2603.07024}
}
```
