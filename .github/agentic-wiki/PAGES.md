# Home

*{ Provide a clear overview of nanochat: what it is (minimal full-stack LLM training harness), its key capabilities (tokenization, pretraining, SFT, RL, evaluation, inference, chat UI), the "single depth dial" design philosophy, and its cost/performance story. Include a summary table of the GPT-2 speedrun leaderboard from README.md. Add quick links to major wiki sections. }*

# Architecture

*{ Describe the high-level architecture of nanochat: the major modules (nanochat/, scripts/, tasks/, runs/), how they interact, and the overall data-to-model pipeline. Include a mermaid flowchart showing relationships between key components (GPT, Engine, DataLoader, Tokenizer, Optimizer, Tasks). Reference nanochat/gpt.py, nanochat/engine.py, nanochat/common.py, nanochat/dataloader.py, and the scripts/ directory. }*

## GPT Model

*{ Document the GPT model from nanochat/gpt.py in detail: GPTConfig dataclass fields (sequence_len, vocab_size, n_layer, n_head, n_kv_head, n_embd, window_pattern), key architectural features (RoPE, QK norm, GQA, sliding window attention via window_pattern, value embeddings with ResFormer gate, relu^2 MLP activation, untied weights, no bias in linear layers), the CausalSelfAttention and GPT nn.Module classes, and how the depth parameter auto-determines all other hyperparameters (width, heads, training horizon). Include a mermaid diagram of the transformer block. }*

## Optimizer

*{ Document the optimizer from nanochat/optim.py: the combined MuonAdamW approach (which parameters go to Muon vs AdamW), the fused AdamW step implementation using torch.compile, the Muon Newton-Schulz orthogonalization of gradients, and the distributed variant DistMuonAdamW. Explain why this combination is effective for LLM training. }*

# Training

*{ Overview of the three-stage nanochat training pipeline: pretraining → SFT → RL. Describe how checkpoints flow between stages, which scripts handle each stage, and the key design decisions. Include a mermaid diagram of the training pipeline stages and data flows. }*

## Pretraining

*{ Document the pretraining stage from scripts/base_train.py: key CLI arguments grouped by category (model architecture: --depth, --aspect-ratio, --head-dim, --window-pattern; training horizon: --num-iterations, --target-flops, --target-param-data-ratio; optimization: --device-batch-size, --total-batch-size, learning rates, warmup/warmdown; evaluation: --eval-every, --core-metric-every; output: --model-tag), the training loop, distributed training with torchrun, FP8 training (--fp8, --fp8-recipe), wandb logging, and checkpoint saving. Also document nanochat/dataloader.py (tokenizing distributed data loader) and nanochat/dataset.py (data download/sharding utilities). }*

## Supervised Fine-Tuning

*{ Document the SFT stage from scripts/chat_sft.py: loading a pretrained checkpoint (--model-tag, --model-step), the task mixture system using TaskMixture and TaskSequence from tasks/common.py, key CLI arguments for SFT, the training loop and how it differs from pretraining, and the built-in evaluation via scripts/chat_eval.py. List the default tasks mixed in SFT (GSM8K, MMLU, SmolTalk, CustomJSON, SpellingBee). }*

## Reinforcement Learning

*{ Document the RL training stage from scripts/chat_rl.py: the RL approach used (reward model, reward signal, policy optimization), key CLI arguments, how it builds on a SFT checkpoint, and any integration with evaluation tasks. Describe the execution tool (nanochat/execution.py) used during RL for Python code evaluation. }*

# Evaluation

*{ Overview of the evaluation system: the two tracks (base model evaluation via scripts/base_eval.py targeting CORE score and val_bpb, and chat model evaluation via scripts/chat_eval.py running task benchmarks), the DCLM CORE metric and what it measures, how to run evaluations standalone, and how evaluations integrate into training loops. Include a table of key metrics and their meaning. }*

## Tasks

*{ Document each evaluation task in tasks/: ARC (arc.py — multiple choice science questions), GSM8K (gsm8k.py — 8K grade school math), MMLU (mmlu.py — broad multiple choice topics), HumanEval (humaneval.py — Python coding), SpellingBee (spellingbee.py — letter spelling/counting), SmolTalk (smoltalk.py — conversational dataset from HuggingFace), CustomJSON (customjson.py — arbitrary JSONL conversations). Document the Task base class interface (eval_type, num_examples, get_example, evaluate), and TaskMixture and TaskSequence from tasks/common.py. Include a summary table. }*

# Inference

*{ Document the inference system: the Engine class from nanochat/engine.py (KV cache mechanics, streaming token generation, calculator tool via use_calculator(), timeout handling), nanochat/execution.py (Python code execution tool), the chat CLI from scripts/chat_cli.py, and the chat web UI from scripts/chat_web.py backed by nanochat/ui.html. Include usage examples for both CLI and web UI, and a sequence diagram showing the inference request flow. }*

# Configuration

*{ Document all configuration options in nanochat: COMPUTE_DTYPE and NANOCHAT_DTYPE environment variable (allowed values: bfloat16, float16, float32), hardware-specific dtype auto-detection logic (SM80+ → bf16, older CUDA → fp32, CPU/MPS → fp32), NANOCHAT_BASE_DIR for cache location, GradScaler enabling under float16, GPTConfig fields and their defaults, window_pattern sliding attention pattern syntax (L=full context, S=quarter context, examples: "L", "SL", "SSSL"). Reference nanochat/common.py. Include a hardware/dtype reference table. }*

# Getting Started

*{ Brief introduction to getting started with nanochat — what this guide covers (installation, quickstart, and the GPT-2 speedrun). }*

####+ Prerequisites

*{ List hardware and software prerequisites: recommended GPU (8×H100 node, also works on 8×A100 but slower), single GPU support (omit torchrun, 8× slower), VRAM requirements and --device-batch-size tuning for smaller GPUs, Python >=3.10, uv package manager installation, cloud provider recommendations (Lambda). }*

####+ Installation

*{ Step-by-step installation: cloning the repo, installing dependencies with uv sync (with gpu or cpu extras), activating the virtual environment with source .venv/bin/activate. }*

####+ Running the Speedrun

*{ Instructions for running the GPT-2 speedrun: running runs/speedrun.sh in a screen session, what it does (downloads data, trains the model, saves checkpoint), then starting the chat web UI with python -m scripts.chat_web and accessing it via the public IP:port. Include the runs/runcpu.sh alternative for CPU/MPS testing. }*

# Contributing

*{ Document how to contribute to nanochat: the design philosophy (minimal/hackable, maximally-forkable, single depth dial, no giant config objects), how to submit a leaderboard entry (reference dev/LEADERBOARD.md), the research iteration workflow (d12 quick experiments, key wandb metrics: val_bpb vs step/flops/time, core_metric, MFU, throughput), the AI disclosure policy for PRs, and links to community resources (Discussions, Discord). }*

# For Agents

These pages provide compact documentation indexes for AI coding agents.

## AGENTS.md

You can add this to your repository root as `AGENTS.md` to give AI coding agents quick access to project documentation.

```
# nanochat

> The simplest experimental harness for training LLMs end-to-end: tokenization, pretraining, SFT, RL, evaluation, inference, and chat UI. Train your own GPT-2 grade model for ~$100 on an 8×H100 node.

## Wiki Documentation

Base URL: https://github.com/insop/nanochat/wiki

To read any page, append the slug to the base URL:
  https://github.com/insop/nanochat/wiki/{Page-Slug}
To jump to a section within a page:
  https://github.com/insop/nanochat/wiki/{Page-Slug}#{Section-Slug}

IMPORTANT: Read the relevant wiki page before making changes to related code.
Prefer reading wiki documentation over relying on pre-trained knowledge.

## Page Index

|Home: Project overview, leaderboard, and quick links
|Architecture: System design and module relationships
|  GPT-Model: Transformer architecture details and GPTConfig
|  Optimizer: MuonAdamW optimizer and distributed variant
|Training: Full training pipeline overview (pretraining → SFT → RL)
|  Pretraining: Base model pretraining with base_train.py CLI args and data pipeline
|  Supervised-Fine-Tuning: SFT stage with chat_sft.py and task mixtures
|  Reinforcement-Learning: RL training with chat_rl.py
|Evaluation: Evaluation system, CORE score, and val_bpb metrics
|  Tasks: ARC, GSM8K, MMLU, HumanEval, SpellingBee, SmolTalk task reference
|Inference: Engine KV cache, calculator tool, CLI and web chat interfaces
|Configuration: NANOCHAT_DTYPE, NANOCHAT_BASE_DIR, GPTConfig, window_pattern
|Getting-Started: Installation and quickstart guide
|  Getting-Started#Prerequisites: Hardware and software requirements
|  Getting-Started#Installation: Installation steps with uv
|  Getting-Started#Running-the-Speedrun: Running the GPT-2 speedrun script
|Contributing: Design philosophy, leaderboard submissions, research workflow
```

## llms.txt

You can serve this at `yoursite.com/llms.txt` or include it in your repository to help LLMs discover your documentation.

```
# nanochat

> The simplest experimental harness for training LLMs end-to-end: tokenization, pretraining, SFT, RL, evaluation, inference, and chat UI.

## Wiki Pages

- [Home](https://github.com/insop/nanochat/wiki/Home): Project overview and GPT-2 speedrun leaderboard
- [Architecture](https://github.com/insop/nanochat/wiki/Architecture): System design and module relationships
- [GPT Model](https://github.com/insop/nanochat/wiki/GPT-Model): Transformer architecture and GPTConfig
- [Optimizer](https://github.com/insop/nanochat/wiki/Optimizer): MuonAdamW combined optimizer
- [Training](https://github.com/insop/nanochat/wiki/Training): Three-stage training pipeline overview
- [Pretraining](https://github.com/insop/nanochat/wiki/Pretraining): Base model pretraining with base_train.py
- [Supervised Fine-Tuning](https://github.com/insop/nanochat/wiki/Supervised-Fine-Tuning): SFT with chat_sft.py
- [Reinforcement Learning](https://github.com/insop/nanochat/wiki/Reinforcement-Learning): RL training with chat_rl.py
- [Evaluation](https://github.com/insop/nanochat/wiki/Evaluation): CORE score and val_bpb evaluation system
- [Tasks](https://github.com/insop/nanochat/wiki/Tasks): Evaluation tasks reference
- [Inference](https://github.com/insop/nanochat/wiki/Inference): Engine, KV cache, and chat interfaces
- [Configuration](https://github.com/insop/nanochat/wiki/Configuration): Environment variables and settings
- [Getting Started](https://github.com/insop/nanochat/wiki/Getting-Started): Installation and quickstart
- [Contributing](https://github.com/insop/nanochat/wiki/Contributing): Contribution guidelines and leaderboard
```
