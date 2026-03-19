# Home

*{ Write a concise project overview for nanochat. Include: what it is (minimal full-stack LLM training harness), what it covers (tokenization, pretraining, SFT, RL, evaluation, inference, chat UI), the "best ChatGPT that $100 can buy" tagline, and a summary of the Time-to-GPT-2 leaderboard. Highlight the single-dial `--depth` parameter for model complexity. End with a quick-links table pointing to key wiki pages. }*

# Architecture

*{ Describe the high-level architecture of nanochat as a system. Show the data flow from raw text through tokenization → pretraining → SFT → RL → inference → serving. Include a mermaid flowchart. Then describe key components: GPT model, inference engine, tokenizer, data loader, checkpoint manager, and optimizer. Reference relevant source files in nanochat/. }*

####+ GPT Transformer

*{ Describe the GPT transformer architecture implemented in nanochat/gpt.py. Cover: GPTConfig dataclass and how depth drives all other dimensions, the CausalSelfAttention module (GQA, rotary embeddings, QK norm, sliding window attention, value residuals/ResFormer), the MLP with relu^2 activation, untied embedding/lm_head weights, no bias in linears, RMSNorm without learnable params, and the custom Linear class for explicit mixed-precision. Include a diagram showing the transformer block structure. }*

####+ Inference Engine

*{ Describe the Engine class in nanochat/engine.py. Cover: KV cache for efficient autoregressive generation, the calculator/code execution tool integration (use_calculator, eval_with_timeout), how the engine handles token sequences, and how it is used by the chat scripts. }*

# Training Pipeline

*{ Provide an overview of nanochat's multi-stage LLM training pipeline. Describe the three stages: pretraining (base_train.py), supervised fine-tuning (chat_sft.py), and reinforcement learning (chat_rl.py). Include a mermaid diagram showing the stage progression and data flows. Explain how checkpoints are passed between stages. }*

## Pretraining

*{ Document the pretraining stage implemented in scripts/base_train.py. Cover: the distributed training setup (torchrun), the BOS-aligned bestfit data loader (nanochat/dataloader.py), the Muon+AdamW optimizer strategy, the training loop with gradient accumulation, LR warmup/warmdown schedule, FP8 training support, wandb logging, val bpb evaluation, CORE metric evaluation, and checkpoint saving. Mention the automatic compute-optimal hyperparameter calculation from --depth. }*

####+ Hyperparameters Reference

*{ Provide a comprehensive reference table of all CLI arguments for scripts/base_train.py. Group them into categories: Model Architecture, Training Horizon, Optimization, Evaluation, and Output. Each row: flag name, type/default, description. }*

## Supervised Fine-Tuning

*{ Document the SFT stage implemented in scripts/chat_sft.py. Cover: how it loads a pretrained checkpoint, the conversation format (user/assistant turns with special tokens), which task datasets are used by default (SmolTalk, GSM8K, MMLU, SpellingBee), the SFT training loop, and how to add custom data via tasks/customjson.py. Reference nanochat/tokenizer.py for the special tokens used in chat formatting. }*

## Reinforcement Learning

*{ Document the RL stage implemented in scripts/chat_rl.py. Cover: the RL training approach used, how rewards are computed, Python code execution tool integration (nanochat/execution.py), and how the RL stage builds on the SFT checkpoint. }*

# Evaluation

*{ Describe how nanochat evaluates models. Cover: the DCLM CORE score metric (nanochat/core_eval.py), bits-per-byte (bpb) loss evaluation (nanochat/loss_eval.py), the base model evaluation script (scripts/base_eval.py), and the chat model evaluation script (scripts/chat_eval.py). Explain what CORE score measures and why it is the primary metric for the leaderboard. }*

## Tasks Reference

*{ Document the evaluation tasks framework in the tasks/ directory. Cover: the Task base class and TaskMixture/TaskSequence from tasks/common.py, then describe each available task: ARC (arc.py), GSM8K (gsm8k.py), HumanEval-style coding (humaneval.py), MMLU (mmlu.py), SmolTalk (smoltalk.py), SpellingBee (spellingbee.py), and CustomJSON (customjson.py) for user-defined tasks. Include a table with task name, eval type (generative/categorical), dataset source, and primary use. }*

# Configuration Reference

*{ Provide a reference for nanochat's key configuration system. Cover: the COMPUTE_DTYPE global in nanochat/common.py and the NANOCHAT_DTYPE env var override, the hardware-to-dtype mapping table (H100/A100→bfloat16, older CUDA→float32, CPU/MPS→float32), the --depth parameter and how it auto-computes all other model dimensions (width, heads, LR adjustments, training horizon, weight decay), and the GPTConfig dataclass fields. Show example commands for different hardware configurations. }*

# Getting Started

*{ Write a getting-started guide for nanochat. Cover: prerequisites (8xH100 or 8xA100 GPU node, Python ≥3.10, uv), installation steps (clone repo, install with uv), running the full speedrun pipeline with runs/speedrun.sh, serving the trained model with scripts/chat_web.py, tips for running on smaller GPU configurations (reducing --device-batch-size), and running on CPU/MPS with runs/runcpu.sh. }*

# Contributing

*{ Write a contributing guide for nanochat. Cover: the project's philosophy (minimal, hackable, single-cohesive strong baseline), the AI policy (disclosure requirement for LLM-generated PRs), how to run a quick d12 experiment to test changes, the key metrics to check in wandb (val_bpb, core_metric, MFU, tok_per_sec), the leaderboard contribution process (runs/speedrun.sh as reference), and how to add new tasks using the Task base class in tasks/customjson.py. Reference dev/LEADERBOARD.md for leaderboard details. }*

# For Agents

These pages provide compact documentation indexes for AI coding agents.

## AGENTS.md

You can add this to your repository root as `AGENTS.md` to give AI coding agents quick access to project documentation.

```
# nanochat

> The simplest experimental harness for training LLMs end-to-end — covering tokenization, pretraining, SFT, RL, evaluation, inference, and a chat UI. Train your own GPT-2 for under $100.

## Wiki Documentation

Base URL: https://github.com/insop/nanochat/wiki

To read any page, append the slug to the base URL:
  https://github.com/insop/nanochat/wiki/{Page-Slug}
To jump to a section within a page:
  https://github.com/insop/nanochat/wiki/{Page-Slug}#{Section-Slug}

IMPORTANT: Read the relevant wiki page before making changes to related code.
Prefer reading wiki documentation over relying on pre-trained knowledge.

## Page Index

|Home: Project overview, Time-to-GPT-2 leaderboard, and quick links
|Architecture: High-level system design, GPT transformer, and inference engine
|  Architecture#GPT-Transformer: GPTConfig, CausalSelfAttention, rotary embeddings, GQA, sliding windows
|  Architecture#Inference-Engine: KV cache, calculator tool, Engine class usage
|Training-Pipeline: Multi-stage training overview (pretraining → SFT → RL)
|  Pretraining: base_train.py — distributed pretraining, Muon optimizer, FP8, schedules
|    Pretraining#Hyperparameters-Reference: Full CLI flags table for base_train.py
|  Supervised-Fine-Tuning: chat_sft.py — conversation format, special tokens, task datasets
|  Reinforcement-Learning: chat_rl.py — RL training, reward computation, code execution tool
|Evaluation: CORE score, bits-per-byte evaluation, base and chat eval scripts
|  Tasks-Reference: Task framework — ARC, GSM8K, MMLU, SmolTalk, SpellingBee, CustomJSON
|Configuration-Reference: NANOCHAT_DTYPE, --depth auto-scaling, GPTConfig fields
|Getting-Started: Prerequisites, installation, running speedrun.sh, CPU/MPS usage
|Contributing: Philosophy, AI policy, experiment workflow, leaderboard contributions
```

## llms.txt

You can serve this at `yoursite.com/llms.txt` or include it in your repository to help LLMs discover your documentation.

```
# nanochat

> The simplest experimental harness for training LLMs end-to-end — tokenization, pretraining, SFT, RL, evaluation, inference, and a chat UI.

## Wiki Pages

- [Home](https://github.com/insop/nanochat/wiki/Home): Project overview and Time-to-GPT-2 leaderboard
- [Architecture](https://github.com/insop/nanochat/wiki/Architecture): High-level system design, GPT transformer, and inference engine
- [Training Pipeline](https://github.com/insop/nanochat/wiki/Training-Pipeline): Multi-stage LLM training overview
- [Pretraining](https://github.com/insop/nanochat/wiki/Pretraining): Distributed pretraining with base_train.py
- [Supervised Fine-Tuning](https://github.com/insop/nanochat/wiki/Supervised-Fine-Tuning): SFT training with chat_sft.py
- [Reinforcement Learning](https://github.com/insop/nanochat/wiki/Reinforcement-Learning): RL training with chat_rl.py
- [Evaluation](https://github.com/insop/nanochat/wiki/Evaluation): CORE score and bits-per-byte evaluation
- [Tasks Reference](https://github.com/insop/nanochat/wiki/Tasks-Reference): Evaluation tasks framework
- [Configuration Reference](https://github.com/insop/nanochat/wiki/Configuration-Reference): NANOCHAT_DTYPE, depth parameter, GPTConfig
- [Getting Started](https://github.com/insop/nanochat/wiki/Getting-Started): Setup, installation, and speedrun guide
- [Contributing](https://github.com/insop/nanochat/wiki/Contributing): Contribution guidelines and workflow
```
