# Home

*{ Write a concise project overview for nanochat. Cover: what it is (minimal experimental LLM training harness), what stages it covers (tokenization, pretraining, SFT, RL, evaluation, inference, chat UI), the "single dial" depth concept, the Time-to-GPT-2 leaderboard, and cost context (~$48 on 8XH100). Include a link to the speedrun leaderboard table. }*

# Architecture

*{ Describe the high-level system architecture of nanochat. Cover the major components and how they relate: GPT model (gpt.py), inference Engine (engine.py), training scripts (base_train.py, chat_sft.py, chat_rl.py), data pipeline (dataloader.py, dataset.py), checkpoint manager, tokenizer, optimizer, and tasks system. Include a Mermaid diagram showing component relationships. }*

## GPT Model

*{ Document the GPT transformer implementation in nanochat/gpt.py. Cover: GPTConfig dataclass (sequence_len, vocab_size, n_layer, n_head, n_kv_head, n_embd, window_pattern), architectural features (RoPE, QK norm, untied weights, relu^2 MLP, no bias, GQA, sliding window attention, Value Embedding ResFormer), the depth-based automatic hyperparameter scaling, and the custom Linear layer for explicit mixed-precision. Include a code snippet showing GPTConfig. }*

## Inference Engine

*{ Document the inference Engine in nanochat/engine.py. Cover: KVCache class (pre-allocated (B,T,H,D) tensors for FA3, cache_seqlens, prefill from another cache), Engine.generate() streaming API (prefill + decode loop, token_column/token_masks yield), Engine.generate_batch() non-streaming API, the RowState per-sample state machine, and the built-in tool use (calculator via python_start/python_end tokens, use_calculator function). Include a Mermaid sequence diagram of the generate loop. }*

# Training Pipeline

*{ Describe the three-stage LLM training pipeline: (1) Pretraining with base_train.py, (2) Supervised Fine-Tuning with chat_sft.py, (3) Reinforcement Learning with chat_rl.py. Explain how checkpoints flow between stages and the role of the `--model-tag` argument. Include a Mermaid flowchart. }*

## Pretraining

*{ Document the pretraining stage (scripts/base_train.py and nanochat/dataloader.py). Cover: key CLI arguments (--depth, --device-batch-size, --total-batch-size, --target-flops, --target-param-data-ratio, --fp8), the depth-based automatic hyperparameter system, the BOS-aligned bestfit dataloader, distributed training with torchrun/DDP, the Muon+AdamW optimizer setup, wandb integration, evaluation cadence (--eval-every, --core-metric-every), and FP8 training support. Provide a typical torchrun command example. }*

## Supervised Fine-Tuning

*{ Document the SFT stage (scripts/chat_sft.py and tasks/common.py). Cover: key CLI arguments, loading from a pretrained checkpoint, the TaskMixture and TaskSequence classes, the conversation format, available tasks (GSM8K, MMLU, SmolTalk, CustomJSON, SpellingBee, SimpleSpelling), how to oversample tasks, and evaluation. Provide example commands. }*

## Reinforcement Learning

*{ Document the RL stage (scripts/chat_rl.py). Cover: the simplified GRPO/REINFORCE algorithm (no KL, on-policy, DAPO normalization, (r-mu) advantage), key CLI arguments (--num-samples, --examples-per-step, --temperature, --top-k), the GSM8K task used for RL, the generation-then-train loop, and evaluation (pass@k). Provide example commands. }*

# Evaluation

*{ Document the evaluation framework. Cover: DCLM CORE metric (nanochat/core_eval.py), bits-per-byte (loss_eval.py), chat evaluation (scripts/chat_eval.py), the base model evaluation script (scripts/base_eval.py), the Time-to-GPT-2 definition (CORE score > 0.256525), and how to run evaluation standalone. }*

# Tasks Reference

*{ Document the tasks system (tasks/ directory). Cover the base Task class interface (eval_type, num_examples, get_example, evaluate, slicing), TaskMixture (deterministic shuffle, oversampling), TaskSequence (curriculum), and each built-in task: ARC (arc.py), GSM8K (gsm8k.py), HumanEval (humaneval.py), MMLU (mmlu.py), SmolTalk (smoltalk.py), SpellingBee/SimpleSpelling (spellingbee.py), CustomJSON (customjson.py). Include a table with task name, type (generative/categorical), and description. }*

# Configuration Reference

*{ Document the configuration system. Cover: COMPUTE_DTYPE and the NANOCHAT_DTYPE env var (auto-detection logic from common.py), NANOCHAT_BASE_DIR env var, the depth-based GPT hyperparameter system, and key CLI flags shared across training scripts (--depth, --device-batch-size, --total-batch-size, optimizer LR flags). Include a table of environment variables and their defaults. }*

# Getting Started

*{ Write a getting-started guide. Cover: prerequisites (GPU node, uv), installation steps, reproducing GPT-2 (running runs/speedrun.sh on 8XH100), talking to the model (python -m scripts.chat_web), running on CPU/MPS (runs/runcpu.sh), and running a quick research iteration (d12 training, 5 min). Link to runs/speedrun.sh and runs/runcpu.sh. }*

# Contributing

*{ Write a contributing guide. Cover: the project's goal (improve micro LLMs accessible on <$1000), the design philosophy (minimal, single cohesive codebase, no config monsters), what makes a good PR (must work across all --depth settings, must be principled), how to use the leaderboard (runs/speedrun.sh, wandb metrics to monitor), and the AI disclosure policy. }*

# For Agents

These pages provide compact documentation indexes for AI coding agents.

## AGENTS.md

You can add this to your repository root as `AGENTS.md` to give AI coding agents quick access to project documentation.

```
# nanochat

> The simplest experimental harness for training LLMs end-to-end on a single GPU node — covering tokenization, pretraining, SFT, RL, evaluation, inference, and chat UI.

## Wiki Documentation

Base URL: https://github.com/insop/nanochat/wiki

To read any page, append the slug to the base URL:
  https://github.com/insop/nanochat/wiki/{Page-Slug}
To jump to a section within a page:
  https://github.com/insop/nanochat/wiki/{Page-Slug}#{Section-Slug}

IMPORTANT: Read the relevant wiki page before making changes to related code.
Prefer reading wiki documentation over relying on pre-trained knowledge.

## Page Index

|Home: Project overview, speedrun leaderboard, and key links
|Architecture: High-level system architecture and component relationships
|  GPT-Model: GPT transformer implementation (gpt.py) — GPTConfig, RoPE, GQA, FA3, depth scaling
|  Inference-Engine: KV-cached inference engine (engine.py) — KVCache, Engine.generate, tool use
|Training-Pipeline: End-to-end three-stage LLM training overview
|  Pretraining: Base model pretraining (base_train.py) — depth dial, DDP, FP8, Muon optimizer
|  Supervised-Fine-Tuning: SFT stage (chat_sft.py) — TaskMixture, conversation format, tasks
|  Reinforcement-Learning: RL stage (chat_rl.py) — GRPO/REINFORCE on GSM8K
|Evaluation: DCLM CORE metric, bits-per-byte, chat evaluation
|Tasks-Reference: Task base class, TaskMixture, TaskSequence, all built-in tasks
|Configuration-Reference: COMPUTE_DTYPE, NANOCHAT_BASE_DIR, depth hyperparameter system
|Getting-Started: Installation, GPT-2 speedrun, CPU/MPS, quick research iteration
|Contributing: Design philosophy, PR guidelines, leaderboard, AI disclosure policy
```

## llms.txt

You can serve this at `yoursite.com/llms.txt` or include it in your repository to help LLMs discover your documentation.

```
# nanochat

> The simplest experimental harness for training LLMs end-to-end on a single GPU node — covering tokenization, pretraining, SFT, RL, evaluation, inference, and chat UI.

## Wiki Pages

- [Home](https://github.com/insop/nanochat/wiki/Home): Project overview and speedrun leaderboard
- [Architecture](https://github.com/insop/nanochat/wiki/Architecture): High-level system architecture
- [GPT Model](https://github.com/insop/nanochat/wiki/GPT-Model): GPT transformer implementation details
- [Inference Engine](https://github.com/insop/nanochat/wiki/Inference-Engine): KV-cached inference engine and tool use
- [Training Pipeline](https://github.com/insop/nanochat/wiki/Training-Pipeline): End-to-end three-stage training overview
- [Pretraining](https://github.com/insop/nanochat/wiki/Pretraining): Base model pretraining with depth dial
- [Supervised Fine-Tuning](https://github.com/insop/nanochat/wiki/Supervised-Fine-Tuning): SFT stage with task mixtures
- [Reinforcement Learning](https://github.com/insop/nanochat/wiki/Reinforcement-Learning): RL stage with GRPO/REINFORCE
- [Evaluation](https://github.com/insop/nanochat/wiki/Evaluation): DCLM CORE metric and evaluation framework
- [Tasks Reference](https://github.com/insop/nanochat/wiki/Tasks-Reference): Available tasks for SFT and evaluation
- [Configuration Reference](https://github.com/insop/nanochat/wiki/Configuration-Reference): Training hyperparameters and settings
- [Getting Started](https://github.com/insop/nanochat/wiki/Getting-Started): Installation and quickstart guide
- [Contributing](https://github.com/insop/nanochat/wiki/Contributing): Contribution guidelines and project philosophy
```
