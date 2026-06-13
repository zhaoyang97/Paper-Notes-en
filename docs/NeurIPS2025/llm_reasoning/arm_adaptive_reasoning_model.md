---
title: >-
  [Paper Note] ARM: Adaptive Reasoning Model
description: >-
  [NeurIPS 2025][LLM Reasoning][Adaptive Reasoning] ARM enables models to adaptively select among four reasoning formats (Direct Answer, Short CoT, Code…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "Adaptive Reasoning"
  - "Overthinking"
  - "GRPO"
  - "Reasoning Format Selection"
  - "Test-Time Compute"
date: 2026-05-08
content_hash: dba16b862b3f9ad9
---

# ARM: Adaptive Reasoning Model

**Conference**: NeurIPS 2025
**arXiv**: [2505.20258](https://arxiv.org/abs/2505.20258)  
**Code**: [https://team-arm.github.io/arm](https://team-arm.github.io/arm)  
**Area**: LLM Reasoning Efficiency / Reinforcement Learning
**Keywords**: Adaptive Reasoning, Overthinking, GRPO, Reasoning Format Selection, Test-Time Compute

## TL;DR
ARM enables models to adaptively select among four reasoning formats (Direct Answer, Short CoT, Code, Long CoT) and introduces Ada-GRPO to address format collapse during training, achieving comparable accuracy to pure Long CoT models while reducing token usage by ~30% on average and up to ~70% on simple tasks.

## Background & Motivation

**Background**: Large reasoning models (LRMs) such as OpenAI-o1 and DeepSeek-R1 achieve significant breakthroughs on complex tasks via Long Chain-of-Thought (Long CoT). These models improve reasoning capability through test-time scaling—generating more tokens.

**Limitations of Prior Work**: LRMs apply Long CoT uniformly across all tasks, leading to severe overthinking. Even trivial commonsense questions (e.g., "Where does a maid dump the garbage?") trigger hundreds of reasoning tokens, wasting computational resources; verbose outputs can also introduce noise and cause incorrect answers. Experiments show that DeepSeek-R1 consumes nearly 4× more tokens on simple tasks without performance gains, and sometimes even suffers degradation.

**Key Challenge**: Long CoT is not universally optimal—simple tasks require no elaborate reasoning, yet standard GRPO training optimizes only for accuracy, causing the model to collapse onto Long CoT as a single dominant format (format collapse) and abandon all more efficient alternatives. Existing mitigation strategies (e.g., length penalties) rely on manually estimated token budgets, and inaccurate estimates lead to substantial performance drops.

**Goal**: Train a model that autonomously selects the appropriate reasoning format based on task difficulty, without manual token budget specification.

**Key Insight**: The problem of reasoning efficiency is reformulated as a *reasoning format selection* problem—rather than shortening Long CoT, the model learns to use Direct Answer for simple tasks, Short CoT or Code for medium tasks, and Long CoT for complex tasks.

**Core Idea**: A format diversity reward scaling factor in Ada-GRPO prevents format collapse during GRPO training, enabling the model to adaptively switch among four reasoning formats according to task difficulty.

## Method

### Overall Architecture
ARM adopts a two-stage training framework. Given a question as input, the model outputs an answer produced by a self-selected reasoning format (Direct Answer / Short CoT / Code / Long CoT), with each format identified by special tokens (e.g., `<Code></Code>`).

- **Stage 1 (SFT)**: Supervised fine-tuning on 10.8K questions, each annotated with all four reasoning formats, enabling the model to understand the basic usage of each format.
- **Stage 2 (Ada-GRPO)**: Reinforcement learning on 19.8K verifiable QA pairs, where a format diversity reward mechanism teaches the model to adaptively select formats based on task difficulty.

### Key Designs

1. **Four-Tier Reasoning Format Hierarchy**

    - **Function**: Reasoning strategies are partitioned into four tiers: Direct Answer (zero reasoning), Short CoT (brief reasoning chain), Code (programmatic reasoning), and Long CoT (deep reasoning with reflection and backtracking).
    - **Mechanism**: Tasks of varying difficulty warrant reasoning of varying complexity. Commonsense questions can be answered with Direct Answer (~10 tokens), mathematical problems may call for Code (~300 tokens), and competition-level problems require Long CoT (~3,000 tokens).
    - **Design Motivation**: This is more principled than simply shortening the reasoning chain—the goal is not to "think less" but to "choose the right way to think."

2. **Format Diversity Scaling Factor in Ada-GRPO**

    - **Function**: Addresses format collapse in standard GRPO, where the model converges to Long CoT after approximately 10 training steps.
    - **Mechanism**: The reward for each rollout is multiplied by a scaling factor $\alpha_i(t) = \frac{G}{F(o_i)} \cdot decay_i(t)$, where $F(o_i)$ is the number of occurrences of format $o_i$ within the group and $G$ is the group size. Formats that appear less frequently receive amplified rewards, encouraging the model to explore diverse formats.
    - **Design Motivation**: Standard GRPO optimizes solely for accuracy; since Long CoT typically yields the highest accuracy, it is continuously reinforced. Ada-GRPO counteracts this by inversely amplifying rewards for underrepresented formats, preventing diversity collapse.

3. **Cosine Decay Mechanism**

    - **Function**: Gradually attenuates the influence of the format diversity reward over the course of training.
    - **Mechanism**: $decay_i(t) = \frac{F(o_i)}{G} + 0.5 \cdot (1 - \frac{F(o_i)}{G}) \cdot (1 + \cos(\frac{\pi t}{T}))$. The format diversity reward is maximized at the start of training ($t=0$) and decays to 1 by the end ($t=T$), reverting to pure accuracy optimization.
    - **Design Motivation**: Prevents training instability caused by persistently over-rewarding low-frequency formats. Ablation experiments confirm that removing the decay leads to substantial fluctuations in test accuracy.

4. **Three Inference Modes**

    - **Adaptive Mode** (default): The model autonomously selects the reasoning format.
    - **Instruction-Guided Mode**: The user specifies the format via special tokens, suitable for batch inference where the task type is known.
    - **Consensus-Guided Mode**: Three efficient formats are used independently; if their outputs disagree, the model falls back to Long CoT, prioritizing accuracy.

### Loss & Training
- **Stage 1**: Standard SFT with LoRA + DeepSpeed ZeRO-3, learning rate 2e-4, 6 epochs.
- **Stage 2**: Ada-GRPO uses the same objective as GRPO (PPO-clip style + KL regularization); the key difference is that rewards are scaled by the format diversity factor. Batch size 1024, 8 rollouts per prompt, maximum rollout length 4096 tokens, 9 epochs, 8 × A800 GPUs.

## Key Experimental Results

### Main Results

| Model (7B, maj@8) | CSQA (easy) | GSM8K (medium) | MATH (medium) | AIME'25 (hard) | Avg. Acc. | Avg. Tokens |
|---|---|---|---|---|---|---|
| Qwen2.5-7B Base | 82.0 | 89.9 | 64.7 | 3.3 | 68.3 | 260 |
| Qwen2.5-7B SFT+GRPO | 83.7 | 94.8 | 84.9 | 20.0 | 76.1 | 1164 |
| **ARM-7B** | **85.7** | **93.7** | **82.6** | **20.0** | **75.9** | **786** |
| DS-R1-Distill-7B | 64.9 | 90.0 | 93.6 | 40.0 | 75.5 | 2797 |

ARM-7B vs. SFT+GRPO: accuracy drops by only 0.2% while token usage is reduced by 32.5%.

### Ablation Study

| Configuration | Outcome | Notes |
|---|---|---|
| SFT only → format distribution | Near-uniform distribution across four formats | SFT teaches format usage but not selection; using Direct Answer on medium tasks causes accuracy collapse |
| GRPO → format distribution | ~100% Long CoT | Collapse occurs within ~10 steps; efficient formats entirely abandoned |
| Ada-GRPO → format distribution | DA/Short CoT dominant on easy tasks; Long CoT dominant on hard tasks | Adaptive selection; token savings >70% on simple tasks |
| Remove decay | Large fluctuations in test accuracy | Persistent over-rewarding of rare formats destabilizes training |
| Remove Direct Answer (CSQA) | +0.1% acc, +29.4% tokens | DA is critical for efficiency on simple tasks |
| Remove Long CoT (AIME'25) | −8.4% acc | Long CoT is essential for complex tasks |
| Add function-calling format | +0.4% avg acc | Finer-grained formats are beneficial but increase engineering complexity |

### Key Findings
- The 2× training speedup of Ada-GRPO stems from shorter responses during the rollout phase, as simple tasks no longer trigger Long CoT generation.
- Backbone choice has limited impact: Base and Instruct models perform comparably; R1-Distill is stronger on hard tasks but exhibits overthinking on simple ones.
- Compared to length-penalty methods such as L1 and ThinkPrune, ARM maintains stable performance across all tasks without relying on manually specified token budgets.

## Highlights & Insights
- **Format selection over length control**: Reframing reasoning efficiency from "controlling chain length" to "selecting reasoning format" is a more natural and robust paradigm. Length penalty methods require manual token budget estimation and suffer performance collapse when estimates are inaccurate, whereas ARM's adaptive selection is fully autonomous.
- **Elegant design of Ada-GRPO**: Format collapse is resolved with a single inverse-proportional scaling factor $G/F(o_i)$ combined with cosine decay, requiring no architectural changes to GRPO. This idea generalizes to any multi-objective optimization setting where one objective risks dominating the others.
- **Training efficiency as a by-product**: The 2× training speedup of Ada-GRPO is an unexpected but valuable bonus—because the model more frequently adopts short-format reasoning, rollout generation throughput increases substantially.

## Limitations & Future Work
- The method depends on four predefined reasoning formats and cannot autonomously discover new reasoning strategies. Future work could explore models that self-create reasoning formats.
- Training data excludes competition-level problems such as AIME, limiting performance gains on hard tasks.
- Token savings are less pronounced on the LLaMA backbone than on Qwen (15.7% vs. 55.2%), possibly due to LLaMA's tendency toward repetitive generation.
- Consensus-Guided Mode requires running three efficient formats plus potentially Long CoT, making total token consumption higher than pure Long CoT in some cases.

## Related Work & Insights
- **vs. L1 / ThinkPrune (length penalty)**: These methods use RL to shorten reasoning chains but require manually specified token budgets. ARM selects formats rather than controlling length, yielding greater autonomy and robustness. On CSQA, L1 with a 512-token budget achieves only 45.8% accuracy, whereas ARM achieves 86.1% using only 136 tokens.
- **vs. DeepSeek-R1-Distill**: R1-Distill follows a knowledge distillation paradigm, acquiring stronger reasoning capability but also inheriting the overthinking problem. ARM-7B matches DS-R1-Distill-7B in accuracy while using only 27.8% of its tokens.
- **vs. Qwen3 Thinking Mode**: Qwen3 supports switching between thinking and non-thinking modes, but each switch requires manual specification. ARM's adaptive selection is fully automated.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Redefining reasoning efficiency as format selection rather than length control is a valuable insight, and Ada-GRPO is elegantly designed, though the four formats remain manually predefined.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated at three scales (3B/7B/14B), across 7 benchmarks, with multiple backbones, inference modes, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clearly structured; Figure 2's format distribution visualization is particularly intuitive.
- **Value**: ⭐⭐⭐⭐ High practical value; the 2× training speedup and inference token savings are significant for deployment, though impact is constrained by the predefined format space.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unlocking Multimodal Mathematical Reasoning via Process Reward Model](unlocking_multimodal_mathematical_reasoning_via_process_reward_model.md)
- [\[NeurIPS 2025\] KTAE: A Model-Free Algorithm to Key-Tokens Advantage Estimation in Mathematical Reasoning](ktae_a_model-free_algorithm_to_key-tokens_advantage_estimation_in_mathematical_r.md)
- [\[NeurIPS 2025\] SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning](sql-r1_training_natural_language_to_sql_reasoning_model_by_reinforcement_learnin.md)
- [\[NeurIPS 2025\] The Impact of Quantization on Large Reasoning Model Reinforcement Learning](the_impact_of_quantization_on_large_reasoning_model_reinforcement_learning.md)
- [\[NeurIPS 2025\] Re-FORC: Adaptive Reward Prediction for Efficient Chain-of-Thought Reasoning](re-forc_adaptive_reward_prediction_for_efficient_chain-of-thought_reasoning.md)

</div>

<!-- RELATED:END -->
