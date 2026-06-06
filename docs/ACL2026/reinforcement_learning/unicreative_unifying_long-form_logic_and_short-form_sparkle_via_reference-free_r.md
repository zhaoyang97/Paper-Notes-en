---
title: >-
  [Paper Note] UniCreative: Unifying Long-form Logic and Short-form Sparkle via Reference-Free Reinforcement Learning
description: >-
  [ACL 2026][Reinforcement Learning][Creative Writing] This paper proposes UniCreative, a framework that unifies long-form (plan→write) and short-form (direct generation) creative writing modes through Adaptive Constraint…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Creative Writing"
  - "Reference-Free Reinforcement Learning"
  - "Preference Optimization"
  - "Generative Reward Model"
  - "Metacognition"
date: 2026-05-08
content_hash: 40a145bc7f4aca79
---

# UniCreative: Unifying Long-form Logic and Short-form Sparkle via Reference-Free Reinforcement Learning

**Conference**: ACL 2026
**arXiv**: [2604.05517](https://arxiv.org/abs/2604.05517)  
**Code**: [https://github.com/weixiaolong94-hub/UniCreative](https://github.com/weixiaolong94-hub/UniCreative)  
**Area**: Reinforcement Learning / Creative Writing
**Keywords**: Creative Writing, Reference-Free Reinforcement Learning, Preference Optimization, Generative Reward Model, Metacognition

## TL;DR

This paper proposes UniCreative, a framework that unifies long-form (plan→write) and short-form (direct generation) creative writing modes through Adaptive Constraint Preference Optimization (ACPO) and an Adaptive Criteria Generative Reward Model (AC-GenRM), requiring neither SFT nor reference answers. The trained model exhibits emergent metacognitive ability to autonomously distinguish between task types.

## Background & Motivation

**Background**: LLMs perform well on general text generation, yet creative writing remains afflicted by two fundamental challenges: long-form texts (e.g., novels, scripts) require global structural coherence and are prone to topic drift and repetition, while short-form texts (e.g., poetry, greetings, advertising copy) demand linguistic vibrancy, yet autoregressive models gravitate toward high-probability "safe" tokens, resulting in mediocre outputs.

**Limitations of Prior Work**: (1) Simply expanding the context window cannot resolve structural degradation in long-form generation; explicit planning is required. (2) Imposing planning mechanisms on short-form tasks over-constrains the output and suppresses creative sparks. (3) Existing alignment methods (RLHF/DPO) rely heavily on high-quality annotated data and reference answers, making them prohibitively costly and difficult to scale for open-ended creative tasks.

**Key Challenge**: The "myopia" of long-form generation (structural collapse due to lack of global planning) and the "over-determination" of short-form generation (diversity suppressed by excessive structural constraints) represent two opposing failure modes that cannot be resolved by a single unified generation strategy.

**Goal**: To construct a unified framework that enables the model to autonomously determine when planning is necessary and when direct generation is preferable, without relying on any human-annotated completed texts.

**Key Insight**: Planning is treated as a "dynamically invokable computational resource" rather than a fixed prerequisite step; reinforcement learning is used to teach the model to adaptively switch between the two modes.

**Core Idea**: Skip the SFT stage and directly train the model via reference-free reinforcement learning (ACPO), with AC-GenRM providing fine-grained preference signals based on adaptively generated criteria.

## Method

### Overall Architecture

UniCreative consists of two core components: (1) AC-GenRM — a generative reward model that dynamically generates evaluation criteria based on query semantics and performs debiased pairwise ranking; and (2) ACPO — a GRPO-based policy optimization algorithm that trains the model using three signals: content quality reward, structural paradigm constraint, and length regularization. During training, the model generates $G$ responses per query, and AC-GenRM produces relative reward signals via bootstrapped contrastive comparison.

### Key Designs

1. **AC-GenRM (Adaptive Criteria Generative Reward Model)**:

    - **Function**: Dynamically generates evaluation criteria for each creative query and provides debiased pairwise preference judgments.
    - **Mechanism**: A two-step process — (a) *Dynamic criteria synthesis*: given query $x$, the critic model $\pi_{critic}$ automatically generates query-specific evaluation dimensions (e.g., horror stories emphasize "plot twists" and "atmosphere," while greeting cards emphasize "warmth" and "conciseness"); (b) *Debiased pairwise ranking*: trained with symmetric data augmentation (response order swapped with 50% probability) to eliminate position bias, ensuring preference signals are strictly aligned with dynamic criteria.
    - **Design Motivation**: Static evaluation criteria fail to capture quality differences across creative genres (the quality dimensions for a mystery novel and a love poem are entirely distinct), and position bias in LLM judges is a well-known problem.

2. **Three-Dimensional Reward Combination**:

    - **Function**: Guides the model to simultaneously learn content quality, structural selection, and length control.
    - **Mechanism**: Total reward $R_{total} = R_{rel} + R_{struct} + R_{len}$. $R_{rel}$ is obtained via bootstrapped contrastive comparison (intra-group self-competition: win +2, lose −2); $R_{struct}$ is a paradigm-aware penalty (deducting $\beta_s = 5.0$ points when long-form tasks omit planning or short-form tasks invoke planning); $R_{len}$ is an asymmetric length regularization term (penalizing excessively short long-form outputs and excessively long short-form outputs, with an upper bound $\gamma = 5.0$ to prevent outliers from producing excessively large gradients).
    - **Design Motivation**: Content quality reward alone cannot teach the model to distinguish task types or control output length; orthogonal structural and length constraints are required to jointly guide policy learning.

3. **ACPO Optimization Algorithm**:

    - **Function**: Directly optimizes the policy without SFT or reference answers.
    - **Mechanism**: Built upon GRPO (Group Relative Policy Optimization), sampling $G$ responses per query, computing intra-group normalized advantages $A_i$, and performing policy updates using clipped importance ratios and KL divergence constraints. Planning tokens are removed via a projection operator $\phi$ before passing outputs to AC-GenRM for evaluation, ensuring rewards reflect only the quality of the final generated content.
    - **Design Motivation**: Creative writing has no unique correct answer, rendering conventional SFT ineffective. GRPO avoids training an unstable value network, making it particularly suitable for high-variance gradient estimation in long-form generation.

### Loss & Training

The GRPO objective is used, maximizing clipped advantages minus a KL penalty. The Qwen3 series (1.7B, 4B, 8B) are trained on 8 H800 GPUs, with RL training initiated directly from the thinking model checkpoints, bypassing any SFT intermediate step.

## Key Experimental Results

### Main Results

| Model | WritingBench Avg. | Blessing Excellence Rate |
|--------|------|------|
| Qwen3-8B-Thinking | 77.11 | 68.0% |
| Qwen3-8B-Thinking + RL | 82.42 | 93.6% |
| Qwen3-4B-Thinking + RL | 77.36 | 91.4% |
| Claude-Sonnet-3.7 | 78.48 | - |
| DeepSeek-R1-0528 | 83.22 | - |
| Claude-Sonnet-4.5 | - | 93.2% |

Qwen3-8B + RL approaches DeepSeek-R1-0528 (83.22) on WritingBench and surpasses Claude-Sonnet-4.5 on the short-form Blessing benchmark.

### Ablation Study

| Configuration | WritingBench | Blessing | Notes |
|------|---------|------|------|
| Qwen3-8B Base | 70.75 | 43.6% | No thinking, no RL |
| + Thinking | 77.11 | 68.0% | Thinking mode |
| + Thinking + RL | 82.42 | 93.6% | Full method, +25.6% gain |

### Key Findings

- **Large RL Gains**: RL training alone (without SFT) yields a 5–10 point improvement on WritingBench and raises Blessing performance from 68% to 93.6%.
- **High AC-GenRM Alignment**: Qwen3-8B AC-GenRM achieves agreement rates of 0.807 with expert judgments on LitBench and 0.994 on Blessing, surpassing Claude-Sonnet-3.7 (0.731) and GPT-4.1 (0.702).
- **Emergent Metacognition**: After training, the model autonomously adopts the Plan-then-Write paradigm for long-form tasks and direct generation for short-form tasks, without any explicit task-type labels.
- **Small Models Also Benefit**: The 1.7B model improves from 64.2% to 90.0% on Blessing via RL, approaching the performance of larger models.

## Highlights & Insights

- **Viability of Reference-Free RL**: This work provides the first demonstration that, in the creative writing domain, the SFT stage can be entirely bypassed; RL combined with an adaptive reward model alone achieves or surpasses SFT+RLHF performance, substantially reducing annotation costs.
- **Elegant Design of Structural Paradigm Penalty**: A simple binary penalty (deducting points when long-form tasks omit planning or short-form tasks invoke it) suffices to teach the model to autonomously select its cognitive mode — a design that is remarkably simple yet highly effective.
- **Dynamic Criteria Synthesis**: The approach of having AC-GenRM automatically generate evaluation dimensions based on query semantics is generalizable to the evaluation of all open-ended generation tasks.

## Limitations & Future Work

- Evaluation still relies on LLM judges (WritingBench uses GPT-4o for scoring); automated evaluation of subjective creative content remains an open problem.
- Validation is limited to the Qwen3 series; generalization to other base models has not been tested.
- The task-type classification (long/short) is relatively coarse-grained; the optimal strategy for medium-length tasks (e.g., emails, reports) remains uncertain.
- The structural paradigm penalty requires predefined task-type labels (Long/Short), limiting fully autonomous mode selection.

## Related Work & Insights

- **vs. Writing-Zero/LongWriter-Zero**: These methods focus on RL optimization for long-form text; UniCreative unifies both long-form and short-form modes and requires no SFT stage.
- **vs. DPO/RLHF**: Conventional methods depend on reference answers and annotated preference pairs; UniCreative eliminates reference dependency entirely through bootstrapped contrastive comparison.
- **vs. GRPO (DeepSeek-R1)**: UniCreative augments GRPO with structural paradigm constraints and adaptive length regularization, adapting it for creative writing scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The unified framework for long- and short-form creative writing and the reference-free RL training approach are both original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-scale model comparisons and dual-benchmark evaluation on WritingBench and Blessing are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation and method design are clearly presented.
- Value: ⭐⭐⭐⭐ Provides a practical paradigm for RL optimization of open-ended creative tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form](../../ICLR2026/reinforcement_learning/safe_continuous-time_multi-agent_reinforcement_learning_via_epigraph_form.md)
- [\[ACL 2026\] Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning](easy_samples_are_all_you_need_self-evolving_llms_via_data-efficient_reinforcemen.md)
- [\[ICLR 2026\] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](../../ICLR2026/reinforcement_learning/spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)
- [\[AAAI 2026\] Do It for HER: First-Order Temporal Logic Reward Specification in Reinforcement Learning](../../AAAI2026/reinforcement_learning/do_it_for_her_first-order_temporal_logic_reward_specification_in_reinforcement_l.md)
- [\[ICLR 2026\] LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts](../../ICLR2026/reinforcement_learning/loongrl_rl_for_reasoning_long_contexts.md)

</div>

<!-- RELATED:END -->
