---
title: >-
  [Paper Note] UniCreative: Unifying Long-form Logic and Short-form Sparkle via Reference-Free Reinforcement Learning
description: >-
  [ACL 2026][Reinforcement Learning][Creative Writing] This paper proposes the UniCreative framework, which unifies long-form (planning → writing) and short-form (direct generation) creative writing modes without SFT or re…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Creative Writing"
  - "Reference-Free RL"
  - "Preference Optimization"
  - "Generative Reward Model"
  - "Metacognition"
date: 2026-05-08
content_hash: f19756c3d0fcacce
---

# UniCreative: Unifying Long-form Logic and Short-form Sparkle via Reference-Free Reinforcement Learning

**Conference**: ACL 2026  
**arXiv**: [2604.05517](https://arxiv.org/abs/2604.05517)  
**Code**: [https://github.com/weixiaolong94-hub/UniCreative](https://github.com/weixiaolong94-hub/UniCreative)  
**Area**: Reinforcement Learning/Creative Writing  
**Keywords**: Creative Writing, Reference-Free RL, Preference Optimization, Generative Reward Model, Metacognition

## TL;DR

This paper proposes the UniCreative framework, which unifies long-form (planning → writing) and short-form (direct generation) creative writing modes without SFT or reference answers. By utilizing Adaptive Constraint Preference Optimization (ACPO) and an Adaptive-Criteria Generative Reward Model (AC-GenRM), the model induces an emergent metacognitive ability to autonomously distinguish task types.

## Background & Motivation

**Background**: LLMs exhibit excellent performance in general text generation, but creative writing remains hindered by two fundamental challenges: long-form text (e.g., novels, scripts) requires global structural coherence and is prone to thematic drift or repetition, while short-form text (e.g., poetry, greetings, ad copy) requires linguistic "sparkle," which auto-regressive models often stifle by favoring high-probability "safe" tokens.

**Limitations of Prior Work**: (1) Simply increasing context windows does not solve structural degradation in long-form text, as explicit planning is required; (2) forcing planning mechanisms onto short-form text can over-constrain the model and stifle creative inspiration; (3) existing alignment methods (RLHF/DPO) rely heavily on high-quality labeled data and reference answers, which are costly and difficult to scale for open-ended creative tasks.

**Key Challenge**: The "myopia" of long-form text (structural collapse due to lack of global planning) and the "over-determination" of short-form text (diverse expression suppressed by excessive structural constraints) represent opposing failure modes that cannot be addressed with a unified generation strategy.

**Goal**: Construct a unified framework that allows the model to autonomously determine when to plan and when to generate directly, without relying on human-annotated completed texts.

**Key Insight**: Treat planning as a "dynamically callable computational resource" rather than a fixed prerequisite, enabling the model to learn adaptive switching between modes through reinforcement learning.

**Core Idea**: Skip the SFT stage and directly train the model using reference-free reinforcement learning (ACPO), with fine-grained preference signals provided by an adaptive-criteria generative reward model (AC-GenRM).

## Method

### Overall Architecture

UniCreative consists of two core components: (1) AC-GenRM—a generative reward model that dynamically generates evaluation criteria based on query semantics and performs debiased pairwise ranking; (2) ACPO—a strategy optimization algorithm based on GRPO that trains the model using signals from content quality rewards, structural paradigm constraints, and length regularization. During training, the model generates $G$ responses per query, and AC-GenRM produces relative reward signals via bootstrapped comparison.

### Key Designs

1. **AC-GenRM (Adaptive-Criteria Generative Reward Model)**:

    - **Function**: Dynamically generates evaluation criteria for each creative query and provides debiased pairwise preference judgments.
    - **Mechanism**: Operates in two steps—(a) Dynamic Criteria Synthesis: given a query $x$, the model $\pi_{critic}$ automatically generates evaluation dimensions (e.g., horror stories focus on "plot twists" and "atmosphere," while greeting cards focus on "warmth" and "conciseness"); (b) Debiased Pairwise Ranking: trained with symmetric data augmentation (50% probability of swapping response order) to eliminate positional bias and strictly align preference signals with dynamic criteria.
    - **Design Motivation**: Static evaluation criteria fail to capture quality differences across various creative genres (the criteria for mystery novels versus love poems are entirely different), and positional bias in LLM-as-a-judge is a recognized issue.

2. **3D Reward Combination**:

    - **Function**: Guides the model to simultaneously learn content quality, structural selection, and length control.
    - **Mechanism**: Total reward $R_{total} = R_{rel} + R_{struct} + R_{len}$. $R_{rel}$ is obtained through bootstrapped comparisons (intra-group competition where winning grants +2 and losing grants -2); $R_{struct}$ is a paradigm-aware penalty (deducting $\beta_s=5.0$ points if long-form text lacks planning or short-form text uses planning); $R_{len}$ is asymmetric length regularization (penalizing overly short long-form text and overly long short-form text, capped at $\gamma=5.0$ to prevent outliers from generating excessive gradients).
    - **Design Motivation**: Content quality rewards alone cannot teach the model to distinguish task types or control output length; orthogonal structural and length constraints are required to guide policy learning.

3. **ACPO Optimization Algorithm**:

    - **Function**: Optimizes the policy directly without SFT or reference answers.
    - **Mechanism**: Based on GRPO (Group Relative Policy Optimization), $G$ responses are sampled per query to calculate group-normalized advantage $A_i$. Policy updates are performed using clipped importance ratios and KL divergence constraints. A projection operator $\phi$ removes planning tokens before passing responses to AC-GenRM for evaluation, ensuring rewards reflect only the quality of the final content.
    - **Design Motivation**: Since creative writing has no single correct answer, traditional SFT is ineffective. GRPO avoids training unstable value networks and is particularly suited for high-variance gradient estimation in long-form generation.

### Loss & Training

The GRPO objective is used to maximize clipped advantage minus a KL penalty. Training was conducted on Qwen3 series models (1.7B, 4B, 8B) using 8 H800 GPUs, starting RL training directly from the "thinking" models without intermediate SFT steps.

## Key Experimental Results

### Main Results

| Model | WritingBench Avg | Blessing Excellence Rate |
|--------|------|------|
| Qwen3-8B-Thinking | 77.11 | 68.0% |
| Qwen3-8B-Thinking + RL | 82.42 | 93.6% |
| Qwen3-4B-Thinking + RL | 77.36 | 91.4% |
| Claude-Sonnet-3.7 | 78.48 | - |
| DeepSeek-R1-0528 | 83.22 | - |
| Claude-Sonnet-4.5 | - | 93.2% |

Qwen3-8B + RL approaches DeepSeek-R1-0528 (83.22) on WritingBench and outperforms Claude-Sonnet-4.5 on short-form text (Blessing).

### Ablation Study

| Configuration | WritingBench | Blessing | Description |
|------|---------|------|------|
| Qwen3-8B Base | 70.75 | 43.6% | No thinking, no RL |
| + Thinking | 77.11 | 68.0% | Thinking mode |
| + Thinking + RL | 82.42 | 93.6% | Full method, +25.6% gain |

### Key Findings

- **RL Gain is Significant**: RL training alone (without SFT) yields a 5-10 point improvement on WritingBench and increases the excellence rate on Blessing from 68% to 93.6%.
- **High Alignment of AC-GenRM**: Qwen3-8B AC-GenRM achieves an agreement rate of 0.807 with expert judgments on LitBench and 0.994 on Blessing, surpassing Claude-Sonnet-3.7 (0.731) and GPT-4.1 (0.702).
- **Emergent Metacognitive Ability**: The trained model autonomously learns to use the Plan-then-Write mode for long-form tasks and direct generation for short-form tasks without explicit task labels.
- **Benefits for Small Models**: The 1.7B model improved from 64.2% to 90.0% (Blessing) via RL, approaching the performance of much larger models.

## Highlights & Insights

- **Feasibility of Reference-Free RL**: This work demonstrates for the first time that the SFT phase can be entirely bypassed in creative writing; RL with an adaptive reward model can meet or exceed SFT+RLHF performance, significantly reducing annotation costs.
- **Clever Design of Structural Paradigm Penalties**: A simple binary penalty (penalizing the absence of planning in long-form or the presence of planning in short-form) effectively teaches the model to choose cognitive modes autonomously.
- **Dynamic Criteria Synthesis**: The AC-GenRM approach of generating evaluation dimensions based on query semantics can be generalized to the evaluation of all open-ended generation tasks.

## Limitations & Future Work

- Evaluation still relies on LLM judges (GPT-4o for WritingBench); the automatic evaluation of subjective creation remains an open problem.
- Results are only verified on the Qwen3 series; generalization to other base models has not been tested.
- The categorization of task types (long vs. short) is coarse; the optimal strategy for medium-length tasks (e.g., emails, reports) remains uncertain.
- Structural paradigm penalties require pre-defined task type labels (Long/Short), limiting fully autonomous mode selection.

## Related Work & Insights

- **vs. Writing-Zero/LongWriter-Zero**: While these methods focus on long-form RL optimization, UniCreative unifies both long-form and short-form modes and eliminates the SFT phase.
- **vs. DPO/RLHF**: Traditional methods rely on reference answers and labeled preference pairs; UniCreative removes reference dependency via bootstrapped comparison.
- **vs. GRPO (DeepSeek-R1)**: UniCreative adds structural paradigm constraints and adaptive length regularization to GRPO, making it suitable for creative writing scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The unified framework and reference-free RL training for short/long-form creation are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive evaluation across model sizes and dual benchmarks (WritingBench and Blessing).
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and methodology design.
- Value: ⭐⭐⭐⭐ Provides a practical paradigm for RL optimization in open-ended creative tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LoVeC: Reinforcement Learning for Better Verbalized Confidence in Long-Form Generations](lovec_reinforcement_learning_for_better_verbalized_confidence_in_long-form_gener.md)
- [\[ICLR 2026\] Safe Continuous-time Multi-Agent Reinforcement Learning via Epigraph Form](../../ICLR2026/reinforcement_learning/safe_continuous-time_multi-agent_reinforcement_learning_via_epigraph_form.md)
- [\[ACL 2026\] Free Energy-Driven Reinforcement Learning with Adaptive Advantage Shaping for Unsupervised Reasoning in LLMs](free_energy-driven_reinforcement_learning_with_adaptive_advantage_shaping_for_un.md)
- [\[ACL 2026\] A Goal Without a Plan Is Just a Wish: Efficient and Effective Global Planner Training for Long-Horizon Agent Tasks (EAGLET)](a_goal_without_a_plan_is_just_a_wish_efficient_and_effective_global_planner_trai.md)
- [\[ICLR 2026\] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](../../ICLR2026/reinforcement_learning/spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)

</div>

<!-- RELATED:END -->
