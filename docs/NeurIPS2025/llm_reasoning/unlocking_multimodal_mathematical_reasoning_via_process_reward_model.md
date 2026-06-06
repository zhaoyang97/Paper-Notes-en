---
title: >-
  [Paper Note] Unlocking Multimodal Mathematical Reasoning via Process Reward Model
description: >-
  [NeurIPS 2025][LLM Reasoning][process reward model] This paper proposes URSA, a three-stage framework that sequentially constructs a million-scale multimodal CoT dataset (MMathCoT-1M) for base model training…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "process reward model"
  - "multimodal math"
  - "GRPO"
  - "test-time scaling"
  - "chain-of-thought"
date: 2026-05-08
content_hash: 31969d1094fd8e05
---

# Unlocking Multimodal Mathematical Reasoning via Process Reward Model

**Conference**: NeurIPS 2025
**arXiv**: [2501.04686](https://arxiv.org/abs/2501.04686)  
**Code**: [URSA-MATH](https://github.com/URSA-MATH)  
**Area**: LLM Reasoning
**Keywords**: process reward model, multimodal math, GRPO, test-time scaling, chain-of-thought

## TL;DR

This paper proposes URSA, a three-stage framework that sequentially constructs a million-scale multimodal CoT dataset (MMathCoT-1M) for base model training, a dual-perspective process supervision dataset (DualMath-1.1M) for PRM training, and a PS-GRPO algorithm that integrates the PRM into online RL. The resulting 8B model surpasses GPT-4o by an average of 2.7% across six mathematical benchmarks.

## Background & Motivation

**Gap in multimodal reasoning for PRMs**: Process reward models (PRMs) have demonstrated value in test-time scaling (TTS) and reinforcement learning for text-only LLM mathematical reasoning, yet their application to multimodal settings remains largely unexplored.

**Scarcity of high-quality reasoning data**: Existing multimodal mathematical datasets are predominantly answer-only or lack rigorous step-by-step logic, and the performance ceiling of TTS and RL is constrained by the base model's capability.

**Lack of automated annotation methods for multimodal process supervision**: While MCTS can be used for text-based PRM annotation, multimodal settings require simultaneous attention to logical correctness and visual perception consistency, for which no systematic approach has existed.

**Failure modes of PRMs in online RL**: Directly using scalar process rewards as RL objectives leads to reward hacking (where the model learns to satisfy the PRM rather than reason correctly) and length bias (where the PRM tends to penalize longer reasoning chains).

**Potential of test-time scaling**: Best-of-N sampling combined with PRM verification can substantially improve reasoning accuracy, with significant gains achievable from as few as 4 samples.

**Gap between open-source and closed-source models**: Open-source MLLMs at the 8B parameter scale still lag behind GPT-4o on mathematical reasoning, necessitating systematic advances in both data and training paradigms.

## Method

### URSA Three-Stage Framework

**Stage I: Data Construction + Base Model Training**
- Collects 860K math-intensive vision-language alignment data (URSA-Alignment-860K), training only the MLP projection layer.
- Collects 1.43M samples from five open-source datasets, processed in three categories based on format:
    - Answer-only → CoT expansion (reasoning trajectories generated via Gemini)
    - Analysis-formatted → Rewriting (enhanced step-by-step logic and linguistic diversity)
    - CoT-formatted → Format normalization (naturalization of mathematical language)
- After filtering for correctness and consistency, MMathCoT-1M is obtained; full-parameter fine-tuning yields URSA-8B.

**Stage II: Dual-Perspective Process Supervision Data Synthesis**
- **Binary Error Localization (BEL) Engine**: For ~553K incorrect solutions generated zero-shot by URSA-8B, MCTS is used to annotate the first erroneous step, with binary search to accelerate localization ($mc_i > 0$ denotes "potentially correct"); 180K correct samples are added for balance, yielding 773K samples.
- **Misconception Injection Engine (MIE)**: Targeting perception inconsistencies unique to multimodal settings, this engine automatically inserts visual hallucination errors into correct reasoning chains (by confusing similar conditions in images), labeling steps from the injection point onward as negative; yields 302K samples.
- The two sources are merged into DualMath-1.1M to train URSA-8B-RM (binary per-step correctness prediction).

**Stage III: PS-GRPO (Process-Supervised GRPO)**
- **Problem identified**: Scalar process rewards used directly as RL objectives exhibit two failure modes — reward hacking (test accuracy degrades) and length bias (short answers are incentivized).
- **Key insight**: Although scalar PRM rewards are unreliable as direct objectives, their **relative quality ranking is trustworthy** — BoN verification and error identification capabilities remain stable throughout RL training.
- **"Drop-moment" concept**: A drop in consecutive PRM reward scores exceeding threshold $\rho = 0.3$ signals that the PRM questions the reasoning quality at that step.
- **Reward design**: Correct with no drop-moment → reward 1; correct but with drop-moment → penalized to $1 - \gamma$ ($\gamma = 0.5$); incorrect → 0.

## Key Experimental Results

### Table 1: Average Performance across 6 Benchmarks (vs. SOTA)

| Model | Params | Avg. | MathVerse | MathVision | MathVista-GPS |
|-------|--------|------|-----------|------------|---------------|
| GPT-4o | — | 55.5 | 50.2 | 30.4 | 64.7 |
| Gemma3-12B | 12B | 49.8 | 40.1 | 29.1 | 63.6 |
| AtomThink-EMOVA | 8B | 49.5 | 42.5 | 24.9 | 75.9 |
| **URSA-8B** | 8B | **54.7** | 45.7 | 28.7 | 81.7 |
| **URSA-8B-PS-GRPO** | 8B | **58.2** | **50.9** | **31.5** | **83.2** |

### Table 2: PS-GRPO vs. Vanilla GRPO (Relative Improvement)

| Method | Avg. Gain | WE-MATH | MathVision | MathVerse |
|--------|-----------|---------|------------|-----------|
| Vanilla GRPO | +3.1% | Modest gain | Modest gain | Moderate gain |
| **PS-GRPO** | **+6.8%** | ~2× GRPO | ~2× GRPO | Outperforms GRPO |

**BoN verification**: URSA-8B-RM achieves a 16.6% relative improvement on MathVerse at Best-of-4, and reaches 35.1 on MathVision at Best-of-32, surpassing GPT-4o (30.4). Ablation studies show BEL and MIE are complementary — removing either leads to performance degradation.

## Highlights & Insights

1. **Systematic framework design**: The three-stage pipeline (data → PRM → RL) proceeds in a logically layered manner, with each stage offering independent contributions and reusable artifacts (both million-scale datasets are open-sourced).
2. **Elegant design of PS-GRPO**: Rather than using scalar process rewards directly as RL objectives (which risks reward hacking), the approach leverages the PRM's relative ranking capability as a penalty signal, elegantly circumventing known failure modes of PRMs in RL.
3. **Dual-perspective process supervision**: BEL addresses logical errors while MIE addresses perceptual hallucinations, providing the first systematic coverage of both core error types in multimodal reasoning.
4. **8B model surpassing GPT-4o**: For the first time, an open-source 8B model exceeds GPT-4o on MathVision (31.5 vs. 30.4), demonstrating exceptional parameter efficiency.
5. **Comprehensive large-scale open release**: Code, both million-scale datasets, and model checkpoints are fully released.

## Limitations & Future Work

1. **Notable gap on DynaMath**: Performance on dynamic mathematical reasoning remains behind, indicating that small-scale MLLMs still face a bottleneck in robust problem solving.
2. **PRM quality depends on the base model**: MCTS annotation in BEL requires the base model to generate sufficiently diverse rollouts, limiting quality to what URSA-8B itself can produce.
3. **Manual design components in MIE**: The misconception injection strategy relies on manually defined rules (confusing similar conditions), which may fail to cover all types of visual perception errors.
4. **Computational cost insufficiently discussed**: The combined training cost of three-stage training, MCTS annotation, and PS-GRPO is substantial, limiting experiments at larger scales.
5. **PS-GRPO hyperparameter sensitivity**: The choice of drop-moment threshold $\rho$ and penalty coefficient $\gamma$ may affect performance, yet the paper does not present sufficient ablation on these parameters.

## Rating

| Dimension | Score | Remarks |
|-----------|-------|---------|
| Novelty | ⭐⭐⭐⭐ | First systematic exploration of multimodal PRMs; PS-GRPO is elegantly designed |
| Technical Depth | ⭐⭐⭐⭐⭐ | Three stages covering data, PRM, and RL, each with thorough analysis |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Six benchmarks, multiple ablations, BoN analysis, training curves, and comparison with GRPO |
| Practical Impact | ⭐⭐⭐⭐ | Open-sourced datasets and models provide direct value, though method complexity is high |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DreamPRM: Domain-Reweighted Process Reward Model for Multimodal Reasoning](dreamprm_domain-reweighted_process_reward_model_for_multimodal_reasoning.md)
- [\[NeurIPS 2025\] Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning](stop_summation_minform_credit_assignment_is_all_process_rewa.md)
- [\[NeurIPS 2025\] Know What You Don't Know: Uncertainty Calibration of Process Reward Models](know_what_you_dont_know_uncertainty_calibration_of_process_reward_models.md)
- [\[ICML 2026\] GRPO is Secretly a Process Reward Model](../../ICML2026/llm_reasoning/grpo_is_secretly_a_process_reward_model.md)
- [\[NeurIPS 2025\] ARM: Adaptive Reasoning Model](arm_adaptive_reasoning_model.md)

</div>

<!-- RELATED:END -->
