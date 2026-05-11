---
title: >-
  [Paper Note] Targeted Exploration via Unified Entropy Control for Reinforcement Learning
description: >-
  [ACL 2026 Findings][Multimodal VLM][entropy control] This paper proposes UEC-RL, a unified bidirectional entropy control framework that addresses entropy collapse and training instability in GRPO by performing high-tempe…
tags:
  - "ACL 2026 Findings"
  - "Multimodal VLM"
  - "entropy control"
  - "GRPO"
  - "exploration strategy"
  - "reinforcement learning"
  - "reasoning enhancement"
date: 2026-05-08
content_hash: df6c703e10caf544
---

# Targeted Exploration via Unified Entropy Control for Reinforcement Learning

**Conference**: ACL 2026 Findings
**arXiv**: [2604.14646](https://arxiv.org/abs/2604.14646)
**Code**: [GitHub](https://github.com/597358816/UEC-RL)
**Area**: Multimodal VLM
**Keywords**: entropy control, GRPO, exploration strategy, reinforcement learning, reasoning enhancement

## TL;DR

This paper proposes UEC-RL, a unified bidirectional entropy control framework that addresses entropy collapse and training instability in GRPO by performing high-temperature targeted exploration on difficult prompts (entropy increase) and consolidating high-quality trajectories via an experience replay stabilizer (entropy decrease), achieving a 37.9% relative improvement on Geometry3K.

## Background & Motivation

**Background**: Reinforcement learning (RL) has become a core paradigm for post-training of LLMs/VLMs. GRPO, as a lightweight alternative to PPO, has been widely adopted—it eliminates the critic network, estimates advantages via intra-group normalized rewards, and offers high computational efficiency with competitive reasoning performance.

**Limitations of Prior Work**: GRPO exhibits two prominent issues on complex reasoning tasks: (1) **Entropy collapse**—policy entropy drops rapidly, causing the model to converge prematurely to low-diversity behaviors and failing to discover low-probability but valuable reasoning paths; (2) **Training instability**—in complex scenarios such as multimodal reasoning, the correctness of sampled outputs varies greatly, and GRPO's group-normalized rewards cannot provide sufficient variance reduction, resulting in fragile gradient updates.

**Key Challenge**: GRPO lacks a bidirectional entropy regulation mechanism—it can neither actively increase entropy to enhance exploration nor stabilize entropy in high-variance environments to ensure convergence. Existing remedies either introduce update variance (DAPO's clip-higher) or introduce optimization bias (entropy rewards/entropy-shaping advantage optimize entropy-related objectives rather than task rewards).

**Goal**: To design a unified framework that increases entropy for deep exploration when needed, and decreases entropy to ensure convergence when exploration becomes ineffective.

**Key Insight**: Based on the entropy variation theorem of natural policy gradients—when high-advantage actions have low probability under the current policy (negative covariance), policy updates increase entropy; otherwise they decrease it. Therefore, raising the sampling temperature amplifies the negative covariance effect, selectively increasing exploration for difficult prompts.

**Core Idea**: Two synergistic components realize bidirectional entropy control—a targeted exploration mechanism (high-temperature sampling for difficult prompts with filtering of valuable trajectories) and a controllable entropy stabilizer (experience replay to repeatedly reinforce high-quality trajectories and reduce entropy), dynamically balancing exploration and exploitation throughout training.

## Method

### Overall Architecture

UEC-RL builds upon GRPO. For each batch of prompts, $G$ trajectories are sampled at standard temperature. If all $G$ trajectories for a given prompt receive zero reward (marked as "difficult"), an additional $G'$ trajectories are sampled at an elevated temperature $t' > 1$. Valuable trajectories are filtered for gradient updates, while high-quality trajectories are stored in a replay buffer. Periodic replay from the buffer stabilizes training.

### Key Designs

1. **Targeted Exploration Mechanism**:

    - **Function**: Adaptively expands the exploration space for difficult prompts to discover low-probability but valuable reasoning trajectories.
    - **Mechanism**: If all $G$ trajectories from standard sampling fail ($\max_i R_i = 0$), an additional $G'$ trajectories are sampled from a softened distribution (temperature $t' > 1$). From all trajectories, only two categories are retained: regular samples $O_R$ (non-zero advantage) and exploration samples $O_H$ (trajectories with positive advantage from expanded sampling); noisy samples with low advantage are filtered out.
    - **Design Motivation**: Based on the entropy variation theorem, raising temperature $t'$ narrows the probability gap between high- and low-probability actions, making negative covariance more likely to occur, thereby controllably increasing policy entropy. Critically, this is activated only for difficult prompts, leaving simple prompts unaffected.

2. **Controllable Entropy Stabilizer (Experience Replay)**:

    - **Function**: Prevents uncontrolled entropy growth by guiding stable policy convergence through repeated reinforcement of high-quality trajectories.
    - **Mechanism**: Trajectories with positive advantage (advantage $> $ threshold $A_0$) discovered during exploration are stored in a fixed-size replay buffer $\mathcal{B}_{replay}$, retaining only the most recent $s'$ trajectories. A replay update is performed by sampling from the buffer every $f_{replay}$ steps. Repeatedly reinforcing high-advantage trajectories shifts probability mass toward correct reasoning patterns, which, per Theorem 4.2, produces positive covariance and thus decreases entropy.
    - **Design Motivation**: High-quality trajectories discovered through exploration have initially low probability; using them only once limits their impact. Replay amplifies their gradient signal while producing a theoretically grounded entropy-reduction effect, enabling a natural transition from exploration to convergence.

3. **Selective Trajectory Retention**:

    - **Function**: Filters noisy samples, retaining only trajectories that are valuable for optimization.
    - **Mechanism**: Regular samples retain all trajectories with non-zero advantage ($\hat{A}_{i,t} \neq 0$$); exploration samples retain only those with positive advantage ($\hat{A}_{i,t} \geq 0$). Low-advantage exploration samples introduce noisy gradients that impair optimization and generalization.
    - **Design Motivation**: Effective exploration should emphasize informative and high-quality diversity, rather than indiscriminately encouraging randomness.

### Loss & Training

The method is based on the GRPO objective with a clipped surrogate loss and KL divergence regularization. Key hyperparameters include the exploration temperature $t'$, exploration group size $G'$, replay buffer size $s'$, and replay frequency $f_{replay}$.

## Key Experimental Results

### Main Results (Text Reasoning, Qwen2.5-math-7B)

| Method | AIME24 | MATH | GSM8K | MMLU | Avg. |
|--------|--------|------|-------|------|------|
| GRPO | 25.8 | 77.6 | 87.1 | 45.0 | 50.34 |
| DAPO | 24.3 | 78.3 | 87.6 | 48.5 | 51.77 |
| **UEC-RL** | **28.5** | **80.4** | **87.9** | **50.2** | **53.62** |

Geometry3K Multimodal Reasoning (Qwen2.5-VL-7B):

| Method | Accuracy | vs UEC-RL |
|--------|----------|-----------|
| Baseline | 38.44 | −16.97 |
| GRPO | 50.75 | −4.66 |
| **UEC-RL** | **55.41** | — |

### Ablation Study

| Configuration | Geometry3K | Note |
|---------------|-----------|------|
| UEC-RL (full) | 55.41 | Full model |
| w/o exploration | ~50 | Reverts to GRPO level |
| w/o replay | ~52 | Training instability |
| w/o selective retention | ~51 | Noisy samples impair optimization |

### Key Findings
- UEC-RL consistently outperforms all RL baselines on both text and multimodal reasoning, with average gains of +2.88% (Qwen2.5-math-7B) and +1.07% (VLM).
- A 37.9% relative improvement is achieved on Geometry3K (38.44→55.41), with higher training efficiency (per-step time only 0.79× that of GRPO).
- UEC-RL also consistently leads in Pass@k evaluations, indicating not only higher top-1 accuracy but also better generation diversity.
- Both the exploration and stabilization components are necessary; neither alone achieves optimal performance.

## Highlights & Insights
- **The bidirectional entropy control concept** is elegant: rather than simply "slowing entropy decay" (as in DAPO), the framework actively increases entropy when exploration is needed and actively decreases it when convergence is required. The theoretical grounding via the entropy variation theorem makes the design principled.
- **The "difficult prompt detection → targeted exploration" strategy** is highly practical: difficulty is assessed at zero cost (all samples failing = difficult), and additional computational budget is allocated only to difficult samples, making resource utilization efficient.
- **Application of experience replay to LLM RL**: adapting the classical replay mechanism to GRPO training not only improves sample efficiency but also provides a theoretically guaranteed entropy stabilization effect through repeated reinforcement.

## Limitations & Future Work
- Validation is limited to 7B/8B-scale models; effectiveness on larger models remains unknown.
- Hyperparameters such as exploration temperature $t'$ and replay frequency $f_{replay}$ require tuning.
- Difficult prompt detection (all samples failing) is binary and may miss "partially difficult" samples.
- Theoretical analysis is based on a natural policy gradient approximation, leaving a gap with the actual clipped objective.
- The replay buffer may introduce distribution shift, though retaining only the most recent trajectories partially mitigates this.

## Related Work & Insights
- **vs GRPO**: GRPO lacks entropy regulation; UEC-RL adds bidirectional entropy control while sharing the same base objective.
- **vs DAPO**: DAPO slows entropy decay via clip-higher but introduces update variance; UEC-RL controls entropy more precisely through targeted exploration and the replay stabilizer.
- **vs Entropy Reward / KL-cov**: These methods introduce bias by optimizing entropy-related terms; UEC-RL indirectly influences entropy by adjusting the sampling strategy without modifying the optimization objective.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The bidirectional entropy control framework is conceptually clear and theoretically well-grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers text and multimodal reasoning, multiple benchmarks, and both Pass@1 and Pass@k evaluations.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with coherent theoretical derivations and experimental analysis.
- **Value**: ⭐⭐⭐⭐ Offers practical guidance for improving GRPO training, with open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unified Reinforcement and Imitation Learning for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/unified_reinforcement_and_imitation_learning_for_vision-language_models.md)
- [\[ACL 2026\] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge](multi-task_reinforcement_learning_for_enhanced_multimodal_llm-as-a-judge.md)
- [\[CVPR 2026\] Explore with Long-term Memory: A Benchmark and Multimodal LLM-based Reinforcement Learning Framework for Embodied Exploration](../../CVPR2026/multimodal_vlm/explore_with_long-term_memory_a_benchmark_and_multimodal_llm-based_reinforcement.md)
- [\[CVPR 2026\] Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](../../CVPR2026/multimodal_vlm/reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)
- [\[CVPR 2026\] MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models](../../CVPR2026/multimodal_vlm/moe-grpo_optimizing_mixture-of-experts_via_reinforcement_learning_in_vision-lang.md)

</div>

<!-- RELATED:END -->
