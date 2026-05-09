---
title: >-
  [Paper Note] Adaptive Dual Reasoner: Large Reasoning Models Can Think Efficiently by Hybrid Reasoning
description: >-
  [NeurIPS 2025][LLM Reasoning][Reasoning Efficiency] This paper proposes the Adaptive Dual Reasoner (ADR), which enables reasoning models to dynamically switch between fast thinking (compressing simple reasoning steps) and slow thinking (preserving depth for complex steps). Through SFT cold-start combined with EHPO (Entropy-guided Hybrid Policy Optimization), ADR achieves up to 6.1% accuracy improvement on mathematical reasoning benchmarks while reducing reasoning tokens by 49.5%–59.3%.
tags:
  - NeurIPS 2025
  - LLM Reasoning
  - Reasoning Efficiency
  - Hybrid Reasoning
  - Fast-and-Slow Thinking
  - Entropy Guidance
  - Reinforcement Learning
date: 2026-05-08
content_hash: 837a51b2d711abab
---

# Adaptive Dual Reasoner: Large Reasoning Models Can Think Efficiently by Hybrid Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2510.10207](https://arxiv.org/abs/2510.10207)
**Code**: None (Tencent YouTu Lab)
**Area**: LLM Reasoning / LLM Efficiency
**Keywords**: Reasoning Efficiency, Hybrid Reasoning, Fast-and-Slow Thinking, Entropy Guidance, Reinforcement Learning

## TL;DR
This paper proposes the Adaptive Dual Reasoner (ADR), which enables reasoning models to dynamically switch between fast thinking (compressing simple reasoning steps) and slow thinking (preserving depth for complex steps). Through SFT cold-start combined with EHPO (Entropy-guided Hybrid Policy Optimization), ADR achieves up to 6.1% accuracy improvement on mathematical reasoning benchmarks while reducing reasoning tokens by 49.5%–59.3%.

## Background & Motivation

**Background**: Large Reasoning Models (LRMs) such as DeepSeek-R1 and Qwen3 have achieved remarkable performance via long Chain-of-Thought reasoning, yet suffer from severe "overthinking"—generating verbose reasoning steps even for simple sub-problems.

**Limitations of Prior Work**: (a) Length-driven compression methods may under-explore complex steps; (b) coarse-grained fast/slow mode switching fails to adapt to varying sub-problem complexity within a reasoning trajectory; (c) static rollout strategies limit deep exploration on difficult problems.

**Key Challenge**: A fundamental tension exists between reducing reasoning length (efficiency) and maintaining reasoning depth (accuracy), requiring adaptive allocation of "cognitive resources" based on the complexity of each reasoning step.

**Key Insight**: Decompose the reasoning trajectory into reasoning units, annotate each as easy or hard based on entropy, compress easy units, and preserve hard units.

**Core Idea**: Entropy-guided dynamic branching—when transitioning from easy to hard mode, expand multiple exploration branches to increase reasoning breadth as a compensation for reduced depth.

## Method

### Overall Architecture
Two-stage training: (1) SFT cold-start—teach the model to distinguish easy/hard reasoning units using hybrid reasoning data; (2) EHPO reinforcement learning—optimize reasoning allocation using entropy-guided dynamic rollout and difficulty-aware rewards.

### Key Designs

1. **Hybrid Reasoning Data Construction**:

    - Function: Automatically reformats long reasoning trajectories into a mixed easy/hard format.
    - Mechanism: (a) Decompose CoT into reasoning units; (b) annotate units by token entropy—units containing reflection/verification keywords ("Wait", "However", "Alternatively") with high entropy are labeled hard, and the rest are labeled easy; (c) easy units are compressed in CoD-style, while hard units retain their original depth.
    - Output format: `<think> <easy> u1 </easy> <hard> u2 </hard> ... </think> a`
    - A dataset of 300K training samples is constructed from OpenMathReasoning.

2. **EHPO Reward Design**:

    - Function: Jointly optimize reasoning accuracy and efficiency.
    - Total reward: $R = \mathcal{R}_{format} \times \mathcal{R}_{accuracy} \times \mathcal{R}_{unit} \times \mathcal{R}_{mode}$
    - **Unit semantic reward** $\mathcal{R}_{unit}$: Ensures easy units contain no reflection keywords and hard units do, maintaining semantic consistency across the two modes.
    - **Mode control reward** $\mathcal{R}_{mode}$: Difficulty-aware—encourages more easy-mode usage on simple problems and permits more hard-mode usage on difficult ones.
        - $\mathcal{R}_{mode} = \beta + (1-\beta)(N_{pass}/N \cdot p_{easy} + (1-N_{pass}/N) \cdot p_{hard})$
        - Problem difficulty is estimated via $N_{pass}/N$ (fraction of correct samples).

3. **Entropy-guided Dynamic Rollout (EDR)**:

    - Function: Expands multiple reasoning branches at critical transition points (easy → hard).
    - Core Observation: The terminal entropy of easy units is typically higher than their initial entropy (uncertainty increases as reasoning proceeds), whereas hard units exhibit the opposite pattern—hence easy→hard transitions are high-uncertainty points.
    - Mechanism: At easy→hard transitions, branches are generated with probability $SP = \alpha + \Delta H$, where $\alpha=0.5$ is the base probability and $\Delta H$ is the normalized entropy difference.
    - Design Motivation: The mode control reward compresses the exploration space; EDR compensates by increasing breadth (multiple branches) to offset reduced depth.
    - Effect: EDR improves average AES from 0.51 to 0.70.

### Loss & Training
- Backbone model: DeepSeek-R1-Distill-Qwen-1.5B
- Cold-start SFT: 300K samples from OpenMathReasoning
- EHPO RL: DeepScaleR-Preview dataset, two stages (8K/16K max tokens)
- EDR is enabled only in the 16K stage
- Evaluation: pass@1 (average over 16 samples)

## Key Experimental Results

### Main Results

| Method | AIME2024 Acc. | AIME2024 Tokens | MATH500 Acc. | MATH500 Tokens | Avg AES |
|--------|--------------|-----------------|--------------|----------------|---------|
| Baseline (R1-1.5B) | 30.4% | 12290 | 81.7% | 4802 | - |
| O1-Pruner | - | - | 84.3% | 2913↓39% | 0.35 |
| DRP | 33.3% | 6135↓50% | 82.0% | 2122↓56% | 0.68 |
| ACPO | 30.0% | 6670↓46% | 81.0% | 1679↓65% | 0.50 |
| **ADR** | **36.5%** | **6110↓50%** | 81.0% | **1955↓59%** | **0.70** |

### Ablation Study

| Configuration | AIME2024 Acc. | Avg AES | Notes |
|---------------|--------------|---------|-------|
| ADR w/o EDR | 33.8% | 0.51 | RL without dynamic rollout |
| **ADR (full)** | **36.5%** | **0.70** | With entropy-guided dynamic rollout |

### Key Findings
- **6.1% gain on AIME2024**: ADR achieves the largest accuracy improvement over the baseline on the hardest competition mathematics benchmark, while reducing tokens by 50%.
- **EDR is critical**: Without EDR, RL training yields limited gains (+3.4% on AIME2024); adding EDR produces a significant jump to +6.1%.
- **50–60% reduction in reasoning tokens**: Achieved by compressing simple reasoning steps into easy mode.
- **Difficulty-adaptive behavior**: Simple problems predominantly use easy mode, while difficult problems automatically increase the proportion of hard mode.

## Highlights & Insights
- **Reasoning-unit-level granularity**: Rather than deciding fast/slow at the problem level, ADR dynamically switches modes at each unit within the reasoning process—enabling finer-grained "cognitive resource allocation."
- **Entropy as an uncertainty signal**: The model's own token generation entropy is used to detect when deeper reasoning is needed, providing a lightweight and natural signal that requires no external annotation.
- **Breadth compensating for depth**: When reducing reasoning depth (compressing easy units) risks degrading performance, the approach compensates by expanding the number of branches at critical transition points—an interesting computational efficiency trade-off.
- **Automatic data construction**: Existing reasoning data can be automatically converted into hybrid format using entropy and keyword matching, making the approach scalable to any reasoning dataset.

## Limitations & Future Work
- **Validated only on a 1.5B model**: Effectiveness on larger models (7B, 32B) remains unknown.
- **Mathematics-only evaluation**: Applicability to code reasoning, logical reasoning, and other tasks has not been verified.
- **Limitations of keyword matching**: Using fixed keywords ("Wait", "However") to annotate hard units may lack precision.
- **Future directions**: (1) Scale to larger models and more diverse reasoning tasks; (2) replace keyword matching with a learned classifier; (3) make the easy/hard boundary learnable rather than predefined.

## Related Work & Insights
- **vs. O1-Pruner**: O1-Pruner pre-samples and prunes from complete reasoning traces; ADR distinguishes fast and slow modes during generation itself, making it more proactive.
- **vs. ACPO**: ACPO also performs mode switching but uses standard GRPO; ADR's EHPO incorporates entropy-guided dynamic rollout and achieves superior results (AES 0.70 vs. 0.50).
- **vs. DRP**: DRP relies on distillation and pruning, while ADR uses RL with adaptive strategies—ADR achieves higher accuracy on AIME2024.
- **vs. System 1/System 2 framework**: ADR's fast/slow modes correspond to Kahneman's System 1/System 2, but the distinction is implemented at the reasoning-unit level rather than the problem level.

## Rating
- Novelty: ⭐⭐⭐⭐ — Entropy-guided dynamic rollout is a novel RL training strategy; reasoning-unit-level granularity control is creative.
- Experimental Thoroughness: ⭐⭐⭐ — Validated only on a 1.5B model and mathematical reasoning; limited in scale and task diversity.
- Writing Quality: ⭐⭐⭐⭐ — Method descriptions are clear and ablation studies are thorough.
- Value: ⭐⭐⭐⭐ — Practically significant for reasoning efficiency optimization; the entropy-guided approach has broad applicability.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Let Me Think! A Long Chain-of-Thought Can Be Worth Exponentially Many Short Ones](let_me_think_a_long_chainofthought_can_be_worth_exponentiall.md)
- [\[NeurIPS 2025\] Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models](reasoning_models_hallucinate_more_factuality-aware_reinforcement_learning_for_la.md)
- [\[NeurIPS 2025\] ARM: Adaptive Reasoning Model](arm_adaptive_reasoning_model.md)
- [\[NeurIPS 2025\] Large Language Models Can Learn and Generalize Steganographic Chain-of-Thought under Process Supervision](large_language_models_can_learn_and_generalize_steganographic_chain-of-thought_u.md)
- [\[NeurIPS 2025\] Re-FORC: Adaptive Reward Prediction for Efficient Chain-of-Thought Reasoning](re-forc_adaptive_reward_prediction_for_efficient_chain-of-thought_reasoning.md)

<!-- RELATED:END -->
