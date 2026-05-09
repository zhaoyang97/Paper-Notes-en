---
title: >-
  [Paper Note] ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training
description: >-
  [ACL 2026][LLM Alignment][Generative Reward Model] ConsistRM proposes a consistency-aware self-training framework for generative reward models (GRMs), comprising two modules: temporal consistency pseudo-labels (integrating online state and historical memory preference consistency) and semantic consistency critique rewards (measuring semantic similarity across multiple generated critiques). Without human annotation, the framework achieves an average improvement of 1.5% across five benchmarks while substantially mitigating position bias.
tags:
  - ACL 2026
  - LLM Alignment
  - Generative Reward Model
  - Self-Training
  - Consistency-Aware
  - Pseudo Labels
  - Position Bias
date: 2026-05-08
content_hash: fecfb24ef64b1b71
---

# ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training

**Conference**: ACL 2026
**arXiv**: [2604.07484](https://arxiv.org/abs/2604.07484)
**Code**: [GitHub](https://github.com/yuliangCarmelo/ConsistRM)
**Area**: Alignment RLHF / Reward Modeling
**Keywords**: Generative Reward Model, Self-Training, Consistency-Aware, Pseudo Labels, Position Bias

## TL;DR

ConsistRM proposes a consistency-aware self-training framework for generative reward models (GRMs), comprising two modules: temporal consistency pseudo-labels (integrating online state and historical memory preference consistency) and semantic consistency critique rewards (measuring semantic similarity across multiple generated critiques). Without human annotation, the framework achieves an average improvement of 1.5% across five benchmarks while substantially mitigating position bias.

## Background & Motivation

**State of the Field**: Generative reward models (GRMs) replace traditional scalar reward models by generating textual critiques and preference labels, offering stronger expressiveness and generalization. Representative works include DeepSeek-GRM (generative critiques + self-derived rules) and RM-R1 (distilled reasoning traces + reinforcement learning).

**Limitations of Prior Work**: GRM training faces two major challenges: (1) reliance on costly human-annotated data, limiting scalability; and (2) self-training methods (e.g., majority-vote pseudo-labels in TTRL) are prone to reward hacking and early overfitting to noisy pseudo-labels, as the reward signal is tightly coupled to the policy model.

**Root Cause**: Self-training requires reliable pseudo-labels, yet model-generated pseudo-labels are inherently unstable — single-round voting is susceptible to sampling randomness, and pseudo-label bias accumulates in later training stages.

**Paper Goals**: Design a stable and effective GRM self-training framework that requires no human annotation.

**Starting Point**: Leverage the model's intrinsic "consistency" signals as a source of self-supervision — if a model produces consistent preference judgments for the same sample across multiple samples and training rounds, that judgment is more likely to be correct.

**Core Idea**: Construct reliable pseudo-labels via temporal consistency (current round + historical memory), and provide fine-grained rewards via semantic consistency (similarity across multiple critique texts), enabling stable annotation-free GRM self-training.

## Method

### Overall Architecture

Given a query $q$ and two candidate responses $(a_1, a_2)$, the GRM generates a structured output $o = (c, y)$, where $c$ is a textual critique and $y \in \{-1, 1\}$ is a preference label. ConsistRM provides self-supervised signals for GRPO reinforcement learning through two core modules: Consistency-Aware Answer Reward (CAAR) and Consistency-Aware Critique Reward (CACR).

### Key Designs

1. **Consistency-Aware Answer Reward (CAAR)**:

    - **Function**: Construct reliable pseudo-labels for self-training.
    - **Mechanism**: Fuses two layers of consistency signals. The online state consistency $s_{\text{online}}^{(n)} = \frac{1}{K}\sum_{j=1}^{K} y_j$ aggregates preference predictions from $K$ rollouts in the current round; the memory-driven consistency $s_{\text{memory}}^{(n)} = \frac{1}{n-1}\sum_{i=0}^{n-1} \hat{y}^{(i)}$ aggregates pseudo-labels across all historical rounds. The final pseudo-label is $\hat{y}^{(n)} = \text{sgn}(s_{\text{online}}^{(n)} + s_{\text{memory}}^{(n)})$; when the two signals disagree, the output is 0 (no supervision provided), preventing low-confidence samples from dominating optimization.
    - **Design Motivation**: Online voting alone is unreliable in early training; historical memory provides a stable anchor. The ternary label scheme (+1/−1/0) explicitly handles uncertain samples, making the approach more robust than forced binary classification.

2. **Consistency-Aware Critique Reward (CACR)**:

    - **Function**: Provide fine-grained rewards for critique text quality.
    - **Mechanism**: Each critique $c_j$ is encoded into a vector using Qwen3-4B-Embedding; a cosine similarity matrix is computed and critiques are ranked by semantic consistency. Critiques ranked in the top $p$ with correct preference labels receive an additional reward $r_j^{(c)} = 0.1$. The intuition is that semantically consistent critiques across multiple generations indicate the model has converged to a stable evaluation region.
    - **Design Motivation**: CAAR supervises only the outcome (preference label), while CACR supplements process supervision (critique content); the two are complementary. Semantically consistent critiques are more likely to reflect reliable assessments.

3. **Format Constraints and Combined Reward**:

    - **Function**: Ensure valid output format and integrate multi-level rewards.
    - **Mechanism**: The final reward is $r^{(n)} = r_j^{(a,n)} + r_j^{(c,n)}$ (when format is valid and $\hat{y} \neq 0$); format violations yield $r = -5$, and $\hat{y} = 0$ yields $r = 0$. GRPO is used for reinforcement training with global batch size 64, learning rate 1e-6, 8 rollouts, and KL coefficient 0.001.
    - **Design Motivation**: Format constraints ensure parseable outputs; the combined reward provides consistent optimization signals at different levels of granularity.

### Loss & Training

GRPO is used for training over 4 epochs, with maximum generation lengths of 1024 (training) / 2048 (inference) and temperature 1.0 (training) / 0 (inference). Training follows an SFT phase (on HelpSteer3) followed by an RFT (reinforcement fine-tuning) phase, where ConsistRM replaces the reward signal during RFT.

## Key Experimental Results

### Main Results

**Five-Benchmark Performance on Qwen3-8B**

| Method | RewardBench | PPE Pref | RM-Bench | RMB | JudgeBench | Avg | Δ |
|--------|------------|----------|----------|-----|------------|-----|---|
| Qwen3-8B (Base) | 81.6 | 63.8 | 75.8 | 78.8 | 54.3 | 70.9 | - |
| + SFT | 82.7 | 65.0 | 77.1 | 76.9 | 51.7 | 70.7 | -0.2 |
| + RFT | 85.4 | 65.4 | 78.2 | 78.2 | 55.4 | 72.5 | +1.6 |
| + TTRL | 85.3 | 65.0 | 77.4 | 74.2 | 56.8 | 71.7 | +0.8 |
| + ConsistRM | **85.6** | **67.7** | **78.3** | **79.1** | **56.9** | **73.5** | **+2.6** |

### Ablation Study

| Configuration | RewardBench | PPE | RM-Bench | RMB | JudgeBench | Avg | Δ |
|--------------|------------|-----|----------|-----|------------|-----|---|
| ConsistRM | 85.6 | 67.7 | 78.3 | 79.1 | 56.9 | 73.5 | - |
| w/o CACR | 84.9 | 64.8 | 77.3 | 78.1 | 56.0 | 72.2 | -1.3 |
| w/o Online-State | 85.5 | 64.1 | 78.6 | 76.7 | 56.7 | 72.3 | -1.2 |
| w/o Memory-Driven | 84.3 | 63.1 | 75.4 | 74.2 | 54.8 | 70.4 | **-3.2** |

### Key Findings

- Memory-driven consistency preference is the most critical component (removal causes a 3.2-point drop), demonstrating that historical information is essential for pseudo-label quality.
- ConsistRM achieves substantially greater position bias mitigation (+5.3 vs. +1.4 for RFT), as consistency rewards encourage the model to focus on content rather than position.
- ConsistRM enables a 4B model to reach the performance level of an 8B model through multi-round voting.
- Replacing CACR with token-level confidence (DeepConfidence) leads to reward hacking and performance degradation.

## Highlights & Insights

- The core assumption that "consistency implies reliability" is concise and compelling — leveraging the model's intrinsic consistency signals in place of external annotation.
- The temporal memory mechanism (historical pseudo-label aggregation) is a key innovation, providing a stable anchor across training rounds.
- The ternary label design (+1/−1/0) elegantly handles uncertain samples, avoiding contamination from noisy labels.
- ConsistRM also improves generation efficiency — producing more concise critiques (1,717 vs. 1,924 tokens).

## Limitations & Future Work

- Semantic consistency evaluation operates only at the overall critique level, without fine-grained alignment of individual semantic segments within critiques.
- Validation is limited to Qwen3 and LLaMA-3.1; generalizability across broader model families remains to be verified.
- Hyperparameters of the consistency reward (top-$p$ ratio, CACR reward value of 0.1) may require task-specific tuning.

## Related Work & Insights

- **vs. TTRL**: TTRL uses majority voting for pseudo-labels but lacks temporal consistency across training rounds, leading to bias accumulation in later stages; ConsistRM's historical memory mechanism substantially mitigates this issue.
- **vs. DeepSeek-GRM**: The latter requires human-annotated reward signals, whereas ConsistRM is entirely annotation-free.
- **Insight**: Consistency signals may serve as a general, low-cost quality indicator for self-training scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The consistency-aware self-training paradigm is elegantly designed, particularly the temporal consistency memory mechanism.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five benchmarks, four models, comprehensive ablations, and analyses of position bias and multi-round voting.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear and experimental analysis is thorough.
- **Value**: ⭐⭐⭐⭐ Provides a practical and effective solution for annotation-free GRM training.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] GRAM-R²: Self-Training Generative Foundation Reward Models for Reward Reasoning](../../AAAI2026/llm_alignment/gram-r2_self-training_generative_foundation_reward_models_for_reward_reasoning.md)
- [\[ACL 2026\] Reward Modeling for Scientific Writing Evaluation](reward_modeling_for_scientific_writing_evaluation.md)
- [\[ACL 2026\] Towards Bridging the Reward-Generation Gap in Direct Alignment Algorithms](towards_bridging_the_reward-generation_gap_in_direct_alignment_algorithms.md)
- [\[ACL 2026\] Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling](aligning_agents_via_planning_a_benchmark_for_trajectory-level_reward_modeling.md)
- [\[CVPR 2026\] MapReduce LoRA: Advancing the Pareto Front in Multi-Preference Optimization for Generative Models](../../CVPR2026/llm_alignment/mapreduce_lora_advancing_the_pareto_front_in_multi-preference_optimization_for_g.md)

<!-- RELATED:END -->
