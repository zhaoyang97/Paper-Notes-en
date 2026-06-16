---
title: >-
  [Paper Note] ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training
description: >-
  [ACL 2026][Alignment & RLHF][Paper Note] ConsistRM proposes a consistency-aware self-training framework. By utilizing two modules—temporal consistency pseudo-labels (fusing online state and historical memory preference) and semantic consistency critique rewards (measuring semantic similarity of multiple generated critiques)—it improves the average performance
tags:
  - ACL 2026
  - Alignment & RLHF
date: 2026-05-08
content_hash: e27399cd7ee3a2d2
---
# ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training

**Conference**: ACL 2026  
**arXiv**: [2604.07484](https://arxiv.org/abs/2604.07484)  
**Code**: [GitHub](https://github.com/yuliangCarmelo/ConsistRM)  
**Area**: Alignment RLHF / Reward Models  
**Keywords**: Generative Reward Models, Self-training, Consistency-Aware, Pseudo-labels, Position Bias

## TL;DR

ConsistRM proposes a consistency-aware self-training framework. By utilizing two modules—temporal consistency pseudo-labels (fusing online state and historical memory preference) and semantic consistency critique rewards (measuring semantic similarity of multiple generated critiques)—it improves the average performance of generative reward models across five benchmarks by 1.5% without human annotation, while significantly mitigating the position bias problem.

## Background & Motivation

**Background**: Generative Reward Models (GRMs) replace traditional scalar reward models by generating textual critiques and preference labels, offering stronger expressiveness and generalization. Representative works include DeepSeek-GRM (critique generation + self-derived rules) and RM-R1 (distilled reasoning trajectories + reinforced learning).

**Limitations of Prior Work**: GRM training faces two major challenges: (1) dependency on expensive human-annotated data, which limits scalability; (2) self-training methods (such as majority voting pseudo-labels in TTRL) easily lead to reward hacking and early overfitting to noisy pseudo-labels because reward signals are highly coupled with policy models.

**Key Challenge**: Self-training requires reliable pseudo-labels, but model-generated pseudo-labels are inherently unstable—single votes are susceptible to sampling randomness, and pseudo-label biases accumulate in later training stages.

**Goal**: To design a stable and effective GRM self-training framework that requires no human annotation.

**Key Insight**: Leverage the model's internal "consistency" signal as a source of self-supervision—if the model provides consistent preference judgments for the same sample across multiple samplings and training rounds, the judgment is more likely to be correct.

**Core Idea**: Construct reliable pseudo-labels using temporal consistency (current round + historical memory) and provide fine-grained rewards using semantic consistency (similarity across multiple critique texts) to achieve stable GRM self-training without labels.

## Method

### Overall Architecture
ConsistRM establishes the self-training of generative reward models on the principle that "consistency implies reliability." Given a query $q$ and two candidate answers $(a_1, a_2)$, the GRM generates a structured output $o = (c, y)$, where $c$ is a textual critique and $y \in \{-1, 1\}$ is a preference label. The framework does not rely on any human annotation; instead, it refines supervision from two internal consistency signals: Consistency-Aware Answer Reward (CAAR), which uses current voting and historical memory to determine reliable pseudo-labels; and Consistency-Aware Critique Reward (CACR), which measures process quality via semantic similarity of multiple critique texts. These signals are combined under format constraints into a final reward for GRPO reinforcement learning.

```mermaid
graph TD
    A["Query $q$ + Candidate Answers $(a_1, a_2)$"] --> B["GRM samples $K$ times<br/>Output critique $c$ and preference $y$"]
    B --> C
    B --> D
    subgraph C["Consistency-Aware Answer Reward (CAAR)"]
        direction TB
        C1["Online State Consistency<br/>Current round $K$-vote mean"] --> C3["sgn function<br/>Pseudo-label $\hat{y} \in \{-1, 0, 1\}$"]
        C2["Memory-Driven Consistency<br/>Historical pseudo-label mean"] --> C3
    end
    D["Consistency-Aware Critique Reward (CACR)<br/>Critique vectorization → Cosine similarity → top-$p$ correct +0.1"]
    C --> E["Format Constraint & Combined Reward<br/>Valid format and $\hat{y} \neq 0$: $r = r_a + r_c$<br/>Invalid: -5; $\hat{y} = 0$: 0"]
    D --> E
    E --> F["GRPO RL updates GRM"]
    F -.->|Pseudo-label written to memory| C2
```

### Key Designs

**1. Consistency-Aware Answer Reward (CAAR): Calibrating via Online Voting and Historical Memory**

The greatest risk in self-training is the instability of pseudo-labels—single-round voting is affected by sampling randomness, and biases accumulate later in training. CAAR thus fuses two layers of consistency signals: Online State Consistency $s_{\text{online}}^{(n)} = \frac{1}{K}\sum_{j=1}^{K} y_j$ aggregates preference predictions from $K$ rollouts in the current round, while Memory-Driven Consistency $s_{\text{memory}}^{(n)} = \frac{1}{n-1}\sum_{i=0}^{n-1} \hat{y}^{(i)}$ aggregates pseudo-labels from all historical rounds to provide a stable anchor for current judgment.

The final pseudo-label is obtained by taking the sign of their sum: $\hat{y}^{(n)} = \text{sgn}(s_{\text{online}}^{(n)} + s_{\text{memory}}^{(n)})$; if the online and memory directions are inconsistent, it outputs 0, meaning no supervision is provided for these low-confidence samples. This ternary $+1/-1/0$ design explicitly excludes uncertain samples from optimization, which avoids noise-dominant training better than binary forced classification.

**2. Consistency-Aware Critique Reward (CACR): Process Supervision via Semantic Convergence of Critiques**

While CAAR focuses on preference results, it cannot constrain the quality of the critique text itself; CACR fills this gap in process supervision. It uses Qwen3-4B-Embedding to encode each generated critique $c_j$ into a vector, calculates a cosine similarity matrix, and ranks them by semantic consistency. Additional reward $r_j^{(c)} = 0.1$ is given to critiques that rank in the top $p$ and have correct preference labels. The intuition is that if multiple generated critiques are semantically highly consistent, the model's evaluation of the sample has converged to a stable region, making such critiques more likely to reflect reliable judgment.

Thus, CAAR supervises the result while CACR supervises the process. The two complement each other at different granularities, ensuring the reward system targets both "correctness" and "stability."

**3. Format Constraint & Combined Reward: Ensuring Parsability and Unified Optimization**

To ensure GRM outputs can be parsed and that multi-layer rewards align in direction, ConsistRM combines the answer and critique rewards into $r^{(n)} = r_j^{(a,n)} + r_j^{(c,n)}$, which is active only when the format is valid and $\hat{y} \neq 0$. If the format is invalid, a heavy penalty of $r = -5$ is applied; if $\hat{y} = 0$ (uncertain sample), $r = 0$. Optimization is performed using the GRPO algorithm with a global batch size of 64, learning rate of 1e-6, 8 rollouts, and a KL coefficient of 0.001.

Format constraints ensure outputs are parsable by downstream tasks, and the aggregation of consistency rewards at different granularities allows CAAR and CACR to provide consistent gradient signals for both results and processes, preventing exploitation of a single reward signal.

### Loss & Training
Training utilizes GRPO for 4 epochs, with a maximum generation length of 1024 (training) / 2048 (inference) and temperature of 1.0 (training) / 0 (inference). The overall process starts with SFT on HelpSteer3 followed by RFT (Reinforcement Fine-Tuning); ConsistRM’s dual consistency rewards replace the original reward signals during the RFT stage.

## Key Experimental Results

### Main Results

**Performance on Five Benchmarks using Qwen3-8B**

| Method | RewardBench | PPE Pref | RM-Bench | RMB | JudgeBench | Average | Δ |
|------|------------|----------|----------|-----|------------|------|---|
| Qwen3-8B (Base) | 81.6 | 63.8 | 75.8 | 78.8 | 54.3 | 70.9 | - |
| + SFT | 82.7 | 65.0 | 77.1 | 76.9 | 51.7 | 70.7 | -0.2 |
| + RFT | 85.4 | 65.4 | 78.2 | 78.2 | 55.4 | 72.5 | +1.6 |
| + TTRL | 85.3 | 65.0 | 77.4 | 74.2 | 56.8 | 71.7 | +0.8 |
| + ConsistRM | **85.6** | **67.7** | **78.3** | **79.1** | **56.9** | **73.5** | **+2.6** |

### Ablation Study

| Configuration | RewardBench | PPE | RM-Bench | RMB | JudgeBench | Average | Δ |
|------|------------|-----|----------|-----|------------|------|---|
| ConsistRM | 85.6 | 67.7 | 78.3 | 79.1 | 56.9 | 73.5 | - |
| w/o CACR | 84.9 | 64.8 | 77.3 | 78.1 | 56.0 | 72.2 | -1.3 |
| w/o Online-State | 85.5 | 64.1 | 78.6 | 76.7 | 56.7 | 72.3 | -1.2 |
| w/o Memory-Driven | 84.3 | 63.1 | 75.4 | 74.2 | 54.8 | 70.4 | **-3.2** |

### Key Findings

- Memory-Driven consistency preference is the most critical component (dropping 3.2 points when removed), indicating that historical information is essential for pseudo-label quality.
- ConsistRM achieves significant improvements in mitigating position bias (+5.3 vs. +1.4 for RFT), as consistency rewards encourage the model to focus on content rather than position.
- ConsistRM enables a 4B model to reach the performance level of an 8B model through multi-round voting.
- Replacing CACR with token-level confidence (DeepConfidence) leads to reward hacking and performance degradation.

## Highlights & Insights

- The core assumption that "consistency implies reliability" is simple yet powerful—utilizing internal consistency signals as a substitute for external annotation.
- The temporal dimension memory mechanism (aggregating historical pseudo-labels) is a key innovation, providing a stable anchor across training rounds.
- The ternary label design (+1/-1/0) elegantly handles uncertain samples and avoids polluting training with noisy labels.
- Generation efficiency is also improved—ConsistRM generates more concise critiques (1717 vs. 1924 tokens).

## Limitations & Future Work

- Semantic consistency evaluation is currently at the overall critique level and fails to align various semantic segments of the critique at a fine-grained level.
- Validation was limited to Qwen3 and LLaMA-3.1; generalization across more model families remains to be verified.
- Hyperparameters for consistency rewards (top-p ratio, CACR reward value of 0.1) might require task-specific tuning.

## Related Work & Insights

- **vs TTRL**: TTRL uses majority voting for pseudo-labels but lacks temporal consistency across rounds, leading to bias accumulation; ConsistRM introduces historical memory to significantly alleviate this.
- **vs DeepSeek-GRM**: The latter requires human-annotated reward signals, while ConsistRM is entirely label-free.
- **Insight**: Consistency signals may serve as a general, low-cost quality indicator for self-training scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The consistency-aware self-training paradigm is cleverly designed, particularly the temporal consistency memory mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage with five benchmarks, four models, full ablation, position bias, and multi-round voting analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear method description and detailed experimental analysis.
- Value: ⭐⭐⭐⭐ Provides a practical and effective solution for label-free GRM training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GRAM-R²: Self-Training Generative Foundation Reward Models for Reward Reasoning](../../AAAI2026/llm_alignment/gram-r2_self-training_generative_foundation_reward_models_for_reward_reasoning.md)
- [\[ICML 2026\] Consistency Training Can Entrench Misalignment](../../ICML2026/llm_alignment/consistency_training_can_entrench_misalignment.md)
- [\[ACL 2026\] Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO](mitigating_selection_bias_in_large_language_models_via_permutation-aware_grpo.md)
- [\[ACL 2026\] Debiasing Reward Models via Causally Motivated Inference-Time Intervention](debiasing_reward_models_via_causally_motivated_inference-time_intervention.md)
- [\[ACL 2026\] Team-Based Self-Play With Dual Adaptive Weighting for Fine-Tuning LLMs](team-based_self-play_with_dual_adaptive_weighting_for_fine-tuning_llms.md)

</div>

<!-- RELATED:END -->
