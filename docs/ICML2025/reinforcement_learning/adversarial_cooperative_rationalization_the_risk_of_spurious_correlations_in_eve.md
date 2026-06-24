---
title: >-
  [Paper Note] Adversarial Cooperative Rationalization: The Risk of Spurious Correlations in Even Clean Datasets
description: >-
  [ICML 2025][Reinforcement Learning][Self-explaining models] Reveals a hidden flaw in the cooperative rationalization framework (RNP)—even on clean datasets, the generator's sampling bias introduces spurious correlations between rationales and labels. An adversarial detection and instruction intervention method is proposed, significantly outperforming existing methods on text and graph classification.
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Self-explaining models"
  - "Rationalization"
  - "Spurious correlations"
  - "Adversarial attack"
  - "Sampling bias"
date: 2026-05-08
content_hash: fbe170d5af56291d
---

# Adversarial Cooperative Rationalization: The Risk of Spurious Correlations in Even Clean Datasets

**Conference**: ICML 2025  
**arXiv**: [2505.02118](https://arxiv.org/abs/2505.02118)  
**Code**: [https://github.com/jugechengzi/Rationalization-A2I](https://github.com/jugechengzi/Rationalization-A2I)  
**Area**: Reinforcement Learning/Explainability  
**Keywords**: Self-explaining models, Rationalization, Spurious correlations, Adversarial attack, Sampling bias

## TL;DR
Reveals a hidden flaw in the cooperative rationalization framework (RNP)—even on clean datasets, the generator's sampling bias introduces spurious correlations between rationales and labels. An adversarial detection and instruction intervention method is proposed, significantly outperforming existing methods on text and graph classification.

## Background & Motivation

### Background

**Background**: Cooperative rationalization (RNP) is a mainstream model-agnostic self-explaining framework—a generator selects the most informative subset of the input as the rationale, and a predictor makes predictions based on the rationale. The two are trained cooperatively to maximize accuracy.

**Limitations of Prior Work**: The authors discover a counter-intuitive phenomenon—even when removing the "maximizing accuracy" objective to let the generator select randomly, the predictor can still achieve high accuracy. This indicates that the predictor might exploit spurious correlations introduced by the rationale selection process.

**Key Challenge**: The sampling process of the generator alters the data distribution—$P(Y|Z)$ after sampling may not equal $P(Y|Z|g)$, even if $Y \perp Z$ in the original data.

**Goal**: To detect and eliminate spurious correlations introduced during the rationalization process.

**Key Insight**: Exposing spurious correlations via adversarial attacks $\rightarrow$ preventing the predictor from learning these spurious patterns using an instruction mechanism.

**Core Idea**: A two-stage approach of adversarial inspection + instruction intervention.

## Method

### Overall Architecture
1. Analysis: Prove how the generator's sampling bias introduces spurious correlations
2. Attack: Design adversarial methods to expose these correlations
3. Defense: Introduce an "instruction" mechanism to prevent the predictor from exploiting spurious correlations

### Key Designs

1. **Sampling Bias Analysis**:

    - **Function**: Theoretically analyze how generator sampling alters the data distribution
    - **Mechanism**: $Y \perp T$ in the original dataset does not imply $Y \perp T$ in the sampled $(Z,Y)$ pairs
    - **Design Motivation**: Explains why random rationales can still achieve high accuracy

2. **Adversarial Detection + Instruction Defense**:

    - **Function**: (a) Adversarial attacks identify which patterns are spurious correlations; (b) The instruction mechanism tells the predictor to ignore these patterns
    - **Mechanism**: Construct rationales that exploit spurious correlations using an adversarial generator $\rightarrow$ analyze which features are exploited $\rightarrow$ incorporate instructions in normal training to exclude these features
    - **Design Motivation**: First "know the enemy" and then "tell the model not to follow"

### Loss & Training
- Standard rationalization loss + instruction regularization term
- Applicable to GRU, BERT, and GCN

## Key Experimental Results

### Main Results
6 text + 2 graph classification datasets:

| Method | Text F1 (Avg.) | Graph F1 (Avg.) |
|------|-------------|------------|
| RNP (Original) | 72.3% | 68.5% |
| A2I (Ours) | **81.7%** | **78.2%** |
| LLaMA-3.1-8B | 79.5% | - |

### Key Findings
- Significantly outperforms existing rationalization methods on all datasets
- Even outperforms LLaMA-3.1-8B on some tasks
- Spurious correlations are prevalent across all rationalization methods

## Highlights & Insights
- The finding that **"even clean datasets have spurious correlations"** is counter-intuitive—the sampling process itself is the source of bias
- The two-stage framework of adversarial + instructions is simple and effective

## Limitations & Future Work
- Adversarial detection increases training complexity
- Instruction design requires domain knowledge

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reveals an overlooked fundamental problem
- Experimental Thoroughness: ⭐⭐⭐⭐ Text + graph, multiple architectures
- Writing Quality: ⭐⭐⭐⭐ In-depth analysis
- Value: ⭐⭐⭐⭐ Important caveat for explainable AI research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration](enhancing_cooperative_multi-agent_reinforcement_learning_with_state_modelling_an.md)
- [\[NeurIPS 2025\] Learning to Clean: Reinforcement Learning for Noisy Label Correction](../../NeurIPS2025/reinforcement_learning/learning_to_clean_reinforcement_learning_for_noisy_label_correction.md)
- [\[NeurIPS 2025\] Risk-Averse Total-Reward Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/risk-averse_total-reward_reinforcement_learning.md)
- [\[NeurIPS 2025\] Risk-Averse Constrained Reinforcement Learning with Optimized Certainty Equivalents](../../NeurIPS2025/reinforcement_learning/risk-averse_constrained_reinforcement_learning_with_optimized_certainty_equivale.md)
- [\[AAAI 2026\] Risk-Sensitive Exponential Actor Critic](../../AAAI2026/reinforcement_learning/risk-sensitive_exponential_actor_critic.md)

</div>

<!-- RELATED:END -->
