---
title: >-
  [Paper Note] Addressing Mark Imbalance in Integration-free Neural Marked Temporal Point Processes
description: >-
  [NeurIPS 2025][marked temporal point processes] This paper is the first to systematically reveal the severe impact of mark distribution imbalance on prediction performance in marked temporal point processes (MTPP). It pr…
tags:
  - "NeurIPS 2025"
  - "marked temporal point processes"
  - "mark imbalance"
  - "thresholding method"
  - "integration-free approximation"
  - "event prediction"
date: 2026-05-08
content_hash: 7c4d5bd306730e51
---

# Addressing Mark Imbalance in Integration-free Neural Marked Temporal Point Processes

**Conference**: NeurIPS 2025
**arXiv**: [2510.20414](https://arxiv.org/abs/2510.20414)
**Code**: [GitHub](https://github.com/undes1red/IFNMTPP)
**Area**: Others
**Keywords**: marked temporal point processes, mark imbalance, thresholding method, integration-free approximation, event prediction

## TL;DR

This paper is the first to systematically reveal the severe impact of mark distribution imbalance on prediction performance in marked temporal point processes (MTPP). It proposes a mark-first-then-time prediction strategy, designs a thresholding method to calibrate the predicted probabilities of rare marks, and develops the integration-free IFNMTPP model to efficiently support mark probability estimation and time sampling.

## Background & Motivation

### State of the Field

**Background**: Marked temporal point processes (MTPP) model the type (mark) and occurrence time of each event in an event stream, with broad applications in earthquake prediction, social media retweet modeling, and related domains. Existing MTPP models have overlooked a critical issue: **mark distributions are highly imbalanced**. For example, magnitude-7 earthquakes (rare but important) are far less frequent than magnitude-3 earthquakes (common).

Problems caused by imbalance:

### Limitations of Prior Work

**Limitations of Prior Work**: The conditional probability $p^*(m,t)$ of frequent marks is far higher than that of rare marks at most time points.

### Root Cause

**Key Challenge**: Models almost always predict frequent marks, resulting in extremely low macro-F1 for rare marks (e.g., on the Retweet dataset, rare marks achieve only 0.027 vs. 0.618 for frequent marks).

Existing methods typically follow a "predict time $t$ first, then predict mark $p^*(m|t)$ at time $t$" paradigm, which makes thresholding methods difficult to apply—mark probabilities vary over time, preventing a unified threshold from being learned across all time points.

## Method

### Overall Architecture

1. **Reversed prediction order**: Predict mark $p^*(m)$ first, then predict time $p^*(t|m)$ conditioned on the mark.
2. **Thresholding method**: Apply prior probability normalization to $p^*(m)$ and learn thresholds to improve rare mark prediction.
3. **IFNMTPP model**: Compute $p^*(m)$ and $F^*(t|m)$ without numerical integration.

### Key Designs

1. **Mark-first prediction + thresholding method**:
    - Function: Predict mark based on $p^*(m) = \int_{t_l}^{+\infty} p^*(m,\tau)\,d\tau$, then predict time.
    - Mechanism: Compute the ratio $r_m = p^*(m) / \bar{p}^*(m)$ (probability / prior probability), and learn a threshold $\epsilon_m$ such that $m_p = \arg\max_m (r_m - \epsilon_m)$.
    - Design Motivation: $p^*(m)$ is time-independent, enabling a unified threshold to handle imbalance. Even when $p^*(m)$ is low for a rare mark, $r_m$ can still be high, indicating that the event is more likely to occur relative to its own baseline probability.
    - Threshold learning: For each mark $m$, the optimal $\epsilon_m$ is determined by maximizing the F1 score of $m$ vs. all other marks.

2. **Integration-free approximation (IFNMTPP)**:
    - Function: Avoid computing two costly improper integrals to obtain $p^*(m)$ and $F^*(t|m)$.
    - Mechanism: Define $\Gamma^*(m,t) = \int_t^{+\infty} p^*(m,\tau)\,d\tau$, unifying both integrals under a single modeling target $\Gamma^*$; directly parameterize $\Gamma^*(m,t)$ via a neural network instead of $\lambda^*(m,t)$.
    - Design Motivation: $p^*(m) = \Gamma^*(m,t_l)$ and $F^*(t|m) = 1 - \Gamma^*(m,t)/\Gamma^*(m,t_l)$, unifying the computation of mark probabilities and the time CDF.
    - Key constraint: $\Gamma^*(m,t)$ must be monotonically decreasing in $t$ and converge to 0, enforced through the network architecture design.

### Loss & Training

- Negative log-likelihood loss: $\mathcal{L} = -\sum_{i} \log p^*(m_i, t_i)$
- $p^*(m,t)$ is computed directly through the IFNMTPP parameterization without numerical integration.
- Time prediction uses inverse transform sampling (ITS) for efficient sampling from $F^*(t|m)$.
- Thresholds $\epsilon_m$ are optimized separately on the training set and do not participate in gradient backpropagation.

## Key Experimental Results

### Main Results

| Method | Retweet (macro-F1) | USearthquake (macro-F1) | StackOverflow (macro-F1) |
|------|-------------------|------------------------|-------------------------|
| SAHP | 0.236 | 0.045 | 0.141 |
| THP | 0.242 | 0.044 | 0.148 |
| IFNMTPP (ours, w/o threshold) | 0.293 | 0.056 | 0.155 |
| **IFNMTPP + threshold** | **0.368** | **0.103** | **0.213** |

The thresholding method substantially improves macro-F1 across all datasets, with particularly large gains for rare marks.

### Ablation Study

- The thresholding method generalizes to different base MTPP models (SAHP, THP, etc.) and consistently yields improvements.
- Mark-first vs. time-first prediction: the former cooperates far more effectively with the thresholding method.
- The integration-free approximation of IFNMTPP achieves accuracy comparable to numerical integration at several times the speed.
- Time prediction accuracy improves with larger sample count $N$; $N=100$ achieves a good trade-off.

### Key Findings

- Mark imbalance is pervasive in real-world datasets and severely degrades prediction performance.
- Existing MTPP models yield near-zero macro-F1 on rare marks.
- The improvement from thresholding primarily stems from substantial gains on rare marks, with negligible impact on frequent marks.
- The prediction order (mark-first vs. time-first) is decisive for the feasibility of imbalance-handling methods.

## Highlights & Insights

- **Important problem identification**: The first work to systematically expose the severity of imbalance in MTPP, filling a significant gap in the literature.
- **Elegant methodology**: Reversing the prediction order makes thresholding naturally applicable; unified integration simplifies model design.
- **Strong practicality**: The thresholding method can be applied as post-processing to any MTPP model.
- The unified modeling of $\Gamma^*(m,t)$ is the core technical contribution, simultaneously addressing mark probability estimation and time sampling.

## Limitations & Future Work

- Only categorical marks are addressed; the approach is not extended to continuous mark spaces.
- The thresholding method assumes consistent prior probabilities between training and test sets, and may fail under distribution shift.
- The expressive capacity of IFNMTPP is constrained by the monotonicity requirement imposed on $\Gamma^*$.
- Combinations of the thresholding method with other imbalance-handling approaches (e.g., oversampling/undersampling) remain unexplored.

## Related Work & Insights

- Inspired by imbalance-handling methods in classification tasks (resampling, cost-sensitive learning, threshold adjustment).
- Distinguished from traditional Hawkes processes and Neural TPPs by its explicit focus on mark imbalance.
- The integration-free design of IFNMTPP is generalizable to other probabilistic models where numerical integration is undesirable.

## Rating

⭐⭐⭐⭐ — The problem is important and overlooked; the methodology is elegantly designed (unified integration + reversed prediction order); experimental improvements are substantial.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Deep Continuous-Time State-Space Models for Marked Event Sequences](deep_continuous-time_state-space_models_for_marked_event_sequences.md)
- [\[ICLR 2026\] Addressing Divergent Representations from Causal Interventions on Neural Networks](../../ICLR2026/others/addressing_divergent_representations_causal.md)
- [\[NeurIPS 2025\] Position: There Is No Free Bayesian Uncertainty Quantification](position_there_is_no_free_bayesian_uncertainty_quantification.md)
- [\[NeurIPS 2025\] Alias-Free ViT: Fractional Shift Invariance via Linear Attention](alias-free_vit_fractional_shift_invariance_via_linear_attention.md)
- [\[NeurIPS 2025\] Weight Weaving: Parameter Pooling for Data-Free Model Merging](weight_weaving_parameter_pooling_for_data-free_model_merging.md)

</div>

<!-- RELATED:END -->
