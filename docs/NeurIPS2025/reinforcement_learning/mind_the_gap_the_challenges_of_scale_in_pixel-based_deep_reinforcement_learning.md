---
title: >-
  [Paper Note] Mind the GAP! The Challenges of Scale in Pixel-based Deep Reinforcement Learning
description: >-
  [NeurIPS 2025][Reinforcement Learning][Deep Reinforcement Learning] This paper identifies the "bottleneck connection" between the encoder (convolutional layers $\phi$) and the fully connected layers ($\psi$) as the fundamental obstacle to scaling pixel-based deep RL networks, and proposes Global Average Pooling (GAP) — a minimal architectural change — to directly resolve this bottleneck. GAP achieves performance on par with or superior to complex methods (SoftMoE, sparse training) at substantially lower computational cost.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Deep Reinforcement Learning
  - Network Scaling
  - Global Average Pooling
  - Bottleneck Layer
  - Atari
date: 2026-05-08
content_hash: 794491274c87d07c
---

# Mind the GAP! The Challenges of Scale in Pixel-based Deep Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.17749](https://arxiv.org/abs/2505.17749)
**Code**: [Dopamine](https://github.com/google/dopamine)
**Area**: Reinforcement Learning
**Keywords**: Deep Reinforcement Learning, Network Scaling, Global Average Pooling, Bottleneck Layer, Atari

## TL;DR

This paper identifies the "bottleneck connection" between the encoder (convolutional layers $\phi$) and the fully connected layers ($\psi$) as the fundamental obstacle to scaling pixel-based deep RL networks, and proposes Global Average Pooling (GAP) — a minimal architectural change — to directly resolve this bottleneck. GAP achieves performance on par with or superior to complex methods (SoftMoE, sparse training) at substantially lower computational cost.

## Background & Motivation

Deep RL exhibits a paradoxical "anti-scaling" phenomenon, opposite to that observed in supervised learning: **naively enlarging the network often degrades rather than improves performance**. Recent works have attempted to address this through complex architectural modifications — Mixture of Experts (SoftMoE), network pruning, tokenization, regularization, and others. While effective, these approaches are implementation-heavy and computationally expensive.

More critically, **the root cause of this performance degradation has remained unclear**, with different methods offering different explanations and no unified understanding.

The paper's core insight is that the problem lies in the **bottleneck**: the encoder $\phi$ (convolutional layers) produces a three-dimensional tensor of shape $H \times W \times C$, which is conventionally flattened into a one-dimensional vector before being passed to the fully connected layers $\psi$. The number of parameters in this flattened connection is $H \times W \times C \times \dim(\psi)$. Scaling the width of $\psi$ causes this parameter count to grow multiplicatively, leading to:
- A large proportion of dormant neurons (plasticity loss)
- Difficulty in effectively combining encoder features
- Diffused attention toward task-irrelevant regions

Crucially, the paper argues that prior methods (SoftMoE, sparse training) succeed **not because of their specific architectural innovations, but because they all implicitly reduce the effective parameter density at the bottleneck**. This insight motivates a simpler solution — GAP — that achieves the same effect directly.

## Method

### Overall Architecture

The pixel-based RL network takes the form $Q(x, \cdot) = \psi(\phi(x))$:
- Encoder $\phi$: a series of convolutional layers (typically an Impala ResNet), producing feature maps $F \in \mathbb{R}^{H \times W \times C}$
- Fully connected layers $\psi$: typically 1–2 dense layers
- **Bottleneck**: the connection from $\phi$ to $\psi$, standardly implemented via flattening, with $H \times W \times C \times \dim(\psi)$ parameters

### Key Designs

1. **Diagnostic Analysis — The Bottleneck as the Root Cause of Performance Degradation**:

    - **Dormant neuron analysis**: After scaling the network, the proportion of dormant neurons in $\psi$ is substantially higher than in $\phi$ (Figure 2, left), indicating that scaling primarily impairs plasticity at the bottleneck.
    - **Scaling the bottleneck alone vs. scaling the full network**: Both yield comparable performance degradation (Figure 2, right), demonstrating that the bottleneck drives the majority of performance decline.
    - **Grad-CAM analysis**: Naively scaled networks fail to attend to task-relevant regions of the input, with attention diffused across irrelevant backgrounds.
    - **"Curative" effect of deeper encoders**: Providing $\psi$ with more abstract, high-level features (by deepening $\phi$) substantially recovers performance — indicating that structured representations facilitate feature learning in scaled networks.

2. **Existing Methods Implicitly Address the Bottleneck**:

    - **SoftMoE-1**: Reorganizes the output of $\phi$ into $H \times W$ tokens of dimension $C$, reducing bottleneck parameters to $C \times \dim(\psi)$.
    - **Sparse training**: Applies masks to reduce the effective bottleneck parameter count to $s \times H \times W \times C \times \dim(\psi)$.
    - Validation: Applying sparsity only at the bottleneck (while keeping all other layers dense) improves performance, matching or surpassing full-network sparsification.

3. **Global Average Pooling (GAP) — The Minimal Solution**: GAP averages each feature map of the encoder output over the spatial dimensions:

$$g^c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} F_{ij}^c$$

This produces $\mathbf{g} \in \mathbb{R}^C$, which is then passed to $\psi$. The bottleneck parameter count is reduced from $H \times W \times C \times \dim(\psi)$ to $C \times \dim(\psi)$. The design motivation is straightforward: since a low-density, structured bottleneck is the key, GAP is the most concise way to achieve it.

### Loss & Training

GAP is a purely architectural modification that requires no changes to the training algorithm. Main experiments use a Rainbow agent with the Impala architecture, trained for 200M environment steps averaged over 55 random seeds. GAP is effective across different scaling factors (×1 to ×8) and varying replay ratios.

## Key Experimental Results

### Main Results

IQM (Interquartile Mean) comparison on 20 Atari games at ×4 scale, 100M steps:

| Method | IQM↑ | Median↑ | Mean↑ | GPU-hours/game |
|--------|------|---------|-------|----------------|
| Baseline (×4) | ~0.8 | ~0.8 | ~1.0 | ~160 |
| Gradual Pruning | ~1.1 | ~1.0 | ~1.2 | ~200 |
| RigL (dynamic sparse) | ~1.0 | ~0.9 | ~1.1 | ~190 |
| SoftMoE-1 | ~1.3 | ~1.2 | ~1.4 | ~170 |
| **GAP** | **~1.3** | **~1.3** | **~1.5** | **~140** |

GAP matches or slightly surpasses SoftMoE-1 in performance while incurring **substantially lower computational cost**, as it avoids the token construction and post-projection computation required by SoftMoE.

### Ablation Study

| Configuration | Key Finding | Notes |
|---------------|-------------|-------|
| Varying scale (×1–×8) | Baseline degrades with scale; GAP improves monotonically | GAP unlocks scaling behavior |
| High replay ratio (0.5–2.0) | GAP maintains strong performance at high replay ratios | Effective for sample-efficient training |
| CNN architecture (Mnih 2015) | GAP yields consistent gains | Not restricted to Impala architecture |
| Full 60-game Atari suite | GAP improves performance consistently | Results are not cherry-picked |
| Simultaneous depth + width scaling | Baseline degrades severely; GAP remains strong | Only method enabling joint depth–width scaling |

### Cross-Domain Validation

| Domain | Agent | Effect of GAP |
|--------|-------|---------------|
| Procgen (16 games) | Rainbow | Substantial performance gains with scaled networks |
| Atari 100K (data-efficient) | DER | Effective under sample-limited settings |
| DMC continuous control | SAC | Benefits extend to continuous action spaces |

### Key Findings

- GAP enables scaled RL networks to exhibit, for the first time, a "bigger is better" property analogous to supervised learning.
- Applying sparsity only at the bottleneck ≈ full-network sparsification, further confirming the bottleneck as the central issue.
- Networks trained with GAP exhibit fewer dormant neurons and lower feature norms, indicating that bottleneck structuring improves plasticity and training stability.
- Grad-CAM visualizations show that GAP networks correctly attend to task-relevant regions of the input.

## Highlights & Insights

- **A textbook case of Occam's Razor**: Faced with a complex network scaling problem addressed by intricate MoE and sparse training solutions, this paper identifies a concise root cause (the bottleneck) through careful diagnosis and resolves it with the simplest available technique (GAP, a single line of code) — yielding a method that is faster, simpler, and more effective than prior approaches.
- The diagnostic analysis is exceptionally well-structured: it first localizes the bottleneck (dormant neuron analysis + bottleneck isolation experiments), then explains *why* (unstructured features), then demonstrates that *existing methods all address the same issue* (a unified perspective), and finally derives the minimal solution.
- The findings suggest that much of the perceived complexity in RL network training may have simpler explanations — motivating a re-examination of other architectural choices in deep RL.

## Limitations & Future Work

- GAP is an aggressive form of compression — global averaging over the spatial dimensions may discard local spatial information, potentially limiting applicability to tasks requiring fine-grained spatial reasoning.
- The current work focuses on pixel-based environments with a clear $\phi$-$\psi$ separation; whether the findings generalize to non-pixel inputs or architectures without a well-defined bottleneck remains an open question.
- The combination of GAP with other representation learning methods (e.g., MICo, Proto-value networks) has not been explored.
- More flexible pooling strategies (e.g., attention pooling) could be investigated to achieve a better trade-off between structural regularity and information preservation.

## Related Work & Insights

- This work is complementary to Sokar et al. (2025), who find that SoftMoE's benefit stems from tokenization rather than expert routing; the present paper further reveals that tokenization's benefit is itself attributable to bottleneck structuring.
- Several concurrent works independently propose similar GAP-based solutions, providing additional corroborating evidence.
- Implication: other "taken-for-granted" architectural choices in RL (e.g., the flatten operation) may similarly warrant critical reexamination.

## Rating

- **Novelty**: ⭐⭐⭐⭐☆ — GAP itself is not new, but the diagnostic analysis and unified perspective constitute a significant contribution
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 55 seeds, 60 games, multiple domains and architectures, multiple scaling factors; experiments are exceptionally rigorous
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Narrative is fluid; the logical chain from diagnosis to solution is clear and complete
- **Value**: ⭐⭐⭐⭐⭐ — Provides a simple and effective best practice for scaling RL networks, with both engineering and theoretical merit

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Confounding Robust Deep Reinforcement Learning: A Causal Approach](confounding_robust_deep_reinforcement_learning_a_causal_approach.md)
- [\[NeurIPS 2025\] Solving Continuous Mean Field Games: Deep Reinforcement Learning for Non-Stationary Dynamics](solving_continuous_mean_field_games_deep_reinforcement_learning_for_non-stationa.md)
- [\[NeurIPS 2025\] Enhancing Interpretability in Deep Reinforcement Learning through Semantic Clustering](enhancing_interpretability_in_deep_reinforcement_learning_through_semantic_clust.md)
- [\[NeurIPS 2025\] Time Reversal Symmetry for Efficient Robotic Manipulations in Deep Reinforcement Learning](time_reversal_symmetry_for_efficient_robotic_manipulations_in_deep_reinforcement.md)
- [\[NeurIPS 2025\] Adaptive Cooperative Transmission Design for URLLC via Deep RL](adaptive_cooperative_transmission_design_for_ultra-reliable_low-latency_communic.md)

<!-- RELATED:END -->
