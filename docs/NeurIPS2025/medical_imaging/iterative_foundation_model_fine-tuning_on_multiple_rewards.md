---
title: >-
  [Paper Note] Iterative Foundation Model Fine-Tuning on Multiple Rewards
description: >-
  [NeurIPS 2025][Medical Imaging][Multi-objective fine-tuning] This paper proposes IterativeRS (Iterative Rewarded Soups), which alternates between independent fine-tuning of per-objective expert policies and policy merging. The method unifies reward combination and expert merging approaches, outperforming MORLHF and Rewarded Soups on small molecule design, DNA sequence generation, and text summarization tasks.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - Multi-objective fine-tuning
  - reinforcement learning
  - foundation models
  - iterative merging
  - Rewarded Soups
date: 2026-05-08
content_hash: acbc590b88e0fb5b
---

# Iterative Foundation Model Fine-Tuning on Multiple Rewards

**Conference**: NeurIPS 2025
**arXiv**: [2511.00220](https://arxiv.org/abs/2511.00220)
**Code**: [GitHub](https://github.com/pouyamghari/IterativeRS)
**Area**: Medical Imaging / Multi-Objective Reinforcement Learning Fine-Tuning
**Keywords**: Multi-objective fine-tuning, reinforcement learning, foundation models, iterative merging, Rewarded Soups

## TL;DR

This paper proposes IterativeRS (Iterative Rewarded Soups), which alternates between independent fine-tuning of per-objective expert policies and policy merging. The method unifies reward combination and expert merging approaches, outperforming MORLHF and Rewarded Soups on small molecule design, DNA sequence generation, and text summarization tasks.

## Background & Motivation

**Background**: RLHF fine-tuning of foundation models has become a standard pipeline, yet real-world applications typically involve multiple evaluation objectives (e.g., toxicity vs. helpfulness in text generation, or multiple molecular properties in drug design).

**Limitations of Prior Work**:
   - **MORLHF** (reward combination): Combines multiple rewards into a single scalar, failing to learn objective-specific skills and exhibiting high performance variance across objectives.
   - **Rewarded Soups** (expert merging): Trains independent experts per objective and then linearly combines them; performance degrades when expert policies diverge significantly.

**Key Challenge**: A fundamental tension exists between objective-specific learning (which requires policy divergence) and policy consistency (which requires policy convergence).

**Goal**: To design a flexible multi-objective fine-tuning algorithm that preserves objective-specific learning capacity while controlling divergence among expert policies.

**Key Insight**: Treating independent fine-tuning and policy merging as two adjustable extremes, and smoothly interpolating between them by controlling the merging frequency $m$.

**Core Idea**: Every $m$ steps, the expert policies of all objectives are merged, and training resumes from the merged checkpoint — MORLHF ($m=1$) and Rewarded Soups ($m=T$) are two limiting special cases.

## Method

### Overall Architecture

The IterativeRS training pipeline proceeds as follows:
1. Initialize all expert policies from the pretrained reference policy $\pi_{\text{ref}}$.
2. Each expert corresponds to one objective $R_i$ and performs $m$ independent gradient update steps.
3. Every $m$ steps, all expert policy parameters are linearly merged into shared parameters $\bm{\rho}_t$.
4. All experts are re-initialized from $\bm{\rho}_t$ and training continues for the next round.
5. The final merged policy is returned as output.

### Key Designs

1. **Iterative Update Rule**:
    $\bm{\theta}_{i,t+1} = \begin{cases} \bm{\theta}_{i,t} - \eta \nabla_{\bm{\theta}} \mathcal{L}_i(\pi_{\bm{\theta}_{i,t}}), & \text{if } t \bmod m \neq 0 \\ \bm{\rho}_t - \eta \nabla_{\bm{\rho}} \mathcal{L}_i(\pi_{\bm{\rho}_t}), & \text{if } t \bmod m = 0 \end{cases}$
   At merging steps, a random subset of $M \leq N$ objectives is sampled to reduce computational cost.

2. **Policy Merging**:
    $\bm{\rho}_t = \sum_{i \in \mathbb{S}_t} \lambda_{i,t} \bm{\theta}_{i,t}$
   Weights can be optimized via Monte Carlo search, or simply set to $\lambda_{i,t} = w_i / \sum_{j \in \mathbb{S}_t} w_j$.

3. **Unified Perspective via Hyperparameter $m$**:
    - $m = 1$: Merging at every step → equivalent to MORLHF (reward combination).
    - $m = T$: Merging only at the end of training → equivalent to Rewarded Soups.
    - $1 < m < T$: Smooth interpolation between the two extremes.

### Theoretical Analysis

**Theorem 1** provides a convergence bound for IterativeRS:

$$\mathcal{L}(\pi_{\bm{\rho}_T}) - \mathcal{L}(\pi_{\bm{\theta}^*}) \leq \frac{4L}{\mu^2(\gamma+T)}\left(3L\Delta^* + 2(2(m-1)^2 + \frac{N-M}{N-1}\frac{m^2}{M})G^2\right) + \frac{\gamma L}{2(\gamma+T)}\|\bm{\theta}_{\text{ref}} - \bm{\theta}^*\|^2$$

Key insights:
- $\Delta^*$ measures the dissimilarity among per-objective optimal policies — more similar objectives yield better convergence.
- Larger $m$ increases the $A_2$ term (expert divergence) but may reduce the $A_1$ term, implying the existence of an optimal $m$.
- A stronger reference policy (smaller $\|\bm{\theta}_{\text{ref}} - \bm{\theta}^*\|$) facilitates convergence.

## Key Experimental Results

### Small Molecule Generation

| Method | α Energy | gap | U₀ Energy | Avg Reward | ICV |
|--------|----------|-----|-----------|-----------|-----|
| MORLHF | 1.4229 | 0.9355 | 1.5146 | 1.2910 | 4.19 |
| RS | 1.4134 | 0.9589 | 1.5464 | 1.3062 | 4.27 |
| RiC | 0.5955 | 0.6795 | 0.7544 | 0.6765 | 3.75 |
| **IterativeRS** | **1.5893** | 0.9508 | **1.6649** | **1.4017** | 3.59 |

IterativeRS achieves the highest average reward across all methods, with the best molecule quality on the Pareto frontier.

### DNA Sequence Generation

| Method | K562 | HepG2 | SKNSH | Avg Reward | ICV |
|--------|------|-------|-------|-----------|-----|
| MORLHF | 0.2724 | 0.7096 | 0.7183 | 0.5667 | 3.14 |
| RS | 0.3057 | 0.6808 | 0.7131 | 0.5666 | 3.82 |
| RiC | 0.4221 | 0.6615 | 0.6688 | 0.5842 | 2.47 |
| **IterativeRS** | 0.3032 | **0.7370** | **0.7378** | **0.5927** | **3.83** |

IterativeRS achieves 35% higher ICV than RiC, demonstrating substantially better cross-objective consistency.

### Text Summarization

| Method | faithful | summary | deberta | Avg Score | ICV |
|--------|----------|---------|---------|-----------|-----|
| MORLHF | 0.6530 | 0.5778 | 0.3857 | 0.4525 | 4.55 |
| RS | 0.6732 | 0.5807 | 0.4296 | 0.4732 | 4.59 |
| RiC | 0.6497 | 0.5688 | 0.3455 | 0.4518 | 3.96 |
| **IterativeRS** | **0.6927** | **0.5854** | **0.4398** | **0.4849** | **4.91** |

### Ablation Study

The effect of $m$ on performance reveals a consistent pattern:
- Both $m=1$ (MORLHF) and $m=T$ (RS) are suboptimal.
- Intermediate values of $m$ typically perform best (small molecules: $m=4$; DNA: $m=8$; text: $m=40$).
- The optimal $m$ depends on the degree of conflict among objectives and data characteristics.

### Key Findings

- RL-based methods (IterativeRS/RS/MORLHF) significantly outperform the SFT-based method (RiC) on molecule generation, as RL can explore high-reward regions beyond the training data distribution.
- When the pretraining and fine-tuning data distributions are close (e.g., the DNA task), the performance gap of SFT-based methods narrows.
- IterativeRS produces fewer extremely low-reward samples, yielding a more concentrated reward distribution.

## Highlights & Insights

- **Unified Framework**: A single hyperparameter $m$ unifies MORLHF and Rewarded Soups as special cases — a conceptually elegant formulation.
- **Theoretical Guarantees**: Convergence analysis is provided under convex assumptions; while the loss is non-convex in practice, the theory offers useful intuition.
- **Cross-Domain Validation**: Consistent advantages are demonstrated across three substantially different domains: molecular design, DNA sequences, and text.
- **Practical Simplicity**: The method requires only minor modifications to the training loop (a few lines of code) with no changes to model architecture or training pipeline.

## Limitations & Future Work

- The theoretical analysis assumes convex loss functions and bounded gradients, which are strong assumptions that deviate significantly from practical deep learning settings.
- Only linear interpolation is considered for policy merging; more advanced merging strategies (e.g., Task Arithmetic, TIES) may yield further improvements.
- The optimal value of $m$ requires empirical tuning; the theoretical bound does not provide sufficient guidance for direct determination.
- IterativeRS does not achieve the best ICV in all settings (e.g., ICV is lower than RS on the small molecule task).
- Computational feasibility and performance when scaling to larger numbers of objectives ($N \gg 3$) remain to be verified.

## Related Work & Insights

- Rewarded Soups (Rame et al. 2023) is the most direct predecessor; IterativeRS is its natural generalization.
- The approach bears a close resemblance to FedAvg in federated learning — local updates for multiple steps followed by aggregation.
- The work offers practical guidance for multi-objective RLHF: practitioners need not choose between reward combination and training multiple experts.
- A promising extension is to adaptively adjust $m$ (e.g., dynamically based on the divergence among expert policies).

## Rating

- Novelty: ⭐⭐⭐⭐ The unified perspective is novel, though the core idea of alternating between independent training and merging is relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three domains, though the task setups within each domain are relatively straightforward.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with good integration of theory and experiments; some notation could be simplified.
- Value: ⭐⭐⭐⭐ Addresses a practical pain point in multi-objective fine-tuning with a simple and deployable method.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] NeurIPT: Foundation Model for Neural Interfaces](neuript_foundation_model_for_neural_interfaces.md)
- [\[NeurIPS 2025\] Flow Density Control: Generative Optimization Beyond Entropy-Regularized Fine-Tuning](flow_density_control_generative_optimization_beyond_entropy-regularized_fine-tun.md)
- [\[NeurIPS 2025\] JanusDNA: A Powerful Bi-directional Hybrid DNA Foundation Model](janusdna_a_powerful_bi-directional_hybrid_dna_foundation_model.md)
- [\[ICLR 2026\] Thompson Sampling via Fine-Tuning of LLMs](../../ICLR2026/medical_imaging/thompson_sampling_via_fine-tuning_of_llms.md)
- [\[NeurIPS 2025\] Brain Harmony: A Multimodal Foundation Model Unifying Morphology and Function into 1D Tokens](brain_harmony_a_multimodal_foundation_model_unifying_morphology_and_function_int.md)

<!-- RELATED:END -->
