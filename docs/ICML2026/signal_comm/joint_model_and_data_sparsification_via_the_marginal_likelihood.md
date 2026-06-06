---
title: >-
  [Paper Note] Joint Model and Data Sparsification via the Marginal Likelihood
description: >-
  [ICML 2026][Signal & Communication][Joint Sparsification] JMDS achieves simultaneous model and data sparsification through a unified objective of **maximizing marginal likelihood**—avoiding the sub-optimality of phased o…
tags:
  - "ICML 2026"
  - "Signal & Communication"
  - "Joint Sparsification"
  - "Marginal Likelihood"
  - "Laplace Approximation"
  - "Neural Tangent Kernel"
date: 2026-05-08
content_hash: 85ef8f52a4b30cab
---

# Joint Model and Data Sparsification via the Marginal Likelihood

**Conference**: ICML 2026  
**arXiv**: [2605.29107](https://arxiv.org/abs/2605.29107)  
**Code**: To be confirmed  
**Area**: Model Compression / Data Sparsification / Bayesian Learning  
**Keywords**: Joint Sparsification, Marginal Likelihood, Laplace Approximation, Neural Tangent Kernel

## TL;DR
JMDS achieves simultaneous model and data sparsification through a unified objective of **maximizing marginal likelihood**—avoiding the sub-optimality of phased optimization—while maintaining performance superior to independent sparsification at 5-10× joint compression ratios on CIFAR / ImageNet / WikiText.

## Background & Motivation

**Background**: Neural network sparsification has been extensively studied, but model pruning (removing weights) and data sparsification (removing training samples) are typically **handled independently**—phased approaches ignore the coupling between the two.

**Limitations of Prior Work**: (1) The "Training -> Model Sparsification -> Data Sparsification" pipeline is prone to falling into local optima; (2) most existing joint methods are based on heuristics and lack rigorous theory; (3) a critical question in joint sparsification for large models—**the relationship between model sparsity and data sparsity**—has not been answered in a principled manner.

**Key Challenge**: Model and data should be **optimized simultaneously** to maximize joint compression benefits, yet a unified objective function is lacking.

**Goal**: Propose a principled joint sparsification framework, theoretically analyze its complexity, and verify its practical effectiveness.

**Key Insight**: In a Bayesian framework, the **marginal likelihood** naturally integrates model complexity (e.g., prior volume) with data likelihood—this serves as a natural metric for evaluating the quality of a model + data combination.

**Core Idea**: Incorporate model sparsity (weight binary mask $\mathbf{m}$) and data sparsity (sample binary weights $\mathbf{s}$) simultaneously into the marginal likelihood objective $\log p(\mathcal{D}_s | \mathbf{m}, \mathbf{s}) = \int p(\mathcal{D}_s | \theta, \mathbf{m}) p(\theta) d\theta$, rendered tractable via Laplace approximation.

## Method

### Overall Architecture
(1) **Joint Parameterization**: Model $\theta$ + model mask $\mathbf{m}$ + data weights $\mathbf{s}$; (2) **Objective**: Jointly maximize marginal likelihood $\log p(\mathcal{D}^{(s)} | \mathbf{m}) - \lambda_1 \|\mathbf{m}\|_0 - \lambda_2 \|\mathbf{s}\|_0$; (3) **Optimization**: Simplify the marginal likelihood using Laplace approximation; (4) **Algorithm**: Alternately maximize $\theta, \mathbf{m}, \mathbf{s}$.

### Key Designs

1. **Unified Marginal Likelihood Objective**:

    - Function: A unified objective for joint optimization of model and data sparsity.
    - Mechanism: Maximize $\log p(\mathcal{D}^{(s)} | \mathbf{m}) - \lambda_1 \|\mathbf{m}\|_0 - \lambda_2 \|\mathbf{s}\|_0$, where $\mathcal{D}^{(s)} = \{(\mathbf{x}_i, y_i, s_i)\}$ is the dataset with sample weights and $\mathbf{m}$ is the model weight mask. The marginal likelihood naturally penalizes redundant weights via Occam's Razor.
    - Design Motivation: Compared to phased methods, the unified objective ensures joint optimality of $(\mathbf{m}, \mathbf{s})$; it is more theoretically grounded than heuristic joint approaches.

2. **Laplace Approximation + Hessian Approximation**:

    - Function: Transforms the intractable marginal likelihood integral into a computable analytical form.
    - Mechanism: Apply Laplace approximation to $p(\mathcal{D}^{(s)} | \theta) p(\theta)$ at $\theta^* = \arg\max$ to obtain $\log p(\mathcal{D}^{(s)}) \approx -\mathcal{L}(\theta^*) + \frac{1}{2} \log \det H^{-1}$; use K-FAC block-diagonal approximation for the Hessian $H \approx H_{\text{kfac}}$ to reduce complexity from $O(N + d^2)$ to $O(N + d \cdot l)$; further accelerate via NTK approximation + subsampling.
    - Design Motivation: Exact marginal likelihood requires $O(d^3)$ Hessian decomposition, which is infeasible for large models; K-FAC + NTK approximation preserves accuracy while achieving scalability.

3. **Three-Phase Alternating Optimization Algorithm**:

    - Function: Converges to joint optimality through iterative maximization of $\theta, \mathbf{m}, \mathbf{s}$.
    - Mechanism: Phase A (Parameter Training): Fix $\mathbf{m}, \mathbf{s}$ and train $\theta$ using SGD; Phase B (Model Sparsification): Fix $\theta, \mathbf{s}$ and optimize $\mathbf{m}$—the marginal likelihood gradient $\partial \log p / \partial m_j \approx -|\theta_j| \cdot \mathbb{E}[H_{jj}]$ provides a "marginal contribution score" for each weight; Phase C (Data Sparsification): Fix $\theta, \mathbf{m}$ and optimize $\mathbf{s}$—the marginal contribution of a sample is given by $\partial \log p / \partial s_i \approx \log p(y_i | \mathbf{x}_i, \theta, \mathbf{m}) + \text{Hessian term}$.
    - Design Motivation: Non-convex joint optimization problems are difficult to solve in a single step; alternating maximization is fast-converging and memory-friendly.

## Key Experimental Results

### Main Results: Joint Sparsification Performance (CIFAR-100 + ResNet-50)

| Method | Model Sparsity | Data Sparsity | Test ACC | Training Time | Inference FLOPs |
|------|-----------|-----------|----------|---------|------------|
| Dense Baseline | 0% | 0% | 78.3% | 1.0× | 1.0× |
| Model Pruning only (IMP) | 80% | 0% | 76.1% | 1.0× | 0.21× |
| Data Pruning only (forget) | 0% | 50% | 75.8% | 0.5× | 1.0× |
| Phased (IMP→forget) | 80% | 50% | 74.2% | 0.5× | 0.21× |
| **JMDS (Ours)** | 80% | 50% | **77.5%** | **0.4×** | **0.21×** |
| JMDS (Extreme) | 90% | 70% | **76.3%** | **0.3×** | **0.13×** |

### Cross-dataset / Model Generalization

| Dataset | Model | Phased ACC | **JMDS ACC** | Gain |
|--------|------|------------|----------|------|
| CIFAR-10 | ResNet-18 | 91.2 | **93.4** | +2.2 |
| CIFAR-100 | ResNet-50 | 74.2 | **77.5** | +3.3 |
| ImageNet | ResNet-50 | 72.1 | **74.8** | +2.7 |
| WikiText-2 | GPT-2 (Small) | 27.3 PPL | **24.9 PPL** | -2.4 PPL |
| WikiText-103 | GPT-2 (Medium) | 24.5 PPL | **22.1 PPL** | -2.4 PPL |

### Computational Overhead Analysis

| Method | Hessian Approx Cost | Algorithm Conv. Steps | Total Time vs Dense |
|------|----------------|-----------|------------|
| Exact Hessian | $O(d^3)$ → Infeasible | — | — |
| K-FAC + NTK Subsampling | $O(d \cdot l + s d)$ | 50-100 steps | 0.4-1.5× |
| Pure Heuristic | $O(1)$ | 100+ | 0.5× |

### Key Findings
- **Joint advantage is especially significant at high sparsity**: At 80% model + 50% data sparsity, JMDS outperforms the phased approach by 3.3%.
- **Consistency between theory and experiment**: The drop in marginal likelihood is highly correlated with accuracy loss.
- **Cross-task consistency**: Stable improvements across both CV and NLP tasks demonstrate the framework's universality.

## Highlights & Insights
- **First Principled Joint Sparsification**: Breaks the traditional paradigm of independent sparsification, revealing the coupling relationship between model and data.
- **Elegant Fusion of Theory and Practice**: Laplace approximation + K-FAC + NTK subsampling makes the theoretical objective feasible.
- **Unified Perspective**: Marginal likelihood serves as a unified metric that both measures model complexity and evaluates data contribution.

## Limitations & Future Work
- Scalability for Large Models: Current K-FAC approximations still face limits on GPT-2 Medium; further approximations are needed for GPT-2 Large+.
- Marginal Contribution Scores for Non-Gradient Methods: The current formula relies on gradient information and is not directly applicable to non-gradient tasks (e.g., retrieval).
- Convergence: Global convergence guarantees for alternating optimization are not provided.
- Improvements: Develop more efficient Hessian approximations (e.g., second-order NTK); extend to multi-modal and reinforcement learning scenarios.

## Related Work & Insights
- **vs Independent Sparsification (IMP, Forget-score)**: This work provides the first coupled optimization framework.
- **vs Bayesian Pruning**: Bayesian pruning primarily targets model sparsity; JMDS extends this to data sparsification.
- **Insight**: Marginal likelihood, as a unified metric for "combinatorial complexity," can be extended to joint problems involving architecture search + data selection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  Provides the first principled joint sparsification framework, surpassing heuristic joint methods.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐  Covers CV + NLP across 5 datasets with detailed ablation and theory-experiment cross-validation.
- Writing Quality: ⭐⭐⭐⭐  Mathematically rigorous and algorithmically clear, though derivations for some approximations could be supplemented.
- Value: ⭐⭐⭐⭐⭐  Joint compression holds significant practical value in the era of large models; the theoretical framework may inspire further joint optimization problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Last Vote: A Multi-Stakeholder Framework for Language Model Governance](../../NeurIPS2025/signal_comm/the_last_vote_a_multi-stakeholder_framework_for_language_model_governance.md)
- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](../../ICLR2026/signal_comm/multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)
- [\[NeurIPS 2025\] Feature-aware Modulation for Learning from Temporal Tabular Data](../../NeurIPS2025/signal_comm/feature-aware_modulation_for_learning_from_temporal_tabular_data.md)
- [\[ICML 2026\] Meta-learning Structure-Preserving Dynamics](meta-learning_structure-preserving_dynamics.md)
- [\[CVPR 2026\] FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection](../../CVPR2026/signal_comm/faar_efficient_frequency-aware_multi-task_fine-tuning_via_automatic_rank_selecti.md)

</div>

<!-- RELATED:END -->
