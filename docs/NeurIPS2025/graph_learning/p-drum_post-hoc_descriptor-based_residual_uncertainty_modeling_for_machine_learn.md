---
title: >-
  [Paper Note] P-DRUM: Post-hoc Descriptor-based Residual Uncertainty Modeling for Machine Learning Potentials
description: >-
  [NeurIPS 2025 (Workshop: ML4PS)][Graph Learning][Uncertainty Quantification] This paper proposes P-DRUM, a simple and efficient post-hoc uncertainty quantification framework that leverages descriptors from a trained graph neural network potential to estimate prediction residuals as uncertainty proxies, requiring no modification to the original model architecture or training pipeline.
tags:
  - "NeurIPS 2025 (Workshop: ML4PS)"
  - Graph Learning
  - Uncertainty Quantification
  - Machine Learning Interatomic Potentials
  - Residual Modeling
  - MACE
  - Out-of-Distribution Detection
date: 2026-05-08
content_hash: 91231c55d0f3dab5
---

# P-DRUM: Post-hoc Descriptor-based Residual Uncertainty Modeling for Machine Learning Potentials

**Conference**: NeurIPS 2025 (Workshop: ML4PS)  
**arXiv**: [2509.02927](https://arxiv.org/abs/2509.02927)  
**Code**: None  
**Area**: Graph Neural Networks / Computational Chemistry  
**Keywords**: Uncertainty Quantification, Machine Learning Interatomic Potentials, Residual Modeling, MACE, Out-of-Distribution Detection

## TL;DR

This paper proposes P-DRUM, a simple and efficient post-hoc uncertainty quantification framework that leverages descriptors from a trained graph neural network potential to estimate prediction residuals as uncertainty proxies, requiring no modification to the original model architecture or training pipeline.

## Background & Motivation

**Machine learning interatomic potentials (MLIPs)** are transforming materials science by enabling atomic-scale simulations at quantum mechanical accuracy but with computational efficiency orders of magnitude higher. However, the **predictive reliability** of MLIPs remains a critical concern, particularly for atomic configurations outside the training distribution.

**Limitations of existing uncertainty quantification (UQ) methods**:

| Method | Strengths | Limitations |
|------|------|------|
| **Ensemble methods** | Best performance; regarded as the gold standard | Requires training and running multiple models; high computational cost |
| **MC Dropout** | Exploits dropout at inference time | Not natively supported by some models (e.g., MACE); may degrade accuracy |
| **Deep Kernel Learning** | Combines NNs with Gaussian processes | Requires modification of the training pipeline |
| **kNN / GMM** | Post-hoc; operates in descriptor space | Does not exploit prediction error information |

**Core motivation**: Can one design a **post-hoc** method that estimates prediction errors solely from descriptors of a trained model, without modifying the model or accessing training logs?

## Method

### Overall Architecture

P-DRUM proceeds in three steps:

1. **Extract descriptors**: Given a trained MACE model, extract descriptor $D_{ij} \in \mathbb{R}^{d_{\text{desc}}}$ for each atom $j$ in each structure $X$.
2. **Compute residuals**: Energy residual $\Delta E = E - \hat{E}$ and force residual $\Delta \mathbf{F} = \mathbf{F} - \hat{\mathbf{F}}$.
3. **Train a lightweight MLP**: Using descriptors as input and residuals as targets, train an MLP to predict residuals; the magnitude of predicted residuals serves as the uncertainty indicator.

Training data construction: $\mathcal{S}_\Delta = \{D(X_i), \Delta E_i, \Delta \mathbf{F}_i\}_{i=1}^N$

### Key Designs

#### 1. Energy Residual Learning

To maintain **permutation invariance** and **flexibility** across varying atom counts, a per-atom scalar function $r^s: \mathbb{R}^{d_{\text{desc}}} \to \mathbb{R}$ is designed, modeling the energy residual as a sum of atomic contributions.

**Error norm learning (norm)**:
$$\mathcal{L}_{\text{E-norm}}(X_i) = \left(\sum_j^{n_i} r_{\text{E-norm}}^s(D_{ij}) - |\Delta E_i|\right)^2$$

**Signed error learning (diff)**:
$$\mathcal{L}_{\text{E-diff}}(X_i) = \left(\sum_j^{n_i} r_{\text{E-diff}}^s(D_{ij}) - \Delta E_i\right)^2$$

The distinction lies in whether the sign information of the error is preserved.

#### 2. Force Residual Learning

Force residuals are predicted directly at the per-atom level:

**Error norm learning (norm)**: Predicts the Euclidean norm of the force error
$$\mathcal{L}_{\text{F-norm}}(X_{ij}) = \left(r_{\text{F-norm}}^s(D_{ij}) - \|\Delta \mathbf{F}_{ij}\|\right)^2$$

**Signed error learning (diff)**: Predicts the three-dimensional force error vector
$$\mathcal{L}_{\text{F-diff}}(X_{ij}) = \frac{1}{3}\sum\left(r_{\text{F-diff}}^v(D_{ij}) - \Delta \mathbf{F}_{ij}\right)^2$$

where $r_{\text{F-diff}}^v: \mathbb{R}^{d_{\text{desc}}} \to \mathbb{R}^3$ is a vector-valued function.

### Loss & Training

- **Base model**: MACE with 32 channels, 5 Å cutoff, 2 interaction layers, 64-dimensional descriptors.
- **P-DRUM MLP**: 1–2 hidden layers with ReLU activations (softplus applied before output for norm variants).
- **Learning rate schedule**: Initial rate $10^{-3}$, halved with patience=10, minimum $10^{-7}$.
- **Maximum training**: 1000 epochs with early stopping.
- **Batch size**: 64 atoms (force training), 64 structures (energy training); 2048 for large datasets.
- **Computational overhead**: Requires only a single MACE forward pass plus negligible additional cost (vs. 5 forward passes for ensemble methods).

## Key Experimental Results

### Main Results: In-Distribution Uncertainty–Error Correlation (Spearman's $\rho$)

| Error Type | Method | Uracil | Salicylic | Malondialdehyde | Ni₃Al | HME21 |
|---------|------|--------|-----------|-----------------|-------|-------|
| Energy | Ensemble | 0.04 | 0.08 | -0.01 | 0.39 | 0.27 |
| | MC-dropout | -0.02 | -0.02 | -0.03 | -0.05 | 0.20 |
| | GMM | 0.07 | 0.07 | 0.13 | 0.64 | 0.06 |
| | kNN | 0.06 | 0.06 | 0.09 | 0.64 | -0.05 |
| | **P-DRUM-norm** | 0.12 | -0.01 | -0.09 | 0.62 | **0.30** |
| | **P-DRUM-diff** | **0.18** | **0.16** | **0.21** | **0.87** | 0.26 |
| Force | Ensemble | 0.68 | 0.65 | 0.69 | 0.97 | 0.78 |
| | MC-dropout | 0.24 | 0.27 | 0.27 | 0.87 | 0.68 |
| | GMM | 0.58 | 0.67 | 0.68 | 0.96 | 0.64 |
| | kNN | 0.52 | 0.61 | 0.65 | 0.96 | 0.54 |
| | **P-DRUM-norm** | **0.67** | **0.71** | 0.69 | **0.98** | **0.92** |
| | **P-DRUM-diff** | 0.53 | 0.58 | 0.57 | 0.96 | 0.85 |

### OOD Detection Results (Ni₃Al Dataset)

| Method | High-Temp Corr. | Hexagonal AUC | Cubic AUC | Atom Swap AUC | Overall Corr. |
|------|----------|---------|---------|------------|-----------|
| Ensemble | 0.98 | 0.94 | 1.00 | 1.00 | 0.90 |
| MC-dropout | 0.92 | 0.63 | 0.84 | 0.82 | 0.72 |
| GMM | 0.98 | 1.00 | 1.00 | 1.00 | 0.81 |
| kNN | 0.98 | 0.99 | 1.00 | 1.00 | 0.82 |
| **P-DRUM-norm** | 0.99 | 0.82 | 0.82 | 0.99 | 0.82 |
| **P-DRUM-diff** | 0.97 | 0.97 | 1.00 | 1.00 | **0.87** |

OOD detection covers four out-of-distribution scenarios: high-temperature molecular dynamics (2000K/3000K vs. training range of 500K–1500K), different crystal phases (hexagonal/cubic), and random atomic position swaps.

### Key Findings

1. **P-DRUM-diff achieves the best performance on energy UQ**: Retaining the sign of the error facilitates learning of scalar energy residuals.
2. **P-DRUM-norm achieves the best performance on force UQ**: Compressing the three-dimensional force error into a scalar norm reduces learning difficulty.
3. **P-DRUM shows a clear advantage on HME21 (37 elements)**: When elemental diversity is high, descriptor-space density alone (kNN/GMM) is insufficient to capture error correlations; explicitly leveraging error signals is necessary.
4. **P-DRUM-norm underperforms in OOD detection**: Compressing errors to norms may discard directional information that is important for detecting out-of-distribution inputs.
5. **P-DRUM-diff achieves the best overall performance**: It excels at both in-distribution energy UQ and OOD detection.

## Highlights & Insights

- **Practicality of post-hoc methods**: No modification to the model architecture, training pipeline, or training logs is required; P-DRUM can be directly applied to any trained MACE model.
- **High computational efficiency**: Requires only a single MACE forward pass (vs. 5 for ensemble methods), with negligible additional overhead.
- **PCA analysis reveals the source of P-DRUM's advantage**: In HME21, high-density regions of the descriptor space can exhibit high prediction errors — contrary to intuition — a relationship that kNN/GMM cannot capture, but P-DRUM can by learning from error signals.
- **Natural preservation of permutation invariance**: Symmetry of the molecular system is maintained through per-atom operations followed by summation.

## Limitations & Future Work

1. **Training set reuse**: Using the same dataset to train both the MLIP and P-DRUM may introduce bias; using a separate dataset reduces available samples.
2. **Ambiguity between norm and diff variants**: The two variants exhibit complementary strengths across tasks, with no uniformly optimal choice.
3. **Validation limited to MACE**: Other MLIP architectures (e.g., NequIP, SchNet) have not been tested.
4. **Weakness of P-DRUM-norm in OOD detection**: Information compression may be detrimental for out-of-distribution detection.
5. **Active learning applications unexplored**: P-DRUM's uncertainty estimates could be directly applied to active learning sample selection.

## Related Work & Insights

- **LTAU (Loss Trajectory Analysis for UQ)**: Requires recording per-atom loss trajectories during training; P-DRUM is more lightweight.
- **Orb-v3's pLDDT-style approach**: Jointly optimizes UQ with model training; P-DRUM serves as a post-hoc alternative.
- **AlphaFold's pLDDT**: The idea of discretizing prediction errors was introduced into the MLIP domain by Orb-v3.
- **Effectiveness of descriptors in downstream tasks**: P-DRUM provides evidence for a new use case of descriptors — uncertainty estimation.

## Rating

- Novelty: ⭐⭐⭐⭐ — Simple yet effective post-hoc UQ framework
- Theoretical Contribution: ⭐⭐⭐ — Primarily experiment-driven work
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple datasets, multiple baselines, with OOD evaluation and PCA analysis
- Practical Value: ⭐⭐⭐⭐⭐ — Plug-and-play; directly valuable to the MLIP community
- Overall Recommendation: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Generative Graph Pattern Machine](generative_graph_pattern_machine.md)
- [\[ICLR 2026\] Relatron: Automating Relational Machine Learning over Relational Databases](../../ICLR2026/graph_learning/relatron_automating_relational_machine_learning_over_relational_databases.md)
- [\[NeurIPS 2025\] S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning](smore_structural_mixture_of_residual_experts_for_parameter-efficient_llm_fine-tu.md)
- [\[AAAI 2026\] GSAP-ERE: Fine-Grained Scholarly Entity and Relation Extraction Focused on Machine Learning](../../AAAI2026/graph_learning/gsap-ere_fine-grained_scholarly_entity_and_relation_extraction_focused_on_machin.md)
- [\[NeurIPS 2025\] MoEMeta: Mixture-of-Experts Meta Learning for Few-Shot Relational Learning](moemeta_mixture-of-experts_meta_learning_for_few-shot_relational_learning.md)

<!-- RELATED:END -->
