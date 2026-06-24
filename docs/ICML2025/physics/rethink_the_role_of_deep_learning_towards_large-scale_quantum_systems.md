---
title: >-
  [Paper Note] Rethink the Role of Deep Learning towards Large-scale Quantum Systems
description: >-
  [ICML2025][Physics & Scientific Computing][Quantum System Learning] This paper systematically compares the performance of ML and DL in Quantum System Learning (QSL) tasks under a unified quantum resource constraint. It finds that traditional ML (Lasso/Ridge/kernel methods) often matches or even outperforms DL, challenging the intuition that "large-scale quantum systems must utilize deep learning."
tags:
  - "ICML2025"
  - "Physics & Scientific Computing"
  - "Quantum System Learning"
  - "Ground State Property Estimation"
  - "Quantum Phase Classification"
  - "Deep Learning vs. Machine Learning"
  - "Classical Shadow"
  - "Large-scale Quantum Systems"
date: 2026-05-08
content_hash: 59bf12f370b418c1
---

# Rethink the Role of Deep Learning towards Large-scale Quantum Systems

**Conference**: ICML2025  
**arXiv**: [2505.13852](https://arxiv.org/abs/2505.13852)  
**Code**: [GitHub](https://github.com/) (Some dataset generation code has been open-sourced)  
**Area**: Physics / Quantum System Learning  
**Keywords**: Quantum System Learning, Ground State Property Estimation, Quantum Phase Classification, Deep Learning vs. Machine Learning, Classical Shadow, Large-scale Quantum Systems

## TL;DR

This paper systematically compares the performance of ML and DL in Quantum System Learning (QSL) tasks under a unified quantum resource constraint. It finds that traditional ML (Lasso/Ridge/kernel methods) often matches or even outperforms DL, challenging the intuition that "large-scale quantum systems must utilize deep learning."

## Background & Motivation

- **Core Problem**: Ground state property estimation (correlation functions, entanglement entropy) of quantum systems and quantum phase classification are fundamental problems in quantum physics, but classical exact simulation is limited by the curse of dimensionality.
- **AI for Quantum**: Recent years have witnessed a surge of ML/DL methods applied to quantum system learning (QSL), including linear regression, kernel methods, MLP, CNN, and self-supervised large models (LLM4QPE).
- **Limitations of Prior Work**: Prior studies often construct training data for DL using substantially more quantum measurement resources than for ML (e.g., infinite measurements to generate labels), leading to unfair comparisons.
- **Key Challenge**: Under the realistic constraint of scarce quantum resources, is deep learning truly necessary for QSL tasks?

## Method

### Unified Resource Framework

The core design of this paper is the **Unified Quantum Resource Budget**: the training dataset $\mathcal{D}$ for all models satisfies the same total query constraint of $n \times M$, where $n$ is the number of training samples, and $M$ is the number of measurement snapshots per sample.

For self-supervised models (SSL), it must satisfy:

$$n_{\text{pre}} \times M_{\text{pre}} + n_{\text{sft}} \times M_{\text{sft}} = n \times M$$

### Hamiltonian Families Explored

1. **Heisenberg Model (HB)**: $\mathsf{H}_{\text{HB}}(\mathbf{x}) = \sum_{i<j} J_{ij}(X_iX_j + Y_iY_j + Z_iZ_j)$, where $J_{ij} = 369 / |i-j|^a$, $a \in (1,2)$
2. **Transverse-Field Ising Model (TFIM)**: $\mathsf{H}_{\text{TFIM}}(\mathbf{x}) = -\sum_{i=1}^{N-1} J_i Z_i Z_{i+1} - \sum_{i=1}^{N} h_i X_i$
3. **Rydberg Atom Model**: $\mathsf{H}_{\text{Ryd}}(\mathbf{x}) = \sum_{i<j} \frac{\Omega R_b^6}{a^6|i-j|^6} N_i N_j + \sum_{i=1}^{N} \frac{\Omega}{2}X_i - \Delta_i N_i$

### Ground State Property Targets

- **Correlation Function** $C_{ij}$: $C_{ij} = \frac{\text{tr}(X_iX_j\rho) + \text{tr}(Y_iY_j\rho) + \text{tr}(Z_iZ_j\rho)}{3}$
- **2nd-order Rényi Entanglement Entropy**: $\mathcal{S}_2(\rho_A) = -\log[\text{tr}(\rho_A^2)]$
- **Quantum Phase Classification**: Classifying the Rydberg model into three phases: $Z_2$-ordered phase, $Z_3$-ordered phase, and disordered phase.

### Benchmark Models

| Category | Model | Characteristics |
|------|------|------|
| ML — Linear Regression | Lasso, Ridge | Random Fourier Features + $L_1/L_2$ regularization |
| ML — Kernel Methods | DK, RBFK, NTK | Dirichlet Kernel / Radial Basis Function Kernel / Neural Tangent Kernel |
| ML — Tree Models | RF, GBT, LGBM, XGBoost | Used for classification tasks |
| DL — Supervised Learning | MLP, CNN (and MLP-A, CNN-A) | With/without measurement auxiliary information |
| DL — Self-Supervised | SG, LLM4QPE-F, LLM4QPE-T | Shadow Generator; LLM4QPE (with/without pre-training) |

### Randomization Test

To examine the role of measurement results as input features, the actual measurement values $\mathbf{v}$ are replaced with random integers $\mathbf{v}'$ uniformly sampled from $[0,5]$, and the changes in model performance are observed.

## Key Experimental Results

### Correlation Function Prediction (HB, RMSE $\epsilon(\bar{C})$, $M=64$)

| Model | $N=48, n=100$ | $N=127, n=100$ |
|------|---------------|----------------|
| Classical Shadow | 0.2114 | 0.2145 |
| MLP-4 layers | 0.0352 | 0.0861 |
| CNN-4 layers | 0.0346 | 0.0522 |
| LLM4QPE-T | 0.0320 | 0.0263 |
| **Lasso** | **0.0249** | **0.0208** |
| **Ridge** | **0.0248** | **0.0216** |

**Key Conclusion**: Lasso/Ridge outperforms DL models by a significant margin in almost all settings.

### Scaling Behavior (127-qubit HB, $n=100, M=512$)

- Lasso: $\epsilon(\bar{C}) = 0.011$
- Ridge: $\epsilon(\bar{C}) = 0.012$
- LLM4QPE-F (~18.1M parameters): $\epsilon(\bar{C}) = 0.017$

### Model Scale Experiment

- An MLP with strong regularization (large $\lambda$) requires only **1/36th of the parameters** of LLM4QPE-F to match its performance.
- Without regularization, larger models exhibit more severe overfitting.

### Quantum Phase Classification (31-qubit Rydberg, Accuracy %)

| Model | $M=64, n=100$ | $M=256, n=100$ |
|------|---------------|----------------|
| MLP | 92.79 | 94.50 |
| CNN | 92.50 | 92.79 |
| LLM4QPE-T | — | — |

### Randomization Test Results

- **GSPE Task**: After replacing the real measurement results with random values, the performance of LLM4QPE-T remains largely unchanged $\rightarrow$ measurement results are **redundant** for property estimation.
- **QPC Task**: The performance drops significantly after replacement $\rightarrow$ measurement results are **crucial** for phase classification.

## Highlights & Insights

1. **Unified Fair Comparison Framework**: Posits the first systematic comparison of ML and DL under a unified quantum resource budget (identical $n \times M$), filling the gap in fair benchmarks within the field.
2. **Counter-intuitive Findings**: Simple Lasso/Ridge consistently outperforms complex DL architectures (MLP, CNN, LLM4QPE) on GSPE tasks, challenging the "deep learning is all you need" assumption in this domain.
3. **Measurement Redundancy**: The randomization test reveals that measurement results as inputs are redundant in GSPE but crucial in QPC. This duality provides clear guidelines for future model designs.
4. **Scale Is Not All**: Larger DL models do not necessarily yield better results; regularization strategies are far more critical than blindly increasing parameter counts.
5. **Large-scale Validation**: Experiments are scaled up to 127 qubits, covering three major Hamiltonian families, showcasing highly robust conclusions.

## Limitations & Future Work

1. **Limited Task Coverage**: Only correlation functions, entanglement entropy, and phase classification are investigated; broader tasks like quantum state tomography and fidelity estimation are not covered.
2. **Hamiltonian Scope**: While representative, the three Hamiltonian families do not cover more complex systems such as chemical molecules or topological order.
3. **Real Quantum Hardware**: All experiments are conducted on simulated datasets; noise on real quantum devices might alter the performance trade-offs between ML and DL.
4. **Insufficient Exploration of DL Architectures**: The potential of more modern architectures, such as Transformers or Graph Neural Networks, has not been tested in QSL.
5. **Lack of Theoretical Analysis**: The paper does not provide a theoretical explanation for why linear models are sufficient for these tasks, offering empirical observations only.

## Related Work & Insights

- Huang et al., 2022: Proved that classical ML algorithms utilizing quantum data can achieve efficient GSPE, laying the foundation for ML approaches.
- Lewis et al., 2024; Wanner et al., 2024: Provably efficient ML methods using linear regression with geometric feature maps.
- Wang et al., 2022 (Shadow Generator): Autoregressive generation of classical shadows.
- Tang et al., 2024 (LLM4QPE): Introduced the LLM pre-training paradigm to quantum property estimation.

## Rating

- Novelty: ⭐⭐⭐⭐ — Offers a fresh perspective, presenting the first systematic and fair comparison of ML vs. DL in QSL.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage of multiple models, tasks, and scales (up to 127 qubits), including randomization tests.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured and problem-driven, though some notations are dense.
- Value: ⭐⭐⭐⭐ — Serves as an important advisory to the quantum ML community against blindly pursuing DL complexity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Erwin: A Tree-based Hierarchical Transformer for Large-scale Physical Systems](erwin_a_tree-based_hierarchical_transformer_for_large-scale_physical_systems.md)
- [\[ICML 2026\] Rethink the Role of Neural Decoders in Quantum Error Correction](../../ICML2026/physics/rethink_the_role_of_neural_decoders_in_quantum_error_correction.md)
- [\[ICML 2025\] Liger: Linearizing Large Language Models to Gated Recurrent Structures](liger_linearizing_large_language_models_to_gated_recurrent_structures.md)
- [\[NeurIPS 2025\] TITAN: A Trajectory-Informed Technique for Adaptive Parameter Freezing in Large-Scale VQE](../../NeurIPS2025/physics/titan_a_trajectory-informed_technique_for_adaptive_parameter_freezing_in_large-s.md)
- [\[NeurIPS 2025\] Toward Complete Merger Identification at Cosmic Noon with Deep Learning](../../NeurIPS2025/physics/toward_complete_merger_identification_at_cosmic_noon_with_deep_learning.md)

</div>

<!-- RELATED:END -->
