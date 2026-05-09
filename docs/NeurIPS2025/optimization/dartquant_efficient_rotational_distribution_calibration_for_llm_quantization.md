---
title: >-
  [Paper Note] DartQuant: Efficient Rotational Distribution Calibration for LLM Quantization
description: >-
  [NeurIPS 2025][Optimization][LLM quantization] DartQuant proposes a distribution-calibration-based rotation matrix optimization method that drives activation distributions toward uniformity via a Whip loss to reduce quantization error, and replaces expensive manifold optimizers with QR-Orth, achieving 47× speedup and 10× memory reduction on 70B models — enabling large-model rotation calibration on a single RTX 3090 for the first time.
tags:
  - NeurIPS 2025
  - Optimization
  - LLM quantization
  - rotation matrix
  - distribution calibration
  - orthogonal optimization
  - post-training quantization
date: 2026-05-08
content_hash: 91a46df1ed736904
---

# DartQuant: Efficient Rotational Distribution Calibration for LLM Quantization

**Conference**: NeurIPS 2025
**arXiv**: [2511.04063](https://arxiv.org/abs/2511.04063)
**Code**: [https://github.com/CAS-CLab/DartQuant](https://github.com/CAS-CLab/DartQuant)
**Area**: Optimization
**Keywords**: LLM quantization, rotation matrix, distribution calibration, orthogonal optimization, post-training quantization

## TL;DR
DartQuant proposes a distribution-calibration-based rotation matrix optimization method that drives activation distributions toward uniformity via a Whip loss to reduce quantization error, and replaces expensive manifold optimizers with QR-Orth, achieving 47× speedup and 10× memory reduction on 70B models — enabling large-model rotation calibration on a single RTX 3090 for the first time.

## Background & Motivation

**Background**: Post-training quantization (PTQ) is a key technique for deploying large language models. Extreme outliers in activations are the primary cause of quantization accuracy degradation. Rotation matrix transformations can effectively smooth outliers and can be losslessly fused into model architectures due to their invertibility and norm-preserving properties.

**Limitations of Prior Work**: SpinQuant and OSTQuant treat rotation matrices as network parameters and perform end-to-end fine-tuning, requiring specialized optimizers such as Cayley SGD. Calibrating a 70B model demands hundreds of GiB of GPU memory and tens of GPU hours.

**Key Challenge**: End-to-end fine-tuning faces three compounding challenges: (a) computational resource consumption conflicts with the goal of rapid PTQ deployment; (b) fine-tuning on small calibration sets is prone to overfitting; (c) Cayley/Riemannian SGD incurs approximately 2× the computational overhead of standard optimizers.

**Goal**: How to optimize rotation matrices rapidly under extremely limited resources while avoiding overfitting.

**Key Insight**: The paper reframes rotation optimization from a distribution-transformation perspective — rather than minimizing task loss, it directly constrains the post-rotation activation distribution to approach a uniform distribution.

**Core Idea**: The Whip loss drives activation distributions from Laplace toward uniform to reduce outliers, while QR decomposition replaces manifold optimization to enforce orthogonality.

## Method

### Overall Architecture
The DartQuant pipeline consists of: (1) forward propagation with a small calibration dataset to obtain per-layer activations; (2) token sampling; (3) minimizing the Whip loss with the QR-Orth optimizer to indirectly optimize the rotation matrix $R$; and (4) fusing $R$ into model weights with zero additional inference overhead. The entire process is independent of task loss, with each layer calibrated independently and no end-to-end backpropagation required.

### Key Designs

1. **Rotational Distribution Calibration**:

    - Function: Optimizes the rotation matrix by constraining the distributional shape of post-rotation activations.
    - Mechanism: The original objective $\min_R \sum_{i=1}^{c_{in}} \mathbb{I}(|(Rx)_i| > \tau)$ involves a non-differentiable indicator function. Variance is difficult to optimize under norm-invariance constraints, and kurtosis converges slowly, motivating the need for a new surrogate objective.
    - Design Motivation: To decouple optimization from end-to-end task loss and directly optimize from the distributional level.

2. **Whip Loss**:

    - Function: Drives activation distributions from Laplace toward uniform.
    - Mechanism: Activations are approximated as $\text{Laplace}(0, b)$, whose CDF transform $U_X(x) = \tau[\exp(x/b)-1]$ (for $x \leq 0$) expands the region near zero while compressing the tails. Motivated by this, the Whip loss is defined as: $\text{Whip} = \sum_{i=1}^{c_{in}} \exp(-|x_i|)$.
    - Design Motivation: The Whip loss has the largest gradient near zero, pushing small-magnitude activations away from zero. Under norm-invariance constraints, as small values increase, outliers are naturally compressed, producing a "gathering effect" that drives the distribution toward uniformity.

3. **QR-Orth Optimization**:

    - Function: Replaces Cayley/Riemannian SGD with QR decomposition to enforce orthogonality constraints.
    - Mechanism: QR decomposition of an arbitrary matrix $Z$ yields an orthogonal matrix $R$; the latent parameter $Z$ is optimized via standard gradient descent, and $R$ is re-derived by decomposition at each step. Complexity is $\frac{4}{3}n^3$, far lower than Cayley's $6n^3$.
    - Design Motivation: Eliminates manifold projection operations, enabling the use of any standard optimizer (SGD/Adam), with an empirical speedup of 1.4×.

### Loss & Training
- Calibration data: 128 WikiText2 samples with token length 2048.
- Optimizer: SGD + QR-Orth; weight reconstruction uses GPTQ.
- Rotation matrices $R_1, R_2$ are optimized via QR-Orth; $R_3, R_4$ use random Hadamard matrices.

## Key Experimental Results

### Main Results

| Model | Quant. Config | Method | PPL ↓ | 0-shot Acc ↑ |
|-------|--------------|--------|-------|-------------|
| Llama-2 7B | W4A4KV16 | QuaRot | 20.63 | 57.90 |
| Llama-2 7B | W4A4KV16 | SpinQuant | 19.90 | 57.85 |
| Llama-2 7B | W4A4KV16 | OSTQuant | 19.24 | 57.94 |
| Llama-2 7B | W4A4KV16 | **DartQuant** | **18.53** | **58.05** |
| Llama-3 70B | W4A4KV16 | SpinQuant | 9.61 | 66.06 |
| Llama-3 70B | W4A4KV16 | OSTQuant | 7.67 | 67.94 |
| Llama-3 70B | W4A4KV16 | **DartQuant** | 7.99 | **69.39** |

### Optimization Cost Comparison

| Metric | Method | 7B | 13B | 70B |
|--------|--------|----|-----|-----|
| Time (GPU hr) | SpinQuant | 0.30 | 0.70 | 42.90 |
| Time (GPU hr) | OSTQuant | 0.30 | 0.80 | 44.00 |
| Time (GPU hr) | **DartQuant** | **0.14** | **0.23** | **0.91** |
| Memory (GiB) | SpinQuant | 19.98 | 33.73 | 238.89 |
| Memory (GiB) | OSTQuant | 42.25 | 239.16 | 583.86 |
| Memory (GiB) | **DartQuant** | **17.41** | **21.40** | **23.47** |

For the 70B model: only 0.91 GPU hr and 23.47 GiB are required, enabling calibration on a single RTX 3090 (approximately 3 hours) for the first time.

### Ablation Study

| Optimization Objective | Effect | Notes |
|------------------------|--------|-------|
| Quantization loss | Negligible change | Weak gradient signal |
| Variance | Negligible change | Constrained by norm invariance |
| Kurtosis | Marginal improvement | Slow convergence |
| **Whip** | **Significant improvement** | Distribution approaches uniform; fast convergence |

| Optimizer | Time per 100 steps | Speedup |
|-----------|-------------------|---------|
| Cayley SGD | 8.2h | 1.0× |
| QR-Orth SGD | 5.7h | 1.44× |
| QR-Orth SGD (with convergence acceleration) | — | **41×** |

### Key Findings
- The Whip loss is the only optimization objective that significantly alters activation distributions; all other objectives are largely ineffective.
- DartQuant is insensitive to the choice of calibration dataset (results are consistent across WikiText2/PTB/C4), whereas end-to-end fine-tuning methods exhibit significant variance.
- QR-Orth is both faster per step (1.4×) and converges faster (6 steps ≈ Cayley 100 steps), yielding a combined speedup of 41×.

## Highlights & Insights
- **Distribution Calibration as a Substitute for End-to-End Fine-Tuning**: Shifting from "optimizing task loss" to "optimizing the shape of activation distributions" substantially reduces cost and avoids overfitting; the paradigm is generalizable to other orthogonally constrained settings.
- **Theoretical Grounding of the Whip Loss**: By leveraging the CDF transform from Laplace to uniform distribution, the paper elegantly reduces distributional transformation to a simple exponential loss.
- **Generality of QR-Orth**: Any scenario requiring orthogonal matrix optimization can employ indirect optimization via QR decomposition, entirely bypassing the complexity of manifold optimization.

## Limitations & Future Work
- Validation is limited to the Llama family and a small number of MoE models; generalization to architectures such as Qwen and Gemma remains unverified.
- The Whip loss assumes activations approximately follow a Laplace distribution; performance may degrade for models with significantly different activation distributions.
- Online rotation matrices $R_3, R_4$ still use random Hadamard matrices; distribution-calibration-based optimization for these remains unexplored.
- Performance under more extreme configurations such as 2-bit quantization is not discussed.

## Related Work & Insights
- **vs. QuaRot**: QuaRot applies random Hadamard rotations, whereas DartQuant learns optimal rotation matrices through distribution calibration.
- **vs. SpinQuant**: SpinQuant performs end-to-end fine-tuning of rotation matrices, incurring high resource costs and overfitting risk; DartQuant performs layer-wise calibration, reducing cost by 47×.
- **vs. OSTQuant**: OSTQuant jointly optimizes rotations and scaling factors but relies on end-to-end training; DartQuant operates independently of task loss.

## Rating
- Novelty: ⭐⭐⭐⭐ The distribution calibration perspective offers a fresh approach to rotation-based quantization, though the field already has substantial prior work on rotation matrix quantization.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparisons across multiple models and configurations, detailed ablations, and convincing efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear logical flow with rich figures; motivation derivation is natural and well-structured.
- Value: ⭐⭐⭐⭐⭐ Enabling 70B model calibration on a single RTX 3090 substantially lowers the barrier to quantization deployment; engineering impact is outstanding.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] OrthoGrad Improves Neural Calibration](orthograd_improves_neural_calibration.md)
- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](efficient_adaptive_federated_optimization.md)
- [\[NeurIPS 2025\] Efficient Adaptive Experimentation with Noncompliance](efficient_adaptive_experimentation_with_noncompliance.md)
- [\[NeurIPS 2025\] Oracle-Efficient Combinatorial Semi-Bandits](oracle-efficient_combinatorial_semi-bandits.md)
- [\[NeurIPS 2025\] MeCeFO: Enhancing LLM Training Robustness via Fault-Tolerant Optimization](mecefo_enhancing_llm_training_robustness_via_fault-tolerant_optimization.md)

<!-- RELATED:END -->
