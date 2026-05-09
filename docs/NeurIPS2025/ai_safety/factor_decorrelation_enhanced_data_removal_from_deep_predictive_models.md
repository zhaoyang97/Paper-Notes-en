---
title: >-
  [Paper Note] Factor Decorrelation Enhanced Data Removal from Deep Predictive Models
description: >-
  [NeurIPS 2025][AI Safety][machine unlearning] This paper proposes DecoRemoval, a framework that achieves data removal without full retraining via two modules: discriminability-preserving factor decorrelation (RFF-based spatial mapping with adaptive weighting) and smoothed loss perturbation. The method significantly outperforms existing approaches, particularly under out-of-distribution (OOD) settings.
tags:
  - NeurIPS 2025
  - AI Safety
  - machine unlearning
  - certified removal
  - factor decorrelation
  - OOD robustness
  - random Fourier features
date: 2026-05-08
content_hash: c8487ae19ee4eac2
---

# Factor Decorrelation Enhanced Data Removal from Deep Predictive Models

**Conference**: NeurIPS 2025
**arXiv**: [2509.23443](https://arxiv.org/abs/2509.23443)
**Code**: [GitHub](https://anonymous.4open.science/r/DecoRemoval-770220/)
**Area**: AI Safety / Machine Unlearning
**Keywords**: machine unlearning, certified removal, factor decorrelation, OOD robustness, random Fourier features

## TL;DR
This paper proposes DecoRemoval, a framework that achieves data removal without full retraining via two modules: discriminability-preserving factor decorrelation (RFF-based spatial mapping with adaptive weighting) and smoothed loss perturbation. The method significantly outperforms existing approaches, particularly under out-of-distribution (OOD) settings.

## Background & Motivation

**Background**: Data removal (machine unlearning) is a hard requirement for privacy compliance. Certified Removal (CR) achieves efficient parameter adjustment through gradient updates and calibrated noise injection, representing the most theoretically grounded approach to date.

**Limitations of Prior Work**: Existing data removal methods assume similar data distributions before and after removal. In practice, however, continuous removal requests cause distributional shift. Under OOD settings, the intrinsic correlation between feature representations and labels changes, degrading both the removal accuracy and generalization ability of existing unlearning mechanisms.

**Key Challenge**: Feature dimensionality reduction is an effective strategy for handling OOD, yet it risks discarding discriminative features. If the loss function is not adapted to the reduced representation space, gradient directions deviate from the optimization objective, leading to training instability and generalization degradation.

**Goal**: (1) Reduce feature redundancy and correlation during the removal process to counter distributional shift; (2) Ensure privacy safety during removal to prevent information leakage.

**Key Insight**: The method borrows Random Fourier Features (RFF) from StableNet to perform feature space mapping and decorrelation, while applying randomized linear perturbations to smooth the loss function so that local parameter adjustments can accurately approximate the effect of full retraining.

**Core Idea**: Feature decorrelation + loss perturbation → local parameter updates remain a valid substitute for full retraining even under distributional shift.

## Method

### Overall Architecture
DecoRemoval consists of two main modules:
- **Input**: Trained model $A(D)$, data to be removed $(x_n, y_n)$
- **Module 1**: Discriminability-Preserving Factor Decorrelation (DP-FD) — dimensionality reduction and decorrelation via RFF mapping
- **Module 2**: Smoothed Data Removal — loss perturbation + Newton-step removal
- **Output**: Post-removal model parameters $w_{clf}^-$

### Key Designs

1. **Random Fourier Feature Mapping (RFF Mapping)**:

    - Function: Maps input features to a high-dimensional space to achieve decorrelation
    - Mechanism: $Z_i = \sqrt{2} \cdot \cos(\omega X_i + \phi)$, where $\omega \sim \mathcal{N}(0, I)$, $\phi \sim \text{Uniform}(0, 2\pi)$
    - This transformation is grounded in the Fourier transform of kernel functions, enabling feature embedding without explicitly computing the kernel
    - Design Motivation: Achieves feature dimensionality reduction and decorrelation with linear computational complexity, reducing feature redundancy

2. **Feature Decorrelation via Sample Weighting**:

    - Function: Minimizes statistical dependencies among transformed features by optimizing sample weights
    - Mechanism: Measures feature dependency via the cross-covariance operator $\Sigma_{AB}$, quantified by the Frobenius norm $I_{AB} = \|\hat{\Sigma}_{AB}\|_F^2$
    - Weighted covariance matrix: $\hat{\Sigma}_{AB;w} = \frac{1}{n-1} \sum_{i=1}^n [(w_i u(Z_i) - \bar{w u})^T \cdot (w_i v(Z_i) - \bar{w v})]$
    - Optimal weights: $w^* = \arg\min_{w \in \Delta_n} \sum_{1 \leq i < j \leq m_Z} \|\hat{\Sigma}_{Z_{:,i}Z_{:,j};w}\|_F^2$
    - Design Motivation: Naive RFF may scatter discriminative information; adaptive weighting preserves class separability

3. **Loss Perturbation + Certified Removal**:

    - Function: Injects a random linear term into the training loss to obfuscate gradient signals, followed by a Newton-step update to perform removal
    - Perturbed loss: $L_{\mathbf{p}}(w_{clf}; D) = \sum_{i=1}^n L(w_{clf}^\top x_i, y_i) + \mathbf{b}^\top w_{clf}$
    - Newton-step removal: $w_{clf}^- = w_{clf}^* + H_{w_{clf}^*}^{-1} \nabla L(w_{clf}^*; (x_n, y_n))$
    - where $H_{w_{clf}^*} = \nabla^2 L(w_{clf}^*; D')$ is the Hessian matrix
    - Key property: The linear perturbation $\mathbf{b}^\top w_{clf}$ shifts the gradient by the constant $\mathbf{b}$ without altering the Hessian, so the removal update remains valid
    - Design Motivation: $\mathbf{b}$ obfuscates the gradient signal attributable to specific samples, reducing the risk of information leakage

### Certified Removal Guarantee
Core definition: A removal mechanism $M$ satisfies $\epsilon$-CR if:

$$e^{-\epsilon} \leq \frac{P(M(A(D), D, x) \in S)}{P(A(D \setminus x) \in S)} \leq e^{\epsilon}$$

That is, the influence of removing a single data point on the model output does not exceed the exponential factor $\epsilon$.

## Key Experimental Results

### Main Results: ACC (%) and F1 Comparison under OOD Settings

| Dataset | Removal Size | Retrain | CR | SISA | DP-SGD | SSD | CU | **DR (Ours)** |
|--------|--------|---------|-----|------|--------|-----|-----|-------------|
| MNIST | 1K | 51.75 | 43.13 | 43.64 | 45.78 | 45.45 | 47.35 | **48.97** |
| MNIST | 10K | 51.02 | 41.87 | 42.96 | 44.87 | 45.03 | 46.53 | **48.34** |
| CIFAR-10 | 1K | 50.76 | 43.09 | 43.21 | 45.30 | 45.06 | 46.84 | **48.56** |
| CIFAR-10 | 10K | 50.01 | 41.76 | 42.51 | 44.37 | 44.51 | 46.11 | **47.83** |
| SST-2 | 1K | 91.76 | 89.71 | 89.98 | 90.45 | 89.98 | 89.94 | **90.45** |
| ESS | 1K | 55.43 | 48.61 | 48.64 | 50.47 | 50.15 | 51.34 | **54.97** |
| CGSS | 1K | 51.60 | 41.24 | 43.52 | 46.76 | 47.77 | 48.77 | **50.82** |

### F1 Score Comparison (10K Samples Removed)

| Dataset | Retrain | CR | SSD | CU | **DR (Ours)** |
|--------|---------|-----|-----|-----|-------------|
| MNIST | 0.495 | 0.390 | 0.450 | 0.450 | **0.473** |
| CIFAR-10 | 0.491 | 0.389 | 0.443 | 0.444 | **0.469** |
| SST-2 | 0.839 | 0.796 | 0.816 | 0.805 | **0.821** |
| ESS | 0.530 | 0.390 | 0.478 | 0.470 | **0.510** |
| CGSS | 0.502 | 0.454 | 0.468 | 0.477 | **0.495** |

### Key Findings
- DecoRemoval consistently achieves results closest to full retraining across all 5 datasets and 3 removal scales
- Improvements are most pronounced on social survey datasets (ESS/CGSS): ACC exceeds the second-best method (CU) by 3–4 percentage points, as these datasets exhibit high feature correlation where RFF decorrelation is particularly effective
- The advantage is greater under large-scale removal (10K): competing methods degrade rapidly while DecoRemoval remains stable
- Improvements on the SST-2 text dataset are comparatively modest (~0.5%), as textual features are already well decorrelated through pretraining

## Highlights & Insights
- The **RFF → decorrelation → data removal** pipeline is novel: this is the first work to introduce feature decorrelation into unlearning, addressing the root cause of performance degradation under OOD settings
- **Dual role of loss perturbation**: it both smooths the optimization landscape as regularization (enabling effective Newton updates) and serves as a privacy protection mechanism (obfuscating gradient signals)
- The theoretical property that linear perturbation does not alter the Hessian guarantees that **removal correctness is unaffected** by perturbation
- The OOD construction strategy in the experiments (randomly assigning 10% of class-A samples to the class-B test set) is simple yet effective

## Limitations & Future Work
- The OOD setting is somewhat artificial: distributional shift is constructed by randomly swapping only 10% of samples, whereas real-world distribution drift may be considerably more complex
- Evaluation is limited to classification tasks with an MLP backbone; validation on Transformers or deeper networks is absent
- The effect of RFF mapping dimension $m_Z$ and kernel function selection on performance is not sufficiently ablated
- The theoretical foundation of $\epsilon$-CR guarantees is weaker for non-convex deep networks; the paper circumvents this by decomposing the network into a feature extractor and a linear classifier
- Inference overhead is not adequately reported: the practical time cost of weight optimization and Hessian computation remains unclear

## Related Work & Insights
- **vs. Certified Removal (Guo et al.)**: CR is the foundation but assumes invariant data distribution; DecoRemoval extends it to OOD settings via RFF decorrelation
- **vs. SSD (Foster et al.)**: SSD selectively suppresses parameters using the Fisher information matrix, focusing on parameter sensitivity with respect to the forget set; DecoRemoval approaches the problem from the feature space perspective
- **vs. StableNet**: StableNet applies RFF to stabilize OOD classification; DecoRemoval transfers this technique to the unlearning setting, serving as a representative case of cross-task method adaptation

## Rating
- Novelty: ⭐⭐⭐⭐ — First to introduce feature decorrelation into data removal; the combination of RFF and loss perturbation is creative
- Experimental Thoroughness: ⭐⭐⭐⭐ — 5 datasets, 3 removal scales, 6 baselines, covering image/text/structured data
- Writing Quality: ⭐⭐⭐ — Readable overall, but notation is inconsistent in places and the OOD setup description is insufficiently detailed
- Value: ⭐⭐⭐⭐ — Addresses a practical pain point of unlearning under distributional shift; the proposed pipeline warrants further attention

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Preserving Task-Relevant Information Under Linear Concept Removal](preserving_task-relevant_information_under_linear_concept_removal.md)
- [\[NeurIPS 2025\] Nearly-Linear Time Private Hypothesis Selection with the Optimal Approximation Factor](nearly-linear_time_private_hypothesis_selection_with_the_optimal_approximation_f.md)
- [\[AAAI 2026\] Privacy on the Fly: A Predictive Adversarial Transformation Network for Mobile Sensor Data](../../AAAI2026/ai_safety/privacy_on_the_fly_a_predictive_adversarial_transformation_network_for_mobile_se.md)
- [\[NeurIPS 2025\] Distributional Adversarial Attacks and Training in Deep Hedging](distributional_adversarial_attacks_and_training_in_deep_hedging.md)
- [\[NeurIPS 2025\] Provable Watermarking for Data Poisoning Attacks](provable_watermarking_for_data_poisoning_attacks.md)

<!-- RELATED:END -->
