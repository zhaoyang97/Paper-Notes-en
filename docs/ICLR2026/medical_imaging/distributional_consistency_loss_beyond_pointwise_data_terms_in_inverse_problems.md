---
title: >-
  [Paper Note] Distributional Consistency Loss: Beyond Pointwise Data Terms in Inverse Problems
description: >-
  [ICLR 2026][Medical Imaging][Inverse Problems] This paper proposes the Distributional Consistency (DC) loss, which replaces conventional pointwise data fidelity terms (e.g., MSE/NLL) with distribution-level calibration…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Inverse Problems"
  - "Data Fidelity"
  - "Distributional Consistency"
  - "PET Reconstruction"
  - "Deep Image Prior"
date: 2026-05-08
content_hash: 0693557abf0b07ab
---

# Distributional Consistency Loss: Beyond Pointwise Data Terms in Inverse Problems

**Conference**: ICLR 2026
**arXiv**: [2510.13972](https://arxiv.org/abs/2510.13972)  
**Code**: [Available](https://github.com/GeorgeWebber/Distributional-Consistency-Loss)  
**Area**: Medical Imaging
**Keywords**: Inverse Problems, Data Fidelity, Distributional Consistency, PET Reconstruction, Deep Image Prior

## TL;DR

This paper proposes the Distributional Consistency (DC) loss, which replaces conventional pointwise data fidelity terms (e.g., MSE/NLL) with distribution-level calibration, thereby eliminating overfitting to noise. The approach achieves significant performance gains in DIP-based denoising and PET image reconstruction without requiring early stopping.

## Background & Motivation

The central challenge in inverse problems (medical imaging, geophysics, signal processing, etc.) is recovering the true signal from noisy measurements. Conventional methods decompose the objective into a **data fidelity term** and a **regularization term**. Data fidelity terms such as MSE and negative log-likelihood (NLL) measure pointwise discrepancies between predictions and noisy measurements, giving rise to a fundamental issue: **the optimization objective encourages the model to match individual noise realizations rather than ensuring statistical consistency between measurements and the model**.

Under noisy realizations, the true signal is not the minimizer of pointwise data terms. Regularization is thus forced to simultaneously suppress noise fitting and impose structural priors—two conflicting objectives. Although early stopping and bias criteria can alleviate this tension, they require explicit hyperparameter tuning and do not modify the objective function itself.

**Core Motivation**: Can a data fidelity term be designed that fundamentally eliminates the incentive to fit noise, allowing regularization to focus exclusively on structural constraints?

## Method

### Overall Architecture

The DC loss is grounded in the **Probability Integral Transform (PIT)**: if a model is correctly calibrated, the quantile of each measurement within its predicted noise distribution should follow a uniform distribution. The DC loss serves as a data fidelity term by measuring the deviation of these empirical quantile values from uniformity.

**Intuition for three fitting regimes**:

- **Underfitting**: Most measurements fall in the tails of the predicted distribution; the quantile histogram peaks near 0 or 1.
- **Well-calibrated**: The quantile histogram is approximately uniform.
- **Overfitting**: Measurements concentrate near the center of the predicted distribution; the histogram exhibits a sharp peak near 0.5.

### Key Designs

**Step 1 – CDF Evaluation**: For each measurement $m_i$ and predicted noise distribution $\mathcal{D}_i(\hat{y}_i)$, compute the cumulative probability:

$$s_i = F_i(m_i | \hat{y}_i) = \mathbb{P}_{c \sim \mathcal{D}_i(\hat{y}_i)}(c \leq m_i)$$

**Step 2 – Logit Transformation**: Directly matching $s_i$ to a uniform distribution causes gradient vanishing (as $s_i$ saturates near 0 or 1 far from the solution). A logit transformation is therefore applied:

$$r_i = \text{logit}(s_i) = \ln\frac{s_i}{1-s_i}$$

This maps the uniform target to a Logistic(0,1) distribution while preserving gradient sensitivity.

**Step 3 – Wasserstein-1 Distance**: Given sorted values $r_i$ and reference samples $u_i$ from Logistic(0,1):

$$\mathcal{L}_{\text{DC}}(\hat{\boldsymbol{\theta}}) = \frac{1}{N}\sum_{i=1}^{N}|r_i - u_i|$$

**Behavior far from the solution**: Via a Gaussian tail approximation, the gradient of the DC loss aligns with that of MSE (i.e., $\partial r_i / \partial \hat{y}_i \approx -(m_i - \hat{y}_i)/\sigma^2$), ensuring convergence.

**Behavior near the solution**: The DC loss defines an **equivalence class**—all predictions whose CDF values are approximately uniform attain low loss, forming a manifold near the MLE. Regularization then selects the optimal solution within this manifold, rather than trading off fidelity against regularization.

### Loss & Training

The DC loss serves as a **plug-and-play replacement** for conventional data fidelity terms:

- Compatible with unsupervised regularization methods that require no paired data.
- Employs the same optimization procedure as traditional losses.
- Avoids noise overfitting without early stopping.
- Applicable when: the noise distribution is known and a large number of independent measurements are available.

The synergy with regularization is the key advantage: under MSE, regularization and data fidelity are in opposition; under DC loss, regularization focuses exclusively on structural selection.

## Key Experimental Results

### Main Results

**Experiment 1: DIP Denoising (Gaussian Noise)**

| Method | Early Stopping Required | Peak PSNR (σ=75/255) | Long-term Stability |
|--------|:---:|:---:|:---:|
| DIP-MSE | Yes | Lower | Degrades after 1,000 iterations |
| DIP-DC | **No** | **Higher** | **Stable beyond 10,000 iterations** |

DIP-DC outperforms **optimally early-stopped** DIP-MSE across all noise levels, with larger gains at higher noise.

**Experiment 2: PET Image Reconstruction (Poisson Noise)**

| Method | Performance at 10,000 Iterations | Noise Artifacts | Early Stopping Required |
|--------|:---:|:---:|:---:|
| NLL-Adam | Severe degradation | Heavy noise spikes | Yes |
| MLEM | Gradual degradation | Progressively accumulates | Yes |
| DC-Adam | **Stable after convergence** | **Minimal** | **No** |

**Experiment 3: DC+TV vs. NLL+TV Regularization**

| Metric | NLL+TV | DC+TV |
|--------|:---:|:---:|
| Optimal NRMSE | Higher | **Lower** |
| Optimal β magnitude | Large | **Orders of magnitude smaller** |
| Image detail | Over-smoothed | **Detail preserved** |

### Ablation Study

- **Noise model misspecification**: The DC loss remains robust under biased estimates of noise variance.
- **Effect of overparameterization**: The advantage of DC loss increases with the degree of overparameterization.
- **Real 3D PET brain data**: DC-Adam demonstrates stable behavior on data acquired from a Siemens clinical scanner.

### Key Findings

1. Far from the solution, the DC loss provides the same convergence direction as MSE/NLL; near the solution, it automatically ceases to chase noise.
2. The optimal regularization strength for DC+TV is orders of magnitude smaller than for NLL+TV, as DC inherently suppresses noise.
3. Practical feasibility is validated on real clinical PET data.

## Highlights & Insights

- **Paradigm shift in data fidelity**: Transitioning from "pointwise measurement matching" to "distribution-level calibration consistency" represents a foundational innovation in the inverse problems literature.
- **Redefined role of regularization**: DC loss enables regularization to focus on structural selection rather than simultaneously resisting noise.
- **Theoretical elegance**: Each component—PIT, logit transformation, and Wasserstein distance—is motivated by clear and principled reasoning.
- **Strong practicality**: No modifications to network architecture or optimization pipeline are required; the approach is a true drop-in replacement.

## Limitations & Future Work

- Assumes independent measurements and a known noise distribution; not directly applicable to small-data or unknown-noise settings.
- Discrete noise distributions (e.g., Poisson) require randomized PIT.
- Does not enforce structural properties (e.g., sparsity); prior knowledge is still needed.
- Ill-conditioned forward operators fall outside the scope of the DC loss.
- Computational overhead is slightly higher than pointwise methods.
- Integration with score-based generative models remains largely unexplored and constitutes an important future direction.

## Related Work & Insights

- **vs. Robust losses (Huber/Student-t)**: The latter reduce the influence of outliers but do not prevent noise fitting.
- **vs. Noise2Noise**: N2N requires multiple noisy observations of the same signal; DC requires only a single observation but assumes a large number of independent measurements.
- **Connection to classical goodness-of-fit tests (K-S/CvM)**: DC loss can be viewed as a differentiable optimization counterpart to these tests.
- **Potential extensions**: Integration with plug-and-play priors and score-based generative models.

## Rating

| Dimension | Score |
|-----------|:---:|
| Novelty | ★★★★★ |
| Theoretical Depth | ★★★★☆ |
| Experimental Thoroughness | ★★★★☆ |
| Value | ★★★★★ |
| Writing Quality | ★★★★★ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Benchmarking Endoscopic Surgical Image Restoration and Beyond](../../CVPR2026/medical_imaging/benchmarking_endoscopic_surgical_image_restoration_and_beyond.md)
- [\[CVPR 2026\] CLoE: Expert Consistency Learning for Missing Modality Segmentation](../../CVPR2026/medical_imaging/cloe_expert_consistency_learning_for_missing_modality_segmentation.md)
- [\[CVPR 2026\] Solving a Nonlinear Blind Inverse Problem for Tagged MRI with Physics and Deep Generative Priors](../../CVPR2026/medical_imaging/solving_a_nonlinear_blind_inverse_problem_for_tagged_mri_with_physics_and_deep_g.md)
- [\[AAAI 2026\] Unsupervised Multi-Parameter Inverse Solving for Reducing Ring Artifacts in 3D X-Ray CBCT](../../AAAI2026/medical_imaging/unsupervised_multi-parameter_inverse_solving_for_reducing_ring_artifacts_in_3d_x.md)
- [\[ICCV 2025\] ViCTr: Vital Consistency Transfer for Pathology Aware Image Synthesis](../../ICCV2025/medical_imaging/victr_vital_consistency_transfer_for_pathology_aware_image_synthesis.md)

</div>

<!-- RELATED:END -->
