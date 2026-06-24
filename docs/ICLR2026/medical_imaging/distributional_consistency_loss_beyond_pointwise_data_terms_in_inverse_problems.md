---
title: >-
  [Paper Note] Distributional Consistency Loss: Beyond Pointwise Data Terms in Inverse Problems
description: >-
  [ICLR 2026][Medical Imaging][Inverse Problems] The authors propose the Distributional Consistency (DC) loss, which replaces traditional pointwise data fidelity terms (e.g., MSE/NLL) with distribution-level calibration. This approach avoids overfitting to noise, significantly improving performance in DIP denoising and PET image reconstruction without requiring early stopping.
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Inverse Problems"
  - "Data Fidelity Term"
  - "Distributional Consistency"
  - "PET Reconstruction"
  - "Deep Image Prior"
date: 2026-05-08
content_hash: 482b3130b2bb5938
---

# Distributional Consistency Loss: Beyond Pointwise Data Terms in Inverse Problems

**Conference**: ICLR 2026  
**arXiv**: [2510.13972](https://arxiv.org/abs/2510.13972)  
**Code**: [Available](https://github.com/GeorgeWebber/Distributional-Consistency-Loss)  
**Area**: Medical Imaging  
**Keywords**: Inverse Problems, Data Fidelity Term, Distributional Consistency, PET Reconstruction, Deep Image Prior

## TL;DR

The authors propose the Distributional Consistency (DC) loss, which replaces traditional pointwise data fidelity terms (e.g., MSE/NLL) with distribution-level calibration. This approach avoids overfitting to noise, significantly improving performance in DIP denoising and PET image reconstruction without requiring early stopping.

## Background & Motivation

A core challenge in inverse problems (medical imaging, geophysics, signal processing, etc.) is recovering the true signal from noisy measurements. Traditional methods decompose the objective function into a **data fidelity term** and a **regularization term**. Data fidelity terms (such as MSE or negative log-likelihood, NLL) measure the discrepancy between predictions and noisy measurements pointwise. This leads to a fundamental issue: **the optimization objective encourages the model to match individual noise realizations rather than ensuring the measurements are statistically consistent with the model.**

Under noisy realizations, the true signal is not the minimizer of the pointwise data term. Consequently, regularization must simultaneously "suppress noise fitting" and "impose structural priors," two tasks that are often in conflict. While early stopping or discrepancy principles can mitigate this, they require explicit hyperparameter tuning and do not change the objective function itself.

**Key Motivation**: Can a data fidelity term be designed to fundamentally eliminate the incentive to fit noise, allowing regularization to focus solely on structural constraints?

## Method

### Overall Architecture

The DC loss shifts the data fidelity term from "comparing measurements and predictions pointwise" to "testing whether a batch of measurements is statistically consistent with the model." The theoretical foundation is the **Probability Integral Transform (PIT)**: if the noise distribution predicted by the model is correct, the cumulative probability of each measurement falling within its predicted distribution should follow a uniform distribution. By measuring the degree to which the empirical distribution of these cumulative probabilities deviates from uniformity, the quality of the model can be assessed without pointwise noise fitting. Underfitting causes measurements to cluster at the ends of the predicted distribution (histograms peaking near 0 or 1), while overfitting causes clustering at the center (peaking at 0.5); the histogram is flat only when correctly calibrated.

### Key Designs

**1. PIT Calibration instead of Pointwise Comparison: Functionally eliminating noise-fitting incentives**

The minimum of pointwise data terms (MSE, NLL) coincides exactly with the noisy measurement, causing models to fit noise as optimization progresses. In contrast, DC calculates the cumulative probability $s_i = F_i(m_i \mid \hat{y}_i) = \mathbb{P}_{c \sim \mathcal{D}_i(\hat{y}_i)}(c \leq m_i)$ for each measurement $m_i$ and its predicted noise distribution $\mathcal{D}_i(\hat{y}_i)$, then requires the set $\{s_i\}$ to follow a uniform distribution. Thus, the optimal solution is no longer a single noise realization but an **equivalence class**—all predictions that make $\{s_i\}$ approximately uniform achieve low loss. These form a manifold near the maximum likelihood solution, allowing regularization to select the most structurally reasonable solution within this manifold without competing with the data fidelity term.

**2. Logit Transform to Prevent Gradient Vanishing: Maintaining descent signals far from the solution**

Matching $s_i$ directly to a uniform distribution poses a risk: when predictions are far from the true value, $s_i$ saturates at 0 or 1, leading to vanishing gradients. To address this, cumulative probabilities are stretched to the entire real axis using a logit transform $r_i = \mathrm{logit}(s_i) = \ln\frac{s_i}{1-s_i}$, making the target distribution Logistic(0,1) instead of Uniform(0,1). Even in the distribution tails, gradients do not vanish—approximations using Gaussian tails show $\partial r_i / \partial \hat{y}_i \approx -(m_i - \hat{y}_i)/\sigma^2$, which aligns with the descent direction of MSE. This ensures DC follows the same convergence path as traditional losses when far from the solution.

**3. Wasserstein-1 Distance for Distributional Deviation: A differentiable target for uniformity**

With the transformed $\{r_i\}$, the remaining task is to quantify its distance from Logistic(0,1). The approach involves sorting $r_i$ and taking an equal number of reference samples $u_i$ from Logistic(0,1) (also sorted), then using the 1D Wasserstein-1 distance as the loss:

$$\mathcal{L}_{\text{DC}}(\hat{\boldsymbol{\theta}}) = \frac{1}{N}\sum_{i=1}^{N}\lvert r_i - u_i \rvert$$

The sorted pointwise absolute difference is the closed-form solution for 1D optimal transport, which is differentiable, robust to sample size, and equivalent to a differentiable goodness-of-fit test.

### Loss & Training

DC loss is a **plug-and-play replacement** for traditional data fidelity terms. It requires no changes to network architectures, optimizers (like Adam), or unsupervised regularization terms; one simply replaces MSE/NLL with $\mathcal{L}_{\text{DC}}$. It is compatible with unsupervised methods lacking paired data. The primary benefit lies in training behavior: since the optimal solution is an equivalence class rather than a point noise realization, the model stabilizes upon reaching calibration and stops chasing noise. This means **early stopping is unnecessary**, effectively decoupling "noise suppression" from the responsibilities of regularization. The trade-off is the requirement for a known noise distribution and sufficient independent measurements for the PIT statistics to hold.

## Key Experimental Results

### Main Results

**Experiment 1: DIP Denoising (Gaussian Noise)**

| Method | Early Stopping Needed | Peak PSNR (σ=75/255) | Long-term Stability |
|------|:---:|:---:|:---:|
| DIP-MSE | Yes | Lower | Degenerates after 1000 iter |
| DIP-DC | **No** | **Higher** | Stable at 10000 iter |

DIP-DC outperforms **optimally stopped** DIP-MSE across all noise levels, with greater advantages at higher noise levels.

**Experiment 2: PET Image Reconstruction (Poisson Noise)**

| Method | Performance at 10k Iterations | Noise Artifacts | Early Stopping Needed |
|------|:---:|:---:|:---:|
| NLL-Adam | Severe degradation | Many noise spikes | Yes |
| MLEM | Gradual degradation | Gradual accumulation | Yes |
| DC-Adam | **Stable after convergence** | **Minimal** | **No** |

**Experiment 3: DC+TV Regularization vs. NLL+TV**

| Metric | NLL+TV | DC+TV |
|------|:---:|:---:|
| Optimal NRMSE | Higher | **Lower** |
| Optimal β Magnitude | Large | **Orders of magnitude smaller** |
| Image Detail | Over-smoothed | **Preserved details** |

### Ablation Study

- Noise Model Mis-specification: DC loss remains robust even with biased noise variance estimates.
- Impact of Over-parameterization: The advantages of DC loss become more pronounced as over-parameterization increases.
- Real 3D PET Brain Data Validation: DC-Adam exhibits stable behavior on data from Siemens clinical scanners.

### Key Findings

1. DC loss provides the same convergence direction as MSE/NLL when far from the solution but automatically segments noise chasing as it approaches calibration.
2. The optimal regularization strength for DC+TV is orders of magnitude smaller than for NLL+TV, as DC possesses inherent noise suppression.
3. Practical feasibility was demonstrated on real clinical PET data.

## Highlights & Insights

- **Paradigm Shift in Data Fidelity**: Moving from "pointwise matching" to "distributional calibration consistency" represents a foundational innovation in inverse problems.
- **Redefining the Role of Regularization**: DC allows regularization to focus on structure rather than simultaneously resisting noise.
- **Theoretical Elegance**: PIT + logit transform + Wasserstein distance; each step is clearly motivated.
- **High Practicality**: A genuine drop-in replacement requiring no changes to network structures or optimization workflows.

## Limitations & Future Work

- Assumes independent measurements and a known noise distribution; not applicable to extremely small datasets or unknown noise scenarios.
- Requires randomization of PIT for discrete noise (e.g., Poisson).
- Does not guarantee structural properties (e.g., sparsity) on its own; still requires priors.
- Does not address the ill-posedness of the forward operator itself.
- Slightly higher computational overhead compared to pointwise methods.
- Integration with score-based generative models remains an important future direction.

## Related Work & Insights

- Difference from robust losses (Huber/Student-t): The latter reduce the impact of outliers but do not prevent noise fitting.
- Difference from Noise2Noise: N2N requires multiple noisy observations, whereas DC requires only one but necessitates many independent measurements.
- Connection to classic Goodness-of-fit tests (K-S/CvM): DC can be viewed as a differentiable version for optimization.
- Potential extensions: Combination with plug-and-play priors and score-based generative models.

## Rating

| Dimension | Score |
|------|:---:|
| Novelty | ★★★★★ |
| Theoretical Depth | ★★★★☆ |
| Experimental Thoroughness | ★★★★☆ |
| Value | ★★★★★ |
| Writing Quality | ★★★★★ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Efficient Unrolled Networks for Large-Scale 3D Inverse Problems](../../CVPR2026/medical_imaging/efficient_unrolled_networks_for_large-scale_3d_inverse_problems.md)
- [\[CVPR 2026\] KLIP: localized distribution shift detection via KL-divergence with diffusion priors in Inverse Problems](../../CVPR2026/medical_imaging/klip_localized_distribution_shift_detection_via_kl-divergence_with_diffusion_pri.md)
- [\[ICLR 2026\] Towards Text–Mask Consistency in Medical Image Segmentation](towards_text-mask_consistency_in_medical_image_segmentation.md)
- [\[ICLR 2026\] Moving Beyond Diffusion: Hierarchy-to-Hierarchy Autoregression for fMRI-to-Image Reconstruction](moving_beyond_diffusion_hierarchy-to-hierarchy_autoregression_for_fmri-to-image_.md)
- [\[CVPR 2026\] Benchmarking Endoscopic Surgical Image Restoration and Beyond](../../CVPR2026/medical_imaging/benchmarking_endoscopic_surgical_image_restoration_and_beyond.md)

</div>

<!-- RELATED:END -->
