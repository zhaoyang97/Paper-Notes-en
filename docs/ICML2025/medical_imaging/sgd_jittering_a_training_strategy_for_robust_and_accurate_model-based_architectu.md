---
title: >-
  [Paper Note] SGD Jittering: A Training Strategy for Robust and Accurate Model-Based Architectures
description: >-
  [ICML2025][Medical Imaging][SGD jittering] Proposes the SGD jittering training strategy, which progressively injects zero-mean Gaussian noise during the iterative reconstruction process of the model. Theoretical analysis demonstrates that it simultaneously enhances model robustness and generalization accuracy without the high computational overhead of adversarial training.
tags:
  - "ICML2025"
  - "Medical Imaging"
  - "SGD jittering"
  - "model-based architecture"
  - "loop unrolling"
  - "robustness"
  - "generalization"
  - "inverse problems"
  - "MRI reconstruction"
date: 2026-05-08
content_hash: 42446c9c2a58db40
---

# SGD Jittering: A Training Strategy for Robust and Accurate Model-Based Architectures

**Conference**: ICML2025  
**arXiv**: [2410.14667](https://arxiv.org/abs/2410.14667)  
**Code**: To be confirmed  
**Area**: Model Robustness / Inverse Problems  
**Keywords**: SGD jittering, model-based architecture, loop unrolling, robustness, generalization, inverse problems, MRI reconstruction

## TL;DR

Proposes the SGD jittering training strategy, which progressively injects zero-mean Gaussian noise during the iterative reconstruction process of the model. Theoretical analysis demonstrates that it simultaneously enhances model robustness and generalization accuracy without the high computational overhead of adversarial training.

## Background & Motivation

Inverse problems aim to recover the original signal $\boldsymbol{x}$ from degraded/incomplete observations $\boldsymbol{y} = \boldsymbol{A}\boldsymbol{x} + \boldsymbol{z}$. Model-based architectures (MBAs), such as loop unrolling, unroll classical optimization iterations into trainable networks, offering better interpretability and higher reconstruction quality than black-box methods.

However, existing research focuses primarily on reconstruction accuracy, paying insufficient attention to robustness and generalization:

- **Robustness**: The stability of the model against measurement noise or adversarial attacks.
- **Generalization**: The performance of the model on out-of-distribution (OOD) data.
- **Trade-off between the two**: Adversarial training (AT) improves robustness at the expense of accuracy, and input jittering suffers from a similar issue.

The authors observe that both AT and input jittering inject noise into the observations $\boldsymbol{y}$, causing the learned inverse mapping to deviate from the true inverse process. For instance, in the noiseless scenario, AT learns $H_\theta(\boldsymbol{y}+\boldsymbol{e}) \approx \boldsymbol{x}$, whereas the true solution is $\boldsymbol{A}^{-1}(\boldsymbol{y}+\boldsymbol{e}) = \boldsymbol{x} + \boldsymbol{A}^{-1}\boldsymbol{e}$. Ignoring the latter in AT introduces generalization errors.

## Method

### Core Idea: SGD Jittering

Instead of injecting noise into the input $\boldsymbol{y}$, SGD jittering injects independent, zero-mean Gaussian noise $\boldsymbol{w}_k$ at each gradient update step during the iterative reconstruction of the MBA:

$$\boldsymbol{x}_{k+1} = \boldsymbol{x}_k - \eta \left( \boldsymbol{A}^\top(\boldsymbol{A}\boldsymbol{x}_k - \boldsymbol{y}) + f_\theta(\boldsymbol{x}_k) + \boldsymbol{w}_k \right)$$

where $\boldsymbol{w}_k \sim \mathcal{N}(0, \sigma_{w_k}^2/n \cdot \boldsymbol{I})$ is independently sampled at each iteration. During inference, the noise is removed, and the model runs in the standard manner.

### Training Loss

SGD jittering minimizes the following risk:

$$J^{SGD}_{\sigma_{w_k}}(\theta) = \mathbb{E}_{\bar{\boldsymbol{w}}, (\boldsymbol{x},\boldsymbol{y})\sim\mathcal{D}} \|\boldsymbol{x} - \hat{\boldsymbol{x}}\|_2^2$$

The key difference is that the underlying optimization objective is fully consistent with standard MSE training; noise only affects the optimization path rather than the objective function itself.

### Implicit Regularization

The training loss of SGD jittering implicitly includes a regularization term:

$$\text{regularization} = \mathbb{E}\left\|\sum_{i=0}^{K-1} \eta(1-\eta)^{K-1-i} \left(f_\theta(\boldsymbol{x}_i') - f_\theta(\boldsymbol{x}_i^{sgd})\right)\right\|_2^2$$

This regularization penalizes the intermediate reconstruction deviations caused by noise, encouraging the network $f_\theta$ to have a flatter Hessian in the input space, thereby enhancing stability.

### Theoretical Guarantees

- **Generalization** (Theorem 7.4): $\mathcal{G}(\theta^{sgd}) \leq \mathcal{G}(\theta^{mse})$, indicating that the path-based generalization risk of SGD jittering does not exceed that of standard MSE training.
- **Robustness** (Theorem 7.5): $R_e(\theta^{sgd}) \leq R_e(\theta^{mse})$, showing that SGD jittering is more robust against average-case attacks.
- **Convergence** (Corollary 6.2): MBA-SGD converges to a stationary point at a rate of $O(1/\sqrt{K})$.

### Comparison with Other Methods

| Method | Noise Position | Inference Noise | Generalization | Robustness | Training Speed |
|------|----------|-----------|------|--------|---------|
| MSE Training | None | None | Baseline | Baseline | Fast |
| Adversarial Training (AT) | Input $\boldsymbol{y}$ | None | Poor | Optimal (worst-case) | Slow |
| Input Jittering | Input $\boldsymbol{y}$ | None | Poor | Moderate | Fast |
| **SGD Jittering** | **Iterative Gradient** | **None** | **Optimal** | **Moderate-to-high** | **Fast** |

## Key Experimental Results

### Seismic Deconvolution

| Method | ID Data (PSNR/SSIM) | Adversarial Attack (PSNR/SSIM) | OOD Data (PSNR/SSIM) |
|------|---------------------|---------------------|---------------------|
| MSE Training | 34.64 / 0.921 | 28.85 / 0.829 | 34.57 / 0.918 |
| AT | 33.50 / 0.903 | **30.27 / 0.849** | 33.45 / 0.902 |
| Input Jittering | 32.92 / 0.882 | 30.10 / 0.832 | 32.91 / 0.882 |
| **SGD Jittering** | **35.10 / 0.928** | 29.89 / 0.842 | **34.93 / 0.927** |

### Accelerated MRI Reconstruction (4× Acceleration)

| Method | fastMRI (PSNR/SSIM) | Adversarial Attack (PSNR/SSIM) | Tumor OOD (PSNR/SSIM) |
|------|---------------------|---------------------|---------------------|
| MSE Training | 28.21 / 0.603 | 25.68 / 0.382 | 29.92 / 0.779 |
| AT | 27.68 / 0.564 | **27.17 / 0.549** | 27.74 / 0.597 |
| Input Jittering | 28.18 / 0.595 | 25.05 / 0.420 | 29.97 / 0.740 |
| **SGD Jittering** | **28.22 / 0.607** | 26.77 / 0.552 | **30.36 / 0.788** |

**Key Findings**: SGD jittering achieves the best performance on both ID and OOD data. Under adversarial attacks, its performance is second only to AT and significantly outperforms MSE and input jittering. On OOD tumor data, the PSNR of AT is 2.62 dB lower than that of SGD jittering, demonstrating that AT severely sacrifices generalization accuracy.

## Highlights & Insights

1. **Simple Yet Effective**: Noise is only injected during iterative training, introducing zero inference overhead and requiring only a few lines of code to implement.
2. **Theoretical Integrity**: Provides the first formal definition and theoretical analysis of generalization accuracy in inverse problems, filling the theoretical gap from hidden layer input flatness to generalization.
3. **Deep Insight on Noise Injection Location**: Input-level noise (AT/input jittering) alters the optimization objective, leading to a loss of generalization, whereas iteration-level noise keeps the objective intact and only affects the solver path.
4. **Value in Medical Imaging**: The significant advantage on OOD tumor MRI data is crucial for clinical safety.
5. **SPGD Extension**: Further proposes a Stochastic Proximal Gradient Descent (SPGD) variant, which is applicable to proximal algorithms.

## Limitations & Future Work

1. **Theory Limited to Denoising**: The proofs of Theorem 7.4 and 7.5 assume the forward model is an identity mapping (denoising); the theoretical framework for general inverse problems still needs to be expanded.
2. **Analyzes Only Average Attacks**: Theoretical guarantees cover only average-case robustness, lacking theoretical analysis for worst-case attacks.
3. **Noise Variance Selection**: Although experiments demonstrate the impact of noise levels (Section 8.2), an automatic selection strategy is lacking.
4. **Limited Experimental Scale**: MRI experiments only utilize single-coil 4× acceleration, without validation on multi-coil or higher acceleration scenarios.
5. **Lack of Comparison with Latest Methods**: Recent paradigms such as diffusion models are not included in the comparison.

## Related Work & Insights

- **Krainovic et al. (2023)**: Input jittering applied to black-box networks achieves robustness comparable to AT but with a significant drop in accuracy.
- **Lim et al. (2021)**: Layer-wise noise injection improves classifier generalization but does not explain how the flatness of hidden-layer inputs improves regression.
- **Foret et al. (2021)**: SAM enhances generalization via sharpness-aware optimization but focuses on the parameter space rather than the iterative space.
- **Insight**: The concept of "injecting noise along the solver path rather than onto the objective" can be extended to other iterative architectures, such as the reverse process of diffusion models.

## Rating

- Novelty: ⭐⭐⭐⭐ — A simple yet profound new perspective on noise injection locations.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three tasks with multi-dimensional evaluations, though the MRI experimental scale is somewhat small.
- Writing Quality: ⭐⭐⭐⭐ — Clear integration of theory and experiments, with consistent notation.
- Value: ⭐⭐⭐⭐ — Holds practical significance for safety-critical inverse problems, such as medical imaging.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DiN: Diffusion Model for Robust Medical VQA with Semantic Noisy Labels](../../CVPR2025/medical_imaging/din_diffusion_model_for_robust_medical_vqa_with_semantic_noisy_labels.md)
- [\[CVPR 2025\] Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](../../CVPR2025/medical_imaging/multi-resolution_pathology-language_pre-training_model_with_text-guided_visual_r.md)
- [\[ICLR 2026\] Cross-Timestep: 3D Diffusion Model with Trans-temporal Memory LSTM and Adaptive Priori Decoding Strategy for Medical Segmentation](../../ICLR2026/medical_imaging/cross-timestep_3d_diffusion_model_with_trans-temporal_memory_lstm_and_adaptive_p.md)
- [\[ICML 2025\] Certification for Differentially Private Prediction in Gradient-Based Training](certification_for_differentially_private_prediction_in_gradient-based_training.md)
- [\[ICLR 2026\] SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding](../../ICLR2026/medical_imaging/seed_towards_more_accurate_semantic_evaluation_for_visual_brain_decoding.md)

</div>

<!-- RELATED:END -->
