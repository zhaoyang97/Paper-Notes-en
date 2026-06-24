---
title: >-
  [Paper Note] Unsupervised Multi-Parameter Inverse Solving for Reducing Ring Artifacts in 3D X-Ray CBCT
description: >-
  [AAAI 2026][Medical Imaging][CT ring artifact removal] This paper proposes Riner, which formulates CT ring artifact removal (RAR) as a physics-based multi-parameter inverse problem. By jointly learning artifact-free images and detector physical parameters via implicit neural representation (INR), Riner achieves unsupervised 3D CBCT reconstruction that surpasses supervised state-of-the-art methods.
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "CT ring artifact removal"
  - "implicit neural representation"
  - "unsupervised learning"
  - "multi-parameter inverse problem"
  - "CBCT reconstruction"
date: 2026-05-08
content_hash: 2833bf0735eeb2c3
---

# Unsupervised Multi-Parameter Inverse Solving for Reducing Ring Artifacts in 3D X-Ray CBCT

**Conference**: AAAI 2026
**arXiv**: [2412.05853](https://arxiv.org/abs/2412.05853)  
**Code**: [https://github.com/iwuqing/Riner](https://github.com/iwuqing/Riner)  
**Area**: Medical Imaging
**Keywords**: CT ring artifact removal, implicit neural representation, unsupervised learning, multi-parameter inverse problem, CBCT reconstruction

## TL;DR

This paper proposes Riner, which formulates CT ring artifact removal (RAR) as a physics-based multi-parameter inverse problem. By jointly learning artifact-free images and detector physical parameters via implicit neural representation (INR), Riner achieves unsupervised 3D CBCT reconstruction that surpasses supervised state-of-the-art methods.

## Background & Motivation

1. **Background**: 3D cone-beam computed tomography (CBCT) is widely used in medical diagnosis, biological research, and materials science. Due to non-ideal responses of X-ray detectors, reconstructed images frequently exhibit severe ring artifacts (RAs), which significantly degrade image quality and diagnostic reliability.

2. **Limitations of Prior Work**: Current supervised deep learning SOTA methods (e.g., DeepRAR, Restormer) require large-scale paired datasets for training, but suffer from two critical limitations: (1) data collection is costly and generalization is poor—models are typically trained on simulated data and experience substantial performance drops on real data; (2) poor scalability—2D models are difficult to extend directly to 3D CBCT, and slice-by-slice processing introduces Z-axis discontinuities.

3. **Key Challenge**: Supervised methods treat RAR as an end-to-end post-processing denoising task, lacking explicit modeling of the physical origins of artifacts, which leads to insufficient out-of-domain generalization.

4. **Goal**: To simultaneously recover high-quality images and estimate detector physical parameters from raw CT measurements, without relying on external training data.

5. **Key Insight**: Starting from the physics of X-ray CT, the paper theoretically analyzes two physical causes of ring artifacts—inconsistent response (IR) and invalid measurements (IM)—and reformulates RAR as a multi-parameter inverse problem.

6. **Core Idea**: The non-ideal behavior of detectors is parameterized via a differentiable physical forward model. The spectral bias prior of INR regularizes the ill-posed inverse problem, enabling RAR without any training data.

## Method

### Overall Architecture

Riner adopts a ray-based optimization pipeline: given raw measurement data $\widetilde{\rho}(\theta,s)$, an MLP network $f_\Phi$ takes spatial coordinates $\mathbf{x}$ along X-ray paths as input and predicts the corresponding CT intensity $\mu(\mathbf{x})$. The predicted CT intensities, together with learnable response factors $\alpha_s$ and masks $\beta_s$, are passed through a differentiable physical forward model to generate estimated measurements $\widehat{\rho}(\theta,s)$. All parameters are then jointly optimized by minimizing the loss function.

### Key Designs

1. **Theoretical Modeling of Ring Artifacts**:
    - **Function**: Theoretically analyzes the physical origins of ring artifacts from the Lambert-Beer law.
    - **Mechanism**: Non-ideal detectors ($\alpha_s \neq 1$) introduce an additional nonlinear term $-\ln\alpha_s$ into the measurement data (IR effect); defective detectors ($\alpha_s = 0$) produce invalid measurements (IM effect). These two physical distortions make the inverse problem of reconstructing CT images from real measurements inherently nonlinear.
    - **Design Motivation**: Traditional linear algorithms (e.g., FDK) assume ideal detectors and cannot handle these nonlinear distortions.

2. **Differentiable Physical Forward Model**:
    - **Function**: Converts MLP-predicted CT intensities and physical parameters into estimated measurements while supporting gradient backpropagation.
    - **Mechanism**: The forward model is defined as $\widehat{\rho}(\theta,s) = [-\ln\alpha_s + \sum_{\mathbf{x}\in L(\theta,s)}\mu(\mathbf{x})\cdot\Delta\mathbf{x}]\cdot\beta_s$, where $\alpha_s = \max(\alpha_s^0, \epsilon)$ ensures non-negativity, and $\beta_s = \sigma(\beta_s^0)$ acts as a binary mask to suppress invalid signals from defective detectors.
    - **Design Motivation**: Compared to traditional linear integral models, incorporating additional physical parameters $(\alpha_s, \beta_s)$ enables more accurate acquisition modeling.

3. **INR-Based Image Parameterization**:
    - **Function**: Represents the continuous CT image function as $f_\Phi: \mathbf{x} \to \mu(\mathbf{x})$ using an MLP network.
    - **Mechanism**: A lightweight MLP with hash encoding and 2 fully connected layers is used. The inherent spectral bias of INR (learning low-frequency global structures before recovering high-frequency details) serves as a regularization constraint, mitigating the ill-posedness of the inverse problem introduced by the additional physical parameters.
    - **Design Motivation**: The learned prior of INR naturally constrains the solution space, enabling high-quality reconstruction without external data.

### Loss & Training

The loss function consists of two terms:
$$\mathcal{L} = \underbrace{\sum_{L(\theta,s)\in\bar{\Pi}}\|\widehat{\rho}(\theta,s) - \widetilde{\rho}(\theta,s)\cdot\beta_s\|_1}_{\text{Data Consistency}} + \underbrace{\lambda\cdot\sum_{s\in\bar{S}}-\beta_s^2}_{\text{Negative }\ell_2}$$

- **Data Consistency Term**: Minimizes the L1 distance between predicted and actual measurements.
- **Negative $\ell_2$ Regularization**: Prevents all masks from collapsing to zero (i.e., avoids labeling all detectors as defective); $\lambda=0.01$.

Training strategy: 80 X-rays are randomly sampled per step (from 2 detectors and 40 projection views). The Adam optimizer is used with a learning rate of $10^{-3}$ for 4,000 iterations. The method requires only ~3 GB of GPU memory (NVIDIA RTX 4090).

## Key Experimental Results

### Main Results

| Dataset | Metric | Riner (Ours) | Restormer (Prev. SOTA) | Gain |
|--------|------|------|----------|------|
| DeepLesion (2D) | PSNR | 39.02±2.18 | 37.31±1.92 | +1.71 dB |
| LIDC (2D) | PSNR | 39.11±2.17 | 36.69±1.90 | +2.42 dB |
| AAPM (3D) | PSNR | 36.53±1.28 | 33.94±1.50 | +2.59 dB |
| DeepLesion | SSIM | 0.967±0.019 | 0.947±0.028 | +0.020 |
| LIDC | SSIM | 0.962±0.030 | 0.925±0.051 | +0.037 |
| AAPM | SSIM | 0.949±0.020 | 0.907±0.031 | +0.042 |

### Ablation Study

| Configuration | PSNR | Note |
|------|---------|------|
| w/o β, α (integral model) | 35.34±2.26 | Cannot handle IR or IM effects |
| w/o β (α only) | 31.05±9.16 | Cannot handle IM effect; extremely high variance |
| w/o α (β only) | 35.72±1.93 | Cannot handle IR effect |
| Full model | 38.98±1.41 | Both effects effectively eliminated |
| λ=0 (no regularization) | 31.63±1.77 | β collapses to 0; optimization degenerates |
| λ=0.01 (default) | 38.98±1.41 | Best balance |
| λ=1 (strong regularization) | 31.55±8.97 | β converges to 1; defective detectors undetectable |

### Key Findings

- Unsupervised Riner is the first method to comprehensively outperform supervised approaches on the RAR task, achieving the best performance across all three simulated datasets.
- The advantage is most pronounced on 3D CBCT data (+2.59 dB), where supervised methods suffer significant performance degradation in 3D settings.
- Estimated physical parameters $(\alpha, \beta)$ show high agreement with ground truth.
- The method also demonstrates superior artifact removal on real-world micro-CT data.

## Highlights & Insights

- **The power of physics-based modeling**: Explicitly incorporating domain knowledge (CT physics) into an unsupervised framework eliminates the dependency on large-scale paired training data.
- **Memory efficiency**: The ray-based optimization strategy makes the method naturally scalable to large-scale 3D CBCT, as each iteration only requires forward/backward passes over a small number of sampled voxels.
- **Consistency among theory, method, and experiments**: Ablation results are fully consistent with the theoretical predictions regarding IR/IM effects.
- The spectral bias of INR serves as "free" regularization, elegantly addressing the ill-posedness of the multi-parameter inverse problem.

## Limitations & Future Work

- As an unsupervised method, per-case optimization is required: ~30 seconds for a 2D slice and ~10 minutes for a 3D volume (256×256×100).
- Advanced representations such as 3D Gaussian Splatting could potentially replace INR to improve efficiency.
- Robustness to varying artifact severity levels is not discussed.
- Quantitative evaluation on real data is not feasible due to the absence of ground truth.

## Related Work & Insights

- **vs. DeepRAR/Restormer (supervised)**: Supervised methods rely on simulated training data and suffer from poor out-of-domain generalization; Riner directly infers from raw measurements without external data.
- **vs. NeRF/NAF and other INR methods**: Existing INR-based CT methods assume ideal detectors and do not address nonlinear physical distortions; Riner extends the framework by introducing explicit physical parameters.
- **vs. Super (traditional model-based)**: Super uses hand-crafted algorithms to handle multiple artifact types, but performance is limited by insufficient priors; Riner achieves superior reconstruction through end-to-end optimization.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The first unsupervised method to formulate RAR as a multi-parameter physical inverse problem, supported by rigorous theoretical analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on 5 datasets (3 simulated + 2 real), with 10+ baseline comparisons and multi-dimensional ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logic flows clearly from theoretical analysis to method design to experimental validation, with rigorous mathematical derivations.
- **Value**: ⭐⭐⭐⭐⭐ Unsupervised methods surpassing supervised counterparts carries significant implications in the CT domain, with strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Unsupervised Motion-Compensated Decomposition for Cardiac MRI Reconstruction via Neural Representation](unsupervised_motion-compensated_decomposition_for_cardiac_mri_reconstruction_via.md)
- [\[CVPR 2026\] Solving a Nonlinear Blind Inverse Problem for Tagged MRI with Physics and Deep Generative Priors](../../CVPR2026/medical_imaging/solving_a_nonlinear_blind_inverse_problem_for_tagged_mri_with_physics_and_deep_g.md)
- [\[CVPR 2026\] Efficient Unrolled Networks for Large-Scale 3D Inverse Problems](../../CVPR2026/medical_imaging/efficient_unrolled_networks_for_large-scale_3d_inverse_problems.md)
- [\[AAAI 2026\] Fine-Tuned LLMs Know They Don't Know: A Parameter-Efficient Approach to Recovering Honesty](fine-tuned_llms_know_they_dont_know_a_parameter-efficient_approach_to_recovering.md)
- [\[ECCV 2024\] Unsupervised Multi-modal Medical Image Registration via Invertible Translation](../../ECCV2024/medical_imaging/unsupervised_multi-modal_medical_image_registration_via_invertible_translation.md)

</div>

<!-- RELATED:END -->
