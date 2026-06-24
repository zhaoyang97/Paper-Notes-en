---
title: >-
  [Paper Note] Adaptive Selection of Sampling-Reconstruction in Fourier Compressed Sensing
description: >-
  [ECCV 2024][Model Compression][Fourier Compressed Sensing] This paper proposes the "adaptive selection of sampling-reconstruction pairs" ($\mathcal{H}_{1.5}$) framework. It leverages a super-resolution spatial generative model to quantify high-frequency Bayesian uncertainty and selects the optimal sampling mask-reconstruction network pair for each input data. Theoretically and experimentally, it outperforms both non-adaptive joint optimization ($\mathcal{H}_1$) and adaptive s…
tags:
  - "ECCV 2024"
  - "Model Compression"
  - "Fourier Compressed Sensing"
  - "Adaptive Sampling"
  - "Sampling-Reconstruction Selection"
  - "Bayesian Uncertainty"
  - "MRI Reconstruction"
date: 2026-05-08
content_hash: 7c2c6617001e6179
---

# Adaptive Selection of Sampling-Reconstruction in Fourier Compressed Sensing

**Conference**: ECCV 2024  
**arXiv**: [2409.11738](https://arxiv.org/abs/2409.11738)  
**Code**: [https://smhongok.github.io/ada-sel.html](https://smhongok.github.io/ada-sel.html) (project page included)  
**Area**: Model Compression / Compressed Sensing  
**Keywords**: Fourier Compressed Sensing, Adaptive Sampling, Sampling-Reconstruction Selection, Bayesian Uncertainty, MRI Reconstruction

## TL;DR
This paper proposes the "adaptive selection of sampling-reconstruction pairs" ($\mathcal{H}_{1.5}$) framework. It leverages a super-resolution spatial generative model to quantify high-frequency Bayesian uncertainty and selects the optimal sampling mask-reconstruction network pair for each input data. Theoretically and experimentally, it outperforms both non-adaptive joint optimization ($\mathcal{H}_1$) and adaptive sampling ($\mathcal{H}_2$), achieving significant SSIM improvements in face image and multi-coil MRI reconstruction.

## Background & Motivation
Fourier compressed sensing (Fourier CS) has crucial applications in electromagnetic imaging such as MRI and radar. The core challenge lies in choosing the optimal sampling pattern to achieve the best reconstruction with minimal measurements. Existing methods split into two categories: (1) Jointly optimized sampling-reconstruction ($\mathcal{H}_1$)—optimizes a fixed sampling mask and reconstruction network for the entire dataset, but cannot adapt to individual samples and requires difficult backpropagation optimization in discrete space; (2) Adaptive sampling ($\mathcal{H}_2$)—generates an optimal mask for each input, but suffers from easy-to-fail optimization of mask generators and the Pareto suboptimality of a single reconstruction network (it is impossible for a single network to be simultaneously optimal for all adaptively generated, varying masks). 

* **Key Challenge**: How to achieve data adaptivity while ensuring the Pareto optimality of the reconstruction network?
* **Key Insight**: Rather than generating infinitely many adaptive masks, predefine a finite set of $J$ sampling-reconstruction pairs and select the best-matching pair for each input.
* **Core Idea**: Utilize a super-resolution generative model to quantify the high-frequency uncertainty distribution of the input. Group the uncertainty patterns into $J$ classes via clustering, with each class corresponding to a dedicated mask-reconstructor pair.

## Method

### Overall Architecture
The $\mathcal{H}_{1.5}$ framework consists of training and inference phases:
1. **Training phase**: 
   - (1) Generate super-resolution samples for all training data using a generative model, and calculate the normalized high-frequency variance (uncertainty).
   - (2) Apply k-means++ clustering on the uncertainty vectors to obtain $J$ centroids.
   - (3) Generate corresponding sampling masks $M_j$ for each centroid using rejection sampling.
   - (4) Independently train a dedicated reconstruction network $\theta_j$ for each $M_j$.
2. **Inference phase**: 
   - (1) Acquire the low-frequency components of the input using the low-frequency mask $M_0$.
   - (2) Quantify high-frequency uncertainty using the super-resolution model.
   - (3) Find the nearest centroid to select the corresponding $(M_{j^*}, \theta_{j^*})$ pair.
   - (4) Acquire additional frequency components specified by $M_{j^*}$ and reconstruct using $\theta_{j^*}$.

### Key Designs
1. **High-frequency Bayesian uncertainty quantification (Mask selector $e_\psi$)**:
    - **Function**: Quantify the high-frequency uncertainty distribution for each input to adaptively select the sampling pattern.
    - **Mechanism**: Utilize a conditional normalizing flow super-resolution model to generate multiple high-resolution samples $\{f_\psi^{-1}(z_s; M_0 k)\}$ from the low-frequency reconstructed image, compute the sample variance $v(M_0 k)$ in k-space, and use the normalized $u = v / \|v\|_2$ as the uncertainty direction descriptor.
    - **Design Motivation**: Sample variance is an unbiased estimator of MSE. Conditioned on known low frequencies, high-frequency variance directly reflects the spatial distribution of reconstruction errors; different high-frequency details (e.g., horizontal stripes vs. vertical hair) lead to different uncertainty directions, necessitating distinct sampling strategies.

2. **Sampling-reconstruction pair construction ($(M_j, \theta_j)_{j=1}^J$)**:
    - **Function**: Create dedicated masks and reconstruction networks for each cluster.
    - **Mechanism**: Generate mask $M_j$ using rejection sampling (rather than simple sorting) based on the cluster centroid $c_j$, introducing stochasticity to optimize SSIM; then independently train a reconstruction network $\theta_j$ for each $M_j$ to ensure Pareto optimality.
    - **Design Motivation**: Although sorting maximizes PSNR (Proposition 1), introducing stochasticity is more beneficial for SSIM. Independent training avoids the Pareto suboptimality issue of a single network serving multiple masks.

3. **Theoretical guarantees (Theorem 3.1 & 3.2)**:
    - **Function**: Prove the theoretical advantages of $\mathcal{H}_{1.5}$.
    - **Mechanism**: Theorem 3.1 proves $\mathcal{H}_1 \subseteq \mathcal{H}_{1.5}$ (adaptive selection includes non-adaptive as a special case), hence $\inf_{h \in \mathcal{H}_{1.5}} \mathcal{L}[h] \leq \inf_{h \in \mathcal{H}_1} \mathcal{L}[h]$. Theorem 3.2 proves that under the condition $|\pi_\phi(M_0 \mathcal{K})| \leq J$, $\mathcal{H}_2 \subseteq \mathcal{H}_{1.5}$ holds (each adaptively sampled mask can find its corresponding sub-model).
    - **Design Motivation**: Theoretically demonstrate that $\mathcal{H}_{1.5}$ integrates the adaptivity of $\mathcal{H}_1$ and the Pareto optimality of $\mathcal{H}_2$.

### Loss & Training
Each sub-reconstruction network $\theta_j$ is trained independently, with the loss function $l(I, \hat{I}) = 1 - \mathrm{SSIM}(I, \hat{I})$. Training utilizes the full dataset rather than only the subset corresponding to the cluster, ensuring each $\theta_j$ reaches optimality under its fixed mask $M_j$. The super-resolution generative model is pretrained using existing conditional normalizing flow methods.

## Key Experimental Results

### Main Results
**CelebA Face Image 2D Fourier CS** (SSIM↑):

| Method | Category | 8× | 16× |
|------|------|------|------|
| VD | $\mathcal{H}_1$ | 0.9073 | 0.8734 |
| LOUPE | $\mathcal{H}_1$ | 0.8742 | 0.8673 |
| Policy | $\mathcal{H}_2$ | 0.8501 | 0.8394 |
| **Ours** | $\mathcal{H}_{1.5}$ | **0.9405** | **0.8952** |

**fastMRI Multi-coil Brain 1D Line Sampling** (SSIM↑):

| Method | Category | 4× | 8× |
|------|------|------|------|
| VD | $\mathcal{H}_1$ | 0.9603 | 0.9367 |
| LOUPE | $\mathcal{H}_1$ | 0.9541 | 0.9218 |
| Policy | $\mathcal{H}_2$ | 0.9569 | 0.9240 |
| **Ours** | $\mathcal{H}_{1.5}$ | **0.9624** | **0.9407** |

### Ablation Study
**Impact of the number of segments $J$** (CS-MRI 2D 8×, SSIM difference relative to J=1):

| J Value | Average SSIM Gain | Worst 5% SSIM Gain | Description |
|------|---------------|-------------------|------|
| J=1 | 0 (baseline) | 0 (baseline) | Non-adaptive |
| J=2 | +0.006 | +0.008 | Maximum average gain |
| J=3 | +0.007 | +0.012 | Average gain slows down, but outliers are handled better |
| J=4 | +0.007 | +0.014 | Average plateau, outliers continue to improve |

**Rejection sampling vs Sorting** (CS-MRI 1D 8×, J=3):

| Mask Generation Method | SSIM |
|------------|------|
| kmeans-Sorted | 0.9167 |
| **Rejection sampling (ours)** | **0.9407** |

### Key Findings
- $\mathcal{H}_{1.5}$ consistently outperforms both $\mathcal{H}_1$ and $\mathcal{H}_2$ across all 6 experimental settings (2 datasets × 3 acceleration rates/sampling patterns).
- Under CelebA 8×, the SSIM improves by approximately 0.04 (0.9073 $\rightarrow$ 0.9405), which is a very significant improvement in image reconstruction.
- In clinical-grade multi-coil MRI settings, an SSIM improvement of around 0.004 is considered highly meaningful in the field of MRI reconstruction.
- The sample variance of the super-resolution generative model effectively quantifies high-frequency uncertainty (Sorted-Self PSNR 37.15 vs. Sorted-Another 36.36 vs. VD 33.33).
- Rejection sampling yields a 0.02+ higher SSIM than sorting-based mask generation, validating the importance of introducing stochasticity for SSIM optimization.

## Highlights & Insights
- **Elegant Concept**: The naming of $\mathcal{H}_{1.5}$ reflects the wisdom of a compromise between fixed sampling ($\mathcal{H}_1$) and fully adaptive sampling ($\mathcal{H}_2$)—a finite number of predefined schemes + data-adaptive selection.
- **Practically Deployable**: Compared to $\mathcal{H}_2$, it avoids difficult optimization over discrete spaces, requiring only a single fast uncertainty evaluation and nearest neighbor search at inference time.
- **Pareto Optimality Insight**: Deeply points out an overlooked issue in adaptive sampling—it is impossible for a single reconstruction network to be optimal for all adaptive masks simultaneously. This aligns with findings in the multi-degradation restoration field (blind denoising is inferior to noise-aware denoising).
- **Theoretical Elegance**: The two theorems are concise and powerful, with proofs requiring only a few lines.

## Limitations & Future Work
- Training cost scales linearly with $J$, requiring the training of $J$ independent reconstruction networks.
- Only explored 1D line sampling and 2D point sampling, leaving more complex non-Cartesian trajectories (e.g., spiral, radial) untouched.
- The quality of the super-resolution model directly affects the accuracy of uncertainty quantification.
- The optimal choice of $J$ must be determined heuristically; the trade-off of increasing $J$ yielding diminishing average gains but continuously improving outlier robustness lacks an automated selection method.
- Clustering is performed on the training set, which may require re-clustering in domain transfer scenarios.

## Related Work & Insights
- Compared with joint optimization methods like LOUPE, this work elegantly avoids discrete optimization difficulties through "finite choices."
- Similar to uncertainty-based adaptive sampling methods (e.g., cGAN-based), this paper further addresses the Pareto suboptimality problem.
- Resembles piecewise regression / Mixture of Experts (MoE) concepts—partitioning the data space into sub-regions, with each region handled by an expert model.
- **Insight**: This paradigm of "predefined candidate pool + adaptive selection" can be extended to other tasks requiring adaptive degradation handling, such as adaptive quantization and adaptive compression.

## Rating
- Novelty: ⭐⭐⭐⭐ The $\mathcal{H}_{1.5}$ concept is novel, bringing the MoE concept into sampling-reconstruction joint optimization with elegant theoretical proofs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers two domains (faces and multi-coil MRI) with various acceleration rates and sampling patterns, complemented by systematic ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Well-structured logic, tight integration of theory and practice, and an intuitive comparison framework of $\mathcal{H}_1 / \mathcal{H}_2 / \mathcal{H}_{1.5}$.
- Value: ⭐⭐⭐⭐ Holds direct practical value in clinical MRI scenarios, with excellent generalizability of the proposed methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling](adaptive_compressed_sensing_with_diffusionbased_posterior_sa.md)
- [\[CVPR 2025\] Sampling Innovation-Based Adaptive Compressive Sensing](../../CVPR2025/model_compression/sampling_innovation-based_adaptive_compressive_sensing.md)
- [\[ECCV 2024\] AdaLog: Post-Training Quantization for Vision Transformers with Adaptive Logarithm Quantizer](adalog_post-training_quantization_for_vision_transformers_with_adaptive_logarith.md)
- [\[ECCV 2024\] Adversarially Robust Distillation by Reducing the Student-Teacher Variance Gap](adversarially_robust_distillation_by_reducing_the_student-teacher_variance_gap.md)
- [\[ECCV 2024\] Isomorphic Pruning for Vision Models](isomorphic_pruning_for_vision_models.md)

</div>

<!-- RELATED:END -->
