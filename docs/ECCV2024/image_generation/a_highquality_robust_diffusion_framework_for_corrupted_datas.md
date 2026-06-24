---
title: >-
  [Paper Note] A High-Quality Robust Diffusion Framework for Corrupted Dataset
description: >-
  [ECCV 2024][Image Generation][Diffusion Models] This paper proposes the RDUOT framework, which integrates Unbalanced Optimal Transport (UOT) into a diffusion model (DDGAN) for the first time. By learning $q(x_0|x_t)$ instead of $q(x_{t-1}|x_t)$, it effectively filters outliers in training data, achieving robust generation on corrupted datasets while outperforming the DDGAN baseline on clean datasets.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Diffusion Models"
  - "Unbalanced Optimal Transport (UOT)"
  - "Robust Generation"
  - "Outlier Filtering"
  - "DDGAN"
date: 2026-05-08
content_hash: 0ec5f3c2893bf5ec
---

# A High-Quality Robust Diffusion Framework for Corrupted Dataset

**Conference**: ECCV 2024  
**arXiv**: [2311.17101](https://arxiv.org/abs/2311.17101)  
**Code**: None  
**Area**: Diffusion Models / Image Generation / Robust Generation  
**Keywords**: Diffusion Models, Unbalanced Optimal Transport (UOT), Robust Generation, Outlier Filtering, DDGAN  

## TL;DR
This paper proposes the RDUOT framework, which integrates Unbalanced Optimal Transport (UOT) into a diffusion model (DDGAN) for the first time. By learning $q(x_0|x_t)$ instead of $q(x_{t-1}|x_t)$, it effectively filters outliers in training data, achieving robust generation on corrupted datasets while outperforming the DDGAN baseline on clean datasets.

## Background & Motivation
In real-world scenarios, training data often contains outlier samples due to human errors or collection biases (e.g., face datasets mixed with flipped faces, handwritten digits, etc.), which causes generative models to produce undesirable samples. Previous robust generation methods primarily relied on the GAN framework—RobustGAN utilizes an additional weight network to achieve UOT, but requires the simultaneous optimization of three networks, leading to convergence difficulties; UOTM integrates UOT more naturally by replacing GANs with an OT-based generative model, but its robustness has only been validated on low-resolution, small-scale datasets. Meanwhile, diffusion models have completely bypassed GANs in terms of image quality, but currently, no diffusion model possesses robustness against corrupted data. Therefore, there is an urgent need for a framework that can leverage the high-quality generation capabilities of diffusion models while resisting the influence of outliers.

## Core Problem
How to introduce the outlier filtering capability of UOT into the diffusion model framework? The key difficulties are: (1) GANs use OT to minimize the distance between real and fake distributions, whereas UOT optimizes the mapping cost from source to target; since their objectives differ, they cannot be simply combined. (2) As the diffusion timestep $t$ increases, the noisy versions $x_t$ of clean samples and outliers converge (the Wasserstein distance decreases), making it increasingly difficult for UOT to distinguish between them.

## Method

### Overall Architecture
RDUOT adopts DDGAN as its backbone but replaces the adversarial process in GAN with an OT-based generative model. Given a noisy image $x_t$ as input, the generator $G_\theta(x_t, t, z)$ predicts the clean image $\hat{x}_0$, and then samples $x_{t-1}$ via $q(x_{t-1}|x_t, \hat{x}_0)$. During training, the generator and the potential function network $D_\phi$ are optimized using a semi-dual UOT loss. During inference, high-quality images can be generated in only 2-4 denoising steps.

### Key Designs
1. **Replacing GAN with an OT-based Generative Model**: DDGAN originally used a GAN discriminator to match $q(x_{t-1}|x_t)$ and $p_\theta(x_{t-1}|x_t)$, and UOT cannot be directly embedded into this adversarial framework (as it would require an additional weight network). RDUOT draws on the ideas of OTM/UOTM to let the optimal transport mapping itself act as the generative model, allowing the semi-dual form of UOT to naturally replace the OT objective function without requiring a third network.

2. **Learning $q(x_0|x_t)$ Instead of $q(x_{t-1}|x_t)$**: This is the core insight of this paper. The authors prove via Proposition 1 that the Wasserstein distance between the clean and outlier distributions after $t$ steps of noise addition decreases with $t$—that is, with more noise, outliers and clean samples become more difficult to distinguish. Therefore, performing UOT filtering directly at the $x_{t-1}$ level yields poor results. The solution is to let UOT operate on the zero-noise $x_0$ space, where the difference between outliers and clean samples is maximized, allowing UOT to efficiently remove mass from the outlier samples.

3. **Softplus as the $\Psi^*$ Function (Lipschitz $\Psi$)**: Traditional choices like the conjugate function of the KL divergence are exponential functions (non-Lipschitz, prone to explosion during training), and the conjugate of $\chi^2$ is a quadratic polynomial (also non-Lipschitz). The authors found that using Softplus $\Psi^*(x) = \ln(1+e^x)$ as the conjugate function significantly stabilizes the training process and prevents loss divergence due to its Lipschitz continuity (with a Lip constant of 2).

4. **Potential Network Using $x_{t-1}$ Instead of $x_0$**: Although UOT filters outliers in the $x_0$ space, experiments show that using $(\hat{x}_{t-1}, x_t, t)$ as the input to the potential function $D_\phi$ performs better than $(\hat{x}_0, x_t, t)$. Since the sampling process depends on $x_{t-1} \sim q(x_{t-1}|x_t, \hat{x}_0)$, using $x_{t-1}$ allows the potential function to align directly with the sampling process.

### Loss & Training
- The training objective is the semi-dual UOT loss (Eq. 16), where the generator minimizes the transport cost $\tau c(x_t, \hat{x}_0) - D_\phi(\hat{x}_{t-1}, x_t, t)$ and the potential function $D_\phi$ maximizes this objective.
- The cost function uses the L2 norm: $c(x, y) = \tau \|x - y\|_2^2$
- The UNet-like generator and StyleGAN-style architecture (AdaIN injection of latent variable $z$) inherited from DDGAN are used.
- R1 regularization, EMA, and the Adam optimizer are employed.
- The number of timesteps is only 2-4 (consistent with DDGAN).
- $\tau$ is the key hyperparameter controlling the outlier filtering intensity: if too small, it degenerates to OT (no filtering); if too large, it falsely removes clean samples.

## Key Experimental Results

### Robustness Experiments (Corrupted Dataset)

| Dataset | Metric | RDUOT | DDGAN | Gain |
|--------|------|-------|-------|------|
| CI+3%MT (32×32) | FID↓ | 3.43 | 4.76 | -1.33 |
| CI+5%MT | FID↓ | 4.37 | 8.81 | -4.44 |
| CI+10%MT | FID↓ | 6.98 | 14.77 | -7.79 |
| CE+FT (64×64) | FID↓ | 7.89 | 10.68 | -2.79 |
| CE+MT | FID↓ | 9.29 | 12.95 | -3.66 |
| CE+CH | FID↓ | 7.86 | 9.83 | -1.97 |
| CE+FCE | FID↓ | 5.99 | 6.48 | -0.49 |

*CI=CIFAR10, MT=MNIST, CE=CelebA-HQ, FT=FashionMNIST, CH=LSUN Church, FCE=Vertically flipped CelebA-HQ*

Comparison with other robust methods (5% outliers): RDUOT(CI+5%MT)=4.37 vs UOTM=7.89 vs RobustGAN=10.68

Synthetic outlier rate: Under 10% outlier data, RDUOT achieves only a 3.8% synthetic outlier rate vs. DDGAN's 9.8%.

### Clean Datasets

| Dataset | Metric | RDUOT | DDGAN | Gain |
|--------|------|-------|-------|------|
| CIFAR-10 | FID↓ | 2.95 | 3.75 | -0.80 |
| CIFAR-10 | Recall↑ | 0.58 | 0.57 | +0.01 |
| CelebA-HQ 256 | FID↓ | 5.60 | 7.64 | -2.04 |
| STL-10 | FID↓ | 11.50 | 21.79 | -10.29 |
| STL-10 | Recall↑ | 0.49 | 0.40 | +0.09 |

### Ablation Study

- **Choice of $\Psi$**: Softplus achieves FID=2.95 on the clean set and FID=4.37 on the corrupted set, outperforming $\chi^2$ (3.93/5.04). KL training is unstable, with the best FID reaching only 10.11 before diverging.
- **Validation of Framework Design**: Performing UOT directly with $q(x_{t-1}|x_t)$ (Eq. 13) $\rightarrow$ FID=6.94 (due to difficulty in distinguishing outliers under high noise); using $q(x_0|x_t)$ but without $x_{t-1}$ information in the potential function (Eq. 15) $\rightarrow$ FID=5.93; full design (Eq. 16) $\rightarrow$ FID=4.37.
- **Number of Timesteps**: 2 steps FID=3.84, 4 / 8 steps FID=2.95 / 2.65 (RDUOT consistently improves, whereas DDGAN degrades to 4.36 at 8 steps).
- **Sensitivity to $\tau$**: If too small (1e-4), it does not filter $\rightarrow$ FID=6.74; moderate (1e-2) is optimal $\rightarrow$ FID=4.37; if too large (1e-1), it falsely filters out clean samples $\rightarrow$ FID=5.98.
- Convergence speed is significantly better than DDGAN: At epoch 400, RDUOT FID < 20 while DDGAN FID > 100 (STL-10).

## Highlights & Insights
- **First Robust Diffusion Model**: This work fills the vacancy of diffusion models in robust generation. The proposed scheme is elegant—rather than simply stacking UOT and DDGAN, the correct integration is identified through a deep analysis of the differences in their objectives.
- **Insight of Filtering in $x_0$ Space**: Through theoretical proof (Wasserstein distance decreases as $t$ increases) and experimental validation, the necessity of filtering outliers in the zero-noise space is revealed. This insight is also of reference value for other tasks that require screening/guidance in the diffusion process.
- **Lipschitz $\Psi^*$ Stabilizes Training**: The Lipschitz property of Softplus is key to training stability. This finding is applicable to all generative models utilizing UOT.
- **Better Even on Clean Data**: RDUOT is not only robust but also comprehensively surpasses DDGAN on clean data. The authors hypothesize that UOT can filter out "unexpected outlier" samples caused by giant diffusion steps.

## Limitations & Future Work
- Limited experimental scale: The maximum resolution is only 256×256 (CelebA-HQ), lacking validation on larger-scale data (ImageNet) and higher resolutions.
- Simpler outlier types: The study mainly focuses on images mixed from different datasets, without testing more subtle outliers (such as images in the same domain but with minor artifacts).
- Framework dependency on the DDGAN architecture: DDGAN itself is not the strongest current diffusion model. If UOT could be integrated into modern architectures like latent diffusion or DiT, its impact would be even larger.
- $\tau$ needs manual tuning and is sensitive to different datasets (ranging from 1e-7 to 1e-3), lacking an adaptive mechanism.
- The application potential of UOT in conditional generation (text-to-image) has not been explored.

## Related Work & Insights
- **vs RobustGAN**: RobustGAN requires three networks (generator + discriminator + weight network), making training unstable and achieving a much higher FID than RDUOT. RDUOT only needs two networks, which is simpler and more efficient.
- **vs UOTM**: UOTM directly uses an UOT-based generative model but without a diffusion process, which underperforms RDUOT on high-resolution/complex data. UOTM is even inferior to DDGAN on CE+FCE.
- **vs DDGAN**: RDUOT maintains the few-step sampling advantage of DDGAN. By replacing the GAN discriminator with an OT-based generative model and introducing UOT, it not only gains robustness but also substantially reduces clean-data FID and converges faster.

## Rating
- Novelty: ⭐⭐⭐⭐ First robust diffusion model. The insights of $x_0$ space filtering and Lipschitz $\Psi$ are valuable, though the core components (UOT, DDGAN, OTM) are not original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Robustness and clean data experiments are comprehensive and the ablation is thorough, but large-scale high-resolution validation is lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical analysis, well-articulated motivation, and overall consistent logic.
- Value: ⭐⭐⭐⭐ Fills an important gap, with improvements on clean data being a plus, although the actual impact is constrained by the experimental scale.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] UDiffText: A Unified Framework for High-quality Text Synthesis in Arbitrary Images via Character-aware Diffusion Models](udifftext_a_unified_framework_for_high-quality_text_synthesis_in_arbitrary_image.md)
- [\[ECCV 2024\] EMDM: Efficient Motion Diffusion Model for Fast and High-Quality Motion Generation](emdm_efficient_motion_diffusion_model_for_fast_and_high-quality_motion_generatio.md)
- [\[ECCV 2024\] Toward Tiny and High-quality Facial Makeup with Data Amplify Learning](toward_tiny_and_high-quality_facial_makeup_with_data_amplify_learning.md)
- [\[ECCV 2024\] Robust-Wide: Robust Watermarking against Instruction-driven Image Editing](robust-wide_robust_watermarking_against_instruction-driven_image_editing.md)
- [\[ICML 2025\] Taming Diffusion for Dataset Distillation with High Representativeness (D³HR)](../../ICML2025/image_generation/taming_diffusion_for_dataset_distillation_with_high_representativeness.md)

</div>

<!-- RELATED:END -->
