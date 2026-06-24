---
title: >-
  [Paper Note] Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction?
description: >-
  [NeurIPS 2025][Medical Imaging][sparse-view CT] This paper reveals that pixel-level metrics such as PSNR and SSIM fail to capture anatomical structural completeness in sparse-view CT reconstruction (correlation only 0.16–0.30), and proposes anatomy-aware metrics (NSD/clDice) based on automated segmentation alongside the CARE framework—which incorporates segmentation-guided loss into diffusion model training—achieving 32% improvement in structural completeness for large organs…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "sparse-view CT"
  - "anatomy-aware metrics"
  - "structural completeness"
  - "diffusion model"
date: 2026-05-08
content_hash: 7395f7c89cfe53f2
---

# Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction?

**Conference**: NeurIPS 2025
**arXiv**: [2506.02093](https://arxiv.org/abs/2506.02093)  
**Code**: [GitHub](https://github.com/MrGiovanni/CARE)  
**Area**: Other
**Keywords**: sparse-view CT, anatomy-aware metrics, structural completeness, diffusion model

## TL;DR
This paper reveals that pixel-level metrics such as PSNR and SSIM fail to capture anatomical structural completeness in sparse-view CT reconstruction (correlation only 0.16–0.30), and proposes anatomy-aware metrics (NSD/clDice) based on automated segmentation alongside the CARE framework—which incorporates segmentation-guided loss into diffusion model training—achieving 32% improvement in structural completeness for large organs and 36% for vessels.

## Background & Motivation

**Background**: Sparse-view CT substantially reduces radiation dose by limiting the number of projection views (<50 views). In recent years, neural rendering methods such as NeRF and Gaussian splatting have achieved steady improvements in PSNR/SSIM.

**Limitations of Prior Work**: PSNR/SSIM compute global voxel-wise average differences and are highly insensitive to small yet clinically critical anatomical structures (e.g., gallbladder, adrenal glands, abdominal aorta)—structures that occupy <0.01% of the volume, whose absence alters PSNR/SSIM only at the third or fourth decimal place. A radiologist evaluation involving 21 Johns Hopkins radiologists found that state-of-the-art methods exhibit missing structures, hallucinated anatomy, and severe artifacts.

**Key Challenge**: High pixel-level scores ≠ clinical utility—reconstructions with high PSNR may be missing critical vessels or organs.

**Goal**: How can anatomical integrity in CT reconstruction be reliably evaluated and preserved?

**Key Insight**: Leveraging an nnU-Net segmentor trained on 3,151 CT scans to perform automated assessment—NSD evaluates surface fidelity of compact organs, while clDice evaluates topological continuity of tubular structures.

**Core Idea**: Replace pixel-level metrics with a segmentor for CT reconstruction evaluation, and backpropagate segmentation loss into diffusion model training to preserve anatomical structures.

## Method

### Overall Architecture
CARE consists of two components: (1) **Anatomy-aware evaluation metrics**—a frozen nnU-Net segmentor performs multi-organ segmentation on reconstructed CT volumes, evaluated using NSD (for large/small organs) and clDice (for vessels/intestines); (2) **Anatomy-aware reconstruction framework**—a latent-space diffusion model conditioned on the latent representation of sparse-view reconstructed CT, with a loss comprising noise prediction, pixel reconstruction, and segmentation guidance terms.

### Key Designs

1. **Anatomy-Aware Metrics (NSD + clDice)**

    - Function: Evaluate structural completeness separately by category (large organs / small organs / intestines / vessels).
    - Mechanism: $\text{NSD} = \frac{|\{x \in S_P: d(x,S_G) \leq \tau\}| + |\{x \in S_G: d(x,S_P) \leq \tau\}|}{|S_P|+|S_G|}$ is sensitive to boundary accuracy and scale-invariant; clDice evaluates topological continuity on skeletonized centerlines. Frozen nnU-Net predictions replace manual annotations (correlation >0.92).
    - Design Motivation: DSC is a volumetric measure; minor boundary deviations in large organs produce negligible DSC changes, whereas NSD directly measures boundary distance. The clinical significance of vessels lies in their continuity, which clDice captures through topological information.

2. **CARE Diffusion Enhancement Framework**

    - Function: Enhance sparse-view CT reconstruction using a latent-space diffusion model.
    - Mechanism: $\mathcal{L}_{\text{CARE}} = \mathcal{L}_n + \lambda_p \mathcal{L}_p + \lambda_s \mathcal{L}_s$—noise prediction + L1 pixel + segmentation cross-entropy guidance. The denoising UNet is conditioned on the reconstructed CT latent. Segmentation loss is computed by applying the frozen segmentor to decoded predictions and comparing against ground truth.
    - Design Motivation: Method-agnostic (enhances any reconstruction method) and patient-agnostic (learns anatomical priors from large-scale data).

### Loss & Training
- $\lambda_p = 1,\ \lambda_s = 0.001$

## Key Experimental Results

### Main Results: Decoupling of Pixel Metrics and Anatomical Metrics

| Metric | Correlation with Anatomical Completeness |
|--------|------------------------------------------|
| SSIM   | Corr ≈ 0.16 |
| PSNR   | Corr ≈ 0.30 |

### CARE Enhancement Performance

| Anatomical Structure | Improvement |
|----------------------|-------------|
| Large organs         | +32% |
| Small organs         | +22% |
| Intestines           | +40% |
| Vessels              | +36% |

### Key Findings
- **PSNR/SSIM rankings and anatomical metric rankings can be completely inverted.**
- **CARE's improvements are entirely invisible in PSNR/SSIM.**
- **Segmentor–radiologist annotation correlation >0.92**—a reliable substitute.

## Highlights & Insights
- The systematic evidence that **"high PSNR ≠ clinical utility"** serves as a major warning signal for the CT reconstruction community.
- The closed-loop design using the **segmentor as both an evaluation bridge and a training signal** is particularly elegant.

## Limitations & Future Work
- Segmentor quality constitutes the upper bound of metric reliability.
- Intestine annotation quality is poor (clDice correlation drops to 0.77).
- nnU-Net training relies on private data.

## Related Work & Insights
- **vs. LPIPS/FID**: These improve natural image evaluation but do not encode anatomical semantics.
- **vs. Diffusion-based CT reconstruction**: Optimizing pixel loss alone may generate hallucinations; CARE adds segmentation loss to protect structural integrity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Exposes blind spots of pixel metrics + proposes anatomy-aware metrics + framework
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 methods + 4 anatomical categories + 61 cases + radiologist evaluation
- Writing Quality: ⭐⭐⭐⭐⭐ Problem-driven with compelling argumentation
- Value: ⭐⭐⭐⭐⭐ Direct practical impact on medical image reconstruction

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] NAB: Neural Adaptive Binning for Sparse-View CT Reconstruction](../../ICLR2026/medical_imaging/nab_neural_adaptive_binning_for_sparse-view_ct_reconstruction.md)
- [\[ICLR 2026\] DM4CT: Benchmarking Diffusion Models for Computed Tomography Reconstruction](../../ICLR2026/medical_imaging/dm4ct_benchmarking_diffusion_models_for_computed_tomography_reconstruction.md)
- [\[ICLR 2026\] OpenPros: A Large-Scale Dataset for Limited View Prostate Ultrasound Computed Tomography](../../ICLR2026/medical_imaging/openpros_a_large-scale_dataset_for_limited_view_prostate_ultrasound_computed_tom.md)
- [\[ICCV 2025\] Vector Contrastive Learning for Pixel-wise Pretraining in Medical Vision](../../ICCV2025/medical_imaging/vector_contrastive_learning_for_pixel-wise_pretraining_in_medical_vision.md)
- [\[ICCV 2025\] Coordinate-based Speed of Sound Recovery for Aberration-Corrected Photoacoustic Computed Tomography](../../ICCV2025/medical_imaging/coordinate-based_speed_of_sound_recovery_for_aberration-corrected_photoacoustic_.md)

</div>

<!-- RELATED:END -->
