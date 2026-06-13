---
title: >-
  [Paper Note] Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction?
description: >-
  [NeurIPS 2025][sparse-view CT] This paper reveals that pixel-level metrics such as PSNR and SSIM fail to capture anatomical structural completeness in sparse-view CT reconstruction (correlation only 0.16–0.30)…
tags:
  - "NeurIPS 2025"
  - "sparse-view CT"
  - "anatomy-aware metrics"
  - "structural completeness"
  - "diffusion model"
date: 2026-05-08
content_hash: e0ef3a076dd3cdfe
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

- [\[NeurIPS 2025\] Lagrangian neural ODEs: Measuring the existence of a Lagrangian with Helmholtz metrics](lagrangian_neural_odes_measuring_the_existence_of_a_lagrangian_with_helmholtz_me.md)
- [\[ICCV 2025\] Recover Biological Structure from Sparse-View Diffraction Images with Neural Volumetric Prior](../../ICCV2025/others/recover_biological_structure_from_sparse-view_diffraction_images_with_neural_vol.md)
- [\[NeurIPS 2025\] MiCADangelo: Fine-Grained Reconstruction of Constrained CAD Models from 3D Scans](micadangelo_fine-grained_reconstruction_of_constrained_cad_models_from_3d_scans.md)
- [\[NeurIPS 2025\] Sign-In to the Lottery: Reparameterized Sparse Training from Scratch](sign-in_to_the_lottery_reparameterizing_sparse_training_from_scratch.md)
- [\[CVPR 2026\] Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](../../CVPR2026/others/rooftop_wind_field_reconstruction_using_sparse_sen.md)

</div>

<!-- RELATED:END -->
