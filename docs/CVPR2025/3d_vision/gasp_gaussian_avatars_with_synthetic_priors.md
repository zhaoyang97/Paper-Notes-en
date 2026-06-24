---
title: >-
  [Paper Note] GASP: Gaussian Avatars with Synthetic Priors
description: >-
  [CVPR 2025][3D Vision][Gaussian Avatars] This paper proposes GASP, which utilizes synthetic data to train a generative prior model (auto-decoder) for Gaussian Avatars. It bridges the synthetic-to-real domain gap through a three-stage fitting process and learned per-Gaussian semantic feature correlations, enabling the creation of high-quality, real-time animatable avatars (at 70 fps) supporting 360° rendering from only a single image or a short video.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Gaussian Avatars"
  - "Synthetic Data Priors"
  - "Monocular Reconstruction"
  - "360-degree Rendering"
  - "Real-time Animation"
date: 2026-05-08
content_hash: 0df88fba396e0d62
---

# GASP: Gaussian Avatars with Synthetic Priors

**Conference**: CVPR 2025  
**arXiv**: [2412.07739](https://arxiv.org/abs/2412.07739)  
**Code**: [https://microsoft.github.io/GASP/](https://microsoft.github.io/GASP/) (Project Page)  
**Area**: 3D Vision  
**Keywords**: Gaussian Avatars, Synthetic Data Priors, Monocular Reconstruction, 360-degree Rendering, Real-time Animation

## TL;DR
This paper proposes GASP, which utilizes synthetic data to train a generative prior model (auto-decoder) for Gaussian Avatars. It bridges the synthetic-to-real domain gap through a three-stage fitting process and learned per-Gaussian semantic feature correlations, enabling the creation of high-quality, real-time animatable avatars (at 70 fps) supporting 360° rendering from only a single image or a short video.

## Background & Motivation

1. **Background**: Animatable avatars based on Gaussian Splatting (Gaussian Avatars) have achieved significant progress in quality and speed. Existing methods either require expensive multi-camera setups for free-viewpoint rendering or reconstruct from monocular training videos but can only render from fixed viewpoints.

2. **Limitations of Prior Work**: (a) Multi-camera methods require complex acquisition setups, making them inaccessible to average users; (b) monocular methods suffer from severe artifacts under non-training viewpoints (especially the sides and back of the head); (c) existing few-shot avatar methods (such as the NeRF-based Preface and Cafca) suffer from extremely slow rendering speeds (>20s/frame).

3. **Key Challenge**: Reconstructing 360° avatars from monocular input is a highly under-constrained problem, as the extreme sides and back of the head are completely invisible. A prior model is required to "fill in" the missing regions, but high-quality multi-view real-human datasets are scarce, and annotations (camera calibration, 3DMM parameters) contain significant errors.

4. **Goal**: How to create high-quality 360° avatars supporting real-time rendering from a single image or a short video captured by a webcam or smartphone?

5. **Key Insight**: Synthetic data has the natural advantage of pixel-level accurate annotations and arbitrary multi-view coverage, which can be leveraged to train large-scale prior models. The key challenge lies in bridging the synthetic-to-real domain gap.

6. **Core Idea**: Train an auto-decoder prior of Gaussian Avatars on large-scale synthetic facial data. By utilizing the correlation of per-Gaussian semantic features and a three-stage fitting process, the synthetic-to-real domain gap is crossed, enabling 360° real-time avatars from a single image.

## Method

### Overall Architecture
The proposed method consists of two main stages. **Prior Training Stage**: An auto-decoder model is trained on 1,000 synthetic subjects (with 50 multi-view images per subject) to jointly optimize a Canonical Template, per-identity latent codes, per-Gaussian features, and an MLP decoder. **Fitting Stage**: Given a single real image or short video, the prior is adapted to the real data through a three-step fitting process: (1) inverting the latent code, (2) fine-tuning the MLP, and (3) refining the Gaussian parameters.

### Key Designs

1. **Auto-decoder Prior Model Training (Prior Model Training)**:

    - **Function**: Learning a generative model capable of generating Gaussian Avatars of different identities.
    - **Mechanism**: Each Gaussian is assigned an 8-dimensional learnable feature vector $\mathbf{f}_i$, and each identity is associated with a 512-dimensional latent code $\mathbf{z}_j$. A shared MLP decoder $\mathcal{D}$ maps the features and identity codes to the offsets of Gaussian attributes: $\mathcal{A}_{i,j}=\mathcal{C}_{i,j}+\mathcal{D}(\mathbf{f}_i, \mathbf{z}_j)$, where $\mathcal{C}$ represents the Canonical Template (mean avatar). Initializing using a UV map at $512 \times 512$ resolution yields approximately 188k Gaussians. The training loss incorporates pixel-level L1+SSIM, perceptual loss LPIPS, alpha mask loss, and regularization losses.
    - **Design Motivation**: Directly regressing all Gaussian attributes via an MLP is intractable due to the extremely high dimensionality. Employing a per-Gaussian feature as a joint "positional + semantic encoding" allows the MLP to process each Gaussian independently and in parallel, thereby drastically reducing the parameter count. The template-plus-offset design enables the model to focus solely on identity variations, promoting stability.

2. **Per-Gaussian Semantic Feature Correlations (Learned Feature Correlations)**:

    - **Function**: Enabling updates in visible regions during the fitting process to propagate automatically to invisible regions.
    - **Mechanism**: During training, the MLP is constrained to map semantically similar Gaussians to a similar feature space. PCA visualization shows that learned features exhibit clear semantic clustering (e.g., forehead, lips, and scalp are naturally grouped). During fitting, the features $\mathbf{f}$ are frozen, and only $\mathbf{z}$ and $\mathcal{D}$ are optimized. Consequently, if the MLP learns to turn the forehead Gaussians blonde, other Gaussians with similar features on the back of the head will automatically turn blonde.
    - **Design Motivation**: This is the key mechanism to address the core dilemma of "monocular input to 360° output". It enables information propagation from visible to invisible regions via implicit semantic associations without relying on explicit symmetry assumptions.

3. **Three-stage Fitting Process (Three-stage Fitting)**:

    - **Function**: Adapting the synthetic prior to real user data.
    - **Mechanism**: Stage 1 (Inversion): Freeze everything and only optimize the identity code $\mathbf{z}$ to find the optimal avatar within the prior space (500 steps). Stage 2 (MLP Fine-tuning): Freeze features and template, and fine-tune the MLP $\mathcal{D}$ to bridge the domain gap using feature correlations (500 steps). Stage 3 (Gaussian Refinement): Directly optimize individual Gaussian parameters to best fit the target data (100 steps). A regularization term $L_{prior}$ (L2 distance of Gaussian parameters to the Stage 1 result) is incorporated at each stage to prevent drifting too far from the prior. The entire fitting process takes about 10 minutes on an RTX 4090.
    - **Design Motivation**: Pure inversion can only generate synthetic-looking appearances, whereas direct Gaussian optimization produces severe artifacts in invisible views. The three-stage progressive transition smoothly transfers the representation. Stage 2 is critical—fine-tuning the MLP allows simultaneous updates of both visible and invisible regions through feature correlations, achieving elegant domain adaptation.

### Loss & Training
Prior training loss is defined as: $\mathcal{L}=\lambda_{pix}L_{pix}+\lambda_\alpha L_\alpha+\lambda_{percep}L_{percep}+L_{reg}$. Here, $L_{pix}$ includes L1 and SSIM, $L_{percep}$ is based on LPIPS, and $L_{reg}$ regularizes Gaussian scale and displacement. The displacement regularization on the scalp region is reduced by a factor of 100 to allow for hair modeling. The prior is trained on 4×A100 GPUs for 4 days with a batch size of 8 for 250 epochs. At inference time, no neural networks are required, enabling pure Gaussian splatting.

## Key Experimental Results

### Main Results

| Setup | Metric | GASP | FlashAvatar | GA | DiffusionRig | ROME |
|------|------|------|-------------|-----|-------------|------|
| Monocular Video PSNR↑ | dB | **21.34** | 17.25 | 17.39 | 19.67 | - |
| Monocular Video SSIM↑ | | **0.712** | 0.603 | 0.601 | 0.343 | - |
| Monocular Video LPIPS↓ | | **0.333** | 0.450 | 0.428 | 0.436 | - |
| Monocular Video FID↓ | | **117** | 351 | 366 | 155 | - |
| Monocular Video ID-SIM↑ | | **0.568** | 0.234 | 0.179 | 0.302 | - |
| Single Image PSNR↑ | dB | **20.73** | 13.26 | 14.80 | 16.87 | 15.78 |
| Single Image QUAL↑ | /5 | **3.80** | 2.05 | 2.03 | 3.15 | 3.38 |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | ID-SIM↑ |
|------|-------|-------|--------|------|---------|
| Full model | **21.34** | **0.712** | **0.333** | **117** | 0.568 |
| w/o prior | 19.42 | 0.670 | 0.391 | 212 | 0.478 |
| w/o prior regularization | 20.31 | 0.701 | 0.344 | 122 | **0.620** |
| w/o stage 1 | 19.56 | 0.678 | 0.364 | 127 | 0.588 |
| w/o stage 2 | 20.33 | 0.704 | 0.347 | 118 | 0.490 |
| 1 subject prior | Worse than w/o prior | - | - | - | - |
| 1000 subjects prior | Full model | - | - | - | - |

### Key Findings
- **Synthetic prior contributes significantly**: Removing the prior drops PSNR by nearly 2 dB and increases FID from 117 to 212.
- **Trade-off in prior regularization**: Disabling it yields the highest ID-SIM (0.620) because the model can fit the visible frontal regions with fewer constraints, but degrades FID and LPIPS due to deterioration on invisible views.
- **Training the prior on only 1 subject performs worse than no prior**, indicating that the prior requires sufficient diversity to be effective.
- **Inference speed runs at 70 fps (RTX 4090)** with 15MB storage, completely network-free (pure Gaussian splatting).
- **Comparable to SOTA in multi-camera setups** (PSNR 23.44 vs. GA 23.73), verifying that the synthetic prior does not degrade reconstruction quality when dense observations are available.

## Highlights & Insights
- **Successful Application of Synthetic Priors**: Leveraging perfectly annotated synthetic data combined with a domain-adaptive fitting strategy provides an elegant paradigm to address real-world data scarcity. This pipeline could be generalized to full-body avatars, hand reconstruction, and other areas.
- **Implicit Propagation of Per-Gaussian Semantic Features**: Avoiding explicit symmetry or correspondence constraints, information automatically propagates from visible to invisible regions via learned feature correlations. PCA visualization demonstrates that the features indeed capture consistent semantics.
- **Zero Network Overhead at Inference**: The prior and MLP are only utilized during the fitting stage and discarded at inference, allowing lightweight and real-time rendering.
- **Progressive Domain Adaptation via Three-stage Fitting**: Each stage gradually unlocks degrees of freedom, balancing the regularization from priors and fitting accuracy to the target data.

## Limitations & Future Work
- **Absence of illumination variation and dynamic wrinkles modeling** leads to slightly lower quality than specialized methods in dense multi-camera setups.
- **Domain gap in illumination**: The synthetic data is rendered under uniform white lighting, meaning complex real-world illumination can impede adaptation.
- **Fitting time limit**: Optimization takes about 10 minutes on an RTX 4090, which may be too slow for instant-on applications.
- **Dependence on 3DMM fitting quality**: Occlusion or extreme facial expressions often result in erroneous 3DMM annotations, which directly impacts the output quality.
- **Interactive online fitting** (e.g., progressively refining the avatar during an active video call) remains unexplored.

## Related Work & Insights
- **vs. Cafca/Preface**: These methods also utilize synthetic priors, but they are NeRF-based, requiring >20s/frame for rendering static scenes. GASP employs Gaussian Splatting, enabling real-time (70 fps) dynamic animations.
- **vs. GaussianAvatars**: GA binds Gaussians to a FLAME mesh. While GA achieves high quality in multi-view settings, it degrades under monocular scenarios. GASP solves this monocular overfitting via its learned prior.
- **vs. DiffusionRig**: Diffusion models act as strong priors but suffer from poor identity preservation. GASP achieves much stronger identity consistency (ID-SIM of 0.568 vs. 0.302).
- **vs. Gaussian Morphable Model (Xu et al.)**: Shares a similar concept but only trains on frontal faces, failing to render the back of the head. GASP utilizes synthetic data with full spherical coverage.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of synthetic priors, semantic feature correlation, and three-stage fitting is highly effective, though the individual components are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation under three settings, complete ablation studies, and includes user studies.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with clearly justified motivations and method design.
- Value: ⭐⭐⭐⭐⭐ Exceptionally practical system, representing the first real-time system capable of generating 360° dynamic Gaussian avatars from a single view.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors](../../ICCV2025/3d_vision/strandhead_text_to_hair-disentangled_3d_head_avatars_using_human-centric_priors.md)
- [\[CVPR 2025\] Synthetic Prior for Few-Shot Drivable Head Avatar Inversion](synthetic_prior_for_few-shot_drivable_head_avatar_inversion.md)
- [\[CVPR 2025\] 3D Gaussian Head Avatars with Expressive Dynamic Appearances by Compact Tensorial Representations](3d_gaussian_head_avatars_with_expressive_dynamic_appearances_by_compact_tensoria.md)
- [\[CVPR 2025\] LUCAS: Layered Universal Codec Avatars](lucas_layered_universal_codec_avatars.md)
- [\[CVPR 2025\] PERSE: Personalized 3D Generative Avatars from A Single Portrait](perse_personalized_3d_generative_avatars_from_a_single_portrait.md)

</div>

<!-- RELATED:END -->
