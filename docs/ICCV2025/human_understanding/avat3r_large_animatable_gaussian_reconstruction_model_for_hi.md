---
title: >-
  [Paper Note] Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars
description: >-
  [ICCV 2025][Human Understanding][3D Gaussian] This paper presents Avat3r — the first animatable large reconstruction model (LRM) that regresses high-quality drivable 3D Gaussian head avatars from only 4 input images in a feed-forward manner. By integrating DUSt3R positional maps and Sapiens semantic features as priors, and modeling expression-driven animation via simple cross-attention, Avat3r substantially outperforms existing methods on the Ava256 and NeRSemble datasets.
tags:
  - ICCV 2025
  - Human Understanding
  - 3D Gaussian
  - head avatar reconstruction
  - facial animation
  - large reconstruction model
  - feed-forward inference
date: 2026-05-08
content_hash: 65ac87beabaf0c63
---

# Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars

**Conference**: ICCV 2025
**arXiv**: [2502.20220](https://arxiv.org/abs/2502.20220)
**Code**: No public code
**Area**: Human Understanding
**Keywords**: 3D Gaussian, head avatar reconstruction, facial animation, large reconstruction model, feed-forward inference

## TL;DR
This paper presents Avat3r — the first animatable large reconstruction model (LRM) that regresses high-quality drivable 3D Gaussian head avatars from only 4 input images in a feed-forward manner. By integrating DUSt3R positional maps and Sapiens semantic features as priors, and modeling expression-driven animation via simple cross-attention, Avat3r substantially outperforms existing methods on the Ava256 and NeRSemble datasets.

## Background & Motivation
Creating photorealistic 3D head avatars is in high demand for telepresence, filmmaking, and personalized gaming, yet existing methods each suffer from significant limitations:
- **Studio-grade optimization methods** (e.g., URAvatar) require multi-view capture and expensive test-time optimization (3 hours on 8×A100), making them unsuitable for consumer-grade scenarios.
- **Monocular video reconstruction** (e.g., FlashAvatar) overfits to training viewpoints and generalizes poorly to novel views.
- **3D-aware portrait animation** (e.g., GPAvatar, GAGAvatar) focuses primarily on frontal rendering, sacrificing 3D consistency for image quality.
- **Photorealistic 3D face models** (e.g., GPHM, HeadGAP) are constrained by limited training identities (only a few hundred), making it difficult to learn the full distribution of facial appearance.

**Core Observation**: 3D face data is limited along the identity axis (only hundreds of subjects) but abundant along the expression axis (thousands of frames per subject). This motivates a system conditioned on identity (provided by input images) that generalizes only along the expression axis, avoiding the need to learn the full distribution of facial appearance. This design philosophy of *generalizing only along axes with sufficient data* is the central conceptual contribution of the paper.

## Method

### Overall Architecture
Four images with camera parameters and a target expression code $z_{exp}$ are fed as input. DUSt3R generates positional maps $I^{pos}$ and Sapiens extracts semantic feature maps $I^{feat}$. The image, positional, and ray embeddings are concatenated and patchified, then processed by a Vision Transformer backbone — self-attention handles cross-view matching and cross-attention injects expression information. The tokens are upsampled into per-pixel Gaussian attribute maps $M$, followed by position/color skip connections and confidence-threshold filtering, yielding a 3D Gaussian set $\mathcal{G}$ renderable from arbitrary viewpoints. The entire pipeline is purely feed-forward with no test-time optimization.

### Key Designs
1. **Foundation model prior injection (DUSt3R + Sapiens)**: DUSt3R predicts dense positional maps as coarse 3D position initialization for each Gaussian, added to the final position prediction via skip connection ($M^{pos} \leftarrow M^{pos}+I^{pos}$). The Sapiens 2B model extracts low-resolution semantic feature maps, which are resolution-aligned via GridSample and concatenated with tokens, simplifying the cross-view matching task for subsequent Transformer layers. Both are precomputed offline to reduce training cost. A key finding is that DUSt3R produces reasonable positional maps even under inconsistent inputs (different expressions).

2. **Animatable large reconstruction model architecture**: The method adopts a GRM-style Vision Transformer. Input images $I$, positional maps $I^{pos}$, and Plücker ray coordinates $I^{pluck}$ are concatenated and patchified into tokens. The core consists of 8 self-attention layers (cross-view matching) and 8 cross-attention layers (expression injection). Expression codes are projected via MLP into a token sequence of length $S=4$, i.e., $f_{exp} \in \mathbb{R}^{S \times D}$, and each image token attends to this expression sequence via cross-attention. A notable finding is that this simple cross-attention mechanism suffices to model complex facial animation, without requiring explicit deformation fields or template meshes.

3. **Confidence filtering and adaptive Gaussian count**: DUSt3R confidence maps (threshold $\tau=0.5$) are used to filter low-confidence pixels, naturally determining the number of Gaussians. Subjects with voluminous hair yield more Gaussians while bald subjects yield fewer — an adaptive foreground-segmentation-like effect. A color skip connection ($M^{rgb} \leftarrow M^{rgb}+I$) provides the inductive bias that predicted colors should be close to input pixel colors.

4. **Inconsistent-input training strategy**: During training, the 4 input images are sampled from different timesteps (different expressions) rather than the synchronized multi-view same-expression setup required by conventional LRMs. DUSt3R still produces reasonable results under such inconsistency. This strategy not only enables training and inference on larger monocular video datasets but also improves robustness to accidental head movement during casual smartphone capture.

### Loss & Training
- **Loss function**: $\mathcal{L} = 0.8 \mathcal{L}_{l1} + 0.2 \mathcal{L}_{ssim} + 0.01 \mathcal{L}_{lpips}$; the LPIPS term is introduced only after 3M steps to avoid premature focus on high-frequency details.
- **Training data**: Ava256 dataset — 244 subjects for training, 12 for testing, 80 camera viewpoints, ~5,000 frames per subject, 512×512 crops.
- **Training configuration**: Adam optimizer, lr=5e-5, batch size=1/GPU × 8×A100, 3.5M steps over ~4 days.
- **Supervision strategy**: Each batch consists of 4 randomly sampled expression inputs and 8 target-expression supervision viewpoints.
- **Viewpoint sampling**: k-farthest viewpoint sampling ensures diverse and well-distributed input viewpoints.

## Key Experimental Results

### Main Results

**Few-shot (4-image input) head avatar creation**:

| Dataset | Method | PSNR↑ | SSIM↑ | LPIPS↓ | AKD↓ | CSIM↑ |
|---------|--------|-------|-------|--------|------|-------|
| Ava256 | HeadNeRF | 9.1 | 0.64 | 0.52 | 6.9 | 0.11 |
| Ava256 | InvertAvatar | 14.2 | 0.36 | 0.55 | 15.8 | 0.29 |
| Ava256 | GPAvatar | 19.4 | 0.69 | 0.34 | 5.3 | 0.31 |
| Ava256 | **Avat3r** | **20.7** | **0.71** | **0.33** | **4.8** | **0.59** |
| NeRSemble | HeadNeRF | 9.8 | 0.69 | 0.47 | 4.9 | 0.22 |
| NeRSemble | GPAvatar† | 17.6 | 0.67 | 0.40 | 5.7 | 0.07 |
| NeRSemble | **Avat3r** | **20.5** | **0.75** | **0.33** | **3.7** | **0.50** |

CSIM (identity similarity) improves from 0.31 (GPAvatar) to 0.59, demonstrating a qualitative leap in identity preservation. Strong generalization is also observed on the unseen NeRSemble benchmark.

**Runtime analysis**:

| Method | Creation Time (s)↓ | Driving Speed (fps)↑ |
|--------|-------------------|---------------------|
| HeadNeRF | 6511 | 1 |
| GPAvatar | 0.2 | 9.5 |
| **Avat3r (4-shot)** | **12.3** | **7.9** |
| Avat3r (1-shot) | 1.15 | 53 |

### Ablation Study

| Configuration | PSNR↑ | AKD↓ | Notes |
|---------------|-------|------|-------|
| w/o DUSt3R | 21.1 | 8.31 | Degraded geometric fidelity; difficulty aligning multi-view predictions |
| w/o Sapiens | 20.9 | 8.08 | Reduced sharpness, especially in hair regions |
| w/o random timestep training | 21.2 | 8.86 | Slightly sharper images but fragile to inconsistent inputs |
| w/o position skip | 21.39 | — | Misalignment artifacts and blurring |
| w/o color skip | 21.76 | — | Color shift |
| **Full model** | **22.05** | **8.08** | All components jointly optimal |

### Key Findings
- DUSt3R and Sapiens contribute complementary geometric and semantic priors that are mutually irreplaceable.
- Inconsistent-input training marginally reduces image sharpness (PSNR 22.05→21.2) but is critical for robustness (AKD 8.08→8.86, i.e., increased facial keypoint distance without it).
- Adding 984 neutral-expression-only identities — increasing the dataset by only 0.08% — improves cross-identity generalization.
- Single-image inference (1-shot) runs at 53 fps, far exceeding the 7.9 fps of 4-shot, owing to ~75% fewer Gaussians.

## Highlights & Insights
- **First animatable large 3D reconstruction model**: Extends the LRM paradigm to drivable 3D head avatars for the first time.
- **Minimalist animation mechanism**: Complex facial animation is achieved solely via cross-attention to an expression token sequence, without template meshes or explicit deformation fields.
- **Adaptive Gaussian count**: Per-pixel Gaussian prediction combined with confidence filtering naturally adapts Gaussian density to individual subjects.
- **"Generalize only along well-supported axes" design philosophy**: When data is scarce along the identity axis but abundant along the expression axis, conditioning on identity and generalizing over expressions is a principled and generalizable design strategy.
- **Strong generalization**: On the unseen NeRSemble dataset, the model can even animate AI-generated images and ancient sculptures.

## Limitations & Future Work
- Single-image inference relies on a 3D GAN for 3D lifting, introducing error accumulation.
- Camera poses are required as input — erroneous pose estimation leads to reconstruction artifacts.
- Lighting is baked into the reconstruction, precluding relighting and limiting applicability in virtual environments.
- Training on only 244 subjects poses a risk of identity overfitting.
- The 12.3-second creation time is dominated by DUSt3R inference rather than the Transformer itself, and could be accelerated with a lightweight DUSt3R variant.

## Related Work & Insights
- **vs. GPAvatar**: Predicts a canonical TriPlane on NeRF and drives it with FLAME; expression is constrained to FLAME space. Avat3r learns expression mapping directly via cross-attention. CSIM: 0.31→0.59.
- **vs. FlashAvatar**: Requires full video and severely overfits training viewpoints. Avat3r with only 4 images far surpasses it (PSNR 15.0→20.5 on NeRSemble).
- **vs. HeadGAP/GPHM**: Learn the full facial appearance distribution but are constrained by training identity count. Avat3r conditions on input images to sidestep the identity generalization problem.
- **vs. URAvatar**: 3-hour optimization on 8×A100 vs. 12.3-second feed-forward inference — a vast practical difference.
- The minimalist cross-attention approach to modeling dynamics may be transferable to other dynamic reconstruction tasks such as gesture and body motion.
- The finding that DUSt3R is robust to inconsistent inputs generalizes to other 3D reconstruction scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ First extension of LRM to animatable head reconstruction; design is clean yet represents a clever combination of existing components.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-dataset evaluation, extensive ablations, and full coverage of single-image, multi-image, and application scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, well-motivated design, with experimental support for every design choice.
- Value: ⭐⭐⭐⭐ High practical value (4 smartphone photos → drivable avatar in seconds); meaningful contribution to the digital avatar field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ImHead: A Large-scale Implicit Morphable Model for Localized Head Modeling](imhead_a_large-scale_implicit_morphable_model_for_localized_head_modeling.md)
- [\[NeurIPS 2025\] VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image](../../NeurIPS2025/human_understanding/vasa-3d_lifelike_audio-driven_gaussian_head_avatars_from_a_single_image.md)
- [\[ICCV 2025\] GGTalker: Talking Head Synthesis with Generalizable Gaussian Priors and Identity-Specific Adaptation](ggtalker_talking_head_systhesis_with_generalizable_gaussian_priors_and_identity-.md)
- [\[ICCV 2025\] AdaHuman: Animatable Detailed 3D Human Generation with Compositional Multiview Diffusion](adahuman_animatable_detailed_3d_human_generation_with_compositional_multiview_di.md)
- [\[CVPR 2026\] 4DSurf: High-Fidelity Dynamic Scene Surface Reconstruction](../../CVPR2026/human_understanding/textit4dsurf_high-fidelity_dynamic_scene_surface_reconstruction.md)

</div>

<!-- RELATED:END -->
