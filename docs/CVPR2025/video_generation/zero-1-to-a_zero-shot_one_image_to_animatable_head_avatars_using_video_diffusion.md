---
title: >-
  [Paper Note] Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion
description: >-
  [CVPR 2025][Video Generation][Head Avatars] Zero-1-to-A is proposed to generate high-fidelity animatable 4D head avatars from a single image using a pretrained video diffusion model via Symbiotic Generation (SymGEN) and a progressive learning strategy, effectively addressing the spatio-temporal inconsistency issue in video diffusion.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Head Avatars"
  - "Video Diffusion"
  - "3D Gaussian Splatting"
  - "Progressive Learning"
  - "Zero-Shot Generation"
date: 2026-05-08
content_hash: fa1b15365cb559bf
---

# Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion

**Conference**: CVPR 2025  
**arXiv**: [2503.15851](https://arxiv.org/abs/2503.15851)  
**Code**: [GitHub](https://github.com/ZhenglinZhou/Zero-1-to-A)  
**Area**: Video Generation  
**Keywords**: Head Avatars, Video Diffusion, 3D Gaussian Splatting, Progressive Learning, Zero-Shot Generation

## TL;DR

Zero-1-to-A is proposed to generate high-fidelity animatable 4D head avatars from a single image using a pretrained video diffusion model via Symbiotic Generation (SymGEN) and a progressive learning strategy, effectively addressing the spatio-temporal inconsistency issue in video diffusion.

## Background & Motivation

- Generating animatable head avatars typically requires large amounts of real or synthetic human data, which is difficult to collect.
- Score Distillation Sampling (SDS)-based methods utilize pretrained diffusion models for zero-shot generation, but directly distilling 4D avatars from video diffusion causes over-smoothing due to spatio-temporal inconsistencies.
- Video diffusion models exhibit spatial inconsistency (appearance changes across different views) and temporal inconsistency (incoherent expression sequences) when generating portrait videos.
- SDS loss directly aligns the avatar with the pseudo-ground truth of the diffusion model, but the pseudo-ground truth itself is unstable, leading to degraded results.
- Image-conditioned 4D avatar generation is more challenging than text-conditioned generation but is more critical for practical applications.
- A robust method is required to synthesize consistent datasets from inconsistent video diffusion outputs for avatar reconstruction.

## Method

### Overall Architecture

Zero-1-to-A utilizes animatable Gaussian heads (FLAME + 3DGS) as the 4D representation. The core algorithm contains two components: (1) SymGEN (Symbiotic Generation) — establishing a reciprocal relationship between dataset construction and avatar reconstruction by caching video diffusion results in an updatable dataset for iterative optimization; (2) Progressive Learning Strategy — decoupling video diffusion generation into spatial consistency learning (fixed expression, from frontal to side views) and temporal consistency learning (fixed view, from neutral to exaggerated expressions) to progressively improve quality from simple to complex.

### Key Designs

**1. Symbiotic Generation (SymGEN)**
- **Function**: Establishes a reciprocal enhancement loop between the dataset and the avatar, iteratively improving the quality of both.
- **Mechanism**: (1) Avatar-driven dataset enhancement: render current avatar videos $\rightarrow$ extract Mediapipe facial landmarks $\rightarrow$ VAE encoding + DDIM inversion to obtain noise $\rightarrow$ denoise with landmarks as geometric guidance $\rightarrow$ generate enhanced videos to replace the dataset; (2) Dataset-refined avatar reconstruction: train the avatar on the updated dataset using $\mathcal{L}_1$+LPIPS+position loss+scaling loss; (3) Dataset update every 30 iterations.
- **Design Motivation**: One-time dataset generation is of poor quality due to spatial and temporal inconsistencies. An iterative reciprocal enhancement can progressively eliminate inconsistencies. Higher avatar quality $\rightarrow$ more accurate rendering guidance $\rightarrow$ more consistent video diffusion output $\rightarrow$ in turn improves the avatar.

**2. Spatial Consistency Learning**
- **Function**: Fixes the expression and progressively learns multi-view representations from frontal to side views.
- **Mechanism**: Create $n_s=20$ spatial samples, each containing a fixed ARKit base expression and a camera trajectory from frontal to a random side view. Programmatically introduce larger side views progressively via $p_i = \hat{p}_{\min(i,j)}$, where $j = \min(\lfloor k/d_s \rfloor + 1, n_f)$.
- **Design Motivation**: Video diffusion produces more consistent results under simple camera poses and neutral expressions. Starting from the frontal view establishes a good initialization, followed by progressively adding side views.

**3. Temporal Consistency Learning**
- **Function**: Fixes near-frontal cameras and progressively learns from synthetic neutral expressions to real exaggerated expressions.
- **Mechanism**: Use a fixed near-frontal camera pose. The first $k_s=5000$ iterations focus only on spatial learning, iterations 5000-8000 add $n_{syn}=10$ synthetic neutral expression samples, and after 8000 iterations, $n_{real}=10$ real exaggerated expressions from talk show videos are added. This mimics a test-time training (TTT) strategy.
- **Design Motivation**: Real expression sequences are more exaggerated and varied than synthetic ones; direct learning easily leads to inconsistency. Building a robust baseline on simple synthetic expressions first before introducing challenging data is optimal.

### Loss & Training

$$\mathcal{L} = \lambda_1 \mathcal{L}_1 + \lambda_{lpips} \mathcal{L}_{LPIPS} + \lambda_{pos} \mathcal{L}_{pos} + \lambda_s \mathcal{L}_s$$

where $\lambda_1=10$, $\lambda_{lpips}=10$, $\lambda_{pos}=0.1$, and $\lambda_s=10$. $\mathcal{L}_{pos}$ and $\mathcal{L}_s$ constrain the alignment of the 3D Gaussian points with the position and scaling of the FLAME mesh, respectively.

## Key Experimental Results

### Main Results

| Method | CLIP-Score (ViT-L/14)↑ | Rendering Speed (FPS) |
|------|----------------------|-------------|
| DreamHead | ~0.68 | ~2 |
| HeadStudio | ~0.72 | ~5 |
| Portrait4D-v2 | ~0.70 | ~15 |
| **Zero-1-to-A** | **~0.76** | **~90** |

*Zero-1-to-A leads significantly in both fidelity and rendering speed.*

### Ablation Study

| Variant | CLIP-Score↑ | Visual Quality |
|------|-----------|---------|
| SDS Loss (baseline) | ~0.62 | Over-smoothed |
| One-time Dataset | ~0.66 | Poor quality |
| SymGEN w/o Progressive | ~0.72 | Partial artifacts |
| **Full Zero-1-to-A** | **~0.76** | **Sharp details** |

### Key Findings

- Direct SDS distillation produces over-smoothed results; SymGEN's iterative dataset update significantly improves quality.
- Progressive learning breaks the "chicken-and-egg" dilemma — initial low-quality avatars cannot provide good guidance.
- Spatial and temporal decoupled learning each play their respective roles, with their combination yielding the best performance.
- The 3DGS-based representation achieves real-time rendering at approximately 90 FPS.

## Highlights & Insights

1. **Symbiotic Generation Paradigm**: Systematises the reciprocal relationship between dataset construction and training, progressively improving quality without relying on external data.
2. **Simple-to-Complex Curriculum Learning**: Decouples 4D generation into spatial and temporal stages, progressing from easy to hard within each stage, preventing the instability of a single-step optimization.
3. **Zero-shot + Real-time Rendering**: Generates high-quality animatable avatars using only a single image with ~5 hours of optimization.

## Limitations & Future Work

- The optimization process still requires approximately 5 hours on a single A6000 GPU, making it unsuitable for real-time deployment.
- There is still room for improvement in handling extreme side views and extreme expressions.
- Performance depends heavily on the quality of the video diffusion model; different foundation models may yield varying results.
- Future work can explore more efficient dataset update strategies to reduce optimization time.

## Related Work & Insights

- Compared to SDS-based methods like DreamHead and HeadStudio, replacing distillation with reconstruction avoids over-smoothing.
- The core concept of symbiotic generation can be generalized to other scenarios requiring learning from inconsistent generation results.
- The progressive learning strategy provides a valuable reference for all video diffusion distillation tasks.

## Rating

⭐⭐⭐⭐ — Cleverly solves the spatio-temporal inconsistency issue in video diffusion distillation through the mutual coordination of symbiotic generation and progressive learning. The experimental results are convincing in terms of both fidelity and rendering speed. However, the 5-hour optimization time and dependency on foundation models pose limitations for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ConMo: Controllable Motion Disentanglement and Recomposition for Zero-Shot Motion Transfer](conmo_controllable_motion_disentanglement_and_recomposition_for_zero-shot_motion.md)
- [\[CVPR 2026\] Are Image-to-Video Models Good Zero-Shot Image Editors?](../../CVPR2026/video_generation/are_image-to-video_models_good_zero-shot_image_editors.md)
- [\[CVPR 2025\] Visual Prompting for One-Shot Controllable Video Editing Without Inversion](visual_prompting_for_one-shot_controllable_video_editing_without_inversion.md)
- [\[CVPR 2026\] Scaling Zero-Shot Reference-to-Video Generation](../../CVPR2026/video_generation/scaling_zero-shot_reference-to-video_generation.md)
- [\[ICML 2026\] WIND: Weather Inverse Diffusion for Zero-Shot Atmospheric Modeling](../../ICML2026/video_generation/wind_weather_inverse_diffusion_for_zero-shot_atmospheric_modeling.md)

</div>

<!-- RELATED:END -->
