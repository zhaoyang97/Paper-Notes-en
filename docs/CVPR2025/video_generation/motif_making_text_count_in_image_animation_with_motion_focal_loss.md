---
title: >-
  [Paper Note] MotiF: Making Text Count in Image Animation with Motion Focal Loss
description: >-
  [CVPR 2025][Video Generation][Text-guided image animation] This paper proposes Motion Focal Loss (MotiF), which spatially weights the diffusion loss using motion heatmaps generated from optical flow. This guides the model to focus on high-motion regions, significantly enhancing text-following and motion quality in Text-Image-to-Video generation, and constructs the TI2V-Bench evaluation benchmark.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Text-guided image animation"
  - "motion focal loss"
  - "diffusion models"
  - "optical flow"
date: 2026-05-08
content_hash: 62ad36f84e70e63e
---

# MotiF: Making Text Count in Image Animation with Motion Focal Loss

**Conference**: CVPR 2025  
**arXiv**: [2412.16153](https://arxiv.org/abs/2412.16153)  
**Code**: None (evaluation set is available on the project page)  
**Area**: Video Generation  
**Keywords**: Text-guided image animation, motion focal loss, video generation, diffusion models, optical flow

## TL;DR

This paper proposes Motion Focal Loss (MotiF), which spatially weights the diffusion loss using motion heatmaps generated from optical flow. This guides the model to focus on high-motion regions, significantly enhancing text-following and motion quality in Text-Image-to-Video generation, and constructs the TI2V-Bench evaluation benchmark.

## Background & Motivation

Text-Image-to-Video (TI2V) generation aims to synthesize videos based on an initial image and a text description. The key challenge is that the image provides a strong spatial signal, causing the model to rely excessively on the image conditioning (termed **conditional image leakage**) and ignore motion instructions within the text.

Prior efforts have attempted to address this by weakening the image conditioning (via noise or masking) or by feeding motion priors into the model. However, these methods approach the problem from the perspective of **input signals**, expecting the model to learn motion implicitly. MotiF directly tackles this from the perspective of the **training objective**: in a video, 97% of the pixels may be static, while only 3% contain meaningful motion. The standard L2 loss treats all regions equally, biasedly steering the model to replicate static images. MotiF explicitly guides the model to focus on motion learning by assigning higher loss weights to high-motion regions.

## Method

### Overall Architecture

The framework is built on top of the pretrained T2V model VideoCrafter2. The conditioned image is injected into the denoising U-Net via **concatenation (x-cat)** (rather than cross-attention, cx-attn). The model is jointly trained using the standard diffusion loss $\mathcal{L}_{\text{diffusion}}$ combined with the motion focal loss $\mathcal{L}_{\text{motif}}$. No additional inputs are required during inference.

### Key Designs

1. **Motion Focal Loss**: The core innovation. First, a RAFT optical flow estimator calculates the optical flow intensity $\mathbf{f}_l$ between adjacent frames. This intensity is then normalized to $[0,1]$ using a sigmoid-like function to generate a motion heatmap $\mathbf{m}$. After downsampling the heatmap to latent space resolution, it is utilized as pixel-specific loss weights: $\mathcal{L}_{\text{motif}} = \mathbb{E}\|\mathbf{m}' \cdot (\epsilon - \epsilon_\theta(\mathbf{z}_t, \mathbf{c}, t))\|_2^2$. The final loss is formulated as $\mathcal{L} = \mathcal{L}_{\text{diffusion}} + \lambda \mathcal{L}_{\text{motif}}$, where $\lambda=1$. This design is simple yet effective—it does not alter the model architecture, requires no extra inputs at inference time, and is highly complementary to existing methods.

2. **Analysis of Image Conditioning Injection**: This work systematically compares three injection methods for image conditioning: cross-attention only (cx-attn), concatenation only (x-cat), and a dual-stream design (cx-attn + x-cat). It is observed that using cx-attn alone leads to poor image alignment, whereas the combination of cx-attn + x-cat improves image alignment but degrades text-following—since image and text embeddings compete in the cross-attention layer. Therefore, the x-cat only scheme is selected, retaining spatial alignment while leaving the cross-attention channel dedicated solely to text signals.

3. **TI2V-Bench Evaluation Benchmark**: A benchmark containing 320 image-text pairs is constructed, covering 22 scenes, 88 unique images, and 133 unique text prompts. It is designed with challenging scenarios such as multi-object fine-grained control and the introduction of new objects. A JUICE-style human evaluation protocol is adopted—annotators first select their overall preference and then provide reasons across four categories: object motion, text alignment, image alignment, and quality.

### Loss & Training

- Joint loss: $\mathcal{L} = \mathcal{L}_{\text{diffusion}} + \lambda \mathcal{L}_{\text{motif}}$, with $\lambda=1$.
- Optical flow normalization function: $\sigma(x) = 1/(1+e^{100(0.05-x)})$, producing continuous and highly polarized heatmaps.
- Training with v-prediction mode.
- Text is randomly dropped by 10% to enable classifier-free guidance.
- Learning rate is $5 \times 10^{-5}$, global batch size is 64, linear noise schedule with 1000 diffusion steps.
- Trained for 32K steps on 8 A100-80G GPUs, with a resolution of $320 \times 512$ and a dynamic frame stride of 16 frames.

## Key Experimental Results

### Main Results

| Baselines | MotiF Preference Rate | Competitor Preference Rate | Key Driving Advantages |
|---------|------------|----------|---------|
| DynamiCrafter | ~75% | ~25% | Text alignment + motion quality |
| I2VGen-XL | ~80% | ~20% | Image alignment + motion quality |
| Cinemo | ~72% | ~28% | Text alignment + motion quality |
| ConsistI2V | ~70% | ~30% | Text alignment + motion quality |
| SEINE | ~68% | ~32% | Text alignment + motion quality |
| **Average** | **72%** | **28%** | Text alignment and motion quality are the primary reasons for winning |

### Ablation Study

| Configuration | TI2V Score (MotiF/Competitor) | Text Alignment | Object Motion | Description |
|------|----------------------|---------|---------|------|
| MotiF vs. w/o MotiF loss | 63.1/36.9 | 34.9/16.4 | 32.9/16.4 | MotiF substantially improves motion and text adherence |
| MotiF vs. Inv-MotiF | 61.9/38.1 | 34.8/12.8 | 34.9/15.4 | Focusing on motion regions outperforms focusing on static regions |
| x-cat vs. cx-attn+x-cat | 58.1/41.9 | 31.5/21.7 | 34.0/21.6 | x-cat is more favorable for text adherence |
| x-cat vs. cx-attn only | 92.2/7.8 | 56.8/5.3 | 41.3/4.5 | cx-attn-only performs extremely poorly |

### Key Findings

1. MotiF effectively reduces the relative loss ratio in high-motion domains—across various diffusion timesteps, the model trained with MotiF consistently maintains a lower loss proportion in high-motion regions than the baseline.
2. Human evaluation results consistently show that MotiF's strengths lie in text alignment and object motion, which perfectly aligns with the motivation behind the method.
3. Automated evaluation metrics (e.g., Image/Text Alignment on Animate Bench) are inconsistent with human perception—static videos that simply replicate the first frame unexpectedly achieve the highest Image Alignment scores.
4. The failure of the inverse MotiF loss (focusing on static areas) validates the necessity of prioritizing high-motion areas.

## Highlights & Insights

- Great simplicity: The method only requires the pre-computation of motion heatmaps via optical flow to weight the loss, without changing the architecture or increasing inference cost. It is orthogonal and combinable with existing approaches.
- Notably, the unreliability of automatic evaluation metrics on TI2V tasks is highlighted—static videos scoring highest on image alignment suggests that automatic metrics should be utilized with caution.
- The setup of TI2V-Bench (multiple animation prompts for the same image + challenging scenarios) successfully fills a vacancy in TI2V evaluations.
- The JUICE-style human evaluation protocol (single metric + multi-factor rationale) reconciles both clear-cut conclusions and analytical depth.

## Limitations & Future Work

- Generation quality remains limited under complex scenarios such as multi-object interaction or novel object entrances.
- Built on VideoCrafter2 (U-Net), thus not validated on the more recent DiT architectures.
- The motion heatmap currently relies solely on optical flow; other signal sources like depth changes or semantic saliency could be explored.
- The sensitivity under different $\lambda$ values has not been thoroughly analyzed.
- Lacks a comparison with more advanced closed-source models (such as Sora or Gen-3).

## Related Work & Insights

- MotiF differs from but is complementary to works like LivePhoto (motion word embedding weighting) and Follow-Your-Click (utilizing optical flow masks as input).
- The concept of motion heatmap-weighted losses can be extended to other motion-centric tasks such as video super-resolution and video frame interpolation.
- The analytical findings on image conditioning injection (where x-cat outperforms others) offer valuable references for subsequent work in TI2V domains.

## Rating

- Novelty: ⭐⭐⭐⭐ Addresses the TI2V text-following issue directly from the loss function perspective with a simple and novel approach.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Conducts large-scale human evaluation across 9 models along with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clearly motivated and concisely presented, though certain experimental details reside in the supplementary materials.
- Value: ⭐⭐⭐⭐ Simple, effective, and highly combinable with existing techniques, offering practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MotionPro: A Precise Motion Controller for Image-to-Video Generation](motionpro_a_precise_motion_controller_for_image-to-video_generation.md)
- [\[CVPR 2025\] MotionStone: Decoupled Motion Intensity Modulation with Diffusion Transformer for Image-to-Video Generation](motionstone_decoupled_motion_intensity_modulation_with_diffusion_transformer_for.md)
- [\[CVPR 2025\] Through-The-Mask: Mask-based Motion Trajectories for Image-to-Video Generation](through-the-mask_mask-based_motion_trajectories_for_image-to-video_generation.md)
- [\[CVPR 2025\] Teller: Real-Time Streaming Audio-Driven Portrait Animation with Autoregressive Motion Generation](teller_real-time_streaming_audio-driven_portrait_animation_with_autoregressive_m.md)
- [\[ECCV 2024\] MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model](../../ECCV2024/video_generation/mofa-video_controllable_image_animation_via_generative_motion_field_adaptions_in.md)

</div>

<!-- RELATED:END -->
