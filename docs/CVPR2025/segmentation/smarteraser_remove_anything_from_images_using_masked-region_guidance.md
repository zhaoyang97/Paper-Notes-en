---
title: >-
  [Paper Note] SmartEraser: Remove Anything from Images using Masked-Region Guidance
description: >-
  [CVPR 2025][Segmentation][Object Removal] SmartEraser proposes a new paradigm called Masked-Region Guidance, which retains the masked region as a guide instead of discarding it. Combined with the million-scale synthetic Syn4Removal dataset, it significantly outperforms existing mask-and-inpaint methods on object removal tasks.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Object Removal"
  - "Masked-Region Guidance"
  - "Image Inpainting"
  - "Diffusion Models"
  - "Synthetic Dataset"
date: 2026-05-08
content_hash: 9ad025ac7399f4ac
---

# SmartEraser: Remove Anything from Images using Masked-Region Guidance

**Conference**: CVPR 2025  
**arXiv**: [2501.08279](https://arxiv.org/abs/2501.08279)  
**Code**: [Project Page](https://longtaojiang.github.io/smarteraser.github.io/)  
**Area**: Image Segmentation  
**Keywords**: Object Removal, Masked-Region Guidance, Image Inpainting, Diffusion Models, Synthetic Dataset

## TL;DR

SmartEraser proposes a new paradigm called Masked-Region Guidance, which retains the masked region as a guide instead of discarding it. Combined with the million-scale synthetic Syn4Removal dataset, it significantly outperforms existing mask-and-inpaint methods on object removal tasks.

## Background & Motivation

Object removal is a core function of image editing and has been widely integrated into applications like Photoshop and Google Photos. Current mainstream methods adopt the "mask-and-inpaint" paradigm, which suffers from two fundamental issues:
- **Object Regeneration Problem**: After discarding the masked region, the model lacks precise identification of the target to be removed, potentially generating new objects within the masked area (e.g., generating another car on the road instead of removing it).
- **Contextual Inconsistency**: User-defined masks are often larger than target objects. The "mask-and-inpaint" paradigm needs to synthesize the expanded region, which easily disrupts the visual coherence of the surrounding context.
- Existing datasets are either limited in scale (RORD and ObjectDrop contain fewer than 3.5k unique scenes) or rely on inpainting models to generate pseudo ground truths (like GQA-Inpaint and DEFACTO), thereby limiting performance to that of the inpainting model itself.
- GAN-based methods (e.g., ZITS++, MAT, LaMa) have limited generation diversity and quality; diffusion-based methods (e.g., RePaint, SD-Inpaint) can generate fine textures but the content of the masked region remains uncertain.
- A fundamental paradigm shift is needed to address the inherent defects caused by information discarding.

## Method

### Overall Architecture

SmartEraser consists of three core components: (1) the Masked-Region Guidance paradigm, which takes the full image $[\mathbf{m}, \mathbf{x}]$ rather than $[\mathbf{m}, \mathbf{x} \odot (1-\mathbf{m})]$ as model input; (2) the million-scale synthetic Syn4Removal dataset, constructed by pasting instances onto background images to form (input image, mask, GT background) triplets; (3) a fine-tuning framework based on SD v1.5, integrating mask augmentation and CLIP visual guidance.

### Key Design 1: Masked-Region Guidance Paradigm

**Function**: Retains masked region information as a precise guide for the object to be removed, avoiding object regeneration and contextual destruction.

**Mechanism**: The conditional input for the traditional paradigm is $[\mathbf{m}, \mathbf{x} \odot (1-\mathbf{m})]$, which replaces the masked region with placeholders such as black pixels. This paper changes it to $[\mathbf{m}, \mathbf{x}]$, where the mask $\mathbf{m}$ identifies the removal area but does not discard the original pixels. The model's learning goal changes from $P(\hat{\mathbf{x}} | \mathbf{m}, \mathbf{x} \odot (1-\mathbf{m}); \Theta')$ to $P(\hat{\mathbf{x}} | \mathbf{m}, \mathbf{x}; \Theta)$. Since the mask is usually larger than the target object, the model can directly replicate the context surrounding the target object inside the masked region.

**Design Motivation**: It is more reasonable to let the model "know" what to remove than to let it "guess" what to fill. However, this introduces a shortcut problem—the model might directly copy the masked content instead of removing it, which is resolved by the Syn4Removal dataset.

### Key Design 2: Syn4Removal Synthetic Dataset

**Function**: Provides 1 million high-quality training triplets (input image, mask, ground-truth background) to train the Masked-Region Guidance model without shortcuts.

**Mechanism**: Object instances are cropped from public instance segmentation datasets, and low-quality instances are filtered using CLIP similarity. High-quality background images are selected from COCONut and SAM-1B. Scaling ratios are sampled according to the normalized area ratio $\mathcal{N}(\mu_c, \sigma_c^2)$ of each category. The feasible pasting region $\mathbf{R}_f = \mathbf{R}_1 \cap \mathbf{R}_2$ (IoU constraints + boundary constraints) is computed, and images are blended using Alpha Blending: $\mathbf{x} = \alpha \odot \mathbf{x}_i + (1-\alpha) \odot \mathbf{x}_b$.

**Design Motivation**: The pasted object exists in the input but not in the GT, forcing the model to learn true "removal" rather than "copying." The real background as GT ensures the model learns the realistic image distribution.

### Key Design 3: Mask Augmentation and CLIP Visual Guidance

**Function**: Improves model robustness to varied user mask shapes and leverages visual features to enhance semantic understanding of the object to be removed.

**Mechanism**: Mask augmentation uses 6 deformation methods: original mask, eroded mask, dilated mask, convex hull mask, ellipse mask, and bounding box + Bezier curve mask, denoted as $\mathbf{m} = \text{ME}_i(\bar{\mathbf{m}}), i \in \{1,...,6\}$. CLIP visual guidance extracts features from the masked region $\nu_\theta(\mathbf{x} \odot \mathbf{m})$ utilizing a CLIP visual encoder, maps them to the text feature space via a trainable MLP, and concatenates them with the text embedding of "Remove the instance of": $\hat{\mathbf{c}} = [\tau_\theta(\mathbf{y}), \text{MLP}(\nu_\theta(\mathbf{x} \odot \mathbf{m}))]$.

**Design Motivation**: User mask shapes are highly variable; simulating different shapes during training reduces the train-inference gap. CLIP guidance provides semantic information for the diffusion model regarding "what to remove."

### Loss & Training

Standard diffusion model training loss:
$$\mathcal{L} = \mathbb{E}_{\mathbf{x}, \mathbf{m}, \mathbf{x}_b, t, \epsilon} \| \epsilon - \epsilon_\theta(\mathbf{z}_t, \bar{\mathbf{z}}, \mathbf{m}, \hat{\mathbf{c}}, t) \|_2^2$$
where $\bar{\mathbf{z}} = E(\mathbf{x})$ represents the VAE latent code of the input image, and $\mathbf{z}_t$ represents the noised GT latent code.

## Key Experimental Results

### Main Results: Comparison of Object Removal on RORD-Val

| Method | FID ↓ | CMMD ↓ | ReMOVE ↑ | LPIPS ↓ | SSIM ↑ | PSNR ↑ |
|------|-------|--------|----------|---------|--------|--------|
| **SmartEraser** | **16.03** | **0.092** | **0.937** | **0.276** | **0.612** | **19.99** |
| PowerPaint | 24.06 | 0.294 | 0.926 | 0.308 | 0.602 | 18.10 |
| CLIPAway | 25.46 | 0.123 | 0.915 | 0.333 | 0.577 | 17.43 |
| LaMa | 24.24 | 0.216 | 0.916 | 0.348 | 0.557 | 16.38 |
| SD-Inpaint | 69.50 | 0.324 | 0.857 | 0.369 | 0.537 | 16.11 |

### Ablation Study: Contribution of Each Component

| Component | FID ↓ | LPIPS ↓ | PSNR ↑ |
|------|-------|---------|--------|
| Full Model | **16.03** | **0.276** | **19.99** |
| w/o CLIP Visual Guidance | Worse | Worse | Worse |
| w/o Mask Augmentation | Worse | Worse | Worse |
| mask-and-inpaint Paradigm | Significantly Worse | Significantly Worse | Significantly Worse |

### Key Findings

- SmartEraser outperforms the state-of-the-art (SOTA) on RORD-Val, improving FID by **10.3 points** and PSNR by **1.89 dB**.
- The core value of the Masked-Region Guidance paradigm lies in simultaneously addressing both object regeneration and contextual consistency issues.
- The quality of the synthetic dataset is crucial to model performance—details such as instance filtering (via CLIP scores) and feasible region computation significantly impact the final results.
- The advantage is particularly evident in complex scenarios: removal scenes with large masks and complex compositions are major pain points for traditional methods.

## Highlights & Insights

- **Paradigm-level Innovation**: Fundamentally challenges the default assumption of "mask-and-inpaint," proving that retaining masked-region information is more effective than discarding it.
- **Ingenious Dataset Construction**: Cleverly addresses the shortcut problem of Masked-Region Guidance through a copy-paste strategy.
- **Mask Augmentation Strategy**: Six deformation methods systematically cover various mask shapes that users might provide.

## Limitations & Future Work

- Resolution limitations (512×512) based on SD v1.5 may affect performance in high-resolution scenarios.
- A distribution gap still exists between copy-paste synthetic data and real-world removal scenarios.
- Video object removal and interactive editing scenarios remain unexplored.
- Future work can extend this method to larger foundation models (such as SDXL) and more complex editing tasks.

## Related Work & Insights

- Comparisons with PowerPaint and CLIPAway demonstrate that introducing background information solely at the level of prompts or attention is insufficient to solve the fundamental problem.
- The copy-paste synthesis strategy of Syn4Removal can be generalized to other tasks requiring paired training data.
- The CLIP visual guidance mechanism can be applied to other generative editing tasks that require specifying target objects.

## Rating

⭐⭐⭐⭐ — The paradigm-level innovation is highly convincing. The simple modification of "retaining the masked region" brings significant quality improvements. Combined with a meticulously designed dataset and training strategies, the method achieves SOTA on multiple benchmarks. The overall approach is simple and elegant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)
- [\[CVPR 2025\] PicoSAM3: Real-Time In-Sensor Region-of-Interest Segmentation](picosam3_real-time_in-sensor_region-of-interest_segmentation.md)
- [\[CVPR 2025\] EdgeTAM: On-Device Track Anything Model](edgetam_on-device_track_anything_model.md)
- [\[CVPR 2025\] GleSAM: Segment Any-Quality Images with Generative Latent Space Enhancement](segment_any-quality_images_with_generative_latent_space_enhancement.md)
- [\[CVPR 2025\] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgb-d_scene_understanding_via_multi-task_adaptive_learning_and_cross-d.md)

</div>

<!-- RELATED:END -->
