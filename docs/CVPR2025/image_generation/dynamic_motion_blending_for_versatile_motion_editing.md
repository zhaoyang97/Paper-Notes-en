---
title: >-
  [Paper Note] Dynamic Motion Blending for Versatile Motion Editing (MotionReFit)
description: >-
  [Image Generation] MotionReFit proposes the first versatile text-guided motion editing framework. By dynamically generating training triplets via the MotionCutMix data augmentation technique, combined with an autoregressive diffusion model and a body part coordinator, it achieves spatial and temporal editing encompassing body part replacement, style transfer, and fine-grained adjustment.
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 83dcf4aad3895ccd
---

# Dynamic Motion Blending for Versatile Motion Editing (MotionReFit)

## TL;DR

MotionReFit proposes the first versatile text-guided motion editing framework. By dynamically generating training triplets via the MotionCutMix data augmentation technique, combined with an autoregressive diffusion model and a body part coordinator, it achieves spatial and temporal editing encompassing body part replacement, style transfer, and fine-grained adjustment.

## Background & Motivation

Text-guided motion editing is a fundamental task in computer animation, allowing creators to modify the semantics and style of motions using natural language. However, existing methods face three key limitations:

1. **Scarcity of Training Data**: Existing methods (such as TMED) rely on pre-collected fixed triplets (original motion, edited motion, and editing instructions). Such annotated data is extremely scarce, severely limiting generalization capabilities.
2. **Requirement of Auxiliary Information**: Current models require explicit specification of the body parts to be edited as auxiliary input, failing to autonomously comprehend high-level semantic instructions.
3. **Unsmooth Spatiotemporal Transitions**: It is difficult to guarantee smooth transitions in both spatial and temporal dimensions when generating the edited motion.

Core Problem: How to achieve spatial and temporal editing of arbitrary motions solely through text instructions without requiring extra annotations?

## Method

### Overall Architecture

MotionReFit consists of three components: (1) the MotionCutMix data augmentation strategy that dynamically generates motion editing triplets during training; (2) an autoregressive conditional diffusion model that generates the edited motion segment-by-segment in a sliding window manner; (3) a body part coordinator serving as a discriminator to provide classifier guidance, ensuring natural coordination between body parts. Additionally, the STANCE dataset is proposed to cover three editing scenarios.

### Key Designs

#### 1. MotionCutMix Data Augmentation

- **Function**: Dynamically generates training triplets from large-scale unlabeled motion databases, expanding $N_S$ annotated samples into $N_L \times N_S$ training pairs.
- **Mechanism**: Combines body parts from different motion sequences through Spatial Motion Blending to generate synthetic training samples. For semantic editing, a source motion is randomly selected from a large database, and a target motion with a mask is chosen from the annotated library to be blended. For style editing, the unedited body parts of an editing pair are replaced with HTML/external motions. Soft masks and SLERP interpolation are used to achieve smooth transitions.
- **Design Motivation**: Annotated motion editing triplets are extremely expensive (requiring paired original/edited motions and instructions), whereas large-scale unlabeled motion data is readily available. Inspired by image augmentation (CutMix), MotionCutMix achieves a similar effect in the motion domain, substantially expanding the training distribution.

#### 2. Autoregressive Diffusion Model

- **Function**: Generates edited motions segment-by-segment in a sliding window manner, supporting arbitrary sequence lengths and temporal editing.
- **Mechanism**: Based on the DDPM framework, the model $\epsilon_\theta$ predicts noise on each motion segment. Conditions include: (i) the last two frames of the previous segment $\mathcal{M}_{prev}$ to ensure temporal continuity; (ii) the original motion segment $\mathcal{M}_{ori}$; (iii) the CLIP-encoded editing instruction $\mathcal{E}$; and (iv) a progress indicator $\mathcal{P}$. Classifier-free guidance is used to enhance instruction following.
- **Design Motivation**: Direct training on long sequences is challenging and memory-constrained. Autoregressive segment-by-segment generation reduces the learning difficulty while achieving smooth transitions through conditioning on the last two frames of the previous segment. The progress indicator helps the model understand the position of the current segment in the entire sequence, avoiding repetitive patterns.

#### 3. Body Part Coordinator

- **Function**: Serves as a discriminator to distinguish "synthetic motions" from "natural motions," correcting body part incoordination during the diffusion process via classifier guidance.
- **Mechanism**: A binary discriminator $D$ is trained, with 50% of reference samples coming from unmodified HumanML3D motion segments (positive examples) and 50% synthesized by blending body parts from different motion segments (negative examples). During inference, the gradient of $D$ is applied as classifier guidance in the denoising process.
- **Design Motivation**: Although MotionCutMix expands the training distribution, the randomness introduced by blending motions might generate unnatural coordination patterns (e.g., walking with ipsilateral hand and foot synchronization). The discriminator learns coordination patterns of natural motions, guiding the generation away from "synthetic" patterns during inference.

### Loss & Training

The training loss for the diffusion model is the standard MSE:

$$\mathcal{L} = \mathbb{E}_{\mathcal{M}_0 \sim q(\mathcal{M}_0|\mathcal{C}), t \sim [1,T]} \|\epsilon - \epsilon_\theta(\mathcal{M}_t, t, \mathcal{C})\|_2^2$$

During inference, classifier-free guidance is used:

$$\tilde{\epsilon}_\theta = (1+w)\epsilon_\theta(\mathcal{M}_t, t, \mathcal{C}) - w\epsilon_\theta(\mathcal{M}_t, t, \mathcal{C}')$$

combined with the classifier guidance from the body part coordinator.

## Key Experimental Results

### Main Results (Tab. 1 - Body Part Replacement)

| Method | FID↓ | Diversity→ | FS↓ | Edited-to-Target R@1↑ |
|------|------|-----------|-----|----------------------|
| MDM-BP | 0.44 | 36.71 | 0.91 | 39.05 |
| TMED | 0.52 | 35.37 | 0.90 | 42.70 |
| TMED w/ MCM | 0.54 | 35.67 | 0.90 | 50.62 |
| Ours w/o MCM | 0.23 | 36.34 | 0.96 | 51.18 |
| Ours w/o BC | 0.23 | 36.18 | 0.97 | 60.78 |
| **Ours full** | **0.20** | **36.01** | **0.97** | **61.37** |

### Ablation Study

| Variant | FID↓ | Edited-to-Source AvgR→ | Edited-to-Target R@1↑ |
|------|------|--------------|---------------|
| w/o MotionCutMix | 0.23 | 1.27 | 51.18 |
| w/o Body Coordinator | 0.23 | 7.54 | 60.78 |
| **Full model** | **0.20** | **7.46** | **61.37** |

### Key Findings

1. **Significant Effectiveness of MotionCutMix**: After adding MCM, the Edited-to-Source AvgR increases from 1.27 to 7.46 (closer to the 8.28 of real data), showing that the model learns to preserve the unedited parts of the original motion.
2. **Further Improvement with Body Coordinator**: R@1 boosts from 60.78 to 61.37, while FID decreases from 0.23 to 0.20.
3. **Consistent Improvement for TMED with MCM**: R@1 increases from 42.70 to 50.62, proving the generalizability of the data augmentation strategy.
4. SOTA performance is also achieved in style transfer tasks.

## Highlights & Insights

1. **Ingenious Mechanism of MotionCutMix**: Porting CutMix from the image domain to the motion domain, utilizing soft masks and SLERP to achieve smooth body part blending, which elegantly addresses the scarcity of motion editing triplets.
2. **First Versatile Motion Editing Framework**: Capable of simultaneously handling semantic editing (body part replacement), style editing (style transfer), and fine-grained adjustments, without requiring LLMs or additional user inputs.
3. **Value of the STANCE Dataset**: Comprising manually annotated body masks, MoCap style pairs, and MLD-generated fine-tuning pairs, providing a systematic evaluation benchmark for the motion editing community.
4. **Synergy of Autoregression and Coordinator**: Effectively resolves issues associated with long-sequence generation and the unnaturalness of synthetic motions.

## Limitations & Future Work

1. **SMPL-X Representation Constraints**: The hands are treated as rigid bodies, rendering fine-fingered motion editing impossible.
2. **Dependency on CLIP Text Encoder**: CLIP has restricted understanding of motion semantics, potentially leading to insufficient adherence to complex editing instructions.
3. **Computational Overhead**: Autoregressive segment-by-segment generation coupled with multi-step DDPM denoising makes inference slow for long sequences.
4. **Limited Dataset Scale**: STANCE's style transfer portion contains only 750 sequences (approx. 2 hours of MoCap), which may limit generalization to a wider range of styles.
5. Future directions include exploring larger motion foundation models or introducing paradigms from video diffusion models.

## Related Work & Insights

- **MDM / FineMoGen**: Pioneers of diffusion model-based motion generation, supporting inpainting but lacking support for unified semantic and style editing.
- **TMED**: The most relevant conditional diffusion motion editing work, though limited by training on fixed triplets.
- **CutMix / MixUp**: Image data augmentation methods, which inspired the application of MotionCutMix to the motion domain.
- **Insight**: Data augmentation strategies could be crucial to breaking the bottleneck of annotated data. The idea of soft-masked blending can be extended to other sequential data editing tasks.

## Rating: ⭐⭐⭐⭐

The MotionCutMix augmentation strategy is novel and practical, the problem definition is comprehensive (covering three editing scenarios), and the accompanying STANCE dataset is valuable. One star is deducted due to coarse hand representations, the need for inference efficiency optimizations, and the relatively limited gains from the body part coordinator in the ablation study.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Less is More: Improving Motion Diffusion Models with Sparse Keyframes](../../ICCV2025/image_generation/less_is_more_improving_motion_diffusion_models_with_sparse_keyframes.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](../../ICCV2025/image_generation/motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[ICCV 2025\] EEdit: Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing](../../ICCV2025/image_generation/eedit_rethinking_the_spatial_and_temporal_redundancy_for_efficient_image_editing.md)
- [\[CVPR 2025\] Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fine-grained_erasure_in_text-to-image_diffusion-based_foundation_models.md)
- [\[CVPR 2025\] FlipSketch: Flipping Static Drawings to Text-Guided Sketch Animations](flipsketch_flipping_static_drawings_to_text-guided_sketch_animations.md)

</div>

<!-- RELATED:END -->
