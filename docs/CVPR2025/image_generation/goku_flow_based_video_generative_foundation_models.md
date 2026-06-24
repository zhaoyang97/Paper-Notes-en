---
title: >-
  [Paper Note] Goku: Flow Based Video Generative Foundation Models
description: >-
  [CVPR 2025][Image Generation][rectified flow] Goku is a family of rectified flow Transformer models (2B/8B) proposed by ByteDance and HKU, marking the first application of rectified flow to joint image-video generation. Assisted by a comprehensive data curation pipeline and large-scale training infrastructure optimization, Goku achieves state-of-the-art (SOTA) performance on benchmarks such as VBench (84.85) and GenEval (0.76).
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "rectified flow"
  - "video generation"
  - "joint image-video"
  - "3D VAE"
  - "Transformer"
  - "data curation"
date: 2026-05-08
content_hash: 59d472f0324cd54b
---

# Goku: Flow Based Video Generative Foundation Models

**Conference**: CVPR 2025  
**arXiv**: [2502.04896](https://arxiv.org/abs/2502.04896)  
**Code**: [Project Page](https://saiyan-world.github.io/goku/)  
**Area**: Image Generation  
**Keywords**: rectified flow, video generation, joint image-video, 3D VAE, Transformer, data curation

## TL;DR

Goku is a family of rectified flow Transformer models (2B/8B) proposed by ByteDance and HKU, marking the first application of rectified flow to joint image-video generation. Assisted by a comprehensive data curation pipeline and large-scale training infrastructure optimization, Goku achieves state-of-the-art (SOTA) performance on benchmarks such as VBench (84.85) and GenEval (0.76).

## Background & Motivation

**Background**: Video generation has achieved remarkable progress, benefiting from advanced generative algorithms (GANs, diffusion, flow), scalable architectures (Transformers), massive internet data, and rising computational power. However, industrial-grade joint image-video generative models still face multi-dimensional challenges.

**Limitations of Prior Work**:
- Early methods separate temporal and spatial attention (temporal+spatial), making it difficult to model complex temporal motion.
- DDPM suffers from slow convergence, leading to prohibitive training costs for large-scale models.
- The acquisition cost of high-quality video data is much higher than that of image data, resulting in a prominent data imbalance issue.
- Training long sequences (over 220K tokens) requires highly efficient parallelization and memory management strategies.

**Key Challenge**: Joint image-video training requires simultaneously learning spatial semantics from images and temporal motion dynamics from videos; thus, direct joint optimization is highly challenging.

**Goal**: To build a complete industrial-grade joint image-video generation pipeline, optimizing the full chain across data, model, training formulations, and infrastructure.

**Key Insight**: Adopting rectified flow to replace DDPM, utilizing a full-attention Transformer and a 3D joint VAE, coupled with a multi-stage progressive resolution training strategy.

**Core Idea**: Unifying image-video generation via rectified flow and a full-attention Transformer, achieving industrial-grade quality through precise data curation pipelines and multi-stage training.

## Method

### Overall Architecture

1. **3D Joint VAE**: Compress images/videos from pixel space into a shared latent space (video compression ratio of 8×8×4, image ratio of 8×8).
2. **Rectified Flow Transformer**: Model a linear interpolation flow on the latent space to jointly train image and video tokens.
3. **Multi-stage Training**: Text-semantic pairing $\rightarrow$ joint image-video learning $\rightarrow$ modality-specific fine-tuning.
4. **Efficient Infrastructure**: Sequence parallelism + FSDP + selective activation checkpointing + MegaScale fault tolerance.

### Key Designs

#### 1. Full-Attention Transformer Architecture

Abandoning conventional temporal+spatial factory attention, the model directly applies **plain full attention** over all image and video tokens. Key enhancements include:
- **Patch n' Pack**: Inspired by NaViT, samples of different resolutions and durations are packed into a single batch along the sequence dimension, eliminating the need for data bucketing.
- **3D RoPE**: Applies three-dimensional rotary position embeddings to image/video tokens, supporting resolution extrapolation and converging faster than sinusoidal embeddings.
- **Q-K Normalization**: Applies RMSNorm to queries and keys before attention computation to prevent training instability and model collapse caused by loss spikes.
- Model scale: Goku-2B (28 layers, dim=1792, 28 heads) and Goku-8B (40 layers, dim=3072, 48 heads).

#### 2. Rectified Flow Training Formulation

The forward process is defined as a linear interpolation between data and noise: $\mathbf{x}_t = t \cdot \mathbf{x}_1 + (1-t) \cdot \mathbf{x}_0$. The model learns to predict the velocity $\mathbf{v}_t = d\mathbf{x}_t / dt$. Compared to DDPM, RF provides a more direct interpolation path, superior theoretical properties, and faster convergence.

#### 3. Multi-Stage Progressive Training Strategy

- **Stage 1 (Text-Semantic Pairing)**: Text-to-image pre-training to establish a solid foundation for semantic-to-visual mapping.
- **Stage 2 (Joint Learning)**: Joint image-video training, utilizing full attention to unify cross-modal representations; high-quality image data assists in enhancing video frame quality; cascaded resolutions are employed (288×512 $\rightarrow$ 480×864 $\rightarrow$ 720×1280).
- **Stage 3 (Modality-Specific Fine-Tuning)**: Fine-tuning specifically targetting T2I and T2V independently to improve output quality in each modality.

### Loss & Training

Standard rectified flow velocity prediction loss: $\mathcal{L} = \mathbb{E}_{t,\mathbf{x}_0,\mathbf{x}_1}[\|\mathbf{v}_t - f_\theta(\mathbf{x}_t, t)\|^2]$

### Data Pipeline

- **Scale**: 160M image-text pairs + 36M video-text pairs.
- **Video Processing**: Preprocessing standardization $\rightarrow$ coarse splitting with PySceneDetect $\rightarrow$ fine splitting with DINOv2 frame-to-frame similarity $\rightarrow$ aesthetic scoring/OCR/motion filtering.
- **Captioning**: InternVL2.0 keyframe captioning + Tarsier2 video captioning $\rightarrow$ refinement and merging via Qwen2.
- **Data Balancing**: Semantics annotated using video classification models, performing upsampling/downsampling to balance 9 macro-categories and 86 subcategories.

## Key Experimental Results

### Main Results

| Task | Benchmark | Goku Score | Rank |
|------|-----------|------------|------|
| T2I  | GenEval   | **0.76**   | SOTA |
| T2I  | DPG-Bench | **83.65**  | SOTA |
| T2V  | VBench    | **84.85**  | Rank 1 (2025-01-25) |
| T2V  | UCF-101 Zero-shot | SOTA | - |

**T2I Comparison**: Surpasses SD3 (GenEval 0.74), DALL-E 3 (GenEval 0.67), and Emu 3 (0.66).

### Ablation Study (ImageNet 256×256 Class-Conditional Generation)

Validation of Rectified Flow convergence speed:

| Loss | Steps | FID ↓ | IS ↑ |
|------|------|-------|------|
| DDPM | 400k | 2.52 | 265.1 |
| DDPM | 1000k | 2.26 | 286.6 |
| **RF** | **400k** | **2.16** | **261.1** |

RF reaches the FID of DDPM at 1000k steps with only 400k steps.

### Key Findings

1. Rectified flow converges approximately 2.5 times faster than DDPM.
2. Full-attention outperforms temporal-spatial separated attention, allowing modeling of more complex temporal motion.
3. 3D RoPE converges faster than sinusoidal positional encodings when transitioning across training stages.
4. Data balancing significantly impacts the generation quality of human-focused classes.
5. Checkpoint saving for the 8B model only blocks training for about 4 seconds.

## Highlights & Insights

1. **Industrial-Grade Complete Solution**: Co-designing data, models, training, and infrastructure across the full stack, extending beyond pure algorithmic innovations.
2. **First Application of RF to Joint Image-Video Generation**: Validate the feasibility and advantages of rectified flow in the realm of video generation.
3. **Flexible Packaging via Patch n' Pack**: Completely resolves the batching challenges of variable-resolution and variable-duration data.
4. **Unique Data-Driven Perspective**: Discloses detailed video filtering thresholds (aesthetic score $\ge 4.3/4.5$, motion boundaries $0.3\text{--}20.0$, etc.), providing highly valuable references for engineering practices.
5. **Appending Motion Scores to Captions**: Employs an elegant and effective approach by embedding motion scores inside captions to control generation dynamics.

## Limitations & Future Work

1. Code and model weights are not open-sourced, limiting reproducibility.
2. Subjective evaluation of video quality relies heavily on human rating, reflecting the lack of unified automated video quality metrics.
3. Image-to-video generation only supports first-frame conditioning instead of more flexible multi-frame referencing.
4. The data pipeline heavily depends on massive proprietary internal datasets (60M images + 25M videos), which is hard for the community to replicate.
5. Motion controllability is only steered via the motion score in the caption, which offers limited granularity.

## Related Work & Insights

1. **Sora** (Brooks et al., 2024): First introduced the idea of utilizing 3D VAEs to compress videos into latent spaces, which Goku inherits and refines.
2. **GenTron** (Chen et al., 2024): The origin of the foundation design for the Goku Transformer blocks.
3. **NaViT** (Dehghani et al., 2024): The inspiration for the Patch n' Pack flexible packing strategy.
4. **InternVL2.0**: Used for generating high-quality image and video captions.
5. **MegaScale** (Jiang et al., 2024): Large-scale training fault-tolerance mechanism.

**Insights**: The advantages of rectified flow in ultra-large generative models (rapid convergence) could drive more research to migrate from DDPMs to flow-based formulations. The practice of embedding motion scores in captions within the data pipeline provides highly valuable reference.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ — Employs RF for the first time in joint image-video generation. Although individual components are not entirely novel, their combination represents solid engineering innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensively evaluated across multiple benchmarks, though lacking deep comparisons with open-source models like CogVideoX.
- **Writing Quality**: ⭐⭐⭐⭐ — Written in a technical report style with a clear structure and rich engineering details.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a comprehensive reference for the data, training, and infrastructure of industrial-grade video generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Can Generative Video Models Help Pose Estimation?](can_generative_video_models_help_pose_estimation.md)
- [\[CVPR 2025\] VLog: Video-Language Models by Generative Retrieval of Narration Vocabulary](vlog_video-language_models_by_generative_retrieval_of_narration_vocabulary.md)
- [\[CVPR 2025\] ObjectMover: Generative Object Movement with Video Prior](objectmover_generative_object_movement_with_video_prior.md)
- [\[CVPR 2025\] Visual Persona: Foundation Model for Full-Body Human Customization](visual_persona_foundation_model_for_full-body_human_customization.md)
- [\[CVPR 2025\] FADE: Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fade_fine_grained_erasure_diffusion.md)

</div>

<!-- RELATED:END -->
