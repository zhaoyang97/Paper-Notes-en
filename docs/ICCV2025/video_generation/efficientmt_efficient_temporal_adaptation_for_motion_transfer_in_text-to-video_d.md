---
title: >-
  [Paper Note] EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models
description: >-
  [Video Generation] This paper proposes EfficientMT, an efficient end-to-end video motion transfer framework that reuses a pretrained T2V model backbone to extract temporal motion features, combines a scaler module with a temporal integration mechanism, and achieves zero-shot motion transfer using only a small amount of synthetic paired data. The inference speed is more than 10× faster than optimization-based methods.
tags:
  - Video Generation
date: 2026-05-08
content_hash: 3d832ff8f83097ab
---

# EfficientMT: Efficient Temporal Adaptation for Motion Transfer in Text-to-Video Diffusion Models

## Paper Information

- **Conference**: ICCV 2025
- **arXiv**: 2503.19369
- **Code**: [https://github.com/PrototypeNx/EfficientMT](https://github.com/PrototypeNx/EfficientMT)
- **Area**: Video Generation / Motion Transfer
- **Keywords**: motion transfer, text-to-video, diffusion model, temporal attention, end-to-end

## TL;DR

This paper proposes EfficientMT, an efficient end-to-end video motion transfer framework that reuses a pretrained T2V model backbone to extract temporal motion features, combines a scaler module with a temporal integration mechanism, and achieves zero-shot motion transfer using only a small amount of synthetic paired data. The inference speed is more than 10× faster than optimization-based methods.

## Background & Motivation

Video motion transfer aims to transfer the motion patterns of a reference video to different subjects and scenes. Existing methods fall into two categories:

**Dense visual conditioning methods** (VideoComposer, Control-A-Video, etc.): These methods use dense conditions such as depth maps and optical flow for end-to-end training. Inference is efficient, but the structural constraints are too strong to generalize across scenes.

**Optimization-based methods** (MotionDirector, MotionClone, MOFT, etc.): These methods extract implicit motion representations from T2V models and achieve strong results, but require per-reference-video optimization (ranging from minutes to tens of minutes).

The root cause of this tension is that end-to-end methods are efficient but lack transferability, while optimization-based methods transfer well but incur high computational cost. EfficientMT aims to combine the strengths of both — leveraging implicit motion representations for robust transfer while maintaining end-to-end inference efficiency.

## Method

### Overall Architecture

EfficientMT is built upon two T2V backbone models, AnimateDiff and VideoCrafter2, and consists of three core components:

1. **Motion feature extraction**: Reuses the T2V backbone as a feature extractor for the reference video.
2. **Scaler module**: Performs fine-grained scaling of reference features to filter out motion-irrelevant information.
3. **Temporal integration mechanism**: Injects the filtered reference motion features into the temporal attention layers of the generation process.

### Key Design 1: Motion Representation Extraction

The pretrained T2V backbone $\hat{\epsilon}$ is directly reused to extract inputs to the temporal attention layers of the reference video as motion features:

$$\Gamma = \hat{\epsilon}(x, t, \tau_\theta(y^r)) = \{f\}_{upblocks}^{temporal}$$

where $t$ is set to a late denoising timestep, and the reference prompt $y^r$ is set to empty text to better capture motion dynamics. This design ensures that the extracted features are naturally aligned with the generation process, eliminating the need for a separately trained feature extractor.

### Key Design 2: Scaler Module

Reference features contain substantial motion-irrelevant information (e.g., texture, shape); directly injecting them leads to overfitting. The scaler predicts a fine-grained scaling map:

$$\alpha = \mathcal{S}(f^r \otimes f^g) \in \mathbb{R}^{h \times w \times n \times 1}$$

where $f^r$ and $f^g$ are the reference and generation features, respectively, and $\otimes$ denotes channel-wise concatenation. The scaled feature $\hat{f}^r = \alpha \cdot f^r$ adaptively preserves motion-relevant information while suppressing irrelevant signals.

Visualization analysis shows that in low-resolution blocks, the scaler tends to either fully select or fully discard features (scale values close to 0 or 1), whereas in high-resolution blocks, it selectively attends to motion-relevant regions (e.g., moving heads, walking legs).

### Key Design 3: Temporal Integration Mechanism

The filtered reference features are concatenated with the generation features along the temporal axis:

$$f^{int} = f^g \otimes \hat{f}^r \in \mathbb{R}^{h \times w \times 2n \times c}$$

The query is projected from the original $f^g$, while keys and values are projected from the integrated feature $f^{int}$:

$$Q = W_Q \cdot f^g, \quad K = W_K \cdot f^{int}, \quad V = W_V \cdot f^{int}$$

This is equivalent to treating reference frames as additional video frames, providing motion guidance within temporal attention. During training, only the temporal attention layers in all upsampling blocks are fine-tuned.

### Loss & Training

The standard diffusion loss is used:

$$\mathcal{L} = \mathbb{E}\left[\|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y), \Gamma)\|_2^2\right]$$

**Training data construction**: 217 real reference videos are collected, and paired motion transfer data are synthesized using MotionClone and MotionInversion. After dual filtering based on motion fidelity score and temporal consistency (inter-frame CLIP features) along with manual selection, approximately 150 high-quality samples are retained.

## Key Experimental Results

### Main Results: Quantitative Comparison

| Method | Temporal Consistency ↑ | Text Alignment ↑ | Motion Fidelity ↑ | Time Cost ↓ |
|------|----------------------|-----------------|-------------------|------------|
| ControlVideo | 0.9213 | 0.2483 | 0.5533 | 80s |
| VideoComposer | 0.9192 | 0.2635 | 0.6356 | 18s |
| MotionDirector | 0.9327 | 0.2525 | 0.8361 | 473s |
| MotionClone | 0.9108 | 0.2637 | 0.8569 | 190s |
| MOFT | 0.9283 | 0.2581 | 0.7698 | 127s |
| **Ours (AnimateDiff)** | **0.9291** | **0.2712** | **0.8470** | **16s** |
| DMT | 0.9275 | 0.2479 | 0.6974 | 203s |
| MotionInversion | 0.9329 | 0.2558 | 0.7373 | 418s |
| **Ours (VideoCrafter2)** | **0.9456** | **0.2677** | **0.7116** | **21s** |

EfficientMT achieves the best text alignment, motion fidelity comparable to optimization-based methods, and an inference time of only 16–21 seconds (versus 127–473 seconds for optimization-based approaches).

### Ablation Study

| Configuration | TC ↑ | TA ↑ | MF ↑ |
|------|------|------|------|
| w/o Scaler | 0.9244 | 0.2638 | 0.8135 |
| w/o Data Filter | 0.9237 | 0.2649 | 0.8278 |
| Inject Upblock.1 | 0.9013 | 0.2789 | 0.6374 |
| Inject Upblock.1,2 | 0.9198 | 0.2755 | 0.7236 |
| **Full Model** | **0.9291** | **0.2712** | **0.8470** |

**Key Findings**:
- Removing the scaler degrades motion fidelity (0.847→0.814) and introduces artifacts in the generated results.
- Injecting into only a single upsampling block (upblock1) causes a sharp drop in motion fidelity to 0.637.
- Data quality is more important than data quantity: omitting data filtering degrades all metrics.

### User Study

Compared against VideoComposer (VC), MotionDirector (MD), MotionClone (MC), and MotionInversion (MI), EfficientMT receives majority user preference (55–80%) across temporal consistency, text alignment, and motion fidelity.

## Highlights & Insights

1. **Efficient reuse strategy**: The T2V backbone is reused as a feature extractor without requiring an additional encoder, resulting in a small number of trainable parameters.
2. **Training with minimal data**: Only approximately 150 high-quality synthetic paired samples are needed for training.
3. **10× speedup**: 16 seconds versus 127–473 seconds for optimization-based methods, making motion transfer practically applicable.
4. **Scaler visualization analysis**: Reveals differentiated processing patterns for motion information across layers of different resolutions.

## Limitations & Future Work

- In scenes with drastic motion changes (e.g., rapid rotation, abrupt displacement), the generated results are prone to tearing artifacts.
- Transfer quality depends on the clarity of motion in the reference video.
- The method is built on AnimateDiff/VideoCrafter2 and has not been validated on newer architectures (e.g., DiT-based T2V models).

## Related Work & Insights

- **Relationship with MotionClone/MotionInversion**: These methods are used to generate training data (teacher), which is then used to train the end-to-end model (student) — essentially a form of knowledge distillation.
- **Scaler module design**: Inspired by SmartControl, replacing coarse-grained global injection strength adjustment with fine-grained scaling.
- **KV concatenation in temporal attention**: Conceptually similar to the spatial attention injection approach in IP-Adapter, but applied along the temporal dimension.

## Rating

⭐⭐⭐⭐ — A highly practical work that compresses motion transfer from minute-level optimization to second-level inference. The design with minimal training data is also elegant. The main limitation is that only two relatively dated T2V baselines are evaluated, with no adaptation to more recent video generation architectures.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [\[ICCV 2025\] LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models](leanvae_an_ultra-efficient_reconstruction_vae_for_video_diffusion_models.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[ICCV 2025\] Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](prompt-a-video_prompt_your_video_diffusion_model_via_preference-aligned_llm.md)
- [\[ICCV 2025\] DIVE: Taming DINO for Subject-Driven Video Editing](dive_taming_dino_for_subject-driven_video_editing.md)

<!-- RELATED:END -->
