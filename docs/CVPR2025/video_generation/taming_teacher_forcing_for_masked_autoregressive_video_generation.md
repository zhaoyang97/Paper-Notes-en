---
title: >-
  [Paper Note] Taming Teacher Forcing for Masked Autoregressive Video Generation
description: >-
  [CVPR 2025][Video Generation][Autoregressive] MAGI proposes the Complete Teacher Forcing (CTF) paradigm, which conditions on fully observed frames rather than masked frames during training. This eliminates the training-inference gap, improves FVD by 23%, and enables the generation of over 100 coherent video frames while being trained on only 16 frames.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Autoregressive"
  - "Masked Modeling"
  - "Teacher Forcing"
  - "Exposure Bias"
date: 2026-05-08
content_hash: fe0c5e5df70f4917
---

# Taming Teacher Forcing for Masked Autoregressive Video Generation

**Conference**: CVPR 2025  
**arXiv**: [2501.12389](https://arxiv.org/abs/2501.12389)  
**Code**: [Project Page](https://MAGI-Video-Generation.Github.Io)  
**Area**: Video Understanding / Video Generation  
**Keywords**: Video Generation, Autoregressive, Masked Modeling, Teacher Forcing, Exposure Bias

## TL;DR

MAGI proposes the Complete Teacher Forcing (CTF) paradigm, which conditions on fully observed frames rather than masked frames during training. This eliminates the training-inference gap, improves FVD by 23%, and enables the generation of over 100 coherent video frames while being trained on only 16 frames.

## Background & Motivation

The "generation order" problem in autoregressive video generation has been largely overlooked. Prior methods are categorized into two types based on prediction granularity:
- **Patch-level methods** (VideoGPT, Emu3) use raster-scan order, which has been shown to be suboptimal by image generation studies.
- **Frame-level masked methods** (MAGViT, Genie) use bidirectional attention but cannot leverage KV Cache, incurring high computational overhead.
- Genie and Diffusion Forcing condition on masked/noisy frames, introducing **training-inference inconsistency**: conditioning on masked frames during training, but on fully generated frames during inference.
- GameNGen uses fixed-length conditional frames, lacking the flexibility of variable-length contexts.
- **Exposure Bias**: The model always observes ground-truth (GT) frames during training but must rely on its own predictions during inference, leading to error accumulation and degraded long video quality.
- Key Insight: The implementation of traditional teacher forcing in frame-level video generation (Masked Teacher Forcing, or MTF) fundamentally deviates from the original intent of teacher forcing.

## Method

### Overall Architecture

MAGI is a hybrid video generation framework: it uses causal modeling (autoregressive) across frames and masked modeling (MAR style) within frames. Fully observed frames are prepended before each frame as the complete context, utilizing a cross-attention mask to implement CTF. The Transformer decoder consists of alternating 2D spatial attention and 1D temporal attention layers, with a diffusion head on top to predict masked tokens.

### Key Designs

#### Key Design 1: Complete Teacher Forcing (CTF)

**Function**: To eliminate the training-inference gap in frame-level autoregressive training, ensuring the model conditions on fully observed frames during both training and inference.

**Mechanism**: Traditional Masked Teacher Forcing (MTF) predicts $p(f_j^m | f_1^m, f_2^m, ..., f_{j-1}^m; \theta)$ during training, which conditions on masked frames—a scenario that never occurs during inference (where conditional frames are complete). CTF changes this to $p(f_j^m | f_1, f_2, ..., f_{j-1}; \theta)$, conditioning on **fully observed frames**. Implementation: Fully observed frames are prepended to the input sequence, and a specialized temporal attention mask is designed. Causal attention is applied among observed frames, while the attention range of each masked frame includes all previous fully observed frames and itself.

**Design Motivation**: Although the high masking rate of MTF (70-100%) benefits frame quality (low FID), it severely damages temporal coherence (high FVD) because the model does not observe sufficient historical information during training. CTF trains the model to utilize complete history, yielding a 23% improvement in FVD.

#### Key Design 2: Dynamic Interval Training

**Function**: To enhance the model's ability to handle different temporal frequencies and large motion ranges, alleviating exposure bias.

**Mechanism**: During training, video segments with varying frame intervals are randomly sampled, forcing the model to learn longer temporal dependencies and larger motion ranges. To support controllable generation, a **learnable interval embedding** is introduced (vocabulary size of 25, covering intervals of 1-25 frames), encoding the interval information into a specific embedding added to the hidden states. During inference, the frame interval can be specified to control the motion speed.

**Design Motivation**: Fixed-interval training restricts model generalization; dynamic intervals introduce diversity to the data distribution. The interval embedding resolves the issue of uncontrollable motion ranges caused by naive dynamic intervals.

#### Key Design 3: Dynamic Noise Injection

**Function**: To simulate error accumulation during inference by adding noise to observed frames during training, thereby improving robustness.

**Mechanism**: Random Gaussian noise is added to the observed frames (noise levels 1-5), and a learnable noise level embedding is concatenated to the hidden states, allowing the model to perceive the current noise level. During inference, the noise level is set to 0, and the model automatically adapts to clean inputs.

**Design Motivation**: Out-of-distribution drift caused by teacher forcing—observing clean GT during training but noisy self-predictions during inference. Noise-injected training bridges this gap.

### Loss & Training

MAR-style diffusion head loss: Denoising diffusion training is performed on the masked tokens. A 64-step iterative inference is used to generate the masked tokens for each frame.

## Key Experimental Results

### Main Results: UCF-101 First-Frame Conditional Video Prediction

| Method | FVD ↓ | Description |
|------|-------|------|
| **MAGI (CTF)** | **Best** | ~23% better than MTF |
| MAGI (MTF) | Poor | Good frame quality but poor temporal coherence |
| VideoGPT | Poor | Patch-level autoregression |
| Diffusion Forcing | Medium | Noisy conditional frames |

### Ablation Study: Training Strategy

| Configuration | FVD ↓ | FID ↓ |
|------|-------|-------|
| CTF + Interval Training + Noise Injection | **Best** | **Best** |
| CTF + Interval Training Only | Poor | Poor |
| CTF + Noise Injection Only | Poor | Poor |
| CTF (No Strategy) | Worst | Worst |

### Key Findings

- The FVD of CTF is **23%** better than that of MTF, even though the frame-level FID of MTF is slightly superior. This indicates that CTF captures motion better, whereas MTF generates high-quality static frames but lacks temporal coherence.
- Both dynamic interval training and noise injection are effective for both CTF and MTF, with CTF consistently outperforming MTF.
- MAGI can generate over 100 frames of coherent video while being trained on only 16 frames.
- KV Cache enables the inference speed of MAGI to scale only linearly with the number of frames.

## Highlights & Insights

- **Training-inference consistency** is crucial for autoregressive video generation, which CTF achieves through a simple attention mask design.
- The **trade-off between FVD and FID** reveals an important insight: high single-frame quality does not equate to high-quality video.
- Excellent **length generalization** capability: 16-frame training -> 100+ frame inference, which benefits from the consistency design of CTF.

## Limitations & Future Work

- Current evaluation is primarily conducted on small-scale datasets like UCF-101; performance on larger-scale training remains to be verified.
- The 256×256 resolution limits practical applications.
- The 64-step iterative inference of the diffusion head still poses a speed bottleneck.
- Combination with text-conditional generation has not yet been explored.

## Related Work & Insights

- CTF provides a direct improvement for methods using MTF (e.g., Genie)—simply modifying the training paradigm can significantly enhance temporal coherence.
- The concepts of interval embedding and noise level embedding can be generalized to other conditional control scenarios.
- The hybrid scheme of MAR + causal temporal modeling provides a new design space for autoregressive video generation.

## Rating

⭐⭐⭐⭐ — This work clearly identifies the overlooked issue of the training-inference gap in MTF. The solution proposed by CTF is simple and effective. The 23% FVD improvement and length generalization from 16 to 100+ frames are highly impressive. The dynamic interval training and noise injection strategies are also highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](../../NeurIPS2025/video_generation/self_forcing_bridging_the_train-test_gap_in_autoregressive_video_diffusion.md)
- [\[CVPR 2025\] Parallelized Autoregressive Visual Generation](parallelized_autoregressive_visual_generation.md)
- [\[CVPR 2025\] VideoGuide: Improving Video Diffusion Models without Training Through a Teacher's Guide](videoguide_improving_video_diffusion_models_without_training_through_a_teachers_.md)
- [\[CVPR 2025\] ReCapture: Generative Video Camera Controls for User-Provided Videos Using Masked Video Fine-Tuning](recapture_generative_video_camera_controls_for_user-provided_videos_using_masked.md)
- [\[ICML 2026\] Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](../../ICML2026/video_generation/light_forcing_accelerating_autoregressive_video_diffusion_via_sparse_attention.md)

</div>

<!-- RELATED:END -->
