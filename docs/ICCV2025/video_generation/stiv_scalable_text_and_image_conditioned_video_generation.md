---
title: >-
  [Paper Note] STiV: Scalable Text and Image Conditioned Video Generation
description: >-
  [ICCV 2025][Video Generation][Diffusion Transformer] This paper proposes STIV, a unified text-image conditioned video generation framework based on Diffusion Transformer. It integrates image conditioning via a frame replacement strategy and introduces joint image-text classifier-free guidance, enabling both T2V and TI2V generation within a single model. The 8.7B-parameter model achieves state-of-the-art scores of 83.1 and 90.1 on VBench T2V and I2V, respectively.
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "Diffusion Transformer"
  - "image conditioning"
  - "text-to-video"
  - "scalable training"
date: 2026-05-08
content_hash: a8cee1644ad92430
---

# STiV: Scalable Text and Image Conditioned Video Generation

**Conference**: ICCV 2025
**arXiv**: [2412.07730](https://arxiv.org/abs/2412.07730)  
**Code**: N/A  
**Area**: Video Generation
**Keywords**: Video generation, Diffusion Transformer, image conditioning, text-to-video, scalable training

## TL;DR
This paper proposes STIV, a unified text-image conditioned video generation framework based on Diffusion Transformer. It integrates image conditioning via a frame replacement strategy and introduces joint image-text classifier-free guidance, enabling both T2V and TI2V generation within a single model. The 8.7B-parameter model achieves state-of-the-art scores of 83.1 and 90.1 on VBench T2V and I2V, respectively.

## Background & Motivation
Video generation has advanced rapidly in the wake of Sora, with the Diffusion Transformer (DiT) architecture becoming the dominant paradigm. However, achieving Sora-level video generation still poses multiple challenges:

**Limitation 1: Unclear integration of image conditioning.** How to effectively incorporate image conditions into DiT architectures remains unsettled. U-Net-based approaches (e.g., ConsistI2V) require additional spatial self-attention and windowed temporal attention, which is inelegant.

**Limitation 2: Instability in large-scale training.** As model scale increases, training instability and memory consumption become primary bottlenecks. Techniques such as QK-norm prove insufficient for larger models.

**Limitation 3: Lack of a systematic recipe.** Existing works typically study individual aspects (architectural design, training strategies, data processing) in isolation, without systematically examining their interactions.

**Core Idea: Provide a transparent and scalable video generation recipe** that builds progressively from T2I to T2V and then to TI2V, with frame replacement as the core design for image conditioning, validated through comprehensive ablation studies.

## Method

### Overall Architecture
STIV is built upon the PixArt-α architecture, using a frozen VAE to encode video frames into spatiotemporal latent vectors, processed by stacked DiT-like blocks. T5 and CLIP encoders handle text prompts. The progressive training pipeline proceeds as: T2I → T2V → STIV (TI2V), with decomposed spatial-temporal attention applied over video frames.

### Key Designs
1. **Frame Replacement**:

    - Function: During training, the first frame of the noisy video latent is replaced with the clean image-conditioned latent, and the loss for that frame is masked out.
    - Mechanism: The stacked spatial-temporal attention layers in the DiT architecture naturally propagate image-conditioning information to subsequent frames through attention, without requiring additional cross-attention or projection layers.
    - Design Motivation: The DiT architecture inherently propagates first-frame information via self-attention, making frame replacement a **minimal yet effective** design. Ablations show that adding extra cross-attention or large projection layers improves subject/background consistency but significantly reduces dynamic degree (22.4 vs. 36.6), over-constraining the generated output.

2. **Joint Image-Text Classifier-Free Guidance (JIT-CFG)**:

    - Function: Image and text conditions are randomly dropped out during training; joint CFG is applied at inference.
    - Mechanism: The velocity field correction is formulated as $\hat{F}_\theta(x_t, c_T, c_I, t) = F_\theta(x_t, \emptyset, \emptyset, t) + s \cdot (F_\theta(x_t, c_T, c_I, t) - F_\theta(x_t, \emptyset, \emptyset, t))$, requiring only two forward passes.
    - Design Motivation: Addresses the **motion staleness issue** in high-resolution STIV models. Image condition dropout prevents the model from passively overfitting to image conditions, encouraging it to learn motion information from underlying video data. This also naturally enables multi-task training for both T2V and TI2V.

3. **Stable and Efficient Large-Scale Training**:

    - Function: A combination of techniques ensures training stability and efficiency.
    - Core techniques:
        - **QK-Norm + Sandwich-Norm**: Pre- and post-normalization applied to both MHA and FFN, combined with stateless layer normalization.
        - **MaskDiT**: Randomly masks 50% of spatial tokens during training (followed by unmasked fine-tuning), substantially reducing memory usage.
        - **AdaFactor Optimizer**: Replaces AdamW to reduce memory footprint.
        - **RoPE**: 2D RoPE for spatial attention and 1D RoPE for temporal attention, supporting resolution extrapolation.
    - Design Motivation: No single stability technique suffices for large-scale training. The three efficiency techniques (MaskDiT + AdaFactor + gradient checkpointing) must operate jointly to make training an 8.7B model feasible within reasonable resources.

4. **Progressive Training**:

    - Function: Stepwise training from T2I → T2V → STIV, from low to high resolution, and from short to long duration.
    - Mechanism: The high-resolution T2V model is initialized simultaneously from a high-resolution T2I model (spatial weights) and a low-resolution T2V model (temporal weights), with RoPE interpolation to accommodate new resolutions and durations.
    - Design Motivation: Directly training high-resolution, long-duration models is prohibitively expensive. Progressive training yields better results under the same compute budget.

### Loss & Training
- Flow Matching objective (rather than conventional diffusion loss): $\min_\theta \mathbb{E}[\|F_\theta(x_t, c, t) - v_t\|_2^2]$, where $v_t = x_1 - \epsilon$
- T2I training: 400k steps, batch size 4096; T2V/TI2V training: 400k steps, batch size 1024
- EMA decay rate: 0.9999
- After MaskDiT training with 50% masking, unmasked fine-tuning is performed for 50k/100k steps.

## Key Experimental Results

### Main Results

| Model | Parameters | Resolution | VBench T2V Total | VBench I2V | Notes |
|-------|-----------|------------|-----------------|------------|-------|
| STIV-M | 8.7B | 512² | **83.1** | **90.1** | SOTA |
| CogVideoX-5B | 5B | — | 81.6 | — | Open-source SOTA |
| Pika | — | — | 80.6 | — | Commercial product |
| Kling | — | — | 81.8 | — | Commercial product |
| Gen-3 | — | — | 82.2 | — | Commercial product |

### Ablation Study

| Configuration | VBench Quality | VBench Semantic | VBench Total | Notes |
|--------------|---------------|----------------|-------------|-------|
| Base T2V-XL | 80.19 | 70.51 | 78.25 | Baseline |
| + temporal patch=1 | 80.92 | 71.69 | 79.07 | Best but 2× compute |
| + causal temp atten | 74.59 | 73.13 | 74.30 | Large degradation |
| + temp mask | 77.58 | 65.95 | 75.25 | Severe harm from temporal masking |
| − spatial mask | 80.57 | 70.31 | 78.52 | Slight gain but higher compute |

Ablation on TI2V image conditioning integration:

| Method | I2V Avg Score | Total Avg Score | Dynamic Degree | Notes |
|--------|-------------|----------------|---------------|-------|
| Cross Attention (CA) | 68.2 | 73.0 | 42.4 | Baseline |
| CA + Large Proj | 72.3 | 75.3 | 22.2 | Over-constrained |
| Frame Replace (FR) | **75.8** | **77.3** | 36.6 | Best balance |
| FR + CA | 74.4 | 77.1 | 35.4 | No additional gain |

### Key Findings
- Frame replacement is the optimal strategy for integrating image conditions: simple, efficient, and without sacrificing dynamic degree.
- **Non-causal temporal attention substantially outperforms causal attention** (Total: 78.25 vs. 74.30), contradicting the causal design commonly attributed to Sora.
- Temporal masking severely degrades performance (−3.0 Total), whereas spatial masking incurs negligible loss.
- Image condition dropout is not merely a multi-task training technique; it is also critical for resolving the motion staleness issue at high resolution.
- Flow Matching + CFG-Renormalization constitutes the single largest performance gain factor.
- Progressive initialization leveraging both T2I and low-resolution T2V weights outperforms single-source initialization.

## Highlights & Insights
- **Minimalist design philosophy**: Both core designs—frame replacement and JIT-CFG—are extremely simple yet highly effective, embodying the engineering aesthetic of "simple but correct."
- **Systematic recipe**: The progressive path from T2I to T2V to TI2V, with detailed ablations at each stage, offers high reference value to the community.
- **Flexibility of the unified framework**: A single model supports T2V, TI2V, video prediction, frame interpolation, multi-view generation, and long video generation by varying the conditioning inputs.
- **Data engine**: A complete video data processing pipeline (PySceneDetect + captioning + DSG-Video filtering) was constructed, processing 90M+ video-text pairs.

## Limitations & Future Work
- Reliance on large-scale internal data (42M internal videos) limits reproducibility.
- The performance of frame replacement under multi-image conditioning scenarios (e.g., video editing) is not thoroughly investigated.
- Inference cost of the 8.7B model remains high; inference efficiency optimization is not discussed.
- Evaluation relies primarily on VBench, with limited human evaluation comparisons.
- Quality and consistency of long video generation (>100 frames) require further validation.

## Related Work & Insights
- Frame replacement was explored in the U-Net era (ConsistI2V) with limited effectiveness; it becomes naturally effective in the DiT architecture due to its pure self-attention design—**architecture determines the optimal strategy**.
- The combination of MaskDiT + AdaFactor + gradient checkpointing constitutes a practical solution for efficient large-model training.
- Progressive training across three dimensions (resolution, duration, and architecture) represents a critical scaling pathway for video generation models.

## Rating
- Novelty: ⭐⭐⭐ — All components are combinations of existing techniques; frame replacement is also not novel per se; the core contribution lies in systematic integration.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Ablations are extremely comprehensive, with detailed studies at each stage from T2I to T2V to TI2V.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured; the recipe-style presentation is practitioner-friendly.
- Value: ⭐⭐⭐⭐⭐ — As a systematic recipe for video generation, this work offers exceptional reference value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Generating, Fast and Slow: Scalable Parallel Video Generation with Video Interface Networks](generating_fast_and_slow_scalable_parallel_video_generation_with_video_interface.md)
- [\[ICCV 2025\] TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation](tip-i2v_a_million-scale_real_text_and_image_prompt_dataset_for_image-to-video_ge.md)
- [\[ICCV 2025\] OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human Animation Models](omnihuman-1_rethinking_the_scaling-up_of_one-stage_conditioned_human_animation_m.md)
- [\[ICCV 2025\] MotionShot: Adaptive Motion Transfer across Arbitrary Objects for Text-to-Video Generation](motionshot_adaptive_motion_transfer_across_arbitrary_objects_for_text-to-video_g.md)
- [\[ICML 2026\] CamGeo: Sparse Camera-Conditioned Image-to-Video Generation with 3D Geometry Prior](../../ICML2026/video_generation/camgeo_sparse_camera-conditioned_image-to-video_generation_with_3d_geometry_prio.md)

</div>

<!-- RELATED:END -->
