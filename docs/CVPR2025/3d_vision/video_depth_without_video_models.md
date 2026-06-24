---
title: >-
  [Paper Note] Video Depth Without Video Models
description: >-
  [CVPR 2025][3D Vision][Video Depth Estimation] This paper proposes RollingDepth, which avoids using video diffusion models and instead extends a single-frame latent diffusion model (Marigold) into a multi-frame snippet processor. Combined with multi-scale dilated sampling and a robust global alignment algorithm, it merges short-snippet depths into temporally consistent long-video depth, outperforming specialized video depth models and single-frame models across multiple bench…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Video Depth Estimation"
  - "Monocular Depth"
  - "Latent Diffusion Models"
  - "Temporal Consistency"
  - "Rolling Inference"
date: 2026-05-08
content_hash: 8ca729a57459ba0d
---

# Video Depth Without Video Models

**Conference**: CVPR 2025  
**arXiv**: [2411.19189](https://arxiv.org/abs/2411.19189)  
**Code**: [https://rollingdepth.github.io](https://rollingdepth.github.io)  
**Area**: 3D Vision / Depth Estimation  
**Keywords**: Video Depth Estimation, Monocular Depth, Latent Diffusion Models, Temporal Consistency, Rolling Inference

## TL;DR

This paper proposes RollingDepth, which avoids using video diffusion models and instead extends a single-frame latent diffusion model (Marigold) into a multi-frame snippet processor. Combined with multi-scale dilated sampling and a robust global alignment algorithm, it merges short-snippet depths into temporally consistent long-video depth, outperforming specialized video depth models and single-frame models across multiple benchmarks.

## Background & Motivation

**Background**: Single-frame monocular depth estimation has made tremendous progress recently, benefiting from large-scale pre-trained foundation models (DINOv2, StableDiffusion) and synthetic training data. Methods like Marigold and Depth Anything exhibit excellent zero-shot generalization. However, applying these methods frame-by-frame to videos leads to depth flickering and drift.

**Limitations of Prior Work**: (1) Frame-by-frame methods lack temporal consistency, leading to sudden depth jumps between adjacent frames, especially when the depth range changes abruptly due to camera motion (e.g., foreground objects entering or the camera turning out of a window); (2) Methods based on video diffusion models (ChronoDepth, DepthCrafter) have good local temporal consistency but are expensive to train and infer, can only process fixed short sequences, and require segment merging schemes prone to low-frequency flickering and drift; (3) Video depth models perform poorly on distant scenes.

**Key Challenge**: Temporal consistency requires information exchange between frames, but video diffusion models are too costly and restricted to fixed lengths; single-frame models are precise but lack temporal coherence.

**Goal**: To extend a single-frame LDM into a temporally consistent depth estimator that can handle videos of arbitrary length, without using a video diffusion model.

**Key Insight**: Since the core issues of video diffusion models are fixed length limits and high training costs, can we achieve or even exceed the video model performance by using a small expansion of a single-frame model (processing 2-3 frame short snippets) plus a smart global alignment algorithm? Sampling snippets at different frame rates can cover both short- and long-range temporal relationships.

**Core Idea**: Extend Marigold into a multi-frame LDM that processes extremely short snippets (typically 3 frames), sample overlapping snippets from the video with different dilation rates, and then use robust optimization to align the scale and shift of all snippets to assemble a consistent long-video depth.

## Method

### Overall Architecture

Given an RGB video of arbitrary length, RollingDepth executes three steps: (1) **Snippet Inference**: Sampling numerous overlapping 3-frame snippets from the video using a dilated rolling window with different frame intervals, and performing a 1-step denoising on each snippet using a multi-frame LDM to get initial depth snippets; (2) **Global Alignment**: Jointly optimizing the scale and shift parameters of all snippets to achieve global consistency of depth values on overlapping frames; (3) **Optional Refinement**: Re-denoising the aligned depth video with moderate noise added using the LDM to restore fine details.

### Key Designs

1. **Multi-frame LDM (Snippet Depth Estimator)**:

    - Function: Extended from the Marigold single-frame model to a depth estimator capable of processing short snippets.
    - Mechanism: Modifying the self-attention layer to flatten the tokens of all frames in the snippet into a single sequence, allowing the attention mechanism to operate across frames and capture spatial-temporal interactions. Unlike the factorized spatial-temporal attention in video diffusion models, this design can process frames with arbitrary temporal intervals, making it suitable for snippets of various frame rates. Meanwhile, the affine-invariant depth prediction of Marigold is changed to predicting inverse depth, which is more robust to the far field.
    - Design Motivation: Cross-frame self-attention is the most concise solution to realize frame-to-frame information exchange. Predicting inverse depth is more suitable for video scenarios than affine-invariant depth because depth ranges often change abruptly in videos.

2. **Dilated Rolling Kernel**:

    - Function: Constructing a large number of overlapping short snippets from the video at various temporal resolutions.
    - Mechanism: For 3-frame snippets, construct snippets using different dilation rates $g \in \{1, 10, 25\}$ and strides $h$. Snippets with a dilation rate of 1 capture short-range relationships between adjacent frames, whereas snippets with a dilation rate of 25 span long-range relationships of about 1 second. Snippets with different dilation rates overlap on the same frame, providing constraints for subsequent alignment.
    - Design Motivation: Using only adjacent frames fails to cover long-range temporal dependencies; using only large-interval frames loses local smoothness. Multi-scale sampling balances both while keeping a constant memory footprint.

3. **Robust Global Depth Co-alignment**:

    - Function: Unifying all independently inferred snippet depths into a globally consistent scale and shift.
    - Mechanism: Each snippet $k$ has its independent scale $s_k$ and shift $t_k$. Jointly optimizing these $N_T$ pairs of parameters by minimizing the L1 loss of all overlapping depth predictions across all frames. Solved using Adam gradient descent (2000 steps), with higher weights given to high-dilation-rate snippets to stabilize optimization. After alignment, the pixel-wise mean of all overlapping depths is taken for each frame to obtain the final depth.
    - Design Motivation: L1 loss is more robust than L2 and unaffected by outliers. High-dilation-rate snippets provide long-range constraints, which are critical for stabilizing global scale.

### Loss & Training

- Fine-tuned Marigold on TartanAir (18 synthetic video scenes, 369 sequences) and Hypersim (365 synthetic single-frame scenes).
- Training image size: 480×640, AdamW optimizer, learning rate $3 \times 10^{-5}$, trained on 4 A100 GPUs for about 2 days.
- Key Trick: Jointly normalizing the inverse depth within each snippet (rather than per-frame normalization), enabling the model to handle abrupt changes in depth range.
- Depth range augmentation: randomly compressing and shifting the normalized depth range.

## Key Experimental Results

### Main Results

**Zero-shot Video Depth Estimation Comparison (AbsRel % ↓)**:

| Method | Type | PointOdyssey(250) | ScanNet(90) | Bonn(110) | DyDToF(200) | DyDToF(100) |
|------|------|---------|---------|------|---------|---------|
| Marigold | Single-frame | 14.9 | 14.9 | 10.5 | 25.3 | 16.4 |
| DepthAnythingv2 | Single-frame | 14.4 | 13.3 | 10.5 | 24.8 | 16.0 |
| ChronoDepth | Video | 51.7 | 16.8 | 10.9 | 26.9 | 19.9 |
| DepthCrafter | Video | 36.3 | 12.7 | **6.6** | 22.1 | 16.2 |
| **Ours(fast)** | Extension | **9.6** | 10.1 | 7.9 | 17.7 | 12.7 |
| **Ours** | Extension | **9.6** | **9.3** | 7.9 | **17.3** | **12.3** |

### Ablation Study

| Dilation Rate | PointOdyssey ↓ | ScanNet ↓ | Description |
|--------|---------------|-----------|------|
| {1} | 16.7 | 12.8 | Adjacent frames only, lacking long-range information |
| {1, 25} | 10.2 | 10.6 | Adding long-range constraints brings massive improvement |
| {1, 10, 25} | **10.2** | **9.9** | Adding mid-range further improves performance |

| Alignment | Refinement | PointOdyssey ↓ | ScanNet ↓ |
|------|------|---------------|-----------|
| × | × | 13.0 | 12.4 |
| ✓ | × | 10.2 | 9.9 |
| ✓ | ✓ | **9.6** | **9.3** |

### Key Findings

- Global alignment is the core contribution: ~3 percentage points of error reduction come from alignment, while refinement only brings marginal improvement.
- Adding snippets with a dilation rate of 25 brings a massive boost (PointOdyssey 16.7 -> 10.2); long-range constraints are crucial.
- Video depth models perform extremely poorly on PointOdyssey (ChronoDepth 51.7), worse than single-frame methods. This shows that video priors can hinder performance in scenes with abrupt depth range changes.
- RollingDepth fast version takes only 81 seconds on a 250-frame video, which is faster than ChronoDepth (121s) and DepthCrafter (284s).
- DepthCrafter performs best in indoor scenes with foreground people (Bonn), suggesting its video priors are advantageous for such environments.

## Highlights & Insights

- **Counter-intuitive success**: Doing better than video models without using a video model proves the feasibility of the "lightweight extension + smart post-processing" path.
- **Joint normalization within snippets is a key trick**: It allows the model to understand frame-to-frame variations in depth range, which is crucial for long-video processing.
- **Elegance of global alignment**: Casts complex temporal consistency as a simple scale-shift optimization, using the overlapping relationships of multi-scale snippets to provide sufficient constraints.
- **1-step inference**: The snippet LDM requires only 1-step denoising to obtain initial depth estimates, making it extremely fast.

## Limitations & Future Work

- The alignment step relies on the assumption of depth consistency among overlapping snippets, which may fail during aggressive object motion.
- Slightly inferior to DepthCrafter in close-up indoor human scenes, showing room for improvement in modeling scenes dominated by foreground elements.
- Inference time grows with the number of dilation rates and refinement steps.
- Adaptive selection of dilation rates based on video motion characteristics could be considered.
- Integration with metric depth estimation has not been explored; currently, it still outputs affine-invariant depth.

## Related Work & Insights

- **vs Marigold**: The base model of RollingDepth. This paper shows how to extend it to video with minimal changes.
- **vs DepthCrafter**: Based on the SVD video diffusion model; has good local consistency but is limited by fixed lengths and high training costs. RollingDepth is more flexible and can handle long videos of over 1000 frames.
- **vs ChronoDepth**: Also a diffusion-based method but with poorer results, producing stratified depth maps and performing extremely poorly in scenes with large depth range changes.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The counter-intuitive yet highly effective idea of "video depth without video models".
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6 baselines, 4 datasets, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear methodology motivation with step-by-step contributions of individual components.
- Value: ⭐⭐⭐⭐⭐ Redefines the methodology of video depth estimation, proving the feasibility of a simple extension pathway.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Video Depth Anything: Consistent Depth Estimation for Super-Long Videos](video_depth_anything_consistent_depth_estimation_for_super-long_videos.md)
- [\[ECCV 2024\] FutureDepth: Learning to Predict the Future Improves Video Depth Estimation](../../ECCV2024/3d_vision/futuredepth_learning_to_predict_the_future_improves_video_depth_estimation.md)
- [\[ICCV 2025\] FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution](../../ICCV2025/3d_vision/flashdepth_real-time_streaming_video_depth_estimation_at_2k_resolution.md)
- [\[CVPR 2025\] Generative Omnimatte: Learning to Decompose Video into Layers](generative_omnimatte_learning_to_decompose_video_into_layers.md)
- [\[CVPR 2025\] HD-EPIC: A Highly-Detailed Egocentric Video Dataset](hd-epic_a_highly-detailed_egocentric_video_dataset.md)

</div>

<!-- RELATED:END -->
