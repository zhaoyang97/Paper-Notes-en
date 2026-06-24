---
title: >-
  [Paper Note] Learning Temporally Consistent Video Depth from Video Diffusion Priors
description: >-
  [CVPR 2025][Video Generation][Video Depth Estimation] This work proposes ChronoDepth, a video depth estimation method based on Stable Video Diffusion (SVD). By independently sampling noise levels per frame during training and using noise-free preceding frames as context during inference (Consistent Context-Aware Strategy), the method achieves state-of-the-art (SOTA) temporal consistency while maintaining spatial accuracy, ranking first on average for the MFC metric.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Video Depth Estimation"
  - "Temporal Consistency"
  - "Video Diffusion Models"
  - "Context-Aware Inference"
  - "SVD Fine-Tuning"
date: 2026-05-08
content_hash: b93e56232fca8902
---

# Learning Temporally Consistent Video Depth from Video Diffusion Priors

**Conference**: CVPR 2025  
**arXiv**: [2406.01493](https://arxiv.org/abs/2406.01493)  
**Code**: [https://xdimlab.github.io/ChronoDepth](https://xdimlab.github.io/ChronoDepth)  
**Area**: Video Generation  
**Keywords**: Video Depth Estimation, Temporal Consistency, Video Diffusion Models, Context-Aware Inference, SVD Fine-Tuning

## TL;DR
This work proposes ChronoDepth, a video depth estimation method based on Stable Video Diffusion (SVD). By independently sampling noise levels per frame during training and using noise-free preceding frames as context during inference (Consistent Context-Aware Strategy), the method achieves state-of-the-art (SOTA) temporal consistency while maintaining spatial accuracy, ranking first on average for the MFC metric.

## Background & Motivation

1. **Background**: Single-frame depth estimation has made significant progress recently (e.g., Marigold, DepthAnything). However, constrained by the I.I.D. assumption of independent frame-by-frame prediction, the generated depth videos suffer from flickering and temporal inconsistency. Discriminative video depth methods (such as NVDS or RNN-based approaches) either rely on precise camera poses or still yield suboptimal temporal consistency.
2. **Limitations of Prior Work**:
    - Video diffusion models natively possess temporal modeling capabilities, but directly extending single-frame diffusion depth models to videos is non-trivial.
    - Processing arbitrarily long videos requires clip-by-clip inference, where the key challenge is how to pass context information across clips.
    - The existing "replacement trick" (adding noise to preceding frames and concatenating them into the current clip) is mathematically ungrounded—applying different levels of noise at each sampling step leads to inconsistent context information.
3. **Key Challenge**: How to share consistent context information across clips to achieve truly coherent video depth estimation in the temporal dimension?
4. **Goal**: To design training and inference strategies that guarantee consistent context across clips.
5. **Key Insight**: Inspired by Diffusion Forcing, setting independent noise levels for each frame during training allows the model to learn denoising when frames with different noise levels coexist. During inference, preceding frames are set to near-zero noise (as already predicted results) and subsequent frames are set to normal noise levels.
6. **Core Idea**: Per-frame independent noise training + Noise-free preceding frame context inference = Cross-clip consistent video depth estimation.

## Method

### Overall Architecture
Based on the image-to-video variant of SVD (Stable Video Diffusion). The RGB video is mapped to the latent space via a VAE encoder, concatenated with the depth latents, and fed into a UNet for denoising. Training consists of two phases: first training the spatial layers (single-frame depth), then freezing the spatial layers to train only the temporal layers (multi-frame clips). During inference, a sliding window strategy is used to handle videos of arbitrary length, where the first $W$ frames are initialized with already predicted depths (without adding noise), and the remaining $F-W$ frames undergo denoising starting from Gaussian noise.

### Key Designs

1. **Per-Frame Independent Noise Levels Training**:

    - **Function**: Enables the model to handle cases where different frames within the same clip have different noise levels, laying the foundation for the context injection strategy during inference.
    - **Mechanism**: Standard video diffusion samples a single noise level $\sigma_t$ for the entire clip. ChronoDepth instead samples independently per frame: $\boldsymbol{\sigma}_t = [\sigma_1, \sigma_2, ..., \sigma_F]$, where $\log \sigma_i \sim \mathcal{N}(P_{mean}, P_{std}^2)$ is sampled independently. The preconditioning functions $c_{skip}, c_{out}, c_{in}, c_{noise}$ are also applied frame-by-frame accordingly. The training loss maintains the DSM formulation.
    - **Design Motivation**: This is the critical prerequisite for achieving consistent context-aware inference. Only when the model learns to handle hybrid scenarios—where some frames are almost clean (already predicted results) while others are fully noisy (to be predicted)—can it properly exploit the context of preceding frames during inference.

2. **Consistent Context-Aware Inference**:

    - **Function**: Passes consistent depth information between clips, eliminating flickering and scale jumps across clips.
    - **Mechanism**: Employs a sliding window strategy to handle long videos. Each clip consists of $F$ frames, and adjacent clips overlap by $W$ frames. In the first clip, standard denoising is performed (all frames start from pure noise). In subsequent clips, the first $W$ frames directly use the predicted depth latents from the previous clip (without adding noise), and the remaining $F-W$ frames start from pure noise. The noise conditioning for the denoiser is:
      $$\boldsymbol{\sigma}_t = [\underbrace{\sigma_\epsilon, ..., \sigma_\epsilon}_W, \underbrace{\sigma_t, ..., \sigma_t}_{F-W}]$$
      where $\sigma_\epsilon$ is an extremely small noise value.
    - **Design Motivation**: Compared to the replacement trick (which adds large noise to preceding frames), this method preserves context consistency across each sampling step—the preceding frames remain as the same clean prediction without introducing inconsistencies from varying noise added at each step. $\sigma_\epsilon$ is set to an extremely small but non-zero value because the preceding predictions are not ground truth, and a tiny amount of uncertainty must be retained to mitigate long-range error accumulation.

3. **Sequential Spatial-Temporal Fine-Tuning**:

    - **Function**: Fine-tunes spatial and temporal layers in stages, avoiding conflicts between spatial accuracy and temporal consistency in joint training.
    - **Mechanism**: Phase 1: Freeze temporal layers, train spatial layers on a single-frame depth dataset (Hypersim, 39K samples) for 20K steps. Phase 2: Freeze spatial layers, train temporal layers on multi-frame video datasets (TartanAir + Virtual KITTI2 + MVS-Synth, 938 video sequences in total) for 18K steps, with clip lengths randomly sampled as $F \in [1, F_{max}]$ ($F_{max}=5$). Single-frame data is used throughout both phases.
    - **Design Motivation**: Joint training easily causes the temporal layers to "slack off" by relying heavily on spatial layers, or temporal objectives can interfere with spatial accuracy. Staged training allows the spatial layers to reach optimal spatial accuracy first, enabling the temporal layers to subsequently focus solely on processing temporal consistency. Random clip lengths serve as data augmentation, enhancing the model's robustness to varying movement speeds.

### Loss & Training
- Uses the DSM loss from the EDM framework:
  $$\mathcal{L} = \mathbb{E}[\lambda(\boldsymbol{\sigma}_t) \|\hat{\mathbf{z}}_0 - \mathbf{z}_0\|_2^2]$$
  where the weighting function is $\lambda(\sigma) = (1+\sigma^2)\sigma^{-2}$.
- RGB conditioning is introduced via channel concatenation (disabling the original cross-attention conditioning in SVD).
- The depth map is replicated across three channels to simulate RGB and reuse the VAE. After decoding, the mean of the three channels is taken.
- Adam optimizer with a learning rate of 3e-5, trained on 8 × A100-80GB GPUs for approximately 1.5 days.

## Key Experimental Results

### Main Results

**Zero-Shot Video Depth Estimation (MFC = Multi-Frame Consistency, lower is better):**

| Method | KITTI-360 AbsRel / MFC | ScanNet++ AbsRel / MFC | Sintel AbsRel / MFC | Mean MFC Rank |
|------|---------------------|---------------------|-------------------|-------------|
| Marigold | 0.213 / 0.776 | 0.192 / 0.109 | 0.573 / 1.112 | 4.00 |
| DepthAnything V2 | 0.207 / 0.807 | 0.170 / 0.103 | 0.387 / 1.504 | 5.00 |
| DepthCrafter | 0.293 / 0.655 | 0.199 / 0.094 | 0.374 / 1.270 | 3.00 |
| **ChronoDepth (Ours)** | 0.215 / **0.407** | 0.176 / **0.092** | 0.493 / **0.728** | **1.00** |

### Ablation Study

**Comparison of Inference Strategies (Temporal Consistency MFC $\downarrow$):**

| Inference Strategy | MFC | Description |
|---------|-----|------|
| Naive sliding window | High | No context propagation |
| Replacement trick | Medium | Inconsistent context |
| **Consistent context-aware** | **Lowest** | Consistent context |

**Comparison of Training Strategies:**

| Configuration | AbsRel | MFC | Description |
|------|--------|-----|------|
| Joint spatial+temporal | Higher | Higher | Mutual interference from joint training |
| **Sequential (spatial $\rightarrow$ temporal)** | **Lower** | **Lowest** | Staged training is optimal |
| Fixed clip length | Moderate | Moderate | Lacks robustness |
| **Random clip length** | **Lower** | **Lower** | Effect of data augmentation |

### Key Findings
- **SOTA across MFC Metrics**: ChronoDepth achieves the lowest MFC on all three benchmarks, ranking 1st on average, consistently outperforming DepthAnything V2 (ranking 5.00) and Marigold (ranking 4.00) by a large margin.
- **Slight Trade-off in Spatial Accuracy**: ChronoDepth ranks 4.00 on AbsRel (vs. 1.67 for DepthAnything V2), showing that some spatial accuracy is traded for temporal consistency. However, this trade-off is highly reasonable for video applications.
- **Quantitative Verification of Replacement Trick Issues**: Compared with DepthCrafter (which uses the replacement trick), ChronoDepth reduces MFC from 0.655 to 0.407 on KITTI-360, validating the importance of consistent context.
- **Sequential Training is Essential**: Training spatial layers before temporal layers yields much better temporal consistency than joint training, without sacrificing spatial accuracy.
- **$\sigma_\epsilon$ Cannot Be Zero**: An extremely small but non-zero noise level performs better than zero noise, as it accounts for the uncertainty in preceding predictions.

## Highlights & Insights
- **Elegant Design of Consistent Context-Aware Strategy**: Through the minor modification of "per-frame independent noise" during training, the model naturally supports "clean + noisy frame coexistence" during inference without any architecture modifications. This trick is highly transferable to any video diffusion model task requiring autoregressive processing.
- **True Integration and Resolution of Replacement Trick Pitfalls**: Prior works (including DepthCrafter) heuristically utilized the replacement trick without recognizing its mathematical inconsistency. This paper clearly identifies the root cause (different noise injected at each step induces conditioning inconsistency) and provides an elegant solution.
- **Discovery of Sequential Training Strategy**: The finding that joint spatial-temporal training is inferior to phased training is highly insightful for other video generation/understanding tasks. It suggests that teaching the model spatial concepts before temporal concepts might serve as a better curriculum learning strategy.

## Limitations & Future Work
- **Suboptimal Spatial Accuracy**: ChronoDepth ranks 4th in AbsRel, falling behind DepthAnything V2 and DepthAnything. This might be improved by using larger single-frame training datasets or more powerful foundation models.
- **Inference Speed Limited by Diffusion Steps**: Requiring multi-step denoising makes it slower than discriminative methods. Flow matching or consistency distillation could be investigated for acceleration.
- **Training Constraint of $F_{max}=5$**: The maximum training clip length is limited to 5 frames, leading to a train-test gap when using 10 frames during inference. Longer training clips (requiring more VRAM) may further boost performance.
- **Reliance on SVD's VAE**: Replicating the depth map into three channels is a workaround and is less efficient than a custom VAE designed specifically for depth.
- **Error Accumulation**: Although $\sigma_\epsilon$ mitigates long-range errors, an analysis of error accumulation on ultra-long videos (thousands of frames) is still lacking.

## Related Work & Insights
- **vs. DepthCrafter [Hu et al.]**: Both methods are based on video diffusion models. However, DepthCrafter utilizes the replacement trick for context propagation, which introduces cross-clip inconsistency. ChronoDepth's consistent context strategy fundamentally resolves this issue.
- **vs. Marigold [Ke et al.]**: Marigold pioneered single-frame diffusion-based depth estimation. ChronoDepth can be viewed as its advanced video counterpart; it not only inherits the spatial precision of diffusion models but also gains temporal consistency from video diffusion priors.
- **vs. Diffusion Forcing [Chen et al.]**: Diffusion Forcing proposed a similar "per-frame independent noise" concept but for generative tasks with an RNN base. ChronoDepth applies this to deterministic video depth estimation tasks on attention-based architectures without requiring extra hidden states.

## Rating
- Novelty: ⭐⭐⭐⭐ The per-frame independent noise and consistent context inference ideas are novel, although the core concept is inspired by Diffusion Forcing.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Excellent coverage with three benchmarks, various inference and training strategy comparisons, and y-t slice visualizations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear explanations comparing the three inference strategies, with pseudo-code in Algorithms 1-3 making the method straightforward to understand.
- Value: ⭐⭐⭐⭐ Significant contribution to temporal consistency in video depth estimation; the consistent context strategy can be transferred to other video diffusion tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] NormalCrafter: Learning Temporally Consistent Normals from Video Diffusion Priors](../../ICCV2025/video_generation/normalcrafter_learning_temporally_consistent_normals_from_video_diffusion_priors.md)
- [\[CVPR 2025\] VideoGuide: Improving Video Diffusion Models without Training Through a Teacher's Guide](videoguide_improving_video_diffusion_models_without_training_through_a_teachers_.md)
- [\[CVPR 2025\] Optical-Flow Guided Prompt Optimization for Coherent Video Generation](optical-flow_guided_prompt_optimization_for_coherent_video_generation.md)
- [\[CVPR 2025\] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](streamingt2v_consistent_dynamic_and_extendable_long_video_generation_from_text.md)
- [\[CVPR 2025\] Tracktention: Leveraging Point Tracking to Attend Videos Faster and Better](tracktention_leveraging_point_tracking_to_attend_videos_faster_and_better.md)

</div>

<!-- RELATED:END -->
