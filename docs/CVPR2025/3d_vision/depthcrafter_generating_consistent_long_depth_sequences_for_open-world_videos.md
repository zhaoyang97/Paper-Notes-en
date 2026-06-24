---
title: >-
  [Paper Note] DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos
description: >-
  [CVPR 2025][3D Vision][Video Depth Estimation] Leverages a pretrained video diffusion model (SVD) for video depth estimation. Through a three-stage training strategy, it realizes temporally consistent depth sequence generation of variable lengths (up to 110 frames). By designing a segment-based inference strategy, it supports extremely long videos, comprehensively outperforming existing methods in zero-shot settings.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Video Depth Estimation"
  - "Diffusion Models"
  - "Temporal Consistency"
  - "Open-world"
  - "Long-sequence Depth"
date: 2026-05-08
content_hash: dbbcc82b805e7227
---

# DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos

**Conference**: CVPR 2025  
**arXiv**: [2409.02095](https://arxiv.org/abs/2409.02095)  
**Code**: [https://depthcrafter.github.io](https://depthcrafter.github.io)  
**Area**: 3D Vision  
**Keywords**: Video Depth Estimation, Diffusion Models, Temporal Consistency, Open-world, Long-sequence Depth

## TL;DR

Leverages a pretrained video diffusion model (SVD) for video depth estimation. Through a three-stage training strategy, it realizes temporally consistent depth sequence generation of variable lengths (up to 110 frames). By designing a segment-based inference strategy, it supports extremely long videos, comprehensively outperforming existing methods in zero-shot settings.

## Background & Motivation

Foundation models in monocular depth estimation (such as Depth Anything V2, Marigold, etc.) perform exceptionally well in generating single-frame depth, but applying them directly frame-by-frame to videos causes severe temporal inconsistency (flickering). Existing video depth methods face multiple challenges:

- **Test-time optimization methods** (e.g., Robust CVD) require camera poses or optical flow, making them difficult to apply to open-world videos.
- **Feed-forward prediction methods** (e.g., NVDS) are limited by restricted training data and cannot handle diverse open-world scenes.
- **Inadequate temporal context**: Existing video diffusion models only support a fixed, small number of frames (e.g., SVD supports only 25 frames), which is insufficient to accurately distribute depth across the entire video.
- Open-world videos vary enormously in content, motion, camera movement, and length.

Core Motivation: Leverages the powerful generative capabilities of video diffusion models to learn temporally consistent depth distributions, obtaining both content diversity and depth accuracy through a carefully designed training strategy.

## Method

### Overall Architecture

DepthCrafter is a conditional diffusion model modeling $p(\mathbf{d}|\mathbf{v})$ to generate a depth sequence $\mathbf{d}$ from an input video $\mathbf{v}$. Built upon the pretrained Stable Video Diffusion (SVD), it operates in the latent space, utilizing a VAE for spatial encoding/decoding, and is progressively adapted through a three-stage training strategy.

### Key Designs

1. **Video Conditioning Adaptation**:
    - **Function**: Adapts the single-image conditional generation of SVD into frame-by-frame video-to-depth conditional generation.
    - **Mechanism**: While original SVD only concatenates the latent representation of the first frame to the input, this work instead concatenates the latent representations of all video frames frame-by-frame with the noisy depth latent representation. High-level semantic information is injected frame-by-frame into cross-attention via CLIP embeddings. The depth sequence uses an affine-invariant representation (normalized to $[0,1]$), where the key difference is employing a **scale and shift shared across the entire sequence** instead of per-frame normalization.
    - **Design Motivation**: Frame-by-frame conditioning provides complete video information, while shared normalization ensures temporal consistency.

2. **Three-Stage Training Strategy**:
    - **Function**: Progressive training enables the model to simultaneously acquire content diversity, long-term context, and fine depth details.
    - **Mechanism**:
        - **Stage 1** (80K iterations): The entire model is trained on a large-scale real-world dataset (~200K videos), with sequence lengths randomly sampled within $[1,25]$ frames, to learn the core video-to-depth task and variable-length generation.
        - **Stage 2** (40K iterations): Only temporal layers are fine-tuned, still on real-world data, with the sequence length expanded to $[1,110]$ frames, learning long-term context.
        - **Stage 3** (10K iterations): Only spatial layers are fine-tuned, trained on a small synthetic dataset (~3K videos, DynamicReplica + MatrixCity) with a fixed length of 45 frames, to learn fine-grained depth details.
    - **Design Motivation**: Direct training on long sequences is memory-prohibitive (a 40GB GPU only supports up to 25 frames). Fine-tuning temporal layers alone drastically reduces memory usage; temporal layers are sensitive to sequence length, whereas spatial layers have already been adapted in Stage 1. Furthermore, synthetic data provides more accurate depth ground truth.

3. **Extremely Long Video Inference Strategy**:
    - **Function**: Supports depth estimation for videos of arbitrary length exceeding 110 frames.
    - **Mechanism**: The video is divided into overlapping segments, and depth is estimated segment by segment. The key technique is **noise initialization anchoring**—instead of using pure Gaussian noise for overlapping frames' latent representations, they are initialized by adding noise to the denoised result of the previous segment, anchoring the scale and shift of the depth distribution. Then, **mortise-and-tenon-style latent interpolation** is used to stitch adjacent segments, applying linearly decreasing weights $w_i$ to interpolate the latent representations of the two segments for the overlapping frames.
    - **Design Motivation**: Independent inference for each segment leads to inconsistent depth distributions across segments; noise initialization anchors the scale/shift, while linear interpolation ensures smooth transitions.

### Loss & Training

- Uses the denoising score matching loss under the EDM framework:
$$\mathbb{E}[\lambda_{\sigma_t}\|D_\theta(\mathbf{x}_t; \sigma_t; \mathbf{c}) - \mathbf{x}_0\|_2^2]$$
- The VAE encoder/decoder directly reuses pretrained weights from SVD, as the reconstruction error for depth sequences has been verified to be negligible.
- Trained at a resolution of $320 \times 640$, and can be inferred at any resolution (e.g., $576 \times 1024$).
- 8× A100 GPUs, with a total training time of approximately 5 days.
- Denoising with 5 steps by default during inference.

## Key Experimental Results

### Main Results (Zero-shot Video Depth Estimation)

| Dataset | Metric | DepthCrafter | Depth-Anything-V2 | Gain |
|--------|------|------|----------|------|
| Sintel (~50 frames) | AbsRel↓ | **0.270** | 0.367 | 26.4% |
| Sintel (~50 frames) | $\delta_1$↑ | **0.697** | 0.554 | 25.8% |
| ScanNet (90 frames) | AbsRel↓ | **0.123** | 0.135 | 8.9% |
| KITTI (110 frames) | AbsRel↓ | **0.104** | 0.140 | 25.7% |
| KITTI (110 frames) | $\delta_1$↑ | **0.896** | 0.804 | 11.4% |
| Bonn (110 frames) | AbsRel↓ | **0.071** | 0.078 | 9.0% |

### Ablation Study (Sintel Dataset)

| Training Stage | AbsRel↓ | $\delta_1$↑ | Description |
|------|---------|------|------|
| Stage 1 only | 0.322 | 0.626 | Base adaptation |
| Stage 1+2 | 0.316 | 0.675 | +Long-term context |
| Stage 1+2+3 | **0.270** | **0.697** | +Fine depth details |

### Key Findings

- Achieves SOTA on all four video datasets, with the most significant advantages in large-motion scenes such as Sintel and KITTI.
- Inferences at 465.84ms per frame ($1024 \times 576$), which is more than twice as fast as Marigold (1070ms), but slower than Depth-Anything-V2 (180ms).
- Temporal consistency is significantly superior to all baselines, with no jagged artifacts in temporal profiles.
- Single-frame depth estimation (NYU-v2) is also competitive, with $\delta_1=0.948$.
- Both noise initialization and latent interpolation in the inference strategy are indispensable: using only initialization solves flickering in static regions, but issues remain in dynamic regions.
- A segment length of 40 frames requires only 12GB of VRAM, making it adaptable to most modern GPUs.

## Highlights & Insights

- **Powerful Prior of Video Diffusion Models**: The spatial-temporal attention mechanism of SVD is naturally suited for modeling temporally consistent depth distributions, yielding transition effects that far exceed expectations.
- **Exquisite Design of Three-Stage Training**: The trade-off between dataset quality and quantity is elegantly resolved through selective fine-tuning of network layers across stages.
- **Scale and Shift Shared Across Entire Sequence**: This is a key design ensuring temporal consistency in video depth, which is more challenging but far more practical than per-frame normalization.
- **Mortise-and-Tenon-Style Interpolation in Inference Strategy**: Named after traditional Chinese woodworking techniques, it is actually a simple yet effective linear blending in the latent space.
- **Open-world Generalization Capability**: Performs exceptionally well across highly diverse scenes, such as DAVIS, videos generated by Sora, cartoons, and game footage.

## Limitations & Future Work

- Inference speed (465ms/frame) is still relatively slow, with the main bottleneck being the iterative denoising process of the diffusion model.
- Requires approximately 24GB of GPU memory to process $1024 \times 576$ resolution videos of 110 frames.
- Only predicts relative depth (affine-invariant); metric depth cannot be directly obtained.
- Synthetic training data is limited to ~3K videos; scaling up fine-grained synthetic data could yield further improvements.
- Ground truth depth in real-world datasets is obtained via stereo matching, introducing a certain level of noise.

## Related Work & Insights

- Compared to ChronoDepth: ChronoDepth only supports 10 frames, whereas DepthCrafter supports 110 frames and can perform inference on arbitrary lengths, which represents a decisive advantage.
- Test-time optimization methods (e.g., Robust CVD) require camera poses, limiting practical applications; DepthCrafter does not require any additional information.
- Insight: Video diffusion models serve as a powerful foundation for video geometry estimation, and three-stage progressive training is an effective paradigm for utilizing heterogeneous datasets.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Successfully applies video video diffusion models to long-sequence depth estimation for the first time, with exquisitely designed three-stage training and inference strategies.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Zero-shot evaluation across 6 datasets covering indoor/outdoor, dynamic/static, and real/synthetic scenarios, with comprehensive ablation studies and downstream application demonstrations.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-organized structure, intuitive visualization of temporal profiles, and a complete chain of motivation-solution-validation.
- Value: ⭐⭐⭐⭐⭐ Pioneers a new paradigm for video depth estimation, exerting a profound impact on subsequent depth foundation models and video understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Video Depth Anything: Consistent Depth Estimation for Super-Long Videos](video_depth_anything_consistent_depth_estimation_for_super-long_videos.md)
- [\[CVPR 2025\] Generating 3D-Consistent Videos from Unposed Internet Photos](generating_3d-consistent_videos_from_unposed_internet_photos.md)
- [\[CVPR 2025\] Video Depth Without Video Models](video_depth_without_video_models.md)
- [\[CVPR 2025\] Open-World Amodal Appearance Completion](open-world_amodal_appearance_completion.md)
- [\[CVPR 2025\] HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos](hawor_world-space_hand_motion_reconstruction_from_egocentric_videos.md)

</div>

<!-- RELATED:END -->
