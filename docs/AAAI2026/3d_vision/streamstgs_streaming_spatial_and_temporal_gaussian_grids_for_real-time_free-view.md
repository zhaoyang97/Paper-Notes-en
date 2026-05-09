---
title: >-
  [Paper Note] StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video
description: >-
  [AAAI 2026][3D Vision][Free-viewpoint video] This paper proposes StreamSTGS, a streamable spatial-temporal Gaussian grid representation that encodes canonical 3D Gaussian attributes as 2D images and temporal features as video, enabling real-time free-viewpoint video streaming at only 170 KB per frame. Reconstruction quality is maintained (PSNR 32.30 dB) through Transformer-assisted training and a sliding window mechanism.
tags:
  - AAAI 2026
  - 3D Vision
  - Free-viewpoint video
  - 3D Gaussian splatting
  - real-time streaming
  - dynamic scene reconstruction
  - adaptive bitrate
date: 2026-05-08
content_hash: f64012ea2ff4aea1
---

# StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video

**Conference**: AAAI 2026
**arXiv**: [2511.06046](https://arxiv.org/abs/2511.06046)
**Code**: Unavailable
**Area**: 3D Vision
**Keywords**: Free-viewpoint video, 3D Gaussian splatting, real-time streaming, dynamic scene reconstruction, adaptive bitrate

## TL;DR

This paper proposes StreamSTGS, a streamable spatial-temporal Gaussian grid representation that encodes canonical 3D Gaussian attributes as 2D images and temporal features as video, enabling real-time free-viewpoint video streaming at only 170 KB per frame. Reconstruction quality is maintained (PSNR 32.30 dB) through Transformer-assisted training and a sliding window mechanism.

## Background & Motivation

### Core Problem

Free-viewpoint video (FVV) is a core application of VR, but 4D representations require enormous storage, and real-time transmission faces three key challenges:

**Training efficiency**: Frame-by-frame training strategies lead to accumulated errors.

**Rendering speed**: NeRF-based methods cannot achieve real-time rendering.

**Transmission efficiency**: Existing 3DGS-based FVV methods store up to 10 MB per frame, precluding real-time streaming.

### Limitations of Prior Work

**NeRF-based methods** (StreamRF, ReRF, HPC): Support residual streaming but volumetric rendering prevents real-time performance.

**3DGS-based methods** exhibit the following issues:
- **3DGStream**: Directly streams InstantNGP-predicted attribute offsets, resulting in large model size.
- **HiCoM**: Stores and streams Gaussian attribute offsets but suffers from high spatial-temporal redundancy.
- **QUEEN/4DGC**: Employ entropy coding to compress attribute offsets, but incur high decoding latency.
- **Frame-by-frame training** introduces significant accumulated errors and requires training multiple LoD models for different network conditions.
- Accessing frame $i$ requires waiting for inference of all preceding $i-1$ frames.

### Core Motivation

The key insight is that decoupling dynamic 3DGS into **canonical Gaussians + temporal features + deformation fields**, then storing canonical Gaussian attributes as images and temporal features as video, enables efficient compression and transmission via conventional video codecs while naturally supporting adaptive bitrate control without additional training.

## Method

### Overall Architecture

StreamSTGS adopts a GOP (Group of Pictures) structure, partitioning long videos into independent groups. Each GOP contains:
- **Canonical 3D Gaussian set $\mathcal{G}$**: Represents static geometry.
- **Temporal features $\mathcal{E}$**: Learns dynamic characteristics.
- **Deformation fields**: Composed of multiple MLP decoders predicting per-attribute deformations.

Canonical Gaussian attributes are organized into 2D images (via the PLAS sorting algorithm), and temporal features are encoded as video, compressed and transmitted using standard codecs such as H.264/HEVC.

### Key Designs

#### 1. **Sliding Window Temporal Feature Aggregation**: Capturing Local Motion Relationships

**Function**: Aggregates neighboring temporal features within a window of size $W$ as input to the deformation field.

**Mechanism**: At timestamp $t_i$, features $e_{i-1}, e_i, e_{i+1}$ are concatenated ($W=3$) and processed by a temporal MLP $D_t$ to obtain $f_i$:

$$fe_i = \text{concat}(e_{i-1}, e_i, e_{i+1})$$
$$f_i = D_t(fe_i, \gamma(t_i))$$

where $\gamma(t_i)$ is the positional encoding of the timestamp. The total number of temporal features is $E = G + W - 1$.

**Design Motivation**: Real-world motion exhibits temporal correlation with continuity across adjacent frames. The sliding window allows the model to leverage contextual information for improved motion prediction while reducing the total number of temporal features.

#### 2. **Transformer-Guided Auxiliary Training**: Learning Global Motion

**Function**: A Transformer module is introduced to learn global motion patterns, with knowledge distilled into StreamSTGS via self-distillation. The Transformer is discarded at inference to maintain high FPS.

**Mechanism**: A dual-branch design is adopted — the Gaussian branch and the auxiliary branch share the deformation field. In the auxiliary branch, the Transformer receives features from all timestamps as a sequence and applies Batch Attention to learn global motion:

$$f'_i = \mathcal{F}(f_i, \gamma(t_i), \gamma(X))$$

A self-distillation loss transfers global motion knowledge from the Transformer to the main branch:

$$\mathcal{L}_{sd} = \|f_i - f'_i\|_1$$

**Design Motivation**: The sliding window captures only local motion and struggles with large-scale motion (e.g., walking). The Transformer attends to all timestamps and learns global motion patterns. Since Transformer inference is slow, it is used only during training and discarded at inference.

#### 3. **Spatial-Temporal Consistency Regularization**: Ensuring High Compression Efficiency

**Function**: Applies spatial smoothness regularization to canonical Gaussian attributes and temporal consistency regularization to temporal features.

**Mechanism**:
- **Spatial regularization** $\mathcal{L}_{spatial}$: Applies Gaussian filter smoothing to the sorted 2D attribute images.
- **Temporal regularization** $\mathcal{L}_{temp}$: Measures differences between adjacent temporal features using the Huber loss:

$$L_i = \text{huber}(e_{i-1} - e_i), \quad \mathcal{L}_{temp} = \text{mean}(L_i, L_{i+1})$$

**Design Motivation**: The Huber loss elegantly applies MSE to static Gaussians (small differences) to enforce consistency and reduce data volume, while applying MAE to dynamic Gaussians (large differences) to tolerate outliers and avoid over-smoothing. This directly improves compression efficiency for video codecs.

#### 4. **Dynamic-Aware Density Control and Gaussian Relocation**

**Dynamic-aware density**: Restricts the SSIM loss to dynamic region pixels, encouraging more Gaussians to be allocated to dynamic areas:

$$\mathcal{L}_c = (1-\beta) \cdot \|\mathcal{I}_i - \mathcal{I}_i^{gt}\|_1 + \beta \cdot \text{SSIM}(\mathcal{I}_i \cdot I^{mask} - \mathcal{I}_i^{gt} \cdot I^{mask})$$

**Gaussian relocate**: Once the Gaussian count reaches the upper limit (150k), unnecessary Gaussians are relocated to more optimal positions to avoid local optima.

### Loss & Training

The total loss function is:

$$\mathcal{L} = \mathcal{L}_c + 2\mathcal{L}_t + \mathcal{L}_{spatial} + \alpha_{temp} \cdot \mathcal{L}_{temp} + \alpha_o \cdot \mathcal{L}_o + \alpha_{sd} \cdot \mathcal{L}_{sd}$$

The training pipeline first trains a coarse 3DGS model (3,000 iterations), followed by fine-grained training for each GOP (12,000/7,000 iterations). Small noise ($\lambda=0.001$) is added to sliding window features during training to simulate compression artifacts.

## Key Experimental Results

### Main Results

**N3DV Dataset**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Storage (KB/frame)↓ | K.F. Size (MB)↓ | Decoding (ms)↓ | FPS↑ | Training (s)↓ | Adaptive Bitrate |
|--------|-------|-------|--------|---------------------|-----------------|----------------|------|---------------|------------------|
| TeTriRF | 30.07 | 0.900 | 0.299 | 65.89 | 2.03 | 149 | 1.53 | 32 | ✓ |
| 3DGStream | 30.73 | 0.935 | 0.147 | 8204 | 42.22 | 7×n | 72 | 17 | ✗ |
| HiCoM | 31.32 | 0.939 | 0.147 | 10704 | 83.35 | 0 | 163 | 10 | ✗ |
| 4DGC | 31.52 | 0.941 | 0.143 | 784 | 21.94 | 2.5×n | 78.6 | 62 | ✗ |
| **Ours** | **32.30** | **0.943** | 0.147 | **173.6** | 3.86 | 8 | 100 | 67 | **✓** |

**MeetRoom Dataset**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Storage (KB/frame)↓ | FPS↑ |
|--------|-------|-------|--------|---------------------|------|
| 3DGStream | 26.41 | 0.90 | 0.24 | 4108 | 121 |
| HiCoM | 26.69 | 0.90 | 0.23 | 5535 | 275 |
| 4DGC | 27.11 | 0.91 | 0.23 | 1196 | 110 |
| **Ours** | **27.41** | **0.92** | **0.21** | **142** | 126 |

### Ablation Study

| Configuration | PSNR | Storage (KB) | Training (s) | Note |
|---------------|------|--------------|--------------|------|
| Full model | **32.30** | 173.59 | 67 | Complete model |
| w/o Auxiliary training | 31.99 | 174.51 | 29 | Quality drops without Transformer |
| w/o Dynamic density | 32.07 | 114.15 | 69 | Dynamic regions blurred; better compression |
| w/o Temporal reg. | 32.23 | **319.48** | 64 | Storage increases by ~84% |
| w/o Gaussian relocate | 32.11 | 169.70 | 68 | Susceptible to local optima |

**Sliding Window Size Ablation**:

| Window Size W | PSNR | SSIM | Storage (KB) |
|---------------|------|------|--------------|
| W=1 (no window) | 32.01 | 0.941 | 298.06 |
| W=3 (default) | **32.30** | **0.944** | 173.59 |
| W=5 | 32.26 | 0.944 | 176.05 |

**Compression QP Ablation**:

| QP | N3DV PSNR | N3DV Storage (KB) | MeetRoom PSNR |
|----|-----------|-------------------|---------------|
| 16 | 32.36 | 247.52 | 27.46 |
| 20 (default) | 32.30 | 173.59 | 27.41 |
| 28 | 31.68 | 87.95 | 27.02 |
| 32 | 30.76 | 68.35 | 26.46 |

### Key Findings

1. **Remarkable storage efficiency**: Only 170 KB per frame — 47× smaller than 3DGStream and 62× smaller than HiCoM.
2. **State-of-the-art PSNR**: Outperforms all baselines, surpassing the second-best method by 0.78 dB on N3DV.
3. **Native adaptive bitrate support**: Different bitrates are achieved by adjusting the QP parameter without any additional training.
4. **Importance of temporal consistency regularization**: Removing it increases storage by 84%, confirming its role as a key driver of compression efficiency.
5. **GOP length of 60 is optimal**: Best trade-off among quality, storage, and training time.

## Highlights & Insights

1. **Elegant design philosophy**: Mapping 3DGS attributes and temporal features to images and video — the two most mature media formats — fully leverages existing codec infrastructure.
2. **Training-free adaptive bitrate**: Bitrate is controlled directly via the video codec's QP parameter, eliminating the need to train multiple LoD models.
3. **Sophisticated use of Huber loss**: Different penalty mechanisms for static and dynamic Gaussians elegantly balance consistency and expressiveness.
4. **Noise injection for compression robustness**: Quantization noise introduced by codecs is simulated during training to improve robustness.

## Limitations & Future Work

1. **All Gaussians carry temporal features**: Static Gaussians do not require temporal features; handling them separately could further reduce storage and improve FPS (acknowledged in the paper).
2. **GOP boundary discontinuities**: Independently trained GOPs may introduce visual discontinuities at boundaries.
3. **Dependence on precomputed dynamic masks**: Knowledge of which pixels are dynamic is required, adding a preprocessing step.
4. **Transformer auxiliary training increases training cost**: Although not used at inference, training time increases from 29 s to 67 s.

## Related Work & Insights

- **vs. VideoRF/TeTriRF**: These methods use NeRF with 2D grids to store voxel features compressed as video, but suffer from slow rendering and severe quantization. StreamSTGS uses 3DGS for real-time rendering.
- The **PLAS sorting algorithm** from Compact-3DGS provides key inspiration for spatial grid organization.
- The **Batch Attention** concept from TimeFormer is adopted in the Transformer auxiliary training module.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Mapping 3DGS attributes to images/video is a creative cross-domain idea.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Two datasets, comprehensive quantitative comparisons, and detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Method is clearly described, though the dense notation requires careful reading.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses a practical bottleneck in real-time FVV streaming with strong applicability.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Motion Matters: Compact Gaussian Streaming for Free-Viewpoint Video Reconstruction](../../NeurIPS2025/3d_vision/motion_matters_compact_gaussian_streaming_for_free-viewpoint_video_reconstructio.md)
- [\[AAAI 2026\] Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field](physics-informed_deformable_gaussian_splatting_towards_unified_constitutive_laws.md)
- [\[ICCV 2025\] FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution](../../ICCV2025/3d_vision/flashdepth_real-time_streaming_video_depth_estimation_at_2k_resolution.md)
- [\[AAAI 2026\] MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video](mobgs_motion_deblurring_dynamic_3d_gaussian_splatting_for_blurry_monocular_video.md)
- [\[AAAI 2026\] Generalized Geometry Encoding Volume for Real-time Stereo Matching](generalized_geometry_encoding_volume_for_real-time_stereo_matching.md)

<!-- RELATED:END -->
