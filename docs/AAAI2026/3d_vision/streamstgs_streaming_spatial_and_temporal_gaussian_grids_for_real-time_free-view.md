---
title: >-
  [Paper Note] StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video
description: >-
  [AAAI 2026][3D Vision][Free-Viewpoint Video] The authors propose StreamSTGS, a streamable spatial-temporal Gaussian grid representation. It encodes canonical 3D Gaussian attributes into 2D images and temporal features into videos, enabling real-time free-viewpoint video streaming (frame size of only 170KB) while ensuring reconstruction quality (PSNR 32.30dB) via a Transformer-guided auxiliary training and a sliding window mechanism.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Free-Viewpoint Video"
  - "3D Gaussian Splatting"
  - "Real-Time Streaming"
  - "Dynamic Scene Reconstruction"
  - "Adaptive Bitrate"
date: 2026-05-08
content_hash: 665f1ea2a805d207
---

# StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video

**Conference**: AAAI 2026  
**arXiv**: [2511.06046](https://arxiv.org/abs/2511.06046)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Free-Viewpoint Video, 3D Gaussian Splatting, Real-Time Streaming, Dynamic Scene Reconstruction, Adaptive Bitrate

## TL;DR

The authors propose StreamSTGS, a streamable spatial-temporal Gaussian grid representation. It encodes canonical 3D Gaussian attributes into 2D images and temporal features into videos, enabling real-time free-viewpoint video streaming (frame size of only 170KB) while ensuring reconstruction quality (PSNR 32.30dB) via a Transformer-guided auxiliary training and a sliding window mechanism.

## Background & Motivation

### Problem Definition

Free-viewpoint video (FVV) is a core application of VR, but 4D representations require massive storage, and real-time streaming faces three major challenges:

**Training Efficiency**: Frame-by-frame training strategies lead to cumulative errors.

**Rendering Speed**: NeRF-based methods cannot render in real-time.

**Transmission Efficiency**: Existing 3DGS-based FVV methods require up to 10MB of storage per frame, rendering real-time streaming impossible.

### Limitations of Prior Work

**NeRF-based methods** (StreamRF, ReRF, HPC): Although they can stream residuals, volumetric rendering hinders real-time rendering.

**3DGS-based methods** suffer from the following issues:
- **3DGStream**: Directly streams attribute offsets predicted by InstantNGP, resulting in a large model size.
- **HiCoM**: Stores and streams Gaussian attribute offsets, but exhibits high spatial-temporal redundancy.
- **QUEEN/4DGC**: Uses entropy coding to compress attribute offsets, but suffers from high decoding latency.
- **Frame-by-frame training strategies** introduce significant cumulative errors and require training multiple LoD models for different network conditions.
- To view the $i$-th frame, users must wait for all previous $i-1$ frames to complete inference.

### Core Motivation

The authors' key insight is: if dynamic 3DGS is decoupled into **canonical Gaussians + temporal features + deformation fields**, the canonical Gaussian attributes can be stored as images, and the temporal features can be encoded as videos. This allows leveraging legacy video codecs for highly efficient compression and transmission, while naturally supporting adaptive bitrate control without additional training.

## Method

### Overall Architecture

StreamSTGS adopts a GOP (Group of Pictures) structure, dividing long videos into multiple groups, where each group is reconstructed independently. Inside each GOP, StreamSTGS consists of:
- **Canonical 3D Gaussian set $\mathcal{G}$**: Represents static geometry.
- **Temporal features $\mathcal{E}$**: Learns dynamic features.
- **Deformation fields**: Composed of multiple MLP decoders to predict attribute deformations.

Ultimately, canonical Gaussian attributes are organized into 2D images (via the PLAS sorting algorithm), and temporal features are encoded into videos, which are then compressed and transmitted using standard codecs such as H.264/HEVC.

### Key Designs

#### 1. **Sliding Window Temporal Feature Aggregation**: Capturing Local Motion Relations

**Function**: Aggregates neighboring temporal features using a sliding window of size $W$ to serve as input for the deformation field.

**Mechanism**: At timestamp $t_i$, $e_{i-1}, e_i, e_{i+1}$ are concatenated ($W=3$) and processed through a temporal MLP $D_t$ to obtain the temporal feature $f_i$:

$$fe_i = \text{concat}(e_{i-1}, e_i, e_{i+1})$$
$$f_i = D_t(fe_i, \gamma(t_i))$$

Where $\gamma(t_i)$ represents the positional encoding of the timestamp. The total number of temporal features is $E = G + W - 1$.

**Design Motivation**: In the real world, motion exhibits temporal correlation; consecutive frames possess continuity. The sliding window allows the model to better predict motion for the current frame using contextual information, while simultaneously reducing the total number of temporal features.

#### 2. **Transformer-Guided Auxiliary Training Strategy**: Learning Global Motion

**Function**: Introduces a Transformer module to learn global motion, which then transfers knowledge to StreamSTGS via self-distillation. The Transformer is discarded during inference to maintain a high FPS.

**Mechanism**: A dual-channel design is adopted, where the Gaussian channel and the auxiliary channel share the deformation field. In the auxiliary channel, the Transformer treats the features of all timestamps as a sequence, learning global motion via Batch Attention:

$$f'_i = \mathcal{F}(f_i, \gamma(t_i), \gamma(X))$$

The self-distillation loss transfers the global motion knowledge learned by the Transformer to the main branch:

$$\mathcal{L}_{sd} = \|f_i - f'_i\|_1$$

**Design Motivation**: The sliding window can only capture local motion and struggles with large-scale motions (e.g., walking). The Transformer can attend to all timestamps to capture global motion patterns. However, since the Transformer has slow inference, it is only utilized during training and discarded at inference time.

#### 3. **Spatial-Temporal Consistency Regularization**: Ensuring High Compression Efficiency

**Function**: Imposes spatial smoothness regularization on canonical Gaussian attributes and temporal consistency regularization on temporal features.

**Mechanism**:
- **Spatial Regularization** $\mathcal{L}_{spatial}$: Imposes Gaussian filtering smoothness on the sorted 2D attribute images.
- **Temporal Regularization** $\mathcal{L}_{temp}$: Uses Huber loss to measure the differences between adjacent temporal features.

$$L_i = \text{huber}(e_{i-1} - e_i), \quad \mathcal{L}_{temp} = \text{mean}(L_i, L_{i+1})$$

**Design Motivation**: The subtlety of the Huber loss lies in deploying MSE for static Gaussians (small differences) to maintain consistency and reduce data volume, while utilizing MAE for dynamic Gaussians (large differences) to tolerate outliers and prevent over-smoothing. This directly enhances the compression efficiency of video codecs.

#### 4. **Dynamic-Aware Density Control and Gaussian Relocation**

**Dynamic-aware density**: Restricts the SSIM loss to only dynamic region pixels, forcing more Gaussians to be allocated to dynamic areas:

$$\mathcal{L}_c = (1-\beta) \cdot \|\mathcal{I}_i - \mathcal{I}_i^{gt}\|_1 + \beta \cdot \text{SSIM}(\mathcal{I}_i \cdot I^{mask} - \mathcal{I}_i^{gt} \cdot I^{mask})$$

**Gaussian relocate**: Once the number of Gaussians reaches the upper limit (150k), redundant Gaussians are relocated to better positions, preventing the optimization from falling into local minima.

### Loss & Training

Total loss function:

$$\mathcal{L} = \mathcal{L}_c + 2\mathcal{L}_t + \mathcal{L}_{spatial} + \alpha_{temp} \cdot \mathcal{L}_{temp} + \alpha_o \cdot \mathcal{L}_o + \alpha_{sd} \cdot \mathcal{L}_{sd}$$

Training scheme: A coarse 3DGS model is trained first (3,000 iterations), followed by fine-grained training on each GOP (12,000 / 7,000 iterations). During training, small noise ($\lambda=0.001$) is injected into the sliding window features to simulate compression loss.

## Key Experimental Results

### Main Results

**N3DV Dataset**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Storage (KB/Frame)↓ | K.F. Size (MB)↓ | Decoding (ms)↓ | FPS↑ | Training (s)↓ | Adaptive Bitrate |
|------|-------|-------|--------|------------|-------------|----------|------|---------|----------|
| TeTriRF | 30.07 | 0.900 | 0.299 | 65.89 | 2.03 | 149 | 1.53 | 32 | ✓ |
| 3DGStream | 30.73 | 0.935 | 0.147 | 8204 | 42.22 | 7×n | 72 | 17 | ✗ |
| HiCoM | 31.32 | 0.939 | 0.147 | 10704 | 83.35 | 0 | 163 | 10 | ✗ |
| 4DGC | 31.52 | 0.941 | 0.143 | 784 | 21.94 | 2.5×n | 78.6 | 62 | ✗ |
| **Ours** | **32.30** | **0.943** | **0.147** | **173.6** | 3.86 | 8 | 100 | 67 | **✓** |

**MeetRoom Dataset**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Storage (KB/Frame)↓ | FPS↑ |
|------|-------|-------|--------|------------|------|
| 3DGStream | 26.41 | 0.90 | 0.24 | 4108 | 121 |
| HiCoM | 26.69 | 0.90 | 0.23 | 5535 | 275 |
| 4DGC | 27.11 | 0.91 | 0.23 | 1196 | 110 |
| **Ours** | **27.41** | **0.92** | **0.21** | **142** | 126 |

### Ablation Study

| Configuration | PSNR | Storage (KB) | Training (s) | Description |
|------|------|---------|---------|------|
| Full model | **32.30** | 173.59 | 67 | Full model |
| w/o Auxiliary training | 31.99 | 174.51 | 29 | Slight performance degradation without Transformer assistance |
| w/o Dynamic density | 32.07 | 114.15 | 69 | Dynamic regions get blurry, but compression efficiency increases |
| w/o Temporal reg. | 32.23 | **319.48** | 64 | Storage surges nearly 2-fold |
| w/o Gaussian relocate | 32.11 | 169.70 | 68 | Falls into local minima |

**Ablation on Sliding Window Size**:

| Window Size W | PSNR | SSIM | Storage (KB) |
|-----------|------|------|---------|
| W=1 (No sliding window) | 32.01 | 0.941 | 298.06 |
| W=3 (Default) | **32.30** | **0.944** | 173.59 |
| W=5 | 32.26 | 0.944 | 176.05 |

**Ablation on Compression QP**:

| QP | N3DV PSNR | N3DV Storage (KB) | MeetRoom PSNR |
|----|-----------|-------------|---------------|
| 16 | 32.36 | 247.52 | 27.46 |
| 20 (Default) | 32.30 | 173.59 | 27.41 |
| 28 | 31.68 | 87.95 | 27.02 |
| 32 | 30.76 | 68.35 | 26.46 |

### Key Findings

1. **Astonishing Storage Efficiency**: Only 170KB per frame, which is 47 times smaller than 3DGStream and 62 times smaller than HiCoM.
2. **Superior PSNR**: Outperforms all baselines, achieving 0.78dB higher than the second-best method on the N3DV dataset.
3. **Inherent Support for Adaptive Bitrate**: Requires no extra training and achieves various bitrates by simply adjusting QP parameters.
4. **Crucial Role of Temporal Consistency Regularization**: Storage overhead increases by 84% without it, demonstrating its critical contribution to compression efficiency.
5. **GOP Length of 60 Strikes the Best Balance**: Serves as the optimal trade-off among quality, storage, and training time.

## Highlights & Insights

1. **Elegant Design Philosophy**: Decouples 3DGS attributes and temporal features, mapping them respectively to images and videos—the two most mature media formats. This fully utilizes existing codec infrastructures.
2. **Training-Free Adaptive Bitrate**: Directly controls the bitrate through QP parameters of the video codec, avoiding the need to train multiple LoD models like in previous concurrent approaches.
3. **Exquisite Application of Huber Loss**: Uses differing penalty mechanisms for static and dynamic Gaussians, ingeniously balancing temporal consistency and expressiveness.
4. **Noise Injection for Compression Tolerance**: Injecting noise during training replicates the quantization errors introduced by real-world codecs, enhancing robustness.

## Limitations & Future Work

1. **Applying Temporal Features Inhomogeneously**: Static Gaussians do not require temporal features; categorizing them could further reduce storage and improve the rendering FPS (as discussed in the paper).
2. **Discontinuity at GOP Boundaries**: Since different GOPs are trained independently, visual flickering might occur at boundaries.
3. **Reliance on Pre-computed Dynamic Masks**: Requires knowing which pixels are dynamic, thereby introducing an extra pre-processing step.
4. **Auxiliary Transformer Increases Training Overhead**: Although the Transformer is discarded during inference, training time increases from 29 seconds to 67 seconds.

## Related Work & Insights

- Comparison with VideoRF/TeTriRF: The latter utilize NeRF + 2D grids to store voxel features and compress them into videos, which suffers from slow rendering speed and severe quantization. In contrast, StreamSTGS leverages 3DGS to achieve real-time rendering.
- The PLAS sorting algorithm from Compact-3DGS provides critical inspiration for the spatial gridding design.
- The Batch Attention concept from TimeFormer is borrowed for the Transformer-guided auxiliary training module.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Mapping 3DGS attributes to images/videos is an interesting cross-domain concept.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluated on two datasets with comprehensive quantitative comparisons and detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — The methodology is explained clearly, though the heavy use of notations requires careful reading.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses the practical bottleneck of real-time FVV streaming with high applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Motion Matters: Compact Gaussian Streaming for Free-Viewpoint Video Reconstruction](../../NeurIPS2025/3d_vision/motion_matters_compact_gaussian_streaming_for_free-viewpoint_video_reconstructio.md)
- [\[AAAI 2026\] Physics-Informed Deformable Gaussian Splatting: Towards Unified Constitutive Laws for Time-Evolving Material Field](physics-informed_deformable_gaussian_splatting_towards_unified_constitutive_laws.md)
- [\[ICCV 2025\] FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution](../../ICCV2025/3d_vision/flashdepth_real-time_streaming_video_depth_estimation_at_2k_resolution.md)
- [\[AAAI 2026\] MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video](mobgs_motion_deblurring_dynamic_3d_gaussian_splatting_for_blurry_monocular_video.md)
- [\[AAAI 2026\] Generalized Geometry Encoding Volume for Real-time Stereo Matching](generalized_geometry_encoding_volume_for_real-time_stereo_matching.md)

</div>

<!-- RELATED:END -->
