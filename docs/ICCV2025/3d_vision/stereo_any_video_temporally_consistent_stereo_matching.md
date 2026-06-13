---
title: >-
  [Paper Note] Stereo Any Video: Temporally Consistent Stereo Matching
description: >-
  [ICCV2025][3D Vision][Video stereo matching] This paper proposes Stereo Any Video, a framework that achieves spatially accurate and temporally consistent video stereo matching without relying on camera poses or optical f…
tags:
  - "ICCV2025"
  - "3D Vision"
  - "Video stereo matching"
  - "temporal consistency"
  - "monocular depth prior"
  - "cost volume"
  - "disparity estimation"
date: 2026-05-08
content_hash: b90e47c9e5da3863
---

# Stereo Any Video: Temporally Consistent Stereo Matching

**Conference**: ICCV2025
**arXiv**: [2503.05549](https://arxiv.org/abs/2503.05549)  
**Code**: [Project Page](https://tomtomtommi.github.io/StereoAnyVideo/)  
**Area**: 3D Vision / Stereo Matching
**Keywords**: Video stereo matching, temporal consistency, monocular depth prior, cost volume, disparity estimation

## TL;DR
This paper proposes Stereo Any Video, a framework that achieves spatially accurate and temporally consistent video stereo matching without relying on camera poses or optical flow. It integrates three core modules — monocular video depth foundation model priors (Video Depth Anything), all-to-all-pair correlation, and temporal convex upsampling — attaining state-of-the-art performance under zero-shot settings across multiple benchmarks.

## Background & Motivation

**Task Definition**: Video stereo matching estimates per-frame disparity maps from rectified left-right image sequences for 3D scene reconstruction, which is critical for downstream tasks such as autonomous driving, robotic navigation, and AR/VR.

**Limitations of Prior Work**:
   - **Image-based methods** (RAFT-Stereo, IGEV-Stereo, etc.): Do not exploit temporal information, leading to flickering and artifacts when applied directly to video.
   - **Video-based methods** (CODD, TemporalStereo, BiDAStereo, etc.): Rely on auxiliary signals (camera poses or optical flow) for temporal alignment; in dynamic scenes with complex camera motion, the limited accuracy of these auxiliary modules becomes a performance bottleneck.

**Core Observation**: Research in video generation demonstrates that stable feature representations are key to temporal coherence. Rather than relying on external auxiliary signals, the authors design their framework from the novel perspective of *feature robustness and stability*.

**Empirical Analysis of Monocular Depth Maps**: The authors compare DepthCrafter (monocular video depth) and RAFT-Stereo on Sintel. While DepthCrafter appears visually more consistent (higher human preference scores), its spatial and temporal errors are far greater than those of RAFT-Stereo (EPE: 8.68 vs. 1.42), demonstrating that monocular depth is *consistent but inaccurate*. Accordingly, only its features are used as a prior rather than the depth maps themselves.

## Method

### Overall Architecture
Given a rectified stereo video sequence $\{I_L^t, I_R^t\}_{t=1}^{T}$, the framework adopts a cascaded pipeline that progressively recovers full-resolution disparity from low resolution via three core modules:

### 1. Feature Extraction with Foundation Model Priors
- **Dual-branch feature extraction**: A trainable convolutional encoder (residual block structure) extracts image and context features, while a frozen Video Depth Anything (VDA) model extracts depth features.
- **Feature fusion**: A lightweight convolutional adapter downsamples VDA depth features (32 channels), which are then concatenated with convolutional features (96 channels) to form a stable 128-channel feature representation.
- **Why VDA-S instead of VDA-L**: VDA-L has 13× more parameters than VDA-S (381.8M vs. 28.4M), yet yields negligible performance gains while requiring more adaptation parameters and increasing training complexity.

### 2. All-to-All-Pair Correlation
- **Conventional approach**: At iteration $n$, the right feature is warped to the left feature coordinate using the previous disparity $d_{n-1}$, and a unidirectional dot-product correlation is computed within a local search window: $C_n(x,y) = \langle F_L(x,y), \hat{F}_R(x+r_x, y+r_y) \rangle$
- **Proposed approach**: Bidirectional correspondences are introduced by computing similarity between all potential matching pairs within two search windows: $C_n(x,y) = \langle F_L(x+r_x, y+r_y), \hat{F}_R(x+r_x, y+r_y) \rangle$
- **Advantages**: (1) Enhanced match verification and reduced ambiguity; (2) enforced matching smoothness through dense correspondences; (3) no dependence on optical flow, avoiding its accuracy bottleneck.
- **Cost volume dimensionality**: Expands from $H' \times W' \times (2r_x+1)(2r_y+1)$ to $H' \times W' \times (2r_x+1)^2(2r_y+1)^2$.

### 3. Temporal Cost Aggregation
- **MLP encoder**: Compresses the high-dimensional cost volume into a compact representation $E_n = \text{MLP}(C_n)$.
- **3D-GRU iterative update**: Employs separable 3D convolutions to aggregate information across spatial and temporal dimensions, incorporating super kernels and spatio-temporal attention for iterative disparity refinement.
- **Temporal Convex Upsampling**: A key contribution. Each high-resolution pixel is obtained as a weighted combination of $3 \times 3 \times 3 = 27$ low-resolution neighbors drawn from three frames (current $\pm 1$) using learnable weights:
  $$\mathbf{w} = \text{softmax}(\text{Conv3d}(h_n))$$
  $$D_n^t = \alpha \cdot \sum_{ijk} \mathbf{w}_{ijk} \odot \text{unfold}([d_n^{t-1}, d_n^{t}, d_n^{t+1}])$$
  This design makes the upsampling process itself temporally consistent, in contrast to conventional spatial-only upsampling.

### 4. Loss & Training
- A pure image-level L1 loss with weight $\gamma=0.9$ is adopted, without temporal losses such as OPW or TGM:
  $$\mathcal{L} = \sum_{t=1}^{T} \sum_{n=1}^{N} \gamma^{N-n} \|D_{gt}^t - D_n^t\|$$
- The authors find that temporal losses introduce an accuracy–consistency trade-off in video stereo matching with minimal benefit.

## Key Experimental Results

### Training Configuration
- Pre-trained on SceneFlow for 120K iterations, then fine-tuned on a mixture of Dynamic Replica + Infinigen SV + Virtual KITTI2 for 80K iterations.
- Training sequence length $T=5$; evaluation sequence length $T=20$; GRU iterations: $N=10$ (training), $N=20$ (evaluation).
- Total training time: approximately 6 days on A100 GPUs.

### Main Results

**Zero-Shot Generalization (SceneFlow training only, Table 2)**

| Dataset | Metric | BiDAStereo | MonSter | **Ours** |
|--------|------|------------|---------|----------|
| Sintel Final | TEPE↓ | 1.26 | 1.70 | **1.07** |
| Dynamic Replica | EPE↓ | 0.65 | 0.45 | **0.25** |
| Infinigen SV | TEPE↓ | 1.99 | 1.65 | **1.65** |
| Virtual KITTI2 | TEPE↓ | 1.02 | 0.73 | **0.74** |

**Mixed Data Training (Table 3)**

| Dataset | Metric | BiDAStereo | FoundationStereo | **Ours** |
|--------|------|------------|-----------------|----------|
| Spring | TEPE↓ | 0.90 | 1.78 | **0.77** |
| Sintel Final | TEPE↓ | 1.33 | — | **0.99** |
| KITTI Depth | TEPE↓ | 0.42 | 0.40 | **0.35** |

The model trained exclusively on synthetic data outperforms methods trained on real data (including in-domain data) by at least 15%.

### Ablation Study

**Core Findings (Table 4)**

| Component | Variant | Sintel TEPE↓ | DR TEPE↓ |
|------|------|-------------|----------|
| Prior | No depth prior | 1.47 | 0.092 |
| Prior | VDA-S ✓ | 1.27 | 0.083 |
| Correlation | 1D+2D local | 1.27 | 0.083 |
| Correlation | 1D+2D all-to-all ✓ | 1.21 | 0.076 |
| Upsampling | Bilinear | 1.26 | 0.085 |
| Upsampling | Temporal convex ✓ | 1.04 | 0.067 |
| Attention | Spatio-temporal attention ✓ | 1.07 | 0.057 |

**Efficiency Analysis (Table 6, 720×1280 resolution, 20 frames)**

| Method | Params (M) | VRAM (G) | MACs (T) |
|------|----------|---------|---------|
| DynamicStereo | 20.5 | 35.1 | 182.3 |
| BiDAStereo | 12.2 | 41.1 | 186.6 |
| **Ours** | 9.4 [+28.4 frozen] | 41.1 | 303.4 |

The proposed method has the fewest trainable parameters (9.4M) but the highest computational cost (303.4T MACs) due to the foundation model prior.

## Highlights & Insights

1. **Novel perspective**: Temporal consistency is addressed through *feature stability* rather than *auxiliary signal alignment*, circumventing the accuracy bottleneck of optical flow and pose estimation.

2. **"Consistent but inaccurate" empirical insight**: Quantitative and perceptual experiments demonstrate the limitations of monocular video depth (EPE far exceeding stereo matching), providing a compelling justification for using only features rather than depth maps.

3. **Elegant temporal convex upsampling**: The conventional spatial convex upsampling is extended to 3D; each pixel aggregates 27 temporal-neighborhood samples via softmax weights, enabling end-to-end learnable temporal smoothing.

4. **Strong zero-shot generalization**: Trained exclusively on synthetic data, the method performs well on real indoor and outdoor scenes, even surpassing methods trained on in-domain real data.

5. **Minimal loss design**: Strong temporal consistency is achieved with a simple L1 loss alone, demonstrating that temporal coherence is primarily determined by architectural design — feature extraction, cost aggregation, and upsampling — rather than explicit temporal loss supervision.

## Limitations & Future Work

1. **High computational cost**: MACs are 1.63× those of BiDAStereo (303.4T vs. 186.6T), mainly due to the frozen VDA backbone, limiting real-time applicability.
2. **Single model variant**: Only one model version currently exists; the authors plan to develop a model zoo with large and lightweight variants.
3. **Synthetic training data only**: Despite strong generalization, the effect of training on real data remains unexplored.
4. **Limited sequence length**: Trained with $T=5$ and evaluated with $T=20$; temporal consistency for longer videos has not been thoroughly validated.
5. **Failure of 3D correlation**: Ablations show that a 3D search window (cross-temporal correlation) performs poorly (TEPE 4.51), indicating that temporal cross-frame correlation is infeasible when frames are not aligned. Resolving this issue could further improve performance.

## Related Work & Insights

- **RAFT-Stereo [Lipson2021]**: The foundational iterative stereo matching architecture; this work extends the paradigm to video settings.
- **BiDAStereo [Jing2024]**: An optical-flow-based temporal stereo matching method; the most direct comparison baseline.
- **Video Depth Anything [Chen2025]**: A video monocular depth foundation model providing frozen feature priors.
- **FoundationStereo [Wen2025]**: An image-level method also leveraging Depth Anything features; this work extends the idea to video.
- **MonSter [Cheng2025]**: A dual-branch image-level method fusing monocular depth maps and features.
- **Implications for future research**: (1) The paradigm of foundation model priors with lightweight adaptation is transferable to other video 3D tasks; (2) the strategy of using features rather than depth maps may apply broadly to scenarios requiring depth-stereo fusion.

## Rating
- Novelty: ⭐⭐⭐⭐ (All-to-all-pair correlation and temporal convex upsampling are substantially original)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (7 datasets, detailed ablations, qualitative/quantitative results, and user study)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure and well-motivated arguments)
- Value: ⭐⭐⭐⭐ (New benchmark for video stereo matching with outstanding generalization)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ZeroStereo: Zero-shot Stereo Matching from Single Images](zerostereo_zero-shot_stereo_matching_from_single_images.md)
- [\[ICCV 2025\] Diving into the Fusion of Monocular Priors for Generalized Stereo Matching](diving_into_the_fusion_of_monocular_priors_for_generalized_stereo_matching.md)
- [\[CVPR 2026\] Lite Any Stereo: Efficient Zero-Shot Stereo Matching](../../CVPR2026/3d_vision/lite_any_stereo_efficient_zero-shot_stereo_matching.md)
- [\[ICCV 2025\] BANet: Bilateral Aggregation Network for Mobile Stereo Matching](banet_bilateral_aggregation_network_for_mobile_stereo_matching.md)
- [\[ICCV 2025\] Thermal Polarimetric Multi-view Stereo](thermal_polarimetric_multi-view_stereo.md)

</div>

<!-- RELATED:END -->
