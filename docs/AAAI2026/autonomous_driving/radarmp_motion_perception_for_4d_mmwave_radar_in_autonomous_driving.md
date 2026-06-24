---
title: >-
  [Paper Note] RadarMP: Motion Perception for 4D mmWave Radar in Autonomous Driving
description: >-
  [AAAI 2026][Autonomous Driving][4D mmWave radar] This paper proposes RadarMP — the first unified architecture that jointly addresses mmWave radar object detection and scene flow estimation. It leverages energy flow consistency across adjacent-frame radar echo signals (tesseracts) for self-supervised training, achieving a detection probability of 69.5% (far exceeding the prior best of 44.1%) while enabling accurate 3D scene motion perception.
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "4D mmWave radar"
  - "scene flow estimation"
  - "object detection"
  - "self-supervised learning"
  - "motion perception"
date: 2026-05-08
content_hash: a9a9cc3e33aacf38
---

# RadarMP: Motion Perception for 4D mmWave Radar in Autonomous Driving

**Conference**: AAAI 2026
**arXiv**: [2511.12117](https://arxiv.org/abs/2511.12117)  
**Code**: [github.com/chengrui7/RadarMP](https://github.com/chengrui7/RadarMP)  
**Area**: Autonomous Driving
**Keywords**: 4D mmWave radar, scene flow estimation, object detection, self-supervised learning, motion perception

## TL;DR

This paper proposes RadarMP — the first unified architecture that jointly addresses mmWave radar object detection and scene flow estimation. It leverages energy flow consistency across adjacent-frame radar echo signals (tesseracts) for self-supervised training, achieving a detection probability of 69.5% (far exceeding the prior best of 44.1%) while enabling accurate 3D scene motion perception.

## Background & Motivation

### Advantages and Challenges of mmWave Radar

4D mmWave radar has emerged as a critical sensor for autonomous driving systems due to its all-weather operational capability (penetrating rain, snow, and fog). However, traditional CFAR-based detection methods rely on statistical assumptions and lack the ability to model complex background clutter and dynamic scenes, leading to degraded detection performance and producing sparse, heavily noisy radar point clouds.

### Limitations of Prior Work

**Decoupled detection and motion estimation**: Existing methods treat radar object detection and motion estimation as two independent stages; the sparse, noisy point clouds produced during detection directly degrade subsequent scene flow estimation accuracy.

**Contradiction in optical sensor supervision**: Using LiDAR/cameras to supervise radar detection models (e.g., RPDNet) forces the radar to attend to low-reflectivity features, undermining the complementarity of multimodal perception.

**Scarcity of radar scene flow research**: Only two prior works — RaFlow and CMFlow — have studied scene flow on radar point clouds, with performance significantly lagging behind LiDAR-based methods.

### Core Motivation

**The energy flow direction of target points in adjacent-frame radar echo signals should be consistent with the motion field, whereas the energy flow of noise points is disordered and irregular.** This key observation motivates a joint modeling approach that simultaneously performs object detection and motion estimation.

## Method

### Overall Architecture

RadarMP takes two consecutive 4D radar tesseracts ($\mathbf{S}, \mathbf{T} \in \mathbb{R}^{D \times R \times A \times E}$, corresponding to the Doppler, range, azimuth, and elevation dimensions) as input and jointly outputs:
1. **Segmentation mask** $\mathbf{M} \in \{0,1\}^{R \times A \times E}$: distinguishing target points from noise points
2. **3D scene flow field** $\mathbf{F} = \{\mathbf{f}_i\}$: displacement vectors for each target point along the range–azimuth–elevation axes

The overall pipeline is:

Tesseract → Doppler Channel Encoding → 3D Feature Pyramid → Multi-Scale Deformable Cross-Attention (Correlation Feature Extraction) → Global Motion Pattern-Aware Module → Segmentation + Flow Prediction Decoder

### Key Designs

#### 1. Doppler Channel Encoding

**Function**: Transforms the Doppler dimension of the tesseract from a redundant representation into compact, motion-aware features.

**Mechanism**: Unlike prior works that simply apply average or max pooling over the Doppler dimension, this paper treats the Doppler axis as a feature channel and encodes it via an MLP. Softmax and Gumbel-Softmax are introduced to probabilistically encode Doppler velocities:

$$\digamma_{v1} = \mathrm{sum}(\mathrm{matmul}(Ax_d, \mathrm{Softmax}(P_d)))$$
$$\digamma_{v2} = \mathrm{sum}(\mathrm{matmul}(Ax_d, \mathrm{GumbelSoftmax}(P_d)))$$

where $Ax_d \in \mathbb{R}^D$ is the Doppler axis and $P_d \in \mathbb{R}^D$ is the raw Doppler energy value.

**Design Motivation**: The Doppler axis encodes motion-related attributes at each spatial location — the energy distribution reflects the confidence of each spatial position over different Doppler velocities. Preserving this information is crucial for both segmentation (semantic cues) and scene flow estimation (physical cues). Through this encoding, the Doppler dimension is compressed from $D$ to $D/8$ while retaining key motion features.

#### 2. Correlation Feature Extraction

**Function**: Establishes dense motion correlations between two frames of tesseracts.

**Mechanism**: A multi-scale deformable cross-attention mechanism is employed, using the source frame as Query and the target frame as Value to extract inter-frame correlation features.

**Correlation reference point generation**: The two frames are projected onto three 2D planes (RA, RE, AE); a pretrained PWC-Net predicts the energy flow direction on each plane, yielding three multi-scale 2D flow components. These are averaged to obtain 3D reference point coordinates:

$$r'_l = r_l + \frac{1}{2}(\mathbf{f}^l_{ra}(r) + \mathbf{f}^l_{re}(r))$$
$$a'_l = a_l + \frac{1}{2}(\mathbf{f}^l_{ra}(a) + \mathbf{f}^l_{ae}(a))$$
$$e'_l = e_l + \frac{1}{2}(\mathbf{f}^l_{re}(e) + \mathbf{f}^l_{ae}(e))$$

**Multi-scale deformable cross-attention**: A three-level feature pyramid is extracted via ResNet3D, and multi-scale deformable attention is applied to achieve cross-scale inter-frame correlation:

$$\digamma^{\mathbf{C}}_l = \mathrm{MSDeformAttn}(\mathbf{q}, \mathbf{p}, \{\mathbf{v}_{\mathbf{T}}^l\}), \quad \mathbf{q} \in \mathbf{q}_{\mathbf{S}}^l$$

The correlation representation $\digamma_c \in \mathbb{R}^{C_c \times R \times A \times E}$ is finally aggregated through an FPN.

**Design Motivation**: Performing dense correlation directly in 3D spherical coordinate space incurs prohibitive memory overhead. Deformable attention achieves precise correlation at far lower computational complexity than brute-force search by learning sampling offsets and attention weights.

#### 3. Global Motion Pattern-Aware Module

**Function**: Captures global motion context to distinguish the disordered motion of noise, the globally coherent motion of static targets, and the locally coherent motion of dynamic targets.

**Mechanism**: Two self-attention mechanisms are designed:

- **Global Patch Self-Attention**: Divides the correlation features into $4 \times 4 \times 4$ patches, treats each patch as a token fed into a Transformer encoder, and uses polar coordinate positional encoding.
- **Direction Slice Self-Attention**: Slices along the AE plane, treating all range bins at the same $(a,e)$ position as a single token, with direction vectors used as positional encoding.

**Design Motivation**: Different point types exhibit fundamentally distinct motion patterns — noise is disordered, static targets are globally consistent, and dynamic targets are locally consistent. The two attention mechanisms capture these patterns at the volumetric and directional levels respectively, providing comprehensive segmentation cues.

### Loss & Training

Three self-supervised loss functions specifically tailored to radar characteristics are designed, requiring **no explicit annotations**:

$$\mathcal{L} = \mathcal{L}_{se} + \mathcal{L}_{ef} + \mathcal{L}_{rfs}$$

**1. Segmentation Energy Loss $\mathcal{L}_{se}$**: Supervises segmentation based on energy distribution — higher energy implies higher likelihood of being a target — and enforces consistency of segmentation masks across frames:

$$\mathcal{L}_{se} = \mathbf{M}_s - \mathrm{sigmoid}(E_f^{\mathbf{S}} - \tau_f^{\mathbf{S}}) + \mathbf{M}_s \times (\mathrm{warp}(\mathbf{M}_s, \mathbf{F}_s) - \mathrm{sigmoid}(E_f^{\mathbf{T}} - \tau_f^{\mathbf{T}}))$$

**2. Energy Flow Loss $\mathcal{L}_{ef}$**: Enforces consistency between the flow field of target points and their energy flow direction; energy intensity weighting reduces the influence of noise:

$$\mathcal{L}_{ef} = E_f^{\mathbf{S}} \times (E_f^{\mathbf{S}} - \mathrm{warp}(E_f^{\mathbf{T}}, \mathbf{F}_s))$$

**3. Radial Flow Segmentation Loss $\mathcal{L}_{rfs}$**: The Doppler value multiplied by the inter-frame time interval should approximate the radial projection of the true flow for target points:

$$\delta_v = \digamma_v - \frac{\mathrm{warp}(C, \mathbf{F}_s) - C}{\Delta t} \odot O$$
$$\mathcal{L}_{rfs} = \mathbf{M}_s - \mathrm{sigmoid}(\alpha(\beta - \delta_v^2))$$

**Training details**: Adam optimizer with an initial learning rate of 0.001, decayed by 0.9 every 2 epochs; trained for 250 epochs on 3 × RTX 3090 GPUs; inference speed of 7.6 fps with 7.5 GB GPU memory.

## Key Experimental Results

### Main Results

#### Object Detection Results (K-Radar Dataset)

| Method | $P_d$ (%)↑ | $P_{fa}$ (%)↓ | CD (m)↓ | SNR (dB)↑ |
|------|-----------|--------------|---------|-----------|
| OS-CFAR | 1.643 | **0.311** | 10.030 | 5.477 |
| RPDNet | 9.311 | 1.821 | 7.590 | 5.175 |
| Radelft | 44.121 | 6.200 | 6.553 | 4.329 |
| **RadarMP** | **69.458** | 1.335 | **3.378** | **5.232** |

RadarMP's detection probability (69.5%) substantially surpasses the previous best method Radelft (44.1%), representing a **57.4%** improvement, while maintaining a low false alarm rate (1.34%) and the best Chamfer distance (3.38 m).

#### Scene Flow Estimation Results

| Method | Segmentation | EPE3D (m)↓ | AccS3D (%)↑ | AccR3D (%)↑ | Outlier3D (%)↓ |
|------|----------|-----------|------------|------------|---------------|
| RaFlow + OS-CFAR | Traditional | 0.329 | 11.635 | 20.887 | 82.399 |
| CMFlow + Radelft | Learned | 0.190 | 20.151 | 46.584 | 65.263 |
| CMFlow + RadarMP-P | Ours (det.) | 0.168 | 20.396 | 47.985 | 50.841 |
| **RadarMP (joint)** | **Joint** | **0.157** | **21.365** | **46.872** | **44.734** |

The joint modeling approach (RadarMP) achieves an EPE3D of 0.157 m, outperforming all decoupled solutions, with Outlier3D reduced from 65.3% to 44.7%.

### Ablation Study

#### Loss Function Ablation

| $\mathcal{L}_{se}$ | $\mathcal{L}_{ef}$ | $\mathcal{L}_{rfs}$ | $P_d$ (%)↑ | $P_{fa}$ (%)↓ | EPE3D (m)↓ |
|:---:|:---:|:---:|-----------|--------------|-----------|
| ✓ | ✓ | ✗ | 62.033 | 2.258 | 0.209 |
| ✓ | ✗ | ✓ | 56.224 | 3.847 | 0.788 |
| ✗ | ✓ | ✓ | 19.846 | 17.136 | 0.621 |
| **✓** | **✓** | **✓** | **69.458** | **1.335** | **0.157** |

All three loss components are indispensable: removing the segmentation energy loss causes the detection probability to drop sharply to 19.8%; removing the energy flow loss degrades EPE3D to 0.788 m; the full configuration achieves the best performance across all metrics.

### Key Findings

1. **Joint modeling outperforms decoupled approaches**: Simultaneously performing detection and motion estimation yields mutual benefits for both tasks.
2. **Energy flow consistency is an effective self-supervised signal**: Strong motion perception is achievable without any annotations.
3. **All-weather robustness**: Reliable detection and motion estimation are maintained under conditions such as heavy snow where cameras and LiDAR severely degrade.
4. **Role of PWC-Net reference points**: Removing PWC-Net degrades EPE3D by 0.31 m, confirming that initial motion estimates are critical for deformable attention.

## Highlights & Insights

- **First radar framework for joint object detection and scene flow estimation**: Two long-separated tasks are unified into a single architecture.
- **Fully self-supervised**: No LiDAR supervision is required, preserving radar's sensing independence and complementarity.
- **Operating on low-level signals**: Raw radar tesseracts (4D echo signals) are used directly rather than post-processed point clouds, avoiding the sparsity and noise introduced by conventional preprocessing.
- **Motion-consistency-driven detection**: Physical priors (energy flow direction = motion direction) are exploited in the loss function design.

## Limitations & Future Work

- Radar's limited resolution precludes LiDAR-level textural information.
- Low-RCS targets (e.g., pedestrians wearing low-reflectivity clothing) in proximity to clutter remain challenging.
- Tesseracts consume substantial memory (approximately 300 MB per frame in K-Radar), requiring careful dimension trimming.
- The absence of precise point-level radar annotations makes fine-grained analysis of detection performance difficult.
- Future work may explore multi-frame fusion to improve temporal consistency.

## Related Work & Insights

- **RaFlow/CMFlow**: The only prior radar scene flow works, employing self/cross-modal supervision but with limited performance.
- **Deformable DETR**: The success of deformable attention in reducing computational complexity is transferred to the 3D radar domain.
- **PWC-Net**: A classical optical flow estimation method repurposed for 2D energy flow prediction, providing initialization for 3D reference points.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First radar architecture for joint detection and flow estimation; self-supervised loss design is elegant)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Ablations are comprehensive, but validation is limited to a single dataset)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure with well-articulated motivation)
- Value: ⭐⭐⭐⭐⭐ (Opens a new paradigm for 4D radar motion perception; significant implications for all-weather autonomous driving)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](../../NeurIPS2025/autonomous_driving/v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)
- [\[CVPR 2026\] Hybrid Robust Collaborative Perception with LiDAR-4D Radar Fusion under Adverse Weather Conditions](../../CVPR2026/autonomous_driving/hybrid_robust_collaborative_perception_with_lidar-4d_radar_fusion_under_adverse_.md)
- [\[CVPR 2026\] RPGFusion: 4D Radar Prior-Guided Multi-Modal Fusion for 3D Detection](../../CVPR2026/autonomous_driving/rpgfusion_4d_radar_prior-guided_multi-modal_fusion_for_3d_detection.md)
- [\[CVPR 2026\] AdaRadar: Rate Adaptive Spectral Compression for Radar-based Perception](../../CVPR2026/autonomous_driving/adaradar_rate_adaptive_spectral_compression_for_radar-based_perception.md)
- [\[CVPR 2026\] DrivePI: Spatial-aware 4D MLLM for Unified Autonomous Driving Understanding, Perception, Prediction and Planning](../../CVPR2026/autonomous_driving/drivepi_spatial-aware_4d_mllm_for_unified_autonomous_driving_understanding_perce.md)

</div>

<!-- RELATED:END -->
