---
title: >-
  [Paper Note] CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration
description: >-
  [CVPR 2026][3D Vision][Point cloud registration] CMHANet proposes a three-stage hybrid attention mechanism (geometric self-attention → image aggregation attention → source-target cross-attention) to fuse 2D image texture…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Point cloud registration"
  - "cross-modal fusion"
  - "hybrid attention"
  - "RGB-D"
  - "contrastive learning"
  - "KPConv"
date: 2026-05-08
content_hash: 509c3f84fcf44a34
---

# CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration

**Conference**: CVPR 2026
**arXiv**: [2603.12721](https://arxiv.org/abs/2603.12721)
**Code**: [Available](https://github.com/DongXu-Zhang/CMHANet)
**Area**: 3D Vision / Point Cloud Registration
**Keywords**: Point cloud registration, cross-modal fusion, hybrid attention, RGB-D, contrastive learning, KPConv

## TL;DR

CMHANet proposes a three-stage hybrid attention mechanism (geometric self-attention → image aggregation attention → source-target cross-attention) to fuse 2D image texture semantics with 3D point cloud geometric information, complemented by a cross-modal contrastive loss. The method achieves state-of-the-art registration recall on 3DMatch/3DLoMatch (92.4%/75.5%) and a zero-shot RMSE of only $0.76\times10^{-2}$ on TUM RGB-D.

## Background & Motivation

**Background**: Point cloud registration is a fundamental task in 3D vision, serving as a prerequisite for 3D reconstruction, AR, and scene understanding. Deep learning-based methods have become mainstream, and Transformer architectures (e.g., GeoTransformer) have demonstrated strong performance in capturing global context.

**Limitations of Prior Work**: (1) The majority of existing methods rely solely on 3D geometric information, overlooking the paired 2D images already provided by ubiquitous RGB-D sensors—point clouds lack texture while images lack 3D information, making them naturally complementary. (2) Existing multimodal methods (IMFNet/CMIGNet/PCR-CG) employ generic fusion mechanisms without fine-grained modeling of geometric–visual feature interactions. (3) Noise, sparsity, and low overlap in real-world scenes degrade feature quality.

**Key Challenge**: 3D point clouds offer precise geometry but lack descriptive texture, while images provide dense semantics but no 3D structure. The core challenge lies in designing a fine-grained cross-modal attention mechanism that enables deep complementarity between the two modalities.

**Goal**: Design an intelligent cross-modal attention mechanism that accurately injects 2D visual semantics into 3D geometric features, improving registration accuracy and robustness in challenging scenarios (low overlap, noise).

**Key Insight**: Three attention types are functionally decoupled—self-attention captures global structure, aggregation attention fuses cross-modal information, and cross-attention establishes correspondences—applied in alternating iterations $N$ times for progressive feature enhancement.

**Core Idea**: The three-stage hybrid attention allows each 3D superpoint to simultaneously absorb structural context from its own point cloud, 2D image semantics, and correspondence information from the target cloud.

## Method

### Overall Architecture

The pipeline consists of four stages: **(1) Feature Extraction & Downsampling**—KPConv-FPN extracts superpoints and their features from the point cloud; ResUNet-50 extracts image features. **(2) Hybrid Attention Superpoint Matching**—three attention types alternate for $N$ iterations, followed by Sinkhorn ($L=50$) to generate a doubly stochastic matching matrix with a learnable dustbin. **(3) Dense Point Correspondence Module**—refines from coarse superpoint matches to fine point-to-point correspondences. **(4) Transformation Estimation**—weighted SVD computes local transformations; a Local-to-Global (LGR) verification strategy selects the optimal global transformation.

### Key Designs

1. **Geometric Self-Attention**: Each superpoint attends to all superpoints within the same point cloud. The key innovation lies in fusing learned features with geometric positional encodings in the Key:
$$e_{ij} = \frac{(\hat{F}_i^P W_q)(\hat{F}_j^P W_k + E_{ij}^P W_g^{Key})^\top}{\sqrt{d_k}}$$
The geometric encoding $E_{ij}^P = E_{ij}^D W_D + \max_r\{E_{ijr}^A W_A\}$ aggregates distance encodings (sinusoidal + MLP) and angular encodings. **Design Motivation**: Enables attention to jointly perceive feature similarity and spatial geometric relationships.

2. **Geometric Aggregation-Attention**: The core cross-modal fusion module. 3D superpoints serve as Query; 2D image patches serve as Key/Value. Both Q and K are injected with their respective modality positional encodings—3D coordinate embedding $E_i^P$ and 2D pixel coordinate embedding $E_j^I$—projected into a shared semantic space via independent $W_f$ and $W_g$:
$$e_{ij} = \frac{(\hat{F}_i^P W_q + E_i^P W_g)(\hat{F}_j^I W_k + E_j^I W_f)^\top}{\sqrt{d_k}}$$
**Design Motivation**: Each 3D point selectively absorbs the most relevant 2D semantic cues; positional encoding injection resolves ambiguity from repetitive textures.

3. **Cross-Modal Contrastive Loss ($\mathcal{L}_{cmc}$)**: Contrastive learning is constructed at the superpoint level between 3D geometric features and their corresponding image features. Diagonal entries are positive samples; off-diagonal entries are negatives:
$$\mathcal{L}_{cmc} = -\frac{1}{N_P} \sum_i \log \frac{\exp(s[i,i])}{\sum_j \exp(s[i,j])}$$
This is effective even with batch size = 1, since $N_P$ superpoints provide sufficient positive and negative pairs. **Design Motivation**: Enforces cross-modal feature consistency, aligning 3D and 2D features in a shared space.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_c + \mathcal{L}_f + \lambda \mathcal{L}_{cmc}$ ($\lambda=0.5$). $\mathcal{L}_c$ is an overlap-aware circle loss for coarse matching (overlap $>10\%$ as positive, no overlap as negative); $\mathcal{L}_f$ is a point-level fine matching negative log-likelihood loss; $\mathcal{L}_{cmc}$ is the cross-modal contrastive loss. Sinkhorn runs for $L=50$ iterations to generate the doubly stochastic matrix with a learnable dustbin for outlier handling. Implementation: PyTorch, RTX 3090, Adam optimizer, 50 epochs, lr=$10^{-4}$ with exponential decay of 0.05/epoch, matching radius $\tau_a=5$ cm.

## Key Experimental Results

### Main Results

| Dataset | Metric | CMHANet | GeoTransformer | CoFiNet | PCR-CG |
|---|---|---|---|---|---|
| 3DMatch | RR% (5000) | **92.4** | — | 89.3 | 89.4 |
| 3DLoMatch | RR% (5000) | **75.5** | — | 67.5 | 66.3 |
| 3DMatch | IR% (250) | **86.2** | — | 52.2 | — |
| 3DLoMatch | IR% (250) | **58.3** | — | 26.6 | — |
| 3DMatch | RRE (°) | **1.764** | 1.772 | 2.002 | — |
| 3DMatch | RTE (m) | **0.060** | 0.061 | 0.064 | — |

### Ablation Study

| Ablation | 3DMatch RR% | 3DLoMatch RR% | Change |
|---|---|---|---|
| Full CMHANet | 92.4 | 75.5 | — |
| w/o Image Module | 90.5 | 71.9 | −1.9/−3.6 |
| w/o Hybrid Attention | 90.5 | 72.4 | −1.9/−3.1 |
| w/o Aggregation-Attention | 91.4 | 73.6 | −1.0/−1.9 |
| w/o Contrastive Loss | 91.4 | 73.8 | −1.0/−1.7 |
| LGR Estimation (no RANSAC) | 91.9 | 74.2 | 100× faster |

### Key Findings

- **Large gain in Inlier Ratio**: On 3DLoMatch with 250 samples, IR improves from 33.1% (OIF-PCR) to 58.3% (+76%), indicating a fundamental improvement in feature discriminability.
- **Greater improvement on 3DLoMatch (low overlap)**: RR increases from 66.3% (PCR-CG) to 75.5% (+9.2%), demonstrating higher value of cross-modal fusion in challenging scenarios.
- **Zero-shot TUM RGB-D**: RMSE of $0.76\times10^{-2}$ substantially outperforms Robust ICP (1.69) and Teaser++ (14.06), evidencing strong generalization.
- **LGR vs. RANSAC**: Replacing RANSAC with LGR sacrifices only 0.5%/1.3% RR while achieving 100× speedup, making it suitable for real-time applications.
- **Image backbone**: ResUNet-50 > ResNet-101 ≈ ResNet-34; the multi-scale features from the UNet architecture are more effective for registration.

## Highlights & Insights

- The three attention types are functionally decoupled (self-attention / aggregation attention / cross-attention) and applied in alternating iterations, resulting in a logically coherent and progressively enriching design.
- Fusing features with geometric positional encodings in the Key outperforms simple concatenation, endowing the attention mechanism with spatial awareness.
- The cross-modal contrastive loss is elegantly designed—constructed at the superpoint level, it operates effectively even with batch size = 1.
- The substantial gain in Inlier Ratio (+76%) confirms the fundamental enhancement of feature discriminability through cross-modal fusion.

## Limitations & Future Work

- **Requires paired RGB-D input**: The method is inapplicable in pure LiDAR scenarios, limiting its applicability.
- **Increased inference time**: Image encoding introduces additional latency (0.144 s vs. CoFiNet's 0.115 s).
- **Extremely low overlap or textureless scenes**: Scenarios with $<10\%$ overlap or entirely textureless planar surfaces may cause failure.
- **Large-scale outdoor scenes**: Applicability to outdoor environments such as autonomous driving has not been validated.

## Related Work & Insights

- **vs. IMFNet**: Both are multimodal, but CMHANet substantially outperforms IMFNet on 3DLoMatch RR (75.5 vs. 48.4), demonstrating that hybrid attention is far more effective than simple attention-based fusion.
- **vs. PCR-CG**: Another multimodal method; CMHANet achieves +9.2% higher 3DLoMatch RR, with the core advantage attributed to its three-stage fine-grained fusion.
- **vs. GeoTransformer**: A single-modality SOTA; CMHANet further reduces RRE/RTE by incorporating image information.
- **Insight**: The three-stage attention design paradigm is generalizable to other 3D tasks requiring geometric–semantic fusion, such as 3D detection and segmentation.

## Rating

⭐⭐⭐⭐ (4/5)

**Rationale**: The method is carefully designed with clear motivation—each of the three attention types serves a distinct purpose. The approach achieves comprehensive state-of-the-art performance on 3DMatch/3DLoMatch, with a notably large Inlier Ratio gain (+76%). Both the cross-modal contrastive loss and the LGR-based RANSAC replacement represent practical innovations. Points are deducted for the dependency on paired RGB-D input, the lack of outdoor scene validation, and the fact that the overall approach is somewhat an engineering combination of individually non-novel components.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Hg-I2P: Bridging Modalities for Generalizable Image-to-Point-Cloud Registration via Heterogeneous Graphs](hg-i2p_bridging_modalities_for_generalizable_image-to-point-cloud_registration_v.md)
- [\[CVPR 2026\] AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis](affordgrasp_cross-modal_diffusion_for_affordance-aware_grasp_synthesis.md)
- [\[CVPR 2026\] Cross-Instance Gaussian Splatting Registration via Geometry-Aware Feature-Guided Alignment](cross-instance_gaussian_splatting_registration_via_geometry-aware_feature-guided.md)
- [\[ICCV 2025\] TurboReg: TurboClique for Robust and Efficient Point Cloud Registration](../../ICCV2025/3d_vision/turboreg_turboclique_for_robust_and_efficient_point_cloud_registration.md)
- [\[ICCV 2025\] BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes](../../ICCV2025/3d_vision/bufferx_towards_zeroshot_point_cloud_registration_in_diverse.md)

</div>

<!-- RELATED:END -->
