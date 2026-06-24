---
title: >-
  [Paper Note] MonoWAD: Weather-Adaptive Diffusion Model for Robust Monocular 3D Object Detection
description: >-
  [ECCV 2024][Autonomous Driving][Monocular 3D Detection] MonoWAD is proposed to achieve robust monocular 3D object detection under various weather conditions. It learns clear-weather knowledge as a reference using a weather codebook and performs feature enhancement by modeling foggy effects as noise through a weather-adaptive diffusion model.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Monocular 3D Detection"
  - "Weather Adaptability"
  - "Diffusion Model"
  - "Weather Codebook"
  - "Foggy Weather Detection"
date: 2026-05-08
content_hash: 8a61cd66274f3ab4
---

# MonoWAD: Weather-Adaptive Diffusion Model for Robust Monocular 3D Object Detection

**Conference**: ECCV 2024  
**arXiv**: [2407.16448](https://arxiv.org/abs/2407.16448)  
**Code**: [Available](https://github.com/VisualAIKHU/MonoWAD)  
**Area**: Autonomous Driving  
**Keywords**: Monocular 3D Detection, Weather Adaptability, Diffusion Model, Weather Codebook, Foggy Weather Detection

## TL;DR

MonoWAD is proposed to achieve robust monocular 3D object detection under various weather conditions. It learns clear-weather knowledge as a reference using a weather codebook and performs feature enhancement by modeling foggy effects as noise through a weather-adaptive diffusion model.

## Background & Motivation

Monocular 3D object detection, which relies solely on a single RGB image to estimate 3D bounding boxes, is widely used in autonomous driving due to its low cost. However, existing methods **focus almost exclusively on detection performance under ideal weather conditions (sunny/clear)**, and their performance drops drastically when facing adverse weather such as fog.

Experimental data intuitively illustrate the severity of this issue. In the authors' reproduction experiments, the AP3D of several state-of-the-art (SOTA) methods on foggy KITTI almost dropped to zero:
- MonoGround: Clear Easy 25.24 → Foggy **0.00**
- DID-M3D: Clear Easy 22.98 → Foggy **1.15**
- MonoDETR: Clear Easy 28.84 → Foggy **7.40**

The root cause of this steep performance decline lies in the fact that the **dense scattering and light absorption** of fog severely degrade visual information. Since monocular detection relies entirely on visual features, it lacks the supplementation of depth sensors like LiDAR.

The authors identify two key problems: **(1) How to quantify the extent to which the input image needs to be improved?** Clear weather requires only minimal enhancement, while foggy weather requires substantial enhancement. **(2) How to guide the direction of feature representation enhancement?** The enhancement must be directional—guided toward the "clear weather" domain.

Although existing dehazing methods (either image-level or feature-level dehazing) bring some improvements under foggy conditions, **a one-size-fits-all dehazing approach compromises performance in clear weather** because they cannot dynamically adjust the enhancement intensity according to weather conditions.

## Method

### Overall Architecture

MonoWAD comprises three core components:
1. A backbone network (DLA-102) to extract input features $x^c$ or $x^f$.
2. A **Weather Codebook** $\mathcal{Z}$: It memorizes clear-weather knowledge to generate a weather reference feature $x^r$ for any input.
3. A **Weather-Adaptive Diffusion Model**: It dynamically enhances input features based on $x^r$.
Finally, a detection module (transformer encoder-decoder) outputs the 3D detection results.

### Key Designs

1. **Weather Codebook**:

    - A learnable codebook $\mathcal{Z} = \{z_k\}_{k=1}^K$ with $K=4096$ slots, where each $z_k \in \mathbb{R}^{1 \times c}$ ($c=256$).
    - The input features are passed through convolutional layers to obtain $\hat{x}^c$ or $\hat{x}^f$. For each spatial location, the nearest codeword is found for quantization: $x^{r(c)} = \mathbf{q}(\hat{x}^c) := \arg\min_{z_k \in \mathcal{Z}} \|\hat{x}^c_{ij} - z_k\|$.
    - **Clear Knowledge Embedding (CKE) Loss**: Uses KL divergence to make the codebook output close to the channel distribution of clear features: $\mathcal{L}_{cke} = D_{KL}(s^c \| s^{r(c)})$, where $s^c$ and $s^{r(c)}$ represent the channel probabilities after GAP + softmax, respectively.
    - **Weather-Invariant Guiding (WIG) Loss**: Ensures that foggy and clear inputs generate the same reference features: $\mathcal{L}_{wig} = \|x^{r(c)} - x^{r(f)}\|_2^2$.
    - Total CKR loss: $\mathcal{L}_{ckr} = \mathcal{L}_{cke} + \mathcal{L}_{wig}$.
    - **Design Motivation**: The codebook acts as a "weather memory bank". During training, it memorizes visual knowledge patterns of clear weather. During inference, regardless of whether the input is clear or foggy, it recalls the same "clear reference". This reference feature implicitly encodes the information of "how much enhancement is needed".

2. **Weather-Adaptive Diffusion**:

    - **Core Innovation: Replacing Gaussian noise with fog distribution**. The fog distribution is defined as $\mathcal{F} = x^f - x^c$ (the difference between foggy and clear features) and serves as the noise for the diffusion model.
    - Forward process: Foggy noise is progressively added to the clear feature $x^c_0$: $q(x^c_t | x^c_{t-1}) = \mathcal{F}(x^c_t; \sqrt{1-\beta_t}x^c_{t-1}, \beta_t \mathbf{I})$.
    - Reverse process: A conditional autoencoder $\boldsymbol{\epsilon}_\theta(x^c_t, t, x^r)$ estimates the fog variables, **conditioned on the weather reference feature $x^r$**.
    - Merging current features with reference features via a **cross-attention mechanism**:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$
      where $Q = W_i^q \cdot \bar{x}^c_t$, $K = W_i^k \cdot \bar{x}^r$, and $V = W_i^v \cdot \bar{x}^r$.
    - **Design Motivation**: Traditional diffusion uses Gaussian noise, which lacks weather semantics. Modeling the foggy effect as noise allows the forward/reverse processes of diffusion to directly correspond to the fog-adding/dehazing processes. Serving as the Key/Value in the attention mechanism, the weather reference feature automatically controls the enhancement strength: clear inputs have a small discrepancy with the reference, leading to minimal enhancement, while foggy inputs have a large discrepancy, resulting in substantial enhancement.

3. **Weather-Adaptive Enhancement (WAE) Loss**:

    - $\mathcal{L}_{wae} = \mathbb{E}_{x^c, \boldsymbol{\epsilon}_n \sim \mathcal{F}, t}[\|\boldsymbol{\epsilon}_n - \boldsymbol{\epsilon}_\theta(x^c_t, t, x^r)\|_2^2]$
    - Ensures that the diffusion model accurately estimates the fog variables.
    - **Design Motivation**: The standard DDPM MSE objective, but with the noise distribution replaced by the fog distribution.

### Loss & Training

- Total loss: $\mathcal{L}_{Total} = \mathcal{L}_{OD} + \lambda_1 \mathcal{L}_{ckr} + \lambda_2 \mathcal{L}_{wae}$
- $\lambda_1 = \lambda_2 = 1$, where $\mathcal{L}_{OD}$ is the standard 3D detection loss (classification + regression + depth).
- Training data: Original clean images from KITTI + paired images from synthesized foggy KITTI.
- Foggy images are synthesized based on DORN depth estimation + object distance.
- Diffusion steps $T=15$, trained for 120 epochs on a single RTX 4090 GPU.

## Key Experimental Results

### Main Results (KITTI Validation Set, Car Category)

| Method | Foggy AP3D Easy | Foggy Mod. | Foggy Hard | Clear AP3D Easy | Clear Mod. | Avg. Mod. |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| GUPNet | 2.74| 2.19 | 2.16 | 22.76 | 16.46 | 9.33 |
| DID-M3D | 1.15 | 0.61 | 0.64 | 22.98 | 16.12 | 8.37 |
| MonoGround | 0.00 | 0.00 | 0.06 | 25.24 | 18.69 | 9.35 |
| MonoDTR | 16.89 | 11.86 | 9.87 | 24.52 | 18.57 | 15.22 |
| MonoDETR | 7.40 | 5.74 | 4.53 | 28.84 | 20.61 | 13.18 |
| **MonoWAD** | **27.17** | **19.57** | **16.21** | **29.10** | **21.08** | **20.33** |

Virtual KITTI Multi-weather Results:

| Method | Foggy AP3D Easy | Rainy Easy | Sunset Easy |
|------|:---:|:---:|:---:|
| MonoDTR | 8.79 | 11.73 | 9.86 |
| MonoDETR | 4.50 | 6.61 | 7.08 |
| **MonoWAD** | **13.33** | **14.12** | **13.38** |

### Ablation Study

Contributions of Each Module:

| Configuration | Foggy AP3D Easy | Foggy Mod. | Clear Easy | Clear Mod. | Description |
|------|:---:|:---:|:---:|:---:|------|
| Baseline | 13.75 | 9.61 | 22.63 | 17.16 | No enhancement |
| +WAD | 25.62 | 18.66 | 26.34 | 19.17 | Diffusion model only |
| **+WC+WAD** | **27.17** | **19.57** | **29.10** | **21.08** | Full MonoWAD |

Impact of Diffusion Steps T:

| T | Foggy AP3D Easy | Foggy Mod. | Clear Easy | Clear Mod. |
|---|:---:|:---:|:---:|:---:|
| None | 13.75 | 9.61 | 22.63 | 17.16 |
| 5 | 23.57 | 17.91 | 26.03 | 19.21 |
| 10 | 25.28 | 18.49 | 26.79 | 19.90 |
| **15** | **27.17** | **19.57** | **29.10** | **21.08** |
| 20 | 24.54 | 18.29 | 24.85 | 18.54 |

Comparison with Dehazing Methods (Based on MonoDTR):

| Method | Foggy Mod. | Clear Mod. | Avg. Mod. |
|------|:---:|:---:|:---:|
| MonoDTR | 11.86 | 18.57 | 15.22 |
| +RIDCP (Image-level Dehazing) | 12.41 | 17.89 | 15.15 |
| +DENet (Feature-level Dehazing) | 17.44 | 5.70 | 11.57 |
| **MonoWAD** | **19.57** | **21.08** | **20.33** |

### Key Findings

- While MonoWAD's performance in foggy weather surges, its **clear weather performance also improves** (29.10 vs. baseline 22.63), which is unachievable with traditional dehazing methods.
- Although the dehazing method (DENet) improves foggy performance to 17.44, its clear weather performance plummets to 5.70, indicating that dehazing operations are harmful to clear images.
- The diffusion steps $T=15$ is optimal, while $T=20$ leads to degradation due to overfitting.
- The weather codebook contributes significantly: adding WC on top of WAD improves the foggy AP3D Easy from 25.62 to 27.17.
- t-SNE visualization shows that the features of MonoWAD in clear and foggy weather almost overlap, whereas other methods display clear separation.

## Highlights & Insights

- **The idea of treating foggy effects as diffusion noise is ingenious**: Traditional diffusion relies on semantic-free Gaussian noise, while the fog distribution $\mathcal{F} = x^f - x^c$ naturally carries weather degradation information, allowing the diffusion process to directly learn "dehazing".
- **Adaptive mechanism of the weather codebook**: The WIG Loss ensures that different weather inputs generate identical reference features, allowing the cross-attention of the diffusion model to naturally achieve adaptive enhancement without explicitly determining the weather type.
- **Win-win scenario for both clear and foggy weather**: This is the greatest advantage over traditional dehazing methods, as the enhancement intensity is automatically adjusted by the weather reference features.

## Limitations & Future Work

- **Slow inference speed**: At $T=15$, the speed is 144ms/image, which is significantly slower than MonoDETR's 38ms/image; the iterative nature of diffusion is the main bottleneck.
- **Requirement for paired training data**: The training stage requires paired clear-foggy images to calculate the fog distribution, limiting direct scaling to other weather types.
- **Foggy images are synthetic**: Synthetic foggy images based on depth estimation suffer from a domain gap compared to real-world foggy scenes.
- Future directions: Exploring training methods that do not require paired images, accelerating diffusion inference (e.g., using DDIM), and extending the framework to handle multiple weather types jointly.

## Related Work & Insights

- **MonoDTR** (CVPR 2022): A depth-guided transformer monocular detector, serving as the primary baseline for comparison.
- **MonoDETR** (ICCV 2023): DETR-style monocular 3D detection, showing top performance under clear weather but dropping severely under foggy conditions.
- **Foggy Cityscapes** (Sakaridis et al.): A pioneer in fog synthesis methods, whose protocol is followed by the fog synthesis in this work.
- **DDPM/Conditional Diffusion**: The diffusion model in this paper is adapted with weather-aware modifications in its noise definition and conditioning mechanism.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Modeling weather variation as diffusion noise is an insightful innovation, and the codebook design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Performs comprehensive validations including KITTI validation/test, Virtual KITTI multi-weather, comparisons with dehazing methods, evaluations on various diffusion models, step-count ablations, and t-SNE visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ — The problems are well-defined, and the motivations behind the methodology are thoroughly explained.
- **Value**: ⭐⭐⭐⭐ — Directly addresses actual pain points in autonomous driving, though its adoption is currently limited by inference speed and the requirement for paired data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] GraphBEV: Towards Robust BEV Feature Alignment for Multi-Modal 3D Object Detection](graphbev_towards_robust_bev_feature_alignment_for_multi-modal_3d_object_detectio.md)
- [\[ECCV 2024\] SimPB: A Single Model for 2D and 3D Object Detection from Multiple Cameras](simpb_a_single_model_for_2d_and_3d_object_detection_from_multiple_cameras.md)
- [\[ICCV 2025\] Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts](../../ICCV2025/autonomous_driving/adaptive_dual_uncertainty_optimization_boosting_monocular_3d_object_detection_un.md)
- [\[ECCV 2024\] Rethinking Data Augmentation for Robust LiDAR Semantic Segmentation in Adverse Weather](rethinking_data_augmentation_for_robust_lidar_semantic_segmentation_in_adverse_w.md)
- [\[ECCV 2024\] OPEN: Object-wise Position Embedding for Multi-view 3D Object Detection](open_object-wise_position_embedding_for_multi-view_3d_object_detection.md)

</div>

<!-- RELATED:END -->
