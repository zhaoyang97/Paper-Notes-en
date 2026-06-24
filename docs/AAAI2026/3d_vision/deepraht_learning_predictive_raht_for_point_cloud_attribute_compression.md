---
title: >-
  [Paper Note] DeepRAHT: Learning Predictive RAHT for Point Cloud Attribute Compression
description: >-
  [AAAI 2026][3D Vision][Point cloud compression] This paper proposes DeepRAHT, the first end-to-end differentiable Region Adaptive Hierarchical Transform (RAHT) framework for lossy point cloud attribute compression. By leveraging a learnable prediction model and a Laplace-based rate proxy, it achieves compression performance surpassing the G-PCC standard and existing deep learning methods.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Point cloud compression"
  - "attribute compression"
  - "RAHT"
  - "end-to-end learning"
  - "variable bitrate"
date: 2026-05-08
content_hash: 0dbe14ccba2dfa1f
---

# DeepRAHT: Learning Predictive RAHT for Point Cloud Attribute Compression

**Conference**: AAAI 2026  
**arXiv**: [2601.12255](https://arxiv.org/abs/2601.12255)  
**Code**: [Available](https://github.com/zb12138/DeepRAHT)  
**Area**: 3D Vision  
**Keywords**: Point cloud compression, attribute compression, RAHT, end-to-end learning, variable bitrate

## TL;DR

This paper proposes DeepRAHT, the first end-to-end differentiable Region Adaptive Hierarchical Transform (RAHT) framework for lossy point cloud attribute compression. By leveraging a learnable prediction model and a Laplace-based rate proxy, it achieves compression performance surpassing the G-PCC standard and existing deep learning methods.

## Background & Motivation

Point cloud attribute compression (PCAC) is a critical step in 3D data processing. As the core transform method of the MPEG G-PCC standard, RAHT offers excellent performance and low complexity. However, applying existing RAHT in deep learning faces several challenges:

**Non-differentiability**: RAHT in G-PCC is implemented in C++ and is non-differentiable, making end-to-end training impossible.

**Lack of Prediction**: 3DAC (the first approach to learn RAHT coefficients) only uses hand-crafted RAHT to generate transform coefficients and then learns entropy coding, ignoring the predictive RAHT which is crucial in the G-PCC standard.

**Rate-Only Optimization**: Due to non-differentiability, 3DAC can only optimize the bitrate and cannot jointly optimize distortion.

**Poor Robustness**: Existing methods are sensitive to data variance, requiring multiple models to cover different bitrate points.

**Unexplored Learnability of Predictive RAHT**: Prediction can significantly reduce the uncertainty of transform coefficients, and coding residuals is more efficient than coding coefficients.

## Method

### Overall Architecture

The core pipeline of DeepRAHT:

1. **Multi-Scale Generation**: Performs $s$ times of sum-pooling with a stride of $2 \times 2 \times 2$ on the input point cloud $P_0$ to obtain $\{P_1, ..., P_s\}$.
2. **Top-Down Coding**: Starting from the coarsest scale $s$, a transform model (Haar) and an optional prediction model are applied to each scale.
3. **Simultaneous Coding and Reconstruction**: The reconstructed $\hat{A}_m$ is used for DC reconstruction and prediction in the next layer.
4. **Decoding**: Perfectly consistent with the reconstruction process, ensuring reversibility.

### Key Designs

#### Differentiable RAHT via Sparse Convolution (Transform Model)

The core innovation is implementing differentiable dyadic RAHT using Minkowski sparse tensors and sparse convolutions:

**Haar Transform**: For each $2 \times 2 \times 2$ voxel, eight nodes are decomposed into one Direct Current (DC) coefficient and seven Alternating Current (AC) coefficients. A binary decomposition is performed sequentially along the Z $\rightarrow$ Y $\rightarrow$ X axes:

$$\begin{bmatrix} g_L \\ g_H \end{bmatrix} = \frac{1}{\sqrt{w_1+w_2}} \begin{bmatrix} \sqrt{w_1} & \sqrt{w_2} \\ -\sqrt{w_2} & \sqrt{w_1} \end{bmatrix} \begin{bmatrix} g_1 \\ g_2 \end{bmatrix}$$

where $w_1$ and $w_2$ are the numbers of raw points contained in the nodes, serving as adaptive weights.

**Sparse Convolution Implementation**:
- Z-axis decomposition: $\text{Zconv} \equiv \text{Conv}(i=1, o=2, k=s=(1,1,2))$
- Y-axis decomposition: $\text{Yconv} \equiv \text{Conv}(i=1, o=2, k=s=(1,2,1))$
- X-axis decomposition: $\text{Xconv} \equiv \text{Conv}(i=1, o=2, k=s=(2,1,1))$
- The initial convolution kernel weight is the identity matrix $I_2$.

Key Property: **DC is equivalent to the normalized attribute of the next scale**: $DC_m \equiv g_{LLL} = A_{m+1,i}/\sqrt{w_{m+1,i}}$. Therefore, the DC coefficient does not need to be coded (as it has already been coded at a higher scale), and only the seven AC coefficients need to be coded.

The inverse Haar transform is implemented using ConvolutionTranspose.

#### Prediction Model

G-PCCv14 uses Inverse Distance Weighted (IDW) prediction, but utilizing sibling nodes in the same layer introduces autoregressive dependency and increases decoding time. DeepRAHT performs prediction using only the parent scale:

**IDW Prediction** (implemented via sparse convolution):
$$\text{IDW}(\hat{a}_m) \equiv \text{Conv}(\text{Unpool}(\hat{a}_m), k=3^3, s=1^3)$$

where the convolution kernel weights are scaled by distance proportions: center:face:edge:corner = 4:3:2:1.

**Prediction Compensation Module**: Leverages the prediction error of the grandparent scale ($m+1$) to compensate for the current prediction, thereby avoiding the autoregressive issue:
$$a'_{m-1} = \text{Comp}(\hat{a}_m - \text{IDW}(\hat{a}_{m+1})) + \text{IDW}(\hat{a}_m)$$

The compensation module consists of multiple linear layers and sparse convolutions (hidden dimension of 128, kernel size of $3^3$), including a transpose convolution with a stride of 2. After prediction, the residuals of the AC coefficients are coded: $r_{m-1} = AC_{m-1} - AC'_{m-1}$.

The compensation module can be dynamically enabled or disabled based on prediction performance (signaled to the decoder with $s$ bits), ensuring that the performance lower bound is G-PCCv14.

#### Entropy Coder (Rate Proxy)

Existing methods utilize bottleneck entropy models, but they are sensitive to data variance. DeepRAHT replaces this with **zero run-length coding**, as RAHT residuals are highly concentrated around zero.

Since run-length coding is non-differentiable, a rate proxy based on the Laplace distribution is proposed:
$$q(r) = \int_{r-0.5}^{r+0.5} \mathcal{L}_{\mu,\sigma}(r)dr$$

The parameters $\alpha=0.425, \mu=0, \sigma=0.2$ are obtained by fitting actual data, achieving a coefficient of determination of 0.991.

**Variable Bitrate Advantage**: Different bitrates can be achieved by simply tuning the quantization step $qs$, avoiding the need to train multiple models ($qs = \{8,10,12,...,224\}$), whereas 3DAC and TSC-PCAC require separate models trained for each bitrate point.

### Loss & Training

Total loss function:
$$\ell = \ell_{bits} + \lambda(\ell_{recon} + \ell_{pred})$$

- $\ell_{recon} = \|a_0 - \hat{a}_0\|_2^2$: End-to-end reconstruction error
- $\ell_{pred} = \sum_m \|(a_m - a'_m)\|_2^2$: Prediction loss to accelerate convergence
- $\ell_{bits} = -\sum_m \log_2 q(r_m/qs)$: Rate proxy loss
- $\lambda = 1/255$, $qs = 8$, Adam optimizer, learning rate of 0.0001, batch size of 1
- Training data: RWTT dataset (568 real-world objects)
- Compression in YUV color space

## Key Experimental Results

### Main Results

**BD-BR Gain (%, negative value = bitrate saving, anchor = G-PCCv14):**

| Method | Owlii Avg | 8iVSLF Avg | MPEG Avg | Overall Avg |
|------|-----------|-----------|----------|--------|
| G-PCCv23 | -20.0 | -17.5 | -11.6 | -16.4 |
| 3DAC | -66.6 | -70.9 | -62.7 | -66.7 |
| TSC-PCAC | -12.8 | -68.5 | -73.2 | -51.5 |
| Unicorn | -7.1 | -10.9 | -4.0 | -7.3 |
| **DeepRAHT** | — | — | — | **Anchor** |

Note: DeepRAHT saves 16.4% bitrate on average compared to G-PCCv23, and 7.3% compared to Unicorn, with larger improvements in chroma components (U: 20.5%, V: 20.8%).

**Complexity Comparison (8iVSLF, average 3.25 million points/frame):**

| Method | Encoding Time | Decoding Time | Model Size | GPU Memory |
|------|---------|---------|---------|---------|
| 3DAC | 38.45s | 51.71s | 1MB×5 | 10GB |
| TSC-PCAC | 7.86s | 26.87s | 148MB×5 | 22GB |
| Unicorn | 20.86s | 14.99s | 65MB×3 | 16GB |
| **DeepRAHT** | **6.03s** | **5.74s** | **88MB×1** | **8GB** |

### Ablation Study

**Ablation on loot_viewdep (BD-rate gain vs G-PCCv14):**

| Configuration | BD-rate gain |
|------|-------------|
| Vanilla RAHT (no prediction) | Baseline |
| RAHT+Pred (IDW, ≈G-PCCv14) | -48.2% |
| **RAHT+Pred+Comp (DeepRAHT)** | **-24.6%** (vs G-PCCv14) |
| vs G-PCCv23 | **-16.6%** |

### Key Findings

- The prediction compensation module, without using sibling context, even outperforms the sibling prediction of G-PCCv23.
- The fitting accuracy of the rate proxy is extremely high ($R^2=0.991$), effectively replacing industrial bottleneck entropy models.
- DeepRAHT is the only deep learning method that succeeds in compressing all data (other methods fail on certain large or sparse point clouds).
- A single model covers 10 bitrate points, whereas competing methods require 3-5 models.
- Reversibility ensures that distortion only stems from quantization, visually retaining more texture details than Unicorn.

## Highlights & Insights

1. **First Implementation of End-to-End Differentiable RAHT**: Rebuilds the core algorithm of the G-PCC standard entirely using sparse convolutions, bridging the gap between deep learning and traditional standards.
2. **Guaranteed Performance Lower Bound**: Since the framework structure is perfectly aligned with G-PCCv14, the optional compensation module and signaling bits ensure that the performance is never inferior to G-PCCv14.
3. **Elegant Solution for Variable Bitrate**: By utilizing the robustness of run-length coding to the Laplace distribution, a single model covers a wide bitrate range by simply adjusting the quantization step.
4. The equivalence of **DC = normalized attribute of the next scale** serves as the key theoretical foundation to avoid redundant coding.
5. **Strong Practicality**: Offers the fastest encoding/decoding speeds, lowest GPU memory footprint, and highest robustness.

## Limitations & Future Work

1. Training is solely conducted on the RWTT dataset; generalization to LiDAR and dynamic point clouds remains to be verified.
2. Batch size is limited to 1, introducing a bottleneck in large-scale training efficiency.
3. The prediction model only utilizes parent/grandparent scales without exploring longer-range contexts.
4. It only processes color attributes; the applicability to other attributes like normals and reflectance has not been validated.
5. Integration with Gaussian Splatting data (a potential application mentioned by the authors) has not yet been experimented.

## Related Work & Insights

- **G-PCC (tmc13v23)**: The industrial standard. DeepRAHT aligns with and surpasses its structure, demonstrating the potential of learning-based methods to replace hand-crafted designs.
- **3DAC**: The first method to learn RAHT coefficients, but is non end-to-end and lacks prediction—DeepRAHT is a complete solution directly addressing these two limitations.
- **Unicorn**: The current SOTA deep learning framework, which utilizes average pooling to obtain multiple scales. The RAHT decomposition in DeepRAHT provides a more theoretically justified multi-scale scheme.
- Insight: The deep integration of traditional signal processing tools (such as Haar wavelet transform) with deep learning is an important direction in the field of compression.

## Rating

- Novelty: ⭐⭐⭐⭐ (The designs of end-to-end differentiable RAHT and the rate proxy are novel, though the overall framework follows the G-PCC structure)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Comprehensive evaluation on three datasets + complexity comparison + variable bitrate + robustness verification + ablation studies)
- Writing Quality: ⭐⭐⭐⭐ (Precise technical descriptions and complete mathematical derivations)
- Value: ⭐⭐⭐⭐⭐ (Directly benchmarks against the industrial G-PCC standard, carrying high practical value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Graph Smoothing for Enhanced Local Geometry Learning in Point Cloud Analysis](graph_smoothing_for_enhanced_local_geometry_learning_in_point_cloud_analysis.md)
- [\[AAAI 2026\] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion](dapointmamba_domain_adaptive_point_mamba_for_point_cloud_completion.md)
- [\[CVPR 2026\] Deformation-based In-Context Learning for Point Cloud Understanding](../../CVPR2026/3d_vision/deformation-based_in-context_learning_for_point_cloud_understanding.md)
- [\[ICLR 2026\] RayI2P: Learning Rays for Image-to-Point Cloud Registration](../../ICLR2026/3d_vision/rayi2p_learning_rays_for_image-to-point_cloud_registration.md)
- [\[CVPR 2026\] Adapting Point Cloud Analysis via Multimodal Bayesian Distribution Learning](../../CVPR2026/3d_vision/adapting_point_cloud_analysis_via_multimodal_bayesian_distribution_learning.md)

</div>

<!-- RELATED:END -->
