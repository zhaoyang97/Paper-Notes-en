---
title: >-
  [Paper Note] DeepRAHT: Learning Predictive RAHT for Point Cloud Attribute Compression
description: >-
  [AAAI 2026][3D Vision][Point Cloud Compression] This paper proposes DeepRAHT, the first end-to-end differentiable Region Adaptive Hierarchical Transform (RAHT) framework for lossy point cloud attribute compression. By integrating learnable prediction models with a Laplace distribution-based rate proxy, DeepRAHT achieves compression performance surpassing both the G-PCC standard and existing deep learning methods.
tags:
  - AAAI 2026
  - 3D Vision
  - Point Cloud Compression
  - Attribute Compression
  - RAHT
  - End-to-End Learning
  - Variable Bitrate
date: 2026-05-08
content_hash: 53535f15ad4fc98e
---

# DeepRAHT: Learning Predictive RAHT for Point Cloud Attribute Compression

**Conference**: AAAI 2026
**arXiv**: [2601.12255](https://arxiv.org/abs/2601.12255)
**Code**: [Available](https://github.com/zb12138/DeepRAHT)
**Area**: 3D Vision
**Keywords**: Point Cloud Compression, Attribute Compression, RAHT, End-to-End Learning, Variable Bitrate

## TL;DR

This paper proposes DeepRAHT, the first end-to-end differentiable Region Adaptive Hierarchical Transform (RAHT) framework for lossy point cloud attribute compression. By integrating learnable prediction models with a Laplace distribution-based rate proxy, DeepRAHT achieves compression performance surpassing both the G-PCC standard and existing deep learning methods.

## Background & Motivation

Point cloud attribute compression (PCAC) is a critical component of 3D data processing. RAHT, as the core transform in the MPEG G-PCC standard, offers strong performance at low complexity. However, applying RAHT in deep learning settings faces several challenges:

**Non-differentiability**: The RAHT implementation in G-PCC is written in C++ and is non-differentiable, precluding end-to-end training.

**Absence of prediction**: 3DAC, the first method to learn RAHT coefficients, relies on handcrafted RAHT to generate transform coefficients and then learns entropy coding, neglecting the predictive RAHT that is integral to the G-PCC standard.

**Rate-only optimization**: Due to non-differentiability, 3DAC can only optimize the bitrate and cannot jointly optimize distortion.

**Poor robustness**: Existing methods are sensitive to data variance and require multiple models to cover different rate-distortion operating points.

**Unexplored learnability of predictive RAHT**: Prediction can substantially reduce the uncertainty of transform coefficients, and encoding residuals is more efficient than encoding raw coefficients.

## Method

### Overall Architecture

The core pipeline of DeepRAHT proceeds as follows:

1. **Multi-scale generation**: The input point cloud $P_0$ undergoes $s$ rounds of $2 \times 2 \times 2$ sum-pooling to produce $\{P_1, ..., P_s\}$.
2. **Top-down encoding**: Starting from the coarsest scale $s$, each scale applies a transform model (Haar) and an optional prediction model.
3. **Encode-and-reconstruct**: The reconstructed $\hat{A}_m$ is used for DC reconstruction and prediction at the next finer level.
4. **Decoding**: The decoding process is fully consistent with reconstruction, ensuring invertibility.

### Key Designs

#### Differentiable RAHT via Sparse Convolution (Transform Model)

The core innovation is implementing a differentiable dyadic RAHT using Minkowski sparse tensors and sparse convolutions:

**Haar Transform**: For each $2 \times 2 \times 2$ voxel, the 8 nodes are decomposed into 1 DC coefficient and 7 AC coefficients via sequential binary decomposition along the Z→Y→X axes:

$$\begin{bmatrix} g_L \\ g_H \end{bmatrix} = \frac{1}{\sqrt{w_1+w_2}} \begin{bmatrix} \sqrt{w_1} & \sqrt{w_2} \\ -\sqrt{w_2} & \sqrt{w_1} \end{bmatrix} \begin{bmatrix} g_1 \\ g_2 \end{bmatrix}$$

where $w_1, w_2$ denote the number of original points contained in each node, serving as adaptive weights.

**Sparse convolution implementation**:
- Z-axis decomposition: $\text{Zconv} \equiv \text{Conv}(i=1, o=2, k=s=(1,1,2))$
- Y-axis decomposition: $\text{Yconv} \equiv \text{Conv}(i=1, o=2, k=s=(1,2,1))$
- X-axis decomposition: $\text{Xconv} \equiv \text{Conv}(i=1, o=2, k=s=(2,1,1))$
- Initial convolution kernel weights are set to the identity matrix $I_2$

Key property: **DC is equivalent to the normalized attribute at the next coarser scale**: $DC_m \equiv g_{LLL} = A_{m+1,i}/\sqrt{w_{m+1,i}}$. Therefore, the DC need not be encoded (it is already encoded at the coarser scale); only the 7 AC coefficients require encoding.

The inverse Haar transform is implemented using ConvolutionTranspose.

#### Prediction Model

G-PCCv14 employs inverse distance weighting (IDW) prediction, but using sibling nodes at the same scale introduces autoregressive dependencies and increases decoding time. DeepRAHT performs prediction using only the parent scale:

**IDW prediction** (implemented via sparse convolution):
$$\text{IDW}(\hat{a}_m) \equiv \text{Conv}(\text{Unpool}(\hat{a}_m), k=3^3, s=1^3)$$

Convolution kernel weights are assigned proportionally by distance: center:face:edge:corner = 4:3:2:1.

**Prediction compensation module**: Leverages the prediction error at the grandparent scale ($m+1$) to compensate the current prediction, avoiding autoregressive dependencies:
$$a'_{m-1} = \text{Comp}(\hat{a}_m - \text{IDW}(\hat{a}_{m+1})) + \text{IDW}(\hat{a}_m)$$

The compensation module consists of multiple linear layers and sparse convolutions (hidden dimension 128, kernel size $3^3$), including a transposed convolution with stride 2. After prediction, the AC residuals are encoded: $r_{m-1} = AC_{m-1} - AC'_{m-1}$.

The compensation module can be selectively enabled based on prediction performance (signaled to the decoder with $s$ bits), guaranteeing a performance lower bound of G-PCCv14.

#### Entropy Coder (Rate Proxy)

Existing methods use bottleneck entropy models, which are sensitive to data variance. DeepRAHT instead employs **zero run-length coding**, exploiting the high concentration of RAHT residuals near zero.

Since run-length coding is non-differentiable, a Laplace distribution-based rate proxy is proposed:
$$q(r) = \int_{r-0.5}^{r+0.5} \mathcal{L}_{\mu,\sigma}(r)dr$$

Parameters $\alpha=0.425, \mu=0, \sigma=0.2$ are obtained by fitting to real data, achieving a coefficient of determination of 0.991.

**Variable bitrate advantage**: Different bitrates are achieved simply by adjusting the quantization step $qs$ ($qs = \{8,10,12,...,224\}$), requiring only a single trained model, whereas 3DAC and TSC-PCAC require separate training for each rate-distortion operating point.

### Loss & Training

The total loss function is:
$$\ell = \ell_{bits} + \lambda(\ell_{recon} + \ell_{pred})$$

- $\ell_{recon} = \|a_0 - \hat{a}_0\|_2^2$: end-to-end reconstruction error
- $\ell_{pred} = \sum_m \|(a_m - a'_m)\|_2^2$: prediction loss for accelerating convergence
- $\ell_{bits} = -\sum_m \log_2 q(r_m/qs)$: rate proxy loss
- $\lambda = 1/255$, $qs = 8$, Adam optimizer, learning rate 0.0001, batch size 1
- Training data: RWTT dataset (568 real-world objects)
- Compression performed in YUV color space

## Key Experimental Results

### Main Results

**BD-BR Gain (%, negative = bitrate savings, anchor = G-PCCv14):**

| Method | Owlii Avg | 8iVSLF Avg | MPEG Avg | Overall Avg |
|--------|-----------|------------|----------|-------------|
| G-PCCv23 | -20.0 | -17.5 | -11.6 | -16.4 |
| 3DAC | -66.6 | -70.9 | -62.7 | -66.7 |
| TSC-PCAC | -12.8 | -68.5 | -73.2 | -51.5 |
| Unicorn | -7.1 | -10.9 | -4.0 | -7.3 |
| **DeepRAHT** | — | — | — | **Baseline** |

Note: DeepRAHT saves an average of 16.4% bitrate over G-PCCv23 and 7.3% over Unicorn; improvements are larger on chroma components (U: 20.5%, V: 20.8%).

**Complexity Comparison (8iVSLF, avg. 3.25M points/frame):**

| Method | Enc. Time | Dec. Time | Model Size | GPU Memory |
|--------|-----------|-----------|------------|------------|
| 3DAC | 38.45s | 51.71s | 1MB×5 | 10GB |
| TSC-PCAC | 7.86s | 26.87s | 148MB×5 | 22GB |
| Unicorn | 20.86s | 14.99s | 65MB×3 | 16GB |
| **DeepRAHT** | **6.03s** | **5.74s** | **88MB×1** | **8GB** |

### Ablation Study

**Ablation on loot_viewdep (BD-rate gain vs. G-PCCv14):**

| Configuration | BD-rate Gain |
|---------------|-------------|
| Vanilla RAHT (no prediction) | Baseline |
| RAHT+Pred (IDW, ≈G-PCCv14) | -48.2% |
| **RAHT+Pred+Comp (DeepRAHT)** | **-24.6%** (vs. G-PCCv14) |
| vs. G-PCCv23 | **-16.6%** |

### Key Findings

- The prediction compensation module exceeds the sibling-based prediction of G-PCCv23 without using any sibling context.
- The rate proxy achieves very high fitting accuracy ($R^2=0.991$), effectively replacing the bottleneck entropy model.
- DeepRAHT is the only deep learning method that successfully compresses all test sequences; competing methods fail on certain large or sparse point clouds.
- A single model covers 10 rate-distortion operating points, whereas competing methods require 3–5 separate models.
- Guaranteed invertibility confines distortion to quantization alone, preserving more texture detail than Unicorn.

## Highlights & Insights

1. **First end-to-end differentiable RAHT**: The core algorithm of the G-PCC standard is fully reimplemented using sparse convolutions, bridging deep learning and traditional coding standards.
2. **Guaranteed performance lower bound**: The framework is structurally aligned with G-PCCv14; the optional compensation module and signaling bits ensure that performance never falls below G-PCCv14.
3. **Elegant variable bitrate solution**: Exploiting the robustness of run-length coding to Laplace distributions, a single model covers a wide bitrate range by adjusting the quantization step.
4. The equivalence **DC = normalized attribute at the next coarser scale** is the key theoretical foundation for avoiding redundant coding.
5. **Highly practical**: Fastest encoding and decoding, lowest GPU memory usage, and best robustness among compared methods.

## Limitations & Future Work

1. Training is conducted solely on the RWTT dataset; generalization to LiDAR and dynamic point clouds remains to be validated.
2. Batch size is limited to 1, creating a bottleneck for large-scale training efficiency.
3. The prediction model uses only parent/grandparent scales; longer-range context is unexplored.
4. Only color attributes are handled; applicability to other attributes such as normals and reflectance is unverified.
5. Integration with Gaussian Splatting data (a potential application mentioned by the authors) has not been experimentally evaluated.

## Related Work & Insights

- **G-PCC (tmc13v23)**: The industry standard; DeepRAHT aligns with its structure and surpasses it, demonstrating the potential of learned methods to replace handcrafted designs.
- **3DAC**: The first method to learn RAHT coefficients, but neither end-to-end nor predictive — DeepRAHT directly addresses both shortcomings.
- **Unicorn**: Current state-of-the-art deep learning framework using average pooling for multi-scale representation. DeepRAHT's RAHT decomposition provides a more theoretically grounded multi-scale alternative.
- Insight: Deep integration of classical signal processing tools (e.g., Haar wavelet transforms) with deep learning is a promising direction in compression research.

## Rating

- Novelty: ⭐⭐⭐⭐ (The end-to-end differentiable RAHT and rate proxy design are novel, though the overall framework adheres to the G-PCC structure.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Comprehensive evaluation on three datasets with complexity comparisons, variable bitrate analysis, robustness validation, and ablation studies.)
- Writing Quality: ⭐⭐⭐⭐ (Technical descriptions are precise and mathematical derivations are complete.)
- Value: ⭐⭐⭐⭐⭐ (Directly benchmarked against the G-PCC industry standard; high practical value.)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Graph Smoothing for Enhanced Local Geometry Learning in Point Cloud Analysis](graph_smoothing_for_enhanced_local_geometry_learning_in_point_cloud_analysis.md)
- [\[AAAI 2026\] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion](dapointmamba_domain_adaptive_point_mamba_for_point_cloud_completion.md)
- [\[CVPR 2026\] Deformation-based In-Context Learning for Point Cloud Understanding](../../CVPR2026/3d_vision/deformation-based_in-context_learning_for_point_cloud_understanding.md)
- [\[AAAI 2026\] Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective](rethinking_multimodal_point_cloud_completion_a_completion-by-correction_perspect.md)
- [\[AAAI 2026\] Point Cloud Quantization through Multimodal Prompting for 3D Understanding](point_cloud_quantization_through_multimodal_prompting_for_3d_understanding.md)

<!-- RELATED:END -->
