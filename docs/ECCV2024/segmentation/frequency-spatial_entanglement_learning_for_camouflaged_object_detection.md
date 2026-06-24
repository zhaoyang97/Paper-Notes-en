---
title: >-
  [Paper Note] Frequency-Spatial Entanglement Learning for Camouflaged Object Detection
description: >-
  [ECCV 2024][Segmentation][Camouflaged Object Detection] The Frequency-Spatial Entanglement Learning (FSEL) framework is proposed. By conducting entanglement learning between the frequency and spatial domains, it utilizes global frequency features to compensate for the locality and sensitivity limitations of spatial features, outperforming 21 state-of-the-art (SOTA) methods across three COD benchmarks.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Camouflaged Object Detection"
  - "Frequency Domain Learning"
  - "Spatial-Frequency Entanglement"
  - "Transformer"
  - "Fourier Transform"
date: 2026-05-08
content_hash: 00ccbdc3bef5181e
---

# Frequency-Spatial Entanglement Learning for Camouflaged Object Detection

**Conference**: ECCV 2024  
**arXiv**: [2409.01686](https://arxiv.org/abs/2409.01686)  
**Code**: [Available](https://github.com/CSYSI/FSEL)  
**Area**: Image Segmentation  
**Keywords**: Camouflaged Object Detection, Frequency Domain Learning, Spatial-Frequency Entanglement, Transformer, Fourier Transform

## TL;DR

The Frequency-Spatial Entanglement Learning (FSEL) framework is proposed. By conducting entanglement learning between the frequency and spatial domains, it utilizes global frequency features to compensate for the locality and sensitivity limitations of spatial features, outperforming 21 state-of-the-art (SOTA) methods across three COD benchmarks.

## Background & Motivation

The core challenge of Camouflaged Object Detection (COD) lies in the high similarity between targets and backgrounds in the spatial domain, making identification extremely difficult. Major limitations of existing methods:

**Inherent flaws of spatial domain features**: Existing COD methods mainly rely on single spatial features. Based on pixel-level information, these features focus on local intensity and spatial positions, possessing **locality** and **sensitivity**—pixels are only correlated with their neighbors, making it hard to distinguish subtle differences between camouflaged targets and backgrounds. When facing complex backgrounds, spatial features are highly susceptible to interference.

**Limitations of existing frequency-based methods**: Some methods have begun to introduce frequency clues, which fall into two main categories of limitations:
   - **Category 1** (e.g., FDNet, EVP): Applying frequency transforms directly to input images to extract features. However, **camouflaged images contain abundant background noise**, rendering the extracted frequency features unreliable and introducing unnecessary background noise during spatial feature aggregation.
   - **Category 2** (e.g., FPNet, FEDER): Operating on the initial features of encoders but focusing only on high- and low-frequency information, thereby **ignoring the rich information contained in the middle-frequency band** and missing critical frequency-domain information.

**Global advantages of frequency features**: Frequency features generated via Fourier transforms possess global properties that capture the frequency distribution of the entire image, thereby assisting in breaking through the locality bottleneck of spatial features.

Based on the above analysis, the authors propose a core idea: instead of simply concatenating frequency and spatial features, features from these two domains should undergo **entanglement learning** (reminiscent of the quantum entanglement metaphor). This allows global frequency features and local spatial features to mutually learn and optimize, forming a more powerful integrated representation.

## Method

### Overall Architecture

The FSEL model consists of three core components:
1. **Joint Domain Perception Module (JDPM)**: Captures high-level semantic features to guide localization.
2. **Entanglement Transformer Block (ETB)**: Performs entanglement learning in the frequency and spatial domains to generate discriminative features.
3. **Dual-domain Reverse Parser (DRP)**: Aggregates multi-level feature flows across both domains.

The input image is encoded by a backbone (PVTv2/ResNet50/Res2Net) into four levels of initial features $\{\mathcal{P}_i\}_{i=1}^4$. It passes through JDPM to generate a coarse localization map $\mathcal{P}_5$, and through ETB to generate discriminative features $\{\mathcal{X}_i\}_{i=1}^4$, which are finally processed by DRP to output prediction maps $\{\mathcal{N}_i\}_{i=1}^4$.

### Key Designs

#### 1. Joint Domain Perception Module (JDPM)

JDPM utilizes a hierarchical structure to extract multi-receptive-field information across spatial and frequency domains. Taking the highest-level feature $\mathcal{P}_4$ as input:

- First, dimensionally reduced to 128 channels via a 1×1 convolution.
- A set of 3×3 dilated convolutions with different dilation rates ($z = 2n+1$) is used to obtain local multi-scale spatial features $\{\mathcal{J}_n^s\}_{n=1}^4$.
- Spatial features undergo FFT → weight filtering → IFFT → absolute value operation to obtain global frequency features $\{\mathcal{J}_n^f\}_{n=1}^4$:
$$\mathcal{J}_n^f = \Phi\|ifft(\sigma(fft(\mathcal{J}_n^s)) * fft(\mathcal{J}_n^s))\|$$
- Spatial and frequency features are added element-wise: $\mathcal{J}_n = \mathcal{J}_n^s + \mathcal{J}_n^f$
- All multi-scale features are concatenated, and a residual connection is introduced to generate a 1-channel coarse localization map $\mathcal{P}_5$.

**Design Motivation**: Convolutional receptive fields in the spatial domain are limited, while introducing frequency transforms through FFT achieves global perception. Simultaneously, multi-scale dilated convolutions cover diverse contextual ranges.

#### 2. Entanglement Transformer Block (ETB)

As the core of this paper, ETB comprises three sub-components to achieve frequency-spatial entanglement:

**Frequency Self-Attention (FSA)**: FFT is performed on the input feature to obtain Q/K/V in the frequency domain, constructing a frequency attention map. Since the frequency attention map is complex-valued, it cannot be directly activated using Softmax. Therefore, it is decomposed into real and imaginary parts to be activated separately before merging:

$$a\Lambda_f = \Theta(Sof(\Lambda_f^{re}), Sof(\Lambda_f^{im}))$$

Frequency attention features $\mathcal{X}_f^1$ are obtained via IFFT and absolute value extraction. **Design Motivation**: Modeling the dependence and importance weights among different frequency bands, rather than narrowing focus to only high/low frequencies.

**Spatial Self-Attention (SSA)**: Employs depthwise separable convolutions (3×3 and 5×5) to embed multi-scale spatial context, generating Q/K/V for standard self-attention operations.

**Entanglement Feed-Forward Network (EFFN)**: Two-stage entanglement learning:
- **Stage 1**: Maps the fused features separately into the frequency domain (FFT → weight filtering → GELU gating) and the spatial domain (depthwise separable convolution → GELU gating) to obtain $\hat{\mathcal{X}}_f^2$ and $\hat{\mathcal{X}}_s^2$.
- **Stage 2**: Entangles and interacts features of the two domains again—frequency and spatial features are concatenated with each other and then optimized within their respective domains, followed by final aggregation and a residual connection.

$$\hat{\mathcal{X}}_f^3 = \Phi\|ifft(\sigma(fft(Cat(\hat{\mathcal{X}}_f^2, \hat{\mathcal{X}}_s^2))) * fft(Cat(\hat{\mathcal{X}}_f^2, \hat{\mathcal{X}}_s^2)))\|$$
$$\hat{\mathcal{X}}_s^3 = \mathcal{DC}_3 Cat(\hat{\mathcal{X}}_f^2, \hat{\mathcal{X}}_s^2)$$

**Design Motivation**: The frequency domain focuses on the global energy distribution and signal changes, whereas the spatial domain acts on local pixel-level details; the two are complementary. Entanglement learning enables features in different states to adapt to each other, forming a more robust representation.

#### 3. Dual-domain Reverse Parser (DRP)

DRP is designed as a dual-branch structure that optimizes and aggregates multi-level features across both frequency and spatial domains:

- **Branch 1**: Expands channels of the auxiliary feature $\mathcal{P}_5$ and concatenates it with the ETB output $\mathcal{X}_4$. It optimizes this fused feature in the frequency domain (FFT → filtering → IFFT) and spatial domain (convolutional sequence) separately, adding them to obtain $\mathcal{N}_4^1$.
- **Branch 2**: Generates a hybrid reverse attention map $\mathcal{A}_r$ (containing inverted information of both spatial and frequency domains) and uses it to weight features, thereby obtaining reverse features $\mathcal{N}_4^2$ focused on hard-to-distinguish regions.
- The two branches are concatenated and fused to produce the final output, optimizing lower-level features level-by-level in a densely connected manner.

### Loss & Training

Multi-level supervision is performed using weighted BCE and weighted IoU:

$$\mathcal{L}_{all} = \sum_{i=1}^5 \frac{1}{2^{i-1}} (\mathcal{L}_{bce}^w(\mathcal{N}_i, G) + \mathcal{L}_{iou}^w(\mathcal{N}_i, G))$$

- The 5-level outputs are supervised with exponentially decaying weights of $1, 1/2, 1/4, 1/8, 1/16$.
- Training settings: Adam optimizer, initial lr = 1e-4, decayed by a factor of 10 every 60 epochs, input size of 416×416, batch size of 40, for a total of 180 epochs.
- Trained on 4 NVIDIA GTX 4090 GPUs.

## Key Experimental Results

### Main Results: Three COD Benchmarks (PVTv2 Backbone, Ours-Pvt)

| Method | CAMO $\mathcal{M}$↓ | CAMO $F_\varphi^m$↑ | CAMO $S_m$↑ | COD10K $\mathcal{M}$↓ | COD10K $F_\varphi^m$↑ | COD10K $S_m$↑ | NC4K $\mathcal{M}$↓ | NC4K $S_m$↑ |
|------|-----|-----|-----|------|------|-----|------|------|
| SINet (CVPR'20) | .100 | .762 | .751 | .051 | .708 | .770 | .058 | .807 |
| FEDER (CVPR'23) | .071 | .824 | .802 | .032 | .788 | .820 | .044 | .846 |
| FSPNet (CVPR'23) | .050 | .869 | .855 | .026 | .816 | .847 | .035 | .878 |
| HiNet (AAAI'23) | .055 | .857 | .849 | .023 | .850 | .868 | .037 | .874 |
| FPNet (MM'23) | .056 | .863 | .851 | .029 | .817 | .847 | — | — |
| **Ours-Pvt** | **.040** | **.891** | **.885** | **.021** | **.853** | **.873** | **.030** | **.892** |

Compared with Prev. SOTA: The $\mathcal{M}$ metric on CAMO decreases from 0.050 to 0.040 (20% Gain), and the $\mathcal{M}$ metric on COD10K decreases from 0.023 to 0.021, achieving comprehensive SOTA performance.

### Ablation Study: Contribution of Each Module (ResNet50 Backbone)

| Config | Baseline | ETB | DRP | JDPM | CAMO $\mathcal{M}$↓ | CAMO $S_m$↑ | COD10K $\mathcal{M}$↓ | COD10K $S_m$↑ |
|------|---------|-----|-----|------|-----|-----|------|------|
| (a) | ✓ | | | | .093 | .767 | .046 | .778 |
| (b) | ✓ | ✓ | | | .076 | .801 | .034 | .821 |
| (c) | ✓ | | ✓ | | .074 | .810 | .034 | .826 |
| (d) | ✓ | | | ✓ | .081 | .787 | .039 | .804 |
| (h) | ✓ | ✓ | ✓ | ✓ | **.067** | **.821** | **.031** | **.830** |

Ablation study regarding frequency vs. spatial components within ETB (ETB-S with spatial only, ETB-F with frequency only, and full ETB) indicates that entanglement learning of both domains is complementary and indispensable.

### Efficiency Analysis

| Method | Params (M) | FLOPs (G) |
|------|-----------|----------|
| SINet | 48.95 | 38.75 |
| FEDER | 37.37 | 23.98 |
| FSPNet | 273.79 | 283.31 |
| **Ours-R50** | **29.15** | **35.64** |
| Ours-Pvt | 67.13 | 54.73 |

### Key Findings

1. FSEL comprehensively outperforms 21 SOTA methods on three datasets, showing significant improvements in the $\mathcal{M}$ metric, particularly on the CAMO dataset.
2. Both frequency and spatial features are indispensable—using frequency only (ETB-F) or spatial only (ETB-S) yields inferior performance compared to the complete ETB featuring entanglement learning.
3. The R50 version exhibits a relatively low parameter count and FLOPs (29.15M / 35.64G), yet its performance substantially surpasses other methods of comparable size.
4. The model generalizes effectively to other tasks, including salient object detection and polyp segmentation.

## Highlights & Insights

- **Real/Imaginary decomposition for frequency-domain self-attention**: Since frequency attention maps are complex-valued, separating the real and imaginary parts for independent Softmax activation before merging is a commendable processing strategy worth adopting.
- **Metaphor of entanglement learning**: Borrowing the concept of quantum entanglement to describe the bidirectional interaction between frequency and spatial features, enabling features of both "states" to form stronger representations through information exchange.
- **All-frequency coverage instead of just high/low frequencies**: Unlike methods like FPNet or FEDER that only focus on high and low frequencies, this work models the relationships and importance of all frequencies through self-attention across frequency bands.

## Limitations & Future Work

- The computational overhead of FFT/IFFT operations increases with image resolution, and efficiency under high-resolution scenarios remains to be optimized.
- The number of layers and interaction mechanisms for entanglement learning are relatively fixed (two stages); adaptive interaction strategies could be explored.
- The approach is primarily validated on the COD task; despite demonstrating generalizability on SOD and polyp segmentation, its performance on broader segmentation tasks (e.g., instance segmentation) remains unexplored.
- Systematic comparisons against different frequency transform methods (e.g., DCT vs. FFT vs. wavelets) are currently lacking.

## Related Work & Insights

- The role of frequency domain features in COD has been underestimated; this work demonstrates the significance of all-frequency interactions.
- The concept of entanglement learning can be extended to other tasks requiring multi-domain feature fusion (e.g., medical imaging, remote sensing).
- The technique of handling complex numbers in frequency self-attention (real/imaginary decomposition followed by separate activation) can serve as a plug-and-play general module.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The frequency-spatial entanglement learning framework design is novel, with technical innovation in handling complex numbers for frequency self-attention.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive comparison against 21 methods, covering 3 datasets and 3 backbones, along with detailed ablation studies and extended applications.
- **Writing Quality**: ⭐⭐⭐⭐ — The paper exhibits a clear structure, complete equations, and rich figures and tables.
- **Value**: ⭐⭐⭐⭐ — Content achieves comprehensive SOTA in COD, and the ETB module can be integrated into other methods as a plug-and-play component.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Learning Camouflaged Object Detection from Noisy Pseudo Label](learning_camouflaged_object_detection_from_noisy_pseudo_label.md)
- [\[CVPR 2026\] Beyond Appearance: Camouflaged Object Detection via Geometric Structure](../../CVPR2026/segmentation/beyond_appearance_camouflaged_object_detection_via_geometric_structure.md)
- [\[CVPR 2026\] SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection](../../CVPR2026/segmentation/sddf_specificity-driven_dynamic_focusing_for_open-vocabulary_camouflaged_object.md)
- [\[ICCV 2025\] Beyond Single Images: Retrieval Self-Augmented Unsupervised Camouflaged Object Detection](../../ICCV2025/segmentation/beyond_single_images_retrieval_self-augmented_unsupervised_camouflaged_object_de.md)
- [\[ECCV 2024\] Self-supervised Co-salient Object Detection via Feature Correspondences at Multiple Scales](self-supervised_co-salient_object_detection_via_feature_correspondences_at_multi.md)

</div>

<!-- RELATED:END -->
