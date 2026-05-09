---
title: >-
  [Paper Note] LINR-PCGC: Lossless Implicit Neural Representations for Point Cloud Geometry Compression
description: >-
  [ICCV 2025][3D Vision][Lossless point cloud compression] LINR-PCGC proposes the first implicit neural representation (INR)-based method for lossless point cloud geometry compression. By designing a lightweight multi-scale SparseConv network with Scale Context Extraction (SCE) and Child Node Prediction (CNP) modules, combined with a GoP-level shared decoder and initialization strategy, the method achieves a 21.21% bitrate reduction over G-PCC TMC13v23 and a 21.95% reduction over SparsePCGC on the MVUB dataset, without relying on any specific training data distribution.
tags:
  - ICCV 2025
  - 3D Vision
  - Lossless point cloud compression
  - implicit neural representations
  - multi-scale sparse convolution
  - GoP coding
  - model compression
date: 2026-05-08
content_hash: 6ff258693e9c83c0
---

# LINR-PCGC: Lossless Implicit Neural Representations for Point Cloud Geometry Compression

**Conference**: ICCV 2025
**arXiv**: [2507.15686](https://arxiv.org/abs/2507.15686)
**Code**: [https://huangwenjie2023.github.io/LINR-PCGC/](https://huangwenjie2023.github.io/LINR-PCGC/)
**Area**: 3D Vision / Point Cloud Compression
**Keywords**: Lossless point cloud compression, implicit neural representations, multi-scale sparse convolution, GoP coding, model compression

## TL;DR

LINR-PCGC proposes the first implicit neural representation (INR)-based method for lossless point cloud geometry compression. By designing a lightweight multi-scale SparseConv network with Scale Context Extraction (SCE) and Child Node Prediction (CNP) modules, combined with a GoP-level shared decoder and initialization strategy, the method achieves a 21.21% bitrate reduction over G-PCC TMC13v23 and a 21.95% reduction over SparsePCGC on the MVUB dataset, without relying on any specific training data distribution.

## Background & Motivation

**Background**: Point cloud compression methods fall into two categories: traditional approaches (G-PCC, V-PCC) and AI-driven approaches (PCGCv2, SparsePCGC). Traditional methods rely on hand-crafted tools and parameters, while AI-based methods exploit neural networks to model spatial correlations and achieve state-of-the-art compression performance.

**Limitations of Prior Work**:
- AI-based methods are heavily dependent on training data distributions; distribution shift leads to significant performance degradation (e.g., SparsePCGC underperforms G-PCC on MVUB).
- INR methods address the distribution dependency issue by overfitting to target data, but face two challenges: (1) the decoder network parameters must be encoded into the bitstream, constraining network size and fitting capacity; (2) overfitting is time-consuming.
- Existing INR methods are limited to lossy compression; no INR solution for lossless compression exists.

**Key Challenge**: A fundamental tension exists between compression efficiency and distribution generalizability in AI-based methods, and between generalizability and network capacity/coding efficiency in INR methods.

**Goal**: How to achieve lossless point cloud geometry compression within an INR framework while controlling decoder size and encoding time?

**Key Insight**: The paper draws inspiration from the Group of Pictures (GoP) concept in video coding — adjacent frames share a single lightweight decoder network to amortize parameter overhead; the overfitted network from the previous GoP initializes the next GoP to accelerate convergence.

**Core Idea**: Reduce parameter overhead via GoP-level network sharing + achieve efficient lossless compression via multi-scale SparseConv child node prediction + save approximately 65% of encoding time through an initialization strategy.

## Method

### Overall Architecture

The input is a point cloud sequence $S = \{x_1, ..., x_M\}$, grouped and encoded by GoP (GoP size T=32 frames). Encoding each GoP consists of three steps:
1. **Initialization**: Initialize the current GoP's network parameters using the overfitted parameters from the previous GoP.
2. **Encoding**: Overfit the network parameters → separate into pc-encoder and pc-decoder → encode the point cloud + quantize and compress pc-decoder parameters.
3. **Decoding**: Decompress pc-decoder parameters → decode the point cloud scale by scale.

The final bitstream comprises: lowest-scale point cloud coordinates + decoder network parameters + occupancy coding information at each scale.

### Key Designs

1. **Multi-Scale SparseConv Network**:

    - **Function**: Progressively downsamples the point cloud until only tens to hundreds of points remain, then predicts occupancy probabilities from low to high scales.
    - **Mechanism**: MaxPooling is used for downsampling $x_t^{i+1} = DS(x_t^i)$; at each scale, occupancy probabilities of child nodes are predicted and compressed via arithmetic coding.
    - **Design Motivation**: The multi-scale architecture allows high-scale details to leverage structural priors from lower scales, enabling progressively refined predictions.

2. **Scale Context Extraction (SCE)**:

    - **Function**: Provides discriminative information for point clouds at different spatial scales.
    - **Mechanism**: A scale embedding (SEMB, 8-channel implicit feature expanding scale index $i$) serves as global information and is concatenated with neighborhood occupancy (occupancy states of 7 positions: front/back/left/right/up/down/self), then fused via MLP to generate the scale context feature $l_t^{i+1}$.
    - **Formula**: $l_t^{i+1} = MLP_i(Concat(Nb^{i+1}, SEMB(i)))$
    - **Design Motivation**: Since all scales share the same set of network parameters, a mechanism is needed to inform the network of the current scale; otherwise, scale-specific spatial features cannot be extracted.

3. **Child Node Prediction (CNP)**:

    - **Function**: Upsamples the point cloud from lower to higher scales, i.e., predicts octree child node occupancy.
    - **Mechanism**: The upsampling problem is formulated as octree child node occupancy prediction (8 channels corresponding to 8 child nodes). A channel-wise 8-stage prediction scheme is adopted — already-decoded child nodes serve as context for subsequent stages. Two modules are employed: GDFE (Global Deep Feature Extraction) for global features and LDFE (Local Deep Feature Extraction) for local features from decoded child nodes; their fusion drives occupancy probability prediction.
    - **vs. Transposed Convolution**: Transposed convolutions incur high memory and time complexity; CNP operates directly on the octree structure and is more efficient.
    - **Design Motivation**: Channel-wise sequential prediction resembles an autoregressive approach, where already-decoded child nodes provide additional context for those yet to be decoded, improving prediction accuracy.

4. **Adaptive Quantization (AQ) and Model Compression (MC)**:

    - **AQ**: Normalizes decoder parameters to [0,1] and quantizes to B=8 bits.
    - **MC**: Adds L2 regularization during training to encourage parameters to follow a Laplacian distribution, then encodes them via arithmetic coding using the Laplacian distribution parameters (mean $\mu$ and scale $b$).

### Loss & Training

$$\mathcal{L} = \sum_{i=0}^{N} \sum_{j=0}^{7} L_{BCE}^{i,j} + \lambda \|\boldsymbol{\theta}\|_2^2$$

- $L_{BCE}^{i,j}$ is the binary cross-entropy at scale $i$, stage $j$, estimating the bitstream size at the current stage.
- $\lambda \|\boldsymbol{\theta}\|_2^2$ is L2 regularization, concentrating the parameter distribution for easier compression.
- Adam optimizer is used with learning rate decayed from 0.01 to 0.0004.
- The first GoP is trained for 6 epochs; subsequent GoPs are trained for 1–6 epochs.
- A single RTX 3090 GPU is used.

## Key Experimental Results

### Main Results

**8iVFB Dataset (Tab. 1)**:

| Method | bpp (avg) | Relative bpp | Enc. Time (s) | Dec. Time (s) |
|--------|-----------|--------------|---------------|---------------|
| G-PCC v23 | 0.743 | 100% | 2.72 | 0.923 |
| SparsePCGC | 0.625 | 84.0% | 2.202 | 1.048 |
| V-PCC v23 | 1.415 | 190.4% | 194.261 | 2.304 |
| **Ours** | **0.616** | **82.9%** | 2.464 | **0.501** |
| **Ours 2** | **0.564** | **75.9%** | 16.423 | **0.459** |

**MVUB Dataset (Tab. 3) — Distribution Shift Scenario**:

| Method | bpp (avg) | Relative bpp | Enc. Time (s) | Dec. Time (s) |
|--------|-----------|--------------|---------------|---------------|
| G-PCC v23 | 0.921 | 100% | 3.951 | 1.284 |
| SparsePCGC | 0.930 | **100.9%** | 3.06 | 1.456 |
| V-PCC v23 | 1.543 | 167.6% | 213.192 | 3.071 |
| **Ours** | **0.806** | **87.5%** | 2.712 | **0.554** |
| **Ours 2** | **0.725** | **78.8%** | 18.564 | **0.544** |

Note that on the MVUB dataset, SparsePCGC (trained on ShapeNet) even underperforms G-PCC (100.9%), whereas LINR-PCGC maintains a strong 78.8% relative bitrate, demonstrating the distribution-agnostic nature of INR-based methods.

### Ablation Study

**Initialization Strategy Ablation (Tab. 5)**:

| Initialization | Relative Time (8iVFB) | Relative Time (Owlii) | Relative Time (MVUB) | Average |
|----------------|-----------------------|-----------------------|----------------------|---------|
| Random init. (rand.) | 100% | 100% | 100% | 100% |
| Prev. GoP init. (ini.) | 36.0% | 34.4% | 33.7% | 34.7% |
| Similar-sequence init. (fur. ini.) | 22.9% | 29.2% | 20.0% | 24.0% |

The initialization strategy saves an average of 65.3% encoding time (ini.) and 76.0% (fur. ini.).

**Module Ablation (Tab. 6)**:

| Configuration | Relative bpp ↓ |
|---------------|----------------|
| CNP only | 100.0% |
| CNP + AQ&MC | 91.9% |
| **CNP + AQ&MC + SCE (full)** | **88.8%** |

AQ&MC reduces bpp by 8.1%; SCE provides an additional 3.1% reduction.

**Bitstream Allocation and Time Breakdown (Tab. 4, MVUB)**:

| Component | Bitstream % | Enc. Time % | Dec. Time % |
|-----------|-------------|-------------|-------------|
| Decoder parameters | 0.73% | 0.47% | 0.00% |
| Lowest-scale point cloud | 0.17% | 8.58% | — |
| High scales (scale 2–6) | 5.83% | 30.47% | 31.60% |
| Mid scale (scale 1) | 18.10% | 14.92% | 16.25% |
| Highest scale (scale 0) | 75.17% | 45.56% | 51.63% |

### Key Findings

- **The key advantage of INR methods is distribution agnosticism**: On the MVUB dataset, SparsePCGC (trained on ShapeNet) underperforms G-PCC, whereas LINR-PCGC independently overfits to each sequence and is unaffected by training data distribution.
- **Decoder parameter overhead is negligible**: Accounting for only 0.73% of total bitstream (owing to GoP-level sharing), it does not become a bottleneck.
- **Trade-off between encoding time and compression ratio**: Encoding for 1 epoch (~2.5 s/frame) already matches SparsePCGC in compression ratio; encoding for 6 epochs (~16 s/frame) further reduces bitrate by 15–20%.
- **Fast decoding**: Approximately half the decoding time of G-PCC or SparsePCGC, owing to the lightweight network design.

## Highlights & Insights

- **GoP-level INR framework** — Adapting the GoP concept from video coding to INR-based point cloud compression simultaneously addresses parameter overhead and encoding speed, a design choice that is transferable to other INR compression scenarios (e.g., NeRF scene compression).
- **Child node prediction as a substitute for transposed convolution** — Formulating octree upsampling as staged child node occupancy prediction is both memory-efficient and exploits already-decoded nodes as context, representing an elegant engineering design.
- **L2 regularization → Laplacian distribution → efficient parameter coding** — A simple training technique that shapes the parameter distribution for easier compression, reflecting a deep understanding of INR parameter characteristics.

## Limitations & Future Work

- Inter-frame prediction is not exploited; frames within a GoP are compressed independently, leaving temporal redundancy unaddressed.
- Encoding time remains substantial (approximately 16 s/frame for full encoding), precluding real-time applications.
- Only geometry is addressed; the method has not been extended to attribute (color) compression.
- No comparison with the latest Unicorn-Part I is provided, though a reasonable justification is given.
- The network architecture is fixed; neural architecture search or adaptive architecture selection is not explored.

## Related Work & Insights

- **vs. G-PCC**: The traditional method performs robustly across all datasets but offers limited compression ratio; LINR-PCGC reduces bitrate by 21–28% with sufficient encoding time.
- **vs. SparsePCGC**: SparsePCGC performs well within its training distribution but degrades sharply under distribution shift (even underperforming G-PCC on MVUB); LINR-PCGC is inherently distribution-agnostic due to the INR paradigm.
- **vs. V-PCC**: V-PCC yields the highest bitrate and is extremely slow to encode (194–213 s), making it unsuitable for sparse point clouds.
- **vs. INR compression methods (Hu & Wang 2022, etc.)**: Prior INR methods address only lossy compression; LINR-PCGC extends the paradigm to the lossless setting for the first time.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First INR-based lossless point cloud compression method; the GoP framework and CNP module are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparisons on three datasets, including encoding-time vs. bitrate curves and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured, though notation-heavy in places.
- **Value**: ⭐⭐⭐⭐ Fills the gap in lossless INR-based point cloud compression; distribution-agnostic property has practical deployment value.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Neural Compression for 3D Geometry Sets](neural_compression_for_3d_geometry_sets.md)
- [\[ICCV 2025\] SL2A-INR: Single-Layer Learnable Activation for Implicit Neural Representation](sl2a-inr_single-layer_learnable_activation_for_implicit_neural_representation.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[ICCV 2025\] UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis](upp_unified_point-level_prompting_for_robust_point_cloud_analysis.md)
- [\[ICCV 2025\] TurboReg: TurboClique for Robust and Efficient Point Cloud Registration](turboreg_turboclique_for_robust_and_efficient_point_cloud_registration.md)

<!-- RELATED:END -->
