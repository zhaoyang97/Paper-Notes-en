---
title: >-
  [Paper Note] JPEG Processing Neural Operator for Backward-Compatible Coding
description: >-
  [ICCV 2025][Scientific Computing][JPEG] This paper proposes JPNeO, a next-generation codec that is fully backward-compatible with the JPEG format. By introducing neural operators at both the encoding stage (JENO) and decoding stage (JDNO), along with a trainable quantization matrix, JPNeO significantly improves JPEG reconstruction quality—particularly for chroma components—while maintaining low memory footprint and parameter count.
tags:
  - ICCV 2025
  - Scientific Computing
  - JPEG
  - Neural Operator
  - Backward Compatibility
  - Image Compression
  - Chroma Preservation
date: 2026-05-08
content_hash: c6c647fc70327c8e
---

# JPEG Processing Neural Operator for Backward-Compatible Coding

**Conference**: ICCV 2025
**arXiv**: [2507.23521](https://arxiv.org/abs/2507.23521)
**Code**: [github.com/WooKyoungHan/JPNeO](https://github.com/WooKyoungHan/JPNeO)
**Area**: Scientific Computing
**Keywords**: JPEG, Neural Operator, Backward Compatibility, Image Compression, Chroma Preservation

## TL;DR

This paper proposes JPNeO, a next-generation codec that is fully backward-compatible with the JPEG format. By introducing neural operators at both the encoding stage (JENO) and decoding stage (JDNO), along with a trainable quantization matrix, JPNeO significantly improves JPEG reconstruction quality—particularly for chroma components—while maintaining low memory footprint and parameter count.

## Background & Motivation

### State of the Field

Although DNN-based nonlinear transform coding achieves strong performance in lossy compression, the JPEG standard is deeply embedded in image signal processors (ISPs) and constitutes an unavoidable component of existing image processing pipelines. Standardization of DNN-based compression methods remains a long-term prospect, and a large volume of existing images has already been compressed with legacy codecs.

### Limitations of Prior Work

**Irreversible information loss at the encoder**: JPEG's two primary sources of loss—quantization and chroma subsampling—discard information at the encoding stage, constraining traditional decoders to the mutual information bottleneck imposed by the encoder.

**Poor chroma component recovery**: Existing JPEG artifact removal methods focus primarily on the luma component; chroma components (CbCr) degrade severely due to subsampling.

**Independent optimization of encoder and decoder**: Prior work optimizes either the encoder or the decoder in isolation, without shared information for joint synergy.

**Compatibility issues**: DNN codecs are not interoperable with existing JPEG infrastructure.

### Root Cause

How can neural networks enhance both encoding and decoding quality without altering the JPEG source coding protocol (file format)?

### Starting Point

The paper adopts an information-theoretic perspective, analyzing information loss throughout the JPEG encode–decode pipeline. It embeds image priors at the encoder to increase mutual information $I(\mathbf{X}';\varphi)$, and acquires additional mutual information $I(\tilde{\mathbf{X}};\hat{\theta})$ at the decoder via learned parameters, while maintaining full backward compatibility.

## Method

### Overall Architecture

JPNeO consists of three components:
- **JENO (JPEG Encoding Neural Operator)**: An auxiliary encoder addressing chroma subsampling loss.
- **Trainable quantization matrix** $\mathbf{Q}_\psi$: A pretrained lookup table replacing the standard quantization matrix.
- **JDNO (JPEG Decoding Neural Operator)**: A learned decoder that directly decodes high-quality images from DCT spectra, replacing the conventional JPEG decoder.

Key design principle: JENO, $\mathbf{Q}_\psi$, and JDNO can each flexibly substitute the corresponding components of a conventional JPEG codec.

### Key Designs

#### 1. **Trainable Quantization Matrix ($\mathbf{Q}_\psi$)**

- **Function**: Learns an optimized quantization matrix via a linear layer, replacing the JPEG default matrix.
- **Mechanism**: Takes the standard quantization matrix (quality factor = 50) as input and maps it to an optimized matrix through a linear layer. The non-differentiable rounding operation is approximated by a third-order surrogate: $\lfloor x \rceil \simeq \lfloor x \rceil + (\lfloor x \rceil - x)^3$.
- **Loss**: $\mathcal{L} = \lambda \cdot \|\mathbf{X} - \tilde{\mathbf{X}}\|_2 + \|1/\mathbf{Q}_\psi\|_1$, where $\lambda$ controls the distortion–rate trade-off.
- **Design Motivation**: After training, only the resulting matrix is stored as a lookup table, adding no runtime computation. Seventeen distinct $\mathbf{Q}_\psi$ matrices correspond to seventeen values of $\lambda$.

#### 2. **JENO (JPEG Encoding Neural Operator)**

- **Function**: Mitigates information loss caused by chroma subsampling by learning high-frequency components of the original image.
- **Mechanism**:
    - An EDSR-baseline network extracts RGB image features $\mathbf{z} \in \mathbb{R}^{H \times W \times K}$.
    - Features are sampled at subsampled coordinates and processed via a Galerkin attention mechanism.
    - The output is added to the conventional downsampled result as a residual: $\hat{\mathbf{X}} = \mathcal{G}_\phi(\mathcal{S}(f_\xi(\mathbf{X}), \delta)) + \mathbf{X}'$.
- **Key Property**: JENO effectively learns a high-pass filter—$U(E_\varphi(\mathbf{X})) \simeq HPF(\mathbf{X})$—compensating for high-frequency information discarded during subsampling.
- **Training Objective**: $\hat{\varphi} = \arg\min_\varphi \|\mathbf{X} - U(\hat{\mathbf{X}}_\varphi)\|_1$.

#### 3. **JDNO (JPEG Decoding Neural Operator)**

- **Function**: Directly decodes high-quality images from DCT spectra, replacing the conventional JPEG decoder.
- **Mechanism**:
    - **Group Embedding**: Embeds luma and chroma spectra into a unified representation, supporting 4:2:0/4:2:2/4:4:4 subsampling formats.
    - **Feature Extraction**: Uses SwinV2 attention blocks for feature extraction.
    - **Cosine Neural Operator (CNO)**: Formalizes decoding via continuous cosine functions: $\mathbf{T}_\rho(\mathbf{z}', \delta; \mathbf{Q}) = \mathbf{A} \otimes (\cos(\pi\mathbf{F}_h \otimes \delta_h) \odot \cos(\pi\mathbf{F}_w \otimes \delta_w))$, where $\mathbf{A} = h_q(\mathbf{Q}) \odot h_a(\mathbf{z}')$ integrates quantization matrix priors.
    - Final decoding is completed via Galerkin attention.
- **Design Motivation**: JDNO is quantization-matrix-aware and can adaptively decode according to the degree of compression.

### Loss & Training

- Both JENO and JDNO are trained with L1 loss.
- Training data: DIV2K + Flickr2K (3,450 images), cropped to $112 \times 112$.
- JENO is trained with random selection of 4:2:0 and 4:2:2 chroma subsampling modes.
- JDNO is trained with both standard and pretrained quantization matrices to improve robustness.
- Training runs for 1,000 epochs on 4× RTX 3090 GPUs.

## Key Experimental Results

### Main Results (Comparison with JPEG Artifact Removal Methods, LIVE-1 Dataset)

| Method | Params | q=0 PSNR/PSNR-B | q=10 PSNR/PSNR-B | q=40 PSNR/PSNR-B |
|--------|--------|-----------------|-------------------|-------------------|
| JPEG | — | 20.89/19.73 | 25.69/24.20 | 30.28/28.84 |
| QGAC | 259.4M | 16.33/15.99 | 27.65/27.43 | 32.08/31.64 |
| FBCNN | 70.1M | 21.70/21.19 | 27.77/27.51 | 32.34/31.80 |
| JDEC | 38.9M | 20.76/20.07 | 27.95/27.71 | 32.50/31.98 |
| **JPNeO** | **29.7M** | **23.15/22.64** | **28.15/27.55** | **32.83/31.91** |

**Chroma Component Comparison** (LIVE-1, $\mathbf{X}_C$-PSNR):

| Method | q=0 | q=10 | q=40 |
|--------|-----|------|------|
| DnCNN | 29.27 | 34.47 | 38.98 |
| FBCNN | 29.86 | 37.35 | 41.23 |
| JDEC | 28.75 | 37.95 | 41.92 |
| **JPNeO** | **32.30** | **38.56** | **43.47** |

### Ablation Study

| Configuration | bpp↓ | PSNR↑ | SSIM↑ | Note |
|---------------|------|-------|-------|------|
| JPEG+Q+JPEG | 0.262 | 19.91 | 0.559 | Baseline |
| JPEG+$Q_\psi$+JPEG | 0.260 | 21.21 | 0.581 | Quantization matrix optimization is effective |
| JPEG+Q+JDNO | 0.262 | 22.27 | 0.643 | Decoder optimization yields substantial gains |
| JPEG+$Q_\psi$+JDNO | 0.260 | 23.10 | 0.661 | Combined effect of both |
| JPNeO (Full) | 0.260 | **23.36** | **0.680** | JENO provides further improvement |

**Computational Efficiency Comparison**:

| Method | Params (M) | Memory (GB) | Time (ms) | q=0 PSNR |
|--------|-----------|-------------|-----------|----------|
| FBCNN | 70.1 | 0.61 | 71.95 | 21.70 |
| JDEC | 38.9 | 1.76 | 224.79 | 20.76 |
| JPNeO- (lightweight) | 8.0 | **0.09** | 222.95 | 22.98 |
| JPNeO | 29.7 | 0.26 | 562.42 | **23.15** |

### Key Findings

- **Pronounced advantage at low bit rates**: At q=0, JPNeO surpasses FBCNN by 1.45 dB and JDEC by 2.39 dB.
- **Chroma recovery is the core contribution**: JPNeO leads by 3–5 dB in $\mathbf{X}_C$-PSNR, far exceeding competing methods.
- **JENO is effective at high bpp; JDNO is effective at low bpp**: The encoder raises the upper bound, the decoder raises the lower bound, and the quantization matrix governs the trade-off path between them.
- **Mutual information validation**: Experiments confirm that JENO contributes more mutual information at high quality levels, while JDNO contributes more at low quality levels.
- **The lightweight JPNeO- with only 8M parameters** already outperforms FBCNN (70M) and JDEC (39M).
- **t-SNE visualization**: JENO-encoded images cluster closer to the ground-truth distribution in latent space.

## Highlights & Insights

1. **Full backward compatibility**: JPNeO's encoder and decoder are independently interchangeable—files encoded by JENO can be opened by a standard JPEG decoder, and standard JPEG files can be decoded by JDNO.
2. **Information-theoretic perspective**: The mutual information framework for analyzing information loss and recovery throughout the JPEG pipeline is theoretically elegant.
3. **Systematic focus on chroma components**: This is the first work to systematically address information loss due to JPEG chroma subsampling.
4. **ISP-friendly**: Low memory (0.26 GB) and compact parameter count (29.7M) make JPNeO suitable for integration into edge-device ISP pipelines.
5. **Quantization matrix as a lookup table**: The trained integer matrix is used directly at inference, incurring zero additional runtime overhead.

## Limitations & Future Work

1. **Slow inference**: At 562 ms, JPNeO is substantially slower than FBCNN (72 ms), largely due to Galerkin attention computation.
2. **L1 loss only**: The absence of perceptual or adversarial losses leaves room for improvement in perceptual quality.
3. **Pre-specified $\lambda$ values for training**: The seventeen quantization matrices must be trained and stored separately.
4. **Gap relative to end-to-end learned compression**: While optimal within the JPEG framework, a structural gap remains compared to modern coding standards such as VVC.
5. **Additional computation at the encoder**: JENO increases encoding time, which is unfriendly to real-time photography scenarios.

## Related Work & Insights

- **JDEC** first proposed decoding directly from DCT spectra to bypass the conventional JPEG decoder, but does not support 4:2:2 and 4:4:4 subsampling.
- **FBCNN** proposed blind quality-factor artifact removal and remains the most practical JPEG enhancement method to date.
- **Strümpler et al.** introduced the concept of trainable quantization matrices with pre-editing; this work extends that idea and integrates it with neural operators.
- The transfer of neural operators from PDE solvers to image coding represents a compelling cross-disciplinary direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — A complete solution combining dual-side neural operators with quantization matrix optimization; the information-theoretic analysis is insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive coverage of module ablation, RD curves, chroma analysis, computational efficiency, and mutual information validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Information-theoretic derivations are clear, though the dense notation makes reading demanding.
- **Value**: ⭐⭐⭐⭐ — Offers practical value for the JPEG ecosystem as a plug-and-play backward-compatible enhancement solution.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] From Black Hole to Galaxy: Neural Operator Framework for Accretion and Feedback Dynamics](../../NeurIPS2025/scientific_computing/from_black_hole_to_galaxy_neural_operator_framework_for_accretion_and_feedback_d.md)
- [\[ICLR 2026\] One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers](../../ICLR2026/scientific_computing/one_operator_to_rule_them_all_on_boundary-indexed_operator_families_in_neural_pd.md)
- [\[NeurIPS 2025\] Integration Matters for Learning PDEs with Backward SDEs](../../NeurIPS2025/scientific_computing/integration_matters_for_learning_pdes_with_backward_sdes.md)
- [\[ICLR 2026\] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](../../ICLR2026/scientific_computing/drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [\[CVPR 2026\] NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training](../../CVPR2026/scientific_computing/nestor_a_nested_moe-based_neural_operator_for_large-scale_pde_pre-training.md)

<!-- RELATED:END -->
