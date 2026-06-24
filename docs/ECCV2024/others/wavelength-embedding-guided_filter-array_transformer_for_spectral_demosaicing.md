---
title: >-
  [Paper Note] Wavelength-Embedding-guided Filter-Array Transformer for Spectral Demosaicing
description: >-
  [ECCV 2024][Spectral Demosaicing] This paper proposes WeFAT, which endows the model with "wavelength memory" capability through Wavelength-Embedding-guided Multi-head Self-Attention (We-MSA). Combined with the Masked Attention Mechanism (MaM) to focus on high-quality spectral regions, it maintains stable performance under different cameras and spectral distributions when trained only on the ARAD dataset, outperforming existing SOTA methods.
tags:
  - "ECCV 2024"
  - "Spectral Demosaicing"
  - "Wavelength Embedding"
  - "Filter Array"
  - "Transformer"
  - "Multispectral Imaging"
date: 2026-05-08
content_hash: e67c3c2e544e50ef
---

# Wavelength-Embedding-guided Filter-Array Transformer for Spectral Demosaicing

**Conference**: ECCV 2024  
**PDF**: [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02182.pdf)
**Code**: None  
**Area**: Others (Spectral Imaging/Low-level Vision)  
**Keywords**: Spectral Demosaicing, Wavelength Embedding, Filter Array, Transformer, Multispectral Imaging

## TL;DR

This paper proposes WeFAT, which endows the model with "wavelength memory" capability through Wavelength-Embedding-guided Multi-head Self-Attention (We-MSA). Combined with the Masked Attention Mechanism (MaM) to focus on high-quality spectral regions, it maintains stable performance under different cameras and spectral distributions when trained only on the ARAD dataset, outperforming existing SOTA methods.

## Background & Motivation

**Background**: Spectral imaging captures the reflection/radiation information of scenes over multiple spectral bands and is widely used in remote sensing, food inspection, medical diagnosis, etc. Multispectral Filter Array (MSFA) is a low-cost spectral imaging solution that covers the sensor array with filters of different wavelengths, but each pixel only captures information of a single band, requiring Spectral Demosaicing to reconstruct the full spectral image.

**Limitations of Prior Work**: (1) Existing CNN and attention models struggle to capture inter-spectral similarities and long-range dependencies—there are both correlations (high similarity between neighboring bands) and differences across spectral bands, which the local receptive field of convolution fails to model; (2) When the optical characteristics of cameras change (e.g., different MSFA patterns or wavelength distributions), the performance of existing models degrades severely, requiring retraining; (3) There is a lack of methods to structurally integrate the physical information of imaging systems (such as MSFA patterns and wavelength distributions) into the model.

**Key Challenge**: Different cameras have diverse MSFA arrangements and wavelength distributions, while training data is usually sourced from a single specific camera. Existing methods implicitly model spectral-spatial relationships, failing to generalize to new cameras. The root cause is that the model does not "understand" wavelengths—it only perceives feature channel indices without knowing the physical wavelengths corresponding to each channel.

**Goal**: (1) How to endow the model with wavelength awareness to adapt to spectral distributions of different cameras? (2) How to effectively utilize the spatial pattern information of MSFA to guide demosaicing? (3) How to train a model on a single dataset that generalizes across cameras?

**Key Insight**: Inspired by timestep embedding in diffusion models—where scalar timestep information is injected into the network to alter its behavior—physics wavelength information can be similarly injected into the attention computation through embedding. This allows the model to "remember" the wavelength corresponding to each spectral channel, achieving adaptive adjustments when processing data from different cameras.

**Core Idea**: Inject wavelength information as embeddings into the multi-head self-attention computation to endow the model with cross-camera wavelength adaptivity, while utilizing MSFA patterns to guide attention toward high-quality sampled regions.

## Method

### Overall Architecture

WeFAT (Wavelength Embedding guided Filter Array Attention Transformer) takes the raw mosaic image sampled by MSFA as input, initially expands it into multi-channel initial spectral estimation using an initialization module, and then refines it progressively through multiple Transformer blocks (each containing We-MSA and MaM modules) to output the complete multispectral image. During training, the model accepts the camera's wavelength distribution and MSFA configuration as conditional inputs.

### Key Designs

1. **Wavelength-Embedding-guided Multi-head Self-Attention (We-MSA)**:

    - **Function**: Infuses physical wavelength information into self-attention computation, equipping the model with wavelength memory and cross-camera adaptation capability.
    - **Mechanism**: For each spectral channel $i$, its center wavelength $\lambda_i$ is mapped to an embedding vector $e_i = \text{PE}(\lambda_i) \in \mathbb{R}^d$ via sinusoidal position encoding. In multi-head self-attention, each spectral feature is treated as a token, and wavelength embeddings are directly added to the query and key to participate in attention computation: $\text{Attention}(Q+E_Q, K+E_K, V)$, where $E_Q, E_K$ are linear-transformed wavelength embeddings. This makes attention weights dependent not only on feature content but also on the physical relationships between wavelengths.
    - **Design Motivation**: Inspired by positional encoding in Transformers and timestep embeddings in diffusion models. Wavelength embeddings inform the model that "channel $i$ corresponds to 520nm and channel $j$ corresponds to 650nm", allowing it to leverage physical spectral continuity—where adjacent wavelength channels should exhibit similar features, a prior that remains constant when switching cameras.

2. **Filter-Array Attention Mechanism (MaM)**:

    - **Function**: Uses the spatial sampling patterns of MSFA to guide attention, focusing on spatial positions providing high-quality spectral information.
    - **Mechanism**: For a given MSFA pattern, the spatial sampling locations of each spectral channel are known. MaM generates a sampling mask $M \in \{0, 1\}^{H \times W \times C}$ based on the MSFA pattern, marking which channels are directly sampled (high-quality) and which need interpolation-based reconstruction (low-quality) at each spatial position. In attention computation, the mask is converted into an attention bias to make the model place higher trust on features at directly sampled locations: $A = \text{softmax}((QK^T + \beta M_{attn})/\sqrt{d})V$.
    - **Design Motivation**: In MSFA, certain channels at specific locations are directly measured, while others require interpolation. Revealing this sampling prior to the model prevents misleading attention weights at interpolated positions, thereby improving reconstruction quality.

3. **Spectral Self-Similarity Modeling**:

    - **Function**: Explicitly models the similarity patterns between different spectral bands to improve spectral reconstruction accuracy.
    - **Mechanism**: Introduces a spectral similarity matrix $S \in \mathbb{R}^{C \times C}$ between Transformer layers, where $S_{ij}$ denotes the similarity between the $i$-th and $j$-th bands. This matrix is initialized by the dot product of wavelength embeddings ($S_{ij}^{init} = e_i \cdot e_j / \|e_i\|\|e_j\|$) and fine-tuned during training. The spectral self-similarity matrix acts as residual connection weights, guiding feature propagation across bands.
    - **Design Motivation**: Spectral responses of adjacent bands are highly similar. Leveraging this similarity allows "borrowing" information from known bands to reconstruct unknown bands, which is particularly crucial when MSFA sampling is sparse.

### Loss & Training

The model uses an L1 loss + Spectral Angle Mapper (SAM) loss: $L = L_1 + \lambda L_{SAM}$, where $L_1$ constrains pixel-level reconstruction accuracy, and $L_{SAM}$ constrains the shape fidelity of spectral curves: $L_{SAM} = \arccos(\frac{x \cdot \hat{x}}{\|x\|\|\hat{x}\|})$. It is trained solely on the ARAD dataset (single camera) and evaluated by directly transferring to cameras with different MSFA patterns and wavelength distributions.

## Key Experimental Results

### Main Results

| Camera/Dataset | Metric | WeFAT | PPID | MSFA-Net | Gain |
|------------|------|-------|------|----------|------|
| ARAD (In-distribution) | PSNR ↑ | **42.8** | 40.3 | 39.7 | +2.5 dB |
| ARAD (In-distribution) | SAM ↓ | **1.82** | 2.31 | 2.54 | -21.2% |
| Camera B (4×4 MSFA) | PSNR ↑ | **38.5** | 34.1 | 33.6 | +4.4 dB |
| Camera C (5×5 MSFA) | PSNR ↑ | **37.2** | 32.8 | 31.9 | +5.3 dB |
| Real Data | PSNR ↑ | **35.6** | 31.2 | 30.5 | +4.4 dB |

### Ablation Study

| Configuration | PSNR (In-distribution) | PSNR (Cross-camera) | Description |
|------|-------------|-------------|------|
| WeFAT (Full) | **42.8** | **38.5** | Full model |
| w/o Wavelength Embedding | 41.9 | 33.8 | Wavelength embedding is crucial for cross-camera generalization |
| w/o MaM | 41.5 | 36.7 | MaM contributes more in the in-distribution scenario |
| w/o Spectral Self-Similarity | 42.1 | 37.2 | Stable contribution from self-similarity modeling |
| Replacing wavelength embedding with channel index embedding | 42.3 | 34.5 | Physical wavelength information is significantly superior to sequential indices |

### Key Findings
- Wavelength embedding is crucial for cross-camera generalization: removing wavelength embedding leads to a drop of 0.9 dB in-distribution, but a severe drop of 4.7 dB in cross-camera settings, demonstrating that wavelength embedding successfully endows the model with cross-camera adaptability.
- The superiority of WeFAT in cross-camera scenarios is significantly larger than in in-distribution scenarios (gain of 4-5 dB vs. 2.5 dB), indicating that the core value of this method lies in its generalization capability.
- Replacing physical wavelengths with channel indices for embedding drastically degrades performance, proving the critical importance of injecting physical information rather than arbitrary identifiers.

## Highlights & Insights
- **The design of wavelength embedding is inspired by timestep embedding in diffusion models**. This paradigm of "injecting physical scalar information into attention via embedding" is highly versatile and can be transferred to any scenario requiring the integration of continuous physical parameters (such as temperature, frequency, angle) into networks.
- **The approach of using MSFA patterns as attention priors** directly encodes hardware properties of the imaging system into the network, achieving physics-aware deep learning. This software-hardware co-design paradigm is highly valuable in computational imaging.
- The ability to generalize across cameras with training on only a single dataset is of great significance for practical deployment, avoiding the high cost of collecting training data for every new camera.

## Limitations & Future Work
- Although cross-camera generalization performs well, it assumes that the wavelength distribution and MSFA arrangement of the new camera are known, making it inapplicable to completely unknown camera configurations.
- Experiments are limited to the spectral demosaicing task, without extension to related tasks like hyperspectral image super-resolution or spectral reconstruction.
- Robustness to noise is not sufficiently analyzed—dark current noise and photon noise are significant in real-world spectral imaging.
- The information density of wavelength embeddings can be further enriched by combining physical optical models, such as filter transmittance curves.

## Related Work & Insights
- **vs. PPID**: PPID uses predefined interpolation patterns for spectral demosaicing without wavelength-aware designs, suffering from sharp performance degradation in cross-camera scenarios. WeFAT achieves camera-agnostic spectral reconstruction via wavelength embeddings.
- **vs. MSFA-Net**: MSFA-Net uses CNNs to model spectral-spatial relationships but lacks the capability to model long-range dependencies and wavelength awareness. The Transformer architecture + wavelength embedding of WeFAT entirely outperforms CNN-based methods.
- **vs. Restormer (General Image Restoration)**: Restormer, as a powerful general restoration model, performs excellently on standard demosaicing (RGB Bayer). However, it lacks wavelength and MSFA priors in spectral demosaicing, resulting in inferior performance compared to the specialized WeFAT.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The design of injecting wavelength embedding into attention is highly inspiring, and the integration of MSFA attention prior is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ The cross-camera generalization experiments are cleverly designed, with deep ablation analysis.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, and the physical motivation is fully elucidated.
- Value: ⭐⭐⭐⭐ Provides a general and elegant demosaicing solution for spectral imaging, with the wavelength embedding concept being widely referable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Exploring Guided Sampling of Conditional GANs](exploring_guided_sampling_of_conditional_gans.md)
- [\[AAAI 2026\] Guided Perturbation Sensitivity (GPS): Detecting Adversarial Text via Embedding Stability and Word Importance](../../AAAI2026/others/guided_perturbation_sensitivity_gps_detecting_adversarial_text_via_embedding_sta.md)
- [\[ECCV 2024\] Rethinking Data Bias: Dataset Copyright Protection via Embedding Class-Wise Hidden Bias](rethinking_data_bias_dataset_copyright_protection_via_embedding_class-wise_hidde.md)
- [\[ICLR 2026\] SONIC: Spectral Oriented Neural Invariant Convolutions](../../ICLR2026/others/sonic_spectral_oriented_neural_invariant_convolutions.md)
- [\[CVPR 2025\] CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction](../../CVPR2025/others/care_transformer_linear_attention.md)

</div>

<!-- RELATED:END -->
