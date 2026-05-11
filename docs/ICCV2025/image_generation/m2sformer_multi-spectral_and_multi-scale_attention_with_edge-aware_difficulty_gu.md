---
title: >-
  [Paper Note] M2SFormer: Multi-Spectral and Multi-Scale Attention with Edge-Aware Difficulty Guidance for Image Forgery Localization
description: >-
  [ICCV 2025][Image Generation][Image Forgery Localization] This paper proposes M2SFormer, which unifies multi-spectral (2D DCT frequency-domain) and multi-scale (SIFT-style spatial pyramid) attention mechanisms within enc…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Image Forgery Localization"
  - "Multi-Spectral Attention"
  - "Multi-Scale Attention"
  - "Difficulty Guidance"
  - "Transformer"
date: 2026-05-08
content_hash: 18500a5cdee2d187
---

# M2SFormer: Multi-Spectral and Multi-Scale Attention with Edge-Aware Difficulty Guidance for Image Forgery Localization

**Conference**: ICCV 2025
**arXiv**: [2506.20922](https://arxiv.org/abs/2506.20922)
**Code**: N/A
**Area**: Image Forgery Detection / Image Generation
**Keywords**: Image Forgery Localization, Multi-Spectral Attention, Multi-Scale Attention, Difficulty Guidance, Transformer

## TL;DR

This paper proposes M2SFormer, which unifies multi-spectral (2D DCT frequency-domain) and multi-scale (SIFT-style spatial pyramid) attention mechanisms within encoder-decoder skip connections, and introduces an edge-aware curvature-based difficulty-guided attention decoder. The method achieves state-of-the-art cross-domain generalization in image forgery localization (average unseen-domain DSC 43.0% and mIoU 34.3% under the CASIAv2 training protocol).

## Background & Motivation

### Problem Definition

Image Forgery Localization requires models to accurately segment tampered (spliced or copy-moved) regions at the pixel level. The core challenges are: (1) forgery traces are typically extremely subtle; and (2) models must generalize to unseen forgery types and data domains.

### Limitations of Prior Work

**CNN-based methods (MantraNet, SPAN, RRUNet, etc.)**: High computational cost, limited representational capacity, difficulty in capturing global dependencies, and insufficient generalization.

**Transformer-based methods (TransForensic)**: Model global dependencies via self-attention but do not fully exploit frequency-domain information.

**Frequency-domain methods (FBINet, ObjectFormer)**: Use 2D DCT to reveal hidden forgery traces, but typically require additional dual-encoder or multi-modal training, incurring high computational cost.

**Separation of frequency and spatial domains**: Existing methods generally process spatial and frequency features separately, lacking a unified attention mechanism to jointly exploit both.

### Core Problem

**Key question**: How can spatial and frequency-domain attention be efficiently integrated while effectively capturing subtle forgery cues?

The human visual system detects subtle artifacts using multiple frequency bands (multi-spectral) and captures forgery patterns of varying sizes across multiple spatial scales (multi-scale). Unifying both within skip connections, combined with a difficulty-aware decoder, simultaneously improves accuracy and cross-domain generalization.

## Method

### Overall Architecture

M2SFormer adopts a Transformer encoder-decoder architecture (PVT-v2 backbone) with two core components: (1) M2S attention modules in skip connections that fuse multi-spectral and multi-scale information; and (2) an Edge-Aware DGA decoder that adaptively adjusts attention based on sample difficulty.

### Key Designs

#### 1. **Multi-Spectral Attention**

- **Function**: Applies 2D DCT basis images to perform frequency-domain channel attention recalibration on cross-scale fused features, capturing forgery traces across different frequency components.
- **Mechanism**: Encoder features from each layer are first unified to a target resolution and concatenated to obtain $\mathbf{f}_c \in \mathbb{R}^{C \times H_t \times W_t}$. Frequency-domain feature components are then computed using 2D DCT basis images:
  $$\mathbf{f}_c^k = \sum_{h=0}^{H_t-1} \sum_{w=0}^{W_t-1} (\mathbf{f}_c)_{:,h,w} \mathbf{D}_{h,w}^{u_k,v_k}$$
  The most relevant frequency components are selected via a top-K strategy, and a channel attention map is generated through GAP/GMP combined with a statistical aggregation block:
  $$\mathbf{M}^{\text{spectral}} = \sigma\left(\sum_{d \in \{\text{avg}, \text{max}\}} \sum_{k=1}^K \text{C2D}_{1\times1}(\delta(\text{C2D}_{1\times1}(\mathbf{f}_c^k)))\right)$$
- **Design Motivation**: Unlike FBINet, which applies 2D DCT directly to the input image, the proposed method operates on fused features within skip connections, avoiding the computational redundancy of dual encoders while preserving coarse-to-fine multi-level semantic information.

#### 2. **Multi-Scale Attention**

- **Function**: Inspired by SIFT feature pyramids, an attention pyramid is constructed across multiple spatial scales to capture forgery patterns of varying sizes.
- **Mechanism**: The spectrally recalibrated feature $\bar{\mathbf{f}}_c$ is downsampled at multiple scales; at each scale, dilated convolutions and $1\times1$ convolutions compress the channels:
  $$\bar{\mathbf{f}}_c^l = \text{C2D}_{1\times1}(\text{DC2D}_{3\times3}^{2l+1}(\text{Down}_l(\bar{\mathbf{f}}_c)))$$
  Learnable parameters $\alpha_i^l, \beta_i^l$ are introduced at each pyramid level to control information flow through foreground/background attention:
  $$\hat{\mathbf{f}}_i^l = \text{C2D}_{3\times3}(\alpha_i^l(\bar{\mathbf{f}}_i^l \times \mathbf{F}_i^l) + \beta_i^l(\bar{\mathbf{f}}_i^l \times \mathbf{B}_i^l))$$
  where $\mathbf{F}_i^l = \sigma(\text{C2D}_{1\times1}(\bar{\mathbf{f}}_i^l))$ is the foreground map and $\mathbf{B}_i^l = 1 - \mathbf{F}_i^l$.
- **Design Motivation**: By decomposing foreground and background at multiple scales and weighting them independently, the model flexibly adapts to forgery regions of varying sizes, from small splices to large copy-moved areas.

#### 3. **Edge-Aware Difficulty-Guided Attention (Edge-Aware DGA)**

- **Function**: Automatically assesses the forgery localization difficulty (easy/hard) of each sample, generates a textual description, and guides the decoder via channel attention.
- **Mechanism**: A global prior map $\mathbf{G}$ is generated from the deepest feature layer; Sobel filters are applied to compute first- and second-order derivatives, yielding a curvature map:
  $$\kappa = \frac{G_x^2 G_{yy} - 2G_x G_y + G_y^2 G_{xx}}{(G_x^2 + G_y^2)^{1.5}}$$
  Mean curvature is computed exclusively over edge regions: $s = \sigma(\sum(\kappa \otimes E) / \sum E)$. If $s \geq 0.5$, the sample is labeled "hard"; otherwise "easy." The text label is encoded into a vector $\mathcal{T}$ via BPE, linearly embedded, and used to guide channel-wise attention in the decoder.
- **Design Motivation**: Low-curvature regions are easier to handle, while high-curvature regions require more perceptual resources. Edge-aware curvature provides a more representative difficulty measure than global average curvature, as it avoids dilution of the signal by large zero-curvature areas. Translating difficulty into text-driven attention allows the model to adaptively allocate attentional resources.

### Loss & Training

- Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\mathbf{R}_t, \mathbf{R}_p) + \mathcal{L}_{\text{BCE}}(\mathbf{R}_t, \text{Up}_{32}(\mathbf{G}))$
- End-to-end training for 100 epochs, batch size 32, Adam optimizer, cosine annealing learning rate schedule.
- PVT-v2-B2 serves as the encoder/decoder backbone, initialized with ImageNet-1K pretrained weights.

## Key Experimental Results

### Main Results

**CASIAv2 training protocol — unseen-domain generalization performance (DSC/mIoU %):**

| Method | CASIAv1 | Columbia | IMD2020 | CoMoFoD | In the Wild | MISD |
|--------|---------|----------|---------|---------|-------------|------|
| TransForensic | 44.2/35.0 | 35.9/25.0 | 27.2/19.1 | 21.7/14.3 | 31.9/22.4 | 60.0/46.5 |
| PIMNet | 49.7/42.2 | 32.5/23.1 | 29.6/22.2 | 24.7/16.8 | 31.2/22.9 | 61.1/48.2 |
| EITLNet | 52.9/46.5 | 28.0/20.9 | 25.3/19.7 | 18.1/12.4 | 24.3/19.0 | 58.8/45.9 |
| **M2SFormer** | **58.4/50.1** | **42.4/32.4** | **32.6/24.9** | **24.9/16.8** | **35.0/27.4** | **69.1/56.9** |

### Ablation Study

**Ablation of M2S attention modules (CASIAv2 protocol):**

| Config | Spectral | Scale | Seen DSC | Unseen DSC | Params | FLOPs |
|--------|----------|-------|----------|------------|--------|-------|
| S0 | Single | Single | 56.3 | 27.1 | 26.2M | 13.8G |
| S1 | Single | Multi | 55.5 | 33.6 | 27.4M | 14.2G |
| S2 | Multi | Single | 55.9 | 36.1 | 26.2M | 13.8G |
| S3 (Full) | Multi | Multi | **58.8** | **43.0** | 27.4M | 14.2G |

**Ablation of DGA decoder:**

| Config | Seen DSC/mIoU | Unseen DSC/mIoU |
|--------|--------------|-----------------|
| No DGA | 55.5/49.5 | 32.3/26.1 |
| Simple DC + DGA | — | Limited gain |
| Edge-Aware DC + DGA | **58.8/50.8** | **43.0/34.3** |

### Key Findings

1. **Synergy between multi-spectral and multi-scale attention**: Each component individually contributes +6.5/+9.0 unseen DSC; their combination yields +15.9 (S3 vs. S0), demonstrating high complementarity between frequency-domain and spatial-scale information.
2. **Edge-aware difficulty guidance is critical**: EADC+DGA improves unseen DSC from 32.3 to 43.0 (+33%); simple difficulty conditioning (Simple DC) yields limited benefit.
3. **M2SFormer substantially outperforms baselines in cross-domain generalization**: DSC reaches 69.1% on MISD, surpassing the second-best method PIMNet by 8 percentage points.
4. **Computational efficiency**: Only 27.4M parameters and 14.2G FLOPs, making it lighter than the dual-encoder FBINet.

## Highlights & Insights

1. **Unified attention in skip connections**: Placing frequency-domain and spatial multi-scale attention in skip connections rather than within the encoder or decoder preserves the original feature information flow while injecting rich forgery artifact signals.
2. **Text-driven adaptive difficulty guidance**: Converting curvature measurements into "hard"/"easy" text labels encoded via BPE to drive attention eliminates dependence on external metadata.
3. **No additional training or fine-tuning required**: Strong generalization across multiple unseen domains is achieved with a single training set, without fine-tuning on external data as required by other methods.

## Limitations & Future Work

1. **Coarse binary difficulty partitioning**: Only "hard" and "easy" categories are used; finer-grained difficulty quantization may yield better adaptivity.
2. **Necessity of BPE text encoding is questionable**: Whether directly using curvature scalars for conditional attention would suffice, and whether the textual intermediate representation introduces unnecessary complexity, remains unclear.
3. **Not evaluated on AI-generated image detection**: With the growing prevalence of AI-generated and deepfake imagery, the method's generalization to such content has yet to be verified.
4. **Fixed threshold $\tau=0.5$**: No adaptive thresholding scheme is provided.
5. **High standard deviation in 5-fold cross-validation**: DSC standard deviation exceeds 10% on some datasets, indicating room for improvement in stability.

## Related Work & Insights

- **Comparison with FBINet**: FBINet applies 2D DCT directly to input images and requires a separate frequency-domain encoder; M2SFormer operates on cross-scale fused features, achieving greater efficiency.
- **Comparison with EITLNet**: EITLNet employs a dual encoder and MLP decoder with higher parameter counts and FLOPs; M2SFormer achieves superior accuracy with a single encoder, Transformer decoder, and DGA.
- **Insight**: Incorporating difficulty-aware mechanisms into other pixel-level tasks (e.g., semantic segmentation, depth estimation) may prove equally effective.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The M2S attention module and edge-aware difficulty guidance are creative designs, though the inspiration for each sub-module is clearly traceable to prior work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple training protocols, multiple unseen domains, and 5-fold cross-validation provide comprehensive coverage; ablations are thorough.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear and figures are detailed, though the frequency-domain formulation section is dense.
- **Value**: ⭐⭐⭐⭐ — Provides a lightweight solution with strong generalization capability for image forgery localization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MMAIF: Multi-task and Multi-degradation All-in-One for Image Fusion with Language Guidance](mmaif_multi-task_and_multi-degradation_all-in-one_for_image_fusion_with_language.md)
- [\[ICCV 2025\] Semantic Discrepancy-aware Detector for Image Forgery Identification](semantic_discrepancy-aware_detector_for_image_forgery_identification.md)
- [\[ICCV 2025\] DiTFastAttnV2: Head-wise Attention Compression for Multi-Modality Diffusion Transformers](ditfastattnv2_head-wise_attention_compression_for_multi-modality_diffusion_trans.md)
- [\[ICCV 2025\] Spectral Image Tokenizer](spectral_image_tokenizer.md)
- [\[ICCV 2025\] MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence](mamtiff-cad_multi-scale_latent_diffusion_with_mamba_for_complex_parametric_seque.md)

</div>

<!-- RELATED:END -->
