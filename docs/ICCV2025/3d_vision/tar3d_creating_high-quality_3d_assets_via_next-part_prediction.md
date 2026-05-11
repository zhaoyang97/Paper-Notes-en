---
title: >-
  [Paper Note] TAR3D: Creating High-Quality 3D Assets via Next-Part Prediction
description: >-
  [ICCV2025][3D Vision][3D generation] TAR3D is proposed as the first framework to quantize triplane representations into discrete geometric parts and generate them autoregressively via GPT. A 3D VQ-VAE encodes meshes of a…
tags:
  - "ICCV2025"
  - "3D Vision"
  - "3D generation"
  - "autoregressive"
  - "VQ-VAE"
  - "triplane representation"
  - "next-part prediction"
  - "GPT"
date: 2026-05-08
content_hash: fc9674d0542f047d
---

# TAR3D: Creating High-Quality 3D Assets via Next-Part Prediction

**Conference**: ICCV2025
**arXiv**: [2412.16919](https://arxiv.org/abs/2412.16919)
**Code**: [GitHub](https://github.com/HVision-NKU/TAR3D)
**Area**: 3D Vision
**Keywords**: 3D generation, autoregressive, VQ-VAE, triplane representation, next-part prediction, GPT

## TL;DR
TAR3D is proposed as the first framework to quantize triplane representations into discrete geometric parts and generate them autoregressively via GPT. A 3D VQ-VAE encodes meshes of arbitrary face counts into fixed-length sequences, while TriPE positional encoding preserves 3D spatial information. The method comprehensively outperforms existing approaches on text/image-to-3D tasks.

## Background & Motivation

**Background**: Conditional 3D generation has achieved significant progress along three main directions: (1) SDS-based optimization (e.g., DreamFusion), which is time-consuming and prone to multi-face artifacts; (2) multi-view synthesis (e.g., Zero123), which depends on view consistency and tends to lose fine details; (3) native 3D diffusion (e.g., Direct3D, Clay), which is difficult to unify with LLM-based modeling.

**Limitations of Prior Work**: Approaches such as MeshGPT and MeshXL attempt to introduce autoregressive LLMs into 3D generation, but directly quantizing mesh faces yields a sequence length of $9 \times$ the face count. Industrial-grade models with hundreds of thousands of faces thus produce sequences of intractable length that do not scale.

**Core Problem**: Can meshes of arbitrary face counts be encoded into fixed-length sequences, making GPT-based autoregressive modeling of 3D objects feasible?

**Key Insight**: Triplane representations naturally compress 3D information into fixed-size feature maps, and their 2D feature map structure is amenable to image-based VQ quantization strategies, enabling face-count-agnostic fixed-length discrete sequences.

## Method

### Overall Architecture
TAR3D consists of two core modules: a **3D VQ-VAE** that encodes 3D shapes into discrete triplane features, and a **3D GPT** that autoregressively predicts codebook index sequences. Training proceeds in two stages: the VQ-VAE is first trained to obtain a high-quality discrete representation, followed by GPT training for sequence generation.

### 3D VQ-VAE

#### 3D Shape Encoder
- Input: 81,920 points uniformly sampled from the 3D mesh surface (with normals), $P \in \mathbb{R}^{B \times N_p \times 6}$
- Fourier positional encoding is applied to the point cloud to capture high-frequency details
- Transformer architecture: 1 cross-attention layer + 8 self-attention layers
- Point cloud information is injected into learnable query tokens ($3 \times 32 \times 32 = 3072$ tokens, channel dimension 768) via cross-attention
- Output: triplane latent representation $\hat{z} \in \mathbb{R}^{B \times 3072 \times d_z}$

#### Quantizer
- Learnable codebook $Z = \{z_k\}_{k=1}^{K}$, $K = 16384$, channel dimension $d_q = 8$
- A linear projection maps continuous features to the codebook dimension
- Element-wise nearest-neighbor quantization: $z_q = \arg\min \|\tilde{z}_{ij} - z_k\|$
- **Sequence length is fixed at $3 \times 32 \times 32 = 3072$**, entirely independent of mesh face count

#### 3D Geometry Decoder
- $N_2 = 6$ attention layers, each comprising 2 feature deformation steps and self-attention
- **Intra-plane self-attention**: ignores the plane axis, processing each plane independently
- **Inter-plane self-attention**: concatenates the three planes along the height dimension for cross-plane information interaction (PII)
- Triplane features are upsampled to $256 \times 256$ resolution
- Query point features are sampled from the triplane via bilinear interpolation and fed into an MLP to predict occupancy values
- Query points consist of 20,480 volumetric points + 20,480 near-surface points

### 3D GPT

#### Sequence Construction
- Sequences are built from codebook indices obtained from the pretrained VQ-VAE
- **Intra-plane**: raster scan order
- **Inter-plane**: indices at the same spatial position across the three planes are arranged adjacently
- Conditional prompts are embedded as prefilling tokens

#### TriPE Positional Encoding
- A custom 3D positional encoding combining 2D RoPE and 1D RoPE
- **TriP_2D**: each element of the 2D positional encoding is repeated 3 times and arranged adjacently, preserving intra-plane 2D spatial information
- **TriP_1D**: the 1D positional encoding (3 elements) is each repeated $h \times w$ times, distinguishing the three feature planes
- **TriPE = TriP_2D + TriP_1D** (element-wise addition)
- Retains significantly more 3D spatial information than naive 1D RoPE

#### Autoregressive Generation
- Sequence $s \in \{0, \ldots, K-1, K\}^{3 \cdot h \cdot w}$
- GPT-L architecture (LLamaGen configuration): 24 Transformer layers, 16 heads, hidden dimension 1024
- Condition encoding: DINO (ViT-B16) for images; FLAN-T5 XL for text
- Classifier-free guidance (CFG, scale = 7.5) is applied at inference to improve quality and alignment
- Training objective: maximize the log-likelihood of the triplane index sequence

### Training Details
- VQ-VAE loss: $\mathcal{L} = \lambda_{\text{rec}} \cdot \mathcal{L}_{\text{rec}}(\text{BCE}) + \lambda_{\text{cb}} \cdot \mathcal{L}_{\text{cb}}(\text{codebook learning})$
- $\lambda_{\text{rec}} = 1$, $\lambda_{\text{cb}} = 0.1$, $\beta = 0.25$
- Optimizer: AdamW, learning rate $1 \times 10^{-4}$, cosine annealing schedule
- VQ-VAE: batch size 128, 100K steps, $8 \times$ A100
- GPT: batch size 80, 100K steps

## Key Experimental Results

### Datasets

| Dataset | Scale | Usage |
|---------|-------|-------|
| ShapeNet | 52,472 meshes / 55 categories | 48,597 train + 2,592 test |
| Objaverse | ~100K (filtered from 800K) | ~99K train + 1K evaluation |
| GSO | ~1,000 real scans | Out-of-domain generalization |

### Quantitative Comparison on Image-to-3D (Tab. 1)

| Method | PSNR↑ | SSIM↑ | CLIP↑ | LPIPS↓ | CD↓ | F-Score↑ |
|--------|-------|-------|-------|--------|-----|----------|
| Shap-E | 10.991 | 0.702 | 0.834 | 0.325 | 0.156 | 0.163 |
| SyncDreamer | 11.269 | 0.706 | 0.837 | 0.320 | 0.158 | 0.178 |
| Michelangelo | 11.928 | 0.734 | 0.864 | 0.278 | 0.117 | 0.226 |
| InstantMesh | 11.560 | 0.721 | 0.847 | 0.303 | 0.137 | 0.179 |
| LGM | 11.363 | 0.714 | 0.841 | 0.317 | 0.149 | 0.172 |
| **TAR3D** | **13.626** | **0.763** | **0.868** | **0.216** | **0.066** | **0.303** |

- PSNR improves by 1.7+, CD is reduced by over 43%, and F-Score improves by over 34%, achieving substantial leads across all metrics.

### Ablation Study

| Ablation | CD↓ | F-Score↑ |
|----------|-----|----------|
| 3D VAE (no quantization) | 0.018 | 0.811 |
| **3D VQ-VAE** | **0.016** | **0.822** |
| Without PII | 0.023 | 0.661 |
| **With PII** | **0.016** | **0.822** |

### Triplane Resolution Ablation

| Triplane Size | CD↓ | Inference Time |
|---------------|-----|----------------|
| $3 \times 16 \times 16$ | 0.157 | 17.7s |
| $3 \times 32 \times 32$ | **0.066** | **67.6s** |
| $3 \times 48 \times 48$ | 0.062 | 143.9s |

- $32 \times 32$ achieves the best quality-efficiency trade-off: CD drops from 0.157 to 0.066 at an acceptable inference cost, while $48 \times 48$ yields only marginal improvement at twice the inference time.

## Highlights & Insights

1. **Fixed-length sequences are the key breakthrough**: MeshGPT's sequence length scales with face count, making it non-scalable. TAR3D compresses meshes of arbitrary complexity into 3,072 tokens via triplane VQ quantization, enabling truly scalable 3D autoregressive modeling.

2. **Elegant design of TriPE**: Rather than naively applying 1D positional encoding to a flattened sequence, TriPE fuses 2D (intra-plane spatial) and 1D (inter-plane identity) encodings. Ablation experiments confirm that naive 1D RoPE loses critical geometric detail.

3. **Importance of PII**: F-Score improves from 0.661 to 0.822 (+24%), demonstrating that cross-plane information interaction is essential for high-quality reconstruction and that the three planes cannot be processed independently.

4. **VQ-VAE outperforms VAE**: Contrary to the intuition that quantization degrades information, the 3D VQ-VAE achieves better reconstruction quality than its continuous counterpart (CD: 0.016 vs. 0.018), suggesting that the discrete codebook learns superior geometric part representations.

5. **GPT paradigm unifies multi-modal 3D generation**: Image and text conditions are naturally unified as prefilling tokens, supporting multi-modal conditional generation in a manner compatible with the broader LLM ecosystem.

## Limitations & Future Work

1. **Geometry only, no texture**: The current version generates shape only; texture relies on an external synthesizer (SyncMVD), and end-to-end texture generation remains to be addressed.
2. **Inference efficiency**: The $32 \times 32$ configuration requires 67.6 seconds per sample, which is slower than diffusion-based methods that generate in seconds; autoregressive sequential dependencies limit parallelization.
3. **Occupancy field representation**: The use of occupancy fields rather than SDF or DMTet may constrain topological complexity and thin-structure reconstruction.
4. **Limited data scale**: Only ~100K Objaverse samples are used; the authors indicate that future work will incorporate Objaverse-XL and explore scaling laws.
5. **Larger GPT variants unexplored**: Only GPT-L (24 layers, hidden dim 1024) is evaluated; scaling behavior and emergent capabilities remain to be investigated.

## Related Work & Insights

- **MeshGPT/MeshXL**: Direct quantization of mesh faces yields excessively long sequences; TAR3D circumvents this fundamental bottleneck via triplane representations.
- **Direct3D/Clay**: 3D diffusion-based approaches; TAR3D provides an autoregressive alternative that integrates more naturally with the LLM paradigm.
- **LLamaGen**: Successful autoregressive image generation validates the paradigm; TAR3D extends it to the 3D domain.
- **Inspiration**: The triplane + VQ approach is extensible to 4D (dynamic 3D) generation; the TriPE positional encoding design is applicable to other structured sequence modeling tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First framework combining triplane quantization with GPT autoregressive 3D generation, resolving the sequence length bottleneck
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset, multi-baseline comparisons with thorough ablations, though large-scale experiments are absent
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-motivated, with intuitive illustrations
- Value: ⭐⭐⭐⭐⭐ Opens a viable path toward unifying 3D generation with the LLM ecosystem

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TAPNext: Tracking Any Point (TAP) as Next Token Prediction](tapnext_tracking_any_point_tap_as_next_token_prediction.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[NeurIPS 2025\] ARMesh: Autoregressive Mesh Generation via Next-Level-of-Detail Prediction](../../NeurIPS2025/3d_vision/armesh_autoregressive_mesh_generation_via_next-level-of-detail_prediction.md)
- [\[ICCV 2025\] Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation](bridging_3d_anomaly_localization_and_repair_via_high-quality_continuous_geometri.md)
- [\[ICCV 2025\] Find Any Part in 3D](find_any_part_in_3d.md)

</div>

<!-- RELATED:END -->
