---
title: >-
  [Paper Note] Domain Generalizable Portrait Style Transfer
description: >-
  [ICCV 2025][Image Generation][Portrait Style Transfer] DGPST proposes a diffusion-based portrait style transfer framework that establishes cross-domain dense semantic correspondences via a semantic adapter to warp the reference image, employs AdaIN-Wavelet Transform for latent space initialization to balance stylization and content preservation, and generates final results through a dual-conditional diffusion model combining ControlNet (high-frequency structural guidance) and a style adapter (style guidance). The model is trained solely on 30K real portrait photographs yet generalizes to diverse domains including photos, cartoons, sketches, and anime.
tags:
  - ICCV 2025
  - Image Generation
  - Portrait Style Transfer
  - Semantic Correspondence
  - Wavelet Transform
  - Diffusion Model
  - Cross-Domain Generalization
date: 2026-05-08
content_hash: 385dc56442ddd7a1
---

# Domain Generalizable Portrait Style Transfer

**Conference**: ICCV 2025
**arXiv**: [2507.04243](https://arxiv.org/abs/2507.04243)
**Code**: [https://github.com/wangxb29/DGPST](https://github.com/wangxb29/DGPST)
**Area**: Image Style Transfer / Diffusion Models
**Keywords**: Portrait Style Transfer, Semantic Correspondence, Wavelet Transform, Diffusion Model, Cross-Domain Generalization

## TL;DR
DGPST proposes a diffusion-based portrait style transfer framework that establishes cross-domain dense semantic correspondences via a semantic adapter to warp the reference image, employs AdaIN-Wavelet Transform for latent space initialization to balance stylization and content preservation, and generates final results through a dual-conditional diffusion model combining ControlNet (high-frequency structural guidance) and a style adapter (style guidance). The model is trained solely on 30K real portrait photographs yet generalizes to diverse domains including photos, cartoons, sketches, and anime.

## Background & Motivation

**State of the Field**: Portrait style transfer requires precise local tone adjustment across facial semantic regions (skin, lips, eyes, hair, background) while preserving subject identity and facial structure. Existing approaches include traditional handcrafted methods (Shih et al.), GAN-based methods (StyleGAN series), and diffusion-based methods.

**Limitations of Prior Work**:
- **Traditional methods** (Shih, Chen et al.) rely on explicit semantic region alignment and are effective only when structural differences between input and reference are small; they cannot handle cross-domain scenarios (photo → cartoon).
- **GAN-based methods** (StyleGAN) inevitably alter subject identity.
- **Existing diffusion methods** (StyleID, IP-Adapter+ControlNet, InstantStyle+) primarily target artistic style transfer without considering semantic correspondence, resulting in poor semantic region alignment in portrait style transfer.
- General-purpose style transfer methods perform poorly on portrait tasks that require fine-grained semantic alignment.

**Root Cause**: Portrait style transfer simultaneously requires ① precise cross-domain semantic correspondence (eye-to-eye, lip-to-lip) and ② high-quality style transfer (tone, texture), yet existing methods fall short on one or both fronts.

**Paper Goals**: Construct a portrait style transfer framework that, trained only on real photographs, generalizes to arbitrary domains (cartoon, sketch, anime, vintage photos).

**Starting Point**: Leverage the inherently cross-domain semantic understanding embedded in the feature space of pretrained diffusion models (Stable Diffusion) to establish dense correspondences; exploit wavelet decomposition to separate high- and low-frequency components for balancing content and style.

**Core Idea**: Semantic correspondence via diffusion features → reference image warping → AdaIN-Wavelet latent space initialization → dual-conditional diffusion model generation.

## Method

### Overall Architecture
Given a content image $z_0^c$ and a style reference image $z_0^s$, the model outputs a portrait that preserves the content identity while applying the reference style. The pipeline consists of four steps: ① establish semantic correspondences using SD features and a semantic adapter, then warp the reference image; ② extract high-frequency information from the content image via ControlNet as structural guidance; ③ extract style guidance from the warped reference image via a style adapter; ④ initialize the latent space with AdaIN-Wavelet and perform conditional denoising generation.

### Key Designs

1. **Semantic-Aware Style Alignment**

   - **Function**: Establishes dense semantic correspondences between the content and reference portraits to produce a warped reference image $z_0^{s\_w}$.
   - **Mechanism**:
     - Image features are extracted via a CLIP image encoder and passed through a projection network into the SD U-Net via decoupled cross-attention.
     - Both images are fed into the SD U-Net (with semantic adapter features injected), and features $F_0^c, F_0^s \in \mathbb{R}^{HW \times C}$ are extracted from the third upsampling block.
     - A normalized correlation matrix $\mathcal{M}(i,j)$ is computed, and softmax-weighted warping is applied to the reference image: $z_0^{s\_w}(i) = \sum_j \text{softmax}(\mathcal{M}(i,j)/\tau) \cdot z_0^s(j)$.
   - **Training Loss**: mask warping loss $\mathcal{L}_{mask} = \|M^c - M^{s\_w}\|_1$ (semantic mask alignment) + cyclic warping consistency loss $\mathcal{L}_{cwc} = \mathcal{L}_{LPIPS}(z_0^s, z^{s'\_w})$ (cycle consistency).
   - **Design Motivation**: Raw SD features may yield incomplete semantic region correspondences; the semantic adapter together with the two loss terms constrains correspondence accuracy.

2. **Dual-Conditional Diffusion Model**

   - **Function**: Simultaneously leverages structural and style guidance to generate high-quality portraits.
   - **Structural Guidance (ControlNet)**: Haar discrete wavelet transform (DWT) is applied to the content image $z_0^c$, and the three high-frequency subbands (LH, HL, HH) are used as ControlNet input. Using only high-frequency information (edges/textures) rather than the original image provides style-agnostic structural guidance.
   - **Style Guidance (Style Adapter)**: Following the IP-Adapter architecture, CLIP image features are extracted from the warped reference image, projected, and injected via decoupled cross-attention: $Z^{new} = \text{softmax}(\frac{QK^t}{\sqrt{d}})V^t + \lambda \cdot \text{softmax}(\frac{QK^i}{\sqrt{d}})V^i$.
   - **Design Motivation**: Using high-frequency components rather than the original image in ControlNet avoids transferring the content's color style to the output; the warped reference image, being semantically aligned, provides more precise style guidance than the raw reference.

3. **AdaIN-Wavelet Transform (Latent Space Initialization)**

   - **Function**: Constructs an initial latent space that retains content structural detail while enhancing style tone transfer.
   - **Mechanism**:
     - DDIM inversion of the warped reference image yields $z_T^{s\_w}$ (initializing directly from this enhances color transfer but loses content detail, causing blurriness).
     - AdaIN is first applied: $z_T^{cs'} = \sigma(z_T^{s\_w}) \cdot \frac{z_T^c - \mu(z_T^c)}{\sigma(z_T^c)} + \mu(z_T^{s\_w})$ (channel-wise mean/variance alignment to bring the content latent statistics closer to the style).
     - Wavelet fusion is then performed: the low-frequency component of $z_T^{s\_w}$ is combined with the high-frequency component of $z_T^{cs'}$, and IDWT is applied to synthesize the final initial latent $z_T^{cs}$.
   - A parameter $\gamma$ controls stylization strength via interpolation $z_T^{cs} = \gamma \cdot z_T^{cs} + (1-\gamma) \cdot z_T^c$, enabling continuous style intensity control.
   - **Design Motivation**: Initializing from the content latent preserves the original color tone (insufficient style transfer), while initializing from the reference latent loses detail (excessive blurring). AdaIN aligns statistics, and Wavelet fusion combines the strengths of high- and low-frequency components.

### Loss & Training
Two-stage training:
- **Stage 1** (500K iterations): Train the semantic adapter. Loss = $\mathcal{L}_{sem}$ (noise prediction) + $\mathcal{L}_{cwc}$ + $10 \times \mathcal{L}_{mask}$.
- **Stage 2** (300K iterations): Train ControlNet + style adapter. Loss = $\mathcal{L}_{rec}$ (conditional noise prediction), conditioned on high-frequency ControlNet and the style adapter. The same image is used as both content and style (self-reconstruction).

## Key Experimental Results

### Main Results — CelebAMask-HQ

| Method | Gram loss ↓ | LPIPS ↓ | ID ↓ |
|--------|------------|---------|------|
| Shih et al. | 0.376 | 0.187 | 0.093 |
| Wang et al. | 0.208 | 0.181 | 0.106 |
| IP-A + C.N. | 2.835 | 0.245 | 0.774 |
| StyleID | 0.505 | 0.198 | 0.222 |
| InstantStyle+ | 0.557 | 0.294 | 0.272 |
| **Ours** | **0.274** | **0.116** | **0.057** |

### Ablation Study

| Configuration | Gram loss ↓ | LPIPS ↓ | ID ↓ |
|---------------|------------|---------|------|
| Full model | 0.274 | **0.116** | **0.057** |
| w/o ControlNet | **0.236** | 0.333 | 0.450 |
| w/o style adapter | 0.548 | 0.145 | 0.086 |
| w/ Init AdaIN (no Wavelet) | 1.196 | 0.151 | 0.062 |

### Cross-Domain Mixed Dataset Results

| Method | Gram loss ↓ | LPIPS ↓ | ID ↓ |
|--------|------------|---------|------|
| Wang et al. | 1.488 | 0.119 | 0.096 |
| InstantStyle+ | 0.723 | 0.192 | 0.203 |
| **Ours** | **0.657** | **0.083** | **0.087** |

### Key Findings
- **ControlNet is critical for identity preservation**: Removing it causes ID to surge from 0.057 to 0.450, demonstrating that high-frequency structural guidance is key to maintaining facial identity.
- **The style adapter is essential for style transfer**: Removing it doubles the Gram loss from 0.274 to 0.548.
- **Wavelet fusion outperforms plain AdaIN**: Gram loss drops from 1.196 to 0.274, with LPIPS also improving, validating the effectiveness of high/low-frequency separation and fusion.
- Inference speed is only 6.97 seconds per image (512×512), significantly faster than Deng et al. (24.18s) and InstantStyle+ (67.4s).
- Training solely on CelebAMask-HQ (30K real photographs) generalizes to cartoons, sketches, anime, vintage photos, and more.

## Highlights & Insights
- **Using SD feature space for cross-domain semantic correspondence** is the core innovation: intermediate features of pretrained diffusion models inherently encode cross-domain semantic understanding (eyes in photos and eyes in cartoons are close in feature space), and fine-tuning with the semantic adapter enables high-quality dense correspondences.
- **High-frequency ControlNet input** (DWT subbands LH/HL/HH) elegantly achieves style-agnostic structural guidance, avoiding color/style leakage that would result from using the original image or Canny edges.
- **The $\gamma$ parameter provides continuous style strength control**, simultaneously modulating latent space initialization and style adapter feature blending to achieve intuitive style interpolation.

## Limitations & Future Work
- The implementation is based on SD 1.5; upgrading to SDXL or SD3 may further improve quality.
- Semantic correspondences may still fail under extreme pose differences.
- Training data is limited to portraits; other style transfer scenarios requiring semantic alignment (e.g., architecture, animals) remain unexplored.
- Two-stage training is relatively complex (800K iterations); whether joint training could simplify this warrants investigation.
- Regional control currently requires manually provided masks; automated semantic region selection could be improved.

## Related Work & Insights
- **vs Wang et al.**: The traditional method performs well within the same domain (Gram loss 0.208 on CelebAMask-HQ, slightly better), but cross-domain capability is poor (Gram loss 1.488 on the mixed dataset, far worse than the proposed 0.657).
- **vs IP-Adapter + ControlNet**: This general-purpose combination ignores semantic correspondence, yielding an extremely high ID loss of 0.774, underscoring the necessity of semantic alignment for portrait style transfer.
- **vs StyleID**: This training-free method injects style via self-attention feature injection but lacks semantic correspondence, leading to poor region alignment.
- **vs InstantStyle+**: Although it introduces some structural control, inference is extremely slow (67.4s) and performance is inferior to the proposed method.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Employing diffusion model features for semantic correspondence combined with wavelet-based latent space fusion is a novel combination.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multi-dataset and cross-domain evaluation, comprehensive ablation, inference efficiency comparison, regional control, and style interpolation.
- **Writing Quality**: ⭐⭐⭐⭐ — Method description is clear, illustrations are abundant, and each module is validated individually in ablation.
- **Value**: ⭐⭐⭐⭐ — Highly practical; training on small-scale real data generalizes across domains with fast inference.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] SA-LUT: Spatial Adaptive 4D Look-Up Table for Photorealistic Style Transfer](sa-lut_spatial_adaptive_4d_look-up_table_for_photorealistic_style_transfer.md)
- [\[ICCV 2025\] Balanced Image Stylization with Style Matching Score](balanced_image_stylization_with_style_matching_score.md)
- [\[ICCV 2025\] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)
- [\[CVPR 2026\] FG-Portrait: 3D Flow Guided Editable Portrait Animation](../../CVPR2026/image_generation/fg-portrait_3d_flow_guided_editable_portrait_animation.md)
- [\[ICCV 2025\] Structure-Guided Diffusion Models for High-Fidelity Portrait Shadow Removal](structure-guided_diffusion_models_for_high-fidelity_portrait_shadow_removal.md)

<!-- RELATED:END -->
