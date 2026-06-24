---
title: >-
  [Paper Note] Synthetic Prior for Few-Shot Drivable Head Avatar Inversion
description: >-
  [CVPR 2025][3D Vision][Head Avatar] SynShot proposes training a generative 3D Gaussian prior model using large-scale synthetic head data, allowing high-fidelity, drivable head avatars to be inverted via pivotal fine-tuning using only 3 real images, significantly outperforming monocular and GAN-based methods.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Head Avatar"
  - "Few-Shot Inversion"
  - "Synthetic Data Prior"
  - "3DGS"
  - "VQ-VAE"
date: 2026-05-08
content_hash: 395f5c9ed800ac8b
---

# Synthetic Prior for Few-Shot Drivable Head Avatar Inversion

**Conference**: CVPR 2025  
**arXiv**: [2501.06903](https://arxiv.org/abs/2501.06903)  
**Code**: [Project Page](https://zielon.github.io/synshot/)  
**Area**: 3D Vision / Head Avatar  
**Keywords**: Head Avatar, Few-Shot Inversion, Synthetic Data Prior, 3DGS, VQ-VAE

## TL;DR

SynShot proposes training a generative 3D Gaussian prior model using large-scale synthetic head data, allowing high-fidelity, drivable head avatars to be inverted via pivotal fine-tuning using only 3 real images, significantly outperforming monocular and GAN-based methods.

## Background & Motivation

High-fidelity, drivable digital head avatars are a key technology for VR/MR. Existing methods face multiple challenges:
- **Monocular methods** (INSTA, Flash Avatar, Splatting Avatar) require thousands of video frames for training and fail to generalize well to novel views and expressions.
- **GAN inversion methods** (Next3D, InvertAvatar, Portrait4D) rely on real datasets like FFHQ and are prone to artifacts such as identity shifting during novel view synthesis.
- **Multi-view prior methods** (GPHM, HeadGAP) require expensive multi-view capture hardware, and real-world data is strictly restricted by privacy regulations such as GDPR.
- High maintenance cost of real data: GDPR requires periodic deletion of models and data to accommodate participants withdrawing their consent.
- The expressiveness of prior models is limited by the diversity of training data (ethnicity, age, expression, etc.), camera setups, and data preprocessing quality.

The core idea of SynShot is to **train the prior entirely on synthetic data**, bypassing the legal and financial costs of real data, while bridging the domain gap through a carefully designed inversion pipeline.

## Method

### Overall Architecture

SynShot adopts a two-stage approach: (1) training a VQ-VAE-based generative 3D Gaussian head prior on a large-scale synthetic head dataset (approx. 2,000 identities, 14 million images); (2) inverting a personalized head avatar given a few real images using a pivotal tuning strategy. The prior model uses a convolutional encoder-decoder to output Gaussian parameters in the UV texture space, and handles the modeling complexity differences across different head regions (face vs. hair) through a part-based densification mechanism.

### Key Design 1: Dual-Branch VQ-VAE Identity-Expression Disentanglement

**Function**: Explicitly disentangles static identity (facial shape, appearance) from dynamic expressions (wrinkles, self-shadowing).

**Mechanism**: The encoder is divided into two parallel branches: the identity encoder $E_{\text{id}}(\mathbf{x}_{\text{tex}}, \mathbf{x}_{\text{verts}}) \to \mathbf{z}_{\text{id}}$ and the expression encoder $E_{\text{expr}}(\mathbf{x}_{\text{exp}}) \to \mathbf{z}_{\text{expr}}$. Both latent spaces undergo vector quantization $\mathbf{q}(\cdot)$ before being fed into three decoder branches: the feature map decoder $D_{\text{feat}}$ fuses identity and expression information, the identity decoder $D_{\text{id}}$ predicts texture and vertex positions solely from the identity code, and the expression decoder $D_{\text{expr}}$ predicts expression offsets solely from the expression code.

**Design Motivation**: Explicit disentanglement allows freezing the expression encoder and optimizing only the identity encoder to match the real images during inversion. This avoids the entanglement of identity and expression information, improving generalization.

### Key Design 2: Part-Based Gaussian Primitive Densification

**Function**: Adaptively adjusts the density of Gaussian primitives based on the modeling requirements of different head regions (face, hair).

**Mechanism**: Differing from directly outputting fixed-resolution Gaussian parameter maps from a CNN, SynShot performs region-wise sampling of the decoder outputs in the UV space via bilinear sampling $\mathcal{B}(\cdot, u, v)$. The face and scalp regions utilize distinct UV sampling grids $(u_r, v_r)$. Each region independently computes Gaussian positions $\phi_r$, initial scales $\sigma_r$ (based on nearest-neighbor distance), and rotations $\theta_r$ (based on the tangent-binormal-normal coordinate system derived from the position map gradient). Afterward, lightweight regressors $R_{\text{color}}$ and $R_{\text{gauss}}$ predict the spherical harmonic coefficients and parameter correction fields.

**Design Motivation**: Hair requires a higher density of Gaussian primitives than skin to model fine details. Part-based sampling acts as an adaptive densification that eliminates the bottleneck of fixed-resolution CNNs (ablation studies show a significant decrease in quality if sampling is omitted).

### Key Design 3: Synthetic-to-Real Pivotal Tuning Inversion

**Function**: Adapts the synthetic prior to a personalized digital avatar of a real subject using a few (minimum of 3) real images.

**Mechanism**: Two-stage optimization—In the first stage, other parts of the network are frozen, and only the identity encoder $E_{\text{id}}$ is optimized to recover the identity latent code $\mathbf{z}_{\text{id}}$. In the second stage, $\mathbf{z}_{\text{id}}$ is frozen, and the decoder and regressors are fine-tuned to bridge the synthetic-to-real domain gap. The loss functions include photometric loss $\mathcal{L}_{\text{color}} = \alpha\mathcal{L}_{L1} + \beta\mathcal{L}_{\text{SSIM}} + \gamma\mathcal{L}_{\text{LPIPS}}$, as well as ArcFace-based identity loss $\mathcal{L}_{\text{id}}$ and feature matching loss $\mathcal{L}_{\text{arc}}$.

**Design Motivation**: Inspired by the PTI strategy in GAN inversion, finding the optimal latent code first and then fine-tuning the generator weights achieves high-fidelity reconstruction while maintaining the generalization ability of the prior.

### Loss & Training

Total training loss: $\mathcal{L} = \mathcal{L}_{\text{color}} + \mathcal{L}_{\text{geom}} + \mathcal{L}_{\text{reg}}$, where $\mathcal{L}_{\text{color}} = \alpha\mathcal{L}_{L1} + \beta\mathcal{L}_{\text{SSIM}} + \gamma\mathcal{L}_{\text{LPIPS}}$, and $\mathcal{L}_{\text{geom}} = \delta\mathcal{L}_{L1}$ supervises the reconstruction of position maps and expression maps, while $\mathcal{L}_{\text{reg}}$ is the $L_2$ regularization. The inversion loss additionally incorporates the ArcFace perceptual loss.

## Key Experimental Results

### Main Results: Quantitative Comparison on Self-Reenactment

| Method | Training Data | LPIPS ↓ | Remarks |
|------|-----------|---------|------|
| SynShot (Ours) | 3 images | **0.0236** | Only 3 images |
| InvertAvatar | 3 images | 0.0962 | GAN inversion |
| Portrait4D | 1 image | 0.0843 | Single-image inversion |
| Next3D | 1 image | 0.2274 | GAN inversion |
| INSTA | ~3000 frames | Higher | Monocular method |
| Flash Avatar | ~3000 frames | Higher | Monocular method |

### Ablation Study: VQ-VAE Architecture

| Configuration | L1 ↓ | LPIPS ↓ | SSIM ↑ | PSNR ↑ |
|------|------|---------|--------|--------|
| F=128 (Final Model) | 0.0356 | **0.2686** | 0.8189 | 20.15 |
| No Sampling | 0.0403 | 0.2853 | 0.8158 | 19.98 |
| w/o VQ | 0.0396 | 0.2747 | 0.8122 | 19.29 |
| Single Layer | 0.0369 | 0.2702 | 0.8177 | 19.89 |
| F=32 | 0.0375 | 0.2732 | 0.8146 | 19.70 |

### Key Findings

- SynShot **significantly outperforms** monocular methods trained on 3000 frames in terms of LPIPS, using only 3 images.
- Both part-based sampling and VQ quantization contribute significantly to quality improvement.
- In cross-reenactment evaluations, monocular methods fail catastrophically under out-of-distribution expressions and poses, whereas SynShot performs robustly due to its strong prior.

## Highlights & Insights

- **Pure synthetic data solution**: Bypasses real-data privacy concerns entirely, enabling free experimentation even under strict regulations like GDPR.
- **Extremely few-shot with high quality**: Using only 3 images outperforms SOTA trained on 3000 frames, demonstrating that a strong prior is far more critical than data quantity.
- **Cross-reenactment reveals generalization**: The paper emphasizes the importance of evaluating cross-reenactment rather than self-reenactment alone, which better exposes generalization weaknesses.

## Limitations & Future Work

- All identities in the synthetic dataset share the same teeth geometry and texture, which limits reconstructed teeth details after inversion.
- Lack of diverse expression-dependent wrinkle data affects the overall visual quality.
- Using a single environmental lighting for ray-traced rendering limits generalization to diverse lighting conditions.
- Future directions: Enhancing synthetic data diversity and incorporating richer lighting conditions and expression textures.

## Related Work & Insights

- Unlike MLP direct embedding methods such as HeadGAP and GPHMv2, SynShot learns parameter distributions via VQ-VAE and does not require guiding meshes at test time.
- The PTI (Pivotal Tuning Inversion) strategy, proven effective in GAN inversion, is successfully extended to 3DGS head prior inversion in this work.
- The success of synthetic data in face-related tasks further validates its feasibility in the 3D avatar domain.

## Rating

⭐⭐⭐⭐ — Training a prior using synthetic data to achieve few-shot, high-fidelity avatar inversion is a practical and significant contribution. The method is comprehensively designed with thorough experiments, though the limitations of synthetic data quality cap further improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HRAvatar: High-Quality and Relightable Gaussian Head Avatar](hravatar_high-quality_and_relightable_gaussian_head_avatar.md)
- [\[CVPR 2025\] Vid2Avatar-Pro: Authentic Avatar from Videos in the Wild via Universal Prior](vid2avatar-pro_authentic_avatar_from_videos_in_the_wild_via_universal_prior.md)
- [\[CVPR 2025\] SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation](scope_scene-contextualized_incremental_few-shot_3d_segmentation.md)
- [\[CVPR 2025\] FFaceNeRF: Few-Shot Face Editing in Neural Radiance Fields](ffacenerf_few-shot_face_editing_in_neural_radiance_fields.md)
- [\[CVPR 2026\] OMG-Avatar: One-shot Multi-LOD Gaussian Head Avatar](../../CVPR2026/3d_vision/omg-avatar_one-shot_multi-lod_gaussian_head_avatar.md)

</div>

<!-- RELATED:END -->
