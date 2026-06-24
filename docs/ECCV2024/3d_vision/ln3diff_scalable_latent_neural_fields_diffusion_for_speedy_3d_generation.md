---
title: >-
  [Paper Note] LN3Diff: Scalable Latent Neural Fields Diffusion for Speedy 3D Generation
description: >-
  [ECCV 2024][3D Vision][3D Generation] Proposes the LN3Diff++ framework, which compresses multi-view images into a compact 3D latent space via a 3D-aware VAE, and trains diffusion models (U-Net or DiT) on this space to achieve high-quality, fast, and general conditional 3D generation, including text-to-3D and image-to-3D.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Generation"
  - "Latent Diffusion Models"
  - "Neural Fields"
  - "Tri-plane Representation"
  - "VAE"
date: 2026-05-08
content_hash: 692b79acaa6e6c53
---

# LN3Diff: Scalable Latent Neural Fields Diffusion for Speedy 3D Generation

**Conference**: ECCV 2024  
**arXiv**: [2403.12019](https://arxiv.org/abs/2403.12019)  
**Code**: [Project Page](https://nirvanalan.github.io/projects/ln3diff)  
**Area**: 3D Vision  
**Keywords**: 3D Generation, Latent Diffusion Models, Neural Fields, Tri-plane Representation, VAE

## TL;DR

Proposes the LN3Diff++ framework, which compresses multi-view images into a compact 3D latent space via a 3D-aware VAE, and trains diffusion models (U-Net or DiT) on this space to achieve high-quality, fast, and general conditional 3D generation, including text-to-3D and image-to-3D.

## Background & Motivation

**Background**: 2D diffusion models have surpassed GANs, but a unified 3D diffusion pipeline has not yet been established. Existing methods are divided into two main categories: 2D lifting (SDS/Zero-123) and feed-forward 3D diffusion.

**Limitations of Prior Work**: 
   - **Poor Scalability**: Existing methods use a shared low-capacity MLP decoder for per-instance optimization, requiring 50+ views, with computational cost growing linearly with the dataset size.
   - **Low Efficiency**: High-dimensional 3D latent spaces (e.g., 256×256×96) make diffusion training difficult; auto-decoding produces noisy latent spaces.
   - **Weak Generalization**: Most methods focus on single-class unconditional generation, neglecting cross-category conditional 3D generation.

**Key Challenge**: Need to simultaneously achieve a compact latent space (for efficient diffusion), high-quality 3D reconstruction (for preserving details), and general conditional generation (for cross-category generalization).

**Goal**: Design a 3D representation-agnostic pipeline that supports fast, high-quality, and general conditional 3D generation.

**Key Insight**: Leveraging the successful experience of 2D LDMs to construct a 3D-aware VAE that compresses images into a structured tri-plane latent space.

**Core Idea**: Train diffusion models on a KL-regularized compact tri-plane latent space, decoupling the 3D compression and generation into two stages.

## Method

### Overall Architecture

Two-stage training pipeline:
- **Stage 1 (3D Latent Compression)**: The convolutional encoder $\mathcal{E}_\phi$ encodes the input images into a KL-regularized tri-plane latent $z \in \mathbb{R}^{h \times w \times 3 \times c}$. The Transformer decoder $\mathcal{D}_T$ decodes the latent into high-capacity tri-planes, and the convolutional upsampler $\mathcal{D}_U$ outputs high-resolution tri-planes for volume rendering supervision.
- **Stage 2 (Latent Diffusion Learning)**: Train conditional diffusion models (U-Net or DiT architecture) on the compact latent space, supporting text/image conditions.

### Key Designs

1. **3D-Aware Transformer Decoder**: To facilitate 3D spatial information flow, two attention mechanisms are designed:

    - **Self-Plane Attention**: Computes self-attention within each plane. Feature aggregation is performed independently for each plane in $z \in \mathbb{R}^{l \times 3 \times c}$, resulting in low complexity.
    - **Cross-Plane Attention**: Flattens the three planes into a long sequence $l \times 3 \times c \to 3l \times c$ to perform global attention, where all tokens attend to each other.
    - The two types of attention are alternated. DiT blocks and AdaLN layers are used to inject latent conditions, which is more efficient than Rodin and supports parallel computation.

2. **Compact Tri-Plane Latent Space**: The encoder downsampling factor is $f=8$, outputting $z \in \mathbb{R}^{h \times w \times 3 \times c}$ (in tri-plane format), which is similar to traditional tri-planes but resides in a compact latent space. KL regularization $\mathcal{L}_{\text{KL}}$ ensures the latent space is structured, making it suitable for diffusion training. It requires only V=2 views (ShapeNet) for training, which is far fewer than the 50 views required by SSDNeRF.

3. **Flow Matching Diffusion Framework**: Upgraded from DDPM+U-Net to FM+DiT. The training objective is:

$$\mathcal{L}_{\text{FM}} = -\frac{1}{2}\mathbb{E}_{\mathcal{E}_\phi(I), \epsilon \sim \mathcal{N}(0,I), t}\left[w_t^{\text{FM}} \lambda_t' \|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2\right]$$

where $z_t = (1-t)x_0 + t\epsilon$ defines the straight path, and the network predicts the velocity $v_\Theta$.

4. **Multimodal Condition Injection**: 

    - Text condition: CLIP text encoder outputs $77 \times 768$ tokens, which are injected via cross-attention.
    - Image condition: CLIP image encoder + DINOv2 patch features. DINO features are prepended to self-attention (similar to SD-3) to provide low-level details for improving reconstruction fidelity.
    - Classifier-free guidance: Conditions are randomly dropped with a 15% probability, mixing conditional/unconditional scores during sampling.

### Loss & Training

**Total Loss of the VAE Stage**:

$$\mathcal{L}(\phi, \psi) = \mathcal{L}_{\text{render}} + \lambda_{\text{geo}}\mathcal{L}_{\text{geo}} + \lambda_{\text{kl}}\mathcal{L}_{\text{KL}} + \lambda_{\text{GAN}}\mathcal{L}_{\text{GAN}}$$

- $\mathcal{L}_{\text{render}}$: L1 + perceptual loss, supervising both the input views and randomly sampled novel views.
- $\mathcal{L}_{\text{GAN}}$: Uses a DINOv2 vision-aided GAN, including an input view discriminator and a novel view discriminator.
- **Flexicubes Fine-tuning**: $\mathcal{L}_{\text{flex}} = \lambda_{\text{normal}}\mathcal{L}_{\text{normal}} + \lambda_{\text{reg}}\mathcal{L}_{\text{reg}}$, fine-tuning only the decoder to switch from NeRF to SDF to support high-quality mesh extraction.

**Training Configuration**: BFloat16 + FlashAttention, DiT-L (24 layers, 16 heads, 1024 dimensions), totaling 800K iterations, trained on 8x A100 for about 7 days.

## Key Experimental Results

### Main Results - Unconditional Generation on ShapeNet

| Category | Method | FID↓ | KID(%)↓ | COV(%)↑ | MMD(‰)↓ |
|------|------|------|---------|---------|---------|
| Car | EG3D | 33.33 | 1.4 | 35.32 | 3.95 |
| Car | SSDNeRF(V=3) | 47.72 | 2.8 | 37.84 | 3.46 |
| Car | **LN3Diff++** | **17.6** | **0.49** | 43.12 | **2.32** |
| Plane | EG3D | 14.47 | 0.54 | 18.12 | 4.50 |
| Plane | **LN3Diff++** | **8.84** | **0.36** | **43.40** | **2.71** |
| Chair | EG3D | 26.09 | 1.1 | 19.17 | 10.31 |
| Chair | **LN3Diff++** | **16.9** | **0.47** | 47.1 | **5.28** |

### Image-conditioned 3D Generation (Objaverse)

| Method | CLIP-I↑ | FID↓ | KID(%)↓ | COV(%)↑ | MMD(‰)↓ |
|------|---------|------|---------|---------|---------|
| OpenLRM | 86.37 | 38.41 | 1.87 | 39.33 | 29.08 |
| LGM (V=4) | 87.99 | **19.93** | **0.55** | 50.83 | 22.06 |
| **LN3Diff++** | **88.29** | 23.01 | 0.75 | **55.17** | **19.94** |

### Ablation Study - VAE Architecture Design

| Design | PSNR@100K |
|------|-----------|
| 2D Conv Baseline | 17.46 |
| + ViT Block | 18.92 |
| ViT → DiT Block | 20.61 |
| + Plücker Embedding | 21.29 |
| + Cross-Plane Attention | 21.70 |
| + Self-Plane Attention | **21.95** |

### Key Findings

- LN3Diff++ outperforms all baselines across all three categories on ShapeNet using only V=2 views (FID/KID/MMD).
- GAN methods suffer severely from mode collapse (e.g., EG3D/GET3D generate only white airliners for the Plane category).
- Diffusion sampling speed is 5.7s per instance (V100), which is nearly 3x faster than RenderDiffusion (15.8s).
- The novel view discriminator is crucial for monocular datasets (FFHQ); without it, reasonable novel views cannot be generated.
- DINO features significantly improve the fidelity of image-conditioned generation; using only CLIP leads to unfaithful generation.

## Highlights & Insights

- **3D Representation Agnostic**: The pipeline design is decoupled from specific 3D representations (NeRF/3DGS/SDF), allowing new rendering techniques to be directly plugged in.
- **Amortized Encoding**: The pretrained encoder can encode new data on-the-fly without per-instance optimization, which resolves the scalability bottleneck of existing methods.
- **Unified Framework**: The same framework supports unconditional, text-conditioned, and image-conditioned generation, showing competitiveness across ShapeNet, FFHQ, and Objaverse datasets.
- **FM+DiT Upgrade**: Upgrading from DDPM+U-Net to Flow Matching + DiT brings double improvements in quality and efficiency, aligning with current research trends in video generation.

## Limitations & Future Work

- Volume rendering remains memory-intensive, and the visual quality of the Flexicubes fine-tuned version is lower than that of the NeRF version.
- The tri-plane latent space might not be the optimal choice; point clouds or sparse voxels could be better.
- Unnatural artifacts appear in background areas when trained on monocular datasets (adversarial shortcuts of the novel view discriminator).
- Jointly modeling geometry and texture distributions can produce suboptimal results; a decoupled framework might perform better.
- Trained only on artist-created data from Objaverse; incorporating real-world data could improve generalization.

## Related Work & Insights

- **EG3D** [CVPR 2022]: Pioneer of tri-plane representation $\to$ this work continues this representation at the latent space level.
- **SSDNeRF** [NeurIPS 2023]: Joint reconstruction and diffusion training $\to$ requires 50 views and complex scheduling.
- **RenderDiffusion** [CVPR 2023]: Diffusion without latent 3D space $\to$ volume rendering at each step severely slows down sampling.
- **SD-3** [2024]: Flow Matching + DiT + prepend conditioning $\to$ core reference for the 3D diffusion part of this work.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to demonstrate that a 3D-aware VAE latent space can efficiently support 3D diffusion learning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers ShapeNet, FFHQ, and Objaverse datasets with thorough ablation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear methodology, but the journal version contains a lot of content, some of which is redundant.
- **Value**: ⭐⭐⭐⭐ Provides a scalable paradigm for native 3D diffusion models, with significant impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Compress3D: a Compressed Latent Space for 3D Generation from a Single Image](compress3d_a_compressed_latent_space_for_3d_generation_from_a_single_image.md)
- [\[ECCV 2024\] MALD-NeRF: Taming Latent Diffusion Model for Neural Radiance Field Inpainting](taming_latent_diffusion_model_for_neural_radiance_field_inpainting.md)
- [\[ECCV 2024\] MeshFeat: Multi-Resolution Features for Neural Fields on Meshes](meshfeat_multi-resolution_features_for_neural_fields_on_meshes.md)
- [\[ECCV 2024\] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)
- [\[ECCV 2024\] IDOL: Unified Dual-Modal Latent Diffusion for Human-Centric Joint Video-Depth Generation](idol_unified_dual-modal_latent_diffusion_for_human-centric_joint_video-depth_gen.md)

</div>

<!-- RELATED:END -->
