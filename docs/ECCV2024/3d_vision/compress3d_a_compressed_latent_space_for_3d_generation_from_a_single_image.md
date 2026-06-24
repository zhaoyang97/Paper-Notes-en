---
title: >-
  [Paper Note] Compress3D: a Compressed Latent Space for 3D Generation from a Single Image
description: >-
  [ECCV 2024][3D Vision][3D Generation] This paper proposes a highly compressed triplane latent space autoencoder, paired with a two-stage diffusion model (generating a shape embedding first, followed by a triplane latent). It generates high-quality 3D assets from a single image in just 7 seconds, utilizing significantly less training data and time than comparable methods.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Generation"
  - "Diffusion Models"
  - "Triplane"
  - "Latent Space Compression"
  - "Single-Image 3D Generation"
date: 2026-05-08
content_hash: 7bd23e69a3b2cf8a
---

# Compress3D: a Compressed Latent Space for 3D Generation from a Single Image

**Conference**: ECCV 2024  
**arXiv**: [2403.13524](https://arxiv.org/abs/2403.13524)  
**Code**: Available ([https://compress3d.github.io/](https://compress3d.github.io/))  
**Area**: 3D Vision  
**Keywords**: 3D Generation, Diffusion Models, Triplane, Latent Space Compression, Single-Image 3D Generation

## TL;DR

This paper proposes a highly compressed triplane latent space autoencoder, paired with a two-stage diffusion model (generating a shape embedding first, followed by a triplane latent). It generates high-quality 3D assets from a single image in just 7 seconds, utilizing significantly less training data and time than comparable methods.

## Background & Motivation

Efficiently generating high-quality 3D assets from a single image is an important yet challenging task. Existing methods exhibit the following issues:

**Optimization-based methods** (e.g., DreamFusion, Magic3D) rely on SDS optimization, taking minutes to hours per asset.

**Learning-based methods** (e.g., Shap-E, 3DGen) train diffusion models in latent spaces but do not achieve high compression, limiting training and generation efficiency.

**Reconstruction-based methods** (e.g., LRM, Zero-1-to-3) require multi-view images or additional reconstruction steps.

4. Existing methods only use image embeddings as conditioning, lacking 3D geometric information and resulting in low-quality generated meshes.

Core Motivation: **To design a highly compressed triplane latent space** and simultaneously leverage both image embeddings and shape embeddings as conditioning to improve generation quality.

## Method

### Overall Architecture

Compress3D consists of three core components trained in three stages:

1. **Triplane AutoEncoder**: Encodes a colored point cloud into a low-dimensional triplane latent space, then decodes it back into a high-quality 3D model.
2. **Diffusion Prior Model**: Generates a shape embedding conditioned on the image embedding.
3. **Triplane Diffusion Model**: Generates a triplane latent conditioned on both the image embedding and shape embedding.

Inference pipeline: Input a single image $\rightarrow$ Extract image embedding via CLIP $\rightarrow$ Generate shape embedding via the Prior model $\rightarrow$ Generate triplane latent via the Triplane Diffusion model $\rightarrow$ Decode into a textured 3D model.

### Key Designs

#### 1. Learnable Projection for Triplane Encoder

Prior methods directly project 3D point features onto the triplane using average pooling, which lacks learnable parameters and leads to information loss. The improvements in Compress3D include:

- Extracts 3D point features $F$ using PointNet.
- Builds a 3D feature volume $V \in \mathbb{R}^{r \times r \times r \times c}$ via weighted projection, where weights are inversely proportional to the distance from points to grid voxels.
- Normalizes the feature volume to eliminate the effect of uneven point cloud density.
- Utilizes **3D convolutions** in three directions to obtain high-resolution triplane features, e.g., $T_{xy} = \text{3DConv}(V^n, k=(1,1,r), s=(1,1,r))$.
- Obtains a low-resolution triplane latent ($32 \times 32$, with 32 channels) via ResBlocks and downsampling.

#### 2. 3D-aware Cross-Attention

To enhance the representation capability of the low-resolution triplane latent:

- Downsamples the feature volume to a lower resolution $V_d^n$ (resolution 32).
- Uses the triplane latent as the query, and the 3D feature volume as the key/value.
- Each triplane position $(i,j)$ query only targets its corresponding **local cubic region**, ensuring computational efficiency.
- Generates residual features and adds them to the original latent: $T^e = A + T^l$.
- Querying the low-resolution feature volume reduces the training step time from 2.295s to 0.824s, while slightly improving performance.

#### 3. Triplane Decoder and FlexiCubes

- Decodes into a $128 \times 128$ high-resolution triplane via ResBlocks and upsampling.
- Adopts **FlexiCubes** representation, predicting weights, SDF, and vertex deformations for each cube.
- Extracts the mesh using Dual Marching Cubes, and predicts surface colors with an MLP.
- Trains end-to-end via a differentiable renderer, without pre-computing SDF.

#### 4. Diffusion Prior Model

- Extracts shape embeddings $e_s \in \mathbb{R}^{1280}$ and image embeddings $e_i \in \mathbb{R}^{1280}$ using OpenShape.
- Uses an MLP with skip connections as the diffusion backbone (25.8M parameters).
- Directly predicts the denoised $e_s$ using L1 loss: $L_{prior} = \mathbb{E}[\|f_\theta^p(e_s^{(t)}, t, e_i) - e_s\|]$.

#### 5. Triplane Diffusion Model

- UNet backbone (864M parameters) with 3D-aware convolution.
- Shape and image embeddings are injected via cross-attention.
- Classifier-free guidance: 5% dropout for each, using dual guidance during inference.

### Loss & Training

**AutoEncoder Rendering Loss**:

$$L_R = \lambda_1 L_{rgb} + \lambda_2 L_{mask} + \lambda_3 L_{depth} - \lambda_{kl} D_{KL}(N(\mu,\sigma) \| N(0,1))$$

- $\lambda_1{=}10, \lambda_2{=}10, \lambda_3{=}0.1, \lambda_{kl}{=}1e{-}6$
- Supervised by rendering $512 \times 512$ images from 40 random viewpoints.

**Dataset**: Objaverse is filtered using an MLP classifier trained on 2,500 manually labeled good/bad models, obtaining **100K** high-quality models.

**Training Configuration**: AutoEncoder: 8 $\times$ A100 for 6 days | Prior: 2 $\times$ A100 for 18 hours | Diffusion: 8 $\times$ A100 for 4 days.

## Key Experimental Results

### Main Results

| Metric | Shap-E | OpenLRM | **Compress3D** |
|---|---|---|---|
| FID (↓) | 146.14 | 94.47 | **53.21** |
| CLIP Similarity (↑) | 0.731 | 0.756 | **0.776** |
| Latent Space Dimension (↓) | 1.05M | 0.98M | **0.10M** |
| Seconds per Shape | 11 | 5 | 7 |
| Training Dataset Size | $\ge$ 1M | 0.951M | **0.095M** |
| Training GPU Hours | - | 9200 | **1900** |

| 3D-aware Cross-Attention Ablation | $L_{rgb}{\times}10^3$↓ | $L_{mask}{\times}10^3$↓ | $L_{depth}{\times}10^2$↓ | Seconds/Step |
|---|---|---|---|---|
| w/o attention | 3.798 | 6.953 | 2.637 | 0.789 |
| **w/ attention** | **2.485** | **5.059** | **2.095** | 0.824 |

### Ablation Study

**Prior Model**: FID improves from 66.46 to 53.21, and CLIP Sim improves from 0.745 to 0.776, showing particularly significant gains under unconventional viewpoints.

**Guidance Scale**: The optimal setting is $s_p{=}5.0, s_s{=}1.0$, yielding FID=53.21; overly large or small guidance scales reduce performance.

**Feature Volume Resolution**: Adjusting resolution from 128 to 64 then 32 results in a slight increase in reconstruction quality, while training time per step reduces from 2.295s to 0.961s and then 0.824s.

### Key Findings

1. Compressing the latent space down to **0.10M** (1/10 of Shap-E) surprisingly improves generation quality.
2. Utilizing only 95K training models, the FID drops by nearly 100 points.
3. Training requires only 1,900 GPU hours (1/5 of OpenLRM).
4. 3D-aware cross-attention adds only 0.035s per step but significantly reduces reconstruction loss.

## Highlights & Insights

1. **Extremely High Compression**: Compressing the latent space from million-scale to 100K-scale demonstrates that proper architectural design is more vital than brute-force model scaling.
2. **Shape Embedding Bridge**: The two-stage image $\rightarrow$ shape $\rightarrow$ triplane pipeline outperforms the direct image $\rightarrow$ triplane approach, as shape embeddings contain richer 3D geometric information.
3. **Local 3D Cross-Attention**: The triplane queries its corresponding local cube instead of calculating global attention, balancing representational capacity with computational efficiency.
4. **Data Quality > Quantity**: Only 2,500 manual annotations were used to train a classifier to filter Objaverse, yielding 100K high-quality data samples.

## Limitations & Future Work

1. Reliance on pre-trained CLIP/OpenShape models may limit effectiveness for objects outside the training distribution.
2. The grid resolution of FlexiCubes (90) limits the capture of extremely fine geometric details.
3. The accuracy of shape embedding prediction has an upper bound, which may still fail under extreme viewpoints.
4. Trained only on 100K samples; scaling up with more high-quality datasets could yield further improvements.

## Related Work & Insights

- **Shap-E**: Transformer encoder encodes directly into the implicit function parameter space, resulting in an excessively large latent space (1.05M).
- **3DGen**: Uses a UNet to refine the triplane but suffers from heavy computation; this work instead integrates learnable parameters into the projection.
- **OpenShape**: Aligned shape-text-image embeddings are elegantly leveraged as conditioning inputs for generation.
- Insight: **Two-stage conditional generation** serves as a general strategy to enhance quality.

## Rating

| Dimension | Score (1-10) |
|---|---|
| Novelty | 7 |
| Technical Depth | 8 |
| Experimental Thoroughness | 8 |
| Writing Quality | 7 |
| Practical Value | 8 |
| **Total Score** | **7.6** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] CRM: Single Image to 3D Textured Mesh with Convolutional Reconstruction Model](crm_single_image_to_3d_textured_mesh_with_convolutional_reconstruction_model.md)
- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)
- [\[ECCV 2024\] LN3Diff: Scalable Latent Neural Fields Diffusion for Speedy 3D Generation](ln3diff_scalable_latent_neural_fields_diffusion_for_speedy_3d_generation.md)
- [\[ECCV 2024\] ZeST: Zero-Shot Material Transfer from a Single Image](zest_zero-shot_material_transfer_from_a_single_image.md)
- [\[ICCV 2025\] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion](../../ICCV2025/3d_vision/sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)

</div>

<!-- RELATED:END -->
