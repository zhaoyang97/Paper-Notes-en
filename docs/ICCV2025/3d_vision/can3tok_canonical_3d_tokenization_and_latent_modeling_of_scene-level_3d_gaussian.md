---
title: >-
  [Paper Note] Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes Can3Tok, the first variational autoencoder capable of encoding scene-level 3DGS into a low-dimensional latent space. It achieves efficient tokenization via cross-attention with canonical queries, and addresses scale inconsistency through 3DGS normalization and semantic-aware filtering, successfully generalizing to novel scenes on DL3DV-10K.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D Gaussian Splatting
  - Variational Autoencoder
  - Scene-Level 3D Generation
  - Canonical Tokens
  - Latent Space Modeling
date: 2026-05-08
content_hash: 62cac3757d51641e
---

# Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians

**Conference**: ICCV 2025  
**arXiv**: [2508.01464](https://arxiv.org/abs/2508.01464)  
**Code**: [https://github.com/Zerg-Overmind/Can3Tok](https://github.com/Zerg-Overmind/Can3Tok)  
**Area**: 3D Vision / 3D Generation  
**Keywords**: 3D Gaussian Splatting, Variational Autoencoder, Scene-Level 3D Generation, Canonical Tokens, Latent Space Modeling

## TL;DR

This paper proposes Can3Tok, the first variational autoencoder capable of encoding scene-level 3DGS into a low-dimensional latent space. It achieves efficient tokenization via cross-attention with canonical queries, and addresses scale inconsistency through 3DGS normalization and semantic-aware filtering, successfully generalizing to novel scenes on DL3DV-10K.

## Background & Motivation

Significant progress has been made in 3D generation, but efforts have largely focused on the object level — NeRF/3DGS-based methods can generate individual objects with high quality. Scene-level 3D generation, however, faces fundamental challenges:

**Incompatibility of 3DGS data structure with existing VAEs**: 3DGS is inherently highly unstructured — it contains heterogeneous attributes (geometry, appearance, lighting) and is irregular like point clouds. Scene-level 3DGS contains a large number of Gaussian primitives (>10K), making compression into low-dimensional embeddings extremely difficult.

**Scale inconsistency**: Caused by COLMAP SfM initialization, both the global scale and per-Gaussian scaling values vary across scenes, preventing direct use in large-scale training.

**Noisy artifacts**: Due to insufficient observation coverage, scene-level 3DGS reconstructions frequently contain large numbers of floaters.

Experiments demonstrate that existing 3D VAE methods (PointNet VAE, L3DG, etc.) completely fail to converge on scene-level 3DGS data — even with hundreds of scenes — let alone generalize to novel ones.

## Method

### Overall Architecture

Can3Tok is a Transformer-based VAE consisting of an encoder and a decoder. The encoder compresses a large number of 3DGS primitives (40K) into a low-dimensional latent space via cross-attention; the decoder reconstructs the original 3DGS parameters from the latent. The overall pipeline: input $\mathcal{G} \in \mathbb{R}^{N \times (2L_B + C)}$ → cross-attention → self-attention ×8 → latent space ($64 \times 64 \times 4$) → self-attention ×16 → MLP → output 3DGS.

### Key Designs

1. **Canonical Query Cross-Attention Tokenization**:

    - The input 3DGS has $N=40\text{K}$ primitives; direct self-attention is computationally prohibitive.
    - Learnable queries $query \in \mathbb{R}^{M \times (P+Q)}$ ($M=256$) perform cross-attention to compress $N$ inputs into $M$ tokens.
    - Key innovation: queries are initialized with regular voxel grid coordinates (canonical space), introducing structured geometric priors.
    - Input features include Fourier positional encodings $\gamma(\mathbf{x})$ and nearest voxel coordinate encodings $\gamma(\mathbf{v})$.
    - Voxel coordinates serve as "anchor" positional information, reducing the learning burden on the encoder.

2. **3DGS Normalization**: The core solution to scale inconsistency.

    - Translation: shift the 3D scene center to the origin $translate = -\frac{1}{n}\sum_{i=1}^n \mathbf{x}_i$
    - Scaling: constrain all Gaussians within a sphere of radius $r$: $scale = \frac{r}{\max|\mathbf{x}+translate|_2 \times 1.1}$
    - Synchronously scale each Gaussian's scaling parameter: $\hat{\mathbf{s}} = \mathbf{s} \times scale$
    - Adjust camera positions accordingly: $\hat{T}_i = (T_i + translate) \times scale$
    - Other attributes (rotation, opacity, color, SH) remain unchanged.
    - An additional benefit: real-world scale can be recovered via monocular depth estimation.

3. **Semantic-aware Filtering**:

    - LangSam (a text-guided SAM variant) detects the "most salient region" in the middle frame of the scene.
    - A Gaussian within the segmentation mask is selected as a seed and iteratively expanded via K-NN to a preset count of $N=40\text{K}$.
    - This removes floaters and non-salient regions, retaining the cleanest and most semantically meaningful 3DGS subset.
    - Experiments show that without filtering, high-frequency details are severely degraded.

### Loss & Training

$$\mathcal{L} = \text{Dist}(GS_{output}, GS_{input}) + \lambda \mathcal{L}_{KL}(\mathbf{z}, \mathcal{N}(\mathbf{0}, \mathbf{I}))$$

- $\text{Dist}$: L2 distance over all 3DGS feature channels.
- $\lambda = 1 \times 10^{-6}$: the KL divergence weight is very small, prioritizing reconstruction quality.
- Data augmentation: random SO(3) rotations applied to input 3DGS.
- Training details: 8× A100 GPUs, 5 days; inference (encode + decode) takes only ~0.06s.

Architecture: encoder — 1 linear layer + 1 cross-attention layer + 8 self-attention layers + 2 projection layers; decoder — 1 linear layer + 16 self-attention layers + 3 MLP layers. Attention uses Flash-Attention with 12 heads × 64 dimensions. Latent space $\mathbf{z} \in \mathbb{R}^{64 \times 64 \times 4}$, identical in size to the Stable Diffusion latent space.

## Key Experimental Results

### Main Results

Quantitative comparison on the DL3DV-10K test set:

| Method | L2 Error↓ | Failure Rate↓ |
|--------|-----------|---------------|
| L3DG (3DGS encoder, convolutional) | 1200.4 | 100% |
| PointNet VAE | 1823.0 | 100% |
| PointTransformer | 230.7 | 70% |
| **Can3Tok (Ours)** | **30.1** | **2.5%** |

Failure rate is defined as the proportion of scenes with reconstruction L2 error exceeding 1000. All baseline methods nearly completely fail; only Can3Tok successfully generalizes. PointNet and L3DG fail to converge even on training sets of more than 500 scenes.

### Ablation Study

| Configuration | L2 Error↓ | Failure Rate↓ |
|---------------|-----------|---------------|
| w/o Learnable Query | $10^{25}$ | 100% |
| w/o Normalization | 1889.7 | 100% |
| w/o Voxel Appending | 50.5 | 4.3% |
| w/o Data Filtering | 73.3 | 6.1% |
| w/o Data Augmentation | 53.3 | 4.6% |
| **Full (Ours)** | **30.1** | **2.5%** |

### Key Findings

- **Normalization is a prerequisite**: Without normalization, even Can3Tok fails to generalize entirely (100% failure rate), demonstrating that scale inconsistency is a fundamental barrier to scene-level 3D representation learning.
- **Learnable queries are indispensable**: Removing them causes the error to explode to $10^{25}$; cross-attention tokenization is the cornerstone of the model's success.
- **Semantic filtering substantially improves quality**: L2 error drops from 73.3 to 30.1; filtering noisy Gaussians prevents high-frequency details from being overwhelmed in the latent space.
- **The latent space preserves spatial information**: t-SNE visualizations show that latents of the same scene under different SO(3) rotations form closed loops, and similar scenes cluster together in latent space.
- **The latent space encodes semantics**: Latents from different subsamples of the same scene (covering the same content) are close to each other, while those from different scenes are well-separated.
- Fast inference (~0.06s encode/decode) enables seamless integration with diffusion models for feed-forward generation.

## Highlights & Insights

- **First scene-level 3DGS VAE**: All prior 3D VAE methods (PointNet, L3DG, etc.) completely fail on scene-level data; Can3Tok is the only successful approach.
- **Simple yet effective normalization strategy**: Drawing inspiration from normalizing 2D RGB images to $[-1, 1]$, the proposed center-translation and sphere-scaling normalization resolves an open problem in 3D scene representation learning.
- **SD-compatible latent space**: The $64 \times 64 \times 4$ latent shape is identical to that of Stable Diffusion, enabling direct use of existing diffusion architectures (UNet/DiT) for conditional generation.
- **End-to-end pipeline from data to model**: Beyond the model architecture, a complete 3DGS preprocessing pipeline (normalization + filtering + augmentation) is proposed, offering significant reference value to the community.

## Limitations & Future Work

- Applicable only to 3DGS representations; cannot be directly extended to NeRF, mesh, or other 3D representations.
- The 2.5% failure rate stems primarily from low-quality 3DGS reconstructions in the training data (motion blur, imbalanced near/far viewpoints).
- Semantic filtering retains only the most salient region, potentially discarding complete scene information (foreground-only).
- Generation quality is bounded by VAE reconstruction fidelity; detail recovery still has room for improvement.
- Text-to-3DGS generation relies on short BLIP-annotated captions; richer text conditioning remains to be explored.
- The DL3DV-10K dataset is of limited scale; larger datasets may further improve generalization.

## Related Work & Insights

- The cross-attention compression idea from PerceiverIO is cleverly adapted for 3DGS tokenization.
- The 3DGS normalization problem is analogous to coordinate system unification in NeRF/SfM, but more complex due to the inclusion of scaling parameters.
- The SD-compatible latent design lowers the barrier for downstream generation tasks, enabling direct reuse of text/image encoders.
- Concurrent work such as Bolt3D also identifies the failure of convolutional VAEs on 3DGS, corroborating the generality of Can3Tok's findings.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to address scene-level 3DGS latent space modeling; core innovations (canonical query + normalization) are concise and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage of quantitative, qualitative, t-SNE latent analysis, and ablation studies, though the generation application demonstration is relatively preliminary.
- **Writing Quality**: ⭐⭐⭐⭐ Thorough problem analysis with clear explanation of why existing methods fail.
- **Value**: ⭐⭐⭐⭐⭐ Opens a new direction for scene-level 3DGS generation; the latent space + diffusion model paradigm holds broad prospects.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [\[CVPR 2026\] LoST: Level of Semantics Tokenization for 3D Shapes](../../CVPR2026/3d_vision/lost_level_of_semantics_tokenization_for_3d_shapes.md)
- [\[ICCV 2025\] AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering](aaa-gaussians_anti-aliased_and_artifact-free_3d_gaussian_rendering.md)
- [\[ICCV 2025\] TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction](timeformer_capturing_temporal_relationships_of_deformable_3d_gaussians_for_robus.md)
- [\[ICCV 2025\] Representing 3D Shapes with 64 Latent Vectors for 3D Diffusion Models](representing_3d_shapes_with_64_latent_vectors_for_3d_diffusion_models.md)

<!-- RELATED:END -->
