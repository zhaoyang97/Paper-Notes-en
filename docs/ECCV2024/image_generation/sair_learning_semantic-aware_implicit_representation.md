---
title: >-
  [Paper Note] SAIR: Learning Semantic-aware Implicit Representation
description: >-
  [ECCV 2024][Image Generation][Implicit Neural Representation] This paper proposes Semantic-Aware Implicit Representation (SAIR). By constructing two modules, Semantic Implicit Representation (SIR) and Appearance Implicit Representation (AIR), SAIR integrates text-aligned semantic embeddings extracted by CLIP into implicit neural functions. This enables it to significantly outperform methods relying solely on appearance information in image inpainting tasks with large missing…
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Implicit Neural Representation"
  - "Semantic-aware"
  - "Image Inpainting"
  - "CLIP"
  - "Image Reconstruction"
date: 2026-05-08
content_hash: a17112a222b000cc
---

# SAIR: Learning Semantic-aware Implicit Representation

**Conference**: ECCV 2024  
**arXiv**: [2310.09285](https://arxiv.org/abs/2310.09285)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Implicit Neural Representation, Semantic-aware, Image Inpainting, CLIP, Image Reconstruction

## TL;DR
This paper proposes Semantic-Aware Implicit Representation (SAIR). By constructing two modules, Semantic Implicit Representation (SIR) and Appearance Implicit Representation (AIR), SAIR integrates text-aligned semantic embeddings extracted by CLIP into implicit neural functions. This enables it to significantly outperform methods relying solely on appearance information in image inpainting tasks with large missing regions, achieving a PSNR improvement of 1.65-2.69dB on CelebA-HQ.

## Background & Motivation

**Background**: Implicit neural representations (such as LIIF) achieve image reconstruction by mapping continuous coordinates to color values, performing outstandingly in tasks like super-resolution. These methods utilize an encoder to extract appearance features, and then map coordinates and neighborhood features to RGB colors using an MLP.

**Limitations of Prior Work**: Existing implicit representation methods **only establish continuous appearance mappings**, completely ignoring the semantic information behind pixels. When the input image is severely degraded (e.g., large missing regions), neighborhood appearance features become unavailable, leading to obvious artifacts and semantic inconsistency in reconstruction results. For instance, when an eye is occluded, relying solely on the appearance features of the surrounding "facial skin" cannot infer that an "eye" should be reconstructed.

**Key Challenge**: Appearance features are local and low-level, failing to provide a global semantic understanding of "what object this position should belong to." While appearance features in missing regions tend to be zero, semantic information can be inferred from the context.

**Goal**: (a) Map an implicit representation that can infer semantics in missing regions; (b) leverage semantic information to compensate for missing appearance information in image inpainting.

**Key Insight**: Utilizing the text-aligned embeddings of pre-trained CLIP models as semantic information. The features output by the CLIP visual encoder are naturally aligned with semantic categories (enabling direct zero-shot classification using text), but they have low resolution and cannot handle missing regions.

**Core Idea**: First employ an implicit function to complete the missing semantics in the semantic embedding space (SIR), and then inject the completed semantic information into the Appearance Implicit Representation (AIR) to guide color reconstruction.

## Method

### Overall Architecture
SAIR consists of two cascaded modules:
- **Input**: An image with missing regions $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$ and a mask $\mathbf{M}$
- **SIR**: Uses a modified version of CLIP to extract low-resolution semantic embeddings $\mathbf{Z}^{sem} \in \mathbb{R}^{h \times w \times c}$, and then uses an implicit function to upsample and complete the semantics of the missing regions.
- **AIR**: Integrates appearance features (extracted by a CNN) with the semantic features output by SIR, using another implicit function to predict the color at arbitrary coordinates.
- **Output**: The fully reconstructed image.

### Key Designs

1. **Modified CLIP Encoder (MaskCLIP style)**:

    - **Function**: Extracts spatially-aware, text-aligned semantic embeddings from the image.
    - **Mechanism**: The query and key projections in the final attention layer of the CLIP ViT are removed, and the value projection and output linear layer are replaced with two $1 \times 1$ convolutions. This introduces no extra parameters and does not alter the feature space, but transitions the CLIP output from a single global vector to an $h \times w$ spatial feature map.
    - **Design Motivation**: The original CLIP output consists only of a global "CLS token" and lacks spatial position information. The MaskCLIP modification enables each position to have an independent semantic embedding.

2. **Semantic Implicit Representation (SIR)**:

    - **Function**: Upsamples low-resolution, potentially incomplete semantic embeddings into full-resolution complete semantic features.
    - **Mechanism**: $\mathbf{z}_{\mathbf{p}}^{sem} = \sum_{\mathbf{q} \in \mathcal{N}_{\mathbf{p}}} \omega_{\mathbf{q}} f_\theta([\mathbf{z}_{\mathbf{q}}^{sem}, \mathbf{M}[\mathbf{q}]], \text{dist}(\mathbf{p}, \mathbf{q}))$, which concatenates the mask information with the semantic features, allowing the MLP to perceive which neighboring areas are missing.
    - **Design Motivation**: Simple interpolation/resize upsampling cannot complete the semantics of missing regions. Through neighborhood weighted aggregation and MLP transformation, the implicit function can infer the semantics of missing locations from existing semantics (e.g., inferring that an obscured position is an "eye" from "cheeks").
    - Three key properties: (a) Semantic embeddings are aligned with text categories; (b) non-integer coordinates are interpolatable; (c) missing regions can be completed.

3. **Appearance Implicit Representation (AIR)**:

    - **Function**: Integrates appearance and semantic information to predict the color at arbitrary coordinates.
    - **Mechanism**: $c_{\mathbf{p}} = \sum_{\mathbf{q} \in \mathcal{N}_{\mathbf{p}}} \omega_{\mathbf{q}} f_\beta([\mathbf{z}_{\mathbf{q}}^{app}, \text{SIR}(\mathbf{I}, \mathbf{q})], \text{dist}(\mathbf{p}, \mathbf{q}))$
    - Three input components: the appearance embedding $\mathbf{z}^{app}$ extracted by the CNN (at the same resolution as the image), the enhanced semantic embedding output by SIR, and coordinate distance.
    - **Design Motivation**: When pixels are missing, $\mathbf{z}_{\mathbf{p}}^{app}$ tends to zero, but the semantic information completed by SIR tells the network "what should be here."

### Loss & Training
Only the $L_1$ reconstruction loss is used: $\mathcal{L} = \|c_{pred} - c_{gt}\|_1$. Both the MLPs $f_\theta$ (SIR) and $f_\beta$ (AIR) are 4-layer ReLU networks with a hidden dimension of 256. Trained using the Adam optimizer with a learning rate of 0.0001 for 200 epochs.

## Key Experimental Results

### Main Results (CelebA-HQ Dataset)

| Method | 0-20% Mask PSNR↑ | 20-40% PSNR↑ | 40-60% PSNR↑ | 20-40% SSIM↑ | 40-60% LPIPS↓ |
|------|----------------|-------------|-------------|-------------|-------------|
| EdgeConnect | 34.53 | 27.30 | 22.32 | 0.889 | 0.195 |
| RFRNet | 34.93 | 27.50 | 22.77 | 0.890 | 0.185 |
| LAMA | 36.04 | 29.14 | 22.94 | 0.932 | 0.152 |
| MISF | 36.32 | 29.85 | 23.91 | 0.932 | 0.133 |
| LIIF | 35.27 | 28.80 | 23.30 | 0.923 | 0.136 |
| **SAIR** | **37.97** | **31.49** | **24.87** | **0.944** | **0.124** |

SAIR achieves optimal performance across all mask ratios and metrics. Under the 20-40% mask ratio, its PSNR is 1.64 dB higher than MISF and 2.69 dB higher than LIIF.

### ADE20K Dataset Comparison

| Method | 20-40% PSNR↑ | 40-60% PSNR↑ | 40-60% LPIPS↓ |
|------|-------------|-------------|--------------|
| MISF | 24.97 | 20.59 | 0.233 |
| LIIF | 24.57 | 19.79 | 0.274 |
| **SAIR** | **26.44** | **21.88** | **0.193** |

It also achieves optimal performance on the scene-level dataset ADE20K, demonstrating that the method is applicable not only to human faces but also to complex scenes.

### Key Findings
- **Semantic information is particularly crucial in the presence of large-area missing regions**: The larger the mask ratio (40-60%), the larger the improvement of SAIR relative to LIIF (1.57 vs 2.69 PSNR), suggesting that the less appearance information is available, the more important the compensatory role of semantics becomes.
- **SIR effectively completes missing semantics**: Visualization displays that the feature quality output by the CLIP encoder on mask-covered areas is poor, but the SIR module successfully restores the semantic embeddings of these regions.
- **Semantic information accelerates convergence**: The training losses of SAIR and LIIF converge at similar steps, but the PSNR of SAIR is higher than LIIF from the very beginning of training.
- Pure appearance-based implicit representation (LIIF) generates blurry results with obvious boundaries under large-area missing regions; SAIR generates semantically coherent and visually natural results.

## Highlights & Insights
- **Semantic + appearance dual-layer implicit representation architecture**: The cascaded design of SIR $\rightarrow$ AIR elegantly solves the "understand first, reconstruct later" problem. It first completes the missing parts in a low-dimensional semantic space, and then uses the completed semantics to guide high-dimensional pixel reconstruction. This two-stage approach can be generalized to other tasks requiring the understanding of missing region contents.
- **Zero-overhead spatial semantics via MaskCLIP modification**: Passing the CLIP output to form a spatial-aware embedding without adding parameters or modifying the feature space is a highly practical trick.
- **Integration of mask info into SIR**: Concatenating $\mathbf{M}[\mathbf{q}]$ into the MLP input allows the network to explicitly perceive which neighboring regions are missing, which is simple yet effective.

## Limitations & Future Work
- Evaluated only on the image inpainting task. The paper claims applicability to tasks like generation, editing, and segmentation, but no experiments were conducted.
- The resolution of CLIP semantic embeddings is very low (about $\frac{H}{16} \times \frac{W}{16}$ for ViT-B/16), which may be insufficient for scenes requiring fine semantic boundaries.
- The semantic completion in SIR is achieved through a local neighborhood MLP, which offers limited global understanding performance for ultra-large missing areas (>60%).
- Training requires two V100 GPUs and 200 epochs, reflecting a relatively high training cost compared to some other inpainting methods.
- Lacks comparison with diffusion model-based inpainting methods (e.g., RePaint).

## Rating
- Novelty: ⭐⭐⭐⭐ Combines semantic-level implicit representation with appearance implicit representation for the first time, offering a clear and novel approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across two datasets and three mask ratios, though lacking comparison with diffusion-based methods.
- Writing Quality: ⭐⭐⭐⭐ The framework diagram and mathematical notations are clear, but the abundance of formulas slightly impacts readability.
- Value: ⭐⭐⭐⭐ The dual-layer (semantic + appearance) implicit representation framework holds potential for generalization and provides inspiring insights to the inpainting field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Learning Semantic Latent Directions for Accurate and Controllable Human Motion Prediction](learning_semantic_latent_directions_for_accurate_and_controllable_human_motion_p.md)
- [\[ECCV 2024\] Idempotent Unsupervised Representation Learning for Skeleton-Based Action Recognition](idempotent_unsupervised_representation_learning_for_skeleton-based_action_recogn.md)
- [\[ECCV 2024\] Implicit Style-Content Separation using B-LoRA](implicit_style-content_separation_using_b-lora.md)
- [\[ECCV 2024\] Implicit Concept Removal of Diffusion Models](implicit_concept_removal_of_diffusion_models.md)
- [\[ICML 2026\] Compression as Adaptation: Implicit Visual Representation with Diffusion Foundation Models](../../ICML2026/image_generation/compression_as_adaptation_implicit_visual_representation_with_diffusion_foundati.md)

</div>

<!-- RELATED:END -->
