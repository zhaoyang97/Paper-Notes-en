---
title: >-
  [Paper Note] ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models
description: >-
  [CVPR 2025][Image Generation][concept learning] A two-stage framework, ICE, is proposed to automatically localize object-level concepts from a single image and decompose them into intrinsic properties (category, color, texture) using a single T2I diffusion model, achieving label-free and model-free hierarchical visual concept extraction.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "concept learning"
  - "intrinsic concepts"
  - "diffusion model"
  - "Stable Diffusion"
  - "triplet loss"
  - "concept decomposition"
date: 2026-05-08
content_hash: 358ec4bd1c086bcc
---

# ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2503.19902](https://arxiv.org/abs/2503.19902)  
**Code**: [https://visual-ai.github.io/ice](https://visual-ai.github.io/ice)  
**Area**: Image Generation  
**Keywords**: concept learning, intrinsic concepts, diffusion model, Stable Diffusion, triplet loss, concept decomposition

## TL;DR

A two-stage framework, ICE, is proposed to automatically localize object-level concepts from a single image and decompose them into intrinsic properties (category, color, texture) using a single T2I diffusion model, achieving label-free and model-free hierarchical visual concept extraction.

## Background & Motivation

**Background**: T2I diffusion models (such as Stable Diffusion) have accumulated rich visual knowledge of the world and have been used for tasks like image classification, segmentation, and semantic correspondence. Generative concept learning aims to decompose and understand the fundamental elements that constitute complex scenes from images.

**Limitations of Prior Work**: (1) Textual Inversion/DreamBooth requires multiple images of the same concept for learning, making extraction from a single image impossible; (2) Break-A-Scene relies on manual mask annotations; (3) Methods like ConceptExpress only extract object-level concepts, neglecting intrinsic properties like color/texture; (4) LangInt requires training a separate encoder for each concept axis, which lacks scalability.

**Key Challenge**: The definition of visual concepts is inherently ambiguous and hierarchical—a "red metal ball" contains multiple layers of intrinsic concepts such as object category (ball), color (red), and material (metal). Existing methods lack a systematic decomposition mechanism.

**Key Insight**: Fully exploit the intrinsic capabilities of pretrained T2I models (CLIP encoder for text retrieval, self-attention layers for segmentation) without introducing additional pretrained models.

## Method

### Overall Architecture

A two-stage architecture:
1. **Stage 1 - Automatic Concept Localization**: Training-free extraction of text concepts + corresponding masks from images.
2. **Stage 2 - Structured Concept Learning**: Learning object-level concepts and intrinsic concepts in two phases.

### Key Designs

**1. Automatic Concept Localization (Stage 1, Training-free)**
- **Function**: Iteratively extracts concepts and masks from the image until no remaining objects are found in the image.
- **Mechanism**:
  1. Image-to-Text Retriever $\mathcal{T}(\mathbf{x})$: Decomposes dense embeddings into sparse, explainable semantic concepts using the SpLiCE framework on the CLIP encoder to obtain the top-1 text concept $c_i$.
  2. Zero-shot Segmentor $\mathcal{S}(\mathbf{x}, c_i)$: Generates corresponding mask $\mathbf{m}_i$ using self-attention layers of Stable Diffusion (DiffSeg).
  3. Iterative Removal: $\mathbf{x}' = \mathbf{x} \odot (1 - \mathbf{m}_i)$, repeating after updating the image.
- **Design Motivation**: Fully reuse internal components of the T2I model without introducing external models, ensuring the framework is lightweight and consistent.

**2. Object-level Concept Learning (Stage 2, Phase 1)**
- **Function**: Learns two tokens for each object-level concept $c_i$: concept-specific $c_i^{conspec}$ (representing general semantic category) and instance-specific $c_i^{inspec}$ (representing instance-specific attributes).
- **Mechanism**: A triplet loss is utilized to pull the concept-specific token closer to the anchor (the text concept extracted in Stage 1) and push the instance-specific token away:
  $$\mathcal{L}_{triplet}^{obj} = \max(0, \|\mathcal{E}(anchor) - \mathcal{E}(c_i^{conspec})\|_2^2 - \|\mathcal{E}(anchor) - \mathcal{E}(c_i^{inspec})\|_2^2 + \gamma)$$
- **Design Motivation**: The anchor of the triplet loss is derived from the high-quality text concept initialization of Stage 1, providing a better starting point for learning than random initialization.

**3. Intrinsic Concept Learning (Stage 2, Phase 2)**
- **Function**: Further decomposes the instance-specific token into intrinsic property tokens $c_j^{intrinsic}$ (e.g., color, texture).
- **Mechanism**: Constructs exclusive anchor text for each intrinsic concept (e.g., "the colour of $c_i^{inspec}$"), constrained by an intrinsic triplet loss:
  $$\mathcal{L}_{triplet}^{intrinsic} = \max(0, \|\mathcal{E}(anchor_j) - \mathcal{E}(c_j^{intrinsic})\|_2^2 - \|\mathcal{E}(anchor_j) - \mathcal{E}(c_k^{intrinsic})\|_2^2 + \gamma)$$
- **Concept Refinement**: Fine-tunes the U-Net and text encoder for a small number of steps (300 steps) after Phase 2 to ensure accurate alignment between intrinsic concepts and visual attributes.
- **Design Motivation**: Hierarchical decomposition (object-level → intrinsic-concept-level) is significantly more stable than learning all concepts at once.

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda_{att}\mathcal{L}_{att} + \lambda_{triplet}\mathcal{L}_{triplet}$$

- $\mathcal{L}_{recon}$: Denoising reconstruction loss
- $\mathcal{L}_{att}$: Attention mask loss (aligns attention maps with segmentation masks using Wasserstein distance)
- $\mathcal{L}_{triplet}$: Triplet loss (object-level or intrinsic-level, switched by phase)
- Hyperparameters: $\lambda_{att} = 1 \times 10^{-5}$, $\lambda_{triplet} = 1$

Training Configuration: Phase 1 (400 steps) → Phase 2 (400 steps) → Refinement (300 steps), on a single 3090 GPU.

## Key Experimental Results

### Main Results — UCE Benchmark (CLIP Encoder)

| Method | SIMI↑ | SIMC↑ | ACC1↑ | ACC3↑ |
|---|---|---|---|---|
| Break-A-Scene | 0.627 | 0.773 | 0.174 | 0.282 |
| ConceptExpress | 0.689 | 0.784 | 0.263 | 0.385 |
| **ICE** | **0.738** | **0.822** | **0.325** | **0.518** |

### Evaluation with DINO Encoder

| Method | SIMI↑ | SIMC↑ | ACC1↑ | ACC3↑ |
|---|---|---|---|---|
| ConceptExpress | 0.319 | 0.568 | 0.324 | 0.470 |
| **ICE** | **0.677** | **0.755** | **0.476** | **0.638** |

The advantage of ICE in the DINO feature space is even more significant (112% gain in SIMI), indicating that the learned concepts have cross-encoder generalization.

### Ablation Study

| Variant | CLIP SIMI↑ | CLIP SIMC↑ | DINO SIMI↑ | DINO ACC3↑ |
|---|---|---|---|---|
| ConceptExpress | 0.689 | 0.784 | 0.319 | 0.470 |
| ICE w. mask only | 0.710 | 0.781 | 0.493 | 0.604 |
| ICE w/o Stage Two | 0.726 | 0.807 | 0.501 | 0.604 |
| ICE w/o text init | 0.722 | 0.814 | 0.548 | 0.627 |
| **ICE Full** | **0.738** | **0.822** | **0.677** | **0.638** |

### Mask Quality Comparison

| Method | mIoU↑ | Recall↑ | Precision↑ |
|---|---|---|---|
| ConceptExpress | 0.483 | 0.676 | 0.657 |
| **ICE** | **0.635** | **0.893** | **0.720** |

The recall of ICE increases from 0.676 to 0.893, indicating that automatic concept localization is capable of capturing more object regions.

### Key Findings

1. **Every module contributes**: Ablation shows that mask, text init, and Stage Two learning each provide incremental gains.
2. **Mask quality is foundational**: ICE's mIoU=0.635 significantly outperforms ConceptExpress's 0.483. Better localization leads to better concept learning.
3. **Greater advantages in DINO space**: This indicates that the learned concepts of ICE represent generic visual attributes rather than just CLIP semantics.
4. **Zero extra models**: All internal components of the T2I model (CLIP encoder + self-attention segmentation) are reused, achieving end-to-end consistency.

## Highlights & Insights

- The hierarchical concept decomposition approach (object → category + instance → intrinsic properties) is clean and elegant.
- Stage 1 is fully training-free, utilizing only internal components of the T2I model for concept localization and segmentation.
- The triplet loss constrained with text anchors gracefully embeds concept constraints into the learning process.
- It is the first unsupervised framework to simultaneously extract object-level and intrinsic-level concepts from a single image.

## Limitations & Future Work

- Dependent on the zero-shot segmentation quality of DiffSeg; extraction might be incomplete in complex, highly overlapping scenes.
- Types of intrinsic concepts (e.g., color, texture) need to be predefined and cannot automatically discover new attribute dimensions.
- The number of training steps is small (1100 steps in total), which might limit the adequacy of concept refinement.
- Evaluated only on the Unsplash dataset, lacking validation across diverse scenarios.
- There is still room for improvement in the interpretability of concept decomposition (some intrinsic concepts may remain entangled).

## Related Work & Insights

- **Textual Inversion / DreamBooth**: Pioneers in concept learning but require multiple images and are limited to a single concept.
- **ConceptExpress**: The state-of-the-art baseline for single-image multi-concept learning, but restricted to the object level.
- **Inspiration Tree**: Attempts concept decomposition but lacks structured guidance.
- **Insights**: Internal components of T2I models (CLIP, self-attention) contain rich exploitable signals. The paradigm of "model as a tool" is worth further investigation.

## Rating

⭐⭐⭐⭐ — The framework design is elegant, offering a novel path to tap into the concept decomposition capability of T2I models. It achieves comprehensive leadership on the UCE benchmark, though the requirement to predefine intrinsic concept types remains a limitation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Intrinsic Concept Extraction Based on Compositional Interpretability](../../CVPR2026/image_generation/intrinsic_concept_extraction_based_on_compositional_interpretability.md)
- [\[CVPR 2025\] ScribbleLight: Single Image Indoor Relighting with Scribbles](scribblelight_single_image_indoor_relighting_with_scribbles.md)
- [\[CVPR 2025\] DiffLocks: Generating 3D Hair from a Single Image using Diffusion Models](difflocks_generating_3d_hair_from_a_single_image_using_diffusion_models.md)
- [\[CVPR 2025\] Six-CD: Benchmarking Concept Removals for Text-to-Image Diffusion Models](six-cd_benchmarking_concept_removals_for_text-to-image_diffusion_models.md)
- [\[CVPR 2025\] Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)

</div>

<!-- RELATED:END -->
