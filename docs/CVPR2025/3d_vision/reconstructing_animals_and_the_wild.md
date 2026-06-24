---
title: >-
  [Paper Note] Reconstructing Animals and the Wild
description: >-
  [CVPR 2025][3D Vision][Single-image 3D Reconstruction] This paper proposes the RAW method, which uses an LLM to autoregressively decode CLIP image embeddings into structured compositional 3D scene representations (animals + natural environments). It innovatively introduces a CLIP projection head to replace discrete asset name predictions, enabling the model to generalize across larger-scale asset collections. This achieves the first simultaneous reconstruction of both animals…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Single-image 3D Reconstruction"
  - "Animal Pose"
  - "Natural Scenes"
  - "Inverse Graphics with LLMs"
  - "Synthetic Data"
date: 2026-05-08
content_hash: 00fac60067d847b5
---

# Reconstructing Animals and the Wild

**Conference**: CVPR 2025  
**arXiv**: [2411.18807](https://arxiv.org/abs/2411.18807)  
**Code**: [https://raw.is.tue.mpg.de/](https://raw.is.tue.mpg.de/)  
**Area**: 3D Vision / Scene Reconstruction  
**Keywords**: Single-image 3D Reconstruction, Animal Pose, Natural Scenes, Inverse Graphics with LLMs, Synthetic Data

## TL;DR
This paper proposes the RAW method, which uses an LLM to autoregressively decode CLIP image embeddings into structured compositional 3D scene representations (animals + natural environments). It innovatively introduces a CLIP projection head to replace discrete asset name predictions, enabling the model to generalize across larger-scale asset collections. This achieves the first simultaneous reconstruction of both animals and their environment from a single natural image.

## Background & Motivation

**Background**: Extensive work exists in 3D animal reconstruction, advancing from 2D pose estimation to 3D deformable models (such as SMAL). However, almost all methods solely reconstruct the animal body, neglecting the environmental context. Recently, inverse graphics methods have leveraged LLMs to infer graphic codes from images (e.g., IG-LLM).

**Limitations of Prior Work**: (1) Existing animal reconstructions ignore the environment, whereas animal behavior analysis requires environmental context (such as occlusions, physical boundaries, and natural interactions); (2) Natural scene reconstruction is highly challenging, as trees, shrubs, and rocks lack regular geometric shapes unlike man-made objects; (3) 3D ground truth data for compositional natural scenes is extremely scarce.

**Key Challenge**: IG-LLM uses discrete tokens to predict asset names. When the asset collection expands (requiring various animals and plants in this work), the model tends to confuse visually similar but highly distinct semantic objects (e.g., misclassifying a tiger as a shrub) due to the lack of a semantic distance metric between discrete tokens.

**Goal**: To generate editable and animatable compositional 3D scenes containing both animals and plants/terrain from a single natural image.

**Key Insight**: To replace discrete asset names with continuous CLIP embeddings, enabling the LLM to retrieve assets via semantic similarity rather than memorizing token sequences.

**Core Idea**: Render specialized `[CLIP]` tokens in the LLM. When predicting an asset's appearance, the model bypasses the discrete tokenizer and directly projects the hidden state into the CLIP space. Cosine similarity is then utilized to match this projection against the CLIP embeddings of rendered templates in the asset library.

## Method

### Overall Architecture
Based on instruction-tuned LLaMA-7B + a frozen CLIP vision encoder. The input image is encoded by CLIP, projected into the LLM token space, and then the LLM autoregressively generates structured graphics code. First, scene-level properties (solar parameters, atmosphere) are generated, followed by each object sorted from largest to smallest pixel area (position, height, rotation, appearance). The generated code can be rendered directly in Blender.

### Key Designs

1. **CLIP Projection Head (Replacing Discrete Asset Names)**:

    - **Function**: Replaces discrete tokens with continuous semantics to predict object appearance/identity.
    - **Mechanism**: Introduces a special token `[CLIP]`. When the LLM predicts this token, the current hidden state is projected into the CLIP embedding space via a linear layer. The training objective is to measure the cosine similarity against the CLIP embedding of the asset rendered at the corresponding yaw angle. During inference, nearest-neighbor search is performed to match the asset in the library.
    - **Design Motivation**: Discrete asset name tokens lack semantic distance (e.g., "tiger" and "bush" are equidistant in the token space), leading to severe confusion in large asset collections. In contrast, assets with similar appearances naturally lie close to each other in the CLIP space, providing useful gradient signals.

2. **Million-scale Synthetic Dataset**:

    - **Function**: Provides image-code training pairs for compositional 3D scenes.
    - **Mechanism**: Modified based on the Infinigen framework, pre-generating 6,000 assets (1,000 each for birds, carnivores, herbivores, shrubs, rocks, and trees). 100 images are rendered for each of the 10,000 distinct scenes, totaling 1 million images. For each non-directional asset (trees, rocks, shrubs), 72 yaw directions are labeled in $5^\circ$ increments, resulting in 432,000 effective assets.
    - **Design Motivation**: 3D data for real natural scenes cannot be acquired at a large scale, making procedural generation the only viable path. Multi-view rendering increases data diversity.

3. **Structured Graphic Code Representation**:

    - **Function**: Represents 3D scenes as parsable, editable, and renderable sequences of code.
    - **Mechanism**: The scene code sequentially contains solar/atmospheric parameters, the `[CLIP]` embedding of the ground texture, and a list of objects (sorted by pixel area). Each object consists of pixel count, position, height, rotation `[ROT]`, and appearance `[CLIP]`.
    - **Design Motivation**: Sorting from largest to smallest allows the model to first focus on salient objects and then handle the background. The graphic code format can be directly utilized for Blender rendering and editing.

### Loss & Training
Three losses are employed: the standard next-token prediction (for text parts), MSE loss for the rotation matrix (after symmetric orthogonalization), and cosine similarity of the CLIP embeddings combined with norm regularization.

## Key Experimental Results

### Main Results

| Method | LPIPS↓ | CLIP Similarity↑ | BioCLIP↑ | DINOv2↑ |
|------|--------|------------|----------|---------|
| IG-LLM (baseline) | 0.720 | 0.748 | 0.421 | 0.833 |
| + CLIP head (RAW) | **0.696** | **0.762** | **0.456** | **0.842** |

### Ablation Study

| Configuration | Effect Description |
|------|---------|
| Discrete Names vs. CLIP Embeddings | The CLIP version improves across all metrics and avoids semantically unrelated errors (e.g., mistaken tiger $\rightarrow$ bush). |
| Synthetic-only Training $\rightarrow$ Real Images | Successfully generalizes to real-world natural images, verifying cross-domain transfer. |
| Object Sorting (by pixel size) | Helps the model prioritize salient objects. |

### Key Findings
- The CLIP projection head eliminates semantic confusion from discrete tokens, allowing the model to match correctly across 6,000+ assets.
- Training solely on synthetic data generalizes effectively to animal and environment reconstruction in real-world natural images.
- Structured output can be directly edited and animated, supporting downstream behavior analysis.

## Highlights & Insights
- The design of the `[CLIP]` token is elegant—the paradigm of replacing discrete tokens with continuous semantic embeddings can generalize to any LLM task requiring retrieval from large-scale libraries.
- First to incorporate natural environments into 3D animal reconstruction, addressing an important gap in computational ethology.
- Labeling assets in 72 directions at $5^\circ$ yaw increments simply and effectively resolves appearance ambiguity for non-directional objects.

## Limitations & Future Work
- Scene complexity is constrained by the context length of the LLM (up to 25 objects).
- Reconstructed geometry is relatively coarse, employing a predefined asset library rather than precise shape estimation.
- Animal pose estimation accuracy is limited, without using dedicated parametric body models.
- Can be combined with fine-grained animal reconstruction methods (e.g., SMAL) in a coarse-to-fine manner.

## Related Work & Insights
- **vs IG-LLM**: A direct extension whose core contribution is using the CLIP projection head to address scalability in large asset sets.
- **vs SMAL/3D Animal Reconstruction**: A complementary relationship—the present work focuses on overall scene layout rather than fine-grained shapes. Combining both is a future research direction.
- **vs Infinigen**: This work introduces significant simplifications on top of Infinigen to achieve million-scale data generation.

## Rating
- Novelty: ⭐⭐⭐⭐ Clever implementation of the CLIP projection head, first to reconstruct animals and environments simultaneously.
- Experimental Thoroughness: ⭐⭐⭐ Primarily evaluated on synthetic data; real-world images are mostly presented via qualitative demonstrations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and detailed methodology description.
- Value: ⭐⭐⭐⭐ Pioneering significance for computational ethology and natural scene understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)
- [\[CVPR 2025\] Reconstructing People, Places, and Cameras](reconstructing_people_places_and_cameras.md)
- [\[CVPR 2025\] Reconstructing Humans with a Biomechanically Accurate Skeleton](reconstructing_humans_with_a_biomechanically_accurate_skeleton.md)
- [\[CVPR 2025\] PICO: Reconstructing 3D People In Contact with Objects](pico_reconstructing_3d_people_in_contact_with_objects.md)
- [\[CVPR 2025\] Extreme Rotation Estimation in the Wild](extreme_rotation_estimation_in_the_wild.md)

</div>

<!-- RELATED:END -->
