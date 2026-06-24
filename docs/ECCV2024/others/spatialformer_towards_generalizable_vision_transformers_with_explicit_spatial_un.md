---
title: >-
  [Paper Note] SpatialFormer: Towards Generalizable Vision Transformers with Explicit Spatial Understanding
description: >-
  [ECCV 2024][Vision Transformer] This work proposes the SpatialFormer architecture, which explicitly models the global spatial relations of scenes by introducing adaptive spatial tokens. It adopts a decoder-only architecture and bilateral cross-attention blocks to achieve efficient interaction between context and spatial information, demonstrating excellent generalization and transferability across classification, segmentation, and detection tasks.
tags:
  - "ECCV 2024"
  - "Vision Transformer"
  - "Spatial Understanding"
  - "Spatial Token"
  - "Bilateral Cross-Attention"
  - "Transferable Representation Learning"
date: 2026-05-08
content_hash: cb23c0f1de3a7077
---

# SpatialFormer: Towards Generalizable Vision Transformers with Explicit Spatial Understanding

**Conference**: ECCV 2024  
**Institution**: Tsinghua University  
**Code**: [https://github.com/Euphoria16/SpatialFormer](https://github.com/Euphoria16/SpatialFormer)  
**Area**: Vision Transformer  
**Keywords**: Vision Transformer, Spatial Understanding, Spatial Token, Bilateral Cross-Attention, Transferable Representation Learning

## TL;DR

This work proposes the SpatialFormer architecture, which explicitly models the global spatial relations of scenes by introducing adaptive spatial tokens. It adopts a decoder-only architecture and bilateral cross-attention blocks to achieve efficient interaction between context and spatial information, demonstrating excellent generalization and transferability across classification, segmentation, and detection tasks.

## Background & Motivation

**Background**: Vision Transformers (ViTs) have become core components in computer vision, achieving remarkable success in tasks such as classification, detection, and segmentation. Existing ViTs primarily extract contextual features through patch embedding and introduce spatial information using additional position encodings (such as absolute, relative, or rotary position encodings).

**Limitations of Prior Work**: The way current ViTs introduce spatial information has fundamental limitations: position encodings only encode the local position of each image token (i.e., the coordinates of the token within the image), failing to effectively model the global spatial relationships of the underlying scene. For example, changes in camera intrinsic and extrinsic parameters, 3D scene layout, and relative spatial relationships between objects are crucial for tasks like detection and segmentation, yet existing position encoding mechanisms cannot capture them.

**Key Challenge**: Existing ViTs couple spatial information and contextual features within the same set of tokens, implicitly encoding space through position embeddings. This makes it difficult for the model to specifically learn and leverage global spatial structures. Spatial information is submerged within semantic features, leading to inadequate spatial understanding during cross-task transfer.

**Goal**: (1) How to explicitly and decouple-ly represent global spatial relationships in ViT? (2) How to make spatial representations possess both generic priors and adaptive capabilities for specific images? (3) How to design an efficient architecture to achieve interaction between spatial and contextual information while maintaining good generalization?

**Key Insight**: The authors observe that by decoupling spatial information from image tokens and representing it with a dedicated set of "spatial tokens", the model can separately learn "what is in the image" (context) and "where it is" (space). These two types of information can then mutually reinforce each other through interaction. This is analogous to the separation of the "what" and "where" pathways in the human visual system.

**Core Idea**: Adaptive spatial tokens are introduced to process in parallel with image tokens, achieving decoupled spatial-context interaction via bilateral cross-attention. The output spatial tokens can directly serve as enhanced initial queries for downstream task decoders.

## Method

### Overall Architecture

SpatialFormer receives image patch embeddings as image tokens (context tokens) while initializing a set of spatial tokens. The entire architecture adopts a decoder-only style, where each layer contains a Bilateral Cross-Attention Block, enabling interaction between spatial tokens and image tokens at every layer. After multi-layer processing, the output image tokens contain spatially enhanced semantic features, while the output spatial tokens encode explicit scene spatial structure information. These spatial tokens can be directly used as initial queries for downstream task-specific decoders (e.g., detection heads, segmentation heads), thereby achieving better task adaptation.

### Key Designs

1. **Adaptive Spatial Tokens**:

    - **Function**: Explicitly represent global spatial relationships of the scene corresponding to the image.
    - **Mechanism**: Spatial tokens are initialized with standard position encodings (e.g., 2D sinusoidal encodings) to introduce generic spatial prior knowledge. On top of this, learnable embeddings are superimposed, allowing them to adaptively learn spatial patterns in specific data distributions. This design of "fixed priors + learnable offsets" enables spatial tokens to possess both generalization capabilities (from the generic priors of position encodings) and adaptability (from the data-driven adjustment of learnable parts). The number of spatial tokens can be set independently of the number of image tokens, providing high flexibility.
    - **Design Motivation**: Purely learnable tokens (like DETR's object queries) learn from scratch, converging slowly and generalizing poorly, whereas purely fixed position encodings lack adaptability. Combining both represents a superior design choice.

2. **Bilateral Cross-Attention Block**:

    - **Function**: Achieve efficient bidirectional information interaction between spatial tokens and image tokens.
    - **Mechanism**: Within each Transformer layer, a bidirectional cross-attention mechanism is designed. In the first step, spatial tokens act as queries and image tokens as keys and values, allowing spatial tokens to extract spatially-relevant patterns from image features. In the second step, image tokens act as queries and spatial tokens as keys and values, allowing image features to be guided by the global spatial structure. This two-step interaction mutually enhances both sets of tokens. Compared to simple concatenation followed by self-attention, bilateral cross-attention has lower computational cost (avoiding the quadratic complexity growth from doubling the number of tokens) while maintaining explicit control over information flow.
    - **Design Motivation**: In self-attention, the information interaction between spatial tokens and image tokens is implicit, which may cause spatial information to be overwhelmed by semantic information. Explicit bidirectional cross-attention ensures balanced interaction between the two types of information.

3. **Decoder-only Architecture and Downstream Adaptation**:

    - **Function**: Provide a transferable unified backbone and supply enhanced initialization for downstream tasks.
    - **Mechanism**: The entire backbone is designed as a decoder-only architecture (each layer contains only cross-attention and FFN, without an encoder), enabling the model to learn context and spatial representations in a unified manner during pre-training. When transferring to downstream tasks, the output spatial tokens can directly serve as initial queries for task-specific decoders (e.g., DETR-style detection heads, Mask2Former segmentation heads). Compared to randomly initialized queries, these pre-trained spatial tokens already encode rich spatial priors, which accelerates decoder convergence and boosts performance.
    - **Design Motivation**: Features output by existing backbones contain only contextual information, forcing downstream decoders (such as DETR) to learn the spatial distribution of object queries from scratch. If the backbone can directly output tokens encoding spatial structures to act as initial queries, a better backbone-decoder synergy can be achieved.

### Loss & Training

The model is pre-trained on ImageNet using standard classification loss. When transferring to downstream tasks, standard training strategies for each task are utilized (e.g., DETR loss for detection, mask loss for segmentation). Spatial tokens naturally learn spatial priors during pre-training through interactions with image tokens.

## Key Experimental Results

### Main Results

| Task | Dataset | Model | Ours (SpatialFormer) | Baseline ViT | Gain |
|------|--------|------|---------------------|-------------------|------|
| Classification | ImageNet-1K | Small | 83.6% Top-1 | ~83.1% (DeiT-S) | +0.5% |
| Semantic Segmentation | ADE20K | Small | 48.5 mIoU | ~47.2 (Swin-S) | +1.3 |
| 2D Detection | COCO | Small | 50.1 AP | ~49.0 (Swin-S) | +1.1 |
| 3D Detection | nuScenes | Small | Significant NDS improvement | Baseline methods | Significant |

### Ablation Study

| Configuration | ImageNet Top-1 | ADE20K mIoU | Description |
|------|---------------|-------------|------|
| Full SpatialFormer | Best | Best | Full model |
| w/o spatial tokens | Decline ~0.5% | Decline ~1.0 | No explicit spatial modeling |
| w/o learnable embeddings | Decline ~0.3% | Decline ~0.6 | Fixed position encodings only |
| w/o position encoding initialization | Decline ~0.4% | Decline ~0.8 | Purely learnable tokens |
| Self-attention replacing Bilateral CA | Decline | Decline | Insufficient information interaction |
| Spatial tokens not passed to decoder | - | Decline ~0.5 | Validates the value of query initialization |

### Key Findings

- The gains from spatial tokens on spatial-sensitive tasks like segmentation and detection ($\sim 1.0+$ mIoU/AP) are larger than on classification ($\sim 0.5\%$ Top-1), validating that explicit spatial understanding is more crucial for spatially dense prediction tasks.
- Both position encoding initialization and learnable embeddings are indispensable, each contributing roughly half of the gains.
- Using spatial tokens as initial queries for downstream decoders brings additional improvements, indicating that pre-trained spatial priors provide substantial benefits for decoder initialization.
- The advantages of SpatialFormer are even more pronounced in tasks requiring strong spatial understanding, such as 3D detection (nuScenes).

## Highlights & Insights

- **The concept of decoupling spatial information from image tokens for explicit modeling is highly inspiring.** Analogous to the separation of position and content in NLP, explicitly performing this separation in vision is a natural but overlooked direction. This decoupling can be transferred to scenarios such as video understanding (spatiotemporal decoupling) and 3D vision (geometric-semantic decoupling).
- **The design allowing spatial tokens to directly serve as initial queries for downstream decoders achieves seamless backbone-decoder integration.** This addresses the long-standing issue in DETR-style methods where object queries are difficult to learn, and could potentially become a new standard design for detection transformers.
- **Bilateral cross-attention is more efficient than concatenated self-attention**, enabling effective bidirectional interaction without significantly increasing computation.

## Limitations & Future Work

- The current number of spatial tokens is fixed, which may lack flexibility for scenes with different resolutions or complex layouts.
- The "spatial understanding" learned by the spatial tokens remains somewhat implicit, lacking detailed visualization and interpretability analysis.
- Performance on larger-scale models (Large/Huge) has not been fully explored.
- The code repository currently only contains a README without the full implementation, presenting a high barrier of reproduction.
- Future work could explore combining spatial tokens with explicit 3D geometric information (such as depth estimation and camera parameters) to further enhance spatial understanding capabilities.

## Related Work & Insights

- **vs Swin Transformer**: Swin addresses spatial locality by restricting the attention range using window mechanisms, but it still encodes space implicitly through position encodings. SpatialFormer explicitly models global spatial relationships through dedicated spatial tokens, representing a different spatial processing philosophy.
- **vs DETR**: The object queries in DETR are essentially a type of spatial representation but are learned from scratch. SpatialFormer's spatial tokens, once pre-trained, can be directly transferred to a DETR-style decoder, providing a warm start for the queries.
- **vs ViTDet**: ViTDet directly uses plain ViTs for detection without introducing additional spatial inductive biases. SpatialFormer demonstrates that incorporating explicit spatial tokens yields meaningful gains without sacrificing efficiency.

## Rating

- Novelty: ⭐⭐⭐⭐ The spatial-context decoupled token design is a meaningful extension to the ViT architecture.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple tasks including classification, segmentation, and 2D/3D detection, with fairly comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and systematic method description, though it is slightly regrettable that the code is not fully open-sourced.
- Value: ⭐⭐⭐⭐ The spatial token concept is inspiring for subsequent ViT designs, although it requires broader validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] AttnZero: Efficient Attention Discovery for Vision Transformers](attnzero_efficient_attention_discovery_for_vision_transformers.md)
- [\[ICML 2026\] Spatial Priors via Space Filling Curves for Small and Limited Data Vision Transformers](../../ICML2026/others/spatial_priors_via_space_filling_curves_for_small_and_limited_data_vision_transf.md)
- [\[ACL 2025\] Enhancing Transformers for Generalizable First-Order Logical Entailment](../../ACL2025/others/enhancing_fol_entailment.md)
- [\[ECCV 2024\] Synergy of Sight and Semantics: Visual Intention Understanding with CLIP](synergy_of_sight_and_semantics_visual_intention_understanding_with_clip.md)
- [\[ICLR 2026\] Hippoformer: Integrating Hippocampus-inspired Spatial Memory with Transformers](../../ICLR2026/others/hippoformer_integrating_hippocampus-inspired_spatial_memory_with_transformers.md)

</div>

<!-- RELATED:END -->
