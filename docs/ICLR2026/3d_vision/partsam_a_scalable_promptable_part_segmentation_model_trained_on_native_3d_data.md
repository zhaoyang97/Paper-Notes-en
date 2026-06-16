---
title: >-
  [Paper Note] PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data
description: >-
  [ICLR 2026][3D Vision][3D part segmentation] This paper proposes PartSAM, the first promptable part segmentation model trained on large-scale native 3D data. It employs a triplane dual-branch encoder (frozen SAM priors +…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "3D part segmentation"
  - "SAM"
  - "prompt-based"
  - "native 3D data"
  - "open-world"
date: 2026-05-08
content_hash: 09469d8e7364090a
---

# PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data

**Conference**: ICLR 2026
**arXiv**: [2509.21965](https://arxiv.org/abs/2509.21965)  
**Code**: [https://czvvd.github.io/PartSAMPage/](https://czvvd.github.io/PartSAMPage/)  
**Area**: 3D Vision
**Keywords**: 3D part segmentation, SAM, prompt-based, native 3D data, open-world

## TL;DR
This paper proposes PartSAM, the first promptable part segmentation model trained on large-scale native 3D data. It employs a triplane dual-branch encoder (frozen SAM priors + learnable 3D branch) and a SAM-style decoder. A model-in-the-loop annotation pipeline is used to construct 5M+ shape–part pairs. Under open-world settings, a single click from PartSAM outperforms Point-SAM by over 90% in IoU@1.

## Background & Motivation

**Background**: 3D part segmentation is a classical problem in computer vision. Early methods trained on closed-set datasets such as ShapeNet-Part and PartNet, limiting generalization to open-world scenarios. Recent approaches (e.g., SAMPart3D, PartField) leverage 2D priors from SAM via multi-view lifting.

**Limitations of Prior Work**: (1) 2D→3D lifting discards internal structural information, restricting understanding to object surfaces; (2) clustering-based methods (e.g., PartField) lack interactive controllability; (3) training data bottleneck — large-scale 3D part annotations are scarce; (4) heavy reliance on mesh connectivity causes performance collapse on AI-generated shapes.

**Key Challenge**: How can one train a model that supports flexible interaction and understands 3D internal structures, given the absence of large-scale 3D part annotations?

**Goal**: Construct large-scale native 3D part data (5M+ pairs), design a novel architecture that exploits both 2D priors and 3D knowledge, and achieve SAM-style interactive and automatic part segmentation.

**Key Insight**: A dual-channel design — a frozen SAM channel retains 2D knowledge, while a learnable channel adapts to native 3D annotations.

**Core Idea**: A dual-branch triplane encoder combined with a SAM-style decoder is trained on millions of native 3D part annotations, enabling the first promptable part segmentation model that genuinely understands 3D internal structures.

## Method

### Overall Architecture
The input consists of a point cloud $P_{in} \in \mathbb{R}^{N \times 9}$ (coordinates, normals, RGB) and prompt points $P_{prompt}$. The encoder extracts triplane feature fields and samples patch embeddings $F_c$. The decoder combines prompt embeddings $F_p$ through a bidirectional Transformer to generate segmentation masks. The model supports both interactive mode (user clicks) and automatic mode (Segment Every Part).

### Key Designs

1. **Dual-Branch Triplane Encoder**:

    - **Function**: Simultaneously preserves SAM's 2D priors while learning native 3D representations.
    - **Mechanism**: Two parallel branches, both built with PVCNN+Transformer to construct triplane feature fields. The frozen branch retains PartField-pretrained SAM contrastive learning features; the learnable branch accepts additional normal/RGB inputs via zero convolution and is trained on native 3D annotations.
    - **Design Motivation**: The frozen branch prevents forgetting of 2D knowledge, while the learnable branch adapts to new 3D supervision signals.

2. **SAM-Inspired Prompt-Guided Decoder**:

    - **Function**: Generates segmentation masks from prompts and features.
    - **Mechanism**: Output token $T_{out}$ and IoU token $T_{iou}$ are introduced; bidirectional cross-attention interaction is defined as $F_c' = \text{CrossAttn}(F_c \leftrightarrow [F_p; T_{out}; T_{iou}])$. For a single prompt, three candidate masks are decoded in parallel, and the IoU token predicts quality scores to select the optimal mask.
    - **Design Motivation**: SAM's parallel decoding and IoU prediction handle ambiguity at part boundaries.

3. **Model-in-the-Loop Data Annotation**:

    - **Function**: Scales training data from fragmented Objaverse assets.
    - **Mechanism**: Stage 1 extracts 22M parts from 180k shapes via scene graphs and connected components. Stage 2 uses PartField to generate pseudo-labels (K=10/20/30 clustering), with PartSAM performing 10-round interactive verification; samples are accepted if IoU@1 > 60% or IoU@10 > 90%, yielding 55M parts from 500k shapes.
    - **Design Motivation**: High IoU@1 indicates intrinsically unambiguous parts; low IoU@1 but high IoU@10 indicates parts that can be refined through interaction.

### Loss & Training
$$\mathcal{L} = \mathcal{L}_{focal} + \alpha \mathcal{L}_{dice} + \mathcal{L}_{IoU} + \lambda \mathcal{L}_{triplet}$$

## Key Experimental Results

### Main Results (Interactive Segmentation)

| Dataset | Method | IoU@1 | IoU@5 | IoU@10 |
|--------|------|-------|-------|--------|
| PartObjaverse-Tiny | Point-SAM | 29.4 | 68.7 | 73.9 |
| | **PartSAM** | **56.1** | **84.1** | **87.6** |
| PartNetE | Point-SAM | 35.9 | 75.1 | 79.2 |
| | **PartSAM** | **59.5** | **86.5** | **89.9** |

### Automatic Segmentation

| Method | PartObjaverse-Tiny | PartNetE |
|------|-------------------|----------|
| PartField | 51.5 | 59.1 |
| **PartSAM** | **69.5** | **72.4** |

### Key Findings
- **91% improvement in IoU@1**: A single click suffices for accurate segmentation, whereas Point-SAM heavily depends on iterative correction.
- **Internal structure understanding**: The model can segment occluded contents inside handbags and car seats, which SAMesh fails to handle.
- **Generalization to AI-generated shapes**: Strong performance is maintained on irregular meshes generated by Hunyuan3D.

## Highlights & Insights
- **Paradigm shift**: From "2D lifting → clustering" to "native 3D + interactive decoding," enabling genuine understanding of 3D internal structures for the first time.
- **Model-in-the-loop annotation resolves the chicken-and-egg problem**: A two-tier filtering strategy combining PartField pseudo-labels and PartSAM interactive verification.
- **Elegant dual-branch design**: The frozen branch retains 2D boundary priors; the learnable branch adapts to coarse 3D signals; summation fusion is simple yet effective.

## Limitations & Future Work
- Interactive inference incurs higher computational cost compared to clustering-based methods.
- The 500k shapes remain far behind the billion-scale data of 2D foundation models.
- Performance degrades on very fine-grained parts (e.g., buttons) due to point sampling resolution constraints.

## Related Work & Insights
- **vs. PartField**: Post-clustering processing is the bottleneck; prompted decoding bypasses this issue and does not rely on mesh connectivity.
- **vs. Point-SAM**: Both data scale (5× more data) and architecture (dual-branch encoder) are substantially improved, yielding a significant performance leap.
- Implication: 3D foundation models require truly large-scale native 3D data.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First promptable part segmentation model trained on native 3D data.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-benchmark evaluation; ablation details are in the appendix.
- Writing Quality: ⭐⭐⭐⭐⭐ Systematic method description with outstanding visualizations.
- Value: ⭐⭐⭐⭐⭐ Opens a new direction toward a 3D counterpart of SAM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] S2AM3D: Scale-controllable Part Segmentation of 3D Point Clouds](../../CVPR2026/3d_vision/s2am3d_scale-controllable_part_segmentation_of_3d_point_cloud.md)
- [\[ICLR 2026\] GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation](geopurify_a_data-efficient_geometric_distillation_framework_for_open-vocabulary_.md)
- [\[CVPR 2026\] GeoSAM2: Unleashing the Power of SAM2 for 3D Part Segmentation](../../CVPR2026/3d_vision/geosam2_unleashing_the_power_of_sam2_for_3d_part_segmentation.md)
- [\[CVPR 2026\] Action-guided Generation of 3D Functionality Segmentation Data](../../CVPR2026/3d_vision/action-guided_generation_of_3d_functionality_segmentation_data.md)
- [\[CVPR 2026\] Learning Hierarchical Hyperbolic Mixture Model for Part-aware 3D Generation](../../CVPR2026/3d_vision/learning_hierarchical_hyperbolic_mixture_model_for_part-aware_3d_generation.md)

</div>

<!-- RELATED:END -->
