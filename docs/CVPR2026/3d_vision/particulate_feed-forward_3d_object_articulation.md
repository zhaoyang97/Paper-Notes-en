---
title: >-
  [Paper Note] Particulate: Feed-Forward 3D Object Articulation
description: >-
  [CVPR 2026][3D Vision][Articulated Objects] Particulate proposes a feed-forward model that infers complete articulation structures (part segmentation, kinematic tree, and motion constraints) from a static 3D mesh within seconds. Built upon the Part Articulation Transformer and trained end-to-end on public datasets, it significantly outperforms existing per-object optimization methods and can be combined with 3D generative models to enable single-image-to-articulated-3D-object generation.
tags:
  - CVPR 2026
  - 3D Vision
  - Articulated Objects
  - 3D Part Segmentation
  - Motion Constraint Prediction
  - Feed-Forward Inference
  - Transformer
date: 2026-05-08
content_hash: f7b9948a0149fdc6
---

# Particulate: Feed-Forward 3D Object Articulation

**Conference**: CVPR 2026
**arXiv**: [2512.11798](https://arxiv.org/abs/2512.11798)
**Code**: [https://ruiningli.com/particulate](https://ruiningli.com/particulate)
**Area**: 3D Vision
**Keywords**: Articulated Objects, 3D Part Segmentation, Motion Constraint Prediction, Feed-Forward Inference, Transformer

## TL;DR
Particulate proposes a feed-forward model that infers complete articulation structures (part segmentation, kinematic tree, and motion constraints) from a static 3D mesh within seconds. Built upon the Part Articulation Transformer and trained end-to-end on public datasets, it significantly outperforms existing per-object optimization methods and can be combined with 3D generative models to enable single-image-to-articulated-3D-object generation.

## Background & Motivation

1. **Background**: Most real-world objects possess not only shape but also motion capabilities (e.g., cabinet door rotation, drawer sliding). Understanding articulation structure is critical for robotic manipulation, game simulation, and digital twins. Existing methods either rely on rule-based procedural generation that struggles to cover long-tail objects, or require per-object multi-view optimization with prohibitively long runtimes (10–20+ minutes).

2. **Limitations of Prior Work**: Learning-based methods fall into three categories: (a) 3D part segmentation methods predict semantic segmentation without modeling articulation relationships; (b) 3D articulated object generation methods cover only a few categories and assume known kinematic structures; (c) VLM-based methods (e.g., Articulate AnyMesh) offer good generalization but require per-object optimization lasting tens of minutes and cannot handle internal or occluded parts.

3. **Key Challenge**: How to achieve fast feed-forward inference while maintaining generalization, and handling internally invisible parts?

4. **Goal**: Directly predict complete articulation structures (part segmentation + kinematic tree + motion parameters) from static 3D meshes in a feed-forward manner, supporting multi-joint, multi-category, and AI-generated 3D assets.

5. **Key Insight**: Leverage the flexibility and scalability of Transformers by training end-to-end on large-scale multi-category articulation datasets, using learnable part queries and multi-head decoders to predict each articulation attribute independently.

6. **Core Idea**: Employ a standard Transformer with learnable part queries for end-to-end training on point clouds, enabling single-pass feed-forward inference to predict all articulation attributes, including part segmentation, kinematic tree, and motion constraints.

## Method

### Overall Architecture
The input is a 3D mesh (converted to point cloud $\mathcal{P}$), and the output is a complete articulation structure $\mathcal{A} = (P, S, K, M)$: the number of parts $P$, face-to-part segmentation mapping $S$, kinematic tree $K$, and motion constraints $M$ (motion type, direction, revolute axis, motion range). The model consists of a Transformer backbone with multiple specialized decoder heads, achieving feed-forward inference in approximately 10 seconds after end-to-end training.

### Key Designs

1. **Part Articulation Transformer**:

    - **Function**: Extracts latent representations of point tokens and part tokens from point clouds.
    - **Mechanism**: Each point $\mathbf{p}_i$ is encoded via three separate MLPs over its coordinates, normal vectors, and PartField semantic features, which are then summed to form the point token $\tilde{\mathbf{p}}_i$. A set of $P_{max}$ learnable part queries $\mathcal{Q}$ (far exceeding the actual number of parts) is initialized. The backbone consists of $B=8$ attention blocks, each alternating between query self-attention and query-to-point cross-attention. Self-attention among point tokens is omitted to save memory, given that $N \gg P_{max}$.
    - **Design Motivation**: DETR-style part queries elegantly address the unknown part count problem, while the Transformer's attention mechanism flexibly captures inter-part and point-to-part relationships. PartField features introduce 2D semantic part priors that enhance generalization to novel categories.

2. **Multi-Head Decoder**:

    - **Function**: Decodes each articulation attribute from point/part tokens independently.
    - **Mechanism**: **Part segmentation** uses MLP $h_S(\tilde{\mathbf{p}}_i, \tilde{\mathbf{q}}_j)$ to predict an $N \times P_{max}$ logit matrix. **Kinematic tree** uses MLP $h_K(\tilde{\mathbf{q}}_i, \tilde{\mathbf{q}}_j)$ to predict a $P_{max} \times P_{max}$ parent-child probability matrix, from which Edmonds' algorithm extracts the maximum spanning tree at inference time. **Motion type, range, and prismatic direction** are each predicted from part tokens via independent MLPs.
    - **Design Motivation**: Decomposing articulation structure into independently predictable attributes, each decoded by a dedicated MLP, simplifies the learning problem.

3. **Over-parameterized Revolute Axes**:

    - **Function**: Accurately predict the direction and position of revolute axes.
    - **Mechanism**: The revolute axis direction $\tilde{\mathbf{d}}_{ra}^i$ is directly predicted by an MLP from the part token and normalized. However, the axis position is not regressed directly (which is prone to overfitting); instead, each 3D point belonging to the part votes via MLP $h_{cp}(\tilde{\mathbf{p}}_j, \tilde{\mathbf{q}}_i)$ to predict its orthogonal projection onto the revolute axis. The median of all votes is taken as the final axis position at inference time.
    - **Design Motivation**: Revolute axis directions are typically axis-aligned and relatively easy to learn, whereas axis position prediction requires high precision. The over-parameterized per-point voting scheme leverages spatial priors for more robust axis position estimation, while median aggregation provides robustness against outliers.

### Loss & Training

The multi-task loss is $\mathcal{L} = \mathcal{L}_S + \mathcal{L}_K + \mathcal{L}_M$. Part segmentation uses cross-entropy loss, and the kinematic tree uses binary cross-entropy. Motion constraint losses include cross-entropy for motion type, L1 losses for prismatic/revolute ranges and directions, and L1 losses for revolute axis direction and position. During training, Hungarian matching assigns the $P_{max}$ predicted part queries to $P$ ground-truth parts (following DETR). Training data comes from PartNet-Mobility (3,800 objects, 50 categories) and GRScenes; each iteration randomly samples an articulation state and computes PartField features online. The model is trained with the AdamW optimizer at a global batch size of 128 on 8 H100 GPUs for 100K iterations.

## Key Experimental Results

### Main Results (Articulated Part Segmentation)

| Method | Lightwheel gIoU↑ | Lightwheel PC↓ | PartNet gIoU↑ | PartNet PC↓ |
|---|---|---|---|---|
| Naive Baseline | 0.018 | 0.285 | 0.296 | 0.210 |
| PartField† | 0.079 | 0.106 | 0.183 | 0.123 |
| SINGAPO (1@10)† | -0.050 | 0.221 | 0.271 | 0.117 |
| Articulate AnyMesh† | 0.172 | 0.190 | 0.383 | 0.104 |
| **Particulate†** | **0.332** | **0.168** | **0.880** | **0.003** |

†: With mesh connected component refinement

### Articulation Motion Prediction (Full Articulated Geometry Comparison)

| Method | Lightwheel gIoU↑ | Lightwheel OC↓ | PartNet gIoU↑ | PartNet OC↓ |
|---|---|---|---|---|
| SINGAPO (1@10)† | -0.056 | 0.019 | 0.264 | 0.041 |
| Articulate AnyMesh† | 0.158 | 0.010 | 0.378 | 0.022 |
| **Particulate†** | **0.305** | **0.009** | **0.843** | **0.003** |

### Ablation Study

| Configuration | gIoU↑ | Notes |
|---|---|---|
| Full model | 0.332 | Complete model (Lightwheel, with connectivity) |
| w/o PartField features | Lower | Generalization degrades without semantic features |
| w/o connected comp. refinement | 0.183 | Significant drop without mesh connected component refinement |
| w/o over-parameterized axis | Lower | Direct axis position regression causes offset |

### Key Findings
- Particulate achieves a gIoU of 0.880 on PartNet-Mobility, far exceeding the second-best method Articulate AnyMesh at 0.383.
- The advantage remains clear on the more challenging Lightwheel dataset (0.332 vs. 0.172).
- PartField and P3SAM predict semantic parts rather than articulation parts, resulting in definition mismatches.
- VLM-based methods (Articulate AnyMesh) cannot handle internally invisible parts (e.g., the turntable inside a microwave).
- Particulate generalizes well to AI-generated 3D assets (objects generated by Hunyuan3D).

## Highlights & Insights
- **The over-parameterized voting mechanism for revolute axes is highly elegant**: having each point vote for the axis position and aggregating via median cleverly exploits the geometric constraint that the axis position must be consistent across all orthogonal projections, avoiding the overfitting issues of direct regression.
- **DETR-style part queries adapted for articulation prediction**: learnable part queries elegantly handle the unknown part count problem while simultaneously enabling prediction of inter-part kinematic relationships.
- **Transferable data augmentation strategy**: randomly sampling different articulation states each iteration effectively provides extensive data augmentation, enabling the model to understand objects across diverse poses.

## Limitations & Future Work
- The constraint $P_{max}=16$ limits the maximum number of parts, which may be insufficient for highly complex articulated objects (e.g., robotic arms).
- Only rigid articulation joints (revolute/prismatic) are considered; soft deformations are not supported.
- Training data comprises only 3,800 objects; scaling up the dataset may further improve generalization.
- PartField feature computation at inference time introduces additional overhead.
- The newly introduced Lightwheel benchmark contains only 243 objects, limiting its scale.

## Related Work & Insights
- **vs. SINGAPO**: SINGAPO assembles articulated objects via part retrieval, constrained by part library coverage and trained on only a few categories. Particulate predicts end-to-end without relying on retrieval.
- **vs. Articulate AnyMesh**: The latter uses VLM-based articulation reasoning with good generality, but requires 15 min/object and cannot handle internal parts. Particulate completes inference in 10 seconds and handles internal structures.
- **vs. PartField**: PartField performs semantic segmentation rather than articulation segmentation, and the two definitions differ. Particulate uses PartField as an input feature, combining the strengths of both approaches.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First method to feed-forwardly predict complete articulation structures from static 3D meshes; the over-parameterized revolute axis design is creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Two datasets, detailed ablations, new evaluation protocol, and rich visualizations; comparisons are very comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Formal definitions are clear, method descriptions are thorough, and the Related Work summary table is well-organized.
- **Value**: ⭐⭐⭐⭐ Significant practical impact for 3D articulation understanding; combined with 3D generative models, it enables end-to-end object creation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Speed3R: Sparse Feed-forward 3D Reconstruction Models](speed3r_sparse_feed-forward_3d_reconstruction_models.md)
- [\[CVPR 2026\] PanoVGGT: Feed-Forward 3D Reconstruction from Panoramic Imagery](panovggt_feed-forward_3d_reconstruction_from_panoramic_imagery.md)
- [\[CVPR 2026\] VGG-T3: Offline Feed-Forward 3D Reconstruction at Scale](vgg-t3_offline_feed-forward_3d_reconstruction_at_scale.md)
- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)
- [\[CVPR 2026\] PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis](physgm_large_physical_gaussian_4d_synthesis.md)

</div>

<!-- RELATED:END -->
