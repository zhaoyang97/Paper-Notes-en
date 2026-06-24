---
title: >-
  [Paper Note] The Four Color Theorem for Cell Instance Segmentation
description: >-
  [ICML 2025][Medical Imaging][Four Color Theorem] This paper introduces the four-color theorem to cell instance segmentation, treating each cell as a "country" and the background as the "ocean", thereby replacing instance segmentation with a constrained 4-class semantic segmentation task. It designs a progressive training strategy and an encoding transformation method to resolve the non-uniqueness of four-color coding, achieving SOTA performance across diverse imaging modaliti…
tags:
  - "ICML 2025"
  - "Medical Imaging"
  - "Four Color Theorem"
  - "Cell Instance Segmentation"
  - "Semantic Segmentation"
  - "Greedy Coloring"
  - "Progressive Training"
  - "Encoding Transformation"
date: 2026-05-08
content_hash: ef392be1272234b1
---

# The Four Color Theorem for Cell Instance Segmentation

**Conference**: ICML 2025  
**arXiv**: [2506.09724](https://arxiv.org/abs/2506.09724)  
**Code**: [GitHub](https://github.com/zhangye-zoe/FCIS)  
**Area**: Medical Image Segmentation / Instance Segmentation / Graph Theory  
**Keywords**: Four Color Theorem, Cell Instance Segmentation, Semantic Segmentation, Greedy Coloring, Progressive Training, Encoding Transformation  

## TL;DR

This paper introduces the four-color theorem to cell instance segmentation, treating each cell as a "country" and the background as the "ocean", thereby replacing instance segmentation with a constrained 4-class semantic segmentation task. It designs a progressive training strategy and an encoding transformation method to resolve the non-uniqueness of four-color coding, achieving SOTA performance across diverse imaging modalities while significantly reducing model complexity.

## Background & Motivation

Core limitations of existing cell instance segmentation methods:

**Detection-based methods** (Mask R-CNN, DoNet): Depend on object detection frameworks, leading to high computational complexity and potential miss-detections of elongated cells and overlapping regions.

**Contour prediction methods** (UNet, DCAN): Introduce boundary classes to distinguish instances, but performance is highly sensitive to contour threshold settings.

**Distance map methods** (StarDist, HoverNet, CellViT): Utilize multiple decoding branches combined with complex post-processing, significantly increasing computational overhead.

The four-color theorem states that only four colors are required to ensure adjacent regions are colored differently. Applying this theorem to cell images allows the instance segmentation task to be transformed into a constrained 4-class semantic segmentation problem.

## Method

### Greedy Coloring Algorithm

A cell graph $G = (V, E)$ is constructed, where nodes represent cells and edges denote adjacency. The minimum available color is greedily assigned to each node:

$$C(v) = \min(\mathcal{C} \setminus \mathcal{C}_{used})$$

where $\mathcal{C} = \{1, 2, 3, 4\}$ and $\mathcal{C}_{used} = \{C(u) | u \in N(v), C(u) \neq 0\}$.

### Analysis of Encoding Non-uniqueness

Three equivalent transformations can cause training instability:
- **Replacement**: A color is replaced by another color.
- **Permutation**: The colors of two cells are swapped.
- **Rule Modification**: Increasing the number of colors used.

### Global Optimality of Greedy Coloring

**Theorem 1**: When the cell graph $G$ is a planar graph, with maximum degree $\Delta(G) \leq 4$, and is arranged in a chain or rectangular layout, greedy coloring achieves global optimality: $\chi_{greedy}(G) = \chi(G)$.

Empirical statistics reveal that the vast majority of cell images require only 2 colors, and almost no images require 4 colors.

### Progressive Training Strategy

**Step 1: Foreground/Background Binary Classification**

Among the 5-channel outputs: the 1st channel represents background probability $\hat{Y}_b$, and the subsequent 4 channels are fused via convolution to obtain the foreground probability $\hat{Y}_f$:

$$\mathcal{L}_{sem} = CE(\hat{Y}_{b,i}, Y_i) + Dice(\hat{Y}_{b,i}, Y_i)$$

**Step 2: Negative Sampling Constraint for Adjacent Cells**

Features of adjacent cell pairs $(v_i, v_j)$ are sampled to impose an orthogonality constraint:

$$\mathcal{L}_{ort} = \frac{1}{|E|}\sum_{(v_i, v_j) \in E} \text{Cos}(F_i, F_j)$$

### Encoding Transformation Method

**Theorem 2**: A mapping $f: \mathbf{P} \to \mathbf{C}$ exists between the network-predicted encoding matrix $\mathbf{P}$ and the greedy encoding $\mathbf{C}$.

Two convolutional layers are employed to perform the encoding transformation, mapping predictions to the minimum color representation to eliminate encoding non-uniqueness:

$$\mathcal{L}_{cls} = CE(\hat{Y}_t, Y_f) + Dice(\hat{Y}_t, Y_f)$$

### Total Loss

$$\mathcal{L}_{total} = \mathcal{L}_{sem} + \lambda_1 \mathcal{L}_{ort} + \lambda_2 \mathcal{L}_{cls}$$

where $\lambda_1 = 2$ and $\lambda_2 = 1$.

## Key Experimental Results

### Model Complexity Comparison

| Method Type | Representative Method | Parameters | FLOPs |
|---------|---------|--------|-------|
| Detection-based | DoNet | 67.71M | 221.64G |
| Distance map | HoverNet | 49.70M | 192.70G |
| Distance map | CellViT | 96.81M | 124.25G |
| **Four Color Theorem** | **FCIS** | **39.75M** | **58.03G** |

### DSB2018 Dataset

| Method | DICE | AJI | PQ |
|------|------|-----|-----|
| DCAN | 0.795 | 0.676 | 0.626 |
| HoverNet | 0.898 | 0.762 | 0.762 |
| CPP-Net | 0.914 | 0.813 | 0.758 |
| **FCIS** | **Best** | **Best** | **Best** |

### Key Findings

1. FCIS substantially outperforms detection-based and distance map methods in terms of parameters and FLOPs.
2. FCIS achieves state-of-the-art or near-SOTA performance on four datasets: DSB2018, PanNuke, BBBC006v1, and YeaZ.
3. Progressive training and encoding transformation are crucial for training stability.
4. Empirical statistics validate the theoretical analysis showing that cell coloring is simpler than map coloring.

## Highlights & Insights

1. **Elegant Problem Reformulation**: Transforms instance segmentation into 4-class semantic segmentation using a classical theorem in graph theory, eliminating specialized instance-differentiation modules.
2. **Combination of Theory and Practice**: Demonstrates rigorous reasoning, from the theoretical guarantees of the four-color theorem to empirical statistics on cell distributions.
3. **Intuitive Progressive Training**: Mimics a progressive cognitive process by first learning "what is a cell" and then "how to distinguish adjacent cells".
4. **Extremely Low Complexity**: Reduces parameters and FLOPs to only 1/2 or 1/3 of distance map methods.

## Limitations & Future Work

- Edge cases in extremely dense or highly overlapping cell clusters may face issues where 3 to 4 colors are insufficient.
- Post-processing is still required to recover instance labels from the 4-color semantic map.
- The method has not been validated on 3D voxel data or ultra-high-resolution images.
- The order-sensitivity of greedy coloring is not thoroughly discussed in the theoretical section.

## Related Work & Insights

- **Detection-based**: Mask R-CNN, IRNet, DoNet
- **Contour prediction**: UNet, UNet++, DCAN, Micro-Net
- **Distance map**: StarDist, HoverNet, CellViT, CPP-Net, RepSNet
- **Graph-theoretic coloring**: Four Color Theorem, Greedy Coloring Algorithm

## Rating

⭐⭐⭐⭐ (4/5)

An exceptionally unique, innovative perspective (Four Color Theorem + instance segmentation) backed by sufficient theoretical analysis and experiments covering multiple imaging modalities. The primary limitations are that the robustness to extremely dense scenes is not fully validated, and the post-processing steps are not discussed in detail.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DISCO: Densely-overlapping Cell Instance Segmentation via Adjacency-aware Collaborative Coloring](../../ICLR2026/medical_imaging/disco_densely-overlapping_cell_instance_segmentation_via_adjacency-aware_collabo.md)
- [\[ICCV 2025\] COIN: Confidence Score-Guided Distillation for Annotation-Free Cell Segmentation](../../ICCV2025/medical_imaging/coin_confidence_score-guided_distillation_for_annotation-free_cell_segmentation.md)
- [\[ICML 2025\] iDPA: Instance Decoupled Prompt Attention for Incremental Medical Object Detection](idpa_instance_decoupled_prompt_attention_for_incremental_medical_object_detectio.md)
- [\[ICML 2025\] Do Multiple Instance Learning Models Transfer?](do_multiple_instance_learning_models_transfer.md)
- [\[CVPR 2025\] TopoCellGen: Generating Histopathology Cell Topology with a Diffusion Model](../../CVPR2025/medical_imaging/topocellgen_generating_histopathology_cell_topology_with_a_diffusion_model.md)

</div>

<!-- RELATED:END -->
