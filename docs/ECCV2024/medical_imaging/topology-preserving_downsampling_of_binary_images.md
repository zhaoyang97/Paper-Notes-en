---
title: >-
  [Paper Note] Topology-Preserving Downsampling of Binary Images
description: >-
  [ECCV 2024][Medical Imaging][Binary image downsampling] This paper proposes the first topology-preserving binary image downsampling method based on discrete optimization (integer programming). By encoding the black-and-white decisions of downsampled pixels as Boolean variables, framing topology preservation as hard constraints, and using similarity to the original image as the objective function, the method guarantees that the downsampled results have the exact same Betti num…
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "Binary image downsampling"
  - "topology preservation"
  - "discrete optimization"
  - "integer programming"
  - "segmentation masks"
date: 2026-05-08
content_hash: 142f42bc5126f61b
---

# Topology-Preserving Downsampling of Binary Images

**Conference**: ECCV 2024  
**arXiv**: [2407.17786](https://arxiv.org/abs/2407.17786)  
**Code**: [https://github.com/pengchihan/BinaryImageDownsampling](https://github.com/pengchihan/BinaryImageDownsampling)  
**Area**: Medical Images  
**Keywords**: Binary image downsampling, topology preservation, discrete optimization, integer programming, segmentation masks

## TL;DR
This paper proposes the first topology-preserving binary image downsampling method based on discrete optimization (integer programming). By encoding the black-and-white decisions of downsampled pixels as Boolean variables, framing topology preservation as hard constraints, and using similarity to the original image as the objective function, the method guarantees that the downsampled results have the exact same Betti numbers (number of connected components and holes) as the original image, while maintaining competitive pixel-level similarity compared to traditional methods.

## Background & Motivation
Binary images (containing only foreground and background pixels) are widely used in applications such as segmentation masks, document analysis, machine vision, and map encoding. Unlike grayscale/color images, binary images naturally correspond to 2D graph structures, where topology (the number of connected components $\beta_0$ and holes $\beta_1$, i.e., the zeroth and first Betti numbers) plays a critical role—for instance, in medical image segmentation, topological correctness directly affects diagnostic accuracy.

The Key Challenge is that: **all existing binary image downsampling methods (bilinear/bicubic interpolation + thresholding, various pooling, and ACN methods) cannot guarantee topological invariance**. Even if the original patch corresponding to a downsampled pixel consists entirely of black pixels, it might need to be set to white for topological correctness. This demonstrates that correct downsampling decisions require global information, making any local method inherently unviable.

Key Insight: This work formulates the topology-preserving downsampling problem as a **discrete optimization problem**—the color of each macro-pixel is represented as a Boolean variable, topology preservation is formulated as hard constraints, and similarity to the original image is defined as the objective function. Consequently, any feasible solution is automatically guaranteed to be topologically correct, and the optimizer will find the one most similar to the original image among all topologically correct solutions. Furthermore, infeasible problems (where no topologically correct solution exists) can be automatically detected.

## Method

### Overall Architecture
Given an input $W \times H$ binary image, it is downsampled by a factor of $A \times B$. The pipeline consists of: (1) analyzing the connected components and Region Adjacency Graph (RAG) of the original image; (2) creating Boolean variables for each macro-pixel-component pair; (3) constructing four sets of constraints to ensure topology preservation; (4) designing an objective function to maximize similarity to the original image; and (5) solving the problem using an integer programming solver (Gurobi).

### Key Designs
1. **Boolean Variables and Coverage Modeling**: 
    - For each connected component $C_i$ (including black and white components) of the original image, collect all candidate macro-pixels $P_j^i$ that could cover it.
    - The coverage area can be expanded beyond the macro-pixel block (with distance thresholds $dx = \lfloor A/4 \rfloor$, $dy = \lfloor B/4 \rfloor$) to increase flexibility.
    - Each $P_j^i$ is a Boolean variable indicating whether the macro-pixel is assigned to component $C_i$.
    - Design Motivation: Simple black/white decisions cannot encode which component covers a macro-pixel, whereas component-level coverage modeling serves as the foundation for topological constraints.

2. **Four Sets of Topology-Preserving Hard Constraints**: 
    - **Complete Non-Overlapping Coverage Constraint**: Each macro-pixel is covered by exactly one component $\rightarrow \sum P_j^i(X,Y) = 1$.
    - **Non-Empty Component Constraint**: Each component is covered by at least one macro-pixel $\rightarrow \sum P_j^i \geq 1$ (otherwise the component disappears).
    - **Local Neighborhood Correctness Constraint**: Different black components cannot be adjacent in an 8-neighborhood, and different white components cannot be adjacent in a 4-neighborhood $\rightarrow (P_0^m + P_1^m) \leq 1$.
    - **Boundary-Preserving Constraint (Key Innovation)**: Ensures that exactly one closed boundary loop exists between each pair of adjacent black and white components.
        - Enumerate 12 possible corner configurations, each having left-hand side (LHS) and right-hand side (RHS) macro-pixels.
        - A corner exists if and only if all RHS macro-pixels belong to the black component and all LHS macro-pixels belong to the white component.
        - Introduce a "Dist" integer variable and a "last" Boolean variable to ensure that corners on the boundary are arranged in monotonically increasing distances, with exactly one "last" exception.
        - This effectively rules out the possibility of multiple closed loops.
    - Design Motivation: Inspired by the "no-island" constraint in network design, this preserves global topological correctness by encoding the single-loop structure of boundaries.

3. **Objective Function Design**: 
    - Calculate a score $S_j^i$ for each candidate macro-pixel: points are awarded when original pixels overlap with an active macro-pixel from the same component, with the reward inversely proportional to distance.
    - Specifically, each original pixel within a $(2dx+1) \times (2dy+1)$ window contributes 1 point to the overlapping candidate macro-pixels.
    - Maximize the total score of all active macro-pixels: $\max \sum S_j^i \times P_j^i$.
    - Design Motivation: This simple score design naturally encourages macro-pixels to cover positions that overlap most with their corresponding components in the original image.

### Loss & Training
This paper does not involve deep learning training. The integer programming problem is solved using the Gurobi 11.0 solver.

Supplement: Dilation baseline method for comparison—
- Creates a topologically correct initial representation (points or bounding boxes) for each component in the downsampled image.
- Iteratively dilates black components and then white components.
- Guarantees topological correctness, but the greedy strategy leads to poor similarity.

## Key Experimental Results

### Main Results
Comparison of downsampling on 542 segmentation masks ($512 \times 512$) from the CNCB dataset:

| Method | Factor | IoU↑ | Dice↑ | Betti Error↓ | PH Dist↓ |
|------|--------|------|-------|-----------|---------|
| Bicubic | 2 | 93.08% | 96.34% | 0.061 | 0.018 |
| Pooling | 2 | 93.41% | 96.53% | 0.046 | 0.015 |
| ACN | 2 | 91.78% | 95.61% | 0.092 | 0.051 |
| Dilation | 2 | 59.71% | 73.78% | **0** | 0.026 |
| **Ours** | **2** | **93.10%** | **96.34%** | **0** | **0.005** |
| Bicubic | 8 | 73.27% | 83.46% | 0.339 | 0.091 |
| Pooling | 8 | 73.33% | 83.21% | 0.404 | 0.115 |
| Dilation | 8 | 53.40% | 68.07% | **0** | 0.051 |
| **Ours** | **8** | **73.31%** | **83.55%** | **0** | **0.030** |

### Ablation Study
Impact of different downsampling factors on PH calculation speedup:

| Image Size | Original (512) | 256 (×2) | 128 (×4) | 64 (×8) | 32 (×16) |
|---------|-----------|---------|---------|--------|---------|
| PH Computation Time (s) | 0.965 | 2.157 | 0.864 | 0.624 | 0.568 |
| PH Distance | - | 0.005 | 0.012 | 0.030 | 0.062 |

Shortest path computation speedup (average of 200 Dijkstra runs):

| Method| Factor | Distance Error↓ | FP↓ | FN↓ | Time (s)↓ |
|------|--------|---------|-----|-----|---------|
| Bicubic | 4 | 1.65 | 0.00 | 0.78 | 2.49 |
| Pooling | 4 | 2.63 | 0.00 | 0.00 | 2.96 |
| Dilation | 4 | 0.38 | 0.00 | 0.00 | 3.66 |
| **Ours** | **4** | **0.17** | **0.00** | **0.00** | 3.63 |
| Bicubic | 8 | 5.05 | 0.00 | 2.12 | 0.95 |
| **Ours** | **8** | **0.19** | **0.00** | **0.00** | 1.34 |
| Bicubic | 16 | 11.57 | 0.00 | 1.85 | 0.96 |
| **Ours** | **16** | **10.47** | **0.00** | **0.00** | 0.99 |

### Key Findings
- **Topology guarantees and similarity can be achieved simultaneously**: The proposed method achieves zero Betti number error while keeping IoU/Dice almost identical to the best non-topology-preserving methods.
- **Consistently lowest PH distance**: Even on persistent homology, a continuous topological metric, the proposed method outweighs all other methods.
- **Dilation method achieves correct topology but poor similarity**: Its IoU is only 59.71% ($\times 2$) vs 93.10% (Ours), demonstrating that greedy strategies cannot satisfy both objectives.
- **Infeasible cases are extremely rare**: Only 2, 1, 3, and 8 infeasible cases occurred for $\times 2, \times 4, \times 8,$ and $\times 16$, respectively, meaning a topologically correct solution almost always exists.
- **Computation speed increases with larger downsampling factors**: As candidate macro-pixels decrease, computation takes $<1$ second for factors $\ge 4$, enabling interactive usage.
- **Shortest path application highlights unique topology guarantee**: Non-topology-preserving methods produce false positives and false negatives, making pathfinding results unreliable.
- **PH calculation speedup yields net time savings for factors $\geq 4$**: The total time of downsampling + computing PH is less than the time of directly computing PH on the original image.

## Highlights & Insights
- **The power of formal problem definition**: Formulating the seemingly impossible task of "guaranteeing topological correctness" as hard constraints in integer programming is elegant and powerful.
- **Innovative design of boundary-preserving constraints**: Inspired by the "no-island" constraint, ensuring a single closed boundary loop via distance integer variables and a "last" marker is the core technical contribution of this work.
- **Leveraging infeasibility**: When no feasible solution exists, the IP solver detects it immediately, which is far more reliable than heuristic approaches.
- **Practical application-driven**: The two applications of PH speedup and shortest-path calculation demonstrate the unique practical value of topological guarantees.
- An counter-intuitive finding: even when a corresponding block is entirely composed of black pixels, the downsampled pixel may need to be set to white—a situation that only global optimization can handle.
- Although the enumeration of 12 corner configurations appears heavily engineered, it provides a sound and complete formulation for the problem.

## Limitations & Future Work
- **Inability to handle extremely fine structures**: Thin structures like road networks and blood vessels may not be satisfactorily preserved, as the objective function treats black and white components with equal preference.
- **Dependency on IP solver**: The reliance on the commercial Gurobi solver increases deployment barriers.
- **Constraints on downsampling factors**: $A$ and $B$ must divide $W$ and $H$, which limits flexibility.
- **No support for continuous or non-uniform downsampling**: The method only supports integer-factor downsampling.
- Future work: (1) designing targeted objective functions for fine structures, (2) integrating constraints into the multi-scale representations of neural segmentation networks, and (3) extending the method to 3D volumetric data.

## Related Work & Insights
- Complementary to topological loss functions in neural segmentation (e.g., PH-based loss, homotopy warping loss): this method provides post-processing topological guarantees, whereas those methods introduce topological prior during training.
- Research on topology-preserving affine transformations of binary images only addresses translation/rotation but lacks scaling, a gap this paper fills via downscaling (downsampling).
- ACN methods aim for topology preservation but cannot guarantee it, highlighting the fundamental limitation of local methods.
- Provides a new tool for visualizing and efficiently processing segmentation masks—enlarged macro-pixels make micro-details easier for the human eye to recognize.
- An exemplary application of integer programming in computer vision, showcasing the indispensable nature of traditional optimization methods for specific problems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (The first to address the topology-preserving downsampling problem, with highly original formulations and constraint designs)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Evaluated on medical images and two application scenarios, with comprehensive quantitative metrics, but limited to a single dataset)
- Writing Quality: ⭐⭐⭐⭐ (Mathematically rigorous, but notation-heavy, requiring readers to have a background in topology)
- Value: ⭐⭐⭐⭐ (Addresses a very clear need, though the application scenario remains relatively niche)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Learning with Preserving for Continual Multitask Learning](../../AAAI2026/medical_imaging/learning_with_preserving_for_continual_multitask_learning.md)
- [\[ECCV 2024\] A Rotation-Invariant Texture ViT for Fine-Grained Recognition of Esophageal Cancer Endoscopic Ultrasound Images](a_rotation-invariant_texture_vit_for_fine-grained_recognition_of_esophageal_canc.md)
- [\[CVPR 2025\] TopoCellGen: Generating Histopathology Cell Topology with a Diffusion Model](../../CVPR2025/medical_imaging/topocellgen_generating_histopathology_cell_topology_with_a_diffusion_model.md)
- [\[ICML 2026\] DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning](../../ICML2026/medical_imaging/dp-kfc_data-free_preconditioning_for_privacy-preserving_deep_learning.md)
- [\[AAAI 2026\] Graph-Theoretic Consistency for Robust and Topology-Aware Semi-Supervised Histopathology Segmentation](../../AAAI2026/medical_imaging/graph-theoretic_consistency_for_robust_and_topology-aware_semi-supervised_histop.md)

</div>

<!-- RELATED:END -->
