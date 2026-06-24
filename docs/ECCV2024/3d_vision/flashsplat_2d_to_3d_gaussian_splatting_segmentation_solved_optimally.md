---
title: >-
  [Paper Note] FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally
description: >-
  [ECCV 2024][3D Vision][3D Gaussian Splatting] This paper reformulates the 2D-to-3D segmentation problem of 3D Gaussian Splatting as an integer linear program. Leveraging the linearity of alpha blending, it obtains a closed-form optimal solution in just 30 seconds, achieving a 50x speedup over existing methods.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "3D Segmentation"
  - "Linear Programming"
  - "Optimal Solution"
  - "Object Removal"
date: 2026-05-08
content_hash: cda3913b1c9bdd55
---

# FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally

**Conference**: ECCV 2024  
**arXiv**: [2409.08270](https://arxiv.org/abs/2409.08270)  
**Code**: [https://github.com/florinshen/FlashSplat](https://github.com/florinshen/FlashSplat)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, 3D Segmentation, Linear Programming, Optimal Solution, Object Removal

## TL;DR

This paper reformulates the 2D-to-3D segmentation problem of 3D Gaussian Splatting as an integer linear program. Leveraging the linearity of alpha blending, it obtains a closed-form optimal solution in just 30 seconds, achieving a 50x speedup over existing methods.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3D-GS) has rapidly emerged as an efficient 3D scene representation, but lifting segmentation results from 2D masks to 3D (i.e., assigning labels to each 3D Gaussian) remains a crucial and unresolved problem.

**Limitations of Prior Work**: Existing methods (e.g., SAGA, Gaussian Grouping) rely on iterative gradient descent optimization to train the label/feature of each Gaussian, requiring 30,000 iterations and 18-37 minutes of extra training time, which easily suffers from sub-optimal solutions.

**Key Challenge**: Intuitively, 3D-GS segmentation should be simple—the scene is already reconstructed, and it is merely a label assignment problem. However, prior methods treat it as a learning task requiring heavy optimization, which is highly inefficient.

**Goal**: Discover the mathematical structure of the 3D-GS segmentation problem and provide a globally optimal and highly efficient solver.

**Key Insight**: In a reconstructed 3D-GS scene, the rendered 2D mask is a **linear function** of each Gaussian's label; thus, the optimal label assignment can be solved in closed form via linear programming.

**Core Idea**: Capitalize on the property that $\alpha_i T_i$ is a constant in alpha blending to reformulate 3D-GS segmentation as a linear program, achieving the global optimum in one step through weighted majority voting.

## Method

### Overall Architecture

Input: Reconstructed 3D-GS scene $\{G_i\}$ + multi-view 2D binary/instance masks $\{M^v\}$.
Goal: Assign label $P_i$ to each 3D Gaussian $G_i$.
Pipeline: (1) Use the 3D-GS rasterizer to traverse all mask views, accumulating the weighted contribution $A_0, A_1$ of each Gaussian to foreground/background pixels; (2) Determine the optimal label assignment via argmax in a single step; (3) Optionally introduce a background bias $\gamma$ to reduce noise. The entire process takes about 30 seconds.

### Key Designs

1. **Binary Segmentation as ILP**:

    - **Function**: Strictly formulate the 2D-to-3D segmentation problem as an integer linear program (ILP).
    - **Mechanism**: In 3D-GS rendering, the pixel value $X = \sum_i x_i \alpha_i T_i$, where $\alpha_i T_i$ is constant once scene reconstruction is completed. Treating the label $P_i$ as the attribute $x_i$, the optimization goal is to minimize the MAE between the rendered mask and the given mask:
    $$\min_{\{P_i\}} \mathcal{F} = \sum_{v \in L} \sum_{M_{jk}^v \in M^v} \left| \sum_i P_i \alpha_i T_i - M_{jk}^v \right|, \quad P_i \in \{0,1\}$$
   By the properties of alpha blending (**Lemma 1**): $0 \leq \sum P_i \alpha_i T_i \leq \sum \alpha_i T_i \leq 1$, the absolute value can be expanded as:
    $$\min \mathcal{F} = C + \sum_i P_i (A_0^i - A_1^i)$$
   where $A_n^i = \sum_{v,j,k} \alpha_i T_i \mathbb{I}(M_{jk}^v, n)$ is the cumulative contribution of the $i$-th Gaussian over all pixels labeled as $n$.
    - **Optimal Solution**: $P_i = \arg\max_n A_n$, which is a weighted majority vote—if a Gaussian's weighted contribution to the foreground pixels is larger than its contribution to the background pixels, it is labeled as foreground.
    - **Design Motivation**: Because the objective function is linear with respect to $P_i$ (each $P_i$ is independent), and $P_i \in \{0,1\}$, it can be solved independently for each Gaussian in parallel. This yields the globally optimal solution.

2. **Regularized ILP with Background Bias**:

    - **Function**: Handle noise in 2D masks by introducing a background bias parameter $\gamma \in [-1, 1]$.
    - **Mechanism**: First, perform L1-normalization on the contributions $\bar{A_e} = A_e / \sum_t A_t$, then adjust $\hat{A_0} = \bar{A_0} + \gamma$. A positive $\gamma$ tends to classify more Gaussians as background (reducing foreground noise), while a negative $\gamma$ retains more foreground Gaussians.
    - **Design Motivation**: 2D masks generated by models like SAM often contain noise, which mistakenly labels background Gaussians as foreground and causes jagged boundaries. $\gamma$ provides a simple yet flexible control knob for noise, and since $\{A_e\}$ only needs to be computed once, adjusting $\gamma$ is instantaneous (0.4ms).

3. **Binary to Scene Segmentation**:

    - **Function**: Extend the method to segment all object instances in the scene simultaneously.
    - **Mechanism**: Re-interpret multi-instance segmentation as a combination of multiple binary segmentations. For target instance $t$, treat all other instances as background: $A_0 = A_{\text{others}} = \sum_{e \neq t} A_e$, and then $P_i = \arg\max_n \{A_0, A_t\}$. The set $\{A_e\}$ only needs to be accumulated once, and segmenting each instance takes only one argmax.
    - **Design Motivation**: 3D Gaussians are naturally not mutually exclusive—a Gaussian can contribute to multiple objects simultaneously (approximately 20% of Gaussians are shared by multiple objects). Thus, allowing overlapping Gaussian subsets for different instances is reasonable.

4. **Depth-Guided Novel View Mask Rendering**:

    - **Function**: Render 2D masks of the segmentation results from novel views.
    - **Mechanism**: In binary segmentation, only the accumulated alpha values $\rho_{jk}$ of foreground Gaussians are rendered and thresholded by $\tau$ to obtain the mask. In scene segmentation, if the alpha values of multiple instances exceed the threshold, the closest instance is selected using the rendered depth.
    - **Design Motivation**: The semi-transparent nature of 3D Gaussians causes many holes in foreground masks if quantization is not applied (the alpha contribution of background Gaussians may exceed that of foreground Gaussians). Depth guidance resolves the ambiguity of multi-instance overlaps.

### Implementation Details

- The core computation is implemented in CUDA kernels: utilizing tile-based rasterization to traverse all masks and accumulate $A_e$ via atomic operations for each Gaussian.
- Accumulating $\{A_e\}$ takes around 26 seconds (traversing all pixels in all views).
- Argmax assignment takes only 0.4ms.
- Peak GPU memory usage is only 8GB, which is half of SAGA.

## Key Experimental Results

### Main Results: Quantitative Comparison on NVOS Dataset

| Method | mIoU (%)↑ | mAcc (%)↑ |
|------|----------|----------|
| NVOS | 39.4 | 73.6 |
| ISRF | 70.1 | 92.0 |
| SGISRF | 83.8 | 96.4 |
| SA3D | 90.3 | 98.2 |
| SAGA | 90.9 | 98.3 |
| **FlashSplat** | **91.8** | **98.6** |

### Computational Cost Comparison

| Method | Extra Training Time | Optimization Steps | Single Segmentation Time | Peak GPU Memory |
|------|------------|---------|------------|---------|
| SAGA | 18 mins | 30000 | 0.5s | 15G |
| Gaussian Grouping | 37 mins | 30000 | 0.3s | 34G |
| **FlashSplat** | **26s** | **1** | **0.4ms** | **8G** |

### Ablation Study

| Background Bias \gamma | mIoU (Truck Scene) | Description |
|----------|-----------------|------|
| -0.8 | 82.4% | Excessive foreground noise |
| -0.4 | 89.6% | - |
| 0 | 92.3% | No bias baseline |
| 0.4 | **94.2%** | Optimal, reduces SAM noise |
| 0.8 | 93.8% | Slightly over-cleaned |

### Key Findings

- FlashSplat achieves an mIoU of 91.8% on the NVOS dataset, outperforming all NeRF and 3D-GS baselines.
- The speedup is about **50x** (26 seconds vs. 18-37 minutes), with **750x** speedup in single segmentation time (0.4ms vs. 0.3-0.5s).
- Only about 10% of the view masks are needed to produce good segmentation results (1/8 views in 360-degree scenes are still usable).
- Background bias of $\gamma=0.4$ improves mIoU from 92.3% to 94.2% on the Truck scene.
- Around 20% of 3D Gaussians are shared by 2 or more objects—3D Gaussians are naturally not mutually exclusive.
- Object removal quality is superior to Gaussian Grouping, which exhibits severe artifacts near the removal regions.

## Highlights & Insights

- The **elegance of problem modeling** is the biggest highlight of this paper—turning an optimization problem into one with a closed-form solution. The key insight is that $\alpha_i T_i$ is constant after alpha blending, making the objective function linear with respect to labels.
- Although the derivation of Lemma 1 is simple, it is crucial—ensuring that the rendered mask values reside within $[0,1]$, allowing the absolute value to be expanded.
- The design of the parameter $\gamma$ is intuitive and practical—once $\{A_e\}$ is computed, users can interactively adjust $\gamma$ to see the segmentation changes in real-time.
- Decomposing scene segmentation into multiple binary segmentations and allowing overlap shows a correct understanding of 3D-GS characteristics.
- The code is concise—the PyTorch implementation of the core algorithm takes only about 20 lines.

## Limitations & Future Work

- Requires traversing all pixels of all masks to accumulate $\{A_e\}$, which may present scalability issues for extremely large scenes.
- 3D-GS lacks explicit geometric supervision, causing the learned geometry to be partially inaccurate; depth-guided mask rendering sometimes yields fuzzy results.
- Relies on the quality of 2D segmentation models like SAM and the accuracy of mask association (video trackers).
- For forward-facing views (e.g., LLFF dataset), object removal exposes unobserved background regions.
- Segmentation at object boundaries might not be fine-grained enough—linear programming assumes independence for each Gaussian's contribution, ignoring spatial continuity priors.
- Not combined with semantic/open-vocabulary segmentation methods.

## Related Work & Insights

- **vs SAGA**: SAGA requires training extra feature dimensions + multiple loss functions (about 30,000 steps) for each 3D Gaussian, which is an over-engineered solution. FlashSplat proves that a closed-form solution exists, completely eliminating the need for iterative optimization.
- **vs Gaussian Grouping**: Distills object features + trains classifiers after associating 2D masks using a video tracker, which is more complex but not necessarily better. FlashSplat directly goes from masks to 3D labels without learning features.
- **vs SAGS**: Projects 3D Gaussian centers to 2D masks to determine foreground/background, which is training-free but overly simplified—ignoring spatial extent and alpha blending weights of Gaussians. FlashSplat considers the full rendering contribution.
- **Insights**: When analyzing attribute assignment problems for reconstructed representations, one should first inspect the mathematical structure of the objective function—a linear structure implies a closed-form solution, bypassing gradient descent.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The insight of transforming an optimization problem into a closed-form solution is brilliant, simple, elegant, and globally optimal.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative + qualitative + ablation + computational cost comparisons are comprehensive, though the quantitative evaluation is somewhat limited to NVOS (8 forward-facing scenes).
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations, and the extension from binary to scene segmentation is natural, though there is some redundancy in some parts.
- Value: ⭐⭐⭐⭐⭐ 50x speedup + global optimum + less GPU memory + better quality. It holds high practical value and is a near-definitive work for 3D-GS segmentation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] 3×2: 3D Object Part Segmentation by 2D Semantic Correspondences](3x2_3d_object_part_segmentation_by_2d_semantic_correspondenc.md)
- [\[ECCV 2024\] Learning 3D Geometry and Feature Consistent Gaussian Splatting for Object Removal](learning_3d_geometry_and_feature_consistent_gaussian_splatting_for_object_remova.md)
- [\[ECCV 2024\] Click-Gaussian: Interactive Segmentation to Any 3D Gaussians](click-gaussian_interactive_segmentation_to_any_3d_gaussians.md)
- [\[ECCV 2024\] GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting](gaussianimage_1000_fps_image_representation_and_compression_by_2d_gaussian_splat.md)
- [\[ECCV 2024\] Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic Integration](analytic-splatting_anti-aliased_3d_gaussian_splatting_via_analytic_integration.md)

</div>

<!-- RELATED:END -->
