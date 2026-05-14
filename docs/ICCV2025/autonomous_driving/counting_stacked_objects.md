---
title: >-
  [Paper Note] Counting Stacked Objects
description: >-
  [ICCV 2025][Autonomous Driving][3D object counting] The paper decomposes the stacked object counting problem into two sub-problems — volume estimation and occupancy ratio estimation — solving the former via multi-view 3D…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "3D object counting"
  - "occupancy estimation"
  - "multi-view reconstruction"
  - "3D Gaussian Splatting"
  - "depth estimation"
date: 2026-05-08
content_hash: 83d24f5fc2bc2017
---

# Counting Stacked Objects

**Conference**: ICCV 2025
**arXiv**: [2411.19149](https://arxiv.org/abs/2411.19149)
**Code**: [https://corentindumery.github.io/projects/stacks.html](https://corentindumery.github.io/projects/stacks.html)
**Area**: Autonomous Driving
**Keywords**: 3D object counting, occupancy estimation, multi-view reconstruction, 3D Gaussian Splatting, depth estimation

## TL;DR

The paper decomposes the stacked object counting problem into two sub-problems — volume estimation and occupancy ratio estimation — solving the former via multi-view 3D reconstruction and the latter via a depth-map-driven neural network that infers interior occupancy from visible surfaces. This is the first method to accurately count largely invisible stacked objects, significantly outperforming humans.

## Background & Motivation

Visual object counting is a fundamental task in computer vision, with applications in crowd counting, cell counting, and traffic surveillance. However, existing methods share a fundamental assumption: the objects being counted must be **visible**. In practice, many real-world scenarios involve objects stacked inside containers (e.g., fruit in boxes, parts on trays), where the majority of objects are occluded by upper layers and only a small fraction is visible at the surface. This "tip of the iceberg" setting renders conventional 2D counting methods entirely ineffective.

Notably, the authors conducted a human study demonstrating that this task is also extremely difficult for humans — 1,485 guesses from 33 participants were far less accurate than the proposed algorithm. The commonly adopted strategy of "counting objects along each axis and multiplying" was shown to be highly inefficient.

## Core Problem

**How can multi-view images be used to accurately count homogeneous objects stacked inside a container, where the vast majority are invisible?**

The key challenges are: (1) most objects are occluded and invisible; (2) stacking patterns are irregular with variable inter-object gaps; (3) individual detection or localization of each object is infeasible.

## Method

### Overall Architecture

The proposed method, **3DC (3D Counting)**, is built on an elegant decomposition. Rather than detecting objects individually, the problem is reduced to a simple formula:

$$\mathcal{N} = \frac{\gamma \cdot \mathcal{V}}{v}$$

where $\mathcal{V}$ is the total volume occupied by the stack, $v$ is the volume of a single object, and $\gamma$ is the **occupancy ratio** — the fraction of the stack volume actually filled by objects (as opposed to gaps).

The pipeline proceeds in three stages:
1. **Segmentation and View Selection**: SAM2 propagates segmentation masks across multi-view images; a key viewpoint where the stack surface is most clearly visible is automatically selected.
2. **Occupancy Ratio $\gamma$ Estimation**: Depth Anything V2 generates a depth map for the key view, which is fed into a trained network $\Phi$ to regress $\gamma$.
3. **Volume Estimation**: COLMAP calibrates camera poses → 3D Gaussian Splatting reconstructs the scene → voxel carving computes the total volume $\mathcal{V}$, with container wall thickness estimated and subtracted.

### Key Designs

**Design of the Occupancy Network $\Phi$**: This is the most elegant contribution of the paper. The core intuition is that if the depth map reveals objects deep within the stack (i.e., large inter-object gaps), the occupancy ratio is low; conversely, a flat surface indicates dense packing and a high occupancy ratio. This "texture" of the depth map is strongly correlated with $\gamma$, and crucially, this relationship **does not depend on the specific shape of the objects**.

Network architecture: a frozen DinoV2 (ViT-B/14) encoder takes a 448×448 depth map as input, producing 32×32×768 features; the decoder applies successive convolutional layers to progressively reduce spatial resolution (768→512→256→128→64 channels, 32×32→2×2), followed by adaptive average pooling to 1×1×64 and a final linear layer with sigmoid activation outputting a scalar $\gamma \in [0,1]$.

**Using Depth Maps Instead of RGB as Input**: Transferring a synthetically trained model to real data introduces domain gap. The authors observe that depth maps estimated by Depth Anything V2 appear nearly identical across synthetic and real data (Fig. 4), substantially reducing the domain discrepancy. Furthermore, training is performed with estimated rather than ground-truth depth maps, allowing the model to adapt to the noise characteristics of depth estimation.

**Container Wall Thickness Prediction**: A separate decoder $\Psi$ predicts the container wall thickness (normalized by container size) using dilated convolutions for a larger receptive field. Predictions are averaged across all views at inference. The corresponding voxels are eroded from the volumetric reconstruction, retaining only the interior volume.

**Volume Estimation Pipeline**: Segmentation masks are used to crop images (with alpha channel) → 3DGS reconstruction → voxel grid initialization → per-view projection: voxels projecting outside the mask or in front of the reconstructed surface are carved away → a closed volume is obtained.

### Loss & Training

- $\Phi$ network training: MSE loss minimizing the squared error between predicted and ground-truth $\gamma$.
- Training data: 400,000 synthetic images from 14,000 physically simulated scenes. CAD models from the ABC dataset are used; Blender's physics engine simulates object falling and stacking, and Cycles renders the scenes with ray tracing. Scenes vary in container shape, material (metal/wood/plastic), and fill level.
- The DinoV2 encoder is frozen; only the decoder is trained.
- Training and inference are lightweight, performed on a 4080 Mobile GPU.

## Key Experimental Results

### Synthetic Dataset (100 unseen scenes)

| Method | NAE↓ | SRE↓ | MAE↓ | sMAPE↓ |
|--------|------|------|------|--------|
| BMNet+ | 0.91 | 0.87 | 320.50 | 158.87 |
| SAM+CLIP | 0.73 | 0.61 | 259.22 | 102.77 |
| CNN (direct prediction) | 0.66 | 0.48 | 235.74 | 98.44 |
| ViT+H (direct prediction) | 0.42 | 0.24 | 149.90 | 47.36 |
| **3DC (Ours)** | **0.22** | **0.09** | **79.48** | **27.65** |

### Real-World Dataset (45 scenes, 2,381 images)

| Method | NAE↓ | SRE↓ | MAE↓ | sMAPE↓ |
|--------|------|------|------|--------|
| BMNet+ | 0.93 | 0.98 | 966.76 | 131.44 |
| SAM+CLIP | 0.94 | 0.99 | 980.33 | 124.31 |
| Human (avg. of 33) | 0.79 | 0.84 | 823.23 | 76.85 |
| Human-Vote (crowd average) | 0.60 | 0.30 | 621.46 | 57.91 |
| LlamaVision 3.2 11B | 1.00 | 1.00 | 1037.50 | 190.48 |
| 3DC (Color, RGB input) | 0.57 | 0.27 | 607.98 | 74.33 |
| **3DC (Ours, Depth input)** | **0.36** | **0.06** | **382.59** | **53.31** |

On real-world data, 3DC achieves 40% lower NAE than human voting (0.36 vs. 0.60) and 80% lower SRE (0.06 vs. 0.30).

### Ablation Study

1. **Depth map source ablation**: Training and inference with estimated depth maps ($\mathcal{T}^-, \mathcal{D}^-$) yields the best performance, even surpassing the use of ground-truth depth. The mild smoothing inherent in estimated depth maps acts as regularization, preventing overfitting to shape-specific features — which also explains why purely synthetic training generalizes to real scenes.
2. **RGB vs. Depth input**: Replacing depth maps with RGB images (Ours Color) increases NAE from 0.36 to 0.57, confirming that depth maps are key to bridging the synthetic-to-real domain gap.
3. **Effect of shape complexity**: As shape complexity (curvature + convex hull volume ratio) increases, $\gamma$ estimation error increases only marginally, demonstrating strong shape generalization.

## Highlights & Insights

1. **The art of problem decomposition**: Direct end-to-end prediction of object count is nearly infeasible in this setting (CNN/ViT baselines perform poorly), but decomposing the problem into $\gamma$ estimation and volume estimation renders each sub-problem tractable — a classic "divide and simplify" strategy.
2. **Depth maps as a domain-invariant bridge**: The synthetic-to-real domain gap is addressed by selecting a domain-invariant representation (estimated depth maps), a more elegant solution than complex domain adaptation techniques.
3. **Counterintuitive finding**: Training with estimated depth maps outperforms training with ground-truth depth — noise provides an implicit regularization effect.
4. **Value of the human baseline**: The counting experiment with 33 participants not only validates task difficulty but also demonstrates the failure of the naive "multiply along axes" strategy, indirectly supporting the paper's decomposition methodology.
5. **Complete synthetic data pipeline**: The fully automated chain from physical simulation to rendering to depth map generation requires only CPU resources, making it environmentally efficient.

## Limitations & Future Work

1. **Real-world performance lags behind synthetic**: NAE = 0.36 on real data vs. 0.22 on synthetic, primarily due to greater complexity in real scenes (thousands of objects, more complex container geometries).
2. **No per-object localization**: The method outputs only a total count and cannot provide the position of individual objects. The authors suggest that integrating visible instance localization with invisible instance configuration estimation is a promising future direction.
3. **Uniform stacking assumption**: The method assumes $\gamma$ is approximately uniform throughout the container; non-uniform stacking (e.g., objects packed more tightly on one side) may introduce larger errors.
4. **Requires known unit volume**: $v$ must be known or estimable, limiting applicability to completely unknown objects.
5. **Multi-view capture requirement**: 30–60 images captured around the container are needed for 3D reconstruction, posing efficiency constraints.
6. **Failure of LLMs/VLMs**: LlamaVision 3.2 completely fails on this task, indicating that current VLMs remain weak in 3D spatial reasoning and quantity estimation.

## Related Work & Insights

- **vs. 2D counting methods (BMNet+, SAM+CLIP)**: Traditional methods can only count visible objects, yielding NAE > 0.9 on stacked scenarios — essentially complete failure.
- **vs. direct end-to-end prediction (CNN, ViT+H)**: Skipping problem decomposition and directly regressing counts yields NAE > 0.9 on real data, underscoring the importance of the decomposition strategy.
- **vs. CountNet3D [Jenkins 2023]**: The only known related work on 3D counting, but it requires LiDAR and is restricted to a specific category (beverages with known volumes); the proposed method is substantially more general.
- **vs. multi-view counting (Zhang 2020a/b)**: Multi-view counting methods assume every object is visible in at least one view, making them inapplicable to stacked scenarios.

## Relevance to My Research

- The core idea of occupancy ratio estimation (inferring global properties from visible surface features) is transferable to other partially observable settings, such as attribute estimation of occluded objects.
- The use of depth maps as a domain-invariant bridge is worth adopting in other sim-to-real transfer tasks.
- The 3DGS + voxel carving volume estimation pipeline constitutes a practical paradigm for multi-view 3D understanding.
- The synthetic data generation pipeline (physics simulation + rendering) offers a useful reference for constructing training data for rare scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to define and solve the stacked object counting problem; the decomposition methodology is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive synthetic and real datasets, convincing human baseline, thorough ablation; however, only 45 real-world scenes are included.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Logic is clear and fluent; the narrative from intuition to formulation to implementation reads like a coherent story.
- **Value to My Research**: ⭐⭐⭐ The problem decomposition strategy and synthetic data pipeline design are inspiring, though direct overlap with current research directions is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Referring Expression Comprehension for Small Objects](referring_expression_comprehension_for_small_objects.md)
- [\[CVPR 2026\] Plant Taxonomy Meets Plant Counting: A Fine-Grained, Taxonomic Dataset for Counting Hundreds of Plant Species](../../CVPR2026/autonomous_driving/plant_taxonomy_meets_plant_counting_a_fine-grained_taxonomic_dataset_for_countin.md)
- [\[CVPR 2026\] SABER: Spatially Consistent 3D Universal Adversarial Objects for BEV Detectors](../../CVPR2026/autonomous_driving/saber_spatially_consistent_3d_universal_adversarial_objects_for_bev_detectors.md)
- [\[ICCV 2025\] 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation](3d_gaussian_splatting_driven_multiview_robust_physical_adver.md)
- [\[ICCV 2025\] EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting](emd_explicit_motion_modeling_for_high-quality_street_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
