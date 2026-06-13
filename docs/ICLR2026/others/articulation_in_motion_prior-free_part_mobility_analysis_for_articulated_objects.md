---
title: >-
  [Paper Note] Articulation in Motion: Prior-Free Part Mobility Analysis for Articulated Objects
description: >-
  [ICLR 2026][articulated objects] This paper proposes AiM (Articulation in Motion), a framework that reconstructs articulated objects from interaction videos and initial-state scans without requiring prior knowledge of th…
tags:
  - "ICLR 2026"
  - "articulated objects"
  - "Gaussian splatting"
  - "part segmentation"
  - "joint estimation"
  - "sequential RANSAC"
  - "prior-free"
  - "interaction video"
date: 2026-05-08
content_hash: 631cf6dc101ca885
---

# Articulation in Motion: Prior-Free Part Mobility Analysis for Articulated Objects

**Conference**: ICLR 2026
**arXiv**: [2603.02910](https://arxiv.org/abs/2603.02910)  
**Project Page**: [AiM](https://haoai-1997.github.io/AiM/)
**Area**: Other
**Keywords**: articulated objects, Gaussian splatting, part segmentation, joint estimation, sequential RANSAC, prior-free, interaction video

## TL;DR

This paper proposes AiM (Articulation in Motion), a framework that reconstructs articulated objects from interaction videos and initial-state scans without requiring prior knowledge of the number of parts. It achieves dynamic-static decoupling via a dual-Gaussian representation (Static GS + Deformable GS), combines sequential RANSAC for prior-free part segmentation and joint estimation, and incorporates an SDMD module to handle newly exposed static regions. On complex 6-part objects (Storage), AiM achieves 79.34% mean IoU, substantially outperforming the prior-dependent ArtGS (52.23%).

## Background & Motivation

**Core demand for articulated object understanding**: Robot manipulation, AR/VR, and embodied intelligence all require understanding the part structure and joint parameters of articulated objects (e.g., drawer cabinets, doors, laptops).

**Prior dependency of existing methods**: Methods such as DTA and ArtGS require the number of parts to be specified in advance, which is typically unknown in real-world scenarios; an incorrect specification leads to severe segmentation failures.

**Challenge of dynamic-static decoupling**: During interaction, some parts move while others remain static; however, the displacement of moving parts exposes previously occluded static regions, which conventional methods struggle to handle.

**Limitation of single representations**: Purely static or purely dynamic 3D Gaussian representations cannot simultaneously accommodate the mixed nature of fixed and moving parts in articulated objects.

**Diversity of joint types**: Articulated objects contain multiple joint types including revolute and prismatic joints, necessitating a unified prior-free estimation approach.

**Practicality of video input**: Recovering articulation information from a single interaction video is more practical and natural than methods requiring multi-view static scans.

## Method

### Overall Architecture

AiM takes as input an interaction video of a human manipulating an articulated object and a 3D scan of the object in its initial (static) state, and outputs part segmentation, joint parameters, and a complete articulated object reconstruction. The pipeline consists of three stages: dual-Gaussian dynamic-static decoupling → sequential RANSAC part discovery → joint parameter estimation.

### Key Designs

1. **Dual-Gaussian Representation**

    - **Function**: Maintains two sets of 3D Gaussians — Static GS representing invariant background and stationary parts, and Deformable GS representing moving parts.
    - **Mechanism**: Gradient signals from pixel-level rendering losses automatically assign Gaussians to static or dynamic sets; Static GS remains fixed while Deformable GS learns per-frame deformation fields.
    - **Design Motivation**: Explicit dynamic-static separation prevents moving parts from corrupting static geometry, and allows subsequent part segmentation to focus exclusively on dynamic Gaussians.

2. **Sequential RANSAC Part Segmentation**

    - **Function**: Automatically discovers parts from the motion trajectories of dynamic Gaussians without presetting the number of parts.
    - **Mechanism**: Fits rigid body motion to the deformation trajectories of all dynamic Gaussians; the largest consensus set corresponds to one part. That part is then removed and the process iterates over the remaining Gaussians until the residual falls below a threshold.
    - **Design Motivation**: RANSAC is naturally suited to the setting of "an unknown number of mixed rigid body motions"; sequential execution ensures parts are discovered in descending order of size.

3. **SDMD Module (Static Dynamic Merging with Discovery)**

    - **Function**: Handles static regions newly exposed after moving parts are displaced (e.g., the interior walls of a cabinet revealed when a drawer is opened).
    - **Mechanism**: Detects discrepancy regions between rendered and real images, initializes new Static Gaussians at those locations, and merges them with the existing Static GS.
    - **Design Motivation**: Conventional methods cannot handle static geometry that is initially invisible but later becomes observable; SDMD fills this critical gap.

## Key Experimental Results

### Main Results

| Method | Part Prior | Mean IoU (%) | Revolute JE (°) | Prismatic JE (mm) |
|--------|-----------|-------------|-----------------|-------------------|
| DTA | Required | 71.45 | 8.32 | 12.7 |
| ArtGS | Required | 76.99 | 5.61 | 8.9 |
| AiM (Ours) | **Not Required** | **80.21** | **4.23** | **7.1** |

### Ablation Study

| Component | Mean IoU (%) | Note |
|-----------|-------------|------|
| Full AiM | **80.21** | Complete method |
| w/o SDMD | 74.85 | Newly exposed regions incorrectly assigned |
| Single GS (no decoupling) | 68.32 | Moving parts corrupt static reconstruction |
| K-means instead of RANSAC | 72.56 | Requires preset K and is sensitive to noise |
| ArtGS with ground-truth part count | 76.99 | Still underperforms AiM even with correct prior |

### Key Findings

1. **Prior-free surpasses prior-dependent**: AiM achieves 80.21% mean IoU without part-count priors, exceeding prior-dependent ArtGS (76.99%), demonstrating that adaptive discovery is more robust than fixed assumptions.
2. **Decisive advantage on complex objects**: On the 6-part Storage object, AiM (79.34%) vs. ArtGS (52.23%) shows a gap of 27%; ArtGS degrades sharply as the number of parts increases.
3. **SDMD is indispensable**: Removing SDMD causes a 5.36% IoU drop, confirming the importance of handling newly exposed regions.
4. **Dynamic-static decoupling is foundational**: The single-GS variant underperforms the full method by nearly 12%, establishing the dual-Gaussian design as the cornerstone of success.

## Highlights & Insights

1. **Complete elimination of priors**: AiM is the first method to achieve part segmentation and joint estimation for articulated objects without requiring prior knowledge of the number of parts, better matching real-world application demands.
2. **Elegant dual-Gaussian decoupling**: Embedding dynamic-static separation into the 3DGS representation simultaneously benefits reconstruction quality and downstream analysis.
3. **Practical innovation of SDMD**: Addresses the progressive exposure of previously occluded static regions — a critical yet often overlooked detail in articulated object understanding.
4. **Natural fit of sequential RANSAC**: Cleverly exploits the iterative stripping property of RANSAC to achieve adaptive part-count discovery.
5. **Overwhelming advantage on complex objects**: The 27% improvement on 6-part scenes demonstrates the scalability of the approach.

## Limitations & Future Work

1. **Single-interaction assumption**: The current method requires that all parts be actuated within the video; parts that are not manipulated cannot be discovered.
2. **Rigid body motion assumption**: Sequential RANSAC assumes each part undergoes rigid body motion and cannot handle flexible hinges or elastic deformations.
3. **Computational cost**: The combination of dual-Gaussian representation and sequential RANSAC incurs substantial computational overhead, precluding real-time operation.
4. **Dependency on video quality**: Low-quality videos with severe motion blur or occlusion may lead to inaccurate dynamic Gaussian estimation.

## Related Work & Insights

- **Articulated object reconstruction**: Gaussian splatting-based methods including DTA (Liu et al., 2024) and ArtGS (Huang et al., 2024).
- **3D Gaussian Splatting**: 3DGS (Kerbl et al., 2023), Dynamic 3DGS (Luiten et al., 2024).
- **Part segmentation**: Supervised methods such as PartNet (Mo et al., 2019); unsupervised methods such as SAM3D.
- **RANSAC**: The classic framework of Fischler & Bolles (1981); application of sequential RANSAC to multi-model fitting.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Prior-free part discovery + dual-Gaussian decoupling + SDMD are all novel designs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple object categories with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Method pipeline is clearly presented; experimental results are detailed.
- Value: ⭐⭐⭐⭐⭐ Prior-free articulated object understanding has significant practical value for robotics and embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Bayesian Influence Functions for Hessian-Free Data Attribution](bayesian_influence_functions_for_hessian-free_data_attribution.md)
- [\[AAAI 2026\] HybriDLA: Hybrid Generation for Document Layout Analysis](../../AAAI2026/others/hybridla_hybrid_generation_for_document_layout_analysis.md)
- [\[CVPR 2026\] DirPA: Addressing Prior Shift in Imbalanced Few-shot Crop-type Classification](../../CVPR2026/others/dirpa_addressing_prior_shift_in_imbalanced_fewshot.md)
- [\[ICML 2026\] Torus Graphs for Large-Scale Neural Phase Analysis](../../ICML2026/others/torus_graphs_for_large_scale_neural_phase_analysis.md)
- [\[AAAI 2026\] Cost-Free Neutrality for the River Method](../../AAAI2026/others/cost-free_neutrality_for_the_river_method.md)

</div>

<!-- RELATED:END -->
