---
title: >-
  [Paper Note] LayeredFlow: A Real-World Benchmark for Non-Lambertian Multi-Layer Optical Flow
description: >-
  [ECCV 2024][Video Understanding][non-Lambertian] Proposes LayeredFlow—the first real-world non-Lambertian benchmark dataset containing multi-layer optical flow annotations (150k optical flow pairs, 185 scenes, 360 objects), while defining the multi-layer optical flow task, introducing a large-scale synthetic training dataset, and presenting a RAFT-based multi-layer optical flow baseline.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "non-Lambertian"
  - "multi-layer optical flow"
  - "real-world benchmark"
  - "transparent objects"
  - "synthetic dataset"
date: 2026-05-08
content_hash: 49c8d62e127d79b7
---

# LayeredFlow: A Real-World Benchmark for Non-Lambertian Multi-Layer Optical Flow

**Conference**: ECCV 2024  
**arXiv**: [2409.05688](https://arxiv.org/abs/2409.05688)  
**Code**: Yes ([https://layeredflow.cs.princeton.edu](https://layeredflow.cs.princeton.edu))  
**Area**: Video Understanding / Optical Flow Estimation  
**Keywords**: non-Lambertian, multi-layer optical flow, real-world benchmark, transparent objects, synthetic dataset

## TL;DR

Proposes LayeredFlow—the first real-world non-Lambertian benchmark dataset containing multi-layer optical flow annotations (150k optical flow pairs, 185 scenes, 360 objects), while defining the multi-layer optical flow task, introducing a large-scale synthetic training dataset, and presenting a RAFT-based multi-layer optical flow baseline.

## Background & Motivation

### Importance and Challenges of Non-Lambertian 3D Understanding

Non-Lambertian surfaces (such as glass, metal, specular objects) are ubiquitous in the real world. Autonomous driving requires identifying the 3D geometry of glass walls and reflective road surfaces, while robotic manipulation depends on accurate depth information of plastic and metallic materials. However, current data-driven algorithms for optical flow and stereo matching perform exceptionally well on diffuse (Lambertian) objects but **fail catastrophically** on non-Lambertian ones. The root causes are:

**Training Data Bias**: In mainstream training datasets (such as FlyingThings3D and Sintel), Lambertian surfaces dominate, while non-Lambertian surfaces are severely underrepresented.

**Inadequate Evaluation Benchmarks**: Existing non-Lambertian benchmarks have severe limitations:
   - **Low Scene Diversity**: Most are limited to a small number of indoor tabletop scenes.
   - **Difficult Object Acquisition**: They require pre-scanning 3D models or painting objects with Lambertian paint.
   - **Lack of Multi-Layer Annotation**: No benchmarks provide multi-layer 3D annotations under transparent occlusions.

### Core Needs of Multi-Layer Perception

In the presence of transparent objects, a single pixel may capture information from multiple 3D points in the scene—one on the transparent surface, and another on the occluded object. Humans can typically infer 3D information from multiple depth layers, but existing algorithms lack this capability. Current "see-through" methods only focus on the diffuse objects behind the transparent surface and only consider a single layer of transparent occlusion.

### Technical Bottlenecks of Data Collection Methods

Existing benchmarks for obtaining 3D annotations of non-Lambertian objects suffer from significant drawbacks:
- **3D Scan Alignment**: Limited to shapes that can be 3D scanned, and scenes are restricted to tabletop objects.
- **Lambertian Paint Method (Booster)**: Requires intensive manual labor to apply and remove paint, and is limited to indoor environments.
- **Occlusion-Overlay Method**: Pasting opaque patches on glass walls and interpolating, which is restricted to planar surfaces.

This paper overcomes these limitations using the AprilTag visual fiducial marker system, enabling large-scale, diverse, and multi-layer data collection.

## Method

### Overall Architecture

LayeredFlow comprises three core contributions: (1) an AprilTag-based real-world multi-layer optical flow benchmark dataset; (2) a large-scale synthetic training dataset; and (3) a multi-layer optical flow task definition along with a baseline method.

### Key Designs

#### 1. **AprilTag Multi-Layer Data Collection Pipeline**

**Function**: Utilizes the AprilTag visual fiducial system in a stereo camera setup to obtain ground-truth multi-layer optical flow annotations for non-Lambertian objects.

**Mechanism**: AprilTags are barcode-like visual fiducial markers printed on matte vinyl stickers that can be easily pasted and peeled off. The acquisition pipeline consists of four steps:
1. Capture a pair of stereo images of the scene without markers using a calibrated stereo camera (a).
2. Apply AprilTags and capture a pair of marked stereo images (b)—providing stereo correspondences.
3. Rearrange scene objects and camera viewpoints, then capture another pair of marked images (c)—the correspondences of markers between (b) and (c) yield the optical flow.
4. Remove the markers and capture the final unmarked image pair (d).

**Key to Multi-Layer Annotation**: Even when pasted behind transparent objects, AprilTags can still be detected by the camera (albeit with refractive distortion). This allows markers to be placed on a desk behind a glass door while simultaneously placing markers on the glass surface itself, thereby obtaining ground-truth annotations for multiple depth layers.

**Design Motivation**: Unlike methods requiring 3D scanning or painting, AprilTags can be applied to non-Lambertian objects of almost any scene and scale (from cups to cars) while retaining instructions of physical refraction and distortion.

#### 2. **Large-Scale Synthetic Training Dataset**

**Function**: Generates 60k synthetic images containing multi-layer optical flow and 3D position annotations to provide training data for multi-layer optical flow tasks.

**Mechanism**: Based on 30 high-quality BlendSwap indoor scenes (10 kitchens + 5 bathrooms + 5 offices + 5 living rooms + 5 bedrooms), data augmentation is performed via the Blender Python API:
- **Camera Selection**: Positions, orientations, and focal lengths are randomly selected from manually specified parameter subsets.
- **Lighting Randomization**: Light source colors and intensities are randomly assigned, and environment textures are selected from 50 HDR images.
- **Material Randomization**: Some objects are randomly assigned glass or metallic materials (with varying colors and roughness).
- **Adding Flying Objects**: Additional non-Lambertian objects are randomly placed from 100 BlendSwap categories.

**Multi-Layer Annotation Generation**: The Blender ray-tracing source code was modified to embed ground-truth collection during rendering. Only materials with low roughness are considered transparent. The number of times a ray passes through transparent surfaces is tracked to determine the layer index. Reflective rays are disabled to ensure only rays from actual physical surfaces are traced.

**Design Motivation**: Acquiring dense annotations for non-Lambertian objects in the real world is nearly impossible. Synthetic data provides pixel-wise multi-layer annotations, and rich randomized augmentations enhance domain generalization capabilities.

#### 3. **Multi-Layer Optical Flow Task Definition and Multi-RAFT Baseline**

**Function**: Defines the multi-layer optical flow estimation problem and provides a RAFT-based baseline method.

**Task Definition**: Given two images and a query pixel $p$, predict an ordered sequence of layer-by-layer optical flows $\hat{\mathcal{F}} = \{\hat{\mathbf{f}}_1, ..., \hat{\mathbf{f}}_n\}$, where the layer count $n$ varies per pixel.

**Evaluation Metrics**:
- **Layer Count Correctness**: Whether the predicted layer count matches the ground truth (allows $\geq m_k$ if the last layer is transparent, but must $= m_k$ if opaque).
- **Multi-layer bad-$\tau$**: The percentage of pixels where the L2 optical flow error of all layers is within the threshold $\tau$.
- **Layer-count-aware bad-$\tau$**: Requires both the layer count to be correct and the optical flow predictions to be accurate.

**Baseline Method Multi-RAFT**: Uses $n$ context encoders with independent weights (replacing the single encoder in RAFT). They share the feature encoder and correlation volume, and are separately fed into ConvGRU update blocks to generate $n$-layer optical flow predictions. During inference, duplicate predictions with distances less than $\delta=0.5$ pixels in adjacent layers are pruned.

### Loss & Training

- When a training sample only provides $k$ layers of ground truth, the last layer is duplicated $n-k$ times to match the $n$ heads.
- Uses the standard optical flow training loss (L1 sequence loss, consistent with RAFT).
- Fine-tuning strategy: L (synthetic data only), S (Sintel only), S+L (joint training, yields best performance).

## Key Experimental Results

### Main Results: Single-Layer Optical Flow (First Layer)

| Method | All EPE↓ | All bad-1px↓ | Transparent EPE↓ | Reflection EPE↓ | Diffuse EPE↓ |
|------|---------|-------------|----------|----------|-----------|
| FlowNet-C | 21.14 | 94.88 | 24.01 | 13.85 | 17.04 |
| RAFT | 16.49 | 78.45 | 20.11 | 8.51 | 10.76 |
| GMA | 16.58 | 79.26 | 20.35 | 8.18 | 12.00 |
| FlowFormer | 18.49 | 78.83 | 22.56 | 9.54 | 5.01 |
| RAFT-ft.(S) | 17.94 | 79.53 | 21.96 | 8.89 | 9.07 |
| RAFT-ft.(S+L) | **15.63** | **77.81** | **18.39** | **11.73** | **6.95** |

Jointly fine-tuned on synthetic data, RAFT (S+L) reduces EPE and bad-τ across all non-Lambertian categories, whereas RAFT fine-tuned solely on Sintel shows no improvement.

### Ablation Study: Multi-Layer Baseline Evaluation (Multi-Layer Count-Aware bad-τ)

| Method | Layer 1 bad-1px↓ | Layer 1 bad-3px↓ | Layer 2 bad-1px↓ | Layer 2 bad-∞↓ | Layer 3 bad-∞↓ |
|------|-------------------|-------------------|-------------------|----------------|----------------|
| RAFT | 78.45 | 55.64 | 100.0 (No multi-layer capability) | 100.0 | 100.0 |
| Multi-RAFT (L) | 76.51 | 51.82 | 91.91 | **47.19** | 38.88 |
| Multi-RAFT (S+L) | 77.83 | 54.85 | **88.85** | 40.56 | **21.62** |

Multi-RAFT outperforms the original RAFT on almost all Layer 1 metrics and is the first to achieve optical flow predictions for Layer 2 and Layer 3.

### Key Findings

1. **All existing methods suffer huge errors on non-Lambertian surfaces**: Compared to benchmarks like Sintel/KITTI, errors on LayeredFlow are several times higher, demonstrating the difficulty of the benchmark.
2. **The first layer of transparent objects is the most difficult**: The Layer 1 error is much larger than the last layer error because existing methods tend to "see through" transparent surfaces and ignore their geometry.
3. **Synthetic data is effective**: Fine-tuning solely on synthetic data significantly boosts non-Lambertian performance without degrading the accuracy of diffuse objects.
4. **"See-Through" scenes show the most notable improvement**: In the "Behind Transparent" category, EPE drops from 8.48 to 4.34 (↓49%) after fine-tuning.

## Highlights & Insights

1. **Ingenuity of the AprilTag Method**: Exploiting the fact that visual markers are detectable even behind transparent objects elegantly solves the long-standing challenge of multi-layer annotation.
2. **Systemic Task Definition**: The joint evaluation metrics of Layer Count Correctness + Flow τ-Accuracy are soundly designed, effectively separating "structural understanding" from "precision."
3. **Breakthrough in Dataset Scale and Diversity**: With 185 indoor/outdoor scenes, 360 objects, and 150k annotated pairs, it vastly exceeds all existing non-Lambertian benchmarks.

## Limitations & Future Work

1. **Annotation Sparsity**: Constrained by the physical dimensions of AprilTags, each scene has only 20-500 sparse corresponding points, rendering dense pixel-wise ground truth infeasible.
2. **Simplistic Baseline Method**: Multi-RAFT achieves multiple layers merely by duplicating context encoders, lacking inter-layer interaction mechanisms.
3. **Synthetic-to-Real Domain Gap**: Though synthetic fine-tuning is effective, a distinct domain gap remains; future work can explore more robust domain adaptation strategies.
4. **Weak Layer Count Prediction**: Multi-RAFT still has significant room for growth in Layer Count accuracy (with Layer 2 bad-∞ still at 40%+).

## Related Work & Insights

- **Booster [ECCV 2022]**: Obtains annotations by painting targets with Lambertian coatings, which is not scalable. LayeredFlow's AprilTag approach is much more flexible.
- **RAFT [ECCV 2020]**: The baseline in this paper directly extends the RAFT architecture, demonstrating its potential for multi-layer extension.
- **See-Through Methods**: Existing see-through methods focus exclusively on diffuse objects behind a single transparent occlusion layer, whereas LayeredFlow is the first to address multi-layer transparency.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introduces the first multi-layer optical flow benchmark and task definition; the AprilTag acquisition method is highly unique.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Features comprehensive comparisons against 10+ optical flow methods with multi-dimensional evaluations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation of the problem, and highly detailed descriptions of the data collection pipeline.
- **Value**: ⭐⭐⭐⭐⭐ — Fills the benchmark gap in non-Lambertian multi-layer understanding, acting as a strong catalyst for the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow](sea-raft_simple_efficient_accurate_raft_for_optical_flow.md)
- [\[CVPR 2026\] OmniGround: A Comprehensive Spatio-Temporal Grounding Benchmark for Real-World Complex Scenarios](../../CVPR2026/video_understanding/omniground_a_comprehensive_spatio-temporal_grounding_benchmark_for_real-world_co.md)
- [\[CVPR 2026\] FlowFM: Advancing Dark Optical Flow Estimation with Flow Matching](../../CVPR2026/video_understanding/flowfm_advancing_dark_optical_flow_estimation_with_flow_matching.md)
- [\[ICCV 2025\] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](../../ICCV2025/video_understanding/memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)
- [\[ICCV 2025\] PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View](../../ICCV2025/video_understanding/prior-flow_enhancing_primitive_panoramic_optical_flow_with_orthogonal_view.md)

</div>

<!-- RELATED:END -->
