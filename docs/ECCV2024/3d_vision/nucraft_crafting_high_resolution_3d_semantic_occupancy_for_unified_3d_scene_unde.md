---
title: >-
  [Paper Note] nuCraft: Crafting High Resolution 3D Semantic Occupancy for Unified 3D Scene Understanding
description: >-
  [ECCV 2024][3D Vision][3D Semantic Occupancy Prediction] This paper constructs nuCraft, a high-precision 3D semantic occupancy dataset based on nuScenes (with a resolution up to 0.1m voxels, $8\times$ denser than existing benchmarks), and proposes VQ-Occ, which uses VQ-VAE to encode occupancy data into a compact latent space for prediction, achieving direct generation of high-resolution semantic occupancy without post-processing upsampling for the first time.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D Semantic Occupancy Prediction"
  - "High-Resolution Voxel"
  - "VQ-VAE"
  - "Autonomous Driving Scene Understanding"
  - "nuScenes"
date: 2026-05-08
content_hash: 688d7887c4815326
---

# nuCraft: Crafting High Resolution 3D Semantic Occupancy for Unified 3D Scene Understanding

**Conference**: ECCV 2024  
**Code**: None  
**Area**: 3D Vision / Autonomous Driving  
**Keywords**: 3D Semantic Occupancy Prediction, High-Resolution Voxel, VQ-VAE, Autonomous Driving Scene Understanding, nuScenes

## TL;DR
This paper constructs nuCraft, a high-precision 3D semantic occupancy dataset based on nuScenes (with a resolution up to 0.1m voxels, $8\times$ denser than existing benchmarks), and proposes VQ-Occ, which uses VQ-VAE to encode occupancy data into a compact latent space for prediction, achieving direct generation of high-resolution semantic occupancy without post-processing upsampling for the first time.

## Background & Motivation

**Background**: 3D Semantic Occupancy Prediction is a core task in 3D scene understanding for autonomous driving, aiming to discretize the 3D space into voxel grids and predict a semantic class for each voxel. Existing mainstream benchmarks, such as Occ3D and OpenOccupancy, are built on nuScenes with a resolution limit of 0.2m voxels ($512 \times 512 \times 40$ grids), and the annotations are automatically generated from LiDAR point clouds.

**Limitations of Prior Work**: Existing benchmarks suffer from two major limitations: (1) **Low resolution**—the 0.2m voxel scale is insufficient to depict fine-grained scene structures; small objects such as thin walls, railings, and pedestrians suffer from severe information loss under coarse resolutions. (2) **Inaccurate annotations**—the raw LiDAR point clouds are sparse and noisy, leading to numerous errors and omissions in the automatically generated labels. Furthermore, existing methods typically perform prediction at a 0.4m resolution and then upsample to 0.2m via post-processing, which introduces additional errors.

**Key Challenge**: High resolution brings finer scene descriptions but simultaneously causes massive GPU memory consumption—a $1024 \times 1024 \times 80$ voxel grid at 0.1m resolution contains over 80 million voxels, making direct prediction and supervision in the raw voxel space computationally prohibitive.

**Goal**: (1) Construct higher-resolution and more accurate 3D semantic occupancy annotations; (2) Design a prediction method that can efficiently process high-resolution occupancy data.

**Key Insight**: The authors observe that occupancy data exhibits highly structured redundancy—most of the space is empty, and neighboring voxels typically share the same semantics. This inspires the idea of using VQ-VAE to compress occupancy data into a low-dimensional discrete latent space: first learning a codebook to efficiently encode the occupancy data, then allowing the prediction model to operate within the compact latent space, and finally restoring high-resolution predictions using a decoder.

**Core Idea**: Using VQ-VAE to compress high-resolution occupancy data into a compact, discrete latent space, transforming semantic occupancy prediction into a feature modeling problem in the latent space.

## Method

### Overall Architecture
The system consists of two parts: (1) **nuCraft Dataset Construction**—generating high-precision semantic occupancy annotations at 0.1m resolution from nuScenes through multi-frame point cloud accumulation and a refined annotation pipeline; (2) **VQ-Occ Prediction Method**—first training an encoder-decoder and a codebook on occupancy data using VQ-VAE, then predicting codebook indices in the latent space using image/LiDAR features, and finally recovering the high-resolution occupancy field via the decoder. The input is multi-view camera images or LiDAR point clouds, and the output is a $1024 \times 1024 \times 80$ resolution semantic occupancy grid.

### Key Designs

1. **nuCraft High-Resolution Annotation Construction**:

    - **Function**: Providing high-precision semantic occupancy annotations that are $8\times$ denser than existing benchmarks.
    - **Mechanism**: First, point cloud density is increased by accumulating multi-frame (past and future) LiDAR scans to eliminate single-frame sparsity. Then, an improved voxelization and semantic propagation algorithm is utilized to assign semantic labels to each 0.1m voxel. Key steps include: removing temporal misalignment of dynamic objects (by tracking dynamic objects with 3D bounding boxes and accumulating point clouds in their local coordinate systems), visibility reasoning (using ray-tracing to determine which voxels are occupied, free, or unobserved), and annotation refinement (removing noisy annotations through geometric consistency checks).
    - **Design Motivation**: High-quality annotation is the prerequisite for high-resolution prediction, as noise and errors in existing automatic annotation pipelines would be amplified at finer resolutions.

2. **VQ-VAE Occupancy Data Compression**:

    - **Function**: Encoding high-dimensional occupancy data into a compact discrete latent representation.
    - **Mechanism**: Training a 3D VQ-VAE whose encoder compresses the $1024 \times 1024 \times 80$ occupancy grid into a low-resolution latent feature map (e.g., $128 \times 128 \times 10$), and mapping each latent vector to the nearest code in a learned codebook via vector quantization. The decoder restores the original high-resolution occupancy from the quantized latent representation. The training utilizes reconstruction loss and commitment loss. The compressed latent space only needs to store codebook indices (integers), reducing the data size by approximately $512\times$.
    - **Design Motivation**: Direct prediction in the high-resolution voxel space requires massive GPU memory (>100GB), whereas VQ-VAE leverages the structural redundancy of occupancy data to compress it to a manageable scale, while the discretized codebook provides structured priors.

3. **Latent Space Occupancy Prediction**:

    - **Function**: Directly predicting the compressed occupancy representation from sensor inputs.
    - **Mechanism**: Using standard image/LiDAR backbones to extract features and generating Bird’s-Eye-View (BEV) features via a BEV feature extraction module. Then, a lightweight prediction head maps the BEV features to the latent space of the VQ-VAE, predicting which codebook index should be used for each latent position. Training uses cross-entropy loss, transforming the prediction problem into a classification problem (selecting from classes of the codebook size). During inference, the predicted indices directly reconstruct the high-resolution occupancy field through the pre-trained decoder, without requiring any post-processing upsampling.
    - **Design Motivation**: Predicting in the latent space rather than the original space makes high-resolution prediction computationally feasible; classification-based prediction avoids the misfit of regression methods on discrete occupancy data.

### Loss & Training
The training consists of two stages: (1) VQ-VAE pre-training stage, using occupancy reconstruction loss $\mathcal{L}_{recon}$ (cross-entropy) + commitment loss $\mathcal{L}_{commit}$ + codebook update (via EMA); (2) Prediction model training stage, using latent space classification loss $\mathcal{L}_{cls}$ to align predicted logits with the codebook indices output by the VQ-VAE encoder.

## Key Experimental Results

### Main Results

| Method | Resolution | mIoU | GPU Memory | Post-processing Upsampling |
|------|--------|------|---------|-------------|
| VQ-Occ (Ours) | 0.1m | 30.2 | ~16GB | No |
| TPVFormer | 0.4m→0.2m | 27.8 | ~12GB | Yes |
| SurroundOcc | 0.4m→0.2m | 28.5 | ~14GB | Yes |
| CTF-Occ | 0.4m→0.2m | 28.9 | ~18GB | Yes |
| OccFormer | 0.4m→0.2m | 27.2 | ~16GB | Yes |

### Ablation Study

| Configuration | mIoU | Description |
|------|------|------|
| VQ-Occ (Full) | 30.2 | Full model, direct 0.1m prediction |
| Direct Voxel Prediction (0.1m) | OOM | Out of memory, unable to train |
| Direct Voxel Prediction (0.2m) | 28.0 | Trainable after downscaling resolution |
| VQ-Occ w/o VQ | 29.1 | Using continuous latent instead of discrete codebook |
| Codebook=256 | 28.8 | Codebook too small, insufficient representational capability |
| Codebook=4096 | 30.0 | Near optimal |
| Codebook=8192 | 30.2 | Default configuration |

### Key Findings
- The annotation quality of nuCraft is significantly superior to Occ3D and OpenOccupancy, with an accuracy improvement of approximately 15% in human evaluation, particularly on small objects and boundary regions.
- The compression ratio of VQ-VAE is extremely high (approx. $512\times$), yet the reconstruction quality is excellent, indicating that occupancy data indeed possesses substantial exploitable structural redundancy.
- The choice of codebook size has a clear sweet spot—too small limits expressivity, while too large increases computation with diminishing returns.
- The 0.1m resolution shows particularly evident improvements on small object categories like pedestrians and bicycles compared to 0.2m (+3-5 IoU), validating the value of high resolution.
- Needing no post-processing upsampling is a key advantage of VQ-Occ, which eliminates artifacts introduced by upsampling.

## Highlights & Insights
- **Elegant transformation of a generative model (VQ-VAE) for a discriminative task**: Instead of directly predicting voxel labels, it first learns a robust representation of occupancy data and then performs predictions in this compact representation space. This concept essentially leverages the compression capability of generative models to bypass computational bottlenecks in discriminative tasks, which is both clever and generalizable.
- **Co-design of dataset and method**: nuCraft does not simply scale up the resolution; it simultaneously improves the annotation pipeline, which ensures the quality of high-resolution annotations. This co-design of dataset and method avoids the common pitfall where "poor data quality holds back the method."
- **Classification instead of regression**: Transforming occupancy prediction from "predicting semantic labels for each voxel" to "predicting codebook indices in the latent space" is fundamentally a representation learning approach, which can be migrated to other large-scale 3D prediction tasks.

## Limitations & Future Work
- **Two-stage training of VQ-VAE**: It requires pre-training the VQ-VAE prior to training the prediction model. This pipeline is complex, and the quality of the VQ-VAE directly caps the system's performance ceiling. Joint end-to-end training could yield better performance.
- **Codebook utilization**: VQ-VAE typically suffers from the codebook collapse problem (where only a subset of codes is frequently used), potentially limiting representational capacity. Future work could explore improved VQ variants (e.g., FSQ, LFQ).
- **Unused temporal information**: The current method relies on single-frame prediction, neglecting multi-frame temporal information that could enhance prediction consistency and accuracy.
- **Annotations still dependent on LiDAR**: The annotation pipeline of nuCraft still relies on LiDAR as the ground truth source. The inherent limitations of LiDAR (e.g., sparse point clouds at a distance, invisibility of transparent objects) still impact annotation quality.
- Future directions: exploring diffusion models instead of VQ-VAE for occupancy data generation; introducing temporal consistency constraints.

## Related Work & Insights
- **vs TPVFormer**: TPVFormer uses tri-perspective planes to approximate 3D voxel representations to reduce computational cost, but sacrifices the completeness of 3D information; nuCraft/VQ-Occ preserves full 3D structures via latent space compression.
- **vs SurroundOcc**: SurroundOcc utilizes multi-scale features and upsampling strategies, but the upsampling process introduces artifacts; VQ-Occ directly predicts in the latent space and recovers the occupancy in one step via the decoder, providing cleaner results.
- **vs Occ3D Dataset**: nuCraft comprehensively outperforms Occ3D in annotation quality and resolution, positioning it to become a new standard benchmark.

## Rating
- Novelty: ⭐⭐⭐⭐ VQ-VAE for occupancy prediction is a novel concept, and the dataset contribution is solid.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional comparisons and ablation studies, with dataset quality validated by human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated, and the dataset construction pipeline is highly detailed.
- Value: ⭐⭐⭐⭐⭐ High-resolution occupancy dataset + efficient prediction method, providing a significant boost to the 3D perception field in autonomous driving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SemanticHuman-HD: High-Resolution Semantic Disentangled 3D Human Generation](semantichuman-hd_high-resolution_semantic_disentangled_3d_human_generation.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] LGM: Large Multi-View Gaussian Model for High-Resolution 3D Content Creation](lgm_large_multi-view_gaussian_model_for_high-resolution_3d_content_creation.md)
- [\[ECCV 2024\] SceneVerse: Scaling 3D Vision-Language Learning for Grounded Scene Understanding](sceneverse_scaling_3d_vision-language_learning_for_grounded_scene_understanding.md)
- [\[ECCV 2024\] Open Vocabulary 3D Scene Understanding via Geometry Guided Self-Distillation](open_vocabulary_3d_scene_understanding_via_geometry_guided_self-distillation.md)

</div>

<!-- RELATED:END -->
