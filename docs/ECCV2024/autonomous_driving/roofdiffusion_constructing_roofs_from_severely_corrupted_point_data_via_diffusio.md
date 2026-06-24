---
title: >-
  [Paper Note] RoofDiffusion: Constructing Roofs from Severely Corrupted Point Data via Diffusion
description: >-
  [ECCV 2024][Autonomous Driving][Diffusion Models] RoofDiffusion proposes an end-to-end self-supervised method based on conditional diffusion probabilistic models to restore complete and clean elevation information from severely sparse (up to 99% missing), incomplete (80% area occluded), and noisy roof height maps. It significantly outperforms traditional interpolation methods and existing depth completion methods on the self-created PoznanRD dataset and BuildingNet.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Diffusion Models"
  - "Height Map Inpainting"
  - "Roof Reconstruction"
  - "Depth Completion"
  - "Remote Sensing"
date: 2026-05-08
content_hash: 934f7daa20b56888
---

# RoofDiffusion: Constructing Roofs from Severely Corrupted Point Data via Diffusion

**Conference**: ECCV 2024  
**arXiv**: [2404.09290](https://arxiv.org/abs/2404.09290)  
**Code**: [https://github.com/kylelo/RoofDiffusion](https://github.com/kylelo/RoofDiffusion)  
**Area**: Autonomous Driving  
**Keywords**: Diffusion Models, Height Map Inpainting, Roof Reconstruction, Depth Completion, Remote Sensing

## TL;DR

RoofDiffusion proposes an end-to-end self-supervised method based on conditional diffusion probabilistic models to restore complete and clean elevation information from severely sparse (up to 99% missing), incomplete (80% area occluded), and noisy roof height maps. It significantly outperforms traditional interpolation methods and existing depth completion methods on the self-created PoznanRD dataset and BuildingNet.

## Background & Motivation

**Background**: Digital Surface Models (DSMs, i.e., height maps) are core data sources for reconstructing 3D urban buildings. OpenStreetMap annotates over 500 million buildings globally; even if only 1% are problematic, it means 5 million buildings need repairing. Empirical investigation shows that up to 34%-50% of roof height maps in USGS 3DEP LiDAR data suffer from some degree of corruption.

**Limitations of Prior Work**: Real-world roof height maps face three major challenges: (1) **Sparsity**—low-resolution sensors or poor surface reflectivity result in extremely low point cloud density; (2) **Incompleteness**—environmental occlusions, surrounding high buildings, or non-nadir perspectives lead to large-area roof data loss; (3) **Noise**—tree canopies invade building footprints, causing LiDAR to capture trees instead of roofs. Existing DEM inpainting methods (IDW, Kriging, Spline) are effective in small areas but fail on large, complex regions; depth completion methods are designed for uniformly distributed sparse depth maps and cannot handle large missing areas.

**Key Challenge**: The corruption patterns of roof height maps (extreme sparsity + large regional missing areas + tree noise) do not match the assumptions of existing methods—traditional interpolation lacks geometric priors, while depth completion assumes uniform sparse distributions.

**Key Insight**: Conceptualizing roof height map restoration as an image inpainting task, leveraging diffusion models to learn the strong prior distribution of roof height maps, and using footprints as conditioning information to guide the restoration process. This is inspired by the JPEG image restoration method, Palette.

**Core Idea**: Training a diffusion model conditioned on corrupted height maps and footprints to learn the prior distribution of roof structures, combined with carefully designed data synthesis strategies (tree noise simulation, multi-Gaussian mask incompleteness simulation) to achieve self-supervised training.

## Method

### Overall Architecture

The pipeline of RoofDiffusion: (1) the input corrupted height map $\mathbf{z}$ is transformed to $\mathbf{x}$ within the range of $[-1, 1]$ via roof-focused normalization; (2) the forward process gradually adds noise to the clean height map $\mathbf{x}_0$; (3) the reverse process progressively denoises using a conditional diffusion model $\boldsymbol{\epsilon}_\theta$, with the corrupted height map $\mathbf{x}$ and noise schedule parameter $\bar{\alpha}_t$ as conditioning inputs; (4) the recovered output $\hat{\mathbf{x}}_0$ is denormalized back to the real height. Training data are obtained by automatically synthesizing corrupted height maps, requiring no manual annotations.

### Key Designs

1. **Roof-Focused Height Map Normalization**:

    - **Function**: Unifying various building heights by normalizing them to $[-1,1]$ to adapt to the diffusion model.
    - **Mechanism**: First identifying the lowest pixel of the roof and subtracting this value (focusing on the relative roof structure), then normalizing using the global maximum height difference $\overline{\underline{\mathbf{z}}}$ (excluding the top 1% outliers):
    $$\mathbf{x} = \frac{2}{\overline{\underline{\mathbf{z}}}} \left(\mathbf{z} - \frac{1}{2}(\mathbf{z}_{\max} + \mathbf{z}_{\min})\right)$$
      Based on an analysis of 13k buildings, the truncation threshold is set to 10 meters.
    - **Design Motivation**: Autonomous driving depth maps can be normalized with a fixed range, but roof heights vary drastically (bungalows vs. skyscrapers), requiring adaptive normalization. Subtracting the lowest point allows the model to focus solely on roof structural variations.

2. **Footprint Embedded Forward Process**:

    - **Function**: Encoding building footprint information during the forward diffusion process, allowing the model to perceive the footprint without requiring extra input channels.
    - **Mechanism**: The forward diffusion process does not add noise to the entire image uniformly, but restricts noise addition to the footprint region and sets pixels outside the footprint to -1:
    $$\mathbf{x}_t = \mathbf{m} \odot (\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}) - \mathbf{m}'$$
      where $\mathbf{m}$ is the footprint mask and $\mathbf{m}'$ is its complement.
    - **Design Motivation**: Two advantages: (1) the model can directly infer the footprint location from the noisy regions in $\mathbf{x}_t$ without extra input channels; (2) pixels in non-building regions are fixed to -1, preventing them from interfering with predictions.

3. **Footprint Masked Loss**:

    - **Function**: Computing the noise prediction loss strictly within the footprint region.
    - **Mechanism**: Using L1 loss restricted within the footprint:
    $$L = \mathbb{E}_{(\mathbf{x}_0, \mathbf{x}, \mathbf{m}), t, \boldsymbol{\epsilon}} \|\mathbf{m} \odot (\boldsymbol{\epsilon} - \tilde{\boldsymbol{\epsilon}}_\theta)\|_1$$
    - **Design Motivation**: There is no need to predict regions outside the roof, thereby concentrating model capacity on roof structure restoration.

4. **Data Synthesis Strategy**:

    - **Tree Planting (tree noise simulation)**: Collecting 1k real tree height maps, randomly placing them around the footprint, and merging the canopy with the roof using a max operation.
    - **Multi-Gaussian mask incompleteness simulation**: Masking the roof area with multiple Gaussian masks of varying positions and variances to simulate various occlusion patterns—multiple Gaussians on the same side simulate entire side occlusions, scattered small Gaussians simulate small feature occlusions, and large variances simulate soft boundaries.
    - **No-FP variant**: The footprint-free version, which simultaneously predicts both the footprint and the height.

5. **PoznanRD Dataset**:

    - **Function**: Providing a high-quality dataset of 13k buildings with complex roof geometries to support training and benchmarking.
    - **Source**: 16k LoD 2.2 level roof meshes from Poznań, Poland; reduced flat roofs to 2k, retaining 13k in total (10k for training, 3k for testing).
    - **Design Motivation**: Existing datasets are either small in quantity (BuildingNet 2k) or geometrically simple (LoD 2.0, missing dormers and gables), and lack a focus on corrupted data. PoznanRD focuses on complex "long-tail" geometries.

### Loss & Training

- Loss: Footprint-masked L1 loss (as formulated above)
- Self-supervised training: Render height maps from clean meshes as ground truth (GT), dynamically synthesizing corrupted versions as conditional inputs
- Noise injection: Global Gaussian noise $\sigma \in [0, 0.05]$, outliers with a probability of 0.01%, and 1-3 tree noise instances with a probability of 30%
- Reverse process: Uses a conditional DDPM with $(\mathbf{x}_t, \mathbf{x}, \bar{\alpha}_t)$ as the conditioning input to the U-Net

## Key Experimental Results

### Main Results

**PoznanRD Height Map Completion (with Footprint):**

| Method | s95 i30 MAE | s95 i30 RMSE | s99 i80 MAE | s99 i80 RMSE |
|------|------------|-------------|------------|-------------|
| Linear | 0.236 | 0.461 | 0.868 | 1.218 |
| IDW | 0.239 | 0.449 | 0.827 | 1.172 |
| Spline | 0.278 | 0.508 | 0.888 | 1.260 |
| P.M. Diff. | 0.266 | 0.473 | 3.085 | 3.548 |
| **RoofDiffusion** | **0.162** | **0.342** | **0.603** | **0.916** |

Under the extreme conditions of 99% sparsity + 80% incompleteness, RoofDiffusion's MAE is 27% lower than the best-performing baseline.

**Footprint-free Version vs. Depth Completion Methods (PoznanRD):**

| Method | s95 i30 MAE | s99 i60 MAE |
|------|------------|------------|
| pNCNN | 1.635 | 2.172 |
| CU-Net | 1.246 | 1.923 |
| **No-FP RoofDiffusion** | **0.319** | **1.200** |

**3D Reconstruction Enhancement (City3D Preprocessor):**

| Preprocessing Method | s99 i30 RMSE | s99 i80 RMSE | Average Face Count |
|-----------|-------------|-------------|---------|
| City3D + IDW | 0.352 | 0.708 | 105-124 |
| City3D + P.M. Diff | 0.577 | 3.016 | 89-97 |
| **City3D + Ours** | **0.244** | **0.534** | **80-83** |

When GT point clouds are input to City3D, the RMSE is 0.104 and the face count is 82.68; the face count after RoofDiffusion preprocessing is extremely close to the GT.

### Ablation Study

| Configuration | Description |
|------|------|
| Footprint Guidance vs. Footprint-Free | The footprint version can handle 99% sparsity + 80% incompleteness, whereas the footprint-free version is effective below s95 |
| Tree Noise Augmentation | Ablation shows that tree simulation augmentation is crucial for robustness against real-world tree noise |
| Footprint Prediction IoU | The No-FP version achieves a 92.14% IoU under s95 i30, significantly outperforming CU-Net's 82.12% |

### Key Findings

- RoofDiffusion shows the most significant advantage in incompleteness restoration, indicating that the diffusion model has learned strong structural priors of roofs.
- Models trained on PoznanRD generalize well to the unseen BuildingNet dataset.
- Demonstrates strong synthetic-to-real transfer capability on real-world LiDAR data (AHN3, Dales3D, USGS 3DEP).
- Perona-Malik diffusion completely fails under extreme conditions (with MAE soaring from 0.266 to 3.085), while RoofDiffusion remains stable.

## Highlights & Insights

1. **Precise Problem Definition**: Elegantly reformulates the remote sensing/3D reconstruction problem of roof restoration as conditional image inpainting, leveraging the strong prior capability of diffusion models to tackle extreme corruptions.
2. **Elegant Footprint Encoding**: Naturally embeds footprint information through masking operations in the forward process, requiring no additional input channels.
3. **Comprehensive Data Synthesis Strategy**: Simulates tree noise by overlaying real tree height maps and incompleteness using multi-Gaussian masks, covering major real-world corruption patterns.
4. **Closed Engineering Loop**: Beyond height restoration, it validates its end-to-end utility as a preprocessor for City3D.

## Limitations & Future Work

- It may produce "hallucinations" in regions with extremely sparse data—generating plausible but non-existent roof structures (such as dormers).
- Currently only processes individual buildings, neglecting spatial relationships among building groups.
- The normalization strategy may not be robust enough for buildings with extreme height differences (such as skyscrapers).
- Inference speed is limited by the number of diffusion sampling steps, requiring acceleration strategies for real-world deployment.

## Related Work & Insights

- Inspired by Palette (JPEG image restoration), it utilizes conditional diffusion models but implements extensive domain-specific customizations for height maps.
- Compared with depth completion methods (pNCNN, CU-Net), the generative prior of diffusion models shows a massive advantage in handling "structural missingness."
- Insight: For domain-specific restoration/completion tasks, conditioning diffusion models with domain priors (such as footprints) combined with self-supervised training on synthetic data is a highly effective paradigm.
- The PoznanRD dataset itself holds independent value for research in roof reconstruction.

## Rating

- Novelty: ⭐⭐⭐⭐ First to apply diffusion models to DSM height map completion; the footprint encoding approach is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Synthetic + real data, quantitative + qualitative, multiple baselines, and downstream 3D reconstruction validation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-defined problem formulation, and valuable dataset contribution.
- Value: ⭐⭐⭐⭐ Holds practical application value for large-scale urban 3D reconstruction, and the dataset drives field development.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SFPNet: Sparse Focal Point Network for Semantic Segmentation on General LiDAR Point Clouds](sfpnet_sparse_focal_point_network_for_semantic_segmentation_on_general_lidar_poi.md)
- [\[CVPR 2025\] SuperPC: A Single Diffusion Model for Point Cloud Completion, Upsampling, Denoising, and Colorization](../../CVPR2025/autonomous_driving/superpc_a_single_diffusion_model_for_point_cloud_completion_upsampling_denoising.md)
- [\[ECCV 2024\] Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation](optimizing_diffusion_models_for_joint_trajectory_prediction_and_controllable_gen.md)
- [\[ECCV 2024\] Reliability in Semantic Segmentation: Can We Use Synthetic Data?](reliability_in_semantic_segmentation_can_we_use_synthetic_data.md)
- [\[ECCV 2024\] Safe-Sim: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries](safe-sim_safety-critical_closed-loop_traffic_simulation_with_diffusion-cont.md)

</div>

<!-- RELATED:END -->
