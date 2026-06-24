---
title: >-
  [Paper Note] Divide and Fuse: Body Part Mesh Recovery from Partially Visible Human Images
description: >-
  [ECCV 2024][3D Vision][Human Mesh Recovery] A "divide-and-conquer" bottom-up human mesh recovery method is proposed, which reconstructs individual body parts independently and then fuses them, effectively solving the failure mode of traditional top-down methods (such as SMPL) when large areas of the human body are invisible.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Human Mesh Recovery"
  - "Partially Visible"
  - "Occlusion"
  - "Parametric Model"
  - "Bottom-Up"
date: 2026-05-08
content_hash: 903f3df72d39f020
---

# Divide and Fuse: Body Part Mesh Recovery from Partially Visible Human Images

**Conference**: ECCV 2024  
**arXiv**: [2407.09694](https://arxiv.org/abs/2407.09694)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Human Mesh Recovery, Partially Visible, Occlusion, Parametric Model, Bottom-Up

## TL;DR

A "divide-and-conquer" bottom-up human mesh recovery method is proposed, which reconstructs individual body parts independently and then fuses them, effectively solving the failure mode of traditional top-down methods (such as SMPL) when large areas of the human body are invisible.

## Background & Motivation

**Background**: Mainstream human mesh recovery methods adopt a top-down design, relying on full-body parametric models such as SMPL to regress full-body parameters after extracting global features from images. This works well when most of the body is visible.

**Limitations of Prior Work**: When large areas of the human body are occluded or out of view in the input image (e.g., only legs are visible), the performance of these methods degrades severely. This is due to two reasons: (a) the network struggles to recognize the human body from a few visible parts; (b) the parameters of different parts in SMPL are coupled, and the lack of information for invisible parts interferes with the reconstruction of visible parts.

**Key Challenge**: The parameter coupling of top-down models means local reconstruction must rely on global information, which is unreliable in partially visible scenarios.

**Goal**: Accurately reconstruct the 3D meshes of visible body parts even when only a few parts are visible.

**Key Insight**: If body parts are reconstructed independently, the issues of "difficulty in capturing information" and "interference between parts" can be naturally avoided.

**Core Idea**: Design independent Human Part Parametric Models (HPPM) to reconstruct individual parts and then fuse them, achieving robust reconstruction for partially visible humans.

## Method

### Overall Architecture

Input monocular partially visible human image → Swin Transformer extracts features → MLP regresses HPPM parameters (shape, rotation, translation) for each part → individual HPPMs independently generate part meshes → Fusion module connects adjacent visible parts → Output human meshes of visible parts.

### Key Designs

#### 1. Human Part Parametric Models (HPPM)

- **Function**: Design independent parametric mesh models for each of the 15 body parts to reconstruct individual parts without inter-part dependence.
- **Mechanism**:
    - Starting from the SMPL template mesh, vertices are assigned to corresponding bones based on the blend weight matrix $W$: $p_i = \arg\max_j W_{ij}$
    - Manually merge adjacent near-rigid parts (e.g., neck-shoulder and torso), resulting in 15 body parts.
    - Dilate each part so that **overlapping regions** exist between adjacent parts, facilitating subsequent fusion.
    - Perform PCA on a large amount of SMPL ground truth data for dimensionality reduction to train a linear mapping matrix $\mathcal{U}_p \in \mathbb{R}^{k \times n}$, mapping $k$-dimensional shape parameters to part meshes.
    - The parameter dimension of each part is adjustable (16 to 42 dimensions), adaptively set based on fitting accuracy to control the maximum error within 2mm.
- **Design Motivation**: Unlike SMPL, the parts in HPPM are completely decoupled. They can be reconstructed independently via simple global rigid transformations and a small number of shape parameters, without explicit pose modeling.

#### 2. Divide: Independent Part Reconstruction Network

- **Function**: Regress HPPM parameters for each part from image features, and independently reconstruct each part's mesh.
- **Mechanism**:
    - Swin Transformer extracts image features, and an MLP predicts the shape parameter $\hat{S}_p$, 6D rotation $\hat{R}_p$, and translation $\hat{T}_p$ for each part respectively.
    - Part mesh generation: $\hat{v}_p = \hat{M}_p(\mathcal{U}_p \hat{S}_p + \mathcal{M}_p)$, where $\hat{M}_p$ is the global transformation matrix.
    - Joint locations are computed via a regressor: $\hat{J}_p = \mathcal{J}_p \hat{v}_p$.
- **Design Motivation**: All supervision signals are defined on individual parts, ensuring that invisible parts do not affect the reconstruction of visible parts.

#### 3. Fuse: Adjacent Part Fusion Module

- **Function**: Seamlessly connect multiple adjacent visible parts into a unified mesh.
- **Mechanism**:
    - Vertices in overlapping regions are processed using a weighted average strategy based on topological distance: $v_k^c = \frac{\hat{v}_{p_1 i} d_{2j} + \hat{v}_{p_2 i} d_{1i}}{d_{1i} + d_{2j}}$.
    - Where $d_{1i}$ is the topological distance from the vertex to the nearest non-overlapping vertex, achieving a smooth transition.
    - Non-overlapping regions directly use the vertices of corresponding parts.
- **Design Motivation**: The overlapping region design combined with gradient weighting avoids unnatural distortion at boundaries.

### Loss & Training

**Divide Stage Loss**:
$$\mathcal{L}_{div} = \lambda_v \mathcal{L}_v + \lambda_{j3d} \mathcal{L}_{j3d} + \lambda_{j2d} \mathcal{L}_{j2d} + \lambda_s \mathcal{L}_s + \lambda_r \mathcal{L}_r + \lambda_t \mathcal{L}_t$$

Includes part vertex loss, 3D joint loss, 2D projected joint loss, shape parameter loss, rotation loss, and translation loss, all weighted by the visibility mask $\delta_p$.

**Fuse Stage Loss**:
$$\mathcal{L}_{fu} = \lambda_{ol} \mathcal{L}_{ol} + \lambda_{dc} \mathcal{L}_{dc}$$

- Overlap loss $\mathcal{L}_{ol}$: Constrains overlapping vertices of adjacent parts to approach their average position.
- Depth consistency loss $\mathcal{L}_{dc}$: Constrains the z-direction depth consistency of non-directly adjacent parts in the same image.

**Total Loss**: $\mathcal{L} = \mathcal{L}_{div} + \mathcal{L}_{fu}$

**Training Data Augmentation**: During training, a similar image cropping strategy is applied to public datasets to simulate partially visible inputs. Evaluation is conducted using self-constructed PV-Human3.6M and PV-3DPW benchmarks.

## Key Experimental Results

### Main Results

| Method | PV-H36M MPVE↓ | PV-H36M MPJPE↓ | PV-3DPW MPVE↓ | PV-3DPW MPJPE↓ |
|------|---------------|-----------------|---------------|-----------------|
| MotionBERT | 196.9 | 169.6 | 185.5 | 155.4 |
| SEFD | 276.0 | 198.9 | 241.1 | 203.0 |
| GLoT | 214.1 | 199.0 | 235.0 | 213.5 |
| CycleAdapt | 249.4 | 231.9 | 189.0 | 137.1 |
| **D&F (Ours)** | **63.3**| **55.9** | **109.9** | **102.7** |

(The above are fine-tuned results; direct testing also shows that ours significantly outperforms other methods)

### Ablation Study

| Configuration | PV-H36M MPVE/MPJPE | PV-3DPW MPVE/MPJPE | Description |
|------|---------------------|---------------------|------|
| w/o 2D projection loss | 64.2/56.7 | 111.8/104.9 | Slight drop |
| w/o 3D joint loss | 63.9/56.2 | 112.4/105.8 | Slight drop |
| w/o 3D vertex loss | 68.4/63.5 | 120.2/108.3 | Obvious drop |
| w/o shape parameter loss | 70.1/57.0 | 119.7/103.9 | Obvious drop |
| w/o 6D rotation loss | 95.6/87.5 | 138.5/127.4 | **Largest drop** |
| w/o translation loss | 75.1/69.9 | 123.0/115.3 | Significant drop |
| w/o overlap loss | 74.5/64.2 | 125.2/111.8 | Significant drop |
| Fixed parameter dimension | 67.5/61.7 | 114.0/105.4 | Adjustable dimensions are better |
| **Full D&F** | **63.3/55.9** | **109.9/102.7** | — |

### Key Findings

- The 6D rotation loss has the greatest impact on performance; removing it increases MPVE by over 50%.
- Adjustable parameter dimensions perform better than fixed dimensions; a total parameter size of 360 can represent all 15 body parts.
- HPPM training fitting error is only 1.11mm (vertices) and 1.46mm (joints) on average.
- It still outperforms SOTA when only 5-10 parts are visible (MPVE 117.4 vs CycleAdapt 169.9).

## Highlights & Insights

- **Novelty**: Proposes the first bottom-up learning-based human mesh recovery method, specifically designed for scenarios with large-scale invisibility.
- **Elegant HPPM Design**: The overlapping region design kills two birds with one stone—it contains joint deformation information and simplifies fusion.
- **Adjustable Parameter Dimensions**: Shape variation differs greatly across parts (torso > limbs/hands/feet); adaptive dimension allocation is more efficient.
- **Huge Gain**: Significantly reduces MPVE from ~200mm to ~60-110mm in partially visible scenarios.

## Limitations & Future Work

- Currently only reconstructs visible parts, failing to utilize inter-part priors to infer invisible parts.
- The fusion module is designed as post-processing and is not optimized end-to-end jointly.
- The division into 15 parts is empirical and may not be fine-grained enough for certain extreme poses.
- Future work can consider incorporating temporal information or diffusion model priors to complete invisible regions.

## Related Work & Insights

- **vs. SMPL-based Methods**: Parameter coupling in SMPL causes global regression to fail when partially visible; this work thoroughly solves this via part decoupling.
- **vs. Occlusion-handling Methods** (e.g., OCHMR): These methods focus on inferring occluded parts from visible parts, requiring high global recognition performance; this work focuses on the accurate reconstruction of the visible parts themselves.
- **vs. Bottom-up Pose Estimation**: Bottom-up approaches are mature in pose estimation, but inferring complete meshes from sparse keypoints is an ill-posed problem; our part parametric model fills this gap.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ A brand new paradigm for bottom-up human mesh recovery, with a unique HPPM design
- Experimental Thoroughness: ⭐⭐⭐⭐ Main experiment + ablation + adjustable parameter analysis, reasonable self-constructed benchmarks; but lacks testing on real occluded scenes
- Writing Quality: ⭐⭐⭐⭐ Clear structure and complete mathematical derivations
- Value: ⭐⭐⭐⭐ Practical value for partially visible human reconstruction in AR/VR, medical, and other scenarios

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot](multi-hmr_multi-person_whole-body_human_mesh_recovery_in_a_single_shot.md)
- [\[ECCV 2024\] Global-to-Pixel Regression for Human Mesh Recovery](global-to-pixel_regression_for_human_mesh_recovery.md)
- [\[CVPR 2026\] MetricHMSR: Metric Human Mesh and Scene Recovery from Monocular Images](../../CVPR2026/3d_vision/metrichmsr_metric_human_mesh_and_scene_recovery_from_monocular_images.md)
- [\[CVPR 2025\] PromptHMR: Promptable Human Mesh Recovery](../../CVPR2025/3d_vision/prompthmr_promptable_human_mesh_recovery.md)
- [\[CVPR 2025\] MEGA: Masked Generative Autoencoder for Human Mesh Recovery](../../CVPR2025/3d_vision/mega_masked_generative_autoencoder_for_human_mesh_recovery.md)

</div>

<!-- RELATED:END -->
