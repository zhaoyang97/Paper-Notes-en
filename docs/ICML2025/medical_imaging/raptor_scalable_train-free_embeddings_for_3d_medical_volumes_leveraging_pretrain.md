---
title: >-
  [Paper Note] Raptor: Scalable Train-Free Embeddings for 3D Medical Volumes Leveraging Pretrained 2D Foundation Models
description: >-
  [ICML2025][Medical Imaging][3D medical volumes] Proposes Raptor (Random Planar Tensor Reduction), a completely train-free method that leverages a frozen 2D foundation model (DINOv2-L) to extract visual tokens from 3D medical volumes along three orthogonal axes, and then substantially compresses the dimensions via random projection, outperforming all SOTA methods requiring large-scale pre-training across 10 medical tasks.
tags:
  - "ICML2025"
  - "Medical Imaging"
  - "3D medical volumes"
  - "train-free embeddings"
  - "random projection"
  - "foundation models"
  - "DINOv2"
  - "dimension reduction"
date: 2026-05-08
content_hash: d1c74f99d8e1e9c7
---

# Raptor: Scalable Train-Free Embeddings for 3D Medical Volumes Leveraging Pretrained 2D Foundation Models

**Conference**: ICML2025  
**arXiv**: [2507.08254](https://arxiv.org/abs/2507.08254)  
**Code**: [github.com/sriramlab/raptor](https://github.com/sriramlab/raptor)  
**Area**: 3D Medical Imaging  
**Keywords**: 3D medical volumes, train-free embeddings, random projection, foundation models, DINOv2, dimension reduction

## TL;DR

Proposes Raptor (Random Planar Tensor Reduction), a completely train-free method that leverages a frozen 2D foundation model (DINOv2-L) to extract visual tokens from 3D medical volumes along three orthogonal axes, and then substantially compresses the dimensions via random projection, outperforming all SOTA methods requiring large-scale pre-training across 10 medical tasks.

## Background & Motivation

Foundation models for 3D medical imaging (MRI/CT) face two major bottlenecks:

**Computational Complexity**: Extending convolutional or Transformer architectures from 2D to 3D results in a cubic or higher-order growth in computational overhead, leading to extremely high training costs (e.g., VoCo requires 8×H100 and SuPreM requires 8×A100 to train for over 7 days).

**Data Scarcity**: The largest 3D medical datasets contain only about 160K volumes, which is several orders of magnitude smaller than 2D image datasets (e.g., 1.2B images).

Meanwhile, 2D image foundation models (such as DINOv2, trained on 1.2B images) have become highly mature. The Core Problem is: **Is it possible to process 3D volume data by directly repurposing the capabilities of 2D foundation models without training any 3D models?**

## Method

### Overall Process

The core idea of Raptor consists of three steps: **tri-planar slicing $\rightarrow$ 2D foundation model encoding $\rightarrow$ random projection compression**, completely train-free throughout.

### Step 1: Tri-Planar Volume Sampling

For an input volume $\mathbf{x} \in \mathbb{R}^{D \times D \times D}$, $D$ slices are extracted along each of the three orthogonal directions: axial, coronal, and sagittal, obtaining $\mathbf{S} \in \mathbb{R}^{3 \times D \times (D \times D)}$ for a total of $3 \times D$ 2D images.

### Step 2: 2D Foundation Model Encoding

A frozen DINOv2-L (304M parameters, ViT architecture, patch size $T=16$) is used to extract tokens for each slice:

$$\mathbf{z} = \text{concat}_{1 \leq i \leq 3}[\phi(\mathbf{S}_i)] \in \mathbb{R}^{3 \times D \times d \times p^2}$$

where $d=1024$ represents the token dimension, and $p = D/T = 16$ is the number of patches per side. For $D=256$, the raw representation contains approximately 201M values (383MB), which is 127 times the size of the original volume.

### Step 3: Random Projection Compression

- **Mean Pooling**: Average along the slice dimension to compress $3 \times D \times d \times p^2$ to $3 \times d \times p^2$.
- **Random Projection**: Sample $\mathbf{R} \in \mathbb{R}^{K \times d}$, where $R_{kl} \sim \mathcal{N}(0,1)$, compressing the token dimension from $d$ to $K$.
- **Flattening**: The final Raptor embedding is:

$$\mathbf{v} = \text{flatten}\left(\text{concat}_{1 \leq i \leq 3}\left[\mathbf{R} \frac{1}{D}\sum_{j=1}^{D}\mathbf{z}_{ij}\right]\right) \in \mathbb{R}^{3Kp^2}$$

Under the typical configuration ($K=100, p=16$), the embedding size is $768 \times K = 76800$ dimensions, which is approximately 99% smaller than the original volume. The lightweight variant Raptor-B uses $K=10$, which is an additional 10 times smaller.

### Theoretical Foundation

The efficacy of random projection is guaranteed by the **Johnson–Lindenstrauss lemma**: after mapping high-dimensional points to $\mathbb{R}^K$, pairwise distances are preserved with $(1 \pm \varepsilon)$ accuracy with high probability. The temporal complexity is $\mathcal{O}(p^2 d N(D+K))$, outperforming PCA's $\mathcal{O}(p^2 d^2 N)$.

## Key Experimental Results

### Classification Tasks (3D MedMNIST, 6 Datasets, AUROC/ACC)

| Method | Organ | Nodule | Fracture | Adrenal | Vessel | Synapse |
|------|-------|--------|----------|---------|--------|---------|
| SuPreM | 0.999/0.968 | 0.891/0.848 | 0.645/0.492 | 0.906/0.869 | 0.964/0.929 | 0.907/0.879 |
| VoCo | 0.992/0.870 | 0.797/0.836 | 0.699/0.535 | 0.913/0.872 | 0.799/0.880 | 0.844/0.830 |
| Merlin | 0.976/0.766 | 0.809/0.861 | 0.691/0.549 | 0.836/0.801 | 0.870/0.879 | 0.833/0.825 |
| **Raptor** | **0.999/0.961** | **0.929/0.870** | 0.677/0.502 | 0.926/0.845 | **0.966/0.922** | **0.943/0.911** |

### Additional Classification Datasets

| Method | CC-CCII (AUC) | CTRG-C (AUC) | CTRG-B (AUC) |
|------|---------------|--------------|--------------|
| SuPreM | 0.988 | 0.613 | 0.717 |
| **Raptor** | **0.997** | **0.620** | 0.711 |

### Regression Tasks (UK Biobank Brain MRI, $r^2$, Average of 10 Brain Regions)

| Method | Average $r^2$ |
|------|-----------|
| Merlin | 0.313 |
| SuPreM | 0.299 |
| Raptor-B | 0.356 |
| **Raptor** | **0.389** |

On the regression task, Raptor improves by an average of +24% over Merlin, +30% over SuPreM, and +47% over SLIViT.

### Data Efficiency

On the Synapse dataset, using only 10 samples achieves 77% of the performance of the full dataset (1230 samples), and 100 samples achieve 88%.

### Embedding Efficiency

| Method | Parameters | Embedding Size |
|------|--------|---------|
| VoCo | 294.9M | $3072 \times 3^3$ |
| Merlin | 124.7M | $2048 \times 14 \times 7^2$ |
| SuPreM | 5.1M | $128 \times 12^3$ |
| **Raptor** | 304.4M (DINOv2-L) | $3 \times 100 \times 16^2$ |
| **Raptor-B** | Same as above | $3 \times 10 \times 16^2$ (28.8× smaller than SuPreM) |

## Highlights & Insights

1. **Completely Train-Free**: Requires zero training on 3D data, strictly utilizing a frozen 2D DINOv2 combined with random projection, which drastically lowers the computational barrier (approx. 6.5s per volume on a single RTX 2080 Ti).
2. **Counter-Intuitive Strength**: A general-purpose 2D model with no domain-specific clinical pre-training surprisingly outperforms all dedicated 3D models pre-trained on medical data through simple tri-planar slicing and compression.
3. **Model Agnosticism**: Raptor can seamlessly swap the underlying 2D foundation model, automatically benefiting from advancements in 2D models.
4. **Extreme Compression**: The Raptor-B ($K=10$) embedding is only 7,680 dimensions—28.8x smaller than SuPreM—yet achieves comparable or even superior performance.
5. **Tri-Axial Complementarity**: Ablation studies verify strong complementarity among tri-axial samplings; feature loss in one axis can be recovered by the check-and-balance of other axes (similar to triangulation).

## Limitations & Future Work

1. **Suboptimal Performance on Certain Datasets**: Performance is mediocre on certain datasets like Fracture3D (AUC 0.677 vs. VoCo 0.699), indicating that some tasks still require domain-specific priors.
2. **Limited Spatial Resolution**: Simulation experiments reveal a sharp drop in detection capability (AUC drops to ~0.5) when target features are smaller than 16px, bounded by the ViT patch size.
3. **Information Loss from Mean Pooling**: Averaging along the slice direction discards fine-grained spatial position information.
4. **Restricted to Voxel Data**: The method is not directly applicable to other 3D representations like point clouds or meshes.
5. **Downstream Classifier Training Still Required**: Although the embeddings are train-free, downstream tasks still necessitate fitting a logistic regression or MLP.

## Related Work & Insights

- **SuPreM / VoCo / Merlin / MISFM**: Represent the mainstream paradigm of current 3D medical pre-training, all requiring substantial computation and data.
- **DINOv2**: This work successfully demonstrates the transferability of general-purpose 2D foundation models to the 3D medical domain.
- **Johnson–Lindenstrauss Lemma**: Provides rigorous theoretical guarantees for random projection dimensionality reduction.
- **Insights**: For data-scarce, high-dimensional modalities (such as 4D fMRI or video), a similar 'slicing + 2D encoding + compression' paradigm is highly worth exploring.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Processing 3D volumes with a train-free approach and random projection constitutes a brand-new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 10 datasets + 6 baselines + comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, balancing theory and experiments.
- Value: ⭐⭐⭐⭐⭐ — Drastically lowers the barrier to entry for 3D medical image analysis, runnable on a single GPU.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification](../../CVPR2026/medical_imaging/revisiting_2d_foundation_models_for_scalable_3d_medical_image_classification.md)
- [\[CVPR 2025\] Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](../../CVPR2025/medical_imaging/developing_foundation_models_for_universal_segmentation_from_3d_whole-body_posit.md)
- [\[CVPR 2025\] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation?](../../CVPR2025/medical_imaging/are_general-purpose_vision_models_all_we_need_for_2d_medical_image_segmentation_.md)
- [\[NeurIPS 2025\] PolyPose: Deformable 2D/3D Registration via Polyrigid Transformations](../../NeurIPS2025/medical_imaging/polypose_deformable_2d3d_registration_via_polyrigid_transformations.md)
- [\[ICLR 2026\] Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity](../../ICLR2026/medical_imaging/improving_2d_diffusion_models_for_3d_medical_imaging_with_inter-slice_consistent.md)

</div>

<!-- RELATED:END -->
