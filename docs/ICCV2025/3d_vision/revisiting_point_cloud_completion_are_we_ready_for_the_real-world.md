---
title: >-
  [Paper Note] Revisiting Point Cloud Completion: Are We Ready For The Real-World?
description: >-
  [ICCV 2025][3D Vision][Point cloud completion] Using algebraic topology and persistent homology ($\mathcal{PH}$) tools, this paper reveals that existing synthetic point cloud datasets lack the rich topological features present in real-world data. It contributes the first real-world industrial point cloud completion dataset RealPC (~40,000 pairs, 21 categories), and proposes BOSHNet, which samples proxy homology skeletons as topological priors to achieve significant improvements on real-world point cloud completion.
tags:
  - ICCV 2025
  - 3D Vision
  - Point cloud completion
  - real-world dataset
  - persistent homology
  - topological prior
  - industrial point cloud
date: 2026-05-08
content_hash: 7e5be38b78a927f2
---

# Revisiting Point Cloud Completion: Are We Ready For The Real-World?

**Conference**: ICCV 2025
**arXiv**: [2411.17580](https://arxiv.org/abs/2411.17580)
**Code**: Coming soon (RealPC dataset + BOSHNet code)
**Area**: 3D Vision
**Keywords**: Point cloud completion, real-world dataset, persistent homology, topological prior, industrial point cloud

## TL;DR

Using algebraic topology and persistent homology ($\mathcal{PH}$) tools, this paper reveals that existing synthetic point cloud datasets lack the rich topological features present in real-world data. It contributes the first real-world industrial point cloud completion dataset RealPC (~40,000 pairs, 21 categories), and proposes BOSHNet, which samples proxy homology skeletons as topological priors to achieve significant improvements on real-world point cloud completion.

## Background & Motivation

**Point cloud completion** is a fundamental task in 3D vision: recovering complete 3D shapes from partial point clouds. However, the field suffers from a fundamental **data–method disconnect**:

**Synthetic datasets are too "clean"**: Mainstream datasets (PCN, ShapeNet55, MVP, etc.) are uniformly sampled from CAD models, lacking noise, non-uniform sparsity, and complex topological structures. Existing methods are approaching saturation on these benchmarks.

**Real-world point clouds are fundamentally different**: Point clouds acquired by real sensors exhibit three distinctive characteristics:
   - **Noise**: Point displacement due to measurement error
   - **Non-uniform sparsity**: Sparse at distance, dense nearby, missing in occluded regions
   - **Rich topological features**: Complex industrial structures contain connected components (0-dimensional homology) and loops/holes (1-dimensional homology)

**Lack of real-world paired datasets**: Real-world datasets such as OmniObject3D and ScanNet are either captured in controlled environments or do not provide ground-truth complete shapes, making them unsuitable for supervised training.

**Core finding**: Persistent homology ($\mathcal{PH}$) analysis reveals that real-world point clouds contain abundant and significant 0- and 1-dimensional topological features (points far from the diagonal in persistence diagrams), whereas such features are nearly absent in PCN/ShapeNet. This discrepancy is the key reason existing methods fail on real data.

## Method

### I. RealPC Dataset Construction

Individual industrial structures are extracted from four open-source scene-level railway point cloud datasets to construct paired complete–partial point cloud data:

**Construction pipeline** (five steps):
1. **(A) HDBSCAN clustering**: Separate individual industrial structures from scene-level point clouds
2. **(B) Manual inspection**: Extract complete structures as ground truth (GT)
3. **(C) Non-uniform incompleteness**: Select a random viewpoint and remove the $N$ points farthest from it
4. **(D) Non-uniform sparsity**: Select a random viewpoint and sample $N$ points with probability proportional/inversely proportional to the cube of distance
5. **(E) Uniform sparsity**: Randomly sample $N$ points

**Scale**: ~40,000 pairs, 21 categories of industrial structures (from multiple countries, different sensors).

### II. Quantitative Analysis: RealPC vs. Existing Datasets

Three metrics are used to quantify differences (mean ×10⁻⁴):

| Metric | PCN | ShapeNet | **RealPC** |
|--------|-----|----------|------------|
| Noise | 12.2 | 23.7 | **113.7** |
| Non-uniformity | 19.7 | 31.8 | **173.8** |
| PH (H₀; H₁) | 86.0; 37.5 | 101.3; 41.0 | **345.2; 155.5** |

RealPC substantially exceeds existing datasets across all three metrics: 9× higher noise, 9× higher non-uniformity, and 4× richer topological features.

### III. TopODGNet: PH-Regularized Completion

ODGNet is selected as the base model (its decoder generates multiple sparse seed point clouds, suitable for computing PH priors). 0-dimensional $\mathcal{PH}$ priors are extracted from the seed point clouds as global topological skeletons:

**Topological loss**: Minimizes the total persistence of all 0-dimensional persistence pairs, ensuring that only a single connected component remains at the end of the filtration:

$$\text{TopoLoss} = \sum_{i=k+1}^n (b_i - d_i)$$

where $k$ allows retaining multiple connected components (when the input partial point cloud is itself fragmented into multiple segments). The 0-dimensional $\mathcal{PH}$ skeleton summarizes the global topology of the complete point cloud, guiding the network to generate points along the skeleton.

**Limitation**: $\mathcal{PH}$ computation is expensive (the number of simplices in the Vietoris–Rips complex grows exponentially with point cloud size), and the resulting improvement is limited.

### IV. BOSHNet: Backbone Outline Sampler for PH

**Core hypothesis**: The 0-dimensional $\mathcal{PH}$ skeleton is essentially a sparse sampling of the GT shape surface. Rather than incurring the high computational cost of extracting $\mathcal{PH}$, proxy skeletons can be obtained by directly **multi-scale sampling** from the GT surface.

**BOSH sampler** (Backbone Outline Sampler for PH): Samples $k$ skeletons from the GT point cloud surface at varying sparsity levels, which are provided as additional inputs during training. These skeletons supply accurate global shape information from the very beginning of training—unlike TopODGNet, where skeletons only become well-defined at later training stages.

**Loss function**:
$$\sum_{i=1}^n \sum_{j=1}^k M(\text{Net}(\text{BOSH}(c_i, j)), c_i) + \sum_{i=1}^n M(\text{Net}(p_i), c_i)$$

The first term trains with sampled skeletons; the second term trains with the original partial point cloud. $M$ denotes Chamfer Distance.

**Dual benefits**: (1) Forces the model to attend to precise shape details at multiple resolutions; (2) completely bypasses expensive $\mathcal{PH}$ computation.

## Key Experimental Results

### Main Results I: RealPC Benchmark (Chamfer Distance ×10⁻³)

| Method | RealPC Avg (L1) | RealPC Avg (L2) | PCN Avg (L1) |
|--------|----------------|----------------|--------------|
| ODGNet | 119 | 111 | 6 |
| FoldingNet | 167 | 127 | 14 |
| PCN | 143 | 92 | 10 |
| PointTr | 114 | 58 | 8 |
| AdaPoinTr | 69 | 26 | 7 |
| SnowflakeNet | 60 | 72 | 7 |
| GRNet | 84 | 27 | 9 |
| AnchorFormer | 72 | 28 | 7 |

All methods exhibit errors on RealPC that are **an order of magnitude** higher than on PCN (e.g., SnowflakeNet: 60 vs. 7), demonstrating that existing methods cannot handle real-world point clouds.

### Main Results II: TopODGNet vs. BOSHNet

| Method | CD-L1 | CD-L2 |
|--------|-------|-------|
| ODGNet (baseline) | 119 | 111 |
| TopODGNet (+PH regularization) | 103 | 80 |
| SnowflakeNet | 60 | 72 |
| **BOSHNet** | **69** | **5.4** |

BOSHNet achieves a dramatic improvement on CD-L2 (5.4 vs. SnowflakeNet's 72, a **13× gain**), with CD-L1 comparable to the best baseline. Topological priors yield meaningful but limited gains in TopODGNet (−16), while BOSHNet achieves substantially larger improvements by replacing PH computation with skeleton sampling.

### Non-Neural Baselines (RealPC vs. ShapeNet)

| Task | ShapeNet CD | RealPC CD |
|------|------------|-----------|
| Simplification (WLOP) | 0.522 | **8.817** |
| Upsampling | 0.043 | **0.838** |

Non-neural methods also exhibit 15–20× higher errors on RealPC.

### Generation & Reconstruction Benchmarks

| Task | ShapeNet 1-NNA | RealPC 1-NNA |
|------|---------------|-------------|
| Generation (CD) | 67 | **90** |
| Generation (EMD) | 61 | **90** |

Diffusion-based generative models also perform poorly on RealPC.

## Highlights & Insights

1. **TDA tools reveal a fundamental data gap**: Persistent homology analysis quantitatively demonstrates the topological discrepancy between synthetic and real-world point clouds—not merely "more noise," but a fundamentally higher level of structural complexity.
2. **RealPC fills the gap in real-world paired data**: ~40K pairs, 21 categories, multiple sensors, multiple incompleteness modes—lowering the barrier to real-world point cloud research.
3. **BOSHNet's elegant simplicity**: Multi-scale surface sampling replaces expensive PH computation while achieving superior performance. The key insight is that "the 0-dimensional PH skeleton ≈ a sparse sampling of the GT."
4. The paper's primary contribution lies in **problem formulation and dataset construction** rather than a single methodological breakthrough—calling on the community to reconsider the point cloud completion task as a whole.

## Limitations & Future Work

1. RealPC covers only railway/industrial structures, with limited shape diversity (lacking everyday objects such as furniture and vehicles).
2. BOSHNet requires GT complete point clouds to sample skeletons during training and cannot be directly applied in unpaired data settings.
3. Higher-dimensional topological features (e.g., 2-dimensional PH) remain unexplored.
4. The improvement in CD-L1 from BOSHNet is modest (69 vs. SnowflakeNet's 60); the primary advantage lies in CD-L2.

## Related Work & Insights

- **Point cloud completion datasets**: PCN (ShapeNet 8 classes), ShapeNet55/34, MVP (16 classes, 100K+), KITTI (no GT)
- **Completion methods**: PCN, PointTr/AdaPoinTr (Transformer-based), SnowflakeNet (snowflake point generation), ODGNet (orthogonal dictionary seeds)
- **Persistent homology in vision**: TopologyNet (topological regularization for segmentation), topological autoencoders, PH in surface reconstruction

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Value | 4 |
| Overall | 4.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective](../../AAAI2026/3d_vision/rethinking_multimodal_point_cloud_completion_a_completion-by-correction_perspect.md)
- [\[AAAI 2026\] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion](../../AAAI2026/3d_vision/dapointmamba_domain_adaptive_point_mamba_for_point_cloud_completion.md)
- [\[NeurIPS 2025\] PointMAC: Meta-Learned Adaptation for Robust Test-Time Point Cloud Completion](../../NeurIPS2025/3d_vision/pointmac_meta-learned_adaptation_for_robust_test-time_point_cloud_completion.md)
- [\[ICCV 2025\] Demeter: A Parametric Model of Crop Plant Morphology from the Real World](demeter_a_parametric_model_of_crop_plant_morphology_from_the_real_world.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)

</div>

<!-- RELATED:END -->
