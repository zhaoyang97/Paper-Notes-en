---
title: >-
  [Paper Note] GPSFormer: A Global Perception and Local Structure Fitting-Based Transformer for Point Cloud Understanding
description: >-
  [ECCV 2024][3D Vision][Point Cloud Understanding] This paper proposes GPSFormer, which learns short-range and long-range dependencies through a Global Perception Module (GPM) and accurately models local geometric information using a Taylor-series-inspired Local Structure Fitting Convolution (LSFConv). With only 2.36M parameters, it achieves 95.4% accuracy on ScanObjectNN, outperforming all supervised learning and pre-training methods.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Point Cloud Understanding"
  - "Global Perception"
  - "Deformable Graph Convolution"
  - "Local Structure Fitting"
  - "Taylor Series"
date: 2026-05-08
content_hash: a59138d2fea6fd9f
---

# GPSFormer: A Global Perception and Local Structure Fitting-Based Transformer for Point Cloud Understanding

**Conference**: ECCV 2024  
**arXiv**: [2407.13519](https://arxiv.org/abs/2407.13519)  
**Code**: [https://github.com/changshuowang/GPSFormer](https://github.com/changshuowang/GPSFormer)  
**Area**: 3D Vision  
**Keywords**: Point Cloud Understanding, Global Perception, Deformable Graph Convolution, Local Structure Fitting, Taylor Series

## TL;DR

This paper proposes GPSFormer, which learns short-range and long-range dependencies through a Global Perception Module (GPM) and accurately models local geometric information using a Taylor-series-inspired Local Structure Fitting Convolution (LSFConv). With only 2.36M parameters, it achieves 95.4% accuracy on ScanObjectNN, outperforming all supervised learning and pre-training methods.

## Background & Motivation

1. **Background**: Point cloud understanding has been widely applied in fields such as autonomous driving, robotics, and public safety. However, due to the disorder and irregularity of point clouds, effectively extracting shape information remains a challenge. Existing methods include indirect methods based on multi-views/voxels, direct methods based on local feature aggregation, and pre-training-based methods.
2. **Limitations of Prior Work**: (1) Methods like PointNet overlook local structural information; (2) Local aggregation methods like PointNet++ neglect long-range dependencies between points; (3) Existing Transformer methods rarely consider short-range dependencies, long-range dependencies, and local structure modeling simultaneously; (4) Although pre-training methods improve performance, they require large amounts of external data and long training times.
3. **Key Challenge**: How to simultaneously capture global contextual information and fine-grained local geometric structures without relying on external data.
4. **Goal**: Design a purely supervised Transformer that simultaneously learns global perception and local structure fitting to precisely capture point cloud shape information.
5. **Key Insight**: Decompose global perception into short-range dependencies (ADGConv) and long-range dependencies (MHA), and transform local structure modeling into a polynomial fitting problem based on Taylor series.
6. **Core Idea**: Use adaptive deformable graph convolution and multi-head attention to capture multi-scale global dependencies, and employ Taylor series to fit the low-order baseline and high-order details of local geometric structures.

## Method

### Overall Architecture

GPSFormer is constructed by stacking two core components: the **Global Perception Module (GPM)** and the **Local Structure Fitting Convolution (LSFConv)**. The classification task uses a cascade of 3-level GPS blocks (feature dimensions 64→128→256), and the segmentation task uses a 5-level GPS block encoder and a U-Net decoder. Within each GPS block, global perception is first performed using GPM, followed by FPS sampling and LSFConv for local structure fitting.

### Key Designs

1. **Adaptive Deformable Graph Convolution (ADGConv)**: Learns short-range dependencies of similar features in the feature space. Its **Mechanism** is to learn a feature offset $\Delta(f_i) = \phi(f_i)$ for each sampled point, transform the feature into $\hat{f}_i = f_i + \Delta(f_i)$, and then construct a dynamic graph centered on the transformed feature for feature aggregation. This "roaming" mechanism enables point features to navigate flexibly across the entire feature space to construct suitable local neighborhoods, avoiding the influence of the choice of K on the receptive field. The **Design Motivation** is that dynamic graphs constructed by traditional KNN are highly sensitive to K: a small K limits the receptive field locally, while a large K introduces semantically irrelevant points.

2. **Residual Cross-Attention (RCA)**: Fuses feature information before and after transformation. The formula is $f_i^r = f_i^a + \text{Attn}(\hat{f}_i, f_i^a, f_i^a)$, where $\hat{f}_i$ acts as the Query, and the ADGConv output $f_i^a$ serves as the Key and Value. Contextual structural understanding is enhanced through a residual connection.

3. **Multi-Head Attention (MHA)**: Captures long-range dependencies among all positions in the feature space. The output of RCA, $f_i^r$, is processed through a standard self-attention mechanism $f_i^g = \text{Softmax}(\frac{Q_i K^T}{\sqrt{h}})V$ to further enhance global representation capability.

4. **Local Structure Fitting Convolution (LSFConv)**: Inspired by the Taylor series $f(x) = f(a) + \sum_{n=1}^{\infty} a_n(x-a)^n$, the local structure representation is decomposed into low-order and high-order components:

    $f(\{f_j\}_{j=1}^{K}) \approx f_i^L + f_i^H = \mathcal{A}(\{\phi(f_j)\}_{j=1}^{K}) + \mathcal{A}(\{\mathcal{T}(f_i, f_j)\}_{j=1}^{K})$

   where $f_i^L$ is the **Low-Order Convolution (LOConv)**, which fits the flat regions and overall trend of the local structure, and $f_i^H$ is the **High-Order Convolution (HOConv)**, which fits the boundaries and detailed parts. HOConv utilizes a novel affine basis function:

    $\mathcal{T}(f_i, f_j) = \left(\frac{w_j \cdot (f_j - f_i)}{|w_j \cdot (f_j - f_i)|}\right)^s \cdot |w_j \cdot (f_j - f_i)|^p$

   When $s=1, p=1$, it degenerates to ABF. When $s=0, p=2$, it degenerates to RBF. A stronger expressiveness is achieved through the learnable parameter $p$.

5. **Explicit introduction of geometric structure**: The weight $w_j = \xi(h(p_i, p_j))$, where $h(p_i, p_j) = [p_i, p_j, p_j - p_i, \|p_i, p_j\|]$. It learns weights using coordinate interaction information between sampled points and neighbor points, enhancing local shape perception.

### Loss & Training

- The classification task uses cross-entropy loss and supports a voting mechanism (majority voting over multiple random samplings).
- LSFConv adopts a multi-scale strategy at each stage: ball query constructs multi-scale radii of $\{0.1, 0.2, 0.4\}$, corresponding to neighbor counts of $\{8, 16, 32\}$.
- Multi-scale parameters remain consistent across all stages to avoid hand-tuning of parameters.

## Key Experimental Results

### Main Results

| Dataset | Metric | GPSFormer | Prev. SOTA | Gain |
|--------|------|-----------|----------|------|
| ScanObjectNN (PB_T50_RS) | OA | **95.4%** | DeLA 90.4% (Supervised) / PointGPT 93.6% (Pre-trained) | +5.0% / +1.8% |
| ScanObjectNN (PB_T50_RS) | mAcc | **93.8%** | DeLA 89.3% (Supervised) | +4.5% |
| ModelNet40 | OA | 94.2% | PointGPT 94.9% (Pre-trained) | Near saturation |
| ShapeNetPart | class mIoU | **85.4%** | SPoTr 85.4% | On par with best |
| ShapeNetPart | inst mIoU | 86.8% | SPoTr 87.2% | -0.4% |

### Ablation Study

| Configuration | OA (%) | Description |
|------|--------|------|
| ADGConv only | 93.2 | Effectively extracts local features |
| RCA only | 88.7 | Contributes global relationships |
| MHA only | 89.6 | Captures dependency relationships |
| ADGConv + RCA | 94.4 | Significant complementarity |
| ADGConv + MHA | 95.0 | Verifies effectiveness |
| ADGConv + RCA + MHA (Full) | **95.4** | Optimal synergy among all three |

| HOConv Parameter Settings | OA (%) | Description |
|---------------|--------|------|
| ABF (s=1, p=1 fixed) | 92.8 | Degenerates to affine basis function |
| RBF (s=0, p=2 fixed) | 93.2 | Degenerates to radial basis function |
| s=0, learnable p | 94.6 | Adaptive parameter is effective |
| s=1, learnable p | **95.4** | Optimal configuration |

### Key Findings

- GPSFormer-elite (0.68M parameters) still achieves 93.3% OA, indicating high efficiency in the model design.
- The neighborhood size K=20 for ADGConv is optimal; being too large or too small degrades performance.
- The model has only 2.36M parameters and 0.7G FLOPs, which is significantly lower than MVTN (11.2M) and PointMLP (12.6M).
- Under the few-shot setting (5-way 10-shot), GPSFormer achieves 89.3% on ScanObjectNN, far surpassing PointNeXt's 72.4%.

## Highlights & Insights

- **Pure Supervised Learning Outperforms Pre-trained Methods**: Without relying on any external data, the model surpasses pre-trained methods like PointGPT solely through ingenious architectural design. This demonstrates that the representation capability of the model structure itself is crucial.
- **Clever Application of Taylor Series**: Formulating the local structure fitting problem as a polynomial fitting task, where low-order terms fit flat regions and high-order terms fit edge details. This approach is elegant and highly effective.
- **Extreme Parameter Efficiency**: Achieving 95.4% accuracy with only 2.36M parameters, demonstrating outstanding parameter efficiency.
- **Innovation in Deformable Graph Convolution**: Implementing feature-space roaming by learning feature offsets, which offers greater flexibility than traditional KNN.

## Limitations & Future Work

- Performance on ModelNet40 is near saturation (94.2%), and this dataset is no longer sufficient to distinguish the strengths and weaknesses of different methods.
- Performance on segmentation tasks is slightly lower than SPoTr, suggesting a need to optimize the network architecture specifically for segmentation tasks.
- The multi-scale strategy parameters are fixed (radii of 0.1/0.2/0.4); adaptive multi-scale mechanisms could be explored.
- The combination of pre-training and GPSFormer has not been explored, which could potentially further raise the performance upper bound.

## Related Work & Insights

- Compared to the dynamic graph convolution in DGCNN, ADGConv achieves more flexible neighborhood construction through feature offsets.
- The Taylor series fitting concept can be generalized to other tasks requiring local structure modeling.
- The GPM framework, which models both short-range and long-range dependencies, can inspire model designs for other 3D understanding tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ Both the Taylor-series-based local structure fitting and the adaptive deformable graph convolution are highly novel designs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering three tasks, detailed ablation studies, few-shot evaluations, model complexity analysis, and visualizations.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with complete mathematical derivations and rich figures and tables.
- Value: ⭐⭐⭐⭐ The fact that a purely supervised method outperforms pre-trained methods is highly significant, showing excellent parameter efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] DG-PIC: Domain Generalized Point-In-Context Learning for Point Cloud Understanding](dg-pic_domain_generalized_point-in-context_learning_for_point_cloud_understandin.md)
- [\[CVPR 2026\] 4D Local Modeling Toward Dynamic Global Perception for Ambiguity-free Rotation-Invariant Point Cloud Analysis](../../CVPR2026/3d_vision/4d_local_modeling_toward_dynamic_global_perception_for_ambiguity-free_rotation-i.md)
- [\[ECCV 2024\] TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos](tram_global_trajectory_and_motion_of_3d_humans_from_in-the-wild_videos.md)
- [\[CVPR 2026\] Mamba Learns in Context: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding](../../CVPR2026/3d_vision/mamba_learns_in_context_structure-aware_domain_generalization_for_multi-task_poi.md)
- [\[CVPR 2025\] PMA: Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter](../../CVPR2025/3d_vision/pma_towards_parameter-efficient_point_cloud_understanding_via_point_mamba_adapte.md)

</div>

<!-- RELATED:END -->
