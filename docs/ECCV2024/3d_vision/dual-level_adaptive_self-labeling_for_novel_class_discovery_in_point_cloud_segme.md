---
title: >-
  [Paper Note] Dual-level Adaptive Self-Labeling for Novel Class Discovery in Point Cloud Segmentation
description: >-
  [ECCV 2024][3D Vision][Novel Class Discovery] This paper proposes a dual-level adaptive self-labeling method that addresses the class imbalance problem through semi-relaxed optimal transport and incorporates region-level representations to enhance pointwise classifier learning, achieving efficient novel class discovery in point cloud segmentation.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Novel Class Discovery"
  - "Point Cloud Segmentation"
  - "Self-Labeling"
  - "Optimal Transport"
  - "Imbalanced Learning"
date: 2026-05-08
content_hash: a1acb50b782a4bfb
---

# Dual-level Adaptive Self-Labeling for Novel Class Discovery in Point Cloud Segmentation

**Conference**: ECCV 2024  
**arXiv**: [2407.12489](https://arxiv.org/abs/2407.12489)  
**Code**: [https://github.com/RikkiXu/NCD_PC](https://github.com/RikkiXu/NCD_PC)  
**Area**: 3D Vision  
**Keywords**: Novel Class Discovery, Point Cloud Segmentation, Self-Labeling, Optimal Transport, Imbalanced Learning

## TL;DR

This paper proposes a dual-level adaptive self-labeling method that addresses the class imbalance problem through semi-relaxed optimal transport and incorporates region-level representations to enhance pointwise classifier learning, achieving efficient novel class discovery in point cloud segmentation.

## Background & Motivation

**Background**: Novel Class Discovery (NCD) aims to identify and segment novel classes by leveraging the semantic knowledge of known classes. In point cloud segmentation, existing methods (such as NOPS) employ online pointwise clustering and avoid degenerate solutions using simplified equal-class-size constraints.

**Limitations of Prior Work**:
   - **Class Imbalance**: Point cloud data inherently has highly imbalanced distributions of novel classes. The equal-class-size constraints do not match the actual distribution, and enforcing uniform constraints generates pseudo-labels that fail to reflect the true distribution.
   - **Lack of Spatial Context**: Pointwise clustering disregards the rich spatial context of objects, which leads to suboptimal representation capacity for semantic segmentation.
   - **NOPS's bi-level optimization strategy** is computationally intensive and introduces extra hyperparameters.

**Key Challenge**: The uniform distribution constraint is too strong in the late stages of training, forcing pseudo-labels towards uniformity rather than reflecting the true imbalanced distribution. Additionally, point-level learning lacks spatial context, meaning points within a region that should possess consistent semantics are processed independently.

**Goal**: To generate high-quality pseudo-labels adapted to the imbalanced class distributions in point cloud NCD tasks, while utilizing regional consistency to improve segmentation quality.

**Key Insight**: Relaxing the equality constraint of optimal transport to a penalty term with KL divergence to adaptively adjust constraint strength, and introducing region-level representations using DBSCAN clustering.

**Core Idea**: Adaptively generate imbalanced pseudo-labels via semi-relaxed optimal transport, combining pointwise and region-level dual-level representation learning to mutually enhance novel class segmentation performance.

## Method

### Overall Architecture

The model consists of an encoder $f_\theta$ and a classifier $h = [h^s, h^u]$ (corresponding to known and novel classes, respectively). During training, known classes are supervised using cross-entropy loss, while novel classes undergo dual-level self-labeling: (1) pointwise self-labeling uses semi-relaxed OT to generate pseudo-labels; (2) region-level self-labeling first clusters the point cloud into regions using DBSCAN before generating region-level pseudo-labels. The two levels share prototype/classifier parameters. Adaptive regularization dynamically adjusts the KL constraint strength $\gamma$.

### Key Designs

1. **Imbalanced Self-Labeling (ISL)**:

    - **Function**: Generates pseudo-labels for novel classes that reflect the true imbalanced distribution.
    - **Mechanism**: Relaxes the two equality constraints in standard OT into one equality constraint plus a KL penalty:
    $$\min_{\mathbf{Q}} \langle \mathbf{Q}, \mathbf{C} \rangle_F + \gamma \text{KL}(\mathbf{Q}^\top \mathbf{1}_M, \boldsymbol{\mu})$$
    $$\text{s.t. } \mathbf{Q} \in \{\mathbf{Q} \in \mathbb{R}^{M \times N} | \mathbf{Q}\mathbf{1}_N = \boldsymbol{\nu}\}$$
    - Here, $\mathbf{C} = -\log \mathbf{P}$ is the cost matrix (based on model predictions), $\boldsymbol{\mu}$ is the marginal class prior (uniform distribution), and $\boldsymbol{\nu}$ is the marginal sample distribution.
    - The row constraint remains an equality (ensuring each point is assigned to exactly one class), while the column constraint is relaxed to a KL penalty (allowing class sizes to deviate from a uniform distribution).
    - By introducing an entropy constraint $-\epsilon \mathcal{H}(\mathbf{Q})$, it can be solved efficiently with a scaling algorithm:
    $$\epsilon \langle \mathbf{Q}, \log \frac{\mathbf{Q}}{e^{-\mathbf{C}/\epsilon}} \rangle_F + \gamma \text{KL}(\mathbf{Q}^\top \mathbf{1}_M, \boldsymbol{\mu})$$
    - **Design Motivation**: The standard OT forcing uniform distribution is unsuitable for imbalanced data. Compared to the bi-level optimization of NOPS, directly using a scaling algorithm is faster and more concise.

2. **Adaptive Regularization (AR)**:

    - **Function**: Dynamically adjusts the strength of the KL constraint $\gamma$ during training.
    - **Mechanism**: Monitors the KL distance between the pseudo-label distribution $\frac{1}{M}\mathbf{Q}^\top \mathbf{1}_M$ and the uniform distribution $\boldsymbol{\nu}$. When the KL distance remains below a threshold $\rho$ for $T$ consecutive iterations, $\gamma$ is decreased:
    $$\gamma_{t+1} = \lambda \cdot \gamma_t, \quad \text{if } \text{KL}(\frac{1}{M}\mathbf{Q}^\top \mathbf{1}_M, \boldsymbol{\nu}) \leq \rho \text{ for } T \text{ iters}$$
    - Where $\lambda < 1$ is the decay coefficient.
    - **Design Motivation**: A fixed $\gamma$ is either over-constrained (overly uniform distribution) or too relaxed (leading to degenerate solutions). Stronger constraints are needed in the early stages of training to prevent degeneration, while relaxation is necessary in later stages to let the model learn the true distribution.

3. **Region-level Learning**:

    - **Function**: Exploits the spatial context of point clouds to perform self-labeling at the region level.
    - **Mechanism**:
        - Cluster the point cloud of each scene into several continuous regions $\{r_k\}_{k=0}^K$ using the DBSCAN algorithm.
        - Average-pool pointwise features within the same region to obtain region-level representations: $\mathbf{z}_r = \text{AvgPool}(\mathbf{z}_p | r_k \text{ is same})$
        - Predict regional representations using the same classifier as the pointwise level (shared prototypes).
        - Apply semi-relaxed OT to region-level predictions to generate region-level pseudo-labels $\mathbf{Q}_r^u$ likewise.
    - DBSCAN parameters: $\epsilon_{\text{dbscan}} = 0.5$ (ensuring 95% of points participate in region learning), min-samples = 2
    - **Design Motivation**: Points within the same region have a high probability of belonging to the same category. Region-level representations reduce point-level noise and provide more robust semantic signals; shared prototypes ensure consistent learning directions for both levels.

### Loss & Training

Overall Loss:
$$\mathcal{L} = \mathcal{L}_s + \alpha \mathcal{L}_u^p + \beta \mathcal{L}_u^r$$

- $\mathcal{L}_s = -\frac{1}{N}\sum_{i=1}^N y_i^s \log p_i^s$: Cross-entropy loss for known classes.
- $\mathcal{L}_u^p = \frac{1}{M}\langle \mathbf{Q}_p^u, -\log \mathbf{P}_r^u \rangle_F$: Point-level self-labeling loss (pseudo-labels are from pointwise OT, predictions are from region-level — cross-supervision).
- $\mathcal{L}_u^r = \frac{1}{K}\langle \mathbf{Q}_r^u, -\log \mathbf{P}_r^u \rangle_F$: Region-level self-labeling loss.
- Use data augmentation to create two views for transformation invariance learning.
- Optimizer: AdamW, initial learning rate 1e-3, cosine decay to 1e-5.

## Key Experimental Results

### Main Results (SemanticPOSS, average mIoU over 4 splits)

| Method | Novel mIoU | Known mIoU | All mIoU |
|------|-----------|-----------|---------|
| NOPS | 24.0 | 38.3 | 35.4 |
| NOPS* (optimized training setup) | 24.5 | 47.5 | 41.2 |
| **Ours** | **29.2** | **47.7** | **43.7** |
| Full (fully supervised upper bound) | - | - | 48.5 |

### Main Results (SemanticKITTI, average mIoU over 4 splits)

| Method | Novel mIoU | Known mIoU | All mIoU |
|------|-----------|-----------|---------|
| NOPS | 22.9 | 42.4 | 37.8 |
| NOPS* | 21.4 | 47.8 | 40.6 |
| **Ours** | **26.8** | **47.2** | **42.4** |
| Full (fully supervised upper bound) | - | - | 47.9 |

### Ablation Study (SemanticPOSS Split 0)

| ISL | AR | Region | Building | Car | Ground | Plants | mIoU |
|-----|-----|--------|----------|-----|--------|--------|------|
| ✗ | ✗ | ✗ | 16.1 | 4.2 | 54.9 | 37.9 | 28.3 |
| ✓ | ✗ | ✗ | 57.4 | 32.1 | 18.9 | 37.2 | 34.1 |
| ✓ | ✓ | ✗ | 70.8 | 34.7 | 16.8 | 57.9 | 40.7 |
| ✓ | ✓ | ✓ | **74.6** | **41.4** | **22.5** | **66.4** | **45.7** |

### Head/Medium/Tail Class Analysis (SemanticPOSS)

| Method | Head | Medium | Tail | Description |
|------|------|--------|------|------|
| NOPS | 37.5 | 21.9 | 4.4 | Extremely poor tail performance |
| **Ours** | **45.0** | **30.8** | **11.2** | Comprehensive improvements |
| Gain | +7.5% | +8.9% | +6.8% | Most significant improvement on tail classes |

### Key Findings
- Adaptive regularization prevents the pseudo-label distribution from being forced into uniformity, allowing it to reflect the true imbalanced nature of the data.
- Region-level learning yields a ~10% gain when prototypes are shared; not sharing prototypes actually degrades performance, indicating the need for unified learning directions between the two levels.
- The epsilon parameter of DBSCAN is not highly sensitive: good results are obtained within the range of 0.3-0.7, with 0.7 being optimal (mIoU 53.5).
- This method exhibits clear computational advantages in speed compared to the bi-level optimization strategy of NOPS, with the scaling algorithm being more suitable for large-scale problems.

## Highlights & Insights
- The formulation of **semi-relaxed OT** is more elegant than that of NOPS: replacing the equality constraint with a KL penalty avoids degradation while preserving flexibility, making it mathematically more concise.
- **Adaptive $\gamma$ adjustment** achieves self-regulation of training dynamics via a simple decay strategy—stronger constraints in the early stage to prevent degradation, and weaker constraints in the late stage to learn the true distribution.
- The design of **dual-level representation + shared prototypes** allows pointwise and region-level features to complement each other: region-level features provide a smoothed signal directing global alignment, while pointwise features provide fine-grained segmentation details.
- Creating two views via data augmentation for cross-supervision (pointwise pseudo-labels + region-level predictions) enhances the information complementarity between labels and predictions.

## Limitations & Future Work
- Although tail class segmentation accuracy is improved, it remains low (11.2%), indicating that the long-tailed imbalance problem is still extremely challenging.
- DBSCAN exhibits some dependency on the epsilon parameter: too large a value results under-segmentation (mixing different categories), while too small a value provides insufficient context.
- The number of novel classes needs to be specified in advance ($|C^u|$ is known); in real-world scenarios, the number of novel classes might be unknown.
- Region-level representations use simple average pooling; more complex aggregation methods (such as attention pooling) could be explored.

## Related Work & Insights
- **vs NOPS**: NOPS utilizes standard OT with equality constraints + multi-head + overclustering, which incurs high computational overhead, and its uniform constraint is unsuitable for imbalanced data. Ours employs semi-relaxed OT with adaptive regularization, which is simpler and more efficient.
- **vs Zhang et al.**: The bi-level optimization by Zhang et al. requires alternating updates of auxiliary variables $\mathbf{w}$ and pseudo-labels $\mathbf{Q}$, introducing extra hyperparameters and slowing down computation. In contrast, ours directly solves the semi-relaxed OT using a scaling algorithm, leading to faster execution.
- **vs GrowSP**: GrowSP is used for region segmentation in unsupervised point cloud pre-training. Ours borrows the spatial prior idea of regionalization but designs a dual-level self-labelling learning specifically for the NCD task.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of semi-relaxed OT and adaptive regularization is novel, and the dual-level design is sensible.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, with two datasets, 4 splits, detailed ablation studies, parameter sensitivity analysis, and head/medium/tail analysis.
- Writing Quality: ⭐⭐⭐⭐ The mathematical derivations are clear, the experimental analysis is rigorous, and the appendix provides rich supplementary materials.
- Value: ⭐⭐⭐⭐ Successfully addresses the core imbalance issue in point cloud NCD, offering strong generalizability that can be extended to other NCD scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Geometric-Aware Hypergraph Reasoning for Novel Class Discovery in Point Cloud Segmentation](../../CVPR2026/3d_vision/geometric-aware_hypergraph_reasoning_for_novel_class_discovery_in_point_cloud_se.md)
- [\[NeurIPS 2025\] Novel Class Discovery for Point Cloud Segmentation via Joint Learning of Causal Representation and Reasoning](../../NeurIPS2025/3d_vision/novel_class_discovery_for_point_cloud_segmentation_via_joint_learning_of_causal_.md)
- [\[ECCV 2024\] P2P-Bridge: Diffusion Bridges for 3D Point Cloud Denoising](p2p-bridge_diffusion_bridges_for_3d_point_cloud_denoising.md)
- [\[ECCV 2024\] AEDNet: Adaptive Embedding and Multiview-Aware Disentanglement for Point Cloud Completion](aednet_adaptive_embedding_and_multiview-aware_disentanglement_for_point_cloud_co.md)
- [\[ECCV 2024\] RISurConv: Rotation Invariant Surface Attention-Augmented Convolutions for 3D Point Cloud Classification and Segmentation](risurconv_rotation_invariant_surface_attention-augmented_convolutions_for_3d_poi.md)

</div>

<!-- RELATED:END -->
