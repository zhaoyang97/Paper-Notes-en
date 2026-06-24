---
title: >-
  [Paper Note] RoadPainter: Points Are Ideal Navigators for Topology Transformer
description: >-
  [ECCV 2024][LLM (Other)][Topology Inference] RoadPainter is proposed, which adopts a two-stage strategy of first regressing lane centerline points and then refining them using instance masks. Combined with a hybrid attention mechanism and a real-virtual lane separation strategy, it achieves SOTA topology inference performance on the OpenLane-V2 dataset.
tags:
  - "ECCV 2024"
  - "LLM (Other)"
  - "Topology Inference"
  - "Lane Centerline Detection"
  - "BEV Perception"
  - "Instance Segmentation"
  - "SD Map"
date: 2026-05-08
content_hash: acbf65c1b5d31324
---

# RoadPainter: Points Are Ideal Navigators for Topology Transformer

**Conference**: ECCV 2024  
**arXiv**: [2407.15349](https://arxiv.org/abs/2407.15349)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Topology Inference, Lane Centerline Detection, BEV Perception, Instance Segmentation, SD Map

## TL;DR

RoadPainter is proposed, which adopts a two-stage strategy of first regressing lane centerline points and then refining them using instance masks. Combined with a hybrid attention mechanism and a real-virtual lane separation strategy, it achieves SOTA topology inference performance on the OpenLane-V2 dataset.

## Background & Motivation

**Background**: Topology inference in autonomous driving aims to extract lane centerlines and their topological connection relationships from multi-view images to provide routing information for downstream trajectory prediction and planning. In recent years, this task has evolved from 2D lane detection to online vectorized map construction in the BEV space.

**Limitations of Prior Work**: Methods like TopoNet, which directly regress centerline points, suffer from insufficient geometric precision in high-curvature areas (such as curves and ramps). Regression methods tend to learn straight centerline shapes, making it difficult to capture complex geometric details. Meanwhile, virtual lanes in intersections differ greatly in characteristics from real lanes, and handling them uniformly limits performance.

**Key Challenge**: Regression provides stable initial localization but poor geometric accuracy, while segmentation provides precise geometric details but unstable localization—how to combine the advantages of both?

**Goal**: Improve the centerline detection accuracy in high-curvature regions while enhancing topology inference performance.

**Key Insight**: First use a transformer decoder to regress coarse centerline points, then use these points to guide the generation of instance masks. New points are then sampled from these masks to be fused and refined with the regressed points.

**Core Idea**: "Points Are Ideal Navigators" — use regressed points to guide mask generation, and utilize mask feedback to optimize point positions, achieving complementary advantages of regression and segmentation.

## Method

### Overall Architecture

Multi-view images → Image backbone (ResNet-50) extracts multi-scale features → BEVFormer constructs BEV features → (Optional) SD Map Interaction enhances BEV features → Hybrid Attention Transformer Decoder regresses centerline points + topological relationships → Points-Guided Mask Generation generates instance masks → Points-Mask Fusion refines centerlines → Output precise centerlines and topology matrices

### Key Designs

1. **Hybrid Attention Transformer Decoder + Real-Virtual Separation Strategy (RVS)**:

    - Function: Detect lane centerline instances from BEV features and establish topological connections.
    - Mechanism: The decoder contains three types of attention—masked cross-attention aggregating mask region features, deformable cross-attention aggregating learnable sampling point features, and self-attention facilitating query interaction. The key innovation is the RVS self-attention: real lanes and virtual lanes use independent queries. In the self-attention, virtual queries can see real queries, but real queries do not look at virtual queries:
    $\text{RVSelfAttn} = \text{softmax}\left(\frac{\begin{bmatrix} Q^r Q^{rT} & -\infty \\ Q^v Q^{rT} & Q^v Q^{vT} \end{bmatrix}}{\sqrt{C}}\right) \begin{bmatrix} Q^r \\ Q^v \end{bmatrix}$
    - Design Motivation: The positions of virtual lanes (intersection connection lines) depend on real lanes, but real lane positions do not depend on virtual lanes. This prior knowledge is encoded through an asymmetric attention mask.

2. **Points-Guided Mask Generation (PGM)**:

    - Function: Guided by the regressed centerline points to generate instance segmentation masks for each centerline.
    - Mechanism: The regressed centerline points $\mathbf{l}_i \in \mathbb{R}^{K \times 3}$ generate mask queries $\mathbf{Q}_i'$ via a positional encoding MLP and a query encoding MLP, which then undergo dot product with BEV features: $\mathbf{M}_i = \mathbf{B} \cdot \mathbf{Q}_i'$. Compared to Mask2Former, which uses learnable queries with no positional priors, PGM utilizes the spatial position information of the regressed points to guide mask generation.
    - Design Motivation: Segmentation masks can capture fine geometric shapes of centerlines (especially in high-curvature regions), but pure segmentation methods have unstable localization. Using regressed points as positional priors for masks achieves both stability and accuracy.

3. **Points-Mask Fusion (PMF)**:

    - Function: Sample points from masks and fuse them with regressed points to obtain refined centerlines.
    - Mechanism: This is done in two steps—(1) Mask Points Sampling: regress a point column-by-column on the mask $\mathbf{C}_{i,j} = [0,1,...,H-1]^T \cdot \text{softmax}(\mathbf{M}_i(:,j))$, while predicting the existence probability $\mathbf{P}_i$ and direction probability $D_i$ for each point; (2) Points Fusion: filter abnormal points (distance to neighbors >1.5m), resample valid mask points into $K$ points, and average them with the regressed points. Note that virtual lanes do not undergo mask refinement (due to the lack of visual information).
    - Design Motivation: Masks provide finer geometric information to compensate for the deficiency of regression. Simple averaging can effectively fuse the two representations.

4. **SD Map Interaction (Optional Module)**:

    - Function: Utilize prior information from Standard Definition maps (SD Maps) to enhance BEV features.
    - Mechanism: Convert vectorized instances of the SD map into BEV semantic features $\mathbf{E}_S$, and interact them with online BEV features via a transformer decoder: $\hat{\mathbf{B}} = \text{TrDec}(\mathbf{B}, \mathbf{E}_S + \mathbf{E}_P)$.
    - Design Motivation: Address issues of occlusion and limited perception range; SD Maps provide road shape priors beyond the line of sight.

5. **Topology Association Head**:

    - Function: Predict the topology connection matrix $\mathbf{A}_{ll} \in [0,1]^{N_L \times N_L}$ between centerlines.
    - Mechanism: Fuse query and position information $\mathbf{E}_i = \psi_1(\mathbf{Q}_i) + \psi_2(\mathbf{l}_i)$, concatenate them, and predict the topological relationships via a binary classifier.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{top}(\mathbf{A}_{ll}) + \mathcal{L}_{cls}(\mathbf{S}) + \mathcal{L}_{det}(\mathbf{L}_V, \mathbf{L}_R) + \mathcal{L}_{mask}(\mathbf{M}) + \mathcal{L}_{mp}(\mathbf{C}, \mathbf{P}, D)$$

- $\mathcal{L}_{top}$, $\mathcal{L}_{cls}$: Focal Loss supervises topological relationships and centerline confidence.
- $\mathcal{L}_{det}$: L1 Loss supervises centerline geometry.
- $\mathcal{L}_{mask}$: BCE + Dice Loss supervises instance masks.
- $\mathcal{L}_{mp}$: L1 + BCE + Focal Loss supervises mask sampling points, existence probability, and direction.
- Bipartite matching and loss calculations are conducted separately for real/virtual centerlines.
- AdamW optimizer, initial learning rate $2 \times 10^{-4}$, training for 24 epochs, cosine annealing.
- Gradient clipping (max norm = 35), backbone learning rate is one-tenth of other modules.

## Key Experimental Results

### Main Results

| Dataset/Method | DET_l ↑ | DET_t ↑ | TOP_ll ↑ | TOP_lt ↑ | OLS ↑ |
|------------|---------|---------|----------|----------|-------|
| **subset_A** | | | | | |
| TopoNet | 28.5 | 48.1 | 4.1 | 20.8 | 35.6 |
| TopoMLP | 28.3 | 50.0 | 7.2 | 22.8 | 38.2 |
| RoadPainter | **30.7** | 47.7 | **7.9** | **24.3** | **38.9** |
| SMERF* (SD map) | 33.4 | 48.6 | 7.5 | 23.4 | 39.4 |
| RoadPainter* (SD map) | **36.9** | 47.1 | **12.7** | **25.8** | **42.6** |
| **subset_B** | | | | | |
| TopoNet | 24.3 | 55.0 | 2.5 | 14.2 | 33.2 |
| TopoMLP | 26.6 | 58.3 | 7.6 | 17.8 | 38.7 |
| RoadPainter | **28.7** | 54.8 | **8.5** | 17.2 | 38.5 |

### Ablation Study

| PGM | PMF | SD | DET_l | TOP_ll | OLS | AP_l | Description |
|-----|-----|-----|-------|--------|-----|------|------|
| ✗ | ✗ | ✗ | 26.9 | 7.7 | 37.2 | - | Baseline |
| ✓ | ✗ | ✗ | 28.1 | 7.9 | 37.6 | 13.5 | +PGM: mask supervision improves DET_l by +1.2 |
| ✓ | ✓ | ✗ | 30.7 | 7.9 | 38.9 | 14.1 | +PMF: mask refinement improves DET_l by +2.6 |
| ✓ | ✓ | ✓ | 36.9 | 12.7 | 42.6 | 15.4 | +SD Map: significantly improves DET_l by +6.2 |

| Attention Ablation | DET_l | TOP_ll | Description |
|-----------|-------|--------|------|
| RoadPainter (full) | 30.7 | 7.9 | - |
| w/o hybrid attention | 29.6 | 7.2 | Hybrid attention contributes to DET_l (+1.1) |
| w/o real-virtual self-attn | 29.6 | 7.5 | RVS has a pronounced impact on TOP_ll (-0.4) |

### Key Findings

- **Points-Mask Fusion is the core contribution**: The PMF module (sampling points from masks and fusing with regressed points) contributes a +2.6 improvement on DET_l, far exceeding PGM's +1.2, showing that the "regression + segmentation complementary" strategy is indeed effective.
- **SD Map provides the most significant boost**: DET_l +6.2, TOP_ll +4.8, showing that beyond-line-of-sight information is highly valuable for topology inference.
- **RVS strategy mainly improves topology inference**: It contributes +0.4 to TOP_ll, conforming to the prior that virtual lanes depend on real lanes.
- **Segmentation masks show a clear advantage in high-curvature scenes**: Visualization results indicate significant improvements in centerline accuracy of curves and intersections.
- **The baseline's TOP_ll (7.7) has already exceeded TopoNet (4.1)**, indicating that the design of the topology association head itself is highly effective.

## Highlights & Insights

- **The complementary regression and segmentation approach** is highly generalizable and can be transferred to other tasks requiring precise geometric prediction (such as 3D object detection, keypoint detection, etc.).
- **Real-virtual lane separation** precisely models domain knowledge, implemented via an asymmetric attention mask, which is simple and elegant.
- The **column-wise mask point regression** design cleverly solves the transition from heatmaps to ordered point sets, avoiding complex post-processing.
- The prediction of **existence probability + direction probability** resolves the issue of variable mask point counts and uncertain orientations, enabling end-to-end training.
- The **SD Map interaction module** features a plug-and-play design, keeping the approach competitive even in scenarios without SD Maps.

## Limitations & Future Work

- DET_t (traffic element detection) and TOP_lt (lane-traffic topology) metrics are inferior to TopoMLP, as a 3-layer BEVFormer was used instead of a 6-layer PETR, which is more suitable for detecting traffic elements.
- Mask refinement is only valid for real lanes; virtual lanes (which lack visual information) do not benefit.
- Completely vertical centerlines require an additional row-wise sampling mechanism, increasing complexity.
- The FPS is only 6.5 (RTX3090, FP32), leaving room for real-time performance optimization.
- Points-Mask Fusion uses a simple average; a more adaptive weighting strategy might further improve performance.

## Related Work & Insights

- **vs TopoNet**: RoadPainter improves DET_l by +2.2 (+7.7%), where the key lies in mask refinement compensating for pure regression.
- **vs TopoMLP**: Slightly better on OLS but weaker on DET_t and TOP_lt than TopoMLP, due to the different BEV construction methods (BEVFormer vs PETR).
- **vs Mask2Former**: Borrows mask attention as a BEV feature aggregation approach, but innovates with points-guided mask queries (providing a better positional prior than learnable queries).
- **vs MapTR**: MapTR focuses on vectorized map elements but does not handle topological relationships. RoadPainter adds topology inference on top of this.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of complementary regression and segmentation is not entirely new, but the closed-loop design of points-guided mask + mask-based refinement is innovative in topology inference.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual subset comparison, multiple groups of ablation studies, attention ablations, and visualization analyses are provided, though comparisons with a wider range of methods are lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, intuitive diagrams, and standardized formulas.
- Value: ⭐⭐⭐⭐ Substantial practical improvements and transferable ideas; the SD Map integration serves as a valuable reference for the industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention for Transformer](../../CVPR2025/llm_nlp/spiking_transformer_introducing_accurate_addition-only_spiking_self-attention_fo.md)
- [\[CVPR 2025\] Spiking Transformer with Spatial-Temporal Attention](../../CVPR2025/llm_nlp/spiking_transformer_with_spatial-temporal_attention.md)
- [\[NeurIPS 2025\] Spectral Conditioning of Attention Improves Transformer Performance](../../NeurIPS2025/llm_nlp/spectral_conditioning_of_attention_improves_transformer_performance.md)
- [\[NeurIPS 2025\] Characterizing the Expressivity of Fixed-Precision Transformer Language Models](../../NeurIPS2025/llm_nlp/characterizing_the_expressivity_of_fixed-precision_transformer_language_models.md)
- [\[ACL 2025\] A Systematic Study of Compositional Syntactic Transformer Language Models](../../ACL2025/llm_nlp/a_systematic_study_of_compositional_syntactic_transformer_language_models.md)

</div>

<!-- RELATED:END -->
