---
title: >-
  [Paper Note] CasP: Improving Semi-Dense Feature Matching Pipeline Leveraging Cascaded Correspondence Priors for Guidance
description: >-
  [ICCV 2025][3D Vision][Feature Matching] This paper proposes CasP, a cascaded matching pipeline that decomposes the matching stage into a one-to-many prior matching at 1/16 scale and a one-to-one fine matching at 1/8 sca…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Feature Matching"
  - "Cascaded Matching"
  - "Semi-Dense"
  - "Efficiency"
  - "Cross-Domain Generalization"
date: 2026-05-08
content_hash: 349d483bae360bfd
---

# CasP: Improving Semi-Dense Feature Matching Pipeline Leveraging Cascaded Correspondence Priors for Guidance

**Conference**: ICCV 2025
**arXiv**: [2507.17312](https://arxiv.org/abs/2507.17312)
**Code**: [GitHub](https://github.com/pq-chen/CasP)
**Area**: 3D Vision / Feature Matching
**Keywords**: Feature Matching, Cascaded Matching, Semi-Dense, Efficiency, Cross-Domain Generalization

## TL;DR

This paper proposes CasP, a cascaded matching pipeline that decomposes the matching stage into a one-to-many prior matching at 1/16 scale and a one-to-one fine matching at 1/8 scale, achieving up to 2.2× speedup while maintaining accuracy and significantly improving cross-domain generalization.

## Background & Motivation

Semi-dense feature matching methods (exemplified by LoFTR) treat every token in the feature map as a candidate match, bypassing explicit feature detection, and demonstrate strong performance in low-texture and repetitive-pattern scenarios. However, their efficiency bottleneck lies in the matching stage, which requires global search over the entire feature map:

**Latency Bottleneck**: As input resolution increases, the number of tokens in the matching stage grows rapidly, occupying an increasingly large fraction of total runtime. Although ELoFTR introduces aggregated attention, the matching stage still consumes a substantial portion of time at 1152 resolution.

**Accuracy–Efficiency Trade-off Failure**: ELoFTR attempts to accelerate by directly removing the dual-softmax (DS) operator, but this causes significant accuracy degradation, as the matching stage relies solely on descriptor similarity without global confidence.

**Insufficient Cross-Domain Generalization**: Existing methods yield limited gains on cross-domain benchmarks such as ScanNet (indoor).

The core design insight is to defer the main computational operations to coarser scales and use cascaded priors to constrain the search range of fine matching.

## Method

### Overall Architecture

The CasP pipeline consists of four stages:
1. **Feature Extraction**: Lightweight CNN (low-level, 1/2–1/8) + Context Cluster (high-level, 1/16–1/32)
2. **Feature Interaction**: Hybrid interaction module (aggregated attention + Cross-CoC)
3. **Cascaded Matching**: One-to-many matching (1/16) → RSCA → One-to-one matching (1/8)
4. **Match Refinement**: Two-stage homography refinement (pixel-level + sub-pixel-level)

### Key Designs

1. **High-Level Feature Extraction (Self-CoC)**:

    - Adopts the Context Cluster mechanism to replace convolution for extracting 1/16 and 1/32 scale features.
    - Performs indirect global point-to-point interaction via a three-stage cluster–aggregate–dispatch process.
    - Controls computational cost by adjusting the number of anchors, enabling global-receptive-field context understanding.

2. **Cascaded Matching Module**:

    - **One-to-Many Matching (1/16 scale)**: Constructs a correlation matrix $S_{1/16}$ and selects the top-$k$ ($k=8$) correspondence priors $\pi_{1/16}$ per token. During training, the DS operator generates a confidence matrix and ground-truth is injected to accelerate convergence; during inference, raw correlations are used directly for top-$k$ selection.
    - **RSCA (Region-based Selective Cross-Attention)**: Partitions the feature map into $r \times r$ blocks, with each block performing cross-attention only over prior regions to enhance discriminability among prior candidates. Each query has length $r^2$ and key/value have length $k \cdot r^2$.
    - **One-to-One Matching (1/8 scale)**: Introduces Partial Softmax—computing softmax only over the tokens attended by RSCA and zeroing out the rest. Matches must satisfy a bidirectional prior constraint: $j \in \phi_r(\pi^A)[i]$ and $i \in \phi_r(\pi^B)[j]$.

3. **Train–Inference Decoupling Strategy**:

    - Training: Both stages use the DS operator for supervision; ground-truth priors are injected into the top-$k$ set to accelerate RSCA learning.
    - Inference: The DS operator is omitted in the one-to-many stage; Partial Softmax replaces full softmax in the one-to-one stage.
    - Effect: Maximizes representation capacity during training while maximizing efficiency during inference.

### Loss & Training

$$L = \lambda_1 L_{1/16}^c + \lambda_2 L_{1/8}^c + \lambda_3 L_{1/1}^f + \lambda_4 L_{\text{sub}}^f$$

- $L_{1/s}^c$: Coarse matching negative log-likelihood loss ($s \in \{8, 16\}$)
- $L_{1/1}^f$: Pixel-level refinement loss
- $L_{\text{sub}}^f$: Sub-pixel $\ell_2$ loss
- Weights: $\lambda_1=0.5, \lambda_2=0.5, \lambda_3=0.25, \lambda_4=1.0$
- Trained on MegaDepth, 8×V100, batch size 8, 30 epochs

## Key Experimental Results

### Main Results (Relative Pose Estimation AUC + Runtime)

| Method | MD-1500 @5° | @10° | @20° | SN-1500 @5° | @10° | @20° | MD Time(ms) | SN Time(ms) |
|--------|-------------|------|------|-------------|------|------|-------------|-------------|
| LoFTR | 52.8 | 69.2 | 81.2 | 16.9 | 33.6 | 50.6 | 347.6 | 71.7 |
| ELoFTR | 56.4 | 72.2 | 83.5 | 19.2 | 37.0 | 53.6 | 238.3 | 45.2 |
| AffineFormer | 57.3 | 72.8 | 84.0 | 22.0 | 40.9 | 58.0 | ≥347.6 | ≥71.7 |
| **Ours-full** | **57.1** | **72.7** | **83.9** | **23.0** | **41.6** | **58.7** | **147.2** | **40.1** |
| **Ours-lite** | 55.6 | 71.7 | 83.3 | 21.6 | 40.1 | 57.0 | **108.1** | **33.1** |

### Ablation Study (ETH3D Cross-Domain Evaluation + Efficiency)

| Method | ETH3D[O] @5° | @10° | @20° | ETH3D[I] @5° | @10° | @20° | GMACs | Runtime(ms) |
|--------|--------------|------|------|--------------|------|------|-------|-------------|
| EL w/ DS | 56.7 | 63.2 | 69.1 | 49.1 | 55.0 | 59.4 | 909.1 | 238.3 |
| EL w/o DS | 53.4 | 60.1 | 66.3 | 44.7 | 50.8 | 55.7 | 909.1 | 185.3 |
| EL+CM-full | 60.1 | 65.6 | 70.6 | 52.3 | 57.2 | 60.7 | 708.1 | 144.6 |
| **Ours-full** | **61.8** | **66.8** | **71.5** | **56.1** | **60.7** | **64.0** | **691.0** | **147.2** |
| **Ours-lite** | 60.3 | 65.9 | 71.2 | 54.3 | 59.2 | 62.7 | 365.1 | 108.1 |

**Low-Level Feature Extraction Parameter Comparison**:

| Method | Type | Channel Config | Params (M) |
|--------|------|----------------|------------|
| LoFTR | ResNet | [128,196,256] | 5.9 |
| ELoFTR | RepVGG | [64,128,256] | 9.5 |
| Ours-full | RepVGG | [64,128,192] | 2.0 |
| Ours-lite | RepVGG | [64,64,128] | 0.8 |

### Key Findings

- **Significant Cross-Domain Generalization**: On SN-1500, the full model surpasses ELoFTR by 3.8% at @5°; on ETH3D[I] the gain reaches 7.0% at @5°—demonstrating that cascaded prior constraints effectively improve generalization.
- Directly removing the DS operator causes a 4.4% drop on ETH3D[I] at @5°, confirming that naive acceleration is not viable.
- The lite variant of EL+CM (replacing only the matching module with cascaded matching) already outperforms the original EL-full on ETH3D[O], validating the contribution of the CasP pipeline itself.
- The lite model's feature extraction contains only 0.8M parameters (1/12 of ELoFTR) while achieving comparable accuracy.
- At 1152 resolution, the lite model is **2.2×** faster than ELoFTR and **3.2×** faster than LoFTR.
- On HPatches homography estimation, the @3px metric reaches 71.8%, approaching the dense method DKM.

## Highlights & Insights

- **Precise Core Acceleration Idea**: Deferring compute-intensive operations to coarser scales and using cheap one-to-many priors to constrain the search range of expensive one-to-one matching.
- **Elegant Train–Inference Decoupling**: Supervision signals from the DS operator and GT injection are retained during training; both are fully removed at inference to maximize efficiency.
- **Partial Softmax as a Drop-in for Global Softmax**: Computing normalization only within prior regions substantially reduces irrelevant computation.
- **Surprising Cross-Domain Generalization**: Cascaded priors not only accelerate matching but also act as a regularizer for generalization—prior constraints reduce the search space for erroneous matches.

## Limitations & Future Work

- The top-$k$ value is fixed at $k=8$; the optimal value may vary across scenes, and adaptive $k$ selection is a potential improvement direction.
- The MegaDepth-trained model shows limited improvement on InLoc (indoor), leaving room for further gains in indoor matching scenarios.
- The two-stage homography refinement assumes local regions satisfy rigid transformation, which may be unsuitable for scenes with large deformations.
- Integration with learned RANSAC and other post-processing methods has not been explored.

## Related Work & Insights

- **LoFTR**: The foundational work in semi-dense matching; this paper analyzes the efficiency bottleneck in its matching stage.
- **ELoFTR**: Introduces aggregated attention and RepVGG; this paper further addresses its residual bottleneck via a cascaded pipeline.
- **ASpanFormer / AffineFormer**: Multi-level cross-attention improves accuracy but increases runtime.
- **EcoMatcher**: The source of the Cross-CoC mechanism, reused in this paper for hybrid interaction and high-level feature extraction.
- **HomoMatcher**: The inspiration for the two-stage homography refinement.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The systematic design of cascaded priors, train–inference decoupling, and Partial Softmax is innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multi-task evaluation across pose estimation, homography, and localization; ETH3D cross-domain ablation; detailed efficiency analysis and visualization.
- **Writing Quality**: ⭐⭐⭐⭐ Pipeline diagrams are clear, mathematical derivations are complete, and efficiency analysis is thorough.
- **Value**: ⭐⭐⭐⭐⭐ An important engineering contribution for real-time systems such as SLAM and UAV, achieving 2.2× speedup without accuracy loss.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ArgMatch: Adaptive Refinement Gathering for Efficient Dense Matching](argmatch_adaptive_refinement_gathering_for_efficient_dense_matching.md)
- [\[ICCV 2025\] Diving into the Fusion of Monocular Priors for Generalized Stereo Matching](diving_into_the_fusion_of_monocular_priors_for_generalized_stereo_matching.md)
- [\[ICCV 2025\] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion](sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)
- [\[ICCV 2025\] Do It Yourself: Learning Semantic Correspondence from Pseudo-Labels](do_it_yourself_learning_semantic_correspondence_from_pseudo-labels.md)
- [\[ICCV 2025\] S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction](s3r-gs_streamlining_the_pipeline_for_large-scale_street_scene_reconstruction.md)

</div>

<!-- RELATED:END -->
