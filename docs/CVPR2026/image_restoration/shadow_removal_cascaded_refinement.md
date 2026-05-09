---
title: >-
  [Paper Note] Winner of CVPR2026 NTIRE Challenge on Image Shadow Removal: Semantic and Geometric Guidance for Shadow Removal via Cascaded Refinement
description: >-
  [CVPR 2026 Workshop (NTIRE)][Image Restoration][Shadow removal] A three-stage cascaded refinement pipeline built upon OmniSR, combining frozen DINOv2 semantic features with monocular depth/normal geometric guidance and a contraction constraint loss to stabilize multi-stage training, achieving first place in the NTIRE 2026 Image Shadow Removal Challenge.
tags:
  - CVPR 2026 Workshop (NTIRE)
  - Image Restoration
  - Shadow removal
  - cascaded refinement
  - semantic guidance
  - geometric guidance
  - progressive restoration
date: 2026-05-08
content_hash: f01ed54116432573
---

# Winner of CVPR2026 NTIRE Challenge on Image Shadow Removal: Semantic and Geometric Guidance for Shadow Removal via Cascaded Refinement

**Conference**: CVPR 2026 Workshop (NTIRE)  
**arXiv**: [2604.16177](https://arxiv.org/abs/2604.16177)  
**Code**: None  
**Area**: Image Restoration  
**Keywords**: Shadow removal, cascaded refinement, semantic guidance, geometric guidance, progressive restoration

## TL;DR

A three-stage cascaded refinement pipeline built upon OmniSR, combining frozen DINOv2 semantic features with monocular depth/normal geometric guidance and a contraction constraint loss to stabilize multi-stage training, achieving first place in the NTIRE 2026 Image Shadow Removal Challenge.

## Background & Motivation

**Background**: Image shadow removal is a fundamental low-level vision task. Recent methods have progressively evolved from handcrafted illumination models to Transformer-based end-to-end restoration systems such as ShadowFormer, HomoFormer, and OmniSR, where OmniSR achieves strong results by combining RGB appearance with semantic and geometric auxiliary information.

**Limitations of Prior Work**: Even powerful single-stage systems such as OmniSR still exhibit residual color shifts, illumination biases, and boundary artifacts after a single forward pass. One-shot inference is insufficient to fully eliminate shadow effects in complex scenes, particularly in texture-rich regions and along shadow boundaries.

**Key Challenge**: Shadow removal is fundamentally better framed as a progressive refinement problem rather than a one-shot prediction task. Single-stage methods attempt to disentangle illumination variation from intrinsic appearance in a single step, which proves inadequate for complex scenes.

**Goal**: (1) Extend OmniSR into a multi-stage cascaded refinement architecture; (2) design a loss function that stabilizes multi-stage training; (3) leverage cross-dataset progressive pretraining to improve generalization.

**Key Insight**: The authors observe that successive stages of shadow removal can correct residual errors from preceding predictions, analogous to iterative refinement strategies in inverse problems.

**Core Idea**: Replace a single forward pass with a three-stage direct refinement cascade, where each stage receives the output of the previous stage and further corrects residual artifacts, with a contraction constraint ensuring monotonically decreasing error across stages.

## Method

### Overall Architecture

Given a shadow image $\boldsymbol{x}$, the system first extracts frozen DINOv2 semantic features $S(\boldsymbol{x})$ and monocular depth-based geometric features $G(\boldsymbol{x})$ (depth channel + surface normals) from the original input in a single pass. These auxiliary signals are then fed alongside the image into three cascaded OmniSR stages. The first stage processes the original input directly; subsequent stages process the output of the preceding stage, with the final output $\hat{\boldsymbol{y}}^{(3)}$ serving as the shadow-free result.

### Key Designs

1. **Semantic and Geometric Guidance with Feature Reuse**:

    - **Function**: Provides the restoration network with scene-level semantic understanding and spatial structural constraints.
    - **Mechanism**: Frozen DINOv2 ViT-L/14 extracts four intermediate feature maps, which are projected and fused at the bottleneck layer and injected into deep Transformer blocks at 1/4 and 1/8 resolution. Depth is estimated via Depth Anything V2, normalized, and concatenated with RGB to form an RGB-D input; depth-derived point maps and normals are additionally injected into the deep blocks.
    - **Design Motivation**: Semantic features help distinguish cast shadows from intrinsically dark materials, while geometric features prevent appearance corrections from propagating across inconsistent scene regions. Features are computed once and reused across all stages to avoid redundant inference.

2. **Contraction-Constrained Multi-Stage Supervision**:

    - **Function**: Ensures that reconstruction error decreases monotonically across cascaded stages.
    - **Mechanism**: Stage-wise error is defined as $d_k = \|\hat{\boldsymbol{y}}^{(k)} - \boldsymbol{y}^*\|_2$, and the contraction loss is formulated as $\mathcal{L}_{\text{contraction}} = \sum_{k=2}^{K} [d_k - \text{sg}(d_{k-1})]_+$, where $\text{sg}$ denotes the stop-gradient operator. Only error increases are penalized; no fixed decay rate is enforced.
    - **Design Motivation**: Multi-stage training is prone to instability, as later stages may inadvertently degrade results. The contraction constraint treats the preceding stage's error as a fixed reference, requiring only that the current stage perform no worse than the previous one.

3. **Cross-Dataset Progressive Pretraining Strategy**:

    - **Function**: Learns robust priors from imperfectly aligned data before progressively adapting to precisely aligned data.
    - **Mechanism**: Phase 1 trains a single-stage model on WSRD (approximately aligned) for 500 epochs; Phase 2 transfers to WSRD+ (precisely aligned) and expands to a two-stage model for 1500 epochs; Phase 3 further expands to three stages and fine-tunes on WSRD+ 2026 for 100 epochs, with final predictions averaged over 5 checkpoints.
    - **Design Motivation**: Imperfectly aligned data provides robust training signals; the model learns tolerance for spatial misalignment and boundary imprecision, which functions as a form of data augmentation.

### Loss & Training

The total loss comprises five terms: MSE reconstruction loss + LPIPS perceptual loss (primary) + Hessian structural consistency + per-stage supervision loss (direct supervision on intermediate outputs) + contraction loss. The AdamW optimizer is used with cosine annealing learning rate scheduling, and the final prediction is the ensemble average of 5 temporally sampled checkpoints.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | 2nd (RAS) | 3rd (SNU-ISPL-B) |
|--------|------|------|------------|-------------------|
| WSRD+ 2026 Test | PSNR↑ | **26.68** | 26.14 | 25.94 |
| WSRD+ 2026 Test | SSIM↑ | **0.874** | 0.866 | 0.867 |
| WSRD+ 2026 Test | LPIPS↓ | **0.058** | 0.071 | 0.085 |
| WSRD+ 2026 Test | FID↓ | **26.14** | 30.47 | 28.05 |

### Ablation Study

| Configuration | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| 1 stage | 27.077 | 0.873 | 0.0605 |
| 2 stages | 27.274 | 0.877 | 0.0599 |
| 3 stages (Full) | **27.356** | 0.877 | 0.0631 |
| w/o contraction loss | 27.173 | 0.877 | 0.0608 |
| w/o DINOv2 guidance | 25.859 | 0.871 | 0.0711 |
| w/o depth + normals | 27.105 | 0.876 | 0.0634 |

### Key Findings

- DINOv2 semantic guidance is the most critical component; its removal causes a 1.5 dB drop in PSNR.
- Three stages is the optimal configuration; additional stages yield diminishing or negative returns in LPIPS.
- The contraction loss primarily acts as a stabilizing regularizer; its removal degrades PSNR but slightly improves LPIPS, indicating a bias toward fidelity over perceptual sharpness.

## Highlights & Insights

- Framing shadow removal as iterative refinement rather than one-shot prediction aligns conceptually with plug-and-play approaches in inverse problem solving. The contraction constraint loss is particularly elegant—penalizing only degradation without enforcing a fixed improvement rate.
- The exploitation of imperfectly aligned data is insightful: spatial misalignment, typically treated as noise, can serve as an implicit form of data augmentation that enhances model robustness.
- At 74.3M parameters, the proposed system is lightweight by competition standards (2nd-place RAS uses 1500M; 4th-place APRIL-AIGC uses 9105M), demonstrating that principled design can compensate for scale differences.

## Limitations & Future Work

- As a competition solution, the system is heavily optimized for the WSRD+ dataset; generalization to in-the-wild shadows remains uncertain.
- Three-stage inference incurs approximately 3× the computational cost of a single-stage model, limiting real-time applicability.
- The ensemble strategy (averaging 5 checkpoints) further increases inference overhead.
- Future directions include adaptive cascade depth (fewer stages for simple shadows, more for complex ones) to improve efficiency.

## Related Work & Insights

- **vs. OmniSR (single-stage)**: This work directly extends OmniSR into a three-stage pipeline, demonstrating the effectiveness of multi-stage refinement for shadow removal.
- **vs. diffusion-based methods (RAS/APRIL-AIGC)**: The second- and third-place competitors employ large diffusion models as the first stage of shadow removal. The proposed method relies entirely on a restoration architecture, achieving superior performance with one to two orders of magnitude fewer parameters.

## Rating

- Novelty: ⭐⭐⭐ Primarily combines existing components; innovation lies in systematic engineering design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated by competition rankings with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed experimental descriptions.
- Value: ⭐⭐⭐⭐ Provides a best-practice reference for multi-stage refinement in shadow removal.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors](phasr_generalized_image_shadow_removal_with_physically_aligned_priors.md)
- [\[CVPR 2026\] NTIRE 2026 The Second Challenge on Day and Night Raindrop Removal for Dual-Focused Images](ntire_2026_raindrop_removal_challenge.md)
- [\[CVPR 2026\] NTIRE 2026 The 3rd RAIM Challenge: AI Flash Portrait (Track 3)](ntire_2026_ai_flash_portrait_challenge.md)
- [\[CVPR 2026\] Flickerformer: A Duet of Periodicity and Directionality for Burst Flicker Removal](it_takes_two_a_duet_of_periodicity_and_directionality_for_burst_flicker_removal.md)
- [\[ICLR 2026\] Mechanism of Task-oriented Information Removal in In-context Learning](../../ICLR2026/image_restoration/mechanism_of_task-oriented_information_removal_in_in-context_learning.md)

</div>

<!-- RELATED:END -->
