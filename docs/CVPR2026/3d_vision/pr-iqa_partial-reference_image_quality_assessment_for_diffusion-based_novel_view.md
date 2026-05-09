---
title: >-
  [Paper Note] PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis
description: >-
  [CVPR 2026][3D Vision][Image quality assessment] This paper proposes PR-IQA, a cross-reference image quality assessment method that first computes geometrically consistent local quality maps in multi-view overlapping regions, then propagates quality information to non-overlapping regions via a reference-conditioned cross-attention network, producing dense quality maps approaching full-reference accuracy. Integrated into a 3DGS pipeline with a dual-filtering strategy, it significantly improves sparse-view 3D reconstruction quality.
tags:
  - CVPR 2026
  - 3D Vision
  - Image quality assessment
  - cross-reference
  - novel view synthesis
  - 3D Gaussian splatting
  - diffusion models
date: 2026-05-08
content_hash: a6c36e2bcebce0f8
---

# PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis

**Conference**: CVPR 2026
**arXiv**: [2604.04576](https://arxiv.org/abs/2604.04576)
**Code**: [https://github.com/Kakaomacao/PR-IQA](https://github.com/Kakaomacao/PR-IQA)
**Area**: 3D Vision / Image Quality Assessment
**Keywords**: Image quality assessment, cross-reference, novel view synthesis, 3D Gaussian splatting, diffusion models

## TL;DR

This paper proposes PR-IQA, a cross-reference image quality assessment method that first computes geometrically consistent local quality maps in multi-view overlapping regions, then propagates quality information to non-overlapping regions via a reference-conditioned cross-attention network, producing dense quality maps approaching full-reference accuracy. Integrated into a 3DGS pipeline with a dual-filtering strategy, it significantly improves sparse-view 3D reconstruction quality.

## Background & Motivation

**State of the Field**: Diffusion models are increasingly important for sparse-view novel view synthesis (NVS)—they can generate pseudo-ground-truth images to compensate for missing viewpoints in 3D reconstruction pipelines such as 3DGS. However, diffusion-generated images frequently contain photometric and geometric inconsistencies, and using them directly as supervision degrades reconstruction quality.

**Limitations of Prior Work**: Full-reference IQA (FR-IQA, e.g., PSNR/SSIM/LPIPS) requires pixel-aligned ground-truth images, which are unavailable in NVS settings. No-reference IQA (NR-IQA) requires no reference but struggles to capture high-level geometric inconsistencies in diffusion-generated images. Cross-reference IQA (CR-IQA) leverages reference images from different poses, but existing methods either perform only patch-level similarity matching (e.g., CrossScore using SSIM) without semantic understanding, or are effective only in overlapping regions (e.g., MEt3R), leaving evaluation blind spots.

**Root Cause**: CR-IQA faces a dilemma—geometrically aligned overlapping regions yield reliable quality estimates, whereas non-overlapping regions cannot be directly evaluated. Simple patch-similarity methods do not require overlap but are low in accuracy; geometry-consistent methods are accurate but lack full coverage.

**Paper Goals**: Design a CR-IQA method that simultaneously exploits the geometric reliability of overlapping regions and the contextual reasoning capability for non-overlapping regions, producing dense quality maps over the entire image.

**Starting Point**: The quality assessment of non-overlapping regions is reformulated as a "quality map completion" problem—analogous to image inpainting, but completing quality scores rather than pixel values. Cross-view context provided by reference images guides the completion process.

**Core Idea**: Reliable local quality maps are first computed in overlapping regions, then a reference-conditioned cross-attention network completes the local quality into a full-image dense quality map, achieving full-reference-level accuracy without ground-truth images.

## Method

### Overall Architecture

PR-IQA operates in two stages: (1) **Local quality map generation**—3D point clouds obtained via VGGT establish geometric correspondences, DINOv2 features from the reference image are warped to the query view, and cosine similarity at overlapping pixels yields a local quality map $\hat{Q}$; (2) **Quality map completion**—a three-stream encoder-decoder network takes the query image $I_q$, reference image $I_r$, and local quality map $\hat{Q}$ as inputs, and predicts a full-image dense quality map $Q$ via reference-conditioned cross-attention.

### Key Designs

1. **Local Quality Map Generation**

   - **Function**: Obtain reliable pixel-level quality estimates in geometrically aligned overlapping regions.
   - **Mechanism**: VGGT is used to obtain dense 3D point clouds of the query and reference images; DINOv2 features of the reference image are warped to the query view coordinate system via back-projection and re-projection. LoftUp upsamples features to high resolution. Normalized cosine similarity is computed at each overlapping pixel $i$: $\hat{Q}(i) = \text{CosSim}(F_q^{\text{DINO}}(i), F_{r \to q}^{\text{DINO}}(i))$. Quality values in non-overlapping regions are left empty.
   - **Design Motivation**: Geometric alignment ensures spatial consistency in feature comparison, while DINOv2 features provide high-level semantic information. Their combination yields highly reliable quality estimates in overlapping regions, serving as "anchors" for subsequent completion.

2. **Three-Stream Encoder-Decoder Quality Completion Network**

   - **Function**: Complete the local quality map into a full-image dense quality map.
   - **Mechanism**: Three separate encoders process the reference image (self-attention encoder $\text{Enc}_{\text{self}}^r$), query image (cross-attention encoder $\text{Enc}_{\text{cross}}^q$), and local quality map (cross-attention encoder $\text{Enc}_{\text{cross}}^p$). The two cross-attention encoders use reference image features as keys/values, enabling explicit view alignment. After each stage, query and quality map features are fused via channel-wise concatenation. The decoder progressively upsamples to generate a full-resolution quality map.
   - **Design Motivation**: The three-stream design decouples "cross-view alignment" (via the reference image) from "quality propagation" (via the local quality map), allowing the network to learn these two capabilities independently. The fusion operation ensures quality propagation is anchored to geometrically verified regions.

3. **Dual-Gated Attention Block**

   - **Function**: Achieve decoupled channel and spatial attention within each encoding stage.
   - **Mechanism**: Based on the CBAM design, channel attention (max/avg pooling + MLP channel recalibration) and spatial attention (Q/K/V projection + softmax spatial refinement) are applied sequentially, each followed by normalization, residual connection, and FFN. Channel attention determines "what features are relevant"; spatial attention determines "where to propagate."
   - **Design Motivation**: For the quality completion task, decoupling "what" and "where" is critical—channel attention selects quality-relevant feature channels, while spatial attention propagates quality information from reliable regions to non-overlapping areas.

### Loss & Training

The total loss consists of three terms: $\mathcal{L} = 0.5 \cdot \mathcal{L}_1^{\text{IQA}} + 1.0 \cdot \mathcal{L}_{\text{JSD}} + 0.25 \cdot \mathcal{L}_{\text{PLCC}}$. $\mathcal{L}_1^{\text{IQA}}$ ensures pixel-level accuracy; Jensen-Shannon divergence $\mathcal{L}_{\text{JSD}}$ aligns the global score distribution; Pearson correlation coefficient loss $\mathcal{L}_{\text{PLCC}}$ enforces linear consistency. Two variants are trained targeting DINOv2 similarity maps and SSIM maps, respectively. Training data uses the MFR dataset, with 3 variants generated per frame via VDM, yielding 120k training pairs.

## Key Experimental Results

### Main Results: IQA Performance (PLCC/SRCC, higher is better)

| Method | Type | Mip-NeRF 360 PLCC | Mip-NeRF 360 SRCC | Tanks&Temples PLCC | Tanks&Temples SRCC |
|--------|------|-----------|-----------|-----------|-----------|
| LPIPS | FR-IQA† | 0.557 | 0.472 | 0.591 | 0.590 |
| PIQE | NR-IQA | 0.144 | 0.161 | 0.194 | 0.201 |
| PaQ-2-PiQ | NR-IQA | -0.088 | -0.107 | 0.039 | 0.118 |
| CrossScore | CR-IQA | 0.094 | 0.090 | 0.237 | 0.272 |
| PuzzleSim | CR-IQA | 0.304 | 0.327 | 0.351 | 0.369 |
| MEt3R* | CR-IQA | 0.105 | 0.129 | 0.142 | 0.153 |
| **Ours (DINOv2)** | CR-IQA | **0.555** | **0.622** | **0.573** | **0.650** |

### IQA-Guided 3DGS Reconstruction

| IQA Method | Mip-NeRF PSNR↑ | Mip-NeRF SSIM↑ | Mip-NeRF LPIPS↓ | T&T PSNR↑ | T&T SSIM↑ |
|------------|----------------|----------------|-----------------|-----------|-----------|
| Vanilla 3DGS | 16.08 | 0.461 | 0.415 | 15.30 | 0.509 |
| ViewCrafter (no IQA) | 16.18 | 0.474 | 0.453 | 15.77 | 0.523 |
| CrossScore | 16.31 | 0.476 | 0.431 | 15.86 | 0.537 |
| PuzzleSim | 16.35 | 0.482 | 0.423 | 15.94 | 0.541 |
| **Ours (DINOv2)** | **16.76** | **0.493** | **0.414** | **16.24** | **0.551** |
| DINOv2† (FR) | 17.18 | 0.498 | 0.399 | 16.78 | 0.562 |

### Ablation Study

| Variant | Mip-NeRF PLCC | Mip-NeRF SRCC | T&T PLCC | T&T SRCC | Note |
|---------|--------------|--------------|----------|----------|------|
| Reversed attention order | 0.540 | 0.609 | 0.517 | 0.584 | Channel→spatial order is superior |
| w/o channel attention | 0.554 | 0.611 | 0.571 | 0.633 | Channel attention is beneficial |
| w/o reference image branch | 0.544 | 0.613 | 0.553 | 0.637 | Reference image provides useful context |
| w/o local quality map branch | 0.421 | 0.464 | 0.452 | 0.438 | **Most critical component** |
| Full model | 0.555 | 0.622 | 0.573 | 0.650 | — |

### Key Findings

- **The local quality map is the most critical input**: Removing it causes SRCC to drop from 0.622 to 0.464 (−25.4%), far exceeding the impact of removing the reference image branch (−1.4%), confirming that geometrically aligned quality estimates in overlapping regions are the cornerstone of the entire method.
- **PR-IQA approaches full-reference accuracy**: On Mip-NeRF 360, PR-IQA achieves SRCC 0.622 vs. LPIPS (FR) at 0.472—PR-IQA even surpasses some FR metrics in correlation, demonstrating that effective exploitation of cross-view information can compensate for the absence of ground truth.
- **NR-IQA is largely ineffective in NVS settings**: PaQ-2-PiQ yields negative correlation, indicating that general no-reference quality metrics cannot detect geometric inconsistencies in diffusion-generated images.
- **PR-IQA significantly outperforms other CR methods in 3DGS reconstruction**: On T&T, PSNR 16.24 vs. PuzzleSim 15.94, while approaching the upper bound guided by FR-IQA (DINOv2) at 16.78.

## Highlights & Insights

- The **reformulation as "quality map completion"** is particularly elegant. The core difficulty of CR-IQA (inability to directly evaluate non-overlapping regions) is recast as an inpainting-like problem—completing quality scores rather than pixels. This formulation allows mature inpainting techniques (cross-attention, multi-scale fusion, etc.) to be directly repurposed.
- The **dual-filtering strategy** tightly integrates IQA with 3DGS training—image-level selection of the best candidates combined with pixel-level quality masking to supervise only high-confidence regions. This coarse-to-fine filtering is highly practical in real applications.
- The **three-stream encoder design** decouples "cross-view alignment" from "quality propagation," enabling the network to learn these two capabilities independently. In particular, reference-conditioned cross-attention injected at every scale is more effective than fusing only at the highest level.

## Limitations & Future Work

- The method relies on VGGT for 3D correspondences and DINOv2 for feature extraction; the quality of these pretrained models directly affects the reliability of local quality maps.
- The quality map completion network must be trained separately for different FR targets (DINOv2-SIM or SSIM), lacking a unified quality representation.
- The quality threshold $\tau=50$ is set heuristically and may require adjustment for different scenes—adaptive thresholding strategies remain to be explored.
- Validation is conducted only on pseudo-ground-truth generated by a video diffusion model (ViewCrafter); applicability to other diffusion models (e.g., Zero123++, SV3D) is unknown.

## Related Work & Insights

- **vs. MEt3R**: MEt3R also uses geometric alignment for CR-IQA but is limited to overlapping regions. PR-IQA extends evaluation to the full image via a quality completion network, eliminating blind spots, and substantially outperforms MEt3R on SRCC (0.622 vs. 0.129).
- **vs. CrossScore**: CrossScore estimates SSIM maps via cross-attention but operates at the patch level without geometric awareness. PR-IQA also significantly outperforms CrossScore on the SSIM target (0.556 vs. 0.325).
- **vs. PuzzleSim**: PuzzleSim uses feature-level cosine similarity, which correlates partially with the DINOv2 target but lacks sufficient accuracy. PR-IQA's geometric alignment plus completion strategy provides substantially more accurate quality estimates.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Reformulating CR-IQA as quality map completion is a novel perspective; the three-stream encoder with reference-conditioned cross-attention is well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets, two FR targets, comprehensive IQA comparisons, 3DGS application validation, and detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear, visual comparisons are intuitive, and problem motivation is well-articulated.
- **Value**: ⭐⭐⭐⭐ Directly addresses the practical pain point of quality assessment for diffusion-generated views; 3DGS integration demonstrates a clear application pathway.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] QD-PCQA: Quality-Aware Domain Adaptation for Point Cloud Quality Assessment](qd-pcqa_quality-aware_domain_adaptation_for_point_cloud_quality_assessment.md)
- [\[CVPR 2026\] GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis](geodesicnvs_probability_density_geodesic_flow_matching_for_novel_view_synthesis.md)
- [\[AAAI 2026\] 3D-Free Meets 3D Priors: Novel View Synthesis from a Single Image with Pretrained Diffusion Guidance](../../AAAI2026/3d_vision/3d-free_meets_3d_priors_novel_view_synthesis_from_a_single_image_with_pretrained.md)
- [\[CVPR 2026\] Physically Inspired Gaussian Splatting for HDR Novel View Synthesis](physically_inspired_gaussian_splatting_for_hdr_novel_view_synthesis.md)
- [\[CVPR 2026\] DMAligner: Enhancing Image Alignment via Diffusion Model Based View Synthesis](dmaligner_enhancing_image_alignment_via_diffusion_model_based_view_synthesis.md)

<!-- RELATED:END -->
