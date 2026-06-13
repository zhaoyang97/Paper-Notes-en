---
title: >-
  [Paper Note] STMI: Segmentation-Guided Token Modulation with Cross-Modal Hypergraph Interaction for Multi-Modal Object Re-Identification
description: >-
  [AAAI 2026][3D Vision][Multi-Modal ReID] STMI proposes a three-component multi-modal object re-identification framework that suppresses background noise via SAM segmentation-guided feature modulation (SFM)…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Multi-Modal ReID"
  - "Cross-Modal Fusion"
  - "Hypergraph Interaction"
  - "Segmentation Guidance"
  - "Token Modulation"
date: 2026-05-08
content_hash: 507342f3349ff978
---

# STMI: Segmentation-Guided Token Modulation with Cross-Modal Hypergraph Interaction for Multi-Modal Object Re-Identification

**Conference**: AAAI 2026
**arXiv**: [2603.00695](https://arxiv.org/abs/2603.00695)  
**Code**: None  
**Area**: 3D Vision / Multi-Modal
**Keywords**: Multi-Modal ReID, Cross-Modal Fusion, Hypergraph Interaction, Segmentation Guidance, Token Modulation

## TL;DR
STMI proposes a three-component multi-modal object re-identification framework that suppresses background noise via SAM segmentation-guided feature modulation (SFM), extracts compact representations through semantic token reallocation (STR), and captures high-order semantic relationships via cross-modal hypergraph interaction (CHI), achieving significant improvements on benchmarks such as RGBNT201.

## Background & Motivation

**Background**: Multi-modal object re-identification (Multi-Modal ReID) leverages complementary information from different modalities (RGB, near-infrared NIR, thermal infrared TIR) to retrieve specific targets. Existing methods extract per-modality features using ViT and then perform fusion.

**Limitations of Prior Work**: (1) Hard token filtering risks discarding discriminative information, and simple fusion strategies fail to effectively exploit cross-modal complementarity; (2) Background noise manifests differently across modalities, and effective foreground/background separation mechanisms are lacking.

**Key Challenge**: How to achieve compact representation while retaining all token information, and how to effectively model high-order cross-modal semantic relationships.

**Goal**: Design a unified multi-modal learning framework that simultaneously addresses background suppression, information compression, and cross-modal high-order interaction.

**Key Insight**: Employ SAM-generated masks for soft modulation rather than hard filtering; use learnable query tokens for adaptive reallocation; model high-order inter-modal relationships via hypergraphs.

**Core Idea**: Three modules—SFM (foreground enhancement) + STR (representation compression) + CHI (multi-modal high-order interaction)—form a complete information processing pipeline.

## Method

### Overall Architecture
Multi-modal images are fed as input, and patch tokens are extracted per modality via ViT. The SFM module modulates token attention using SAM-pregenerated segmentation masks. The STR module compresses the modulated tokens into compact representations via learnable query tokens. The CHI module constructs a unified hypergraph over the compact representations of all modalities to capture high-order cross-modal semantic relationships.

### Key Designs

1. **Segmentation-Guided Feature Modulation (SFM)**:

    - **Function**: Enhances foreground representation and suppresses background noise using SAM segmentation masks.
    - **Mechanism**: SAM is used offline to generate foreground masks, which are converted into learnable attention weights to softly modulate tokens at each ViT layer. Unlike hard token filtering, SFM retains all tokens while redistributing importance weights.
    - **Design Motivation**: Hard filtering may erroneously discard discriminative foreground tokens; soft modulation reduces background interference while preserving information integrity.

2. **Semantic Token Reallocation (STR)**:

    - **Function**: Compresses variable-length patch tokens into a fixed number of compact semantic representations.
    - **Mechanism**: Learnable query tokens interact with modulated patch tokens via cross-attention to achieve adaptive semantic reallocation. The number of query tokens is much smaller than that of patch tokens; no tokens are discarded—all information is aggregated into query tokens via attention.
    - **Design Motivation**: Traditional top-$k$ selection inevitably loses information; STR reallocates all information into compact representations through attention.

3. **Cross-Modal Hypergraph Interaction (CHI)**:

    - **Function**: Models high-order semantic relationships across all modalities.
    - **Mechanism**: Compact representations from each modality are treated as hypergraph nodes to construct a unified cross-modal hypergraph. Hyperedges connect multiple nodes, capturing ternary and higher-order semantic associations. Information propagation is performed via hypergraph convolution.
    - **Design Motivation**: Multi-modal information contains high-order relationships beyond pairwise interactions; hypergraphs naturally model such relationships.

### Loss & Training
Standard ReID training strategy is adopted: identity classification loss (Cross-Entropy) + metric learning loss (Triplet Loss).

## Key Experimental Results

### Main Results

| Dataset | Metric | STMI | Prev. SOTA | Gain |
|--------|------|------|--------|------|
| RGBNT201 | mAP | Best | — | Significantly surpasses all baselines |
| RGBNT100 | mAP | Best | — | Clear advantage of multi-modal fusion |
| MSVR310 | mAP | Best | — | Hypergraph interaction proven effective |

### Ablation Study

| Configuration | Performance | Notes |
|------|------|------|
| Full STMI | Best | Three modules in synergy |
| w/o SFM | Notable drop | Increased background interference |
| w/o STR | Drop | Token redundancy reduces discriminability |
| w/o CHI | Drop | Absence of cross-modal high-order interaction |

### Key Findings
- SFM contributes the most, validating that background noise is the primary bottleneck in multi-modal ReID.
- CHI hypergraph interaction shows clear advantages over standard graph convolution, demonstrating the necessity of high-order relationship modeling.
- STR retains more discriminative information compared to hard top-$k$ selection.

## Highlights & Insights
- **SAM as a universal foreground extractor**: The zero-shot segmentation capability of SAM provides foreground masks for ReID, transferable to tasks such as pedestrian attribute recognition and vehicle ReID.
- **Hypergraph vs. ordinary graph**: Hypergraphs can model high-order semantics beyond pairwise relationships, offering greater expressiveness in multi-modal scenarios.
- **Compression without token loss**: The "reallocation rather than discarding" philosophy of STR balances efficiency and information completeness.

## Limitations & Future Work
- Reliance on SAM for offline mask generation introduces additional preprocessing overhead.
- Hypergraph construction and convolution increase computational complexity.
- Validation is limited to the RGB-NIR-TIR three-modality setting; performance on other modality combinations remains unknown.
- SAM may fail under low-quality or extreme conditions, potentially degrading SFM effectiveness.

## Related Work & Insights
- **vs. TOP-ReID**: TOP-ReID employs token pruning which may cause information loss; STMI's STR avoids this through reallocation.
- **vs. traditional multi-modal fusion**: Simple concatenation/summation cannot capture high-order cross-modal relationships; CHI is more effective.
- **vs. TransReID**: Multi-modal inputs provide complementary information; STMI demonstrates the advantages of structured fusion.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of SAM and hypergraph is novel, with individual innovations in each of the three modules.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across three benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear.
- Value: ⭐⭐⭐⭐ Advances the field of multi-modal ReID.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Multi-Modal Assistance for Unsupervised Domain Adaptation on Point Cloud 3D Object Detection](multi-modal_assistance_for_unsupervised_domain_adaptation_on_point_cloud_3d_obje.md)
- [\[CVPR 2026\] Glove2Hand: Synthesizing Natural Hand-Object Interaction from Multi-Modal Sensing Gloves](../../CVPR2026/3d_vision/glove2hand_synthesizing_natural_hand-object_interaction_from_multi-modal_sensing.md)
- [\[AAAI 2026\] MR-CoSMo: Visual-Text Memory Recall and Direct Cross-Modal Alignment Method for Query-Driven 3D Segmentation](mr-cosmo_visual-text_memory_recall_and_direct_cross-modal_alignment_method_for_q.md)
- [\[CVPR 2026\] CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration](../../CVPR2026/3d_vision/cmhanet_a_cross-modal_hybrid_attention_network_for_point_cloud_registration.md)
- [\[CVPR 2026\] AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis](../../CVPR2026/3d_vision/affordgrasp_cross-modal_diffusion_for_affordance-aware_grasp_synthesis.md)

</div>

<!-- RELATED:END -->
