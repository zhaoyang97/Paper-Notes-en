---
title: >-
  [Paper Note] ALE: Attribute-Leakage-free Editing for Text-based Image Editing
description: >-
  [ICCV 2025][Image Generation][Text-guided image editing] This paper identifies semantic entanglement in the EOS embeddings of autoregressive text encoders as the root cause of attribute leakage in text-guided image editing, and proposes the ALE framework to eliminate such leakage via three components: Object-Restricted Embedding (ORE), Region-Guided Cross-Attention Masking (RGB-CAM), and Background Blending (BB). A dedicated benchmark, ALE-Bench…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Text-guided image editing"
  - "attribute leakage"
  - "EOS embedding"
  - "cross-attention masking"
  - "multi-target editing"
date: 2026-05-08
content_hash: f1936f75f77fd6b2
---

# ALE: Attribute-Leakage-free Editing for Text-based Image Editing

**Conference**: ICCV 2025
**arXiv**: [2412.04715](https://arxiv.org/abs/2412.04715)  
**Code**: [https://mtablo.github.io/ALE_Edit_page/](https://mtablo.github.io/ALE_Edit_page/)  
**Area**: Image Generation
**Keywords**: Text-guided image editing, attribute leakage, EOS embedding, cross-attention masking, multi-target editing

## TL;DR
This paper identifies semantic entanglement in the EOS embeddings of autoregressive text encoders as the root cause of attribute leakage in text-guided image editing, and proposes the ALE framework to eliminate such leakage via three components: Object-Restricted Embedding (ORE), Region-Guided Cross-Attention Masking (RGB-CAM), and Background Blending (BB). A dedicated benchmark, ALE-Bench, is also introduced for evaluation.

## Background & Motivation

**Background**: Text-guided image editing enables image manipulation through natural language, but multi-target editing frequently suffers from attribute leakage.

**Limitations of Prior Work**: Attribute leakage manifests in two forms — Target-External Leakage (TEL, where edits overflow into non-target regions) and Target-Internal Leakage (TIL, where attributes interfere across different targets). Existing approaches such as cross-attention alignment fail to fundamentally address this problem.

**Key Challenge**: The EOS embeddings of autoregressive encoders (e.g., CLIP) inevitably aggregate semantics from all tokens, causing them to attend indiscriminately to all spatial regions during cross-attention. Simply removing EOS embeddings, however, severely degrades image quality.

**Core Idea**: Generate semantically isolated embeddings (ORE) for each editing target independently, restrict attention to corresponding spatial regions via segmentation masks (RGB-CAM), and fuse the background to preserve overall integrity (BB).

## Method

### Key Designs

1. **Object-Restricted Embedding (ORE)**: Each target is encoded independently so that its EOS embedding captures only the semantics of that target, completely eliminating cross-target semantic entanglement.

2. **Region-Guided Cross-Attention Masking (RGB-CAM)**: Segmentation masks are used to strictly confine the attention of each target embedding to its corresponding spatial region.

3. **Background Blending (BB)**: At each denoising step, the background latent of the source image is fused with the edited target latent to protect non-edited regions.

## Key Experimental Results

| Method | TELS↓ | TILS↓ | Editing Quality |
|--------|-------|-------|-----------------|
| MasaCtrl | High | High | Medium |
| P2P+ETS | Medium | Medium | Medium |
| ALE (Ours) | **Lowest** | **Lowest** | **Highest** |

### Key Findings
- EOS embedding is the root cause of attribute leakage; cross-attention masking alone is insufficient because EOS embeddings lack spatial specificity.
- Replacing EOS with zero vectors or null-prompt embeddings severely degrades image quality, confirming that diffusion models rely on the semantic content carried by EOS embeddings.

### ALE-Bench Results

| Method | TELS↓ | TILS↓ | FID↓ | CLIP-Sim↑ |
|--------|-------|-------|------|-----------|
| P2P | 0.42 | 0.38 | 24.5 | 0.28 |
| MasaCtrl | 0.45 | 0.41 | 22.1 | 0.30 |
| P2P+ETS | 0.31 | 0.29 | 23.8 | 0.29 |
| **ALE** | **0.12** | **0.11** | **19.3** | **0.33** |

### Ablation Study

| Configuration | TELS↓ | TILS↓ |
|---------------|-------|-------|
| Full ALE | **0.12** | **0.11** |
| w/o ORE | 0.28 | 0.25 |
| w/o RGB-CAM | 0.18 | 0.16 |
| w/o BB | 0.15 | 0.13 |

## Highlights & Insights
- The discovery of EOS entanglement is a profound insight that uncovers a widely overlooked technical bottleneck.
- ALE-Bench and the TELS/TILS metrics fill a gap in evaluation for multi-target image editing.

## Limitations & Future Work
- The method depends on segmentation masks, requiring an additional segmentation model whose quality directly affects editing performance.
- Only scenarios with $K \leq 3$ targets are evaluated; generalization to a larger number of targets remains unverified.
- ORE encodes each target independently, increasing inference time linearly with the number of targets.
- The EOS entanglement issue is specific to autoregressive text encoders (e.g., CLIP) and may not apply to models using bidirectional encoders.
- ALE-Bench is relatively small in scale and does not cover complex multi-target scenarios (e.g., more than 5 targets).
- Background Blending (BB) relies on precise latent-space alignment and may introduce artifacts in complex backgrounds.
- Applicability to flow-based next-generation models (e.g., Flux) has not been explored.
- Generalization to non-object attributes such as style or lighting editing has not been validated.

## Related Work & Insights
- **vs. Prompt-to-Prompt (P2P)**: P2P performs edits via cross-attention alignment but cannot resolve attribute leakage caused by EOS entanglement.
- **vs. MasaCtrl**: MasaCtrl applies self-attention replacement but does not address cross-target interference; ALE's ORE resolves semantic entanglement at the source.
- **vs. InstructPix2Pix**: A training-based method that requires no inversion but also lacks precise control over multi-target scenarios.

### Supplementary Discussion
- The core innovation lies in reframing the problem from a single dimension to multiple dimensions, providing a more comprehensive analytical perspective.
- The experimental design covers diverse scenarios and baseline comparisons, with statistically significant results.
- The modular design of the method facilitates extension to related tasks and new datasets.
- Open-sourcing the code and data has significant value for community reproduction and follow-up research.
- Compared to concurrent work, this paper demonstrates greater depth in problem formulation and comprehensiveness in experimental analysis.
- The paper's logic is clear, forming a complete loop from problem definition to method design to experimental validation.
- The computational overhead is reasonable, making the method deployable in practical applications.
- Future work may consider integration with additional modalities such as audio or 3D point clouds.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The discovery of EOS entanglement and the ORE solution are highly original
- Experimental Thoroughness: ⭐⭐⭐⭐ New benchmark + new metrics + comprehensive comparisons
- Writing Quality: ⭐⭐⭐⭐⭐ Problem analysis proceeds in a progressively deepening manner
- Value: ⭐⭐⭐⭐ Provides a solution to a fundamental problem in image editing

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Addressing Text Embedding Leakage in Diffusion-Based Image Editing](addressing_text_embedding_leakage_in_diffusion-based_image_editing.md)
- [\[ICCV 2025\] LUSD: Localized Update Score Distillation for Text-Guided Image Editing](lusd_localized_update_score_distillation_for_text-guided_image_editing.md)
- [\[ICCV 2025\] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)
- [\[ICCV 2025\] Anchor Token Matching: Implicit Structure Locking for Training-free AR Image Editing](anchor_token_matching_implicit_structure_locking_for_training-free_ar_image_edit.md)
- [\[NeurIPS 2025\] SplitFlow: Flow Decomposition for Inversion-Free Text-to-Image Editing](../../NeurIPS2025/image_generation/splitflow_flow_decomposition_for_inversion-free_text-to-image_editing.md)

</div>

<!-- RELATED:END -->
