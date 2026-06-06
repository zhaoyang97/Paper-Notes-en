---
title: >-
  [Paper Note] Inverse Virtual Try-On: Generating Multi-Category Product-Style Images from Clothed Individuals
description: >-
  [ICLR 2026][Human Understanding][Virtual Try-Off] This paper proposes TEMU-VTOFF, a Dual-DiT architecture for the Virtual Try-Off (VTOFF) task. A feature extractor and a garment generator collaborate in a division-of-lab…
tags:
  - "ICLR 2026"
  - "Human Understanding"
  - "Virtual Try-Off"
  - "Garment Extraction"
  - "Dual-DiT"
  - "Multimodal Attention"
  - "Garment Alignment"
date: 2026-05-08
content_hash: 3494953939ab7db6
---

# Inverse Virtual Try-On: Generating Multi-Category Product-Style Images from Clothed Individuals

**Conference**: ICLR 2026  
**arXiv**: [2505.21062](https://arxiv.org/abs/2505.21062)

**Code**: [Project Page](https://temu-vtoff-page.github.io/)

**Area**: Human Understanding  
**Keywords**: Virtual Try-Off, Garment Extraction, Dual-DiT, Multimodal Attention, Garment Alignment

## TL;DR

This paper proposes TEMU-VTOFF, a Dual-DiT architecture for the Virtual Try-Off (VTOFF) task. A feature extractor and a garment generator collaborate in a division-of-labor design; Multimodal Hybrid Attention (MHA) fuses image, text, and mask signals to resolve visual ambiguity; and a DINOv2-driven garment aligner preserves high-frequency details. The method achieves state-of-the-art performance on both VITON-HD and the multi-category Dress Code benchmark.

## Background & Motivation

**Task Definition**: Virtual Try-Off (VTOFF) aims to recover a standardized flat-lay product image from a photo of a clothed person. Unlike virtual try-on (VTON), VTOFF produces outputs in a consistent format (flat-lay display), yet operates under constrained input conditions (only a clothed-person photo is available).

**Commercial Value**: Fashion e-commerce requires large quantities of standardized catalog images for retrieval and recommendation, but manual photography is costly. VTOFF can automatically convert customer or model photos into standardized product images, enabling scalable catalog generation.

**Limitation of Prior Work (1) — Architectural Mismatch**: Methods such as TryOffDiff and TryOffAnyone simply invert the VTON pipeline without designing architectures tailored to VTOFF, resulting in structural artifacts around necklines, waists, and silhouettes.

**Limitation of Prior Work (2) — Single-Modality Bottleneck**: Relying solely on visual cues from a single image leads to high ambiguity under occlusion or complex poses. CLIP pooled vectors ($\mathbb{R}^{2048}$) are too coarse to encode fine-grained garment features.

**Limitation of Prior Work (3) — Category Restriction**: TryOffDiff and TryOffAnyone support only upper-body garments; MGT extends to multiple categories but still suffers from texture and color distortion.

**Technical Trend**: DiT with Flow Matching has surpassed U-Net with DDPM in diffusion models. SD3 demonstrates the effectiveness of joint text–image attention (MMDiT) within DiT, providing a foundational architecture for multimodal conditioning.

## Method

### 1. Dual-DiT Overall Architecture

The system consists of two DiTs with distinct roles:

- **Feature Extractor $F_E$**: Processes the clothed-person image and extracts multi-layer intermediate features (keys and values).
- **Garment Generator $F_D$**: Receives features from $F_E$ and denoises via MHA to generate the flat-lay garment image.
- Two-stage training: $F_E$ is first trained independently (diffusion loss), followed by joint training of $F_D$ (diffusion loss + alignment loss).

### 2. Feature Extractor $F_E$ Design

$F_E$ takes the following inputs:

- **Global Input**: The clothed-person image is encoded by CLIP into $e^v_{pool} \in \mathbb{R}^{2048}$ and modulated via AdaLN.
- **Local Spatial Input**: Channel-wise concatenation $z'_t = [z_t, M, x_M] \in \mathbb{R}^{h \times w \times 33}$
    - $z_t$: noisy latent (16 channels)
    - $M$: binary mask (1 channel)
    - $x_M = \mathcal{E}(x_{model} \odot M)$: VAE encoding of the masked person image (16 channels)

Keys and values $K^l_{extractor}, V^l_{extractor}$ are extracted at $t=0$ (rather than at each denoising step) to ensure features are derived from clean data.

Three key advantages: (i) Spatial tokens of dimension $S \times d$ are obtained, surpassing the $\mathbb{R}^{2048}$ of CLIP; (ii) $L$ layers capture multi-granularity information from coarse to fine; (iii) Features from the same DiT architecture are naturally aligned.

### 3. Multimodal Hybrid Attention (MHA)

The core innovation unifies text, latent variables, and extractor features within a single attention mechanism:

$$Q = [Q_{z_t}, Q_{text}], \quad K = [K_{z_t}, K_{extractor}, K_{text}], \quad V = [V_{z_t}, V_{extractor}, V_{text}]$$

This produces three key interaction types:

| Interaction | Role |
|-------------|------|
| $A_{text \leftrightarrow z_t}$ | Preserves pretrained language–image alignment |
| $A_{z_t \leftrightarrow extractor}$ | Transfers fine-grained features from the clothed person to the garment image |
| $A_{text \leftrightarrow extractor}$ | Anchors text semantics to structural features from the extractor |

Text embedding construction: $e_{text} = [\text{CLIP}(c), \text{T5}(c)] \in \mathbb{R}^{77 \times 4096}$

Complementary roles of mask and text:
- **Mask = Hard Discriminator**: Precisely indicates the pixel region occupied by the target garment.
- **Text = Soft Discriminator**: Provides category semantics (e.g., "upper-body shirt" / "lower-body pants").
- Together they are complementary, enabling unified multi-category processing.

### Key Designs

- **Dual-Path AdaLN Conditioning**: CLIP pooled text features $e_{pool} \in \mathbb{R}^{2048}$ provide high-level style and appearance information via AdaLN; the full text embedding provides local semantics via MHA.
- **Garment Aligner**: Addresses the insensitivity of the diffusion loss to high-frequency details due to optimization in noise space. A lightweight CNN downsamples DiT layer-8 features to align them with DINOv2 representations, supervised by a cosine similarity loss:

$$\mathcal{L}_{align} = -\mathbb{E}_{z_g, \epsilon_t, t}\left[\frac{1}{N}\sum_{i=1}^{N}\cos(\tilde{h}_i^{DiT}, h_i^{enc})\right]$$

- **Training-Only Module**: The aligner is discarded at inference, introducing zero additional computational overhead.
- **Total Loss**: $\mathcal{L}_{total} = \mathcal{L}_{DiT} + \lambda \cdot \mathcal{L}_{align}$

## Key Experimental Results

### Table 1: Main Results on Dress Code

| Method | SSIM↑ | LPIPS↓ | DISTS↓ | FID↓ | KID↓ |
|--------|-------|--------|--------|------|------|
| Any2AnyTryon | 77.56 | 35.17 | 25.17 | 12.32 | 3.65 |
| MGT | 77.77 | 35.37 | 27.28 | 13.47 | 5.28 |
| **TEMU-VTOFF** | **75.95** | **31.46** | **18.66** | **5.74** | **0.65** |

FID improves by 53.4% over the second-best method Any2AnyTryon (12.32→5.74); DISTS improves by 25.9%.

### Table 2: Main Results on VITON-HD

| Method | SSIM↑ | LPIPS↓ | DISTS↓ | FID↓ | KID↓ |
|--------|-------|--------|--------|------|------|
| TryOffDiff | 75.53 | 39.56 | 25.53 | 17.49 | 5.30 |
| TryOffAnyone | 75.90 | 35.26 | 23.47 | 12.74 | 2.85 |
| One Model for All | — | 22.50 | 19.20 | 9.12 | 1.49 |
| **TEMU-VTOFF** | **77.21** | **28.44** | **18.04** | **8.71** | **1.11** |

### Table 3: Ablation Study (Dress Code, Selected Key Results)

| Configuration | DISTS↓ | FID↓ |
|---------------|--------|------|
| w/o Feature Extractor $F_E$ | 23.56 | 9.11 |
| w/o Garment Aligner | 20.63 | 5.91 |
| w/o Text and Mask | 25.20 | 9.63 |
| w/o Text Modulation | 22.54 | 7.75 |
| w/o Fine-Grained Mask | 20.87 | 6.58 |
| **Full TEMU-VTOFF** | **18.66** | **5.74** |

Each component contributes clearly. Removing $F_E$ raises FID from 5.74 to 9.11 (+58.7%), validating the core value of the Dual-DiT design.

## Key Findings

- The **text↔extractor interaction in MHA** is critical for resolving occlusion ambiguity — text provides semantic anchors for structurally invisible regions.
- **Joint use of mask and text far outperforms either alone**: removing both raises FID from 5.74→9.63; removing only the mask yields 6.58, removing only text yields 7.75.
- **Cross-dataset generalization**: Training on Dress Code and testing on VITON-HD yields FID 20.39 vs. MGT's 23.11; the reverse transfer also demonstrates clear advantages (FID 18.63 vs. TryOffDiff's 41.91).
- **Downstream gains**: Augmenting training data with synthetic garment images generated by TEMU-VTOFF reduces FID for CatVTON across all categories, validating the practical utility of the generation quality.

## Highlights & Insights

- **VTOFF-specific architectural design**: Rather than naively inverting the VTON pipeline, the paper addresses the limited-input nature of VTOFF by designing a Dual-DiT with a dedicated separation between feature extraction and generation.
- **Mask-as-hard / text-as-soft discriminator complementarity**: A clear analytical framework — the mask determines "which pixels," and the text determines "which category" — with both being indispensable.
- **Elegant use of DINOv2 alignment**: Applied only during training with zero inference overhead; rather than reconstructing in pixel space, alignment is enforced in semantic space, yielding more robust high-frequency detail preservation.
- **Practical validation**: Beyond evaluating VTOFF itself, the paper demonstrates that synthetic data generated by the model improves downstream VTON task performance.
- **Cross-dataset experimental design** is rigorous, demonstrating genuine generalization rather than dataset-specific overfitting.

## Limitations & Future Work

- SSIM is not optimal (75.95 vs. Any2AnyTryon's 77.56), indicating room for improvement in pixel-level alignment accuracy.
- Lower-body category performance is weaker (Dress Code lower-body: ~9k samples vs. upper-body: ~15k vs. full-body dress: ~29k), reflecting a data imbalance issue.
- Reliance on text descriptions as input conditions requires an additional captioning module in deployment scenarios.
- Mask extraction quality directly affects results, necessitating a reliable segmentation frontend.
- Inference speed is constrained by the dual forward passes of Dual-DiT (though $F_E$ runs only once).
- Validation is limited to fashion datasets; generalization to broader item categories (accessories, footwear, bags) remains unexplored.

## Related Work & Insights

### vs. TryOffDiff (Velioglu et al., 2024)
TryOffDiff pioneered the VTOFF task using a SigLIP-conditioned diffusion model to recover garment images, but supports only a single category (upper-body) and reuses VTON architectures, leading to structural artifacts. TEMU-VTOFF fundamentally redesigns the VTOFF pipeline via Dual-DiT and MHA, reducing FID on VITON-HD from 17.49 to 8.71 (DISTS: 25.53→18.04) and achieving multi-category unified processing for the first time.

### vs. MGT (Velioglu et al., 2025)
MGT extends to multiple categories via category embeddings but remains constrained by coarse-grained visual encoding. TEMU-VTOFF substantially outperforms it on Dress Code FID (5.74 vs. 13.47) and in cross-dataset evaluation (20.39 vs. 23.11). The key distinction is TEMU-VTOFF's dual-modality conditioning (text + mask) and dedicated feature extractor, as opposed to merely appending category labels.

### vs. One Model for All (Liu et al., 2025)
One Model for All unifies VTON and VTOFF into a single framework, achieving competitive LPIPS (22.50) and DISTS (19.20). Nevertheless, as a VTOFF-specialized architecture, TEMU-VTOFF remains superior on FID (8.71 vs. 9.12) and KID (1.11 vs. 1.49), demonstrating the continued value of task-specific design.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Dual-DiT division of labor + three-way cross-attention in MHA + mask/text complementary disambiguation → a clearly motivated and original VTOFF-specific architectural design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Two datasets, six metrics, full ablation, cross-dataset generalization, and downstream VTON augmentation experiments constitute a comprehensive evaluation framework.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation analysis is thorough (mask as hard / text as soft discriminator); method description is well-structured; experimental comparisons are fair.
- **Value**: ⭐⭐⭐⭐ Directly applicable to e-commerce platforms; downstream gain experiments validate real-world utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RefTon: Reference Person Shot Assist Virtual Try-on](../../CVPR2026/human_understanding/refton_reference_person_shot_assist_virtual_try-on.md)
- [\[CVPR 2026\] Mobile-VTON: High-Fidelity On-Device Virtual Try-On](../../CVPR2026/human_understanding/mobile_vton_ondevice_virtual_tryon.md)
- [\[CVPR 2026\] Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](../../CVPR2026/human_understanding/reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)
- [\[AAAI 2026\] Generating Attribute-Aware Human Motions from Textual Prompt](../../AAAI2026/human_understanding/generating_attribute-aware_human_motions_from_textual_prompt.md)
- [\[CVPR 2026\] WildCap: Facial Albedo Capture in the Wild via Hybrid Inverse Rendering](../../CVPR2026/human_understanding/wildcap_facial_albedo_capture_in_the_wild_via_hybrid_inverse_rendering.md)

</div>

<!-- RELATED:END -->
