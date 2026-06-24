---
title: >-
  [Paper Note] Addressing Text Embedding Leakage in Diffusion-Based Image Editing
description: >-
  [ICCV 2025][Image Generation][Image Editing] This work identifies the root cause of attribute leakage in text-driven diffusion-based image editing — semantic entanglement in EOS embeddings of autoregressive text encoders — and proposes the ALE framework (ORE + RGB-CAM + BB) to comprehensively eliminate attribute leakage through embedding disentanglement, attention masking, and background blending.
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Image Editing"
  - "Attribute Leakage"
  - "Diffusion Models"
  - "Text Embeddings"
  - "Cross-Attention"
date: 2026-05-08
content_hash: 425e69ddf2949404
---

# Addressing Text Embedding Leakage in Diffusion-Based Image Editing

**Conference**: ICCV 2025
**Code**: [https://mtablo.github.io/ALE_Edit_page/](https://mtablo.github.io/ALE_Edit_page/)  
**Area**: Image Generation
**Keywords**: Image Editing, Attribute Leakage, Diffusion Models, Text Embeddings, Cross-Attention

## TL;DR

This work identifies the root cause of attribute leakage in text-driven diffusion-based image editing — semantic entanglement in EOS embeddings of autoregressive text encoders — and proposes the ALE framework (ORE + RGB-CAM + BB) to comprehensively eliminate attribute leakage through embedding disentanglement, attention masking, and background blending.

## Background & Motivation

### Core Problem
Text-driven image editing based on diffusion models (e.g., Prompt-to-Prompt) enables users to modify images through natural language prompts. However, existing methods universally suffer from **Attribute Leakage**: edits targeting a specific object unintentionally affect unrelated regions or other objects.

### Two Types of Attribute Leakage

**Target-External Leakage (TEL)**: Editing a target object unintentionally affects non-target regions. For example, when editing "red pepper" to "golden apple," the adjacent "green pepper" also turns golden.

**Target-Internal Leakage (TIL)**: In multi-object editing, the attributes of one target object unintentionally influence another. For example, when editing "yellow pepper → red pumpkin," a region intended to become "golden apple" instead exhibits mixed attributes of red pumpkin.

### Why Do Existing Methods Fall Short?

**Root Cause Analysis** (one of the paper's core contributions):

The problem lies in the **EOS (End-of-Sequence) embedding** of autoregressive text encoders such as CLIP. CLIP pads prompts to a fixed length (77 tokens) using EOS tokens, and EOS embeddings **indiscriminately aggregate semantic information from all attributes and objects** in the entire prompt.

For instance, when encoding "a red diamond and a golden apple," the EOS embedding conflates the semantics of "diamond," "golden," and "apple." In cross-attention layers, these EOS embeddings activate indiscriminately across the entire image, causing leakage.

**Limitations of Prior Approaches**:
- **Object-wise embeddings** (e.g., Structured Diffusion): Encode prompts by noun phrases independently, but only address entanglement among original tokens — EOS entanglement remains unresolved.
- **End-Token-Substitution (ETS)**: Replaces original EOS embeddings with attribute-free ones, but the simplified EOS still aggregates semantics from multiple objects.
- **Cross-attention masking**: Constrains the spatial attention range of EOS embeddings, but EOS inherently lacks spatial specificity (it integrates the entire prompt), rendering spatial constraints ineffective.
- **Zero-vector / empty-prompt EOS replacement**: Eliminates semantic entanglement but severely degrades image quality and editing precision — demonstrating that diffusion models **inherently rely** on semantic information in EOS embeddings.

**Key Insight**: A strategy is needed that both preserves the semantic content of EOS embeddings (ensuring editing quality) and prevents cross-object attribute interference (eliminating leakage).

## Method

### Overall Architecture

ALE is built upon a **Dual-Branch editing framework** combined with a DDCM virtual inversion scheme:
- **Source Branch**: Reconstructs source image $x_{src}$ conditioned on source prompt $y_{base}^{src}$, capturing structural and spatial information.
- **Target Branch**: Denoises conditioned on target prompt $y_{base}^{tgt}$ to produce the edited image.
- **Structure Preservation**: Injects self-attention Q and K from the source branch into corresponding layers of the target branch.

Three complementary components are introduced upon this foundation: **ORE** (embedding-level disentanglement), **RGB-CAM** (attention-level spatial constraint), and **BB** (latent-space-level background protection).

### Key Designs

#### 1. DDCM Virtual Inversion

The Denoising Diffusion Consistent Model (DDCM) employs a special variance schedule such that any noisy latent $z_\tau$ maintains a closed-form relationship with the clean latent $z_0$ at every timestep. This enables:
- Elimination of costly DDIM / Null-text inversion
- Editing in only 4–20 steps
- Compatibility with the multi-step consistency sampler of Latent Consistency Models

**Self-attention injection schedule $\mathcal{S}$**: Controls the strength of structure preservation. A shorter schedule injects only in early denoising steps, permitting larger-scale edits; a longer schedule enforces stricter structural fidelity.

#### 2. Object-Restricted Embeddings (ORE) — Embedding-Level Disentanglement

**Core Idea**: Each target object prompt is encoded independently, so that each embedding matrix contains the semantics of only a single object.

Each object prompt $y_i^{tgt}$ is encoded independently to obtain:
$$E_i = [e_{BOS}, e_{token_1}, ..., e_{EOS}, ...]$$

For example, given "a red diamond and a golden apple," $E_1$ is encoded solely from "a red diamond" and $E_2$ solely from "a golden apple."

A base embedding $E_{base}$ is also constructed by encoding the full prompt $y_{base}^{tgt}$ and splicing in the corresponding spans from each $E_i$.

**Why does ORE resolve EOS entanglement?** The EOS embedding in $E_i$ contains semantic information only from the single object prompt $y_i^{tgt}$, with no aggregation of other objects' attributes. The cross-attention layers thus receive semantically fully disentangled embeddings — eliminating leakage **at the source**.

#### 3. RGB-CAM (Region-Guided Blending for Cross-Attention Masking) — Attention-Level Spatial Constraint

**Problem**: Standard cross-attention layers accept only a single value tensor $V$ and cannot leverage multiple OREs simultaneously.

**Solution**: Replace the vanilla cross-attention output with a spatially blended tensor:
$$A = \sum_{i=1}^{K} (M \odot m_i) V_i + (M \odot m_{back}) V_{base}$$

where:
- $M$ is the base cross-attention map computed from $Q$ and $K$
- $V_i = W_v(E_i)$ is the value tensor for the $i$-th object
- $\{m_i\}$ and $m_{back}$ are segmentation masks for target objects and background, generated by Grounded-SAM
- Masks are slightly dilated to handle boundary imprecision

**Spatial precision**: $(M \odot m_i) V_i$ confines each ORE to its designated region, eliminating TIL; the background term protects non-edited areas.

**Key point**: ORE and RGB-CAM must be **used jointly** to achieve leakage-free results — neither component alone is sufficient.

#### 4. Background Blending (BB) — Latent-Space-Level Background Protection

Even with perfect cross-attention control, the background remains vulnerable because $\{y_i^{tgt}\}$ describes only the target objects. At each timestep $\tau$, source latents are blended using the background mask:
$$z_\tau^{tgt} = m_{back} \odot z_\tau^{src} + (1 - m_{back}) \odot z_\tau^{tgt}$$

BB guarantees the fidelity of non-edited regions and suppresses TEL without the costly threshold tuning required by prior methods such as P2P.

### Loss & Training

ALE is a **tuning-free** framework — no additional training or fine-tuning is required. The entire process is an inference-time editing pipeline:

1. Preprocessing: parse prompts, encode OREs, obtain segmentation masks via Grounded-SAM.
2. Initialization: sample initial noise $z_T^{src}$; set $z_T^{tgt} = z_T^{src}$.
3. Iterate from $T$ to 1: source branch predicts noise → target branch (with RGB-CAM) predicts noise → BB blends latents.
4. Decode $z_0^{tgt}$ to obtain the edited image.

Only 4–20 inference steps are required (enabled by DDCM), making the approach computationally efficient.

## Key Experimental Results

### Main Results

**Comparison on ALE-Bench**:

| Method | TELS↓ | TILS↓ | Structure Distance↓ | Editing Performance↑ | PSNR↑ | SSIM↑ |
|--------|-------|-------|---------------------|----------------------|-------|-------|
| P2P | 21.52 | 17.26 | 0.1514 | 20.67 | 11.15 | 0.5589 |
| MasaCtrl | 20.18 | 16.74 | 0.0929 | 20.01 | 14.99 | 0.7346 |
| FPE | 21.07 | 17.38 | 0.1164 | 21.89 | 12.82 | 0.6052 |
| InfEdit | 19.59 | 16.69 | 0.0484 | 21.78 | 16.74 | 0.7709 |
| **ALE** | **16.03** | **15.28** | **0.0167** | **22.20** | **30.04** | **0.9228** |

ALE outperforms all baselines across every metric: TEL is reduced by 3.56 over InfEdit, PSNR improves by 13.3 dB, and SSIM increases from 0.77 to 0.92.

### Ablation Study

**Analysis by number of edited objects**:

| # Edited Objects | TELS↓ | TILS↓ | Editing Perf↑ | PSNR↑ | SSIM↑ |
|-----------------|-------|-------|---------------|-------|-------|
| 1 | 16.41 | - | 22.62 | 30.01 | 0.9049 |
| 2 | 16.00 | 15.42 | 22.06 | 30.06 | 0.9235 |
| 3 | 15.89 | 15.36 | 22.19 | 30.01 | 0.9426 |

**Analysis by editing type**:

| Editing Type | TELS↓ | TILS↓ | Editing Perf↑ | PSNR↑ |
|-------------|-------|-------|---------------|-------|
| Color | 17.63 | 16.21 | 23.12 | 32.97 |
| Material | 17.15 | 15.96 | 22.94 | 30.63 |
| Object | 15.86 | 16.25 | 21.82 | 29.03 |
| Color+Object | 15.30 | 14.01 | 22.15 | 28.60 |
| Object+Material | 14.55 | 14.51 | 21.42 | 28.88 |

### Key Findings

1. **Stable performance under multi-object editing**: As the number of edited objects increases from 1 to 3, ALE's leakage metrics and background preservation quality remain stable or even improve slightly (SSIM increases from 0.90 to 0.94).
2. **Composite editing types are more challenging**: Color+Object and Object+Material editing yields slightly lower editing performance than single-type edits, yet their leakage metrics are lower.
3. **Color editing is most prone to leakage yet easiest to execute**: TELS = 17.63 is the highest among editing types, but Editing Performance = 23.12 is also the highest.
4. **Remarkable PSNR advantage**: ALE achieves PSNR = 30.04, far exceeding the second-best InfEdit at 16.74, demonstrating outstanding background preservation.
5. **Superior structural fidelity**: Structure Distance = 0.0167, only one-third that of InfEdit (0.0484).

## Highlights & Insights

1. **Root-cause resolution**: Rather than applying surface-level patches (e.g., adjusting attention masks), the paper traces the problem to its fundamental origin — semantic entanglement in EOS embeddings — and designs targeted solutions accordingly.
2. **Elegantly complementary three-component design**: ORE addresses embedding-level entanglement, RGB-CAM addresses attention-level spatial issues, and BB addresses latent-space-level background degradation; all three levels are indispensable.
3. **A complete evaluation framework**: ALE-Bench and the TELS/TILS metrics fill a gap in the assessment of multi-object editing.
4. **Tuning-free**: No additional training or fine-tuning is required; the method operates directly on pretrained models, conferring broad applicability.
5. **Efficient inference**: DDCM virtual inversion requires only 4–20 steps, avoiding the computational overhead of DDIM inversion.
6. **Philosophy of EOS embedding treatment**: Simply discarding EOS embeddings degrades quality; simply replacing them retains entanglement. The correct approach is **isolation at the source** — an insight with broad implications beyond this work.

## Limitations & Future Work

1. **Restricted to rigid edits**: The current framework supports only local edits such as color, object, and material changes; non-rigid transformations including style transfer, pose variation, and object addition/removal are not supported.
2. **Limited benchmark scale**: ALE-Bench comprises only 20 carefully curated source images (3,000 editing scenarios in total), limiting image diversity.
3. **Dependence on segmentation models**: RGB-CAM relies on Grounded-SAM for segmentation masks; segmentation quality directly affects editing quality.
4. **Maximum of $K=3$ objects**: Validation is limited to multi-object editing scenarios involving at most three objects.
5. **Focused on CLIP encoders**: The analysis is primarily grounded in the autoregressive CLIP text encoder; EOS-related issues in bidirectional encoders such as T5 are not discussed.

## Related Work & Insights

- **Prompt-to-Prompt (P2P)**: Achieves editing via cross-attention control but is severely affected by EOS entanglement.
- **MasaCtrl**: A tuning-free approach based on mutual attention control that preserves layout but cannot prevent leakage.
- **InfEdit**: An efficient editing method integrating DDCM; serves as the direct base framework for ALE.
- **ETS (NeurIPS 2024)**: The first work to address EOS embedding issues, though its solution is incomplete.
- **Broader implication**: Semantic entanglement in EOS embeddings may not be exclusive to image editing — analogous issues likely arise in all CLIP-dependent tasks including text-to-image generation and text-to-video generation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (The in-depth analysis of EOS entanglement is highly insightful; the ORE+RGB-CAM+BB three-component design is elegant)
- Experimental Thoroughness: ⭐⭐⭐⭐ (A complete benchmark is proposed, though the number of source images is limited)
- Writing Quality: ⭐⭐⭐⭐⭐ (Problem analysis is thorough, visualizations are rich, and concepts are clearly defined)
- Value: ⭐⭐⭐⭐⭐ (Addresses a key pain point in multi-object editing; tuning-free nature facilitates practical deployment)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ALE: Attribute-Leakage-free Editing for Text-based Image Editing](ale_attribute_leakage_free_editing.md)
- [\[ICCV 2025\] Text Embedding Knows How to Quantize Text-Guided Diffusion Models](text_embedding_knows_how_to_quantize_text-guided_diffusion_models.md)
- [\[ICCV 2025\] Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing](early_timestep_zero-shot_candidate_selection_for_instruction-guided_image_editin.md)
- [\[ICCV 2025\] Fair Generation without Unfair Distortions: Debiasing Text-to-Image Generation with Entanglement-Free Attention](fair_generation_without_unfair_distortions_debiasing_text-to-image_generation_wi.md)
- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](../../NeurIPS2025/image_generation/training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
