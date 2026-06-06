---
title: >-
  [Paper Note] Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception
description: >-
  [CVPR 2026][Image Generation][Image Aesthetic Enhancement] DIAE proposes a Multimodal Aesthetic Perception (MAP) module to convert vague aesthetic instructions into explicit control signals (HSV + contour maps + text). I…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Image Aesthetic Enhancement"
  - "Multimodal Aesthetic Perception"
  - "Weakly Supervised Diffusion Models"
  - "Imperfectly Paired Data"
  - "ControlNet"
date: 2026-05-08
content_hash: baeedf73f54bb6f1
---

# Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception

**Conference**: CVPR 2026  
**arXiv**: [2603.11556](https://arxiv.org/abs/2603.11556)  
**Code**: None  
**Area**: Image Generation / Image Aesthetic Enhancement  
**Keywords**: Image Aesthetic Enhancement, Multimodal Aesthetic Perception, Weakly Supervised Diffusion Models, Imperfectly Paired Data, ControlNet

## TL;DR

DIAE proposes a Multimodal Aesthetic Perception (MAP) module to convert vague aesthetic instructions into explicit control signals (HSV + contour maps + text). It constructs a "imperfectly paired" dataset, IIAEData, and utilizes a dual-branch supervision framework for weakly supervised training, achieving content-consistent aesthetic enhancement with a 17.4% improvement in LAION aesthetic scores.

## Background & Motivation

**Background**: Image aesthetic enhancement requires models to possess aesthetic perception capabilities to identify deficiencies in color, composition, and lighting for corresponding editing. While diffusion models have succeeded in image editing, existing methods focus on semantic editing and lack aesthetic perception.

**Limitations of Prior Work**: (1) **Difficulty in understanding aesthetic instructions**—evaluations like "low saturation" or "rule of thirds" are highly abstract, and simple text encoders cannot translate them into generative directions; (2) **Lack of training data**—enhancement requires "perfectly paired" images with identical content but different aesthetic quality, which is extremely costly to annotate.

**Key Challenge**: Aesthetics is a high-level human visual capability influenced by culture and experience, lacking paired data for supervised learning. Artificial degradations (blur, noise) in quality assessment datasets reflect technical quality rather than aesthetics.

**Goal**: (1) Enable diffusion models to understand and execute vague aesthetic instructions; (2) Train aesthetic enhancement models without perfectly paired data.

**Key Insight**: Decomposing aesthetic perception into color and structure dimensions, using HSV color maps and HED contour maps as visual representations combined with text descriptions. For data, weakly supervised training is conducted using "imperfectly paired" images with same semantics but different aesthetics.

**Core Idea**: Materialize vague aesthetic instructions using multimodal visual representations (HSV + contour) and achieve weakly supervised enhancement using "imperfectly paired" data with a dual-branch supervision framework.

## Method

### Overall Architecture

Three main components: (1) IIAEData construction—pairing high/low quality images from datasets like AVA/TAD66K using LLaVA for semantic matching and UNIAA-LLaVA for aesthetic text generation; (2) MAP Multimodal Aesthetic Perception—converting aesthetic evaluations into HSV maps + contour maps + text control signals injected via ControlNet; (3) Dual-branch supervision framework—input images supervise semantics (early denoising), while reference images supervise aesthetics (throughout), enabling weakly supervised training.

### Key Designs

1. **Multimodal Aesthetic Perception (MAP)**:

    - **Function**: Converts vague aesthetic instructions into explicit control signals understandable by the diffusion model.
    - **Mechanism**: Splits aesthetic evaluation into color attributes (saturation, lighting) and structural attributes (focus, composition). Color attributes use HSV maps (more intuitive for color perception than RGB), and structural attributes use HED contour maps to emphasize focus/composition. Two CNN branches $\Phi_i$ extract visual features $F_{col}^I, F_{str}^I$, while a CLIP text encoder extracts $F_{col}^T, F_{str}^T$. These are combined into control signals $\{cond_h, cond_c\}$ for UNet via ControlNet.
    - **Design Motivation**: Abstract aesthetic text is hard for simple encoders to parse, but HSV and contour maps are intuitive representations of aesthetic attributes. Both lose some semantic info, which is supplemented by text.

2. **Imperfectly Paired Dataset (IIAEData)**:

    - **Function**: Builds an aesthetic enhancement dataset for weakly supervised training.
    - **Mechanism**: Selects high MOS images as references and low MOS images as inputs from AVA, TAD66K, etc. (excluding median scores). LLaVA-13b matches pairs based on captions. UNIAA-LLaVA generates standardized aesthetic text. Experts filter incorrect pairs. Result: 47.5K samples (45K training + 1.5K testing).
    - **Design Motivation**: Perfect pairs are nearly impossible to obtain. Imperfect pairs provide "semantic-consistent but aesthetic-different" signals sufficient for weakly supervised learning.

3. **Dual-Branch Supervision Framework**:

    - **Function**: Solves training issues when input and reference image contents are inconsistent.
    - **Mechanism**: Leverages frequency stratification in diffusion denoising—early steps build semantics, later steps create aesthetic attributes. Given threshold $t_s$ (default 900), steps $t \leq t_s$ use input image for semantic consistency $L_{inp}$, while the high-MOS reference image supervises aesthetic attributes $L_{ref}$ throughout. Total loss $L = L_{ref} + \lambda L_{inp}$.
    - **Design Motivation**: Using inconsistent reference images as the sole supervision causes content drift. The dual-branch design maintains input semantics while learning reference aesthetics.

### Loss & Training

Based on SD-v1.5; UNet and ControlNet are trainable, CLIP text encoder is frozen. $t_s=900$, AdamW optimizer, learning rate 1e-5, trained on 4×A800 for 100K iterations.

## Main Experiments

### Main Results

| Method | LAION Score (256) | LAION Score (512) | MLLM Score (256) | MLLM Score (512) | CLIP-I (256) | CLIP-I (512) |
|------|-------------|-------------|------------|------------|----------|----------|
| Original Image | 4.962 | 5.123 | 3.243 | 3.300 | 1.000 | 1.000 |
| ControlNet | 4.979 | 5.522 | 3.271 | 3.415 | 0.628 | 0.617 |
| InstructPix2Pix | 4.991 | 5.396 | 3.264 | 3.325 | 0.764 | 0.690 |
| MGIE | 4.947 | 5.519 | 3.045 | 3.411 | 0.557 | 0.770 |
| DOODL | 5.102 | 5.140 | 3.255 | 3.297 | 0.775 | 0.703 |
| **Ours (DIAE)** | **5.324** | **6.012** | **3.339** | **3.662** | **0.772** | **0.784** |

### Ablation Study

| Configuration | LAION Score | MLLM Score | CLIP-I | Note |
|------|----------|---------|--------|------|
| DIAE (w/o v) | 5.250 | 3.343 | 0.623 | Removed visual modality (degrades to ControlNet) |
| DIAE (w/o t) | 5.428 | 3.410 | 0.792 | Removed text modality |
| DIAE (Full) | 5.668 | 3.501 | 0.778 | Text + Visual |

### Key Findings

- At 512 resolution, DIAE's LAION score improved by 17.4% (5.123→6.012) and MLLM score by 11.0%, while CLIP-I remained at 0.784, indicating content preservation.
- Improvement is most significant for low-aesthetic quality images (MOS < 4.0), effectively correcting color and lighting flaws.
- Removing visual modalities caused CLIP-I to drop to 0.623, highlighting that HSV/contour maps are critical for content consistency.
- A larger $t_s$ retains more input semantics—this parameter provides explicit control over content preservation vs. aesthetic enhancement.

## Highlights & Insights

- **Decomposing aesthetic perception into color + structure**: Using HSV maps for color and contour maps for composition ground abstract aesthetic concepts into concrete visual signals. This approach can be transferred to other tasks needing materialization of abstract controls.
- **Clever weakly supervised strategy**: Utilizing the frequency stratification of the denoising process to apply different supervision signals at different timesteps essentially decouples "content" and "style" in the time dimension.
- **IIAEData construction**: Using existing aesthetic datasets + LLM semantic matching to build weakly paired data is low-cost and scalable, providing a template for tasks lacking paired data.

## Limitations & Future Work

- Portrait/crowd scenes are not covered—facial features and poses are key aesthetic factors but were excluded.
- Based on SD-v1.5 rather than newer models (e.g., SD3.5), limiting generative capacity.
- Quality of IIAEData depends on LLaVA's matching accuracy; mismatch issues may exist.
- Aesthetic evaluation is limited to color and structure, missing micro-attributes like texture or lighting gradients.
- $t_s$ is fixed; adaptive adjustment might be needed for different images.

## Related Work & Insights

- **vs InstructPix2Pix**: IP2P focuses on semantic editing; it lacks aesthetic understanding and shows limited results on aesthetic tasks.
- **vs DOODL**: DOODL uses aesthetic classifier gradients during sampling but only changes the overall score without correcting specific attributes.
- **vs ControlNet**: ControlNet provides structural control but lacks aesthetic semantic understanding; DIAE adds aesthetic perception capabilities on top of it.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of multimodal aesthetic perception + weakly supervised paired data + dual-branch training is novel, though individual components are technically standard.
- **Experimental Thoroughness**: ⭐⭐⭐ Lacks user studies; CLIP-I does not fully reflect human perception of content consistency; ablation is not deep enough.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem definition, smooth motivation derivation, and rich charts.
- **Value**: ⭐⭐⭐⭐ Aesthetic enhancement is a practical task, and the weakly supervised data construction idea has generalizable value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](prototype-guided_concept_erasure_in_diffusion_models.md)
- [\[ICLR 2026\] Dual-Solver: A Generalized ODE Solver for Diffusion Models with Dual Prediction](../../ICLR2026/image_generation/dual-solver_a_generalized_ode_solver_for_diffusion_models_with_dual_prediction.md)
- [\[ICLR 2026\] Intention-Conditioned Flow Occupancy Models](../../ICLR2026/image_generation/intention-conditioned_flow_occupancy_models.md)

</div>

<!-- RELATED:END -->
