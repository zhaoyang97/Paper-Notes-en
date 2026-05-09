---
title: >-
  [Paper Note] HouseMind: Tokenization Allows MLLMs to Understand, Generate and Edit Architectural Floor Plans
description: >-
  [CVPR 2026][Multimodal VLM][Architectural Floor Plans] This paper proposes HouseMind, a framework that discretizes architectural floor plans into structured sequences of contour tokens and room instance tokens via a hierarchical VQ-VAE. Combined with three-stage multimodal alignment and instruction fine-tuning on Qwen3-0.6B as the backbone, HouseMind achieves unified modeling of floor plan understanding, generation, and editing, substantially outperforming existing methods in geometric validity and controllability.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Architectural Floor Plans
  - VQ-VAE
  - Multimodal LLM
  - Spatial Reasoning
  - Unified Generation
date: 2026-05-08
content_hash: 379ad6d167b17f1e
---

# HouseMind: Tokenization Allows MLLMs to Understand, Generate and Edit Architectural Floor Plans

**Conference**: CVPR 2026
**arXiv**: [2603.11640](https://arxiv.org/abs/2603.11640)
**Code**: [https://housemind.github.io/](https://housemind.github.io/)
**Area**: Multimodal VLM
**Keywords**: Architectural Floor Plans, VQ-VAE, Multimodal LLM, Spatial Reasoning, Unified Generation

## TL;DR
This paper proposes HouseMind, a framework that discretizes architectural floor plans into structured sequences of contour tokens and room instance tokens via a hierarchical VQ-VAE. Combined with three-stage multimodal alignment and instruction fine-tuning on Qwen3-0.6B as the backbone, HouseMind achieves unified modeling of floor plan understanding, generation, and editing, substantially outperforming existing methods in geometric validity and controllability.

## Background & Motivation

**Background**: AI-assisted architectural floor plan design has seen diverse approaches including GANs, graph neural networks, and diffusion models. Recent LLM/MLLM-based paradigms, such as Tell2Design and ChatHouseDiffusion, have explored language-driven layout generation.

**Limitations of Prior Work**: (1) Diffusion and autoregressive models treat layout generation as a purely visual process, lacking explicit room-instance-level reasoning; (2) large models largely operate as black boxes with poor spatial controllability; (3) understanding, generation, and editing cannot be unified; (4) high computational overhead makes local deployment impractical.

**Key Challenge**: Floor plan design requires joint processing of geometric and semantic information. Existing methods either excel at geometry but lack semantic reasoning, or possess language capabilities but sacrifice spatial precision.

**Goal**: To build an efficient, locally deployable unified framework that simultaneously supports spatial understanding, conditional generation, and controllable editing of floor plans.

**Key Insight**: The key breakthrough is identified at the representation level — using VQ-VAE to discretize floor plans into structured token sequences so that an LLM can process spatial layouts as it does language.

**Core Idea**: A hierarchical VQ-VAE represents floor plans as discrete sequences of contour tokens and room instance tokens, enabling an MLLM to autoregressively unify all three design tasks.

## Method

### Overall Architecture
A floor plan is decomposed into a contour and $N$ rooms. Two independent VQ-VAEs encode them into discrete token sequences, which together with room labels form a structured sequence $Z = [\mathbf{z}_o, \ell_{r_1}, \mathbf{z}_{r_1}, \dots]$. This sequence is interleaved with text tokens and fed into Qwen3-0.6B, which is trained via a three-stage procedure for unified autoregressive modeling.

### Key Designs

1. **Hierarchical VQ-VAE (Contour + Conditional Room Discretization)**:

    - Function: Encodes the global contour and each room instance separately into discrete tokens.
    - Mechanism: The contour is quantized with a CNN encoder against codebook $\mathcal{Z}_o$; each room is encoded conditionally using the room mask and contour as joint input: $z_{i,j}^{(r)} = \arg\min_k \|E_r(x_{r_i}, x_o)_j - e_k^{(r)}\|_2$.
    - Design Motivation: Conditional encoding allows room tokens to capture both geometric shape and spatial position relative to the contour simultaneously.

2. **Three-Stage Multimodal Alignment Training**:

    - Function: Progressively establishes alignment between spatial tokens and language tokens.
    - Mechanism: Stage 1 integrates VQ-VAE codebook embeddings into the LLM vocabulary; Stage 2 performs autoregressive pretraining on large-scale paired data; Stage 3 applies supervised fine-tuning (SFT) on instruction data covering all three tasks.
    - Design Motivation: Progressive alignment avoids optimization instability that arises from directly training on complex tasks.

3. **Unified Sequence Modeling (Understanding / Generation / Editing)**:

    - Function: Unifies all three tasks as conditional autoregressive prediction.
    - Mechanism: Generation follows $p(Z|\mathbf{z}_o, s) = \prod_t p(Z_t|Z_{<t}, \mathbf{z}_o, s)$; understanding outputs textual descriptions; editing takes the original sequence plus an instruction and outputs the modified sequence.
    - Design Motivation: A unified format enables the same model to share knowledge across different tasks.

### Loss & Training
The VQ-VAE is trained with reconstruction loss and codebook loss. The LLM component uses cross-entropy autoregressive loss. The model is built on Qwen3-0.6B and trained on the RPLAN dataset (78,738 samples), with inference on a single RTX 3090.

## Key Experimental Results

### Main Results

| Method | Micro IoU | FID↓ | Node F1 | Edge Overlap | Inference (s) |
|---|---|---|---|---|---|
| Tell2Design | 0.390 | 30.5 | 0.808 | 0.197 | ~15 |
| ChatHouseDiffusion | 0.589 | 11.3 | 0.985 | 0.710 | ~30 |
| **HouseMind-G** | **0.709** | **1.91** | **0.994** | **0.880** | ~2 |

### Ablation Study (Understanding Task)

| Method | RMR | LocAcc | AreaDiff↓ | AdjAcc | RelAcc |
|---|---|---|---|---|---|
| Qwen3-VL-8B | 0.698 | 0.347 | 5.837 | 0.382 | 0.128 |
| InternVL3.5-8B | 0.847 | 0.546 | 12.234 | 0.469 | 0.157 |
| **HouseMind-U** | **0.998** | **0.969** | **0.549** | **0.990** | **0.808** |

### Key Findings
- FID drops from 11.3 (ChatHouseDiffusion) to 1.91, marking a substantial improvement in generation quality.
- On the understanding task, room localization accuracy of 0.969 exceeds the best VLM baseline by over 40 points, with an area error of only 0.549 m².
- On the editing task, ΔIoU reaches 0.608, far surpassing FLUX (0.053) and Qwen-Edit (0.088).
- The unified model HouseMind-O achieves performance close to single-task variants across all tasks.

## Highlights & Insights
- **Room-level tokenization**: Naturally mirrors the cognitive process of architectural design (global outline first → room layout second).
- **Extreme efficiency**: A 0.6B-parameter model outperforms 7–8B VLMs, with inference of only ~2 seconds per sample.
- **Comparison with GPT-5/Gemini Pro**: Domain-specific tokenization achieves superior precision over general-purpose large models.

## Limitations & Future Work
- Editing is limited to room addition and deletion; complex topological transformations are not supported.
- Functional components such as doors, windows, and furniture are not modeled.
- Evaluation is conducted solely on the RPLAN dataset.

## Related Work & Insights
- **vs MaskPLAN**: MaskPLAN encodes the layout holistically, whereas HouseMind encodes hierarchically by room, preserving structural information.
- **vs FloorPlanLLaMA**: VQ-VAE encoding of the entire layout leads to boundary inconsistencies; HouseMind maintains geometric precision through conditional VQ-VAE encoding.

## Rating
- Novelty: ⭐⭐⭐⭐ — Hierarchical tokenization concept is novel; three-task unified design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation across three tasks, though limited to RPLAN.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure.
- Value: ⭐⭐⭐⭐ — Direct practical value for AI-assisted architectural design.

## Related Papers

- [\[CVPR 2026\] Tokenization Allows Multimodal Large Language Models to Understand, Generate and Edit Architectural Floor Plans (HouseMind)](tokenization_allows_multimodal_large_language_models_to_understand_generate_and_.md)
- [\[CVPR 2026\] HiSpatial: Taming Hierarchical 3D Spatial Understanding in Vision-Language Models](hispatial_taming_hierarchical_3d_spatial_understanding_in_vision-language_models.md)
- [\[CVPR 2026\] ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps](reasonmap_towards_finegrained_visual_reasoning_fro.md)
- [\[CVPR 2026\] SoPE: Spherical Coordinate-Based Positional Embedding for 3D LVLMs](sope_spherical_positional_encoding_3d_lvlm.md)
- [\[CVPR 2026\] Generate, Analyze, and Refine: Training-Free Sound Source Localization via MLLM Meta-Reasoning](generate_analyze_and_refine_training-free_sound_source_localization_via_mllm_met.md)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Tokenization Allows Multimodal Large Language Models to Understand, Generate and Edit Architectural Floor Plans (HouseMind)](tokenization_allows_multimodal_large_language_models_to_understand_generate_and_.md)
- [\[CVPR 2026\] Token Warping Helps MLLMs Look from Nearby Viewpoints](token_warping_helps_mllms_look_from_nearby_viewpoints.md)
- [\[CVPR 2026\] EgoMind: Activating Spatial Cognition through Linguistic Reasoning in MLLMs](egomind_activating_spatial_cognition_through_linguistic_reasoning_in_mllms.md)
- [\[CVPR 2026\] ENC-Bench: A Benchmark for Evaluating MLLMs in Electronic Navigational Chart Understanding](enc-bench_a_benchmark_for_evaluating_multimodal_large_language_models_in_electro.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)

</div>

<!-- RELATED:END -->
