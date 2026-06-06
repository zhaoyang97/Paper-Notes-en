---
title: >-
  [Paper Note] Tokenization Allows Multimodal Large Language Models to Understand, Generate and Edit Architectural Floor Plans (HouseMind)
description: >-
  [CVPR 2026][Multimodal VLM][Multimodal large language models] This paper presents HouseMind, which discretizes architectural floor plans into room-level spatial tokens via a hierarchical VQ-VAE…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Multimodal large language models"
  - "VQ-VAE"
  - "spatial tokenization"
  - "floor plan generation"
  - "floor plan editing"
  - "instruction tuning"
date: 2026-05-08
content_hash: 01fa30a58f8124c2
---

# Tokenization Allows Multimodal Large Language Models to Understand, Generate and Edit Architectural Floor Plans (HouseMind)

**Conference**: CVPR 2026
**arXiv**: [2603.11640](https://arxiv.org/abs/2603.11640)  
**Code**: [housemind.github.io](https://housemind.github.io/)  
**Area**: Multimodal VLM / Architectural Floor Plan Design
**Keywords**: Multimodal large language models, VQ-VAE, spatial tokenization, floor plan generation, floor plan editing, instruction tuning

## TL;DR

This paper presents HouseMind, which discretizes architectural floor plans into room-level spatial tokens via a hierarchical VQ-VAE, enabling floor plan understanding, generation, and editing within a unified MLLM framework. The approach comprehensively outperforms diffusion model and general-purpose VLM baselines in geometric validity and controllability.

## Background & Motivation

**High cognitive complexity of architectural floor plan design**: Floor plan design requires simultaneous reasoning over geometric, semantic, and spatial hierarchical relationships. Patterns are not sequential but embedded in complex relational structures, posing a significant challenge for AI.

**Lack of global spatial consistency in existing methods**: Diffusion models and autoregressive models have achieved improvements in visual fidelity but treat layout synthesis as a purely visual process, lacking explicit room-instance-level reasoning. This leads to locally plausible but globally spatially incoherent results (e.g., inconsistent adjacency and circulation relationships).

**Insufficient interpretability and controllability**: Large-scale vision-language models commonly function as black-box generators, with limited spatial controllability and interpretability.

**Inability to unify understanding, generation, and editing**: Existing frameworks struggle to simultaneously handle understanding, generation, and editing tasks within a single architecture, particularly given the geometric and semantic complexity of architectural layouts.

**High computational overhead and difficulty of local deployment**: Most AI systems demand substantial computational resources, making integration into practical design workflows difficult.

**Existing LLM-driven design approaches remain modular**: Methods such as Tell2Design, ChatHouseDiffusion, and FloorPlanLLaMA improve interpretability but operate as independent modules, lacking unified multi-task reasoning.

## Method

### Overall Architecture

HouseMind consists of two core components: **Room-Instance Tokenization** and **Multimodal Alignment & Instruction Tuning**.

A floor plan is decomposed into an outline $x_o$ and $N$ room instances $\{x_{r_i}\}_{i=1}^N$, which are encoded into discrete token sequences via two separate VQ-VAEs and then interleaved into a unified sequence:

$$Z = [\boldsymbol{z}_o, \ell_{r_1}, \boldsymbol{z}_{r_1}, \dots, \ell_{r_N}, \boldsymbol{z}_{r_N}]$$

where $\ell_{r_i}$ denotes the semantic label token of room $i$, and $\boldsymbol{z}_o$ and $\boldsymbol{z}_{r_i}$ are the discrete tokens for the outline and each room, respectively.

### Key Designs

1. **Outline Discretization**: A CNN encoder $E_o$ extracts features from the binary outline mask, which are vector-quantized into discrete tokens via the outline codebook $\mathcal{Z}_o$; a decoder reconstructs the outline.
2. **Conditional Room Discretization**: The room encoder $E_r$ jointly encodes each room mask conditioned on the outline, and quantizes it via the room codebook $\mathcal{Z}_r$. Conditional encoding enables room representations to be globally context-aware, capturing geometric and spatial adjacency relationships.
3. **Three-stage Multimodal Training**:

    - **Stage 1 – Embedding Initialization**: Spatial codes from the VQ-VAE codebook are mapped to trainable token embeddings in the LLM vocabulary, establishing a one-to-one correspondence between discrete spatial codes and text tokens.
    - **Stage 2 – Multimodal Pre-training**: The model is trained on large-scale paired data consisting of text descriptions, outline tokens, and room tokens using an autoregressive language modeling objective, achieving bidirectional alignment between language and geometry.
    - **Stage 3 – Instruction Tuning (SFT)**: Supervised fine-tuning is performed on instruction data covering understanding, generation, and editing tasks, endowing the model with task awareness and spatial reasoning capability.
4. **Unified Task Modeling**: Understanding (inferring room functions and topology from $Z$), generation (autoregressively generating a layout given text $s$ and outline $\boldsymbol{z}_o$), and editing (generating a modified layout $Z^{tgt}$ given a source layout $Z^{src}$ and instruction $s$) are all formulated as a unified sequence modeling problem.

### Backbone & Efficiency

HouseMind is built upon Qwen3-0.6B as the language model backbone. Its small parameter count supports real-time inference and local deployment on a single RTX 3090 GPU.

## Experiments

### Dataset & Benchmark

A unified benchmark for evaluating floor plan understanding, generation, and editing is constructed based on the RPLAN dataset, comprising 80,738 samples: 76,122 for training, 2,308 for validation, and 2,308 for testing. Each floor plan includes a JSON representation and both simple and detailed text descriptions (generated by Qwen3-30B-A3B).

### Understanding Task Results

| Method | RMR | LocAcc | AreaDiff↓ | AdjAcc | RelAcc | Time (s) |
|--------|-----|--------|-----------|--------|--------|----------|
| LLaVA-v1.6-Mistral-7B | 0.616 | 0.225 | 3.649 | 0.134 | 0.056 | ~6 |
| Qwen3-VL-8B | 0.698 | 0.347 | 5.837 | 0.382 | 0.128 | ~8 |
| InternVL3.5-8B | 0.847 | 0.546 | 12.234 | 0.469 | 0.157 | ~13 |
| MiniCPM-V 4.5 | 0.904 | 0.492 | 13.765 | 0.597 | 0.208 | ~14 |
| **HouseMind-U** | **0.998** | **0.969** | **0.549** | **0.990** | **0.808** | **~3** |

HouseMind improves room localization accuracy and adjacency accuracy by more than 40 absolute percentage points, reducing area error from several square meters to below 0.6 m².

### Generation Task Results

| Method | Micro IoU | Macro IoU | FID↓ | GED↓ | Node F1 | Edge Ovl. | Time (s) |
|--------|-----------|-----------|------|------|---------|-----------|----------|
| Tell2Design | 0.390 | 0.307 | 30.5 | 6.94 | 0.808 | 0.197 | ~15 |
| ChatHouseDiffusion | 0.589 | 0.521 | 11.3 | 2.36 | 0.985 | 0.710 | ~30 |
| FloorPlanLLaMA | 0.607 | 0.511 | 49.3 | 2.68 | 0.922 | 0.574 | ~1 |
| **HouseMind-G** | **0.709** | **0.653** | **1.91** | **1.01** | **0.994** | **0.880** | **~2** |

IoU improves by more than 10% over ChatHouseDiffusion, and FID drops from 11.3 to 1.9.

### Editing Task Results

| Method | ΔIoU | ΔMSE↓ | Node F1 | Edge Ovl. |
|--------|------|-------|---------|-----------|
| FLUX.1-Kontext-dev | 0.053 | 0.0162 | 0.765 | 0.222 |
| Qwen-Image-Edit | 0.088 | 0.0074 | 0.915 | 0.426 |
| **HouseMind-E** | **0.608** | **0.0019** | **0.998** | **0.934** |

### Ablation Study

| Configuration | Train Loss↓ | Eval Loss↓ |
|---------------|-------------|------------|
| w/o Stage 1&2 | 0.0729 | 0.0836 |
| w/o Stage 1 | 0.0659 | 0.0840 |
| w/o Stage 2 | 0.0712 | 0.0831 |
| **Full** | **0.0644** | **0.0830** |

### Key Findings

- Removing Stage 1 (embedding initialization) leads to unstable optimization, preventing spatial tokens from settling into a stable embedding space.
- Removing Stage 2 (multimodal pre-training) deprives the model of high-level text–layout correspondences.
- The unified variant HouseMind-O achieves performance close to or on par with task-specific models trained independently on each task.
- In qualitative comparisons with GPT-5 and Gemini 2.5 Pro, HouseMind demonstrates superior spatial consistency and controllability.

## Highlights & Insights

- **Room-level discrete tokenization** is the core innovation: it bridges continuous geometric layouts to discrete sequence modeling, enabling LLMs to perform interpretable spatial reasoning directly in token space.
- **Unified three-task modeling**: a single model simultaneously handles understanding, generation, and editing without modular composition.
- **Extremely lightweight and deployable**: built on Qwen3 with 0.6B parameters, inference runs on a single RTX 3090, requiring only 2–3 seconds per sample.
- **Conditional room encoding**: room encoding is conditioned on the outline, naturally capturing global context and adjacency relationships.
- **First unified benchmark**: a standardized evaluation protocol covering all three tasks is established.

## Limitations & Future Work

- The editing functionality supports only simple addition and deletion operations, without handling complex topological transformations such as global reorganization.
- Functional components such as doors, windows, and furniture are not modeled, limiting applicability to detailed interior design.
- Generated results are not aligned with human design preferences and aesthetic constraints, leaving a gap relative to professional design standards.
- The dataset is based on RPLAN (predominantly Chinese residential buildings); generalizability to other architectural typologies and cultural styles remains to be verified.

## Related Work & Insights

- **GAN-based**: Graph-constrained GANs and similar approaches improve realism but overfit to local geometry.
- **Graph/GNN-based**: Methods such as Graph2Plan model room connectivity, but discrete graph representations limit geometric fidelity.
- **Diffusion-based**: GSDiff, FloorPlan Diffusion, and related methods are stable but computationally expensive and limited to single tasks.
- **LLM-driven**: Tell2Design establishes a text-to-floor-plan benchmark; ChatHouseDiffusion and FloorPlanLLaMA introduce language control but remain modular architectures.
- **Positioning of this work**: HouseMind is the first unified multi-task multimodal framework to jointly learn geometric, semantic, and topological representations.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of room-level VQ-VAE tokenization with a unified three-task LLM framework is novel, recasting spatial design as token sequence modeling.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation across three tasks with multiple baselines (including GPT-5/Gemini) and ablation studies validating the training strategy.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, formalized problem definitions, and rich figures and tables.
- Value: ⭐⭐⭐⭐ — Establishes a unified paradigm in architectural design AI; the lightweight deployable design has practical application potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] See, Hear, and Understand: Benchmarking Audiovisual Human Speech Understanding in Multimodal Large Language Models](see_hear_and_understand_benchmarking_audiovisual_human_speech_understanding_in_mul.md)
- [\[CVPR 2026\] Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models](video-only_tom_enhancing_theory_of_mind_in_multimodal_large_language_models.md)
- [\[CVPR 2026\] Predictive Regularization Against Visual Representation Degradation in Multimodal Large Language Models](predictive_regularization_against_visual_representation_degradation_in_multimoda.md)
- [\[CVPR 2026\] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)
- [\[ACL 2026\] "I See What You Did There": Can Large Vision-Language Models Understand Multimodal Puns?](../../ACL2026/multimodal_vlm/i_see_what_you_did_there_can_large_vision-language_models_understand_multimodal_.md)

</div>

<!-- RELATED:END -->
