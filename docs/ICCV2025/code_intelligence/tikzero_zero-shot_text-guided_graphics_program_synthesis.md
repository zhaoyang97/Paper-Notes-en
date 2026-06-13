---
title: >-
  [Paper Note] TikZero: Zero-Shot Text-Guided Graphics Program Synthesis
description: >-
  [ICCV 2025 (Highlight)][Code Intelligence][Graphics Program Synthesis] This paper proposes TikZero, which decouples graphics program generation from text understanding by using image representations as an intermediate br…
tags:
  - "ICCV 2025 (Highlight)"
  - "Code Intelligence"
  - "Graphics Program Synthesis"
  - "TikZ"
  - "Zero-Shot"
  - "Text-Guided"
  - "Multimodal Language Model"
  - "Image Bridging"
date: 2026-05-08
content_hash: 716492808f35f93a
---

# TikZero: Zero-Shot Text-Guided Graphics Program Synthesis

**Conference**: ICCV 2025 (Highlight)
**arXiv**: [2503.11509](https://arxiv.org/abs/2503.11509)  
**Code**: [potamides/DeTikZify](https://github.com/potamides/DeTikZify)  
**Area**: Code Intelligence
**Keywords**: Graphics Program Synthesis, TikZ, Zero-Shot, Text-Guided, Multimodal Language Model, Image Bridging

## TL;DR

This paper proposes TikZero, which decouples graphics program generation from text understanding by using image representations as an intermediate bridge, enabling zero-shot text-guided TikZ graphics program synthesis without text-aligned training data. TikZero substantially outperforms baseline methods, and its end-to-end fine-tuned variant TikZero+ matches or surpasses large commercial models such as GPT-4o.

## Background & Motivation

**Automatically generating scientific diagrams from text descriptions** is a highly desirable capability. Such diagrams require high geometric precision and editability, necessitating representation as programs in graphics languages such as TikZ rather than raster images. However, the field faces a core data bottleneck:

**Severe scarcity of aligned training data**: Ideal training data consists of paired (text description, TikZ program) examples, which are extremely difficult to obtain at scale. Manually annotating TikZ code with descriptions is prohibitively costly, and existing datasets remain small.

**Abundant but disjoint unaligned data**: Large collections of unannotated TikZ programs (e.g., extracted from arXiv papers) and large collections of captioned raster images (e.g., natural image datasets) exist independently, but without correspondence. Conventional end-to-end methods cannot directly exploit this disjoint data.

**Limitations of Prior Work**:
- **End-to-end models** (e.g., DeTikZify) require text–program aligned data for training and are thus limited by data scale.
- **General-purpose large models** (e.g., GPT-4o) are powerful but entail enormous parameter counts, high inference costs, and are not specialized for graphics program synthesis.

**Core Idea**: The text-to-graphics-program process can be decomposed into two steps — text to image representation, then image representation to graphics program. Since both the "image → TikZ code" and "text → image" directions have abundant independent training data, they can be connected via image representations as an intermediate bridge.

## Method

### Mechanism: Image Representations as a Bridge

The core innovation of TikZero is a **decoupling** strategy that decomposes the "text → graphics program" task into two independently trainable subtasks:

- **Image → Graphics Program**: Trained on large collections of unannotated TikZ programs, using compiled rendered images as input conditions.
- **Text → Image Representation**: Trained on large collections of captioned raster images, learning to map text into an image embedding space.

At inference time, the two components are chained: a text description is first mapped into the image embedding space, and the graphics program generation model then produces TikZ code, achieving **zero-shot** text-guided graphics program synthesis.

### Architecture

TikZero is built upon DeTikZifyv2 (8B), a multimodal language model based on the Idefics3/LLaMA3 architecture specialized for image-to-TikZ code generation. TikZero introduces the following key components:

#### 1. Image Encoder and Graphics Program Decoder (from DeTikZifyv2)

The core pipeline of DeTikZifyv2:

- **Visual Encoder**: Encodes the input image into a sequence of visual tokens.
- **Cross-Modal Projection**: Projects visual tokens into the language model's embedding space.
- **LLM Decoder**: Based on LLaMA3-8B, autoregressively generates TikZ code tokens.

This model is trained on the DaTikZv2/v3 dataset (large collections of TikZ programs extracted from arXiv along with their compiled rendered images), providing strong image-to-code capability.

#### 2. Text–Image Adapter (Core Contribution of TikZero)

The key innovation of TikZero is a lightweight **adapter module** that maps text embeddings into the image embedding space of DeTikZifyv2. The architecture is inspired by the cross-attention mechanisms in Flamingo and LLaMA 3.2-Vision:

- **Text Encoder**: A separate language model (e.g., LLaMA 3.2-1B) encodes the text description.
- **Cross-Attention Layers**: Map text embeddings into the image embedding space via cross-attention, so that text-derived embeddings are distributionally close to real image embeddings.
- **Training Objective**: Minimize the distance between text embeddings and the corresponding image embeddings (supporting both cosine distance and MSE training variants).

The adapter contains approximately 0.4B parameters and can be plugged into DeTikZifyv2 without retraining the base model.

#### 3. TikZero+: End-to-End Fine-Tuning

Building on the zero-shot TikZero framework, TikZero+ further leverages the limited available text–program aligned data for end-to-end fine-tuning:

- The TikZero adapter and DeTikZifyv2 are merged into a unified 10B-parameter model.
- End-to-end training on aligned data enables joint optimization of both submodules.
- This allows the model to retain zero-shot generalization while further improving performance on annotated distributions.

### Training Data

TikZero leverages three types of data sources:

| Data Type | Source | Purpose |
|-----------|--------|---------|
| Unannotated TikZ programs | DaTikZv2/v3 (extracted from arXiv) | Train image → code model |
| Captioned raster images | General image–text datasets | Train text → image embedding adapter |
| Aligned text–TikZ pairs | Small annotated dataset | TikZero+ end-to-end fine-tuning |

### Inference Pipeline

1. The user inputs a text description (e.g., "A multi-layer perceptron with two hidden layers").
2. The text encoder and adapter map the description to an image embedding.
3. DeTikZifyv2 treats this embedding as a "virtual image input" and autoregressively generates TikZ code.
4. The TikZ code is compiled by LaTeX into a high-quality vector graphic.

## Key Experimental Results

### Main Results

- **Zero-shot setting**: TikZero (trained exclusively on non-aligned data) substantially outperforms baselines that can only train on aligned data.
- **Supervised setting**: TikZero+ (additionally fine-tuned on aligned data) matches or surpasses large commercial systems such as GPT-4o.
- **Model efficiency**: The TikZero adapter has only 0.4B parameters, and the full TikZero+ model has 10B parameters, far smaller than hundred-billion-scale models such as GPT-4o.

### Evaluation Metrics

Experiments employ a multi-dimensional evaluation framework:

- **Compilation Success Rate**: Whether the generated TikZ code compiles successfully.
- **Visual Similarity**: Pixel-level and semantic-level similarity between generated and reference graphics.
- **Semantic Fidelity**: Whether the generated graphic accurately conveys the semantic content of the text description.

### Comparison with Baselines

| Method | Data Requirement | Model Scale | Performance |
|--------|-----------------|-------------|-------------|
| Baseline (aligned data only) | Aligned data | 8B | Low |
| TikZero (zero-shot) | Non-aligned data | 8B + 0.4B adapter | Substantially better than baseline |
| TikZero+ | Non-aligned + aligned data | 10B | Matches/surpasses GPT-4o |
| GPT-4o | General pre-training data | >>100B | Strong but not specialized |

### Follow-up: DeTikZifyv2.5

Building on TikZero, the authors further train DeTikZifyv2.5 via **Reinforcement Learning from Self-Feedback (RLSF)**, achieving additional performance gains. The GRPO (Group Relative Policy Optimization) training script is also released as open source.

## Highlights & Insights

1. **Elegance of the decoupled training paradigm**: TikZero reframes the data scarcity problem as a bridging problem — using images as a junction between two abundant but disjoint data sources. This idea is not limited to TikZ and can generalize to any scenario where cross-domain aligned data is scarce but domain-specific data is plentiful (e.g., code generation, CAD modeling).
2. **Lightweight plug-and-play adapter**: A 0.4B adapter enables zero-shot text conditioning on an 8B base model without retraining the entire model, resulting in very low deployment cost.
3. **ICCV 2025 Highlight paper**: The work was selected as an ICCV 2025 highlight, reflecting strong reviewer recognition of its methodological novelty and experimental rigor.
4. **Comprehensive open-source ecosystem**: Code (GitHub 1.8k stars), model weights (HuggingFace), datasets, Web UI, and Colab Demo are all publicly released, ensuring strong reproducibility.
5. **Smooth transition from zero-shot to supervised**: The TikZero → TikZero+ progression demonstrates that decoupled training is effective not only in the zero-shot setting but also as initialization for end-to-end training, showing that the two paradigms are complementary.

## Limitations & Future Work

1. **Information loss through image bridging**: The two-step mapping of text → image embedding → code may introduce an information bottleneck — precise numerical information in text (e.g., coordinates, dimensions) may be lost when projected into the image embedding space.
2. **Limitations of the TikZ language**: Although TikZ is highly expressive, its user base is smaller than that of more general vector formats such as SVG, and its applications are primarily concentrated in academic publishing.
3. **Compilation dependency**: Generated TikZ code requires a full TeX Live environment to compile, increasing deployment complexity.
4. **Generalization boundaries of the adapter**: How the adapter performs on out-of-distribution text descriptions (e.g., highly abstract or non-scientific diagram descriptions) remains to be evaluated.
5. **Continued value of aligned data**: The improvement of TikZero+ over the purely zero-shot TikZero indicates that aligned data remains important; efficient acquisition of more high-quality aligned data (e.g., via automatic LLM annotation) is a direction worth exploring.

## Related Work & Insights

- **DeTikZify (NeurIPS 2024 Spotlight)**: The predecessor of TikZero, focusing on image-to-TikZ multimodal language models with MCTS-based iterative inference optimization.
- **AutomaTikZ**: An earlier work on automatic TikZ generation, upon which the DeTikZify series builds.
- **Flamingo / LLaMA 3.2-Vision**: Sources of architectural inspiration for TikZero's cross-attention adapter design.
- **Idefics3**: The base architecture for DeTikZifyv2.

**Implications for Future Research**:

- The image-as-modality-bridge idea can be extended to a broader range of cross-modal generation tasks (e.g., text → SVG, text → CAD, text → music).
- The paradigm of lightweight adapters for zero-shot cross-modal transfer provides a new approach to model reuse in low-resource settings.
- The two-stage strategy of decoupled training followed by end-to-end fine-tuning has broad applicability in scenarios where training data is only partially aligned.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Program Synthesis via Test-Time Transduction](../../NeurIPS2025/code_intelligence/program_synthesis_via_test-time_transduction.md)
- [\[AAAI 2026\] TAPA: Training-Free Adaptation of Programmatic Agents via LLM-Guided Program Synthesis in Dynamic Environments](../../AAAI2026/code_intelligence/tapas_are_free_training-free_adaptation_of_programmatic_agen.md)
- [\[NeurIPS 2025\] Once Upon an Input: Reasoning via Per-Instance Program Synthesis](../../NeurIPS2025/code_intelligence/once_upon_an_input_reasoning_via_per-instance_program_synthesis.md)
- [\[AAAI 2026\] Extracting Events Like Code: A Multi-Agent Programming Framework for Zero-Shot Event Extraction](../../AAAI2026/code_intelligence/extracting_events_like_code_a_multi-agent_programming_framework_for_zero-shot_ev.md)
- [\[NeurIPS 2025\] Searching Latent Program Spaces](../../NeurIPS2025/code_intelligence/searching_latent_program_spaces.md)

</div>

<!-- RELATED:END -->
