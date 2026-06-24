---
title: >-
  [Paper Note] Perception Tokens Enhance Visual Reasoning in Multimodal Language Models
description: >-
  [CVPR 2025][3D Vision][Perception tokens] This paper proposes Perception Tokens, a method to encode intermediate visual representations (e.g., depth maps, bounding boxes) into auxiliary reasoning tokens. This enables multimodal language models to enhance visual reasoning capabilities by generating perception tokens as intermediate steps, analogous to textual chain-of-thought.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Perception tokens"
  - "Multimodal Language Models"
  - "chain-of-thought"
  - "depth estimation"
  - "object counting"
date: 2026-05-08
content_hash: 356296e292abd12b
---

# Perception Tokens Enhance Visual Reasoning in Multimodal Language Models

**Conference**: CVPR 2025  
**arXiv**: [2412.03548](https://arxiv.org/abs/2412.03548)  
**Code**: None (will be released on the project page)  
**Area**: 3D Vision / Multimodal  
**Keywords**: Perception tokens, Multimodal Language Models, chain-of-thought, depth estimation, object counting

## TL;DR

This paper proposes Perception Tokens, a method to encode intermediate visual representations (e.g., depth maps, bounding boxes) into auxiliary reasoning tokens. This enables multimodal language models to enhance visual reasoning capabilities by generating perception tokens as intermediate steps, analogous to textual chain-of-thought.

## Background & Motivation

**Background**: Multimodal language models (MLMs) like LLaVA perform exceptionally well on high-level vision-language tasks, but still struggle with tasks requiring fundamental visual perception capabilities (such as depth reasoning and object counting). Task-specific vision models perform better on these tasks, but MLMs cannot natively generate depth maps or detection bounding boxes to assist their reasoning.

**Limitations of Prior Work**: (1) Directly fine-tuning MLMs on perception tasks yields limited performance and poor generalization; (2) calling external visual tools (e.g., depth estimators, detectors) incurs additional computational and memory overhead, and cascaded multi-model pipelines are prone to error accumulation; (3) the vocabulary of MLMs contains only text and CLIP image tokens, failing to represent low-to-mid-level visual features like depth and segmentation.

**Key Challenge**: Visual reasoning requires intermediate visual representations (e.g., depth maps) to support the reasoning stage, but the token space of MLMs is restricted to language tokens—natural language cannot accurately describe pixel-level depth relations or precise physical locations.

**Goal**: Extend the token space of MLMs to introduce auxiliary perception tokens, enabling models to generate and utilize visual perception representations during reasoning.

**Key Insight**: Analogous to chain-of-thought (CoT) reasoning in language models—where textual CoT supports reasoning via intermediate text steps—visual tasks can be aided by generating intermediate visual representations encoded as tokens.

**Core Idea**: Use VQVAE to tokenize visual representations such as depth maps into discrete tokens incorporated into the MLM vocabulary. Train the MLM to first generate these perception tokens as intermediate reasoning steps when answering visual questions (e.g., "The depth map is <<<perception tokens>>>, therefore point D is the closest"), and then derive the final answer based on these tokens.

## Method

### Overall Architecture

The Aurora framework extends the vocabulary of LLaVA as $V' = V \cup V_{\text{aux}}$. First, a VQVAE is used to tokenize depth maps (pixel-level representation) or directly encode bounding boxes (structural representation) into auxiliary tokens. Then, a curriculum learning strategy is used to train the MLM: starting from simple token prediction tasks and progressively transitioning to chain-of-thought visual reasoning with these tokens.

### Key Designs

1. **Tokenization of Perception Tokens**:

    - **Function**: Standardizes intermediate visual representations into discrete tokens that MLMs can generate and process.
    - **Mechanism**: For pixel-level representations (depth maps, segmentation masks), a VQVAE/VQGAN is used to encode them into discrete codebook indices as tokens. For structural representations (bounding boxes, coordinates), tokens are directly defined based on the domain range (e.g., coordinate ranges from 0 to the maximum image width/height). All tokens are integrated into $V_{\text{aux}}$ to form the extended vocabulary.
    - **Design Motivation**: A unified tokenization space allows different types of visual representations to be seamlessly handled within the same autoregressive framework without modifying the model architecture.

2. **Expert-to-Generalist Distillation + Reconstruction Loss**:

    - **Function**: Trains the MLM to generate accurate auxiliary tokens.
    - **Mechanism**: A pre-trained task-specific model (such as a depth estimator) provides the target distribution $q_i$. The MLM is trained using a cross-entropy distillation loss $\ell_{dist} = \min_M (-\sum_i q_i \log p_{M(i)})$ to align target auxiliary token predictions. Simultaneously, a lightweight decoder $g$ is introduced to map tokens back to the feature space, utilizing a reconstruction loss $\ell_{rec} = \|g(t) - f\|_2^2$ to enhance token interpretability and prediction accuracy.
    - **Design Motivation**: Distillation ensures that the generated tokens are semantically consistent with expert models, while reconstruction ensures that the tokens retain high fidelity when decoded back to the original representation. Combining both prevents the degradation of token prediction.

3. **Curriculum Learning + Progressive CoT**:

    - **Function**: Avoids catastrophic forgetting and incrementally builds multi-step reasoning capabilities.
    - **Mechanism**: Defining task difficulties $d_1 < d_2 < \cdots < d_T$, a temperature-annealed Softmax sampling probability $p(d_t, s) = \exp(-d_t/\tau(s)) / \sum_i \exp(-d_i/\tau(s))$ controls the training progress, where $\tau(s) = \tau_0 / (1 + \lambda \cdot s/S)$ gradually decreases with training steps. Three data subsets are used: (a) atomic tasks: learning to generate auxiliary tokens; (b) CoT data: generating tokens first, then answering the question; (c) direct annotation: answering directly without generating tokens. Both reasoning styles are sequentially presented on the same image.
    - **Design Motivation**: Executing fixed-data mixture training directly leads to a trade-off between token prediction accuracy and reasoning capability. Curriculum learning enables the model to first master the basics (token generation) before gradually learning complex reasoning, thereby effectively preventing catastrophic forgetting.

### Loss & Training

Auxiliary token prediction is trained by combining distillation loss and reconstruction loss. Constrained decoding (restricting sampling to only auxiliary tokens) and information bottlenecks (truncating the CoT chain to retain only auxiliary tokens) are applied to force the model to rely on perception tokens for reasoning. Implemented based on LLaVA 1.5 13B.

## Key Experimental Results

### Main Results

Relative depth estimation (Accuracy %):

| Method | BLINK 2-Point | HardBLINK 3-Point | HardBLINK 4-Point | HardBLINK 5-Point | Average |
|---|---|---|---|---|---|
| LLaVA 1.5 13B | 54.0 | 35.5 | 37.9 | 29.0 | 39.1 |
| Fine-tuned LLaVA | 68.5 | 58.9 | 52.4 | 41.1 | 55.2 |
| GPT-4o | 53.2 | 58.9 | 50.0 | 36.3 | 49.6 |
| **LLaVA-Aurora** | **64.5** | **66.9** | **60.5** | **54.8** | **61.6** |

Object counting (Accuracy %):

| Method | BLINK | CVBench | SEED-Bench |
|---|---|---|---|
| LLaVA 1.5 13B | 34.7 | 43.3 | 54.2 |
| Fine-tuned LLaVA | 35.2 | 48.5 | 57.5 |
| **LLaVA-Aurora** | **45.5** | **54.6** | **62.5** |

### Ablation Study

| Configuration | BLINK Depth | BLINK Count | Explanation |
|---|---|---|---|
| Baseline (No perception token) | 39.1 | 34.7 | Original LLaVA |
| Fine-tune only | 55.2 | 35.2 | Limited improvement |
| Token prediction only (No CoT) | Lower than full | Lower than full | Tokens not exploited for reasoning |
| **Full Aurora** | **61.6** | **45.5** | Perception tokens + CoT |

### Key Findings

- LLaVA-Aurora achieves an average improvement of +6.4% in depth reasoning (vs fine-tune) and a +13.7% improvement on the more challenging 5-point setup.
- Consistently improves counting performance across three benchmarks (+10.8 / +11.3 / +8.3 percentage points).
- Perception tokens demonstrate a more pronounced advantage as task difficulty increases—simple tasks may not require intermediate reasoning.
- The curriculum learning strategy is critical for avoiding catastrophic forgetting; performance drops significantly without it.
- Even without external tools, end-to-end perception token reasoning outperforms GPT-4 Turbo + Tool.

## Highlights & Insights

1. **"Visual Chain-of-Thought" paradigm**: Extends CoT from language to vision modality, utilizing generated depth map tokens to assist in depth reasoning and bounding box tokens to assist in counting—this constitutes a novel reasoning paradigm.
2. **Unification of VQVAE tokenization**: Unifies various visual representations into discrete tokens that share the same autoregressive space with language tokens, elegantly resolving compatibility issues of multimodal representations.
3. **Curriculum learning strategy**: The temperature-annealed sampling strategy elegantly balances token learning and reasoning learning, serving as an effective solution for handling multi-task heterogeneous data.

## Limitations & Future Work

- Currently, the approach is only validated on two types of tasks (depth and counting) and has not been extended to other perception tasks like segmentation or pose estimation.
- VQVAE tokenization introduces information loss, which can affect fine-grained reasoning.
- Generating extra perception tokens during inference adds to the sequence length and inference time.
- Future work can extend this to video understanding, embodied AI, etc., which require richer intermediate representations.

## Related Work & Insights

- **vs Unified-IO**: Unified-IO can generate visual outputs but cannot reason over its own generations; the core of Aurora is enabling the model to reason upon its own generated intermediate visual tokens.
- **vs LISA**: LISA generates segmentation tokens for grounding; Aurora's perception tokens are general-purpose intermediate steps designed for reasoning.
- **vs Tool-using MLMs**: Models like Visual ChatGPT invoke external tools; Aurora internalizes tool capacities as tokens, eliminating the need for extra models.
- **Insight**: The reasoning ability of MLMs is constrained by their token space. Extending the token space is an effective path to enhance reasoning capacity.

## Rating

- **Novelty**: 8/10 — The "Visual CoT" concept is novel, and encoding perceptual capabilities into reasoning tokens is pioneering.
- **Experimental Thoroughness**: 7/10 — Validated across multiple benchmarks, but limited to depth and counting tasks.
- **Writing Quality**: 8/10 — Clear framework description, with intuitive analogies to language CoT.
- **Value**: 8/10 — Opens a new direction for visual reasoning in MLMs, as the perception token framework is highly extensible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] InteractVLM: 3D Interaction Reasoning from 2D Foundational Models](interactvlm_3d_interaction_reasoning_from_2d_foundational_models.md)
- [\[CVPR 2025\] DepthCues: Evaluating Monocular Depth Perception in Large Vision Models](depthcues_evaluating_monocular_depth_perception_in_large_vision_models.md)
- [\[ICCV 2025\] GeoProg3D: Compositional Visual Reasoning for City-Scale 3D Language Fields](../../ICCV2025/3d_vision/geoprog3d_compositional_visual_reasoning_for_city-scale_3d_language_fields.md)
- [\[CVPR 2025\] Empowering Large Language Models with 3D Situation Awareness](empowering_large_language_models_with_3d_situation_awareness.md)
- [\[CVPR 2025\] Feat2GS: Probing Visual Foundation Models with Gaussian Splatting](feat2gs_probing_visual_foundation_models_with_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
