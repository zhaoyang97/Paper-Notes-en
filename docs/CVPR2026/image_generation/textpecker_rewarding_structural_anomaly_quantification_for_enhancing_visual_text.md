---
title: >-
  [Paper Note] TextPecker: Rewarding Structural Anomaly Quantification for Enhancing Visual Text Rendering
description: >-
  [CVPR 2026][Image Generation][visual text rendering] This paper proposes TextPecker—a plug-and-play structural anomaly-aware RL strategy that constructs a character-level structural anomaly annotation dataset to train a structure-aware recognizer, replacing the noisy reward signals of conventional OCR. By jointly optimizing semantic alignment and structural fidelity, TextPecker achieves significant improvements in visual text rendering quality across multiple text-to-image models (FLUX, SD3.5, Qwen-Image).
tags:
  - CVPR 2026
  - Image Generation
  - visual text rendering
  - structural anomaly
  - reinforcement-learning
  - reward model
  - OCR
date: 2026-05-08
content_hash: efe0f18d16a9f535
---

# TextPecker: Rewarding Structural Anomaly Quantification for Enhancing Visual Text Rendering

**Conference**: CVPR 2026  
**arXiv**: [2602.20903](https://arxiv.org/abs/2602.20903)  
**Code**: [GitHub](https://github.com/CIawevy/TextPecker)  
**Area**: Image Generation  
**Keywords**: visual text rendering, structural anomaly, reinforcement-learning, reward model, OCR

## TL;DR

This paper proposes TextPecker—a plug-and-play structural anomaly-aware RL strategy that constructs a character-level structural anomaly annotation dataset to train a structure-aware recognizer, replacing the noisy reward signals of conventional OCR. By jointly optimizing semantic alignment and structural fidelity, TextPecker achieves significant improvements in visual text rendering quality across multiple text-to-image models (FLUX, SD3.5, Qwen-Image).

## Background & Motivation

**Visual Text Rendering (VTR) remains a key challenge in T2I generation**: Even state-of-the-art models (e.g., FLUX, GPT-4o, BAGEL) frequently produce structural anomalies such as distortion, blurring, misalignment, and missing characters.

**OCR/MLLM evaluators have fundamental limitations**: Existing evaluation and RL optimization pipelines rely on OCR models or MLLMs to recognize generated text and compute edit-distance rewards. However, these models cannot perceive fine-grained structural anomalies, manifesting as two failure modes: (a) **Misinterpretation**: over-reliance on linguistic priors to "correct" structural defects, ignoring glyph-level flaws such as missing or misplaced strokes; (b) **Invisibility**: directly ignoring severely blurred or distorted regions as if they do not exist.

**Evaluator blind spots lead to misleading rewards**: The "auto-correction" behavior of OCR reduces the edit distance $N_e$ and artificially inflates the reward score $S$, causing RL optimization to deviate from the correct direction. Even highly optimized models such as Qwen-Image and Seedream4.0 still struggle to render structurally faithful text.

**Scarcity of structural anomaly annotation data**: Training data with character-level structural anomaly annotations is scarce, particularly for Chinese characters, where the two-dimensional spatial composition and 8,000+ character inventory introduce combinatorial explosion.

## Method

### Overall Architecture

TextPecker adopts the GRPO (Group Relative Policy Optimization) framework, with the core improvement being the replacement of OCR rewards with a structure-aware composite reward. The pipeline is as follows:
1. Sample $G$ candidate outputs $\{o_i\}_{i=1}^G$ from the reference policy model $\pi_{\theta_{\text{ref}}}$
2. The structure-aware recognizer extracts fine-grained generated text and flags structurally anomalous characters
3. Compute the joint reward $\mathcal{R}_i$ (semantic alignment + structural quality)
4. Normalize into group-relative advantages $A_i$ and optimize the policy model $\pi_\theta$ under KL divergence constraints

### Key Designs 1: Structural Quality Score $\mathcal{S}_Q$

- **Function**: Quantifies the proportion of structurally anomalous characters in generated text, and amplifies the penalty for rare but severe defects via a scaling factor.
- **Formula**:

$$\mathcal{S}_Q = \text{clip}\left(1 - \omega \frac{N_a}{N_P},\ 0,\ 1\right)$$

where $N_P$ is the total number of characters in the generated text, $N_a$ is the number of characters flagged as structurally anomalous, and $\omega > 1$ is a scaling factor ($\omega=5$ in experiments).
- **Design Motivation**: For strong generators, structural errors are rare but visually conspicuous. The factor $\omega$ amplifies the penalty for infrequent errors, preventing the policy from receiving high scores despite sporadic defects.

### Key Designs 2: Semantic Alignment Score $\mathcal{S}_E$

- **Function**: Performs Hungarian matching at the word level, computes the normalized edit distance between target and generated text, and penalizes unmatched words.
- **Formula**:

$$\mathcal{S}_E = 1 - \frac{\sum_{(t_i, p_j) \in \mathcal{M}} \text{NED}(t_i, p_j) + \text{Penalty}(\mathcal{T}, \mathcal{P}, \mathcal{M})}{\max(|\mathcal{T}|, |\mathcal{P}|)}$$

where $\mathcal{T}$ and $\mathcal{P}$ denote the target and generated word sets respectively, $\mathcal{M}$ is the Hungarian optimal matching based on NED, and $\text{Penalty}(\cdot)$ counts unmatched words.
- **Design Motivation**: The word order in generated text may differ from the prompt, necessitating word-level matching rather than simple string comparison; penalizing surplus or missing words ensures comprehensive evaluation.

### Key Designs 3: Composite Reward $\mathcal{R}$

$$\mathcal{R} = w_E \mathcal{S}_E + w_Q \mathcal{S}_Q, \quad w_E + w_Q = 1$$

In experiments, $w_E = w_Q = 0.5$, jointly optimizing semantic accuracy and structural fidelity.

### Key Designs 4: Structure-Aware Data Construction

A three-step pipeline is used to construct a character-level structural anomaly annotation dataset (1.4M samples in total):

1. **Text image generation**: Multiple T2I models are used (AnyText, SD1.5, SD3.5, FLUX, Seedream3.0, Qwen-Image for English; Cogview4, Kolors, Seedream3.0, Qwen-Image for Chinese) to generate large-scale text images. Chinese prompts are sampled from WanJuan1.0, with font style descriptions generated by Qwen3-235B.
2. **Structural anomaly annotation**: OCR is first applied to obtain preliminary recognition results; annotators then label structural defects character by character (blurring, distortion, missing strokes, extraneous strokes), with severely merged characters marked using placeholders.
3. **Synthetic data augmentation**: A stroke-editing synthesis engine is introduced to perform three stroke-level operations on Chinese characters:
    - **Stroke deletion**: Remove a subset of strokes
    - **Stroke swapping**: Swap positions of disjoint stroke pairs (aligned by centroid)
    - **Stroke insertion**: Sample strokes from other characters and insert them

Both synthetic anomalous and normal characters are rendered onto diverse backgrounds and layouts using the SynthTIGER rendering engine.

| Data Type | Level | Samples | Proportion |
|-----------|-------|---------|------------|
| Human-annotated | Box | 559.6K | 39.32% |
| Human-annotated | Image | 131.1K | 9.21% |
| Synthetic anomalous text | Box | 452.5K | 31.80% |
| Synthetic anomalous text | Image | 100.0K | 7.03% |
| Synthetic normal text | Box | 150.0K | 10.54% |
| Synthetic normal text | Image | 30.0K | 2.10% |
| **Total** | – | **1.4M** | 100% |

### RL Optimization Backbone

- Flow-GRPO is adopted to extend GRPO to the rectified-flow setting, converting deterministic dynamics into stochastic differential equations by injecting randomness:

$$dx_t = \left(v_t + \frac{\sigma_t^2}{2t}(x_t + (1-t)v_t)\right)dt + \sigma_t\,dw_t$$

- Recognizer backbones: Qwen3-VL-8B and InternVL3-8B, supporting bounding-box-level input, with full-parameter fine-tuning for 2 epochs.

## Key Experimental Results

### Text Structural Anomaly Perception (TSAP) vs. Conventional Text Recognition (CTR)

| Method | English TSAP F1 | English CTR Recall | Chinese TSAP F1 | Chinese CTR Recall |
|--------|----------------|-------------------|----------------|-------------------|
| PP-OCRv5 | 0.000 | 0.720 | 0.024 | 0.921 |
| GOT-OCR-2.0 | 0.000 | 0.610 | 0.008 | 0.853 |
| GPT-5 | 0.170 | 0.556 | 0.226 | 0.758 |
| Qwen3-VL-8B | 0.032 | 0.807 | 0.017 | 0.943 |
| InternVL3-8B | 0.183 | 0.759 | 0.153 | 0.927 |
| **TextPecker (InternVL3)** | **0.870** | **0.944** | **0.927** | **0.962** |
| **TextPecker (Qwen3-VL)** | **0.862** | **0.918** | **0.925** | **0.972** |

- Existing OCR and MLLM approaches almost completely fail on TSAP (F1 ≈ 0), while TextPecker achieves F1 above 0.87.
- TextPecker simultaneously improves conventional text recognition, with CTR Recall exceeding 0.94.

### VTR RL Optimization

- **FLUX**: Compared to baseline, Sem. +38.3%, Qua. +31.6%; compared to OCR reward, GenTextEval Sem. +11.7%.
- **Qwen-Image Chinese rendering**: Semantic alignment +8.7%, structural fidelity +4.0%, achieving new SOTA.
- **SD3.5-M**: Qua. improves from 0.671 to 0.959, Sem. from 0.265 to 0.506.

## Ablation Study

- Removing synthetic data augmentation leads to a significant drop in Chinese recognition performance, validating the necessity of the stroke-editing engine for covering Chinese structural anomalies.
- Training with human-annotated data alone results in poor generalization to unseen anomaly types.
- $\omega=5$ achieves the optimal balance in the scaling factor ablation.

## Highlights & Insights

**Strengths**:
- First systematic identification of structural anomaly perception as a critical bottleneck in VTR, providing a new perspective for both evaluation and optimization
- Plug-and-play design requires no modification to the generator architecture and is applicable to arbitrary T2I models
- The stroke-editing synthesis engine elegantly addresses the combinatorial explosion of Chinese character structural anomalies
- Achieves significant gains even on the already highly optimized Qwen-Image

**Limitations**:
- High annotation cost (559.6K box-level annotations)
- The structure-aware recognizer is based on an 8B-parameter VLM, incurring substantial inference overhead
- Validation is primarily conducted on Chinese and English; other writing systems (e.g., Arabic, Japanese kana) are not covered

## Rating

⭐⭐⭐⭐

This paper conducts a thorough analysis of and effectively addresses a critical pain point in VTR (the structural blind spot of OCR evaluators). Starting from the finding that "OCR and MLLM achieve F1 ≈ 0 on TSAP," the complete pipeline from dataset construction → recognizer training → composite reward design → RL optimization is highly coherent. The design of the stroke-editing synthesis engine reflects a deep understanding of Chinese character properties. The fact that gains of +8.7% semantic and +4.0% structural quality are still achievable on the already highly optimized Qwen-Image demonstrates the practical value of the proposed method. Limitations include high annotation costs and substantial inference overhead; nevertheless, as a contribution that fills a critical evaluation gap, the work is of outstanding significance.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Enhancing Spatial Understanding in Image Generation via Reward Modeling](enhancing_spatial_understanding_in_image_generation_via_reward_modeling.md)
- [\[CVPR 2026\] RenderFlow: Single-Step Neural Rendering via Flow Matching](renderflow_single-step_neural_rendering_via_flow_matching.md)
- [\[CVPR 2026\] Learning to Generate via Understanding: Understanding-Driven Intrinsic Rewarding for Unified Multimodal Models](learning_to_generate_via_understanding_understanding-driven_intrinsic_rewarding_.md)
- [\[AAAI 2026\] AnoStyler: Text-Driven Localized Anomaly Generation via Lightweight Style Transfer](../../AAAI2026/image_generation/anostyler_text-driven_localized_anomaly_generation_via_light.md)
- [\[CVPR 2026\] Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning](spatial-ssrl_enhancing_spatial_understanding_via_self-supervised_reinforcement_l.md)

<!-- RELATED:END -->
