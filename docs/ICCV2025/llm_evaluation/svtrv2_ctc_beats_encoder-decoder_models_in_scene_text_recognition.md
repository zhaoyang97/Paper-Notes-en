---
title: >-
  [Paper Note] SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition
description: >-
  [ICCV 2025][LLM Evaluation][Scene Text Recognition] SVTRv2 is proposed with three key designs — Multi-Size Resize (MSR), Feature Rearrangement Module (FRM), and Semantic Guidance Module (SGM) — enabling a CTC-based model to comprehensively outperform encoder-decoder methods across multi-scene benchmarks for the first time, while retaining inference speed advantages.
tags:
  - "ICCV 2025"
  - "LLM Evaluation"
  - "Scene Text Recognition"
  - "CTC"
  - "Irregular Text"
  - "Semantic Guidance"
  - "Multi-Size Resize"
date: 2026-05-08
content_hash: 63209d34a61b49c9
---

# SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition

**Conference**: ICCV 2025
**arXiv**: [2411.15858](https://arxiv.org/abs/2411.15858)  
**Code**: [https://github.com/Topdu/OpenOCR](https://github.com/Topdu/OpenOCR)  
**Area**: LLM Evaluation
**Keywords**: Scene Text Recognition, CTC, Irregular Text, Semantic Guidance, Multi-Size Resize

## TL;DR

SVTRv2 is proposed with three key designs — Multi-Size Resize (MSR), Feature Rearrangement Module (FRM), and Semantic Guidance Module (SGM) — enabling a CTC-based model to comprehensively outperform encoder-decoder methods across multi-scene benchmarks for the first time, while retaining inference speed advantages.

## Background & Motivation

Scene text recognition (STR) methods fall into two main categories:

**CTC-based**: Vision-only models with a linear CTC classifier. Simple in structure and fast at inference, they dominate commercial OCR systems but underperform on irregular text.

**Encoder-Decoder (EDTR)**: Leverage attention-based decoders to handle multimodal cues (visual, linguistic, positional). Higher accuracy but slower speed.

The limitations of CTC models stem from two core issues:
- **Difficulty handling irregular text**: CTC alignment assumes approximately horizontal text layout; curved, rotated, or perspective text violates this assumption. Existing methods uniformly resize images to a fixed size (e.g., 32×128), causing severe distortion for low aspect-ratio text.
- **Lack of linguistic context modeling**: CTC directly classifies visual features without encoding language information. In occluded or low-quality scenarios, the absence of language priors leads to high error rates.

The paper aims to **equip CTC models with the ability to handle irregular text and model linguistic context**, while maintaining a lightweight inference architecture.

## Method

### Overall Architecture

SVTRv2 = Multi-Size Resize (MSR) + three-stage visual feature extraction + Feature Rearrangement Module (FRM) + Semantic Guidance Module (SGM, training-only). At inference, SGM is discarded, and the model remains a pure CTC architecture.

### Key Designs

1. **Multi-Size Resize Strategy (MSR)**
   Images are bucketed into 4 groups based on aspect ratio $R = W/H$:
    - $R < 1.5$: resize to [64, 64]
    - $1.5 \leq R < 2.5$: resize to [48, 96]
    - $2.5 \leq R < 3.5$: resize to [40, 112]
    - $R \geq 3.5$: resize to [32, $\lfloor R \rfloor \times 32$]

   The first three buckets use fixed sizes for batch training; the fourth uses dynamic width to handle long text. **Design Motivation**: Avoid stretching low aspect-ratio images (e.g., vertical text) into fixed-width images, eliminating unnecessary distortion. Experiments show MSR improves accuracy by **15.3%** on the $R_1$ bucket compared to fixed 32×128.

2. **Feature Rearrangement Module (FRM)**
   Rearranges 2D visual features $\mathbf{F} \in \mathbb{R}^{(H/8 \times W/4) \times D_2}$ into a 1D sequence conforming to CTC alignment, in two steps:
    - **Horizontal rearrangement**: Self-attention over each row of features to learn a horizontal rearrangement matrix $\mathbf{M}_i^h$, aligning feature order with text reading direction.
    - **Vertical rearrangement**: Selecting tokens interact with column features via cross-attention to learn a vertical rearrangement matrix $\mathbf{M}_j^v$, selecting the most relevant features across rows.

   **Design Motivation**: Characters in rotated text do not follow horizontal reading order. The two-step rearrangement (horizontal then vertical) makes the module sensitive to text orientation, effectively addressing CTC alignment challenges. Experiments show the combined two-step approach improves accuracy by 2.46% on the Multi-Oriented subset.

3. **Semantic Guidance Module (SGM)**
   During training, contextual character strings from ground-truth labels are used to guide the visual model in encoding linguistic information. Specifically, for target character $c_i$, its left $l_s$ and right $l_s$ neighboring characters are taken as context, encoded into hidden representation $\mathbf{Q}_i^l$, and then cross-attended with visual features to produce attention map $\mathbf{A}_i^l$ for classifying $c_i$. **Mechanism**: The attention map can correctly focus on the target character position only when the visual model has incorporated contextual information into the target character's visual features; the training signal backpropagates to encourage the visual model to learn linguistic context. At inference, **SGM is completely discarded**, adding zero computational overhead.

### Loss & Training

$$\mathcal{L} = \lambda_1 \mathcal{L}_{ctc} + \lambda_2 \mathcal{L}_{sgm}$$

- $\mathcal{L}_{ctc}$: Standard CTC loss, weight $\lambda_1 = 0.1$
- $\mathcal{L}_{sgm}$: Mean cross-entropy of left and right context predictions, weight $\lambda_2 = 1$
- Training proceeds in two stages: first without SGM, then with SGM added

## Key Experimental Results

### Main Results

**Union14M-Benchmark subset accuracy + speed (trained on U14M-Filter)**

| Method | Type | Curve | MO | Artistic | Com Avg | U14M Avg | FPS |
|------|------|-------|------|----------|---------|---------|-----|
| SVTR-B | CTC | 76.2 | 44.5 | 67.8 | 94.58 | 71.17 | 161 |
| MAERec | EDTR | 89.1 | 87.1 | 79.0 | 96.36 | 85.17 | 17.1 |
| PARSeq | EDTR | 87.6 | 88.8 | 76.5 | 96.40 | 84.26 | 52.6 |
| CPPD | EDTR | 86.2 | 78.7 | 76.5 | 96.40 | 81.91 | 125 |
| **SVTRv2-B** | **CTC** | **90.6** | **89.0** | **79.3** | **96.57** | **86.14** | **143** |

SVTRv2-B outperforms all EDTR methods on U14M (surpassing MAERec by +0.97%) with an inference speed of 143 FPS, **8× faster** than MAERec. Compared to the previous SVTR, improvements on Curve and Multi-Oriented reach **14.4% and 44.5%**, respectively.

### Ablation Study

**MSR and FRM ablation (U14M Curve + MO)**

| Configuration | Curve | MO | Com Avg | U14M Avg |
|------|-------|------|---------|---------|
| No MSR, No FRM (Fixed 32×128) | 82.89 | 65.59 | 95.28 | 77.78 |
| + MSR | 87.35 | 83.73 | 95.44 | 82.22 |
| + FRM (H+V) | 88.05 | 85.76 | 95.98 | 82.94 |
| + MSR + FRM | **88.17** | **86.19** | **96.16** | **83.86** |
| + MSR + FRM + SGM | **90.64** | **89.04** | **96.57** | **86.14** |

MSR contributes the most (+4.44 U14M Avg); FRM yields significant gains on MO (+2.46); SGM shows the most prominent improvement in occlusion scenarios (OST +5.11).

### Key Findings

- SGM delivers a 5.11% absolute improvement on occluded scenarios (OST Avg), far exceeding alternatives such as GTC, ABINet, and VisionLAN.
- SVTRv2-B also achieves state-of-the-art on Chinese text recognition (83.31 Avg) and supports long text (SceneL>25 accuracy 52.8%).
- With pretraining, SVTRv2 reaches 97.83% on Common Benchmarks, surpassing large-scale pretrained methods such as CLIP4STR while using only 14% of their parameters.

## Highlights & Insights

- **CTC comprehensively defeats EDTR for the first time**: Winning on both speed and accuracy, this overturns the prevailing belief in the STR community that "CTC is inferior to EDTR."
- The "use during training, discard at inference" design of SGM is particularly elegant: training signals indirectly compel the visual model to learn linguistic context, improving accuracy at zero inference cost.
- The simplicity of MSR is impressive: merely bucketing images by aspect ratio and selecting appropriate sizes eliminates most irregular text issues.
- Additional contribution: A deduplicated U14M-Filter dataset is constructed and 24 methods are retrained for comparison, providing a fair and reliable new benchmark for STR.

## Limitations & Future Work

- SGM requires character-level annotations; extension to word-level or line-level annotations requires further adaptation.
- The bucket boundaries in MSR are manually designed (4 fixed sizes); adaptively learned sizes may yield better results.
- The maximum training length is limited to 25 characters (English); very long text remains a limitation.
- FRM introduces additional attention computation, which may need to be simplified for extremely latency-sensitive applications.

## Related Work & Insights

- SVTR [IJCAI 2022] is the predecessor of this work; the main improvements consist of the three modules MSR, FRM, and SGM.
- GTC [AAAI 2020] first proposed using an auxiliary decoder to guide CTC training; SGM can be viewed as a more refined semantic variant of this idea.
- The approach presented in this paper is transferable to other sequence recognition tasks (e.g., speech recognition, gesture recognition, and other CTC-based settings).

## Rating

- Novelty: ⭐⭐⭐⭐ (Each of the three modules is innovative; collectively, CTC comprehensively surpasses EDTR for the first time)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (6+ benchmarks, 24 methods retrained for comparison, covering Chinese, long text, occlusion, and pretraining scenarios)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, thorough experiments)
- Value: ⭐⭐⭐⭐⭐ (Highly significant for the OCR community, with a fair benchmark and open-source code)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Fixed Psychological Personas: State Beats Trait, but Language Models are State-Blind](../../ACL2026/llm_evaluation/beyond_fixed_psychological_personas_state_beats_trait_but_language_models_are_st.md)
- [\[ECCV 2024\] EvSign: Sign Language Recognition and Translation with Streaming Events](../../ECCV2024/llm_evaluation/evsign_sign_language_recognition_and_translation_with_streaming_events.md)
- [\[ICLR 2026\] Culture in Action: Evaluating Text-to-Image Models through Social Activities](../../ICLR2026/llm_evaluation/culture_in_action_evaluating_text-to-image_models_through_social_activities.md)
- [\[ACL 2026\] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models](../../ACL2026/llm_evaluation/attribution_citation_and_quotation_a_survey_of_evidence-based_text_generation_wi.md)
- [\[ACL 2026\] Gated Tree Cross-Attention for Checkpoint-Compatible Syntax Injection in Decoder-Only LLMs](../../ACL2026/llm_evaluation/gated_tree_cross-attention_for_checkpoint-compatible_syntax_injection_in_decoder.md)

</div>

<!-- RELATED:END -->
