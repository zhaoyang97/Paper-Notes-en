---
title: >-
  [Paper Note] CAP: Evaluation of Persuasive and Creative Image Generation
description: >-
  [ICCV 2025][Image Generation][advertising image generation] This paper proposes three novel evaluation metrics (creativity, alignment, and persuasiveness) for the task of advertising image generation…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "advertising image generation"
  - "creativity evaluation"
  - "persuasiveness evaluation"
  - "text-image alignment"
  - "implicit message"
date: 2026-05-08
content_hash: a36e7e67bed5cece
---

# CAP: Evaluation of Persuasive and Creative Image Generation

**Conference**: ICCV 2025
**arXiv**: [2412.10426](https://arxiv.org/abs/2412.10426)
**Code**: [https://aysanaghazadeh.github.io/CAP/](https://aysanaghazadeh.github.io/CAP/)
**Area**: Image Generation
**Keywords**: advertising image generation, creativity evaluation, persuasiveness evaluation, text-image alignment, implicit message

## TL;DR
This paper proposes three novel evaluation metrics (creativity, alignment, and persuasiveness) for the task of advertising image generation, and leverages LLMs to expand implicit messages into explicit visual descriptions to improve T2I model performance on advertisement generation, achieving significantly higher agreement with human annotations than baselines such as CLIPScore.

## Background & Motivation
Advertising image generation is an underexplored sub-domain of text-to-image (T2I) generation. Effective advertisements must simultaneously satisfy three conditions: **clearly conveying a message, presenting it in a creative manner, and effectively influencing the audience**. However, while current T2I models (e.g., Stable Diffusion, DALL-E 3) can generate high-quality images from detailed explicit descriptions, they tend to produce images that are topically relevant but lack creativity and persuasiveness when given *implicit text prompts* — prompts that describe intent and message rather than directly specifying the objects to be depicted.

Existing evaluation metrics (FID, CLIPScore, VQAScore, etc.) focus solely on explicit text-image alignment and cannot measure creativity or persuasiveness. For instance, Fig. 1 shows that for a Gatorade advertisement, baseline T2I models generate images showing only a bottle and can, whereas human-created advertisements feature an athlete running alongside a slogan. Existing metrics fail to distinguish this difference in quality.

The key starting point is: **incorporating rhetorical persuasion theory (Aristotle's Pathos/Ethos/Logos) into computational evaluation**, and leveraging LLM reasoning to score advertising images across multiple dimensions of creativity and persuasiveness.

## Method

### Overall Architecture
Given an advertising message in action-reason format (e.g., "I should drink Gatorade because it would help me win"), the framework evaluates generated advertisement images along three dimensions: alignment (AIM), creativity ($C_{obj}$), and persuasiveness ($P_{comp+AIM}$). A generation strategy is also proposed that uses an LLM to expand implicit messages into explicit visual descriptions before feeding them into a T2I model.

### Key Designs

1. **AIM (Alignment of Image and Message)**:

    - Function: Evaluates the semantic alignment between a generated image and the implicit advertising message.
    - Mechanism:
      1. Use an MLLM (e.g., InternVL-V2-26B) to generate an image caption.
      2. Fine-tune an LLM (e.g., LLAMA3-8B) with CPO (Contrastive Preference Optimization) to infer action-reason statements from image captions.
      3. Compute semantic similarity between the inferred action/reason and the original message, then compute a weighted average: $AIM = \frac{Sim(A_{gen}, A_m) + \alpha \cdot Sim(R_{gen}, R_m)}{1+\alpha}$, where $\alpha=4$.
    - Design Motivation: Metrics such as CLIPScore cannot handle implicit text; fine-tuning an LLM enables more accurate inference of the deep intent behind an advertisement rather than simple object matching.

2. **$C_{obj}$ (Creativity Metric)**:

    - Function: Quantifies the degree of creativity in a generated image.
    - Mechanism: Creativity = high alignment + low object similarity (i.e., the image does not simply display the objects mentioned in the message).
    - Formula: $C_{obj} = \frac{AIM(I_{gen}, AR_m)}{\frac{1}{n}\sum_{obj} Sim(I_{gen}, obj) + 0.01}$
    - Design Motivation: Creative advertisements should convey a message in an atypical manner, rather than merely listing product images.

3. **$P_{comp+AIM}$ (Persuasiveness Metric)**:

    - Function: Evaluates the persuasive effect of an advertisement across multiple dimensions.
    - Mechanism: Combines seven rhetorical dimensions (audience targeting AU, benefit transfer B, appeal category AP, elaboration E, originality O, imagination I, and synthesis S) by prompting an LLM to score image captions on each dimension, averaging the scores, and combining the result with AIM.
    - Design Motivation: A single dimension cannot capture the complexity of persuasiveness; combining multiple dimensions resembles an ensemble model in which small errors from individual dimensions cancel out.

### Loss & Training
LLM fine-tuning for AIM uses a CPO (Contrastive Preference Optimization) loss. Training data consists of 250 images and 3,500 data points (captions paired with correct/incorrect action-reason pairs), with batch size = 4, lr = 5e-5, and 3,000 training steps.

## Key Experimental Results

### Main Results

| Metric | Agreement with Human Annotations (Krippendorff's Alpha) |
|---|---|
| CLIPScore | 0.17 |
| VQAScore | 0.31 |
| ImageReward | 0.11 |
| AIM (zero-shot LLM) | 0.18 |
| **AIM (InternVL + LLAMA3, fine-tuned)** | **0.68** |
| Inter-annotator agreement | 0.86 |

Improvement from LLM expansion on T2I generation (InternVL + LLAMA3 + AuraFlow):

| Configuration | AIM(COM) | C_obj(COM) | P_c+A(COM) | AIM(PSA) | C_obj(PSA) | P_c+A(PSA) |
|---|---|---|---|---|---|---|
| $I_{AR}$ (direct) | 0.50 | 2.12 | 0.64 | 0.31 | 1.36 | 0.42 |
| $I_{LLAMA3}$ (expanded) | 0.53 | 2.25 | 0.70 | 0.43 | 1.87 | 0.60 |
| $I_{QwenLM}$ (expanded) | 0.55 | 2.34 | 0.59 | 0.48 | 2.05 | 0.54 |

### Ablation Study

| Metric Configuration | Human Agreement on COM Ads | Human Agreement on PSA Ads |
|---|---|---|
| $C_{LLM}$ (direct LLM creativity score) | -0.03 | 0.15 |
| **$C_{obj}$ (proposed creativity metric)** | **0.57** | **0.53** |
| $P_{LLM}$ (direct LLM persuasiveness score) | 0.27 | 0.26 |
| $P_{comp}$ (multi-component, w/o AIM) | 0.83 | 0.54 |
| **$P_{comp+AIM}$ (full persuasiveness metric)** | **0.85** | **0.75** |
| Inter-annotator agreement | 0.80 | 0.56 |

The human–AI agreement of $P_{comp+AIM}$ on commercial advertisements (0.85) even surpasses inter-annotator agreement (0.80).

### Key Findings
- PSA (public service announcement) advertisements are harder to generate and evaluate than commercial ones, as their messages are more implicit.
- T2I models exhibit a severe lack of creativity and persuasiveness under implicit prompts.
- The simple LLM expansion strategy yields especially notable improvements on PSA advertisements (AIM +39%, creativity +38%, persuasiveness +43%).
- Multi-component persuasiveness evaluation outperforms direct LLM scoring, as sub-questions guide finer-grained reasoning.

## Highlights & Insights
- Incorporating rhetorical theory (Pathos/Ethos/Logos) into computational evaluation reflects refreshing interdisciplinary thinking.
- The design of the AIM metric is elegant: the LLM first interprets the image, then infers the underlying message, which is subsequently compared against the original.
- Human–AI agreement surpassing human–human agreement demonstrates the advantage of multi-component ensemble evaluation.
- The LLM expansion strategy is simple yet highly effective, revealing a fundamental deficit of T2I models in understanding implicit intent.

## Limitations & Future Work
- The creativity metric $C_{obj}$ still has room for improvement in human agreement (0.54 vs. 0.73 inter-annotator).
- Reliance on the PittAd dataset limits the diversity of advertisement types and message formats.
- The LLM expansion strategy, while effective, is biased toward textual description and cannot capture purely visual creativity.
- Cultural differences in the perception of advertising creativity and persuasiveness are not considered.
- The evaluation pipeline involves multiple models in series (MLLM + LLM + CLIP), resulting in substantial computational overhead.

## Related Work & Insights
- This work is complementary to NLP research on persuasive text generation, extending the scope from text to images.
- The problem of implicit message alignment also arises in tasks such as meme generation and poster design.
- The multi-component evaluation paradigm can be generalized to other generation tasks requiring multi-dimensional assessment.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to propose computational evaluation metrics for creativity and persuasiveness in advertising image generation; the problem formulation is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers diverse MLLM/LLM/T2I combinations, human annotation experiments, and separate analyses for commercial and PSA advertisements.
- Writing Quality: ⭐⭐⭐⭐ The paper is clearly structured with intuitive examples, though the dense notation requires careful reading.
- Value: ⭐⭐⭐⭐ Provides valuable evaluation tools and directions for improvement in advertising and creative content generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation](adiee_automatic_dataset_creation_and_scorer_for_instruction_guided_image_editing_evaluation.md)
- [\[ICCV 2025\] Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](holistic_unlearning_benchmark_a_multi-faceted_evaluation_for_text-to-image_diffu.md)
- [\[NeurIPS 2025\] OVERT: A Benchmark for Over-Refusal Evaluation on Text-to-Image Models](../../NeurIPS2025/image_generation/overt_a_benchmark_for_over-refusal_evaluation_on_text-to-image_models.md)
- [\[NeurIPS 2025\] Hallucination as an Upper Bound: A New Perspective on Text-to-Image Evaluation](../../NeurIPS2025/image_generation/hallucination_as_an_upper_bound_a_new_perspective_on_text-to-image_evaluation.md)
- [\[ICCV 2025\] Enhancing Reward Models for High-quality Image Generation: Beyond Text-Image Alignment](enhancing_reward_models_for_high-quality_image_generation_beyond_text-image_alig.md)

</div>

<!-- RELATED:END -->
