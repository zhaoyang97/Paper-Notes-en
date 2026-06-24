---
title: >-
  [Paper Note] Unveiling the Ignorance of MLLMs: Seeing Clearly, Answering Incorrectly
description: >-
  [CVPR 2025][Multimodal VLM][MLLM evaluation] Reveals the prevalent phenomenon in MLLMs of "**understanding visual content but still giving incorrect answers**", constructs the **MMVU benchmark** consisting of 12 categories of positive-negative question pairs, discovers that the root causes lie in training data bias towards positive samples and insufficient attention on visual tokens, and proposes a three-pronged solution: the **MMVU-Train dataset** (112K positive-negative pai…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "MLLM evaluation"
  - "visual understanding"
  - "misleading questions"
  - "positive-negative sample pairs"
  - "attention analysis"
  - "robustness"
date: 2026-05-08
content_hash: ea91a0c7cd016951
---

# Unveiling the Ignorance of MLLMs: Seeing Clearly, Answering Incorrectly

**Conference**: CVPR 2025  
**arXiv**: [2406.10638](https://arxiv.org/abs/2406.10638)  
**Code**: [https://github.com/BAAI-DCAI/MMVU](https://github.com/BAAI-DCAI/MMVU)  
**Area**: Multimodal VLM  
**Keywords**: MLLM evaluation, visual understanding, misleading questions, positive-negative sample pairs, attention analysis, robustness

## TL;DR
Reveals the prevalent phenomenon in MLLMs of "**understanding visual content but still giving incorrect answers**", constructs the **MMVU benchmark** consisting of 12 categories of positive-negative question pairs, discovers that the root causes lie in training data bias towards positive samples and insufficient attention on visual tokens, and proposes a three-pronged solution: the **MMVU-Train dataset** (112K positive-negative pairs) + **Content Guided Refinement (CGR)** + **Visual Attention Refinement (VAR)**.

## Background & Motivation
MLLMs perform exceptionally well in visual understanding tasks, yet an overlooked phenomenon is that models can **correctly answer directly related positive questions** (proving they understand the image) but fail on **indirect or misleading negative questions**. For example, a model can successfully identify "a black cat" in an image (positive question), but when asked "What color is the red part of this cat?", it takes the premise for granted and outputs an incorrect option.

**Limitations of Prior Work**:
1. Existing hallucination benchmarks (such as POPE, MADBench, etc.) do not distinguish between "errors caused by lack of understanding" and "errors made despite understanding."
2. Instruction tuning datasets predominantly focus on positive, direct visual question answering, lacking negative/misleading samples, which leads to systematic bias in models toward positive responses.
3. When generating tokens, the model's attention on visual tokens is far lower than its attention on system prompts and question tokens.

**Key Challenge**: Models possess visual understanding capabilities (answering positive questions correctly) but lack the ability to leverage this understanding to resist misleading information. The issue is not "failing to see", but "misapplying knowledge".

**Key Insight**: Construct paired positive/negative questions to precisely quantify this "understanding but making errors" phenomenon, analyze attention distribution to pinpoint the root causes, and propose solutions from both the data and inference perspectives.

## Method

### Overall Architecture
This work consists of three parts: (1) MMVU benchmark test set (893 paired positive-negative questions across 12 categories, manually annotated) + new evaluation metrics; (2) MMVU-Train training set (112K positive-negative pairs, automatically constructed pipeline); and (3) CGR and VAR strategies during the inference stage. These three components are complementary—the training data reduces bias from the source, while the inference strategies enhance visual focus during deployment.

### Key Designs

1. **MMVU Benchmark Design and Evaluation Metrics**
    - Function: Precisely evaluate the extent of "understanding but making errors" in MLLMs.
    - Mechanism: Each image is paired with two multiple-choice questions—**positive questions** directly examine visual understanding (e.g., "What color is the flower in the picture?"), while **negative questions** introduce misleading information to test robustness (e.g., "How many red petals does the flower in the picture have?"—when the flower is actually blue). It covers 12 categories across 3 levels:
        - Character level: Character/digit recognition
        - Attribute level: Color/texture, quantity, shape, pose, position
        - Context level: Abstract knowledge, concrete knowledge, specialized knowledge, actions, relationships
    - Evaluation metrics:
        - **RA (Response Accuracy)**: Synthetic accuracy across both positive and negative questions, ↑ higher is better.
        - **MR (Misresponse Rate)**: The proportion of negative questions answered incorrectly, conditional on answering positive questions correctly, ↓ lower is better. MR directly quantifies the extent of "understanding but making errors".
    - Design Motivation: Existing benchmarks equate "answering incorrectly" with "lack of understanding", whereas MMVU distinguishes "true lack of understanding" from "understanding but being misled" through positive-negative pairs.

2. **Positive-Negative Paired Data Construction Pipeline (MMVU-Train)**
    - Function: Constructs 112K positive-negative training pairs to reduce the systematic bias of MLLMs toward positive responses from the data source.
    - Mechanism: Automatically extracts visual information (text, digits, objects, attributes, relationships, local/global contexts) from LLM-enhanced image descriptions, and then constructs positive questions (directly examining observable content) and negative questions (introducing hypothetical modifications, irrelevant descriptions, or false premises) based on this information, with 4 options per question (including distractors with explanatory details).
    - Design Motivation: Manual annotation is prohibitively expensive (the MMVU test set contains only 893 pairs), whereas an automated pipeline scales up generation. Crucially, negative questions are not random noise, but rather carefully constructed counter-arguments based on actual image content—ensuring the model learns to "make judgments based on visual understanding" rather than simply "rejecting all negative questions".

3. **Content Guided Refinement (CGR) + Visual Attention Refinement (VAR)**
    - Function: Enhances model attention on and utilization of visual content during the inference stage.
    - **CGR (Content Guided Refinement)**:
        - Two-step inference: First prompts the model to conduct a detailed content analysis of the image (extracting structured information), and then answers the question based on the analysis output and the original image.
        - Motivation: Forces the model to "see before thinking" before answering, making visual understanding explicit and reducing the influence of language bias on responses.
    - **VAR (Visual Attention Refinement)**:
        - Extracts attention scores between question tokens and visual tokens, highlights high-score regions and masks low-score ones, generating a focused mask to guide the model to pay attention to visual regions relevant to the question.
        - Motivation: Analysis reveals that MLLM attention on visual tokens is far lower than on system/question tokens (Fig. 4a), and visual attention is even lower during negative questions (Fig. 4b)—VAR mitigates this imbalance by explicitly enhancing visual attention.

## Key Experimental Results

### Performance of 15 MLLMs on MMVU (Tab. 1)

| Model | Avg. RA↑ | Avg. MR↓ |
|------|---------|---------|
| GPT-4o | 65.06 | 19.53 |
| Ovis1.6-Gemma2-9B | 66.74 | 19.78 |
| Llama3.2-90B | 66.74 | 18.47 |
| LLaVA-OneVision-7B | 60.58 | 27.77 |
| InternVL2-8B | 49.72 | 30.63 |
| VILA1.5-13B | 25.76 | 62.30 |
- Even the strongest model, GPT-4o, has a misresponse rate of 19.53%—failing once out of every five times it correctly understands the visual content.
- VILA1.5-13B has an MR of up to 62.30%—more than half of the cases are "understanding but answering incorrectly".

### Fine-Tuning Results with MMVU-Train (Tab. 2)

| Model | Baseline RA | +MMVU-Train RA | Gain |
|------|--------|---------------|------|
| InternVL2-8B | 49.72 | 58.01 | **+8.29** |
| VILA1.5-13B | 25.76 | 37.74 | **+11.98** |
| LLaVA-OneVision-7B | 60.58 | 63.49 | +2.91 |

| Model | Baseline MR | +MMVU-Train MR | Reduction |
|------|--------|---------------|------|
| LLaVA-OneVision-7B | 27.77 | 21.03 | **-6.74** |
| InternVL2-8B | 30.63 | 25.04 | **-5.59** |

### Key Findings from Attention Analysis (Fig. 4)
- Attention from answer tokens to visual tokens accounts for only ~10%, far lower than that to system tokens (~55%) and question tokens (~35%).
- In the presence of negative questions, question-to-visual attention is even lower compared to positive questions (approx. 15-25% lower).
- During negative questions, the model's output probability decreases (confidence drops), yet it still outputs incorrect answers—indicating that the model is "hesitant but still biased towards making errors".

### Performance of CGR + VAR Strategies
- CGR improves RA by an average of 2-4 percentage points, and the combination of CGR + VAR yields even better results.
- Performance does not degrade on general benchmarks (MMBench, SEED, etc.), indicating that enhancing robustness does not compromise general capabilities.

## Highlights & Insights
- **Unique Perspective**: Distinguishes between "errors caused by lack of understanding" and "errors made despite understanding", revealing an overlooked systematic flaw in MLLMs.
- **Causal Analysis**: Pinpoints two root causes (data bias + insufficient visual attention) through attention map and logit distribution analyses, rather than relying solely on empirical speculation.
- **Two-Pronged Approach (Data & Inference)**: MMVU-Train reduces bias from the data source, while CGR + VAR enhances visual utilization during inference—the two approaches complement each other.
- **Exquisite Positive-Negative Pairing**: Negative questions are not randomly generated, but are "counter-parts" carefully constructed based on the visual content of positive questions—ensuring that errors indeed stem from "understanding but failing" rather than "lack of understanding".
- **Revelatory finding**: Even GPT-4o displays an MR of ~20%—proving that this is not unique to small models but is a systematic flaw in the current training paradigm of MLLMs.

## Limitations & Future Work
- The MMVU test set size is relatively small (893 pairs), with certain categories having very few samples (e.g., only 71 pairs for the relationship category).
- It only utilizes multiple-choice question format, and its performance in open-ended Q&A scenarios has not been verified.
- The VAR strategy relies on accessing intermediate attention maps, making it inapplicable to closed-source API models.
- The automatically constructed MMVU-Train may contain noisy samples, as no manual quality audit was performed.
- The two-step inference of CGR increases inference latency (approximately doubling it).

## Related Work & Insights
- POPE, MADBench $\rightarrow$ Focus on object hallucination and adversarial misleading, but do not distinguish between "lack of understanding" and "understanding but making errors".
- NaturalBench $\rightarrow$ Focuses on consistency evaluation, but does not analyze the root causes of attention allocation.
- VCD (Visual Contrastive Decoding) $\rightarrow$ Modifies decoding strategies to mitigate hallucination, presenting a complementary idea to VAR.
- Insights: The "understanding" and "application of understanding" in MLLMs are distinct capability dimensions—even if the model “sees” the correct information, improper attention allocation or training data bias can still lead to incorrect answers. This finding provides profound implications for the construction of instruction-tuning data for MLLMs.

## Rating
⭐⭐⭐⭐ — Discovers an important yet overlooked vulnerability in MLLMs, utilizing rigorous and objective analysis methodologies (positive-negative pairing + attention analysis). The MMVU benchmark and MMVU-Train dataset make practical contributions to the community. Although the proposed solutions (data + inference strategies) are not entirely disruptive, they are highly practical and effective. The overall work is complete and rigorous.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Seeing Justice Clearly: Handwritten Legal Document Translation with OCR and Vision-Language Models](../../AAAI2026/multimodal_vlm/seeing_justice_clearly_handwritten_legal_document_translation_with_ocr_and_visio.md)
- [\[CVPR 2025\] Seeing the Abstract: Translating the Abstract Language for Vision Language Models](seeing_the_abstract_translating_the_abstract_language_for_vision_language_models.md)
- [\[ACL 2025\] Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension](../../ACL2025/multimodal_vlm/unveiling_cultural_blind_spots_analyzing_the_limitations_of_mllms_in_procedural_.md)
- [\[CVPR 2025\] MARTEN: Visual Question Answering with Mask Generation for Multi-Modal Document Understanding](marten_visual_question_answering_with_mask_generation_for_multi-modal_document_u.md)
- [\[ACL 2025\] VLSBench: Unveiling Visual Leakage in Multimodal Safety](../../ACL2025/multimodal_vlm/vlsbench_unveiling_visual_leakage_in_multimodal_safety.md)

</div>

<!-- RELATED:END -->
