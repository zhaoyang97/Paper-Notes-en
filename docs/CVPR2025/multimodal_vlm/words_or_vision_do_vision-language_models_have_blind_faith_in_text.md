---
title: >-
  [Paper Note] Words or Vision: Do Vision-Language Models Have Blind Faith in Text?
description: >-
  [CVPR 2025][Multimodal VLM][Modality Preference] This paper identifies the phenomenon of "blind faith in text" in VLMs—where models systematically favor text (even when incorrect) when visual and textual inputs are inconsistent. By constructing a benchmark with three text variants (Match, Corruption, and Irrelevance), this work evaluates 10 VLMs, analyzes five influencing factors, demonstrates that SFT with text augmentation effectively mitigates this issue…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Modality Preference"
  - "Text Bias"
  - "VLM Robustness"
  - "Multimodal Inconsistency"
  - "Security Risks"
date: 2026-05-08
content_hash: 1662d7094142bc40
---

# Words or Vision: Do Vision-Language Models Have Blind Faith in Text?

**Conference**: CVPR 2025  
**arXiv**: [2503.02199](https://arxiv.org/abs/2503.02199)  
**Code**: [https://github.com/d-ailin/blind-faith-in-text](https://github.com/d-ailin/blind-faith-in-text)  
**Area**: Multimodal VLMs  
**Keywords**: Modality Preference, Text Bias, VLM Robustness, Multimodal Inconsistency, Security Risks

## TL;DR

This paper identifies the phenomenon of "blind faith in text" in VLMs—where models systematically favor text (even when incorrect) when visual and textual inputs are inconsistent. By constructing a benchmark with three text variants (Match, Corruption, and Irrelevance), this work evaluates 10 VLMs, analyzes five influencing factors, demonstrates that SFT with text augmentation effectively mitigates this issue, and provides a theoretical explanation tracing the root cause to the imbalance between text-only and multimodal training data.

## Background & Motivation

VLMs perform exceptionally well on vision-centric tasks. However, in practical applications like RAG and multimodal agents, textual inputs often conflict with or mislead the visual information. Existing VLM evaluations are primarily vision-centric, where text only serves as question inputs, lacking assessment of the model's ability to handle multimodal inconsistencies. 

Key Challenge: **Since VLMs are built on top of pretrained language models, text-only training data heavily dominates multimodal data ($N \gg M$), introducing a natural bias toward the textual modality**. 

Key Insight: Systematically injecting three types of text variants (Match, Corruption, and Irrelevance) into vision tasks to quantify the model's modality preference. 

Core Idea: The "blind faith" in text exhibited by VLMs is a structural issue arising from the modality imbalance in training data.

## Method

### Overall Architecture

An evaluation benchmark spanning four domains (VQAv2, DocVQA, MathVista, and Brand Recognition) is constructed, where each sample is expanded with three text variants. Match and Corruption texts are generated using GPT-4o, while Irrelevance texts are randomly sampled from WikiText. On samples with inconsistent visual and textual answers, the modality that the model follows is analyzed. Text Preference Ratio (TPR) is defined to quantify the degree of text preference. Ten VLMs are evaluated, and five influencing factors are analyzed.

### Key Designs

1. **Construction of Three Text Variants**:
    - Function: Comprehensively evaluate VLM behavior under different types of textual interference.
    - Mechanism: Match text $T_m$ provides a description of the correct answer, Corruption text $T_c$ provides a misleading description (relying on text leads to an incorrect answer), and Irrelevance text $T_{irr}$ provides task-irrelevant Wikipedia paragraphs. The three variants, along with the image-only base, constitute the full evaluation.
    - Design Motivation: Using Corruption alone cannot distinguish between "ignoring text" and "correctly utilizing text"—as models might simply learn to reject all text. Introducing Match ensures that the model must evaluate visual-text consistency rather than blindly rejecting all text.

2. **Text Preference Ratio (TPR) Metric**:
    - Function: Quantify the modality preference of the model when visual and textual answers are inconsistent.
    - Mechanism: For samples where $\hat{Y}_{img} \neq \hat{Y}_{txt}$, TPR is calculated as $$\text{TPR} = \frac{p_{txt}}{p_{txt} + p_{img}}$$ where $p_{txt}$ represents the proportion of final answers that match the text-only answers.
    - Design Motivation: Simply looking at accuracy cannot distinguish whether "the model failed because it trusted the incorrect text" or "the model itself is incapable". TPR directly measures the direction and intensity of modality preference.

3. **SFT Text-Augmented Mitigation Scheme**:
    - Function: Reduce text bias through Supervised Fine-Tuning (SFT).
    - Mechanism: Collect 1,000 samples, consisting of text-only data, original VQA, and 200 samples each for Match/Corruption/Irrelevance. Fine-tune for 3 epochs using LoRA. It is crucial to include text-only data to preserve language capabilities.
    - Design Motivation: Instruction tuning yields limited improvement (TPR decreases by only 2.6%), whereas SFT directly trains the model under diverse textual conditions, enabling it to make correct decisions regardless of textual quality.

### Loss & Training

SFT employs the standard next-token prediction loss with a learning rate of $1.0 \times 10^{-4}$, cosine decay, warmup ratio of 0.1, and LoRA parameter-efficient fine-tuning. The key lies in the **data composition**: it must include both text-only data (lest the model over-rejects all text) and all three text variants.

## Key Experimental Results

### Main Results (Performance Impact under Corruption)

| Model | VQAv2 Base | VQAv2 Corr | Norm↑ | TPR↓ | DocVQA Base | DocVQA Corr | Norm↑ |
|------|-----------|-----------|-------|------|-----------|-----------|-------|
| GPT-4o | 78.39 | 70.75 | 90.25 | 27.09 | 85.00 | 73.60 | 86.59 |
| Claude Sonnet | 66.88 | 68.17 | **101.93** | **9.58** | 87.00 | **84.60** | **97.24** |
| LLaVA-NeXT-7B | 79.45 | 28.69 | 36.10 | 85.52 | 53.60 | 10.00 | 18.60 |
| Qwen2-VL-7B | 85.51 | 50.79 | 59.41 | 29.22 | 90.50 | 57.50 | 63.63 |
| Molmo-7B-D | 76.33 | 49.29 | 64.50 | 59.40 | 74.00 | 38.40 | 51.90 |

The performance degradation under Corruption is significantly higher for open-source models than for closed-source models. LLaVA-NeXT-7B retains only 36.1% of its baseline performance on VQAv2.

### SFT Performance (In-Domain VQAv2)

| Model | Base↑ | Match↑ | Corr↑ | Irr↑ | Macro↑ |
|------|-------|--------|-------|------|--------|
| LLaVA-NeXT-7B Original | 79.45 | 92.32 | 28.69 | 79.43 | 66.81 |
| + Instruction | 79.45 | 92.25 | 34.27 | 78.15 | 68.22 |
| + **SFT** | 77.48 | 87.56 | **71.25** | 77.32 | **78.71** |
| Qwen2-VL-7B Original | 85.51 | 92.76 | 50.79 | 83.70 | 75.75 |
| + **SFT** | 84.18 | 87.01 | **82.72** | 84.00 | **84.58** |

SFT improves the Corruption accuracy of LLaVA from 28.69% to 71.25%, with a 12% improvement in Macro-average accuracy.

### Key Findings

- **Blind faith in text is a pervasive phenomenon**: Almost all VLMs exhibit a TPR >50% under both Corruption and Match, showing bias even toward incorrect texts.
- **Open-source models exhibit much stronger text bias than closed-source models**: Claude Sonnet is the most robust (VQAv2 TPR of only 9.58%), while LLaVA-NeXT-7B is the worst (85.52%).
- **Instruction tuning has limited effect**: A "Focus on Image" instruction only reduces Qwen's TPR from 16.8% to 14.2%.
- **Model scaling provides limited help**: Scaling from 7B to 34B reduces text bias, but the improvement saturates.
- **Token order is highly influential**: Placing text tokens before image tokens exacerbates the text bias.
- **Positive correlation with text relevance**: Highly relevant texts retrieved by BM25 (even if useless) are more likely to influence the model.
- **Security risks**: In the brand recognition task, Molmo-7B-D's performance drops from 87.44% to 41.44% under HTML corruption, suggesting that phishing websites could exploit this vulnerability.

## Highlights & Insights

- **Unveiling structural blind spots in VLMs**: Text bias is not a bug in specific models, but rather a systematic consequence of building VLMs on top of LLMs.
- **Profound theoretical explanation**: $N \gg M$ (where text-only data is much larger than multimodal data) causes a smaller upper bound for text-only loss and a larger upper bound for multimodal loss, explaining the root of the bias from an information-theoretic perspective.
- **Practical security warning**: The brand recognition task directly demonstrates the safety hazards of text bias—malicious HTML injection can bypass VLM-based phishing detection.
- **Low-cost mitigation**: Significant improvements are achieved with only 1,000 SFT samples and LoRA, making the solution highly practical.

## Limitations & Future Work

- Evaluation is limited to classification/short-answer tasks, leaving long-text generation scenarios unexplored.
- Although SFT is effective, it slightly degrades the baseline performance (e.g., LLaVA drops from 79.45% to 77.48%). Achieving zero-cost mitigation remains to be studied.
- The theoretical analysis relies on strong assumptions (e.g., bounded loss, ERM convergence), whereas actual training environments are far more complex.
- The effect of balancing the ratio of text-only and multimodal data during the pretraining stage has not been explored.

## Related Work & Insights

- **vs. Hallucination Research**: While hallucination focuses on cases where the model "hallucinates non-existent content", this study focuses on cases where "the model selects the incorrect modality when faced with conflicting information".
- **vs. Multimodal RAG**: The findings directly warn against RAG scenarios, where retrieved but "relevant yet incorrect" text can severely mislead VLMs.
- **vs. Textual Robustness Research**: While extensive text adversarial attack research exists in the NLP domain, this work extends the scope to multimodal inconsistency scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ Defining and systematically quantifying the "blind faith in text" phenomenon is a novel contribution, backed by a deep theoretical explanation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensively evaluated across 10 models, 4 domains, 3 variants, with factor analysis and SFT validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition, well-designed metrics, and a logical structure.
- Value: ⭐⭐⭐⭐ Provides crucial warning signs for the safe deployment of VLMs, with a highly practical SFT mitigation strategy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation](../../ACL2025/multimodal_vlm/do_vision-language_models_have_internal_world_models_towards_an_atomic_evaluatio.md)
- [\[CVPR 2025\] Vision-Language Models Do Not Understand Negation](vision-language_models_do_not_understand_negation.md)
- [\[CVPR 2025\] It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data](its_a_blind_match_towards_vision-language_correspondence_without_parallel_data.md)
- [\[CVPR 2025\] PARC: A Quantitative Framework Uncovering the Symmetries within Vision Language Models](parc_a_quantitative_framework_uncovering_the_symmetries_within_vision_language_m.md)
- [\[CVPR 2025\] FastVLM: Efficient Vision Encoding for Vision Language Models](fastvlm_efficient_vision_encoding_for_vision_language_models.md)

</div>

<!-- RELATED:END -->
