---
title: >-
  [Paper Note] Toward Robust Hyper-Detailed Image Captioning: A Multiagent Approach and Dual Evaluation Metrics for Factuality and Coverage
description: >-
  [ICML 2025][Multimodal VLM][Hyper-Detailed Captioning] Proposed the CapMAS multi-agent system, which corrects hallucinations through LLM-MLLM collaboration by decomposing detailed image-text descriptions into atomic propositions and verifying their truthfulness one by one. It also introduces a framework to evaluate detailed captions from the dual dimensions of factuality and coverage, significantly improving the description quality of various MLLMs, including GPT-4V.
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "Hyper-Detailed Captioning"
  - "Hallucination"
  - "Multiagent System"
  - "Factuality"
  - "Coverage Evaluation"
date: 2026-05-08
content_hash: 09796e6a694aacce
---

# Toward Robust Hyper-Detailed Image Captioning: A Multiagent Approach and Dual Evaluation Metrics for Factuality and Coverage

**Conference**: ICML 2025  
**arXiv**: [2412.15484](https://arxiv.org/abs/2412.15484)  
**Code**: [github.com/adobe-research/CapMAS](https://github.com/adobe-research/CapMAS)  
**Area**: Multimodal VLM  
**Keywords**: Hyper-Detailed Captioning, Hallucination, Multiagent System, Factuality, Coverage Evaluation

## TL;DR

Proposed the CapMAS multi-agent system, which corrects hallucinations through LLM-MLLM collaboration by decomposing detailed image-text descriptions into atomic propositions and verifying their truthfulness one by one. It also introduces a framework to evaluate detailed captions from the dual dimensions of factuality and coverage, significantly improving the description quality of various MLLMs, including GPT-4V.

## Background & Motivation

MLLMs can generate long and detailed image descriptions but suffer from severe **hallucination issues**: descriptions include objects, or incorrect attributes/relations, that do not exist in the images.

Key finding: **Existing hallucination detection methods fail on long sequences**.
- Confidence and Consistency methods fail to detect hallucinations after the 192nd token.
- Reason: As the MLLM output grows longer, the model increasingly relies on its self-generated text rather than the input image (attention weights shift from image tokens to text tokens).

Experimental validation: By "isolating" objects in long descriptions into independent queries (the Isolation method), the AUROC improves from 57.5 (Confidence) and 73.5 (Consistency) to **81.4**.

## Method

### CapMAS Multi-agent System

Three-step pipeline (training-free):
1. **Decomposer LLM**: Decomposes the detailed description into atomic propositions (the smallest units that can be verified as true or false).
2. **Fact Check MLLM**: Converts each proposition into a True/False question and queries the MLLM independently.

Definition of hallucination score:
$$H(u) = -\log(\min(p(\text{T}|x, Q(u)) - p(\text{F}|x, Q(u)), \epsilon))$$

Propositions are classified into a True set $\mathcal{T}$ and a False set $\mathcal{F}$ based on a threshold $\pi$.

3. **Corrector LLM**: Corrects the original description based on $\mathcal{T}$ and $\mathcal{F}$.

### Evaluation Framework

**Factuality Evaluation**:
- GPT-4o decomposes descriptions into atomic propositions, then determines truthfulness by referring to both the image and reference descriptions.
- Factuality = $T / (T + F)$

**Coverage Evaluation**:
- Constructs a high-granularity VQA dataset (averaging 49.8 multiple-choice questions per image, 19,899 questions in total).
- Assumption: If the description completely covers the image information, visual questions can be answered using only the description.
- Employs an LLM to answer questions based on the generated description, and uses the accuracy as the coverage.

### Meta-evaluation of Evaluation Metrics

Three types of hallucinations (Object/Attribution/Relation) were introduced into the DOCCI dataset to test whether each metric could detect them:

| Metric | Clean | Object | Attrib | Relation | Can Detect? |
|------|-------|--------|--------|----------|---------|
| CIDEr | 6.4 | 4.8 | 6.2 | **6.7** | ✗ |
| CLIP-S | 81.3 | 81.0 | 80.9 | **81.4** | ✗ |
| CLAIR | 86.9 | 85.2 | 80.0 | 83.5 | Partially |
| **Ours** | **62.8** | 52.3 | 60.9 | 51.9 | **✓** |

## Key Experimental Results

### Improvement of CapMAS on Different Models

| Description Model | CapMAS | CLAIR | Factuality | Coverage | Average |
|---------|--------|-------|--------|--------|------|
| LLaVA-NeXT-7B | — | 68.8 | 59.9 | 47.9 | 58.9 |
| LLaVA-NeXT-7B | LLaMA-3 + 7B | 74.1 | **72.2** | 46.9 | **64.4** |
| GPT-4V | — | 82.4 | 77.1 | 53.5 | 71.0 |
| GPT-4V | LLaMA-3 + InternVL | **84.6** | **82.1** | **53.5** | **73.4** |

### Comparison with Other Methods

| Method | CLAIR | Factuality | Coverage | Average |
|------|-------|--------|--------|------|
| Base (LLaVA-1.5-7B) | 62.1 | 52.8 | 34.3 | 49.7 |
| VCD | 59.7 | 44.6 | 39.3 | 47.9 |
| OPERA | 59.1 | 53.0 | 34.1 | 48.7 |
| LURE | 57.2 | 51.9 | 27.6 | 45.6 |
| **CapMAS** | **66.3** | **63.4** | 33.1 | **54.3** |

### Key Findings

- Existing decoding methods (VCD, OPERA) are **ineffective or even harmful** for detailed descriptions (VCD reduces factuality).
- CapMAS improves the factuality of GPT-4V descriptions (77.1→82.1), even when using a much weaker model for checking than GPT-4V.
- VQA benchmark performance is **uncorrelated** with detailed description capabilities, questioning the VQA-centric evaluation paradigm.

## Highlights & Insights

1. **Isolation verification outperforms Confidence/Consistency**: Confirms the necessity of the decompose-then-check strategy.
2. **Plug-and-play + training-free**: Can be applied to any description model, including closed-source GPT-4V.
3. **Dual evaluation of Factuality × Coverage**: Systematically decouples and evaluates these two dimensions for the first time.
4. **Revelation of VQA benchmark limitations**: Strong performance of an MLLM on VQA does not imply superior description ability.

## Limitations & Future Work

- Factuality improvement is accompanied by a slight decrease in coverage (conservative corrections lead to information loss).
- Relies on the MLLM's own visual understanding capabilities to check for hallucinations.
- The quality of the LLM decomposer affects the final results.
- Hyperparameter $\pi$ controls the trade-off between factuality and coverage.

## Related Work & Insights

- Decoding methods (VCD, OPERA)
- Training methods (LRV)
- Correction methods (LURE, Volcano)
- Evaluation methods (CLIPScore, CLAIR, ALOHa, FaithScore)

## Rating

⭐⭐⭐⭐ — Precise problem formulation (failure of long-sequence hallucination detection), and a well-designed evaluation framework. The CapMAS method is intuitive and highly effective. The dual-dimensional evaluation and the revelation of VQA benchmark limitations possess independent value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DisCode: Distribution-Aware Score Decoder for Robust Automatic Evaluation of Image Captioning](../../AAAI2026/multimodal_vlm/discode_distribution-aware_score_decoder_for_robust_automatic_evaluation_of_imag.md)
- [\[ICLR 2026\] ScaleCap: Scalable Image Captioning via Dual-Modality Debiasing](../../ICLR2026/multimodal_vlm/scalecap_scalable_image_captioning_via_dual-modality_debiasing.md)
- [\[ICCV 2025\] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models](../../ICCV2025/multimodal_vlm/simplevqa_multimodal_factuality_evaluation_for_multimodal_large_language_models.md)
- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](../../ICCV2025/multimodal_vlm/sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)
- [\[ICML 2025\] Robust Multimodal Large Language Models Against Modality Conflict](robust_multimodal_large_language_models_against_modality_conflict.md)

</div>

<!-- RELATED:END -->
