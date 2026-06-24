---
title: >-
  [Paper Note] Ensemble Watermarks for Large Language Models
description: >-
  [ACL 2025][LLM Safety][watermarking] Proposes an ensemble watermarking method that combines stylometric features (acrostics + sensorimotor norms) with existing red-green watermarks, achieving a 95% detection rate for the three-feature ensemble after paraphrasing attacks, compared to only 49% for the red-green watermark alone.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "watermarking"
  - "LLM"
  - "stylometry"
  - "paraphrasing attack"
  - "acrostic"
date: 2026-05-08
content_hash: e66a8258537aa9ae
---

# Ensemble Watermarks for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2411.19563](https://arxiv.org/abs/2411.19563)  
**Code**: [GitHub](https://github.com/CommodoreEU/ensemble-watermark)  
**Area**: AI Safety  
**Keywords**: watermarking, LLM, stylometry, paraphrasing attack, acrostic

## TL;DR
Proposes an ensemble watermarking method that combines stylometric features (acrostics + sensorimotor norms) with existing red-green watermarks, achieving a 95% detection rate for the three-feature ensemble after paraphrasing attacks, compared to only 49% for the red-green watermark alone.

## Background & Motivation

### Background

**Background**: LLM watermarking (e.g., the red-green watermark by Kirchenbauer et al.) embeds hidden signals by modifying token generation probabilities.

**Limitations of Prior Work**: Single-feature watermarks possess poor robustness under paraphrasing attacks—the detection rate of the red-green watermark drops sharply to ~49%.

**Key Challenge**: Single watermark features are easily defeated by a single attack strategy, lacking redundancy and diversity.

**Goal**: How to improve robustness by combining multiple different watermarking features?

**Key Insight**: Borrowing author identification features from stylometry, acrostics (spelling secret messages with sentence-initial letters) and sensorimotor norms (words biased toward specific sensory categories) are selected as new features to ensemble with the red-green watermark.

**Core Idea**: Embodying backups via an ensemble of multiple orthogonal watermark features, making it difficult for attackers to eliminate all signals simultaneously.

## Method

### Overall Architecture
During generation, three features are simultaneously embedded via logit manipulation: (1) red-green watermark (token-level), (2) sensorimotor word bias (word-level), and (3) acrostic patterns (sentence-level). During detection, a unified statistical test function is used to detect any combination of features without modification.

### Key Designs
1. **Acrostic Feature**:

    - Function: The first token of each new sentence is biased to start with a specific letter (controlled by a secret key).
    - Implementation: $\text{logits}[t] += \delta_{\text{acro}} \cdot \mathbf{1}\{\text{starts\_with\_target\_letter}\}$
    - Characteristics: Has minimal impact on perplexity but is less robust to paraphrasing (as sentence structures are easily altered).

2. **Sensorimotor Feature**:

    - Function: Biases generation to contain words belonging to specific sensory categories (e.g., olfactory, haptic).
    - Implementation: Based on Lancaster Sensorimotor Norms (an 11-dimensional sensory rating of 40K words), biasing words belonging to target categories.
    - Characteristics: Most robust to paraphrasing (>80%), because paraphrasing tools struggle to alter sensorimotor semantics.

3. **Key Management**:

    - Dynamically generates keys based on the SHA256 hash of the previous word/sentence.
    - Token text is first preprocessed via stop-word removal and lemmatization before hashing to enhance robustness.
    - Used to control the target letter of the acrostic and the target category of the sensorimotor feature.

### Unified Detection Function
A statistical test score is calculated for each feature, and the scores of all features can be combined for collective judgment without modifying the detection logic for different feature combinations.

## Key Experimental Results

### Paraphrasing Attack Detection Rate (≥10% of text paraphrased)

### Main Results

| Feature Combination | Llama 3.1 8B (Strong) | Llama 3.2 3B (Strong) |
|----------|----------------------|----------------------|
| Red-Green Only | 49.14% | 54.05% |
| Sensorimotor Only | 80.41% | 85.11% |
| Acrostic Only | 28.52% | 31.39% |
| All Three | **95.19%** | **95.79%** |
| Human Text False Positive | 0.34% | 0.97% |

### Detection Rate without Attack
- The all-feature ensemble achieves a ~98% detection rate under the Strong setting.
- Even under the Weak setting, it outperforms any single feature in the Strong setting.

### Key Findings
- Sensorimotor features are the most robust to paraphrasing (80%+), while the red-green watermark is the least robust (~49%).
- Acrostic features have the minimal impact on perplexity but yield low detection rates when used alone.
- The three-feature ensemble is consistently optimal across all models and intensity settings.
- The ensemble advantage is more pronounced on shorter texts (<5 sentences).

## Highlights & Insights
- Introducing stylometric features to LLM watermarking is an innovative cross-domain integration.
- The sensorimotor feature exploits deep semantic structures of human cognition, which paraphrasing tools struggle to eliminate.
- Flexibility of the ensemble approach: feature combinations can be selected according to needs, and the same detection function is applicable to all configurations.

## Limitations & Future Work
- Limited to English (acrostics require Latin letters, and the sensorimotor database is also in English).
- Extensively validated only on decoder models (Llama, Mistral).
- Stronger attacks (e.g., translation, rewriting, LLM paraphrasing) were not fully considered.
- Increasing watermark intensity degrades perplexity.

## Related Work & Insights
- **vs Kirchenbauer et al. (2023) Red-Green Watermark**: Drops to only 49% after paraphrasing when used alone, whereas the ensemble reaches 95%.
- **vs Duwak**: Duwak modifies the sampling strategy; this work modifies the feature types. The underlying approaches differ.
- **vs Post-processing Methods**: Logit manipulation is more natural than post-processing and does not require auxiliary models.

## Rating
- Novelty: ⭐⭐⭐⭐ Creative cross-domain integration of stylometric features and watermarking.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation using 3 models × 3 intensities × multiple feature combinations under paraphrasing attacks.
- Writing Quality: ⭐⭐⭐⭐ Clear method description and dense information in tables.
- Value: ⭐⭐⭐⭐ Direct contribution to the research on LLM watermarking robustness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Improved Unbiased Watermark for Large Language Models](improved_unbiased_watermark_for_large_language.md)
- [\[ACL 2026\] Topic-Based Watermarks for Large Language Models](../../ACL2026/llm_safety/topic-based_watermarks_for_large_language_models.md)
- [\[ICLR 2026\] In-Context Watermarks for Large Language Models](../../ICLR2026/llm_safety/in-context_watermarks_for_large_language_models.md)
- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)
- [\[ACL 2025\] MEGen: Generative Backdoor into Large Language Models via Model Editing](megen_generative_backdoor_into_large_language_models_via_model_editing.md)

</div>

<!-- RELATED:END -->
