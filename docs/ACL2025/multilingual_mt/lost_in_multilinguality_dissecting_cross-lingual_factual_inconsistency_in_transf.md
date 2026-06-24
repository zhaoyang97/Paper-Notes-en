---
title: >-
  [Paper Note] Lost in Multilinguality: Dissecting Cross-lingual Factual Inconsistency in Transformer Language Models
description: >-
  [ACL 2025][Multilingual & Machine Translation][cross-lingual consistency] This work dissects the cross-lingual factual inconsistency in multilingual LLMs using mechanistic interpretability. It reveals that while models process knowledge in a language-agnostic concept space across most layers, inconsistencies arise from failures during the "language transition" process in the final few layers. A linear shortcut method is proposed to bypass these final layers…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "cross-lingual consistency"
  - "mechanistic interpretability"
  - "language transition"
  - "factual knowledge"
  - "multilingual LM"
date: 2026-05-08
content_hash: 5a34cb76eabb22b1
---

# Lost in Multilinguality: Dissecting Cross-lingual Factual Inconsistency in Transformer Language Models

**Conference**: ACL 2025  
**arXiv**: [2504.04264](https://arxiv.org/abs/2504.04264)  
**Code**: [https://github.com/boschresearch/KLAR-CLC](https://github.com/boschresearch/KLAR-CLC)  
**Area**: Multilingual Translation  
**Keywords**: cross-lingual consistency, mechanistic interpretability, language transition, factual knowledge, multilingual LM

## TL;DR
This work dissects the cross-lingual factual inconsistency in multilingual LLMs using mechanistic interpretability. It reveals that while models process knowledge in a language-agnostic concept space across most layers, inconsistencies arise from failures during the "language transition" process in the final few layers. A linear shortcut method is proposed to bypass these final layers, improving both consistency and accuracy.

## Background & Motivation

**Background**: Multilingual LLMs often provide inconsistent answers when responding to the same factual question in different languages.

**Limitations of Prior Work**: Prior work identified the inconsistency phenomenon but failed to analyze the underlying internal causes; interpretability research has primarily focused on correctly predicted cases.

**Key Challenge**: The model already "knows" the correct answer in its intermediate layers (in the concept space), yet it makes errors when transitioning to the target language—why is this the case?

**Goal**: Track the information flow through interpretability to pinpoint the internal causes of cross-lingual factual inconsistency.

**Key Insight**: Utilize Logit Lens and causal tracing to compare the internal mechanism differences between "consistently correct" and "cross-lingual inconsistent" scenarios.

**Core Idea**: The root cause of cross-lingual inconsistency lies in the failure of the "language transition" mechanism in the final few layers—the model knows the correct answer in the concept space but fails to correctly translate it into the target language.

## Method

### Overall Architecture
Construct the KLAR dataset (17 languages × 20 relation types) -> Evaluate cross-lingual consistency -> Trace representations across layers using Logit Lens -> Compare consistent and inconsistent cases -> Discover language transition failures -> Propose a linear shortcut method.

### Key Designs

1. **KLAR Dataset**

    - 17 languages, 20 relation types
    - Knowledge probing format designed for autoregressive models
    - **Design Motivation**: Cover more languages and relations than existing datasets

2. **Logit Lens Analysis**

    - Project intermediate hidden states of each layer into the vocabulary space to observe how "current predictions" evolve across layers
    - Finding: Intermediate layer predictions align closely with the correct English answer (concept space), while the final layers transition to the target language
    - **Design Motivation**: Unveil the internal levels of language processing within the model

3. **Linear Shortcut Method**

    - Learn a linear mapping to directly project representations from intermediate layers (concept space) to target language predictions
    - Bypass the final few layers where language transition fails
    - **Design Motivation**: If the correct answer is already present in the concept space, jumping directly can bypass transition errors

## Key Experimental Results

### Main Results — LLaMA2 Cross-lingual Consistency

| Language | Accuracy | Consistency with English |
|------|--------|------------|
| English | ~75% | 100% (Baseline) |
| German | ~55% | ~65% |
| Chinese | ~45% | ~55% |
| Arabic | ~35% | ~45% |

### Linear Shortcut Performance

| Configuration | Accuracy | Consistency |
|------|--------|--------|
| Original Model | Baseline | Baseline |
| **Linear Shortcut** | **+5-10%** | **+8-12%** |

### Logit Lens Hierarchy Analysis

| Layers | Representation Space | Description |
|------|---------|------|
| First 1/3 | Language-dependent | Processes syntax/vocabulary of the input language |
| Middle 1/3 | **Language-agnostic** | Concept space, stores factual knowledge |
| Last 1/3 | **Language Transition** | Transitions from the concept space to the target language |

### Key Findings
- **The concept space in middle layers already contains the correct answer**—even if the final prediction is incorrect.
- **Language transition is the critical failure point**: Correct knowledge is not successfully mapped to the target language.
- **Low-resource languages exhibit higher transition failure rates**: Consistent with the training data distribution.
- **The linear shortcut method is effective**: Bypassing the failed transition layers improves both accuracy and consistency.
- **LLaMA2's concept space is biased toward English**: Reflecting its English-centric pre-training.

## Highlights & Insights
- **First to locate the specific failure mechanism of cross-lingual inconsistency**: The concept of "language transition layers" provides a new perspective for understanding multilingual LLMs.
- **Linear shortcut method** is lightweight and effective, requiring no model retraining.
- **The dichotomy of concept space vs. language transition** aligns with the "Mentalese" (internal language) hypothesis in cognitive science.

## Limitations & Future Work
- The linear shortcut is a post-processing method, which does not address the root architectural issue.
- The analysis is limited to LLaMA2 and BLOOM.
- Future directions: Improving the language transition capability of the final layers and implementing multilingual-aware layer-wise training.

## Related Work & Insights
- **vs. Wendler et al. (2024)**: They analyze the multilingual mechanisms of correct predictions, whereas this work focuses on failure cases.
- **vs. CogSteer (Wang et al.)**: CogSteer selects optimal intervention layers based on cognitive findings, while this work identifies the final layers as the main failure point.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First mechanism-level explanation of cross-lingual inconsistency
- Experimental Thoroughness: ⭐⭐⭐⭐ 17 languages + layer-wise analysis + shortcut method
- Writing Quality: ⭐⭐⭐⭐⭐ Exceptionally clear visualization and analysis
- Value: ⭐⭐⭐⭐⭐ Profound impact on multilingual LLM research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cross-Lingual Optimization for Language Transfer in Large Language Models](cross-lingual_optimization_for_language_transfer_in_large_language_models.md)
- [\[ACL 2025\] Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](crosslingual_pitfalls.md)
- [\[ACL 2025\] Bridging the Language Gaps in Large Language Models with Inference-Time Cross-Lingual Intervention](bridging_the_language_gaps_in_large_language_models_with_inference-time_cross-li.md)
- [\[ACL 2025\] Cross-Lingual Generalization and Compression: From Language-Specific to Shared Neurons](cross_lingual_neurons_compression.md)
- [\[ACL 2025\] Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-Lingual Transfer](semantic_aware_linear_transfer_by_recycling_pre-trained_language_models_for_cros.md)

</div>

<!-- RELATED:END -->
