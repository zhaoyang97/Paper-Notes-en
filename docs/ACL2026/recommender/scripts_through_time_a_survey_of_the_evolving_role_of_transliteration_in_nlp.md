---
title: >-
  [Paper Note] Scripts Through Time: A Survey of the Evolving Role of Transliteration in NLP
description: >-
  [ACL 2026][Recommender Systems][transliteration] This paper presents a systematic survey of the evolving role of transliteration in cross-lingual NLP. It proposes a five-category motivation taxonomy (named entity/OOV han…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "transliteration"
  - "cross-script transfer"
  - "romanization"
  - "script barrier"
  - "multilingual language models"
date: 2026-05-08
content_hash: dd3a85cd8e0c3121
---

# Scripts Through Time: A Survey of the Evolving Role of Transliteration in NLP

**Conference**: ACL 2026
**arXiv**: [2604.18722](https://arxiv.org/abs/2604.18722)
**Code**: None
**Area**: Recommender Systems
**Keywords**: transliteration, cross-script transfer, romanization, script barrier, multilingual language models

## TL;DR

This paper presents a systematic survey of the evolving role of transliteration in cross-lingual NLP. It proposes a five-category motivation taxonomy (named entity/OOV handling, code-mixing, cross-script similarity exploitation, English-centric transfer, and unified preprocessing), compares six integration strategies, and discusses whether transliteration remains necessary in the era of modern LLMs.

## Background & Motivation

**State of the Field**: Cross-lingual transfer is a central driver of multilingual NLP, yet the "script barrier"—the absence of lexical overlap across different writing systems—severely hinders knowledge transfer between languages. Transliteration, the technique of converting one writing system into another, has emerged as a practical solution to mitigate cross-script incompatibility.

**Limitations of Prior Work**: (1) Transliteration research is scattered across disparate motivations and methodologies, lacking a unified taxonomic framework; (2) there is no systematic summary of when transliteration is beneficial versus harmful (e.g., romanizing logographic scripts such as Chinese leads to semantic information loss); (3) whether large-scale pretraining in the LLM era has rendered transliteration unnecessary warrants re-examination.

**Root Cause**: Transliteration promotes transfer by increasing lexical overlap, but may simultaneously discard critical information such as semantics and morphology. Clear conditions under which transliteration yields net benefit must be established.

**Paper Goals**: To provide a comprehensive taxonomic framework guiding researchers in selecting the most appropriate transliteration strategy given a specific language, task, and resource setting.

**Starting Point**: A two-dimensional taxonomy is constructed along motivation (why) and method (how), tracing the technical evolution from the statistical MT era to the LLM era.

**Core Idea**: Transliteration is not merely a preprocessing technique but a bridge enabling knowledge transfer across writing systems. Its effectiveness is determined by the interaction among language relatedness, script type, and downstream task characteristics.

## Method

### Overall Architecture

The survey covers 50+ papers organized along the following dimensions: (1) five motivation categories—why transliteration is used; (2) six integration strategies—how transliteration is incorporated into NLP pipelines; (3) conditional analysis—when transliteration is beneficial; (4) positioning in the LLM era—whether modern large models still require transliteration.

### Key Designs

1. **Five-Category Motivation Taxonomy**:

    - **Function**: Systematically understanding the role of transliteration in NLP.
    - **Mechanism**: (1) Named entity and OOV handling—the earliest application; (2) code-mixed text processing—handling multi-script mixing within a single document; (3) cross-script linguistic similarity exploitation—for related languages written in different scripts (e.g., Hindi/Urdu); (4) English-centric transfer—leveraging English-centric pretrained models; (5) unified preprocessing—reducing vocabulary size in multilingual models.
    - **Design Motivation**: Existing literature organizes work by method or task, overlooking the temporal evolution of motivations. A motivation-based taxonomy reveals a paradigm shift from "patching" to "systematic design."

2. **Comparison of Six Integration Strategies**:

    - **Function**: Guiding practitioners in selecting the most suitable transliteration strategy.
    - **Mechanism**: (1) full replacement with transliterated data; (2) transliteration augmentation (retaining the original alongside the transliterated version); (3) vocabulary augmentation; (4) embedding concatenation/fusion; (5) prompting strategies (incorporating transliteration into in-context learning); (6) multi-encoder/ensemble architectures.
    - **Design Motivation**: The six strategies differ substantially in complexity, effectiveness, and applicability—simple replacement may be the most effective yet carries the highest risk.

3. **Re-examination in the LLM Era**:

    - **Function**: Assessing whether modern large-scale pretraining renders transliteration redundant.
    - **Mechanism**: Even in the LLM era, (1) tokenizer coverage for low-resource scripts remains insufficient (a single word may be segmented into 10+ tokens); (2) romanization can substantially improve inference efficiency by reducing token counts; (3) for scripts with minimal training data, transliteration remains the most practical solution.
    - **Design Motivation**: To counter the misconception that "LLMs solve everything"—the tokenization bottleneck persists in multilingual LLMs.

### Loss & Training

Not applicable; this is a survey paper.

## Key Experimental Results

### Main Results

The survey does not include original experiments, but summarizes the following key findings:

| Condition | Effect of Transliteration | Reason |
|-----------|--------------------------|--------|
| Related languages with different scripts | Strongly positive | Maximizes lexical overlap gain |
| Logographic scripts (e.g., Chinese) | Negative | Transliteration discards semantic information |
| Code-mixed text | Positive | Unified script reduces distributional mismatch |
| Poor LLM tokenizer coverage | Positive | Reduces token fragmentation |

### Key Findings

- Transliteration is most effective when related languages use different scripts (e.g., Devanagari Hindi → romanized form aligned with Urdu).
- Romanization offers practical value for LLM inference efficiency—romanizing low-resource languages can reduce token counts by a factor of 2–5.
- The primary risk of transliteration is information loss, particularly for tonal languages and logographic scripts.
- Augmenting training with both transliterated and original text is generally safer than pure replacement.

## Highlights & Insights

- Organizing the survey around the temporal evolution of motivations is an insightful perspective—the shift from "patching" to "systematic design" reflects the maturation of the field.
- The argument that transliteration remains necessary in the LLM era is compelling—the tokenization bottleneck is an easily overlooked yet practically significant issue.
- The concrete practical recommendations constitute a key contribution of the survey—researchers can directly consult the taxonomy to select strategies based on language, task, and resource conditions.

## Limitations & Future Work

- The survey's scope emphasizes romanization/latinization, with comparatively limited discussion of non-Latin target scripts.
- The treatment of transliteration quality (tool selection, impact of errors) lacks sufficient depth.
- A quantitative meta-analysis is absent.

## Related Work & Insights

- **vs. multilingual pretraining surveys**: Those focus on model architecture and training strategies, whereas this paper targets data/input-level transliteration interventions.
- **vs. code-mixing surveys**: Code-mixing surveys cover the full phenomenon; this paper focuses on cross-script conversion as an orthogonal dimension.

## Rating

- Novelty: ⭐⭐⭐ The organizational framework is valuable, though no new methods are proposed.
- Experimental Thoroughness: ⭐⭐⭐ No original experiments, as expected of a survey.
- Writing Quality: ⭐⭐⭐⭐ The taxonomy is clear and the summary tables are useful.
- Value: ⭐⭐⭐⭐ A practically useful reference for multilingual NLP researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VisualLens: Personalization through Task-Agnostic Visual History](../../NeurIPS2025/recommender/visuallens_personalization_through_task-agnostic_visual_history.md)
- [\[ICLR 2026\] ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation](../../ICLR2026/recommender/propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis.md)
- [\[NeurIPS 2025\] MMPB: It's Time for Multi-Modal Personalization](../../NeurIPS2025/recommender/mmpb_its_time_for_multi-modal_personalization.md)
- [\[NeurIPS 2025\] Inference-Time Reward Hacking in Large Language Models](../../NeurIPS2025/recommender/inference-time_reward_hacking_in_large_language_models.md)
- [\[NeurIPS 2025\] TV-Rec: Time-Variant Convolutional Filter for Sequential Recommendation](../../NeurIPS2025/recommender/tv-rec_time-variant_convolutional_filter_for_sequential_recommendation.md)

</div>

<!-- RELATED:END -->
