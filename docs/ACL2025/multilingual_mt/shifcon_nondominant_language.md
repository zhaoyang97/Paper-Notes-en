---
title: >-
  [Paper Note] ShifCon: Enhancing Non-Dominant Language Capabilities with a Shift-based Multilingual Contrastive Framework
description: >-
  [ACL 2025][Multilingual & Machine Translation][Multilingual LLM] This paper proposes the ShifCon framework, which significantly improves the performance of low-resource languages by shifting representation of non-dominant languages to the subspace of dominant languages to access richer model knowledge, then shifting it back to the original language subspace for generation, combined with multilingual contrastive learning.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Multilingual LLM"
  - "Language Subspace"
  - "Shift Projection"
  - "Contrastive Learning"
  - "Non-Dominant Language"
date: 2026-05-08
content_hash: 52c5a8971b5db57c
---

# ShifCon: Enhancing Non-Dominant Language Capabilities with a Shift-based Multilingual Contrastive Framework

**Conference**: ACL 2025  
**arXiv**: [2410.19453](https://arxiv.org/abs/2410.19453)  
**Code**: None  
**Area**: LLM NLP / Multilingual  
**Keywords**: Multilingual LLM, Language Subspace, Shift Projection, Contrastive Learning, Non-Dominant Language  

## TL;DR

This paper proposes the ShifCon framework, which significantly improves the performance of low-resource languages by shifting representation of non-dominant languages to the subspace of dominant languages to access richer model knowledge, then shifting it back to the original language subspace for generation, combined with multilingual contrastive learning.

## Background & Motivation

**Background**:
   - LLMs have demonstrated powerful multilingual capabilities, but a significant performance gap remains between dominant languages (e.g., English) and non-dominant languages.
   - This gap mainly stems from the severe imbalance among languages in pre-training data (with English data vastly outnumbering others).
   - A common mitigation strategy is to translate dominant language data into non-dominant languages for multilingual supervised fine-tuning (MSFT).

**Limitations of Prior Work**:
   - High-quality data annotation for non-dominant languages is extremely costly.
   - Translation errors tend to propagate in downstream processes.
   - MSFT is limited by data scale, leading to a performance ceiling.
   - Even if intermediate-layer representations appear language-agnostic, LDA visualization reveals that different languages still occupy distinct subspaces.

**Key Challenge**:
   - Most of the model's knowledge is encoded in its parameters in the format of the dominant language, making it difficult for non-dominant language representations to effectively access this knowledge.
   - However, target-language-specific information is mandatory for generating output, meaning the entire processing flow cannot simply be conducted in the dominant language space.

**Goal**:
   - To enhance the performance of non-dominant languages under limited MSFT data conditions by manipulating internal language representations within the model.

**Key Insight**:
   - Starting from the model's internal representation space, leverage language vectors to perform shift operations across language subspaces.
   - Combine this with subspace distance measurement to automatically determine the optimal shift layers.

**Core Idea**:
   - Temporarily routing the representation of non-dominant languages through the dominant language subspace to acquire rich knowledge, then returning to the original language subspace to complete generation.

## Method

### Overall Architecture

ShifCon consists of two core modules:
1. **Shift Projection**: Includes shift-toward (mapping to the dominant language space) and shift-backward (mapping back to the original language space).
2. **Multilingual Contrastive Learning (MCL)**: Enhances the alignment between the shifted representations and the dominant language representations.

### Key Designs

1. **Shift-toward Projection**:

    - **Function**: At layer $L_{\text{to}}$, maps the representation of a non-dominant language $l$ to the dominant language (English) subspace.
    - Core formula: $\tilde{h}_l^{L_{\text{to}}} = h_l^{L_{\text{to}}} - v_l^{L_{\text{to}}} + v_d^{L_{\text{to}}}$
    - That is: subtract the original language vector and add the dominant language vector.
    - The language vector $v_l^i$ is obtained by averaging the sentence representations of that language at the $i$-th layer of the model.
    - **Design Motivation**: By entering the dominant language subspace, the representations of non-dominant languages can better access the knowledge encoded in the model parameters in the dominant language format.

2. **Shift-backward Projection**:

    - **Function**: At layer $L_{\text{bk}}$, maps the dominant-like representation back to the original language subspace.
    - Core formula: $h'_l^{L_{\text{bk}}} = \tilde{h}_l^{L_{\text{bk}}} - v_d^{L_{\text{bk}}} + v_l^{L_{\text{bk}}}$
    - **Design Motivation**: Language-specific information is crucial for generating target-language output and must be restored prior to generation.

3. **Language Subspace Distance**:

    - **Function**: Automatically determines the optimal layer positions for shift-toward and shift-backward.
    - **Mechanism**: Uses a Riemannian distance-based metric to measure the alignment between the dominant-like subspace and the dominant language subspace.
    - Formula: $\text{Dist}(S^{D'}, S^D) = \sqrt{\sum \log^2(\lambda_i)} + \|\mu_{D'} - \mu_{D}\|_2$
    - Perform SVD to obtain the principal directions of each language subspace, then select the continuous layer region with the minimum distance (low subspace distance region).
    - Sorting distances and selecting the top-$\beta\%$ (e.g., $30\%$) reveals that these layers are continuous intermediate layers across different models.

4. **Multilingual Contrastive Learning (MCL)**:

    - **Function**: Further aligns dominant-like representations with their corresponding dominant language representations.
    - **Mechanism**: Employs multilingual translation pairs as positive samples to pull the dominant-like representation of a non-dominant language closer to the dominant language representation, while pushing away other representations.
    - **Design Motivation**: Shift projection alone is insufficient to fully align the representation space; MCL provides a stronger alignment signal.

### Loss & Training

- MCL uses standard contrastive learning loss.
- Training data: A small amount of MSFT data + FLORES translation data for calculating language vectors.
- Important finding: Applying MCL directly to the original representations (instead of the shifted ones) harms language-specific information, thereby degrading target language generation.

## Key Experimental Results

### Main Results

Results on Llama-2 7B (High-resource vs. Low-resource languages):

| Method | MGSM-High | MGSM-Low | FLORES(en→xx)-High | FLORES(en→xx)-Low |
|------|-----------|----------|--------------------|--------------------|
| Base | 35.2 | 5.1 | 33.5 | 15.9 |
| +MSFT | 44.9 | 29.5 | 34.7 | 18.4 |
| +AFP | 46.3 | 31.7 | 35.2 | 19.1 |
| **+ShifCon** | **48.2** | **35.1** | **35.6** | **19.7** |

- Low-resource language MGSM: improved from 5.1 to 35.1 (a relative improvement of +18.9% over the MSFT baseline).
- ShifCon outperforms both MSFT and AFP across all tasks and language settings.

**Other Models** (XGLM 7.5B displays similar improvements)

### Key Findings

1. **Low-resource languages benefit the most**: ShifCon achieves far greater improvements for low-resource languages than for high-resource ones.
2. **Intermediate layers are the optimal shift region**: Through the subspace distance metric, it is observed that the language subspace distance is minimized in the intermediate layers.
3. **Continuous property**: The low subspace distance layers form a continuous block of intermediate layers across different models and scales.
4. **MCL must be performed on shifted representations**: Applying contrastive learning directly to the original representations destroys language-specific information.
5. **$\beta=30\%$ is a preferable choice**: The best performance is achieved when the low subspace distance region covers approximately $30\%$ of the model layers.
6. **Conjecture: Low-distance layers primarily handle information aggregation**: These layers may focus on cross-lingual semantic fusion.

## Highlights & Insights

- **Intuition-driven Elegant Design**: Modeling the representation spaces of different languages as subspaces that can be transformed via vector shift is simple yet effective.
- **Contribution of Subspace Distance Metric**: Provides a principled approach to automatically select the optimal shift layers, avoiding blind searching.
- **Deepened Understanding of LLM's Internal Multilingual Mechanisms**:
    - Even if intermediate layers appear language-agnostic on the surface, they still retain language-specific information in other projection directions.
    - Model knowledge is predominantly encoded in parameters in the dominant language format.
- **High Practical Value**: No extra multilingual data annotation is required; significant improvements can be achieved by utilizing existing MSFT data alone.

## Limitations & Future Work

1. Language vectors are acquired via simple mean pooling; more refined extraction methods for language representations could be more effective.
2. The shift operation assumes a linear relationship among language subspaces (vector addition/subtraction), which may be more complex in practice.
3. MCL employs translation pairs as positive samples, meaning translation quality directly affects the performance of contrastive learning.
4. Mainly validated on 7B-scale models; the effectiveness and optimal hyperparameters on larger models might differ.
5. There is a discrepancy in the improvement magnitude between generation and classification tasks; cross-task robustness remains to be enhanced.
6. Calculating language vectors requires a certain amount of data in the respective language, which may not be feasible for extremely low-resource languages.

## Related Work & Insights

- **Work Related to Language Vectors** (Libovický et al., 2020; Xu et al., 2023; Tang et al., 2024): Language vectors act as effective tools for subspace mapping.
- **MSFT Methods** (Chen et al., 2023; Zhang et al., 2023): Standard multilingual fine-tuning methods, serving as the baseline for this work.
- **LLM Internal Representation Alignment** (Yoon et al., 2024; Li et al., 2024): Improving multilingual performance via aligning internal representations.
- **Insights**: The multilingual capability of LLMs depends not only on data volume but can also be enhanced by manipulating the internal representation space.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of shift projection + subspace distance metric is relatively novel, though the language vector manipulation itself is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multiple models, tasks, and languages (high/low resource), with extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear framework illustrations, intuitive LDA visualizations, and complete mathematical formulations.
- Value: ⭐⭐⭐⭐⭐ — Highly practical value for low-resource multilingual LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models](just_go_parallel_improving_the_multilingual_capabilities_of_large_language_model.md)
- [\[ACL 2025\] Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning](cross-lingual_representation_alignment_through_contrastive_image-caption_tuning.md)
- [\[ACL 2025\] Alleviating Distribution Shift in Synthetic Data for Machine Translation Quality Estimation](alleviating_distribution_shift_in_synthetic_data_for_machine_translation_quality.md)
- [\[AAAI 2026\] How Does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective](../../AAAI2026/multilingual_mt/how_does_alignment_enhance_llms_multilingual_capabilities_a_language_neurons_per.md)
- [\[ACL 2025\] Group then Scale: Dynamic Mixture-of-Experts Multilingual Language Model](group_then_scale_dynamic_mixture-of-experts_multilingual_language_model.md)

</div>

<!-- RELATED:END -->
