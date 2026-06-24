---
title: >-
  [Paper Note] Using Shapley Interactions to Understand How Models Use Structure
description: >-
  [ACL 2025][Shapley interaction] Using the Shapley Taylor Interaction Index (STII) to systematically analyze cross-modality (text and speech) how language models encode syntactic structure, non-compositional semantics, and phonetic coarticulation through non-linear interactions, it is found that autoregressive models significantly outperform masked models in syntactic encoding.
tags:
  - "ACL 2025"
  - "Shapley interaction"
  - "syntactic structure"
  - "multi-word expression"
  - "speech model"
  - "non-linear representation"
date: 2026-05-08
content_hash: fd1279445fadec31
---

# Using Shapley Interactions to Understand How Models Use Structure

**Conference**: ACL 2025  
**arXiv**: [2403.13106](https://arxiv.org/abs/2403.13106)  
**Code**: None  
**Area**: Others  
**Keywords**: Shapley interaction, syntactic structure, multi-word expression, speech model, non-linear representation

## TL;DR

Using the Shapley Taylor Interaction Index (STII) to systematically analyze cross-modality (text and speech) how language models encode syntactic structure, non-compositional semantics, and phonetic coarticulation through non-linear interactions, it is found that autoregressive models significantly outperform masked models in syntactic encoding.

## Background & Motivation

### Background

**Background**: **Background**: Feature attribution methods like Shapley values are important tools for understanding neural networks, but they assume that features are independent and linearly additive, ignoring non-linear interactions. **Limitations of Prior Work**: Existing work on Shapley interactions is limited to older architectures like LSTM and simple classification tasks, failing to extend to modern Transformers and multimodal scenarios. **Key Challenge**: Language data is highly structured, and linear attribution cannot reveal how models encode dependencies within the structures. **Goal**: To verify whether STII can capture the model's encoding of linguistic structures across modalities. **Key Insight**: Associating and analyzing STII with three known linguistic structures (syntax, semantic compositionality, and phonetic coarticulation). **Core Idea**: Structurally closely associated features exhibit stronger non-linear interactions.

## Method

### Overall Architecture

STII is used to measure the intensity of non-linear interactions between pairwise features. Under the condition of controlling for positional distance, the relationships between STII and syntactic distance, multi-word expression attribution, phoneme types, etc., are examined.

### Key Designs

1. **STII Computation and Position Control**:
    - Function: Computes the Shapley Taylor Interaction Index (STII) for pairwise features and controls for positional effects.
    - Mechanism: $\text{STII}_{A,B} = \frac{\| \phi(\emptyset) - \phi(A) - \phi(B) + \phi(A,B) \|_2}{\| \phi(\emptyset) \|_2}$, approximated using Monte Carlo permutation sampling. Stratified control is applied by defining interaction pair distance $d_i$ and prediction distance $d_p$.
    - Design Motivation: STII measures the part of the joint effect that exceeds the sum of independent effects—which is precisely the non-linear structural encoding signal. Stratified control eliminates the confounding of positional effects.

2. **Three-Level Structural Association Analysis**:
    - Function: Associates STII with syntactic structure, non-compositional semantics (MWE), and phonetic coarticulation, respectively.
    - Mechanism: (a) Syntax: spaCy dependency tree + Spearman correlation; (b) Semantics: AMALGrAM labeled strong/weak MWEs, comparing STII differences inside and outside MWEs; (c) Speech: Wav2Vec 2.0 + Montreal Forced Aligner, comparing STII at consonant-vowel vs. consonant-consonant boundaries.
    - Design Motivation: Verification across all three levels proves the value of STII as a general interpretability tool.

3. **Autoregressive vs. Masked Model Comparison**:
    - Function: Compares GPT-2 and BERT-base under the same experiments.
    - Mechanism: Compares the sensitivity of the two training objectives to syntax under identical STII analysis.
    - Design Motivation: To verify whether training objectives lead models to encode syntactic relations in different ways.

### Loss & Training

This is an analytical study that directly analyzes pre-trained models without involving training. Inputs are truncated to 20 tokens, and softmax is applied to the logit outputs to ensure comparability.

## Key Experimental Results

### Main Results

| Experiment | GPT-2 (Autoregressive) | BERT (Masked) |
|------|:---:|:---:|
| Positional Effect | STII decreases monotonically with distance ✓ | STII decreases monotonically with distance ✓ |
| Syntactic Distance vs STII | All significant cells are **negatively correlated** | Inconsistent mix of positive and negative |
| Strong MWE Interaction Enhancement | Strong MWE > Weak MWE > General pair ✓ | Strong MWE > Weak MWE > General pair ✓ |

Speech Model (Wav2Vec 2.0):

| Comparison | Mean STII |
|------|:---:|
| Consonant-Vowel Boundary | **Significantly higher** |
| Consonant-Consonant Boundary | Lower |
| High Sonority Consonant | Higher (similar to Vowel) |
| Low Sonority Consonant | Lower |

### Ablation Study

Positional effect baseline:

| Distance Type | GPT-2 | BERT |
|---------|:---:|:---:|
| $d_i$ ↑ | STII monotonic ↓ | STII monotonic ↓ |
| $d_p$ ↑ | STII sharp ↓ | STII sharp ↓ |

### Key Findings

1. **Autoregressive vs. Masked Difference**: In GPT-2, syntactic distance is consistently negatively correlated with STII, whereas BERT is inconsistent—autoregressive training objectives are more inclined to learn syntax.
2. **Non-compositional Semantics Reflected as Non-linear Interactions**: Strong MWEs (e.g., *kick the bucket*) have stronger interactions than weak MWEs—and this holds true across both models.
3. **Speech Models Capture Coarticulation**: Consonant-vowel interactions are stronger than consonant-consonant interactions, and consonants with higher sonority have higher STII—perfectly aligning with phonetic theories.

## Highlights & Insights

- **Unified Cross-Modality Analysis**: Text + speech, generation + recognition—STII serves as a general interpretability tool.
- **Revealing Deep Impacts of Training Objectives on Structural Encoding**—not differences in performance, but differences in encoding mechanisms.
- **Phonetic Experiments Utilizing the IPA Consonant Chart as Heatmap Layout**—visually presenting phonetic rules.

## Limitations & Future Work

- Only small models like GPT-2/BERT-base are investigated; findings may not generalize to large language models.
- Only pairwise interactions are explored, without investigating the hierarchical structures corresponding to higher-order interactions.
- Correlation rather than causal analysis.

## Related Work & Insights

- **vs. Structural Probe**: Detects linearly extractable structural information; STII detects non-linear encoding—complementary.
- **vs. Saphra & Lopez (2020)**: This work extends to Transformer + multiple linguistic structures + speech.
- **Insights**: Non-linear processing of models is the core missing piece of interpretability—linear analyses only scratch the surface.

## Rating

- Novelty: ⭐⭐⭐⭐ First to systematically associate Shapley interactions with multiple linguistic structures.
- Experimental Thoroughness: ⭐⭐⭐ In-depth analysis but with limited model scale.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical framework and clever experimental design.
- Value: ⭐⭐⭐⭐ Provides a new methodological perspective for NLP interpretability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ProxAnn: Use-Oriented Evaluations of Topic Models and Document Clustering](proxann_topic_model_eval.md)
- [\[ACL 2025\] LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models](latim_measuring_latent_token-to-token_interactions_in_mamba_models.md)
- [\[AAAI 2026\] HyperSHAP: Shapley Values and Interactions for Explaining Hyperparameter Optimization](../../AAAI2026/others/hypershap_shapley_values_and_interactions_for_explaining_hyperparameter_optimiza.md)
- [\[AAAI 2026\] How to Marginalize in Causal Structure Learning?](../../AAAI2026/others/how_to_marginalize_in_causal_structure_learning.md)
- [\[ACL 2025\] The Harmonic Structure of Information Contours](the_harmonic_structure_of_information_contours.md)

</div>

<!-- RELATED:END -->
