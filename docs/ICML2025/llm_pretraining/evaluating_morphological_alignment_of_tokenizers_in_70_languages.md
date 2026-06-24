---
title: >-
  [Paper Note] Evaluating Morphological Alignment of Tokenizers in 70 Languages
description: >-
  [ICML 2025][LLM Pretraining][Tokenizer Evaluation] This work extends the MorphScore evaluation framework to 70 languages to systematically investigate the correlation between the morphological boundary alignment of tokenizers and downstream task performance. The results show that morphological alignment explains only a minimal amount of performance variance and exhibits a negative correlation, challenging the mainstream assumption that morphologically aligned tokenization ben…
tags:
  - "ICML 2025"
  - "LLM Pretraining"
  - "Tokenizer Evaluation"
  - "Morphological Alignment"
  - "Multilingual NLP"
  - "MorphScore"
  - "BPE"
  - "Language Model Performance"
date: 2026-05-08
content_hash: f6d39eeac62027f4
---

# Evaluating Morphological Alignment of Tokenizers in 70 Languages

**Conference**: ICML 2025  
**arXiv**: [2507.06378](https://arxiv.org/abs/2507.06378)  
**Code**: [GitHub](https://github.com/catherinearnett/morphscore)  
**Area**: LLM Pre-training  
**Keywords**: Tokenizer Evaluation, Morphological Alignment, Multilingual NLP, MorphScore, BPE, Language Model Performance

## TL;DR

This work extends the MorphScore evaluation framework to 70 languages to systematically investigate the correlation between the morphological boundary alignment of tokenizers and downstream task performance. The results show that morphological alignment explains only a minimal amount of performance variance and exhibits a negative correlation, challenging the mainstream assumption that morphologically aligned tokenization benefits model performance.

## Background & Motivation

Tokenization is the first step in language modeling, significantly impacting training efficiency, model performance, and inference costs. However, effectively evaluating tokenizer quality remains an open question.

Existing intrinsic evaluation metrics for tokenizers primarily include:

**Compression**: e.g., fertility (tokens per word) and CTC (total corpus tokens), but prior research shows no robust correlation between compression and performance.

**Rényi Efficiency**: accounts for frequency distribution, but subsequent work argues that it fails to comprehensively measure tokenization quality.

**Morphological Alignment**: measures whether token boundaries align with morpheme boundaries. For instance, the ideal segmentation of the English word "books" is [book + s] rather than [boo + ks].

Existing literature remains highly polarized on whether morphological alignment benefits model performance. Some studies argue that alignment aids performance (Park 2020, Hofmann 2021, etc.), while others find no significant benefit (Macháček 2018, Saleva & Lignos 2021). The original MorphScore only covers 22 languages and suffers from multiple limitations: it excludes high-resource languages like French and German, lacks contextual information, and disregards word frequency.

The core motivation of this paper is to determine more accurately whether morphological alignment truly impacts model performance by substantially expanding language coverage and parameter flexibility.

## Method

### Overall Architecture

The proposed workflow consists of three stages: (1) creating a morphological alignment evaluation dataset for 70 languages based on Universal Dependencies (UD) treebanks; (2) designing a scoring function with various parameter settings; (3) analyzing the correlation between alignment scores and downstream task performance.

### Evaluation Dataset Creation

For each language, multi-morphemic words (excluding mono-morphemic words) are extracted from UD treebanks, and segmentation gold standards are determined using wordforms and lemmas:

- The stem is identified by finding the longest common subsequence between the wordform and the lemma.
- Extra characters before and after the stem are treated as prefixes and suffixes, respectively.
- Only regular forms that can be reconstituted via concatenation are retained (excluding irregular variants and non-concatenative morphology).
- This method is only applicable to **inflecting and agglutinative languages**, and is not suitable for Semitic languages (e.g., Arabic) and isolating languages (e.g., Chinese).

Consequently, datasets for 86 languages were created, retaining 70 after filtering out languages with fewer than 100 entries.

### Scoring Function

The original MorphScore is extended to introduce two classes of metrics: boundary-level and subword-level metrics:

- **Boundary Metrics**: evaluate whether the predicted segmentation correctly identifies morpheme boundaries.
    - Macro-averaged boundary precision and recall.
- **Subword Metrics**: evaluate whether predicted subwords precisely match the gold-standard morphemes.
    - Micro/macro-averaged subword precision, recall, and F1.

Example: For a gold segmentation of [book + s] and a predicted segmentation of [boo + k + s]:
- Boundary Precision = 1/2 (only the k|s boundary is correct), Boundary Recall = 1/1
- Subword Precision = 1/3 (only "s" matches exactly), Subword Recall = 1/2

### Parameter Settings Experiment

**Frequency Weighting**: whether to weight alignment scores by word frequency. Experiments show that high-frequency words are more likely to be segmented with morphological alignment (Spearman $\rho = 0.119$, $p < 0.0001$).

**Single-token Word Handling**: whether to include words that are stored entirely as a single token in the scoring. Including single-token words generally yields higher scores. High-frequency words are more likely to be stored as a single token ($\rho = -0.108$, $p < 0.0001$).

**Optimal Default Settings**: linear mixed-effects model analysis reveals that frequency weighting combined with excluding single-token words offers slightly stronger predictive power for model performance.

### Correlation Analysis with Model Performance

Using the performance of five pre-trained models (Llama2 8B, BLOOM, XGLM 7.5B, Llama3, Gemma3) across seven downstream tasks (XCOPA, XNLI, SIB-200, MultiBLiMP, etc.), linear mixed-effects models are employed to test whether morphological alignment explains additional variance.

Control variables include model parameter size and the proportion of training data for each language. ANOVA is used to test whether morphological alignment provides additional explanatory power.

## Key Experimental Results

### Main Results on Tokenizer Morphological Alignment

| Tokenizer | Recall | Precision |
|-----------|--------|-----------|
| BLOOM | 0.33 ± 0.00 | 0.11 ± 0.00 |
| Gemma3 | 0.35 ± 0.00 | 0.12 ± 0.00 |
| Llama2 | 0.56 ± 0.00 | 0.13 ± 0.00 |
| Llama3 | 0.45 ± 0.00 | 0.12 ± 0.00 |
| XGLM | 0.52 ± 0.00 | 0.23 ± 0.00 |

XGLM consistently performs best in precision; Llama2 achieves the highest recall, but this is primarily driven by over-segmentation.

### Correlation Analysis Results

- When recall is used as the morphological alignment metric, it explains additional variance ($\chi^2(1) = 391.42, p < 0.001$), whereas precision does not ($\chi^2(1) = -6.99, p = 1$).
- The overall explanatory power is extremely low: recall $R^2 = 0.024$, precision $R^2 = 0.005$.
- **The direction of correlation is negative**—higher morphological alignment is associated with lower performance.
- This is consistent with the findings of Arnett & Bergen (2025).

### Over-segmentation and Precision

When using accuracy as a metric, character-level segmentation can achieve perfect scores, which is misleading. Llama tokenizers frequently over-segment non-Latin script languages (segmenting down to the byte level), yielding high recall but low precision. Consequently, this work recommends using precision and recall instead of accuracy.

## Highlights & Insights

1. **Challenging Mainstream Assumptions**: In joint experiments spanning 70 languages, 5 models, and 7 tasks, morphological alignment explains less than 2.5% of performance variance, and in a negative direction. This directly queries the widespread assumption that morphologically aligned tokenization benefits model performance.

2. **The Choice of Evaluation Metrics is Crucial**: Using accuracy to measure morphological alignment severely misleads results (as over-segmentation yields high scores). Precision effectively penalizes over-segmentation and serves as a more reasonable metric.

3. **Morphological Alignment May Need Contextual Metrics**: Morphological alignment alone is insufficient to evaluate tokenization quality. Future work might need to combine it with other metrics like compression and Rényi efficiency.

4. **High Dataset Flexibility**: The new dataset includes context, part-of-speech (POS) information, and morphological annotations, facilitating fine-grained analysis such as POS-specific evaluations.

## Limitations & Future Work

- The language sample remains dominated by European languages; Semitic and isolating languages are excluded due to non-concatenative morphology.
- The operationalization of morphological boundaries is relatively coarse, primarily covering inflectional morphology.
- The number of downstream tasks is limited, and they are mostly concentrated in high-resource languages.
- Only autoregressive LMs are considered, excluding encoder models.
- The model sample is limited since most models do not disclose their training data proportions.

## Related Work & Insights

- **Tokenizer Evaluation**: Rust 2021 (fertility), Schmidt 2024 (CTC), Zouhar 2023 (Rényi efficiency)
- **Morphological Alignment**: Batsuren 2024 (taxonomic evaluation), Arnett & Bergen 2025 (original MorphScore)
- **Multilingual Tokenization Fairness**: Ahia 2023, Petrov 2023

## Rating

⭐⭐⭐ — Large experiment scale and clear conclusions, but the core findings lean toward negative results (morphological alignment is ineffective), and the methodological novelty is limited. The contributions of the dataset and the evaluation framework hold practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unsupervised Morphological Tree Tokenizer](../../ACL2025/llm_pretraining/unsupervised_morphological_tree_tokenizer.md)
- [\[ICML 2025\] Tokenized Bandit for LLM Decoding and Alignment](tokenized_bandit_for_llm_decoding_and_alignment.md)
- [\[ACL 2025\] Splintering Nonconcatenative Languages for Better Tokenization](../../ACL2025/llm_pretraining/splintering_nonconcatenative_languages_for_better_tokenization.md)
- [\[ICLR 2026\] Identifying and Evaluating Inactive Heads in Pretrained LLMs](../../ICLR2026/llm_pretraining/identifying_and_evaluating_inactive_heads_in_pretrained_llms.md)
- [\[NeurIPS 2025\] Composition and Alignment of Diffusion Models using Constrained Learning](../../NeurIPS2025/llm_pretraining/composition_and_alignment_of_diffusion_models_using_constrai.md)

</div>

<!-- RELATED:END -->
