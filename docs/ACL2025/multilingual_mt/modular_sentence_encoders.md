---
title: >-
  [Paper Note] Modular Sentence Encoders: Separating Language Specialization from Cross-Lingual Alignment
description: >-
  [ACL 2025][Multilingual & Machine Translation][multilingual sentence encoders] This paper proposes a modular training scheme for multilingual sentence encoders: it first trains language-specific modules (embedding + language adapter + sentence encoder adapter) to alleviate the curse of multilinguality, and then trains cross-lingual alignment adapters using both parallel and paraphrase data to resolve performance trade-offs among different cross-lingual tasks. This approach co…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "multilingual sentence encoders"
  - "modular training"
  - "curse of multilinguality"
  - "cross-lingual alignment"
  - "adapters"
date: 2026-05-08
content_hash: 5f2344ff6b5830e7
---

# Modular Sentence Encoders: Separating Language Specialization from Cross-Lingual Alignment

**Conference**: ACL 2025  
**arXiv**: [2407.14878](https://arxiv.org/abs/2407.14878)  
**Area**: Multilingual Translation  
**Keywords**: multilingual sentence encoders, modular training, curse of multilinguality, cross-lingual alignment, adapters  

## TL;DR

This paper proposes a modular training scheme for multilingual sentence encoders: it first trains language-specific modules (embedding + language adapter + sentence encoder adapter) to alleviate the curse of multilinguality, and then trains cross-lingual alignment adapters using both parallel and paraphrase data to resolve performance trade-offs among different cross-lingual tasks. This approach consistently outperforms monolithic model training across 4 tasks and 23 languages.

## Background & Motivation

- Multilingual Sentence Encoders (MSE) map sentences of different languages into a shared semantic space, widely used in cross-lingual retrieval, clustering, and classification.
- **Two Core Problems**:
  1. **Curse of Multilinguality (CoM)**: Parameter sharing leads to a degradation in the quality of monolingual representations for each language, which is particularly severe for low-resource languages.
  2. **Performance Trade-offs Across Tasks**:
     - Cross-lingual alignment training destroys monolingual semantic structures $\rightarrow$ Monolingual vs. Cross-lingual performance conflict.
     - Training with parallel data suits bitext mining but is ill-suited for semantic similarity $\rightarrow$ Conflict between different cross-lingual tasks.
     - Training with paraphrase data suits semantic similarity but is ill-suited for bitext mining.
- Existing modular approaches (e.g., LASER3) only solve a subset of these problems, and their teacher models themselves are already affected by CoM.

## Method

### Overall Architecture

Three-step modular training (only the parameters of the corresponding modules are updated in each step):

1. **Language Adaptation (LA)**: Train a language-specific embedding layer + LoRA language adapter for each language.
2. **Sentence Encoding (SE) Training**: Stack a LoRA SE adapter on top of the language adapter and train it using monolingual paraphrase data.
3. **Cross-Lingual Alignment (CLA)**: Train a parallel adapter by alternating between cross-lingual paraphrase data and parallel data.

### Key Designs

**Language Adaptation**:
- Train a dedicated tokenizer for each language.
- Initialize new embeddings using the FOCUS method (copy existing token embeddings, and use similar token interpolation for new tokens).
- Perform continuing MLM training using only LoRA for parameter efficiency.

**Sentence Encoding Retraining**:
- The MLM objective degrades the sentence encoding capability of the pretrained MSE, which necessitates SE retraining.
- Train with MNRL (Multiple Negatives Ranking Loss) on **monolingual paraphrase data obtained via machine translation**.
- Freeze the LA module and only update the SE adapter.

**Cross-Lingual Alignment**:
- Perform bilingual alignment with English as the pivot language (English embeddings have the highest quality, trained on gold-standard paraphrase data).
- **Alternating Training**: Use cross-lingual paraphrase pairs + MNRL loss in one batch, and parallel pairs + cosine similarity loss in the next batch.
- Use Parallel Adapters to prevent alignment training from interfering with monolingual SE capabilities.
- Do not train a CLA adapter for English: force other languages to project into the English space.

**Training Data**:
- Translate 5 English paraphrase datasets (MNLI, SentenceCompression, etc., totaling ~600k pairs) into 22 languages using NLLB 3.3B.
- Leverage multilingual parallel paraphrase data to construct both paraphrase pairs and parallel pairs simultaneously.

## Key Experimental Results

### Main Results

Results based on LaBSE (23 languages):

**Monolingual Tasks**:

| Model Config | STS ↑ | STR ↑ | Classification ↑ |
|---------|-------|-------|--------|
| LaBSE Original | 76.7 | 69.2 | 82.7 |
| Full_mc (Best Monolithic) | 80.0 | 75.4 | 86.0 |
| Mod_mc-jt (Ours) | **83.9** | **79.0** | **86.4** |

**Cross-Lingual Tasks**:

| Model Config | STS ↑ | Classification ↑ | FLORES Mining ↓ | Tatoeba Mining ↓ |
|---------|-------|--------|-------------|--------------|
| LaBSE Original | 74.5 | 83.6 | 0.14 | 3.87 |
| Full_c (Best Cross-Lingual Monolithic) | 77.8 | 85.3 | 0.20 | 4.00 |
| Mod_mc-jt (Ours) | **81.4** | **86.7** | **0.10** | **3.12** |

**Alignment Metrics**:
- Language Bias (lower is better): Mod_mc-jt achieves 0.49 on STSB (vs. Full_mc 0.53) and 0.65 on SICK (vs. 0.64).
- RSIM (higher is better): Mod_mc-jt reaches 0.79 (vs. Full_mc 0.77).

### Key Findings

- **Substantial Improvement in Monolingual Performance**: The modular scheme improves by 3.9 percentage points over the strongest monolithic model on STS (83.9 vs. 80.0).
- **Comprehensive Cross-Lingual Superiority**: Achieves SOTA simultaneously in STS, classification, and bitext mining, resolving the trade-offs between tasks.
- **Low-Resource Languages Benefit Most**: Language-specific modules effectively alleviate CoM.
- **MT Data is Effective**: High-quality training is achievable using only machine-translated paraphrase data.
- **Alternating Paraphrase and Parallel Training for CLA is Optimal**: Using only paraphrases (Mod_mc-pp) performs poorly on bitext mining, while using only parallel data (Mod_mc-pl) performs poorly on STS.
- **Equally Effective on mE5**: The modular scheme brings consistent improvements on the mE5 base as well.

## Highlights & Insights

- Clearly decomposes multiple conflicts in MSE training: monolingual vs. cross-lingual, STS vs. bitext mining vs. classification.
- Elegantly resolves conflicts through parameter isolation (modularity), where each module focuses on a single responsibility.
- The bilingual alignment strategy with English as the pivot is simple and effective, avoiding the training overhead of $N^2$ language pairs.
- Demonstrates that MT data can fully replace human-annotated paraphrase data for MSE training, substantially lowering data acquisition costs.
- Combining FOCUS embedding initialization and LoRA makes language adaptation both parameter-efficient and sample-economical.

## Limitations & Future Work

- The language adaptation step (tokenizer training + MLM continuing training) requires computational resources for each language, incurring costs when scaling to hundreds of languages.
- Inferencing requires knowing the input language to activate the corresponding module (can be solved via language identification but increases latency).
- Validated only on LaBSE and mE5-base; larger models (e.g., mE5-large) have not been tested.
- CLA relies solely on English as a pivot, introducing a strong dependency on the quality of English embeddings.
- The batch ratio of alternating training (paraphrase vs. parallel) was not systematically tuned.

## Related Work & Insights

- Multilingual Sentence Encoding: LaBSE (Feng et al., 2022), mE5 (Wang et al., 2024), LASER3 (Heffernan et al., 2022)
- Alleviating the Curse of Multilinguality: Language adapters (Pfeiffer et al., 2020, 2021), FOCUS (Dobler & de Melo, 2023)
- Parameter-Efficient Fine-Tuning: LoRA (Hu et al., 2022), Parallel Adapter (He et al., 2022)
- Contrastive Learning for Sentence Embeddings: SimCSE (Gao et al., 2021), mSimCSE (Wang et al., 2022)
- Cross-Lingual Alignment: Reimers & Gurevych (2020), Artetxe & Schwenk (2019)

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to systematically utilize a modular approach to resolve both CoM and task-related trade-offs in MSE.
- **Technical Depth**: ⭐⭐⭐⭐ — Rigorous three-step modular design, with each step backed by clear motivation and experimental validation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Highly comprehensive, spanning 4 tasks $\times$ 23 languages $\times$ 2 bases $\times$ multiple variant comparisons.
- **Clarity**: ⭐⭐⭐⭐ — Well-structured, supported by clear diagrams, and follows a systematic naming convention for model variants.
- **Impact**: ⭐⭐⭐⭐ — Offers direct guidance to the multilingual NLP community with a generalizable scheme.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning](cross-lingual_representation_alignment_through_contrastive_image-caption_tuning.md)
- [\[ACL 2025\] Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)
- [\[ACL 2025\] Implicit Cross-Lingual Rewarding for Efficient Multilingual Preference Alignment](implicit_cross-lingual_rewarding_for_efficient_multilingual_preference_alignment.md)
- [\[ACL 2025\] Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](crosslingual_pitfalls.md)
- [\[ACL 2025\] Cross-Lingual Optimization for Language Transfer in Large Language Models](cross-lingual_optimization_for_language_transfer_in_large_language_models.md)

</div>

<!-- RELATED:END -->
