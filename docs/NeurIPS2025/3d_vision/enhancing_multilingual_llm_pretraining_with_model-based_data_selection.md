---
title: >-
  [Paper Note] Enhancing Multilingual LLM Pretraining with Model-Based Data Selection
description: >-
  [NeurIPS 2025][3D Vision][multilingual pretraining] This paper proposes a transparent, simple, and efficient model-based data selection framework for multilingual pretraining. It leverages FastText and Transformer (XLM-RoBERTa) embedding classifiers to identify structured and knowledge-rich samples. On the FineWeb-2 dataset, the framework matches baseline MMLU scores using only 15% of tokens, and is extended to 20 languages with publicly released curated pretraining datasets.
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "multilingual pretraining"
  - "data selection"
  - "model-based filtering"
  - "FineWeb-2"
  - "data curation"
date: 2026-05-08
content_hash: 1e6ec0202f2be803
---

# Enhancing Multilingual LLM Pretraining with Model-Based Data Selection

**Conference**: NeurIPS 2025
**arXiv**: [2502.10361](https://arxiv.org/abs/2502.10361)  
**Code**: [github.com/epfml/fineweb2-hq](https://github.com/epfml/fineweb2-hq)  
**Area**: Multilingual Translation
**Keywords**: multilingual pretraining, data selection, model-based filtering, FineWeb-2, data curation

## TL;DR

This paper proposes a transparent, simple, and efficient model-based data selection framework for multilingual pretraining. It leverages FastText and Transformer (XLM-RoBERTa) embedding classifiers to identify structured and knowledge-rich samples. On the FineWeb-2 dataset, the framework matches baseline MMLU scores using only 15% of tokens, and is extended to 20 languages with publicly released curated pretraining datasets.

## Background & Motivation

Data curation has become a cornerstone of LLM performance. While prior work such as FineWeb-Edu and DCLM has demonstrated the substantial potential of model-based filtering for English data — matching full-dataset training with only 10% of tokens — these advances **remain almost entirely English-centric**. This bias risks widening the LLM performance gap across languages.

The current state of multilingual data curation:
- **FineWeb-2** (SOTA): primarily relies on heuristic filters
- **CCNet**: applies language model perplexity filtering, but is suboptimal
- Systematic model-based multilingual filtering methods are lacking

The paper's core goal is to extend model-based filtering from English to multilingual settings, defining quality in terms of **structured data** and **knowledge-rich samples**. Chinese, German, French, Arabic, and Danish are selected as representative languages, covering diverse language families, writing systems, and resource availability.

## Method

### Overall Architecture

The framework consists of two steps: (1) selecting appropriate classifier training datasets, and (2) training classifiers for document scoring and filtering.

### Key Designs

#### 1. Classifier Training Dataset Construction

Two quality criteria are defined: samples must be informative and well-structured, and datasets must be available in multiple languages.

Five representative datasets are selected as positive sample sources:
- **Aya Collection**: ~514M samples, 101 languages, AI-generated, no quality guarantee but largest scale
- **Aya Dataset**: ~202K samples, 65 languages, human-annotated
- **MMLU**: ~14K multiple-choice questions, original English + professional translations in 14 languages
- **OpenAssistant-2**: ~14K conversations, 28 languages
- **Include-Base-44**: ~23K samples, 44 languages, academic and professional examinations

Two configurations are formed to balance quality and scale:
- **MKC**: Include-Base-44 + OpenAssistant-2 + MMLU + Aya Dataset
- **MKC+**: MKC + Aya Collection

Binary classification datasets are constructed with at most 80K positive samples per language (all samples from smaller datasets plus random samples from Aya Collection), with an equal number of negative samples randomly drawn from FineWeb-2. Datasets are constructed independently per language to prevent language-bias leakage.

#### 2. FastText Filtering (FT)

A binary FastText classifier is trained and runs efficiently on CPU:
- Training features: 2-grams (4-grams for Chinese)
- All documents are scored and thresholded according to a target retention ratio

#### 3. Transformer Embedding Filtering

A pretrained XLM-RoBERTa base model (279M parameters, 100 languages) is used without fine-tuning to preserve general-purpose embeddings.

The first 512 tokens of each document are encoded; mean pooling yields a 768-dimensional embedding. Two methods are applied:

**MLP Classifier**: Single hidden layer (256 units), ReLU + 20% Dropout + Sigmoid, AdamW optimizer (lr=0.0003), trained for 6 epochs with BCE loss.

**Cosine Similarity (CS)**: The maximum cosine similarity between a document embedding and $K=8192$ randomly sampled positive embeddings is computed as the score.

### Loss & Training

- LLM evaluation: 1B-parameter Llama model trained on 70B or 119B tokens
- Optimizer: AdamW with WSD learning rate schedule
- Batch size: 1.6M tokens; learning rate: 0.0008; 2,000 warmup steps
- Tokenizer: Multilingual Mistral v3 (Tekken)
- Model selection: global ranking across languages via FineTasks

## Key Experimental Results

### Main Results

**Method ranking** (averaged across Chinese, German, French, Arabic, and Danish, at 70B + 119B tokens):

| Method | Average Rank |
|--------|-------------|
| **MLP MKC+** | **4.35** |
| MLP MKC | 6.11 |
| FT MKC+ | 7.17 |
| FT MKC | 8.04 |
| CS MKC | 8.10 |
| **Baseline (FineWeb-2)** | **8.72** |
| CS MKC+ | 8.79 |

MLP MKC+ achieves a decisive lead, substantially outperforming the baseline.

**English validation** (119B tokens, compared against DCLM and FineWeb-Edu):

| Dataset | Average Rank |
|---------|-------------|
| **Ours (MLP MKC+)** | **1.83** |
| DCLM | 2.39 |
| FineWeb-Edu | 2.44 |
| FineWeb | 3.33 |

The proposed method also achieves the best overall ranking on English.

**Token efficiency**: On high-resource languages, retaining only 10% of the data suffices to match the baseline at approximately 20B tokens (16.7% of the total). Baseline MMLU scores are matched using only 15% of tokens.

### Ablation Study

**Threshold selection** (Chinese, German, French; MLP/FT; 10%/15%/20%):
- MLP MKC+ achieves the best ranking at 10% retention (8.85)
- MKC training data performs better at higher retention rates (15–20%)

**Impact of training data source** (30B tokens, MLP, 10% retention):
- MKC+ (all data combined) ranks best (2.52)
- Aya Collection alone is also strong (2.91), despite lacking quality guarantees
- Include-Base-44 and OpenAssistant-2 individually underperform the baseline

**Data contamination analysis** (13-gram deduplication):
- Performance drops only slightly after decontamination; the method still substantially outperforms the undecontaminated baseline
- This rules out the possibility that performance gains are primarily attributable to data contamination

### Key Findings

1. **Mitigation of the multilingual curse**: After quality filtering, a multilingual model (5 languages × 119B tokens = 595B tokens) **outperforms** the corresponding monolingual models. On unfiltered data, multilingual training still suffers from the curse. This finding is highly significant.

2. **FastText as a resource-efficient alternative**: Although MLP MKC+ is the strongest, FastText classifiers run on CPU without requiring GPU-based embedding computation, offering a favorable cost-performance trade-off.

3. **Cross-lingual generalizability of the framework**: The approach is effective from high-resource languages (Chinese, German, French) to low-resource ones (Arabic, Danish), supporting diverse writing systems and language families.

## Highlights & Insights

1. **Exceptional data efficiency**: Retaining 10% of data matches or surpasses models trained on 100% of data, yielding enormous resource savings.
2. **Mitigation of the multilingual curse**: Quality filtering transforms multilingual training from a disadvantage into an advantage, a finding with important theoretical and practical implications.
3. **Transparency and reproducibility**: Public release of curated datasets (20 languages), code, and XLM-RoBERTa embeddings advances open science.
4. **Clever design of positive samples**: Multilingual benchmark datasets serve as proxy signals for "high quality," eliminating the need for manual annotation.
5. **Systematic ablation study**: Comprehensive, cross-lingual controlled experiments over classifier types, training data sources, and retention thresholds.

## Limitations & Future Work

1. **Quality definition skewed toward knowledge/structure**: Using benchmark datasets as positive samples may favor academic/exam-style text, not necessarily covering all dimensions of "high quality" (e.g., creative writing, dialogue).
2. **Limited gains on low-resource languages**: Arabic and Danish exhibit relatively high retention rates (56%/65%), with less pronounced filtering gains compared to high-resource languages.
3. **Computational cost of XLM-RoBERTa embeddings**: Computing full embeddings for 20 languages requires approximately 4K H100 GPU hours (though publicly released to amortize this cost).
4. **Evaluation limited to 1B-parameter models**: Whether the same data efficiency advantage holds at larger scales remains to be verified.
5. **Coverage of only 20 languages**: Far more languages are represented on the internet; the framework's behavior on extremely low-resource languages is unknown.

## Related Work & Insights

- **DCLM**: The FastText approach in this paper is directly inspired by DCLM, extending it to multilingual settings.
- **FineWeb-Edu**: Filters by LLM-assessed educational value, but is restricted to English and computationally more expensive.
- **FineWeb-2**: The baseline dataset used in this work, primarily relying on heuristic filtering.
- **Insights**: (1) Model-based filtering is a critical lever for multilingual LLM pretraining; (2) data quality improvements can "cure" the multilingual curse, suggesting that the phenomenon is fundamentally driven by data noise rather than language interference.

## Rating

- Novelty: ⭐⭐⭐ — The methods themselves (FastText/MLP classifiers) are relatively straightforward; the innovation lies in systematically extending them to multilingual settings.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers 5 languages, multiple classifier types, multiple thresholds, decontamination analysis, and multilingual training; extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Clear and systematic, with detailed experimental descriptions.
- Value: ⭐⭐⭐⭐⭐ — Public release of curated datasets in 20 languages offers exceptional practical value; the finding on mitigating the multilingual curse has far-reaching implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] COOPERTRIM: Adaptive Data Selection for Uncertainty-Aware Cooperative Perception](../../ICLR2026/3d_vision/coopertrim_adaptive_data_selection_for_uncertainty-aware_cooperative_perception.md)
- [\[ICCV 2025\] Bootstrap3D: Improving Multi-view Diffusion Model with Synthetic Data](../../ICCV2025/3d_vision/bootstrap3d_improving_multi-view_diffusion_model_with_synthetic_data.md)
- [\[NeurIPS 2025\] SceneForge: Enhancing 3D-text alignment with Structured Scene Compositions](sceneforge_enhancing_3d-text_alignment_with_structured_scene_compositions.md)
- [\[NeurIPS 2025\] Web-Scale Collection of Video Data for 4D Animal Reconstruction](web-scale_collection_of_video_data_for_4d_animal_reconstruction.md)
- [\[NeurIPS 2025\] VA-GS: Enhancing the Geometric Representation of Gaussian Splatting via View Alignment](va-gs_enhancing_the_geometric_representation_of_gaussian_splatting_via_view_alig.md)

</div>

<!-- RELATED:END -->
