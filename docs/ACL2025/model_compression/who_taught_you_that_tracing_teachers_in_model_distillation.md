---
title: >-
  [Paper Note] Who Taught You That? Tracing Teachers in Model Distillation
description: >-
  [Model Compression] This paper introduces a novel problem of "teacher model attribution": given a distilled student model, can its training teacher be identified from a pool of candidate teachers? It is found that n-gram similarity and perplexity are unreliable, whereas Part-of-Speech (PoS) syntactic templates provide effective signals for teacher identification.
tags:
  - "Model Compression"
date: 2026-05-08
content_hash: 02b292ad4e071c6e
---

# Who Taught You That? Tracing Teachers in Model Distillation

| Conference | arXiv | Code | Area | Keywords |
|------|-------|------|------|--------|
| ACL 2025 | 2502.06659 | - | Model Compression / LLM Safety | Knowledge Distillation, Teacher Attribution, Syntactic Templates, PoS Tags, Model Provenance |

## TL;DR

This paper introduces a novel problem of "teacher model attribution": given a distilled student model, can its training teacher be identified from a pool of candidate teachers? It is found that n-gram similarity and perplexity are unreliable, whereas Part-of-Speech (PoS) syntactic templates provide effective signals for teacher identification.

## Background & Motivation

**Research Question:** In the context of model distillation (using large models to train small models), is it possible to infer the teacher model by analyzing the output of the student model?

**Limitations of Prior Work:** 
- Model distillation has become the mainstream method for training efficient small models using large proprietary LLMs.
- Distillation may violate the terms of service of model providers (e.g., the controversy over whether DeepSeek distilled ChatGPT).
- There is a lack of effective methods to detect unauthorized distillation behaviors.
- Existing data provenance methods (such as watermarking) require embedding at generation time and cannot be detected post-hoc.

**Core Motivation:** LLM providers require tools to identify unauthorized distillation use. Furthermore, understanding the transfer of "linguistic fingerprints" from teachers to students helps elucidate the mechanisms of knowledge distillation.

## Method

### Overall Architecture

Three teacher attribution strategies are systematically compared:
1. **Perplexity Method:** Computes the perplexity of candidate teachers on the student's output, with the expectation that the true teacher will yield a lower perplexity.
2. **Similarity Method:** Measures the text similarity between the student's and candidate teachers' outputs.
3. **Syntactic Template Method:** Trains a classifier based on Part-of-Speech (PoS) sequence patterns to identify the teacher.

### Key Designs

**Experimental Setup:**
- **Student Models:** GPT-2 (124M) and OLMo-1B
- **Candidate Teachers $\mathcal{M}$:** {Llama3-8B, Llama3-70B, Mistral-7B, Mixtral, Gemma2-9B}, all of which are open-source models.
- **Tasks:** Summarization (CNN-DailyMail, Rotten Tomatoes, PubMed), Question Answering (OpenbookQA, CommonsenseQA), Instruction Following (Alpaca 10K).

**PoS Template Method:**
- Extracts PoS templates of length 4 using the `diversity` package.
- Selects the top 50 most common PoS patterns across all teacher outputs.
- Constructs PoS template indicator features (a 50-dimensional binary vector).
- Trains a logistic regression classifier (5-class), trained on teacher data and tested on student data.

**Core Hypothesis:** Student models internalize the syntactic preferences of their teachers during distillation. These high-level linguistic structural features are more discriminative than surface lexical similarity.

### Evaluation Metrics

Classification accuracy (5 classes, random baseline of 0.20), BoW cosine similarity, BERTScore, and AUC-ROC.

## Experiments

### Main Results

| Student Model | Feature Type | C-D | P-M | R-T | CSQA | OBQA | QRe | Alpaca |
|----------|----------|-----|-----|-----|------|------|-----|--------|
| GPT-2 | BERT | 0.46 | 0.55 | 0.40 | 0.44 | 0.38 | 0.35 | 0.51 |
| GPT-2 | n-grams | 0.58 | 0.68 | 0.44 | 0.56 | 0.48 | 0.50 | 0.56 |
| GPT-2 | **PoS Templates** | **0.60** | **0.71** | **0.54** | **0.69** | **0.51** | **0.59** | 0.55 |
| OLMo-1B | BERT | 0.45 | 0.65 | 0.41 | 0.40 | 0.42 | 0.31 | 0.46 |
| OLMo-1B | n-grams | 0.60 | 0.62 | 0.48 | 0.55 | 0.42 | 0.58 | 0.50 |
| OLMo-1B | **PoS Templates** | **0.61** | **0.74** | 0.45 | **0.59** | 0.43 | **0.61** | 0.53 |

> 5-class classification accuracy, with a random baseline of 0.20. PoS templates outperform n-gram and BERT features on most datasets.

### Ablation Study

| Method | Effect |
|------|------|
| Perplexity | Teacher perplexity cannot reliably distinguish (the true teacher does not necessarily yield the lowest PPL) |
| BoW + BERTScore Similarity | AUC $\approx$ 0.49-0.53, close to random, lacking discriminative power |
| Logistic Regression + Similarity Features | AUC $\approx$ 0.52, almost no discriminative power |
| PoS Templates (Core Method) | Significantly outperforms random, reaching 0.69 on CSQA, but is still far from perfect |

### Key Findings

- **Surface Similarity Fails:** BoW and BERTScore cannot distinguish students taught by different teachers, demonstrating that distillation does not transfer surface lexical patterns.
- **Perplexity Also Fails:** Teacher models do not always prefer the output of their own student models (e.g., Gemma actually assigns a higher PPL to its own distilled student).
- **Syntactic Templates Work:** PoS templates capture higher-level syntactic structural preferences, which are retained by students during distillation.
- **Task Dependency:** PoS templates perform best on reasoning tasks (CSQA: 0.69) and show smaller gains on instruction-following tasks (Alpaca: 0.55).
- **Accuracy, while far exceeding random choice, remains far from perfect**, indicating that while teacher fingerprints exist, they are not strong enough, necessitating further improvements for practical applications.

## Highlights & Insights

- Introduces a novel and practically valuable problem—post-hoc attribution of distilled teacher models.
- Systematically rules out intuitively plausible approaches (perplexity, text similarity), highlighting the challenging nature of the problem.
- The effectiveness of PoS templates reveals that distillation transfers implicit preferences at the syntactic level rather than surface-level lexical features.
- Requires no access to the internal states of the teacher model and operates without watermarking strategies.

## Limitations & Future Work

- Although the classification accuracy of PoS templates is far above random, it is still far from practical utility (maximum of 0.74).
- Assumes a closed-set scenario (the true teacher must be in the candidate set) and cannot handle out-of-candidate-set teachers.
- Additional fine-tuning, data augmentation, or multi-teacher distillation might blur the attribution signals.
- Different teachers trained on the same data might share footprints, increasing the difficulty of attribution.
- Only two student models (GPT-2, OLMo-1B) are analyzed, and generalizability remains to be verified.

## Related Work & Insights

- **LLM Distillation:** Knowledge distillation framework by Hinton (2015); Reasoning tutoring by Ho et al. (2023); Symbolic CoT distillation by Li et al. (2023b); CoT-enhanced distillation by Wadhwa et al. (2024a).
- **Provenance Tracking:** Statistical watermark detection by Li et al. (2024b); Generation-time watermarking methods by Li et al. (2024a); and text source detection using perplexity and contrastive training by Li et al. (2023a).
- **LLM Text Detection:** Shaib et al. (2024b) find that LLMs prefer specific syntactic templates (which serves as the direct inspiration for this work).

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Variance-Based Pruning for Accelerating and Compressing Trained Networks](../../ICCV2025/model_compression/variance-based_pruning_for_accelerating_and_compressing_trained_networks.md)
- [\[ICML 2025\] Come Together, But Not Right Now: A Progressive Strategy to Boost Low-Rank Adaptation](../../ICML2025/model_compression/come_together_but_not_right_now_a_progressive_strategy_to_boost_low-rank_adaptat.md)
- [\[ICLR 2026\] UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation](../../ICLR2026/model_compression/uniflow_a_unified_pixel_flow_tokenizer_for_visual_understanding_and_generation.md)
- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[ICLR 2026\] S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](../../ICLR2026/model_compression/s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)

</div>

<!-- RELATED:END -->
