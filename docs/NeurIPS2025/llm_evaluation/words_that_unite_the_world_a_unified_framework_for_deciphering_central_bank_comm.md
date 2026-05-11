---
title: >-
  [Paper Note] Words That Unite The World: A Unified Framework for Deciphering Central Bank Communications Globally
description: >-
  [NeurIPS 2025 (Main Conference)][LLM Evaluation][central bank communication] This paper constructs WCB, the most comprehensive central bank monetary policy corpus to date (380,000+ sentences, 25 central banks, spanning 28 years), defines three NLP tasks (stance detection, temporal classification, uncertainty estimation), and through 15,075 benchmark experiments demonstrates that models trained on aggregated multi-bank data significantly outperform single-bank training, confirming the principle that "the whole is greater than the sum of its parts."
tags:
  - NeurIPS 2025 (Main Conference)
  - LLM Evaluation
  - central bank communication
  - monetary policy
  - stance detection
  - large language models
  - dataset benchmark
date: 2026-05-08
content_hash: 344a8cd201883e5a
---

# Words That Unite The World: A Unified Framework for Deciphering Central Bank Communications Globally

**Conference**: NeurIPS 2025 (Main Conference)  
**arXiv**: [2505.17048](https://arxiv.org/abs/2505.17048)  
**Code**: HuggingFace + GitHub (available, CC-BY-NC-SA 4.0)  
**Area**: LLM Evaluation  
**Keywords**: central bank communication, monetary policy, stance detection, large language models, dataset benchmark

## TL;DR

This paper constructs WCB, the most comprehensive central bank monetary policy corpus to date (380,000+ sentences, 25 central banks, spanning 28 years), defines three NLP tasks (stance detection, temporal classification, uncertainty estimation), and through 15,075 benchmark experiments demonstrates that models trained on aggregated multi-bank data significantly outperform single-bank training, confirming the principle that "the whole is greater than the sum of its parts."

## Background & Motivation

**Background**: Central banks worldwide convey monetary policy signals through policy statements, press conferences, and meeting minutes. The NLP community has produced sentiment and stance analysis work targeting specific central banks (e.g., the Federal Reserve, the European Central Bank), and the financial NLP field has developed specialized models such as FinBERT.

**Limitations of Prior Work**: (1) Existing central bank NLP datasets cover only a single institution or a small number of developed economies, with severe geographic underrepresentation; (2) different studies adopt different annotation schemes and task definitions, making cross-study comparisons impossible; (3) misinterpretation of central bank communications can impose disproportionate economic harm on vulnerable populations, yet systematic evaluation tools remain lacking.

**Key Challenge**: Central bank communications span diverse languages, cultures, and economic contexts. Data from any single bank is too narrow to capture the global diversity of monetary policy expression. Large-scale cross-regional coverage must be achieved while maintaining annotation consistency.

**Goal**: (1) Construct a unified dataset with broad global coverage, long temporal span, and high annotation quality; (2) establish a systematic, multi-model, multi-setting benchmark evaluation; (3) answer the practical question of whether cross-bank aggregated training outperforms single-bank training.

**Key Insight**: One thousand sentences per bank are sampled uniformly by year from 25 central banks (25,000 sentences total), with quality controlled through a three-tier pipeline of dual annotation, disagreement resolution, and expert review.

**Core Idea**: Construct a unified NLP benchmark spanning 25 central banks and 28 years of history, and through 15,075 experiments demonstrate that aggregated multi-bank training significantly outperforms single-bank models.

## Method

### Overall Architecture

The WCB framework comprises three major components: (1) **Data Collection**—policy documents from 25 global central banks spanning 28 years, totaling 380,000+ sentences; (2) **Annotation Pipeline**—25,000 sentences sampled uniformly, each independently annotated by two annotators across three tasks, followed by disagreement resolution and expert review; (3) **Benchmark Evaluation**—large-scale experiments on 7 PLMs and 9 LLMs covering zero-shot, few-shot, and annotation-guide-augmented settings.

### Key Designs

1. **WCB Dataset Construction and Annotation**:

    - Function: Provide high-quality annotated central bank communication data with global coverage
    - Mechanism: 25 central banks span multiple geographic regions including the Americas, Europe, and Asia-Pacific. Year-stratified sampling of 1,000 sentences per bank eliminates temporal bias. Annotation employs a three-tier quality control process: dual annotators, disagreement resolution, and secondary review by domain experts. Each sentence is labeled for all three tasks simultaneously, maximizing data utility.
    - Design Motivation: Uniform sampling prevents data from more active central banks from dominating the corpus; the rigorous annotation pipeline ensures cross-cultural annotation consistency—the central challenge of any large-scale, multi-regional dataset.

2. **Three-Task Definition**:

    - Function: Systematically characterize central bank communications along orthogonal dimensions
    - Mechanism: **Stance Detection**—classifying sentences as dovish (accommodative), hawkish (restrictive), or neutral; **Temporal Classification**—determining whether a sentence addresses past, present, or future economic conditions; **Uncertainty Estimation**—assessing the degree of policy uncertainty expressed in a sentence. The three tasks capture policy direction, temporal perspective, and degree of certainty respectively.
    - Design Motivation: Stance detection alone is insufficient for a comprehensive understanding of central bank communications. The policy orientation, temporal focus, and certainty of a sentence constitute three orthogonal yet complementary dimensions; using them jointly yields a more complete reading of policy signals.

3. **Large-Scale Systematic Benchmark Evaluation**:

    - Function: Comprehensively assess the capabilities of existing NLP models
    - Mechanism: Seven PLMs (BERT, RoBERTa, FinBERT, etc.) are evaluated via fine-tuning; nine LLMs (GPT-4, Llama, etc.) are evaluated under zero-shot, few-shot, and annotation-guide-augmented prompting. The total experiment count reaches 15,075. The core comparison is **aggregated training** (all bank data pooled) versus **single-bank training** (each bank trained independently).
    - Design Motivation: An experiment matrix of this scale yields reliable model capability assessments and avoids the statistical fragility of small-scale evaluations.

### Loss & Training

PLMs follow a standard fine-tuning protocol (classification head + cross-entropy loss). LLMs are evaluated via prompting without parameter updates. The primary evaluation metrics are classification accuracy and F1 score.

## Key Experimental Results

### Main Results

| Dimension | Key Finding |
|-----------|-------------|
| Aggregated vs. single-bank | Aggregated models **consistently and significantly outperform** single-bank models across all three tasks |
| PLM vs. LLM | Fine-tuned PLMs remain competitive on specific tasks, but the strongest LLMs with annotation guides approach or exceed them |
| LLM prompting strategy | Zero-Shot < Few-Shot < Few-Shot + annotation guide |
| Cross-bank generalization | Training on one bank and testing on another degrades performance, but aggregated training substantially mitigates this |
| Economic prediction validation | Model-predicted stances exhibit statistically significant predictive correlations with actual policy rate changes |

### Evaluation Scale Statistics

| Dimension | Data |
|-----------|------|
| Central bank coverage | 25 (spanning the Americas, Europe, Asia-Pacific, and other regions) |
| Raw corpus | 380,000+ sentences |
| Annotated data | 25,000 sentences (high-quality, three-tier review) |
| Historical span | 28 years (approx. 1996–2024) |
| Benchmark experiments | 15,075 |
| PLMs evaluated | 7 |
| LLMs evaluated | 9 |

### Key Findings

- **"The whole is greater than the sum of its parts" is the central finding**: aggregated training consistently outperforms single-bank training, indicating that monetary policy expression shares transferable linguistic regularities across central banks—the "policy grammar" of central banks is more universal than previously assumed.
- LLM few-shot performance improves substantially when annotation guides are provided, suggesting that domain knowledge injection is critical for financial NLP.
- Human evaluation and error analysis confirm the economic utility of the framework: model-predicted stance classifications exhibit a statistically significant predictive relationship with subsequent policy rate adjustments.
- Financial domain pre-trained models such as FinBERT perform strongly on stance detection but show no clear advantage on uncertainty estimation.

## Highlights & Insights

- **Breakthrough contribution in scale and coverage**: 25 central banks, 28 years, 380,000+ sentences—far exceeding any comparable prior dataset. Scale itself is the foundation for scientific discovery, lending statistical reliability to the finding that aggregation outperforms individual training.
- **Rigor of the annotation pipeline**: dual annotation, disagreement resolution, and expert review represent the highest standard in financial NLP. Cross-cultural annotation consistency is the central challenge of large-scale multi-regional datasets; the solution proposed here (unified annotation guidelines combined with domain expert supervision) is transferable to other cross-lingual and cross-cultural annotation tasks.
- **Practical implications of "aggregation beats isolation"**: the direct takeaway for financial NLP practitioners is that even when the target is a single central bank, models should be trained on global central bank data.

## Limitations & Future Work

- All data are in English or English translations; original-language documents (e.g., Bank of Japan statements in Japanese) are not processed directly, potentially losing language-specific policy signals and nuanced expressions.
- Although 25 central banks provide broad coverage, the corpus remains concentrated in G20 and major economies; central banks in Africa, Central Asia, and similar regions are underrepresented.
- The three tasks, while practical, are relatively coarse-grained—finer-grained analyses such as specific policy instrument identification (e.g., quantitative easing, rate adjustment magnitude) or economic indicator extraction are not addressed.
- Dataset currency is a concern: central bank communication styles evolve over time (e.g., the forward guidance style that emerged in the post-financial-crisis era), and a static dataset may not reflect the latest trends.
- Multilingual joint modeling is not explored; training is conducted exclusively on English text.

## Related Work & Insights

- **vs. FedSpeak and other single-bank datasets**: prior work focused on the Federal Reserve or the ECB; WCB expands coverage from 1–2 institutions to 25, representing an order-of-magnitude increase.
- **vs. financial NLP models such as FinBERT/BloombergGPT**: WCB provides a more systematic evaluation platform for assessing these models on policy analysis tasks, rather than evaluating them solely on general financial text.
- **vs. traditional policy analysis methods**: economists have historically relied on manual reading of central bank statements; WCB demonstrates the value of NLP models for analysis at scale, while also revealing that models still fall short in understanding subtle policy signals.

## Rating

- Novelty: ⭐⭐⭐ The primary contribution lies in the dataset and benchmark rather than methodological innovation; however, the scale and systematicity carry intrinsic value as potential field infrastructure.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 15,075 experiments, 7 PLMs + 9 LLMs + multiple prompting strategies + human evaluation + economic prediction validation—exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is articulated clearly; the dataset construction pipeline is well documented.
- Value: ⭐⭐⭐⭐ Significant value for the financial NLP community and policy analysis practice; the finding that aggregation outperforms isolation carries direct practical guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Unified Framework for Provably Efficient Algorithms to Estimate Shapley Values](a_unified_framework_for_provably_efficient_algorithms_to_estimate_shapley_values.md)
- [\[ICCV 2025\] Discontinuity-aware Normal Integration for Generic Central Camera Models](../../ICCV2025/llm_evaluation/discontinuity-aware_normal_integration_for_generic_central_camera_models.md)
- [\[NeurIPS 2025\] Unlocking Transfer Learning for Open-World Few-Shot Recognition](unlocking_transfer_learning_for_open-world_few-shot_recognition.md)
- [\[ICCV 2025\] A Real-world Display Inverse Rendering Dataset](../../ICCV2025/llm_evaluation/a_real-world_display_inverse_rendering_dataset.md)
- [\[CVPR 2026\] Unified Primitive Proxies for Structured Shape Completion](../../CVPR2026/llm_evaluation/unified_primitive_proxies_for_structured_shape_completion.md)

</div>

<!-- RELATED:END -->
