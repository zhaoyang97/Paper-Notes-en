---
title: >-
  [Paper Note] FitCF: A Framework for Automatic Feature Importance-guided Counterfactual Example Generation
description: >-
  [ACL 2025][Causal Inference][Counterfactual Generation] This paper proposes the FitCF framework, which leverages BERT-based feature attribution methods (such as LIME/IG/SHAP) to extract important words to guide Large Language Models (LLMs) in generating counterfactual examples under a zero-shot setting (ZeroCF). After filtering through label-flip validation, these examples are utilized as few-shot demonstrations. This approach consistently outperforms three baseline methods (…
tags:
  - "ACL 2025"
  - "Causal Inference"
  - "Counterfactual Generation"
  - "Feature Attribution"
  - "Few-shot Prompting"
  - "Label Flip Validation"
  - "Explainability"
date: 2026-05-08
content_hash: b8cd55fcc1c42f1c
---

# FitCF: A Framework for Automatic Feature Importance-guided Counterfactual Example Generation

**Conference**: ACL 2025  
**arXiv**: [2501.00777](https://arxiv.org/abs/2501.00777)  
**Code**: [https://github.com/qiaw99/FitCF](https://github.com/qiaw99/FitCF)  
**Authors**: Qianli Wang, Nils Feldhus, Simon Ostermann, Luis Felipe Villa-Arenas, Sebastian Möller, Vera Schmitt  
**Affiliations**: TU Berlin, DFKI, Deutsche Telekom, Saarland Informatics Campus  
**Area**: Causal Inference / Explainable AI  
**Keywords**: Counterfactual Generation, Feature Attribution, Few-shot Prompting, Label Flip Validation, Explainability  

## TL;DR

This paper proposes the FitCF framework, which leverages BERT-based feature attribution methods (such as LIME/IG/SHAP) to extract important words to guide Large Language Models (LLMs) in generating counterfactual examples under a zero-shot setting (ZeroCF). After filtering through label-flip validation, these examples are utilized as few-shot demonstrations. This approach consistently outperforms three baseline methods (Polyjuice, BAE, FIZLE) on news classification and sentiment analysis tasks.

## Background & Motivation

**Background**: Counterfactual examples are widely utilized in Natural Language Processing (NLP) for data augmentation and model explainability. Existing automatic generation methods include Polyjuice (fine-tuned GPT-2), BAE (BERT-based mask-and-replace), and FIZLE (LLM zero-shot generation).

**Limitations of Prior Work**:
   - Acquiring counterfactuals via crowdsourcing is expensive and scales poorly.
   - FIZLE relies on the LLM itself to extract important words, which is highly prone to hallucinations. Experiments show that for Llama3-8B on AG News, in **64.5%** of the instances, the important words extracted by the LLM do not exist in the original input at all.
   - Counterfactuals generated in a zero-shot setting exhibit unstable quality and suffer from low label flip rates.

**Core Motivation**: This work coordinates feature attribution (post-hoc explainability) and counterfactual generation (another explainability paradigm) in a complementary manner, using verified structured feature importance to guide LLMs toward generating higher-quality counterfactual instances.

## Method

### 3.1 ZeroCF (Zero-shot Counterfactual Generation)

The core workflow of ZeroCF consists of three steps:

1. **Prediction**: The input $x$ is classified using a BERT model fine-tuned on the target dataset to obtain the predicted label.
2. **Feature Attribution Scoring**: An explainer (Ferret framework) is deployed to compute the importance score of each word in the input $x$ using four feature attribution methods (Gradient, Integrated Gradients, LIME, and SHAP).
3. **Counterfactual Generation**: The top important words $w$ are extracted. The task instructions, important words, predicted label, and original input are formatted into a prompt, which is then sent to the LLM in a zero-shot manner to generate counterfactuals.

**Key Difference from FIZLE**: While FIZLE relies on the LLM to identify important words natively (which is prone to hallucinations), ZeroCF utilizes the feature attribution scores of a BERT model to extract more faithful and robust important words.

### 3.2 FitCF (Few-shot Counterfactual Generation Framework)

Building upon ZeroCF, FitCF introduces three core components:

1. **Top-k Exemplar Sampling**: SBERT is employed to encode all inputs into sentence embeddings. $k$-means clustering is performed, and $c$ samples closest to each cluster centroid are selected to guarantee exemplar diversity.
2. **Label Flip Validation**: The ZeroCF-generated counterfactuals are re-evaluated using the BERT model. Only counterfactuals that successfully trigger a label flip are retained, filtering out invalid outputs.
3. **Few-shot Counterfactual Generation**: The validated input-counterfactual pairs are adopted as demonstrations. These, along with the extracted important words of the target input, are prompted to the LLM in a few-shot manner to generate final counterfactuals.

**Design Choices**:

- Number of exemplars $l = 2k$ (specifically, $l = 10$ for AG News, and $l = 8$ for SST2).
- BERT is simultaneously used as both the feature attribution model and the label-flip validator due to its optimal performance-efficiency trade-off.
- The framework is modular; other classification models including encoder-decoder or decoder-only models can also be substituted, with Inseq utilized to extract attribution scores.

## Key Experimental Results

### Experimental Setup

- **Datasets**: AG News (4-class news classification), SST2 (binary sentiment classification)
- **LLMs**: Llama3-8B, Qwen2.5-32B, Qwen2.5-72B
- **Feature Attribution Methods**: Gradient, Integrated Gradients (IG), LIME, SHAP
- **Evaluation Metrics**: Soft Label Flip Rate (SLFR↑), Perplexity (PPL↓), Textual Similarity (TS↓)

### Main Results (Table 1, AG News + SST2)

| Method | Model | Attribution Method | SLFR(AG)↑ | PPL(AG)↓ | TS(AG)↓ | SLFR(SST2)↑ | PPL(SST2)↓ | TS(SST2)↓ |
|------|------|----------|-----------|----------|---------|-------------|------------|-----------|
| Polyjuice | GPT2 | - | 18.60% | 121.76 | 0.50 | 29.00% | 258.32 | 0.71 |
| BAE | BERT | - | 19.50% | 168.44 | 0.12 | 47.00% | 367.06 | 0.09 |
| FIZLE | Llama3-8B | - | 93.50% | 123.67 | 0.61 | 95.50% | 202.22 | 0.52 |
| ZeroCF | Llama3-8B | SHAP | **98.00%** | **99.08** | 0.27 | 94.00% | **204.76** | 0.46 |
| ZeroCF | Llama3-8B | IG | 95.50% | 109.09 | **0.27** | **99.50%** | 222.51 | **0.42** |
| FitCF | Llama3-8B | LIME | 95.50% | **75.15** | **0.19** | **100.00%** | **151.22** | 0.48 |
| FitCF | Llama3-8B | IG | 96.00% | 87.67 | 0.23 | **100.00%** | 161.88 | 0.48 |
| FIZLE | Qwen2.5-72B | - | 21.50% | 84.09 | 0.22 | 92.00% | 257.91 | 0.43 |
| FitCF | Qwen2.5-72B | Gradient | **77.00%** | 62.13 | 0.99 | **96.00%** | 595.71 | 0.38 |
| FitCF | Qwen2.5-72B | LIME | 45.00% | **61.54** | **0.35** | **96.50%** | 240.94 | 0.41 |

**Key Findings**: FitCF consistently outperforms the three baselines across all LLM and dataset configurations; Llama3-8B (the smallest evaluated model) surprisingly achieves the strongest overall performance.

### Ablation Study (Qwen2.5-72B, Table 2-4)

| Ablation Component | AG News SLFR Change | SST2 SLFR Change | Impact Level |
|----------|-------------------|----------------|----------|
| w/o Important Words | ↓1.96%~35.50% | ↓0%~2.50% | Medium |
| Halved Exemplars ($l=k$) | ↓22.00%~63.50% | ↓1.50%~7.00% | **Largest** |
| w/o Label Flip Validation | ↓1.50%~43.00% | ↓0.50%~2.00% | Medium |

**Key Findings**: The number of exemplars is the most critical driver of FitCF's performance; the pipelines incorporating LIME and SHAP exhibit the strongest robustness.

### Correlation between Faithfulness and Counterfactual Quality (Table 5)

| Attribution Method | AG News comp.↑ | AG News suff.↓ | SST2 comp.↑ | SST2 tau(loo)↑ |
|----------|----------------|----------------|-------------|----------------|
| Gradient | 0.12~0.20 | 0.12~0.13 | 0.20~0.21 | -0.03 |
| IG | 0.32~0.38 | 0.03 | 0.50~0.52 | 0.21~0.22 |
| LIME | 0.53~0.61 | -0.02~-0.01 | 0.67~0.68 | 0.29 |
| SHAP | 0.53~0.62 | -0.02~-0.01 | 0.59~0.60 | 0.25 |

The feature attribution faithfulness of LIME and SHAP significantly outperforms Gradient and IG, showing a strong positive correlation with the final counterfactual quality.

## Highlights & Insights

1. **Complementary Combination of Two Explainability Paradigms**: This work systematically integrates feature attribution and counterfactual generation for the first time, solving the hallucination issue where LLMs misidentify important words.
2. **Fully Automated Pipeline**: From exemplar sampling and ZeroCF generation to label flip validation and few-shot generation, the framework operates autonomously without requiring manually annotated counterfactual data.
3. **Rigorous Ablation Analysis**: Fine-grained ablations on the three core components successfully decouple their individual contributions, revealing that the importance of resources ranks as: exemplar count > feature attribution > label flip validation.
4. **Faithfulness-Quality Correlation**: Shows that the faithfulness of feature attribution scores is strongly positively correlated with downstream counterfactual quality, offering direct guiding principles for future work.

## Limitations & Future Work

1. **Limitation to English**: Evaluation is only performed on English datasets; applicability to multilingual settings remains unverified.
2. **Dependence on Fine-tuned BERT**: Both feature attribution and label flip validation are bound to a task-specific fine-tuned BERT model, without exploring other architectures.
3. **Narrow Task Scope**: The evaluation is restricted to text classification (binary and 4-class), lacking validation on more complex generative tasks (e.g., QA, reasoning).
4. **Over-reliance on Automated Metrics**: Faithfulness is evaluated extensively, but the plausibility and naturalness of the counterfactuals—which typically require human annotation—are not fully addressed.
5. **Trade-off in Edit Distance**: In some configurations (e.g., Qwen2.5-32B on AG News), FitCF yields high edit distances, indicating that the perturbed texts deviate substantially from the original inputs.

## Related Work & Insights

- **Counterfactual Generation**: MICE (Ross et al., 2021), Polyjuice (Wu et al., 2021), DISCO (Chen et al., 2023), Tigtec (Bhan et al., 2023), FIZLE (Bhattacharjee et al., 2024)
- **Combining Explainability Methods**: CREST (Treviso et al., 2023), Krishna et al. (2023), Gressel et al. (2023)
- **Feature Attribution Tools**: Ferret (Attanasio et al., 2023), Inseq (Sarti et al., 2023)
- **Auto-CoT**: Zhang et al. (2023), which conceptually inspired the automated exemplar structure design in FitCF.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | The combination of feature attribution and LLM counterfactual generation is highly sensible, and the pipeline is well-designed. |
| Experimental Thoroughness | 4 | Evaluates 3 LLMs x 4 attribution methods over 2 datasets, coupled with detailed ablation and correlation studies. |
| Writing Quality | 4 | Well-structured, highly informative tables, and logically sound presentation. |
| Practicality | 3 | High system complexity, demanding fine-tuned classifiers, explainer frameworks, and LLM inference. |
| Impact | 3 | Highly solid methodology, though counterfactual generation remains a moderately niche subdomain. |

**Overall Score**: 3.6/5 — A well-executed and comprehensive study. Integrating feature attribution into LLM prompting is a valuable contribution to automatic counterfactual generation. Its main drawback is the narrow focus on conventional classification, lacking validation on more challenging generative tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Parallel Universes, Parallel Languages: A Comprehensive Study on LLM-based Multilingual Counterfactual Example Generation](../../ACL2026/causal_inference/parallel_universes_parallel_languages_a_comprehensive_study_on_llm-based_multili.md)
- [\[ACL 2025\] CausalRAG: Integrating Causal Graphs into Retrieval-Augmented Generation](causalrag_integrating_causal_graphs_into_retrieval-augmented_generation.md)
- [\[ACL 2025\] IRIS: An Iterative and Integrated Framework for Verifiable Causal Discovery](iris_an_iterative_and_integrated_framework.md)
- [\[ICCV 2025\] A Visual Leap in CLIP Compositionality Reasoning through Generation of Counterfactual Sets](../../ICCV2025/causal_inference/a_visual_leap_in_clip_compositionality_reasoning_through_gen.md)
- [\[ACL 2025\] Counterfactual Explanations for Aspect-Based Sentiment Analysis](counterfactual_explanations_for_aspect-based_sentiment_analysis.md)

</div>

<!-- RELATED:END -->
