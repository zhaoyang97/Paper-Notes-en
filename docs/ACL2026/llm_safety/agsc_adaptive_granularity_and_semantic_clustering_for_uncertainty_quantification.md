---
title: >-
  [Paper Note] AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation
description: >-
  [ACL 2026][LLM Safety][Uncertainty Quantification] AGSC proposes an uncertainty quantification framework for long-text generation that triggers adaptive granularity decomposition via NLI neutral probability (reducing inf…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Uncertainty Quantification"
  - "Long-text Generation"
  - "Adaptive Granularity"
  - "Semantic Clustering"
  - "GMM"
date: 2026-05-08
content_hash: f93e2df1c1d49f9e
---

# AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation

**Conference**: ACL 2026  
**arXiv**: [2604.06812](https://arxiv.org/abs/2604.06812)  
**Code**: None  
**Area**: LLM Security  
**Keywords**: Uncertainty Quantification, Long-text Generation, Adaptive Granularity, Semantic Clustering, GMM

## TL;DR

AGSC proposes an uncertainty quantification framework for long-text generation that triggers adaptive granularity decomposition via NLI neutral probability (reducing inference time by $60\%$) and utilizes GMM soft clustering to capture latent semantic themes for theme-aware weighted aggregation, achieving SOTA factuality correlation on BIO and LongFact benchmarks.

## Background & Motivation

**Background**: Hallucination issues in LLMs make uncertainty quantification (UQ) critical for enhancing trustworthiness. Existing UQ methods mainly target short responses, while long-text UQ (e.g., LUQ) attempts to decompose responses into atomic facts for fine-grained evaluation.

**Limitations of Prior Work**: (1) Fine-grained decomposition significantly increases computational overhead; (2) Long texts involve multiple mixed semantic themes, where simple pooling aggregation is overly influenced by minor or off-topic segments; (3) LUQ simply discards NLI neutral labels, yet neutrality often reflects epistemic uncertainty.

**Key Challenge**: Long-text UQ needs to strike a balance between granularity, efficiency, and thematic heterogeneity.

**Goal**: Design an accurate and efficient long-text UQ framework that simultaneously addresses thematic heterogeneity.

**Key Insight**: Utilize the NLI neutral category as a trigger for adaptive granularity and combine GMM soft clustering for theme-aware aggregation.

**Core Idea**: Neutrality is not noise to be discarded but a signal requiring finer granularity analysis; semantic thematic clustering can effectively reduce interference from minor parts on the overall UQ.

## Method

### Overall Architecture

AGSC consists of three stages: (1) **Diversity Generation**—sampling multiple responses; (2) **NLI Calculation and Adaptive Decomposition**—sentence-level NLI analysis where high neutral probability triggers atomic fact decomposition or noise filtering; (3) **Semantic Clustering and Aggregation**—UMAP dimensionality reduction + GMM soft clustering for theme-aware weighted aggregation.

### Key Designs

1.  **Adaptive Granularity Strategy (Adaptive Granularity)**:
    - **Function**: Balances granularity and efficiency.
    - **Mechanism**: Conducts NLI analysis for each sentence; when the neutral probability exceeds a threshold, it triggers finer atomic fact decomposition (indicating the sentence may contain mixed information). Extremely high neutrality leads to filtering as irrelevant information. This avoids expensive atomic decomposition for all sentences.
    - **Design Motivation**: Neutrality can signify either irrelevance (should be filtered) or mixed uncertainty (requires further decomposition); the adaptive trigger mechanism distinguishes between these two cases.

2.  **GMM Semantic Clustering (Semantic Clustering)**:
    - **Function**: Handles thematic heterogeneity in long texts.
    - **Mechanism**: Embeddings of all evaluation units are reduced via UMAP and then processed with GMM for soft clustering, where each cluster corresponds to a latent semantic theme. Theme-aware weights are assigned based on cluster size to down-weight the influence of minor or noisy parts.
    - **Design Motivation**: Different samples of open-ended prompts (e.g., "Tell me about Einstein") may organize content around different themes, leading to structural confusion.

3.  **Thematically-weighted Uncertainty Aggregation**:
    - **Function**: Produces the final uncertainty score.
    - **Mechanism**: Calculates NLI-based uncertainty for each unit first, then performs weighted aggregation based on cluster weights, where major themes contribute higher weights.
    - **Design Motivation**: Prevents minor or off-topic segments from disproportionately affecting the overall UQ score.

### Loss & Training

No model training is involved. Pre-trained NLI and embedding models are used. The number of GMM clusters is automatically selected via BIC.

## Key Experimental Results

### Main Results

- AGSC achieves SOTA correlation with factuality on BIO and LongFact benchmarks.
- Reduces inference time by approximately $60\%$ compared to full atomic decomposition methods.

### Ablation Study

- Both adaptive granularity and semantic clustering components contribute significantly to final performance.
- GMM clustering outperforms K-means hard clustering, as soft assignments better accommodate the fuzzy boundaries of semantic themes.

### Key Findings

- NLI neutrality is a valuable signal and should not be discarded.
- Theme-aware aggregation significantly outperforms simple pooling.
- Adaptive granularity maintains or improves accuracy while reducing computation by $60\%$.

## Highlights & Insights

- Transforming the NLI neutral category from "waste" into a valuable trigger signal is a clever insight.
- GMM soft clustering naturally handles the ambiguity of semantic boundaries.
- The $60\%$ reduction in inference time is statistically significant for practical deployment.

## Limitations & Future Work

- Automatic selection of the number of GMM clusters may be unstable in extreme cases.
- Performance depends on the quality of the NLI model; erroneous NLI judgments will propagate.
- Future work could explore combining AGSC with other UQ methods.

## Related Work & Insights

- Provides systematic solutions to the three limitations of LUQ.
- The GMM clustering approach can be generalized to other NLP tasks requiring the handling of thematic heterogeneity.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of neutrality triggers and semantic clustering is novel and practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparison across two benchmarks and multiple baselines.
- **Writing Quality**: ⭐⭐⭐⭐ The framework description is clear, and the problem motivation is well-justified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Uncertainty Quantification in LLMs is Just Unsupervised Clustering](../../ICML2026/llm_safety/position_uncertainty_quantification_in_llms_is_just_unsupervised_clustering.md)
- [\[ACL 2026\] From Passive Metric to Active Signal: The Evolving Role of Uncertainty Quantification in Large Language Models](from_passive_metric_to_active_signal_the_evolving_role_of_uncertainty_quantifica.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)
- [\[ICLR 2026\] Resource-Adaptive Federated Text Generation with Differential Privacy](../../ICLR2026/llm_safety/resource-adaptive_federated_text_generation_with_differential_privacy.md)
- [\[ACL 2026\] Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization](adaptive_text_anonymization_learning_privacy-utility_trade-offs_via_prompt_optim.md)

</div>

<!-- RELATED:END -->
