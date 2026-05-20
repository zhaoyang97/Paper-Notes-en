---
title: >-
  [Paper Note] AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation
description: >-
  [ACL 2026][LLM Safety][Uncertainty Quantification] AGSC proposes an uncertainty quantification framework for long-text generation that uses NLI neutral probability to trigger adaptive granularity decomposition (reducing…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Uncertainty Quantification"
  - "Long-text Generation"
  - "Adaptive Granularity"
  - "Semantic Clustering"
  - "GMM"
date: 2026-05-08
content_hash: a6f3a38e6ec7352a
---

# AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation

**Conference**: ACL 2026
**arXiv**: [2604.06812](https://arxiv.org/abs/2604.06812)  
**Code**: None  
**Area**: LLM Safety
**Keywords**: Uncertainty Quantification, Long-text Generation, Adaptive Granularity, Semantic Clustering, GMM

## TL;DR

AGSC proposes an uncertainty quantification framework for long-text generation that uses NLI neutral probability to trigger adaptive granularity decomposition (reducing inference time by 60%) and employs GMM soft clustering to capture latent semantic topics for topic-aware weighted aggregation, achieving state-of-the-art factuality correlation on the BIO and LongFact benchmarks.

## Background & Motivation

**Background**: The hallucination problem in LLMs makes uncertainty quantification (UQ) a critical technique for enhancing trustworthiness. Existing UQ methods primarily target short responses, while long-text UQ approaches (e.g., LUQ) attempt to decompose responses into atomic facts for fine-grained evaluation.

**Limitations of Prior Work**: (1) Fine-grained decomposition substantially increases computational overhead; (2) Long texts mix multiple semantic topics, and simple pooling aggregation is overly influenced by secondary or off-topic content; (3) LUQ naively discards NLI neutral labels, yet neutrality often reflects epistemic uncertainty.

**Key Challenge**: Long-text UQ must balance granularity, efficiency, and topical heterogeneity.

**Goal**: Design an accurate and efficient long-text UQ framework that simultaneously handles topical heterogeneity.

**Key Insight**: Leverage the NLI neutral category as an adaptive granularity trigger, combined with GMM soft clustering for topic-aware aggregation.

**Core Idea**: Neutrality is not noise to be discarded but a signal calling for finer-grained analysis; semantic topic clustering effectively reduces the interference of secondary content on overall UQ.

## Method

### Overall Architecture

AGSC consists of three stages: (1) **Diverse Generation** — sampling multiple responses; (2) **NLI Computation and Adaptive Decomposition** — sentence-level NLI analysis, where sentences with high neutral probability trigger atomic fact decomposition or noise filtering; (3) **Semantic Clustering and Aggregation** — UMAP dimensionality reduction followed by GMM soft clustering for topic-weighted aggregation.

### Key Designs

1. **Adaptive Granularity**:

    - Function: Balance granularity and efficiency.
    - Mechanism: Each sentence undergoes NLI analysis; when the neutral probability exceeds a threshold, finer-grained atomic fact decomposition is triggered (indicating the sentence may contain mixed information). Sentences with extremely high neutrality rates are filtered as irrelevant. This avoids expensive atomic decomposition for every sentence.
    - Design Motivation: Neutrality may indicate irrelevance (to be filtered) or mixed uncertainty (to be further decomposed); the adaptive trigger mechanism distinguishes between these two cases.

2. **GMM Semantic Clustering**:

    - Function: Handle topical heterogeneity in long texts.
    - Mechanism: Embeddings of all evaluation units are reduced via UMAP and then soft-clustered using GMM, where each cluster corresponds to a latent semantic topic. Topic-aware weights are assigned based on cluster size, down-weighting secondary or noisy components.
    - Design Motivation: Under open-ended prompts (e.g., "Tell me about Einstein"), different samples may organize content around different topics, causing structural inconsistency.

3. **Topic-Weighted Uncertainty Aggregation**:

    - Function: Produce the final uncertainty score.
    - Mechanism: NLI-based uncertainty is computed for each unit and then aggregated with cluster-derived weights, granting greater influence to dominant topics.
    - Design Motivation: Prevent secondary or off-topic content from disproportionately affecting the overall UQ score.

### Loss & Training

No model training is involved. Pre-trained NLI and embedding models are used. The number of GMM clusters is selected automatically via BIC.

## Key Experimental Results

### Main Results

- AGSC achieves state-of-the-art factuality correlation on the BIO and LongFact benchmarks.
- Inference time is reduced by approximately 60% compared to full atomic decomposition methods.

### Ablation Study

- Both the adaptive granularity and semantic clustering components contribute significantly to final performance.
- GMM clustering outperforms K-means hard clustering, as soft assignment better accommodates the fuzzy boundaries of semantic topics.

### Key Findings

- NLI neutrality is a valuable signal that should not be discarded.
- Topic-aware aggregation substantially outperforms simple pooling.
- Adaptive granularity reduces computation by 60% while maintaining or improving accuracy.

## Highlights & Insights

- Repurposing the NLI neutral category from "noise" into a valuable trigger signal is an elegant insight.
- GMM soft clustering naturally handles the fuzziness of semantic boundaries.
- The 60% inference time reduction has significant implications for practical deployment.

## Limitations & Future Work

- Automatic selection of the GMM cluster count may be unstable in extreme cases.
- The approach relies on the quality of the NLI model; erroneous NLI judgments can propagate and accumulate.
- Future work may explore combining AGSC with other UQ methods.

## Related Work & Insights

- The paper provides systematic solutions to three limitations of LUQ.
- The GMM clustering idea can be generalized to other NLP tasks requiring handling of topical heterogeneity.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of neutrality-triggered decomposition and semantic clustering is novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparisons across two benchmarks and multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ The framework is clearly described with well-motivated problem formulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Resource-Adaptive Federated Text Generation with Differential Privacy](../../ICLR2026/llm_safety/resource-adaptive_federated_text_generation_with_differential_privacy.md)
- [\[ACL 2026\] Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization](adaptive_text_anonymization_learning_privacy-utility_trade-offs_via_prompt_optim.md)
- [\[ACL 2026\] Toward Consistent World Models with Multi-Token Prediction and Latent Semantic Enhancement](toward_consistent_world_models_with_multi-token_prediction_and_latent_semantic_e.md)
- [\[ACL 2026\] Synthia: Scalable Grounded Persona Generation from Social Media Data](synthia_scalable_grounded_persona_generation_from_social_media_data.md)
- [\[ACL 2026\] FACTS: Table Summarization via Offline Template Generation with Agentic Workflows](facts_table_summarization_via_offline_template_generation_with_agentic_workflows.md)

</div>

<!-- RELATED:END -->
