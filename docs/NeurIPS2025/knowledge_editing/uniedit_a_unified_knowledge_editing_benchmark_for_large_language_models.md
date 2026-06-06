---
title: >-
  [Paper Note] UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models
description: >-
  [NeurIPS 2025 (Datasets & Benchmarks Track)][Knowledge Editing][LLM benchmark] This paper presents UniEdit — the first unified LLM knowledge editing benchmark built upon an open-domain knowledge graph (Wikidata)…
tags:
  - "NeurIPS 2025 (Datasets & Benchmarks Track)"
  - "Knowledge Editing"
  - "LLM benchmark"
  - "ripple effect"
  - "knowledge graph"
  - "multi-hop reasoning"
date: 2026-05-08
content_hash: 5c8466f5ea7f7c03
---

# UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models

**Conference**: NeurIPS 2025 (Datasets & Benchmarks Track)  
**arXiv**: [2505.12345](https://arxiv.org/abs/2505.12345)  
**Code**: Available  
**Area**: NLP / Knowledge Editing  
**Keywords**: knowledge editing, LLM benchmark, ripple effect, knowledge graph, multi-hop reasoning

## TL;DR
This paper presents UniEdit — the first unified LLM knowledge editing benchmark built upon an open-domain knowledge graph (Wikidata), covering 311K samples across 25 domains in 5 major categories. By introducing the Neighborhood Multi-hop Chain Sampling (NMCS) algorithm, UniEdit integrates diverse generalization and locality evaluation criteria into a single framework, systematically revealing the shortcomings of existing editing methods under complex ripple effect evaluations.

## Background & Motivation

**Background**: Model editing aims to efficiently correct the internal knowledge of large language models, avoiding the high cost of full retraining and catastrophic forgetting. Existing editing methods fall into two main paradigms: Locate-then-Edit (e.g., ROME, MEMIT, AlphaEdit) and external module-based approaches (e.g., SERAC, GRACE, MEND).

**Limitations of Prior Work**: Existing knowledge editing benchmarks suffer from three core limitations. First, narrow domain coverage — most benchmarks sample from a limited set of knowledge graph triples, confined to a small range of relations and domains. Second, one-sided evaluation criteria — each benchmark independently constructs data focusing on specific evaluation dimensions (e.g., MQuAKE focuses on multi-hop reasoning, RippleEdit on ripple effects), lacking a unified scheme that integrates all criteria into a single dataset. Third, insufficient scale — the data volume is too small to support methods that require editing-specific training (e.g., SERAC, RECIPE).

**Key Challenge**: Real-world knowledge editing requirements span broad domains, and the post-edit ripple effects (relation inversion, multi-hop reasoning, alias recognition, etc.) exhibit complex combinatorial patterns. Evaluation conclusions drawn from constrained datasets may not generalize to diverse open-domain scenarios.

**Core Idea**: By leveraging Wikidata — the largest open-source knowledge graph — the paper constructs a unified benchmark spanning 25 domains and designs the NMCS algorithm to integrate multiple generalization and locality evaluation criteria into a single sampling framework, allowing each sample to naturally encompass combinations of multiple evaluation criteria.

## Method

### Overall Architecture
UniEdit is constructed via a five-step pipeline: (1) cleaning Wikidata (retaining 29.9M entities and 2,400 properties from 113.7M entities and 12,300 properties); (2) retrieving entities from Wikidata using domain keywords, covering 25 domains across 5 major categories; (3) selecting edit triples via weighted sampling, dynamically reducing the sampling probability of semantically similar entities to enhance diversity; (4) applying the NMCS algorithm to sample generalization and locality subgraphs from edit triples; (5) converting structured knowledge subgraphs into natural language using DeepSeek-V3.

### Key Designs

1. **Domain Coverage and Entity Sampling**:

    - Function: Ensures broad and diverse knowledge domain coverage in the benchmark data
    - Mechanism: Knowledge domains are divided into 5 major categories — natural sciences, humanities, social sciences, applied sciences, and interdisciplinary fields — comprising 25 sub-domains (e.g., astronomy, biology, computer science, data science). For each domain, approximately 100 keywords are generated via GPT-4 to retrieve Wikidata entities, with 30,000 head entities sampled per domain.
    - Design Motivation: Existing benchmarks typically cover only a few relation types and fail to reflect performance differences across knowledge domains. Weighted sampling (with dynamic probability decay for semantically similar entities) ensures both sufficient domain coverage and avoidance of over-concentration.

2. **Neighborhood Multi-hop Chain Sampling (NMCS) Algorithm**:

    - Function: Constructs generalization and locality evaluation samples in a unified manner, automatically covering combinations of multiple evaluation criteria
    - Mechanism: Given an edit triple $t_\varepsilon = (s_\varepsilon, r_\varepsilon, o_\varepsilon)$, NMCS samples multi-hop chain subgraphs within its neighborhood. Generalization subgraphs must contain the complete edit triple; locality subgraphs only partially contain or exclude components of the edit triple. NMCS operates in two stages: first sampling chain structures around the initial triple, then selecting a node as the prediction target and extending from both sides to form multi-hop chains.
    - Design Motivation: Prior benchmarks independently construct specific types of evaluation data (e.g., MQuAKE for multi-hop, RippleEdit for alias recognition), making it impossible to evaluate combinations of criteria. NMCS naturally generates samples containing combinations of multiple criteria (e.g., simultaneously covering multi-hop + relation inversion + subject alias generalization) through a unified graph sampling framework.

3. **Natural Language Conversion and Quality Control**:

    - Function: Converts structured knowledge subgraphs into natural and diverse natural language test samples
    - Mechanism: DeepSeek-V3 is used to first generate single-hop sentences for each triple, which are then merged into multi-hop descriptions. Quality is ensured through automatic checks (verifying that each generated prompt contains the correct subject and points to the correct object) and human evaluation.
    - Design Motivation: The triple structure of knowledge graphs is insufficiently natural for direct evaluation; conversion into diverse natural language forms is necessary to simulate real-world scenarios.

### Loss & Training
UniEdit contains a total of 311K samples, each comprising an edit sample, generalization samples, and locality samples. The dataset is publicly released along with a complete construction toolkit.

## Key Experimental Results

### Main Results

| Editor | GPT2-XL (1.5B) |  |  | GPT-J (6B) |  |  | LLaMA-3.1 (8B) |  |  |
|--------|------|------|------|------|------|------|------|------|------|
|  | Rel. | Gen. | Loc. | Rel. | Gen. | Loc. | Rel. | Gen. | Loc. |
| W/O (Unedited) | 29.69 | 28.04 | 100.0 | 35.34 | 33.04 | 100.0 | 43.68 | 51.81 | 100.0 |
| FT | 100.0 | 49.46 | 89.72 | 100.0 | 57.25 | 91.26 | 100.0 | 69.00 | 93.54 |
| IKE | 99.93 | **76.46** | 83.35 | 99.80 | **79.05** | 84.31 | 93.54 | **89.52** | 80.79 |
| ROME | 92.02 | 35.84 | 96.76 | 98.98 | 45.33 | 96.41 | 75.81 | 51.38 | 95.12 |
| SERAC | 99.46 | 78.79 | 88.06 | 99.16 | 81.32 | 86.59 | 98.96 | 83.66 | 84.25 |
| T-Patcher | 82.28 | 45.40 | 97.27 | 91.24 | 48.16 | 93.23 | 73.03 | 49.83 | 83.27 |
| GRACE | 99.68 | 28.00 | **99.99** | 99.99 | 33.16 | **99.97** | 99.92 | 51.89 | **99.97** |
| AlphaEdit | 92.26 | 37.20 | 95.90 | 99.77 | 43.91 | 97.60 | 84.09 | 55.10 | 98.72 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| SERAC trained on individual domains | Significant generalization drop | SERAC trained on a single domain performs well only in that domain; cross-domain generalization is poor |
| Single criterion vs. combined criteria | Increasing generalization difficulty | As evaluation criteria combinations become more complex (e.g., Rep+OA+SA), editor generalization scores decrease |
| Locality + MH | Locality improves | More complex sentences reduce overlap between locality inputs and edited knowledge, decreasing interference |

### Key Findings
- Locate-and-Edit (L&E) methods (ROME, AlphaEdit) report success on simple rephrase tasks but perform poorly on UniEdit's complex generalization evaluations (combinations of multi-hop + alias + relation inversion), with generalization scores of only 35–55%.
- IKE and SERAC achieve the best generalization (76–89%) by leveraging in-context learning and edit-training priors, but at the cost of reduced locality (80–88%).
- GRACE achieves the highest locality (~100%) via a token-based linear distance retrieval mechanism, but its strong assumption of linear structure in representation space severely limits generalization.
- Editors perform slightly better in natural sciences and humanities, and worse in social sciences and applied sciences, reflecting distributional biases in pretraining corpora.

## Highlights & Insights
- The first open-domain knowledge editing benchmark covering 25 domains, with a scale (311K) significantly surpassing prior benchmarks.
- The NMCS algorithm elegantly unifies multiple evaluation criteria within a single sampling framework, allowing criterion combinations to emerge naturally rather than being artificially constructed.
- The work systematically exposes the fragility of L&E methods under complex ripple effect evaluations, providing clear directions for improving editing approaches.
- Domain-level analysis reveals that knowledge editing in low-resource and niche domains is more challenging, warranting greater attention in future research.

## Limitations & Future Work
- Currently covers English only; multilingual knowledge editing is not addressed.
- Focuses on the text modality; multimodal scenarios such as visual LLM editing are not included.
- Subgraph structures are limited to simple chains; more complex graph structures (e.g., star, mesh) are not explored.
- Future work could leverage Wikidata's multimodal content (e.g., images, videos) to construct more comprehensive multimodal editing benchmarks.

## Related Work & Insights
- **ZSRE / CounterFact**: Early editing benchmarks that only evaluate reliability and simple rephrase; UniEdit substantially expands the evaluation scope.
- **MQuAKE / RippleEdit**: Focus on multi-hop reasoning and ripple effects respectively; UniEdit unifies these criteria through NMCS.
- **ROME / MEMIT / AlphaEdit**: L&E methods are shown on UniEdit to generalize less effectively than previously expected.
- Insight: Knowledge editing evaluation must account for the "propagation scope" of edited knowledge — after modifying a fact, the model should maintain consistency across various reasoning chains involving that fact.

## Rating
- Novelty: ⭐⭐⭐⭐ The NMCS approach to unifying multiple evaluation criteria is novel, and the 25-domain coverage is the broadest among comparable benchmarks.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 editors × 3 LLM backbones × 25 domains × multiple evaluation criterion combinations; the experiments are highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The construction pipeline is clearly presented and the experimental analysis is systematic and in-depth.
- Value: ⭐⭐⭐⭐ Provides the knowledge editing field with a much-needed standardized, large-scale evaluation infrastructure.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] KScope: A Framework for Characterizing the Knowledge Status of Language Models](kscope_a_framework_for_characterizing_the_knowledge_status_of_language_models.md)
- [\[ACL 2026\] Aligning Language Models with Real-time Knowledge Editing](../../ACL2026/knowledge_editing/aligning_language_models_with_real-time_knowledge_editing.md)
- [\[ACL 2026\] The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models](../../ACL2026/knowledge_editing/the_model_agreed_but_didn39t_learn_diagnosing_surface_compliance_in_large_langua.md)
- [\[ICML 2026\] KORE: Enhancing Knowledge Injection for Large Multimodal Models via Knowledge-Oriented Controls](../../ICML2026/knowledge_editing/kore_enhancing_knowledge_injection_for_large_multimodal_models_via_knowledge-ori.md)
- [\[ICLR 2026\] When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations](../../ICLR2026/knowledge_editing/when_large_multimodal_models_confront_evolving_knowledge_challenges_and_explorat.md)

</div>

<!-- RELATED:END -->
