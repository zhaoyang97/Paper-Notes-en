---
title: >-
  [Paper Note] Search Arena: Analyzing Search-Augmented LLMs
description: >-
  [ICLR 2026][Recommender Systems][search-augmented LLM] This paper presents Search Arena — the first large-scale human preference dataset for search-augmented LLMs (24,069 conversations + 12,652 preference votes…
tags:
  - "ICLR 2026"
  - "Recommender Systems"
  - "search-augmented LLM"
  - "benchmark"
  - "human preference"
  - "citation analysis"
  - "Chatbot Arena"
date: 2026-05-08
content_hash: c4a60b5b46a8828b
---

# Search Arena: Analyzing Search-Augmented LLMs

**Conference**: ICLR 2026
**arXiv**: [2506.05334](https://arxiv.org/abs/2506.05334)  
**Code**: [Project Page](https://github.com/) (open-source dataset)  
**Area**: Recommender Systems
**Keywords**: search-augmented LLM, benchmark, human preference, citation analysis, Chatbot Arena

## TL;DR
This paper presents Search Arena — the first large-scale human preference dataset for search-augmented LLMs (24,069 conversations + 12,652 preference votes, 71 languages). Key findings include: user preference is positively influenced by citation quantity even when citations do not support the claims; community-driven platforms are preferred over Wikipedia; search augmentation does not degrade general chat performance, whereas general-purpose LLMs degrade significantly in search scenarios.

## Background & Motivation
**Background**: Search-augmented LLMs (e.g., Perplexity, Gemini Search, ChatGPT Search), which combine web retrieval with LLM reasoning, are increasingly popular. Existing benchmarks such as SimpleQA (4,326 instances) and BrowseComp (1,266 instances) are small-scale, single-turn, English-only, and oriented toward factual queries.

**Limitations of Prior Work**:
   - **Insufficient coverage**: Factual queries constitute only ~19% of real user queries; the majority require information synthesis, analysis, recommendation, creativity, and other higher-order capabilities.
   - **Lack of preference understanding**: It remains unclear what users prefer in search scenarios — the role of citations, the influence of source domains, the value of reasoning.
   - **Cross-scenario evaluation gap**: How do search-augmented LLMs perform in general settings? How do general-purpose LLMs perform in search settings?

**Key Challenge**: Evaluating search-augmented LLMs requires large-scale, naturalistic, and diverse interaction data, yet existing datasets are small-scale and expert-constructed.

**Core Idea**: Crowdsource real user interactions with and preferences over search-augmented LLMs via the Chatbot Arena platform, and conduct systematic multi-dimensional analysis.

## Method

### Overall Architecture
Search Arena platform (search tab of Chatbot Arena) → anonymously display two search model responses side by side → user votes for preferred response → collect 7 weeks of data: 24,069 conversations + 12,652 preference votes → model preferences via Bradley-Terry model → multi-dimensional analysis.

### Key Designs

1. **Data Collection and Scale**

    - Function: Crowdsource real user interaction data with search-augmented LLMs.
    - Scale: 24,069 conversations, 12,652 preference votes, 11,650 users, 136 countries, 71 languages (English 58.3%, Russian 11.8%, Chinese 7.0%), 13 models.
    - Full system traces included: retrieved URLs, reasoning traces, model responses, multi-turn conversation history.
    - 22.4% multi-turn conversations; 11% multilingual queries.

2. **User Intent Taxonomy**

    - Function: Define 9 intent categories for search-augmented conversation scenarios.
    - Categories: Factual Lookup (19.3%), Information Synthesis, Analysis, Recommendation, Explanation, Creative Generation, Guidance, Text Processing, Other.
    - Annotation method: GPT-4.1 automatic annotation; Cohen's kappa = 0.812 (strong agreement) on 150 multilingual samples.
    - Key finding: Factual queries account for only one-fifth of all queries; the majority require higher-order capabilities.

3. **Preference Analysis (Bradley-Terry + Feature Analysis)**

    - **General features**:
        - Reasoning models perform better (top-3 models average win rate >60%).
        - Larger search context window → more preferred (sonar-pro high context 63.9% vs. medium 57.6%).
        - Longer responses → more preferred ($\beta_{length} = 0.334$), but length preference is halved for factual queries.
    - **Citation features** (core findings):
        - Citation quantity is positively correlated with preference ($\beta = 0.334$).
        - **Irrelevant citations are also positively correlated with preference ($\beta_{irrelevant} = 0.273$)** — users equate the mere presence of citations with credibility.
        - The preference coefficient for correctly attributed citations ($\beta_{correct} = 0.285$) is close to that of irrelevant citations — a concerning result.
    - **Source domain preference**: Community blogs, technical platforms, and social networks are preferred over Wikipedia. Wikipedia is unsuitable for time-sensitive topics such as sports news.

4. **Cross-Scenario Analysis**

    - Function: Test search-augmented LLMs in general chat scenarios and general-purpose LLMs in search scenarios.
    - Finding 1: Search augmentation does not degrade general performance; it even improves performance on factual queries (p=0.012), with only a marginal decrease on text processing tasks (p=0.077).
    - Finding 2: General-purpose LLMs degrade significantly in search scenarios (p=0.009) — parametric knowledge alone is insufficient.

### Methodological Tools
- Bradley-Terry preference model + standardized feature difference coefficients.
- LLM-based dataset difference analysis framework (Dunlap et al.).
- 100-sample expert validation with 3 annotators: expert-user agreement rate 68% (ties excluded), substantially above the random baseline of 50%.

## Key Experimental Results

### Factors Influencing Preference (Bradley-Terry Coefficients)

| Feature | Coefficient $\beta$ | Statistical Significance | Interpretation |
|------|-------------|-----------|------|
| Response length | 0.334 | ✓ | Longer responses preferred |
| Citation count | positive | ✓ | More citations preferred |
| Correctly attributed citations | 0.285 | ✓ | Expected |
| **Irrelevant citations** | **0.273** | ✓ | Concerning — nearly equivalent to correct citations |
| Search context size | positive | ✓ (select models) | Larger context preferred |
| Reasoning capability | positive | ✓ | Reasoning models achieve higher win rates |

### Cross-Scenario Analysis

| Model Type | Search Scenario | General Scenario |
|---------|---------|---------|
| Search-augmented LLM | Normal | **No degradation** (even gains on factual queries) |
| General-purpose LLM | **Significant degradation** (p=0.009) | Normal |

### Comparison with Existing Benchmarks

| Benchmark | Scale | Languages | Multi-turn | Intent Coverage |
|------|------|------|------|---------|
| SimpleQA | 4,326 | English | ✗ | Factual queries |
| BrowseComp | 1,266 | English | ✗ | Constrained challenges |
| **Search Arena** | **24,069** | **71** | **✓** | **9 categories** |

### Key Findings
- **Citation quantity bias** is the most important finding: users equate the presence of citations with credibility, without distinguishing whether citations actually support the claims. This has profound implications for search-augmented LLM design — models are incentivized to inflate citations.
- Factual queries account for only one-fifth of real user queries; existing benchmarks severely underestimate the operational complexity of search-augmented LLMs.
- Search augmentation is strictly beneficial — general performance is maintained or improved with added real-time capability; conversely, general-purpose models are inadequate in search scenarios.
- Community-driven platforms (e.g., Reddit) are preferred over Wikipedia, likely reflecting the value of information recency and discussion depth.

## Highlights & Insights
- **Systematic revelation of "citation padding"**: This is an important safety and alignment finding — if irrelevant citations receive nearly the same preference boost as correct citations, search-augmented LLMs are incentivized to add spurious citations to improve user satisfaction scores.
- **Unique dataset value**: Complete system traces (URLs + reasoning traces + multi-turn history) enable a wide range of downstream research — citation verification, reasoning quality assessment, search strategy analysis.
- **Practical implications of cross-scenario analysis**: Search augmentation is a unidirectional improvement — it can be enabled by default without concern for performance degradation.

## Limitations & Future Work
- User preference is inherently subjective; preference does not equal correctness or quality.
- Crowdsourced data may suffer from selection bias, as Chatbot Arena users are not representative of the general population.
- Confounding factors cannot be controlled — citation count is highly correlated with response length, search depth, and other features.
- The analysis establishes correlation rather than causation; controlled experiments are needed to establish causal relationships.
- Coverage is limited to 13 models, not encompassing all mainstream search-augmented LLMs.

## Related Work & Insights
- **vs. SimpleQA/BrowseComp**: Search Arena is 5–19× larger, multilingual, multi-turn, multi-intent, and based on preference votes rather than gold-standard answers.
- **vs. Chatbot Arena**: Search Arena operates as a dedicated search tab; differing user expectations lead to a distinct query distribution.
- **vs. CORAL/WildChat**: These datasets lack search augmentation and citation metadata.

## Rating
- Novelty: ⭐⭐⭐⭐ First large-scale preference dataset for search-augmented LLMs; citation bias revelation is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 24K conversations + 12K votes + multi-dimensional in-depth analysis + cross-scenario evaluation.
- Writing Quality: ⭐⭐⭐⭐ Analysis is progressively structured with rich figures and tables.
- Value: ⭐⭐⭐⭐⭐ Profound impact on search-augmented LLM evaluation and design; the open-source dataset is of exceptional value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RAE: A Neural Network Dimensionality Reduction Method for Nearest Neighbors Preservation in Vector Search](rae_a_neural_network_dimensionality_reduction_method_for_nearest_neighbors_prese.md)
- [\[AAAI 2026\] CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search](../../AAAI2026/recommender/crops_improving_dense_retrieval_with_cross-perspective_positive_samples_in_short.md)
- [\[AAAI 2026\] Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling](../../AAAI2026/recommender/semi-supervised_synthetic_data_generation_with_fine-grained_relevance_control_fo.md)
- [\[ICML 2026\] Prompts for Public-Sector LLMs Should Be Governed as Commons](../../ICML2026/recommender/prompts_for_public-sector_llms_should_be_governed_as_commons.md)
- [\[ACL 2026\] MemRec: Collaborative Memory-Augmented Agentic Recommender System](../../ACL2026/recommender/memrec_collaborative_memory-augmented_agentic_recommender_system.md)

</div>

<!-- RELATED:END -->
