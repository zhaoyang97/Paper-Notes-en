---
title: >-
  [Paper Note] Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation
description: >-
  [AAAI 2026][Recommender Systems][POI Recommendation] This paper is the first to introduce the tool-augmented LLM paradigm to the next POI recommendation task. Through three modules—preference extraction, multi-round candidate retrieval, and reranking—the framework enables LLMs to retrieve recommendations from the full POI pool. It achieves over 40% accuracy in Out-of-History (OOH) scenarios (where existing methods yield 0%), with average Acc@5/10 improvements of 20%/30%.
tags:
  - AAAI 2026
  - Recommender Systems
  - POI Recommendation
  - Tool-Augmented LLM
  - Agent
  - Open-Set Recommendation
  - Location-Based Services
date: 2026-05-08
content_hash: 966b96a377447a2f
---

# Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation

**Conference**: AAAI 2026
**arXiv**: [2511.06405](https://arxiv.org/abs/2511.06405)
**Code**: N/A
**Area**: Recommender Systems
**Keywords**: POI Recommendation, Tool-Augmented LLM, Agent, Open-Set Recommendation, Location-Based Services

## TL;DR
This paper is the first to introduce the tool-augmented LLM paradigm to the next POI recommendation task. Through three modules—preference extraction, multi-round candidate retrieval, and reranking—the framework enables LLMs to retrieve recommendations from the full POI pool. It achieves over 40% accuracy in Out-of-History (OOH) scenarios (where existing methods yield 0%), with average Acc@5/10 improvements of 20%/30%.

## Background & Motivation

**Background**: Next Point-of-Interest (POI) recommendation is a core task in location-based services. Traditional approaches (RNN/Transformer/GCN) model user trajectory sequences via embedding representations. Recently, the contextual reasoning capability of LLMs has been introduced to this task—e.g., LLM-Mob (in-context learning) and LLM4POI (supervised fine-tuning)—demonstrating potential for understanding spatiotemporal dynamics.

**Limitations of Prior Work**: LLM-based methods face two fundamental constraints: (1) heavy reliance on historical completeness—they can only recommend POIs the user has previously visited and cannot handle Out-of-History (OOH) scenarios (where users intend to visit places they have never been), which account for over 30% of real-world cases; (2) context window limitations—a city may contain hundreds of thousands of POIs, making it infeasible to encode all candidates into a prompt for open-set recommendation.

**Key Challenge**: User behavior exhibits both regularity (commuting patterns) and exploratory tendencies (trying new restaurants). Existing LLM methods overfit to visited POIs and cannot support exploratory behavior; fine-tuning approaches (e.g., GNPR-SID) further exacerbate this bias.

**Goal**: Design a plug-and-play, fine-tuning-free framework that enables LLMs to retrieve recommendations from the full POI pool via external tools, overcoming both the OOH limitation and the large-scale candidate space constraint.

**Key Insight**: The observation that humans select destinations through progressive filtering (by category, region, and distance) suggests that this narrowing process can be simulated via multi-round tool calls by an LLM agent.

**Core Idea**: Equip LLMs with external tool-calling capabilities and implement a three-stage pipeline—preference extraction → multi-round tool-based retrieval → reranking—to enable open-set POI recommendation.

## Method

### Overall Architecture
Tool4POI comprises three modules, all built on Qwen2.5-14B and applicable in a plug-and-play manner without fine-tuning: (1) **Preference Extraction Module**—extracts user preferences along region, category, and temporal dimensions from long-term check-in history; (2) **Tool-Augmented Candidate Retrieval Module**—the LLM acts as a retrieval agent, interacting with six external tools across multiple rounds to retrieve relevant candidates from the full POI pool; (3) **Reranking Module**—reorders candidates based on the user's recent check-in behavior to reflect current intent.

### Key Designs

1. **Preference Extraction Module**:

    - *Function*: Extracts a structured representation of long-term user preferences from historical check-in trajectories.
    - *Mechanism*: A structured prompt is designed to feed the user's temporally ordered check-in sequence (converted to regional codes via Google Maps Plus Codes) into the LLM, which outputs preference keywords along Region, Category, and Time dimensions. Plus Codes aggregate latitude/longitude coordinates into region-level encodings, enabling spatially proximate POIs to share the same code, thereby simplifying geographic feature representation.
    - *Design Motivation*: Historical data is voluminous yet rich in implicit patterns; the reasoning capability of LLMs is well suited for extracting multi-dimensional preference summaries.

2. **Tool-Augmented Candidate Retrieval Module**:

    - *Function*: Enables the LLM to autonomously retrieve relevant candidates from the full POI pool, overcoming context window limitations.
    - *Mechanism*: Six external tools are defined: query tools (*getPOIinfo* for POI metadata retrieval), retrieval tools (*filterByCategories*/*filterByRegions* for filtering by category/region), auxiliary tools (*findPotential* for generating initial candidates via POI-level collaborative filtering; *sortByDistance* for distance-based ranking), and a control tool (*finish* to terminate retrieval). The LLM acts as a RetrievalAgent, autonomously determining tool call order and parameters based on extracted preferences. Termination conditions: explicit invocation of *finish*, candidate set size falling below threshold $\tau=10$, or reaching the maximum call count $K=6$.
    - *Design Motivation*: (1) *findPotential* introduces collective behavioral priors via a directed co-occurrence graph $G=(\mathcal{P}, \mathcal{E})$, enabling OOH POIs to be retrieved; (2) multi-round interaction simulates the human decision-making process, progressively narrowing the candidate space to produce a high-quality candidate set.

3. **Reranking Module**:

    - *Function*: Reranks retrieved candidates based on the user's recent behavior to capture short-term intent.
    - *Mechanism*: The user's recent check-in trajectory $R_u$, the target timestamp $t_{i+1}$, and the candidate set $\mathcal{C}$ are jointly fed into the LLM, which ranks candidate POIs by visit likelihood based on recent behavioral patterns, using natural language reasoning rather than embedding similarity.
    - *Design Motivation*: Preference extraction captures long-term interests while reranking captures short-term dynamics (seasonal changes, life-stage transitions, etc.); the two are complementary.

### Loss & Training
Tool4POI is entirely training-free and requires no fine-tuning. During inference, the three modules execute sequentially. The top-20 candidates from the retrieval module are passed to the reranking module.

## Key Experimental Results

### Main Results

| Method | NYC Acc@5 | NYC Acc@10 | TKY Acc@5 | TKY Acc@10 | CA Acc@5 | CA Acc@10 |
|------|-----------|------------|-----------|------------|----------|-----------|
| Tool4POI | **0.6346** | **0.7623** | Best | Best | Best | Best |
| GNPR-SID (FT LLM) | Low | Low | Low | Low | Low | Low |
| GETNext | 0.4815 | 0.5811 | 0.4045 | 0.4961 | 0.3278 | 0.3946 |
| STAN | 0.4582 | 0.5734 | 0.3798 | 0.4464 | 0.2348 | 0.3018 |

### Ablation Study

| Configuration | All Acc@1 | All Acc@10 | OOH Acc@1 | OOH Acc@10 |
|------|-----------|------------|-----------|------------|
| Tool4POI (Full) | 0.3164 | 0.7623 | 0.0522 | 0.5863 |
| w/o Retrieval Module | 0.2545 | 0.5559 | 0 | 0 |
| w/o Reranking Module | 0.1655 | 0.7145 | 0.0963 | 0.6024 |

### Key Findings
- Existing LLM methods achieve 0% accuracy in OOH scenarios, while Tool4POI reaches 40%+ Acc@10, demonstrating the critical value of tool-augmented retrieval.
- The retrieval module contributes most to top-k recommendation (by introducing diverse candidates), while the reranking module contributes most to top-1 precision (by capturing current intent).
- The most significant improvements are observed on the sparse dataset CA (averaging only 10 check-ins per user), with gains as high as 100%, demonstrating robustness in data-sparse settings.
- Model scale effects: even a 3B model outperforms 7B fine-tuned methods, indicating that tool augmentation matters more than model scale.

## Highlights & Insights
- This is the first work to introduce the agent tool-calling paradigm to recommender systems, pioneering the integration of recommendation systems with LLM agents. The *findPotential* tool, which leverages a POI co-occurrence graph as a collective prior, is particularly elegant.
- The training-free, plug-and-play design allows direct deployment across any city or dataset without retraining, making it highly practical.
- The iterative candidate narrowing via multi-round interaction mirrors the cognitive process humans use when selecting destinations, and this paradigm is transferable to other open-set recommendation scenarios (e.g., product recommendation, content recommendation).

## Limitations & Future Work
- Retrieval quality depends on the LLM's tool-calling accuracy; smaller models may produce errors that break the tool chain.
- The six-tool design is relatively hand-crafted; additional domain-specific tools (e.g., weather queries, event calendars) may further improve performance.
- In OOH scenarios, reranking may actually degrade performance due to lack of contextual information; adaptive decisions on whether to perform reranking could be considered.
- Inference latency is high due to multi-round LLM calls, which requires optimization for real-time recommendation settings.

## Related Work & Insights
- **vs. LLM4POI**: Supervised fine-tuning of LLMs for QA-style recommendation leads to severe overfitting toward high-frequency POIs. Tool4POI requires no training and retrieves open-set candidates via tools.
- **vs. GNPR-SID**: Uses semantic IDs to train LLMs but still cannot recommend OOH POIs. Tool4POI's *findPotential* tool leverages collective behavior to transcend individual history limitations.
- **vs. Traditional Methods (e.g., GETNext)**: Fixed embedding representations lack flexibility. Tool4POI leverages LLM reasoning for context-aware recommendation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First tool-augmented LLM POI recommendation framework; groundbreaking resolution of the OOH problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation on three datasets with in-depth IH/OOH analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear method description with complete algorithmic pseudocode.
- Value: ⭐⭐⭐⭐⭐ — Pioneering contribution to the intersection of recommender systems and LLM agents.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization](from_ids_to_semantics_a_generative_framework_for_cross-domain_recommendation_wit.md)
- [\[AAAI 2026\] Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation](tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi-domain_llm-.md)
- [\[AAAI 2026\] Evaluating LLMs for Police Decision-Making: A Framework Based on Police Action Scenarios](evaluating_llms_for_police_decision-making_a_framework_based_on_police_action_sc.md)
- [\[AAAI 2026\] Preference is More Than Comparisons: Rethinking Dueling Bandits with Augmented Human Feedback](preference_is_more_than_comparisons_rethinking_dueling_bandits_with_augmented_hu.md)
- [\[NeurIPS 2025\] Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](../../NeurIPS2025/recommender/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)

<!-- RELATED:END -->
