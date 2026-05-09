---
title: >-
  [Paper Note] Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation
description: >-
  [NeurIPS 2025][Recommender Systems] This paper proposes TagCF, a framework that employs MLLM to extract user role tags and item topic tags, then uses LLM reasoning to construct U2I/I2U logic graphs (causal associations between user roles and item types). Three integration strategies — a tag encoder, contrastive learning augmentation, and logic-based scoring — are used to enhance recommendations. On an industrial platform with hundreds of millions of users, online A/B testing yields a 0.946% improvement in engagement metrics and a 0.102% gain in diversity; offline experiments show an 8.06% improvement in NDCG@10.
tags:
  - NeurIPS 2025
  - Recommender Systems
  - User Role Modeling
  - LLM Knowledge Extraction
  - Tag-based Collaborative Filtering
  - Logical Reasoning Graph
date: 2026-05-08
content_hash: 4f3b639c2132f2b5
---

# Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation

**Conference**: NeurIPS 2025
**arXiv**: [2505.10940](https://arxiv.org/abs/2505.10940)
**Code**: [https://github.com/Code2Q/TagCF](https://github.com/Code2Q/TagCF)
**Area**: Recommender Systems / LLM Augmentation
**Keywords**: Recommender Systems, User Role Modeling, LLM Knowledge Extraction, Tag-based Collaborative Filtering, Logical Reasoning Graph

## TL;DR
This paper proposes TagCF, a framework that employs MLLM to extract user role tags and item topic tags, then uses LLM reasoning to construct U2I/I2U logic graphs (causal associations between user roles and item types). Three integration strategies — a tag encoder, contrastive learning augmentation, and logic-based scoring — are used to enhance recommendations. On an industrial platform with hundreds of millions of users, online A/B testing yields a 0.946% improvement in engagement metrics and a 0.102% gain in diversity; offline experiments show an 8.06% improvement in NDCG@10.

## Background & Motivation

**Background**: Mainstream recommender systems follow the learning-to-rank paradigm, ranking items by discovering item topics (e.g., categories) and capturing user preferences over those topics. User representations rely primarily on implicit vectors derived from historical behavior, with little explicit modeling of user roles or characteristics.

**Limitations of Prior Work**: (a) Item-item correlations are difficult to explain using item types alone — the classic "beer and diapers" association requires the user role "father of a newborn" as a logical explanation. (b) Interest-based recommendation models only interest relations (item–item), whereas user–item logical relations (e.g., "symphony musician → headphones", "symphony musician → instruments") are more expressive and interpretable. (c) User role identification is difficult to obtain directly in practice.

**Key Challenge**: Recommendation requires understanding the logical relation "what kind of person prefers what kind of content," yet existing methods model only statistical correlations and lack causal or logical-level modeling.

**Goal**: Two new tasks are introduced: (1) user role identification (e.g., "symphony musician", "new mother"); and (2) behavioral logic modeling — bidirectional U2I and I2U logic between user roles and item topics.

**Key Insight**: Leverage MLLM multimodal understanding to extract tags, and leverage LLM world knowledge and logical reasoning to construct virtual logic graphs.

**Core Idea**: MLLM-extracted user/item tags + LLM-reasoned U2I/I2U logic graphs + three integration strategies = user-role-aware logical recommendation.

## Method

### Overall Architecture
TagCF consists of three modules: (1) **MLLM Tag Extraction**: the M3 model extracts user role tags and item topic tags from multimodal item features; (2) **LLM Collaborative Logic Filtering**: Qwen2.5-7B reasons over tag relationships to construct a bidirectional logic graph; (3) **Tag-Logic Integration**: tags and logic are incorporated into the recommendation model via a tag encoder, contrastive learning augmentation, and logic-based scoring.

### Key Designs

1. **MLLM Tag Extraction + Coverage-Set Reduction**:

    - Function: Extract "what kind of user would like this item" and "what type of item this is" from each item.
    - Mechanism: The M3 model receives multimodal item features and generates tags via customized prompts. A greedy dynamic minimum coverage-set algorithm identifies a small set of approximately 300 core tags with high coverage and semantic diversity. A distilled model then processes items at the million scale.
    - Design Motivation: Directly applying MLLM to hundreds of millions of items is infeasible; the coverage-set plus distillation approach enables industrial-scale deployment.

2. **LLM Collaborative Logic Filtering**:

    - Function: Construct a bidirectional logic graph between user roles and item topics.
    - Mechanism: For each item topic tag $c$, the LLM reasons about "what users are interested in this type of content" (I2U); for each user role tag $t$, it reasons about "what content this type of user prefers" (U2I). A distilled model maps results to the coverage-set space.
    - Design Motivation: LLM world knowledge inherently encodes logical associations such as "symphony musician → instruments," eliminating the need to mine such relations from behavioral data.

3. **Three Integration Strategies**:

    - **Tag Encoder**: Attention-aggregated item tag embeddings are concatenated with ID embeddings and fed into SASRec.
    - **Contrastive Learning Augmentation**: User-level (aligning user representations with relevant role tags) and item-level (aligning item embeddings with relevant topic tags) contrastive objectives.
    - **Logic-based Scoring**: At inference time, the logic graph expands the user's tag set, and a tag-based matching score is computed.

### Loss & Training
- Primary loss: binary cross-entropy with reward weighting.
- Augmented loss: linear combination of the primary loss and two contrastive losses.
- Two configurations: TagCF-util (accuracy-oriented) and TagCF-expl (exploration-oriented).

## Key Experimental Results

### Online A/B Testing (Platform with Hundreds of Millions of Users, 14 Days)

| Strategy | Engagement Metric | Diversity |
|---|---|---|
| TagCF-util vs. Baseline | **+0.946%*** | +0.001% |
| TagCF-expl vs. Baseline | +0.143% | **+0.102%*** |

After 40 days of expanded deployment, TagCF-expl yields a statistically significant +0.037% improvement in LT7 (next-week DAU / retention).

### Offline Experiments

| Dataset | Metric | TagCF | Prev. SOTA | Gain |
|---|---|---|---|---|
| Industry | NDCG@10 | **0.0201** | 0.0186 (SAID) | +8.06% |
| Industry | Cover@10 | **0.4013** | 0.3558 (LRURec) | +12.78% |
| Books | NDCG@10 | **0.1881** | 0.1705 (SAID) | +10.32% |
| Movies | NDCG@10 | **0.1490** | 0.1300 | +14.62% |

### Ablation Study

| Configuration | NDCG@10 | Cover@10 | Note |
|---|---|---|---|
| TagCF-ut (full) | **0.0201** | 0.3832 | Full model |
| w/o tag encoder | Decrease | Decrease | Tag semantics lost |
| w/o contrastive learning | Decrease | Decrease | Embedding space misaligned |
| w/o logic reasoning | Decrease | Significant decrease | Exploration capability lost |

### Key Findings
- **User roles > item topics**: TagCF-ut outperforms TagCF-it on NDCG — "knowing who the user is" matters more than "knowing what the item is."
- **Logic graphs are transferable**: A logic graph extracted from one scenario can be directly applied to other recommendation tasks.
- **Long-term online effects**: Diversity gains translate into improvements in DAU and retention after 7 days.
- **Coverage-set stability**: Daily update change rate is approximately 5%.

## Highlights & Insights
- **Treating "user roles" as a core component of recommendation** is a profound insight — conventional approaches model users as "sequences of behaviors," whereas TagCF models users as "collections of social roles."
- **LLM logic graphs constitute general, transferable knowledge** — extracted once and serving multiple tasks, reducing the marginal cost of LLM-augmented recommendation.
- **Coverage-set reduction** compresses open-world tags into approximately 300 core tags, making industrial deployment feasible.

## Limitations & Future Work
- Tag extraction depends on MLLM quality; performance may be limited for non-multimodal items.
- LLM-reasoned logic graphs may encode biases (e.g., gender–occupation stereotypes).
- Operational cost of daily updates to tags and logic graphs.
- Dynamic update mechanisms for user roles that evolve over time warrant further development.

## Related Work & Insights
- **vs. RLMRec / SAID / GENRE**: These methods extract only item-side features and neglect user-side logic. TagCF jointly models both sides and connects them via the logic graph.
- **vs. LLM-as-Recommender (P5 / TALLRec)**: These incur extremely high computational costs. TagCF uses LLMs only for offline knowledge extraction, with zero LLM overhead at inference time.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ User role modeling and virtual logic graph concepts are highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Online A/B testing (hundreds of millions of users, 14 days) + industrial offline evaluation + 2 public datasets.
- Writing Quality: ⭐⭐⭐⭐ Framework is clear, though notation is dense.
- Value: ⭐⭐⭐⭐⭐ Validated on an industrial-scale platform; provides important reference for LLM-augmented recommendation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The More You Automate, the Less You See: Hidden Pitfalls of AI Scientist Systems](the_more_you_automate_the_less_you_see_hidden_pitfalls_of_ai_scientist_systems.md)
- [\[NeurIPS 2025\] Think before Recommendation: Autonomous Reasoning-enhanced Recommender](think_before_recommendation_autonomous_reasoning-enhanced_recommender.md)
- [\[NeurIPS 2025\] Measuring What Matters: Construct Validity in Large Language Model Benchmarks](measuring_what_matters_construct_validity_in_large_language_model_benchmarks.md)
- [\[AAAI 2026\] Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation](../../AAAI2026/recommender/wavelet_enhanced_adaptive_frequency_filter_for_sequential_re.md)
- [\[ICLR 2026\] In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations](../../ICLR2026/recommender/in_agents_we_trust_but_who_do_agents_trust_latent_source_preferences_steer_llm_g.md)

</div>

<!-- RELATED:END -->
