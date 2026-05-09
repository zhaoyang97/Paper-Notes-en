---
title: >-
  [Paper Note] Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring
description: >-
  [ACL 2026][Passage Retrieval] This paper proposes BAGEL, a Bayesian active learning framework based on Gaussian Processes (GP) that propagates sparse LLM relevance signals across the embedding space via an exploration–exploitation strategy under a limited LLM budget, enabling global passage retrieval that substantially outperforms conventional LLM re-ranking methods.
tags:
  - ACL 2026
  - Passage Retrieval
  - Gaussian Processes
  - Active Learning
  - LLM Re-ranking
  - Bayesian Optimization
date: 2026-05-08
content_hash: 3ff2a39d189b3779
---

# Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring

**Conference**: ACL 2026
**arXiv**: [2604.17906](https://arxiv.org/abs/2604.17906)
**Code**: [GitHub](https://github.com/junieberry/BAGEL)
**Area**: Information Retrieval
**Keywords**: Passage Retrieval, Gaussian Processes, Active Learning, LLM Re-ranking, Bayesian Optimization

## TL;DR

This paper proposes BAGEL, a Bayesian active learning framework based on Gaussian Processes (GP) that propagates sparse LLM relevance signals across the embedding space via an exploration–exploitation strategy under a limited LLM budget, enabling global passage retrieval that substantially outperforms conventional LLM re-ranking methods.

## Background & Motivation

**Background**: LLMs exhibit strong zero-shot relevance modeling capabilities, but their high computational cost renders passage retrieval a budget-constrained global optimization problem. The dominant paradigm employs LLM re-ranking: a dense retriever first retrieves top-$K$ candidates, which are then re-ranked by an LLM.

**Limitations of Prior Work**: (1) Relevant passages are often distributed across multiple disjoint clusters in the semantic space, yet dense retrievers only retrieve neighbors near the query embedding, failing to discover distant relevant clusters. (2) Existing methods cannot propagate relevance signals from already-scored passages to unseen ones, thereby ignoring the semantic structure of the embedding space.

**Key Challenge**: Exploring the entire embedding space under a limited LLM inference budget is necessary, yet conventional methods passively rely on a single-stage retriever and are incapable of global exploration.

**Goal**: To leverage GP's kernel-based relevance propagation and uncertainty estimation to actively navigate the embedding space and discover multimodal relevance distributions.

**Key Insight**: Passage retrieval is formulated as a Bayesian optimization problem, where the GP provides predictive mean and uncertainty, and an acquisition function balances exploration and exploitation.

**Core Idea**: GPs are naturally suited to this task—the kernel function propagates relevance signals, and the posterior variance guides active learning toward uncertain regions.

## Method

### Overall Architecture

BAGEL proceeds in two stages: (1) **Warm-start initialization**—the query itself is treated as the highest-relevance observation, supplemented by LLM scores for top-$M$ densely retrieved passages; (2) **Active learning exploration**—an acquisition function (UCB) iteratively selects the next passage for LLM scoring, updates the GP posterior, and ultimately generates rankings over all passages.

### Key Designs

1. **Query-Specific Gaussian Process**:

   - **Function**: Models the query–passage relevance function over the embedding space.
   - **Mechanism**: The GP takes passage embeddings $\mathbf{x}_p$ as input and LLM relevance scores as output; the posterior predictive mean $\mu_q(\mathbf{x}_{p_*})$ and variance $\sigma_q^2(\mathbf{x}_{p_*})$ provide relevance estimates and uncertainty, respectively.
   - **Design Motivation**: The GP kernel (RBF) naturally models smooth relevance structure in the embedding space and supports signal propagation.

2. **UCB Acquisition Function for Active Exploration**:

   - **Function**: Balances exploration of high-uncertainty regions with exploitation of high predicted relevance.
   - **Mechanism**: $a^{\text{UCB}}(\mathbf{x}) = \mu_q(\mathbf{x}) + \sqrt{\beta}\,\sigma_q(\mathbf{x})$, where $\beta$ controls the exploration–exploitation trade-off; at each step, the unlabeled passage with the highest acquisition score is selected.
   - **Design Motivation**: Pure exploitation risks local optima, while pure exploration wastes the budget; UCB naturally balances both.

3. **Warm-Start Initialization Strategy**:

   - **Function**: Mitigates the cold-start problem and provides high-quality initial signals.
   - **Mechanism**: The query embedding $\mathbf{x}_q$ is treated as the maximum-relevance observation, combined with LLM scores for top-$M$ densely retrieved passages to form the initial observation set $\mathcal{D}_q^{(0)}$.
   - **Design Motivation**: The query itself is naturally the most relevant "passage," providing a strong positive signal and an initial anchor for the GP.

### Loss & Training

No training is required. GP hyperparameters (kernel length-scale $\ell$, noise $\alpha$) are set via standard procedures. Two LLM scoring modes are supported: Expected Relevance (ER) and Peak Relevance (PR). Anytime prediction is supported—the GP can generate rankings over all passages after any iteration.

## Key Experimental Results

### Main Results (LLM budget = 50/query)

| Dataset | Metric | BM25 | Dense Retr. | LLM Point. | BAGEL (Qwen3) | BAGEL (GPT-4o) |
|--------|------|------|-------------|-----------|---------------|----------------|
| Covid | N@50 | 42.8 | 48.7 | 52.9 | **61.4** | **62.1** |
| Robust04 | N@50 | 34.9 | 33.2 | 38.2 | **44.4** | **48.7** |
| TravelDest | N@10 | 21.1 | 22.3 | 45.8 | 49.8 | **57.0** |
| NFCorpus | N@50 | 27.7 | 29.0 | 32.7 | 32.8 | **35.9** |

### Ablation Study

| Configuration | Key Findings |
|------|------|
| RBF vs. Linear vs. Matérn kernel | RBF and Matérn perform best; Linear underperforms |
| UCB vs. EI vs. PI acquisition function | Uncertainty-aware acquisition functions (UCB) are critical |
| With/without warm-start | Warm-start substantially improves early-stage performance |

### Key Findings

- BAGEL outperforms LLM re-ranking baselines (under the same LLM budget) on all four datasets.
- NDCG@50 improves from 29.3 to 41.6 (+42%) on the TravelDest dataset.
- Stationary kernels (RBF, Matérn) effectively capture multimodal relevance structures.
- Uncertainty-guided exploration is critical for discovering relevant clusters distant from the query.

## Highlights & Insights

- Passage retrieval is elegantly reformulated as a Bayesian optimization problem, with GPs naturally fitting this setting.
- The framework addresses two core limitations of existing methods: inability to propagate relevance signals and failure to explore distant clusters.
- Anytime prediction support accommodates varying budget constraints.
- The warm-start design—treating the query as the maximum-relevance observation—is both simple and effective.

## Limitations & Future Work

- The $O(n^3)$ computational complexity of GPs limits scalability to large observation sets.
- The framework assumes that semantically similar passages in the embedding space share similar relevance, which may not always hold.
- Evaluation is conducted exclusively on English retrieval benchmarks.
- Future work may explore sparse GPs or neural kernel functions to improve scalability.

## Related Work & Insights

- **LLM re-ranking** (Zhuang et al., 2024; Sun et al., 2023): The dominant paradigm, yet fundamentally constrained by the first-stage candidate set.
- **Bayesian optimization / GP**: A classical methodology innovatively applied to a new domain (retrieval).
- **Active learning for document annotation**: Typically employed for classification rather than ranking.
- The application of GPs to information retrieval represents a direction worthy of further investigation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Combining GP and active learning for passage retrieval offers a uniquely original perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four datasets, two LLMs, and ablations over kernels and acquisition functions.
- **Writing Quality**: ⭐⭐⭐⭐ Intuitive visualizations; the connection between GP and retrieval is clearly articulated.
- **Value**: ⭐⭐⭐⭐ Substantially improves retrieval effectiveness under budget-constrained settings.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)
- [\[ACL 2026\] An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs](an_iterative_utility_judgment_framework_inspired_by_philosophical_relevance_via_.md)
- [\[ICLR 2026\] Fine-tuning with RAG for Improving LLM Learning of New Skills](../../ICLR2026/information_retrieval/fine-tuning_with_rag_for_improving_llm_learning_of_new_skills.md)
- [\[ACL 2026\] From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)
- [\[ACL 2026\] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)

<!-- RELATED:END -->
