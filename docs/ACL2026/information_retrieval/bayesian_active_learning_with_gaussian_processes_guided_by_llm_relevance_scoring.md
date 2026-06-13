---
title: >-
  [Paper Note] Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring
description: >-
  [ACL 2026][Information Retrieval & RAG][Passage Retrieval] Ours proposes BAGEL, a Bayesian active learning framework based on Gaussian Processes (GP). Under limited LLM budgets…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Passage Retrieval"
  - "Gaussian Processes"
  - "Active Learning"
  - "LLM Reranking"
  - "Bayesian Optimization"
date: 2026-05-08
content_hash: 6de14335c7ac9e0d
---

# Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.17906](https://arxiv.org/abs/2604.17906)  
**Code**: [GitHub](https://github.com/junieberry/BAGEL)  
**Area**: Information Retrieval  
**Keywords**: Passage Retrieval, Gaussian Processes, Active Learning, LLM Reranking, Bayesian Optimization

## TL;DR

Ours proposes BAGEL, a Bayesian active learning framework based on Gaussian Processes (GP). Under limited LLM budgets, it propagates sparse LLM relevance signals through an exploration-exploitation balancing strategy, achieving passage retrieval across the global embedding space and significantly outperforming traditional LLM reranking methods.

## Background & Motivation

**Background**: LLMs possess excellent zero-shot relevance modeling capabilities, but high computational costs turn passage retrieval into a budget-constrained global optimization problem. Prevailing methods adopt an LLM reranking paradigm: first using a dense retriever to obtain top-K candidates, then performing LLM reranking.

**Limitations of Prior Work**: (1) Relevant passages are often distributed across multiple distinct clusters in the semantic space, while dense retrievers only retrieve local neighborhoods near the query embedding, failing to discover distant relevant clusters; (2) existing methods cannot propagate relevance signals from scored passages to unseen ones, ignoring the semantic structure of the embedding space.

**Key Challenge**: The need to explore the entire embedding space under a limited LLM inference budget, whereas traditional methods passively rely on first-stage retrievers and cannot perform global exploration.

**Goal**: Utilize the kernel correlation propagation and uncertainty estimation capabilities of GPs to actively navigate the embedding space and discover multimodal relevance distributions.

**Key Insight**: Modeling passage retrieval as a Bayesian optimization problem, where the GP provides predicted means and uncertainties, and an acquisition function balances exploration and exploitation.

**Core Idea**: GPs are naturally suited for this task—kernels propagate relevance signals, and posterior variance guides active learning to explore uncertain regions.

## Method

### Overall Architecture

BAGEL consists of two stages: (1) Warm-up initialization—treating the query itself as the highest relevance observation, combined with LLM scores of the top-M dense-retrieved passages; (2) Active learning exploration—iteratively selecting the next passage for LLM scoring via an acquisition function (UCB), updating the GP posterior, and finally generating rankings for all passages.

### Key Designs

1.  **Query-specific Gaussian Process**:
    - **Function**: Models the query-passage relevance function over the embedding space.
    - **Mechanism**: The GP takes passage embeddings $\mathbf{x}_p$ as input and LLM relevance scores as output. The posterior predictive mean $\mu_q(\mathbf{x}_{p_*})$ and variance $\sigma_q^2(\mathbf{x}_{p_*})$ provide relevance estimation and uncertainty, respectively.
    - **Design Motivation**: The GP kernel (e.g., RBF) naturally models the smooth relevance structure in the embedding space, supporting signal propagation.

2.  **UCB Acquisition Function Guided Active Exploration**:
    - **Function**: Balances the exploration of high-uncertainty regions and the exploitation of high-predicted-relevance regions.
    - **Mechanism**: $a^{\text{UCB}}(\mathbf{x}) = \mu_q(\mathbf{x}) + \sqrt{\beta}\,\sigma_q(\mathbf{x})$, where $\beta$ controls the exploration-exploitation trade-off, selecting the highest-scoring unlabeled passage at each step.
    - **Design Motivation**: Pure exploitation traps the model in local optima, while pure exploration wastes budget; UCB naturally balances both.

3.  **Warm-up Initialization Strategy**:
    - **Function**: Mitigates the cold-start problem and provides high-quality initial signals.
    - **Mechanism**: Uses the query embedding $\mathbf{x}_q$ as a maximum relevance observation, combined with LLM scores of top-M dense-retrieved passages to form the initial observation set $\mathcal{D}_q^{(0)}$.
    - **Design Motivation**: The query is inherently the most "relevant" passage, providing a strong positive signal and an initial anchor for the GP.

### Loss & Training

Training-free. GP hyperparameters (kernel length scale $\ell$, noise $\alpha$) are set via standard methods. Supports two LLM scoring modes: Expected Relevance (ER) and Peak Relevance (PR). Supports anytime prediction—the GP can generate rankings for all passages after any number of iterations.

## Key Experimental Results

### Main Results (LLM Budget = 50 per query)

| Dataset | Metric | BM25 | Dense Retr. | LLM Point. | BAGEL (Qwen3) | BAGEL (GPT-4o) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Covid | N@50 | 42.8 | 48.7 | 52.9 | **61.4** | **62.1** |
| Robust04 | N@50 | 34.9 | 33.2 | 38.2 | **44.4** | **48.7** |
| TravelDest | N@10 | 21.1 | 22.3 | 45.8 | 49.8 | **57.0** |
| NFCorpus | N@50 | 27.7 | 29.0 | 32.7 | 32.8 | **35.9** |

### Ablation Study

| Configuration | Finding |
| :--- | :--- |
| RBF vs Linear vs Matérn Kernels | RBF and Matérn perform best; Linear is inferior. |
| UCB vs EI vs PI Acquisition | Uncertainty-related acquisition functions (UCB) are critical. |
| With vs Without Warm-up | Warm-up significantly improves early-stage performance. |

### Key Findings

- BAGEL outperforms LLM reranking baselines on all four datasets under the same LLM budget.
- On the TravelDest dataset, NDCG@50 increased from 29.3 to 41.6 (+42%).
- Stationary kernels (RBF, Matérn) effectively capture multimodal relevance structures.
- Uncertainty-guided exploration is crucial for discovering relevant clusters far from the query.

## Highlights & Insights

- Elegantly transforms passage retrieval into a Bayesian optimization problem, with GPs naturally fitting this scenario.
- Addresses two major limitations of existing methods: the inability to propagate relevance signals and the failure to explore distant clusters.
- Supports anytime prediction, adapting to different budget constraints.
- The design of warm-up plus using the query as the maximum relevance observation is simple yet effective.

## Limitations & Future Work

- The $O(n^3)$ computational complexity of GPs limits the size of the observation set.
- The assumption that semantically close passages in the embedding space have similar relevance may not always hold.
- Evaluation was limited to English retrieval.
- Future work could explore sparse GPs or neural kernels to improve scalability.

## Related Work & Insights

- LLM Reranking (Zhuang et al., 2024; Sun et al., 2023): Dominant but limited by the first-stage candidate set.
- Bayesian Optimization/GP: Innovative application of classic methods in a new scenario (retrieval).
- Active Learning for Document Annotation: Usually applied to classification rather than ranking.
- The application of GPs in Information Retrieval is a direction worth further exploration.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Unique perspective using GP + Active Learning for passage retrieval.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, two LLMs, and ablations on kernels/acquisition functions.
- Writing Quality: ⭐⭐⭐⭐ Intuitive diagrams; clear explanation of the link between GP and retrieval.
- Value: ⭐⭐⭐⭐ Significantly improves retrieval effectiveness in budget-constrained scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)
- [\[ACL 2026\] An Iterative Utility Judgment Framework Inspired by Philosophical Relevance via LLMs](an_iterative_utility_judgment_framework_inspired_by_philosophical_relevance_via_.md)
- [\[ACL 2026\] From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)
- [\[ICLR 2026\] Fine-tuning with RAG for Improving LLM Learning of New Skills](../../ICLR2026/information_retrieval/fine-tuning_with_rag_for_improving_llm_learning_of_new_skills.md)
- [\[ACL 2026\] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation](learning_to_extract_rational_evidence_via_reinforcement_learning_for_retrieval-a.md)

</div>

<!-- RELATED:END -->
