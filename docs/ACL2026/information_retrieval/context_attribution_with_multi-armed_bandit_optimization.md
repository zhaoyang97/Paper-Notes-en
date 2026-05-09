---
title: >-
  [Paper Note] Context Attribution with Multi-Armed Bandit Optimization
description: >-
  [ACL 2026 (Findings)][Context Attribution] This paper proposes CAMAB, which frames context attribution in RAG — identifying which context segments contribute to the generated answer — as a Combinatorial Multi-Armed Bandit (CMAB) problem. Using Linear Thompson Sampling to adaptively explore the space of context subsets, CAMAB reduces model queries by up to 30% compared to SHAP and ContextCite while matching or surpassing attribution quality on HotpotQA, CNN/DM, and TyDi QA.
tags:
  - ACL 2026 (Findings)
  - Context Attribution
  - Multi-Armed Bandit
  - Thompson Sampling
  - Retrieval-Augmented Generation
  - Query Efficiency
date: 2026-05-08
content_hash: b2ba827f5e7d4c53
---

# Context Attribution with Multi-Armed Bandit Optimization

**Conference**: ACL 2026 (Findings)
**arXiv**: [2506.19977](https://arxiv.org/abs/2506.19977)
**Code**: [https://github.com/pd90506/camab](https://github.com/pd90506/camab)
**Area**: Information Retrieval / Interpretability
**Keywords**: Context Attribution, Multi-Armed Bandit, Thompson Sampling, Retrieval-Augmented Generation, Query Efficiency

## TL;DR

This paper proposes CAMAB, which frames context attribution in RAG — identifying which context segments contribute to the generated answer — as a Combinatorial Multi-Armed Bandit (CMAB) problem. Using Linear Thompson Sampling to adaptively explore the space of context subsets, CAMAB reduces model queries by up to 30% compared to SHAP and ContextCite while matching or surpassing attribution quality on HotpotQA, CNN/DM, and TyDi QA.

## Background & Motivation

**Background**: RAG enhances the factual accuracy of LLMs, yet verifying that generated answers are genuinely grounded in retrieved context remains difficult. LLMs frequently hallucinate or incorporate unsupported information, necessitating precise attribution of which context segments contribute to the answer.

**Limitations of Prior Work**: (1) Methods that train models to explicitly cite context cannot guarantee that citations faithfully reflect the underlying reasoning; (2) post-hoc perturbation methods such as SHAP and ContextCite require a large number of model queries — through uniform sampling or exhaustive feature selection — making computational costs prohibitive in long-context settings; (3) under strict query budgets, existing methods suffer significant performance degradation.

**Key Challenge**: Precise attribution requires testing a large number of context subset combinations, yet LLM inference is expensive and query budgets are limited. Uniform sampling is wasteful, as many tested subsets carry little information.

**Goal**: Achieve high-quality segment-level context attribution within a limited query budget.

**Key Insight**: The attribution problem is recast as a CMAB, where each context segment corresponds to an "arm," selecting a subset constitutes an "action," and Thompson Sampling is used to adaptively prioritize informative subsets.

**Core Idea**: Linear Thompson Sampling is employed to efficiently explore the exponentially large space of context subsets. Bayesian posterior estimation adaptively balances exploration and exploitation, converging to high-quality attributions faster than uniform random perturbation.

## Method

### Overall Architecture

CAMAB treats the $N$ segments of context $C = \{s_1, ..., s_N\}$ as the base arms of a bandit. At each iteration: (1) a weight vector is sampled from the posterior distribution; (2) segments with positive weights are selected to form a subset; (3) the LLM is queried with this subset to obtain a reward (the average log probability of the target response); (4) the posterior is updated via Bayesian updating. The final posterior mean serves as the attribution score.

### Key Designs

1. **Token Log-Probability Reward Function**:

    - Function: Quantifies the degree to which a context subset supports the original response.
    - Mechanism: Given subset $S$, the reward is $V(S) = \frac{1}{T}\sum_{t=1}^{T}\log P_M(r_t|Q, S, r_1,...,r_{t-1})$, i.e., the average log probability of the original response tokens under the given subset. This is applicable to both open-source and black-box API models (requiring only a log-probability interface).
    - Design Motivation: Log-probability is a direct internal measure of how well a context subset supports the original response, and is more accurate than external metrics such as text matching.

2. **Linear Thompson Sampling (LinTS)**:

    - Function: Efficiently balances exploration and exploitation in the combinatorial space.
    - Mechanism: The reward is assumed to follow $V(S) = \mathbf{w}^\top \mathbf{x} + \epsilon$, where $\mathbf{x}$ is a binary inclusion vector over segments. A Gaussian posterior $\mathcal{N}(\hat{\boldsymbol{\mu}}_t, \mathbf{B}_t^{-1})$ is maintained over the weight vector $\mathbf{w}$. At each round, weights are sampled, segments with positive weights are queried, and the posterior is updated. The off-diagonal elements of the precision matrix $\mathbf{B}_t$ implicitly capture inter-segment interactions.
    - Design Motivation: LinTS is among the most exploration-efficient bandit algorithms. Compared to SHAP's uniform sampling and ContextCite's Lasso regression, Bayesian adaptive sampling converges more rapidly to important segments.

3. **Combinatorial Super-Arm Representation**:

    - Function: Encodes the exponentially large subset space into a tractable linear form.
    - Mechanism: Each subset is represented as an $(N+1)$-dimensional binary vector (with a bias term in the first dimension). The linear assumption decomposes the combinatorial problem into estimation of each segment's marginal contribution. The precision matrix $\mathbf{B}_t$ accumulates co-selection statistics across iterations, implicitly capturing substitutability and complementarity among segments.
    - Design Motivation: Direct search over the $2^N$ space is infeasible. Although the linear assumption simplifies interaction effects, this limitation is partially compensated by the correlation structure encoded in the precision matrix.

### Loss & Training

CAMAB is an inference-time method and involves no model training. The algorithm operates iteratively within a given query budget $T_{max}$. The prior is set as $\mathbf{w} \sim \mathcal{N}(\hat{\boldsymbol{\mu}}_0, \sigma_p^2 \mathbf{I})$, with noise variance $\sigma^2$ as a hyperparameter. The $O(N^3)$ posterior update is negligible compared to the cost of LLM inference.

## Key Experimental Results

### Main Results

**Attribution Performance on LLaMA-3.1-8B (Query Budget 40)**

| Dataset | Metric | CAMAB | SHAP | ContextCite | Random |
|--------|------|-------|------|-------------|--------|
| HotpotQA | Log-P Drop@5 ↑ | **0.717** | 0.648 | 0.632 | 0.103 |
| HotpotQA | BERTScore@5 ↓ | **0.407** | 0.453 | 0.496 | 0.703 |
| CNN/DM | Log-P Drop@5 ↑ | **1.129** | 1.041 | 1.025 | 0.389 |
| TyDi QA | Log-P Drop@5 ↑ | **0.893** | 0.872 | 0.631 | 0.373 |

### Ablation Study

| Query Budget | CAMAB BERTScore@1 | SHAP | ContextCite |
|---------|-------------------|------|-------------|
| 20 | **0.525** | 0.668 | 0.605 |
| 40 | **0.509** | 0.562 | 0.601 |
| 60 | **0.511** | 0.527 | 0.598 |

**Alignment with Human Annotations (HotpotQA, 200 samples)**

| Method | P@1 | AUROC | AP |
|------|-----|-------|-----|
| **CAMAB** | **0.780** | **0.855** | **0.688** |
| SHAP | 0.680 | 0.806 | 0.598 |
| Random | 0.055 | 0.516 | 0.162 |

### Key Findings

- At a budget of 40, CAMAB already surpasses SHAP at a budget of 60, yielding approximately 30% improvement in sampling efficiency.
- The advantage is largest under extremely low budgets (20): SHAP degrades sharply, while CAMAB maintains high fidelity.
- The performance gap on CNN/DM is smaller (~1%), as the lead bias in news summarization allows all methods to converge rapidly.
- CAMAB aligns closely with gold-standard supporting facts from human annotations (P@1=0.780, AUROC=0.855).
- The correlation structure of the precision matrix demonstrably captures inter-segment interactions — clustering of topically related segments reveals substitutability.

## Highlights & Insights

- The problem formulation is elegant — recasting attribution as CMAB represents a natural and effective transfer of framework.
- The adaptive exploration of LinTS is the key driver — efficiency gains stem not from "more exploration" but from "smarter exploration."
- The method requires only a log-probability interface, making it applicable to black-box APIs and thus highly practical.

## Limitations & Future Work

- The linear assumption may miss strong interaction effects, such as cases where two segments are only meaningful in conjunction.
- A token-level log-probability interface is required, which is not available from all APIs.
- The method may converge to suboptimal solutions in high-noise or highly ambiguous settings.
- Future work may explore non-linear bandits or attention-guided initialization strategies.

## Related Work & Insights

- **vs. ContextCite**: ContextCite performs feature selection via Lasso regression, requiring more samples in high-dimensional spaces; CAMAB's Bayesian approach is more robust under small-sample conditions.
- **vs. SHAP**: SHAP's uniform random sampling does not exploit historical information; CAMAB learns adaptively to guide subsequent sampling.
- **vs. LIME**: LIME relies on local linear approximation, whereas CAMAB's global linear assumption is more appropriate in the attribution setting.

## Rating

- Novelty: ⭐⭐⭐⭐ The CMAB formulation is a clever problem reframing, though Linear Thompson Sampling itself is an established algorithm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, two models, multi-budget comparisons, and human annotation validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is clear and algorithm description is concise.
- Value: ⭐⭐⭐⭐ Provides an efficient and practical solution for RAG attribution.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation](../../ICLR2026/information_retrieval/attributing_response_to_context_a_jensen-shannon_divergence_driven_mechanistic_s.md)
- [\[ACL 2026\] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits](mab-dqa_addressing_query_aspect_importance_in_document_question_answering_with_m.md)
- [\[ACL 2026\] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)
- [\[ACL 2026\] How Retrieved Context Shapes Internal Representations in RAG](how_retrieved_context_shapes_internal_representations_in_rag.md)
- [\[ICLR 2026\] Q-RAG: Long Context Multi-Step Retrieval via Value-Based Embedder Training](../../ICLR2026/information_retrieval/q_rag_long_context_multi_step_retrieval.md)

<!-- RELATED:END -->
