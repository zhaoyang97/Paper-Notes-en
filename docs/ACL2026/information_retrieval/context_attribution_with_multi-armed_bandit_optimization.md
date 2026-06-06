---
title: >-
  [Paper Note] Context Attribution with Multi-Armed Bandit Optimization
description: >-
  [ACL 2026 (Findings)][Information Retrieval & RAG][Context Attribution] This paper proposes CAMAB, which models context attribution in RAG (identifying which context fragments contribute to generated answers) as a Combin…
tags:
  - "ACL 2026 (Findings)"
  - "Information Retrieval & RAG"
  - "Context Attribution"
  - "Multi-Armed Bandit"
  - "Thompson Sampling"
  - "Retrieval-Augmented Generation"
  - "Query Efficiency"
date: 2026-05-08
content_hash: 2326e79d3e9ff013
---

# Context Attribution with Multi-Armed Bandit Optimization

**Conference**: ACL 2026 (Findings)  
**arXiv**: [2506.19977](https://arxiv.org/abs/2506.19977)  
**Code**: [https://github.com/pd90506/camab](https://github.com/pd90506/camab)  
**Area**: Information Retrieval / Explainability  
**Keywords**: Context Attribution, Multi-Armed Bandit, Thompson Sampling, Retrieval-Augmented Generation, Query Efficiency

## TL;DR

This paper proposes CAMAB, which models context attribution in RAG (identifying which context fragments contribute to generated answers) as a Combinatorial Multi-Armed Bandit (CMAB) problem. By using Linear Thompson Sampling to adaptively explore the space of context subsets, it reduces model queries by up to 30% compared to SHAP and ContextCite while matching or exceeding attribution quality on HotpotQA, CNN/DM, and TyDi QA.

## Background & Motivation

**Background**: RAG enhances factual accuracy of LLMs, but verifying that generated answers are indeed based on retrieved context remains difficult. LLMs often hallucinate or incorporate ungrounded information, necessitating a precise attribution of which context fragments contribute to the answer.

**Limitations of Prior Work**: (1) Training models to explicitly cite context cannot guarantee that citations reflect actual reasoning; (2) Post-hoc perturbation methods such as SHAP and ContextCite require a large volume of model queries (uniform sampling or full feature selection), resulting in computational costs that are unacceptable in long-context scenarios; (3) Performance drops sharply when budgets are strictly limited.

**Key Challenge**: Precise attribution requires testing many combinations of context subsets, but LLM inference is expensive and query budgets are finite. Uniform sampling is wasteful—many tested subsets yield low informative value.

**Goal**: Achieve high-quality segment-level context attribution within a limited query budget.

**Key Insight**: Reframe the attribution problem into CMAB—each context fragment is treated as an "arm," selecting a subset is an "action," and Thompson Sampling is used to adaptively prioritize the exploration of informative subsets.

**Core Idea**: Use Linear Thompson Sampling to explore the exponential space of context subsets efficiently. By leveraging Bayesian posterior estimation to adaptively balance exploration and exploitation, it converges to high-quality attribution faster than uniform random perturbations.

## Method

### Overall Architecture

CAMAB treats $N$ fragments of the context $C = \{s_1, ..., s_N\}$ as the base arms of the bandit. In each iteration: (1) Sample a weight vector from the posterior distribution; (2) Select fragments with positive weights to form a subset; (3) Query the LLM with the subset to obtain a reward (average log-probability of the target response); (4) Perform a Bayesian update on the posterior. The final posterior mean serves as the attribution score.

### Key Designs

1.  **Token log-probability based reward function**:

    - **Function**: Quantifies the extent to which a context subset supports the original response.
    - **Mechanism**: Given a subset $S$, the reward is defined as $V(S) = \frac{1}{T}\sum_{t=1}^{T}\log P_M(r_t|Q, S, r_1,...,r_{t-1})$, which is the average log-probability of the original response tokens under that subset. This applies to both open-source and black-box API models (requiring only a log-probability interface).
    - **Design Motivation**: Log-probability is a direct internal measure of how well a context subset can support the original response, making it more accurate than external metrics like text matching.

2.  **Linear Thompson Sampling (LinTS)**:

    - **Function**: Efficiently balances exploration and exploitation within the combinatorial space.
    - **Mechanism**: Assumes the reward follows $V(S) = \mathbf{w}^\top \mathbf{x} + \epsilon$, where $\mathbf{x}$ is the binary inclusion vector of the fragments. It maintains a Gaussian posterior $\mathcal{N}(\hat{\boldsymbol{\mu}}_t, \mathbf{B}_t^{-1})$ for the weights $\mathbf{w}$. Each round, it samples weights, selects fragments with positive weights, queries the model, and updates the posterior. The non-diagonal elements of the precision matrix $\mathbf{B}_t$ implicitly capture interactions between fragments.
    - **Design Motivation**: LinTS is among the most exploration-efficient bandit algorithms. Compared to the uniform sampling of SHAP and the Lasso regression of ContextCite, Bayesian adaptive sampling locks onto important fragments faster.

3.  **Combinatorial Super-arm Representation**:

    - **Function**: Encodes the exponential subset space into a tractable linear form.
    - **Mechanism**: Each subset is represented by an $(N+1)$-dimensional binary vector (with the first dimension as a bias term). The linear assumption decomposes the combinatorial problem into marginal contribution estimations for each fragment. The precision matrix $\mathbf{B}_t$ accumulates co-selection statistics during iterations, implicitly capturing the substitutability and complementarity of fragments.
    - **Design Motivation**: Direct searching in $2^N$ space is infeasible. While the linear assumption simplifies interaction effects, the correlation structure of the precision matrix partially compensates for this limitation.

### Loss & Training

CAMAB is an inference-time method and does not involve model training. The algorithm runs iteratively within a given query budget $T_{max}$. The prior is set as $\mathbf{w} \sim \mathcal{N}(\hat{\boldsymbol{\mu}}_0, \sigma_p^2 \mathbf{I})$, and the noise variance $\sigma^2$ is a hyperparameter. $O(N^3)$ posterior updates are negligible compared to LLM inference overhead.

## Key Experimental Results

### Main Results

**Attribution Performance on LLaMA-3.1-8B (Query Budget 40)**

| Dataset | Metric | CAMAB | SHAP | ContextCite | Random |
|---------|--------|-------|------|-------------|--------|
| HotpotQA | Log-P Drop@5 ↑ | **0.717** | 0.648 | 0.632 | 0.103 |
| HotpotQA | BERTScore@5 ↓ | **0.407** | 0.453 | 0.496 | 0.703 |
| CNN/DM | Log-P Drop@5 ↑ | **1.129** | 1.041 | 1.025 | 0.389 |
| TyDi QA | Log-P Drop@5 ↑ | **0.893** | 0.872 | 0.631 | 0.373 |

### Ablation Study

| Query Budget | CAMAB BERTScore@1 | SHAP | ContextCite |
|--------------|-------------------|------|-------------|
| 20 | **0.525** | 0.668 | 0.605 |
| 40 | **0.509** | 0.562 | 0.601 |
| 60 | **0.511** | 0.527 | 0.598 |

**Alignment with Human Annotation (HotpotQA, 200 samples)**

| Method | P@1 | AUROC | AP |
|--------|-----|-------|----|
| **CAMAB** | **0.780** | **0.855** | **0.688** |
| SHAP | 0.680 | 0.806 | 0.598 |
| Random | 0.055 | 0.516 | 0.162 |

### Key Findings

- CAMAB at a budget of 40 already exceeds the performance of SHAP at a budget of 60, improving sampling efficiency by approximately 30%.
- The advantage is most significant at extremely low budgets (20)—where SHAP's performance drops sharply, CAMAB maintains high fidelity.
- The gap is smaller on CNN/DM (~1%) because the head bias in news summaries allows all methods to converge quickly.
- Highly aligned with human-annotated gold support facts (P@1=0.780, AUROC=0.855).
- The correlation structure of the precision matrix effectively captures fragment interactions—clustering of same-topic fragments indicates substitutability.

## Highlights & Insights

- The problem formulation is elegant—transforming attribution into CMAB is a natural and effective framework transfer.
- The adaptive exploration of LinTS is key—efficiency gains come from "smarter exploration" rather than "more exploration."
- Requiring only a log-probability interface makes it highly practical for black-box APIs.

## Limitations & Future Work

- Linear assumptions may miss strong interaction effects (e.g., when two fragments are only meaningful when combined).
- Requires a token-level log-probability interface, which is not provided by all APIs.
- May converge to sub-optimal solutions in high-noise or high-ambiguity scenarios.
- Future work can explore non-linear bandits or initialization strategies guided by attention.

## Related Work & Insights

- **vs ContextCite**: ContextCite uses Lasso regression for feature selection, requiring more samples in high-dimensional spaces; CAMAB's Bayesian approach is more robust under small sample sizes.
- **vs SHAP**: SHAP's uniform random sampling does not utilize historical information; CAMAB adaptively learns to guide subsequent sampling.
- **vs LIME**: LIME requires local linear approximation; CAMAB's global linear assumption is more appropriate in the attribution context.

## Rating

- Novelty: ⭐⭐⭐⭐ CMAB formulation is a clever problem reconstruction, though Linear Thompson Sampling itself is an existing algorithm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, two models, multi-budget comparisons, and validation with human annotation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition and concise algorithm description.
- Value: ⭐⭐⭐⭐ Provides an efficient and practical solution for RAG attribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation](../../ICLR2026/information_retrieval/attributing_response_to_context_a_jensen-shannon_divergence_driven_mechanistic_s.md)
- [\[ACL 2026\] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits](mab-dqa_addressing_query_aspect_importance_in_document_question_answering_with_m.md)
- [\[ACL 2026\] IF-GEO: Conflict-Aware Instruction Fusion for Multi-Query Generative Engine Optimization](if-geo_conflict-aware_instruction_fusion_for_multi-query_generative_engine_optim.md)
- [\[ACL 2026\] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)
- [\[ACL 2026\] BRIEF-Pro: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning](brief-pro_universal_context_compression_with_short-to-long_synthesis_for_fast_an.md)

</div>

<!-- RELATED:END -->
