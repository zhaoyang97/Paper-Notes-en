---
title: >-
  [Paper Note] Optimizing User Profiles via Contextual Bandits for Retrieval-Augmented LLM Personalization
description: >-
  [ACL 2026][Information Retrieval & RAG][User Profile Optimization] The PURPLE framework is proposed, modeling the user profiling problem in retrieval-augmented LLM personalization as a contextual bandit problem. It captu…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "User Profile Optimization"
  - "Contextual Bandits"
  - "RAG Personalization"
  - "Plackett-Luce Ranking"
  - "Policy Gradient"
date: 2026-05-08
content_hash: c926713978489342
---

# Optimizing User Profiles via Contextual Bandits for Retrieval-Augmented LLM Personalization

**Conference**: ACL 2026  
**arXiv**: [2601.12078](https://arxiv.org/abs/2601.12078)  
**Code**: [GitHub](https://github.com/LinfengDu/PURPLE)  
**Area**: Reinforcement Learning  
**Keywords**: User Profile Optimization, Contextual Bandits, RAG Personalization, Plackett-Luce Ranking, Policy Gradient

## TL;DR

The PURPLE framework is proposed, modeling the user profiling problem in retrieval-augmented LLM personalization as a contextual bandit problem. It captures dependencies between records using a Plackett-Luce ranking model and directly optimizes retrieval to match generation quality using the LLM's log-likelihood of reference responses as the reward signal.

## Background & Motivation

- **Background**: LLM personalization is a prominent research direction. Parametric fine-tuning methods based on RLHF are computationally expensive and unsuitable for large-scale real-time personalization. Retrieval-augmented personalization (RAG) guides LLM generation by injecting user history into prompts, offering a lightweight, transparent, and deployable alternative.
- **Limitations of Prior Work**: Existing methods select historical records for user profiles based on semantic relevance, but relevance is not a reliable proxy for utility. A record might be semantically similar to a query but harm generation quality due to redundancy or information conflict. For example, when a user searches for "relaxing Friday night movies," keyword matching might prioritize suspenseful thrillers containing "Friday night" over comedies that truly reflect "relaxing" intent.
- **Key Challenge**: (1) The utility of an individual record depends on the context of other records—combinatorial utility is non-additive, making greedy top-k selection suboptimal; (2) existing list-wise re-rankers can model dependencies but remain constrained by relevance-oriented supervision signals.
- **Goal**: Design a re-ranking mechanism that directly optimizes downstream generation quality and is sensitive to interactions between records.
- **Key Insight**: relevance $\neq$ utility; use the LLM's log-likelihood of reference responses as a semantically rich reward signal to train a policy network that accounts for inter-record dependencies.
- **Core Idea**: Treat user profiling as an order-sensitive combinatorial selection problem, optimized directly via policy gradients within a contextual bandit framework.

## Method

### Overall Architecture

PURPLE acts as a re-ranking module layered on top of initial retrieval (e.g., Contriever retrieving 20 candidates). A user record encoder receives the query and candidate records, outputting a propensity score for each record. During training, a probability distribution is generated from these scores using a Plackett-Luce model, and $M=32$ profiles are sampled for policy gradient estimation. During inference, the $K$ records with the highest propensity scores are selected.

### Key Designs

1.  **Plackett-Luce Ranking Policy ($\pi_\theta$)**:
    - **Function**: Transforms propensity scores into a probability distribution over ordered profiles to support order-sensitive sampling.
    - **Mechanism**: Each record obtains a propensity score $f_\theta(h_i; C) \in [0, 1]$ via the encoder. The PL model converts these into the probability of a $K$-permutation: $\pi_\theta(P|C) = \prod_{k=1}^{K} f_\theta(p_k) / [S - \sum_{j<k} f_\theta(p_j)]$. Training involves sampling $K$ records without replacement, while inference uses top-K selection.
    - **Design Motivation**: The PL model inherently models order sensitivity—different permutations have different probabilities—and supports efficient sampling, making it suitable for policy gradient optimization.

2.  **User Record Encoder ($f_\theta$)**:
    - **Function**: Captures interaction relationships between query-record and record-record pairs.
    - **Mechanism**: Employs a late-interaction strategy—first using a pre-trained Contriever to obtain token embeddings. Each record undergoes cross-attention with the query at the token level to produce a query-fused representation, which is pooled into a fixed-size record embedding. Finally, a Transformer encoder (without positional encodings) models dependencies between records.
    - **Design Motivation**: Processing all records jointly at the token level would exceed the encoder's context window; late interaction maintains fine-grained interaction while controlling computational complexity.

3.  **Log-Likelihood Reward Function**:
    - **Function**: Provides semantically rich training signals that directly reflect generation quality.
    - **Mechanism**: $R(LLM(P\|x), y) = \log p_\phi(y|P,x) = \sum \log p_\phi(y_j|P,x,y_{<j})$, representing the token-level log-likelihood of the LLM given the reference response. Compared to coarse-grained metrics like Accuracy or ROUGE, log-likelihood distinguishes between "feasible" and "optimal" profiles.
    - **Design Motivation**: The authors further demonstrate that using log-likelihood reward is equivalent to maximizing the ELBO of the RAG marginalization formula, providing theoretical guarantees.

### Loss & Training

- Uses REINFORCE policy gradient: $\nabla_\theta J(\theta) = E[\nabla_\theta \log \pi_\theta(P|C) \cdot R(LLM(P\|x), y)]$.
- Samples $M=32$ profiles per instance, applying z-score normalization to rewards to stabilize training.
- LLM parameters are frozen; only the record encoder parameters $\theta$ are trained.
- Selects $K=5$ records from $N=20$ candidates to construct the profile.

## Key Experimental Results

### Main Results (LaMP Benchmark, 6 Tasks)

| Method | Citation Acc/F1 | Movie Acc/F1 | Rating MAE/RMSE | News RG1/RGL/MT | Scholar RG1 | Tweet RG1 |
|---|---|---|---|---|---|---|
| **Phi-4-Mini (3.84B)** |
| Contriever | 64.6/64.5 | 36.0/31.1 | 0.424/0.830 | 14.6/13.1/12.2 | 39.7 | 38.6 |
| ICR (Llama-3-8B) | 65.2/65.0 | 34.1/29.8 | 0.424/0.830 | 15.0/13.4/12.5 | 39.5 | 38.6 |
| **PURPLE** | **66.0/65.6** | **38.6/34.2** | **0.419/0.808** | **15.1/13.5/12.6** | **40.0** | 39.0 |
| **Llama-3-8B (8.03B)** |
| Contriever | 58.5/58.1 | 47.2/39.1 | 0.314/0.631 | 17.2/15.6/15.1 | 41.1 | 32.1 |
| ICR (Llama-3-8B) | 58.4/57.3 | 48.0/39.3 | 0.312/0.631 | 17.1/15.4/14.9 | 41.3 | 31.8 |
| **PURPLE** | 59.2/**58.8** | **49.6/41.6** | **0.307/0.624** | **17.6/15.9/15.3** | 41.4 | **32.5** |

PURPLE consistently outperforms all baselines across 3 LLM scales (3.84B/8B/70B) and 9 tasks.

### Ablation Study (Phi-4-Mini)

| Variant | Citation Acc | Movie Acc | Rating MAE | News RG1 |
|---|---|---|---|---|
| PURPLE (Full) | 66.2 | 38.2 | 0.405 | 15.2 |
| w/o Cross-Attention | 64.8 | 35.1 | 0.440 | 14.8 |
| w/o Inter-record dependency modeling | 61.3 | 35.0 | 0.449 | 14.5 |
| w/ Metric reward replacement | 64.8 | 38.0 | 0.433 | 15.0 |

Removing inter-record dependency modeling (Transformer encoder) resulted in the largest performance drop, validating the necessity of profile-level holistic optimization.

### Key Findings

- **Relevance $\neq$ Utility**: PURPLE's propensity scores provide a more effective ranking signal than raw relevance, even when using a much smaller model than RankGPT.
- **Order Sensitivity Matters**: The record order selected by PURPLE was most frequently ranked as optimal among 120 permutations, indicating its scores capture relative dependencies between records.
- **Log-likelihood Reward is Universal**: Even on regression tasks (Rating), log-likelihood reward outperforms task-specific metric rewards.
- **Human Evaluation Lead of 14.4%**: In a blind test of the Tweet task, evaluators preferred PURPLE's generated results 57.2% vs 42.8%.
- **Optimal Profile Size $K=5$**: Performance slightly decreased when increasing to 10 or 15, validating the assumption of non-monotonic utility.

## Highlights & Insights

- **Elegant Problem Formulation**: User profiling is transformed into a combinatorial selection problem using contextual bandits, with the Plackett-Luce model naturally handling order sensitivity and combinatorial dependencies.
- **Deep Theoretical Connection**: Proving that log-likelihood reward corresponds to maximizing the ELBO of the RAG marginalization formula elevates it from an experimental heuristic to a theoretically grounded method.
- **High Practical Value**: No LLM fine-tuning is required. The encoder is lightweight, and inference requires only a single forward pass to retrieve top-K, balancing performance and efficiency.
- **Core Insight on Relevance vs. Utility**: This distinction is applicable not only to personalization but offers inspiration for all RAG scenarios.

## Limitations & Future Work

- Reliance on high-quality reference responses to calculate log-likelihood rewards; in actual deployment, explicit supervision may be sparse or unavailable (e.g., only implicit feedback exists).
- Currently, policies are trained independently for each task; cross-task/domain generalization has not been verified.
- The candidate pool size is fixed at 20; effectiveness and efficiency with larger candidate pools remain to be tested.
- Future Work: Exploration of training under weak supervision/implicit feedback, unified multi-task policies, and deeper integration with RAG pipelines.

## Related Work & Insights

- **REPLUG (Shi et al., 2024)**: Marginalizes over multiple retrieved records but processes each record independently, failing to model inter-record dependencies.
- **IC-RALM (Ram et al., 2023)**: Periodically triggers retrieval and replaces context during decoding, also handling records independently.
- **RankGPT (Sun et al., 2023)**: A zero-shot LLM re-ranker with high inference costs that optimizes for relevance rather than utility.
- **ICR (Chen et al., 2025)**: Utilizes attention mechanisms for zero-shot re-ranking; efficient but still relevance-oriented.
- **LaMP / LongLaMP (Salemi et al., 2024; Kumar et al., 2024)**: Personalization benchmarks covering classification, regression, and generation tasks.
- Insight: Shifting RAG retrieval optimization from relevance-oriented to utility-oriented is a direction worth broader exploration.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The combination of Contextual Bandits + Plackett-Luce + Log-likelihood rewards is very elegant; the insight into relevance vs. utility is profound.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 tasks, 3 LLM scales, multiple baselines, ablations, human evaluation, and sensitivity analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ The movie recommendation example for motivation is intuitive; methodological derivation is clear with solid theoretical links.
- Value: ⭐⭐⭐⭐⭐ Proposes a new paradigm for retrieval-augmented personalization with broad potential impact on the RAG community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AuthorityBench: Benchmarking LLM Authority Perception for Reliable Retrieval-Augmented Generation](authoritybench_benchmarking_llm_authority_perception_for_reliable_retrieval-augm.md)
- [\[AAAI 2026\] Exposing the Cracks: Vulnerabilities of Retrieval-Augmented LLM-Based Machine Translation](../../AAAI2026/information_retrieval/exposing_the_cracks_vulnerabilities_of_retrieval-augmented_llm-based_machine_tra.md)
- [\[ACL 2026\] How Large Language Models Balance Internal Knowledge with User and Document Assertions](how_large_language_models_balance_internal_knowledge_with_user_and_document_asse.md)
- [\[ACL 2026\] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits](mab-dqa_addressing_query_aspect_importance_in_document_question_answering_with_m.md)
- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](feedback_adaptation_for_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
