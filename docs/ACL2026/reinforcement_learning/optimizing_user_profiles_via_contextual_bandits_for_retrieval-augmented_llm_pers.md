---
title: >-
  [Paper Note] Optimizing User Profiles via Contextual Bandits for Retrieval-Augmented LLM Personalization
description: >-
  [ACL 2026][Reinforcement Learning][user profile optimization] This paper proposes PURPLE, a framework that models user profile construction in retrieval-augmented LLM personalization as a contextual bandit problem. It em…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "user profile optimization"
  - "contextual bandits"
  - "RAG personalization"
  - "Plackett-Luce ranking"
  - "policy gradient"
date: 2026-05-08
content_hash: 0248857e917c20d2
---

# Optimizing User Profiles via Contextual Bandits for Retrieval-Augmented LLM Personalization

**Conference**: ACL 2026
**arXiv**: [2601.12078](https://arxiv.org/abs/2601.12078)
**Code**: [GitHub](https://github.com/LinfengDu/PURPLE)
**Area**: Reinforcement Learning
**Keywords**: user profile optimization, contextual bandits, RAG personalization, Plackett-Luce ranking, policy gradient

## TL;DR

This paper proposes PURPLE, a framework that models user profile construction in retrieval-augmented LLM personalization as a contextual bandit problem. It employs the Plackett-Luce ranking model to capture inter-record dependencies, uses the LLM's log-likelihood over reference responses as a reward signal, and directly optimizes retrieval to align with generation quality.

## Background & Motivation

- **Background**: LLM personalization is an active research area. Parameter fine-tuning via RLHF is computationally expensive and poorly suited for large-scale real-time personalization. Retrieval-augmented personalization injects user history into the prompt to guide personalized generation, offering a lightweight, transparent, and deployable alternative.
- **Limitations of Prior Work**: Existing methods select historical records based on semantic relevance to construct user profiles, but relevance is not a reliable proxy for utility. A record may be semantically similar to a query yet degrade generation quality due to redundancy or conflicting information. For example, when a user searches for "a relaxing Friday night movie," keyword-matching approaches would prioritize thriller records containing "Friday night" over comedy records that better reflect the user's intent to unwind.
- **Key Challenge**: (1) The utility of an individual record depends on the context of other selected records—combinatorial utility is non-additive, making greedy top-$k$ selection suboptimal; (2) existing listwise rerankers can model dependencies but remain constrained by relevance-oriented supervision signals.
- **Goal**: To design a reranking mechanism that directly optimizes downstream generation quality while remaining sensitive to inter-record interactions.
- **Key Insight**: User profile construction is treated as an order-sensitive combinatorial selection problem, directly optimized via policy gradient within a contextual bandit framework.
- **Core Idea**: Relevance $\neq$ utility. The LLM's log-likelihood over reference responses serves as a semantically rich reward signal for training a policy network that accounts for inter-record dependencies.

## Method

### Overall Architecture

PURPLE functions as a reranking module stacked on top of an initial retrieval stage (e.g., Contriever retrieving 20 candidates). A user record encoder takes the query and candidate records as input and outputs a propensity score for each record. During training, the Plackett-Luce model generates a probability distribution over scores, from which $M=32$ profiles are sampled for policy gradient estimation. At inference, the $K$ records with the highest propensity scores are selected directly.

### Key Designs

1. **Plackett-Luce Ranking Policy ($\pi_\theta$)**:
    - **Function**: Converts propensity scores into a probability distribution over ordered profiles, enabling order-sensitive sampling.
    - **Mechanism**: Each record obtains a propensity score $f_\theta(h_i; C) \in [0,1]$ from the encoder. The PL model converts these into probabilities over $K$-permutations: $\pi_\theta(P|C) = \prod_{k=1}^{K} f_\theta(p_k)/[S - \sum_{j<k} f_\theta(p_j)]$. Training uses sampling without replacement of $K$ records; inference takes the top-$K$.
    - **Design Motivation**: The PL model naturally captures order sensitivity—different permutations carry different probabilities—and supports efficient sampling, making it well-suited for policy gradient optimization.

2. **User Record Encoder ($f_\theta$)**:
    - **Function**: Captures query-record and record-record interactions.
    - **Mechanism**: A late interaction strategy is adopted. Pre-trained Contriever first produces token embeddings; each record then performs token-level cross-attention with the query to obtain a query-fused representation, which is pooled into a fixed-size record embedding. A Transformer encoder (without positional encoding) subsequently models inter-record dependencies.
    - **Design Motivation**: Jointly processing all records at the token level would exceed the encoder's context window. Late interaction preserves fine-grained interactions while controlling computational complexity.

3. **Log-Likelihood Reward Function**:
    - **Function**: Provides a semantically rich training signal that directly reflects generation quality.
    - **Mechanism**: $R(\text{LLM}(P \| x), y) = \log p_\phi(y|P,x) = \sum \log p_\phi(y_j|P,x,y_{<j})$, i.e., the token-level log-likelihood of the reference response under the LLM. Compared to coarse-grained metrics such as Accuracy or ROUGE, log-likelihood distinguishes between "viable" and "optimal" profiles.
    - **Design Motivation**: The authors further demonstrate that using log-likelihood reward is equivalent to maximizing the ELBO of the RAG marginalization objective, providing a theoretical guarantee.

### Loss & Training

- REINFORCE policy gradient is used: $\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(P|C) \cdot R(\text{LLM}(P \| x), y)]$
- $M=32$ profiles are sampled per training instance; rewards are $z$-score normalized to stabilize training.
- LLM parameters are frozen; only the record encoder parameters $\theta$ are trained.
- $K=5$ records are selected from $N=20$ candidates to construct each profile.

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
| w/o Inter-Record Dependency Modeling | 61.3 | 35.0 | 0.449 | 14.5 |
| w/ Metric Reward Substitute | 64.8 | 38.0 | 0.433 | 15.0 |

Removing inter-record dependency modeling (the Transformer encoder) causes the largest performance drop, validating the necessity of profile-level holistic optimization.

### Key Findings

- **Relevance $\neq$ Utility**: PURPLE's propensity scores provide a more effective ranking signal than raw retrieval scores, even with a substantially smaller model than RankGPT.
- **Order Sensitivity Is Meaningful**: The record orderings selected by PURPLE are most frequently ranked as optimal among 120 permutations, confirming that the scores genuinely capture relative inter-record dependencies.
- **Log-Likelihood Reward Generalizes Across Tasks**: Even on regression tasks (Rating), log-likelihood reward outperforms task-specific metric rewards.
- **Human Evaluation Advantage of 14.4%**: In a blind evaluation on the Tweet task, annotators preferred PURPLE-generated outputs at 57.2% vs. 42.8%.
- **Optimal Profile Size at $K=5$**: Increasing $K$ to 10 or 15 yields marginal or negative gains, corroborating the hypothesis that utility is non-monotonic in profile size.

## Highlights & Insights

- **Elegant Problem Formulation**: Casting user profile construction as a combinatorial selection problem under the contextual bandit framework allows the Plackett-Luce model to naturally handle order sensitivity and combinatorial dependencies.
- **Theoretically Grounded**: The log-likelihood reward is shown to correspond to maximizing the ELBO of the RAG marginalization objective, providing theoretical justification beyond empirical effectiveness.
- **High Practical Value**: No LLM fine-tuning is required; the encoder is lightweight; inference requires only a single forward pass to retrieve the top-$K$ records, balancing effectiveness and efficiency.
- **Core Insight on Relevance vs. Utility**: This distinction extends beyond personalization and carries broad implications for all RAG scenarios.

## Limitations & Future Work

- The framework relies on high-quality reference responses to compute log-likelihood rewards; in deployment, explicit supervision may be sparse or unavailable (e.g., only implicit feedback exists).
- The policy is currently trained independently per task; cross-task and cross-domain generalization remain unvalidated.
- The candidate pool size is fixed at 20; performance and efficiency under larger candidate pools remain to be examined.
- Future directions include: training under weak supervision or implicit feedback, unified multi-task policies, and deeper integration with RAG pipelines.

## Related Work & Insights

- **REPLUG (Shi et al., 2024)**: Marginalizes over multiple retrieved records but processes each record independently, precluding inter-record dependency modeling.
- **IC-RALM (Ram et al., 2023)**: Periodically triggers retrieval and replaces context during decoding, also processing records independently.
- **RankGPT (Sun et al., 2023)**: A zero-shot LLM reranker with high inference cost; its optimization target is relevance rather than utility.
- **ICR (Chen et al., 2025)**: An attention-based zero-shot reranker offering reasonable efficiency, yet still oriented toward relevance.
- **LaMP / LongLaMP (Salemi et al., 2024; Kumar et al., 2024)**: Personalization benchmarks covering classification, regression, and generation tasks.
- **Insight**: Shifting retrieval optimization in RAG from relevance-oriented to utility-oriented is a direction that warrants broader exploration.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The combination of contextual bandits, Plackett-Luce ranking, and log-likelihood reward is highly elegant; the relevance vs. utility insight is profound.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 9 tasks, 3 LLM scales, multiple baselines, ablations, human evaluation, and sensitivity analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The movie recommendation example vividly motivates the problem; the method derivation is clear; the theoretical connections are rigorous.
- **Value**: ⭐⭐⭐⭐⭐ Introduces a new paradigm for retrieval-augmented personalization with broad impact on the RAG community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)
- [\[ACL 2026\] ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning](rerec_reasoning-augmented_llm-based_recommendation_assistant_via_reinforcement_f.md)
- [\[ACL 2026\] The Stackelberg Speaker: Optimizing Persuasive Communication in Social Deduction Games](the_stackelberg_speaker_optimizing_persuasive_communication_in_social_deduction_.md)
- [\[ICLR 2026\] Single Index Bandits: Generalized Linear Contextual Bandits with Unknown Reward Functions](../../ICLR2026/reinforcement_learning/single_index_bandits_generalized_linear_contextual_bandits_with_unknown_reward_f.md)
- [\[AAAI 2026\] Bi-Level Contextual Bandits for Individualized Resource Allocation under Delayed Feedback](../../AAAI2026/reinforcement_learning/bi-level_contextual_bandits_for_individualized_resource_allocation_under_delayed.md)

</div>

<!-- RELATED:END -->
