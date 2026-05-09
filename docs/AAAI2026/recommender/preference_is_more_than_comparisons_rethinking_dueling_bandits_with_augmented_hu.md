---
title: >-
  [Paper Note] Preference is More Than Comparisons: Rethinking Dueling Bandits with Augmented Human Feedback
description: >-
  [AAAI 2026][Recommender Systems][Dueling Bandits] This paper proposes IPEA-HF, a model-free Dueling Bandit framework based on augmented human feedback. It integrates contextual similarity and dependency relations through Augmented Confidence Bounds to calibrate uncertainty, achieving superior performance across multiple benchmarks including recommendation, multi-objective optimization, and LLM response optimization.
tags:
  - AAAI 2026
  - Recommender Systems
  - Dueling Bandits
  - Preference Elicitation
  - Augmented Human Feedback
  - Confidence Bounds
  - Multi-Objective Optimization
date: 2026-05-08
content_hash: 936a2745abc52332
---

# Preference is More Than Comparisons: Rethinking Dueling Bandits with Augmented Human Feedback

**Conference**: AAAI 2026
**arXiv**: [2511.09047](https://arxiv.org/abs/2511.09047)
**Code**: [COLA-Laboratory/IPEA-HF](https://github.com/COLA-Laboratory/IPEA-HF)
**Area**: Recommender Systems / Preference Learning
**Keywords**: Dueling Bandits, Preference Elicitation, Augmented Human Feedback, Confidence Bounds, Multi-Objective Optimization

## TL;DR

This paper proposes IPEA-HF, a model-free Dueling Bandit framework based on augmented human feedback. It integrates contextual similarity and dependency relations through Augmented Confidence Bounds to calibrate uncertainty, achieving superior performance across multiple benchmarks including recommendation, multi-objective optimization, and LLM response optimization.

## Background & Motivation

### Efficiency Bottleneck in Interactive Preference Elicitation (IPE)

In personalized systems such as recommender systems, multi-objective optimization, and LLM response optimization, acquiring user preferences requires substantial human effort. **Interactive Preference Elicitation (IPE)** reduces this burden by selectively querying users. Dueling Bandits (DB), an online decision-making framework based on pairwise comparisons, serves as the theoretical foundation of IPE.

**Core Problem**: DB frameworks are inefficient when human feedback is sparse. Existing methods address this through two main approaches:

**Parametric reward models** (e.g., the Bradley-Terry model): Impose overly rigid assumptions, are prone to model misspecification, and cannot handle non-transitive preferences.

**Candidate partitioning** (e.g., clustering methods): Assume candidates can be clearly divided into distinguishable subsets, which is often unverifiable in practice.

### An Overlooked Perspective: Feedback Augmentation

The authors propose a key insight — **human preferences are not merely isolated pairwise comparisons**, but are also shaped by contextual information and latent dependency relations. The root cause of inefficiency in DB frameworks lies in treating each feedback instance independently. By augmenting feedback through contextual similarity and dependencies, efficiency can be improved without relying on parametric assumptions.

### Three Core Research Questions

- **RQ1**: How can augmented human feedback be integrated in a model-free setting?
- **RQ2**: Does augmented feedback always improve efficiency, or can it lead to degradation?
- **RQ3**: Can the DB framework go beyond pairwise comparisons to incorporate richer forms of feedback?

## Method

### Overall Architecture

The IPEA-HF algorithm consists of four core components:

1. **AugConfidenceBound**: Computes confidence bounds based on augmented observations.
2. **DuelingBanditAlgo**: Selects candidate pairs based on confidence bounds (supports strategies such as RUCB and DTS).
3. **DependencyExtract**: Extracts dependency relations from the contextual space.
4. **FeedbackAug**: Augments feedback based on observations and dependency relations.

### Key Designs

#### 1. Augmented Confidence Bounds

**Function**: Incorporates related observations (comparison results from similar candidate pairs) into confidence bound estimation.

**Mechanism**: For a candidate pair $(a_i, a_j)$, let $n^d_{i,j}(t)$ denote the number of direct comparisons and $n^r_{i,j}(t)$ the number of related observations, with total observations $n_{i,j}(t) = n^d_{i,j}(t) + n^r_{i,j}(t)$. The augmented UCB/LCB are defined as:

$$\hat{u}_{i,j}=\hat{p}_{i,j}+\frac{1}{\eta}\sqrt{\frac{\alpha\ln t}{n_{i,j}(t)}}, \quad \hat{l}_{i,j}=\hat{p}_{i,j}-\frac{1}{\eta}\sqrt{\frac{\alpha\ln t}{n_{i,j}(t)}}$$

where $\eta = (n^d_{i,j}+\sum_{k}w^k_{i,j})/n_{i,j}$, and $w^k_{i,j}\in[0,1]$ denotes the dependency weight.

**Design Motivation**: When $n^r_{i,j}=0$ (no augmentation), the formula reduces to the standard DB confidence bound. Related observations follow $X^k_{i,j}\sim\text{Bernoulli}(w^k_{i,j}p_{i,j})$, where the dependency weight $w^k_{i,j}$ controls the reliability of related observations.

**Concentration Property (Theorem 3.1)**: Under the condition $\alpha>0.5$, with probability $1-\delta$, for all sufficiently large $t$ and all candidate pairs, the true preference probability $p_{i,j}$ is contained within the augmented confidence interval.

#### 2. Calibration Threshold and Multi-Factor Trade-offs

**Function**: Quantitatively analyzes when augmented feedback is beneficial and when it is harmful.

**Core Findings**: The condition under which augmented feedback is effective is:

$$w^r_{i,j} > \eta n_{i,j}(t)\left(\sqrt{1+\frac{1}{n_{i,j}(t)}}-1\right)$$

- When $w^r_{i,j}=1$, related observations are equivalent to direct observations and the confidence interval shrinks.
- When $w^r_{i,j}=0$, the confidence interval **widens** instead.
- As the number of direct observations increases, the marginal contribution of related observations diminishes.

**Design Motivation**: This addresses RQ2 — the effectiveness of augmented feedback depends on dependency strength, and weak dependencies may lead to degradation. This provides practical guidance on when to leverage augmented feedback and how to calibrate it.

#### 3. A Unified Perspective on Existing Methods

**Function**: Demonstrates that augmented confidence bounds constitute a unified framework encompassing multiple DB methods.

**Relation to Partitioning Methods**: Partitioning methods are a special case — within-group $w^r_{i,j}=1$ (full dependency), with no sharing across groups.

**Relation to Parametric Reward Estimation**: In structured reward methods, the Mahalanobis-norm confidence bound $\|x_i-x_j\|_{V^{-1}}$ narrows as directional weights decrease (i.e., with more related observations), which is consistent with the proposed mechanism. However, structured methods lack formal concentration guarantees.

### Loss & Training

#### Sample Complexity (Theorem 3.2)

$$P(\exists t, i,j\in\mathcal{A}, n_{i,j}(t) > C(\delta) \lor D^w_{i,j}\ln t) < \delta$$

where $D^w_{i,j} = \frac{4\alpha}{\min_r w^r_{i,j}{}^2 \min\{\Delta_i^2, \Delta_j^2\}}$.

**Trade-off**: Strong dependencies reduce the need for direct observations, whereas weak dependencies increase the coefficient $D^w_{i,j}$.

#### Regret Analysis (Theorem 3.3)

Assuming bidirectional dependencies and $C$ soft clusters, the cumulative regret bound is:

$$\mathcal{O}\left(\frac{1}{\min_{i,j,r}w^r_{i,j}}\hat{K}^2\log T\right), \quad \hat{K}=\max\{C,K_1,\dots,K_C\}$$

This subsumes the partitioning method bound $\mathcal{O}(C^2\log T)$ as a special case with $w^r_{i,j}\equiv 1$.

#### Computational Design

- **Dependency Extraction**: Constructs a similarity graph (based on Gower/Euclidean distance) and applies graph partitioning to obtain candidate groups.
- **Feedback Augmentation**: After each round of comparison, dependency conditions for related pairs are annotated via LLM or by users.

## Key Experimental Results

### Main Results

#### Recommendation Tasks (Sushi/Car Preference, 2000 Interaction Rounds)

| Algorithm | Type | Sushi Final Regret | Car Final Regret | Notes |
|-----------|------|--------------------|------------------|-------|
| RUCB | No context | Medium-high | Medium-high | Basic DB |
| DTS | No context | Medium | Medium | Stochastic strategy |
| MaxInP | Parametric | High | High | BT model |
| COLSTIM | Parametric | High | High | Convergence issues |
| VACDB | Parametric | Continuously growing | High | Insufficient exploration |
| **IPEA-RUCB** | **Augmented** | **Lowest** | **Lowest** | **Ours** |
| **IPEA-DTS** | **Augmented** | **Second lowest** | **Second lowest** | **Ours** |

Parametric methods perform poorly due to model misspecification; VACDB even exhibits continuously growing regret (exploration failure).

#### Multi-Objective Optimization (DTLZ7, 200 Interaction Rounds)

IPEA-RUCB demonstrates clear advantages under large candidate sets ($100^2$ pairs) with sparse feedback. Parametric methods slightly outperform context-free DB but suffer from repeatedly querying the same small subset (over-exploitation, insufficient exploration).

#### LLM Response Optimization (Anthropic H-H Dataset)

DTS and IPEA-DTS achieve the best performance (stochastic strategies are more suitable for LLM settings). IPEA-DTS further outperforms standard DTS by leveraging augmented feedback across prompts. Parametric methods are inefficient due to the high computational cost of the 768-dimensional feature space.

### Ablation Study

| Configuration | Description | Effect |
|---------------|-------------|--------|
| No augmentation (standard DB) | Direct observations only | Baseline |
| Augmented + similarity graph | Utilizes contextual similarity | Improved sample efficiency |
| Augmented + LLM-annotated dependencies | Additional dependency weight estimation | Further improvement |
| Varying $\alpha$ | Sensitivity of confidence parameter | $\alpha=0.1$ performs best |

### Key Findings

1. **Model-free > Parametric**: In recommendation settings, context-free DB consistently outperforms parametric DB, which suffers from model misspecification.
2. **Value of augmented feedback is scenario-dependent**: Gains are larger in recommendation settings (low-dimensional feature space, meaningful similarity) and exist but are more limited in LLM settings due to cross-prompt comparability constraints.
3. **Deterministic vs. stochastic strategies**: IPEA-RUCB performs better on multi-objective tasks with sparse feedback; IPEA-DTS performs better on LLM tasks.
4. **Query diversity is a key metric**: Query frequency analysis shows that IPEA methods achieve a better exploration-exploitation balance.

## Highlights & Insights

1. **Solid theoretical contributions**: Three theorems (concentration property, sample complexity, regret bound) provide provable efficiency guarantees and reveal explicit multi-factor trade-offs.
2. **Unified perspective**: Partitioning methods and parametric methods are both subsumed under the augmented confidence bound framework, deepening the understanding of DB methodology.
3. **Practical value of calibration threshold**: Explicit conditions are provided for determining when augmented feedback is beneficial or harmful.
4. **Beyond pairwise comparisons**: The framework can integrate richer feedback forms such as feature-level comparisons, expert demonstrations, and LLM reasoning signals, offering a more flexible foundation for IPE.
5. **Cross-domain validation**: Three substantially different application scenarios — recommendation, multi-objective optimization, and LLM — are evaluated.

## Limitations & Future Work

1. **Estimation of dependency weights**: The current approach obtains $w^k_{i,j}$ via LLM annotation, which may introduce noise; although theoretically robust, practical performance may be limited.
2. **Bidirectional dependency assumption**: The regret analysis requires bidirectional (symmetric) dependencies; the analysis of asymmetric dependencies is left open.
3. **Scalability to large candidate sets**: When $K$ is very large (e.g., thousands of candidates), the overhead of similarity graph construction and dependency annotation warrants attention.
4. **Relation to RLHF**: The framework has conceptual connections to DPO/RLHF, but it is not directly demonstrated how it integrates into modern LLM training pipelines.
5. **Active learning** strategies for selecting the most informative augmentation annotations are worth exploring.

## Related Work & Insights

- **RUCB (Zoghi et al. 2014)**: Theoretical foundation of the DB framework; this paper directly extends its concentration property.
- **DTS (Wu & Liu 2016)**: Stochastic-strategy DB; this paper also implements IPEA-DTS based on it.
- **DPO (Rafailov et al. 2023)**: Simplifies preference learning as a classification problem, but assumes a static parametric reward.
- **PBEMO (Huang et al. 2024)**: Partition-based multi-objective DB; subsumed as a special case in this paper.
- Insight: **Return to model-free methods** — in preference learning, reducing reliance on model assumptions may be a more robust path.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Unified framework of augmented confidence bounds + multi-factor trade-off analysis + perspective beyond pairwise comparisons)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Benchmarks across three different domains, though each domain covers relatively few datasets)
- Writing Quality: ⭐⭐⭐⭐ (Rigorous theoretical presentation, but the paper is lengthy and has a moderate entry barrier for first-time readers)
- Value: ⭐⭐⭐⭐⭐ (Strong in both theory and practice; the unified framework carries far-reaching significance)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Moral Change or Noise? On Problems of Aligning AI With Temporally Unstable Human Feedback](moral_change_or_noise_on_problems_of_aligning_ai_with_temporally_unstable_human_.md)
- [\[AAAI 2026\] Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation](tool4poi_a_tool-augmented_llm_framework_for_next_poi_recommendation.md)
- [\[AAAI 2026\] When Top-ranked Recommendations Fail: Modeling Multi-Granular Negative Feedback for Explainable and Robust Video Recommendation](when_top-ranked_recommendations_fail_modeling_multi-granular_negative_feedback_f.md)
- [\[ACL 2026\] Decisive: Guiding User Decisions with Optimal Preference Elicitation from Unstructured Documents](../../ACL2026/recommender/decisive_guiding_user_decisions_with_optimal_preference_elicitation_from_unstruc.md)
- [\[ICLR 2026\] Search Arena: Analyzing Search-Augmented LLMs](../../ICLR2026/recommender/search_arena_analyzing_search-augmented_llms.md)

<!-- RELATED:END -->
