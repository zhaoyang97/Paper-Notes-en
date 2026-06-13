---
title: >-
  [Paper Note] Root Cause Analysis of Outliers with Missing Structural Knowledge
description: >-
  [NeurIPS 2025][Causal Inference][Root Cause Analysis] Two simple and efficient algorithms are proposed for root cause analysis using only **marginal anomaly scores**: SMOOTH TRAVERSAL (known causal graph — finds the node…
tags:
  - "NeurIPS 2025"
  - "Causal Inference"
  - "Root Cause Analysis"
  - "Anomaly Detection"
  - "Information-Theoretic Anomaly Score"
  - "Polytree"
date: 2026-05-08
content_hash: 871aabb144b021a5
---

# Root Cause Analysis of Outliers with Missing Structural Knowledge

**Conference**: NeurIPS 2025
**arXiv**: [2406.05014](https://arxiv.org/abs/2406.05014)  
**Code**: [amazon-science/RCAWithMissingStructuralKnowledgeCode](https://github.com/amazon-science/RCAWithMissingStructuralKnowledgeCode)  
**Area**: Causal Inference
**Keywords**: Root Cause Analysis, Causal Inference, Anomaly Detection, Information-Theoretic Anomaly Score, Polytree

## TL;DR
Two simple and efficient algorithms are proposed for root cause analysis using only **marginal anomaly scores**: SMOOTH TRAVERSAL (known causal graph — finds the node with the largest score jump along causal paths) and SCORE ORDERING (unknown causal graph — ranks nodes by score and returns the top-$k$). Both algorithms provide nonparametric probabilistic guarantees under polytree structure and operate on a single anomalous sample.

## Background & Motivation
**Background**: Root cause analysis (RCA) is widely applied in cloud service monitoring, industrial fault diagnosis, healthcare, and related domains. From a causal perspective, RCA is framed as identifying which causal mechanism has undergone a soft intervention.

**Limitations of Prior Work**:
   - Most methods assume access to or estimation of the post-intervention distribution, yet in practice only a **single anomalous sample** is often available.
   - Even without relying on multiple samples, conventional methods require estimating **conditional probabilities $P(X_i | PA_i)$**, which is statistically ill-posed in high-dimensional or low-density regions.
   - Traversal-based methods require manually specifying an anomaly threshold and lack theoretical guarantees.

**Key Challenge**: The core tension lies in performing theoretically guaranteed root cause analysis under minimal information — a single anomalous sample and a potentially unknown causal graph.

**Goal**: Avoid conditional probability estimation and identify a single root cause using only marginal anomaly scores $S(x_i)$.

**Key Insight**: Exploit the mathematical properties of information-theoretic (IT) anomaly scores to prove that "anomalies rarely cause larger anomalies" along causal paths.

**Core Idea**: In a polytree causal graph, differences between marginal anomaly scores can substitute for conditional anomaly scores to test whether a causal mechanism has been disrupted.

## Method

### Overall Architecture
The paper builds on the Causal Bayesian Network (CBN) framework. The normal distribution $P(X_1, \ldots, X_n)$ factorizes over a causal DAG as $\prod P(X_i | PA_i)$. An anomaly is modeled as the causal mechanism of some node $X_j$ being replaced by $\tilde{P}(X_j | PA_j)$, and the goal is to identify $j$ from a single observation $(x_1, \ldots, x_n)$.

The central tool is the **information-theoretic (IT) anomaly score**:
- Marginal anomaly score: $S(x_n) = -\log P(\tau(X_n) \geq \tau(x_n))$, the "surprise" of the observed value.
- Conditional anomaly score: $S(x_n | pa_n) = -\log P(\tau(X_i) \geq \tau(x_n) \mid PA_n = pa_n)$, the residual surprise after conditioning on parent nodes.
- Estimator: $\hat{S}(x_n) = -\log\!\left(\tfrac{1}{k} \cdot |\{i : \tau(x^i) \geq \tau(x^n)\}|\right)$, computed from $k$ samples drawn from the normal period.

### Key Designs
1. **Score Typicality Condition**: Defined as $S(y|x) \geq |S(y) - S(x)|_+$, i.e., the conditional anomaly score is no less than the positive part of the difference between marginal scores. This serves as the bridge between marginal and conditional scores. Lemma 3.4 shows this condition holds approximately for a randomly drawn $x$; Lemma 3.5 shows it holds exactly under monotone invertible mappings.

2. **"Anomalies Rarely Cause Larger Anomalies" (Lemma 3.3)**: For a causal pair $X \to Y$, under score typicality, the probability that an anomaly of strength $S(x)$ at $X$ induces an anomaly of strength $S(y)$ at $Y$ is at most $e^{-|S(y)-S(x)|_+}$. This implies that marginal scores should not exhibit abrupt increases along causal paths; if they do, it signals a disrupted mechanism at that node.

3. **Generalization to Polytrees**: In a polytree, the parents of any node are marginally independent, allowing the multi-parent problem to be reduced to a binary problem. A joint score $S_{\text{joint}}(pa_i) = \sum S(pa_{ij}) - \log\!\left(\sum (\sum S)^l / l!\right)$ is used to aggregate parent information (analogous to Fisher's method for combining $p$-values).

4. **SMOOTH TRAVERSAL (requires causal graph)**: For each ancestor $X_i$ of the target node, compute $\delta_i = |S(x_i) - \max_{\text{parent}} S(\text{parent})|_+$ and select the node with the largest $\delta$ as the root cause. This eliminates the need for a manually specified anomaly threshold. Theorem 3.12 provides the $p$-value upper bound $p \leq 1 - (1 - e^{-\delta_{\max}})^{n-1}$.

5. **SCORE ORDERING (requires no causal graph)**: All nodes are ranked in descending order of marginal anomaly score and the top-$k$ nodes are returned. Theorem 3.13 proves that the probability of the root cause falling outside the top-$k$ is $\leq n \cdot d_{\max} \cdot e^{-\Delta_k}$, where $\Delta_k = S(x_{\pi(1)}) - S(x_{\pi(k+1)})$. Given confidence level $1-\alpha$ and maximum in-degree $d_{\max}$, the required $k$ can be determined.

### Loss & Training
No model training is required. The two algorithms only need:
- Normal-period data to estimate marginal IT anomaly scores (using the simple rank-based estimator in Eq. 5).
- SMOOTH TRAVERSAL additionally requires the causal graph.
- SCORE ORDERING additionally requires an upper bound estimate of the maximum in-degree $d_{\max}$.

## Key Experimental Results

### Main Results: Effect of Anomaly Strength on Top-1 Recall
(Synthetic data, random DAG with $n$ nodes, nonlinear SCM, 100 repetitions)

| Method | Requires Causal Graph | Requires SCM | TPR @ $x{=}2.0$ | TPR @ $x{=}3.0$ | Notes |
|---|---|---|---|---|---|
| **SMOOTH TRAVERSAL** | ✓ | ✗ | ~0.65 | ~0.72 | On par with Traversal and Counterfactual |
| Traversal | ✓ | ✗ | ~0.65 | ~0.72 | Requires manual anomaly threshold |
| Counterfactual | ✓ | ✓ | ~0.65 | ~0.72 | Requires full SCM |
| Circa | ✓ | ✗ | ~0.35 | ~0.40 | Assumes linearity; poor in nonlinear settings |
| Cholesky | ✗ | ✗ | ~0.30 | ~0.35 | Assumes linearity; poor in nonlinear settings |
| **SCORE ORDERING** | ✗ | ✗ | ~0.45 | ~0.50 | Requires no graph structure |

### Ablation Study

| Dimension | Result |
|---|---|
| Graph size $n{=}20{\to}100$ | SMOOTH TRAVERSAL / Traversal / Counterfactual remain stable; SCORE ORDERING / Cholesky / Circa degrade as $n$ grows |
| Causal graph misspecification (edge additions/deletions/reversals) | SMOOTH TRAVERSAL and Traversal show comparable robustness; Circa degrades faster |
| Linear SCM only | Circa and Cholesky improve substantially, approaching SMOOTH TRAVERSAL |
| DAG restricted to polytree | SCORE ORDERING performance improves; theoretical guarantees hold exactly |
| Runtime as $n$ increases | SMOOTH TRAVERSAL and Traversal are fastest; other methods have comparable runtimes |

### Real-World Data

| Dataset | Best-Performing Method | Notes |
|---|---|---|
| PetShop (Hardt et al.) | Varies by method | Microservice monitoring |
| Sock-shop 2 (Pham et al.) | SCORE ORDERING performs well | Containerized application fault localization |
| ProRCA semi-synthetic | Methods alternate in rank | Multiple fault injection scenarios |

### Key Findings
- Among methods requiring a causal graph, SMOOTH TRAVERSAL is the **only one that requires no anomaly threshold**, while matching the best-performing methods.
- SCORE ORDERING has the **weakest input requirements** (no causal graph, no SCM) yet provides nonparametric probabilistic guarantees.
- In nonlinear settings, methods assuming linearity (Circa, Cholesky) suffer sharp performance degradation.
- On real-world data, SCORE ORDERING is robust as a first-pass heuristic.

## Highlights & Insights
- **Theoretical elegance**: The intuition that "anomalies rarely cause larger anomalies" is rigorously formalized through a concise derivation grounded in the information-theoretic interpretation of $p$-values.
- **Practical simplicity**: Both algorithms are straightforward to implement — SCORE ORDERING requires only sorting anomaly scores, and SMOOTH TRAVERSAL requires a single graph traversal.
- **Guarantees under the weakest assumptions**: SCORE ORDERING is arguably the theoretically grounded RCA method with the fewest assumptions to date.
- **Score Typicality as a bridge**: The key technique of substituting marginal scores for conditional scores avoids the statistical difficulties of conditional density estimation.

## Limitations & Future Work
- **Polytree assumption**: The core theory relies on polytree structure (i.e., the skeleton is a tree); in general DAGs, downstream anomaly scores may exceed those of the root cause (Li et al. 2024 characterize conditions under which this occurs).
- **Single root cause assumption**: Only one causal mechanism is assumed to be disrupted; extensions to multi-root-cause scenarios are needed.
- **Inconsistent real-world performance**: The proposed methods are outperformed by Cholesky or Circa on certain real-world datasets, indicating a cost to simplicity.
- **Estimation accuracy of IT anomaly scores**: Sufficient normal-period samples are required — at least $e^s$ samples to reliably estimate a score of value $s$.

## Related Work & Insights
- **vs. Traversal (Chen et al. 2014)**: Both are graph-traversal-based, but Traversal requires a manually specified anomaly threshold, which SMOOTH TRAVERSAL entirely avoids.
- **vs. Counterfactual (Budhathoki et al. 2022)**: Counterfactual methods require a complete SCM; this paper requires only marginal scores.
- **vs. Cholesky (Li et al. 2024)**: Assumes a linear SCM and leverages the Cholesky decomposition of the covariance matrix; this paper makes no parametric assumptions.
- **vs. Circa (Li et al. 2022)**: Fits a linear SCM and takes the largest residual; this paper avoids regression models entirely.
- **Insight**: The property that IT anomaly scores do not amplify along causal paths may generalize to other causal-statistical testing problems.

## Rating
- Novelty: ⭐⭐⭐⭐ — The theoretical contribution of replacing conditional scores with marginal scores for RCA is solid; Score Typicality is a novel technical tool.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Synthetic experiments cover multiple dimensions; real-world evaluation is comprehensive, though performance is not always state-of-the-art.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear; the progressive exposition from binary to polytree settings is particularly smooth.
- Value: ⭐⭐⭐⭐ — Directly applicable to industrial RCA practice; SCORE ORDERING can serve as a fast pre-screening step in any RCA pipeline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] An Analysis of Causal Effect Estimation Using Outcome Invariant Data Augmentation](an_analysis_of_causal_effect_estimation_using_outcome_invariant_data_augmentatio.md)
- [\[ICLR 2026\] Self-Supervised Learning from Structural Invariance](../../ICLR2026/causal_inference/self-supervised_learning_from_structural_invariance.md)
- [\[ICLR 2026\] Resisting Contextual Interference in RAG via Parametric-Knowledge Reinforcement](../../ICLR2026/causal_inference/resisting_contextual_interference_in_rag_via_parametric-knowledge_reinforcement.md)
- [\[AAAI 2026\] KTCF: Actionable Recourse in Knowledge Tracing via Counterfactual Explanations for Education](../../AAAI2026/causal_inference/ktcf_actionable_recourse_in_knowledge_tracing_via_counterfactual_explanations_fo.md)
- [\[AAAI 2026\] CaDyT: Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis](../../AAAI2026/causal_inference/causal_structure_learning_for_dynamical_systems_with_theoretical_score_analysis.md)

</div>

<!-- RELATED:END -->
