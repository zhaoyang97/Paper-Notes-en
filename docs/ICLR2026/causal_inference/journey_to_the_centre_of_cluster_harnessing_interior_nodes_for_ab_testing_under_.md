---
title: >-
  [Paper Note] Journey to the Centre of Cluster: Harnessing Interior Nodes for A/B Testing under Network Interference
description: >-
  [ICLR2026][Causal Inference][A/B testing] To address the high-variance problem in GATE estimation for A/B testing under network interference…
tags:
  - "ICLR2026"
  - "Causal Inference"
  - "A/B testing"
  - "network interference"
  - "cluster randomization"
  - "GATE estimation"
date: 2026-05-08
content_hash: 454065308f66c081
---

# Journey to the Centre of Cluster: Harnessing Interior Nodes for A/B Testing under Network Interference

**Conference**: ICLR2026
**arXiv**: [2602.04457](https://arxiv.org/abs/2602.04457)
**Code**: [GitHub](https://github.com/Cqyiiii/AMII-Harnessing-Interior-Nodes-for-Network-Experiments)
**Area**: Causal Inference
**Keywords**: A/B testing, network interference, causal inference, cluster randomization, GATE estimation

## TL;DR

To address the high-variance problem in GATE estimation for A/B testing under network interference, this paper proposes the Mean-in-Interior (MII) estimator—which averages only over interior nodes within each cluster to substantially reduce variance—and further introduces a counterfactual predictor to correct for covariate shift, yielding the augmented AMII estimator that achieves low bias and low variance simultaneously.

## Background & Motivation

In online platform A/B testing, the classical SUTVA assumption (each user's outcome depends only on their own treatment) is frequently violated: in social networks, a user's behavior is influenced by the treatment status of their neighbors, a phenomenon known as **network interference**. The **Global Average Treatment Effect (GATE)**—the difference in average outcomes under full treatment versus full control—serves as the standard estimand.

To handle network interference, **cluster-level randomization** is the industry standard: the graph is partitioned into communities via community detection, and treatment is assigned at the cluster level. Within this framework, **interior nodes** (nodes whose entire 1-hop neighborhood lies within the same cluster) naturally experience local environments that approximate global treatment or control, making them ideal candidates for GATE estimation.

Nevertheless, existing methods suffer from a fundamental variance problem:

- **HT estimator**: applies inverse probability weighting to each node satisfying the exposure condition, with weights of the form $(1/p)^c$ where $c$ is the number of distinct clusters a node connects to. In real social networks, $c$ can reach tens, causing weight explosion and extremely high variance.
- **CAE estimator**: employs a two-level averaging scheme (within cluster → across clusters), replacing inverse probability weighting with means, but its two-tier structure introduces unnecessary complexity when clustering quality is poor.
- **DIM estimator**: ignores graph structure entirely, resulting in severe bias when interference is present.

Analysis on the Facebook social network (11,586 nodes, 568,309 edges) shows that interior nodes constitute only about 8% of all nodes, yet after exposure-condition filtering they account for the vast majority of retained samples—an observation that directly motivates the proposed approach.

## Core Problem

1. **Variance explosion**: Existing network-aware estimators (e.g., HT) require exponentially large inverse probability weights for boundary nodes, leading to extremely high variance in early-stage experiments with low treatment probability ($p \le 10\%$).
2. **Selection bias**: Interior nodes differ systematically from the full population in covariate distributions (e.g., degree, number of connected clusters), so naively averaging over interior nodes introduces bias.
3. **Difficulty of interference function extrapolation**: Under low treatment probability, estimators must extrapolate from local observations to the global treatment scenario; nonlinear interference functions (e.g., square-root, quadratic) make pure regression approaches perform poorly.

## Method

### Mean-in-Interior (MII) Estimator

The core idea is remarkably simple: discard all boundary nodes and compute a difference-in-means solely over interior nodes:

$$\hat{\tau}_{MII} = \frac{\sum_{i \in \text{Int}} z_i Y_i}{\sum_{j \in \text{Int}} z_j} - \frac{\sum_{i \in \text{Int}} (1-z_i) Y_i}{\sum_{j \in \text{Int}} (1-z_j)}$$

In contrast to the exponential weights of the HT estimator, MII employs uniform weights. Under the NIA assumption and mild technical conditions (asymptotically uniform interior-node proportions across clusters and representativeness of interior nodes within clusters), MII is consistent: $\hat{\tau}_{MII} - \tau = o_p(1)$.

A key advantage: for a canonical potential outcome model $Y_i(\mathbf{z}) = \beta z_i + h(\sum_j z_j / \deg_i, v_i)$, each interior node provides an unbiased estimate of the global treatment/control mean, and uniform averaging is the minimum-variance unbiased estimator by the Cauchy-Schwarz inequality.

### Augmented MII (AMII) Estimator

To correct for covariate shift between interior nodes and the full population, a counterfactual predictor $f(\mathbf{z}, X, A)$ (e.g., a GNN) is trained on the full graph and extrapolated to global treatment/control scenarios, constructing a correction term:

$$\hat{\tau}_{AMII} = \hat{\tau}_{MII} + \underbrace{\left(\frac{1}{n}\sum_j f(\mathbf{1},X,A)_j - \frac{1}{s_1}\sum_{i \in \text{Int}} z_i f(\mathbf{1},X,A)_i\right)}_{\text{treatment group correction}} - \underbrace{\left(\frac{1}{n}\sum_j f(\mathbf{0},X,A)_j - \frac{1}{s_0}\sum_{i \in \text{Int}} (1-z_i) f(\mathbf{0},X,A)_i\right)}_{\text{control group correction}}$$

The correction term captures the discrepancy between predicted values for the full population and the interior subgroup, compensating for covariate shift.

**Bias analysis** (Theorem 4.1): Under a partially linear potential outcome model $Y_i(\mathbf{z}) = (\beta + \alpha u_i)z_i + h(\cdot)$:

- MII bias $= \alpha(\mu_{\text{Int}} - \mu)$, i.e., the covariate mean difference scaled by the interaction coefficient.
- AMII bias $= (\mathbb{E}[\hat{\alpha}_n] - \alpha)(\mu - \mu_{\text{Int}})$, which is substantially reduced when the linear component of the regression model is correctly estimated.

### Semi-Supervised Learning Perspective

Rearranging the AMII expression yields:

$$\hat{\tau}_{AMII,1} = \frac{1}{n}\sum_j f(\mathbf{1},X,A)_j + \frac{1}{s_1}\sum_{i \in \text{Int}} z_i (Y_i - f(\mathbf{1},X,A)_i)$$

This is precisely the point estimate of the **Prediction-Powered Inference (PPI)** framework. Interior nodes serve as labeled data (albeit with selection bias), while boundary nodes provide representative covariates with only partial label information. AMII can be interpreted as "debiasing using predictions"—correcting the selection bias of interior nodes induced by clustering—which is complementary to the classical doubly-robust estimator's strategy of "debiasing predictions using labels."

## Key Experimental Results

Monte Carlo simulations (1,000 repetitions) on the Facebook network (11,586 nodes, 95 clusters via Louvain, ~8% interior nodes):

| Setting | Method | Key Performance |
|---------|--------|----------------|
| NIA holds ($r_2=0$), $p=0.1$ | AMII | Significantly lowest MSE; bias far below MII/CAE/Hájek |
| NIA holds, $p=0.3$ | MII | Consistently lowest variance across all methods |
| 2-hop interference ($r_2=1$), $p=0.1$ | AMII | Bias-correction advantage more pronounced |
| No interaction, no 2-hop interference | MII | Nearly unbiased with lowest variance; optimal choice |
| Low treatment probability $p=0.1$ | GNN | Poor performance due to insufficient information for extrapolation |

Key findings:

- **AMII achieves the lowest MSE across all settings**, even when $p=0.1$ and the GNN itself performs poorly.
- **MII consistently achieves the lowest variance**, though its bias is affected by network covariate distributional differences.
- As the interaction coefficient increases from 0.5 to 1, the advantage of AMII becomes more pronounced.
- Increasing clustering resolution $\gamma$ from 2 to 5 reduces variance, but further increasing to 10 yields marginal improvement (irreducible variance floor).
- Training $f$ on the full graph substantially outperforms training on boundary nodes only at low $p$; the gap narrows as $p$ increases.

## Highlights & Insights

- **Conceptual simplicity**: The core of MII is simply "average over interior nodes only," yet it is backed by rigorous theoretical guarantees.
- **PPI perspective of AMII**: The paper elegantly unifies counterfactual prediction adjustment with the semi-supervised learning framework, endowing the estimator with richer theoretical foundations.
- **High relevance to industrial practice**: The paper demonstrates on a billion-scale social platform that covariate distributional differences between interior and boundary nodes are empirically substantial.
- **Harmlessness property**: When interior nodes and the full population share the same distribution, the AMII correction term introduces no additional bias.
- **Weaker theoretical assumptions than CAE**: MII's consistency requires only asymptotically uniform interior-node proportions and representativeness, without requiring constant within-cluster means.

## Limitations & Future Work

- **Only consistency is established, not a CLT**: Hypothesis testing and confidence interval construction are not directly supported; bootstrap or other alternatives are needed for practical deployment.
- **Low interior-node proportion (~8%)**: In extremely sparse graphs or with poor clustering quality, this proportion may be even lower, limiting data utilization efficiency.
- **Choice of GNN predictor**: The paper uses a simple 3-layer Chebyshev convolution; more powerful architectures could further improve AMII performance.
- **Only NIA and linear 2-hop interference are considered**: More complex interference patterns (e.g., higher-order interference, dynamic networks) are not addressed.
- **Experiments conducted solely on simulated data**: Despite providing distributional difference evidence from an industrial platform, no real-world A/B test results are reported.

## Related Work & Insights

| Method | Bias | Variance | Applicable Scenario |
|--------|------|----------|---------------------|
| DIM | Severe (ignores network structure) | Low | No interference |
| HT/Hájek | Unbiased/low (under NIA) | Extremely high (weight explosion) | Theoretical analysis |
| CAE | Moderate | Moderate | Good clustering quality |
| GNN regression | Model- and $p$-dependent | Unstable | High treatment probability |
| **MII (Ours)** | Moderate (covariate shift) | **Lowest** | NIA + good clustering |
| **AMII (Ours)** | **Lowest** | Comparable to MII | **Generally recommended** |

Distinction from the PPI framework (Angelopoulos et al., 2023): PPI assumes labeled samples are MCAR (missing completely at random), whereas interior nodes in this work exhibit selection bias. AMII requires additional handling of this distributional shift.

**Broader connections**:

- **Intersection of semi-supervised learning and causal inference**: Mapping the interior/boundary partition in network experiments to the labeled/unlabeled split in semi-supervised learning opens up the PPI framework as a new toolbox for network causal inference.
- **Practical value for large-scale online experimentation**: Platforms such as Tencent run thousands of A/B tests weekly; the computational simplicity of MII/AMII (no complex weighting required) is deployment-friendly.
- **Transferability to other selection bias settings**: The "debiasing using predictions" idea is transferable to selection bias correction in survey sampling, missing data, and related domains.

## Rating

- Novelty: ⭐⭐⭐⭐ — The MII idea, though simple, had not been formally proposed; the PPI perspective of AMII is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Extensive ablation studies across diverse settings, but real-platform A/B test results are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clearly articulated; theory and intuition are well balanced; the progression from MII to AMII is logically coherent.
- Value: ⭐⭐⭐⭐ — Directly applicable to online experimentation; theoretical contributions align well with practical needs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Resisting Contextual Interference in RAG via Parametric-Knowledge Reinforcement](resisting_contextual_interference_in_rag_via_parametric-knowledge_reinforcement.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Perturbations](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_perturbations.md)
- [\[NeurIPS 2025\] Cyclic Counterfactuals under Shift–Scale Interventions](../../NeurIPS2025/causal_inference/cyclic_counterfactuals_under_shift-scale_interventions.md)
- [\[ICLR 2026\] On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)
- [\[ICLR 2026\] Direct Doubly Robust Estimation of Conditional Quantile Contrasts](direct_doubly_robust_estimation_of_conditional_quantile_contrasts.md)

</div>

<!-- RELATED:END -->
