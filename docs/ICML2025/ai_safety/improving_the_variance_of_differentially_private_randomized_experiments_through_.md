---
title: >-
  [Paper Note] Improving the Variance of Differentially Private Randomized Experiments through Clustering
description: >-
  [ICML 2025][AI Safety][Differential Privacy] The Cluster-DP mechanism is proposed to leverage non-sensitive clustering structure information to improve the privacy-variance trade-off of causal effect estimation in differentially private randomized experiments. Without sacrificing privacy guarantees, a more homogeneous clustering structure significantly reduces the variance loss of ATE estimation.
tags:
  - "ICML 2025"
  - "AI Safety"
  - "Differential Privacy"
  - "Causal Inference"
  - "Randomized Experiments"
  - "Clustering"
  - "Variance Reduction"
  - "Label Differential Privacy"
date: 2026-05-08
content_hash: ec5e80086214314a
---

# Improving the Variance of Differentially Private Randomized Experiments through Clustering

**Conference**: ICML 2025  
**arXiv**: [2308.00957](https://arxiv.org/abs/2308.00957)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: Differential Privacy, Causal Inference, Randomized Experiments, Clustering, Variance Reduction, Label Differential Privacy

## TL;DR

The Cluster-DP mechanism is proposed to leverage non-sensitive clustering structure information to improve the privacy-variance trade-off of causal effect estimation in differentially private randomized experiments. Without sacrificing privacy guarantees, a more homogeneous clustering structure significantly reduces the variance loss of ATE estimation.

## Background & Motivation

### Scenario

Consider a typical online advertising scenario: an advertising platform holds user data, and advertisers wish to obtain user-level differentially private data to analyze the causal effects of ads. The platform must protect user privacy while enabling advertisers to conduct effective causal analysis (e.g., A/B testing).

### Key Challenge

Differential privacy (DP) protects individual privacy by adding noise, which inevitably increases the variance of statistical estimators. In causal inference, increased variance means a reduced ability to detect small causal effects. This constitutes the fundamental contradiction of the **privacy-variance trade-off**.

### Key Insight

Not all attributes are sensitive. If **non-sensitive clustering information** (such as user country, DMA region) can be utilized to improve the privacy mechanism, it is possible to reduce variance without degrading privacy. Intuitively, if the response values of users within a cluster are highly similar (high homogeneity), replacing the uniform distribution with the noisy intra-cluster empirical distribution to generate pseudo-responses will introduce less noise.

### Limitations of Prior Work

**Uniform-Prior-DP** (a generalization of Kancharla & Kang, 2021): Replaces true responses with uniform random responses with a fixed probability $\lambda$. It does not utilize prior information about the response distribution, resulting in large variance loss.

**Aggregation Methods** (Noisy Horvitz-Thompson / Noisy Histogram): Only output aggregated statistics without providing user-level data, failing to satisfy the advertisers' need for fine-grained analysis.

## Method

### Overall Architecture

The **Cluster-DP Mechanism** (Algorithm 2) consists of the following steps:

1. **Construct Empirical Intra-Cluster Distributions**: For each cluster $c$ and treatment group $a \in \{0,1\}$, compute the empirical distribution of responses for the control/treatment group within that cluster, $\hat{p}_a(y|c)$.
2. **Add Laplace Noise**: Add $\text{Laplace}(\sigma/n_{a,c})$ noise to each response probability to obtain $q_a(y|c)$.
3. **Truncation and Normalization**: Truncate the probabilities to $[\gamma, 1]$ ($\gamma \le 1/K$), and then re-normalize to make it a valid distribution $\tilde{q}_a(y|c)$.
4. **Response Perturbation**: Each user's true response is replaced with a random sample from the distribution $\tilde{q}$ with probability $\lambda$, and remains unchanged with probability $1-\lambda$.
5. **Debiased Estimation**: Construct an unbiased ATE estimator $\hat{\tau}_Q$ by inverting the response randomization matrix $Q$.

### Key Designs

#### Key Design 1: Data-Dependent Prior Distribution

Unlike Uniform-Prior-DP which uses a fixed uniform distribution, Cluster-DP uses the **noisy empirical intra-cluster distribution** as the replacement distribution. When the intra-cluster responses have high homogeneity, this distribution is closer to the true distribution, thereby introducing less variance during the replacement operation.

When the truncation parameter $\gamma = 1/K$, Cluster-DP degenerates to Uniform-Prior-DP (where the distribution becomes uniform). Thus, Uniform-Prior-DP is a special case of Cluster-DP.

#### Key Design 2: Cluster-Free-DP

When no good clustering structure is available, all users are treated as a single large cluster. Cluster-Free-DP is a special case of Cluster-DP, which can still leverage the overall empirical distribution information.

#### Key Design 3: Debiased Estimator

Construct the response randomization matrix $Q_{c,a}[y',y] = (1-\lambda) \cdot \mathbb{I}(y'=y) + \lambda \cdot \tilde{q}_a(y'|c)$. This matrix describes the probability that the true response $y$ is corrupted into $y'$. By inverting the matrix $Q$, an unbiased estimate can be recovered from the noisy responses:

$$\hat{\tau}_Q = \sum_c \frac{n_c}{n} \sum_{i \in c} \left( \frac{y^T Q^{-1}_{c,z_i}[\tilde{y}_i] \cdot z_i}{n_{1,c}} - \frac{y^T Q^{-1}_{c,z_i}[\tilde{y}_i] \cdot (1-z_i)}{n_{0,c}} \right)$$

This estimator is conditionally unbiased with respect to the stratified difference-in-means estimator under the randomness of the DP mechanism.

### Loss & Training

**Theorem 3.1**: Cluster-DP satisfies $(\varepsilon, \delta)$-label differential privacy, where:
- $\varepsilon = \min(1/\sigma, 2/\gamma) + \tilde{\varepsilon}$
- The first term is the privacy budget for privatizing the empirical distribution.
- The second term is the privacy budget for generating privatized responses (combined via composition theorems).
- Setting $\tilde{\varepsilon} = \log(1 + (1-\lambda)/(\lambda\gamma))$ yields a pure DP guarantee with $\delta = 0$.

**Important Property**: The privacy guarantee **does not depend on the quality or structure of the clustering**; clustering only affects the variance.

### Variance Analysis

**Theorem 3.5**: The variance gap is bounded by:

$$0 \le \text{Var}_{DP,z}[\hat{\tau}_Q] - \text{Var}_z[\hat{\tau}_{\text{No-DP}}] \le \left(\frac{1}{(1-\lambda)^2} - 1\right) \cdot \sum_a \phi_a + \sum_a \sum_c \frac{n_c^2}{n^2} \cdot \frac{A(n_{a,c})}{n_{a,c}}$$

where:
- **$\phi_a$** (clustering homogeneity) $= \sum_c \frac{n_c^2}{n^2} \cdot \frac{S^2(y_c(a))}{n_{a,c}}$, i.e., the weighted average of intra-cluster variances.
- The first term depends on clustering homogeneity—the more homogeneous the clustering (smaller $\phi_a$), the smaller the variance loss.
- The second term $A(x)$ depends on the truncation parameter $\gamma$ and noise $\sigma$, and is independent of clustering.

## Key Experimental Results

### main Results

#### Experiment 1: Privacy-Variance Trade-off (Synthetic Data)

| Mechanism | Variance Performance at Fixed $\varepsilon=0.2$ |
|---|---|
| Cluster-DP (small $\gamma$) | **Significantly lowest** |
| Cluster-Free-DP | Second lowest |
| Uniform-Prior-DP (stratified) | High |
| Uniform-Prior-DP (non-stratified) | Highest |

- Gaussian Mixture Model setup: $C=3$ clusters, $n=3500$, $K=12$ outcomes.
- Over the range $\gamma \in [0.1/K, 1/K]$, Cluster-DP consistently outperforms other methods.
- Privacy-variance Pareto frontier after optimizing $\sigma$ and $\gamma$: Cluster-DP dominates the baselines.

#### Experiment 2: Effect of Clustering Quality

| Clustering Quality Metric $\beta$ | Variance Ratio $\text{Var}(\text{Cluster-DP})/\text{Var}(\text{Cluster-Free-DP})$ |
|---|---|
| $\beta=0$ (no clustering effect) | $\approx 1.0$ |
| $\beta=2$ | $\approx 0.8$ |
| $\beta=4.5$ (high homogeneity) | $\approx 0.4 \text{ to } 0.6$ |

- The more homogeneous the clustering (larger $\beta$), the more pronounced the advantage of Cluster-DP over Cluster-Free-DP.
- Smaller $\lambda$ (greater retention of true responses) leads to a more significant clustering effect.

#### Experiment 3: Semi-synthetic Data from YouTube Social Network

- 50 largest communities, 22,179 users, $K=8$ discrete outcomes.
- Features are based on graph structure (number of nodes, number of edges, density, etc.).
- Cluster-DP also achieves a better privacy-variance trade-off on YouTube's natural clustering structure.

## Highlights & Insights

1. **Elegant Theoretical Unification**: Uniform-Prior-DP and Cluster-Free-DP are both special cases of Cluster-DP (corresponding to $\gamma=1/K$ and $C=1$, respectively), forming a hierarchical family of methods.
2. **Decoupling of Privacy and Variance**: The privacy guarantee does not rely on clustering quality, while the variance improves with clustering homogeneity—this is the core design advantage.
3. **Label DP Perspective**: Focuses on protecting outcome labels (rather than features), which highly aligns with practical scenarios (e.g., an ad platform protecting click behaviors while sharing geographic information).
4. **User-Level Data Output**: Unlike aggregation methods, Cluster-DP outputs user-level differentially private data, enabling advertisers to perform any follow-up analysis (with privacy guaranteed by post-processing properties).
5. **Quantitative Definition of Intra-Cluster Homogeneity** (Definition 3.4) is concise and practically interpretable: in the super-population sense, it is $\text{Var}(y(a)) - \text{Var}(\mathbb{E}[y(a)|c])$.

## Limitations & Future Work

1. **Requirement for Discrete Outcome Space**: The response space $\mathcal{Y}$ must be a finite set; continuous outcomes need to be binned beforehand. The choice of binning granularity increases the hyperparameter tuning burden.
2. **Prior Access to Clustering Quality**: A well-defined clustering structure must be pre-determined. Although Remark 2.1 discusses the possibility of DP clustering, it would consume additional privacy budget.
3. **Tightness of the Variance Bound**: Theorem 3.5 provides an upper bound; actual variance improvement might be larger, but the theoretical bound is not necessarily tight.
4. **Fixed Clustering Assumption**: Assuming static clustering makes it unsuitable for dynamic user population scenarios.
5. **Focus Only on ATE**: The method has not been extended to estimate Conditional Average Treatment Effects (CATE) or heterogeneous treatment effects.
6. **Communication Complexity**: Requires quadratic communication overhead for seed exchange among workers (though only one-time)—however, this is primarily an issue for CafCor, while Cluster-DP only requires a central unit.

## Related Work & Insights

- **Esfandiari et al. (2022)**: The inspiration for Cluster-DP, though their original work focused on excess risk minimization in classification tasks rather than causal inference. This paper adapts it to ATE estimation and redesigns the variance analysis.
- **Kancharla & Kang (2021)**: Pioneers of DP causal inference with binary outcomes, which is equivalent to a special case of Uniform-Prior-DP ($K=2$, no clustering).
- **Ohnishi & Awan (2023)**: Minimax optimal estimators under local DP, which is a special case of Algorithm 3 (single cluster).
- **Label DP Literature** (Chaudhuri & Hsu, 2011): This paper adopts the label DP definition, protecting only responses rather than features.
- **Insights for Causal Inference Research**: Non-sensitive covariates can serve as clustering signals to improve privacy mechanisms. This approach can be generalized to other causal estimators (such as CATE or quantile treatment effects).

## Rating

- ⭐ Novelty: 4/5 — Introducing clustering information into DP causal inference is a novel approach, and the hierarchical design of the method family is elegant.
- ⭐ Theoretical Depth: 4/5 — Derivations of privacy guarantees and variance bounds are complete, and the analysis of privacy-variance decoupling is clean.
- ⭐ Experimental Thoroughness: 3.5/5 — Good validation on synthetic experiments, and the YouTube semi-synthetic experiment adds credibility, though it lacks large-scale real-world A/B testing validation.
- ⭐ Writing Quality: 4/5 — Clear structure, well-explained motivation, and standardized algorithm descriptions.
- ⭐ Practical Value: 4/5 — Directly addresses the actual needs of advertising platforms, and the method is easy to deploy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mitigating Disparate Impact of Differentially Private Learning through Bounded Adaptive Clipping](../../NeurIPS2025/ai_safety/mitigating_disparate_impact_of_differentially_private_learning_through_bounded_a.md)
- [\[ICML 2025\] Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off](clients_collaborate_flexible_differentially_private_federated_learning_with_guar.md)
- [\[ICLR 2026\] Dual Randomized Smoothing: Beyond Global Noise Variance](../../ICLR2026/ai_safety/dual_randomized_smoothing_beyond_global_noise_variance.md)
- [\[ICML 2025\] Private Model Personalization Revisited](private_model_personalization_revisited.md)
- [\[ICML 2025\] Faster Rates for Private Adversarial Bandits](faster_rates_for_private_adversarial_bandits.md)

</div>

<!-- RELATED:END -->
