---
title: >-
  [Paper Note] Networked Information Aggregation for Binary Classification
description: >-
  [ICML 2026][Others][vertical federated learning] This paper generalizes the conclusion from Kearns-Roth-Ryu 2026—which states that linear regression agents on a DAG can approach global optimality by sequentially passing prediction columns—to binary classification. Under the $M$-coverage condition, where each agent observes only a subset of feature columns and sequent
tags:
  - ICML 2026
  - Others
  - vertical federated learning
date: 2026-05-08
content_hash: e11b3167f4ca8639
---
# Networked Information Aggregation for Binary Classification

**Conference**: ICML 2026  
**arXiv**: [2605.01082](https://arxiv.org/abs/2605.01082)  
**Code**: None  
**Area**: Distributed Learning / Networked Aggregation / Binary Classification Theory  
**Keywords**: vertical federated learning, logistic regression, DAG sequential learning, Bregman divergence, excess loss lower bound

## TL;DR
This paper generalizes the conclusion from Kearns-Roth-Ryu 2026—which states that linear regression agents on a DAG can approach global optimality by sequentially passing prediction columns—to binary classification. Under the $M$-coverage condition, where each agent observes only a subset of feature columns and sequentially forwards its logit to downstream neighbors, an excess BCE loss of $O(M/\sqrt{D})$ relative to the global logistic regression optimum is achieved. Furthermore, the authors construct hard instances to prove an $\Omega(k/D)$ lower bound, characterizing network depth as the fundamental bottleneck for information aggregation.

## Background & Motivation

**Background**: Social and networked learning has a history spanning half a century, including the DeGroot model, Bayesian observational learning, information cascades, Vertical Federated Learning (VFL), and Split Learning. These models address whether dispersed partial information across nodes can be aggregated into globally correct decisions. Kearns-Roth-Ryu (2026) provided clean results under linear regression with squared loss: on a DAG where each agent sees a subset of feature columns and passes local linear predictions to downstream neighbors, the excess loss of the final agent is controlled by the network depth $D$ and coverage parameter $M$.

**Limitations of Prior Work**: Classification is more common than regression in practical deployments (e.g., medical diagnosis, fraud detection). However, the proofs by Kearns et al. rely heavily on "residual orthogonality + Pythagoras variance decomposition" for squared loss, tools that do not exist under the BCE loss with a sigmoid link. Practical schemes in VFL literature (SecureBoost, Split Learning) rely on multi-round communication to exchange gradients or activations; few have addressed whether "one-shot unidirectional logit passing" can achieve aggregation.

**Key Challenge**: The probability space for classification is not Euclidean. Linear combinations in probability space do not equal linear combinations in feature space. This is why the authors emphasize passing logits instead of probabilities, though it also means the original geometry no longer applies.

**Goal**: To clearly define the "sequential logit passing on a DAG" protocol; prove it achieves global MLE under $M$-coverage conditions; and provide a matching lower bound to prove that depth is indeed the bottleneck.

**Key Insight**: Substitute the Euclidean geometry of squared loss with the Bregman/KL geometry of BCE. The loss difference is equivalent to the KL divergence between predicted distributions, and Pinsker’s inequality is used to translate KL progress into prediction error. The technical core lies in finding that the optimal BCE solution still satisfies residual orthogonality $\mathbb{E}[x(p^*(x) - y)] = 0$ (while the geometry changes, the first-order necessary condition remains).

**Core Idea**: Construct a telescoping sum of the loss reduction at each stage of the chain. Combine Pinsker ($KL \rightarrow$ squared error) with the orthogonality of the segment on the path where "feature $x_l$ is observed by agent $j$" to bound the prediction residuals for all features.

## Method

### Overall Architecture
Each agent $A_i$ on a DAG holds a feature subset $S_i \subseteq [d]$ and learns in topological order. When it is an agent’s turn, it receives logits $\{z_j : A_j \in \mathrm{Pa}(A_i)\}$ from all parent nodes, concatenates them with local features $x_{S_i}$ to train a logistic regression $z_i(x) = w_i^T x_{S_i} + \sum_{j} v_{ij} z_j(x)$ minimizing BCE, and passes its logit $z_i$ to successors. The final output is given by the sink agent (or the end of the path $A_D$). Logits are passed instead of probabilities to preserve the information geometry of the exponential family, allowing downstream agents to continue linear refinement without loss.

### Key Designs

**1. Residual Orthogonality Lemma + Bregman Loss Decomposition: Rewriting BCE loss difference as KL divergence between predicted distributions.**

The original Kearns proof relied on the quadratic structure of squared loss (where $\|p-q\|^2=$ loss diff). This does not exist for BCE. The authors first prove Lemma 3.1: the optimal BCE solution still satisfies residual orthogonality $\mathbb{E}[x(p^*(x) - y)] = 0$, derived by taking the expectation of $\nabla_\theta L = 0$. Then, using Lemma 3.3 and the identity $\log \sigma(z) = z - \log(1 + e^z)$, the loss difference is expanded. By adding and subtracting $p^*(x)(\theta - \theta^*)^T x$ and applying orthogonality, they obtain $L(q) = L(p^*) + D(p^* \| q)$, where $D$ is the Bernoulli KL. This is the equivalent of "variance decomposition" for BCE: the excess loss of any suboptimal predictor exactly equals its KL divergence from the optimum, allowing the telescoping argument to proceed.

**2. Path Residual Control (Lemma 3.5): Controlling the discrepancy between global optimal logit and current prediction through agents on the coverage path.**

The core is to reduce the "global residual of any feature $x_l$" to the "accumulated loss reduction across segments of the chain." For any linear logit $z_g(x) = \sum \alpha_l x_l$, the triangle inequality splits $|\mathbb{E}[(p_k - y) z_g]|$ into correlation terms $\sum |\alpha_l| |\mathbb{E}[x_l (p_k - y)]|$. For each $x_l$, an agent $A_j$ that observed it is identified, where orthogonality gives $\mathbb{E}[x_l (p_j - y)] = 0$. Using Cauchy-Schwarz and Pinsker’s inequality ($D(p \| q) \geq 2 \mathbb{E}[(p-q)^2]$), $\|p_k - p_j\|_2 \leq \sqrt{k \varepsilon / 2}$ is established, bounding the discrepancy over a path of length $k$ by $O(\sqrt{k\varepsilon})$. This "telescoping reduction via intermediate agents" converts the networked learning problem into an analysis of cumulative progress along the chain.

**3. Pigeonhole Parameter Selection + Global Convergence (Theorem 3.7): Finding a stable block with average progress via the pigeonhole principle.**

The path of length $D$ is partitioned into $K = \lfloor D/M \rfloor$ disjoint blocks. By the pigeonhole principle, there must exist a block where the total loss reduction is $\leq L(p_1) / K \leq 2M L(p_1) / D$. Let this stable block span indices $s..t$. Applying Lemmas 3.4 and 3.5 yields $L(p_t) \leq L(p^*) + B_{p^*} B_X \sqrt{M \varepsilon / 2}$. Since $L(p_1) \leq \log 2 < 1$ (achievable at $\theta = 0$), the final result is $L(p_D) - L(p^*) \leq B_{p^*} B_X M / \sqrt{D} = O(M / \sqrt{D})$. The beauty of the pigeonhole argument is avoiding fine-grained control over every segment—stating that "there is always a segment with at least average progress" is sufficient for global convergence.

### Loss & Training
The local optimization objective for all agents is standard BCE without regularization or additional structure. The communication protocol involves each agent passing the scalar logit (inner part of the sigmoid) rather than the resulting probability. This maintains linear additivity within the exponential family, allowing downstream nodes to perform linear regression on parent logits without the nonlinearity of the sigmoid destroying the information geometry.

## Key Experimental Results
This is a theoretical paper and does not include numerical experimental tables. However, the authors provide a conceptual complexity comparison table for the upper and lower bounds.

### Main Results

| Method | Task | Loss | Upper Bound | Lower Bound |
| :--- | :--- | :--- | :--- | :--- |
| Kearns-Roth-Ryu 2026 | Regression | MSE | $O(M/\sqrt{D})$ | — |
| **Ours** | Binary Classification | BCE | $O(M/\sqrt{D})$ | $\Omega(k/D)$ |

Upper bound conditions: path length $D$, every $M$ consecutive agents cover all features; constants depend on $\mathbb{E}[x_l^2] \leq B_X^2$ and $\|\alpha^*\|_1 \leq B_{p^*}$.

### Ablation Study
Key designs for the lower bound construction (Theorem 4.5):

| Design | Function | Key Lemma |
| :--- | :--- | :--- |
| Latent variable $Z_i \sim \mathcal{N}(0,1)$ iid, features $x_i = Z_i - Z_{i-1}$ | Ensures $Z_k = \sum x_j$; any prefix of features is independent of the label $y \sim \text{Ber}(\sigma(Z_k))$ | 4.1 (Correlation Recursion) |
| Agents on the path observe dimensions in cyclic order $\ell = ((i-1) \mod k) + 1$ | Forces the unlocking of only one effective feature per pass | — |
| Optimal logit after pass $p$ follows $z_D = c(Z_k + \xi/\sqrt{p})$, $\xi \sim \mathcal{N}(0, V_p)$ | Noise variance can only decay at a rate of $1/p$ | 4.2, 4.3 |
| Optimal $c \in (0,1)$ from 2nd-order sigmoid smoothness, then MVT translates prob diff to logit diff | Derives $L(p_D) - L(p^*) \geq C/(p+1) = \Omega(k/D)$ | 4.4, 4.5 |

### Key Findings
- The upper bound $O(M/\sqrt{D})$ is of the same order as the original Kearns regression result, suggesting the $\sqrt{D}$ rate is not exclusive to squared loss and is also accessible via BCE.
- For fixed $M = O(k)$, the lower bound $\Omega(k/D)$ differs from the upper bound $O(k/\sqrt{D})$ only by a $\sqrt{D}$ factor—an open gap acknowledged by the authors.
- The "differential encoding" of features in the lower bound ($x_i = Z_i - Z_{i-1}$) creates a scenario where any single $x_i$ is independent of $y$. Data must be chained across sufficient length to solve for $Z_k$; this construction shows that "feature-by-feature disentanglement" inherent to the protocol is the root cause of the limited rate, not loose analysis.
- The paper discusses the "regression-to-classification" gap across multiple dimensions (compressed sensing, second-order acceleration, Conformal Prediction), arguing that BCE analysis is not a trivial modification of MSE.

## Highlights & Insights
- Substituting Euclidean decomposition with Bregman/KL is a standard approach for generalizing regression to classification, but the author's use of "orthogonality on segments of the coverage path" to decompose "global feature residuals" into "local sub-path KL progress" is an ingenious maneuver that could be reused in other distributed GLM problems.
- The emphasis on passing logits rather than probabilities is an undervalued design principle: the sigmoid maps the exponential family to $(0,1)$, but logits are the natural coordinates for downstream linear combinations. This provides practical guidance for industrial VFL system design.
- The differential encoding $x_i = Z_i - Z_{i-1}$ in the lower bound is a highly efficient instance of an "information bottleneck," demonstrating incontrovertibly that $k$ passes are required for disentanglement using only $k$-dimensional Gaussians.

## Limitations & Future Work
- A $\sqrt{D}$ gap remains between the upper and lower bounds, which the authors hope for future work to tighten.
- The protocol itself is "non-interactive + unidirectional + single logit," which is quite restrictive for practical VFL—real-world systems often prefer multi-round exchanges of gradients or activations for better performance. Theoretical results show that if one insists on the weakest communication, $1/\sqrt{D}$ is the best achievable rate, though its engineering utility is limited.
- It does not consider real VFL issues like privacy, noise, or partial alignment, remaining a pure statistical learning rate analysis.
- The assumptions of bounded second moments for features and bounded $\ell_1$ norm for optimal logit coefficients may not hold for real industrial data.

## Related Work & Insights
- **vs Kearns-Roth-Ryu 2026 (regression)**: Same protocol, but the proof framework is completely rewritten—Bregman replaces Euclidean, KL replaces variance, and Pinsker replaces Pythagoras; this is a clean example of "standard transformation" for regression-to-classification.
- **vs VFL (SecureBoost, etc.)**: Industrial VFL relies on multi-round interaction + encrypted summation. This paper's single-pass approach provides a theoretical guarantee that "aggregation is possible even in worst-case single-pass," though its empirical accuracy likely cannot compete with multi-round schemes.
- **vs Split Learning**: Split learning uses intermediate activations as communication, which is theoretically closer to this paper (passing representations downstream). However, this paper provides rigorous convergence rates specifically for linear logistic models, whereas split learning lacks clean corresponding results for deep networks.
- **Insight**: Could the protocol be modified to "pass sufficient statistics rather than logits"—such as score + Fisher info in GLMs—to achieve a better trade-off between interaction rounds and depth? This remains an open direction.

## Rating
- Novelty: ⭐⭐⭐⭐ The effort to generalize regression results to classification is a solid extension; the lower bound construction is particularly elegant.
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical with no numerical experiments; the $\sqrt{D}$ gap between bounds is not empirically tested to see which is tighter.
- Writing Quality: ⭐⭐⭐⭐⭐ The lemma dependency chain is very clear, and Section 1.2's discussion on "why regression-to-classification is non-trivial" is highly insightful.
- Value: ⭐⭐⭐ Primarily targeting the theory community; industrial VFL systems are unlikely to change architectures because of this, but it establishes a baseline for theoretical analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Structure-Induced Information for Rerooting Levin Tree Search](structure-induced_information_for_rerooting_levin_tree_search.md)
- [\[CVPR 2026\] Advancing Image Classification with Discrete Diffusion Classification Modeling](../../CVPR2026/others/advancing_image_classification_with_discrete_diffusion_classification_modeling.md)
- [\[ICML 2026\] Coupled Training with Privileged Information and Unlabeled Data](coupled_training_with_privileged_information_and_unlabeled_data.md)
- [\[AAAI 2026\] Improved Differentially Private Algorithms for Rank Aggregation](../../AAAI2026/others/improved_differentially_private_algorithms_for_rank_aggregation.md)
- [\[ICML 2025\] Sampling from Binary Quadratic Distributions via Stochastic Localization](../../ICML2025/others/sampling_from_binary_quadratic_distributions_via_stochastic_localization.md)

</div>

<!-- RELATED:END -->
