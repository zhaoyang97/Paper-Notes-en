---
title: >-
  [Paper Note] Networked Information Aggregation for Binary Classification
description: >-
  [ICML 2026][vertical federated learning] This paper extends the conclusion from Kearns-Roth-Ryu 2026—stateing that "linear regression agents on a DAG can approach global optimality by passing prediction sequences"—to bin…
tags:
  - "ICML 2026"
  - "vertical federated learning"
  - "logistic regression"
  - "DAG sequential learning"
  - "Bregman divergence"
  - "excess loss lower bound"
date: 2026-05-08
content_hash: 9bcea1c5e82d6866
---

# Networked Information Aggregation for Binary Classification

**Conference**: ICML 2026  
**arXiv**: [2605.01082](https://arxiv.org/abs/2605.01082)  
**Code**: None  
**Area**: Distributed Learning / Network Aggregation / Binary Classification Theory  
**Keywords**: vertical federated learning, logistic regression, DAG sequential learning, Bregman divergence, excess loss lower bound

## TL;DR
This paper extends the conclusion from Kearns-Roth-Ryu 2026—stateing that "linear regression agents on a DAG can approach global optimality by passing prediction sequences"—to binary classification. Each agent observes only a subset of feature columns and sequentially forwards its logit downstream. Under $M$-coverage conditions, they achieve the global logistic regression optimum with an $O(M/\sqrt{D})$ excess BCE loss. Simultaneously, a hard instance is constructed to prove an $\Omega(k/D)$ lower bound, characterizing network depth as the fundamental bottleneck for information aggregation.

## Background & Motivation

**Background**: Social and network learning have a lineage spanning half a century, including the DeGroot model, Bayesian observational learning, information cascades, Vertical Federated Learning (VFL), and Split Learning. These models address whether partial information dispersed across different nodes can be aggregated into a globally correct decision. Kearns-Roth-Ryu (2026) provided a clear result for linear regression with squared loss: on a DAG, each agent sees a subset of feature columns and passes local linear prediction sequences to downstream neighbors; the excess loss of the final agent can be controlled by network depth $D$ and coverage parameter $M$.

**Limitations of Prior Work**: In practical deployment, classification is more common than regression (e.g., medical diagnosis, fraud detection). However, the proofs by Kearns et al. rely heavily on "residual orthogonality + Pythagoras variance decomposition" for squared loss, both of which do not exist under BCE + sigmoid links. Practical schemes in VFL literature (SecureBoost, Split Learning) rely on multi-round communication to exchange gradients or activations, and none address whether "one-way logit passing" can achieve aggregation.

**Key Challenge**: The probability space for classification is not Euclidean—linear combination in probability space $\neq$ linear combination in feature space. This is why the authors emphasize passing logits rather than probabilities, but it also means the original geometry no longer applies.

**Goal**: To define the "sequential logit passing on a DAG" protocol; prove it achieves global MLE under the $M$-coverage condition; and provide a matching lower bound proving that depth is indeed the bottleneck.

**Key Insight**: Replace Euclidean geometry of squared loss with the Bregman/KL geometry of BCE—where the loss difference is the KL divergence between predicted distributions—then use Pinsker's inequality to translate KL progress into prediction error. The technical core is the discovery that the optimal solution for BCE still satisfies residual orthogonality $\mathbb{E}[x(p^*(x) - y)] = 0$ (though the geometry differs, the first-order necessary condition remains).

**Core Idea**: Formulate the "loss reduction per segment on the chain" into a telescoping sum. Combine Pinsker (KL progress $\to$ squared error) with orthogonality from segments where "feature $x_l$ is observed by agent $j$" to bound the prediction residuals for all features.

## Method

### Overall Architecture
On a DAG, each agent $A_i$ holds a feature subset $S_i \subseteq [d]$ and learns in topological order. Upon their turn, they receive logits $\{z_j : A_j \in \mathrm{Pa}(A_i)\}$ from all parent nodes, concatenate them with local features $x_{S_i}$ to train a logistic regression $z_i(x) = w_i^T x_{S_i} + \sum_{j} v_{ij} z_j(x)$, minimize BCE, and forward their own logit $z_i$ to successors. The final output is given by the sink agent (or $A_D$ at the end of the path). Note that logits are passed instead of probabilities to preserve the information geometry of the exponential family, allowing downstream agents to continue linear combination without loss.

### Key Designs

1.  **Residual Orthogonality Lemma + Bregman Loss Decomposition**:
    *   Function: Rewrites the "binary cross-entropy loss difference of logistic regression" as expected KL divergence, bridging loss reduction and predicted distribution approximation.
    *   Mechanism: Lemma 3.1 proves $\mathbb{E}[x(p^*(x) - y)] = 0$ holds for the optimal BCE solution (derived from $\nabla_\theta L = 0$). Lemma 3.3 uses the identity $\log \sigma(z) = z - \log(1 + e^z)$ to expand the loss difference, then adds/subtracts $p^*(x)(\theta - \theta^*)^T x$, eliminating one term via orthogonality to obtain $L(q) = L(p^*) + D(p^* \| q)$, where $D$ is the Bernoulli KL. This is the equivalent of "variance decomposition" under BCE.
    *   Design Motivation: The original Kearns proof relied on $\|p - q\|^2 = \text{loss diff}$, which does not exist for BCE. The Bregman decomposition shows that the excess loss of any sub-optimal predictor is exactly the KL divergence to the optimum, enabling the telescoping sum.

2.  **Path Residual Control (Lemma 3.5)**:
    *   Function: Bounds the "difference between global optimal logit and current predictor" by $O(\sqrt{k \varepsilon})$ on a coverage path of length $k$, where $\varepsilon$ is the total loss reduction on that path.
    *   Mechanism: For any linear logit $z_g(x) = \sum \alpha_l x_l$, bound $|\mathbb{E}[(p_k - y) z_g]|$ using the triangle inequality over correlation terms $\sum |\alpha_l| |\mathbb{E}[x_l (p_k - y)]|$. For each $x_l$, find an agent $A_j$ who previously observed it; the orthogonality of that step gives $\mathbb{E}[x_l (p_j - y)] = 0$. Use Cauchy-Schwarz and Pinsker ($D(p \| q) \geq 2 \mathbb{E}[(p-q)^2]$) to get $\|p_k - p_j\|_2 \leq \sqrt{k \varepsilon / 2}$.
    *   Design Motivation: The core is using the "intermediate agent who observed the feature" to perform a telescoping reduction on any feature residual, reducing the global information aggregation problem to the "cumulative reduction on a segment of the chain."

3.  **Pigeonhole Parameter Selection + Global Convergence (Theorem 3.7)**:
    *   Function: Partitions a path of length $D$ into $K = \lfloor D/M \rfloor$ disjoint blocks. By the pigeonhole principle, there must exist a block where the total loss reduction is $\leq L(p_1) / K \leq 2M L(p_1) / D$.
    *   Mechanism: Let this stable block span indices $s..t$. Applying Lemma 3.4 and 3.5 on this path yields $L(p_t) \leq L(p^*) + B_{p^*} B_X \sqrt{M \varepsilon / 2}$. Since $L(p_1) \leq \log 2 < 1$ (achievable with $\theta = 0$), the final $L(p_D) - L(p^*) \leq B_{p^*} B_X M / \sqrt{D} = O(M / \sqrt{D})$.
    *   Design Motivation: The pigeonhole argument avoids the difficulty of fine-grained control over every segment; it suffices to show that "there is always one segment where progress is no worse than average."

### Loss & Training
The local optimization objective for all agents is standard BCE without regularization or additional structure. The communication method involves each agent forwarding the scalar logit inside the sigmoid rather than the probability after the sigmoid. This is to maintain linear additivity within the exponential family: downstream agents can perform linear regression directly on parent logits, avoiding the destruction of information geometry by sigmoid nonlinearity.

## Key Experimental Results

This is a purely theoretical paper with no numerical experimental tables. However, the authors place upper and lower bounds in a conceptual "complexity comparison table."

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
| Latent variables $Z_i \sim \mathcal{N}(0,1)$ iid, features $x_i = Z_i - Z_{i-1}$ | Ensures $Z_k = \sum x_j$, but any feature prefix is independent of label $y \sim \text{Ber}(\sigma(Z_k))$ | 4.1 (Info Correlation Recursion) |
| Agents on path look at 1D features in cyclic order $\ell = ((i-1) \mod k) + 1$ | Forces only one effective feature to be "unlocked" per pass | — |
| Optimal logit after pass $p$ is $z_D = c(Z_k + \xi/\sqrt{p})$, $\xi \sim \mathcal{N}(0, V_p)$ | Noise variance only decays at rate $1/p$ | 4.2, 4.3 |
| Optimal $c \in (0,1)$ from sigmoid second-order smoothness + MVT | Derives $L(p_D) - L(p^*) \geq C/(p+1) = \Omega(k/D)$ | 4.4, 4.5 |

### Key Findings
- The $O(M/\sqrt{D})$ upper bound is of the same order as the original Kearns regression result, showing that the $\sqrt{D}$ rate is not exclusive to squared loss and can be achieved by BCE.
- The $\Omega(k/D)$ lower bound differs from the $O(k/\sqrt{D})$ upper bound only by a factor of $\sqrt{D}$ when $M = O(k)$—an open gap acknowledged by the authors.
- In the lower bound construction, features are "differential encoded"—any $x_i$ alone is independent of $y$. Only a sufficiently long sequence can decode $Z_k$. This construction shows that the "feature-by-feature disentanglement" of the protocol itself is the fundamental cause limiting the rate.
- The paper also discusses that the "regression-to-classification" gap is non-trivial across several directions, including compressed sensing, second-order acceleration, and conformal prediction, arguing that "BCE is not a minor adjustment to MSE."

## Highlights & Insights
- Replacing Euclidean decomposition with Bregman/KL is a standard approach for extending regression to classification, but the author cleverly uses "orthogonality of a segment on the coverage path" to decompose the "global residual of feature $x_l$" into "cumulative KL progress of local sub-paths."
- Emphasizing logits over probabilities is an underrated design principle: sigmoid maps the exponential family to $(0,1)$, but when downstream agents want to perform further linear combinations, the logit is the natural coordinate.
- The differential encoding $x_i = Z_i - Z_{i-1}$ in the lower bound construction is a highly economical instance of a "bottleneck": it demonstrates that $k$ passes are required for disentanglement.

## Limitations & Future Work
- There remains a $\sqrt{D}$ gap between the upper and lower bounds, which future work might tighten.
- The protocol is "non-interactive + one-way single logit," which is restrictive compared to industrial VFL systems that exchange gradients or activations for better performance.
- Real VFL issues like privacy, noise, and partial alignment are not considered; this is purely a statistical learning rate analysis.
- Assumes bounded second moments of features and bounded $\ell_1$ norm of optimal logit coefficients.

## Related Work & Insights
- **vs Kearns-Roth-Ryu 2026 (regression)**: Same protocol, but the proof framework is completely rewritten—Bregman instead of Euclidean, KL instead of variance, Pinsker instead of Pythagoras.
- **vs VFL (SecureBoost, etc.)**: Industrial VFL relies on multi-round interaction. This paper is single-pass, providing theoretical guarantees that aggregation can happen in a single-pass even in worst-case scenarios.
- **vs Split Learning**: Split learning uses intermediate activations as communication. This paper provides strict convergence rates within linear logistic models, whereas split learning lacks clean corresponding results in deep networks.
- **Insight**: Could the protocol be modified to "pass sufficient statistics instead of logits"—e.g., passing scores + Fisher info in GLMs—to achieve a better trade-off between interaction rounds and depth?

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Structure-Induced Information for Rerooting Levin Tree Search](structure-induced_information_for_rerooting_levin_tree_search.md)
- [\[AAAI 2026\] Improved Differentially Private Algorithms for Rank Aggregation](../../AAAI2026/others/improved_differentially_private_algorithms_for_rank_aggregation.md)
- [\[ICML 2026\] Multi-Level Strategic Classification: Incentivizing Improvement Through Promotion and Relegation Dynamics](multi-level_strategic_classification_incentivizing_improvement_through_promotion.md)
- [\[ICML 2026\] Coupled Training with Privileged Information and Unlabeled Data](coupled_training_with_privileged_information_and_unlabeled_data.md)
- [\[ICML 2026\] ParalESN: Enabling Parallel Information Processing in Reservoir Computing](paralesn_enabling_parallel_information_processing_in_reservoir_computing.md)

</div>

<!-- RELATED:END -->
