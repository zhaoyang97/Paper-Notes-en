---
title: >-
  [Paper Note] Networked Information Aggregation for Binary Classification
description: >-
  [ICML 2026][Others][vertical federated learning] This work extends the conclusion of Kearns-Roth-Ryu 2026—which states that linear regression agents on a DAG can approach global optimality by sequentially passing prediction columns—to binary classification. Under the $M$-coverage condition, each agent observes a subset of feature columns and sequentially forwards its
tags:
  - ICML 2026
  - Others
  - vertical federated learning
date: 2026-05-08
content_hash: 60175f84562961c0
---
# Networked Information Aggregation for Binary Classification

**Conference**: ICML 2026  
**arXiv**: [2605.01082](https://arxiv.org/abs/2605.01082)  
**Code**: None  
**Area**: Distributed Learning / Network Aggregation / Binary Classification Theory  
**Keywords**: vertical federated learning, logistic regression, DAG sequential learning, Bregman divergence, excess loss lower bound

## TL;DR
This work extends the conclusion of Kearns-Roth-Ryu 2026—which states that linear regression agents on a DAG can approach global optimality by sequentially passing prediction columns—to binary classification. Under the $M$-coverage condition, each agent observes a subset of feature columns and sequentially forwards its logits downstream, achieving global logistic regression optimality with an excess BCE loss of $O(M/\sqrt{D})$. Simultaneously, a hard instance is constructed to prove an $\Omega(k/D)$ lower bound, characterizing network depth as the fundamental bottleneck for information aggregation.

## Background & Motivation

**Background**: Social and network learning have a lineage spanning half a century, including the DeGroot model, Bayesian observational learning, information cascades, Vertical Federated Learning (VFL), and Split Learning. These models address whether decentralized information across different nodes can be aggregated into a globally correct decision. Kearns-Roth-Ryu (2026) provided a clean result under linear regression with squared loss: on a DAG where each agent sees a subset of feature columns and passes local linear prediction columns to downstream neighbors, the excess loss of the final agent is controlled by the network depth $D$ and coverage parameter $M$.

**Limitations of Prior Work**: Classification is more common in practical deployments (e.g., medical diagnosis, fraud detection) than regression. However, the proofs by Kearns et al. heavily rely on "residual orthogonality + Pythagorean variance decomposition" of the squared loss, tools that do not exist under the BCE loss with a sigmoid link. Practical schemes in VFL literature (SecureBoost, Split Learning) rely on multi-round communication to exchange gradients or activations; no one has answered whether "one-shot unidirectional logit passing" is sufficient for aggregation.

**Key Challenge**: The probability space for classification is non-Euclidean—linear combinations of probabilities are not equivalent to linear combinations of features. This is why the authors emphasize passing logits rather than probabilities, but it also means the original geometry no longer applies.

**Goal**: To formally define the "sequential logit passing on DAG" protocol, prove that it achieves global MLE under $M$-coverage, and provide a matching lower bound proving that depth is indeed the bottleneck.

**Key Insight**: Replace the Euclidean geometry of squared loss with the Bregman/KL geometry of BCE. The loss difference is exactly the KL divergence between prediction distributions. Then, use Pinsker's inequality to translate KL progress into prediction error. The technical core is the discovery that the optimal BCE solution still satisfies residual orthogonality, $\mathbb{E}[x(p^*(x) - y)] = 0$ (the first-order necessary condition remains despite the change in geometry).

**Core Idea**: Construct the loss reduction across each segment of the chain into a telescoping sum. Combine this with Pinsker's inequality to translate KL progress $\rightarrow$ squared error $\rightarrow$ control the prediction residuals for all features using the orthogonality of the segment where "feature $x_l$ was observed by agent $j$."

## Method

### Overall Architecture
Each agent $A_i$ on a DAG holds a feature subset $S_i \subseteq [d]$ and learns in topological order. When it is their turn, they receive logits $\{z_j : A_j \in \mathrm{Pa}(A_i)\}$ from all parent nodes, concatenate them with their local features $x_{S_i}$ to train a logistic regression $z_i(x) = w_i^T x_{S_i} + \sum_{j} v_{ij} z_j(x)$ by minimizing BCE, and then pass their logit $z_i$ to successors. The final output is provided by the sink agent (or $A_D$ at the end of the path). Note that logits are passed instead of probabilities to preserve the information geometry of the exponential family, allowing downstream agents to continue linear combinations without loss.

### Key Designs

**1. Residual Orthogonality Lemma + Bregman Loss Decomposition: Rewriting BCE loss difference as KL divergence between distributions to bridge loss reduction and prediction approximation.**

The original Kearns proof relies on the quadratic structure of squared loss where $\|p-q\|^2=$ loss difference. Under BCE, this does not exist. This paper first proves Lemma 3.1: the optimal BCE solution still satisfies residual orthogonality $\mathbb{E}[x(p^*(x) - y)] = 0$, derived from $\nabla_\theta L = 0$. Then, Lemma 3.3 uses the identity $\log \sigma(z) = z - \log(1 + e^z)$ to expand the loss difference. By adding and subtracting $p^*(x)(\theta - \theta^*)^T x$ and applying orthogonality, it arrives at $L(q) = L(p^*) + D(p^* \| q)$, where $D$ is the Bernoulli KL. This is the BCE equivalent of "variance decomposition": the excess loss of any suboptimal predictor is exactly its KL divergence from the optimum, allowing the telescoping argument to proceed.

**2. Path Residual Control (Lemma 3.5): Controlling the discrepancy between the global optimal logit and the current prediction via agents that observed specific features along the coverage path.**

The core is to reduce the "global residual of any feature $x_l$" to the cumulative loss reduction on a segment of the chain. For any linear logit $z_g(x) = \sum \alpha_l x_l$, the triangle inequality decomposes $|\mathbb{E}[(p_k - y) z_g]|$ into correlation terms for each feature $\sum |\alpha_l| |\mathbb{E}[x_l (p_k - y)]|$. For each $x_l$, an agent $A_j$ is found that previously observed it; the orthogonality of that step gives $\mathbb{E}[x_l (p_j - y)] = 0$. Using Cauchy-Schwarz and Pinsker's inequality ($D(p \| q) \geq 2 \mathbb{E}[(p-q)^2]$), the difference $\|p_k - p_j\|_2 \leq \sqrt{k \varepsilon / 2}$ is bounded, thus controlling the discrepancy over a path of length $k$ as $O(\sqrt{k\varepsilon})$. This "telescoping reduction via intermediate agents" converts the networked learning problem into an analysis of cumulative progress.

**3. Pigeonhole Parameter Selection + Global Convergence (Theorem 3.7): Partitioning the path and using the pigeonhole principle to identify a stable block where progress is at least average.**

The path of length $D$ is divided into $K = \lfloor D/M \rfloor$ disjoint blocks. By the pigeonhole principle, there must exist a block where the total loss reduction is $\leq L(p_1) / K \leq 2M L(p_1) / D$. Let this stable block span indices $s..t$. Applying Lemmas 3.4 and 3.5 over it yields $L(p_t) \leq L(p^*) + B_{p^*} B_X \sqrt{M \varepsilon / 2}$. Since $L(p_1) \leq \log 2 < 1$ (achievable by setting $\theta = 0$), the final result is $L(p_D) - L(p^*) \leq B_{p^*} B_X M / \sqrt{D} = O(M / \sqrt{D})$. The beauty of the pigeonhole argument is that it avoids the need for fine-grained control over every segment—stating that "there is always a segment with progress no worse than the average" is sufficient to prove global convergence.

### Loss & Training
The local optimization goal for all agents is standard BCE without regularization or additional structures. The communication protocol involves each agent passing the logit scalar (the input to the sigmoid) rather than the probability after the sigmoid. This is to maintain linear additivity within the exponential family, allowing downstream agents to perform linear regression directly on parent logits without the sigmoid nonlinearity destroying the information geometry.

## Key Experimental Results
This is a purely theoretical paper and does not contain numerical experimental tables. However, the authors provide a conceptual complexity comparison table.

### Main Results

| Method | Task | Loss | Upper Bound | Lower Bound |
|------|------|------|------|------|
| Kearns-Roth-Ryu 2026 | Regression | MSE | $O(M/\sqrt{D})$ | — |
| **Ours** | Binary Classification | BCE | $O(M/\sqrt{D})$ | $\Omega(k/D)$ |

Upper bound conditions: path length $D$, every $M$ consecutive agents collectively cover all features; constants depend on $\mathbb{E}[x_l^2] \leq B_X^2$ and $\|\alpha^*\|_1 \leq B_{p^*}$.

### Ablation Study
Key designs for the lower bound construction (Theorem 4.5):

| Design | Function | Key Lemma |
|------|------|----------|
| Latent variables $Z_i \sim \mathcal{N}(0,1)$ iid, features $x_i = Z_i - Z_{i-1}$ | Ensures $Z_k = \sum x_j$; any prefix of features is independent of the label $y \sim \text{Ber}(\sigma(Z_k))$ | 4.1 (Information correlation recursion) |
| Agents on the path observe one dimension of features in cyclic order $\ell = ((i-1) \mod k) + 1$ | Forces the unlocking of only one effective feature per pass | — |
| The optimal logit after each pass $p$ takes the form $z_D = c(Z_k + \xi/\sqrt{p})$, where $\xi \sim \mathcal{N}(0, V_p)$ | Noise variance can only decay at a rate of $1/p$ | 4.2, 4.3 |
| Optimal $c \in (0,1)$ derived from second-order smoothness of sigmoid; MVT converts probability difference back to logit difference | Results in $L(p_D) - L(p^*) \geq C/(p+1) = \Omega(k/D)$ | 4.4, 4.5 |

### Key Findings
- The upper bound $O(M/\sqrt{D})$ is of the same order as the original Kearns regression result, showing that the $\sqrt{D}$ rate is not unique to squared loss and is also enjoyed by BCE.
- For fixed $M = O(k)$, the lower bound $\Omega(k/D)$ differs from the upper bound $O(k/\sqrt{D})$ only by a $\sqrt{D}$ factor—an open gap acknowledged by the authors.
- Features in the lower bound construction are "difference encoded"; any single $x_i$ is independent of $y$, requiring a sufficiently long chain to decode $Z_k$. This construction suggests that the protocol's inherent "feature-by-feature disentanglement" is the fundamental limit on the rate, rather than loose analysis.
- The paper also discusses why the regression-to-classification gap is non-trivial across compressed sensing, second-order acceleration, and Conformal Prediction, providing a broader context that "BCE is not merely a minor modification of MSE."

## Highlights & Insights
- Using Bregman/KL to replace Euclidean decomposition is a standard approach when moving from regression to classification, but the author's clever use of "orthogonality on a segment of the coverage path" to decompose "global feature residuals" into "local sub-path cumulative KL progress" is a technique that can be reused in other distributed GLM problems.
- Emphasizing logit passing over probability passing is an underrated design principle: the sigmoid maps the exponential family to $(0,1)$, but logits are the natural coordinates when downstream agents want to perform further linear combinations. this experience provides guidance for real-world industrial VFL system design.
- The difference encoding $x_i = Z_i - Z_{i-1}$ in the lower bound is a highly efficient "information bottleneck" instance: it uses only $k$-dimensional Gaussians to irrefutably demonstrate that $k$ passes are required for disentanglement.

## Limitations & Future Work
- There remains a $\sqrt{D}$ gap between the upper and lower bounds, which the authors hope future work will close.
- The protocol is "non-interactive + unidirectional single logit," which is quite restrictive for practical VFL—real-world systems often prefer multi-round exchanges of gradients or activations for better performance. Theoretical results show that if one insists on the weakest communication, $1/\sqrt{D}$ is the best achievable rate, though its engineering value is limited.
- Privacy, noise, and partial alignment in VFL are not considered; this is a pure statistical learning rate analysis.
- The assumptions of bounded second moments for features and bounded $\ell_1$ norm for optimal logit coefficients may not hold for real industrial data.

## Related Work & Insights
- **vs Kearns-Roth-Ryu 2026 (regression)**: Same protocol, but the proof framework is completely rewritten—Bregman instead of Euclidean, KL instead of variance, Pinsker instead of Pythagoras. This is a clean example of a "standard transformation" from regression to classification.
- **vs VFL (SecureBoost, etc.)**: Industrial VFL relies on multi-round interaction + encrypted summation. This paper is single-pass, providing a theoretical guarantee that "worst-case single-pass aggregation is possible," even if accuracy cannot match multi-round schemes.
- **vs Split Learning**: Split Learning uses intermediate network activations for communication, which is theoretically closer to this work (passing intermediate representations downstream). However, this work provides a rigorous convergence rate within linear logistic models, whereas Split Learning lacks clean corresponding results for deep networks.
- Insight: Could the protocol be modified to "pass sufficient statistics instead of logits"—for instance, passing score + Fisher info in GLMs? This might achieve a better trade-off between interaction rounds and depth, representing an open research direction.

## Rating
- Novelty: ⭐⭐⭐⭐ The effort to extend regression to classification is a competent expansion; the lower bound construction is particularly elegant.
- Experimental Thoroughness: ⭐⭐⭐ A pure theory paper with no numerical experiments; the $\sqrt{D}$ gap between bounds was not experimentally verified.
- Writing Quality: ⭐⭐⭐⭐⭐ The chain of lemma dependencies is very clear; Section 1.2 on "why regression-to-classification is non-trivial" is highly insightful.
- Value: ⭐⭐⭐ Primarily aimed at the theory community. Industrial VFL systems will not change architecture due to this, but it establishes a baseline for theoretical analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Structure-Induced Information for Rerooting Levin Tree Search](structure-induced_information_for_rerooting_levin_tree_search.md)
- [\[ICML 2026\] Coupled Training with Privileged Information and Unlabeled Data](coupled_training_with_privileged_information_and_unlabeled_data.md)
- [\[AAAI 2026\] Improved Differentially Private Algorithms for Rank Aggregation](../../AAAI2026/others/improved_differentially_private_algorithms_for_rank_aggregation.md)
- [\[ICML 2025\] Sampling from Binary Quadratic Distributions via Stochastic Localization](../../ICML2025/others/sampling_from_binary_quadratic_distributions_via_stochastic_localization.md)
- [\[ICML 2026\] ParalESN: Enabling Parallel Information Processing in Reservoir Computing](paralesn_enabling_parallel_information_processing_in_reservoir_computing.md)

</div>

<!-- RELATED:END -->
