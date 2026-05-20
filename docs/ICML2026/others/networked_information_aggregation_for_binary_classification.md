---
title: >-
  [Paper Note] Networked Information Aggregation for Binary Classification
description: >-
  [ICML 2026][vertical federated learning] Extends the Kearns-Roth-Ryu 2026 result—"sequentially passing prediction columns among linear regression agents on a DAG nearly achieves global optimum"—to binary classification:…
tags:
  - "ICML 2026"
  - "vertical federated learning"
  - "logistic regression"
  - "DAG sequential learning"
  - "Bregman divergence"
  - "excess loss lower bound"
date: 2026-05-08
content_hash: 8ae43cdacb585812
---

# Networked Information Aggregation for Binary Classification

**Conference**: ICML 2026  
**arXiv**: [2605.01082](https://arxiv.org/abs/2605.01082)  
**Code**: None  
**Area**: Distributed Learning / Network Aggregation / Binary Classification Theory  
**Keywords**: vertical federated learning, logistic regression, DAG sequential learning, Bregman divergence, excess loss lower bound

## TL;DR
Extends the Kearns-Roth-Ryu 2026 result—"sequentially passing prediction columns among linear regression agents on a DAG nearly achieves global optimum"—to binary classification: each agent observes only a subset of feature columns and sequentially forwards its logit to downstream agents. Under the $M$-coverage condition, this achieves global logistic regression optimum with $O(M/\sqrt{D})$ excess BCE loss; a matching hard instance proves an $\Omega(k/D)$ lower bound, characterizing network depth as the fundamental bottleneck for information aggregation.

## Background & Motivation

**Background**: Social learning/networked learning has a half-century lineage—DeGroot model, Bayesian observational learning, information cascades, Vertical Federated Learning (VFL), Split Learning, etc. These models address whether "dispersed partial information at different nodes can be aggregated for globally correct decisions." Kearns-Roth-Ryu (2026) provided a clean result for linear regression with squared loss: on a DAG, each agent observes a subset of features and passes local linear predictions to downstream neighbors; the final agent's excess loss is controlled by network depth $D$ and coverage parameter $M$.

**Limitations of Prior Work**: In practice, classification is more common than regression (e.g., medical diagnosis, fraud detection are binary labels), but Kearns et al.'s proof crucially relies on squared loss's "residual orthogonality + Pythagoras variance decomposition," which do not exist for BCE + sigmoid. Practical VFL schemes (SecureBoost, Split Learning) rely on multi-round communication exchanging gradients/activations; none address whether "one-shot unidirectional logit passing can aggregate."

**Key Challenge**: The probability space for classification is not Euclidean—linear combination in probability does not equal linear combination in features. This is why the authors emphasize passing logits rather than probabilities, but it also means the original geometry does not apply.

**Goal**: To formalize the "sequential logit passing on DAG" protocol; prove that under $M$-coverage it achieves global MLE; and provide matching lower bounds showing depth is indeed the bottleneck.

**Key Insight**: Replace the Euclidean geometry of squared loss with the Bregman/KL geometry of BCE—the loss difference is the KL divergence between predictive distributions, and Pinsker's inequality translates KL progress into prediction error. The technical core is the observation that the BCE optimum still satisfies residual orthogonality $\mathbb{E}[x(p^*(x) - y)] = 0$ (the geometry differs, but the first-order condition remains).

**Core Idea**: Aggregate "loss decrease along each chain segment" into a telescoping sum; use Pinsker to translate KL progress into squared error, and control all feature prediction residuals via the orthogonality at the segment where feature $x_l$ is observed by agent $j$.

## Method

### Overall Architecture
Each agent $A_i$ on the DAG holds a feature subset $S_i \subseteq [d]$ and learns in topological order; when it is its turn, it receives all parent logits $\{z_j : A_j \in \mathrm{Pa}(A_i)\}$, concatenates them with its local features $x_{S_i}$, and trains a logistic regression $z_i(x) = w_i^T x_{S_i} + \sum_{j} v_{ij} z_j(x)$ minimizing BCE, then passes its logit $z_i$ to successors. The final output is given by the sink agent (or terminal $A_D$). Note that logits, not probabilities, are passed: this preserves exponential family information geometry, allowing downstream agents to continue linear combinations without loss.

### Key Designs

1. **Residual Orthogonality Lemma + Bregman Loss Decomposition**:

    - **Function**: Rewrites "BCE loss difference in logistic regression" as expected KL divergence, bridging loss decrease and predictive distribution approximation.
    - **Mechanism**: Lemma 3.1 proves $\mathbb{E}[x(p^*(x) - y)] = 0$ holds for BCE optimum (directly by taking expectation of $\nabla_\theta L = 0$). Lemma 3.3 expands the loss difference using the identity $\log \sigma(z) = z - \log(1 + e^z)$, adds and subtracts $p^*(x)(\theta - \theta^*)^T x$, and uses orthogonality to cancel a term, yielding $L(q) = L(p^*) + D(p^* \| q)$, where $D$ is Bernoulli KL. This is the BCE analogue of squared loss "variance decomposition."
    - **Design Motivation**: The original Kearns proof relies on $\|p - q\|^2 = $ loss diff, which lacks quadratic structure under BCE; Bregman decomposition provides that "any suboptimal predictor's excess loss equals its KL to the optimum," enabling telescoping.

2. **Path Residual Control (Lemma 3.5)**:

    - **Function**: On a length-$k$ coverage path, controls the "difference between global optimal logit and current predictor" to $O(\sqrt{k \varepsilon})$, where $\varepsilon$ is total loss decrease along the path.
    - **Mechanism**: For any linear logit $z_g(x) = \sum \alpha_l x_l$, decomposes $|\mathbb{E}[(p_k - y) z_g]|$ via triangle inequality into $\sum |\alpha_l| |\mathbb{E}[x_l (p_k - y)]|$; for each $x_l$, finds the agent $A_j$ that observed it, whose orthogonality gives $\mathbb{E}[x_l (p_j - y)] = 0$; Cauchy-Schwarz and Pinsker ($D(p \| q) \geq 2 \mathbb{E}[(p-q)^2]$) yield $\|p_k - p_j\|_2 \leq \sqrt{k \varepsilon / 2}$.
    - **Design Motivation**: The core is to reduce "any feature's residual" via "intermediate agent that observed it," telescoping the global information aggregation problem to "cumulative decrease along a chain segment."

3. **Pigeonhole Parameter Selection + Global Convergence (Theorem 3.7)**:

    - **Function**: Divides the length-$D$ path into $K = \lfloor D/M \rfloor$ disjoint blocks; by the pigeonhole principle, some block's total loss decrease $\leq L(p_1) / K \leq 2M L(p_1) / D$.
    - **Mechanism**: For the stable block spanning indices $s..t$, applying Lemmas 3.4 and 3.5 yields $L(p_t) \leq L(p^*) + B_{p^*} B_X \sqrt{M \varepsilon / 2}$; since $L(p_1) \leq \log 2 < 1$ (achievable with $\theta = 0$), finally $L(p_D) - L(p^*) \leq B_{p^*} B_X M / \sqrt{D} = O(M / \sqrt{D})$.
    - **Design Motivation**: The pigeonhole argument avoids the need for fine-grained control over each segment—"there must exist a segment whose progress is no worse than average" suffices.

### Loss & Training
Each agent's local optimization target is standard BCE, with no regularization or extra structure; communication is each agent passing the scalar logit (pre-sigmoid) rather than the probability. This maintains linear additivity in the exponential family: downstream agents can directly perform linear regression on parent logits, avoiding the nonlinearity of sigmoid that would disrupt information geometry.

## Key Experimental Results
This is a purely theoretical paper with no numerical experiment tables. However, the authors present upper and lower bounds in a conceptual "complexity comparison table."

### Main Results

| Method | Task | Loss | Upper Bound | Lower Bound |
|--------|------|------|-------------|-------------|
| Kearns-Roth-Ryu 2026 | Regression | MSE | $O(M/\sqrt{D})$ | — |
| **Ours** | Binary Classification | BCE | $O(M/\sqrt{D})$ | $\Omega(k/D)$ |

Upper bound conditions: path length $D$, every $M$ consecutive agents jointly cover all features; constants depend on $\mathbb{E}[x_l^2] \leq B_X^2$ and $\|\alpha^*\|_1 \leq B_{p^*}$.

### Ablation Study
Key designs in the lower bound construction (Theorem 4.5):

| Design | Function | Key Lemma |
|--------|----------|-----------|
| Latent variables $Z_i \sim \mathcal{N}(0,1)$ iid, features $x_i = Z_i - Z_{i-1}$ | Ensures $Z_k = \sum x_j$, and any prefix of features is independent of label $y \sim \text{Ber}(\sigma(Z_k))$ | 4.1 (information relevance recursion) |
| Agents on the path observe one feature each in cyclic order $\ell = ((i-1) \mod k) + 1$ | Forces only one effective feature to be unlocked per pass | — |
| After $p$ passes, optimal logit is $z_D = c(Z_k + \xi/\sqrt{p})$, $\xi \sim \mathcal{N}(0, V_p)$ | Noise variance can only decay at $1/p$ rate | 4.2, 4.3 |
| Optimal $c \in (0,1)$ comes from sigmoid's second-order smoothness; MVT translates probability gap back to logit gap | Yields $L(p_D) - L(p^*) \geq C/(p+1) = \Omega(k/D)$ | 4.4, 4.5 |

### Key Findings
- The $O(M/\sqrt{D})$ upper bound matches the order in Kearns regression, indicating that the $\sqrt{D}$ rate is not unique to squared loss—BCE also benefits.
- The $\Omega(k/D)$ lower bound, for fixed $M = O(k)$, differs from the upper bound $O(k/\sqrt{D})$ by only a $\sqrt{D}$ factor—an acknowledged open gap.
- In the lower bound construction, features are "differentially encoded," so any single $x_i$ is independent of $y$; only a sufficiently long chain can recover $Z_k$. This shows that the protocol's "feature-by-feature disentanglement" is the essential rate-limiting factor, not a loose analysis.
- The paper also discusses that the regression-to-classification gap is nontrivial in compressed sensing, second-order acceleration, conformal prediction, etc., providing a broader context that "BCE is not a minor tweak of MSE."

## Highlights & Insights
- Replacing Euclidean decomposition with Bregman/KL is standard for extending regression to classification, but the authors' Lemma 3.5 cleverly uses "orthogonality on a coverage path segment" to decompose "global residual of feature $x_l$" into "local subpath cumulative KL progress." This telescoping technique can be reused for other distributed GLM problems.
- Emphasizing logit transmission over probability is an underrated design principle: sigmoid projects the exponential family to $(0,1)$, but for downstream linear combinations, logit is the natural coordinate. This insight is instructive for practical industrial VFL system design.
- The lower bound's differential encoding $x_i = Z_i - Z_{i-1}$ is an economical "information bottleneck" instance: just $k$-dimensional Gaussian suffices to irrefutably demonstrate that "at least $k$ passes are needed to disentangle."

## Limitations & Future Work
- There remains a $\sqrt{D}$ gap between upper and lower bounds, which the authors clearly hope future work will close.
- The protocol is "non-interactive + unidirectional single logit," which is quite restrictive for practical VFL—real systems prefer multi-round gradient or activation exchange for better performance. The theoretical result shows "if you insist on minimal communication, then $\sqrt{D}$ is the best rate achievable," but engineering value is limited.
- Does not consider privacy/noise/partial alignment and other real VFL issues; it is purely a statistical learning rate analysis.
- Assumes bounded second moment of features and bounded $\ell_1$ norm of optimal logit coefficients, which may not hold for real industrial data.

## Related Work & Insights
- **vs Kearns-Roth-Ryu 2026 (regression)**: Same protocol, but proof framework is completely rewritten—Bregman replaces Euclidean, KL replaces variance, Pinsker replaces Pythagoras; this is a clean example of the "standard reduction" from regression to classification.
- **vs VFL (e.g., SecureBoost)**: Industrial VFL relies on multi-round interaction + encrypted summation; this work is single-pass, providing a theoretical guarantee that "even in the worst case, single-pass can aggregate," but empirical accuracy cannot match multi-round schemes.
- **vs Split Learning**: Split learning communicates intermediate activations, theoretically closer to this work (also passing intermediate representations downstream), but this work gives convergence rates strictly for linear logistic models, whereas split learning in deep networks lacks such clean results.
- **Inspiration**: Could the protocol be modified to "pass sufficient statistics instead of logits"—e.g., in GLMs, pass score + Fisher information? This might enable better trade-offs between interaction rounds and depth; this is an open direction.

## Rating
- Novelty: ⭐⭐⭐⭐ Extending regression to classification is a solid contribution; the lower bound construction is especially elegant
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical, no numerical experiments; the $\sqrt{D}$ gap between bounds is not empirically tested for tightness
- Writing Quality: ⭐⭐⭐⭐⭐ Lemma dependency chain is very clear; Section 1.2's discussion of "why regression-to-classification is nontrivial" is insightful
- Value: ⭐⭐⭐ Mainly for the theory community; industrial VFL systems are unlikely to change architecture, but this sets a baseline for theoretical analysis

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Principled Algorithms for Optimizing Generalized Metrics in Binary Classification](../../ICML2025/others/principled_algorithms_for_optimizing_generalized_metrics_in_binary_classificatio.md)
- [\[AAAI 2026\] Improved Differentially Private Algorithms for Rank Aggregation](../../AAAI2026/others/improved_differentially_private_algorithms_for_rank_aggregation.md)
- [\[ICLR 2026\] On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets](../../ICLR2026/others/on_the_lipschitz_continuity_of_set_aggregation_functions_and_neural_networks_for.md)
- [\[AAAI 2026\] LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval](../../AAAI2026/others/leanrag_knowledge-graph-based_generation_with_semantic_aggregation_and_hierarchi.md)
- [\[AAAI 2026\] Tractable Weighted First-Order Model Counting with Bounded Treewidth Binary Evidence](../../AAAI2026/others/tractable_weighted_first-order_model_counting_with_bounded_treewidth_binary_evid.md)

</div>

<!-- RELATED:END -->
