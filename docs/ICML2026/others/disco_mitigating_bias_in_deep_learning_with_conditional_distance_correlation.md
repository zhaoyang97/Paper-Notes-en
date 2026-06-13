---
title: >-
  [Paper Note] DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation
description: >-
  [ICML2026][Anti-causal models] This paper unifies confounding, collider, and mediator biases into a single conditional independence criterion $\hat{Y} \perp \mathbf{B} \mid Y$ using an anti-causal graph. It introduces sD…
tags:
  - "ICML2026"
  - "Anti-causal models"
  - "Conditional distance correlation"
  - "Shortcut learning"
  - "Causal stability"
  - "Single-step differentiable estimation"
date: 2026-05-08
content_hash: b07c37201439e52c
---

# DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation

**Conference**: ICML2026  
**arXiv**: [2506.11653](https://arxiv.org/abs/2506.11653)  
**Code**: https://github.com/yakamoz5/DISCO  
**Area**: AI Safety / Fairness and Bias Mitigation / Causal Representation Learning  
**Keywords**: Anti-causal models, Conditional distance correlation, Shortcut learning, Causal stability, Single-step differentiable estimation

## TL;DR
This paper unifies confounding, collider, and mediator biases into a single conditional independence criterion $\hat{Y} \perp \mathbf{B} \mid Y$ using an anti-causal graph. It introduces sDISCO, a single-step differentiable estimator with $O(n^2)$ memory complexity, designed as a regularizer to penalize conditional distance correlation in any gradient-trained network, thereby mitigating various biases and scaling efficiently to multi-bias scenarios.

## Background & Motivation
**Background**: Dataset biases cause deep models to learn task-irrelevant shortcuts rather than true signals. Typical examples include age becoming a common cause for Alzheimer's prediction in medical imaging, background correlations with waterbirds in CV, and negation words as pseudo-related to entailment labels in NLP. Current mitigation methods usually target specific bias structures (e.g., confounder-only) or use empirical independence regularization (IRM, GDRO, Fishr, C-MMD), but they often lack a unified causal foundation.

**Limitations of Prior Work**: (1) Different bias types (confounder/collider/mediator) are typically handled separately using incompatible methods; (2) Existing conditional independence regularizers, such as conditional MMD, lack full support for combinations of bias and target types (binary/categorical/continuous); (3) Stronger nonlinear independence criteria like conditional distance correlation (Wang et al. 2015) require $O(n^3)$ memory in their V-statistic implementation, causing OOM errors in deep learning batches; (4) Most methods scale poorly to multi-bias scenarios.

**Key Challenge**: The "causal stability" of a model should be characterized by the observable conditional independence criterion $\hat{Y} \perp \mathbf{B} \mid Y$. However, non-parametric conditional independence measures with high expressive power are computationally expensive, forcing a trade-off between weak criteria (linear covariance) or imprecise/unscalable approximations.

**Goal**: (i) Provide a unified causal explanation for why confounder, collider, and mediator biases can be addressed via the same conditional independence criterion; (ii) Develop a differentiable estimator for conditional distance correlation that is compatible with any target-bias type combination; (iii) Reduce memory complexity from $O(n^3)$ to $O(n^2)$ while supporting exact global estimation.

**Key Insight**: Borrowing from the path-specific fairness analysis framework (Plecko & Bareinboim), the authors abstract anti-causal prediction $Y \to X$ into a Standard Anti-Causal Model (SAM). Using counterfactual decomposition (TV = ctf-stable − ctf-IE − ctf-SE), they prove that $\hat{Y} \perp \mathbf{B} \mid Y$ simultaneously nullifies ctf-IE and ctf-SE. They further expand the V-statistic of Wang et al. (2015) into a batch-computable Hadamard form to avoid explicit $n \times n \times n$ tensors.

**Core Idea**: Transform the problem of bias type identification into a single conditional independence constraint and implement an $O(n^2)$ memory-efficient matrix decomposition for conditional distance correlation estimation, allowing it to be integrated as a regularizer into any loss function.

## Method

### Overall Architecture
The approach consists of three layers. The top layer is **theory**: establishing the SAM causal graph (target $Y$ → input $X$ → prediction $\hat{Y}$, with auxiliary variables $\mathbf{Z}$ and mediators $\mathbf{W}$ collectively termed bias $\mathbf{B}$) and proving that $\hat{Y} \perp \mathbf{B} \mid Y$ implies causal stability. The middle layer is **estimation**: using conditional distance covariance $\mathrm{dCov}^2(X, Y \mid Z) = \mathbb{E}_Z[\mathrm{dCov}^2(X, Y \mid Z = z)]$, which equates to conditional independence in strong negative-type metric spaces. Two estimators are proposed: DISCO$_m$ (memory-efficient via $m$ reference points) and sDISCO (exact global estimation via algebraic decomposition in $O(n^2)$). The bottom layer is **training**: adding sDISCO as a regularizer to the ERM loss: $\min_\theta \sum L(Y, \hat{Y}) + \lambda \cdot \mathrm{sDISCO}(\hat{Y}, \mathbf{B} \mid Y)$.

### Key Designs

1.  **SAM Anti-Causal Model + Unified Criterion**:
    *   **Function**: Unifies confounder (fork), collider (latently conditioned), and mediator (stable/unstable) structures into one graph, proving causal stability is equivalent to $\hat{Y} \perp \mathbf{B} \mid Y$ ($\mathbf{B} = \mathbf{W} \cup \mathbf{Z}$).
    *   **Mechanism**: Defines counterfactual stable effects (ctf-stable via $Y \to X \to \hat{Y}$), counterfactual indirect effects (ctf-IE via $Y \to \mathbf{W} \to \hat{Y}$), and counterfactual spurious effects (ctf-SE via $Y$—$\mathbf{Z}$—$\hat{Y}$). Theorem 2.3 proves that if $\hat{Y} \perp \mathbf{W}, \mathbf{Z} \mid Y$, then ctf-IE and ctf-SE vanish.
    *   **Design Motivation**: Existing methods often fragment by bias type. Path-specific analysis reveals that the specific bias structure is irrelevant—as long as a ctf-stable path exists and biases are observed, a single constraint suffices.

2.  **Conditional Distance Correlation as Nonlinear Measure**:
    *   **Function**: Leverages the property that $\mathrm{dCov}^2(X, Y \mid Z) = 0$ if and only if $X \perp Y \mid Z$ in strong negative-type metric spaces (including Euclidean space) to provide a regularizer that captures arbitrary nonlinear and high-dimensional dependencies.
    *   **Mechanism**: For samples $\{(X_i, Y_i, Z_i)\}$, it calculates conditional probability proxy weights $w_{ij} = K_h(Z_i, Z_j) / \sum_k K_h(Z_i, Z_k)$ using an RBF kernel $K_h$, followed by pairwise distance matrices.
    *   **Design Motivation**: Unlike linear conditional covariance or type-restricted C-MMD, distance correlation is valid for any combination of $X, Y, Z$ types and does not require modeling the conditional distribution explicitly.

3.  **sDISCO Algebraic Decomposition: $O(n^3) \to O(n^2)$ Exact Estimation**:
    *   **Function**: Rewrites the global V-statistic (naive implementation requires $(n, n, n)$ tensors) into three terms $T_1, T_2, T_3$ using Hadamard products and matrix multiplications, allowing exact global estimation within $O(n^2)$ memory.
    *   **Mechanism**: Uses the property that weighted marginal sums of centered matrices are zero. By defining $M^X = WA$ and $g^X = (W \circ M^X) \mathbf{1}$, the local covariance for $n$ reference points is calculated as $\mathcal{V}_{XY} = T_1 + T_2 - 2T_3$, where $T_1 = (W \circ (W(A \circ B))) \mathbf{1}$.
    *   **Design Motivation**: Avoids the accuracy-memory trade-off and hyperparameter tuning required by DISCO$_m$ (sampling $m$ points) by processing the entire batch precisely with only two parameters (bandwidth and $\lambda$).

### Loss & Training
The combined objective is $\min_\theta \sum L(Y, \hat{Y}) + \lambda \cdot \mathrm{sDISCO}(\hat{Y}, \mathbf{B} \mid Y)$, where $L$ is the standard MLE loss (MSE/CE). Inference relies solely on $X$; bias $\mathbf{B}$ is only required during training for the regularization term.

## Key Experimental Results

### Main Results
Evaluation across six datasets (regression/classification, synthetic/real, vision/NLP) using OOD/unbiased test sets.

| Dataset | Task/Bias Type | Key Metric | DISCO$_m$ / sDISCO | Prev. SOTA |
| :--- | :--- | :--- | :--- | :--- |
| dSprites | y-pos reg, X-pos confounding | OOD MSE ↓ | **Lowest** | Comparable to IRM/Fishr |
| Blob | Causal intensity reg, mediator bias | OOD MSE ↓ | **Lowest** | C-MMD/GDRO higher |
| YaleB | Pose class, lighting bias | OOD acc ↑ | **Lead** | Adversarial baselines |
| FairFace | Gender class, skin tone bias | Worst-group acc ↑ | **Competitive** | GDRO strong baseline |
| Waterbirds | Bird class, background spurious | Worst-group acc ↑ | **Competitive** | GDRO/JTT strong baseline |
| MNLI | Entailment class, negation bias | Worst-group acc ↑ | **Competitive** | GDRO strong baseline |

Across all datasets, DISCO variants achieved SOTA or comparable performance while requiring significantly less hyperparameter tuning than GDRO or IRM.

### Ablation Study
| Configuration | Key Property | Result |
| :--- | :--- | :--- |
| Full sDISCO | Global exact + $O(n^2)$ | Seamless extension to multi-bias without overhead. |
| DISCO$_m$ | Approximation | Close to sDISCO at small batches, but $m$ requires tuning. |
| Linear Covariance | Linear only | Significant performance drop on nonlinear biases (dSprites/Blob). |
| C-MMD | Restricted types | Degrades on mixed continuous/categorical inputs. |
| No Regularizer (ERM) | Shortcut learning | Significant OOD degradation (control group). |

### Key Findings
*   sDISCO scales seamlessly to multi-bias scenarios (e.g., skin tone + age in FairFace) by concatenating them into the distance matrix.
*   Counterfactual analysis on controlled simulations confirms that DISCO-trained models make decisions primarily through the ctf-stable path (ctf-IE and ctf-SE near zero).
*   Computation: sDISCO increases wall-time by ~1.5–2× per batch but maintains a strict $O(n^2)$ memory footprint, making it viable for standard batch sizes (128–512).

## Highlights & Insights
*   Unifying bias types into a single criterion via SAM provides a cleaner abstraction than enumerating specific algorithms for each bias type.
*   The $O(n^2)$ algebraic decomposition of high-order distance statistics is a reusable engineering contribution applicable to any non-parametric method using third-order statistics.
*   The use of counterfactual decomposition (ctf-stable/IE/SE) as an evaluation tool rather than just a motivation allows for verifiable and falsifiable explanations of model performance.

## Limitations & Future Work
*   The criterion $\hat{Y} \perp \mathbf{B} \mid Y$ depends on the "positivity" assumption; if $\mathbf{B}$ is a deterministic function of $Y$, observational debiasing fails.
*   Unobserved confounding: The method only masks paths involving observed $\mathbf{B}$. It cannot handle biases leaking through unobserved variables $\mathbf{Z}'$.
*   Distinguishing mediators: The approach treats mediators as shortcuts by default. In practice, identifying which mediators are "stable" remains a domain-specific modeling choice.
*   Bandwidth sensitivity: $\sigma_Y$ in sDISCO can be sensitive, necessitating heuristics like the median heuristic during engineering.

## Related Work & Insights
*   **vs. Veitch et al. 2021**: While Veitch proves counterfactual invariance under stricter conditions, this work generalizes the conclusion to the SAM framework, covering more bias structures.
*   **vs. IRM / GDRO**: These depend on environment/group labels and often require extensive hyperparameter search; DISCO uses continuous/discrete bias variables directly with fewer parameters.
*   **vs. C-MMD**: sDISCO is superior in handling mixed data types and providing a single-step exact implementation.

## Rating
*   Novelty: ⭐⭐⭐⭐ (SAM framework and sDISCO decomposition are highly innovative).
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Broad dataset coverage and rigorous counterfactual analysis).
*   Writing Quality: ⭐⭐⭐⭐ (Formal causal notation is clear, though high entry barrier for non-causal experts).
*   Value: ⭐⭐⭐⭐⭐ (Solid theoretical foundation combined with a practical, memory-efficient estimator).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets](../../ICLR2026/others/mitigating_spurious_correlation_via_distributionally_robust_learning_with_hierar.md)
- [\[ICML 2026\] Possibilistic Predictive Uncertainty for Deep Learning](possibilistic_predictive_uncertainty_for_deep_learning.md)
- [\[ICML 2026\] Sequential Group Composition: A Window into the Mechanics of Deep Learning](sequential_group_composition_a_window_into_the_mechanics_of_deep_learning.md)
- [\[AAAI 2026\] How Wide and How Deep? Mitigating Over-Squashing of GNNs via Channel Capacity Constrained Estimation](../../AAAI2026/others/how_wide_and_how_deep_mitigating_over-squashing_of_gnns_via_channel_capacity_con.md)
- [\[ICML 2026\] Riemannian Networks over Full-Rank Correlation Matrices](riemannian_networks_over_full-rank_correlation_matrices.md)

</div>

<!-- RELATED:END -->
