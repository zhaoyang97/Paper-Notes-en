---
title: >-
  [Paper Note] Unifying Formal Explanations: A Complexity-Theoretic Perspective
description: >-
  [ICLR 2026][Optimization & Theory][Paper Note] Proposes a unified framework reducing sufficient and contrastive reasons (local/global, probabilistic/non-probabilistic) to a minimization problem of a unified probability value function. It reveals that global value functions possess key combinatorial optimization properties like monotonicity, submodularity, and super
tags:
  - ICLR 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: 3891093c9c7895bf
---
# Unifying Formal Explanations: A Complexity-Theoretic Perspective

**Conference**: ICLR 2026  
**arXiv**: [2602.18160](https://arxiv.org/abs/2602.18160)  
**Area**: Optimization  
**Keywords**: Explainable AI, Computational Complexity, Sufficient Reasons, Contrastive Reasons, Submodular/Supermodular Functions

## TL;DR

Proposes a unified framework reducing sufficient and contrastive reasons (local/global, probabilistic/non-probabilistic) to a minimization problem of a unified probability value function. It reveals that global value functions possess key combinatorial optimization properties like monotonicity, submodularity, and supermodularity, proving that global explanations are computable in polynomial time—even when the corresponding local explanations are NP-hard.

## Background & Motivation

### Two Fundamental Forms of Explainability

**Sufficient Reasons**: Prediction remains invariant after fixing the values of a feature subset $S$ — answering "What is sufficient to support this prediction?"

**Contrastive Reasons**: Prediction changes after modifying the values of a feature subset $S$ — answering "What changes would flip the prediction?"

Smaller reasons provide greater explanatory value; thus, **minimality** is the core objective.

### From Local to Global, Deterministic to Probabilistic

Explanation problems in the literature extend along two dimensions:

- **Local vs. Global**: Targeting a single prediction vs. targeting model behavior across the entire input space.
- **Non-probabilistic vs. Probabilistic**: Absolute guarantees vs. guarantees with probability $\delta$.

**Existing Hardness Results**:

| Setting | Sufficient Reasons | Contrastive Reasons |
|------|----------|----------|
| Decision Trees (Local Non-probabilistic) | NP-hard | P |
| Neural Networks (Local Non-probabilistic) | $\Sigma_2^P$-hard | NP-hard |
| Decision Trees (Local Probabilistic) | NP-hard | NP-hard |

These discouraging complexity results lead to the core question: **Does there exist a form of explanation that remains efficiently computable while maintaining meaningful guarantees?**

## Method

### Overall Architecture

The paper reduces all "explanation-finding" tasks to the same combinatorial optimization problem: finding the smallest feature subset $S$ that satisfies a quality threshold under a value function $v$, i.e., $\min |S|$ s.t. $v(S) \geq \delta$. The differences across the four dimensions—sufficient/contrastive, local/global, and probabilistic/non-probabilistic—are absorbed into the specific functional form of $v$. Once $v$ is defined, the problem difficulty is determined solely by its combinatorial structure (monotonicity, submodularity/supermodularity). The core discovery is that switching from "local" to "global" transforms $v$ from unstructured to highly regular, bringing NP-hard explanation problems back into polynomial time.

### Key Designs

**1. Unified Value Function: Four Classes of Explanations in One Template**

The fundamental difference between explanation types lies in "which features are fixed and what probability is queried." Fixing values in $S$ to check prediction maintenance yields sufficient reasons; fixing values in the complement $\bar S$ to check prediction flipping yields contrastive reasons. Replacing a single point $\boldsymbol{x}$ with the expectation over the data distribution upgrades the explanation from local to global. The four functions are: Local Sufficient $v_{\text{suff}}^\ell(S) = \Pr_{\boldsymbol{z} \sim \mathcal{D}}(f(\boldsymbol{z}) = f(\boldsymbol{x}) \mid \boldsymbol{z}_S = \boldsymbol{x}_S)$, Global Sufficient $v_{\text{suff}}^g(S) = \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}[v_{\text{suff}}^\ell(S)]$; Local Contrastive $v_{\text{con}}^\ell(S) = \Pr_{\boldsymbol{z}}(f(\boldsymbol{z}) \neq f(\boldsymbol{x}) \mid \boldsymbol{z}_{\bar S} = \boldsymbol{x}_{\bar S})$, and Global Contrastive $v_{\text{con}}^g(S) = \mathbb{E}_{\boldsymbol{x}}[v_{\text{con}}^\ell(S)]$. This template allows all complexity conclusions to be derived from the properties of $v$.

**2. Three Combinatorial Properties: Trading Structure for Algorithms**

Problem tractability depends on three properties. Monotonicity $v(S \cup \{i\}) \geq v(S)$ implies adding features never decreases the value, allowing "greedy pruning" to converge to a subset-minimal solution. Supermodularity $v(S \cup \{i\}) - v(S) \leq v(S' \cup \{i\}) - v(S')$ for $S \subseteq S'$ indicates increasing marginal contributions, while submodularity indicates decreasing marginals. Submodularity is the prerequisite for greedy algorithms (Wolsey 1982) to achieve logarithmic approximation guarantees. This translates abstract explanation problems into the maturity of combinatorial optimization tools.

**3. Local-to-Global "Phase Transition": Expectation Smooths Irregularities**

Checking these properties reveals a counter-intuitive result: local value functions fail all properties in almost all settings, whereas global value functions (the expectation over the distribution) become monotonic. Furthermore, global sufficient reasons become supermodular and contrastive reasons become submodular.

| Property | Local Suff $v_{\text{suff}}^\ell$ | Global Suff $v_{\text{suff}}^g$ | Local Con $v_{\text{con}}^\ell$ | Global Con $v_{\text{con}}^g$ |
|------|:---:|:---:|:---:|:---:|
| Monotonicity | ✗ | ✓ | ✗ | ✓ |
| Supermodularity | ✗ | ✓ (indep dist) | ✗ | ✗ |
| Submodularity | ✗ | ✗ | ✗ | ✓ (indep dist) |

This occurs because expectation operations have a smoothing effect; while adding a feature for a single $\boldsymbol{x}$ can cause drastic probability jumps, taking the expectation over the distribution smooths these jumps, allowing the underlying regular structure to emerge.

**4. Two Greedy Algorithms: Executing Properties into Solutions**

For subset-minimal explanations, a top-down Algorithm 1 is used: starting from the full set, it iteratively removes the feature $j$ with the highest remaining value $j = \arg\max_{i \in S} v(S \setminus \{i\})$. Monotonicity ensures exact convergence in global settings. For cardinality-minimal explanations, a bottom-up Algorithm 2 is used: starting from an empty set, it iteratively adds the feature with the highest gain $j = \arg\max_{i \notin S} v(S \cup \{i\})$. Submodularity/supermodularity provide approximation bounds—$O(\ln|D|)$ for global contrastive and a constant factor for global sufficient using bounded curvature techniques (Shi et al. 2021).

## Key Experimental Results

This is a theoretical work; the core contributions are complexity analysis results.

### Summary of Complexity Results

| Explanation Type | Local (Arbitrary Model) | Global (Empirical Distribution) |
|----------|:---:|:---:|
| Subset-minimal Sufficient | NP-hard (even trees) | **P** |
| Subset-minimal Contrastive | NP-hard (NN/Ensembles) | **P** |
| Approx Cardinality-minimal Suff | No bounded approx | $O\left(\frac{1}{1-k^f} + \ln\frac{v([n])}{\min_i v(\{i\})}\right)$ |
| Approx Cardinality-minimal Con | No bounded approx | $O\left(\ln\frac{v([n])}{\min_i v(\{i\})}\right)$ |

### Approximation Guarantee Comparison

| Setting | Global | Local |
|------|------|------|
| Subset-minimal | Poly-time Exact | NP-hard (even trees + uniform) |
| Cardinality-min (Con) | $O(\ln |D|)$ Approx | No bounded approx (even $|D|=1$) |
| Cardinality-min (Suff) | Constant Factor Approx | No bounded approx (even $|D|=1$) |

### Key Findings

1. While non-probabilistic global sufficient reasons are unique (Bassan et al. 2024), there can be exponentially many subset-minimal global sufficient reasons in probabilistic settings: $\Theta(2^n/\sqrt{n})$.
2. The submodularity of global contrastive reasons allows the direct use of classic Wolsey (1982) greedy guarantees.
3. The supermodularity of global sufficient reasons requires bounded curvature techniques from Shi et al. (2021).
4. Approximation guarantees under empirical distributions grow only logarithmically with the dataset size $|D|$.

## Highlights & Insights

1. **Elegant Unified Framework**: Consolidates four dimensions (Sufficient/Contrastive × Local/Global × Probabilistic/Non-probabilistic × Subset/Cardinality Minimal) into a single value function minimization task.
2. **Local-Global "Phase Transition"**: Local functions lack structure (non-monotonic, non-submodular), whereas global functions possess all useful properties due to the smoothing effect of expectations.
3. **Sufficient-Contrastive Duality**: The supermodular vs. submodular nature reflects the symmetry between "fixing features to maintain predictions" and "varying features to change them."
4. **Practical Significance**: Efficient computation of global explanations implies that "which features the model depends on overall" is an easier question to answer than "which features this specific prediction depends on."
5. **Broad Theoretical Scope**: Results apply to decision trees, neural networks, and tree ensembles, covering the spectrum from "interpretable" to "black-box" models.

## Limitations & Future Work

1. Purely theoretical; lacks empirical validation regarding the actual runtime efficiency and explanation quality of the greedy algorithms.
2. Submodular/supermodular properties hold strictly under feature independence assumptions, which are often violated in reality.
3. The dataset size $|D|$ in the approximation guarantees may be practically very large.
4. The semantics of global explanations may not align perfectly with users' interests in specific local predictions.
5. The complexity of computing the value function itself is not discussed—for neural networks, even a single inference pass can be costly.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The unified framework and the discovery of the local/global property contrast are highly original.
- **Experimental Thoroughness**: ⭐⭐ — Theoretical work with no empirical experiments.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Rigorous definitions, well-structured theorems, and clear proof sketches.
- **Value**: ⭐⭐⭐⭐⭐ — Establishes a fundamental complexity-theoretic foundation for XAI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Directional Sheaf Hypergraph Networks: Unifying Learning on Directed and Undirected Hypergraphs](directional_sheaf_hypergraph_networks_unifying_learning_on_directed_and_undirect.md)
- [\[ICLR 2026\] Neural Networks Learn Generic Multi-Index Models Near Information-Theoretic Limit](neural_networks_learn_generic_multi-index_models_near_information-theoretic_limi.md)
- [\[ICML 2026\] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective](../../ICML2026/optimization/learning_dynamics_of_zeroth-order_optimization_a_kernel_perspective.md)
- [\[ICML 2025\] Learning Mixtures of Experts with EM: A Mirror Descent Perspective](../../ICML2025/optimization/learning_mixtures_of_experts_with_em_a_mirror_descent_perspective.md)
- [\[ICML 2026\] Rethinking the Flow-Based Gradual Domain Adaptation: A Semi-Dual Optimal Transport Perspective](../../ICML2026/optimization/rethinking_the_flow-based_gradual_domain_adaptation_a_semi-dual_optimal_transpor.md)

</div>

<!-- RELATED:END -->
