---
title: >-
  [Paper Note] Unifying Formal Explanations: A Complexity-Theoretic Perspective
description: >-
  [ICLR 2026][Optimization][Explainable AI] This paper proposes a unified framework that reduces sufficient reasons and contrastive reasons (local/global…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Explainable AI"
  - "Computational Complexity"
  - "Sufficient Reasons"
  - "Contrastive Reasons"
  - "Submodular/Supermodular Functions"
date: 2026-05-08
content_hash: eee097af87f9f6d5
---

# Unifying Formal Explanations: A Complexity-Theoretic Perspective

**Conference**: ICLR 2026
**arXiv**: [2602.18160](https://arxiv.org/abs/2602.18160)  
**Area**: Optimization
**Keywords**: Explainable AI, Computational Complexity, Sufficient Reasons, Contrastive Reasons, Submodular/Supermodular Functions

## TL;DR

This paper proposes a unified framework that reduces sufficient reasons and contrastive reasons (local/global, probabilistic/non-probabilistic) to the problem of minimizing a unified probabilistic value function. It reveals that global value functions possess key combinatorial optimization properties—monotonicity, submodularity/supermodularity—and proves that global explanations are computable in polynomial time, even when their local counterparts are NP-hard.

## Background & Motivation

### Two Fundamental Forms of Explanations

**Sufficient Reasons**: Fixing the values of a feature subset $S$ leaves the model prediction unchanged—answering "what is sufficient to support this prediction?"

**Contrastive Reasons**: Modifying the values of a feature subset $S$ changes the model prediction—answering "what change would flip the prediction?"

Smaller reasons are more interpretable; thus **minimality** is a central objective.

### From Local to Global, From Deterministic to Probabilistic

Explanation problems in the literature extend along two dimensions:

- **Local vs. Global**: Targeting a single prediction vs. the model's behavior over the entire input space
- **Non-probabilistic vs. Probabilistic**: Absolute guarantees vs. guarantees with probability $\delta$

**Known Hardness Results**:

| Setting | Sufficient Reasons | Contrastive Reasons |
|---|---|---|
| Decision Trees (local, non-probabilistic) | NP-hard | P |
| Neural Networks (local, non-probabilistic) | $\Sigma_2^P$-hard | NP-hard |
| Decision Trees (local, probabilistic) | NP-hard | NP-hard |

These discouraging complexity results raise the core question: **Does there exist a form of explanation that admits efficient computation while preserving meaningful guarantees?**

## Method

### Unified Framework

All explanation problems are unified as:

$$\text{find the minimal subset } S \subseteq [n] \text{ such that } v(S) \geq \delta$$

where $v$ is the value function under different settings:

- **Local sufficient value function**: $v_{\text{suff}}^\ell(S) = \Pr_{\boldsymbol{z} \sim \mathcal{D}}(f(\boldsymbol{z}) = f(\boldsymbol{x}) \mid \boldsymbol{z}_S = \boldsymbol{x}_S)$
- **Global sufficient value function**: $v_{\text{suff}}^g(S) = \mathbb{E}_{\boldsymbol{x} \sim \mathcal{D}}[\Pr_{\boldsymbol{z}}(f(\boldsymbol{z}) = f(\boldsymbol{x}) \mid \boldsymbol{z}_S = \boldsymbol{x}_S)]$
- **Local contrastive value function**: $v_{\text{con}}^\ell(S) = \Pr_{\boldsymbol{z} \sim \mathcal{D}}(f(\boldsymbol{z}) \neq f(\boldsymbol{x}) \mid \boldsymbol{z}_{\bar{S}} = \boldsymbol{x}_{\bar{S}})$
- **Global contrastive value function**: $v_{\text{con}}^g(S) = \mathbb{E}_{\boldsymbol{x}}[\Pr_{\boldsymbol{z}}(f(\boldsymbol{z}) \neq f(\boldsymbol{x}) \mid \boldsymbol{z}_{\bar{S}} = \boldsymbol{x}_{\bar{S}})]$

### Three Key Properties

**1. Monotonicity**: $v(S \cup \{i\}) \geq v(S)$ (adding a feature does not decrease the value function)

**2. Supermodularity**: $v(S \cup \{i\}) - v(S) \leq v(S' \cup \{i\}) - v(S')$ (increasing marginal contributions)

**3. Submodularity**: $v(S \cup \{i\}) - v(S) \geq v(S' \cup \{i\}) - v(S')$ (diminishing marginal contributions)

### Core Finding: A Striking Contrast Between Local and Global

| Property | Local Sufficient $v_{\text{suff}}^\ell$ | Global Sufficient $v_{\text{suff}}^g$ | Local Contrastive $v_{\text{con}}^\ell$ | Global Contrastive $v_{\text{con}}^g$ |
|---|:---:|:---:|:---:|:---:|
| Monotonicity | ✗ | ✓ | ✗ | ✓ |
| Supermodularity | ✗ | ✓ (independent distribution) | ✗ | ✗ |
| Submodularity | ✗ | ✗ | ✗ | ✓ (independent distribution) |

These findings are **counter-intuitive**:

- Local value functions satisfy none of the three properties under any setting
- Global value functions satisfy monotonicity in all settings
- The global value function for sufficient reasons is supermodular, while that for contrastive reasons is submodular—the two are precisely "dual" to each other

### Greedy Algorithm Design

**Algorithm 1 (Subset-Minimal Explanation)**: Top-down, starting from the full feature set, iteratively removing the feature whose removal minimally decreases the value function:

$$j = \arg\max_{i \in S} v(S \setminus \{i\}), \quad S \leftarrow S \setminus \{j\}$$

Monotonicity guarantees convergence to a subset-minimal explanation in the global setting, but provides no such guarantee locally.

**Algorithm 2 (Cardinality-Minimal Explanation Approximation)**: Bottom-up, starting from the empty set, iteratively adding the feature that maximally increases the value function:

$$j = \arg\max_{i \notin S} v(S \cup \{i\}), \quad S \leftarrow S \cup \{j\}$$

## Key Experimental Results

This paper is a purely theoretical contribution; its core results concern complexity-theoretic analysis.

### Summary of Complexity Results

| Explanation Type | Local (arbitrary model) | Global (empirical distribution) |
|---|:---:|:---:|
| Subset-minimal sufficient reasons | NP-hard (even for decision trees) | **P** |
| Subset-minimal contrastive reasons | NP-hard (neural networks/tree ensembles) | **P** |
| Cardinality-minimal sufficient reason approximation | No bounded approximation | $O\left(\frac{1}{1-k^f} + \ln\frac{v([n])}{\min_i v(\{i\})}\right)$ |
| Cardinality-minimal contrastive reason approximation | No bounded approximation | $O\left(\ln\frac{v([n])}{\min_i v(\{i\})}\right)$ |

### Comparison of Approximation Guarantees

| Setting | Global | Local |
|---|---|---|
| Subset-minimal | Exact solution in polynomial time | NP-hard (even for decision trees + uniform distribution) |
| Cardinality-minimal (contrastive) | $O(\ln |D|)$ approximation | No bounded approximation (even for $|D|=1$) |
| Cardinality-minimal (sufficient) | Constant-factor approximation | No bounded approximation (even for $|D|=1$) |

### Key Theoretical Findings

1. Although non-probabilistic global sufficient reasons are unique (Bassan et al. 2024), in the probabilistic setting there may exist exponentially many subset-minimal global sufficient reasons: $\Theta(2^n/\sqrt{n})$
2. The submodularity of global contrastive reasons directly admits the classical greedy guarantee of Wolsey (1982)
3. The supermodularity of global sufficient reasons requires the bounded-curvature technique of Shi et al. (2021)
4. Approximation guarantees under the empirical distribution grow only logarithmically in the dataset size $|D|$

## Highlights & Insights

1. **The unified framework is remarkably elegant**: It subsumes four major dimensions (sufficient/contrastive × local/global × probabilistic/non-probabilistic × subset/cardinality-minimal) into a single value function minimization task.
2. **A "phase transition" between local and global**: Local value functions are entirely unstructured (non-monotone, neither submodular nor supermodular), whereas global value functions possess all useful properties—this is not coincidental, but rather a smoothing effect of the expectation operator.
3. **Duality between sufficient and contrastive reasons**: One is supermodular and the other submodular, reflecting the symmetric relationship between "fixing features to maintain a prediction" and "varying features to change a prediction."
4. **Far-reaching practical implications**: The tractability of global explanations implies that the question "which features does this model rely on overall?" is easier to answer than "which features does this prediction rely on?"
5. **Theoretical contributions span the explainability spectrum**: Results apply to decision trees, neural networks, and tree ensembles, covering models from "interpretable" to "black-box."

## Limitations & Future Work

1. As a purely theoretical work, empirical validation is absent—the practical runtime efficiency and explanation quality of the greedy algorithms remain unknown.
2. The submodular/supermodular properties hold only under feature independence assumptions, whereas real-world features are often highly correlated.
3. The dataset size $|D|$ appears as an implicit constant in the approximation guarantees for the empirical distribution and may be large in practice.
4. The semantics of global explanations may not fully align with users' actual needs for local explanations.
5. The computational complexity of evaluating the value function itself is not discussed—even a single inference pass may be costly for neural networks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The unified framework and the discovery of the striking local/global property contrast are highly original.
- **Experimental Thoroughness**: ⭐⭐ — A purely theoretical work with no empirical support.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Definitions are rigorous, theorems are well-organized, and proof strategies are clearly presented.
- **Value**: ⭐⭐⭐⭐⭐ — Lays an important foundation for the complexity theory of explainable AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Directional Sheaf Hypergraph Networks: Unifying Learning on Directed and Undirected Hypergraphs](directional_sheaf_hypergraph_networks_unifying_learning_on_directed_and_undirect.md)
- [\[ICLR 2026\] Neural Networks Learn Generic Multi-Index Models Near Information-Theoretic Limit](neural_networks_learn_generic_multi-index_models_near_information-theoretic_limi.md)
- [\[ICML 2026\] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective](../../ICML2026/optimization/learning_dynamics_of_zeroth-order_optimization_a_kernel_perspective.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[ICLR 2026\] Test-Time Meta-Adaptation with Self-Synthesis](test-time_meta-adaptation_with_self-synthesis.md)

</div>

<!-- RELATED:END -->
