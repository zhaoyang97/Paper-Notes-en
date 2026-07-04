---
title: >-
  [Paper Note] Maximum Coverage in Turnstile Streams with Applications to Fingerprinting Measures
description: >-
  [ICML2025][Streaming Algorithms][maximum coverage] This work presents the first single-pass streaming algorithm for the maximum coverage problem under the turnstile streaming model (supporting arbitrary insertions/deletions) with $\tilde{O}(d/\varepsilon^3)$ space and $\tilde{O}(1)$ update time. The algorithm is further extended to the privacy fingerprinting scenario, achieving up to a 210× speedup compared to prior methods in experiments.
tags:
  - "ICML2025"
  - "Streaming Algorithms"
  - "Submodular Optimization"
  - "Privacy Risk Measurement"
  - "maximum coverage"
  - "turnstile streaming"
  - "linear sketch"
  - "fingerprinting"
  - "frequency moment"
  - "submodular maximization"
date: 2026-05-08
content_hash: b5223d80005c7236
---

# Maximum Coverage in Turnstile Streams with Applications to Fingerprinting Measures

**Conference**: ICML2025  
**arXiv**: [2504.18394](https://arxiv.org/abs/2504.18394)  
**Code**: None  
**Area**: Streaming Algorithms / Submodular Optimization / Privacy Risk Measurement  
**Keywords**: maximum coverage, turnstile streaming, linear sketch, fingerprinting, frequency moment, submodular maximization

## TL;DR

This work presents the first single-pass streaming algorithm for the maximum coverage problem under the turnstile streaming model (supporting arbitrary insertions/deletions) with $\tilde{O}(d/\varepsilon^3)$ space and $\tilde{O}(1)$ update time. The algorithm is further extended to the privacy fingerprinting scenario, achieving up to a 210× speedup compared to prior methods in experiments.

## Background & Motivation

**Maximum Coverage**: Given $d$ sets (drawn from a universe of size $n$), the goal is to select $k$ sets to maximize the size of their union. The classical greedy algorithm achieves a tight approximation ratio of $(1-1/e)$, but requires $O(knd)$ time and $O(nd)$ space, which is impractical for large-scale data.

**Requirements for Streaming Models**:
- Prior works [BEM17, MV19] only support **insertion-only** streams (allowing only insertions) and do not support deletions.
- Practical scenarios (e.g., dynamic updates of private datasets) require **turnstile streams**, which support both insertions and deletions.
- This work introduces the first single-pass algorithm for maximum coverage under the turnstile model.

**Applications in Fingerprinting**: In privacy auditing, it is necessary to identify $k$ features that maximize the risk of user re-identification. This task can be reduced to the maximum coverage or submodular maximization problem. Prior algorithms, such as [GÁC16], require $O(nd)$ space and $O(knd)$ time.

## Method

### Core Idea: Linear Sketch

Represent the input as an $n \times d$ matrix $\boldsymbol{A}$ (where $A_{ij} \neq 0$ indicates that element $i$ belongs to set $j$). Use a random sketch matrix $\boldsymbol{S}$ ($r \times n$) to compress the input:

$$\boldsymbol{S}(\boldsymbol{A} + c_{ij}) = \boldsymbol{S}\boldsymbol{A} + \boldsymbol{S} c_{ij}$$

Linearity guarantees that both insertions and deletions can update the sketch efficiently without storing the full matrix.

### Maximum Coverage Algorithm (Max-Coverage-LS)

**Step 1: Subsampling to reduce the universe.** Apply the technique from [MV19] to perform random subsampling on rows, yielding $\boldsymbol{A}'$ such that $\text{OPT} = O(k \log d / \varepsilon^2)$ . This step only incurs an $\varepsilon/4$ loss in approximation quality.

**Step 2: Hash bucketing + non-zero entry sampling.** The rows of $\boldsymbol{A}'$ are mapped into $b = O(k \log d / \varepsilon^2)$ buckets using $t = O(\log(d/\varepsilon))$ independent hash functions. Within each bucket, at most $O(d \log(1/\varepsilon)/(\varepsilon k))$ non-zero entries are retained.

**Step 3: Constructing the small matrix $\boldsymbol{A}_*$.** Process the rows in a random permutation, extracting them one by one from the stored non-zero entries until $\boldsymbol{A}_*$ contains $\tilde{O}(d/\varepsilon^3)$ non-zero entries.

**Step 4: Greedy solver.** Run a standard greedy algorithm (or any $1-1/e$ approximation algorithm) on $\boldsymbol{A}_*$ to obtain the final solution.

A key lemma ensures that entries of both heavy rows ($\geq d/k$ non-zero entries) and light rows can be recovered correctly.

### Submodular Maximization Framework (Theorem 3)

To address general fingerprinting, a more generalized submodular maximization framework is designed: if a monotone submodular function $f$ can be estimated within a $(1 \pm \gamma)$ approximation by a linear sketch with $O(s)$ space, a $(1-1/e-\varepsilon)$ approximation can be obtained in $O(sk)$ space.

### Frequency Moment Complement Estimation (Theorem 4)

A novel linear sketch is designed to estimate $n^p - F_p$ ($p \geq 2$), where $F_p = \sum_i f_i^p$ is the $p$-th frequency moment:

$$\text{Sketch size} = \tilde{O}(\gamma^{-2/(p-1)}), \quad \text{approximation ratio} = (1 \pm \gamma^{1/(p-1)})$$

This result holds independent theoretical value and is applied to instantiate the submodular framework for solving general fingerprinting.

## Theoretical Results Overview

| Problem | Approximation Ratio | Space | Update Time | Prior Best |
|------|--------|------|----------|----------|
| Maximum Coverage (turnstile) | $(1-1/e-\varepsilon)$ | $\tilde{O}(d/\varepsilon^3)$ | $\tilde{O}(1)$ | Insertion-only only |
| Targeted Fingerprinting | $(1-1/e-\varepsilon)$ | $\tilde{O}(d/\varepsilon^3)$ | $\tilde{O}(1)$ | $O(nd)$ space [GÁC16] |
| General Fingerprinting | $(1-1/e-\varepsilon)$ | $\tilde{O}(dk^3/\varepsilon^2)$ | $\tilde{O}(k^3/\varepsilon^2)$ | $O(nd)$ space [GÁC16] |
| $n^p - F_p$ Estimation | $(1 \pm \gamma^{1/(p-1)})$ | $\tilde{O}(\gamma^{-2/(p-1)})$ | $\tilde{O}(\gamma^{-2/(p-1)})$ | New Result |

**Lower Bound Matching**: The space bound of $\tilde{O}(d/\varepsilon^3)$ is close to the $\Omega(d/\varepsilon^2)$ lower bound [Ass17], and the approximation ratio of $(1-1/e)$ is optimal (assuming P ≠ NP) [Fei98].

## Key Experimental Results

- **Targeted Fingerprinting**: Achieves a **49×** speedup compared to [GÁC16] with comparable accuracy.
- **General Fingerprinting**: Achieves a **210×** speedup compared to [GÁC16] with comparable accuracy.
- Evaluated the practicality on two different datasets.
- The general fingerprinting algorithm can also serve as a **dimensionality reduction technique** for feature selection in clustering algorithms such as $k$-means, significantly boosting efficiency with minimal accuracy loss.

## Highlights & Insights

1. **First maximum coverage algorithm under turnstile streams**: Breaks through the limitation of prior works that only support insertion-only streams, achieving highly efficient polylogarithmic update time.
2. **Generality of linear sketches**: Since the algorithm itself is a linear sketch, it directly applies to distributed (coordinator model) and parallel computing scenarios.
3. **New estimator for $n^p - F_p$**: The linear sketch for the frequency moment complement is an independent and significant theoretical contribution, complementing classic studies on $F_p$ estimation (such as AMS99).
4. **Complete pipeline from theory to application**: Progresses step-by-step from maximum coverage to the submodular maximization framework, the frequency moment sketch, and finally to fingerprinting.
5. **Significant experimental speedup** (210×), demonstrating the practical feasibility of the theoretical algorithm.

## Limitations & Future Work

- The space complexity of general fingerprinting contains a $k^3$ factor, raising overhead when $k$ is large. Can the dependency on $k$ be reduced?
- The trade-off between the precision parameter $\varepsilon$ and space ($1/\varepsilon^3$) in the approximation ratio is relatively steep. Does a better sketch exist?
- Experiments only validate the fingerprinting scenario and have not tested other classic maximum coverage applications such as information retrieval or influence maximization.
- The greedy algorithm in the post-processing phase still requires $O(k \cdot |\boldsymbol{A}_*|)$ time, although $|\boldsymbol{A}_*|$ has already been compressed.
- The error form $\gamma^{1/(p-1)}$ of the frequency moment complement estimation amplifies as $p$ increases, resulting in limited accuracy for higher-order moments.

## Related Work & Insights

- **[BEM17]** Bateni et al.: $(1-1/e-\varepsilon)$ approximation under insertion-only streams with $\tilde{O}(d/\varepsilon^3)$ space, which serves as the foundation of this work.
- **[MV19]** McGregor & Vu: Subsampling framework under set-arrival streams.
- **[GÁC16]** Gulyás et al.: Greedy algorithms for fingerprinting, of which this work significantly improves the efficiency.
- **[KNW10]** Kane, Nelson, Woodruff: $L_0$ sketch with $O(1)$ update time.
- **[AMS99]** Alon, Matias, Szegedy: Pioneering work on frequency moment estimation.
- **[CCF04]** CountSketch: The fundamental tool for frequency estimation used in this work.

## Rating

- Novelty: ⭐⭐⭐⭐ (First turnstile maximum coverage algorithm + novel frequency moment complement sketch)
- Theoretical Depth: ⭐⭐⭐⭐⭐ (Nested sketch design and rigorous proofs)
- Experimental Thoroughness: ⭐⭐⭐ (Fingerprinting scenario only, limited datasets)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, well-organized from merits to details)
- Value: ⭐⭐⭐⭐ (210× speedup, applicable to real-time privacy monitoring)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Sampling Data Structures for Tensor Products in Turnstile Streams](../../ICLR2026/learning_theory/towards_sampling_data_structures_for_tensor_products_in_turnstile_streams.md)
- [\[ICML 2025\] Improved Generalization Bounds for Transductive Learning by Transductive Local Complexity and Its Applications](improved_generalization_bounds_for_transductive_learning_by_transductive_local_c.md)
- [\[ICLR 2026\] "Noisier" Noise Contrastive Estimation is (Almost) Maximum Likelihood](../../ICLR2026/learning_theory/noisier_noise_contrastive_estimation_is_almost_maximum_likelihood.md)
- [\[ICLR 2026\] On the Wasserstein Geodesic Principal Component Analysis of probability measures](../../ICLR2026/learning_theory/on_the_wasserstein_geodesic_principal_component_analysis_of_probability_measures.md)
- [\[ICML 2025\] On Fine-Grained Distinct Element Estimation](on_fine-grained_distinct_element_estimation.md)

</div>

<!-- RELATED:END -->
