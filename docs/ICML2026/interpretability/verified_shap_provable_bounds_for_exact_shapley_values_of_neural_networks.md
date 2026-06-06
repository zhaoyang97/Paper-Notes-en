---
title: >-
  [Paper Note] Verified SHAP: Provable Bounds for Exact Shapley Values in Neural Networks
description: >-
  [ICML 2026][Interpretability][SHAP] By combining branch-and-bound with neural network verification techniques, VERISHAP provides the first provable bounds for neural network SHAP value calculation—scaling to feature sear…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "SHAP"
  - "Neural Network Verification"
  - "Branch-and-Bound"
  - "Provable Bounds"
date: 2026-05-08
content_hash: e69439ca7bfc74bd
---

# Verified SHAP: Provable Bounds for Exact Shapley Values in Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2605.24084](https://arxiv.org/abs/2605.24084)  
**Code**: https://github.com/sen-uni-kn/verishap  
**Area**: Interpretability / Neural Network Verification  
**Keywords**: SHAP, Interpretability, Neural Network Verification, Branch-and-Bound, Provable Bounds

## TL;DR
By combining branch-and-bound with neural network verification techniques, VERISHAP provides the first provable bounds for neural network SHAP value calculation—scaling to feature search spaces several orders of magnitude larger than existing exact methods.

## Background & Motivation

**Background**: SHAP is the most widely used feature attribution method. While exact values can be efficiently calculated for tree and linear models, exact computation for neural networks faces exponential complexity.

**Limitations of Prior Work**: SHAP calculation for neural networks results in an exponential search space over feature subsets ($2^n$ coalitions). Existing methods provide only statistical approximations (KernelSHAP, DeepSHAP). These methods have two fundamental limitations: (1) insufficient accuracy under highly non-linear models; (2) lack of a principled evaluation framework (inability to obtain "ground truth" for verification).

**Key Challenge**: The contradiction between the #P-hard nature of exact SHAP calculation and practical interpretability needs—researchers cannot verify the quality of approximation methods using exact values.

**Goal**: (1) Expand the feasible scale of exact SHAP calculation; (2) Provide provable bounds at arbitrary precision; (3) Establish "ground truth" benchmarks for evaluating SHAP approximation methods.

**Key Insight**: Neural network verification fields (branch-and-bound, bound propagation like CROWN) can be used to calculate complex functional properties but have not been applied to SHAP before. The SHAP marginal contribution $\Delta_i(S)$ can be approximated as linear within partitions of the feature space, which perfectly aligns with the piecewise linear structure of ReLU networks.

**Core Idea**: Use branch-and-bound to transform SHAP calculation into an interval optimization problem—recursively partitioning the feature set and using bound propagation from neural network verification to compute upper and lower bounds for marginal contributions within each partition, eventually recovering exact values.

## Method

### Overall Architecture
(1) Initialize a trivial partition; (2) Perform **recursive partitioning** on branches—selecting an unassigned feature $j$ as the split axis; (3) Use **bound propagation** for each branch to calculate the upper and lower bounds of the marginal contribution; (4) Aggregate bounds for SHAP values according to Shapley weights; (5) Converge to exact values when upper and lower bounds are equal.

### Key Designs

1.  **Theoretical Foundation for SHAP Bounds (Theorem 3.1)**:
    - **Function**: Derives global bounds for SHAP values from partition-level marginal contribution bounds.
    - **Mechanism**: Groups feature subsets into branches $B_k$. By calculating $\Delta_{i,B}^{lower} \leq \Delta_i(S) \leq \Delta_{i,B}^{upper}$ for each branch, it follows that $\sum_{B} \Lambda_B \Delta_{i,B}^{lower} \leq \varphi_i \leq \sum_{B} \Lambda_B \Delta_{i,B}^{upper}$.
    - **Design Motivation**: Decomposes the exponential summation into branch-level summations, requiring only one pair of bounds to be computed per branch.

2.  **Recursive Partitioning and Closed-form Weight Calculation**:
    - **Function**: Efficiently partitions the feature space and calculates the Shapley weight for each branch.
    - **Mechanism**: A branch is defined by $(I, E)$. Let $r = |I|, s = |I| + |E|$, then $\Lambda_B = \binom{s}{r}^{-1} (s+1)^{-1}$ (closed-form formula). Recursive refinement uses $\Lambda_{B'} = \frac{r+1}{s+2} \Lambda_B$ (including feature) and $\Lambda_{B''} = \frac{s+1-r}{s+2} \Lambda_B$ (excluding feature).
    - **Design Motivation**: Avoids repeated combinatorial calculations for each branch by using combinatorial identities for constant-time updates.

3.  **Adaptation of Masking and Bound Propagation**:
    - **Function**: Maps discrete feature subsets to a continuous domain to enable the application of existing neural network verification tools.
    - **Mechanism**: Represents a feature subset $S$ using a binary mask $m \in \{0,1\}^n$, defining $\Delta_i(m) = v(m^{+i}) - v(m)$. A branch $(I,E)$ corresponds to a mask interval $[m, \bar{m}] \subseteq [0,1]^n$. Use Linear Bound Propagation (LBP) (e.g., CROWN) to calculate $[\Delta_{i,B}, \Delta_{i,B}]$.
    - **Design Motivation**: Bridges discrete combinatorial optimization (SHAP) and continuous convex optimization (neural network verification).

## Key Experimental Results

### Main Results

| Dataset | Features $n$ | Coalitions | EXACTSHAP (s) | VERISHAP Exact (s) | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Obesity | 17 | $2^{17} > 10^4$ | 4 | 18 | ✓ Feasible |
| German | 20 | $2^{20} > 10^6$ | 9 | 16 | ✓ Feasible |
| Mushroom | 22 | $2^{22} > 10^6$ | – (OOM) | 17 | **Incomparable** |
| Default | 23 | $2^{23} > 10^6$ | – (OOM) | 127 | **Incomparable** |
| Auto | 25 | $2^{25} \approx 3 \times 10^7$ | – (OOM) | 81 | **Incomparable** |
| Sonar | 60 | $2^{60} > 10^{18}$ | – | 13 (10% HR) | **Significant Scaling** |

### MNIST CNN Convergence

| Iterations | Runtime (s) | Key Metric |
| :--- | :--- | :--- |
| t=1 | 1s | Bound width > 200% of network output |
| t=121 (25% HR) | 25s | Bound width shrinks to 25% |
| t=278 (10% HR) | 29s | Importance of image regions clearly displayed |
| t=512 Exact | 34s | Upper and lower bounds equal, recovering exact value |

### Key Findings
- VERISHAP scales exact calculation for tabular data with $n \geq 22$ by 1-2 orders of magnitude.
- On the Sonar dataset with $n=60$ features, it reduces OOM to a bound calculation of 13s.
- CROWN-IBP detects the constancy of the value function across multiple coalitions (Application of Theorem 3.5), avoiding the exhaustion of $2^{64}$ coalitions.
- Bounds are usable after the first few dozen iterations—attribution patterns emerge at 10% relative error.
- Splitting strategy ablation: SMEARS > SmartBranching > StrongBranching > InOrder.

## Highlights & Insights
- **Paradigm Shift from Verification to Interpretation**: Systematically introduces neural network verification tools to SHAP calculation for the first time, demonstrating the deep connection between adversarial robustness and interpretability.
- **Theory-Practice Gap**: Theorem 3.5 proves that piecewise linear networks can degrade into constant function queries after sufficient partitioning, circumventing combinatorial explosion.
- **Provable Bounds as Intermediate Products**: The generated sequence of bounds is practical dozens of iterations before reaching the exact value, providing a principled solution for the time-accuracy tradeoff.
- **Inversion of the Evaluation Framework**: For the first time, exact values can be obtained on real tabular data with 20-25 features to evaluate KernelSHAP and TreeMSR.

## Limitations & Future Work
- High-dimensional bottlenecks—exact calculation still requires > 100s when $n > 25$; RGB pixels in images remain impractical.
- Dependency on value function tractability—if bounds are too loose, the recursion explores exponentially many branches.
- Implicit assumptions on background distribution—different background choices affect the tightness of the verifier's bounds.
- Improvements: Integrating stronger neural network verification algorithms; research into feature aggregation theory; adapting to other attribution methods (IG, DeepLIFT).

## Related Work & Insights
- **vs ExactSHAP**: Enumerates all $2^{n-1}$ coalitions, which often leads to OOM for neural networks; VERISHAP avoids exhaustive enumeration via branch-and-bound + bound propagation.
- **vs KernelSHAP/DeepSHAP**: MC sampling approximations have no accuracy guarantees; VERISHAP provides a "ground truth benchmark" and upper/lower bounds.
- **Inspiration Chain**: Neural network verification evolved from adversarial robustness to general bound propagation; this paper allows that flow to reverse into the field of interpretation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic solution of SHAP using verification technology, featuring both paradigm and technical innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive tabular data + complete ablation + MNIST visualization; high-dimensional image experiments are missing.
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous with clear algorithm pseudocode; the section on splitting strategies is slightly too short.
- Value: ⭐⭐⭐⭐⭐ Significant theoretical breakthrough and practical value, with profound implications for the fusion of Explainable AI and Formal Verification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks](probabilistic_modeling_of_latent_agentic_substructures_in_deep_neural_networks.md)
- [\[NeurIPS 2025\] SHAP Values via Sparse Fourier Representation](../../NeurIPS2025/interpretability/shap_values_via_sparse_fourier_representation.md)
- [\[ICML 2026\] ShaplEIG: Bayesian Experimental Design for Shapley Value Estimation](shapleig_bayesian_experimental_design_for_shapley_value_estimation.md)
- [\[ICLR 2026\] Modal Logical Neural Networks for Financial AI](../../ICLR2026/interpretability/modal_logical_neural_networks_for_financial_ai.md)
- [\[ICML 2026\] Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in Large Language Models](dual_mechanisms_of_value_expression_intrinsic_vs_prompted_values_in_large_langua.md)

</div>

<!-- RELATED:END -->
