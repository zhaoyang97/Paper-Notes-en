---
title: >-
  [Paper Note] Estimating Causal Effects in Gaussian Linear SCMs with Finite Data
description: >-
  [ICML 2025][Causal Inference][Structural Causal Models] This work proposes the Centralized Gaussian Linear SCM (CGL-SCM), which significantly reduces the parameter space by standardizing exogenous variables to $\mathcal{N}(0,1)$, and designs an EM-based estimation algorithm to accurately recover identifiable causal effects under finite observational data.
tags:
  - "ICML 2025"
  - "Causal Inference"
  - "Structural Causal Models"
  - "Gaussian Linear SCMs"
  - "EM Algorithm"
  - "Causal Effect Estimation"
  - "Latent Confounders"
date: 2026-05-08
content_hash: 35c744835fb69d8c
---

# Estimating Causal Effects in Gaussian Linear SCMs with Finite Data

**Conference**: ICML 2025  
**arXiv**: [2601.04673](https://arxiv.org/abs/2601.04673)  
**Code**: None  
**Area**: Causal Inference  
**Keywords**: Structural Causal Models, Gaussian Linear SCMs, EM Algorithm, Causal Effect Estimation, Latent Confounders

## TL;DR

This work proposes the Centralized Gaussian Linear SCM (CGL-SCM), which significantly reduces the parameter space by standardizing exogenous variables to $\mathcal{N}(0,1)$, and designs an EM-based estimation algorithm to accurately recover identifiable causal effects under finite observational data.

## Background & Motivation

Estimating causal effects from observational data is a core challenge in causal inference, especially when latent confounders exist. Existing works mainly fall into two paradigms:

**Non-parametric methods**: Represented by Pearl's do-calculus, these methods theoretically solve the identifiability of L2 (intervention) and L3 (counterfactual) queries, but typically assume infinite data or do not involve concrete parameter estimation.

**Parametric methods**: Linear Structural Causal Models (Linear SCMs) are commonly used in econometrics and statistics, but prior works often assume infinite data, partially known distribution parameters, or the Markovian assumption (excluding latent confounders).

**Limitations of Prior Work**: Although Gaussian Linear SCMs (GL-SCMs) are analytically tractable, they suffer from overparameterization when modeling observed variables and latent confounders, making parameter estimation infeasible with finite data. The exogenous variables $\mathbf{U'} \sim \mathcal{N}(\boldsymbol{\mu_{U'}}, \boldsymbol{\Sigma^2})$ introduce a total of $2|\mathbf{U}|$ additional parameters for mean and variance, which, combined with edge weights and biases, far exceeds the information that finite observational data can constrain.

**Goal**: To estimate identifiable causal effects in GL-SCMs using only finite observational samples, given a known causal diagram.

## Method

### Overall Architecture

This paper proposes a two-step technical approach:

1. **Model simplification**: Define the CGL-SCM subclass and prove its equivalence to GL-SCMs in terms of causal effect identifiability.
2. **Parameter estimation**: Design an EM algorithm to learn CGL-SCM parameters from finite data, thereby calculating causal effects.

The key insight is that the identifiability of causal effects implies that all SCMs sharing the same observational distribution $P(\mathbf{X})$ and causal graph $G$ will yield the same results for an identifiable query $Q$. Thus, instead of recovering the true data-generating model, it suffices to find a CGL-SCM that matches the observational distribution.

### Key Designs

#### 1. Centralized Gaussian Linear SCM (CGL-SCM)

CGL-SCM simplifies the parameter space via two key constraints:

- Exogenous confounding variables: $\mathbf{U} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ (standard normal)
- Exogenous non-confounding variables: $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Psi^2})$ (zero-mean)

The structural equation for endogenous variables is:

$$X_i = \sum_{X_j \in Pa^o(X_i)} \alpha_{ji} X_j + \sum_{U_k \in Pa^u(X_i)} \alpha_{ki} U_k + \mu_i + \varepsilon_i$$

Compared to GL-SCMs, CGL-SCMs "absorb" the mean and variance information of exogenous variables into the edge weights $\alpha_{ki}$ and biases $\mu_i$, achieving $\alpha_{ki} = \alpha'_{ki} \cdot \sigma^2_{U'_k}$ and $\mu_i = \sum \alpha'_{ki} \mu_{U'_k} + \mu'_i + \mu_{\varepsilon'_i}$.

**Equivalence Theorem (Theorem 2.3)**: For any GL-SCM $M'$, there exists a CGL-SCM $M$ with the same causal graph such that $P^{M'}(\mathbf{X}) = P^{M}(\mathbf{X})$.

**Identifiability Transfer Theorem (Theorem 2.4)**: If a query $Q$ is identifiable in causal graph $G$, then $P^{M'}(Q) = P^{M}(Q)$. This implies that the causal effect estimated on the CGL-SCM is completely consistent with the original GL-SCM.

#### 2. Vectorized Representation

To make the EM algorithm feasible, the CGL-SCM is converted into a matrix form. Define:

- **Edge weight matrix** $T$: $t_{ij} = \alpha_{ij}$ if $X_i \to X_j$, otherwise 0.
- **Path aggregation matrix** $B$: $B = I + \sum_{i=1}^{d} T^i$, where $d$ is the maximum path length in the graph.
- **Exogenous influence matrix** $C$: $c_{ij}$ represents the direct influence of $U_i$ on $X_j$.

The final vectorized form:

$$\mathbf{X} = B^T \boldsymbol{\mu} + B^T C^T \mathbf{U} + B^T \boldsymbol{\varepsilon}$$

#### 3. Masking Mechanism for Causal Graph Constraints

Directly applying EM may produce parameters that violate the causal graph structure. The paper defines mask matrices $B_m$ and $C_m$:

- $B_m$: Calculated from $A = \sum_{i=1}^d T_m^i$, where $b^m_{ij} = 1$ when $a_{ij} > 0$ (i.e., a directed path exists).
- $C_m$: The $(i,j)$ element is 1 if and only if $U_i \to X_j$ exists.

During gradient updates, masks are used to enforce $B \leftarrow B + \eta \nabla_B l \cdot B_m$, ensuring that weights at missing edges remain zero.

#### 4. Recovering the Edge Weight Matrix T from the Aggregation Matrix B (Algorithm 2: CGL-Edge)

Matrix $B$ encodes the total influence between variables (the sum of all paths), which needs to be restored to single-edge weights $T$. For each node $X_i$, their direct successors $\bar{X_i}$ are traversed in topological order, recursively computing:

$$t_{ik} = b_{ik} - \sum_{\tau_i(X_j) < \tau_i(X_k)} t_{ij} \cdot b_{jk}$$

This essentially strips the contribution of indirect paths from the total effect, progressively extracting the weights of direct causal edges.

### Loss & Training

#### EM Algorithm (Algorithm 1: CGL-Go)

**E-step**: Compute the posterior distribution of latent variables $\mathbf{U}^i | \mathbf{x}^i$:

$$\boldsymbol{\mu}_{\mathbf{U}^i | \mathbf{x}^i} = CB \big((CB)^T CB + B^T B\big)^{-1} (\mathbf{x}^i - B^T \boldsymbol{\mu})$$

$$\Sigma_{\mathbf{U}^i | \mathbf{x}^i} = \mathbf{I} - CB \big((CB)^T CB + B^T B\big)^{-1} B^T C^T$$

**M-step**: Maximize the expected log-likelihood:

$$\max_{B,C,\mu} -n \log |B^T B| - \sum_{i=1}^N \mathbb{E}_{\mathbf{U}^i|\mathbf{x}^i} \Big[ (\mathbf{x}^i - B^T \boldsymbol{\mu} - B^T C^T \mathbf{U}^i)^T (B^T B)^{-1} (\mathbf{x}^i - B^T \boldsymbol{\mu} - B^T C^T \mathbf{U}^i) \Big]$$

Since there are no closed-form solutions for $B$ and $C$, optimization is performed using gradient ascent coupled with mask constraints. $\boldsymbol{\mu}$ has a closed-form update:

$$\boldsymbol{\mu} = \frac{1}{N} \sum_{i=1}^N \Big( (B^T)^{-1} \mathbf{x}^i - C^T \boldsymbol{\mu}_{\mathbf{U}^i | \mathbf{x}^i} \Big)$$

**Key Strategy**: $B$ and $C$ are optimized alternately. Within each round, gradient updates are performed on all samples until convergence, and then the other matrix is updated, followed by the update of $\boldsymbol{\mu}$, iterating the entire loop until convergence.

## Key Experimental Results

### Main Results

Synthetic data was generated in two classic causal graphs (Frontdoor and Napkin) with a sample size of 10,000 to compare the original and estimated intervention distributions.

| Causal Graph | Intervention Query | Original Distribution | Estimated Distribution | Mean Error |
|--------------|--------------------|-----------------------|------------------------|------------|
| Frontdoor | $P(X_3 \| do(X_2=1))$ | $\mathcal{N}(1.1, 1.09)$ | $\mathcal{N}(1.1018, 1.069)$ | 0.0018 |
| Frontdoor | $P(X_3 \| do(X_1=1))$ | $\mathcal{N}(0.74, 1.9)$ | $\mathcal{N}(0.7391, 1.881)$ | 0.0009 |
| Napkin | $P(X_4 \| do(X_3=1))$ | $\mathcal{N}(0.3, 1.16)$ | $\mathcal{N}(0.3051, 1.1692)$ | 0.0051 |
| Napkin | $P(X_4 \| do(X_1=1))$ | $\mathcal{N}(-1.068, 2.3248)$ | $\mathcal{N}(-0.9721, 2.3274)$ | 0.0959 |

### Ablation Study

| Configuration | Key Metrics | Description |
|---------------|-------------|-------------|
| Direct Estimation of GL-SCM | Infeasible | Too many parameters, unidentifiable with finite data |
| CGL-SCM + EM | Mean Error < 0.1 | Parameters reduced after standardization, estimation is feasible |
| Mask Constraint Enabled | Maintains Graph Structure | Eliminates spurious weights at non-existing edges |
| Mask Constraint Disabled | Graph Structure Destroyed | Non-existing edges appear after optimization |

### Key Findings

1. **High-precision recovery**: The mean estimation error of the intervention distribution on the Frontdoor graph is less than 0.002, and the variance error is less than 0.02.
2. **Napkin graph is more challenging**: The mean error of $P(X_4|do(X_1=1))$ reaches 0.096, as this query involves more complex paths and more latent variables.
3. **Equivalence of CGL-SCM verified**: Despite the simplified parameter space, the learned model still accurately recovers the causal distribution.

## Highlights & Insights

1. **Concise yet profound theoretical contribution**: The proof of equivalence for CGL-SCM is elegant—by simply "absorbing" the mean and variance information of exogenous variables into the coefficients of structural equations, the parameters are significantly reduced without losing any causal information.
2. **Practical methodology**: Combining vectorized representation, masked gradients, and the EM algorithm, the theoretical identifiability problem is successfully transformed into a practically solvable optimization problem.
3. **Insight of "no need to recover the true model"**: Leveraging the definition of causal effect identifiability, it is only necessary to find a model consistent with the observational distribution, avoiding the fundamental difficulty of parameter unidentifiability.

## Limitations & Future Work

1. **Limited to Gaussian linear models**: It cannot handle non-linear relationships or non-Gaussian noise, restricting its applicability.
2. **Requires a known causal graph**: In practice, the causal graph is often unknown and needs to be integrated with causal discovery methods.
3. **Small experimental scale**: Validated only on classic small graphs with 3-4 variables, lacking scalability analysis on large-scale causal graphs.
4. **Missing convergence analysis**: No convergence rate or finite-sample error bounds are provided for the EM algorithm.
5. **Unexplored non-identifiable queries**: Although the authors mention in the conclusion that finite-data bounds could be established for non-identifiable queries, this was not implemented.
6. **Lack of comparison with machine learning methods**: There is no comparison with existing finite-data causal estimation methods such as Double Machine Learning.

## Related Work & Insights

- **Non-parametric causal inference**: Pearl's do-calculus, the c-component decomposition of Tian & Pearl (2002), and the counterfactual testability of Shpitser & Pearl (2007).
- **Linear SCM identifiability**: The graphical criteria of Brito & Pearl (2002), and the instrumental/auxiliary cutsets of Kumor et al. (2019, 2020).
- **Causal estimation with finite data**: The Double Machine Learning method of Jung et al. (2021), which is however limited to non-parametric settings.
- **Insights**: Combining parametric assumptions with non-parametric identifiability theory is an important direction for causal inference with finite data; the standardization technique can be generalized to other parametric families.

## Rating

| Dimension | Score (1-5) | Description |
|-----------|-------------|-------------|
| Novelty | 3 | The introduction of CGL-SCM is somewhat novel, but not a major technical breakthrough. |
| Theoretical Depth | 4 | The equivalence proof is rigorous, and the EM derivation is complete. |
| Experimental Thoroughness | 2 | Only two small synthetic graphs, lacking real-world data and scalability experiments. |
| Writing Quality | 3 | Clear structure but relatively short, with some details left in the appendix. |
| Value | 3 | Useful for Gaussian linear scenarios, but the scope of application is narrow. |
| **Total Score** | **3.0** | Solid theoretical contribution but insufficient experimental validation. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Independence Test for Linear Non-Gaussian Data and Applications in Causal Discovery](../../ICLR2026/causal_inference/independence_test_for_linear_non-gaussian_data_and_applications_in_causal_discov.md)
- [\[ICML 2025\] Causal Abstraction Inference under Lossy Representations](causal_abstraction_inference_under_lossy_representations.md)
- [\[ICLR 2026\] Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models](../../ICLR2026/causal_inference/distributional_equivalence_in_linear_non-gaussian_latent-variable_cyclic_causal_.md)
- [\[ICML 2025\] Isolated Causal Effects of Natural Language](isolated_causal_effects_of_natural_language.md)
- [\[NeurIPS 2025\] An Analysis of Causal Effect Estimation Using Outcome Invariant Data Augmentation](../../NeurIPS2025/causal_inference/an_analysis_of_causal_effect_estimation_using_outcome_invariant_data_augmentatio.md)

</div>

<!-- RELATED:END -->
