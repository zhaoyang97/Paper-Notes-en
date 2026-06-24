---
title: >-
  [Paper Note] Latent Variable Causal Discovery under Selection Bias
description: >-
  [ICML 2025][Causal Inference][Latent Causal Discovery] Extends rank constraints to selection bias scenarios for the first time, proving that under linear selection mechanisms, the rank of the biased covariance matrix still preserves information about the causal structure and selection mechanism. It proposes a generalized t-separation graphical criterion, proves identifiability in one-factor models, and validates effectiveness on both synthetic and real-world datasets (World V…
tags:
  - "ICML 2025"
  - "Causal Inference"
  - "Latent Causal Discovery"
  - "Selection Bias"
  - "Rank Constraints"
  - "Linear Gaussian Models"
  - "One-Factor Models"
date: 2026-05-08
content_hash: 752d83c49ce8c022
---

# Latent Variable Causal Discovery under Selection Bias

**Conference**: ICML 2025  
**arXiv**: [2512.11219](https://arxiv.org/abs/2512.11219)  
**Code**: [https://github.com/MarkDana/Latent-Selection](https://github.com/MarkDana/Latent-Selection)  
**Area**: Causal Inference  
**Keywords**: Latent Causal Discovery, Selection Bias, Rank Constraints, Linear Gaussian Models, One-Factor Models

## TL;DR

Extends rank constraints to selection bias scenarios for the first time, proving that under linear selection mechanisms, the rank of the biased covariance matrix still preserves information about the causal structure and selection mechanism. It proposes a generalized t-separation graphical criterion, proves identifiability in one-factor models, and validates effectiveness on both synthetic and real-world datasets (World Value Survey, Big Five Personality).

## Background & Motivation

**Background**: Latent variable causal discovery aims to recover causal relationships between unobserved latent variables from observational data, which is crucial in scenarios such as psychological questionnaires and socio-economic surveys. Existing tools have expanded from basic conditional independence (CI) constraints to various statistical tools including rank constraints, high-order moment constraints, and matrix factorization, significantly enhancing the capability of latent causal discovery.

**Limitations of Prior Work**: Selection bias (e.g., people with specific traits being more willing to participate in surveys) is ubiquitous in real-world data. However, all new tools that go beyond CI constraints are designed solely for latent variables, and none of them can handle selection bias. Currently, the only algorithm capable of handling both latent variables and selection bias is FCI. Yet, FCI relies exclusively on CI constraints, which are insufficient for latent causal discovery—it can only discover causal relationships among observed variables and cannot identify the causal structure among latent variables.

**Key Challenge**: This creates a major gap: although stronger methods for latent causal discovery exist, once selection bias is introduced, these new tools must be set aside, forcing a regression to FCI which only uses CI constraints. The core difficulty lies in the fact that selection bias completely alters the data distribution—even if the original model is linear Gaussian, it becomes truncated Gaussian after selection, making covariance and higher-order moments difficult to express and interpret.

**Goal**: Develop statistical tools that go beyond CI constraints and process both latent variables and selection bias simultaneously.

**Key Insight**: Look for invariant statistical patterns—specifically, the rank of sub-covariance matrices—rather than explicitly modeling the full distribution after selection. The intuition stems from the discovery of the "reversed Tetrad structure": even though four originally independent variables no longer follow a Gaussian distribution after linear truncation, the low-rank structure of the biased covariance matrix remains identical to the classical Tetrad.

**Core Idea**: Selection bias, like latent variables, leaves traces of "dimensionality bottlenecks" in the rank of the covariance matrix. These traces can be read from the graphical representation using a generalized t-separation criterion.

## Method

### Overall Architecture

The core contribution of this paper is a theoretical tool—**generalized rank constraints**. The process consists of: (1) defining a linear selection mechanism and constructing a selection-augmented graph $\mathcal{G}^{(\mathcal{S})}$; (2) establishing a generalized t-separation criterion on the augmented graph to precisely characterize the rank of the biased covariance matrix; (3) applying this tool to one-factor models to prove that CI relationships among latent variables can still be recovered from biased observational data; (4) integrating with the FCI algorithm to complete latent causal discovery.

### Key Designs

1. **Linear Selection Mechanism and Selection-Augmented Graph**:

    - **Function**: Define a sufficiently general mathematical model of selection bias and integrate it into the causal diagram representation.
    - **Mechanism**: The linear selection mechanism is defined as $\{(V_i, \beta_i, \epsilon_i, \mathcal{Y}_i)\}_{i=1}^k$, where each selection condition has a participating subset of variables $V_i$, linear coefficients $\beta_i$, a noise term $\epsilon_i$, and an acceptance set $\mathcal{Y}_i$. The response variable is $Y_i = \beta_i^\top V_i + \epsilon_i$, and samples are included or excluded based on whether $Y_i \in \mathcal{Y}_i$. The selection-augmented graph adds response nodes $Y$ and corresponding edges to the original DAG.
    - **Design Motivation**: This framework covers classic parametric models such as hard truncation ($\epsilon=0, \mathcal{Y}=(a,b)$), Logistic selection ($\epsilon \sim \text{Logistic}, \mathcal{Y}=(a,\infty)$), Probit selection ($\epsilon \sim \mathcal{N}(0,1), \mathcal{Y}=(a,\infty)$), and stabilization selection ($\epsilon \sim \mathcal{N}(0,1), \mathcal{Y}=\{a\}$).

2. **Graphical Criterion of Generalized Rank Constraints (Theorem 1)**:

    - **Function**: Precisely characterize the rank of sub-covariance matrices in selection-biased data.
    - **Mechanism**: $\text{rank}(\Sigma_{A,B}^{(\mathcal{S})}) = \min\{|C|+|D| : C,D \subset X \cup Y, (C,D) \text{ t-separates } (A \cup Y, B \cup Y) \text{ in } \mathcal{G}^{(\mathcal{S})}\} - k$. When $\mathcal{S}=\emptyset$ (no selection), it degenerates to the original rank constraint (Proposition 2). Core proof idea: Even under a non-Gaussian truncated distribution, block matrix analysis of conditional covariance is used to prove that $\text{rank}(\Sigma_{A,B}^{(\mathcal{S})}) = \text{rank}(\Sigma_{A \cup Y, B \cup Y}) - |Y|$.
    - **Design Motivation**: Similar to CI constraints, it aims for an exact characterization using graphical criteria, but rank constraints are a strict generalization of CI. Specifically, CI corresponds to rank $|C|$ (Proposition 1), while rank constraints can capture low-rank structures beyond CI (e.g., in a Tetrad structure where there is no CI but rank=1).

3. **Identifiability under One-Factor Models (Proposition 3)**:

    - **Function**: Proving that generalized rank constraints can recover CI relationships among latent variables in one-factor models.
    - **Mechanism**: For any disjoint subsets $A, B, C \subset L$ (latent variables), $A \perp B | C$ holds if and only if the rank of the sub-covariance matrix of observed variables $\Sigma_{\mathbf{X}_A \cup \mathbf{X}_C^{(1)}, \mathbf{X}_B \cup \mathbf{X}_C^{(2)}}^{(\mathcal{S})}$ is $|C|$ (where $\mathbf{X}_C^{(1)}$ and $\mathbf{X}_C^{(2)}$ are disjoint partitions of $\mathbf{X}_C$, each containing $\geq |C|$ observed variables).
    - **Design Motivation**: One-factor models are a classic way to model questionnaire data (each latent variable has $\geq 2$ observed measurements), where selection bias is particularly common (e.g., personality traits influencing the willingness to participate in a questionnaire).

### Loss & Training

This paper is a theoretical/algorithmic work and does not involve neural network training. The actual algorithm workflow: Assuming the one-factor clustering is known (which observed variables measure which latent variables), the generalized rank constraints are used to recover CI relationships among latent variables from the biased covariance matrix, and then the FCI algorithm is called to obtain the Partial Ancestral Graph (PAG). Rank testing is conducted using a statistical hypothesis testing framework.

## Key Experimental Results

### Synthetic Data Experiments (PAG Recovery Quality)

| Number of Latents $n$ | Number of Selection Variables | Method | Total Edge Mark Differences (↓ Better) |
|-------------|-----------|------|-------------------|
| 5 | 1 | FCI | ~8 |
| 5 | 1 | PC | ~10 |
| 5 | 1 | BOSS | ~12|
| 5 | 1 | **Ours** | **~3** |
| 10 | 2 | FCI | ~20 |
| 10 | 2 | **Ours** | **~8** |
| 15 | 3 | FCI | ~35 |
| 15 | 3 | **Ours** | **~12** |
| 20 | 4 | FCI | ~55 |
| 20 | 4 | **Ours** | **~18** |

### Computational Efficiency

| Number of Latents | Running Time | Hardware |
|---------|---------|------|
| 5 | < 1 second | 2 CPU, 16GB |
| 10 | < 1 second | 2 CPU, 16GB |
| 15 | < 1 second | 2 CPU, 16GB |
| 20 | < 5 minutes | 2 CPU, 16GB |

### Real-World Data (World Value Survey)

| Country | Detected Selection Bias Variables | Key Causal Discoveries |
|------|-----------------|------------|
| Canada | Social Trust (tail mark $\rightarrow$ selection ancestor) | Level of trust affects survey participation tendency |
| China | Social Trust + Perception of Science | Positive correlation of science perception $\rightarrow$ participation in social science research |
| Germany | 5 variables involved in selection bias | More complex multivariate selection mechanisms |
| Big Five | Agreeableness | High agreeableness $\rightarrow$ likely more willing to cooperate with questionnaires |

### Summary of Theoretical Contributions

| Tool | Latent Variables | Selection Bias | Beyond CI | First Time |
|------|--------|---------|--------|------|
| CI (d-separation) | ✓ | ✓ | ✗ | - |
| Original Rank Constraint (t-separation) | ✓ | ✗ | ✓ | sullivant2010 |
| **Generalized Rank Constraint (Ours)** | **✓** | **✓** | **✓** | **✓** |

### Key Findings
- As the number of latent variables increases, the performance gap between our method and baseline methods continues to widen—showing a roughly 3-fold advantage at 20 latent variables, which demonstrates that generalized rank constraints provide a more pronounced informational advantage in complex scenarios.
- The experimental results are consistent under both Gaussian and exponential noise selection mechanisms, validating the generality of the theory to non-Gaussian selection noise.
- Social Trust is detected as a potential ancestor of selection bias across the WVS data of three different countries, with different countries exhibiting nation-specific selection patterns.
- The "reversed Tetrad" phenomenon reveals that selection bias and latent variables are sometimes indistinguishable (rank equivalent) at the rank constraint level, but Spider structure variants demonstrate that they can be distinguished under more complex topologies.

## Highlights & Insights

- **Elegant Theorization of the "Dimensionality Bottleneck" Intuition**: While original rank constraints capture "how dependence flows through minimal dimensions," this paper proves that selection bias produces a similar dimensionality bottleneck effect, and both are unified within the same graphical criterion framework. This intuition is simple yet profound.
- **Discovery of the Reversed Tetrad Structure**: After four independent variables undergo linear truncation, the low-rank structure of the biased covariance is completely identical to the classical Tetrad (four measurements of a single latent variable)—this duality phenomenon serves as the starting point of the entire paper and is highly inspiring.
- **Ingenious Proof Strategy**: Instead of explicitly solving the covariance expressions of the truncated Gaussian (which is extremely complex), it translates the problem back to the rank constraint without selection via block matrix elimination of the conditional covariance: $\text{rank}(\Sigma^{(\mathcal{S})}_{A,B}) = \text{rank}(\Sigma_{A \cup Y, B \cup Y}) - |Y|$.

## Limitations & Future Work

- The authors acknowledge that a complete characterization of the rank equivalence class—similar to the characterization of CI equivalence classes by MAG—has not yet been established, which is key to understanding the distinguishable boundaries between latent variables and selection bias.
- The assumption of linear selection mechanisms and linear Gaussian causal models makes generalization to non-linear scenarios non-trivial.
- The one-factor model assumes that clustering is known (which observations belong to which latent variable), whereas clustering discovery itself remains an unsolved problem in practical scenarios.
- The average degree of the graphs in synthetic experiments is 2, which is relatively sparse; its effectiveness on denser causal graphs remains to be validated.
- The practical algorithm depends on statistical tests of covariance matrix ranks; the power and robustness of these tests under finite samples need further study.

## Related Work & Insights

- **vs. FCI**: FCI is currently the only algorithm capable of handling both latent variables and selection bias, but it relies solely on CI constraints, which provides insufficient information for latent causal discovery (e.g., no CI but a low-rank structure in a Tetrad structure). The generalized rank constraints in this paper strictly generalize CI constraints and can be viewed as an upgrade of FCI to the rank constraint level.
- **vs. Original Rank Constraint (sullivant2010)**: The original rank constraint handles latent variables by extending d-separation to t-separation, but completely ignores selection bias. This paper extends t-separation to selection-augmented graphs, achieving a unified framework.
- **vs. Heckman Selection Model**: Heckman's classic work focuses on correcting selection bias in causal inference rather than causal discovery. This paper is the first to introduce selection bias into the toolbox of causal structure learning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Extends rank constraint tools to selection bias scenarios for the first time, filling an important theoretical gap, with highly original intuition regarding the reversed Tetrad.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation with synthetic data and two real-world datasets, though larger-scale synthetic experiments are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Clearly structured from intuitive examples to rigorous theoretical derivations, with a sharp and precise contribution map in Figure 1.
- Value: ⭐⭐⭐⭐⭐ Provides a brand-new theoretical tool for the causal discovery community, opening up a research direction for causal discovery under both latent variables and selection bias.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards a Holistic Understanding of Selection Bias for Causal Effect Identification](../../ICML2026/causal_inference/towards_a_holistic_understanding_of_selection_bias_for_causal_effect_identificat.md)
- [\[ICML 2025\] Causal Abstraction Inference under Lossy Representations](causal_abstraction_inference_under_lossy_representations.md)
- [\[ICML 2026\] Causal Modeling of Selection in Evolution](../../ICML2026/causal_inference/causal_modeling_of_selection_in_evolution.md)
- [\[ICLR 2026\] Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models](../../ICLR2026/causal_inference/distributional_equivalence_in_linear_non-gaussian_latent-variable_cyclic_causal_.md)
- [\[ICLR 2026\] Characterization and Learning of Causal Graphs with Latent Confounders and Post-treatment Selection from Interventional Data](../../ICLR2026/causal_inference/characterization_and_learning_of_causal_graphs_with_latent_confounders_and_post-.md)

</div>

<!-- RELATED:END -->
