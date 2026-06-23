---
title: >-
  [Paper Note] Finite-Width Neural Tangent Kernels from Feynman Diagrams
description: >-
  [ICML 2026][learning_theory][Paper Note] This paper adapts Feynman diagrams from quantum field theory to neural network analysis, providing a graphical framework of rules for the "finite-width statistical corrections of NTK." This transforms extremely tedious layer-wise recursive derivations into a "draw and translate" process. It proves the critical stabilit
tags:
  - ICML 2026
  - learning_theory
date: 2026-05-08
content_hash: 0e0a9e52b98b932d
---
# Finite-Width Neural Tangent Kernels from Feynman Diagrams

**Conference**: ICML2026  
**arXiv**: [2508.11522](https://arxiv.org/abs/2508.11522)  
**Code**: https://github.com/PhilippMisofCH/ntk-unlimited  
**Area**: Learning Theory / Neural Tangent Kernel / Finite-width corrections  
**Keywords**: Neural Tangent Kernel, Finite-width, Feynman diagrams, 1/n expansion, Critical initialization

## TL;DR
This paper adapts Feynman diagrams from quantum field theory to neural network analysis, providing a graphical framework of rules for the "finite-width statistical corrections of NTK." This transforms extremely tedious layer-wise recursive derivations into a "draw and translate" process. It proves the critical stability of NTK and demonstrates that scale-invariant activations like ReLU have no finite-width corrections on the diagonal. Numerically, the results align with sampled networks at widths $n \gtrsim 20$.

## Background & Motivation

**Background**: The Neural Tangent Kernel $\Theta(x,x')=J(x)J(x')^\top$ (where $J$ is the Jacobian of the network with respect to parameters) characterizes the first-order training dynamics of a network. In the **infinite-width** limit, the NTK collapses to its mean and remains frozen during training (frozen NTK), allowing for closed-form recursive calculation across layers. Consequently, gradient descent and Bayesian inference can be solved analytically.

**Limitations of Prior Work**: The convenience of infinite width is also its weakness—at this limit, the network is effectively linearized into a Gaussian process, where only the last layer evolves during training, meaning there is **no feature learning**. Multiple studies have found significant deviations between the behavior of real finite-width networks and the predictions of infinite-width NTKs.

**Key Challenge**: To incorporate phenomena like feature learning and NTK evolution into theory, one must perform a $1/n$ Taylor expansion (where $n$ is the hidden layer width) beyond the strict infinite-width limit, correcting Gaussian statistics toward non-Gaussianity. However, the algebraic derivation of such expansions is **extraordinarily long** and uses language unfamiliar to the machine learning community, hindering its adoption.

**Goal**: To provide a **systematic and concise** set of tools for calculating the finite-width statistical corrections of NTK (and related high-order tensors), enabling the mechanical derivation of layer-wise recursions for any $1/n$ order correction.

**Key Insight**: Infinite-width statistics are Gaussian; a $1/n$ expansion is essentially a "perturbative expansion around a Gaussian distribution." This is the standard approach in perturbative quantum field theory (QFT), where the action (log-probability) is expanded around a Gaussian lead term (non-interacting particles), and non-Gaussian corrections correspond to interactions. Physicists use **Feynman diagrams** as an intuitive shorthand for these calculations.

**Core Idea**: Tailor a set of "Feynman rules" specifically for NTK statistics. The expectation values to be calculated are drawn as a set of compatible diagrams, which are then translated into algebraic expressions according to the rules. This reduces complex recursive derivations to the task of drawing diagrams.

## Method

### Overall Architecture
The paper investigates two kernels of an $L$-layer MLP at initialization: the empirical NTK $\widehat\Theta^{(\ell)}_{ij}(x,x')=\sum_\mu \frac{\partial z_i^{(\ell)}(x)}{\partial\theta_\mu}\frac{\partial z_j^{(\ell)}(x')}{\partial\theta_\mu}$ (capturing gradient correlations) and the NNGP kernel $\widehat K^{(\ell)}_{ij}(x,x')=z_i^{(\ell)}(x)z_j^{(\ell)}(x')$ (capturing pre-activation correlations). Both are random variables due to stochastic initialization. At infinite width, the NTK is frozen and fluctuations $\widehat{\Delta\Theta}=0$. To go beyond this, the paper expands in $1/n$, using **joint cumulants** (connected correlators in physics) to characterize mixed moments. These are decomposed by neural indices into a set of rank-4 tensors ($A,B,D,F$), a four-point cumulant $V$, and first-order mean corrections $K^{\{1\}},\Theta^{\{1\}}$. These objects fully determine the statistics of pre-activations and NTK at the $1/n$ order. The method assigns Feynman rules to these cumulants, allowing their layer-wise recursion relations to be derived graphically.

### Key Designs

**1. $1/n$ Expansion + Tensor Decomposition: Reducing statistics to countable blocks**

Addressing the "lack of feature learning in infinite width," the paper performs a Taylor expansion in $1/n$. The leading term $1/n \to 0$ represents Gaussian process behavior, while first-order corrections introduce nonlinearity and feature evolution. Since NTK fluctuations $\widehat{\Delta\Theta}=0$ hold at infinite width, mixed moments like $\mathbb{E}_\theta[z_{i_1}^{(\ell)}z_{i_2}^{(\ell)}\widehat{\Delta\Theta}^{(\ell)}_{i_3i_4}]$ are naturally of order $1/n$. The paper decomposes joint cumulants $\mathbb{E}^c$ into rank-4 tensors. For example, the joint pre-activation-NTK cumulant is written as:

$$\mathbb{E}^c_\theta[z^{(\ell+1)}_{i_1},z^{(\ell+1)}_{i_2},\widehat{\Delta\Theta}^{(\ell+1)}_{i_3i_4}]=\tfrac{1}{n_\ell}\big(D^{(\ell+1)}_{1234}\delta_{i_1i_2}\delta_{i_3i_4}+F^{(\ell+1)}_{1324}\delta_{i_1i_3}\delta_{i_2i_4}+F^{(\ell+1)}_{1423}\delta_{i_1i_4}\delta_{i_2i_3}\big),$$

where $D,F$ are Gram tensors over sample indices. This decomposition compresses a vast number of mixed moments into a finite set of recursive tensors.

**2. NTK Feynman Rules: Turning expectation calculations into "Drawing + Translation"**

Previous work relied heavily on the Gaussianity of conditional distributions $P(z^{(\ell+1)}\mid z^{(\ell)})$, which fails for NTK because it is quadratic in weights. This paper redefines rules based on the cumulant-tensor decomposition: external vertices are solid dots (solid line = pre-activation $z_\alpha$, dotted line = NTK fluctuation $\widehat{\Delta\Theta}_{\alpha\beta}$); external lines connect to **cubic interaction vertices** (carrying factors of order $C_W^{(\ell+1)}/n_\ell$); internal lines connect to **propagators** representing Gaussian expectations (white blobs). Rank-4 tensors are represented as **quartic interaction vertices**. To calculate an expectation, one draws all compatible diagrams at a given $1/n$ order and sums their translated algebraic values.

**3. Completeness Theorem: Rules providing correct recursions at all orders**

The rigor of these rules is established via three theorems: Theorem 4.1 proves that the Feynman rules **uniquely determine** the recursions for $D,F,A,B$ at order $1/n$. Theorem 4.2 extends the rules to higher-order derivative tensors (dNTK / ddNTK). Theorem 4.3 demonstrates that by adding higher-order generalizations of the tensors, the rules can fully characterize the statistics of NTK and its derivatives at **all orders** of $1/n$.

**4. Three Applications: New recursions, Stability, and ReLU precision**

- **NTK Mean Recursion (5.1)**: Graphically derived the first-order correction $\Theta^{\{1\}}$ for the NTK mean—a recursion relation that, according to the authors, has **never been derived before**.
- **Finite-Width Gradient Stability (5.2)**: Theorem 5.1 proves that **if the NNGP is critical, any cumulant involving NTK is also critical**, extending previous forward-pass stability results to the backward pass (gradients).
- **No Correction for Scale-Invariant Activations (5.3)**: Theorem 5.2 proves that for scale-invariant activations like ReLU ($ \sigma(\lambda z)=\lambda\sigma(z) $), the **diagonal components of the NTK mean receive no finite-width corrections**.

### Loss & Training
This work focuses on statistical analysis at initialization and does not involve training loss directly. Numerically, the recursions involve high-dimensional Gaussian integrals without analytical solutions. The paper uses a custom SymPy routine to reduce symbolic expressions, utilizes multivariate Gaussian marginals to simplify 4D integrals, and employs tensor contraction to reduce terms, resulting in a flexible, activation-agnostic framework.

## Key Experimental Results

### Main Results
Numerical solutions of the recursions were compared against statistics from sampled networks.

| Validation Item | Setting | Result |
|:---|:---|:---|
| NNGP/NTK 1st-order Correction | GeLU-MLP Layer 4 off-diagonal, MC samples $10^6$/$10^5$ | $K+K^{\{1\}}/n$ and $\Theta+\Theta^{\{1\}}/n$ curves fit sampled means significantly better than infinite-width results |
| Convergence Width | Various hidden widths $n$ | Corrected statistics match sampled networks starting at $n \gtrsim 20$ |
| Critical Stability | Sampled tensors across depths | Critical initialization stabilizes all NTK-involved statistics across all orders of $1/n$ |

### Ablation Study

| Activation Function | Scale Invariant | $\Theta(x,x)$ Finite-Width Correction |
|:---|:---|:---|
| ReLU | Yes | None (Matches Theorem 5.2) |
| LeakyReLU | Yes | None |
| GeLU | No | Exists (As a counter-example) |

### Key Findings
- **First-order corrections capture finite-width behavior**: Adding $1/n$ corrections results in kernels that are significantly closer to the true statistics of sampled networks, even at small widths ($n \gtrsim 20$).
- **Critical stability extends to the backward pass**: Proved that all statistics involving NTK remain stable at the NNGP criticality, covering the gradient statistics upon which training depends.
- **Scale invariance is a source of structural precision**: The diagonal NTK of ReLU/LeakyReLU has zero finite-width correction, suggesting that the infinite-width approximation is "fortuitously exact" for these specific quantities.

## Highlights & Insights
- **Interdisciplinary Method Transfer**: Bringing Feynman diagrams from QFT into NTK analysis changes the "accounting method" of the theory—replacing pages of algebraic derivations with diagrammatic enumeration.
- **Comprehensive Coverage**: Unlike previous work limited to pre-activation statistics, this framework covers NTK and joint statistics, which are essential for describing **training dynamics**.
- **Full-Order Completeness**: The rules are proven to be correct at any order of $1/n$, and the provided open-source SymPy framework allows for the practical calculation of these corrections.

## Limitations & Future Work
- **Limited to MLPs**: The tensor recursions and Feynman rules are derived specifically for fully connected networks; rules for CNNs or Transformers are not yet provided.
- **Focus on Initialization**: The analysis targets the kernels at initialization. While this informs first-order dynamics, a full description of training trajectories requires even higher-order tensors.
- **Numerical Cost**: Although reduction techniques are used, the cost of high-dimensional Gaussian integration and tensor contraction may become expensive at higher $1/n$ orders.

## Related Work & Insights
- **vs. Roberts et al. (2022) / Yaida (2020)**: They systematically developed the $1/n$ expansion using direct algebra; this paper provides equivalent but **significantly simplified** graphical tools.
- **vs. Banta et al. (2024)**: This work expands beyond their **pre-activation-only** Feynman rules to include NTK and joint statistics—a crucial jump for describing training.
- **vs. Dyer & Gur-Ari (2020)**: While they used diagrams to study scaling behavior, this paper provides **explicit values** and numerically implementable recursions.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First expansion of Feynman diagrams to joint NTK statistics with proven full-order completeness.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Strong validation on MLPs and stability, though limited to initialization.
- **Writing Quality**: ⭐⭐⭐⭐ Clear integration of physical intuition and mathematical theorems, though the bar for understanding Feynman rules remains high.
- **Value**: ⭐⭐⭐⭐⭐ Provides a reusable, extensible, and open-source computational tool for finite-width theory.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>
<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] Function Spaces Without Kernels: Learning Compact Hilbert Space Representations](../../ICLR2026/learning_theory/function_spaces_without_kernels_learning_compact_hilbert_space_representations.md)
- [\[ICLR 2026\] A Statistical Learning Perspective on Semi-dual Adversarial Neural Optimal Transport Solvers](../../ICLR2026/learning_theory/a_statistical_learning_perspective_on_semi-dual_adversarial_neural_optimal_trans.md)
- [\[NeurIPS 2025\] Finite-Time Analysis of Stochastic Nonconvex Nonsmooth Optimization on the Riemannian Manifolds](../../NeurIPS2025/learning_theory/finite-time_analysis_of_stochastic_nonconvex_nonsmooth_optimization_on_the_riema.md)
- [\[ICML 2026\] A Perturbation Approach to Unconstrained Linear Bandits](a_perturbation_approach_to_unconstrained_linear_bandits.md)
- [\[ICML 2026\] Correcting Split Selection in Online Decision Trees via Anytime-Valid Inference](correcting_split_selection_in_online_decision_trees_via_anytime-valid_inference.md)

</div>

<!-- RELATED:END -->
