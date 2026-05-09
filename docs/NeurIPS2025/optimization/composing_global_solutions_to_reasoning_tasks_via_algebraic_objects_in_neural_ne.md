---
title: >-
  [Paper Note] Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets
description: >-
  [NeurIPS 2025][Optimization][Algebraic structure] This paper proposes the CoGS framework, demonstrating that the weight space of two-layer quadratic-activation networks on Abelian group multiplication reasoning tasks admits a semiring algebraic structure. The Sum Potentials in the loss function are ring homomorphisms, enabling global optimal solutions to be algebraically composed from partial solutions—each satisfying only a subset of loss constraints—via ring addition and multiplication. Approximately 95% of gradient descent solutions match the theoretical constructions exactly.
tags:
  - NeurIPS 2025
  - Optimization
  - Algebraic structure
  - modular arithmetic reasoning
  - two-layer networks
  - grokking
  - weight-space semiring
  - global optimal solution construction
date: 2026-05-08
content_hash: adbb8292590bca1d
---

# Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets

**Conference**: NeurIPS 2025
**arXiv**: [2410.01779](https://arxiv.org/abs/2410.01779)
**Code**: [facebookresearch/luckmatters](https://github.com/facebookresearch/luckmatters/tree/yuandong3/ssl/real-dataset)
**Area**: Optimization
**Keywords**: Algebraic structure, modular arithmetic reasoning, two-layer networks, grokking, weight-space semiring, global optimal solution construction

## TL;DR

This paper proposes the CoGS framework, demonstrating that the weight space of two-layer quadratic-activation networks on Abelian group multiplication reasoning tasks admits a semiring algebraic structure. The Sum Potentials in the loss function are ring homomorphisms, enabling global optimal solutions to be algebraically composed from partial solutions—each satisfying only a subset of loss constraints—via ring addition and multiplication. Approximately 95% of gradient descent solutions match the theoretical constructions exactly.

## Background & Motivation

- **Background**: Large language models excel at complex reasoning yet still make surprisingly basic arithmetic errors; understanding how models execute reasoning is a central open problem.
- **Limitations of Prior Work**: Modular addition (predicting $a+b \bmod d$) is a widely adopted benchmark due to its structural simplicity and rich training dynamics (e.g., grokking, Fourier basis representations). Prior work can construct or reverse-engineer Fourier solutions (Gromov 2023, Nanda 2023) but relies on infinite-width approximations and lacks a systematic algebraic construction methodology.
- **Key Challenge**: The algebraic structure of the weight space itself during training has not been explored; geometric deep learning exploits symmetries in data but does not open the black box to study the weight space. Directly solving global optimality conditions is intractable due to high nonlinearity.
- **Goal**: Develop a systematic algebraic framework that (1) characterizes the semiring structure of the weight space, (2) identifies loss components as ring homomorphisms, and (3) composes global solutions from local partial solutions algebraically. Additionally, provide rigorous theoretical explanations for the benefits of overparameterization and the role of weight decay.

## Method

### Overall Architecture: CoGS (Composing Global Solutions)

The core idea of CoGS is to avoid solving the highly nonlinear global optimization problem directly, and instead **decompose global solutions into local solutions satisfying only partial loss constraints, then compose them into a global solution via algebraic operations**. The technical pipeline proceeds in three steps: (1) prove the weight space admits a semiring structure; (2) prove the loss decomposes into Sum Potentials that are ring homomorphisms; (3) leverage ring homomorphism properties to compose global solutions from lower-order local solutions.

### Key Design 1: Semiring Structure of the Weight Space

- **Function**: Define addition (concatenation along the hidden dimension) and multiplication (Khatri–Rao product) on the weight space $\mathcal{Z} = \bigcup_{q \geq 0} \mathcal{Z}_q$ (where $q$ denotes the number of hidden nodes), and prove that $\langle \mathcal{Z}, +, * \rangle$ forms a commutative semiring.
- **Mechanism**: Addition corresponds to concatenating networks of different widths; multiplication corresponds to outer-product expansion of hidden nodes. Addition satisfies commutativity (via permutation equivalence); multiplication satisfies associativity and distributivity.
- **Design Motivation**: The semiring structure enables solutions to be constructed in the weight space analogously to polynomial arithmetic—low-order "generators" are combined via ring operations to produce high-order global solutions.

### Key Design 2: Sum Potentials and Ring Homomorphisms

- **Function**: Analytically decompose the $L_2$ loss as $\ell = d^{-1} \sum_{k \neq 0} \ell_k$, where each $\ell_k$ is expressed in terms of Sum Potentials (SP) $r_{k_1 k_2 k}(\mathbf{z})$ and $r_{p k_1 k_2 k}(\mathbf{z})$, and prove that all SPs are ring homomorphisms.
- **Mechanism**: An SP is defined as $r(\mathbf{z}) := \sum_j \prod_{p,k \in \text{idx}(r)} z_{pkj}$, a monomial summed over hidden nodes. Ring homomorphism properties imply $r(\mathbf{z}_1 + \mathbf{z}_2) = r(\mathbf{z}_1) + r(\mathbf{z}_2)$ and $r(\mathbf{z}_1 * \mathbf{z}_2) = r(\mathbf{z}_1) \cdot r(\mathbf{z}_2)$.
- **Design Motivation**: Ring homomorphisms guarantee that the zero-sets of local solutions can be intersected via addition and unioned via multiplication, progressively expanding until all constraints are satisfied, thereby yielding a global solution.

### Key Design 3: Polynomial Construction and Global Solution Composition

- **Function**: Select order-1 generators $\mathbf{u}$ with favorable symmetry (e.g., $\mathbf{u}_{\text{syn}}, \mathbf{u}_\nu$), construct polynomials $\boldsymbol{\rho}(\mathbf{u}) = \prod_{s}(\mathbf{u} + \hat{\mathbf{s}})$ as local solutions, and compose them into order-4 ($2 \times 2$) and order-6 ($2 \times 3$) Fourier global solutions.
- **Mechanism**: The order-6 solution $\mathbf{z}_{F6}$ combines order-3 and order-2 local solutions for each frequency via ring multiplication to cover all SP constraints; the mixed order-4/6 solution $\mathbf{z}_{F4/6}$ further reduces the total order to $2d$ by exploiting SP cancellations across different frequencies to achieve global optimality.
- **Design Motivation**: Gradient descent naturally favors lower-order solutions (Theorem 5), motivating the construction of the lowest-order global solutions possible to explain empirical observations.

### Key Design 4: Training Dynamics Analysis

- **Function**: Prove that high- and low-order global solutions are topologically connected on the zero-loss manifold (Theorem 5), implying that gradient descent under weight decay favors lower-order solutions; prove that SP dynamics decouple in the infinite-width limit (Theorem 6), explaining the benefits of overparameterization.
- **Mechanism**: If $\mathbf{z} = \mathbf{y} * \mathbf{z}'$ and both are global solutions, then a zero-loss path connecting them exists, and weight decay regularization drives dynamics toward the lower-order end.
- **Design Motivation**: Addresses the key question of why gradient descent converges to order-4/6 Fourier solutions rather than order-$d^2$ perfect memorization solutions.

## Loss & Training

- **Loss Function**: Standard $L_2$ mean squared error loss, computed after zero-mean projection of the predicted Abelian group multiplication outputs.
- **Optimizer**: Adam, learning rate 0.01.
- **Regularization**: $L_2$ weight decay (multiple values tested in experiments, including $5 \times 10^{-5}$).
- **Training Duration**: 10,000 epochs.
- **Data Split**: 90% training / 10% test, entirely synthetically generated.

## Key Experimental Results

### Table 1: Matching Rate Between Gradient Descent Solutions and CoGS Theoretical Constructions ($q=512$, weight decay $5 \times 10^{-5}$)

| Group size $d$ | Indecomposable (%) | Decomposition error ($\times 10^{-2}$) | Order-4 (%) | Order-6 (%) | $\mathbf{z}_{\nu=i} * \mathbf{z}_\xi$ (%) | $\mathbf{z}_\nu * \mathbf{z}_{\text{syn}}$ (%) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 23 | 0.0 | 0.05 | 47.07 | 39.80 | 47.07 | 39.80 |
| 71 | 0.0 | 0.03 | 72.57 | 21.14 | 72.57 | 21.14 |
| 127 | 1.50 | 0.26 | 82.96 | 14.13 | 82.96 | 14.13 |

> Approximately 95% of gradient descent solutions are decomposable and match the theoretical constructions exactly. The proportion of order-4 solutions increases with $d$, consistent with the theoretical prediction for $\mathbf{z}_{F4/6}$ mixed solutions.

### Table 2: Effect of Weight Decay Strength on Solution Distribution ($d \in \{23, 71, 127\}$, $q=512$)

| Setting | Effect |
|:--|:--|
| Increasing weight decay | Solution distribution shifts toward lower-order (order-4) solutions, until model collapse (weights approach zero) |
| Substantial overparameterization ($q=512$ vs. minimum $\sim 2d$ nodes required) | Final solution order remains constant at 4/6, independent of network width |
| Removing the $R_*$ constraint term | Only 3 hidden nodes per frequency required (order-3 $\mathbf{z}_{\text{syn}}$ solution) |
| Grokking phenomenon | Training accuracy reaches 100% first; test accuracy jumps to 100% after thousands of additional epochs |

## Highlights & Insights

1. **Pioneering algebraic perspective**: This work is the first to rigorously identify and exploit a semiring algebraic structure in the weight space of neural networks during training, reformulating a nonlinear optimization problem as an algebraic composition problem.
2. **Strong theory-practice agreement**: Approximately 95% of empirical gradient descent solutions match the algebraically constructed solutions exactly, validating the practical relevance of the theory.
3. **Elegant explanations for complex phenomena**: Ring homomorphisms and topological connectivity elegantly explain the benefits of overparameterization, the preference for simpler solutions under weight decay, and aspects of the grokking phenomenon.
4. **Potential for a new training paradigm**: CoGS suggests the possibility of bypassing gradient descent entirely and constructing solutions directly via algebraic decomposition and composition, potentially with greater efficiency.

## Limitations & Future Work

1. **Restricted to quadratic activations**: The theory strictly requires $\sigma(x)=x^2$; extensions to practical activations such as ReLU or SiLU rely on Taylor approximations.
2. **Restricted to two-layer networks**: The framework has not been extended to deeper networks or Transformer architectures.
3. **Restricted to Abelian groups**: Reasoning tasks over non-commutative groups (e.g., permutation groups) are not covered.
4. **Grokking not fully explained**: The work provides partial insights but does not directly model dynamics under non-uniform training data distributions.
5. **Limited experimental scale**: The largest $d$ tested is 127; the theoretical predictions have not been validated at larger scales.

## Related Work & Insights

- **Gromov (2023)** constructs analytic Fourier solutions for modular arithmetic but relies on infinite-width approximations; the constructions in this paper are more concise and exact at finite width.
- **Morwani et al. (2023)** analyze algebraic tasks using a max-margin framework with $L_{2,3}$ norms, but neither identify algebraic structure in the weight space nor analyze training dynamics.
- **Nanda et al. (2023)** extract circuits through mechanistic interpretability in a bottom-up empirical manner; this paper adopts a top-down theoretical construction approach.
- **Insight**: The paradigm of "decomposing nonlinear optimization into composable subproblems via algebraic structure" may generalize to broader representation learning theory.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to discover and exploit a semiring algebraic structure in neural network training; entirely novel perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across multiple values of $d$, multiple weight decay settings, and 95% matching rate; coverage of model scales and architectures remains limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematically rigorous with clear figures, though the dense algebraic notation poses a high barrier for non-specialist readers.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a fundamentally new algebraic theoretical toolkit for understanding neural network reasoning mechanisms, with potential to open new research directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DynaAct: Large Language Model Reasoning with Dynamic Action Spaces](dynaact_large_language_model_reasoning_with_dynamic_action_spaces.md)
- [\[NeurIPS 2025\] A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning](a_theoretical_study_on_bridging_internal_probability_and_sel.md)
- [\[NeurIPS 2025\] OrthoGrad Improves Neural Calibration](orthograd_improves_neural_calibration.md)
- [\[NeurIPS 2025\] Verbalized Algorithms: Zero-shot Classical Algorithmic Reasoning for Correctness and Runtime Guarantees](verbalized_algorithms.md)
- [\[NeurIPS 2025\] Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers](generalization_or_hallucination_understanding_out-of-context_reasoning_in_transf.md)

</div>

<!-- RELATED:END -->
