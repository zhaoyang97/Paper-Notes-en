---
title: >-
  [Paper Note] Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets
description: >-
  [NeurIPS 2025][Optimization][algebraic structures] This work reveals that the weight space of two-layer quadratic-activation networks trained on Abelian group reasoning tasks possesses a semiring algebraic structure, and proposes the CoGS framework that composes partial solutions into globally optimal solutions via ring operations. Approximately 95% of gradient descent solutions match the theoretical constructions exactly.
tags:
  - NeurIPS 2025
  - Optimization
  - algebraic structures
  - semiring
  - ring homomorphism
  - modular addition
  - Fourier basis
  - global solution composition
  - sum potentials
  - two-layer networks
  - grokking
  - reasoning tasks
date: 2026-05-08
content_hash: 5d4d18f19a4b3758
---

# Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets

**Conference**: NeurIPS 2025
**arXiv**: [2410.01779](https://arxiv.org/abs/2410.01779)
**Code**: [facebookresearch/luckmatters](https://github.com/facebookresearch/luckmatters/tree/yuandong3/ssl/real-dataset)
**Area**: Optimization
**Keywords**: algebraic structures, semiring, ring homomorphism, modular addition, Fourier basis, global solution composition, sum potentials, two-layer networks, grokking, reasoning tasks

## TL;DR

This work reveals that the weight space of two-layer quadratic-activation networks trained on Abelian group reasoning tasks possesses a semiring algebraic structure, and proposes the CoGS framework that composes partial solutions into globally optimal solutions via ring operations. Approximately 95% of gradient descent solutions match the theoretical constructions exactly.

## Background & Motivation

- **Background**: Large language models frequently make surprising errors on elementary reasoning tasks (e.g., the reversal curse), and the nature of reasoning capability remains an open question.
- **Limitations of Prior Work**: Modular addition prediction ($a+b \mod d$) is a canonical probe for studying reasoning; it is structurally simple yet exhibits complex learning phenomena such as grokking. Prior work identified that trained networks employ Fourier basis representations and hand-crafted analytic solutions, but lacked a unified theoretical framework to explain and generalize these findings. Gromov (2023) required an infinite-width assumption, inconsistent with finite-width training in practice. Furthermore, no theoretical explanation was provided for why gradient descent converges to low-order Fourier solutions rather than memorization solutions.
- **Key Challenge**: The internal algebraic structure of the weight space during training had never been explored; geometric deep learning exploits data symmetry but does not open the black box to analyze the weight space itself.
- **Goal**: Establish a systematic algebraic framework that explains the emergence of Fourier solutions, generalizes analytic constructions to finite-width networks, and connects solution structure to gradient dynamics.

## Method

### Overall Architecture: CoGS (Composing Global Solutions)

For $L_2$ loss optimization of two-layer quadratic-activation ($\sigma(x)=x^2$) networks on Abelian group multiplication prediction, CoGS establishes: (1) the weight space $\mathcal{Z}$ across networks of varying widths admits a semiring algebraic structure; (2) the loss decomposes into sum potentials (SP) that are ring homomorphisms; (3) leveraging ring homomorphism properties, partial solutions—each satisfying only a subset of loss constraints—can be algebraically composed via ring addition and multiplication into a globally optimal solution.

### Key Design 1: Semiring Structure of the Weight Space

- **Function**: Defines addition and multiplication operations over the union of weight sets $\mathcal{Z} = \bigcup_{q \geq 0} \mathcal{Z}_q$ across two-layer networks of varying width $q$.
- **Mechanism**: Addition is defined as concatenation along the hidden dimension; multiplication is defined as the Khatri–Rao product (Kronecker product along the hidden dimension). Both operations satisfy commutativity and distributivity, forming a commutative semiring $\langle \mathcal{Z}, +, * \rangle$.
- **Design Motivation**: This unifies networks of different widths within a single algebraic framework, serving as the foundation for all subsequent compositional constructions. Prior work analyzed only fixed-width networks, missing the rich cross-width structure.

### Key Design 2: Sum Potentials as Ring Homomorphisms

- **Function**: Proves that the central term in the $L_2$ loss—the sum potentials $r(\mathbf{z}) = \sum_j \prod_{p,k} z_{pkj}$—is a ring homomorphism from the weight semiring $\mathcal{Z}$ to the complex field $\mathbb{C}$.
- **Mechanism**: Ring homomorphisms satisfy $r(\mathbf{z}_1 + \mathbf{z}_2) = r(\mathbf{z}_1) + r(\mathbf{z}_2)$ and $r(\mathbf{z}_1 * \mathbf{z}_2) = r(\mathbf{z}_1) \cdot r(\mathbf{z}_2)$. Consequently, if partial solution $\mathbf{z}_1$ annihilates some SPs and $\mathbf{z}_2$ annihilates others, the ring product $\mathbf{z}_1 * \mathbf{z}_2$ annihilates the union of both sets.
- **Design Motivation**: This reduces a highly nonlinear global optimization problem to a compositional problem over an algebraic structure, bypassing direct non-convex optimization.

### Key Design 3: Polynomial Construction of Low-Order Global Solutions

- **Function**: Constructs polynomials $\mathbf{z} = \mathbf{u}^L + \mathbf{c}_1 * \mathbf{u}^{L-1} + \cdots + \mathbf{c}_L$ over the semiring $\mathcal{Z}$, systematically generating partial solutions from a single-frequency generator $\mathbf{u}$ and composing them into global solutions.
- **Mechanism**: Symmetry-favorable order-1 generators are selected (e.g., using 3rd- or 4th-order roots of unity $\omega_3, \omega_4$) so that their SP values exhibit extensive repetition (non-unity), allowing low-degree polynomials to cover more constraints. This yields the order-6 Fourier solution $\mathbf{z}_{F6}$ (6 hidden nodes per frequency) and the mixed order-4/6 solution $\mathbf{z}_{F4/6}$ (lower overall order).
- **Design Motivation**: Low-order solutions (fewer hidden nodes) are preferred because gradient dynamics under weight decay naturally favor them—algebraically connected solutions are also topologically connected (Theorem 5), whereas the high-order perfect memorization solution (order $d^2$) is not favored by the dynamics.

### Key Design 4: Gradient Dynamics and Overparameterization Analysis

- **Function**: Analyzes why training favors Fourier solutions over perfect memorization, and characterizes the role of overparameterization.
- **Mechanism**: (1) Theorem 5 proves that if a global solution $\mathbf{z} = \mathbf{y} * \mathbf{z}'$, there exists a zero-loss path between $\mathbf{z}$ and the lower-order $\mathbf{z}'$, so weight decay naturally drives the solution toward lower order. (2) Theorem 6 proves that as width tends to infinity the SP dynamics decouple, explaining why overparameterization improves training.
- **Design Motivation**: Bridges the gap between static solution structure analysis and actual training dynamics, providing a partial explanation for phenomena such as grokking.

## Loss & Training

- **Loss**: Projected $L_2$ loss with quadratic activation; the loss admits an analytic decomposition into sum potentials over each frequency $k$ (Theorem 1).
- **Training**: Adam optimizer, learning rate 0.01, 10,000 epochs, with varying weight decay regularization strengths.
- **Data**: Full synthetic datasets for Abelian group multiplication, with a 90%/10% train/test split.

## Key Experimental Results

### Main Results

#### Table 1: Order Distribution of Gradient Descent Solutions ($q=512$, weight decay $5 \times 10^{-5}$)

| Group size $d$ | Irreducible % | Order-4 ($\mathbf{z}_{\nu=i} * \mathbf{z}_\xi$) | Order-6 ($\mathbf{z}_\nu * \mathbf{z}_{syn}$) | Other % |
|:---:|:---:|:---:|:---:|:---:|
| 23 | 0.0% | 47.07% | 39.80% | 1.82% |
| 71 | 0.0% | 72.57% | 21.14% | 2.29% |
| 127 | 1.50% | 82.96% | 14.13% | 0.66% |

→ The proportion of order-4 solutions increases with $d$, consistent with the theoretical $\mathbf{z}_{F4/6}$ mixed solution (only one order-6 frequency is required to form a global solution).

#### Table 2: Theoretical Matching Accuracy

| Metric | Result |
|:---|:---|
| Decomposability rate | ~95% (decomposition error ~0.04, solution norm on the order of 1) |
| CoGS structure prediction success rate | ~98% (remaining ~2% attributable to insufficient training) |
| Number of order-6 frequencies at $d=127$ | empirically ~1.26, theoretically predicted 1 |

### Ablation Study

- Larger weight decay concentrates the solution distribution toward lower orders (until model collapse), validating the "Occam's Razor" effect of Theorem 5.
- Removing the $R_*$ constraint allows order-3 solutions to suffice, confirming the hierarchical structure of partial solution composition.

## Highlights & Insights

- **First discovery of algebraic structure during training**: Reveals the semiring structure of the weight space and the ring homomorphism property of the loss, establishing an entirely new analytical paradigm.
- **High concordance between theory and experiment**: Approximately 95% of gradient descent solutions are matched exactly to theoretical constructions, not merely in a statistical sense.
- **No infinite-width assumption required**: More concise and practical than Gromov (2023); the order-6 Fourier solution requires only finite width.
- **Explains multiple practical phenomena**: Weight decay preference for simpler solutions, the benefit of overparameterization, and the gradient dynamics' aversion to perfect memorization.

## Limitations & Future Work

- **Restricted to quadratic activation**: Extending to practical activations such as SiLU requires Taylor expansion to generate higher-order SPs; generalization of the framework remains to be verified.
- **Restricted to Abelian groups**: Reasoning tasks over non-commutative groups (e.g., permutation groups) are not addressed.
- **Restricted to two-layer networks**: The extension to deep networks and Transformer architectures is unclear.
- **Evaluated on synthetic tasks only**: Validation is limited to synthetic benchmarks such as modular addition; the connection to real-world reasoning tasks remains to be explored.
- **Does not fully explain grokking**: The dynamical analysis captures only coarse-grained features; the phase transition from memorization to generalization is not completely characterized.

## Related Work & Insights

- **Gromov (2023)**: Manually constructs Fourier solutions but relies on an infinite-width approximation → CoGS requires no such assumption and is more systematic.
- **Morwani et al. (2023)**: Analyzes algebraic tasks via max-margin and $L_{2,3}$ norms → CoGS uses standard $L_2$ loss and discovers the semiring structure.
- **Nanda et al. (2023)**: Extracts circuits via mechanistic interpretability → CoGS provides an algebraic explanation for why circuits exhibit Fourier structure.
- **Geometric deep learning (Bronstein et al., 2021)**: Incorporates symmetry into architectures → CoGS discovers algebraic structure that spontaneously emerges in the weight space during training.
- **Li et al. (2024), Liu et al. (2022)**: Construct Transformer weights to implement automata → Do not verify that the constructions coincide with gradient descent solutions.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to reveal the semiring algebraic structure of the weight space during training, inaugurating a new analytical paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 95% exact matching across multiple values of $d$, but limited to synthetic modular addition tasks.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are rigorous and clear, though the dense algebraic notation poses a high entry barrier.
- **Value**: ⭐⭐⭐⭐ — Provides a novel algebraic perspective for understanding network reasoning capabilities, with potential future impact on training algorithm design.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] DynaAct: Large Language Model Reasoning with Dynamic Action Spaces](dynaact_large_language_model_reasoning_with_dynamic_action_spaces.md)
- [\[NeurIPS 2025\] A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning](a_theoretical_study_on_bridging_internal_probability_and_sel.md)
- [\[NeurIPS 2025\] Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers](generalization_or_hallucination_understanding_out-of-context_reasoning_in_transf.md)
- [\[NeurIPS 2025\] OrthoGrad Improves Neural Calibration](orthograd_improves_neural_calibration.md)
- [\[NeurIPS 2025\] Verbalized Algorithms: Zero-shot Classical Algorithmic Reasoning for Correctness and Runtime Guarantees](verbalized_algorithms.md)

<!-- RELATED:END -->
