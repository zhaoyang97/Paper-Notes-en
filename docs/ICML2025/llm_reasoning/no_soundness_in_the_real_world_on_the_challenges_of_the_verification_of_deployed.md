---
title: >-
  [Paper Note] No Soundness in the Real World: On the Challenges of the Verification of Deployed Neural Networks
description: >-
  [ICML 2025][Reasoning][Neural network verification] This paper demonstrates that all current state-of-the-art neural network verifiers only provide "theoretical soundness" (bounding exact-precision output) rather than "practical soundness" (bounding floating-point outputs in deployment environments), and empirically verifies that all tested verifiers can be deceived by constructing environment-sensitive adversarial backdoor networks.
tags:
  - "ICML 2025"
  - "Reasoning"
  - "Neural network verification"
  - "floating-point arithmetic"
  - "soundness"
  - "deployment environment"
  - "adversarial backdoor"
  - "interval analysis"
date: 2026-05-08
content_hash: 3bae76e0daebf7c9
---

# No Soundness in the Real World: On the Challenges of the Verification of Deployed Neural Networks

**Conference**: ICML 2025  
**arXiv**: [2506.01054](https://arxiv.org/abs/2506.01054)  
**Code**: [https://github.com/szasza1/no_soundness](https://github.com/szasza1/no_soundness)  
**Area**: Audio and Speech  
**Keywords**: Neural network verification, floating-point arithmetic, soundness, deployment environment, adversarial backdoor, interval analysis

## TL;DR

This paper demonstrates that all current state-of-the-art neural network verifiers only provide "theoretical soundness" (bounding exact-precision output) rather than "practical soundness" (bounding floating-point outputs in deployment environments), and empirically verifies that all tested verifiers can be deceived by constructing environment-sensitive adversarial backdoor networks.

## Background & Motivation

### Goals of Neural Network Verification

The goal of neural network verification is to provide mathematical proofs guaranteeing that a network possesses certain safety properties (such as adversarial robustness). For classification networks, the core verification task is: given an input $x^*$ and its $\epsilon$-neighborhood $D_{\epsilon,p}(x^*)$, prove that all inputs within the neighborhood are assigned to the same class.

### Theoretical Model vs. Deployed Model

All current verification methods target the **theoretical model** of the network, which is a mathematical abstraction using real-precision real arithmetic. However, in actual deployment:
- Hardware utilizes **floating-point arithmetic** (IEEE 754), introducing rounding errors.
- Floating-point addition is **non-associative**: different summation orders for the same addition operations can yield different results.
- Parallelization, different frameworks (PyTorch vs. Flux), varying hardware (CPU vs. GPU), and different batch sizes all alter the expression tree of the operations.
- The deployment environment can be **stochastic** (due to non-deterministic operation order in parallel computing).

### Key Insight

$$\text{Theoretical Soundness} \neq \text{Practical Soundness}$$

That is, a verifier that correctly bounds the exact-precision output does not necessarily bound the actual output under floating-point deployment. This implies that an attacker can design a network backdoor that is completely invisible during theoretical model verification, yet becomes activated to exhibit malicious behavior in specific deployment environments.

### Why does this problem matter?

- Verifiers are used in safety-critical applications (such as autonomous driving and medical diagnostics); if verification results are unreliable, the consequences are severe.
- All current mainstream verifiers (such as β-CROWN, MIPVerify, DeepPoly, etc.) suffer from this issue.
- The problem is not merely floating-point precision errors, but the fundamental nature of the **non-associativity** and **environmental dependency** of floating-point arithmetic.

## Method

### Overall Architecture

The paper's demonstration framework is divided into three levels:

1. **Formal Modeling**: Defining the deployed network $r(x;\theta,\mathcal{E})$, as distinct from the theoretical network $f(x;\theta)$.
2. **Theoretical Proof**: Proving that verification methods based on Interval Bound Propagation (IBP) and symbolic propagation cannot guarantee practical soundness.
3. **Empirical Attack**: Constructing adversarial backdoor networks to verify that all mainstream verifiers can be deceived.

### Key Design 1: Formalization of the Deployment Verification Problem

**Theoretical verification problem** (solved by existing work):

$$\forall x \in D_{\epsilon,p}(x^*),\ f(x;\theta) \geq 0$$

**Deployment verification problem** (proposed to be solved in this paper):

$$\forall x \in D_{\epsilon,p,\mathcal{E}}(x^*)\ \forall z \in r(x;\theta,\mathcal{E}),\ z \geq 0$$

Key differences between the two:
- **Different functions**: $r \neq f$, where the deployed version produces different outputs due to floating-point rounding.
- **Different domains**: $D_{\epsilon,p,\mathcal{E}} = D_{\epsilon,p} \cap X$, where $X$ depends on the numerical representation of the environment.
- **Different properties**: $P_\mathcal{E} \neq P \cap X$, as some negative outputs of $f$ may become zero in $r$.
- **Stochasticity**: In stochastic environments, $r(x;\theta,\mathcal{E})$ may be a **set** (the set of all output vectors with probability > 0).

### Key Design 2: Theoretical Proof - Unsoundness of Interval Propagation

The paper focuses on the simplest class of functions—the **sum of input variables**—to prove that even for such simple functions, verification cannot guarantee practical soundness.

**Proposition 6.1 (Positive Result)**: If the deterministic expression tree of the deployment environment is known, and IBP is executed according to this expression tree, then IBP is practically sound.

**Proposition 6.2 (Core Negative Result)**: For any environment $\mathcal{E}$ using IEEE 754 floating-point arithmetic that allows all legal expression trees, there exists an expression tree $o$ and an input $x$ such that the IBP interval $[a,b]_o$ cannot cover the actual minimum/maximum output of the deployment.

**Proposition 6.3 (Ruling out Simple Fixes)**: Summing in decreasing order or decreasing absolute-value order **cannot** guarantee obtaining the minimum output.

**Corollary**: Finding an expression tree that minimizes/maximizes the floating-point output is highly likely to be an NP-hard problem (based on a similar conclusion by Kao & Wang 2000).

### Key Design 3: Symbolic Propagation Methods are Also Unsound

- **Polyhedra methods** (DeepPoly, CROWN): Degenerate to IBP when computing the sum of input variables, and are thus also practically unsound.
- **Zonotope methods** (DeepZ, RefineZono): Utilize widening techniques of affine arithmetic, resulting in wider intervals than IBP, but **still cannot guarantee covering all possible outputs** (Propositions 6.4 and 6.5).

### Key Design 4: Adversarial Backdoor Networks

The paper designs a **detector neuron** that exploits floating-point arithmetic properties to detect the deployment environment:

#### Precision Detector

Leveraging a special number $\omega$ (the smallest positive number such that the next representable number is $\omega+2$):
- binary32 format: $\omega = 2^{24}$
- binary64 format: $\omega = 2^{53}$

The detector $\omega + 1 - \omega$ outputs 0 under the target precision, and 1 under higher precision.

#### Expression Tree Detectors (Three Increasing Difficulty Levels)

| Detector | Principle | Difficulty for the Verifier |
|--------|------|---------------|
| Order1 | $(2h_1+1)h_2$ summation terms alternately arranging positive and negative $\omega/h_1$ | Easy: Covered by IBP with default ordering |
| Order2 | $h+2$ summation terms $2/h + ... + \omega - \omega$ | Medium: IBP with default ordering does not cover 0 |
| Order3 | $h+2$ summation terms $1 + ... + 1 + \omega - \omega$ | Hard: Very few expression trees cover 0 |

Backdoor embedding method: Adopts the scheme from Zombori et al. (2021) to insert the detector neuron into a normal MNIST classification network. When the detector outputs the trigger value, the network behavior changes arbitrarily (e.g., flipping the classification result).

### Theoretical Analysis

The paper's theoretical contributions can be summarized with the following hierarchical logic:

1. **Floating-point non-associativity $\rightarrow$ Operation order affects results**: The same mathematical expression can produce different floating-point outputs under different operation trees.
2. **Deployment environment $\rightarrow$ Determines operation order**: Different hardware, software, frameworks, and batch sizes select different operation trees.
3. **IBP/Symbolic propagation only tracks a constitutional operation tree**: The verifier's intervals might not cover the outputs of other operation trees.
4. **Attackers can exploit this gap**: Designing backdoors that activate under specific operation trees while remaining invisible under the verifier's expression tree.
5. **Fundamental difficulty**: Exhaustively searching all possible operation trees is NP-hard.

## Key Experimental Results

### Experimental Settings

- **Host Network**: The MNIST network from Wong & Kolter (2018), 2 convolutional layers (16 and 32 filters, stride 2) + a 100-neuron fully connected layer, ReLU activation, adversarially trained under an $l_\infty$ radius of 0.1.
- **Five Adversarial Networks**: 2 precision attacks + 3 expression tree attacks (Order1/2/3).
- **Tested Verifiers**: MIPVerify, MN-BAB, β-CROWN BaB, GCP-CROWN, DeepPoly, RefinePoly, DeepZ, RefineZono (8 in total).
- **Verification Object**: The first 100 samples of the MNIST test set.

### Backdoor Network Effectiveness

| Attack Type | 32-bit Precision | 64-bit Precision |
|---------|-----------|-----------|
| Precision Attack (32-bit adversarial) | 0.11% | 98.11% |
| Precision Attack (64-bit adversarial) | 98.11% | 0.11% |

The precision attack succeeds perfectly: under the target precision, the accuracy drops to 0.11% (almost completely flipping the classification), while maintaining a normal accuracy of 98.11% under non-target precision.

### Performance of Expression Tree Attacks in Different Environments

| Attack | Batch | PyTorch CPU | PyTorch GPU | Flux CPU | Flux GPU |
|------|-------|-------------|-------------|----------|----------|
| Order1 | 1 | 0.11% | 0.11% | 98.11% | 0.11% |
| Order2 | 1 | 98.11% | 98.11% | 98.11% | 0.11% |
| Order3 | 1 | 98.11% | 98.11% | 98.11% | 98.11% |
| Order1 | 10 | 0.11% | 98.11% | 0.11% | 98.11% |
| Order2 | 10 | 98.11% | 98.11% | 98.11% | 98.11% |
| Order3 | 10 | 98.11% | 98.11% | 98.11% | 98.11% |

Order1 is activated in multiple environments (0.11%), Order2 is activated on Flux GPU with batch=1, and Order3 is designed to be the hardest to detect, being activated in almost no environment.

### Verifier Attacks Results

| Verifier | Verification Environment | Propagation Method | Precision Attack | Order1 | Order2 | Order3 |
|--------|---------|---------|--------------|--------|--------|--------|
| MIPVerify | 64-bit CPU | IBP | **unsound** | sound | **unsound** | **unsound** |
| MN-BAB | 64-bit GPU | Polyhedra | **unsound** | sound | **unsound** | **unsound** |
| β-CROWN (32-bit) | 32-bit CPU | Polyhedra | **unsound** | sound | **unsound** | **unsound** |
| β-CROWN (64-bit CPU) | 64-bit CPU | Polyhedra | **unsound** | sound | **unsound** | **unsound** |
| β-CROWN (64-bit GPU) | 64-bit GPU | Polyhedra | **unsound** | sound | **unsound** | **unsound** |
| GCP-CROWN | 64-bit CPU | Polyhedra | **unsound** | sound | **unsound** | **unsound** |
| DeepPoly | 64-bit CPU | Polyhedra | **unsound** | sound | sound | **unsound** |
| DeepZ | 64-bit CPU | Zonotope | **unsound** | sound | sound | **unsound** |

**Core Findings**: All 8 verifiers are **unsound** against both the Precision attack and the Order3 attack—failing to detect the backdoor and erroneously certifying the network as safe.

## Highlights & Insights

1. **Conceptual Breakthrough**: First to clearly distinguish between "theoretical soundness" and "practical soundness", revealing a fundamental blind spot shared by all existing verifiers.
2. **High Exploitability of Attack**: The constructed backdoor network remains completely invisible under standard verification, yet can completely flip classification results (dropping accuracy from 98% to 0.1%) in specific deployment environments.
3. **Root Cause is Not Precision**: Even if a verifier utilizes infinite-precision arithmetic, it cannot resolve this issue—because the problem lies in verifying the "theoretical model" rather than the "deployed model".
4. **Exquisite Incremental Attack Design**: Order1 $\rightarrow$ Order2 $\rightarrow$ Order3 progressively increases detection difficulty, systematically exposing the weaknesses of verifiers.
5. **Insights on Fixed-Point Arithmetic**: The paper points out that fixed-point arithmetic, with its controllable quantization rounding errors, could serve as an alternative path to achieve practically sound verification.
6. **Implications for Actual Deployment Security**: In safety-critical systems, verification must incorporate the deployment environment (hardware, precision, arithmetic libraries, and parallelization strategies).

## Limitations & Future Work

1. **No Solutions Proposed**: The paper only exposes the problem and does not propose any verification method to achieve practical soundness.
2. **Strong Assumptions in Attack Scenarios**: Attackers need to understand the floating-point characteristics (precision, operation order) of the deployment environment, which increases the attack difficulty in black-box deployment scenarios.
3. **Only Tested on ReLU Networks**: Experiments strictly use MNIST networks with ReLU activations, without covering more complex architectures (such as Transformers, BatchNorm, etc.) or larger-scale networks.
4. **Computational Complexity Unresolved**: It is conjectured that finding the optimal expression tree is NP-hard, but no rigorous proof is provided (only citing known results of similar problems).
5. **Insufficient Assessment of Actual Threats**: Whether an attacker can construct an effective backdoor without knowing exact deployment details in real-world scenarios is not thoroughly discussed.
6. **Backdoor Embedding Relies on Existing Methods**: Employs the embedding scheme from Zombori et al. (2021) rather than a completely independent attack framework.

## Related Work

- **Classification of Verification Methods**: Sound methods (IBP, DeepPoly, CROWN, DeepZ) are based on bound propagation; sound+complete methods (MIPVerify, Reluplex) are based on SMT/MILP; the BaB framework (β-CROWN) merges both.
- **Floating-point Numerical Vulnerabilities**: Zombori et al. (2021) exploit numerical errors of MILP solvers to attack verifiers (affecting only optimization-based verifiers); Jia & Rinard (2021) exploit the mismatch in precision between verification and deployment.
- **Discrepancies in Deployment Environments**: Schlögl et al. (2023) investigate the impact of different platforms on deployed network outputs; Shanmugavelu et al. (2024/2025) show that adversarial examples can be generated solely by permuting the order of operations.
- **Implementation Gap**: Cordeiro et al. (2025) propose the concept of "implementation gap", positioning this problem as a programming language challenge.
- **Fixed-point Arithmetic Alternatives**: Lohar et al. (2023) study sound verification under fixed-point quantization, where rounding errors are controllable.

## Rating

| Dimension | Score (1-10) | Description |
|------|-----------|------|
| Novelty | 9 | First to systematically reveal the fundamental gap between theoretical and practical soundness |
| Theoretical Depth | 8 | Rigorous formal proofs, although the NP-hard conjecture is not fully proven |
| Experimental Thoroughness | 8 | Covers 8 mainstream verifiers and multiple deployment environments, highly convincing |
| Practical Value | 7 | Exposes an important security issue but does not provide solutions |
| Writing Quality | 8 | Clear logic, with a well-balanced integration of intuitive explanations and formal proofs |
| Overall Rating | **8.0** | Proposes a fundamental open problem in the field of neural network verification |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Neural Theorem Proving for Verification Conditions: A Real-World Benchmark](../../ICLR2026/llm_reasoning/neural_theorem_proving_for_verification_conditions_a_real-world_benchmark.md)
- [\[NeurIPS 2025\] Clip-and-Verify: Linear Constraint-Driven Domain Clipping for Accelerated Neural Network Verification](../../NeurIPS2025/llm_reasoning/clip-and-verify_linear_constraint-driven_domain_clipping_for_accelerating_neural.md)
- [\[NeurIPS 2025\] TimE: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios](../../NeurIPS2025/llm_reasoning/time_a_multilevel_benchmark_for_temporal_reasoning_of_llms_i.md)
- [\[ICLR 2026\] OpenEstimate: Evaluating LLMs on Reasoning Under Uncertainty with Real-World Data](../../ICLR2026/llm_reasoning/openestimate_evaluating_llms_on_reasoning_under_uncertainty_with_real-world_data.md)
- [\[NeurIPS 2025\] Two-Stage Learning of Stabilizing Neural Controllers via Zubov Sampling and Iterative Domain Expansion](../../NeurIPS2025/llm_reasoning/two-stage_learning_of_stabilizing_neural_controllers_via_zubov_sampling_and_iter.md)

</div>

<!-- RELATED:END -->
