---
title: >-
  [Paper Note] Learning Randomized Reductions
description: >-
  [ICML 2026][Optimization][Randomized Self-Reduction] This work formalizes the discovery of Randomized Self-Reductions (RSR) for a function $f$—a manual task dormant for forty years—as a learning problem with correlated s…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Randomized Self-Reduction"
  - "Self-correcting programs"
  - "Neuro-symbolic"
  - "Symbolic Regression"
  - "LLM Agent"
date: 2026-05-08
content_hash: 41be54f5e36f7fb1
---

# Learning Randomized Reductions

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2412.18134](https://arxiv.org/abs/2412.18134)  
**Code**: https://github.com/ferhaterata/learning-randomized-reductions (Available)  
**Area**: Optimization / Neuro-symbolic Learning / Symbolic Regression  
**Keywords**: Randomized Self-Reduction, Self-correcting programs, Neuro-symbolic, Symbolic Regression, LLM Agent  

## TL;DR
This work formalizes the discovery of Randomized Self-Reductions (RSR) for a function $f$—a manual task dormant for forty years—as a learning problem with correlated sampling. It introduces the Bitween framework: initially employing sparse linear regression to mine RSR within a fixed query set $\{x+r, x-r, x \cdot r, x, r\}$, and subsequently utilizing an LLM agent to search across a broader query function space. Bitween increases RSR coverage from 54% to 80% on the 80-function RSR-Bench and provides the first known RSR expression for the sigmoid function.

## Background & Motivation

**Background**: Since its introduction by Goldwasser & Micali (1984), Randomized Self-Reduction (RSR) has been a technique to reconstruct the value $f(x)$ for a hard input $x$ using a linear combination of $f$ evaluated at several random but correlated points $u_i = q_i(x, r)$. It is widely used in self-correcting programs, instance hiding, average/worst-case complexity reductions, and interactive proofs.

**Limitations of Prior Work**: For over forty years, discovering RSRs for specific functions has relied almost exclusively on manual mathematical derivation. Furthermore, the field has long restricted "query functions" to five fixed forms: $\{x+r, x-r, x \cdot r, x, r\}$, leaving many functions (e.g., sigmoid, Gudermannian, special functions) without known RSRs.

**Key Challenge**: The search space for RSR explodes in two dimensions: the choice of the query function family $Q$ and the algebraic structure of the recovery function $p$. Purely symbolic methods cannot freely explore the query family, while purely neural methods frequently hallucinate "pseudo-RSRs" that cannot be symbolically verified.

**Goal**: To decompose RSR discovery into two sub-problems: (1) efficiently inferring the recovery function $p$ given a query function family $Q$ using sample data; (2) dynamically proposing new, meaningful query functions for a given function $f$ beyond $Q$.

**Key Insight**: The authors observe that Lipton (1989) proved that for polynomials of degree $d$ over finite fields, only $d+1$ **linear** query functions and one **linear** recovery function are needed to construct an RSR. This implies that in many scenarios, RSR is essentially a sparse linear regression problem, where "heavy" methods like symbolic regression, MILP, or genetic programming may not be optimal.

**Core Idea**: First, optimize the regression backend on a fixed query set (Vanilla Bitween), then deploy an LLM as a "query function generator" to explore new queries while utilizing the same verification tools (Agentic Bitween), creating a synergy between neural creativity and symbolic verifiability.

## Method

### Overall Architecture
Bitween takes as input a program $\Pi$ suspected of implementing an unknown function $f$ (potentially with floating-point errors), an input domain $X$, a query function class $Q$, and an upper bound for the recovery function degree $d$. The process is: (1) for each candidate query $q \in Q$, introduce a symbolic variable $v_q$ representing $\Pi(q(x,r))$; (2) enumerate all monomials $V$ of degree $\le d$ as linear features; (3) randomly sample $m$ pairs of $(x_i, r_i)$, calling $\Pi$ to obtain $\Pi(x_i)$ and all $\Pi(q(x_i, r_i))$; (4) fit $\Pi(x_i) = \sum_V C_V \cdot V(x_i, r_i)$ via sparse linear regression, removing monomials with coefficients near zero; (5) convert residual coefficients into rational numbers using maximum denominator constraints; (6) simplify the candidate expression to zero using SymPy for formal verification. Agentic Bitween adds an external LLM agent that iteratively calls `infer_property_tool` (triggering the Vanilla backend but allowing new queries like $f(x+\log k)$), `symbolic_verify_tool` (SymPy verification), and `sequential_thinking_tool` (chain-of-thought logging).

### Key Designs

1.  **Formalization of RSR Learning and "Correlated Sampling" Access Model**:
    *   Function: Defines RSR discovery as a PAC-style learning problem: given a function class $F \subseteq \mathrm{RSR}_k(Q, P)$ and $m$ samples, output a $(\rho, \xi)$-approximate RSR with probability $\ge 1 - \delta$.
    *   Mechanism: Introduces a third access type between "i.i.d. samples" and "arbitrary oracle queries"—**correlated random samples**, where the marginal distribution of each $x_j$ is uniform, but they are correlated across $j$. This matches the natural RSR sampling mode (e.g., $x$ and $x+r$ are marginally uniform but jointly correlated).
    *   Design Motivation: Traditional PAC learning aims to approximate $f$ itself, whereas RSR learning seeks an equality constraint $p(x, r, f(u_1), \ldots, f(u_k)) = 0$. These objectives have distinct sample complexities (compared in Claims A.1/A.2).

2.  **Vanilla Bitween: Supervised Regression + Sparsity + Rationalization**:
    *   Function: Reduces RSR search under a fixed query set to controlled sparse linear regression sub-problems.
    *   Mechanism: For each candidate target variable $v_q$, a regression task is constructed with loss $L_q(C) = \frac{1}{m}\sum_i (\Pi_i - \sum_V C_V V_i)^2 + \lambda R(C)$, where $R(C)$ toggles between Lasso and Ridge, and $\lambda$ is found via 5-fold CV. Coefficients below a threshold are pruned iteratively. Finally, floating-point coefficients are converted to readable fractions via rational approximation (essential for subsequent SymPy verification).
    *   Design Motivation: The authors compared this to PySR (Genetic Programming), GPLearn, and Gurobi (MILP). While more "powerful," these symbolic backends often timed out or yielded unverifiable approximations. Simple sparse linear regression (V-Bitween-LR) dominated in coverage, runtime, and the number of RSRs found (54% vs ≤32%).

3.  **Agentic Bitween: LLM as Query Function Generator with Symbolic Verification**:
    *   Function: Breaks the limitations of the fixed query set by allowing the LLM to propose arbitrary terms like $f(x+\log k)$, $f(\sqrt{x^2+y^2})$, etc., followed by Bitween inference and SymPy verification.
    *   Mechanism: The agent is called once per function but can invoke three tools multiple times: `infer_property_tool` (propose new queries for the regression backend), `symbolic_verify_tool` (hard verification), and `sequential_thinking_tool` (CoT). Models used include GPT-OSS-120B, Claude-Sonnet-4, and Claude-Opus-4.1.
    *   Design Motivation: Pure neural baselines (N-Research) produce many unverifiable "pseudo-RSRs". Agentic Bitween uses symbolic tools to suppress the unverifiable ratio (793 verified RSRs vs 26 unverified for Opus-4.1), achieving a true loop of neural creativity and symbolic verifiability.

### Loss & Training
The regression stage employs Lasso/Ridge with regularization $\lambda$ (selected via 5-fold CV) and iterative pruning. Sampling uses a uniform distribution over $[-10, 10]$ with error tolerance $\delta = 10^{-3}$. Each experiment is repeated 5 times with an 1800s budget. Degree $d=3$ for trigonometric/hyperbolic/exponential functions and $d=2$ otherwise.

## Key Experimental Results

### Main Results
RSR-Bench contains 80 functions across 8 categories. The table below compares 5 symbolic backends, 3 neural baselines, and 3 Agentic Bitween configurations (Format: RSR Count / Verified | Unverified).

| Method | RSR / Verified | Unverified | RSR Coverage | Avg. Runtime |
| :--- | :--- | :--- | :--- | :--- |
| V-Bitween-PySR | 61 / 61 | 60 | 38% | 335 s |
| V-Bitween-GPLearn | 48 / 48 | 54 | 32% | 140 s |
| V-Bitween-MILP | 74 / 74 | 29 | 51% | 11 s |
| **V-Bitween-LR** | **87 / 87** | 46 | **54%** | **5 s** |
| N-Research-Opus-4.1| 250 / 539 | 172 | 64% | 286 s |
| A-Bitween-Sonnet-4 | 293 / 729 | 14 | 66% | 160 s |
| **A-Bitween-Opus-4.1**| **793 / 1628** | **26** | **80%** | 378 s |

The linear regression backend achieved the highest RSR count, coverage, and shortest runtime among symbolic methods. Agentic Bitween pushed coverage to 80% and discovered the first RSR for sigmoid: $\sigma(x) = \frac{\sigma(x+r)(\sigma(r) - 1)}{2\sigma(x+r)\sigma(r) - \sigma(x+r) - \sigma(r)}$.

### Ablation Study
Based on the number of RSRs by category (comparing A-Bitween with its corresponding N-Research model).

| Category | V-Bitween-LR | N-Research (Opus) | A-Bitween (Opus) | Tool Gain |
| :--- | :--- | :--- | :--- | :--- |
| Basic | 1.2 | 11.0 | 23.5 | +12.5 |
| Exponential | 1.2 | 12.3 | 29.6 | +17.3 |
| Trigonometric | 1.8 | 9.1 | 19.4 | +10.3 |
| Hyperbolic | 0.7 | 7.0 | 20.0 | +13.0 |
| ML Functions | 1.0 | 4.1 | 18.7 | +14.6 |
| Special | 0.7 | 4.9 | 20.2 | +15.3 |

### Key Findings
- In symbolic methods, "model class matching" outperforms "algorithmic complexity"—sparse linear regression yields 1.4–1.7x the coverage of PySR/GPLearn with 28–67x faster runtimes.
- The primary benefit of Agentic Bitween over N-Research is the drastic reduction of pseudo-RSRs: unverified properties dropped from 172 to 26 (−85%) for Opus-4.1, proving the symbolic verification tool is crucial.
- Stronger models yield higher gains: RSR counts grew exponentially from GPT-OSS-120B (157) to Sonnet-4 (293) to Opus-4.1 (793).
- The framework generalizes to non-scalar domains: RSRs were successfully found for matrices, quaternions, and Clifford/Lie algebras by flattening structures into scalars.

## Highlights & Insights
- **Dual Formalization & Engineering**: Formalizing a 40-year task as a falsifiable learning problem and pairing it with a simple, effective backend (linear regression) is a high-value paradigm for ML for Science.
- **LLM as "Query Generator"**: Restricting the LLM to "proposing" rather than "solving" while using symbolic tools for validation is a clean neuro-symbolic fusion paradigm applicable to invariant discovery or quantum circuit simplification.
- **The first sigmoid RSR** is a significant scientific byproduct, enabling self-correction for sigmoid via two calls ($\sigma(x+r)$, $\sigma(r)$) plus rational operations, which is valuable for instance-hiding in privacy-preserving inference.

## Limitations & Future Work
- The current theory assumes realizability ($F \subseteq \mathrm{RSR}_k(Q, P)$); the agnostic setting is left for future work.
- The maximum degree $d$ is limited to 3 due to the exponential growth of monomials; high-order RSRs remain unreachable.
- The "Fundamental Theorem of RSR Learning" relating sample complexity to VC-dimension is left as an open problem.
- High sensitivity to LLM scale and significant token costs/latency for Agentic Bitween (up to 900s per function).

## Related Work & Insights
- **vs. Symbolic Regression**: While general SR discovers expressions from data, this work fixes the function and finds RSR equalities. It demonstrates that sparse linear regression is often a better fit for this specific task.
- **vs. Program Invariants (Daikon/DIG)**: These focus on dynamic program-level invariants; Bitween focuses on mathematical RSRs from a PAC-learning perspective.
- **vs. Pure LLM Reasoning**: Unlike models that write definitive answers prone to hallucination, A-Bitween uses the LLM solely for query proposal within a verification loop.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First formalization of RSR discovery as a learning problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive comparison across 80 functions and multiple backends, though repetitions were relatively low (5x).
- Writing Quality: ⭐⭐⭐⭐ Clear parallel between theory and system design.
- Value: ⭐⭐⭐⭐⭐ Provides a concrete scientific discovery (sigmoid RSR) and a reusable neuro-symbolic paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning Locally, Revising Globally: Global Reviser for Federated Learning with Noisy Labels](learning_locally_revising_globally_global_reviser_for_federated_learning_with_no.md)
- [\[ICML 2026\] Interpretability and Generalization Bounds for Learning Spatial Physics](interpretability_and_generalization_bounds_for_learning_spatial_physics.md)
- [\[ICML 2026\] Ubiquity of Emergent Hebbian Dynamics in Regularized Learning](ubiquity_of_emergent_hebbian_dynamics_in_regularized_learning.md)
- [\[ICML 2026\] Bayesian Gated Non-Negative Contrastive Learning](bayesian_gated_non-negative_contrastive_learning.md)
- [\[ICML 2026\] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective](learning_dynamics_of_zeroth-order_optimization_a_kernel_perspective.md)

</div>

<!-- RELATED:END -->
