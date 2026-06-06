---
title: >-
  [Paper Note] Provably Data-Driven Lagrangian Relaxation for Mixed Integer Linear Programming
description: >-
  [ICML 2026][Optimization][Lagrangian relaxation] This paper provides the first rigorous statistical learning theory for the empirical approach of "predicting Lagrangian multipliers to accelerate MILP." It derives an ERM…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Lagrangian relaxation"
  - "MILP"
  - "generalization bounds"
  - "minimax lower bounds"
  - "learning to warm-start"
date: 2026-05-08
content_hash: 89058c20d3e38111
---

# Provably Data-Driven Lagrangian Relaxation for Mixed Integer Linear Programming

**Conference**: ICML 2026  
**arXiv**: [2605.19052](https://arxiv.org/abs/2605.19052)  
**Code**: Not released by authors  
**Area**: Mathematical Optimization / Learn-to-Optimize / Data-driven Algorithm Design  
**Keywords**: Lagrangian relaxation, MILP, generalization bounds, minimax lower bounds, learning to warm-start  

## TL;DR
This paper provides the first rigorous statistical learning theory for the empirical approach of "predicting Lagrangian multipliers to accelerate MILP." It derives an ERM generalization upper bound of $\mathcal{O}(s^{1.5}/\sqrt{N})$, a minimax lower bound of $\Omega(s/\sqrt{N})$, and constructively achieves the optimal rate of $\Theta(s/\sqrt{N})$ using a Stochastic Subgradient Ascent (SGA) averaging algorithm. Furthermore, it proves that the sample complexity can be improved to $\Theta(s/N)$ when the paradigm is shifted to "learning warm-start initial values."

## Background & Motivation

**Background**: Mixed-Integer Linear Programming (MILP) is ubiquitous in logistics, energy, and finance, but solving these problems exactly often leads to combinatorial explosion. For problems with "few coupling constraints and many decomposable sub-structures" (e.g., Vehicle Routing Problem (VRP), Unit Commitment), Lagrangian relaxation (LR) is a classical accelerator. By dualizing $s$ coupling constraints $Ax \geq b$ into the objective, one obtains the dual function $u(\pi, P) = \min_x c^\top x + \pi^\top (b - Ax)$ s.t. $x \in \mathbb{R}_+^m \times \{0,1\}^p, Cx \geq d$. This enables parallel sub-problem decomposition and provides tighter bounds than continuous relaxation, thus significantly pruning the branch-and-bound tree.

**Limitations of Prior Work**: The practical efficiency of LR depends almost entirely on how quickly a good multiplier $\pi$ can be found. Since optimal multipliers are solutions to non-smooth concave optimization, traditional subgradient methods iterate slowly, often consuming the time gains provided by the relaxation itself. Recently, work such as Demelas et al. (2024, ICML) began using neural networks to predict multipliers from historical instances. While empirically effective, these methods lack theoretical guarantees regarding sample complexity dependencies on $s$ and $N$, and the "optimal" algorithm choice remains unknown.

**Key Challenge**: Data-driven LR learns a function $u_\pi(P)$ defined by an embedded optimization problem. While concave in $\pi$ for a fixed $P$, it is piecewise linear in $P$ for a fixed $\pi$, with a number of segments that grows exponentially with $s$. Traditional covering number arguments fail to provide a clean dependence on $s$. Moreover, it was previously unknown whether algorithms could achieve rates independent of the number of coupling constraints $s$.

**Goal**: Characterize three aspects of the "learning LR multipliers" statistical problem: the ERM upper bound, an algorithm-independent minimax lower bound, and a specific constructive algorithm that achieves said lower bound. Additionally, extend this analysis to the alternative paradigm of learning-to-warm-start.

**Key Insight**: The authors adopt the "dual perspective" from data-driven algorithm design (Balcan 2020). Instead of analyzing the complex $u_\pi(P)$, they fix $P$ and treat $u_P(\pi)$ as a concave Lipschitz function over $\pi$ to perform covering. For the lower bound, they apply Fano’s inequality combined with a Varshamov-Gilbert construction to build a "geometrically separated and statistically indistinguishable" family of hard distributions.

**Core Idea**: By utilizing concavity and the $2B\sqrt{s}$-Lipschitz property, the function class is compressed into a parameter space covering, yielding $\mathcal{O}(s^{1.5}/\sqrt{N})$. An $s$-dimensional binary packing reduces the estimation to high-dimensional parameter estimation, yielding $\Omega(s/\sqrt{N})$. Using SGA with averaging from an Online Convex Optimization (OCO) perspective closes the $\sqrt{s}$ gap. Finally, switching to squared Euclidean distance as a warm-start loss transforms the problem into strongly convex mean estimation, jumping the convergence rate from $1/\sqrt{N}$ to $1/N$.

## Method

### Overall Architecture
The paper formalizes data-driven LR as a statistical learning problem: "learning a multiplier $\pi$ on a distribution $\mathcal{D}$ that maximizes $\mathbb{E}_{P\sim\mathcal{D}}[u(\pi, P)]$." It provides four pillars: (1) Geometric characterization of the function class $\mathcal{U}=\{P\mapsto u(\pi,P)\}$ (concavity + Lipschitzness); (2) Rademacher complexity analysis leading to the ERM generalization upper bound; (3) Fano construction for the minimax lower bound; and (4) SGA + averaging to constructively match the lower bound. These components are subsequently transferred to the learning-to-warm-start setting.

### Key Designs

1.  **Geometric Properties + ERM Generalization Upper Bound $\mathcal{O}(s^{1.5}/\sqrt{N})$**:
    *   **Function**: Converts the complexity of function classes containing embedded MILPs into a tractable covering problem.
    *   **Mechanism**: It is proven that $u(\cdot, P)$ is concave in $\pi$. Under Assumption 4.1, the subgradient $g(\pi, P) = b - A x^*(\pi, P)$ satisfies $\|g\|_2 \leq 2B\sqrt{s}$, implying $u(\cdot, P)$ is $L=2B\sqrt{s}$-Lipschitz. Using Lipschitzness, the $\delta$-covering of the function class reduces to a covering of the parameter space $\Pi$: $\log \mathcal{N}(\delta, \mathcal{U}, \|\cdot\|_{2,N}) \leq s \log(1 + 2B\pi_{\max} s / \delta)$. Dudley’s entropy integral then yields $\mathscr{R}_N(\mathcal{U}) = \mathcal{O}(s^{1.5}/\sqrt{N})$.
    *   **Design Motivation**: Direct piecewise-linear analysis of $u_\pi(P)$ leads to exponential explosions. Fixing $P$ and using the geometry over $\pi$ (the dual perspective) allows the application of standard OCO and empirical process tools.

2.  **Minimax Lower Bound $\Omega(s/\sqrt{N})$ via Hard Instance Construction**:
    *   **Function**: Demonstrates that the linear dependence on $s$ is unavoidable for any algorithm, identifying the $\sqrt{s}$ gap as a result of a loose upper bound rather than a weak lower bound.
    *   **Mechanism**: The estimation problem is reduced to multi-hypothesis testing using Fano’s method and Varshamov-Gilbert. $P$ is restricted such that the dual value decomposes into $u(\pi, P) = \sum_k \min(\pi_k/2, c_k - \pi_k/2)$. By designing binary Bernoulli distributions for $c_k$ indexed by $v \in \{0,1\}^s$, the optimal multipliers satisfy $\pi^*(\mathcal{D}_v) = \mu \mathbf{1}_s + \sigma v$. The VG lemma provides a packing of size $2^{s/8}$ with Hamming distance $\geq s/8$, ensuring geometric separation $\|\pi^*(\mathcal{D}_v) - \pi^*(\mathcal{D}_{v'})\|_1 \geq \Omega(\sigma s)$.
    *   **Design Motivation**: Simplifying the embedded optimization into a coordinate-separable form reduces the problem to direction estimation for $s$ independent biased coins.

3.  **SGA + Averaging to Close the $\sqrt{s}$ Gap**:
    *   **Function**: Constructively proves that while ERM might not be optimal, a simple online-to-batch algorithm is.
    *   **Mechanism**: Instead of static ERM, the authors use Stochastic Subgradient Ascent (SGA). For each instance $P_t$, the Lagrangian sub-problem is solved for $x_t^*$, an unbiased subgradient $g_t = b_t - A_t x_t^*$ is obtained, and $\pi$ is updated via $\pi_{t+1} = \mathrm{Proj}_\Pi(\pi_t + \eta g_t)$. The final output is the average $\bar{\pi}_N = \frac{1}{N}\sum_t \pi_t$. Standard OCO regret bounds with $\|g_t\|_2 \leq 2B\sqrt{s}$ and $\eta = \frac{\pi_{\max}}{2B\sqrt{N}}$ yield $\mathbb{E}[\mathcal{E}(\bar{\pi}_N)] \leq \mathcal{O}(s/\sqrt{N})$, matching the lower bound. This is extended to warm-starting where the objective becomes minimizing proximity to optimal multipliers, attaining a fast rate of $\Theta(s/N)$.
    *   **Design Motivation**: The $\sqrt{s}$ gap between ERM and the lower bound suggested an algorithmic bottleneck. SGA naturally matches the optimal rate for non-smooth concave maximization.

### Loss & Training
Two objectives are analyzed: directly learning multipliers using $\mathbb{E}_{P}[u(\pi, P)]$ (non-smooth concave maximization, requiring SGA + averaging for optimality) and learning warm-start initial values using $\mathbb{E}_P\|\phi - \pi^*(P)\|_2^2$ (strongly convex minimization, where the empirical mean is optimal). The factor of difference in optimal sample complexity between these two is a major practical insight of the paper.

## Key Experimental Results

This is a purely theoretical paper; "Main Results" consist of theorem comparisons and algorithmic rate tables.

### Main Results (Rate Comparison)

| Objective | Upper Bound Algorithm | Upper Bound | Minimax Lower Bound | Match |
| :--- | :--- | :--- | :--- | :--- |
| Direct Learning (ERM) | ERM over $\Pi$ | $\mathcal{O}(s^{1.5}/\sqrt{N})$ | $\Omega(s/\sqrt{N})$ | No (Gap $\sqrt{s}$) |
| Direct Learning (SGA) | SGA + Averaging | $\mathcal{O}(s/\sqrt{N})$ | $\Omega(s/\sqrt{N})$ | Yes $\Theta(s/\sqrt{N})$ |
| Learning Warm-start | Empirical Mean | $\mathcal{O}(s/N)$ | $\Omega(s/N)$ | Yes $\Theta(s/N)$ |

### Source/Resource Dependencies

| Quantity | Expression | Source |
| :--- | :--- | :--- |
| Lipschitz Constant $L$ | $2B\sqrt{s}$ | $\ell_2$-norm of subgradient $b - Ax^*$ |
| Parameter Space Diameter $D$ | $\pi_{\max}\sqrt{s}$ | Diameter of the $[0, \pi_{\max}]^s$ hypercube |
| ERM Constant | $\propto B \pi_{\max} s^{1.5}$ | Dudley integral $\sqrt{s/N} \cdot LD$ |
| SGA Step-size $\eta$ | $\pi_{\max} / (2B\sqrt{N})$ | Standard OCO formulation |

### Key Findings
- The extra $\sqrt{s}$ in the $\mathcal{O}(s^{1.5}/\sqrt{N})$ upper bound is not reducible through piecewise-linear segment analysis; the bottleneck lies in the ERM algorithm itself rather than the function class complexity.
- Switching from ERM to SGA provides a $\sqrt{s}$ improvement in sample efficiency at zero additional computational cost, as SGA uses the same sub-problem solutions required for LR execution.
- The gap between direct prediction and warm-start ($\Theta(s/\sqrt{N})$ vs. $\Theta(s/N)$) provides a theoretical explanation for why warm-starting is empirically more robust.

## Highlights & Insights
- A sophisticated technique used in Lemma 5.7 (diagonal constraint family) forces $A$ to be the identity matrix $\mathbf{I}_s$, making dual values coordinate-separable. This transforms an $s$-dimensional hard problem into $s$ independent 1-dimensional hard problems for the Fano lower bound.
- The paper acknowledges the "looseness" of the ERM upper bound while proving the lower bound "touches the ceiling," making the constructive SGA results particularly significant.
- The warm-start analysis is elegant because it changes the loss function without changing the underlying problem, showing that quadratic loss in a strongly convex setting naturally improves the convergence rate.

## Limitations & Future Work
- The assumption that the distribution $\mathcal{D}$ is i.i.d. is often violated in real-world VRP or Unit Commitment tasks where intra-day shifts occur. Sample complexity under distribution shift is a noted area for future work.
- Assumption 4.1 requires uniformly bounded constraint violations ($|b_k|, |(Ax)_k|$), which may not hold for economic-scale instances. The dependence of constants on $s$ as $B$ grows is not fully explored.
- The $\Omega(s/\sqrt{N})$ bound is worst-case. For structured problems like VRP, tighter distribution-dependent bounds (e.g., using effective dimension) may exist.
- Lack of numerical validation; future work should verify the theoretical rates of SGA vs. ERM on synthetic MILP instances.

## Related Work & Insights
- **vs. Demelas et al. (2024, ICML)**: Provides the theoretical foundation for their empirical "learn-to-predict" route; the two works are complementary.
- **vs. Balcan et al.**: Extends data-driven algorithm design from cutting planes/Gomory cuts to Lagrangian relaxation, substituting piecewise-linear analysis with dual-perspective covering.
- **vs. Classical OCO (Zinkevich, 2003)**: Integrates online learning with distributed optimization by recognizing that the LR sub-problem naturally provides unbiased subgradients.
- **vs. Learning-to-Warm-start**: Formally establishes that warm-starting is more sample-efficient than direct prediction, guiding the design of L2O modules toward mean-estimation formulations where possible.

## Rating
- Novelty: ⭐⭐⭐⭐ First theoretical treatment of learn-to-predict LR, complete with matching bounds and warm-start acceleration.
- Experimental Thoroughness: ⭐⭐ Purely theoretical; lacks rate-fitting validation on synthetic data.
- Writing Quality: ⭐⭐⭐⭐ Sequential structure from geometry to bounds to constructive algorithms is clear.
- Value: ⭐⭐⭐⭐ Provides a theoretical pillar for a popular engineering branch of L2O and offers clear guidance for warm-start design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Constraint Matters: Multi-Modal Representation for Reducing Mixed-Integer Linear Programming](../../ICLR2026/optimization/constraint_matters_multi-modal_representation_for_reducing_mixed-integer_linear_.md)
- [\[ICML 2026\] RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs](rl-sph_learning_to_achieve_feasible_solutions_for_integer_linear_programs.md)
- [\[ICML 2026\] On the Expressive Power of GNNs to Solve Linear SDPs](on_the_expressive_power_of_gnns_to_solve_linear_sdps.md)
- [\[ICML 2026\] Distilling Linearized Behavior into Non-Linear Fine-Tuning for Effective Task Arithmetic](distilling_linearized_behavior_into_non-linear_fine-tuning_for_effective_task_ar.md)
- [\[ICML 2026\] Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks](dynamics_and_representation_structure_of_local_approximations_to_gradient-based_.md)

</div>

<!-- RELATED:END -->
