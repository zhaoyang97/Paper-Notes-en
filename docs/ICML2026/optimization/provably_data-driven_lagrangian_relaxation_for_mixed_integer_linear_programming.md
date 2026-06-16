---
title: >-
  [Paper Note] Provably Data-Driven Lagrangian Relaxation for Mixed Integer Linear Programming
description: >-
  [ICML 2026][Optimization & Theory][Lagrangian relaxation] This paper provides the first rigorous statistical learning theory for the empirical approach of "learning to predict Lagrangian multipliers to accelerate MILP": it derives an ERM generalization upper bound of $\mathcal{O}(s^{1.5}/\sqrt{N})$, a minimax lower bound of $\Omega(s/\sqrt{N})$, and constructively achieves th
tags:
  - ICML 2026
  - Optimization & Theory
  - Lagrangian relaxation
  - MILP
date: 2026-05-08
content_hash: 2c87ff7c9ba895d1
---
# Provably Data-Driven Lagrangian Relaxation for Mixed Integer Linear Programming

**Conference**: ICML 2026  
**arXiv**: [2605.19052](https://arxiv.org/abs/2605.19052)  
**Code**: Not released  
**Area**: Mathematical Optimization / Learn-to-Optimize / Data-Driven Algorithm Design  
**Keywords**: Lagrangian relaxation, MILP, generalization bounds, minimax lower bounds, learning to warm-start  

## TL;DR
This paper provides the first rigorous statistical learning theory for the empirical approach of "learning to predict Lagrangian multipliers to accelerate MILP": it derives an ERM generalization upper bound of $\mathcal{O}(s^{1.5}/\sqrt{N})$, a minimax lower bound of $\Omega(s/\sqrt{N})$, and constructively achieves the optimal rate of $\Theta(s/\sqrt{N})$ using an SGA averaging algorithm. Furthermore, it proves that the sample complexity can be improved to $\Theta(s/N)$ when the problem is reformulated as "learning a warm-start initial value."

## Background & Motivation

**Background**: MILP is ubiquitous in logistics, energy, and finance, but exact solving easily suffers from combinatorial explosion. When problems involve "a few coupling constraints + many decomposable sub-structures" (e.g., Vehicle Routing Problem VRP, Unit Commitment), Lagrangian relaxation (LR) is a classic accelerator: by dualizing $s$ coupling constraints $Ax \geq b$ into the objective, one obtains the dual function $u(\pi, P) = \min_x c^\top x + \pi^\top (b - Ax)$ s.t. $x \in \mathbb{R}_+^m \times \{0,1\}^p, Cx \geq d$. This allows for parallel decomposition of sub-problems and provides tighter bounds than continuous relaxation, significantly pruning the branch-and-bound tree.

**Limitations of Prior Work**: The practical efficiency of LR depends almost entirely on "how quickly one can find good multipliers $\pi$." Optimal multipliers are solutions to non-smooth concave optimization, where traditional subgradient methods iterate slowly, often consuming the benefits gained from relaxation. Recently, works like Demelas et al. (2024, ICML) began using neural networks to predict multipliers from historical instances. While empirically effective, these lack theoretical guarantees—neither the $s, N$ dependence of generalization bounds nor the "optimal" algorithm is known.

**Key Challenge**: Data-driven LR learns a function $u_\pi(P)$ defined by an internal optimization. While $u$ is concave in $\pi$ for a fixed $P$, it is piecewise linear in $P$ for a fixed $\pi$, with a number of segments that grows exponentially with $s$. Traditional covering number arguments fail to provide clean $s$-dependence, and it remains unknown if algorithms independent of the number of coupling constraints $s$ exist.

**Goal**: To characterize three aspects of the "learning LR multipliers" statistical problem: ERM upper bounds, algorithm-independent minimax lower bounds, and specific implementable algorithms that achieve these bounds; and to extend this analysis to the alternative paradigm of learning-to-warm-start.

**Key Insight**: The authors borrow the "dual perspective" from data-driven algorithm design (Balcan 2020)—instead of analyzing the complex $u_\pi(P)$, they fix $P$ and treat $u_P(\pi)$ as a concave Lipschitz function over $\pi$ for covering. For the lower bound, Fano's inequality and Varshamov-Gilbert construction are used to create a family of distributions that are "geometrically separated + statistically indistinguishable."

**Core Idea**: By using concavity and $2B\sqrt{s}$-Lipschitz properties, the function class is compressed into a parameter space covering, yielding $\mathcal{O}(s^{1.5}/\sqrt{N})$. An $s$-dimensional binary packing reduces the estimation problem to high-dimensional parameter estimation, yielding $\Omega(s/\sqrt{N})$. SGA with averaging from an OCO perspective closes the $\sqrt{s}$ gap. By switching to squared Euclidean distance as a warm-start loss, the problem becomes a strongly convex mean estimation, jumping the rate from $1/\sqrt{N}$ to $1/N$.

## Method

### Overall Architecture
This paper does not propose an empirical algorithm but formalizes data-driven LR as a statistical learning problem of "learning a $\pi$ to maximize $\mathbb{E}_{P\sim\mathcal{D}}[u(\pi, P)]$ over a problem distribution $\mathcal{D}$." It provides four components: (1) Geometric properties of the function class $\mathcal{U}=\{P\mapsto u(\pi,P)\}$ (concavity + Lipschitz); (2) Rademacher complexity leading to ERM generalization upper bounds; (3) Fano construction for minimax lower bounds; and (4) SGA + averaging to constructively match the lower bound. Finally, these four components are transferred to the learning-to-warm-start setting.

### Key Designs

**1. Geometry + ERM Generalization Upper Bound $\mathcal{O}(s^{1.5}/\sqrt{N})$: Avoiding Piecewise-Linear Explosion via Dual Perspective**

The learned function $u_\pi(P)$ is piecewise linear in $P$ for fixed $\pi$, with segments growing exponentially with $s$. Direct segment analysis would fail. Ours adopts the dual perspective of data-driven algorithm design: fixing $P$ and treating $u(\cdot,P)$ as a concave function over $\pi$. It is proved concave (Lagrangian is a pointwise min/max of linear functions), and under Assumption 4.1, the subgradient $g(\pi,P)=b-Ax^*(\pi,P)$ satisfies $\|g\|_2\le 2B\sqrt s$. Thus, $u(\cdot,P)$ is $L=2B\sqrt s$-Lipschitz. Using Lipschitzness reduces the function class covering to a parameter space covering $\log\mathcal{N}(\delta,\mathcal{U},\|\cdot\|_{2,N})\le s\log(1+2B\pi_{\max}s/\delta)$. Finally, the Dudley entropy integral gives the Rademacher complexity $\mathscr{R}_N(\mathcal{U})=\mathcal{O}(s^{1.5}/\sqrt N)$—where one $\sqrt s$ comes from the Lipschitz constant and another from the parameter space diameter $\pi_{\max}\sqrt s$. This allows standard empirical process tools to be applied without handling the explosive number of segments.

**2. Minimax Lower Bound $\Omega(s/\sqrt{N})$ Construction: Identifying the $\sqrt s$ gap as Upper Bound Slackness**

To determine if the extra $\sqrt s$ in the upper bound is intrinsic or due to loose analysis, an algorithm-independent lower bound is needed. Ours uses Fano + Varshamov-Gilbert to construct a hard family of distributions: restricting $P$ such that $A=\mathbf{I}_s$ makes the dual values coordinate-separable $u(\pi,P)=\sum_k\min(\pi_k/2,c_k-\pi_k/2)$. A pair of Bernoulli distributions is designed for each coordinate, indexed by $v\in\{0,1\}^s$, such that $\pi^*(\mathcal{D}_v)=\mu\mathbf{1}_s+\sigma v$. VG provides a packing of size $2^{s/8}$ with Hamming distance $\ge s/8$, ensuring geometric separation $\|\pi^*(\mathcal{D}_v)-\pi^*(\mathcal{D}_{v'})\|_1\ge\Omega(\sigma s)$. Using a $\chi^2$ upper bound for KL divergence yields $\mathrm{KL}\le 4Ns\epsilon^2$. Setting $\epsilon=\Theta(1/\sqrt N)$ gives the Fano term a constant lower bound, which combined with the $\ell_1$ error to risk lower bound in Lemma 5.9, results in $\Omega(s/\sqrt N)$. The key technique is using $A=\mathbf{I}_s$ to decompose an $s$-dimensional hard problem into $s$ independent 1-dimensional hard problems, the ideal setting for Fano—thus confirming the $\sqrt s$ gap resides in the upper bound, not the lower bound.

**3. SGA + Averaging to Close the $\sqrt s$ Gap and Warm-start Improvement to $\Theta(s/N)$: Bottleneck is Algorithm-side**

Given the lower bound is tight, the gap originates from the ERM algorithm itself. Ours switches to online-to-batch Stochastic Subgradient Ascent: for each instance $P_t$, solve the Lagrangian sub-problem for $x_t^*$, take the unbiased subgradient $g_t=b_t-A_tx_t^*$, update via $\pi_{t+1}=\mathrm{Proj}_\Pi(\pi_t+\eta g_t)$, and output the average $\bar\pi_N$. Standard OCO regret combined with $\|g_t\|_2\le 2B\sqrt s$ and $\eta=\pi_{\max}/(2B\sqrt N)$ yields $\mathbb{E}[\mathcal{E}(\bar\pi_N)]\le 2B\pi_{\max}s/\sqrt N=\mathcal{O}(s/\sqrt N)$, matching the lower bound exactly. Since SGA requires solving a sub-problem at each step—exactly what is done during LR deployment—this achieves a $\sqrt s$ improvement at zero extra cost. Furthermore, changing the objective from "maximizing dual value" to "minimizing $\ell_2$ distance to optimal multipliers" $\ell(\phi,P)=\|\phi-\pi^*(P)\|_2^2$ transforms the problem from non-smooth concave maximization to strongly convex mean estimation. The empirical mean $\hat\phi(S)=\frac1N\sum_i\pi^*(P_i)$ (the ERM) combined with Popoviciu’s inequality yields $\mathcal{O}(s/N)$, with Fano providing a matching lower bound of $\Theta(s/N)$. This theoretical foundation explains why warm-starting is fundamentally more sample-efficient.

### Loss & Training
Two sets of objectives: Direct multiplier learning uses $\mathbb{E}_{P}[u(\pi, P)]$ (non-smooth concave max, requires SGA + averaging for optimal rate); learning warm-start initial values uses $\mathbb{E}_P\|\phi - \pi^*(P)\|_2^2$ (strongly convex min, empirical mean is optimal). The corresponding optimal sample complexities differ by one order—a core practical takeaway of the paper.

## Key Experimental Results

This is a purely theoretical paper with no numerical experiments; the "Main Results" consist of a theorem comparison table and an algorithm comparison table.

### Main Results (Rate Comparison)

| Objective | Upper Bound Algorithm | Upper Bound | Minimax Lower Bound | Match |
| :--- | :--- | :--- | :--- | :--- |
| Direct Learning (ERM) | ERM over $\Pi$ | $\mathcal{O}(s^{1.5}/\sqrt{N})$ | $\Omega(s/\sqrt{N})$ | Gap of $\sqrt{s}$ |
| Direct Learning (SGA) | SGA + averaging | $\mathcal{O}(s/\sqrt{N})$ | $\Omega(s/\sqrt{N})$ | ✓ $\Theta(s/\sqrt{N})$ |
| Learning Warm-start | Empirical Mean | $\mathcal{O}(s/N)$ | $\Omega(s/N)$ | ✓ $\Theta(s/N)$ |

### Resource Consumption relative to $B, \pi_{\max}$

| Quantity | Expression | Source |
| :--- | :--- | :--- |
| Lipschitz Constant $L$ | $2B\sqrt{s}$ | $\ell_2$ norm of subgradient $b - Ax^*$ |
| Parameter Space Diameter $D$ | $\pi_{\max}\sqrt{s}$ | $[0, \pi_{\max}]^s$ Hypercube |
| ERM Upper Bound Constant | $\propto B \pi_{\max} s^{1.5}$ | $\sqrt{s/N} \cdot LD$ |
| SGA Step Size $\eta$ | $\pi_{\max} / (2B\sqrt{N})$ | Standard OCO formula |
| Lower Bound w.r.t. $B, \pi_{\max}$ | $\Omega(B \pi_{\max} s / \sqrt{N})$ | Scaled construction in Remark 5.10 |

### Key Findings
- The extra $\sqrt{s}$ in the upper bound $\mathcal{O}(s^{1.5}/\sqrt{N})$ cannot be removed by piecewise-linear segment analysis (as the argument degrades when $K = 2^s$). The bottleneck lies in the ERM algorithm itself rather than the function class complexity.
- Switching from ERM to SGA provides a $\sqrt{s}$ improvement in sample complexity at zero extra cost, as SGA naturally fits the LR deployment process.
- The gap between direct multiplier learning and warm-start learning is $\Theta(s/\sqrt{N})$ vs $\Theta(s/N)$—an order of magnitude difference in sample efficiency, providing a rigorous explanation for why warm-starting is more stable in engineering practice.

## Highlights & Insights
- A clever technique is the "diagonal constraint family" in Lemma 5.7: by setting $A=\mathbf{I}_s$, the dual values become coordinate-separable, turning an $s$-dimensional hard problem into $s$ independent 1-dimensional problems for direct Fano application. This "active structural construction" for lower bounds is noteworthy.
- Attributing the ERM-vs-SGA gap to "problem structure" rather than "analytical methodology," and providing a counterexample ($K = 2^s$) to show the gap cannot be closed by finer covering arguments, makes the constructive SGA results critical.
- The elegance of the warm-start section lies in "changing the loss, not the problem"—the goal of predicting $\pi^*(P)$ remains, but changing the evaluation from dual value to Euclidean distance shifts the geometry from non-smooth concave to strongly convex, naturally increasing the rate.

## Limitations & Future Work
- Assumes problem distribution $\mathcal{D}$ is stable and i.i.d.; in reality, VRP/UC distributions drift significantly. Sample complexity under distribution shift is listed as future work.
- Assumption 4.1 requires constraint violations $|b_k|, |(Ax)_k|$ to be uniformly bounded, which might be restrictive for large-scale instances; the paper also does not discuss constant behavior as $B$ scales with $s$.
- $\Omega(s/\sqrt{N})$ is a worst-case bound; for highly regular problems, tighter distribution-dependent bounds might exist.
- Lack of numerical experiments to corroborate theoretical rates; it is suggested that future work verify SGA vs ERM vs warm-start rates on synthetic MILP instances.

## Related Work & Insights
- **vs Demelas et al. (2024, ICML)**: That work is an empirical learn-to-predict approach; this paper provides the first $\mathcal{O}(s/\sqrt{N})$ theoretical foundation for the same route. They are complementary; Demelas’s network can be used as a warm-start for SGA.
- **vs Balcan et al. series**: Both are instances of data-driven algorithm design for MILP; this paper shifts the focus from cutting plane configurations to predicting LR multipliers.
- **vs Classical OCO (Zinkevich, 2003)**: The SGA + averaging part essentially applies 1classical OCO regret to dual function maximization, with the "surprise" being that the LR sub-problem inherently provides unbiased subgradients.
- **vs Learning Warm-start (Plug-and-play)**: The paper theoretically confirms that warm-starting is more sample-efficient than direct prediction, providing clear guidance for learn-to-optimize: formulate objectives as mean estimation wherever possible.

## Rating
- Novelty: ⭐⭐⭐⭐ First theoretical treatment of learning LR multipliers with matching bounds and warm-start analysis.
- Experimental Thoroughness: ⭐⭐ Purely theoretical; lacks synthetic rate verification.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative structure from geometry to upper bounds, lower bounds, and alternative paradigms.
- Value: ⭐⭐⭐⭐ Provides a theoretical foundation for the L2O × MILP engineering line; warm-start rate analysis provides clear guidance for algorithm design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Constraint Matters: Multi-Modal Representation for Reducing Mixed-Integer Linear programming](../../ICLR2026/optimization/constraint_matters_multi-modal_representation_for_reducing_mixed-integer_linear_.md)
- [\[ICML 2025\] Integer Programming for Generalized Causal Bootstrap Designs](../../ICML2025/optimization/integer_programming_for_generalized_causal_bootstrap_designs.md)
- [\[ICML 2026\] On the Expressive Power of GNNs to Solve Linear SDPs](on_the_expressive_power_of_gnns_to_solve_linear_sdps.md)
- [\[ICML 2026\] Distilling Linearized Behavior into Non-Linear Fine-Tuning for Effective Task Arithmetic](distilling_linearized_behavior_into_non-linear_fine-tuning_for_effective_task_ar.md)
- [\[ICML 2026\] Dynamics and Representation Structure of Local Approximations to Gradient-Based Learning in Linear Recurrent Neural Networks](dynamics_and_representation_structure_of_local_approximations_to_gradient-based_.md)

</div>

<!-- RELATED:END -->
