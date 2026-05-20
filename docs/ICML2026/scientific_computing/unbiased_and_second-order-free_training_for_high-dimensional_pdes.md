---
title: >-
  [Paper Note] Unbiased and Second-Order-Free Training for High-Dimensional PDEs
description: >-
  [ICML 2026][Scientific Computing][BSDE] This paper addresses the discretization bias in EM-BSDE training loss by proposing Un-EM-BSDE: single-step errors are averaged over two independent groups of Monte Carlo subsamples…
tags:
  - "ICML 2026"
  - "Scientific Computing"
  - "BSDE"
  - "High-dimensional PDE"
  - "Euler-Maruyama"
  - "Unbiased Estimation"
  - "Second-order-free"
date: 2026-05-08
content_hash: 4f6021beeac5799b
---

# Unbiased and Second-Order-Free Training for High-Dimensional PDEs

**Conference**: ICML 2026  
**arXiv**: [2605.14643](https://arxiv.org/abs/2605.14643)  
**Code**: https://github.com/seojaemin22/Un-EM-BSDE (available)  
**Area**: Scientific Computing / Neural PDE Solvers  
**Keywords**: BSDE, High-dimensional PDE, Euler-Maruyama, Unbiased Estimation, Second-order-free

## TL;DR
This paper addresses the discretization bias in EM-BSDE training loss by proposing Un-EM-BSDE: single-step errors are averaged over two independent groups of Monte Carlo subsamples and then "multiplied" to form an unbiased estimator, eliminating bias without requiring Hessians. On benchmark PDEs such as HJB/BSB/AC, it matches the accuracy of Heun-BSDE / FS-PINNs but with only 1.79× the training time of EM-BSDE (compared to 42.91× for Heun-BSDE and 32.07× for FS-PINNs).

## Background & Motivation

**Background**: There are two mainstream approaches for high-dimensional PDE solvers—PINNs encode the PDE residual into the loss function but are unstable for high-frequency or multi-scale solutions; Deep BSDE leverages the connection between PDEs and stochastic differential equations (SDEs), reformulating the problem as a probabilistic representation along trajectories, thus avoiding the curse of dimensionality. Deep BSDE uses Euler-Maruyama (EM) for time discretization, constructing a self-consistency loss $\ell_{\text{EM}}=\mathbb{E}[|\text{err}^{\text{EM}}_n|^2]$.

**Limitations of Prior Work**: Park & Tu (2025) proved that the EM-BSDE loss is a discretization-biased estimator at finite step size $\Delta t$—the bias term $\frac{1}{2}\text{Tr}[(\sigma^T(\nabla^2 u_\theta)\sigma)^2]$ directly contaminates the gradient direction. To remove the bias, they proposed Heun-BSDE (using Stratonovich + Heun integration), but this requires explicit computation of second derivatives (Hessian), making training **42.91 times** slower than EM-BSDE. Xu & Zhang (2025)'s Shotgun method only reduces the bias to $1/M$ and does not eliminate it.

**Key Challenge**: Achieving both unbiasedness and efficiency (no second derivatives) in BSDE training seems mutually exclusive—Heun-BSDE sacrifices efficiency for unbiasedness, Shotgun/Multi-Shot EM sacrifices unbiasedness for efficiency, and FS-PINNs use forward SDE sampling but still require Hessians.

**Goal**: (i) Completely eliminate EM discretization bias; (ii) Avoid any computation of $\nabla^2 u_\theta$; (iii) Training time should not be much slower than EM-BSDE; (iv) Should work for complex dynamics such as BZ (fully-coupled FBSDE) and PIDE (with jumps).

**Key Insight**: Leverage the classic sample-splitting principle from statistics—replace the same second moment $\mathbb{E}[X^2]$ with the cross-moment of two independent subsamples $\mathbb{E}[X_1\cdot X_2]$. Since $X_1, X_2$ are independent, $\mathbb{E}[X_1 X_2]=\mathbb{E}[X_1]\mathbb{E}[X_2]=(\mathbb{E}[X])^2$, and the bias term (from $\text{Var}(X)$) naturally vanishes.

**Core Idea**: Replace "square of a single group of samples" with "product of two independent groups of shot subsamples" to form an unbiased estimator of the single-step error: $\ell^{M_1, M_2}_{\text{UEM}}=\mathbb{E}[\text{Shot}_{M_1}[\text{err}^{\text{EM}}_n]\cdot\text{Shot}_{M_2}[\text{err}^{\text{EM}}_n]]$.

## Method

### Overall Architecture
The PDE $\mathcal{L}[u](t,x)=\phi(t,x,u,\nabla u)$ is transformed via Itô's formula into an FBSDE system $dX_t=\mu\,dt+\sigma\,dW_t$, $dY_t=\phi\,dt+Z_t^T\sigma\,dW_t$, where $Y_t=u(t,X_t)$, $Z_t=\nabla u(t,X_t)$. EM discretizes time on grid $t_n=n\Delta t$, yielding forward $F_n(x)=x+\mu\Delta t+\sigma\Delta W_n$ and backward $B_n(x;u)=u(t_n,x)+\phi_u\Delta t+\nabla u\cdot\sigma\Delta W_n$. The single-step error is defined as $\text{err}^{\text{EM}}_n(x;u)=\frac{u(t_{n+1}, F_n(x))-B_n(x;u)}{\Delta t}$. Un-EM-BSDE samples $M_1+M_2$ independent Brownian increments $\Delta W_{n,i}$ at each step, splits them into two groups, averages each, and multiplies to obtain an unbiased single-step loss, then accumulates along the trajectory.

### Key Designs

1. **Sample-splitting Unbiased Estimator**:

    - **Function**: Replaces the biased second moment $\ell_{\text{EM}}=\mathbb{E}[X^2]$ with the unbiased cross-moment $\mathbb{E}[X_1 X_2]$.
    - **Mechanism**: Define $\text{Shot}_M[\xi]=\frac{1}{M}\sum_{m=1}^M \xi_m$; use $M_1+M_2$ i.i.d. Brownian increments to compute $M_1+M_2$ independent single-step errors, split into two non-overlapping groups to compute $\text{Shot}_{M_1}$ and $\text{Shot}_{M_2}$, and the final loss is $\ell^{M_1,M_2}_{\text{UEM}}=\mathbb{E}[\text{Shot}_{M_1}[\text{err}^{\text{EM}}_n]\cdot\text{Shot}_{M_2}[\text{err}^{\text{EM}}_n]]$. Lemma 4.1 proves $\ell^{M_1,M_2}_{\text{UEM}}=([\mathcal{L}[u_\theta]-\phi_{u_\theta}](t_n,x))^2+O(\Delta t^{1/2})$, i.e., exactly the square of the continuous-time PDE residual (up to vanishing terms), completely removing the EM bias term $\text{Tr}[(\sigma^T\nabla^2 u_\theta\sigma)^2]$.
    - **Design Motivation**: Traditional BSDE uses the same noise $\Delta W_n$ for both forward and backward steps, causing the variance of $\text{err}^{\text{EM}}_n$ to be absorbed into $\mathbb{E}[X^2]$ as bias; using two independent noise groups separates the variance and mean-square, so the variance contribution is excluded from $\mathbb{E}[X_1]\mathbb{E}[X_2]$.

2. **Second-order-free (Avoiding Explicit Second Derivatives)**:

    - **Function**: Maintains the EM single-step update structure, thus avoiding $\nabla^2 u_\theta$.
    - **Mechanism**: Heun-BSDE is slow because the Itô-to-Stratonovich conversion introduces a second-order spatial correction term, requiring Hessian computation; Un-EM-BSDE always stays within the Itô framework, and the single-step formula $B_n$ only involves $u, \nabla u$, so the entire pipeline only needs first-order reverse-mode gradients (one line of grad in PyTorch / JAX).
    - **Design Motivation**: In $d$-dimensional PDEs, the Hessian is a $d\times d$ matrix, and AD computation costs $O(d)$ times that of first-order gradients. For high-dimensional problems ($d=100$), this directly determines whether training can be completed on GPU—this is the root cause of Heun-BSDE's 42.91× time.

3. **Variance Control + Shotgun Universal Wrapper**:

    - **Function**: (a) Proves that the variance of the Un-EM estimator with $M_1=1, M_2=2$ is no larger than EM-BSDE; (b) Applies the same sample-splitting idea as a universal debiasing wrapper to any single-step loss.
    - **Mechanism**: Theorem 4.3 proves that under $\alpha=2/M-1/(2M_1)-1/(2M_2)\geq 4/(3M+\beta M^4)$, $\beta=1/(2M^2)-1/(4M_1 M_2)>0$, $\mathbb{V}[\hat\ell^{M_1,M_2}_{\text{UEM}}]\leq\mathbb{V}[\hat\ell^M_{\text{SG}}]=\mathbb{V}[\hat\ell^M_{\text{SEM}}]\leq\mathbb{V}[\hat\ell_{\text{EM}}]$. Applying the same product construction to Shotgun loss yields Un-SG, which reduces RL2 by 2.67× on BSB hard constraint, with only 1.78× increase in training time.
    - **Design Motivation**: Sample-splitting can introduce extra variance (cross-moment is noisier than second moment), so variance analysis is key to ensuring practicality; the universal wrapper amplifies the contribution from a single point to a general technique that can debias any biased single-step loss.

### Loss & Training

Experiments default to $M_1=M_2=5$. Baselines: Shotgun uses $M=50$, Multi-Shot EM uses $M=10$, aligning the internal sampling budget with $M_1+M_2=10$. The loss supports both soft constraint (terminal condition as an extra loss term $L_T$) and hard constraint (trial function form). Algorithm 1 presents batched implementation: for batch size $B$, time steps $N$, and shot number $M_1+M_2$, tensor $X\in\mathbb{R}^{B\times(N+1)\times(M_1+M_2)\times d}$ stores all candidate states at once, computes forward trajectories and single-step predictions $\hat Y[b,n+1,i]$ in parallel, and finally aggregates by group for the product.

## Key Experimental Results

### Main Results

RL2 error ($\times 10^{-2}$) on 5 benchmark PDEs; bold for best, underline for second-best:

| PDE / Constraint | EM-BSDE (Biased) | Shotgun (Biased) | Multi-Shot EM | Heun-BSDE (Unbiased) | FS-PINNs (Unbiased) | **Un-EM-BSDE (Ours)** |
|------------------|------------------|------------------|---------------|----------------------|---------------------|-----------------------|
| HJB soft         | 0.4055           | 1.1409           | 0.1617        | 0.1424               | **0.0867**          | _0.1348_              |
| BSB soft         | 0.3483           | 39.99            | 0.1046        | 0.1030               | **0.0478**          | _0.0814_              |
| AC soft          | 0.0462           | 0.0951           | _0.0206_      | 0.0774               | 0.0325              | **0.0147**            |
| BSB hard         | 0.3456           | 0.1629           | 0.0739        | 0.0201               | **0.0048**          | _0.0120_              |
| PIDE hard        | 0.0374           | 0.4057           | 0.0245        | 0.1874               | **0.0137**          | _0.0226_              |

Training time multiples (Table 1):

| Method                | Unbiased | 2nd-order-free | Training Time |
|-----------------------|----------|----------------|--------------|
| EM-BSDE               | ✗        | ✓              | 1×           |
| Shotgun               | ✗        | ✓              | 0.75×        |
| Multi-Shot EM-BSDE    | ✗        | ✓              | 1.74×        |
| Heun-BSDE             | ✓        | ✗              | **42.91×**   |
| FS-PINNs              | ✓        | ✗              | **32.07×**   |
| **Un-EM-BSDE (ours)** | ✓        | ✓              | **1.79×**    |

### Ablation Study

| Configuration                          | Effect                                                      |
|-----------------------------------------|-------------------------------------------------------------|
| Full Un-EM-BSDE                         | Almost always second-best or best in all settings           |
| Sample-splitting wrapper on Shotgun (Un-SG) | RL2 on BSB hard reduced by 2.67×, time increased by 1.78×   |
| Hard constraint vs Soft constraint      | Hard is significantly more stable for complex dynamics (BZ, PIDE); soft is affected by loss balancing |
| BZ (fully-coupled FBSDE) soft           | Un-EM at 5.18 level, Shotgun spikes to 86.53                |

### Key Findings
- **Efficiency is the killer feature**: In high-dimensional ($d$) scenarios, Heun-BSDE and FS-PINNs may be "unrunnable" due to Hessian computation, while Un-EM-BSDE's 1.79× time is a sweet spot.
- **Wrapper generality is more valuable than the method itself**: Applying the same product construction to Shotgun immediately yields a 2.67× accuracy improvement, indicating this is a class of debiasing techniques (applicable to any "same-noise forward + backward" single-step loss).
- **Hard constraint is more robust for complex dynamics (BZ, PIDE)**: Loss balancing issues with soft constraint are amplified in fully-coupled/jump scenarios; hard constraint is more stable due to no weight tuning, which is a practical engineering tip.
- **Variance does not explode**: Theorem 4.3 and experiments both confirm that the variance of the Un-EM estimator is no greater than EM-BSDE, so the "classic concern" of sample-splitting is not a practical issue here.

## Highlights & Insights
- **Precise application of a classic statistical trick**: Sample-splitting is a well-known technique in statistical inference, but plugging it precisely into the BSDE single-step loss to achieve both debiasing and efficiency demonstrates deep understanding—the bias term is hidden in $\text{Var}(X)$, and independent sampling automatically isolates it.
- **Universal wrapper design**: Sec 5.3 abstracts the method as "any biased single-step loss with parameter $\tau$ can use the same construction", making this a framework-level contribution far beyond a single algorithm.
- **Avoiding Itô vs Stratonovich dilemma**: Heun-BSDE forces Stratonovich for unbiasedness, introducing Hessians; Un-EM achieves the same unbiasedness within the Itô framework via randomization, sidestepping the stochastic calculus dilemma.
- **Tight theory-experiment match**: Lemma 4.1 (unbiasedness), Theorem 4.2 (consistency), and Theorem 4.3 (variance) are all experimentally validated—there is no "theory looks good but experiments fail" problem common in ML papers.

## Limitations & Future Work
- The current theory assumes bounded $\mu, \sigma$ and $u_\theta\in C^{1,2}$; for practical fully-coupled FBSDE and PIDE with unbounded coefficients/jump processes, theoretical guarantees are only partially covered (explicitly acknowledged in the paper).
- The algorithm requires $M_1+M_2$ independent Brownian increments per step (default 10), compared to 1 for EM-BSDE; batched implementation needs $10\times$ more tensor memory, which may be a bottleneck for large $d$ or batch size.
- Experiments only go up to $d\sim 100$; there is no complete ablation for truly large-scale ($d>1000$) PDE solvers.
- Lacks comparison with modern SOTA such as forward-backward dual-network methods (separate networks per step).
- Extension to adaptive time-stepping for complex dynamics is listed as future work; currently, fixed $\Delta t$ may be sub-optimal for stiff/multi-scale PDEs.

## Related Work & Insights
- **vs EM-BSDE (Raissi 2024)**: Base method; Un-EM removes its bias via randomized product, with only 79% more time.
- **vs Heun-BSDE (Park & Tu 2025)**: Also unbiased, but Heun requires Hessians and is 42.91× slower; Un-EM is completely Hessian-free.
- **vs Shotgun (Xu & Zhang 2025)**: Shotgun reduces bias to $1/M$ but does not eliminate it; Un-EM wrapper applied to Shotgun immediately debiases it.
- **vs FS-PINNs (Park & Tu 2025)**: FS-PINNs directly minimize the squared PDE residual sampled along SDE trajectories, unbiased but require Hessians; Un-EM achieves similar accuracy via BSDE-style single-step loss without Hessians.
- **vs Hu et al. (2025) bias-variance trade-off PINNs**: Similar idea (independent samples form product to debias); this paper specializes and extends this idea within the BSDE framework.
- **Insights**: (a) Sample-splitting may also debias other stochastic losses (e.g., contrastive learning, scoring rules); (b) The trick of "splitting noise into two independent groups" may also remove bootstrap bias in RL value estimation.

## Rating
- Novelty: ⭐⭐⭐⭐ Application of sample-splitting within BSDE loss is a clear and nontrivial contribution, though sample-splitting itself is a classic idea.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 standard PDEs + 2 complex extensions (BZ, PIDE) + wrapper generalization experiments, very comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ The "unbiased + 2nd-order-free + time" comparison table in Table 1 makes the contribution immediately clear; Lemma/Theorem numbering is clear.
- Value: ⭐⭐⭐⭐⭐ Heun-BSDE's 42.91× slowdown limits its practical value; Un-EM brings unbiased BSDE back to EM-level training cost, a directly usable advance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PhysicsCorrect: A Training-Free Approach for Stable Neural PDE Simulations](../../AAAI2026/scientific_computing/physicscorrect_a_training-free_approach_for_stable_neural_pde_simulations.md)
- [\[NeurIPS 2025\] Enforcing Governing Equation Constraints in Neural PDE Solvers via Training-free Projections](../../NeurIPS2025/scientific_computing/enforcing_governing_equation_constraints_in_neural_pde_solvers_via_training-free.md)
- [\[NeurIPS 2025\] Neural Emulator Superiority: When Machine Learning for PDEs Surpasses its Training Data](../../NeurIPS2025/scientific_computing/neural_emulator_superiority_when_machine_learning_for_pdes_surpasses_its_trainin.md)
- [\[CVPR 2026\] EHETM: High-Quality and Efficient Turbulence Mitigation with Events](../../CVPR2026/scientific_computing/high-quality_and_efficient_turbulence_mitigation_with_events.md)
- [\[NeurIPS 2025\] EddyFormer: Accelerated Neural Simulations of Three-Dimensional Turbulence at Scale](../../NeurIPS2025/scientific_computing/eddyformer_accelerated_neural_simulations_of_three-dimensional_turbulence_at_sca.md)

</div>

<!-- RELATED:END -->
