---
title: >-
  [Paper Note] Unbiased and Second-Order-Free Training for High-Dimensional PDEs
description: >-
  [ICML 2026][Physics & Scientific Computing][BSDE] To address the discretization bias in the training loss of EM-BSDE, this paper proposes Un-EM-BSDE. By using the "product" of single-step errors averaged from two indepen…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "BSDE"
  - "High-dimensional PDEs"
  - "Euler-Maruyama"
  - "Unbiased Estimation"
  - "Second-order derivative-free"
date: 2026-05-08
content_hash: 6aaad043eedffa09
---

# Unbiased and Second-Order-Free Training for High-Dimensional PDEs

**Conference**: ICML 2026  
**arXiv**: [2605.14643](https://arxiv.org/abs/2605.14643)  
**Code**: https://github.com/seojaemin22/Un-EM-BSDE (Available)  
**Area**: Scientific Computing / Neural PDE Solvers  
**Keywords**: BSDE, High-dimensional PDEs, Euler-Maruyama, Unbiased Estimation, Second-order derivative-free

## TL;DR
To address the discretization bias in the training loss of EM-BSDE, this paper proposes Un-EM-BSDE. By using the "product" of single-step errors averaged from two independent Monte Carlo sub-samples, it forms an unbiased estimator. This approach eliminates bias without requiring the Hessian. On benchmark PDEs such as HJB, BSB, and AC, it achieves the accuracy of Heun-BSDE / FS-PINNs while requiring only 1.79× the training time of EM-BSDE (compared to 42.91× for Heun-BSDE and 32.07× for FS-PINNs).

## Background & Motivation

**Background**: High-dimensional PDE solvers follow two main paradigms: PINNs, which incorporate PDE residuals into the loss function but suffer from training instability for high-frequency or multi-scale solutions; and Deep BSDE, which leverages the connection between PDEs and Stochastic Differential Equations (SDEs) to transform the problem into a probabilistic representation along trajectories, thereby bypassing the curse of dimensionality. Deep BSDE typically employs Euler-Maruyama (EM) for time discretization, constructing a self-consistency loss $\ell_{\text{EM}}=\mathbb{E}[|\text{err}^{\text{EM}}_n|^2]$.

**Limitations of Prior Work**: Park & Tu (2025) proved that the EM-BSDE loss is a **discretization-biased estimator** under a finite step size $\Delta t$—the bias term $\frac{1}{2}\text{Tr}[(\sigma^T(\nabla^2 u_\theta)\sigma)^2]$ directly contaminates the gradient direction. To eliminate this bias, they proposed Heun-BSDE (using Stratonovich + Heun integration), but at the cost of explicitly calculating the second-order derivative (Hessian), resulting in a training time **42.91 times** that of EM-BSDE. The Shotgun method by Xu & Zhang (2025) can only reduce the bias to $1/M$ rather than eliminating it entirely.

**Key Challenge**: The objectives of unbiasedness and efficiency (avoiding second-order derivatives) seem mutually exclusive in BSDE training. Heun-BSDE sacrifices efficiency for unbiasedness, while Shotgun/Multi-Shot EM sacrifices unbiasedness for efficiency. FS-PINNs uses forward SDE sampling but still requires the Hessian.

**Goal**: (i) Completely eliminate EM discretization bias; (ii) avoid any $\nabla^2 u_\theta$ computation; (iii) maintain a training time comparable to EM-BSDE; (iv) ensure compatibility with complex dynamics such as BZ (fully-coupled FBSDE) and PIDE (with jumps).

**Key Insight**: Leveraging the classical principle of sample-splitting in statistics—if a second moment $\mathbb{E}[X^2]$ is replaced by the product of two independent sub-samples $\mathbb{E}[X_1\cdot X_2]$, the bias term (originating from $\text{Var}(X)$) naturally vanishes since $\mathbb{E}[X_1 X_2]=\mathbb{E}[X_1]\mathbb{E}[X_2]=(\mathbb{E}[X])^2$ due to independence.

**Core Idea**: Replace the "square of a single sample set" with the "product of two independent Shot sub-samples" to form an unbiased estimate of the single-step error: $\ell^{M_1, M_2}_{\text{UEM}}=\mathbb{E}[\text{Shot}_{M_1}[\text{err}^{\text{EM}}_n]\cdot\text{Shot}_{M_2}[\text{err}^{\text{EM}}_n]]$.

## Method

### Overall Architecture
The PDE $\mathcal{L}[u](t,x)=\phi(t,x,u,\nabla u)$ is transformed into an FBSDE system $dX_t=\mu\,dt+\sigma\,dW_t$, $dY_t=\phi\,dt+Z_t^T\sigma\,dW_t$ via Itô's formula, where $Y_t=u(t,X_t)$ and $Z_t=\nabla u(t,X_t)$. Using EM discretization on a time grid $t_n=n\Delta t$, the single-step forward mapping is $F_n(x)=x+\mu\Delta t+\sigma\Delta W_n$ and the backward mapping is $B_n(x;u)=u(t_n,x)+\phi_u\Delta t+\nabla u\cdot\sigma\Delta W_n$. The single-step error is defined as $\text{err}^{\text{EM}}_n(x;u)=\frac{u(t_{n+1}, F_n(x))-B_n(x;u)}{\Delta t}$. Un-EM-BSDE samples $M_1+M_2$ independent Brownian increments $\Delta W_{n,i}$ at each step, splits them into two groups, averages them, and computes their product to obtain an unbiased single-step loss, which is then accumulated along the trajectory.

### Key Designs

1. **Sample-splitting Unbiased Estimator**:
    - **Function**: Replaces the biased second moment in $\ell_{\text{EM}}=\mathbb{E}[X^2]$ with an unbiased cross-moment $\mathbb{E}[X_1 X_2]$.
    - **Mechanism**: Define $\text{Shot}_M[\xi]=\frac{1}{M}\sum_{m=1}^M \xi_m$. Using $M_1+M_2$ i.i.d. Brownian increments, compute $M_1+M_2$ independent single-step errors, split them into two non-overlapping groups to calculate $\text{Shot}_{M_1}$ and $\text{Shot}_{M_2}$, resulting in the final loss $\ell^{M_1,M_2}_{\text{UEM}}=\mathbb{E}[\text{Shot}_{M_1}[\text{err}^{\text{EM}}_n]\cdot\text{Shot}_{M_2}[\text{err}^{\text{EM}}_n]]$. Lemma 4.1 proves $\ell^{M_1,M_2}_{\text{UEM}}=([\mathcal{L}[u_\theta]-\phi_{u_\theta}](t_n,x))^2+O(\Delta t^{1/2})$, which equals the square of the continuous-time PDE residual (excluding vanishing terms), completely removing the $\text{Tr}[(\sigma^T\nabla^2 u_\theta\sigma)^2]$ term from the EM bias.
    - **Design Motivation**: Traditional BSDE uses the same noise $\Delta W_n$ for both forward and backward steps, causing the variance of $\text{err}^{\text{EM}}_n$ to be absorbed into $\mathbb{E}[X^2]$ as bias. Using two independent noise sets separates the variance from the squared mean.

2. **Second-order-free**:
    - **Function**: Maintains the EM single-step update structure to avoid calculating $\nabla^2 u_\theta$.
    - **Mechanism**: Heun-BSDE is slow because the Itô-to-Stratonovich conversion introduces second-order spatial derivative correction terms, necessitating the Hessian. Un-EM-BSDE remains within the Itô framework; the single-step formula $B_n$ only contains $u$ and $\nabla u$, requiring only first-order backpropagation.
    - **Design Motivation**: In $d$-dimensional PDEs, the Hessian is a $d\times d$ matrix, and its AD computation cost is $O(d)$ times that of first-order gradients. For high-dimensional problems where $d=100$, this determines feasibility on GPUs—this is the root cause of Heun-BSDE's 42.91× overhead.

3. **Variance Control + Shotgun Universal Wrapper**:
    - **Function**: (a) Prove that the variance of the Un-EM estimator with $M_1=1, M_2=2$ is no larger than that of EM-BSDE; (b) apply the same sample-splitting logic to any single-step loss as a universal debiasing wrapper.
    - **Mechanism**: Theorem 4.3 proves that under certain conditions regarding $\alpha$ and $\beta$, $\mathbb{V}[\hat\ell^{M_1,M_2}_{\text{UEM}}]\leq\mathbb{V}[\hat\ell^M_{\text{SG}}]=\mathbb{V}[\hat\ell^M_{\text{SEM}}]\leq\mathbb{V}[\hat\ell_{\text{EM}}]$. Applying this product construction to the Shotgun loss (Un-SG) reduces RL2 by 2.67× on BSB hard constraints with only a 1.78× increase in training time.
    - **Design Motivation**: Sample-splitting can introduce additional variance since cross-moments are often noisier than second moments; variance analysis is crucial for practical utility.

### Loss & Training
The default experiment uses $M_1=M_2=5$. Baselines include Shotgun ($M=50$) and Multi-Shot EM ($M=10$) to align the sampling budget with $M_1+M_2=10$. The loss supports both soft constraints (terminal condition as an extra loss term $L_T$) and hard constraints (built-in via trial functions). Algorithm 1 shows a batched implementation: for batch size $B$, time steps $N$, and shots $M_1+M_2$, a tensor $X\in\mathbb{R}^{B\times(N+1)\times(M_1+M_2)\times d}$ stores all candidate states, allowing parallel computation of forward trajectories and single-step predictions $\hat Y[b,n+1,i]$, followed by group aggregation for the product.

## Key Experimental Results

### Main Results

RL2 error ($\times 10^{-2}$) across 5 benchmark PDEs (**bold** indicates best, <u>underline</u> indicates second-best):

| PDE / Constraint | EM-BSDE (Biased) | Shotgun (Biased) | Multi-Shot EM | Heun-BSDE (Unbiased) | FS-PINNs (Unbiased) | **Un-EM-BSDE (Ours)** |
|------------------|------------------|------------------|---------------|----------------------|----------------------|-----------------------|
| HJB soft         | 0.4055           | 1.1409           | 0.1617        | 0.1424               | **0.0867**           | <u>0.1348</u>         |
| BSB soft         | 0.3483           | 39.99            | 0.1046        | 0.1030               | **0.0478**           | <u>0.0814</u>         |
| AC soft          | 0.0462           | 0.0951           | <u>0.0206</u> | 0.0774               | 0.0325               | **0.0147**            |
| BSB hard         | 0.3456           | 0.1629           | 0.0739        | 0.0201               | **0.0048**           | <u>0.0120</u>         |
| PIDE hard        | 0.0374           | 0.4057           | 0.0245        | 0.1874               | **0.0137**           | <u>0.0226</u>         |

Training Time Multiplier (Table 1):

| Method               | Unbiased | 2nd-order-free | Training Time |
|----------------------|----------|----------------|---------------|
| EM-BSDE              | ✗        | ✓              | 1×            |
| Shotgun              | ✗        | ✓              | 0.75×         |
| Multi-Shot EM-BSDE   | ✗        | ✓              | 1.74×         |
| Heun-BSDE            | ✓        | ✗              | **42.91×**    |
| FS-PINNs             | ✓        | ✗              | **32.07×**    |
| **Un-EM-BSDE (ours)**| ✓        | ✓              | **1.79×**     |

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| Full Un-EM-BSDE | Achieve second-best or best in almost all settings |
| Sample-splitting wrapper on Shotgun (Un-SG) | RL2 decreased by 2.67× on BSB hard, time increased by 1.78× |
| Hard vs Soft constraint | Hard constraints are significantly more stable for complex dynamics (BZ, PIDE) |
| BZ (fully-coupled FBSDE) soft | Un-EM is at 5.18 magnitude, while Shotgun spikes to 86.53 |

### Key Findings
- **Efficiency is the killer feature**: In high-dimensional $d$ scenarios, Heun-BSDE and FS-PINNs may fail to run due to Hessian computation. Un-EM-BSDE's 1.79× time is the sweet spot.
- **Universality of the Wrapper**: Applying the same product construction to Shotgun immediately yields a 2.67× accuracy improvement, indicating this is a general debiasing technique.
- **Hard constraints are superior for complex dynamics**: Loss balancing issues in soft constraints are amplified in fully-coupled/jump scenarios. Hard constraints are more stable as they eliminate weight tuning.
- **No variance explosion**: Theorem 4.3 and experiments verify that the Un-EM estimator's variance is not greater than EM-BSDE.

## Highlights & Insights
- **Precise Application of Classical Statistics**: While sample-splitting is an old trick in statistical inference, plugging it precisely into the BSDE single-step loss to achieve simultaneous debiasing and efficiency reflects a deep understanding of the problem—bias resides in $\text{Var}(X)$, and independent sampling isolates it.
- **Universal Wrapper Design**: Section 5.3 abstracts the method into a framework where any biased single-step loss with a $\tau$-parameter can be constructed this way, providing value beyond a single algorithm.
- **Avoiding the Itô vs Stratonovich Dilemma**: Heun-BSDE forces Stratonovich integration to achieve unbiasedness, thereby introducing the Hessian. Un-EM achieves the same unbiasedness within the Itô framework through randomization.
- **Tight Theory-Experiment Coupling**: Lemma 4.1 (unbiasedness), Theorem 4.2 (consistency), and Theorem 4.3 (variance) are all supported by corresponding experimental results.

## Limitations & Future Work
- Theoretical assumptions require bounded $\mu, \sigma$ and $u_\theta\in C^{1,2}$, which only partially covers fully-coupled FBSDE and PIDE with unbounded coefficients or jump processes.
- The algorithm requires $M_1+M_2$ independent Brownian increments (default 10) per step. Compared to 1 in EM-BSDE, the batched implementation allocates 10× more tensor memory, which might be a bottleneck for very large $d$ or batch sizes.
- Experiments focused on $d\sim 100$; full ablation for truly large-scale ($d>1000$) PDEs is pending.
- Comparison with modern SOTA such as forward-backward dual-network methods (separate networks per step) is missing.
- Extending adaptive time-stepping for complex dynamics is noted as future work, as fixed $\Delta t$ may be sub-optimal for stiff/multi-scale PDEs.

## Related Work & Insights
- **vs EM-BSDE (Raissi 2024)**: The base method; Un-EM uses randomized products to eliminate bias with only 79% additional time.
- **vs Heun-BSDE (Park & Tu 2025)**: Both are unbiased, but Heun requires the Hessian (42.91× time), whereas Un-EM is Hessian-free.
- **vs Shotgun (Xu & Zhang 2025)**: Shotgun reduces bias by $1/M$ but does not eliminate it; the Un-EM wrapper makes it unbiased.
- **vs FS-PINNs (Park & Tu 2025)**: FS-PINNs minimizes squared PDE residuals sampled along SDE trajectories; it is unbiased but requires the Hessian.
- **vs Hu et al. (2025) bias-variance trade-off PINNs**: Shared logic of using independent samples to form a product for debiasing; this work specializes and extends it to the BSDE framework.

## Rating
- Novelty: ⭐⭐⭐⭐ Application of sample-splitting within BSDE losses is a clear and non-trivial contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered 5 standard PDEs + 2 complex extensions (BZ, PIDE) + wrapper generalization.
- Writing Quality: ⭐⭐⭐⭐⭐ The comparison in Table 1 effectively highlights the "unbiased + 2nd-order-free + time" contribution.
- Value: ⭐⭐⭐⭐⭐ Bringing unbiased BSDE back to EM-level training costs is a significant practical advancement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PhysicsCorrect: A Training-Free Approach for Stable Neural PDE Simulations](../../AAAI2026/physics/physicscorrect_a_training-free_approach_for_stable_neural_pde_simulations.md)
- [\[NeurIPS 2025\] Enforcing Governing Equation Constraints in Neural PDE Solvers via Training-free Projections](../../NeurIPS2025/physics/enforcing_governing_equation_constraints_in_neural_pde_solvers_via_training-free.md)
- [\[ICML 2026\] Hermite-NGP: Gradient-Augmented Hash Encoding for Learning PDEs](hermite-ngp_gradient-augmented_hash_encoding_for_learning_pdes.md)
- [\[ICML 2026\] EqGINO: Equivariant Geometry-Informed Fourier Neural Operators for 3D PDEs](eqgino_equivariant_geometry-informed_fourier_neural_operators_for_3d_pdes.md)
- [\[NeurIPS 2025\] Neural Emulator Superiority: When Machine Learning for PDEs Surpasses its Training Data](../../NeurIPS2025/physics/neural_emulator_superiority_when_machine_learning_for_pdes_surpasses_its_trainin.md)

</div>

<!-- RELATED:END -->
