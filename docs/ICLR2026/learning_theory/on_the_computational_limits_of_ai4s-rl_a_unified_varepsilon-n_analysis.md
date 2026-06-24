---
title: >-
  [Paper Note] On the Computational Limits of AI4S-RL: A Unified $\varepsilon$-$N$ Analysis
description: >-
  [ICLR 2026][Learning Theory][AI4Science] When using AI surrogate models (neural operators) as RL simulation environments to replace expensive PDE solvers, this paper proposes a unified $\varepsilon$-$N$ theoretical framework. By putting the surrogate discretization accuracy, RL agent grid resolution, and policy learning quality into a single probabilistic language, it derives the minimum computational cost $N^*(\varepsilon)$ required for unbiased value function estimation und…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Reinforcement Learning Theory"
  - "AI for Science"
  - "AI4Science"
  - "Surrogate Models"
  - "PAC Sample Complexity"
  - "Discretization Error"
  - "Computational Cost Trade-off"
date: 2026-05-08
content_hash: 33c27aa89ca5f021
---

# On the Computational Limits of AI4S-RL: A Unified $\varepsilon$-$N$ Analysis

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=BZnnIeeQox](https://openreview.net/forum?id=BZnnIeeQox)  
**Code**: To be confirmed  
**Area**: Learning Theory / Reinforcement Learning Theory / AI for Science  
**Keywords**: AI4Science, Surrogate Models, PAC Sample Complexity, Discretization Error, Computational Cost Trade-off

## TL;DR
When using AI surrogate models (neural operators) as RL simulation environments to replace expensive PDE solvers, this paper proposes a unified $\varepsilon$-$N$ theoretical framework. By putting the surrogate discretization accuracy, RL agent grid resolution, and policy learning quality into a single probabilistic language, it derives the minimum computational cost $N^*(\varepsilon)$ required for unbiased value function estimation under a given precision $\varepsilon$ and confidence $1-\delta$. It also provides closed-form optimal allocation ratios $K^*$ for "surrogate accuracy vs. RL accuracy" across different physical systems.

## Background & Motivation

**Background**: Reinforcement Learning is highly promising for control problems constrained by PDEs, such as plasma confinement, climate adaptation, and turbulence control. However, RL typically requires millions of environment interactions, while high-fidelity PDE solvers are extremely expensive per step, making them prohibitively slow. A natural remedy is replacing the PDE solver with an AI4S surrogate model (neural operators, PINNs, etc.) to accelerate simulations by several orders of magnitude.

**Limitations of Prior Work**: Classic RL theory (e.g., PAC bounds for UCB-VI) assumes transition noise is **stochastic** and can be averaged out via repeated sampling. Robust RL treats model uncertainty as worst-case random perturbations. However, errors introduced by AI4S surrogates are **deterministic**—stemming from spatial discretization, time integration, and boundary approximations. Such errors do not disappear with more samples; instead, they systematically bias the policy. An intuitive example provided: if the surrogate's discretization fails to capture the vertical equilibrium in a cart-pole system, the learned controller will consistently overshoot regardless of the RL action resolution—the policy accuracy is ultimately bottlenecked by the surrogate resolution.

**Key Challenge**: There is a neglected coupling between the surrogate model's grid resolution and the RL agent's decision resolution. Refining either side blindly incurs high costs without guaranteed policy improvement. There exists an optimal ratio of "how to distribute computing power," which has not been theoretically characterized until now.

**Goal**: Given a fixed computational budget, answer "how should surrogate resolution be coordinated with RL agent discretization to minimize total cost while guaranteeing policy accuracy." This is decomposed into: (1) conditions for PAC learnability; (2) how errors propagate and couple between the RL $\leftrightarrow$ PDE spaces; and (3) the optimal allocation of total computing power.

**Key Insight**: The authors observe that **deterministic surrogate error can be reformulated as a statistical inference problem**. By perturbing initial conditions within the range of observation uncertainty and sampling multiple trajectories, the question of "which grid cell the next state falls into" becomes a frequency-based classification problem. Thus, numerical analysis (spectral theory, Sobolev estimates) and RL learning theory (PAC sample complexity) can be unified within the same probabilistic framework.

**Core Idea**: Use "perturbation resampling + grid classification" to convert deterministic discretization errors into statistically estimable objects. Subsequently, use spectral analysis to characterize error amplification and PAC bounds to characterize sample requirements. Finally, provide the $\varepsilon$-$N$ framework and system-dependent closed-form optimal resolution ratios $K^*$. The paper focuses on **tabular RL** to establish "necessary conditions for learnability," serving as a lower-bound guide for deep RL hyperparameter design.

## Method

### Overall Architecture
This paper is not an algorithm but a **theoretical derivation chain**: starting from "whether surrogate errors can be statistically identified," moving through "how the two-space errors couple," to "how to optimally distribute total computing power," and finally validated numerically on four real physical systems. The process follows a four-step analysis.

Assume an RL agent interacts with a hybrid system where an AI4S surrogate approximates the PDE environment. Subscript $r$ denotes the RL side and $w$ denotes the physical world side; $K=\Delta x_r/\Delta x_w$ is the spatial resolution ratio, and $H_r, H_w$ are the respective time steps. The surrogate error satisfies $\lVert G(f)-G_\theta(f)\rVert_{L^2}\le C_1 H^s + C_2 X^d$.

The four steps are: ① **Spectral Analysis Layer**: Determining whether a single PDE state can be uniquely identified under observation grid $\Delta y$ (deciding PAC reachability) and providing the required sampling frequency; ② **Error Coupling Layer**: Analyzing the accumulated error from "lifting" RL actions to PDE space and "projecting" physical states back to RL space to obtain the scaling law of forward projection error rate $\rho$ with respect to $H_r, K$; ③ **Cost Allocation Layer**: Aligning RL sample complexity with physical simulation costs to derive the closed-form optimal $K^*$; ④ **Cross-System Verification Layer**: Calculating $K^*$ and cost scaling for tokamak, airfoil, teppanyaki, and cart-pole systems.

### Key Designs

**1. Converting Deterministic Error to Grid Classification: Theorem 1 $\delta$-Confidence Sample Complexity**

Deterministic surrogate error is problematic because it won't be averaged out, precluding the direct application of classic PAC bounds. The solution is: state $y_0$ is only observed at grid precision $\Delta y$. When evolved with deterministic solver $f$, the next state falls within an interval $[y_1^{\min},y_1^{\max}]$. **Criterion**: If this interval spans more than one full PDE grid cell, the next state cannot be uniquely identified, and PAC is unreachable. If it covers only one full cell, the correct cell can be estimated to any confidence by repeatedly perturbing the initial state.

If $p$ is the frequency of falling into the correct cell and $q=p^{(j)}_{\max}$ is the max frequency among competitors, the number of forward predictions required for $1-\delta$ confidence is:

$$n = O\!\left(\frac{\log(1/\delta)}{\min_j\left(\Delta y^{(j)}-p^{(j)}_{\max}\right)^2}\right).$$

**2. Spectral Theory Constraint $\Delta t \lesssim 1/\lambda_1$: Physical Prerequisite for Classification**

What determines if the frequency difference collapses to zero? The spectral properties of the PDE. For a nonlinear solver $f_t(y_0)$, the linearization of perturbation $\eta$ has a modal decomposition $\eta(t)\sim\sum_k \hat\eta_k e^{\mathrm{Re}(\lambda_k)t}\psi_k$. To keep grids separable, the dominant perturbation growth must remain bounded, requiring $e^{\lambda_1\Delta t}\lesssim 1$, where $\lambda_1:=\max_k\{\mathrm{Re}(\lambda_k)\}$. This yields the time step constraint:

$$\Delta t \lesssim 1/\lambda_1.$$

If $\Delta t>1/\lambda_1$, perturbation growth exceeds $\Delta y$, effectively destroying separability. Thus, $\lambda_1$ (system "chaoticity") is a key physical quantity throughout the paper.

**3. $\rho$-$K$ Analysis: Scaling Law $O(1/H_r + 1/K^d)$ (Theorem 2)**

A single RL step's total error $\Delta_{\text{total}}$ can be decomposed into four RL-side terms plus the PDE surrogate error amplified by the time scale ratio $\Delta t_r/\Delta t_w$. Defining $\rho:=1-p$ as the probability of misclassification after one "RL $\to$ AI4S $\to$ Projection" cycle, the high-resolution limit simplifies to:

$$\rho = O\!\left(\frac{1}{H_r}+\frac{1}{K^d}\right),$$

where $d$ is the PDE spatial dimension. This scaling law shows that refining RL temporal resolution ($1/H_r$) and surrogate spatial resolution ($1/K^d$) are **substitutable but cannot compensate for each other infinitely**.

**4. Optimal Computational Cost Allocation: Theorem 3**

Using identification analysis, the sample complexity for UCB-VI is refined from $O(H_r^4 S_r A_r)$ to $O(H_r^3 S_r A_r\cdot\log(1/\delta))$. Aligning costs between both sides, the total cost $\mathrm{Cost}(K)$ is minimized at the optimal resolution ratio:

$$K^* = \left(\frac{\alpha+\beta+2d}{(\alpha+\beta)(1-H_r^{-1})}\right)^{1/d}\approx\left(\frac{\alpha+\beta+2d}{\alpha+\beta}\right)^{1/d}\cdot\exp\!\left(\frac{1}{d\lambda_1}\right),$$

where $\alpha, \beta$ are determined by the "actuation topology." Boundary actuation systems require quadratic refinement due to trace theorem errors.

### Cross-System Closed-Form Results
(Selection from Table 1):

- **Tokamak Control** (Boundary actuation, $d=3$): $K^*=(7/4)^{1/3}\exp\!\left(\frac{1}{3\lambda_1}\right)$.
- **Airfoil Control**: $K^*=(5/3)^{1/2}\exp\!\left(\frac{1}{2\lambda_1}\right)$.
- **Teppanyaki Plate** (Internal actuation, 2D heat): $K^*=(7/3)^{1/2}\exp\!\left(\frac{1}{2\lambda_1}\right)$.
- **Cart-Pole System** (Low-dim ODE): $K^*=2\exp\!\left(\frac{1}{\lambda_1}\right)$.

## Key Experimental Results

### Main Results (Cart-pole Resolution Heatmap)

| Algorithm | Optimal Config $(K,\ \log_{H_w}H_r)$ | Optimal Samples $N$ | Deviation Cost |
|------|------|------|------|
| Tabular RL | $(1.5,\ 1/3)$ | $10^{3.65}\approx4500$ | Worst config needs $\sim 10$x more |
| Q-Learning (DQN) | $(2.0,\ 1/2)$ | $10^{2.65}\approx450$ | One order lower than tabular |

### Key Findings
- **Non-monotonic Optimal Scale**: Resolution is not "the finer the better." Excessive coarseness ($K=8.0$) leads to non-convergence, validating Theorem 1.
- **Guidance for Deep RL**: The optimal $K^*=6$ for PPO on teppanyaki matches Theorem 3, showing tabular lower bounds carry over to deep RL.
- **Infeasibility of Brute Force**: Search space for resolutions in high-dim PDEs is too large, necessitating theoretical prediction.

## Highlights & Insights
- **Innovation**: Converting "deterministic error" into "statistical classification" allows spectral theory and PAC complexity to be merged.
- **Applicability**: The $\rho=O(1/H_r+1/K^d)$ scaling law is reusable for any dual-layer discrete system (surrogate + agent).
- **Physical Grounding**: Embedding $\lambda_1$ into $K^*$ provides a clear "stop signal" for surrogate refinement based on system chaos.

## Limitations & Future Work
- **Tabular Focus**: Strictly proven for tabular RL; extensions to function approximation remain an open problem.
- **Static Surrogates**: Does not consider adaptive grid refinement within the RL loop.
- **Discretization Only**: Ignores approximation and generalization errors of neural networks.

## Related Work & Insights
- **vs. Classic PAC RL**: They assume stochastic noise; this paper addresses **deterministic discretization bias** and improves RL complexity via identification.
- **vs. Robust RL**: This paper **exploits** discretization structure rather than treating it as an adversarial perturbation.
- **vs. Control Theory**: Explicitly quantifies the impact of surrogate resolution on learning costs, which is often ignored in classical adjoint methods.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unified $\varepsilon$-$N$ framework for AI4S-RL.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid verification on four systems, though deep RL scale is somewhat small.
- Writing Quality: ⭐⭐⭐⭐ Clear derivation chain.
- Value: ⭐⭐⭐⭐⭐ Clear engineering guidance for compute-constrained AI4S-RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reshaping Reasoning in LLMs: A Theoretical Analysis of RL Training Dynamics through Pattern Selection](reshaping_reasoning_in_llms_a_theoretical_analysis_of_rl_training_dynamics_throu.md)
- [\[ICLR 2026\] Computational Bottlenecks for Denoising Diffusions](computational_bottlenecks_for_denoising_diffusions.md)
- [\[ICLR 2026\] Language Identification in the Limit with Computational Trace](language_identification_in_the_limit_with_computational_trace.md)
- [\[ICLR 2026\] UniCon: Unified Framework for Efficient Contrastive Alignment via Kernels](unicon_unified_framework_for_efficient_contrastive_alignment_via_kernels.md)
- [\[ICLR 2026\] Characterizing Pattern Matching and Its Limits on Compositional Task Structures](characterizing_pattern_matching_and_its_limits_on_compositional_task_structures.md)

</div>

<!-- RELATED:END -->
