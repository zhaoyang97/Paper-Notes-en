---
title: >-
  [Paper Note] Incorporating Expert Priors into Bayesian Optimization via Dynamic Mean Decay
description: >-
  [ICLR 2026][Optimization & Theory][Bayesian Optimization] The proposed framework, DynMeanBO, directly incorporates expert priors (distribution of the optimum $\pi(x)$) into the **mean function** of a Gaussian Process, utilizing a weight that decays over iterations to prioritize the prior early and phase it out later. This results in a prior-informed Bayesian Optimization fram
tags:
  - ICLR 2026
  - Optimization & Theory
  - Bayesian Optimization
  - Expert Prior
  - Gaussian Process
  - Mean Function
  - Dynamic Decay
  - Robustness
date: 2026-05-08
content_hash: 455c3662a0c758ee
---
# Incorporating Expert Priors into Bayesian Optimization via Dynamic Mean Decay

**Conference**: ICLR 2026  
**Code**: [https://github.com/quchongqi/DynMeanBO](https://github.com/quchongqi/DynMeanBO)  
**Area**: Bayesian Optimization / Black-box Optimization / Probabilistic Methods  
**Keywords**: Bayesian Optimization, Expert Prior, Gaussian Process, Mean Function, Dynamic Decay, Robustness  

## TL;DR
The proposed framework, DynMeanBO, directly incorporates expert priors (distribution of the optimum $\pi(x)$) into the **mean function** of a Gaussian Process, utilizing a weight that decays over iterations to prioritize the prior early and phase it out later. This results in a prior-informed Bayesian Optimization framework that is compatible with any acquisition function, incurs nearly zero additional overhead, and remains robust to poor priors.

## Background & Motivation
**Background**: Bayesian Optimization (BO) is a primary tool for black-box optimization—fitting expensive objective functions with a Gaussian Process (GP) and using acquisition functions (e.g., EI, UCB) to balance exploration and exploitation. In real-world scenarios like HPO, materials discovery, and robotics, evaluation budgets are often limited to a few dozen samples, making "sample efficiency" critical.

**Limitations of Prior Work**: In many scenarios, domain experts possess knowledge of "roughly where the optimal solution lies." Leveraging this prior can significantly accelerate early convergence. However, existing prior-informed BO methods have drawbacks: BOPro is tied to the TPE structure; Ramachandran et al. embed priors into the kernel (where a poor prior can collapse performance); $\pi$BO uses a weighting mechanism in the acquisition function (a heuristic that doesn't explicitly model the prior in the surrogate); and ColaBO is flexible but limited to Monte Carlo acquisition functions with high computational costs.

**Key Challenge**: It is difficult to simultaneously achieve simplicity, generality, and robustness. Methods are often too complex, tied to specific sampling strategies, or overly sensitive to prior quality, leading to optimization failures if the expert's judgment is incorrect.

**Goal**: To develop a **simple, general, and robust** framework—one that accelerates when the prior is good, avoids failure when the prior is bad, requires no specific acquisition function, has negligible overhead, and provides theoretical convergence guarantees.

**Core Idea**: Inject expert priors into the **mean function** of the GP rather than the kernel or acquisition function (`Model-level Injection`), and pair it with an exponentially decaying weight as observations increase (`Dynamic Decay`). This allows BO to benefit from expert knowledge early on while automatically reverting to standard BO behavior later (`Robust Floor`).

## Method

### Overall Architecture
All modifications in DynMeanBO occur strictly within the GP mean function. While standard BO typically uses a zero mean $m(x)=0$, this work employs a dynamic mean $m_n(x)$ constructed from expert priors that fades over iterations. The process starts with mixed initial sampling using the expert prior, followed by updating the dynamic mean, recomputing the GP posterior, and selecting the next point using any acquisition function.

```mermaid
flowchart LR
    A[Expert Prior π(x)<br/>Optimum Location Dist.] --> B[Prior Mean m_prior=A·π+B]
    A --> C[Mixed Initialization<br/>ρ Proportion from π + Sobol]
    B --> D[Dynamic Decay Mean<br/>m_n=γ_n·m_prior+1-γ_n·μ_0]
    C --> E[GP Posterior p_f|D_n]
    D --> E
    E --> F[Any Acquisition α_x<br/>EI/UCB/MES/...]
    F --> G[Select Next x_n]
    G --> H{Update Dataset D_n}
    H -->|γ_n Decays| D
```

### Key Designs

**1. Injecting Prior into GP Mean Function: Knowledge injection at the model level rather than the acquisition level.** Experts typically do not know the analytical form of $f$, but can provide a prior distribution $\pi(x)$ for the location of the optimum (e.g., a single Gaussian for one region, Gaussian Mixture for multiple regions). This is treated directly as the GP prior mean:

$$m_{\text{prior}}(x) = A \cdot \pi(x) + B$$

where $A>0$ scales the magnitude of $\pi(x)$ and $B$ is an additive shift. In the absence of observations, $m_{\text{prior}}(x)$ outlines a rough landscape consistent with expert beliefs. This differs from $\pi$BO (acquisition weighting) and Ramachandran (kernel modification); the mean function is the "softest" component of a GP, ensuring the prior influences the surrogate without being as rigid or destructive as a kernel modification.

**2. Dynamic Mean Decay: Prior dominance early, automatic exit later.** As more points are evaluated, the data itself characterizes $f$ more accurately. Reliance on the expert prior should decrease—both to reflect growing confidence in observations and to provide robustness against bad priors. Inspired by $\pi$BO's decay, the mean at iteration $n$ is defined as a convex combination of the prior mean and the baseline mean $\mu_0(x)$ (usually zero):

$$m_n(x) = \gamma_n \cdot m_{\text{prior}}(x) + (1-\gamma_n)\cdot \mu_0(x), \qquad \gamma_n = \exp\big(-\lambda(n-N_0)\big)$$

where $\lambda>0$ controls the decay rate and $N_0$ is the number of initial evaluations. $\gamma_n$ decays exponentially from 1 to 0. This ensures that even if the expert focuses on a region far from the global optimum, the decay eventually washes out the incorrect bias.

**3. Mixed Initialization: Supplementing starts with priors without sacrificing coverage.** During initialization, a proportion $\rho$ of initial points ($\lfloor\rho N_0\rfloor$) are sampled from the expert prior $\pi(x)$, while the remaining $N_0-\lfloor\rho N_0\rfloor$ points use a Sobol sequence. This hybrid strategy builds an accurate surrogate model faster in promising regions while maintaining global exploration.

**4. Non-intrusive to Acquisition Functions + Convergence Guarantees.** Since modifications are limited to the mean function, DynMeanBO is "plug-and-play" with any acquisition function (PI, EI, LogEI, TS, UCB, KG, MES). Theoretically, the authors provide two guarantees: under **EI** (following RKHS assumptions), DynMeanBO-EI maintains the same asymptotic convergence rate as standard BO-EI; under **UCB**, the cumulative regret satisfies $R_N\le C_1\sqrt{N\beta_N G_N}+C_2\sum_{n=1}^N\gamma_n$. As long as $\sum_n\gamma_n<\infty$ (sufficiently fast decay), it matches the order of BO-UCB $R_N=O(\sqrt{N\beta_N G_N})$.

## Key Experimental Results

### Main Results
Tasks: Synthetic functions Hartmann (4D), Levy (5D), Hartmann (6D), Rosenbrock (6D), Stybtang (7D) + PD1 HPO benchmarks (WMT/CIFAR/LM1B, 4D). Compared against $\pi$BO, ColaBO, and standard BO.

| Setup | DynMeanBO Performance | Description |
|------|----------------|------|
| Good Prior + 7 Acq. Functions | Consistently faster than standard BO | Speedup across PI/EI/LogEI/TS/UCB/KG/MES, especially early on |
| Good Prior vs. πBO/ColaBO | Performance parity | On par with SOTA prior-informed methods |
| Bad Prior (70% deviation) | Robust, converges to standard BO | Significantly more robust than ColaBO; clear advantage over πBO |
| Computational Cost | 1.4×–6.4× faster than πBO/ColaBO | Nearly zero overhead relative to standard BO |

### Ablation Study

| Ablation Dimension | Conclusion |
|----------|--------------|
| Decay Rate $\lambda$ | Sensitivity analysis in Appendix J shows adjustable decay pacing |
| Init. Ratio $\rho$ | Appendix K analyzes the selection of sampling proportions from $\pi(x)$ |
| Prior Scaling $A,B$ | Appendix B provides interpretation and sensitivity analysis |
| Prior Quality (Strong/Weak/Wrong) | Appendix L analyzes three quality levels under different prior variances |
| High-dim Scaling | Verified on Levy (20D) and Rosenbrock (20D) in Appendix I |

### Key Findings
- **Good priors accelerate, bad priors don't break**: DynMeanBO matches SOTA with good priors and shows the strongest robustness with bad priors, validating that "dynamic decay" achieves both speed and safety.
- **Expert dividends are concentrated early**: Since BO budgets are scarce, the gains of DynMeanBO are most prominent in the first few dozen iterations—exactly the phase most critical for expensive real-world tasks.
- **"Bad priors" can still be useful**: On PD1 (LM1B), even shifted priors outperformed vanilla BO because the "bad" prior, while not at the global optimum, still pointed toward a high-quality local region.
- **Cheaper SOTA**: DynMeanBO is several times faster per step than $\pi$BO/ColaBO while maintaining competitive performance.

## Highlights & Insights
- **Optimal Injection Point**: The mean function is the ideal component for "location priors"—it directly shapes the predicted landscape but remains less structurally rigid than the kernel.
- **Unifying Speed and Robustness via Decay**: The same $\gamma_n$ mechanism handles both the early exploitation of priors and the late-stage elimination of bias, solving two contradictory requirements.
- **True Generality**: No modification to the acquisition function means any existing BO toolchain (e.g., BoTorch) can be used directly with negligible implementation cost.

## Limitations & Future Work
- **Prior Form Dependency**: The method assumes experts can provide a distribution $\pi(x)$. Robustly constructing $\pi(x)$ from vague human preferences remains an open problem.
- **Hyperparameter Sensitivity**: $\lambda$ (decay), $\rho$ (init. ratio), and $A, B$ (scaling) need to be set; while sensitivity analyses exist, a self-adaptive mechanism is currently lacking.
- **Scalability**: While 20D results are in the appendix, validation on very high-dimensional HPO or complex combinatorial spaces is still needed.
- **Future Work**: Plans to integrate multi-fidelity optimization and parallel evaluations to further increase speed and scale.

## Related Work & Insights
- **$\pi$BO (Hvarfner 2022)**: Multiplies acquisition functions by a prior weight with decay. DynMeanBO's decay is inspired by this, but placing the prior in the surrogate model is more principled and general.
- **ColaBO (Hvarfner 2024)**: Flexible but limited to MC acquisition functions with high cost. DynMeanBO is significantly faster and compatible with non-MC functions.
- **Kernel Injection (Ramachandran 2020)**: Sensitivity to bad priors highlights the advantage of DynMeanBO's "inject-into-mean + decay" approach for robustness.
- **Insight**: When injecting external knowledge into a probabilistic model, the *point* of injection is often more important than the *strength*. Choosing a soft, reversible, and locally influential component (like the mean) paired with an evidence-based exit strategy provides a clean trade-off between utilization and robustness.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of mean injection and dynamic decay is elegant and addresses a key pain point; the decay mechanism itself is a clever re-application of previous ideas.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid coverage across 7 acquisition functions, multiple tasks, and robustness scenarios, supported by extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, clean derivation, and strong alignment between theory and experiments.
- Value: ⭐⭐⭐⭐ Highly practical for expensive optimization (HPO, materials, control), offering a plug-and-play solution that is faster than SOTA with theoretical guarantees.

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[ICLR 2026\] Local Entropy Search over Descent Sequences for Bayesian Optimization](local_entropy_search_over_descent_sequences_for_bayesian_optimization.md)
- [\[CVPR 2026\] DABO: Difficulty-Aware Bayesian Optimization with Diffusion-Learned Priors](../../CVPR2026/optimization/dabo_difficulty-aware_bayesian_optimization_with_diffusion-learned_priors.md)
- [\[ICLR 2026\] From Sorting Algorithms to Scalable Kernels: Bayesian Optimization in High-Dimensional Permutation Spaces](from_sorting_algorithms_to_scalable_kernels_bayesian_optimization_in_high-dimens.md)
- [\[ICLR 2026\] Generative Bayesian Optimization: Generative Models as Acquisition Functions](generative_bayesian_optimization_generative_models_as_acquisition_functions.md)
- [\[ICLR 2026\] Symmetry-Aware Bayesian Optimization via Max Kernels](symmetry-aware_bayesian_optimization_via_max_kernels.md)

</div>

<!-- RELATED:END -->
