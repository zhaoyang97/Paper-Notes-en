---
title: >-
  [Paper Note] Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler
description: >-
  [ICML 2026][Optimization][SAM] This paper generalizes the Polyak step size to USAM/SAM, providing a sharpness-aware scheduler that does not rely on manual learning rate tuning. Its stability and performance are verified…
tags:
  - "ICML 2026"
  - "Optimization"
  - "SAM"
  - "USAM"
  - "Polyak step size"
  - "Adaptive learning rate"
  - "Convergence theory"
date: 2026-05-08
content_hash: bbb0331eea9553b8
---

# Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler

**Conference**: ICML 2026  
**arXiv**: [2606.01827](https://arxiv.org/abs/2606.01827)  
**Code**: https://github.com/dimitris-oik/sam_sps  
**Area**: Optimization / Sharpness-Aware Minimization  
**Keywords**: SAM, USAM, Polyak step size, Adaptive learning rate, Convergence theory  

## TL;DR
This paper generalizes the Polyak step size to USAM/SAM, providing a sharpness-aware scheduler that does not rely on manual learning rate tuning. Its stability and performance are verified through convex optimization theory and CIFAR experiments.

## Background & Motivation
**Background**: Sharpness-Aware Minimization (SAM) encourages the search for flatter solutions by optimizing the worst-case perturbation loss within a parameter neighborhood, often improving generalization in deep models. Standard SAM constructs a perturbation point along the gradient direction and then computes the gradient update at that point. USAM simplifies this by removing perturbation normalization, facilitating theoretical analysis.

**Limitations of Prior Work**: SAM/USAM is highly sensitive to the learning rate. Practical training usually requires repeated searching for fixed learning rates or schedulers across various datasets, networks, and sharpness radii. Existing adaptive SAM methods either lack strong theoretical foundations or require extra assumptions, such as bounded variance, bounded gradients, or growth conditions.

**Key Challenge**: The generalization benefits of SAM come from a non-zero perturbation radius $\rho$, but optimization theory suggests that a non-zero $\rho$ makes stable updates more difficult. Furthermore, the more fine-tuned the learning rate needs to be, the higher the practical cost of using SAM. This paper attempts to use a Polyak-type closed-form learning rate to reduce tuning while maintaining provable convergence.

**Goal**: To derive Polyak-type schedulers for USAM in deterministic and stochastic scenarios, prove convergence rates for strongly convex and convex objectives, and test whether this scheduler can match or exceed fine-tuned SAM/USAM on deep learning benchmarks.

**Key Insight**: The classic Polyak step size is derived from minimizing the upper bound of $\|x_{t+1}-x^*\|^2$. The update gradient in USAM is not at $x_t$ but at the perturbation point $e_t=x_t+\rho_t\nabla f(x_t)$. By re-deriving this distance bound, a sharpness-aware version of the Polyak learning rate can be obtained.

**Core Idea**: The $f(x_t)-f^*$ and $\nabla f(x_t)$ in the Polyak step size are replaced with the loss and gradient at the perturbation point $e_t$, with a perturbation term $\rho_t\langle\nabla f(e_t),\nabla f(x_t)\rangle$ subtracted.

## Method
The primary contribution is a scheduler rather than a new optimization objective. It starts from the deterministic update of USAM: first taking $e_t=x_t+\rho_t\nabla f(x_t)$, and then updating $x_t$ using $\nabla f(e_t)$. The authors expand $\|x_{t+1}-x^*\|^2$ and utilize convexity to construct a quadratic upper bound with respect to the learning rate $\gamma_t$. Minimizing this bound yields the USAM Polyak scheduler.

### Overall Architecture
In deterministic USAM, the scheduler is defined as $\gamma_t=(f(e_t)-f^*-\rho_t\langle\nabla f(e_t),\nabla f(x_t)\rangle)/\|\nabla f(e_t)\|^2$. If the numerator could be negative, a $[\cdot]_+$ safeguard can be added; however, the paper proves that for convex, $L$-smooth objectives with $\rho_t\le 1/L$, the numerator is naturally non-negative, making the safeguard redundant.

In stochastic finite-sum scenarios, $f$ is replaced by the mini-batch objective $f_{S_t}$, and the global optimum is replaced by a mini-batch lower bound $\ell^*_{S_t}$. To avoid excessively large steps in stochastic settings, the scheduler includes an upper bound $\gamma_b$, resulting in the Capped Stochastic Polyak Scheduler. Specifically, when $\rho_t=0$, these formulas reduce to the classic GD Polyak step size and SPSmax, respectively.

### Key Designs
1.  **Polyak Upper Bound Derivation for USAM**:
    - **Function**: Transitions the learning rate from a hyperparameter requiring tuning to a closed-form quantity determined by current loss and gradients.
    - **Mechanism**: Expands $\|x_{t+1}-x^*\|^2-\|x_t-x^*\|^2$, where $x_{t+1}=x_t-\gamma_t\nabla f(e_t)$. Since $e_t=x_t+\rho_t\nabla f(x_t)$, an extra term $\rho_t\langle\nabla f(e_t),\nabla f(x_t)\rangle$ appears in the bound and must be subtracted from $f(e_t)-f^*$.
    - **Design Motivation**: Ignoring the difference between the perturbation point and the origin causes the Polyak step size to overestimate the safe descent space. This subtraction term is precisely the correction for sharpness-aware updates relative to standard GD.

2.  **Stochastic Capped Scheduler and Lower Bound**:
    - **Function**: Extends the deterministic scheduler to mini-batch training.
    - **Mechanism**: The stochastic step size is $\gamma_t=\min\{(f_{S_t}(e_t)-\ell^*_{S_t}-\rho_t\langle\nabla f_{S_t}(e_t),\nabla f_{S_t}(x_t)\rangle)/\|\nabla f_{S_t}(e_t)\|^2,\gamma_b\}$. For non-negative losses, one can often set $\ell^*_{S_t}=0$, while $\gamma_b$ controls the maximum step size under stochastic noise.
    - **Design Motivation**: Success with SPSmax suggests that stochastic Polyak steps need capping to ensure stability; this paper retains this structure, as USAM-SPS naturally corresponds to existing SGD-SPS theory.

3.  **Deterministic and Stochastic Convergence Theory**:
    - **Function**: Demonstrates that the scheduler is a theory-grounded mechanism rather than just an empirical trick.
    - **Mechanism**: For strongly convex deterministic objectives, USAM-PS satisfies linear convergence $\|x_T-x^*\|^2\le (1-\mu(1-L\rho)^2/(4L))^T\|x_0-x^*\|^2$; for convex objectives, the Cesaro average reaches $O(1/T)$. In stochastic scenarios, the method converges to a neighborhood controlled by $\sigma^2=\mathbb{E}_{S_t}[f_{S_t}(x^*)-\ell^*_{S_t}]$; this neighborhood vanishes in interpolation settings where $\sigma^2=0$.
    - **Design Motivation**: This fills the theoretical gap often found in adaptive SAM methods and avoids assumptions like extra noise or bounded gradients required by some prior work.

### Loss & Training
The theoretical section handles general convex/strongly convex finite-sum objectives. For experiments, cross-entropy is used on DNNs (ResNet-20/32) using CIFAR-10/100, with a mini-batch size of 128, 100 epochs, and standard data augmentation. The Stochastic Polyak Scheduler for USAM/SAM uses $\ell^*_{S_t}=0$ and $\gamma_b=1.0$, and is compared against tuned constant learning rates and cosine annealing.

## Key Experimental Results

### Main Results
The table below extracts ResNet-32 results on CIFAR-100. The metric is test accuracy (higher is better).

| Method | $\rho=0.1$ | $\rho=0.2$ | $\rho=0.3$ | $\rho=0.4$ |
| :--- | :--- | :--- | :--- | :--- |
| Constant USAM | 90.56±0.18 | 90.45±0.34 | 90.25±0.10 | 89.56±0.07 |
| USAM + Cosine Annealing | 90.01±0.32 | 88.77±0.26 | 88.05±0.23 | 86.52±0.04 |
| USAM + Stochastic Polyak Scheduler | 91.81±0.04 | 92.23±0.22 | 92.24±0.30 | 92.01±0.12 |
| Constant SAM | 90.17±0.11 | 90.53±0.02 | 89.61±0.10 | 88.64±0.13 |
| SAM + Cosine Annealing | 90.49±0.02 | 89.03±0.13 | 87.05±0.24 | 84.61±0.34 |
| SAM + Stochastic Polyak Scheduler | 91.61±0.12 | 92.24±0.07 | 91.70±0.15 | 90.79±0.16 |

### Ablation Study
The ablation and analysis primarily compare different schedulers, various sharpness radii, and theoretical conditions.

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| USAM-PS vs. Constant USAM | Fastest convergence in ridge regression | Compared to guaranteed constant steps, Polyak scheduler reaches the optimum faster in deterministic settings. |
| USAM-SPS vs. Stochastic USAM baseline | Convergence to solution neighborhood | Consistent with stochastic theory without requiring per-problem learning rate tuning. |
| Comparison with AdaSAM / LightSAM | Minimum iterations to optimum | While other adaptive SAMs work, this method provides theoretical guarantees for strong convexity. |
| Increasing $\rho$ | Cosine accuracy drops; SPS stays stable | At $\rho=0.4$, USAM-SPS maintains 92.01% while Cosine USAM drops to 86.52%. |
| Normalized SAM version | Similar trends as USAM | Although derived for USAM, Polyak principles transfer to standard SAM. |
| Interpolation scenario $\sigma^2=0$ | Stochastic neighborhood term vanishes | In over-parameterized tasks, the scheduler can theoretically converge to the optimum. |

### Key Findings
- The Stochastic Polyak Scheduler shows the most significant advantage under large sharpness radii. Cosine annealing degrades rapidly as $\rho$ increases, whereas SPS accuracy remains stable.
- Theoretically, the method reduces to existing Polyak/SPSmax when $\rho=0$; thus, it unifies standard adaptive stepping with sharpness-aware updates.
- For the normalized update of SAM, the paper does not offer an equivalent non-negativity proof, but experiments show the safeguard is rarely triggered, and CIFAR performance remains superior to tuned baselines.

## Highlights & Insights
- The derivation is elegant: the USAM Polyak step simply adds an inner-product subtraction term to the GD step, directly corresponding to the geometric discrepancy of "stepping to the perturbation point before calculating the gradient."
- It tightly bridges the gap between the practical pain points of SAM and optimization theory. Unlike many adaptive optimizers that only provide empirical results, this work justifies the scheduler in convex/strongly convex settings.
- The most compelling evidence is the $\rho$ scan. A truly useful scheduler should stabilize training as the sharpness radius varies, rather than only performing well at a single tuned point.

## Limitations & Future Work
- Theory covers smooth convex and strongly convex objectives, whereas deep networks are non-convex. CIFAR experiments show empirical viability but do not directly prove convergence for large-scale non-convex training.
- The scheduler still requires an estimate of the lower bound; while 0 works for cross-entropy, the choice of $\ell^*_{S_t}$ may affect stability in more general tasks.
- The sharpness radius $\rho$ remains a critical hyperparameter. While learning rate tuning is reduced, the adaptive selection of $\rho$ is not fully addressed.
- Systemic validation is missing for large-scale LLM training, AdamW settings, and interactions with momentum or weight decay.

## Related Work & Insights
- **vs. Standard SAM**: Standard SAM requires an external learning rate schedule; this work makes the step size a closed-form quantity based on mini-batch loss and perturbation gradients.
- **vs. SPSmax**: SPSmax is a stochastic Polyak step for SGD; this work reduces to SPSmax when $\rho=0$ and extends it to USAM/SAM when $\rho>0$.
- **vs. AdaSAM / LightSAM / SA-SAM**: These methods also adaptively adjust SAM, but have different theoretical conditions; this work emphasizes clear guarantees in convex problems.
- **Inspiration**: Future work could jointly design Polyak-style schedulers with sharpness radius schedulers to further automate SAM-based methods.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Systematic generalization of Polyak step size to SAM/USAM is natural yet addresses a key pain point.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes theoretical verification and CIFAR experiments, but lacks larger models or non-vision tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Formulas, theorems, and experimental logic are very clear.
- Value: ⭐⭐⭐⭐☆ Highly practical for scenarios where one wants the benefits of SAM without extensive learning rate tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Stability Analysis of Sharpness-Aware Minimization](stability_analysis_of_sharpness-aware_minimization.md)
- [\[ICML 2026\] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization](safeguarded_stochastic_polyak_step_sizes_for_non-smooth_optimization_robust_perf.md)
- [\[ICLR 2026\] Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization](../../ICLR2026/optimization/minor_first_major_last_a_depth-induced_implicit_bias_of_sharpness-aware_minimiza.md)
- [\[ICML 2026\] LoRe: Adaptive Interaction-Evaluation Routing with Per-Step Interaction Budgets for Iterative Graph Solvers](lore_adaptive_interaction-evaluation_routing_with_per-step_interaction_budgets_f.md)
- [\[ICML 2026\] Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm](towards_understanding_continual_factual_knowledge_acquisition_of_language_models.md)

</div>

<!-- RELATED:END -->
