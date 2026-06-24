---
title: >-
  [Paper Note] Derandomized Online-to-Non-convex Conversion for Stochastic Weakly Convex Optimization
description: >-
  [ICLR2026][Optimization][Weakly convex optimization] This paper demonstrates that in stochastic weakly convex optimization, the randomized interpolation or scaling required by O2NC can be removed. By taking stochastic subgradients directly at the current iterate and using online incremental learning with quadratic regularization, the optimal Goldstein stationarity complexity is achieved. This further derives an SGDM variant that is nearly equivalent to periodic momentum resta…
tags:
  - "ICLR2026"
  - "Optimization"
  - "Weakly convex optimization"
  - "Goldstein stationarity"
  - "O2NC"
  - "SGDM"
  - "Momentum restart"
date: 2026-05-08
content_hash: 09125aa0591d5816
---

# Derandomized Online-to-Non-convex Conversion for Stochastic Weakly Convex Optimization

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=Snfqe4lU3G](https://openreview.net/forum?id=Snfqe4lU3G)  
**Code**: None  
**Area**: Stochastic Weakly Convex Optimization / Non-smooth Non-convex Optimization  
**Keywords**: Weakly convex optimization, Goldstein stationarity, O2NC, SGDM, Momentum restart  

## TL;DR
This paper demonstrates that in stochastic weakly convex optimization, the randomized interpolation or scaling required by O2NC can be removed. By taking stochastic subgradients directly at the current iterate and using online incremental learning with quadratic regularization, the optimal Goldstein stationarity complexity is achieved. This further derives an SGDM variant that is nearly equivalent to periodic momentum restart.

## Background & Motivation
**Background**: In non-smooth non-convex optimization, standard "gradient norm approaching zero" is difficult to use as a computable objective because the objective function may lack classical gradients. Moreover, in general Lipschitz non-convex cases, finding approximate stationary points faces unavoidable complexity barriers. A commonly used alternative convergence concept is Goldstein $(\delta, \epsilon)$-stationarity: it does not require a single subgradient at the current point to be small, but checks whether the distance from zero to the convex hull of Clarke subgradients within a radius $\delta$ of the current point is no more than $\epsilon$.

**Limitations of Prior Work**: The online-to-non-convex conversion (O2NC) framework by Cutkosky et al. provides a powerful theoretical framework: by embedding an online convex optimization algorithm into stochastic subgradient iterations, the optimal oracle complexity of $O(\delta^{-1}\epsilon^{-3})$ for finding Goldstein stationary points is achieved. However, original O2NC requires randomized interpolation between adjacent iterates to prove descent, such as taking gradients at $v_n=w_{n-1}+s_n\Delta_n$ where $s_n\sim\mathrm{Uniform}([0,1])$. The subsequent E-O2NC introduced randomized scaling. These randomizations allow theoretical algorithms to explain optimizers like SGDM/Adam but create a gap with the actual SGDM used in deep learning training.

**Key Challenge**: If randomization is completely removed, one encounters the impossibility results of Jordan et al. for general non-smooth non-convex functions: deterministic algorithms cannot achieve dimension-free rates. Thus, "updating deterministically like actual SGDM" and "maintaining the dimension-free optimal guarantee of O2NC" cannot both hold in the most general case.

**Goal**: The authors restrict the problem to a broad but structured class of functions: stochastic weakly convex optimization. The risk function $R$ is $\rho$-weakly convex, meaning $R(\cdot)+\frac{\rho}{2}\|\cdot\|^2$ is convex. The paper seeks to answer three questions: first, whether weak convexity can replace randomized interpolation to yield descent inequalities; second, whether derandomized O2NC can still achieve $O(\delta^{-1}\epsilon^{-3})$ complexity; and third, whether this theoretical algorithm can closer resemble actual SGDM and show visible effects in neural network training.

**Key Insight**: The most critical role of weak convexity is providing a "one-step change control" via a quadratic upper bound. For any $\gamma\ge\rho$, if $g_n\in\partial R(w_n)$ and $\Delta_n=w_n-w_{n-1}$, weak convexity yields:
$$
R(w_n)-R(w_{n-1})\le \langle g_n,\Delta_n\rangle+\frac{\gamma}{2}\|\Delta_n\|^2.
$$
This transforms the selection of each parameter update $\Delta_n$ into a problem of online learning for a sequence of quadratic losses $\langle \hat g_n,\Delta\rangle+\frac{\gamma}{2}\|\Delta\|^2$.

**Core Idea**: Replace the randomized interpolation in O2NC with the quadratic regularization term from weak convexity. This allows the online learner to directly learn the "next increment," recovering optimal Goldstein stationarity complexity without introducing auxiliary randomization, and naturally resulting in clipped SGDM or momentum-restarted SGDM.

## Method

### Overall Architecture
The proposed method is called Derandomized O2NC (D-O2NC). Inputs include an initial point $w_0$, a total iteration budget $N=KT$, a quadratic regularization coefficient $\gamma$ related to weak convexity, an online learning step size $\eta$, and an optional increment radius $D$. In each round, parameters are updated using the current increment $w_n=w_{n-1}+\Delta_n$. A data sample $z_n$ is then drawn at the actual current point $w_n$ to compute a stochastic subgradient $\hat g_n\in\partial \ell(w_n;z_n)$. Finally, $\hat g_n$ is fed to the online learner to update the next increment $\Delta_{n+1}$.

The primary difference from original O2NC is the absence of the randomized intermediate point $v_n=w_{n-1}+s_n\Delta_n$ and exponential randomized scaling. The theoretical analysis no longer relies on randomized interpolation to link function differences with path integrals, but on weak convexity to bound one-step function changes via quadratic online loss. The output follows the O2NC block-averaging idea: $N$ iterations are divided into $K$ blocks of length $T$. Each block outputs an average point $\bar w^{(k)}=\frac{1}{T}\sum_{t=1}^T w_t^{(k)}$, and one is chosen uniformly at random.

The workflow consists of four steps: maintain increment $\Delta_n$; update actual parameter $w_n$; query stochastic subgradient at $w_n$; treat the subgradient as part of an online quadratic loss; and use an OGD-type online algorithm to determine the next increment. Two versions of the online learner are provided: Option-I uses clipped OGD with ball constraints, directly providing Goldstein stationarity; Option-II uses unconstrained OGD with restarts every $T$ steps, which is more similar to actual SGDM and is analyzed using a new regularized stationarity metric.

### Key Designs
**1. Weak Convex Quadratization: Replacing Randomized Interpolation with $\gamma\|\Delta\|^2/2$**

Original O2NC samples points randomly between $w_{n-1}$ and $w_n$ to ensure the stochastic subgradient represents the average change along the path. This paper bypasses this step using weak convexity. Since $R$ is $\rho$-weakly convex, taking $\gamma\ge\rho$ ensures that at the current point $w_n$:
$$
R(w_n)-R(w_{n-1})\le \langle g_n,\Delta_n\rangle+\frac{\gamma}{2}\|\Delta_n\|^2.
$$
This translates "optimizing the objective descent" into "selecting increments online to minimize the cumulative linear plus quadratic loss." Since the stochastic subgradient oracle satisfies $\mathbb E[\hat g_n\mid\mathcal F_{n-1}]=g_n\in\partial R(w_n)$, the $\hat g_n$ seen by the online learner serves as a stochastic estimate of this quadratic loss.

**2. Clipped OGD: Conservative Increments for Direct Goldstein Guarantees**

The first version updates increments using projected OGD:
$$
\Delta_{n+1}=\mathrm{clip}_D\big((1-\eta\gamma)\Delta_n-\eta\hat g_n\big).
$$
Here $D$ is the increment radius, ensuring each update is small. Defining $m_n=-\gamma\Delta_n$ and $\beta=\eta\gamma$, the update is rewritten as:
$$
w_n=w_{n-1}-\gamma^{-1}m_n, \quad m_{n+1}=\mathrm{clip}_D\big((1-\beta)m_n+\beta\hat g_n\big).
$$
This is effectively clipped SGDM, where $m_n$ is the momentum search direction, $\gamma^{-1}$ is the learning rate, and $\beta$ is the momentum coefficient.

**3. Periodic Restart OGD: Bringing Theory Closer to Standard SGDM**

Option-II removes the projection, using:
$$
\Delta_{n+1}=(1-\eta\gamma)\Delta_n-\eta\hat g_n,
$$
but resets the increment to $0$ every $T$ steps. For momentum variable $m_n=-\gamma\Delta_n$, the update is:
$$
w_n=w_{n-1}-\gamma^{-1}m_n, \quad m_{n+1}=\big((1-\beta)m_n+\beta\hat g_n\big)\mathbf 1\{\mathrm{mod}(n+1,T)\ne 1\}.
$$
This is almost standard SGDM, except for periodic momentum clearing. To analyze this without the $D$ constraint, the authors introduce $(\mu, \epsilon)$-regularized stationarity:
$$
\|\partial f(w)\|_{+\mu}=\inf_{V\subseteq\mathbb R^d}\left\{\mathrm{dist}(0, \partial_V f)+\mu\sup_{v\in V}\|v-w\|^2\right\}.
$$

**4. Block Average Output: Transforming Online Regret into Non-smooth Stationary Certificates**

The output is the block average point rather than the final iterate. For the $k$-th block, let $\bar g^{(k)}=\frac{1}{T}\sum_{t=1}^T g_t^{(k)}$. The online learner ensures low regret for the quadratic loss; the weak convexity inequality links cumulative regret to $R(w_T^{(k)})-R(w_0^{(k)})$. Summing across blocks makes the function values telescope, leaving only the initial gap $\Delta R_0$.

### Loss & Training
The "loss" in the theoretical part is not the training loss of the neural network but the internal quadratic loss for the online learner:
$$
f_n(\Delta)=\langle \hat g_n,\Delta\rangle+\frac{\gamma}{2}\|\Delta\|^2.
$$
Option-I performs OGD on $f_n$ over a ball of radius $D$. Corollary 1 provides parameters based on budget $N$ and accuracy $(\delta,\epsilon)$:
$$
T=\lceil(\delta N)^{2/3}\rceil, \quad K=\left\lfloor\frac{N}{T}\right\rfloor, \quad \gamma=N^{1/3}\delta^{-2/3}, \quad \eta=\frac{1}{8N}, \quad D=\delta^{1/3}N^{-2/3}.
$$
When $N\ge O(\delta^{-1}\epsilon^{-3})+\rho^3\delta^2+\delta^{-1}$, it outputs a point satisfying $\mathbb E[\mathrm{dist}(0,\partial_\delta R(\bar w_T))]\le\epsilon$.

Option-II uses the same quadratic loss with periodic restarts. Corollary 2 selects:
$$
T=\lceil N^{4/7}\mu^{-2/7}\rceil, \quad K=\left\lfloor\frac{N}{T}\right\rfloor, \quad \gamma=N^{3/7}\mu^{2/7}, \quad \eta=\frac{1}{8N},
$$
achieving a complexity of $O(\mu^{1/2}\epsilon^{-7/2}+\rho^{7/3}\mu^{-2/3}+\mu^{1/2})$ for $(\mu,\epsilon)$-regularized stationarity.

## Key Experimental Results

### Main Results
The authors train CIFAR-10 image classification models using the periodic restart version of D-O2NC, comparing it against standard SGDM with identical learning rates, momentum, and weight decay.

| Backbone | Metric | SGDM | D-O2NC $(T=20)$ | D-O2NC $(T=50)$ | Main Observation |
|--------|------|------|----------------|----------------|----------|
| ResNet-101 | Train Loss $(\times 10^{-3})$ | $2.55\pm0.08$ | $0.55\pm0.02$ | $1.38\pm0.01$ | Restart version converges to lower training loss |
| ResNet-101 | Test Accuracy | $93.91\pm0.55$ | $94.64\pm0.07$ | $94.81\pm0.34$ | ~0.90% improvement |
| ViT | Train Accuracy | $90.16\pm0.51$ | $92.38\pm0.28$ | $91.36\pm0.22$ | Higher training accuracy on ViT |
| ViT | Test Accuracy | $84.26\pm0.41$ | $84.92\pm0.17$ | $86.22\pm0.29$ | ~1.96% improvement |

### Ablation Study
The ablation focuses on the sensitivity to restart frequency $T$, learning rate, and momentum. If $T$ is too small, momentum is cleared too frequently to accumulate useful directions; if $T$ is too large, it approaches standard SGDM. Performance typically improves then declines as $T$ increases.

| Configuration | Key Metric | Description |
|------|---------|------|
| $T=2$ | Poor performance | Restart too frequent, momentum cannot form |
| $T=20$ | ResNet-101 Test Acc $94.64\pm0.07$ | Lowest training loss, fast convergence |
| $T=50$ | ViT Test Acc $86.22\pm0.29$ | Best overall performance in main experiments |
| $T=196$ | Performance drops | Approaches regular SGDM as interval reaches one epoch |

### Key Findings
- The theoretical result shows clipped D-O2NC achieves $O(\delta^{-1}\epsilon^{-3})$ dominant complexity in stochastic weakly convex optimization. The weak convexity parameter $\rho$ appears only in non-dominant terms.
- Option-II (periodic restart) is practically nearly identical to SGDM, adding only a periodic momentum clearing step.
- On CIFAR-10, D-O2NC improves test accuracy for both ResNet-101 and ViT, with more significant gains for ViT.
- Restart frequency is the primary hyperparameter. Values between $20$ and $50$ are robust in the experiments.

## Highlights & Insights
- The ingenious theoretical move replaces randomized interpolation with a weak convexity quadratic upper bound.
- The explanation for SGDM is closer to actual training code than prior O2NC versions.
- $(\mu,\epsilon)$-regularized stationarity acts as a bridge, allowing unconstrained restart algorithms to achieve Goldstein stationarity guarantees.
- Momentum clearing is a computationally cheap trick that provides faster convergence and better test accuracy in experiments.

## Limitations & Future Work
- Theoretical applicability depends on weak convexity. Deterministic dimension-free impossibility still holds for general Lipschitz non-smooth functions.
- The clipped OGD version requires explicit radius constraints and parameter settings based on target accuracy $(\delta, \epsilon)$ and budget $N$.
- The experimental scale is relatively preliminary (CIFAR-10/100).
- Comparison is mainly against SGDM, lacking systematic comparison with AdamW, schedule-free SGD, or Lion.

## Related Work & Insights
- **vs Original O2NC**: Original O2NC uses randomized interpolation for $O(\delta^{-1}\epsilon^{-3})$ complexity; Ours removes this in weakly convex functions while maintaining dominant complexity and resembling actual SGDM.
- **vs E-O2NC**: E-O2NC uses exponential randomized scaling; Ours uses periodic restarts without randomized scaling or EMA outputs.
- **vs Moreau envelope analysis**: Standard weakly convex analysis often gives $O(\epsilon^{-4})$ for Moreau envelope stationarity; Ours focuses on Goldstein stationarity with $O(\delta^{-1}\epsilon^{-3})$ complexity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐☆ 
- Writing Quality: ⭐⭐⭐⭐☆ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] High Probability Bounds for Non-Convex Stochastic Optimization with Momentum](high_probability_bounds_for_non-convex_stochastic_optimization_with_momentum.md)
- [\[ICLR 2026\] Fast Frank–Wolfe Algorithms with Adaptive Bregman Step-Size for Weakly Convex Functions](fast_frankwolfe_algorithms_with_adaptive_bregman_step-size_for_weakly_convex_fun.md)
- [\[ICLR 2026\] Improving Online-to-Nonconvex Conversion for Smooth Optimization via Double Optimism](improving_online-to-nonconvex_conversion_for_smooth_optimization_via_double_opti.md)
- [\[ICLR 2026\] Non-Convex Federated Optimization under Cost-Aware Client Selection](non-convex_federated_optimization_under_cost-aware_client_selection.md)
- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](../../NeurIPS2025/optimization/stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)

</div>

<!-- RELATED:END -->
