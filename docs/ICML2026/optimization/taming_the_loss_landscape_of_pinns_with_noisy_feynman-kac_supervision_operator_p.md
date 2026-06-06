---
title: >-
  [Paper Note] Taming the Loss Landscape of PINNs with Noisy Feynman-Kac Supervision: Operator Preconditioning and Non-Asymptotic Error Bounds
description: >-
  [ICML 2026][Optimization][PINN] Incorporating a small number of interior pseudo-labels, obtained via Monte Carlo simulation of the Feynman–Kac formula…
tags:
  - "ICML 2026"
  - "Optimization"
  - "PINN"
  - "Feynman-Kac"
  - "operator preconditioning"
  - "condition number"
  - "loss landscape"
date: 2026-05-08
content_hash: 2f44689b2beaa51c
---

# Taming the Loss Landscape of PINNs with Noisy Feynman-Kac Supervision: Operator Preconditioning and Non-Asymptotic Error Bounds

**Conference**: ICML 2026  
**arXiv**: [2606.00643](https://arxiv.org/abs/2606.00643)  
**Code**: None  
**Area**: Optimization Theory / Physics-Informed Neural Networks / Loss Landscape  
**Keywords**: PINN, Feynman-Kac, operator preconditioning, condition number, loss landscape  

## TL;DR
Incorporating a small number of interior pseudo-labels, obtained via Monte Carlo simulation of the Feynman–Kac formula, into the PINN loss essentially acts as preconditioning for the PDE operator. This paper provides an operator-level proof that the "condition number remains bounded" with respect to the number of collocation points $N$, along with non-asymptotic $L^2$ error bounds for networks with $\tanh$ activations. This approach successfully solves problems like Schrödinger, Poisson, and committor equations where standard PINNs typically fail.

## Background & Motivation

**Background**: PINNs approximate PDE solutions $u_\theta$ by penalizing residuals $\mathcal{L}u - f$ and boundary violations. As a mesh-free solver paradigm, they are widely used in forward/inverse problems, parameter estimation, and multi-physics coupling scenarios.

**Limitations of Prior Work**: On moderately stiff or high-frequency problems, PINNs often exhibit extremely slow training or fail to converge entirely. Results vary significantly with different hyperparameters or sampling strategies. Existing mitigations—such as adaptive sampling, curriculum training, residual/gradient weighting, domain decomposition, and specialized architectures—only work on specific problems and lack a general, interpretable solution.

**Key Challenge**: Recent perspectives on operator condition numbers (De Ryck et al. 2024, Rathore et al. 2024, etc.) point out that PINN training difficulties stem from the severely ill-conditioned spectrum of the Hermite square $\mathcal{L}^*\mathcal{L}$ of the PDE operator. This is an inherent property of the problem itself rather than insufficient network capacity; therefore, larger networks or denser collocation points cannot rectify it. Most existing remedies focus on modifying optimizers (natural gradient, second-order methods, implicit preconditioning) rather than the training objective itself.

**Goal**: (1) Propose a preconditioning scheme that **modifies the training objective** instead of the optimizer by adding a data fidelity term from any source (FEM, experiments, or MC simulation); (2) Rigorously prove that this term suppresses the PL$^*$ condition number explosion from polynomial growth in $N$ back to **uniform boundedness**; (3) Utilize the Feynman–Kac formula to provide a mesh-free, compatible label generation scheme; (4) Derive end-to-end non-asymptotic $L^2$ error bounds for networks with $\tanh$ activations.

**Key Insight**: The authors noted that "mass terms" in the operator spectrum are typically ignored. Adding $\sum_k (u_\theta(x_k) - \hat u(x_k))^2$ is equivalent to adding a positive definite term $\lambda_{\mathrm{FK}}M$ to the curvature matrix, which acts as a "mass matrix" preconditioner in classical numerical analysis, compensating for small eigenvalue directions. The Feynman–Kac formula provides an almost cost-free, parallelizable method for generating these pseudo-labels.

**Core Idea**: The probabilistic FK representation of a PDE solution, $u^\star(x) = \mathbb{E}_x[\int_0^\tau r(X_t)\mathrm{d}t + h(X_\tau)]$, is estimated using Euler–Maruyama simulation of several trajectories to generate a small set of useful "interior pseudo-labels." Integrating these as an auxiliary loss term into a standard PINN improves the condition number and stability without altering the architecture or optimizer.

## Method

FK-PINN consists of "offline FK label generation" and "online PINN training with data augmentation." The theoretical analysis proves condition number boundedness under the PL$^*$ framework and provides non-asymptotic error bounds by incorporating MC noise into the approximation-estimation-optimization decomposition.

### Overall Architecture

Inputs consist of the domain $\Omega\subset\mathbb{R}^d$, boundary $\partial\Omega$, a second-order linear elliptic/parabolic operator $\mathcal{L}$, source term $f$, Dirichlet data $g$, and a $\tanh$ network $u_\theta$. The pipeline is as follows:

1.  **Offline**: Select $N_{\mathrm{FK}}$ interior supervision points $\{x_k^{\mathrm{FK}}\}$. For each point, run $N_{\mathrm{MC}}$ diffusion trajectories using Euler–Maruyama with step $\Delta t$ and maximum time $T_{\max}$ to estimate $\hat u^{\mathrm{MC}}(x_k^{\mathrm{FK}})$ using the cumulative formula in Algorithm 1. This step is **fully offline, parallelizable**, and independent of the network.
2.  **Online**: At each step, sample new interior/boundary collocation points and compute the total loss (Eq. 4.2) along with the FK dataset. Update $\theta$ and task weights $\phi$ using Adam/SGD. The loss reverts to a standard PINN when $\lambda_{\mathrm{FK}} = 0$.

### Key Designs

1.  **Feynman–Kac Supervision as Operator Preconditioning**:
    *   **Function**: Uses a small number of interior pseudo-labels to modify the PINN curvature matrix $H$ into $H_{\mathrm{FK}} = H + \lambda_{\mathrm{FK}}M$, where $M$ is the semi-definite matrix contributed by the data fidelity term.
    *   **Mechanism**: Incorporates $\mathcal{R}_{\mathrm{FK}}(\theta) = \frac{1}{N_{\mathrm{FK}}}\sum_k(u_\theta(x_k^{\mathrm{FK}}) - \hat u^{\mathrm{MC}}_\Gamma(x_k^{\mathrm{FK}}))^2$ into the loss, resulting in $\mathcal{R}_{\mathrm{FK\text{-}PINN}} = \lambda_{\mathrm{PDE}}\mathcal{R}_{\mathrm{PDE}} + \lambda_{\partial\Omega}\mathcal{R}_{\partial\Omega} + \lambda_{\mathrm{FK}}\mathcal{R}_{\mathrm{FK}}$. This effectively adds a "mass term" to the $L^2$ inner product, lifting the small eigenvalues of the ill-conditioned $\mathcal{L}^*\mathcal{L}$ spectrum.
    *   **Design Motivation**: While traditional preconditioning requires changing optimizers, modifying the training objective is fully compatible with existing PINN pipelines. The preconditioning effect is theoretically guaranteed regardless of the data source; FK is simply one implementation, while coarse FEM solutions or experimental data are also applicable.

2.  **PL$^*$-based Condition Number Bound (Theorem 5.4)**:
    *   **Function**: Formalizes the improvement in the condition number into a rigorous non-asymptotic inequality.
    *   **Mechanism**: Within the PL$^*$ framework (Liu et al. 2022, Rathore et al. 2024), $\kappa_{\mathrm{PL}}(\mathcal{J}) = L(\mathcal{J})/\mu(\mathcal{J})$ is defined as the ratio of the smoothness constant to the PL$^*$ constant. Previous work proved $\kappa_{\mathrm{PINN}} \geq cN^{\beta/2}$ (exploding with collocation points). By introducing a "compatibility condition" (Assumption 5.2), this paper proves that $\mathcal{R}_{\mathrm{FK\text{-}PINN}}$ is $L_0$-smooth and satisfies a PL$^*$ condition with constant $\mu_0$ independent of $N$, such that $\kappa_{\mathrm{FK}} \leq C$. This reduces the gradient descent complexity from $\mathcal{O}(N^{\beta/2}\log(1/\varepsilon))$ to $\mathcal{O}(\log(1/\varepsilon))$.
    *   **Design Motivation**: Directly proving the Hessian spectrum for deep networks is intractable. PL$^*$ is a weaker condition that holds for non-isolated minima and, combined with interpolation hypotheses, captures the reality of over-parameterized PINNs while providing a clean language for preconditioning gains.

3.  **Non-Asymptotic $L^2$ Error Bound with $\tanh$ (Theorem 6.2)**:
    *   **Function**: Translates operator-level improvements into the final error between the trained solution and the true solution.
    *   **Mechanism**: MC labels are decomposed into $Y_i^{\mathrm{FK}} = u^\star(X_i^{\mathrm{FK}}) + b(X_i^{\mathrm{FK}}) + \zeta_i$, where bias $b(x)$ is controlled by the Euler–Maruyama step size and truncation time. A key technical contribution is providing pseudodimension bounds for the first and second derivatives of $\tanh$ networks. For $u^\star\in W^{s,\infty}$, width $m\asymp N_{\mathrm{FK}}^{d/(4d+8(s-2))}$, and depth $L=3$, the $L^2$ error after $T \gtrsim \log N_{\mathrm{FK}}$ steps is bounded by $C(N_{\mathrm{FK}}^{-\beta} + e^{-c_{\mathrm{opt}}T} + \varepsilon_{\mathrm{bias}} + \varepsilon_{\mathrm{bias}}^{1/2})$.
    *   **Design Motivation**: Most PINN non-asymptotic bounds only hold for piecewise polynomial activations. Since $\tanh$ is the practical standard, this section fills a critical theoretical gap and provides a guide for balancing MC budgets against total error.

### Loss & Training

The total loss employs uncertainty weighting (Kendall et al. 2018), where the log-variance $s_j = \log\sigma_j^2$ of each task is a learnable weight $\lambda_j(\phi) = \exp(-s_j)$. $s_j$ is clipped to a bounded interval to prevent noisy FK terms from being ignored. The optimizer used is a combination of Adam and L-BFGS.

## Key Experimental Results

### Main Results

| Problem | Metric | Standard PINN | FK-PINN | Gain |
| :--- | :--- | :--- | :--- | :--- |
| Poisson | Relative $L^2$ Error | $0.322\pm 0.149$ | $0.118\pm 0.003$ | Error reduced to 1/3; variance reduced by 1 order |
| Schrödinger-type | Relative $L^2$ Error | $0.624\pm 0.195$ | $0.096\pm 0.010$ | PINN fails; FK-PINN recovers wave function structure |
| Mean Exit Time | Relative $L^2$ Error | $1.007\pm 0.003$ | $0.107\pm 0.006$ | PINN is meaningless; FK-PINN error reduced by 1 order |
| Committor | Relative $L^2$ Error | $0.839\pm 0.661$ | $0.030\pm 0.008$ | Gain > 27x; variance dropped from $0.66$ to $0.008$ |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Schrödinger Visualization | Adam + L-BFGS | Standard PINN predictions are distorted; FK-PINN optimizes the landscape. |
| 5 Random Seeds | Variance Analysis | FK-PINN variance is 1–2 orders smaller, confirming boundedness of $\kappa$. |
| FK Budget $N_{\mathrm{FK}}$ vs $N_{\mathrm{MC}}$ | Bias/Variance Trade-off | Budget scaling matches the $m\asymp N_{\mathrm{FK}}$ theoretical prescription. |
| $\lambda_{\mathrm{FK}} = 0$ | Baseline Performance | Performance immediately reverts, confirming gains come from the FK term. |

### Key Findings

*   On all four test problems, FK-PINN significantly outperforms standard PINNs. Specifically, on the **Schrödinger, Mean Exit Time, and Committor problems where PINNs fail**, FK-PINN changes the outcome from "unsolvable" to "solvable" with $10^{-1}$ level accuracy.
*   Standard PINN variance is often as large as the mean, implying near-random results, while FK-PINN variance is 1–2 orders smaller, validating the theoretical prediction of bounded condition numbers.
*   The benefits of operator preconditioning are independent of the data source. FK is simply a mesh-free implementation, implying that even sparse FEM or experimental data can be used with this framework.

## Highlights & Insights

*   The reinterpretation of the data fidelity term as an operator preconditioner is the core narrative—it bridges classical mass-matrix preconditioning with modern PINN operator spectrum analysis.
*   The derivation of pseudodimension bounds for the derivatives of $\tanh$ networks is a valuable independent contribution for any future work on PINN learning theory involving second-order residuals.
*   The experiments are designed to challenge PINNs on multi-scale and long-time diffusion problems. If the FK term truly preconditions the operator, it should make these intractable problems solvable rather than simply slightly improving Poisson errors.
*   The flexibility of the preconditioning effect allows for mixing FK MC, sparse FEM, and real measurements, provided the compatibility assumption is satisfied.

## Limitations & Future Work

*   The theory currently assumes linear second-order elliptic/parabolic operators where FK representations are available; non-linear PDEs require extensions such as branching diffusions.
*   Assumption 5.2 (Compatibility) is natural in theory but lacks an operational tool for verification on a per-case basis.
*   Errors are bounded by the MC bias $\varepsilon_{\mathrm{bias}}$, necessitating a sufficiently small $\Delta t$ to realize the full theoretical benefits.
*   In the "truly sparse" regime where $N_{\mathrm{FK}}$ is fixed but $N_{\mathrm{int}}$ increases, the error is saturated by the FK term—an inherent statistical cost of the preconditioning approach.

## Related Work & Insights

*   **vs De Ryck et al. 2024**: This paper inherits the operator spectrum analysis framework but moves the improvement from the "optimizer" to the "training objective," which is lower in implementation cost.
*   **vs Rathore et al. 2024**: Utilizes the PL$^*$ and kernel integral operator toolchain to prove the dual conclusion: while standard PINN $\kappa$ explodes, adding a data term keeps $\kappa$ bounded.
*   **vs Han et al. 2020**: Unlike methods that use FK as the primary NN PDE solver (requiring MC at every point), this method uses FK only for sparse supervision, maintaining the continuous solution advantages associated with PINNs.
*   **vs Weighting Schemes (Wu et al. 2023)**: These methods fix issues within the existing operator spectrum; this work modifies the spectrum directly with a mass term to solve problems the former cannot.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Non-Asymptotic Analysis of Efficiency in Conformalized Regression](../../ICLR2026/optimization/non-asymptotic_analysis_of_efficiency_in_conformalized_regression.md)
- [\[ICML 2026\] Sharp Description of Local Minima in the Loss Landscape of High-Dimensional Two-Layer ReLU Networks](sharp_description_of_local_minima_in_the_loss_landscape_of_high-dimensional_two-.md)
- [\[ICML 2026\] Interpretability and Generalization Bounds for Learning Spatial Physics](interpretability_and_generalization_bounds_for_learning_spatial_physics.md)
- [\[ICLR 2026\] Rolling Ball Optimizer: Learning by Ironing Out Loss Landscape Wrinkles](../../ICLR2026/optimization/rolling_ball_optimizer_learning_by_ironing_out_loss_landscape_wrinkles.md)
- [\[ICML 2026\] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](rmnp_row-momentum_normalized_preconditioning_for_scalable_matrix-based_optimizat.md)

</div>

<!-- RELATED:END -->
