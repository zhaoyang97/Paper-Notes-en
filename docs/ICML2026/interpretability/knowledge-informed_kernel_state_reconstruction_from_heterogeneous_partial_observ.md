---
title: >-
  [Paper Note] MAAT: Knowledge-Guided Kernel Regression for Heterogeneous Partially Observed State Reconstruction
description: >-
  [ICML 2026][Interpretability][Kernel state reconstruction] MAAT reformulates the task of "recovering a physically consistent latent state trajectory from sparse, heterogeneous…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Kernel state reconstruction"
  - "RKHS"
  - "Heterogeneous observation operators"
  - "Symbolic regression"
  - "Physical priors"
date: 2026-05-08
content_hash: 469952d543d6683f
---

# MAAT: Knowledge-Guided Kernel Regression for Heterogeneous Partially Observed State Reconstruction

**Conference**: ICML 2026  
**arXiv**: [2601.22328](https://arxiv.org/abs/2601.22328)  
**Code**: Not disclosed  
**Area**: Scientific Computing / Dynamical Systems Modeling / Symbolic Regression  
**Keywords**: Kernel state reconstruction, RKHS, Heterogeneous observation operators, Symbolic regression, Physical priors

## TL;DR
MAAT reformulates the task of "recovering a physically consistent latent state trajectory from sparse, heterogeneous, and noisy observations" as a constrained kernel ridge regression problem in Reproducing Kernel Hilbert Space (RKHS). By integrating observation operators, smoothness, and physical priors (non-negativity, conservation, monotonicity) into a single objective function, it provides high-quality trajectories with analytical time derivatives for downstream symbolic regression (SINDy / PySR). It reduces reconstruction MSE by 1–3 orders of magnitude across 9 synthetic benchmarks and real-world COVID-19 data.

## Background & Motivation

**Background**: In fields such as medicine, ecology, and physics, latent dynamical states $x(t)\in\mathbb{R}^d$ are governed by ODEs $\dot{x}=f(x)$, but direct and regular observation of the full state is rare. Practice involves using classical smoothing (splines, RBF, Savitzky–Golay) for continuous trajectories, state-space models (Kalman, GP) for sensor fusion, or deep methods like Neural ODE / UDE to learn latent dynamics.

**Limitations of Prior Work**: Existing methods have significant flaws—classical smoothing ignores observation operators and domain constraints, handling only single-channel regular sampling; GPs provide analytical derivatives but struggle with hard constraints like "mass conservation" or "non-negativity"; Kalman filters require prior transition dynamics; Neural ODEs act as black boxes, making recovered trajectories difficult to feed directly into "mechanism discovery" pipelines (e.g., symbolic regression). The most fatal issue is derivative estimation: finite differences are sensitive to noise, with an irreducible error lower bound of $\Omega(\sigma^2/\Delta t^2)$, while symbolic regression is extremely sensitive to derivative accuracy.

**Key Challenge**: Measurements are **heterogeneous**—sparse direct measurements (e.g., gene expression snapshots) and high-frequency aggregate signals (e.g., blood biomarkers) map to the state space through different linear operators $\mathcal{H}_i$, accompanied by measurement noise. Processing these heterogeneous observations within a unified framework, injecting physical priors, and producing **analytically differentiable** trajectories is a critical bottleneck in the "measurement-to-mechanism discovery" pipeline.

**Goal**: To upgrade state reconstruction from a "numerical preprocessing step" to a "knowledge-guided inference problem in function space," ensuring reconstruction results naturally possess analytical derivatives and can encode mechanistic constraints such as conservation, non-negativity, and monotonicity.

**Key Insight**: The authors note that the RKHS framework simultaneously provides three advantages—the Representer Theorem guarantees the optimal solution is a finite linear combination of kernels; common kernels (like the Gaussian kernel) are $C^{\infty}$ with analytical derivatives; and constraints can be seamlessly injected as regularization terms. This is naturally suited for scientific scenarios characterized by "sparse data + heterogeneity + derivative requirements + prior knowledge."

**Core Idea**: Construct a composite loss in RKHS that integrates "snapshot fidelity + heterogeneous linear observation fidelity + dynamical priors + norm regularization." By solving for the coefficient matrix $U$ through closed-form kernel operations, a **physically consistent trajectory with analytical derivatives** is obtained, serving as a "clean interface" for symbolic regression.

## Method

### Overall Architecture

Input: $N$ observations $\mathcal{D}=\{(t_i, y_i, \mathcal{H}_i)\}$ collected at irregular timestamps $\{t_i\}_{i=1}^N$, where each observation relates to the latent state $x(t_i)$ via a linear operator $\mathcal{H}_i$ with Gaussian noise; optional set of physical constraints $\mathcal{C}$ (non-negativity, conservation, monotonicity); optional dynamical prior $F(\cdot)$.

Mechanism: Uses a Gaussian kernel $\kappa(t,t')$ to construct an RKHS $\mathcal{H}_K$. Each state component is parameterized as $\widehat{x}_j(t)=\sum_{\ell=1}^N u_{\ell j}\,\kappa(t,t_\ell)$, where the coefficient matrix $U\in\mathbb{R}^{N\times d}$ is the sole learnable quantity. The optimization goal is a scalar convex loss (weighted sum of data fidelity, smoothness, and physical priors), solved via closed-form or convex optimization to find $U$.

Output: Continuous trajectories $\widehat{x}(t)$ and their analytical time derivatives $\partial_t \widehat{x}(t)=U\,\partial_t\kappa(t,\boldsymbol{t})$, which can be directly fed into symbolic regression engines like SINDy or PySR for equation discovery.

### Key Designs

1.  **RKHS Composite Loss (Kernel State Reconstruction, KSR)**:
    - **Function**: Simultaneously fits sparse direct snapshots $\mathbf{X}^{\mathrm{obs}}$, heterogeneous linear aggregate signals $\mathbf{Y}$, and dynamical priors within one objective function, avoiding cascaded errors found in traditional "smooth-then-differentiate-then-discover" pipelines.
    - **Mechanism**: The learning of coefficient matrix $U$ is formulated as $\min_U \tfrac{w_s}{N_{\text{obs}}}\|\mathbf{K}^{\mathrm{obs}}U-\mathbf{X}^{\mathrm{obs}}\|_F^2 + \sum_i \tfrac{w_i}{N}\|\mathbf{K}U\mathbf{H}_i^\top-\mathbf{Y}\|_F^2 + \gamma\|\dot{\mathbf{K}}U-F(\mathbf{K}U)\|_F^2 + \lambda\|U\|_F^2$. The second term explicitly applies each heterogeneous operator $\mathbf{H}_i$ to the reconstruction to perform "observation matching," while the third term aligns the time derivative $\dot{\mathbf{K}}U$ with a dynamical prior $F$. Lemma 1 proves this composite loss is a calibrated surrogate for the true $L^2$ error.
    - **Design Motivation**: Heterogeneous observations, physical priors, and smoothness have historically been scattered across different frameworks (GP / State Space / PINN). This unifies them in an RKHS closed-form problem while retaining the critical "analytical derivative" property.

2.  **Analytical Derivative Estimation & Noise Robustness**:
    - **Function**: Directly computes the time derivative of the kernel function to obtain $\partial_t \widehat{x}(t)$, avoiding numerical differentiation of noisy trajectories.
    - **Mechanism**: Because the model is linear with respect to $U$, the differentiation operator only applies to the kernel: $\partial_t \widehat{x}(t)=U\,\partial_t\kappa(t,\boldsymbol{t})$. Proposition 1 provides a theoretical guarantee: while finite difference derivative error is $\mathcal{O}(\Delta t^4)+\Omega(\sigma^2/\Delta t^2)$ (having an **irreducible noise amplification lower bound**), KSR derivative error is $\mathcal{O}(\lambda)+\mathcal{O}(\sigma^2/n)$, representing a standard bias–variance trade-off that decreases with sample size without high-frequency noise amplification issues.
    - **Design Motivation**: Downstream symbolic regression (e.g., SINDy) is extremely sensitive to derivative accuracy. Analytical derivatives without a noise lower bound are why MAAT is fundamentally more suitable as a "mechanism discovery interface" than numerical differentiation methods.

3.  **Physical Priors as Additional Regularization** ($\mathcal{R}_{\text{phys}}(x,\mathcal{C})$):
    - **Function**: Injects domain knowledge (non-negativity $x_j(t)\ge 0$, mass conservation $\sum_j x_j(t)=\text{const}$, monotonicity $R'(t)\ge 0$) as differentiable penalty terms into the RKHS optimization.
    - **Mechanism**: For compartmental models like SEIR, $\mathcal{R}_{\text{phys}}$ is written as a squared penalty on constraint violations (e.g., $\sum_t (\max(0,-x_j(t)))^2$). The overall problem remains solvable via convex optimization. Constraints are checked on a sampling grid rather than everywhere.
    - **Design Motivation**: Traditional GP/Kalman cannot easily ingest hard constraints; deep methods are often unstable. RKHS representation allows constraints to appear naturally as quadratic penalties, preserving convexity while using domain semantics to prune physically impossible trajectories. Table 2 shows that adding priors consistently reduces MSE by 10–15% across various noise types.

### Loss & Training
Total Loss = Snapshot fidelity term $\tfrac{w_s}{N_{\text{obs}}}\|\mathbf{K}^{\mathrm{obs}}U-\mathbf{X}^{\mathrm{obs}}\|_F^2$ + Heterogeneous fidelity term $\sum_i \tfrac{w_i}{N}\|\mathbf{K}U\mathbf{H}_i^\top-\mathbf{Y}\|_F^2$ + Dynamical prior term $\gamma\|\dot{\mathbf{K}}U-F(\mathbf{K}U)\|_F^2$ + RKHS regularization $\lambda\|U\|_F^2$ + Physical prior term $\lambda_2\mathcal{R}_{\text{phys}}(x,\mathcal{C})$. When $F$ is linear and $\mathcal{R}_{\text{phys}}$ is quadratic, the problem has a closed-form solution; otherwise, first-order convex optimization is used. Gaussian kernels are selected, with hyperparameters $\lambda, \gamma, \lambda_2$ tuned via grid search.

## Key Experimental Results

### Main Results

Comparison of state reconstruction MSE across nine synthetic dynamical benchmarks for two symbolic regression backends (PySR / SINDy) (Selected from Table 1):

| Dataset | Backend | Prev. Best Baseline | Prev. SOTA MSE | MAAT MSE | Gain |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CRC | SINDy | Kalman | $1.1\times 10^{-2}$ | $\mathbf{1.5\times 10^{-3}}$ | ~7× |
| Neutralization | SINDy | Kalman | $2.5\times 10^{-3}$ | $\mathbf{4.3\times 10^{-4}}$ | ~6× |
| SEIR | SINDy | Kalman | $8.4\times 10^{-4}$ | $\mathbf{7.9\times 10^{-5}}$ | ~11× |
| SEIRH | SINDy | GP | $8.6\times 10^{-4}$ | $\mathbf{4.1\times 10^{-5}}$ | ~21× |
| TMDD | SINDy | GP | $8.7\times 10^{-2}$ | $\mathbf{4.8\times 10^{-3}}$ | ~18× |
| Tumor | SINDy | Kalman | $8.8\times 10^{-1}$ | $\mathbf{1.2\times 10^{-1}}$ | ~7× |
| TDI | SINDy | Kalman | $4.7\times 10^{1}$ | $\mathbf{1.8\times 10^{0}}$ | ~26× |
| Viral | SINDy | Kalman | $8.1\times 10^{-4}$ | $\mathbf{1.3\times 10^{-4}}$ | ~6× |

COVID-19 Real-world Data (Table 3, SINDy backend, mean ± 95% CI):

| Method | Test MSE | 95% CI |
| :--- | :--- | :--- |
| **MAAT** | $\mathbf{6.33\times 10^{-5}}$ | $\pm 1.07\times 10^{-5}$ |
| RBF | $9.64\times 10^{-4}$ | $\pm 6.51\times 10^{-4}$ |
| Savitzky–Golay | $9.73\times 10^{-4}$ | $\pm 6.47\times 10^{-4}$ |
| TVRegDiff | $9.73\times 10^{-4}$ | $\pm 6.47\times 10^{-4}$ |
| Linear | $9.80\times 10^{-4}$ | $\pm 6.53\times 10^{-4}$ |

MAAT reduces reconstruction error by another order of magnitude on real epidemic data.

### Ablation Study

Ablation of physical priors (Table 2, SEIR / SEIRH across 3 noise types):

| Configuration | SEIR (Gauss, PySR) | SEIRH (Gauss, PySR) | Description |
| :--- | :--- | :--- | :--- |
| Plain | $2.58\times 10^{-5}$ | $1.71\times 10^{-5}$ | Only KSR + heterogeneous obs, no conservation/non-negativity |
| + priors | $\mathbf{2.19\times 10^{-5}}$ | $\mathbf{1.48\times 10^{-5}}$ | Adds conservation + non-negativity + $R'\ge 0$, $S'\le 0$ |
| Plain (Student-t) | $7.69\times 10^{-5}$ | $4.12\times 10^{-5}$ | Heavy-tailed noise |
| + priors (Student-t) | $\mathbf{7.38\times 10^{-5}}$ | $\mathbf{3.68\times 10^{-5}}$ | Priors still yield 5–10% improvement under heavy tails |

### Key Findings

- **Key Modules**: Even without the dynamical prior $F$, KSR + heterogeneous observation operators already outperform all baselines by 1–2 orders of magnitude. Adding structural priors (conservation/non-negativity) provides a further 10–20% improvement, indicating that "heterogeneous observation modeling" is the primary driver of the performance leap, while physical priors refine robustness.
- **Failures of Deep Methods**: Neural ODEs consistently show MSEs 4–10 orders of magnitude higher than MAAT, and even explode to $10^{10}$ on some datasets (Conservation, Tumor), proving pure black-box deep methods are unusable in low-data, high-noise scientific scenarios.
- **Noise Robustness**: MAAT's MSE remains nearly constant under Student-t and correlated Gaussian noise, whereas classical smoothing methods (RBF, Cubic) show 5–10x MSE increases, consistent with Proposition 1 regarding the lack of a noise amplification lower bound for KSR.
- **Downstream Quality**: Trajectories reconstructed by MAAT lead to discovered equations that are much closer to the ground truth across all 9 datasets, proving that high-quality analytical derivatives are indeed the bottleneck for symbolic regression.

## Highlights & Insights

- **Redefining "Reconstruction" as "Inference in Function Space"**: Instead of treating state reconstruction as a mere preprocessing step, this paper elevates it to an RKHS inference problem that systematically incorporates observation operators, physical priors, and smoothness. This is a shift in research perspective, not just a new method.
- **Diagnostic Power of Proposition 1**: A simple bias–variance analysis reveals why finite differences are structurally unsuitable for symbolic regression—the $\Omega(\sigma^2/\Delta t^2)$ noise lower bound means denser sampling can actually worsen results. This insight is transferable to any task requiring derivative estimation from noisy sequences (signal processing, RL advantage estimation).
- **Unified Expression of Heterogeneous Linear Operators**: Using the product $\mathbf{K}U\mathbf{H}_i^\top$ to combine high-frequency aggregate signals and sparse snapshots achieves multimodal fusion in a few lines of formulas. This expression can be transferred to any multi-sensor time-series fusion task.
- **Double Benefits of Convexity + Analytical Derivatives**: In an era dominated by deep learning, this work demonstrates a case of "classical mathematical tools meeting modern problem formulations," where RKHS convex optimization outperforms black-box methods in scientific scenarios requiring interpretability and stability.

## Limitations & Future Work

- **Acknowledged Limitations**: The current framework only supports **linear** observation operators $\mathcal{H}_i$; non-linear sensing models require linearization. The dynamical prior $F$ is assumed known or partially known, which has limited utility for high-dimensional systems where the physics is entirely unknown.
- **Observed Potential Limitations**: Fixed kernels (Gaussian) require grid searching for bandwidth, which might be problematic for multi-scale systems with varying time scales. $O(N^2)$ complexity makes the method unsuitable for very long sequences ($N>10^4$) without approximations like Nyström methods. Physical priors require manual specification.
- **Future Directions**: Extending $\mathcal{H}_i$ to non-linear operators via kernel mapping; introducing learnable kernels (e.g., DKL) for scale adaptation; and partnering with LLMs to automatically extract candidate physical constraints from literature.

## Related Work & Insights

- **vs Gaussian Process**: GPs provide analytical derivatives and uncertainty but struggle with hard constraints and don't naturally support heterogeneous operators. MAAT generalizes the kernel representation to a composite loss, outperforming GP by 1–2 orders of magnitude on compartmental models.
- **vs Kalman Filter**: Kalman filters excel at sensor fusion but require prior transition dynamics. When transitions are unknown, they degrade to simple smoothing. MAAT makes dynamical priors optional and uses physical constraints as an alternative.
- **vs Neural ODE / UDE**: Deep methods are flexible but collapse in low-data regimes (with MSEs often exceeding $10^{10}$). MAAT is fully convex and closed-form solvable, showing superior stability at scientific data scales.
- **vs Physics-Informed Kernel Learning (PIKL)**: PIKL solves forward/hybrid problems (solving PDEs with known operators), whereas MAAT focuses on the inverse/reconstruction problem. They are complementary for "reconstruct then solve PDE" workflows.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Breaking the Simplification Bottleneck in Amortized Neural Symbolic Regression](breaking_the_simplification_bottleneck_in_amortized_neural_symbolic_regression.md)
- [\[AAAI 2026\] Partially Shared Concept Bottleneck Models](../../AAAI2026/interpretability/partially_shared_concept_bottleneck_models.md)
- [\[ICML 2026\] Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution](manifold-aligned_guided_integrated_gradients_for_reliable_feature_attribution.md)
- [\[ICML 2026\] Understanding LoRA as Knowledge Memory: An Empirical Analysis](understanding_lora_as_knowledge_memory_an_empirical_analysis.md)
- [\[NeurIPS 2025\] Towards Scaling Laws for Symbolic Regression](../../NeurIPS2025/interpretability/towards_scaling_laws_for_symbolic_regression.md)

</div>

<!-- RELATED:END -->
