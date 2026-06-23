---
title: >-
  [Paper Note] On the Convergence Behavior of Preconditioned Gradient Descent Toward the Rich Learning Regime
description: >-
  [ICLR 2026][Optimization & Theory][Grokking] Starting from the eigenvalue dynamics of the Neural Tangent Kernel (NTK), this paper demonstrates theoretically and experimentally that preconditioned gradient descent (PGD), such as Gauss-Newton/Levenberg-Marquardt, can flatten the "disparity in convergence rates across frequency modes" caused by spectral bias into un
tags:
  - ICLR 2026
  - Optimization & Theory
  - Grokking
  - Gauss-Newton
date: 2026-05-08
content_hash: a8bf4492364dcda9
---
# On the Convergence Behavior of Preconditioned Gradient Descent Toward the Rich Learning Regime

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=CXlsqTAf1E](https://openreview.net/forum?id=CXlsqTAf1E)  
**Code**: To be open-sourced (Authors promise release at https://github.com/sandialabs)  
**Area**: Optimization / Learning Theory  
**Keywords**: Preconditioned Gradient Descent, Spectral Bias, Grokking, Neural Tangent Kernel, Gauss-Newton

## TL;DR
Starting from the eigenvalue dynamics of the Neural Tangent Kernel (NTK), this paper demonstrates theoretically and experimentally that preconditioned gradient descent (PGD), such as Gauss-Newton/Levenberg-Marquardt, can flatten the "disparity in convergence rates across frequency modes" caused by spectral bias into uniform convergence. This significantly compresses the delayed generalization window of grokking. However, PGD tends to remain trapped in the lazy NTK subspace, leading to weaker final generalization; it is necessary to switch back to first-order methods (e.g., Adam) after the lazy phase is exhausted to restore generalization.

## Background & Motivation

**Background**: Neural networks exhibit "spectral bias" (F-Principle), where low-frequency components are learned before high-frequency ones during training. The mainstream explanation stems from the NTK perspective: the evolution of error in function space is determined by the eigenvalue spectrum of the NTK matrix $K_t = J_t J_t^T$. Modes corresponding to large eigenvalues converge quickly, while those with small eigenvalues converge slowly. In tasks like image classification, spectral bias acts as an implicit regularizer, suppressing high-frequency noise and aiding generalization.

**Limitations of Prior Work**: In scientific computing (e.g., solving PDEs with PINNs or fitting functions with high-frequency structures), the need for fast approximation of high-frequency solutions makes spectral bias a bottleneck. Simultaneously, grokking (delayed generalization, where a model memorizes the training set and suddenly generalizes to the test set much later) is another phenomenon that slows training. Previous works (Wadia 2021, Buffelli 2024) found that although high-order or curvature methods converge faster, they often generalize worse, leading to long-standing controversy over whether high-order optimizers should be used.

**Key Challenge**: The root of spectral bias is the extremely poor condition number of the NTK—only $O(1)$ eigenvalues are "large," while the majority of modes converge at a snail's pace of $1 - \lambda_k/\lambda_N \ll 1$. Meanwhile, grokking is hypothesized by Kumar (2024) and Zhou (2024) to be a product of the transition from the "lazy regime" dominated by a stationary NTK to the "rich feature learning regime." These two issues share a common origin: the model stays in the lazy subspace for a long duration, restricted by spectral bias to learning only low frequencies.

**Goal**: (1) Prove that PGD can fundamentally improve the condition number of the NTK and eliminate spectral bias; (2) Use PGD to "uniformly" and rapidly explore the lazy regime to verify that grokking is indeed a transition from lazy to rich regimes; (3) Explain why high-order methods generalize poorly and provide a remedy.

**Key Insight**: Treating optimization as a preconditioning problem in numerical linear algebra—since spectral bias is equivalent to "anisotropic convergence under ill-conditioned numbers," classical preconditioners (curvature/Hessian information) can be used to reshape the loss landscape into a more isotropic one, allowing all frequency modes to converge at nearly identical rates.

**Core Idea**: Replace first-order GD/Adam with preconditioned gradient descent (GN / LM) to "uniformly explore" the NTK subspace and eliminate delays caused by spectral bias; then switch back to first-order methods after the lazy phase is exhausted to compensate for generalization shortcomings.

## Method

### Overall Architecture
The paper does not propose a new network architecture but rather provides a theoretical characterization and experimental validation of how optimizers reshape the convergence dynamics of error modes. The framework follows a causal chain: ① Under the NTK/infinite-width approximation, decouple the error according to NTK eigenvectors into a set of independent scalar ODEs to obtain convergence rate expressions for each mode; ② Derive these rates for GD, LM, and GN, proving that LM/GN transforms the "rate proportional to eigenvalue $\lambda_i$" into a "flattened rate," improving the condition number from $\kappa_{GD}=\lambda_N/\lambda_1$ to $\kappa_{LM}\ll\kappa_{GD}$ (with $\kappa_{LM}\approx 2$ when $\mu=\lambda_1$); ③ Apply this "uniform exploration of the lazy subspace" to grokking, demonstrating that PGD compresses delayed generalization, but requires switching to first-order methods to overcome the generalization deficiency caused by staying in the lazy subspace.

Consider least-squares regression $\min_\theta L(\theta)$, $L(\theta)=\tfrac12\lVert f(\theta)-y\rVert^2$, where $f$ is an MLP of depth $L$ and width $W$. Taking the continuous gradient flow with step size $\eta\to0$, let $J_t=\nabla_\theta f(\theta(t))$ be the Jacobian. The function space dynamics are $\frac{\partial f}{\partial t} = -J_t J_t^T (f-y) = -K_t\,e$, where error $e(t)=f(\theta(t))-y$ and the NTK matrix $K_t=J_t J_t^T$ is symmetric positive semi-definite (and positive definite with high probability when sufficiently over-parameterized), allowing for orthogonal eigen-decomposition.

### Key Designs

**1. Eliminating Spectral Bias via Preconditioning: Flattening "Rate ∝ Eigenvalue" to Uniform Convergence**

Addressing the root cause of spectral bias—disproportionate convergence rates pulled apart by NTK eigenvalues. The authors first decouple the error in the infinite-width limit (where $K_t\to K_\infty$ is constant): let $\Lambda=\mathrm{diag}(\lambda_i)$ be the eigenvalues of $K_t$ and $\hat e_i$ be the error component on the $i$-th eigenvector. Standard gradient flow gives $\frac{\partial}{\partial t}\hat e_i = -\lambda_i\hat e_i$ (baseline from Lemma 3.1→3.3). This accurately reflects spectral bias: the learning rate must be small enough to stabilize the largest eigenvalue $\lambda_N$, causing small eigenvalue modes to decay slowly at $\sim\lambda_i/\lambda_N$.

Introducing Levenberg-Marquardt (LM) preconditioning, where the update is $\theta_{n+1}=\theta_n-\eta(\mu I + J_t^T J_t)^{-1}J_t^T(f-y)$ (equivalent to ridge regression in least squares), the gradient flow derivation yields (Lemma 3.2):

$$\frac{\partial}{\partial t}\hat e_i = -\frac{\lambda_i}{\mu+\lambda_i}\,\hat e_i.$$

The mapping $\lambda_i \mapsto \frac{\lambda_i}{\mu+\lambda_i}$ "squashes" eigenvalues of different magnitudes into a range near 1, improving the condition number from $\kappa_{GD}=\lambda_N/\lambda_1$ to $\kappa_{LM}=\frac{\lambda_N}{\lambda_1}\cdot\frac{\lambda_1+\mu}{\lambda_N+\mu}\ll\kappa_{GD}$. Specifically, when $\mu=\lambda_1$, $\kappa_{LM}\approx 2$. As $\mu\to0$, LM degrades to Gauss-Newton (GN): $\theta_{n+1}=\theta_n-\eta(J_t^T J_t)^{\dagger}J_t^T(f-y)$ (where $\dagger$ is the pseudo-inverse with truncation $\varepsilon$ due to potential Jacobian ill-conditioning). In this case (Lemma 3.3):

$$\frac{\partial}{\partial t}\hat e_i = -\mathbf 1_{\lambda_i(e)>\varepsilon}\,\hat e_i,$$

meaning that except for the smallest eigenvalues discarded by the pseudo-inverse truncation (usually corresponding to the highest geometric frequencies), **all modes decay exponentially at a uniform rate**—spectral bias is completely eliminated. LM can thus be viewed as a trust-region variant of GN, with $\mu$ acting as an interpolation knob between SGD and GN.

**2. Grokking as a Lazy→Rich Transition: Compressing the Delay Window with PGD**

Addressing the debate—is grokking caused by weight decay, adaptive optimizers, or other factors? The authors adopt the hypothesis of Kumar (2024) and Zhou (2024): the network is initially trapped in the lazy subspace $f(x,\theta)\approx f(x,\theta_0)+J_0(\theta-\theta_0)$, where spectral bias allows it to learn only low-frequency features. It suddenly generalizes only after "escaping" the lazy region and entering the rich regime of feature learning. Since Design 1 proves PGD can uniformly explore the NTK subspace, if grokking truly stems from "spectral bias + prolonged stay in the lazy regime," PGD should significantly shorten the time to generalization by advancing all modes simultaneously and uniformly rather than prioritized by frequency.

Experiments verify this by artificially enlarging the lazy regime (where $\alpha\to\infty$ corresponds to a larger lazy region) using a scaling factor $\alpha$ for initialization or outputs. Across modular addition, high-dimensional polynomial regression, Transformer modular arithmetic, and MNIST, it is observed that under SGD, the grokking gap (delay between train and test) increases with $\alpha$. Conversely, LM compresses this delay "consistently across different $\alpha$," and testing dynamics are nearly insensitive to $\alpha$. This provides direct evidence from the optimization dynamics side that "grokking is a transition phenomenon, and spectral bias is the primary cause rather than overfitting or adaptivity."

**3. PGD Generalization Gap and Hybrid Strategy: PGD to Exhaust the Lazy Region, First-Order to Repair Generalization**

Addressing the counter-intuitive phenomenon—PGD converges extremely fast but ultimately generalizes worse (e.g., in Transformer modular arithmetic, GGN achieves only 45% vs. Adam's 100%). The authors attribute this to high-order methods rapidly reducing error toward the lazy/NTK solution $w^*_{NTK}$ and then "sticking" in the lazy subspace, making it difficult to leave that plane for true non-linear feature learning (rich regime). The NTK solution $w^*_\mu$ is under-generalized relative to the true target $w^*$. The remedy is the opposite of the "second-order refinement" convention in PDE practice: **use PGD to exhaust the linear/lazy region first, then switch back to a first-order method** (e.g., on MNIST, 2,000 steps of LM followed by 20,000 steps of AdamW). This allows the first-order updates to exit the subspace, reaching or even exceeding the original test accuracy. This "PGD start + first-order finish" workflow is the practical conclusion of the paper and explains why prior works observed poor generalization in high-order methods.

> Assumption for applicability: The above theory strictly holds in the NTK / near-linear region (infinite width or specific scaling/initialization that keeps $K_t$ approximately stationary). Once the rich regime dominated by non-linear feature learning is entered, the linear curvature approximation (Eq. $f\approx f_0+J_0(\theta-\theta_0)$) fails, and the advantages of PGD disappear—this is both a theoretical boundary and the reason observed in experiments why "preconditioning effectiveness stops abruptly at the rich regime."

### Loss & Training
Regression tasks use MSE (where the Fisher Information Matrix coincides with Gauss-Newton, requiring only the Jacobian). Transformer classification tasks use cross-entropy (requiring generalized Gauss-Newton, GGN). Inversion of the large matrix $(\mu I + J_t^T J_t)$ is approximated using iterative methods like Conjugate Gradient or the Sherman-Morrison-Woodbury formula, often coupled with line search for step size determination. Hybrid strategy: start with several steps of LM/GN → switch to AdamW/Adam for finishing.

## Key Experimental Results

### Main Results (Qualitative conclusions on convergence and grokking)

| Task | Optimizer Comparison | Key Observation |
|------|-----------|---------|
| Fitting multi-frequency function $u(x)=\tfrac13\sum_{k=1}^3 k\sin((2k{+}1)\pi x-k)$ (2 layers, width 80) | SGD vs. LM ($\mu{=}0.5,0.1$) vs. GN | GN modes decay exponentially and uniformly; LM interpolates between SGD and GN based on $\mu$, confirming Lemma 3.2/3.3 |
| 2D Poisson PINN (width 256, solution $\sin(\pi nx)\sin(\pi my)$) | SGD / Adam / LM ($\mu{=}0.1$) | SGD/Adam drop quickly early on by eliminating low frequencies; LM's advantage increases with frequency, with nearly identical slopes across frequencies |
| MNIST grokking (Initialization $\times\alpha$) | AdamW / Adam / LM | LM shortens generalization delay, but final accuracy is lower than first-order; grokking is unrelated to weight norm increases/decreases |
| Transformer modular arithmetic | GGN vs. Adam | GGN accuracy rises faster but plateaus at 45%, while Adam reaches 100%; increasing GGN iterations leads to overfitting rather than generalization |

### Ablation Study

| Configuration | Phenomenon | Explanation |
|------|------|------|
| SGD only (Increasing $\alpha$) | Grokking delay increases with $\alpha$ | The lazy regime is enlarged; spectral bias slows convergence of unseen modes |
| LM/GN only | Delay is compressed consistently across $\alpha$, but final generalization is weak | Uniform exploration of the NTK subspace, but remains trapped in the lazy region |
| LM 2,000 steps → AdamW 20,000 steps (MNIST) | Restores or exceeds final first-order accuracy | Exhausts the lazy region first, then uses first-order method to exit the subspace |
| "Max Frequency" error in polynomial regression (1D subspace) | Train/test subspaces do not converge simultaneously under SGD; nearly simultaneous under PGD | Directly proves spectral bias causes slow convergence of unseen modes; PGD eliminates this difference |

### Key Findings
- **GN providing "uniform exponential decay across all frequencies"** is the strongest point of theory-experiment alignment (Lemma 3.3's $\mathbf 1_{\lambda_i>\varepsilon}$ manifested as uniform slopes in FFT mode error plots).
- **The dependency of grokking delay on scaling $\alpha$ basically disappears under PGD**, supporting the "grokking = lazy→rich transition + spectral bias as primary cause" theory over overfitting or adaptivity.
- **The generalization shortcoming of high-order methods can be fixed by "switching to first-order for finishing"**—a counter-intuitive conclusion compared to the "second-order refinement" convention in the PDE community.

## Highlights & Insights
- Unifies "spectral bias" and "grokking"—two seemingly distinct phenomena—under a single root cause (being trapped in an ill-conditioned NTK subspace), then explains both using the classical tool of preconditioning from numerical linear algebra.
- Three Lemmas map the error rates of GD→LM→GN into a continuous spectrum $\lambda_i \to \frac{\lambda_i}{\mu+\lambda_i} \to \mathbf 1_{\lambda_i>\varepsilon}$, connected by the single knob $\mu$, creating a minimal and verifiable theory.
- "PGD start + first-order finish" is a directly transferable training trick: for any task suffering from spectral bias or slow high-frequency convergence (e.g., PINNs, Implicit Neural Representations), consider using curvature methods to quickly explore the linear region before switching to Adam.

## Limitations & Future Work
- The theory strictly holds only in the NTK / near-linear region; convergence analysis in the rich regime remains an open problem where PGD loses its advantage.
- The paper does not address another potential cause of grokking—train/test set size (data-side factors), focusing entirely on optimization dynamics.
- High-order methods have non-trivial per-step inversion costs (requiring CG/SMW approximations + line search); GN pseudo-inverse truncation $\varepsilon$ can lose the highest frequency modes.
- Most experiments involve shallow MLPs or small tasks with artificial scaling to induce grokking; extensibility to large-scale real-world models needs verification. Code is yet to be released.

## Related Work & Insights
- **vs. Kumar et al. (2024) / Zhou et al. (2024)**: While they proposed grokking hypotheses based on "lazy→rich transitions" and "spectral mismatch," this paper **actively verifies** these hypotheses using PGD as a controllable optimizer—directly compressing the delay through uniform exploration of the NTK subspace to provide causal evidence from the dynamics side.
- **vs. Wadia (2021) / Buffelli (2024)**: They observed poor generalization in high-order/curvature methods without a clear mechanism; this paper attributes it to "sticking in the lazy subspace" and provides the "first-order switch" fix, turning negative observations into a usable strategy.
- **vs. Adam (First-order adaptive)**: Adam provides per-parameter learning rates (diagonal preconditioning), which shortens the lazy regime but ignores cross-parameter coupling. Ill-conditioned directions remain slow, whereas GN/LM uses full curvature information to handle interactions, significantly outperforming on high frequencies.

## Rating
- Novelty: ⭐⭐⭐⭐ Unifying spectral bias and grokking through preconditioning and providing counter-intuitive hybrid training conclusions. Uses classical tools with a fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐ Tasks cover regression/PINN/modular arithmetic/MNIST and align with theory, but mostly focus on shallow networks and small scales with artificial scaling.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations with one-to-one mapping between Lemmas and figures; complete logical chain.
- Value: ⭐⭐⭐⭐ "PGD start + first-order finish" is practically valuable for high-frequency tasks like PINNs/INR and deepens understanding of optimization dynamics and learning phase transitions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On the Convergence Direction of Gradient Descent](on_the_convergence_direction_of_gradient_descent.md)
- [\[ICLR 2026\] Gradient Descent with Large Step Sizes: Chaos and Fractal Convergence Region](gradient_descent_with_large_step_sizes_chaos_and_fractal_convergence_region.md)
- [\[ICLR 2026\] Egalitarian Gradient Descent: A Simple Approach to Accelerated Grokking](egalitarian_gradient_descent_a_simple_approach_to_accelerated_grokking.md)
- [\[ICLR 2026\] Fast Convergence of Natural Gradient Descent for Over-parameterized Physics-Informed Neural Networks](fast_convergence_of_natural_gradient_descent_for_over-parameterized_physics-info.md)
- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](../../ICML2026/optimization/on_the_convergence_rate_of_lora_gradient_descent.md)

</div>

<!-- RELATED:END -->
