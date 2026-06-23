---
title: >-
  [Paper Note] On the Interpolation Effect of Score Smoothing in Diffusion Models
description: >-
  [ICLR 2026][learning_theory][Diffusion Model] This paper demonstrates through analytical derivations and numerical experiments that diffusion models can "create" new samples not present in the training set because neural networks learn a **smoothed version** of the empirical score function (ESF). This smoothing directly drives denoising trajectories to generate sa
tags:
  - ICLR 2026
  - learning_theory
  - Diffusion Model
date: 2026-05-08
content_hash: ee6bef8b651b939c
---
# On the Interpolation Effect of Score Smoothing in Diffusion Models

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=O33LAUliUF](https://openreview.net/forum?id=O33LAUliUF)  
**Code**: https://github.com/google-research/diffusion-score-smoothing  
**Area**: Learning Theory / Diffusion Models  
**Keywords**: Score Smoothing, Diffusion Models, Memorization, Generalization Mechanism, Denoising Dynamics  

## TL;DR
This paper demonstrates through analytical derivations and numerical experiments that diffusion models can "create" new samples not present in the training set because neural networks learn a **smoothed version** of the empirical score function (ESF). This smoothing directly drives denoising trajectories to generate samples that interpolate between training data points, thereby avoiding the memorization inevitably caused by an exact ESF.

## Background & Motivation

**Background**: Score-based diffusion models train a neural network to fit a family of time-varying score functions $\nabla\log p_t$. During inference, a probability flow ODE $\dot{x}_t = -\tfrac12 s_t(x_t)$ is used for reverse denoising to sample high-quality images or molecules from Gaussian noise. Their most striking feature is the ability to generate new data not found in the training set—a form of "creativity."

**Limitations of Prior Work**: Theoretically, the unique optimal solution for the training objective (score matching loss) is the **empirical score function** (ESF), $\nabla\log p_t^{(n)}$, which is determined by the training set $S=\{y_k\}$ and can be written in closed form. However, if this exact ESF is used for denoising dynamics, the reverse process precisely inverts the forward noise-adding process, inevitably returning to the empirical distribution $p_0^{(n)}$—resulting in **memorization**. In practice, when model capacity is too large relative to the training set, this phenomenon of "learning so well that it only memorizes" is indeed observed.

**Key Challenge**: To enable diffusion models to generate new samples, it is crucial that the neural network **does not** perfectly learn the ESF. But what kind of "perfect imperfection" is required? And how does this imperfection translate into meaningful creativity? This is the theoretical question the paper addresses.

**Key Insight**: The authors are inspired by Scarvelis et al. (2025), which showed that smoothing the score function allows diffusion models to generate the "centroid" of training data. This paper further argues that regularization effects during neural network training **naturally** lead to learning a smoothed version of the ESF. This smoothed score quantitatively guides denoising trajectories toward interpolation points between training data, representing the simplest form of "meaningful creativity."

**Core Idea**: Replace the exact ESF with an analytically tractable "smoothed ESF" to prove: (1) two-layer ReLU networks with regularization indeed approximate this smoothed version; (2) the denoising dynamics driven by this smoothed version interpolate along the data subspace rather than collapsing to training points.

## Method

### Overall Architecture

This is a mechanism-analysis paper with no new trainable modules. The "Method" consists of a series of interconnected theoretical arguments. The authors condense the problem into a clean 1D two-point setting: $d=1, S=\{y_1=-1, y_2=+1\}$. In this setting, the ESF, smoothed score, and denoising trajectories all have closed-form solutions, allowing the "smoothing $\to$ interpolation" causal chain to be fully explained before generalizing to high-dimensional subspaces and non-linear manifolds.

The chain consists of four steps: ① Derive the exact ESF and show that at small $t$, it approximates a piecewise linear ESF (PL-ESF) that pulls samples toward $\pm 1$ (memorization); ② Construct a **Smoothed Piecewise Linear ESF** (S-PL-ESF) parameterized by $\delta$, and prove via a variational problem involving "score matching loss + non-smoothness penalty" that this is exactly what a regularized two-layer ReLU network approximates; ③ Substitute the S-PL-ESF into the denoising ODE to analytically derive the flow map and endpoint distribution, proving the endpoint is an interpolation distribution with positive density on $[-1, 1]$ rather than two Dirac points; ④ Generalize to "subspace recovery" where training data lies on a 1D subspace of $\mathbb{R}^d$, and verify with numerical experiments ($d=2, 4, 20$ and circular/spherical manifolds) that the scores learned by neural networks exhibit the same smoothing-interpolation effect.

### Key Designs

**1. Smoothed PL-ESF: Blunting "Hard Jumps" into Ramps**

In the $d=1, n=2$ setting, the exact ESF is $\frac{d}{dx}\log p_t^{(n)}(x) = (\hat{x}_t^{(n)}(x)-x)/t$, where the posterior mean $\hat{x}_t^{(n)}(x)=\frac{p_N(x-1;\sqrt{t})-p_N(x+1;\sqrt{t})}{p_N(x-1;\sqrt{t})+p_N(x+1;\sqrt{t})}$ stays between $\pm 1$ and shares the sign of $x$. As $t\to0$, the Gaussian peaks sharpen, $\hat{x}_t^{(n)}(x)\to\mathrm{sgn}(x)$, and the ESF degenerates into the **PL-ESF** $\bar{s}_t^{(n)}(x)=(\mathrm{sgn}(x)-x)/t$. The sign jump at $x=0$ pulls all trajectories toward $\pm 1$, causing memorization.

The authors construct the S-PL-ESF by replacing this jump with a ramp of finite slope, parameterized by $\delta\in(0,1]$:

$$\hat{s}_{t,\delta}^{(n)}(x)=\begin{cases}-(x+1)/t, & x\le \delta-1,\\[2pt] -(x-1)/t, & x\ge 1-\delta,\\[2pt] \dfrac{\delta}{1-\delta}\cdot\dfrac{x}{t}, & x\in(\delta-1,\,1-\delta).\end{cases}$$

It remains identical to the PL-ESF on the sides ($\hat{s}_{t,1}^{(n)}\equiv\bar{s}_t^{(n)}$) but uses a gentle positive slope in the interval $[\delta-1, 1-\delta]$ instead of a jump. Smaller $\delta$ indicates stronger smoothing. This modification is the pivot of the paper: it is analytically solvable and quantifies "creativity from imperfect scores."

**2. NN Regularization ≈ Non-smoothness Penalty: Proving the Smoothing Bias**

Why do regularized networks learn something like the S-PL-ESF? Using results from Savarese et al. (2019), regularizing the weights of a two-layer ReLU network (equivalent to weight decay) in 1D fitting is essentially equivalent to penalizing the **non-smoothness** $R[f]=\int_{-\infty}^{\infty}|f''(x)|\,dx$. The learning process is modeled as a variational problem:

$$r_{t,\epsilon}^* := \inf_f R[f]\quad \text{s.t.}\quad L_t^{(n)}[f]<\epsilon,$$

i.e., finding the smoothest function among those with a score matching loss below $\epsilon$. **Proposition 1** provides the core answer: setting $\delta_t=\kappa\sqrt{t}$, for all sufficiently small $t$, S-PL-ESF satisfies both $L_t^{(n)}[\hat{s}_{t,\delta_t}^{(n)}]<\epsilon$ and $R[\hat{s}_{t,\delta_t}^{(n)}]<(1+8\sqrt{\epsilon})\,r_{t,\epsilon}^*$. **Lemma 2** supports this by showing that at small $t$, $p_t^{(n)}$ concentrates near $\pm 1$ and the loss is dominated by these neighborhoods, allowing $\delta_t$ to shrink with $\sqrt{t}$ while keeping the loss constant.

**3. Analytical Solution of Denoising Dynamics: Direct Interpolation via Flow Map**

Substituting S-PL-ESF into the denoising ODE $\dot{x}_t=-\tfrac12\hat{s}_{t,\delta_t}^{(n)}(x_t)$ allows for a closed-form flow map $\phi_{s|t}$ due to piecewise linearity. The differentiability of S-PL-ESF splits the $x$–$\sqrt{t}$ plane into three regions. Points in the side regions converge to $y=\pm 1$, while points in the **middle region** converge to an endpoint between $-1$ and $1$:

$$\phi_{0|t}(x)=\begin{cases}x/(1-\delta_t), & x\in[\delta_t-1,\,1-\delta_t],\\ \mathrm{sgn}(x), & \text{otherwise}.\end{cases}$$

Using the push-forward formula, the evolution of the marginal distribution $\hat{p}_s^{(n,t_0)}$ can be derived. The endpoint distribution is $\hat{p}_0^{(n,t_0)}=a_+\delta_1+a_-\delta_{-1}+(1-a_+-a_-)\tilde{p}_0^{(n,t_0)}$, where $\tilde{p}_0^{(n,t_0)}$ has **positive density** on $[-1, 1]$, representing smooth interpolation. In contrast, the exact ESF results in a purely singular $p_0^{(n)}$.

**4. High-dimensional Subspace Recovery: Tangential Smoothing and Normal Collapse**

When training points lie on a 1D subspace (e.g., the $[x]_1$ axis) in $\mathbb{R}^d$, the ESF is identical to the 1D case in the **tangential** direction and a pure linear function in the **normal** directions. The authors generalize non-smoothness to $R^{(d)}[f]=\sup_{(w,b)}\int\|\nabla_w^2 f(wx+b)\|\,dx$ and define a high-dimensional S-PL-ESF that is smoothed tangentially but linear normally. **Propositions 4/5** prove that the normal component is linear and does not add to the penalty, making the high-dimensional S-PL-ESF approximately optimal. Denoising decouples by dimension: the normal component collapses to the subspace at a $\sqrt{t}$ rate, while the tangential component interpolates.

### Loss & Training
The analysis targets the time-averaged score matching loss $\min_\theta \tfrac1T\int_0^T L_t^{(n)}[s_\theta(\cdot,t)]\,dt$, where $L_t^{(n)}[f]=t\cdot\mathbb{E}_{x\sim p_t^{(n)}}[\,\|f(x)-\nabla\log p_t^{(n)}(x)\|^2\,]$. Numerical experiments train 2/3-layer ReLU networks with and without weight decay to observe score smoothing.

## Key Experimental Results

### Main Results
The paper focuses on visualization and alignment with analytical curves, comparing distributions driven by exact ESF, S-PL-ESF, and learned SF.

| Experiment (Setting) | Comparison | Key Observation |
|--------------|----------|----------|
| 6.1 ($d=2, n=4$) | ESF vs S-PL-ESF vs Network SF | ESF's tangential variance collapses to 0 (memorization); S-PL-ESF maintains positive tangential variance (interpolation), matching analytical predictions; Network SF evolution is closer to S-PL-ESF. |
| 6.2 (Uniform sampling on circle) | Network SF (no weight decay) vs ESF | Network scores are significantly smoother in the tangential (polar angle) direction, generating samples that perform linear interpolation between training points, forming a regular polygon. |
| 6.3 ($d=20$ Spherical manifold) | 2/3-layer Network SF | Visualized via stereographic projection; samples interpolate along the data manifold, showing the effect holds for non-linear, high-dimensional manifolds. |

### Key Findings
- **Networks naturally learn smoothed scores**: In 6.1, Network SF is much closer to S-PL-ESF than the exact ESF in both distribution evolution and the scores themselves.
- **Implicit regularization is sufficient**: In 6.2, even **without weight decay**, scores remained smooth and produced interpolation, suggesting implicit gradient-based regularization is enough.
- **Smoothing $\neq$ Early Stopping**: Analytical and experimental results emphasize that early stopping merely adds noise and does not produce true generalization along the subspace, unlike smoothing.
- $\delta_t\propto\sqrt{t}$ is the critical scaling: It allows the smoothing window to shrink with the noise scale, maintaining loss while approximating smoothness optimality.

## Highlights & Insights
- **Reduction of "Creativity" to a Tractable Scalar**: Using $\delta$ in S-PL-ESF turns the vague notion of "imperfect learning" into a concrete, solvable, and provably near-optimal object.
- **Three-way Verification**: Variational optimality (networks learn it), ODE analytical solutions (it causes interpolation), and numerical experiments (networks do it) create a closed logical loop.
- **Decoupling Perspective**: Viewing scores as tangential (interpolation) and normal (collapse to manifold) provides a reusable framework for understanding generalization in high-dimensional manifolds.
- **Clarification of Misconceptions**: It explicitly identifies that early stopping is just noise addition, whereas score smoothing is the driver of true interpolation.

## Limitations & Future Work
- **Highly Simplified Settings**: Core theorems rely on 1D/subspace settings, uniform training points, and two-layer networks, which are far from real-world UNet/Transformer architectures and complex image manifolds.
- **Shallow Link to Implicit Bias**: While experiments show implicit regularization works, a complete theoretical depth for why gradient training induces this specific smoothing in high dimensions is missing.
- **Single Regularization Mechanism**: The analysis focuses on non-smoothness penalties; whether other regularization forms or diffusion variants (e.g., flow matching) follow this remains unproven.

## Related Work & Insights
- **vs. Memorization Studies (Yi et al. 2023 / Li et al. 2024)**: This paper builds on the fact that exact ESF causes memorization by positively defining what the "correct" imperfection looks like.
- **vs. Scarvelis et al. (2025)**: Advances the "centroid" idea to "continuous interpolation along subspaces" with variational proofs.
- **vs. Early Stopping**: Clearly distinguishes that early stopping is not the same as the generalization provided by smoothing.
- **vs. ReLU Network Theory (Savarese et al. 2019)**: Bridging weight-norm regularization to score matching loss as a mechanism for diffusion generalization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Establishes the first complete causal chain from "score smoothing $\to$ interpolation $\to$ generalization."
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic verification from 1D to 20D manifolds, though restricted to synthetic data.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, moving from simple to complex with strong alignment between formulas and figures.
- Value: ⭐⭐⭐⭐ Provides a clean theoretical anchor for understanding generalization, despite the gap with production architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Provable Separations between Memorization and Generalization in Diffusion Models](provable_separations_between_memorization_and_generalization_in_diffusion_models.md)
- [\[ICLR 2026\] Quotient-Space Diffusion Models](quotient-space_diffusion_models.md)
- [\[ICLR 2026\] Polynomial Convergence of Riemannian Diffusion Models](polynomial_convergence_of_riemannian_diffusion_models.md)
- [\[ICLR 2026\] Score-Based Density Estimation from Pairwise Comparisons](score-based_density_estimation_from_pairwise_comparisons.md)
- [\[ICLR 2026\] A Sharp KL Convergence Analysis for Diffusion Models under Minimal Assumptions](a_sharp_kl_convergence_analysis_for_diffusion_models_under_minimal_assumptions.md)

</div>

<!-- RELATED:END -->
