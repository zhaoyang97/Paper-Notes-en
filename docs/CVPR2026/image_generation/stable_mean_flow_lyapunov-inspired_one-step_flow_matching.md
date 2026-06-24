---
title: >-
  [Paper Note] Stable Mean Flow: Lyapunov-Inspired One-Step Flow Matching
description: >-
  [CVPR 2026][Image Generation][Flow Matching] Adds a "non-expansivity" regularization inspired by Lyapunov stability to Mean Flow, the current strongest one-step generation method. By forcing the one-step transport map not to amplify neighborhood perturbations, this eliminates the instability issue of JVP exploding to NaN/Inf during training, reducing the one-step FID from 2.92 to 2.86 on CIFAR-10 with significantly faster convergence.
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Flow Matching"
  - "Mean Flow"
  - "One-step Generation"
  - "Lyapunov Stability"
  - "Non-expansivity Regularization"
date: 2026-05-08
content_hash: 46f970a370a375cc
---

# Stable Mean Flow: Lyapunov-Inspired One-Step Flow Matching

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Stable_Mean_Flow_Lyapunov-Inspired_One-Step_Flow_Matching_CVPR_2026_paper.html)  
**Code**: To be confirmed  
**Area**: Image Generation / Flow Matching / One-step Generation  
**Keywords**: Flow Matching, Mean Flow, One-step Generation, Lyapunov Stability, Non-expansivity Regularization

## TL;DR
Adds a "non-expansivity" regularization inspired by Lyapunov stability to Mean Flow, the current strongest one-step generation method. By forcing the one-step transport map not to amplify neighborhood perturbations, this eliminates the instability issue of JVP exploding to NaN/Inf during training, reducing the one-step FID from 2.92 to 2.86 on CIFAR-10 with significantly faster convergence.

## Background & Motivation

**Background**: Diffusion/score models produce high image quality but require hundreds or thousands of denoising steps; even with few-step samplers or distillation, they remain iterative. Flow matching instead learns an ODE velocity field that deterministically transports noise to data. Straightening the trajectory enables one-step image generation. Among these, Mean Flow (MeanFlow [9]) is the current SOTA for one-step generation. Instead of learning instantaneous velocity, it learns the **mean velocity** over a time interval, $u(z_t,r,t)=\frac{1}{t-r}\int_r^t v(z_\tau,\tau)\,d\tau$. Since the target is a well-defined physical quantity (mean displacement rate), it can be stably trained from scratch and truly achieves 1-NFE (single forward pass) sampling.

**Limitations of Prior Work**: One-step methods are highly fragile in practice. Training often suffers from "sensitivity explosion" and trajectory ambiguity, leading to instability and collapsed image quality. A specific and fatal failure mode is that the Jacobian-Vector Product (JVP, the total derivative of the mean velocity with respect to time) computed during Mean Flow training periodically produces NaN/Inf, which immediately halts training—as shown in Figure 2 of the paper, where Mean Flow's JVP collapses around 53,000 steps.

**Key Challenge**: The root cause is that the learned one-step map can be **locally expansive**—it amplifies input perturbations, causing adjacent trajectories to cross, squeeze, or explode. Since one-step generation lacks step-by-step ground-truth targets to "correct errors", once the map is locally stretched, the trajectories become ill-posed. Consequently, regression targets at different timesteps contradict each other, and gradient propagation becomes unstable.

**Goal**: Under the premise of **fully retaining the Mean Flow target**, introduce a "no perturbation amplification allowed" constraint to the one-step map, keeping training within a numerically safe zone, while providing a provable upper bound for the error growth of one-step/multi-step generation.

**Key Insight**: Borrowing **Lyapunov stability** from dynamical systems theory. The authors observe that when the conditional target dynamics define a constant vector field $\dot x(t)=a=X_1-X_0$ (the straight-line optimal transport path derived from Brenier's theorem), the distance between any two trajectories $\|x(t)-y(t)\|$ remains permanently frozen—which is precisely Lyapunov stability, where "starting close means staying close forever." The authors aim to endow the learned flow with this "perturbation-confining" property.

**Core Idea**: In one sentence: **supplement the "GPS navigation (Mean Flow direction)" with "trajectory control (Lyapunov term to keep wheels from veering off)" by adding a hinge soft constraint of local non-expansivity (NE) to the one-step map.**

## Method

### Overall Architecture
Stable Mean Flow Matching (SMFM) = Mean Flow regression target + a stability regularization, with total loss $L = L_{MF} + \mu\,\ell_{stab}$. The network architecture, sampler (NFE=1), data augmentation, and optimization schedules are fully inherited from Mean Flow. **The only variable is this stability regularization**, meaning any performance differences can be strictly attributed to the stability mechanism itself.

To lay the groundwork: Flow matching describes the intermediate state using linear interpolation $z_t=a_t x+b_t\epsilon$, training a velocity field $v_\theta$ to predict the direction of $z_t$. The conditional target loss is $L_{CFM}(\theta)=\mathbb{E}_{t,x,\epsilon}\|v_\theta(z_t,t)-v_t(z_t\mid x)\|^2$, and sampling solves the ODE $\frac{d}{dt}z_t=v_\theta(z_t,t)$ from noise. Mean Flow replaces "learning instantaneous velocity" with "learning mean velocity" $u_\theta(z_t,r,t)$, representing the one-step reconstruction as $z_r\approx z_t-(t-r)\,u_\theta(z_t,r,t)$ (setting $(r,t)=(0,1)$ yields $z_0\approx z_1-u_\theta(z_1,0,1)$). Its training loss is:

$$L(\theta)=\mathbb{E}\Big[\big\|u_\theta(z_t,r,t)-\mathrm{sg}\big(v_t-(t-r)\tfrac{d}{dt}u_\theta(z_t,r,t)\big)\big\|_2^2\Big],$$

where the total derivative $\frac{d}{dt}u_\theta=\partial_z u_\theta\,v_t+\partial_t u_\theta$ is computed via JVP, and $\mathrm{sg}$ denotes the stop-gradient. On top of this, SMFM does three things: translates Lyapunov stability into an NE constraint for the one-step map (Design 1), softly incorporates NE into training using a hinge-squared loss and controls the perturbation radius $\delta$ (Design 2), and elevates the "terminal velocity error" to establish a global trajectory error upper bound, guaranteeing sampling quality (Design 3).

### Key Designs

**1. From $\delta$-cap to Local Non-Expansivity (NE): Translating Lyapunov Stability into Geometric Constraints on the One-Step Map**

First, write the one-step map as $\phi_r^\theta(t,\cdot):z\mapsto z-(t-r)\,u_\theta(z,r,t)$, which performs a "reverse Euler" update along the learned velocity field. Translating the Lyapunov definition directly yields a naive $\delta$-cap constraint: as long as the perturbation satisfies $\|\Delta z\|_2\le\delta$, the output displacement must satisfy $\|\phi_r^\theta(t,z_t+\Delta z)-\phi_r^\theta(t,z_t)\|_2\le\delta$. However, the authors point out that the $\delta$-cap is too loose—even if the total displacement falls within radius $\delta$, the map can still stretch violently locally. They instead employ a stronger, cleaner Non-Expansivity (NE) condition:

$$\big\|\phi_r^\theta(t,z_t+\Delta z)-\phi_r^\theta(t,z_t)\big\|_2\le\|\Delta z\|_2,\quad\forall\,\|\Delta z\|_2\le\delta.$$

NE requires the "output distance to not exceed the input distance", which automatically implies $\delta$-cap for any $\delta$ and additionally ensures infinitesimal perturbations are not amplified. This constraint yields two theoretical benefits: **trajectory uniqueness** (Theorem 3.1: under NE, the ODE $\dot z(s)=u_\theta(z(s),r,s)$ has a unique characteristic line for each initial value, preventing trajectory crossing/multivalued transport from causing self-contradictory regression targets, corresponding to deterministic coupling from source to target in optimal transport); and **bounded JVP** (Theorem 3.2: $\|\partial_z u_\theta\,\xi+\partial_t u_\theta\|\le C$, which directly suppresses the most common numerical collapse mode mentioned earlier—the update operator cannot infinitely amplify perturbations, but when the ground truth already satisfies NE, this regularization does not alter the learning of JVP, simply keeping the optimization trajectory within a numerically safe zone).

**2. Hinge-Squared Stability Loss and Trade-off for Perturbation Radius $\delta$: Converting Hard Constraints to Trainable Soft Penalties**

NE is a hard inequality and cannot be used directly as a loss. At each base point $z_t$, the authors randomly sample a perturbation within radius $\delta$: $\Delta z=\delta\,\xi/\|\xi\|_2,\ \xi\sim\mathcal N(0,I_d)$ (identifying a Gaussian vector on the unit sphere and scaling to radius $\delta$). Letting $\Delta u=u_\theta(z_t+\Delta z,r,t)-u_\theta(z_t,r,t)$ and $\alpha=t-r$, they soften the violation using a hinge-squared penalty:

$$\ell_{stab}(\Delta z)=\Big[\max\big(0,\ \|\Delta z-(t-r)\Delta u\|_2-\|\Delta z\|_2\big)\Big]^2.$$

This is identically 0 when NE is satisfied and grows quadratically on violation, meaning it only penalizes regions where "the mapping begins to expand" without disturbing already compliant regions. The final objective is $L=L_{MF}+\mu\,\ell_{stab}$. The radius $\delta$ is the most critical hyperparameter. The authors characterize its impact using Theorem 3.3 (robust bound for small radius): defining the signed violation $V(\Delta z)=\|\phi_r^\theta(t,z_t+\Delta z)-\phi_r^\theta(t,z_t)\|_2-\|\Delta z\|_2$, then $\mathbb{E}[V(\Delta z)]\le 2\delta$ and $P(V(\Delta z)>\tau)\le 2\delta/\tau$. This yields a clear trade-off: if $\delta$ is too large, the stochastic estimation of the objective becomes unreliable, dragging down regression towards the Mean Flow target; if $\delta$ is too small, gradients vanish, degrading back to original Mean Flow. Thus, a "small but non-zero" radius is selected (such as a typical step size $\alpha\|u_\theta\|$ or a fixed ratio of the local state norm), which stabilizes training without hurting learning. As a training detail, the stability term is only activated in early time windows, and the weight $\mu$ decays with iterations to avoid dominating the main Mean Flow target in later stages (see Algorithm 1).

**3. Terminal Error Control: Translating "Terminal Velocity Error" into Global Trajectory Error Upper Bounds**

SMFM has a defining property—sampling depends only on the velocity field at the terminal time $t=1$, with intermediate timesteps acting only through their impact on terminal quantities. Formally, let the terminal velocity error be $e_1:=u_\theta(z_1,r,1)-u^*(1,z_1)$ ($u^*$ is the oracle velocity, i.e., the conditional expectation of path tangents). The one-step reconstruction satisfies $\hat z_r-z_r=-(1-r)\,e_1$, showing that **endpoint accuracy is linearly controlled by terminal velocity mismatch**, and early errors $e_r$ do not enter this single-step relation. Based on this, the authors derive an error recurrence: Theorem 4.1 provides a forward single-step error bound (bounding $e_{t+\Delta t}$ via $e_t$, step size, and model constants); Corollary 4.1 globalizes it to a "non-growing upper bound" $\|e_{t_{k+1}}\|\le\max\{\|e_{t_k}\|,\ T_{t_k}\}$ (where $T_{t_k}=M^*+M_\theta+\alpha_{t_k}\Lambda_\theta$), forming a "safety envelope" where error contraction is triggered once the threshold is exceeded, preventing exploding errors in multi-step generation; Corollary 4.2 provides endpoint control, where the one-step scenario collapses to the explicit bound $\|e_1\|\le(M^*+M_\theta)+\tfrac12\Lambda_\theta$. Intuitively, as long as the model velocity field fits the oracle near a small window around $t=1$, the endpoint $z_1$ remains close to the oracle trajectory, with residuals growing at most linearly with the remaining time—guaranteeing endpoint stability for both one-step and multi-step generation. ⚠️ *Note: Refer to the original paper for the exact constants and proof details of all theorems/corollaries.*

### Loss & Training
The total loss is $L=L_{MF}+\mu\,\ell_{stab}$. In Algorithm 1 (Hybrid Mean Flow with Non-Expansivity), each step: sample $t\sim\mathrm{Unif}[\varepsilon,1],\ r\sim\mathrm{Unif}[0,t]$, and $z_t\sim p_t$; sample unit sphere perturbation $\xi$ and set $\Delta z=\delta\xi$; compute $u=u_\theta(z_t,r,t)$, the target $u_{tgt}=v-(t-r)(\partial_z u_\theta\,v+\partial_t u_\theta)$, and $L_{MF}=\|u-\mathrm{sg}(u_{tgt})\|_2^2$; compute $\Delta u=u_\theta(z_t+\Delta z,r,t)-u$ and $L_{stab}=\max(0,\|\Delta z-(t-r)\Delta u\|_2-\|\Delta z\|_2)^2$; combine into $L=L_{MF}+\mu L_{stab}$ and perform gradient descent. The stability term is activated early and $\mu$ decays across iterations. For CIFAR-10 main experiments, the model is trained for 500k steps with batch size 128 on a single A100; ImageNet uses SMF-XL/2.

## Key Experimental Results

### Main Results
CIFAR-10, one-step inference (NFE=1), lower FID is better. SMFM compared with representative one-step/few-step methods (Table 2):

| Method | NFE | FID |
|------|-----|-----|
| 1-Rectified Flow [26] | 1 | 378 |
| Glow [19] | 1 | 48.9 |
| Residual Flow [7] | 1 | 46.4 |
| GLFlow [37] | 1 | 44.6 |
| DenseFlow [11] | 1 | 34.9 |
| Consistency Model [31] | 2 | 5.83 |
| Consistency Flow Matching [40] | 2 | 5.34 |
| Mean Flow [9] | 1 | 2.92 |
| **Stable Mean Flow (Ours)** | 1 | **2.86** |

While maintaining single-step sampling efficiency, SMFM reduces the FID from Mean Flow's 2.92 to 2.86, indicating that the stability term improves robustness and image quality without sacrificing inference speed. On ImageNet, training Stable MeanFlow-XL/2 with the same pipeline achieved an FID of 3.37 at Epoch 240, slightly lower than the 3.43 reported by the original MeanFlow-XL/2 (a minor gap), but showing clear advantages in early training stages.

### Ablation Study
Hyperparameter sweep (Table 1, CIFAR-10 FID under a reduced budget grid search; bold indicates optimal; "–" as untested). Horizontal axis: stability weight $\mu$; vertical axis: perturbation radius $\Delta z$:

| $\Delta z\backslash\mu$ | 0 | 0.1 | 0.5 | 1 |
|------|------|------|------|------|
| 0.005 | – | 85.41 | 127.40 | 224.53 |
| 0.01 | 86.73 | **79.86** | 134.53 | 253.67 |
| 0.02 | – | 95.17 | 187.43 | 331.23 |

The optimal point is at $\Delta z=0.01,\ \mu=0.1$ with an FID of 79.86; in the same row, $\mu=0$ (disabling the stability term to revert to Mean Flow) yields 86.73, showing a clear improvement when applying a small weight stability term.

### Key Findings
- **SMFM is highly sensitive to both $\Delta z$ and $\mu$, with the effective range forming a narrow band**: A small $\Delta z$ provides localized, informative stability signals, and a moderate $\mu$ balances the stability signal and Mean Flow target; exceeding either parameter collapses performance rapidly—the regularization term dominates, over-contracting the dynamics and driving the velocity field away from the Mean Flow target (see Table 1, rightmost column $\mu=1$ where FID surges to 224~331).
- **Numerical stability is the core benefit**: While Mean Flow's JVP collapses (NaN/Inf) around 53,000 steps, SMFM maintains average and maximum JVPs at much safer, stable levels throughout (Figure 2), validating Theorem 3.2's bounded JVP.
- **Faster early convergence**: Mid-training qualitative comparisons (at 200k steps) show that SMFM converges to visually coherent samples earlier than Mean Flow; however, when NFE increases to 5 steps, the visual quality and FID differences between the two tend to diminish—the stability term is mostly beneficial for "early-stage + one-step" scenarios.
- **2D checkerboard toy verifies the mechanism**: The stability constraint forces the velocity field to "learn the correct directions first, then scale up magnitudes." Particle motion becomes more structured and monotonic, concentrating probability mass into correct grid cells faster with sharper boundaries; in contrast, Mean Flow is more dispersed early on, only catching up later.

## Highlights & Insights
- **Translating dynamical systems theory into a single-line loss**: The progression from Lyapunov stability $\rightarrow$ parallel drift of constant vector fields $\rightarrow$ non-expansivity of the one-step map $\rightarrow$ hinge-squared penalty is a very clean theoretical translation, offering a reusable methodological template.
- **A plug-and-play regularization at virtually zero cost**: With network, sampler, and training schedule completely unchanged, only a perturbation-penalty is added, making performance gains clearly attributable and highly portable to any Mean Flow-style training.
- **The practical observation that "terminal error linearly controls sampling error"**: $\hat z_r-z_r=-(1-r)e_1$ reduces the quality issue to "only needing to fit the oracle near $t=1$", providing a simple grip for the theoretical analysis of one-step generation.
- **Clear explanation of why NE outperforms naive $\delta$-cap**: While $\delta$-cap allows "total displacement to be within bounds despite violent local stretching," NE prohibits even infinitesimal expansion. This contrast illuminates how the choice of constraint formulation affects stability.

## Limitations & Future Work
- **Marginal absolute image quality gains**: The improvement (2.92 $\rightarrow$ 2.86 on CIFAR-10, 3.43 $\rightarrow$ 3.37 on ImageNet) is relatively minor, which the authors candidly acknowledge. The primary selling point is training stability and early convergence speed, rather than pushing SOTA visual limits significantly.
- **Narrow evaluation scope**: Main results are limited to CIFAR-10, a single ImageNet-XL/2 checkpoint, and 2-D toy examples, lacking systematic validation on higher resolutions, larger scales, or class-conditional generations.
- **Extreme hyperparameter sensitivity**: The effective range is a narrow band; slightly larger values of $\Delta z$ or $\mu$ trigger collapse (increasing FID by multiple times), making careful tuning necessary for practical deployment. The "scale-aware" selection of $\delta$ is described somewhat qualitatively.
- **No advantage in multi-step scenarios**: At NFE=5, there is no distinct gap between SMFM and Mean Flow, showing that benefits are constrained to one-step/early-stage settings.
- **Future directions**: The authors plan to amplify endpoint performance gains and optimize the algorithm to reduce training time.

## Related Work & Insights
- **vs Mean Flow [9]**: This work directly builds on Mean Flow, sharing its mean velocity objective and one-step sampling. The only difference is the addition of the NE stability regularization; the advantages are stabler JVPs, faster early convergence, and slightly superior FID, at the cost of introducing two highly sensitive hyperparameters.
- **vs Rectified Flow [26] / Consistency Flow Matching [40]**: These methods reduce sampling steps by "straightening trajectories" or "promoting piecewise velocity consistency," focusing on trajectory geometry. This work does not change the trajectory definition, but restricts the one-step map's ability to "amplify perturbations." This represents an orthogonal stability-focused perspective that can be stacked on top of them.
- **vs Consistency Model [31]**: CM employs 2-NFE multi-scale self-consistency (FID 5.83). This work sticks to true 1-NFE, choosing to "exchange stability regularization for one-step generation reliability" rather than relying on multi-step refinement.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing Lyapunov stability to one-step flow matching in the form of an NE soft regularization is a clean and novel mathematical translation.
- Experimental Thoroughness: ⭐⭐⭐ Main evaluations are centered on CIFAR-10, with only a single ImageNet point and no large-scale, high-resolution validation.
- Writing Quality: ⭐⭐⭐⭐ The theoretical derivations (uniqueness, bounded JVP, and endpoint error bounds) are well-structured, and the GPS/trajectory control analogy is highly intuitive.
- Value: ⭐⭐⭐⭐ A plug-and-play stability regularization requiring virtually zero cost, holding practical value for training Mean Flow-like one-step generation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Functional Mean Flow in Hilbert Space](functional_mean_flow_in_hilbert_space.md)
- [\[CVPR 2026\] RenderFlow: Single-Step Neural Rendering via Flow Matching](renderflow_single-step_neural_rendering_via_flow_matching.md)
- [\[CVPR 2026\] BiFM: Bidirectional Flow Matching for Few-Step Image Editing and Generation](bifm_bidirectional_flow_matching_for_few-step_image_editing_and_generation.md)
- [\[CVPR 2026\] Flow Matching for Multimodal Distributions](flow_matching_for_multimodal_distributions.md)
- [\[CVPR 2026\] LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories](leapalign_post_training_flow_matching_models_at_any_generation_step.md)

</div>

<!-- RELATED:END -->
