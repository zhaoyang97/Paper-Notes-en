---
title: >-
  [Paper Note] Improved Mean Flows: On the Challenges of Fastforward Generative Models
description: >-
  [CVPR 2026][Image Generation][One-step generation] The paper diagnoses two root causes of failures in MeanFlow (a one-step generation framework): the training objective's dependence on the network itself and the hard-coded CFG guidance scale before training. By rewriting the objective as a network-independent v-loss using the predicted marginal velocity as the JVP input, and treating the guidance scale as a variable condition injected via multi-token in-context conditioning…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "One-step generation"
  - "MeanFlow"
  - "flow matching"
  - "classifier-free guidance"
  - "1-NFE"
date: 2026-05-08
content_hash: a802956b5549d7ad
---

# Improved Mean Flows: On the Challenges of Fastforward Generative Models

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Geng_Improved_Mean_Flows_On_the_Challenges_of_Fastforward_Generative_Models_CVPR_2026_paper.html)  
**Code**: Reused original MeanFlow public code (no new repository listed)  
**Area**: Image Generation  
**Keywords**: One-step generation, MeanFlow, flow matching, classifier-free guidance, 1-NFE

## TL;DR
The paper diagnoses two root causes of failures in MeanFlow (a one-step generation framework): the training objective's dependence on the network itself and the hard-coded CFG guidance scale before training. By rewriting the objective as a network-independent v-loss using the predicted marginal velocity as the JVP input, and treating the guidance scale as a variable condition injected via multi-token in-context conditioning, the proposed iMF achieves a 1.72 FID on ImageNet 256×256 with a single function evaluation (1-NFE) trained from scratch. This represents an approximately 50% relative improvement over the original MeanFlow, approaching the performance of multi-step methods without any distillation.

## Background & Motivation
**Background**: Diffusion and flow matching models view generation as solving an ODE that maps a prior distribution to the data distribution, typically requiring multi-step numerical solvers and multiple function evaluations (NFE). Recently, a class of "fastforward generative models" has attempted to build ODE/SDE acceleration directly into the training objective, achieving very few or even one-step generation via "one-hop" jumps across large time intervals. MeanFlow (MF) is a representative of this: instead of learning the instantaneous velocity field $v$, it learns the **average velocity field** $u(z_t,r,t)=\frac{1}{t-r}\int_r^t v(z_\tau)\,d\tau$ between two time points, using the "MeanFlow identity" $u=v-(t-r)\frac{d}{dt}u$ to transform the non-integrable definition into a trainable objective.

**Limitations of Prior Work**: MF faces two unresolved issues. ① **The training objective depends on the network itself**: Since the ground truth $u$ is unavailable, MF substitutes the network's own prediction $u_\theta$ into the objective ($u_{tgt}=(e-x)-(t-r)\,\mathrm{JVP}(u_\theta;e-x)$). This is not a standard regression problem; the target moves with the network, leading to high-variance training loss that may fail to converge. ② **The CFG scale is fixed before training**: MF supports 1-NFE classifier-free guidance but requires the guidance scale $\omega$ to be fixed before training, making it unadjustable during inference. However, the optimal $\omega$ varies with model capability (larger models or those with more NFE prefer smaller $\omega$), making a pre-frozen scale sub-optimal.

**Key Challenge**: To achieve "one-hop" generation, fastforward models must construct look-ahead objectives during training. However, these objectives incorporate "the network's own predictions" and "pre-fixed hyperparameters," creating a direct conflict between **trainability (valid, low-variance objectives)** and the **look-ahead construction method**.

**Goal**: (i) Transform the MF objective into a standard, network-independent regression problem to stabilize training; (ii) Allow the CFG scale to take arbitrary values during both training and inference while maintaining 1-NFE.

**Key Insight**: The authors discovered that the u-loss in MF is mathematically **exactly equivalent** to a v-loss (instantaneous velocity loss) reparameterized by $u_\theta$. Thus, it is preferable to replace the regression target with the network-independent $v$ and subsequently fix the residual "illegal inputs."

**Core Idea**: Treat MeanFlow as a "v-loss reparameterized by $u_\theta$," ensuring the regression target depends only on the ground truth velocity and the prediction function only takes noise samples $z_t$ as input. Furthermore, treat all guidance-related hyperparameters as learnable conditions injected via a multi-token in-context approach.

## Method

### Overall Architecture
iMF retains the skeleton of MeanFlow (one-step generation, average velocity parameterization, ImageNet latent space training) but modifies the **formulation of the training objective** and the **method of condition injection**. Effectively, it is a transformation at the loss and architecture levels rather than a new pipeline. Four transformations progress logically: first, rewriting MF in **v-loss form** (to obtain a network-independent target $v$); this reveals that the prediction function still "leaks" $e-x$, requiring **legal parameterization** (replacing the JVP input from the conditional velocity $e-x$ with the network-predicted marginal velocity $v_\theta$) so the prediction function only depends on $z_t$; then, turning CFG from a fixed scale into **flexible guidance conditions** (treating $\omega$ and even the CFG interval as learnable conditions); and finally, using **improved in-context condition injection** to concatenate heterogeneous conditions ($r,t,c, \Omega$) as multiple tokens, allowing for the removal of the heavy adaLN-zero.

### Key Designs

**1. Rewriting MeanFlow as v-loss: Switching to a network-independent regression target**

The trouble with MF is that it optimizes a u-loss where the ground truth $u$ is replaced by the network's $u_\theta$, causing the target to "drift" with the network. The authors rearrange the MeanFlow identity as $v(z_t)=u(z_t)+(t-r)\frac{d}{dt}u(z_t)$. The instantaneous velocity $v$ on the left can serve as a **fixed regression target** as in standard Flow Matching. The composite function on the right is parameterized by $u_\theta$, denoted as $V_\theta \triangleq u_\theta(z_t)+(t-r)\,\mathrm{JVP_{sg}}(u_\theta;e-x)$, resulting in a Flow-Matching objective $\mathbb{E}\,\lVert V_\theta-(e-x)\rVert^2$. This formulation is **fully equivalent** to the original MF objective but reveals that MeanFlow is essentially a "v-loss reparameterized by $u_\theta$." Crucially, the target $v$ no longer depends on the network, stabilizing training.

**2. Legal Parameterization: Replacing JVP input with predicted marginal velocity**

The v-loss rewrite exposes a flaw: $V_\theta$ does not just consume $z_t$, but also "leaks" $e-x$ (written as $V_\theta(z_t,\,e-x)$), making it an "illegal" prediction function for standard regression. This stems from an approximation in the JVP where the **marginal velocity** $v(z_t)=\mathbb{E}[v_c\mid z_t]$ is replaced by the **conditional velocity** $v_c=e-x$. The authors rectify this by defining $V_\theta(z_t) \triangleq u_\theta(z_t)+(t-r)\,\mathrm{JVP_{sg}}(u_\theta;v_\theta)$, where the JVP tangent vector is also provided by the network's prediction $v_\theta$. Both components now only consume $z_t$. To implement $v_\theta$ at zero cost, the boundary condition $v(z_t,t)\equiv u(z_t,t,t)$ is used, setting $v_\theta(z_t,t)=u_\theta(z_t,t,t)$ without adding parameters. Why it works: The true regression target is the low-variance marginal velocity $v(z_t)$, whereas the conditional velocity $e-x$ has high variance and is amplified by the Jacobian. Replacing it with $v_\theta$ ensures iMF training loss decreases monotonically with much lower variance than original MF.

**3. Flexible Guidance: Turning CFG scale (and interval) into learnable conditions**

In original MF, $\omega$ is fixed, making it unadjustable at inference. The authors treat the guidance scale as a **condition** similar to timesteps $t,r$: $V_\theta(\cdot\mid c,\omega)\triangleq u_\theta(z_t\mid c,\omega)+(t-r)\,\mathrm{JVP_{sg}}$. During training, $\omega$ is sampled from a distribution; during inference, it can take any value, allowing a single model to sweep different $\omega$ in 1-NFE. This framework further incorporates the CFG interval $[t_{min},t_{max}]$, collectively denoted as the condition set $\Omega=\{\omega,t_{min},t_{max}\}$, unlocking full flexibility for one-step sampling.

**4. Improved in-context condition injection: Multi-token concatenation and removing adaLN-zero**

iMF handles many heterogeneous conditions: timesteps $r, t$, class $c$, and guidance set $\Omega$. Standard adaLN-zero approaches become "overburdened" when adding many embedding vectors together. The authors switch to in-context injection—originally considered inferior to adaLN-zero in DiT—but find that **using multiple tokens per condition** bridges the gap. The implementation uses 8 tokens for class and 4 tokens for each other condition, concatenated with image latent tokens along the sequence axis. This allows for the **complete removal of the parameter-heavy adaLN-zero**, reducing model size by about 1/3 (e.g., iMF-Base drops from 133M to 89M) without performance degradation.

### Loss & Training
The final training objective is a Flow-Matching style $\mathbb{E}_{t,r,x,e}\lVert V_\theta(z_t)-(e-x)\rVert^2$ using legal parameterization. A linear schedule $z_t=(1-t)x+t\,e$ is used. In $V_\theta$, $u$ and $\frac{d}{dt}u$ are computed simultaneously via one JVP, with a stop-gradient applied to $\frac{d}{dt}u$. Experimental settings follow the original MeanFlow code, training from scratch on ImageNet 256×256 class-conditional generation in latent space ($32\times32\times4$), primarily evaluating 1-NFE FID-50K.

## Key Experimental Results

### Main Results (System-level, ImageNet 256×256, 1-NFE, Train from scratch)

| Configuration | # params | Gflops | FID ↓ | IS ↑ |
|------|------|------|------|------|
| MF-B/2 | 131M | 23.1 | 6.17 | 208.0 |
| MF-XL/2 | 676M | 119.0 | 3.43 | 247.5 |
| iMF-B/2 | 89M | 24.9 | 3.39 | 255.3 |
| iMF-L/2 | 409M | 116.4 | 1.86 | 276.6 |
| **iMF-XL/2** | 610M | 174.6 | **1.72** | **282.0** |

iMF-XL/2 achieves a 1.72 FID (approx. 50% improvement over MF). Interestingly, iMF-B/2 at 89M matches the performance of the 676M MF-XL/2 (3.43), leading significantly at similar or smaller scales without distillation.

### Ablation Study (MF-B/2 backbone, 240 epochs, FID-50K)

| Configuration | FID (w/o CFG) | FID (w/ CFG) | Description |
|------|------|------|------|
| Original MF | 32.69 | 6.17 | Starting point |
| + $V_\theta$, $v_\theta=u_\theta(z_t,t,t)$ boundary condition | 29.42 | 5.97 | Zero extra parameters (Design 1+2) |
| + $V_\theta$, auxiliary v-head | 30.76 | 5.68 | No param increase at inference |
| + $\omega$-condition (flexible guidance) | 25.15 | 5.52 | Single scale condition (Design 3) |
| + $\Omega$-condition (incl. CFG interval) | 20.95 | 4.57 | Further conditioning |
| + in-context replacing adaLN-zero | — | 4.09 | Reduced from 133M to 89M (Design 4) |
| + Advanced Transformer block | — | 3.82 | Architecture gain |
| + Longer training (640ep) | — | 3.39 | — |

### Key Findings
- **"Legal regression target" is key for stable training**: Replacing the high-variance $e-x$ with the low-variance $v_\theta$ in the JVP tangent vector turns the loss from high-variance to monotonic; the boundary condition variant improves FID from 32.69 to 29.42 (Gain: 3.27) w/o CFG.
- **Stronger models benefit more from legal parameterization**: On MF-XL/2, the boundary condition improves FID from 3.43 to 2.99, confirming that larger networks better learn $v_\theta$ via $u_\theta(z_t,t,t)$.
- **Flexible guidance value extends beyond FID**: While $\Omega$-conditioning drops FID to 4.57, its primary value is enabling hyperparameter sweeps during inference.
- **In-context injection yields dual benefits**: Replacing adaLN-zero reduces FID from 4.57 to 4.09 while simultaneously cutting model size from 133M to 89M.

## Highlights & Insights
- **范式诊断 (Paradigm Diagnosis) via equivalent rewriting is elegant**: Proving that the MF u-loss equals a reparameterized v-loss identifies the "moving target" and "illegal input" issues immediately—a significant cognitive shift.
- **Zero-cost $v_\theta=u_\theta(z_t,t,t)$ is simple yet effective**: Leveraging the boundary condition where average velocity collapses to instantaneous velocity provides a legal, low-variance JVP input without extra parameters.
- **"Treating hyperparameters as conditions" is transferable**: The idea of conditioning on CFG scales/intervals is applicable to any one-step or few-step model requiring hyperparameter tuning at inference.
- **Multi-token in-context conditioning** re-evaluates the conclusion from DiT that in-context is inferior to adaLN-zero; by using more tokens per condition, it matches performance while saving 1/3 of parameters.

## Limitations & Future Work
- Verification is limited to the single benchmark of ImageNet 256×256 class-conditional generation; generalization to text-to-image or higher resolutions has not been tested.
- Specific implementation details for the auxiliary v-head and CFG condition distributions are primarily in the appendix; the choice between boundary conditions and v-heads varies with model scale.
- The continued practical need for stop-gradient despite being "theoretically removable" suggests unresolved optimization nuances.

## Related Work & Insights
- **vs Original MeanFlow**: iMF retains the framework but fixes the two fundamental flaws—network-dependent u-loss and fixed CFG—leading to a ~50% FID improvement.
- **vs Consistency Models / Shortcut / IMM**: While other fastforward methods use different look-ahead approximations, iMF focuses on the objective validity and CFG utility of the MeanFlow branch, offering orthogonal improvements.
- **vs Multistep Diffusion/Flow Matching**: iMF narrows the gap with multi-step methods to 1.72 FID in 1-NFE without relying on distillation, supporting the case for fastforward models as a standalone paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Uses equivalent rewriting to fix fundamental MF flaws with clean solutions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid ablation and scale comparisons, though limited to ImageNet.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous derivation and excellent analysis of loss variance.
- Value: ⭐⭐⭐⭐⭐ 1-NFE 1.72 FID establishes a more stable objective paradigm for one-step generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CaTok: Taming Mean Flows for One-Dimensional Causal Image Tokenization](catok_taming_mean_flows_for_one-dimensional_causal_image_tokenization.md)
- [\[CVPR 2026\] Functional Mean Flow in Hilbert Space](functional_mean_flow_in_hilbert_space.md)
- [\[CVPR 2026\] Stable Mean Flow: Lyapunov-Inspired One-Step Flow Matching](stable_mean_flow_lyapunov-inspired_one-step_flow_matching.md)
- [\[CVPR 2026\] Nonlinear Color Transfer via Learnable Bezier Flows](nonlinear_color_transfer_via_learnable_bezier_flows.md)
- [\[CVPR 2026\] Learning Straight Flows: Variational Flow Matching for Efficient Generation](learning_straight_flows_variational_flow_matching_for_efficient_generation.md)

</div>

<!-- RELATED:END -->
