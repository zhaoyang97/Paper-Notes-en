---
title: >-
  [Paper Note] Saving Foundation Flow-Matching Priors for Inverse Problems
description: >-
  [ICML 2026][Image Generation][warm-start] Addressing the observation that foundation flow-matching (FM) models like Stable Diffusion / Flux significantly underperform compared to domain-specific or even untrained priors in solving inverse problems, the authors propose FMPlug. By using a time-learnable warm-start guided by approximate samples and a sharp Gaussi
tags:
  - ICML 2026
  - Image Generation
  - warm-start
date: 2026-05-08
content_hash: 6463f3ea704c3e96
---
# Saving Foundation Flow-Matching Priors for Inverse Problems

**Conference**: ICML 2026  
**arXiv**: [2511.16520](https://arxiv.org/abs/2511.16520)  
**Code**: https://sun-umn.github.io/xm-plug/ (Project Page)  
**Area**: Diffusion Models / Inverse Problems / Flow Matching  
**Keywords**: Foundation Flow-Matching Priors, Inverse Problems, warm-start, Gaussian Regularization, Plug-and-play

## TL;DR
Addressing the observation that foundation flow-matching (FM) models like Stable Diffusion / Flux significantly underperform compared to domain-specific or even untrained priors in solving inverse problems, the authors propose FMPlug. By using a time-learnable warm-start guided by approximate samples and a sharp Gaussian shell constraint, FMPlug forces the latent variables of the foundation FM back onto the thin shell it truly "understands," significantly restoring its capability as an inverse problem prior.

## Background & Motivation

**Background**: Inverse Problems (IPs) aim to recover an unknown $x$ from measurements $y \approx A(x)$, typically achieved by minimizing $\ell(y, A(x)) + \Omega(x)$. Recently, a mainstream approach involves plugging Deep Generative Priors (DGP) into $\Omega$, particularly diffusion/flow models based on Flow Matching (FM). FM has replaced traditional diffusion in images, videos, and world models, becoming the de facto standard for SOTA generation.

**Limitations of Prior Work**: Existing methods for solving inverse problems based on FM (e.g., D-Flow, FlowDPS, FlowChef) almost entirely rely on domain-specific FMs (e.g., face priors trained on FFHQ). When switching to foundation FMs like Stable Diffusion V3 or Flux, performance degrades severely. The authors found that on AFHQ-Cat Gaussian deblurring, foundation FM priors generally perform several points worse in PSNR/LPIPS/CLIPIQA than domain FMs, and even fail to beat an untrained Deep Image Prior (DIP). On DIV2K, the results are even more extreme: the recovered images are worse than the blurry inputs themselves.

**Key Challenge**: Foundation FM models are strong "generators" but weak "priors"—their constraint on images is merely "looking like natural images," lacking domain-specific structural/semantic information. Previous tricks used to "strengthen" foundation FM priors (such as D-Flow's $z_0 = \sqrt{\alpha} y_0 + \sqrt{1-\alpha} z$ warm-start or log-likelihood regularization on $\|z_0\|^2$) are largely ineffective. From the perspective of Concentration of Measure (CoM), the authors reveal the root cause: samples from the source distribution $z_0 \sim \mathcal{N}(0, I_d)$ of standard FM almost entirely concentrate on an ultra-thin shell $S$ of $S^{d-1}(0, \sqrt{d})$. Consequently, the generator $G_\theta$ is only trained on $S$, and its behavior outside the shell is completely undefined. D-Flow's initialization, which mixes $y_0$ into $z_0$, pushes samples into another shell that is almost disjoint from $S$, effectively pushing $G_\theta$ into the "wild." Similarly, D-Flow's Gaussian likelihood penalty $-(d/2-1)\log\|z_0\|^2 + \|z_0\|^2/2$ changes extremely slowly in regions far from the optimum (e.g., less than a 0.031% relative change in the interval $\|z_0\|^2 \in [62000, 70000]$), failing to push $z$ back to the thin shell.

**Goal**: Without changing the plug-in framework, identify for foundation FM priors: (i) an initialization strategy that utilizes problem-relevant guidance samples while ensuring they fall within the training distribution of $G_\theta$; and (ii) a truly "hard" Gaussian constraint to lock $z$ onto the thin shell around $S^{d-1}(0, \sqrt{d})$.

**Key Insight**: The authors reduce the problem to "why D-Flow's warm-start/regularization fails" and explain it using the Gaussian Concentration of Measure theorem (Vershynin 2018)—the pretrained FM generator only works correctly when $z$ is almost strictly on the sphere of radius $\sqrt{d}$. Additionally, when $x$ and $y$ are close, the intermediate state of FM $z_t = \alpha_t x + \beta_t z$ can be approximated as $z_t \approx \alpha_t y + \beta_t z$; the introduced error is controllable as long as $\alpha_t$ is chosen appropriately.

**Core Idea**: Make $t$ a learnable parameter (finding the optimal "shortcut" starting point along the timeline) and explicitly constrain $z$ to a sharp spherical shell $\{z: \|z\|_2 \in [1-\epsilon, 1+\epsilon]\sqrt{d}\}$, turning a "wandering plug-in" into a "plug-in running on the manifold familiar to the generator."

## Method

### Overall Architecture
FMPlug addresses why foundation FMs fail as inverse problem priors and how to fix them. It does not alter the plug-in framework itself—it still fixes a pretrained foundation FM model $G_\theta$ (Stable Diffusion V3 or Flux) and optimizes $\min_z \ell(y, A \circ G_\theta(z)) + \Omega \circ G_\theta(z)$ in its latent space to fit measurement $y$. Two key modifications are made: the optimization variables are expanded from $z$ alone to $(z, t)$, allowing the flow time index $t \in [0, 1]$ to be learnable and used to construct the starting point via $y$. Simultaneously, a sharp spherical shell constraint is applied to $z$, forcing it to stay on the thin shell seen by $G_\theta$ during training. It employs a "simple-distortion" mode when only $y$ is available and switches to a "few-shot" mode when a few guidance images are available.

### Key Designs

**1. Instance-Guided Time-Learnable Warm-Start: Bringing the Starting Point into $G_\theta$'s Familiar Shell**

D-Flow's initialization, which mixes measurements with noise, fails because it pushes samples into regions where $G_\theta$ was never trained. FMPlug takes a different approach: the standard FM flow is $z_t = \alpha_t x + \beta_t z$ ($z \sim \mathcal{N}(0, I)$). When the image to be recovered $x = y + \epsilon$ and the deviation $\|\epsilon\|$ is small, the intermediate state can be directly approximated as $z_t \approx \alpha_t y + \beta_t z$, with an introduced error of $\alpha_t \epsilon$. Since the true magnitude of $\epsilon$ is unknown, but $\alpha_t$ is determined by $t$, making $t$ a learnable parameter allows the optimizer to adaptively find a "transfer point" along the flow where $\alpha_t \epsilon$ is negligibly small. The optimization problem becomes $\min_{z, t \in [0, 1]} \ell(y, A \circ G_\theta(\alpha_t y + \beta_t z, t))$. Starting from $t > 0$ instead of $t = 0$ results in a shorter path, a starting point closer to the training distribution, and faster convergence by bypassing a portion of the ODE integration.

**2. Sharp Gaussian Shell Constraint: Replacing Soft Regularization with Hard Constraints**

D-Flow uses the negative log-likelihood $h(z_0)$ as Gaussian regularization, but it is nearly flat far from the optimal radius $\sqrt{d-2}$ (empirical tests show less than 0.031% relative change in the $\|z_0\|^2 \in [62000, 70000]$ interval), providing virtually no gradient to pull samples. FMPlug replaces this with a hard constraint. Based on the Gaussian Concentration of Measure theorem, the norm of a $d$-dimensional standard Gaussian vector falls within $[(1-\epsilon)\sqrt{d}, (1+\epsilon)\sqrt{d}]$ with extremely high probability. Thus, a thin shell set $S^{d-1}_\epsilon(0, \sqrt{d}) = \{z: \|z\|_2 \in [1-\epsilon, 1+\epsilon]\sqrt{d}\}$ is defined and added as an optimization constraint: $\min_{z, t} \ell(y, A \circ G_\theta(\alpha_t y + \beta_t z, t)) \;\text{s.t.}\; z \in S^{d-1}_\epsilon(0, \sqrt{d})$. Implementation-wise, this is equivalent to adding a set-indicator regularization term solved via projection or penalty methods—any $z$ with an out-of-bounds norm is pushed directly back into the shell, ensuring $G_\theta$ always operates on the trained manifold.

**3. Few-shot Guidance Extension: Rescuing Scientific Inverse Problems where $y$ is far from $x$**

In scientific imaging like microscopy or astronomy, there is a lack of data to train domain FMs, and measurements $y$ are distant from the ground truth $x$. The $y \approx x$ assumption in simple-distortion collapses. FMPlug suggests a compromise: if a few "neighboring samples" $\{x_i\}$ are available, their mean $\bar{x} = \frac{1}{n}\sum x_i$ (or a random sample) replaces $y$ in the warm-start to construct $z_t \approx \alpha_t \bar{x} + \beta_t z$, while keeping the sharp shell constraint. The data term $\ell(y, A \circ G_\theta(\cdot))$ still constrains the output to fit the actual measurement, so these neighbors only serve to pull the foundation FM toward the correct submanifold without overriding the measurement signal.

### Loss & Training
The process is entirely training-free, with no network parameters updated. The objective is $\min_{z, t} \ell(y, A \circ G_\theta(\alpha_t y + \beta_t z, t))$, where $\ell$ is L2 or perceptual loss. The constraint $z \in S^{d-1}_\epsilon(0, \sqrt{d})$ is implemented via projection or penalties. Joint optimization of $(z, t)$ is performed using optimizers like Adam or L-BFGS, with backpropagation through the $G_\theta$ ODE solver. $\epsilon$ is set to a very small value (e.g., $10^{-2}$) to keep the shell thin.

## Key Experimental Results

### Main Results
The authors conducted a comprehensive comparison of foundation FMs, domain FMs, and untrained priors on AFHQ-Cat (256×256) Gaussian deblurring:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | CLIPIQA↑ |
|---|---|---|---|---|
| DIP (Untrained) | 27.59 | 0.718 | 0.390 | 0.240 |
| D-Flow (Domain FM) | 28.14 | 0.763 | 0.278 | 0.587 |
| D-Flow (Foundation FM) | 25.01 | 0.708 | 0.534 | 0.361 |
| D-Flow (Found. FM-S, Old Enhancement) | 25.15 | 0.683 | 0.521 | 0.323 |
| FlowDPS (Foundation FM) | 22.14 | 0.593 | 0.541 | 0.291 |

It is evident that D-Flow with foundation FM drops 3 PSNR points compared to domain FM, and old enhancement methods are nearly useless. DIP, a "zero-data" prior, actually outperforms the foundation FM baseline.

Image regression tests on DIV2K (directly representing a known image using FM to check $G_\theta$'s coverage):

| Metric | D-Flow | FMPlug |
|---|---|---|
| PSNR | 36.19 | 37.92 |
| LPIPS | 0.181 | 0.093 |

FMPlug nearly halves the LPIPS, indicating it successfully pushes the foundation FM latent variables back to a submanifold capable of accurate image reconstruction. FMPlug restores a reasonable ranking where "foundation FM > untrained prior $\approx$ domain FM" across various simple-distortion tasks and scientific IPs.

### Ablation Study

| Configuration | Observation |
|---|---|
| Full FMPlug | Enabling both warm-start and sharp shell constraint yields the best results. |
| w/o learnable $t$ (fixed $t = 0$) | Degenerates to D-Flow style warm-start; performance drops to near the foundation FM baseline. |
| w/o sharp shell constraint | $z$ norm drifts out of the training shell, severely degrading generation quality; confirms CoM argument. |
| Replacing shell constraint with D-Flow's $h(z_0)$ | Almost no improvement (soft regularization equals no regularization). |
| Different $\epsilon$ | Smaller $\epsilon$ behaves more like an ideal Gaussian shell, but too small makes optimization difficult; an empirical range is provided. |

### Key Findings
- The fundamental difficulty of using foundation FMs as IP priors is that **"the training distribution is a thin shell, but the plug-in optimization moves outside the shell."** The solution is not to change the regularization form but to constrain variables directly to the shell.
- Making $t$ learnable is not just a trick but a theoretical necessity—it corresponds to finding the $\alpha_t$ that suppresses the influence of unknown $\epsilon$, adaptively searching for the best "transfer point."
- In the few-shot scientific imaging setup, FMPlug enables foundation FMs to outperform untrained priors, an achievement not reached by previous foundation FM IP works.

## Highlights & Insights
- **Re-examining FM Priors via Concentration of Measure**: The author uses CoM to clearly explain why foundation FMs underperform. This is a elegant and transferable diagnostic perspective; future high-dimensional flow model plug-ins can use the "training shell vs. optimization trajectory" framework.
- **Elevating $t$ to an Optimization Variable**: While time indices are usually seen as ODE solver "ticks," the author discovers $t$ is actually a key degree of freedom connecting warm-start error to the training shell. This transformation of fixed hyperparameters into learnable ones is worth trying in many plug-in scenarios.
- **Hard Constraints > Soft Regularization**: When the target distribution is highly concentrated on a high-dimensional spherical shell, soft negative log-likelihood regularization is useless. Explicit set-indicator constraints must be used. This serves as a warning for all works using "Gaussian/spherical priors."
- Entirely training-free, only modifying the optimization objective and initialization. **Can be applied to any released foundation FM** (SD3, Flux, etc.), offering high practical value.

## Limitations & Future Work
- Validated only on image IPs; effectiveness on video/3D/molecule foundation FMs (e.g., Sora, Cosmos) is unknown, especially since the semantics of $z_t$ in multi-modal conditional FMs may be more complex.
- The sharp shell constraint requires choosing $\epsilon$; no parameter-free automatic version is provided. Projections/penalties also add engineering complexity.
- The few-shot setup relies heavily on the quality of "neighboring samples"; if $y$ is far from all neighbors, the approximation error $\alpha_t \epsilon$ is no longer small.
- Interaction with measurement noise is not discussed—when $y$ is heavily corrupted, it may no longer serve as a "seed close to $x$," requiring extra likelihood modeling for $\ell$.
- $G_\theta$ is treated as a black box; joint fine-tuning of a small portion of FM parameters (e.g., LoRA) might provide further breakthroughs.

## Related Work & Insights
- **vs D-Flow (Ben-Hamu et al. 2024)**: Inherits the plug-in framework but replaces its failed warm-start and Gaussian regularization; essentially a corrected version based on CoM theory.
- **vs FlowDPS / FlowChef (Kim et al. 2025; Patel et al. 2024)**: These are interleaving methods (ODE steps + measurement gradient steps); manifold and measurement feasibility are not guaranteed. FMPlug is a plug-in where output naturally stays on $G_\theta$'s manifold.
- **vs DIP (Ulyanov et al. 2018)**: DIP is generic but lacks domain information; FMPlug combines universal pre-training with measurement adaptation as a strengthened version of DIP.
- **vs Diffusion Plug-ins (DPS, PSLD, etc.)**: The methodology can be mapped to foundation diffusion priors by treating the noise schedule as a special case of FM flow.

## Rating
- Novelty: ⭐⭐⭐⭐ Re-explaining D-Flow's failure through CoM is elegant; the warm-start + shell constraint combination is simple yet powerful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes cross-prior and cross-task comparisons.
- Writing Quality: ⭐⭐⭐⭐ Motivations and theoretical explanations are very clear.
- Value: ⭐⭐⭐⭐ Elevates "foundation FM for IP" from unusable to highly effective and identifies a previously overlooked fundamental challenge.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] LithoGRPO: Fast Inverse Lithography via GRPO Reinforced Flow Matching](lithogrpo_fast_inverse_lithography_via_grpo_reinforced_flow_matching.md)
- [\[ICCV 2025\] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching](../../ICCV2025/image_generation/unsupervised_imaging_inverse_problems_with_diffusion_distribution_matching.md)
- [\[ICCV 2025\] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](../../ICCV2025/image_generation/flowdps_flow-driven_posterior_sampling_for_inverse_problems.md)
- [\[ICML 2026\] Stage-wise Distortion-Perception Traversal in Zero-shot Inverse Problems with Diffusion Models](stage-wise_distortion-perception_traversal_in_zero-shot_inverse_problems_with_di.md)
- [\[ICML 2026\] Zeroth-Order Non-Log-Concave Sampling with Variance Reduction and Applications to Inverse Problems](zeroth-order_non-log-concave_sampling_with_variance_reduction_and_applications_t.md)

</div>

<!-- RELATED:END -->
