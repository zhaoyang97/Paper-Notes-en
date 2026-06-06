---
title: >-
  [Paper Note] Saving Foundation Flow-Matching Priors for Inverse Problems
description: >-
  [ICML 2026][Image Generation][Foundation Flow-Matching Priors] Addressing the phenomenon where foundation flow-matching models like Stable Diffusion / Flux significantly underperform compared to domain-specific or even u…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Foundation Flow-Matching Priors"
  - "Inverse Problems"
  - "warm-start"
  - "Gaussian Regularization"
  - "Plug-and-Play"
date: 2026-05-08
content_hash: a989a89c988a29d5
---

# Saving Foundation Flow-Matching Priors for Inverse Problems

**Conference**: ICML 2026  
**arXiv**: [2511.16520](https://arxiv.org/abs/2511.16520)  
**Code**: https://sun-umn.github.io/xm-plug/ (Project Page)  
**Area**: Diffusion Models / Inverse Problems / Flow Matching  
**Keywords**: Foundation Flow-Matching Priors, Inverse Problems, warm-start, Gaussian Regularization, Plug-and-Play

## TL;DR
Addressing the phenomenon where foundation flow-matching models like Stable Diffusion / Flux significantly underperform compared to domain-specific or even untrained priors in solving inverse problems, the authors propose FMPlug. By utilizing a time-learnable warm-start guided by approximate samples and a sharp Gaussian shell constraint, FMPlug forces the latent variables of the foundation FM back onto the thin shell it truly "understands," thereby significantly restoring its capability as an inverse problem prior.

## Background & Motivation

**Background**: Inverse Problems (IPs) aim to recover an unknown $x$ from measurements $y \approx A(x)$, typically by minimizing $\ell(y, A(x)) + \Omega(x)$. Recently, a dominant approach is to plug Deep Generative Priors (DGP) into $\Omega$, especially diffusion/flow models based on Flow Matching (FM). FM has surpassed traditional diffusion in images, video, and world models, becoming the de facto standard for SOTA generation.

**Limitations of Prior Work**: Existing FM-based methods for solving IPs (e.g., D-Flow, FlowDPS, FlowChef) almost exclusively rely on domain-specific FMs (e.g., face priors trained on FFHQ). When switching to foundation FMs like Stable Diffusion V3 or Flux, performance degrades severely. The authors found that on AFHQ-Cat Gaussian deblurring, foundation FM priors generally lag behind domain FMs by several points in PSNR/LPIPS/CLIPIQA, failing to even beat an untrained Deep Image Prior (DIP). Results on DIV2K are even more drastic, where recovered images can be worse than the blurred inputs.

**Key Challenge**: Foundation FM models are strong generators but weak priors—their constraint on images is merely that they "look like natural images," lacking domain-specific structural or semantic information. Previous tricks for "strengthening" foundation FM priors (such as D-Flow's $z_0 = \sqrt{\alpha} y_0 + \sqrt{1-\alpha} z$ warm-start or log-likelihood regularization on $\|z_0\|^2$) are largely ineffective. From the perspective of Concentration of Measure (CoM), the authors reveal the root cause: samples from the source distribution of standard FM, $z_0 \sim \mathcal{N}(0, I_d)$, almost entirely concentrate on a ultra-thin shell $S$ of $S^{d-1}(0, \sqrt{d})$. Consequently, the generator $G_\theta$ is only trained on $S$, and its behavior outside this shell is completely undefined. Initializations like D-Flow's, which mix $y_0$ into $z_0$, cause samples to fall into a shell that is nearly disjoint from $S$, pushing $G_\theta$ into "uncharted territory." Similarly, D-Flow's Gaussian likelihood penalty $-(d/2-1)\log\|z_0\|^2 + \|z_0\|^2/2$ changes extremely slowly in regions far from the optimum (e.g., less than a 0.031% relative change in the interval $\|z_0\|^2 \in [62000, 70000]$), failing to push $z$ back to the shell.

**Goal**: Without altering the plug-in framework, find (i) an initialization strategy for foundation FM priors that utilizes problem-related guidance samples while ensuring alignment with the training distribution of $G_\theta$, and (ii) a truly "hard" Gaussian constraint to lock $z$ onto the thin shell around $S^{d-1}(0, \sqrt{d})$.

**Key Insight**: The authors reduce the problem to "why D-Flow's warm-start/regularization fails" and explain it using the Gaussian Concentration of Measure theorem (Vershynin 2018)—the pretrained FM generator only works correctly when $z$ is almost strictly on the sphere of radius $\sqrt{d}$. Furthermore, when $x$ and $y$ are close, the intermediate state of FM $z_t = \alpha_t x + \beta_t z$ can be approximated as $z_t \approx \alpha_t y + \beta_t z$, where the introduced error is controllable given an appropriate choice of $\alpha_t$.

**Core Idea**: Treat $t$ as a learnable parameter (finding the optimal "shortcut" starting point along the timeline) and explicitly constrain $z$ to a sharp spherical shell $\{z: \|z\|_2 \in [1-\epsilon, 1+\epsilon]\sqrt{d}\}$. This transforms a wandering plug-in into one that operates on the manifold familiar to the generator.

## Method

### Overall Architecture
FMPlug follows the plug-in framework $\min_z \ell(y, A \circ G_\theta(z)) + \Omega \circ G_\theta(z)$, where $G_\theta$ is a fixed pretrained foundation FM model (specifically Stable Diffusion V3 or Flux). The pipeline takes measurements $y$ (and potentially a set of guidance samples) as input and outputs the recovered image $\hat{x} = G_\theta(\alpha_{t^*} y + \beta_{t^*} z^*, t^*)$. The optimization variables are $(z, t)$: $z$ is the intermediate state of the FM, and $t \in [0, 1]$ is its corresponding time point. Both are jointly optimized to minimize the data term $\ell$ while forcing $z$ onto the $\sqrt{d}$ shell. A simple-distortion mode is used when only $y$ is available, while a few-shot mode (using mixture distribution constraints on $z$) is used when a small set of guidance images $\{x_i\}$ is provided.

### Key Designs

1. **Instance-Guided Time-Learnable Warm-Start**:
    - **Function**: Replaces purely random initialization with $y$ (or approximate samples) and places it at an "appropriate time point" in the flow, ensuring the starting point falls within the thin shell seen during $G_\theta$ training.
    - **Mechanism**: Standard FM flow is $z_t = \alpha_t x + \beta_t z$ with $z \sim \mathcal{N}(0, I)$. When $x = y + \epsilon$ and $\|\epsilon\|$ is small, it can be approximated as $z_t \approx \alpha_t y + \beta_t z$ with an approximation error of $\alpha_t \epsilon$. While the true magnitude of $\epsilon$ is unknown, optimizing $t$ (and thus $\alpha_t$) alongside the objective allows the error to be adaptively suppressed to a negligible level. The optimization problem becomes $\min_{z, t \in [0, 1]} \ell(y, A \circ G_\theta(\alpha_t y + \beta_t z, t))$.
    - **Design Motivation**: Solves the issue where D-Flow initialization pushes samples out of the training shell. By allowing $t > 0$ instead of starting from $t = 0$, the path is shorter and the starting point more closely matches the distribution $G_\theta$ is familiar with, while also accelerating convergence by skipping part of the ODE integration.

2. **Sharp Gaussian Shell Constraint**:
    - **Function**: Replaces the weak negative log-likelihood regularization of D-Flow with a "hard" constraint, forcing $z$ to lie strictly on a thin shell with radius $\sqrt{d}$.
    - **Mechanism**: According to the Gaussian CoM theorem, the norm of a $d$-dimensional standard Gaussian vector lies within $[(1-\epsilon)\sqrt{d}, (1+\epsilon)\sqrt{d}]$ with extremely high probability. Thus, the shell set is defined as $S^{d-1}_\epsilon(0, \sqrt{d}) = \{z: \|z\|_2 \in [1-\epsilon, 1+\epsilon]\sqrt{d}\}$. This is added as a constraint to the optimization: $\min_{z, t} \ell(y, A \circ G_\theta(\alpha_t y + \beta_t z, t)) \;\text{s.t.}\; z \in S^{d-1}_\epsilon(0, \sqrt{d})$. Implementation-wise, this is equivalent to adding a set-indicator regularization term and solving via projection or penalty methods.
    - **Design Motivation**: Since $h(z_0)$ in D-Flow changes extremely slowly far from $\sqrt{d-2}$ and fails to move samples, the explicit spherical shell ensures that any boundary violation is immediately corrected, forcing $z$ to behave like a standard Gaussian.

3. **Few-Shot Guidance Extension (for Scientific IPs)**:
    - **Function**: Handles scenarios where the gap between $y$ and $x$ is large (making $y$ unsuitable as a direct warm-start seed) but a few "neighboring samples" $\{x_i\}$ are available, such as in data-scarce scientific imaging (microscopy/astronomy).
    - **Mechanism**: The measurement $y$ in the warm-start is replaced with guidance samples, such as using $\bar{x} = \frac{1}{n}\sum x_i$ or random sampling to construct $z_t \approx \alpha_t \bar{x} + \beta_t z$. The sharp shell constraint remains, while the data term $\ell(y, A \circ G_\theta(\cdot))$ still constrains the generated result to fit the actual measurements. This utilizes domain structures from few neighbor samples without letting them override the measurement signal.
    - **Design Motivation**: Scientific applications often lack sufficient data to train domain FMs and lack signals where the simple-distortion assumption $y \approx x$ holds. Few-shot guidance serves as a compromise—even a few similar images can pull the foundation FM toward the correct sub-manifold.

### Loss & Training
No network parameters are trained. The objective function is $\min_{z, t} \ell(y, A \circ G_\theta(\alpha_t y + \beta_t z, t))$, where $\ell$ uses L2 or perceptual loss. The constraint $z \in S^{d-1}_\epsilon(0, \sqrt{d})$ is implemented via projection or penalties. $(z, t)$ are jointly optimized using Adam or L-BFGS, with backpropagation through the ODE solver of $G_\theta$ at each step. $\epsilon$ is set to a very small value (e.g., $10^{-2}$) to keep the shell as thin as possible.

## Key Experimental Results

### Main Results
The authors conducted a comprehensive comparison between foundation FMs, domain FMs, and untrained priors on AFHQ-Cat ($256 \times 256$) Gaussian deblurring:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | CLIPIQA↑ |
|---|---|---|---|---|
| DIP (Untrained) | 27.59 | 0.718 | 0.390 | 0.240 |
| D-Flow (Domain FM) | 28.14 | 0.763 | 0.278 | 0.587 |
| D-Flow (Foundation FM) | 25.01 | 0.708 | 0.534 | 0.361 |
| D-Flow (Foundation FM-S, Old enhancement) | 25.15 | 0.683 | 0.521 | 0.323 |
| FlowDPS (Foundation FM) | 22.14 | 0.593 | 0.541 | 0.291 |

It can be observed that D-Flow using foundation FMs drops 3 PSNR points compared to the domain FM version. Older enhancement methods are largely ineffective. The "zero-data" prior DIP even outperforms the foundation FM baseline.

In "image regression tests" on DIV2K (representing a known image with FM to test $G_\theta$'s coverage):

| Metric | D-Flow | FMPlug |
|---|---|---|
| PSNR | 36.19 | 37.92 |
| LPIPS | 0.181 | 0.093 |

FMPlug nearly halves the LPIPS, indicating it successfully pushes the latent variables of the foundation FM back to the sub-manifold capable of accurate image reconstruction. The paper reports that FMPlug restores the expected ranking—foundation FM superior to untrained priors and close to domain FMs—across multiple simple-distortion tasks (deblurring, super-resolution, inpainting) and scientific IPs.

### Ablation Study

| Configuration | Observation |
|---|---|
| Full FMPlug | Best performance with both warm-start and sharp shell constraint enabled. |
| w/o learnable $t$ (fixed $t = 0$) | Degrades to D-Flow style warm-start; performance drops to near foundation FM baseline. |
| w/o sharp shell constraint | $z$-norm drifts out of the training shell; generation quality significantly deteriorates, validating the CoM argument. |
| soft $h(z_0)$ reg instead of shell constraint | Negligible improvement (verifying "soft regularization equals no regularization" as shown in Figure 4). |
| Different $\epsilon$ | Smaller $\epsilon$ behaves more like an ideal Gaussian shell, but values that are too small make optimization difficult. |

### Key Findings
- The fundamental difficulty in using foundation FMs as IP priors is that while the **training distribution is a thin shell, the plug-in optimization moves out of this shell**. The solution is forcing variables back onto the shell rather than changing the form of regularization.
- Making $t$ learnable is not just a trick but a theoretical necessity—it determines the $\alpha_t$ required to suppress the influence of the unknown $\epsilon$, effectively searching for the optimal entry point along the flow.
- Under the few-shot scientific imaging setup, FMPlug allows foundation FMs to outperform untrained priors, an achievement not seen in previous works utilizing foundation FMs for IPs.

## Highlights & Insights
- **Revisiting FM Priors via Measure Concentration**: The authors use CoM to fully explain "why foundation FMs are difficult to use." This is a elegant and transferable diagnostic perspective that can be used to analyze any high-dimensional flow model plug-in method via the "training shell vs. optimization trajectory" framework.
- **Promoting $t$ to an Optimization Variable**: While time indices in diffusion/FM are usually treated as "scales" for ODE solvers, the authors identified it as a critical degree of freedom connecting warm-start error to the training distribution shell. This conversion of fixed hyperparameters to learnable ones is worth trying in many plug-in scenarios.
- **Hard Constraints > Soft Regularization**: In high-dimensional spaces where the target distribution is highly concentrated on a spherical shell, soft negative log-likelihood regularization is ineffective. Explicit set-indicator constraints must be used, which serves as a warning for all works using Gaussian/spherical priors.
- **Training-free and Versatile**: By only modifying the optimization objective and initialization, FMPlug can be applied to any released foundation FM (SD3, Flux, etc.), offering high practical value.

## Limitations & Future Work
- The method is only validated on image IPs; its efficacy on other foundation FMs (e.g., Sora, Cosmos for video/3D) is unknown, especially where $z_t$ semantics in multimodal conditional FMs may be more complex.
- The sharp shell constraint requires choosing $\epsilon$, and no parameter-free automated version is provided. Adding projection/penalties to optimization increases engineering complexity and may require task-specific tuning.
- The few-shot setup is highly dependent on the quality of "neighboring samples." If $y$ is far from all neighbor samples, the approximation error $\alpha_t \epsilon$ in warm-start is no longer small, and theoretical guarantees weaken.
- Interaction with measurement noise is not discussed. When $y$ is heavily corrupted by noise, it no longer serves as a "seed close to $x$," potentially requiring additional likelihood modeling for $\ell$.
- $G_\theta$ is treated as a black box; jointly fine-tuning a small portion of FM parameters (e.g., LoRA) might provide further breakthroughs.

## Related Work & Insights
- **vs. D-Flow (Ben-Hamu et al. 2024)**: Inherits the plug-in framework but replaces its failed warm-start and Gaussian regularization. Essentially a corrected version of D-Flow backed by measure concentration theory.
- **vs. FlowDPS / FlowChef (Kim et al. 2025; Patel et al. 2024)**: These are interleaving methods (ODE steps + measurement gradient steps) with no guarantees for manifold or measurement feasibility. FMPlug uses a plug-in approach where the output naturally resides on $G_\theta$'s manifold.
- **vs. DIP (Ulyanov et al. 2018) / Implicit Neural Representations**: DIP is zero-data but lacks domain information. FMPlug combines "general pretraining + measurement adaptation," acting as a prioritized version of DIP.
- **vs. Diffusion Plug-ins (DPS, PSLD, etc.)**: The methodology can be translated to foundation diffusion priors by viewing noise schedules as a special case of FM flow; shell constraints and learnable time would remain applicable.

## Rating
- Novelty: ⭐⭐⭐⭐ Re-explaining D-Flow's failure via measure concentration is elegant; the combination of warm-start and shell constraints is simple and effective, though individual technical components are not entirely revolutionary.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes horizontal comparisons across priors (Foundation/Domain/Untrained) and coverage across tasks (Deblurring, SR, Scientific IP). More depth on robustness to extreme noise and different FM models would be beneficial.
- Writing Quality: ⭐⭐⭐⭐ Motivation and theoretical explanations are very clear. The CoM illustrations and the D-Flow regularization plateau plots are highlights.
- Value: ⭐⭐⭐⭐ Successfully bridges the gap to make "IP solving via foundation FMs" viable and identifies a previously overlooked fundamental challenge, providing insights for anyone using foundation generative models for downstream tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance](weak_diffusion_priors_can_still_achieve_strong_inverse-problem_performance.md)
- [\[ICML 2026\] LithoGRPO: Fast Inverse Lithography via GRPO Reinforced Flow Matching](lithogrpo_fast_inverse_lithography_via_grpo_reinforced_flow_matching.md)
- [\[ICCV 2025\] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching](../../ICCV2025/image_generation/unsupervised_imaging_inverse_problems_with_diffusion_distribution_matching.md)
- [\[ICCV 2025\] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](../../ICCV2025/image_generation/flowdps_flow-driven_posterior_sampling_for_inverse_problems.md)
- [\[ICML 2026\] Stage-wise Distortion-Perception Traversal in Zero-shot Inverse Problems with Diffusion Models](stage-wise_distortion-perception_traversal_in_zero-shot_inverse_problems_with_di.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] LithoGRPO: Fast Inverse Lithography via GRPO Reinforced Flow Matching](lithogrpo_fast_inverse_lithography_via_grpo_reinforced_flow_matching.md)
- [\[ICML 2026\] Weak Diffusion Priors Can Still Achieve Strong Inverse-Problem Performance](weak_diffusion_priors_can_still_achieve_strong_inverse-problem_performance.md)
- [\[ICCV 2025\] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching](../../ICCV2025/image_generation/unsupervised_imaging_inverse_problems_with_diffusion_distribution_matching.md)
- [\[ICCV 2025\] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](../../ICCV2025/image_generation/flowdps_flow-driven_posterior_sampling_for_inverse_problems.md)
- [\[ICML 2026\] Stage-wise Distortion-Perception Traversal in Zero-shot Inverse Problems with Diffusion Models](stage-wise_distortion-perception_traversal_in_zero-shot_inverse_problems_with_di.md)

</div>

<!-- RELATED:END -->
