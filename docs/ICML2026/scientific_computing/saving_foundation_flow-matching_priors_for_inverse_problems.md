---
title: >-
  [Paper Note] Saving Foundation Flow-Matching Priors for Inverse Problems
description: >-
  [ICML 2026][Scientific Computing][Foundation Flow-Matching Prior] To address the phenomenon where foundation flow-matching models such as Stable Diffusion / Flux perform significantly worse than domain-specific or even u…
tags:
  - "ICML 2026"
  - "Scientific Computing"
  - "Foundation Flow-Matching Prior"
  - "Inverse Problems"
  - "Warm-Start"
  - "Gaussian Regularization"
  - "Plug-and-Play"
date: 2026-05-08
content_hash: 36afaadbd43ca84b
---

# Saving Foundation Flow-Matching Priors for Inverse Problems

**Conference**: ICML 2026  
**arXiv**: [2511.16520](https://arxiv.org/abs/2511.16520)  
**Code**: https://sun-umn.github.io/xm-plug/ (project page)  
**Area**: Diffusion Models / Inverse Problems / Flow Matching  
**Keywords**: Foundation Flow-Matching Prior, Inverse Problems, Warm-Start, Gaussian Regularization, Plug-and-Play

## TL;DR
To address the phenomenon where foundation flow-matching models such as Stable Diffusion / Flux perform significantly worse than domain-specific or even untrained priors on inverse problems, the authors propose FMPlug: a method that uses a sample-guided, time-learnable warm-start combined with a sharp Gaussian shell constraint to force the latent variables of the foundation FM back onto the thin shell where it was actually "trained," thereby significantly restoring its effectiveness as a prior for inverse problems.

## Background & Motivation

**Background**: Inverse Problems (IPs) aim to recover unknown $x$ from measurements $y \approx A(x)$, typically by minimizing $\ell(y, A(x)) + \Omega(x)$. Recently, the mainstream approach is to plug deep generative priors (DGP) into $\Omega$, especially flow-matching (FM) based diffusion/flow models—FM has already replaced traditional diffusion in images, videos, and world models, becoming the de facto SOTA generative standard.

**Limitations of Prior Work**: Existing FM-based methods for inverse problems (D-Flow, FlowDPS, FlowChef, etc.) almost exclusively rely on domain-specific FMs (e.g., face priors trained on FFHQ). When switching to foundation FMs like Stable Diffusion V3 or Flux, performance degrades severely—the authors find on AFHQ-Cat Gaussian deblurring that foundation FM priors lag domain FMs by several points in PSNR/LPIPS/CLIPIQA, and even lose to untrained DIP (Deep Image Prior). On DIV2k, the recovered images are even worse than the blurry inputs.

**Key Challenge**: Foundation FM models are strong as "generators" but weak as "priors"—their constraint on images is merely "looks natural," lacking domain-specific structure/semantic information. Previous tricks to "strengthen" foundation FM priors (D-Flow's $z_0 = \sqrt{\alpha} y_0 + \sqrt{1-\alpha} z$ warm-start, log-likelihood regularization on $\|z_0\|^2$) are largely ineffective. The authors, from the perspective of concentration of measure (CoM), reveal the root cause: the source distribution $z_0 \sim \mathcal{N}(0, I_d)$ of standard FM is almost entirely concentrated on a thin shell $S^{d-1}(0, \sqrt{d})$, so the generator $G_\theta$ is only trained on $S$; its behavior outside the shell is undefined. D-Flow's initialization, which mixes $y_0$ into $z_0$, places samples on a shell almost disjoint from $S$, effectively pushing $G_\theta$ "into the wild." Similarly, D-Flow's $-(d/2-1)\log\|z_0\|^2 + \|z_0\|^2/2$ Gaussian likelihood penalty changes extremely slowly far from the optimum (e.g., for $\|z_0\|^2 \in [62000, 70000]$, the relative change from the minimum is less than 0.031%), and cannot force $z$ back onto the thin shell.

**Goal**: Without changing the plug-in framework, find for foundation FM priors (i) an initialization strategy that leverages problem-relevant guided samples while ensuring they lie on the training distribution of $G_\theta$; (ii) a truly "hard" Gaussian constraint that locks $z$ onto the thin shell around $S^{d-1}(0, \sqrt{d})$.

**Key Insight**: The authors reduce the problem to "why D-Flow's warm-start/regularization fails," and explain via the Gaussian concentration of measure theorem (Vershynin 2018)—only when $z$ is almost strictly on the sphere of radius $\sqrt{d}$ can the pretrained FM generator function properly. Additionally, when $x, y$ are close, the FM intermediate state $z_t = \alpha_t x + \beta_t z$ can be approximated as $z_t \approx \alpha_t y + \beta_t z$, with controllable error as long as $\alpha_t$ is chosen appropriately.

**Core Idea**: Make $t$ a learnable variable (searching along the time axis for the best "shortcut" starting point), and explicitly constrain $z$ to the sharp spherical shell $\{z: \|z\|_2 \in [1-\epsilon, 1+\epsilon]\sqrt{d}\}$, turning the "blind plug-in" into a plug-in that operates on the familiar manifold of the generator.

## Method

### Overall Architecture
FMPlug follows the plug-in framework $\min_z \ell(y, A \circ G_\theta(z)) + \Omega \circ G_\theta(z)$, where $G_\theta$ is a fixed, pretrained foundation FM model (specifically Stable Diffusion V3 or Flux). The pipeline takes measurement $y$ (and possibly a set of guided samples) as input and outputs the recovered image $\hat{x} = G_\theta(\alpha_{t^*} y + \beta_{t^*} z^*, t^*)$. The optimization variables are $(z, t)$: $z$ is the FM intermediate state, $t \in [0, 1]$ is its corresponding time; both jointly minimize the data term $\ell$, with $z$ forced onto the $\sqrt{d}$ shell. When only $y$ is available, the simple-distortion mode is used; with a few guided images $\{x_i\}$, the few-shot mode (with mixture constraints on $z$) is used.

### Key Designs

1. **Instance-Guided, Time-Learnable Warm-Start**:

    - **Function**: Uses $y$ (or approximate samples) instead of pure random initialization, placing it at the "appropriate time point" in the flow so the starting point lies within the thin shell seen during $G_\theta$'s training.
    - **Mechanism**: Standard FM flow is $z_t = \alpha_t x + \beta_t z$, $z \sim \mathcal{N}(0, I)$. When $x = y + \epsilon$ and $\|\epsilon\|$ is small, $z_t \approx \alpha_t y + \beta_t z$, with approximation error $\alpha_t \epsilon$. The true size of $\epsilon$ is unknown, but by making $t$ (and thus $\alpha_t$) learnable, the optimizer can adaptively suppress the error. The optimization becomes $\min_{z, t \in [0, 1]} \ell(y, A \circ G_\theta(\alpha_t y + \beta_t z, t))$.
    - **Design Motivation**: Addresses the issue in D-Flow initialization where samples are pushed outside the training shell—by starting from $t > 0$ instead of $t = 0$, the path is shorter and the starting point is closer to the distribution $G_\theta$ is familiar with; also saves an ODE integration segment, accelerating convergence.

2. **Sharp Gaussian Shell Constraint**:

    - **Function**: Replaces D-Flow's weak negative log-likelihood regularization with a "hard" constraint, forcing $z$ strictly onto the thin shell of radius $\sqrt{d}$.
    - **Mechanism**: By the Gaussian concentration of measure theorem, the norm of a $d$-dimensional standard Gaussian vector is with high probability in $[(1-\epsilon)\sqrt{d}, (1+\epsilon)\sqrt{d}]$. Define the shell set $S^{d-1}_\epsilon(0, \sqrt{d}) = \{z: \|z\|_2 \in [1-\epsilon, 1+\epsilon]\sqrt{d}\}$, and add it as a constraint: $\min_{z, t} \ell(y, A \circ G_\theta(\alpha_t y + \beta_t z, t)) \;\text{s.t.}\; z \in S^{d-1}_\epsilon(0, \sqrt{d})$. In practice, this is equivalent to adding a set-indicator regularization term to the objective, solved via projection or penalty methods.
    - **Design Motivation**: D-Flow's $h(z_0)$ changes extremely slowly far from $\sqrt{d-2}$ and cannot pull samples back; with an explicit shell, any out-of-bounds sample is immediately projected back, truly enforcing "z looks like standard Gaussian."

3. **Few-Shot Guided Extension (for Scientific Inverse Problems)**:

    - **Function**: Handles scenarios where $y$ and $x$ are far apart (so $y$ cannot be used directly as a warm-start seed), but a few "neighboring samples" $\{x_i\}$ are available, as in data-scarce scientific imaging (microscopy/astronomy).
    - **Mechanism**: Replace $y$ in the warm-start with a guided sample, e.g., use $\bar{x} = \frac{1}{n}\sum x_i$ or random sampling, constructing $z_t \approx \alpha_t \bar{x} + \beta_t z$; keep the sharp shell constraint; the data term $\ell(y, A \circ G_\theta(\cdot))$ still enforces fit to the measurement. This leverages the domain structure of a few neighbor samples without letting them override the measurement signal.
    - **Design Motivation**: In scientific applications, there is insufficient data to train a domain FM, and the simple-distortion assumption $y \approx x$ fails; few-shot guidance is a compromise—just a few similar images can pull the foundation FM onto the correct submanifold.

### Loss & Training

No network parameters are trained. The objective is $\min_{z, t} \ell(y, A \circ G_\theta(\alpha_t y + \beta_t z, t))$, with $\ell$ as L2 or perceptual loss; the constraint $z \in S^{d-1}_\epsilon(0, \sqrt{d})$ is enforced via projection or penalty; Adam/L-BFGS optimizers jointly optimize $(z, t)$, with each step backpropagating through $G_\theta$'s ODE solver. $\epsilon$ is set very small (e.g., $10^{-2}$) to keep the shell as thin as possible.

## Key Experimental Results

### Main Results
The authors conduct comprehensive comparisons of foundation FM, domain FM, and untrained priors on AFHQ-Cat (256×256) Gaussian deblurring:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | CLIPIQA↑ |
|---|---|---|---|---|
| DIP (Untrained) | 27.59 | 0.718 | 0.390 | 0.240 |
| D-Flow (Domain FM) | 28.14 | 0.763 | 0.278 | 0.587 |
| D-Flow (Foundation FM) | 25.01 | 0.708 | 0.534 | 0.361 |
| D-Flow (Foundation FM-S, old enhancement) | 25.15 | 0.683 | 0.521 | 0.323 |
| FlowDPS (Foundation FM) | 22.14 | 0.593 | 0.541 | 0.291 |

D-Flow with foundation FM drops 3 PSNR points compared to domain FM, and old enhancement methods are almost useless; DIP, a "zero-data" prior, even outperforms foundation FM.

On DIV2K, the "image regression test" (directly representing a known image with FM, i.e., testing $G_\theta$'s coverage):

| Metric | D-Flow | FMPlug |
|---|---|---|
| PSNR | 36.19 | 37.92 |
| LPIPS | 0.181 | 0.093 |

FMPlug nearly halves LPIPS, indicating it indeed forces the foundation FM's latent variables back onto the submanifold capable of precise image reconstruction. The paper reports that FMPlug restores the reasonable ranking of "foundation FM > untrained prior, close to domain FM" across various simple-distortion tasks (deblurring, super-resolution, inpainting) and scientific inverse problems.

### Ablation Study

| Configuration | Phenomenon |
|---|---|
| Full FMPlug | Both warm-start and sharp shell constraint enabled, best performance |
| w/o learnable $t$ (fixed $t = 0$) | Degrades to D-Flow style warm-start, performance drops to near foundation FM baseline |
| w/o sharp shell constraint | $z$ norm drifts outside training shell, generation quality degrades sharply, validating CoM argument |
| Replace shell constraint with D-Flow's $h(z_0)$ soft regularization | Almost no improvement (as repeatedly shown in Fig. 4: "soft regularization is as good as none") |
| Different $\epsilon$ | Smaller $\epsilon$ approaches ideal Gaussian shell, but too small makes optimization difficult; empirical range provided in the paper |

### Key Findings
- The fundamental difficulty of using foundation FM as an inverse problem prior is **"the training distribution is a thin shell, but plug-in optimization runs outside the shell"**; the solution is not to change the regularization form, but to directly constrain variables onto the shell.
- Making $t$ learnable is not just a trick but a theoretical necessity—it corresponds to "choosing $\alpha_t$ large enough to suppress the unknown $\epsilon$," i.e., adaptively finding the best "switching point" along the flow.
- In few-shot scientific imaging, FMPlug enables foundation FM to outperform untrained priors, which no previous foundation FM-based IP method has achieved.

## Highlights & Insights
- **Re-examining FM priors from the concentration of measure perspective**: The authors thoroughly explain "why foundation FM is hard to use" via CoM—a beautiful and transferable diagnostic angle; future high-dimensional flow model plug-in methods can analyze with the same "training shell vs optimization trajectory" framework.
- **Promoting $t$ to an optimization variable**: While diffusion/FM time indices are usually seen as ODE solver "ticks," the authors show it is the key degree of freedom connecting warm-start error and the training shell. Turning a fixed hyperparameter into a learnable variable is worth trying in many plug-in scenarios.
- **Hard constraint > soft regularization**: When the target distribution is highly concentrated on a high-dimensional shell, soft negative log-likelihood regularization is ineffective; explicit set-indicator constraints are necessary. This is a warning for all works using Gaussian/spherical priors.
- Completely training-free, only modifying the optimization objective and initialization, **applicable to any released foundation FM** (Stable Diffusion 3, Flux, future larger models), with high practical value.

## Limitations & Future Work
- The method is only validated on image IPs; its effectiveness on video/3D/molecular foundation FMs (e.g., Sora, Cosmos) is unknown, especially as $z_t$ semantics may be more complex in multimodal conditional FMs.
- The sharp shell constraint requires choosing $\epsilon$; the paper does not provide a parameter-free automatic version. Adding projection/penalty in optimization increases engineering complexity and may require task-specific tuning.
- The few-shot setting heavily depends on the quality of "neighboring samples"; when $y$ is far from all neighbors, the warm-start approximation error $\alpha_t \epsilon$ is no longer small, and theoretical guarantees break down.
- The interaction with measurement noise is not discussed—when $y$ is heavily noise-corrupted, it is no longer a "seed close to $x$," and additional likelihood modeling in $\ell$ may be needed.
- The authors treat $G_\theta$ as a black box; jointly fine-tuning a small part of FM parameters (like LoRA) may further improve performance.

## Related Work & Insights
- **vs D-Flow (Ben-Hamu et al. 2024)**: This work inherits the plug-in framework but replaces its ineffective warm-start and Gaussian regularization; essentially a "concentration of measure"-theoretic revision of D-Flow.
- **vs FlowDPS / FlowChef (Kim et al. 2025; Patel et al. 2024)**: Those are interleaving methods (ODE step + measurement gradient step), with no guarantee of manifold or measurement feasibility; FMPlug uses plug-in, with outputs naturally on $G_\theta$'s manifold.
- **vs DIP (Ulyanov et al. 2018) / Implicit Neural Representations**: DIP does not rely on training data, excelling in generality but lacking domain information; FMPlug uses foundation FM to combine "universal pretraining + measurement adaptation," essentially a strengthened DIP.
- **vs Diffusion plug-in (DPS, PSLD, etc.)**: The methodology can be directly transferred to foundation diffusion priors—treating the noise schedule as a special case of FM flow, FMPlug's shell constraint + learnable time are equally applicable.

## Rating
- Novelty: ⭐⭐⭐⭐ The reinterpretation of "why D-Flow fails" via concentration of measure is elegant, and the combination of warm-start + shell constraint is simple and effective; though each technical point alone is not disruptive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes both cross-prior (foundation/domain/untrained) and cross-task (deblurring, super-resolution, scientific IP) comparisons; but lacks depth on extreme measurement noise/robustness to different FM models.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation and theoretical explanation are very clear; CoM illustrations and D-Flow soft regularization platform plots are highlights.
- Value: ⭐⭐⭐⭐ Directly makes "using foundation FM for IP" feasible, and identifies a previously overlooked fundamental challenge, providing valuable insights for anyone seeking to use foundation generative models for downstream tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Unsupervised Imaging Inverse Problems with Diffusion Distribution Matching](../../ICCV2025/image_generation/unsupervised_imaging_inverse_problems_with_diffusion_distribution_matching.md)
- [\[ICCV 2025\] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](../../ICCV2025/image_generation/flowdps_flow-driven_posterior_sampling_for_inverse_problems.md)
- [\[ICML 2026\] Exploring and Exploiting Stability in Latent Flow Matching](exploring_and_exploiting_stability_in_latent_flow_matching.md)
- [\[ICML 2026\] The Coupling Within: Flow Matching via Distilled Normalizing Flows](the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)
- [\[NeurIPS 2025\] A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](../../NeurIPS2025/image_generation/a_gradient_flow_approach_to_solving_inverse_problems_with_latent_diffusion_model.md)

</div>

<!-- RELATED:END -->
