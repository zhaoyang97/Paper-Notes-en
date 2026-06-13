---
title: >-
  [Paper Note] Initialization is Half the Battle: Generating Diverse Images from a Guidance Potential Posterior
description: >-
  [ICML 2026][Image Generation][Diversity enhancement] This paper treats "initial noise" as a random variable that can be sampled from a posterior defined by a conditional guidance potential. It proposes DivIn: using a sin…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Diversity enhancement"
  - "Initial noise"
  - "Langevin dynamics"
  - "Mode collapse"
  - "Guidance potential"
date: 2026-05-08
content_hash: 69370dd4257b9e41
---

# Initialization is Half the Battle: Generating Diverse Images from a Guidance Potential Posterior

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2606.02453](https://arxiv.org/abs/2606.02453)  
**Code**: https://github.com/South7X/divin (Available)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Diversity enhancement, Initial noise, Langevin dynamics, Mode collapse, Guidance potential

## TL;DR
This paper treats "initial noise" as a random variable that can be sampled from a posterior defined by a conditional guidance potential. It proposes DivIn: using a single step of Langevin dynamics to push standard Gaussian noise toward "low-potential, flat" regions. This significantly mitigates mode collapse in diffusion/flow matching models with almost no additional inference overhead and is orthogonally compatible with existing trajectory-based diversity methods.

## Background & Motivation

**Background**: Modern diffusion and flow matching models have achieved high fidelity in text-to-image and class-conditional generation, but they suffer from severe mode collapse—changing the random seed often results in highly similar images, and in extreme cases, they replicate training samples. The mainstream path for enhancing diversity involves modifying the "generation trajectory," such as adjusting CFG scales (CADS, Interval Guidance) or introducing mutual repulsion between trajectories (Particle Guidance).

**Limitations of Prior Work**: All these methods assume the starting point—isotropic Gaussian noise $\mathcal{N}(0, \mathbf{I})$—is sufficient and only intervene thereafter. However, the authors identify an overlooked fact: standard Gaussian noise is completely unaware of the conditional guidance "landscape" and often falls near "steep peaks" of high guidance potential. Consequently, random trajectories from different seeds are attracted to the same strong mode, resulting in nearly identical outputs.

**Key Challenge**: Existing "seed optimization" methods (most representatively SAIL) take the opposite extreme—using deterministic Hessian/sharpness optimization to select a "best seed." This results in the collapse of the initial distribution into a single point, destroying the original Gaussian prior and reducing the overall diversity of the images. Furthermore, high-frequency artifacts appear when the latent norm deviates significantly from $\sqrt{d}$.

**Goal**: Reformulate "initial noise selection" as "sampling from a posterior that preserves the prior while biasing towards diverse regions," utilizing the guidance potential landscape without destroying the volume of the initial distribution.

**Key Insight**: Regions with higher conditional guidance potential correspond to higher probability volume contraction rates in the reverse process (Theorem A.3), making trajectories more likely to be sucked into a single mode. Empirical tests on 1,000 prompts show a Spearman correlation of $-0.4$ ($p < 0.001$) between potential and Vendi score. Thus, actively pushing initial noise toward **low-potential basins** restores diversity, but this must be done in the sense of distribution sampling rather than point estimation.

**Core Idea**: Use a one-step Langevin update to approximate sampling from a "diversity-weighted posterior." The noise is simultaneously subjected to three forces: (1) a diversity force pushing it away from high-potential regions; (2) a prior constraint pulling it back to the Gaussian prior manifold; and (3) a stochastic noise term to prevent entrapment in local minima.

## Method

### Overall Architecture
DivIn adds **only one "initial noise optimization" step** before the standard diffusion/flow matching pipeline, without modifying the subsequent denoising chain:

1. Start from standard Gaussian $\mathbf{x}_T^{(0)} \sim \mathcal{N}(0, \mathbf{I})$.
2. Use $K$ steps (experimentally $K=1$ is sufficient) of Langevin updates to push $\mathbf{x}_T$ toward the guidance potential posterior $p_{\text{diverse}}(\mathbf{x}_T|c)$.
3. Feed the updated $\mathbf{x}_T^{(K)}$ into the original denoiser for standard sampling.

The posterior is defined as $p_{\text{diverse}}(\mathbf{x}_T|c) \propto \exp(-\tau \cdot U(\mathbf{x}_T,c)) \cdot \mathcal{N}(\mathbf{x}_T;0,\mathbf{I})$, where $\tau$ controls the trade-off between "adhering to the prior" and "seeking low potential."

### Key Designs

1.  **Guidance Potential $U(\mathbf{x}_T,c)$ in Tweedie-space**:
    - **Function**: Approximates the pulling strength of conditional guidance on the trajectory at the initial moment using a cheap, stable, and model-agnostic scalar.
    - **Mechanism**: Defined as $U(\mathbf{x}_T,c) = \|\hat{\mathbf{x}}_0(\mathbf{x}_T,c) - \hat{\mathbf{x}}_0(\mathbf{x}_T,\varnothing)\|_2$, the Euclidean distance in $\mathbf{x}_0$ space between conditional and unconditional single-step Tweedie denoising estimates. A larger gap indicates that the condition is pushing harder at $\mathbf{x}_T$, making it more likely to pull different seeds into the same mode.
    - **Design Motivation**: The authors prove (Prop. A.1) that this quantity is a robust proxy for local curvature. By projecting into the $\hat{\mathbf{x}}_0$ pixel/latent space via Tweedie, it absorbs the $\lambda_t$ time-scaling of different samplers, making the optimal $\tau$ insensitive to inference steps (e.g., 30 vs. 50 steps), avoiding the fragility of SAIL's Hessian-score expansion and manual thresholds.

2.  **Langevin One-step Update on the Diversity-weighted Posterior**:
    - **Function**: Samples from $p_{\text{diverse}}$ rather than deterministically optimizing to a "best seed."
    - **Mechanism**: From $\nabla_\mathbf{x}\log p_{\text{diverse}} = -\tau \nabla_\mathbf{x} U(\mathbf{x},c) - \mathbf{x}$, the discrete update rule is $\mathbf{x}_T^{(k+1)} = \mathbf{x}_T^{(k)} - \eta(\tau \nabla U + \mathbf{x}_T^{(k)}) + \sqrt{2\eta}\,\boldsymbol{\xi}^{(k)}$. This dynamically balances three forces: the diversity force $-\tau\nabla U$ pushes the latent away from steep peaks; the prior term $-\mathbf{x}$ pulls it back to the Gaussian manifold where $\|\mathbf{x}\|\approx\sqrt{d}$; and the noise term $\sqrt{2\eta}\boldsymbol{\xi}$ helps escape shallow local minima.
    - **Design Motivation**: This is the fundamental difference from SAIL, which treats seed selection as deterministic optimization, causing latents to fall into the same sharp local minimum (Fig. 5) and collapsing distribution volume. DivIn performs **posterior sampling at the distribution level**, preserving entropy and spreading latents across low-potential basins.

3.  **Minimal Intrusion and Orthogonal Superposition**:
    - **Function**: Ensures DivIn is a truly plug-and-play inference-time module applicable to different paradigms like SD v1.4 (DDPM) and SD v3.5 Medium (Rectified Flow).
    - **Mechanism**: Since $U$ is defined in $\hat{\mathbf{x}}_0$ space, it can be computed via $\hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\epsilon_\theta)/\sqrt{\bar\alpha_t}$ for diffusion or $\hat{\mathbf{x}}_0 = \mathbf{x}_t - t\mathbf{v}_\theta$ for flow matching. At $K=1$, it only adds one conditional and one unconditional forward/backward pass. After correcting the starting point, trajectory methods like PG/CADS/IG can be applied, resulting in additive diversity gains.
    - **Design Motivation**: The authors distinguish "initialization" and "trajectory" as two independent sources of diversity. Since standard methods assume the starting point is already good, correcting it provides "free" orthogonal diversity.

### Loss & Training
DivIn is a **completely training-free** inference-time method that **does not modify model weights**. The "objective function" is the Langevin update rule described above. Key hyperparameters: temperature $\tau$ (default sweep $[0.5, 1.0]$, higher means stronger diversity), step size $\eta$ (e.g., $0.05$, corresponding to noise scale $\sqrt{2\eta}\approx 0.316$), and number of steps $K$ (default $1$; increasing to $3$ offers marginal diversity gains but increases FID).

## Key Experimental Results

### Main Results

**Class-conditional Generation (ImageNet-1K, SD v1.4, 10k images, avg. of 5 seeds)**:

| Method | Recall ↑ | Vendi Score ↑ | Coverage ↑ | FID ↓ |
|------|---------:|--------------:|-----------:|------:|
| Base Model | 0.503 | 4.265 | 0.596 | 16.696 |
| + SAIL | 0.543 | 4.549 | 0.591 | 16.395 |
| **+ DivIn (Ours)** | **0.569** | **4.688** | 0.597 | **16.158** |
| CADS | 0.528 | 4.384 | 0.598 | 16.360 |
| **CADS + DivIn** | **0.553** | **4.548** | 0.602 | 16.336 |
| IG | 0.564 | 4.585 | 0.597 | **15.531** |
| **IG + DivIn** | **0.576** | **4.729** | 0.599 | 15.877 |

**Text-to-Image (500 prompt mix, SD v3.5 Medium / Rectified Flow)**: DivIn alone reduces in-batch Similarity from 0.793 to 0.775 and increases Vendi from 1.803 to 1.864. When combined with CADS, Similarity drops further to 0.761 and Vendi rises to 1.918, while CLIP/Aesthetic scores remain stable or slightly improve. **It is a true orthogonal superposition with trajectory methods, not a simple replacement.**

### Ablation Study

| Configuration | Recall ↑ | Vendi ↑ | Precision ↑ | FID ↓ |
|------|---------:|--------:|------------:|------:|
| Base Model | 0.503 | 4.265 | **0.833** | 16.696 |
| SAIL (Deterministic baseline) | 0.543 | 4.549 | 0.825 | 16.395 |
| DivIn w/o noise (Remove $\sqrt{2\eta}\xi$) | 0.541 | 4.534 | 0.822 | 16.544 |
| DivIn w/o prior (Remove $-\mathbf{x}$) | 0.557 | 4.584 | 0.824 | **16.121** |
| **DivIn (Full)** | **0.569** | **4.688** | 0.825 | 16.158 |

Removing the stochastic term immediately degrades performance to SAIL levels, **indicating that the true source of diversity is the Langevin posterior sampling framework, not the potential $U$ itself**. Removing the prior term results in slightly better short-term FID, but increasing $K$ from $1$ to $3$ causes noise-free FID to jump to 17.53 and prior-free FID to 17.36, while full DivIn remains at 15.98—both components are indispensable.

### Key Findings
- **Manifold preservation is a critical safety net**: SAIL pushes the latent norm down significantly from $\sqrt{d}\approx 128$ during 10 steps of optimization; once it leaves the Gaussian manifold, high-frequency artifacts appear. DivIn's triple-force balance keeps the norm stable near 128, allowing for more Langevin steps without collapse.
- **Zero-overhead, nearly free diversity**: At $K=1$, per-image generation time increases from 0.754s to 0.779s (approx. $+3\%$ wall-clock cost), whereas SAIL is significantly more expensive due to rejection sampling and second-order approximations.
- **Orthogonal performance on the Pareto frontier**: On the diversity-quality Pareto plot, the curves for all baselines + DivIn (solid lines) encapsulate the corresponding baseline curves (dashed lines), pushing the outer frontier **without a quality penalty**.
- **Temperature $\tau$ provides a smooth tunable knob**: Adjusting $\tau$ from $0$ to $1.0$ increases recall from $0.500$ to $0.607$ with only a slight drop in precision. In contrast, SAIL's early-stopping threshold is cliff-like; slightly stricter settings cause FID to skyrocket to 27.11.

## Highlights & Insights
- **Redefining the language of "Seed Selection"**: Transitioning from "finding an optimal seed" (point estimation, deterministic optimization) to "sampling from a diversity-weighted posterior" (distribution, Langevin) is elegant. It preserves the Gaussian prior volume while utilizing potential information through the "prior + energy + noise" decomposition.
- **Tweedie-space potential proxy as a reusable trick**: Using the single-step $\hat{\mathbf{x}}_0$ distance between conditional and unconditional estimates as a proxy for guidance strength avoids second-order Hessian calculations and maintains stable hyperparameters across different schedulers and generation paradigms. This is useful for any task aiming to quantify CFG intensity at specific noise points (e.g., memorization detection).
- **"The prior term serves as distribution volume protection"**: The ablation study shows the prior term is crucial as $K$ increases to prevent norm explosion. This serves as a warning for future seed-optimization work: **diffusion models produce artifacts whenever they stray from the $\|\mathbf{x}\|\approx\sqrt{d}$ shell**.

## Limitations & Future Work
- **Dependence on manual $\tau$ calibration**: $\tau$ needs to be re-scanned for new architectures or specialized domains (e.g., medical imaging, video), which adds deployment cost.
- **Increased variance with multiple Langevin steps**: While $K=1$ captures most gains, multi-step sampling to approach the true posterior introduces higher run-to-run variance and computational cost.
- **Conditional generation requirement**: The potential $U$ is based on the difference between conditional and unconditional estimates, so it does not natively support purely unconditional models. The authors suggest using unconditional score curvature in the future.
- **No explicit constraint on prompt semantic fidelity**: Experiments show ImageReward drops slightly when combining IG/CADS (e.g., IG+DivIn from 0.521 → 0.501), suggesting that pushing toward extreme low-potential regions may sacrifice some prompt precision. Adding a semantic alignment term to the Langevin forces could improve stability.

## Related Work & Insights
- **vs. SAIL (Jeon et al., 2025)**: Both start from "initialization geometry," but SAIL uses **deterministic sharpness optimization** with Hessian-score second-order approximations. This collapses the distribution volume and drifts the latent off the Gaussian manifold. DivIn uses Tweedie-space potential and Langevin sampling, preserving both manifold and diversity, outperforming SAIL in both Vendi and FID (4.688 vs. 4.549; 16.158 vs. 16.395).
- **vs. Particle Guidance / CADS / Interval Guidance**: These methods intervene on the **generation trajectory** (particle repulsion, annealing conditions, CFG filtering) assuming the starting point is sufficiently diverse. DivIn intervenes at the "starting point," allowing for **multiplicative superposition** to push the Pareto frontier further out.
- **vs. Noise-optimization routes (Mao 2023, Guo 2024, etc.)**: While prior noise optimization focused on text-image alignment or finding rare concepts, DivIn extends this to the **diversity** dimension and proposes a paradigm shift from "optimizing to a point" to "sampling from a posterior."

## Rating
- Novelty: ⭐⭐⭐⭐ While Langevin and Tweedie are established tools, the reformulation of initialization as posterior sampling and its clear separation from trajectory methods is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers DDPM and Rectified Flow, class-conditional and text-to-image tasks, averaged results across multiple seeds, and comprehensive sweeps against SAIL/PG/CADS/IG.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear storyline—observation (mode collapse) → geometric insight (curvature contraction) → formulation (diversity posterior) → algorithm (Langevin) → ablation against SAIL. Each step is backed by evidence.
- Value: ⭐⭐⭐⭐ Training-free, $+3\%$ inference overhead, plug-and-play, and stackable with existing methods—very engineer-friendly for any diffusion/FM pipeline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Does Semantic Noise Initialization Transfer from Images to Videos? A Paired Diagnostic Study](../../ICLR2026/image_generation/does_semantic_noise_initialization_transfer_from_images_to_videos_a_paired_diagn.md)
- [\[ICML 2026\] Unified Masked Diffusion Models with Diverse Generation Orders](unifying_masked_diffusion_models_with_various_generation_orders_and_beyond.md)
- [\[CVPR 2026\] HiFi-Inpaint: Towards High-Fidelity Reference-Based Inpainting for Generating Detail-Preserving Human-Product Images](../../CVPR2026/image_generation/hifi-inpaint_towards_high-fidelity_reference-based_inpainting_for_generating_det.md)
- [\[ICML 2026\] PhysForge: Generating Physics-Grounded 3D Assets for Interactive Virtual World](physforge_generating_physics-grounded_3d_assets_for_interactive_virtual_world.md)
- [\[NeurIPS 2025\] Increasing the Utility of Synthetic Images through Chamfer Guidance](../../NeurIPS2025/image_generation/increasing_the_utility_of_synthetic_images_through_chamfer_guidance.md)

</div>

<!-- RELATED:END -->
