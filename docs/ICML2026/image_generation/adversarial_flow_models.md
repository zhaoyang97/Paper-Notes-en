---
title: >-
  [Paper Note] Adversarial Flow Models
description: >-
  [ICML 2026][Image Generation][Adversarial Training] The authors add an optimal transport regularization term $\|G(z)-z\|^2$ to the GAN training objective…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Adversarial Training"
  - "Flow Matching"
  - "One-step generation"
  - "Optimal Transport"
  - "DiT"
date: 2026-05-08
content_hash: 69d69d73ffd3c5a3
---

# Adversarial Flow Models

**Conference**: ICML 2026  
**arXiv**: [2511.22475](https://arxiv.org/abs/2511.22475)  
**Code**: The paper mentions "The code is available at this repository" (Yes)  
**Area**: Image Generation / Diffusion & Flow Matching / GAN  
**Keywords**: Adversarial Training, Flow Matching, One-step generation, Optimal Transport, DiT

## TL;DR
The authors add an optimal transport regularization term $\|G(z)-z\|^2$ to the GAN training objective, constraining the GAN's "arbitrary transport map" into a Wasserstein-2 optimal transport map. This allows adversarial training on pure transformers to achieve stable convergence for the first time for end-to-end single-step generation, reaching 1NFE FID of 2.38 (XL/2) and 1.94 (112 layers) on ImageNet-256.

## Background & Motivation

**Background**: Few-step/single-step image generation primarily follows two paths: (1) Distilling consistency models / sCM / MeanFlow / Shortcut etc., from pre-trained flow matching teachers; (2) Using adversarial training for GAN-style refinement. Both paths typically require maintaining a flow backbone.

**Limitations of Prior Work**: Consistency methods, even when targeting single-step generation, must propagate consistency constraints across all timesteps, which "consumes" model capacity, accumulates propagation errors, and results in blurry images due to pointwise / moment matching losses. Pure GAN training on standard transformers is highly unstable, requiring either convolution + complex tricks (R3GAN) or frozen feature networks (GAT), failing to benefit from the scaling dividends of DiT / large models.

**Key Challenge**: The authors identify the root cause of GAN instability: the adversarial objective only constrains the generated distribution to match the data distribution but does not constrain the specific transport map $z \mapsto x$. Theoretically, infinitely many valid transport maps exist; randomness in initialization and training causes the generator to drift among them.

**Goal**: To use a single objective (independent of distillation / teachers / feature networks) to perform stable single/few-step adversarial training on standard DiT architectures while enjoying the deterministic transport properties of flow.

**Key Insight**: Introduce Brenier’s theorem: under Gaussian source + quadratic cost, the optimal transport map is unique. By adding a loss that encourages $G(z)$ to stay close to $z$ on top of the GAN objective, one can "lock" the unique Wasserstein-2 optimal transport map among all "valid transport maps," thereby eliminating generator drift.

**Core Idea**: Use a GAN with $\mathcal{L}_{\mathrm{ot}}^G = \mathbb{E}_z[\|G(z)-z\|^2/n]$ as an additional regularization term, combined with a backpropagation trick using EMA normalization, to allow adversarial training to train single/few-step generation models from scratch on DiT.

## Method

### Overall Architecture
The model remains a GAN: the generator $G$ directly maps Gaussian noise $z\in\mathbb{R}^n$ to image latents $G(z) \in \mathbb{R}^n$, while a discriminator $D$ distinguishes real from fake using a relativistic loss + R1/R2 gradient penalties (approximated via finite difference) + logit centering penalties. An additional optimal transport loss $\mathcal{L}_{\mathrm{ot}}^G$ is added to the generator side, and gradients from the discriminator are normalized via EMA, making $\lambda_{\mathrm{ot}}$ reusable across model scales. Multi-step/arbitrary-step generation is naturally extended by introducing a source timestep $s$, target timestep $t$, and linear interpolation $x_s = (1-s)x + s z$. Architecturally, it uses unmodified standard DiT; for single-step, timestep projections are removed, and the discriminator is nearly symmetric to the generator, merely adding a [CLS] token.

### Key Designs

1.  **Optimal Transport Regularization + Brenier Anchoring**:
    - **Function**: Locks the "shape" of the transport map on top of the GAN's marginal matching goal, forcing the generator to converge to the unique Wasserstein-2 optimal transport map.
    - **Mechanism**: Adds $\mathcal{L}_{\mathrm{ot}}^G=\mathbb{E}_z\big[\tfrac{1}{n}\|G(z)-z\|^2_2\big]$ to the generator, generalized in multi-step settings to $\mathbb{E}_{x,z,s,t}\big[\tfrac{1}{n\,w(s,t)}\|G(x_s,s,t)-x_s\|^2_2\big]$ with weight $w(s,t)=\max(|s-t|,\delta)$. $\lambda_{\mathrm{ot}}$ requires scheduling: too small fails to escape local minima, while too large pushes towards the identity map; this paper employs a strategy that decays according to training progress.
    - **Design Motivation**: Eliminates the actual cause of "GAN training non-convergence"—an under-determined objective. Brenier’s theorem guarantees a unique optimal transport map; OT regularization transforms GAN optimization into "choosing the closest" among all valid transport maps, making training curves and results stable and reproducible across different random seeds (yielding identical mappings in 1D Gaussian mixture experiments).

2.  **Gradient Normalization in Backward Path**:
    - **Function**: Allows the hyperparameter $\lambda_{\mathrm{ot}}$ to be universal across B/2 → XL/2 → 112-layer models without per-size searching.
    - **Mechanism**: Rewrites $D(G(z))$ as $D(\phi(G(z)))$, where $\phi$ is the identity in the forward pass but normalizes $\partial \mathcal{L}_{\mathrm{adv}}^G/\partial G(z)$ by the gradient norm tracked via EMA in the backward pass, then divides by $\sqrt{n}$. This can be seen as applying the second-moment logic of Adam to the backward path.
    - **Design Motivation**: Gradient magnitudes from $D$ are strongly affected by architecture, initialization, and $\lambda_{\mathrm{gp}}$. Originally, Adam's adaptive scaling could "absorb" magnitude differences; however, the relative ratio between the two losses becomes critical when adding $\lambda_{\mathrm{ot}}$, necessitating the normalization of adversarial gradients to a unified scale.

3.  **Any-step Training + Deep Recursive Single-step Model**:
    - **Function**: Enables the framework to support pure single-step generation, few-step generation, and transport between arbitrary source/target timesteps. It also matches the parameter count of multi-step models by repeating transformer blocks in a single-step model.
    - **Mechanism**: During training $s\sim\mathcal{U}(0,1),\ t\sim\mathcal{U}(0,s)$, the generator receives $(x_s, s, t)$ and is written in residual form $G(x_s,s,t) = x_s - (s-t)\,g(x_s,s,t)$ (similar to velocity prediction). The discriminator depends only on $(x_t, t)$ and must not be conditioned on the source sample—otherwise, the independent sampling of $x, z$ makes the objective unsatisfiable, causing training to diverge. Deep single-step models use transformer block repetition: reusing the hidden state each time with a lightweight "repetition ID embedding" to distinguish iterations, while training remains end-to-end single-step without intermediate supervision.
    - **Design Motivation**: Compared to consistency methods, **Ours** learns the target distribution directly via $D$ without needing to propagate consistency, allowing training on a specific set of timesteps like 1-NFE. Meanwhile, extremely deep single-step models avoid the "repeated entry into data space → projection error" problem, incorporating the capacity advantage of multi-step models into a single-step inference path.

### Loss & Training
The discriminator loss is $\mathcal{L}_{\mathrm{AF}}^D = \mathcal{L}_{\mathrm{adv}}^D + \lambda_{\mathrm{gp}}(\mathcal{L}_{r_1}^D + \mathcal{L}_{r_2}^D) + \lambda_{\mathrm{cp}}\mathcal{L}_{\mathrm{cp}}^D$, where R1/R2 are replaced by finite differences with $\epsilon=0.01$ and calculated for only 25% of the batch; the generator loss is $\mathcal{L}_{\mathrm{AF}}^G = \mathcal{L}_{\mathrm{adv}}^G + \lambda_{\mathrm{ot}}\mathcal{L}_{\mathrm{ot}}^G$. Training uses AdamW, $\beta_1=0,\beta_2=0.9$, lr $1\times10^{-4}$, batch size 256, EMA 0.9999, following MeanFlow size definitions (B/M/L/XL, patch=2). Generator and discriminator share the same size and use independent dataloaders. Guidance is implemented via an additional $\mathcal{L}_{\mathrm{cg}}^G=-\mathbb{E}[C(\mathrm{interp}(G(z,c),z',t'),t',c)]$, which must accumulate gradients across timesteps to replicate CFG behavior.

## Key Experimental Results

### Main Results
Class-conditional generation on ImageNet-256 (32×32×4 VAE latent), FID-50k evaluated against the full training set, primarily comparing 1NFE / 2NFE / 4NFE.

| Model | NFE | Params / Depth | FID-50k | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| AF B/2 (Ours) | 1 | 28 layers | Close to sCM XL/2 | Capacity preserved for single-step |
| AF XL/2 (Ours) | 1 | 28 layers | **2.38** | New 1NFE SOTA |
| AF XL/2 (Ours, Recursive) | 1 | 56 layers | **2.08** | Beats 28-layer 2NFE equivalent |
| AF XL/2 (Ours, Recursive) | 1 | 112 layers | **1.94** | Beats 28-layer 4NFE equivalent |
| sCM / iMM / MeanFlow / AYF | 1 | Same size | Higher than Ours | Consistency family |
| Pure Adv (R3GAN / GAT) | 1 | Conv / Non-standard | Weaker or N/A | Requires frozen features or non-standard arch |

### Ablation Study

| Configuration | Phenomenon | Interpretation |
| :--- | :--- | :--- |
| No $\mathcal{L}_{\mathrm{ot}}$, any $\lambda_{\mathrm{gp}}$ | Training diverges | OT regularization is a necessary condition for adversarial training on stable DiT |
| $\lambda_{\mathrm{ot}}$ too small | Falls into local minima | Insufficient to constrain the transport map, behavior reverts to standard GAN |
| $\lambda_{\mathrm{ot}}$ too large | Pushed towards $G(z)\approx z$ | Distribution matching is sacrificed |
| Fixed $\lambda_{\mathrm{ot}}$ vs Decay | Decay is superior | Early stages constrain transport, later stages let GAN fine-tune the distribution |
| No Gradient Norm | $\lambda_{\mathrm{ot}}$ needs per-size search | EMA normalization makes hyperparameters universal from B → XL → 112 layers |
| $D(\cdot, z)$ (source cond) | Training oscillates / diverges | Objective is mathematically unsatisfiable due to independent $x, z$ sampling |
| Simple $C(G(z,c),c)$ guidance | Nearly identical to no guidance | Classifiers have no gradient at clear boundaries; must use timestep-conditional $C$ + gradient accumulation |

### Key Findings
- Without teacher distillation, feature networks, or architectural changes, pure adversarial training on standard DiT can stably train from scratch and achieve 1NFE SOTA on ImageNet; OT regularization is the critical switch.
- In guidance-free settings, **Ours** even outperforms flow matching; the authors attribute this to $L_2$ not being a manifold metric—forward KL's strong mode-coverage tends to produce OOD samples, whereas a GAN discriminator is closer to a perceptual metric and JS distance is more robust to outliers.
- The success of deep recursive single-step models reveals that effective model depth, rather than the "number of steps" itself, is the bottleneck for single-step generation fidelity—providing a new interpretation for the "single-step vs multi-step" debate.

## Highlights & Insights
- Diagnosing GAN training instability as "objective under-determination" and using Brenier’s theorem to provide a unique optimal transport map as an anchor is a clean, provable, and actionable perspective, lighter than the "propagate consistency" approach of the consistency family.
- Timestep-conditional classifier guidance ($C(x_{t'}, t', c)$) simulates the effect of CFG accumulating gradients along the flow, allowing single-step adversarial models to benefit from CFG-style controllable generation; this trick can be directly applied to any single/few-step GAN framework.
- EMA gradient normalization in the backward path is an underrated engineering trick for reducing hyperparameter search—by decoupling the added loss weight from the scale of $D$'s output, the choice of $\lambda$ becomes independent of model size.
- Deep recursive single-step training conceptually counters the inherent bias that "flow must be multi-step," providing a new design point for trading capacity for NFE.

## Limitations & Future Work
- Datasets are still limited to ImageNet-256 class-conditional generation, without large-scale validation on text-to-image or video; the authors only hint at scalability via motivation citations.
- The decay schedule for $\lambda_{\mathrm{ot}}$ still requires manual design; while gradient normalization makes hyperparameters universal across sizes, the shape of the schedule needs further research.
- In regions where transport map uniqueness fails (e.g., boundaries of multimodal distributions), OT regularization may create tension with the GAN objective; this is not strictly analyzed theoretically.
- Training costs and stability for extremely deep single-step models (112 layers + repetition) still rely on small learning rates and low OT decay floors, maintaining demands on batch size and hardware.

## Related Work & Insights
- **vs Consistency Family (CM / sCM / iMM / MeanFlow / AYF / Shortcut)**: They propagate consistency constraints along the flow, requiring training across all timesteps; **Ours** trains directly at target timesteps, saving capacity and avoiding error accumulation.
- **vs Pure Adversarial (R3GAN / GAT)**: They rely on convolutions + special designs or frozen feature networks; **Ours** uses standard DiT, with the only change being a [CLS] token in $D$.
- **vs Distillation (Salimans & Ho / Liu et al.)**: **Ours** does not require a teacher and can be trained end-to-end from scratch.
- **vs Distillation + Adv Fine-tuning (e.g., Lin et al. 2025)**: They use GANs for final refinement; **Ours** proves that adversarial training alone is sufficient for the main training, eliminating the two-stage process.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of Brenier anchoring + backward gradient normalization is clear, explains instability, and enables from-scratch adversarial training on DiT for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic comparison of multiple sizes and NFEs on ImageNet-256 with extensive ablation, though missing large-scale T2I/video validation.
- Writing Quality: ⭐⭐⭐⭐⭐ From diagnosis → mathematical motivation → implementation tricks → extensive ablations, the paper has a "textbook" flow of argumentation.
- Value: ⭐⭐⭐⭐⭐ Directly challenges the mainstream path that "few-step generation must use distillation/consistency," opening a new path for future large-scale generative model design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows](../../ICLR2026/image_generation/twinflow_realizing_one-step_generation_on_large_models_with_self-adversarial_flo.md)
- [\[ICML 2026\] E²PO: Embedding-perturbed Exploration Preference Optimization for Flow Models](embedding-perturbed_exploration_preference_optimization_for_flow_models.md)
- [\[ICML 2026\] SURF: Separation via Unsupervised Remixing Flow](surf_separation_via_unsupervised_remixing_flow.md)
- [\[ICML 2026\] A Kinetic Energy Perspective of Flow Matching](a_kinetic_energy_perspective_of_flow_matching.md)
- [\[ICML 2026\] Stable Velocity: A Variance Perspective on Flow Matching](stable_velocity_a_variance_perspective_on_flow_matching.md)

</div>

<!-- RELATED:END -->
