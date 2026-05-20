---
title: >-
  [Paper Note] Adversarial Flow Models
description: >-
  [ICML 2026][Image Generation][Adversarial Training] The authors add an optimal transport regularization $\|G(z)-z\|^2$ to the GAN training objective…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Adversarial Training"
  - "Flow Matching"
  - "One-step Generation"
  - "Optimal Transport"
  - "DiT"
date: 2026-05-08
content_hash: e240409ad99e7257
---

# Adversarial Flow Models

**Conference**: ICML 2026  
**arXiv**: [2511.22475](https://arxiv.org/abs/2511.22475)  
**Code**: Mentioned at the end of the paper: "The code is available at this repository" (available)  
**Area**: Image Generation / Diffusion & Flow Matching / GAN  
**Keywords**: Adversarial Training, Flow Matching, One-step Generation, Optimal Transport, DiT

## TL;DR
The authors add an optimal transport regularization $\|G(z)-z\|^2$ to the GAN training objective, constraining the "arbitrary transport map" of GANs to the Wasserstein-2 optimal transport map. This enables stable adversarial training and end-to-end one-step generation on pure transformers for the first time. On ImageNet-256, 1NFE FID reaches 2.38 (XL/2) and 1.94 (112 layers).

## Background & Motivation

**Background**: Few-step/one-step image generation mainly follows two paths: (1) distilling consistency models / sCM / MeanFlow / Shortcut, etc., from pretrained flow matching teachers; (2) adversarial training for final refinement (GAN-style). Both typically retain the flow backbone.

**Limitations of Prior Work**: Consistency methods, even when targeting one-step generation, must propagate consistency constraints across all timesteps, which consumes model capacity, accumulates propagation errors, and leads to blurry images due to pointwise/moment matching losses. Pure GAN training is highly unstable on standard transformers, requiring either convolutions plus complex tricks (R3GAN) or freezing feature networks (GAT), thus missing out on DiT/large model scaling benefits.

**Key Challenge**: The authors identify the root cause of GAN instability: the adversarial objective only enforces distribution matching, not the specific transport map $z \mapsto x$. Theoretically, infinitely many valid transport maps exist, and initialization plus training randomness causes the generator to drift among them.

**Goal**: Using a single objective (without distillation/teacher/feature network), achieve stable one-step/few-step adversarial training on standard DiT architectures, while enjoying the deterministic transport property of flows.

**Key Insight**: Introducing the Brenier theorem: under Gaussian source and quadratic cost, the optimal transport map is unique. Adding a loss encouraging $G(z)$ to stay close to $z$ on top of GAN locks the generator to the unique Wasserstein-2 optimal transport map among all "valid transport maps," eliminating generator drift.

**Core Idea**: Use $\mathcal{L}_{\mathrm{ot}}^G = \mathbb{E}_z[\|G(z)-z\|^2/n]$ as an additional regularization term in GAN, together with an EMA-normalized backward propagation trick, enabling adversarial training to stably train one-step/few-step models from scratch on DiT.

## Method

### Overall Architecture
The model remains a GAN: the generator $G$ maps Gaussian noise $z\in\mathbb{R}^n$ directly to image latent $G(z) \in \mathbb{R}^n$, and the discriminator $D$ distinguishes real from fake using relativistic loss, R1/R2 gradient penalties (finite difference approximation), and logit centering penalty. An additional optimal transport loss $\mathcal{L}_{\mathrm{ot}}^G$ is added to the generator, and gradients from the discriminator are EMA-normalized so that $\lambda_{\mathrm{ot}}$ can be reused across model scales. Multi-step/arbitrary-step is naturally extended by introducing source timestep $s$, target timestep $t$, and linear interpolation $x_s = (1-s)x + s z$. The architecture uses an unmodified standard DiT; for one-step, the timestep projection is removed. The discriminator is nearly symmetric to the generator, with only an extra [CLS] token.

### Key Designs

1. **Optimal Transport Regularization + Brenier Anchoring**:

    - **Function**: On top of GAN's marginal matching, further anchors the "shape" of the transport map, ensuring the generator converges to the unique Wasserstein-2 optimal transport mapping.
    - **Mechanism**: Add $\mathcal{L}_{\mathrm{ot}}^G=\mathbb{E}_z\big[\tfrac{1}{n}\|G(z)-z\|^2_2\big]$ to the generator; in the multi-step setting, generalize to $\mathbb{E}_{x,z,s,t}\big[\tfrac{1}{n\,w(s,t)}\|G(x_s,s,t)-x_s\|^2_2\big]$, with weight $w(s,t)=\max(|s-t|,\delta)$. $\lambda_{\mathrm{ot}}$ must be scheduled: too small, stuck in local minima; too large, pushed toward identity mapping. The paper adopts a decay strategy based on training progress.
    - **Design Motivation**: To eliminate the true cause of "GAN training non-convergence"—objective underdetermination. Brenier's theorem guarantees a unique optimal transport map; OT regularization turns GAN optimization into "selecting the closest one" among all valid transport maps, making training curves and results under different random initializations stable and reproducible (in 1D Gaussian mixture experiments, the mapping is exactly reproducible).

2. **Gradient Normalization in Backward Path**:

    - **Function**: Makes the hyperparameter $\lambda_{\mathrm{ot}}$ transferable across B/2 → XL/2 → 112-layer models, without re-tuning for each size.
    - **Mechanism**: Rewrite $D(G(z))$ as $D(\phi(G(z)))$, where $\phi$ is identity in the forward pass, and in the backward pass normalizes $\partial \mathcal{L}_{\mathrm{adv}}^G/\partial G(z)$ by the EMA-tracked gradient norm, then divides by $\sqrt{n}$. This is akin to bringing Adam's second-moment idea to the backward path.
    - **Design Motivation**: The gradient magnitude of adversarial loss backpropagated from $D$ is strongly affected by architecture, initialization, and $\lambda_{\mathrm{gp}}$; Adam's adaptive scaling originally "absorbs" these differences. However, after adding $\lambda_{\mathrm{ot}}$, the relative scale of the two losses becomes important, so adversarial gradients must be normalized to a unified scale first.

3. **Arbitrary-step Training + Deep Recursive One-step Model**:

    - **Function**: Enables the same framework to support pure one-step generation, few-step generation, and arbitrary source/target timestep transport; also, by repeating transformer blocks, the one-step model can be made very deep to match the parameter count of multi-step models.
    - **Mechanism**: During training, $s\sim\mathcal{U}(0,1),\ t\sim\mathcal{U}(0,s)$; the generator receives $(x_s, s, t)$, with residual form $G(x_s,s,t) = x_s - (s-t)\,g(x_s,s,t)$ (similar to velocity prediction). The discriminator depends only on $(x_t, t)$ and must not condition on the source sample—otherwise, independent sampling of $x,z$ makes the objective unsatisfiable and training diverges. Deep one-step models use transformer block repetition: each time, the hidden state is reused, with a lightweight "repeat ID embedding" to distinguish iterations. The whole process is still end-to-end one-step training, with no intermediate supervision.
    - **Design Motivation**: Compared to consistency methods, here $G$ directly learns the target distribution via $D$, without propagating consistency, so training can be done only at the 1-NFE set of timesteps; meanwhile, deep one-step models avoid "repeated projection into data space → projection error," consolidating the capacity advantage of multi-step models into the one-step inference path.

### Loss & Training

Discriminator loss: $\mathcal{L}_{\mathrm{AF}}^D = \mathcal{L}_{\mathrm{adv}}^D + \lambda_{\mathrm{gp}}(\mathcal{L}_{r_1}^D + \mathcal{L}_{r_2}^D) + \lambda_{\mathrm{cp}}\mathcal{L}_{\mathrm{cp}}^D$, where R1/R2 use finite difference with $\epsilon=0.01$ instead of second derivatives, computed on only 25% of the batch. Generator loss: $\mathcal{L}_{\mathrm{AF}}^G = \mathcal{L}_{\mathrm{adv}}^G + \lambda_{\mathrm{ot}}\mathcal{L}_{\mathrm{ot}}^G$. AdamW, $\beta_1=0,\beta_2=0.9$, lr $1\times10^{-4}$, batch 256, EMA 0.9999, following MeanFlow's size definitions (B/M/L/XL, patch=2). Generator and discriminator are of the same size, each with an independent dataloader. Guidance is implemented via an additional $\mathcal{L}_{\mathrm{cg}}^G=-\mathbb{E}[C(\mathrm{interp}(G(z,c),z',t'),t',c)]$, and gradient accumulation over timesteps is required to reproduce CFG behavior.

## Key Experimental Results

### Main Results
ImageNet-256 (32×32×4 VAE latent) class-conditional generation, FID-50k evaluated on the full train set, mainly comparing 1NFE / 2NFE / 4NFE.

| Model | NFE | Params / Depth | FID-50k | Notes |
|-------|-----|---------------|---------|-------|
| AF B/2 (Ours) | 1 | 28 layers | Close to sCM XL/2 | Capacity preserved for one-step generation |
| AF XL/2 (Ours) | 1 | 28 layers | **2.38** | 1NFE new SOTA |
| AF XL/2 (Ours, deep recursion) | 1 | 56 layers | **2.08** | Surpasses 28-layer 2NFE baseline |
| AF XL/2 (Ours, deep recursion) | 1 | 112 layers | **1.94** | Surpasses 28-layer 4NFE baseline |
| sCM / iMM / MeanFlow / AYF etc. | 1 | Same size | Higher than ours | Consistency family |
| R3GAN / GAT etc. pure adversarial | 1 | Conv / non-standard transformer | Weaker or not comparable | Require frozen feature network or non-standard architecture |

### Ablation Study

| Configuration | Phenomenon | Interpretation |
|---------------|------------|----------------|
| No $\mathcal{L}_{\mathrm{ot}}$, any $\lambda_{\mathrm{gp}}$ | Training diverges | OT regularization is necessary for stable adversarial training on DiT |
| $\lambda_{\mathrm{ot}}$ too small | Easily stuck in local minima | Insufficient to constrain transport map, degenerates to GAN behavior |
| $\lambda_{\mathrm{ot}}$ too large | Pushes toward $G(z)\approx z$ | Distribution matching is sacrificed |
| Fixed vs decayed $\lambda_{\mathrm{ot}}$ | Decay is better | Constrain transport early, let GAN fine-tune distribution later |
| No gradient normalization | $\lambda_{\mathrm{ot}}$ must be re-tuned for each size | EMA normalization makes hyperparameters transferable across B → XL → 112 layers |
| $D(\cdot, z)$ i.e., condition on source | Training oscillates/diverges | Due to independent $x,z$ sampling, the objective is mathematically unsatisfiable |
| Simple classifier guidance $C(G(z,c),c)$ | Almost same as no guidance | When class boundaries are clear, classifier has no gradient, guidance fails; must use timestep-conditional classifier + accumulate gradients along flow |

### Key Findings
- Without teacher distillation, feature networks, or architecture modifications, pure adversarial training on standard DiT can be stably trained from scratch and achieves 1NFE SOTA on ImageNet; OT regularization is the key switch.
- In the guidance-free setting, this method even surpasses flow matching; attributed to $L_2$ not being a manifold metric, forward KL's strong mode-coverage producing OOD samples, while GAN discriminators are closer to perceptual metrics and JS distance is more robust to outliers.
- Deep recursive one-step models surpass multi-step models, indicating that effective model depth is the bottleneck for one-step generation fidelity, not the "number of steps"—offering a new perspective on the "one-step vs multi-step" debate.

## Highlights & Insights
- Clearly attributes GAN instability to "objective underdetermination," and uses Brenier to provide a unique optimal transport map as an anchor—a clean, provable, and directly applicable perspective, lighter than the "propagate consistency" approach of the consistency family.
- Timestep-conditional classifier guidance ($C(x_{t'}, t', c)$) simulates CFG's gradient accumulation along the flow, enabling one-step adversarial models to enjoy CFG-style controllable generation; this trick can be directly applied to any one-step/few-step GAN framework.
- EMA gradient normalization in the backward path is an underrated "hyperparameter search reduction" engineering trick—by adjusting the relative scale of multi-objective losses to the scale of $D$'s output, the choice of new loss weights is decoupled from model size.
- Deep recursive one-step training conceptually challenges the inherent bias that "flow must be multi-step," providing a new design point of "trading capacity for NFE."

## Limitations & Future Work
- Experiments are still limited to ImageNet-256 class-conditional; large-scale text-to-image/video validation is lacking. The authors only cite Lin et al. 2025 in motivation to suggest scalability.
- The decay schedule of $\lambda_{\mathrm{ot}}$ still requires manual design; although gradient normalization makes hyperparameters transferable across sizes, the schedule shape needs further study.
- When the generator enters regions where the uniqueness of the transport map fails (e.g., boundaries of multimodal distributions), OT regularization may conflict with the GAN objective; this is not strictly analyzed theoretically.
- Training extremely deep one-step models (112 layers + repetition) still depends on small learning rates and low OT decay lower bounds; engineering-wise, batch/hardware requirements remain.

## Related Work & Insights
- **vs Consistency Family (CM / sCM / iMM / MeanFlow / AYF / Shortcut)**: They propagate consistency constraints along the flow, requiring training at all timesteps; this work trains directly at the target timestep, saving capacity and avoiding error accumulation.
- **vs R3GAN / GAT and other pure adversarial revival works**: They rely on convolutions + special designs / frozen feature networks; this work uses standard DiT, with the only change being an added [CLS] token in $D$.
- **vs Distillation (Salimans & Ho / Liu et al.)**: This work does not require a teacher and can be trained end-to-end from scratch.
- **vs Distillation + Adversarial Fine-tuning (e.g., Lin et al. 2025)**: They use adversarial training for final refinement; this work shows adversarial training alone suffices for main training, eliminating the two-stage process.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of Brenier anchoring and backward gradient normalization is both clear and explanatory for instability, and enables, for the first time, from-scratch adversarial training on standard DiT.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic comparison on ImageNet-256 across multiple sizes and NFE, with extensive hyperparameter/configuration ablations, but lacks large-scale T2I/video validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Causal analysis → mathematical motivation → implementation tricks → extensive ablation, with a "textbook-style" argument flow throughout.
- Value: ⭐⭐⭐⭐⭐ Directly challenges the mainstream "few-step generation must distill/must be consistent" path, opening a new direction for large-scale generative model design.

## Related Papers

- [\[ICLR 2026\] TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows](../../ICLR2026/image_generation/twinflow_realizing_one-step_generation_on_large_models_with_self-adversarial_flo.md)
- [\[CVPR 2025\] Instant Adversarial Purification with Adversarial Consistency Distillation](../../CVPR2025/image_generation/instant_adversarial_purification_with_adversarial_consistency_distillation.md)
- [\[ICML 2026\] The Coupling Within: Flow Matching via Distilled Normalizing Flows](the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)
- [\[ICML 2026\] Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features](cascaded_flow_matching_for_heterogeneous_tabular_data_with_mixed-type_features.md)
- [\[ECCV 2024\] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](../../ECCV2024/image_generation/learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)

</div>

<!-- RELATED:END -->
