---
title: >-
  [Paper Note] Adversarial Flow Models
description: >-
  [ICML 2026][Image Generation][Adversarial Training] The authors add an optimal transport regularization term $\|G(z)-z\|^2$ to the GAN training objective, constraining the GAN's "arbitrary transport map" to a unique Wasserstein-2 optimal transport map. This allows adversarial training on pure Transformers to stabilize for the first time and perform end-to-end single-ste
tags:
  - ICML 2026
  - Image Generation
  - Adversarial Training
  - Flow Matching
  - Optimal Transport
  - DiT
date: 2026-05-08
content_hash: 2c52505dcbf244d1
---
# Adversarial Flow Models

**Conference**: ICML 2026  
**arXiv**: [2511.22475](https://arxiv.org/abs/2511.22475)  
**Code**: The paper mentions "The code is available at this repository" (Yes)  
**Area**: Image Generation / Diffusion & Flow Matching / GAN  
**Keywords**: Adversarial Training, Flow Matching, One-step Generation, Optimal Transport, DiT

## TL;DR
The authors add an optimal transport regularization term $\|G(z)-z\|^2$ to the GAN training objective, constraining the GAN's "arbitrary transport map" to a unique Wasserstein-2 optimal transport map. This allows adversarial training on pure Transformers to stabilize for the first time and perform end-to-end single-step generation. On ImageNet-256, the 1NFE FID reaches **2.38** (XL/2) and **1.94** (112-layer recursive model).

## Background & Motivation

**Background**: Few-step/single-step image generation currently follows two main paths: (1) Distilling consistency models (CM, sCM, MeanFlow, Shortcut, etc.) from pre-trained flow matching teachers; (2) Using adversarial training for refinement (GAN-style refinement). Both typically require maintaining the flow-based backbone or a teacher model.

**Limitations of Prior Work**: Consistency-based methods, even when targeting single-step generation, must propagate consistency constraints across all timesteps. This "consumes" model capacity, accumulates propagation errors, and often yields blurry images due to pointwise or moment matching losses. Pure GAN training is notoriously unstable on standard Transformers, requiring either convolutional architectures with complex tricks (R3GAN) or frozen feature networks (GAT), thus failing to benefit from the scaling of DiT and large-scale pre-training.

**Key Challenge**: The authors identify the root cause of GAN instability: the adversarial objective only constrains the generated distribution to match the data distribution but does not constrain the specific transport map $z \mapsto x$. Theoretically, infinitely many valid transport maps exist. Randomness in initialization and training causes the generator to drift among these maps, leading to divergence.

**Goal**: Establish a single objective (independent of distillation, teachers, or auxiliary feature networks) to stabilize single-step/few-step adversarial training on standard DiT architectures while retaining the deterministic transport properties of flow models.

**Key Insight**: By introducing Brenier’s theorem, which states that the optimal transport map is unique under Gaussian sources and quadratic costs, the authors propose adding a loss that encourages $G(z)$ to remain close to $z$. This locks the unique Wasserstein-2 optimal transport map among all "valid transport maps," thereby eliminating generator drift.

**Core Idea**: A GAN framework using $\mathcal{L}_{\mathrm{ot}}^G = \mathbb{E}_z[\|G(z)-z\|^2/n]$ as an additional regularization term, combined with a backpropagation trick using EMA-based gradient normalization, enabling the training of single-step/few-step generation models from scratch on DiT.

## Method

### Overall Architecture
The framework remains a GAN: the generator $G$ maps Gaussian noise $z\in\mathbb{R}^n$ directly to an image latent $G(z)\in\mathbb{R}^n$. The discriminator $D$ utilizes a relativistic loss with R1/R2 gradient penalties (approximated via finite difference) and logit centering. The core modification is the inclusion of the optimal transport regularization on the generator to fix the under-determined transport map issue. A gradient normalization scheme on the backward path ensures hyperparameters are robust across various model scales. The system supports pure one-step generation and extends to arbitrary-step transport via source/target timesteps $s, t$ and linear interpolation, utilizing an unmodified standard DiT architecture.

### Key Designs

**1. Optimal Transport Regularization + Brenier Anchoring: Pinning the Adversarial Objective to a Unique Map**

The root of GAN instability is that the adversarial objective matches distributions but does not constrain the shape of the $z\mapsto x$ transport map. This paper introduces an optimal transport loss $\mathcal{L}_{\mathrm{ot}}^G=\mathbb{E}_z\big[\tfrac{1}{n}\|G(z)-z\|^2_2\big]$, encouraging $G(z)$ to stay near the source $z$. In multi-step settings, this generalizes to $\mathbb{E}_{x,z,s,t}\big[\tfrac{1}{n\,w(s,t)}\|G(x_s,s,t)-x_s\|^2_2\big]$, where $w(s,t)=\max(|s-t|,\delta)$. Brenier’s theorem guarantees that under Gaussian sources and quadratic costs, the optimal transport map is unique. Consequently, the OT regularization transforms GAN optimization into "selecting the nearest map among all valid ones," eliminating drift. The weight $\lambda_{\mathrm{ot}}$ must decay during training: if too small, it reverts to standard GAN drift; if too large, $G$ approaches an identity mapping, sacrificing distribution matching.

**2. Backward Gradient Normalization: One $\lambda_{\mathrm{ot}}$ for All Model Scales**

With the addition of $\mathcal{L}_{\mathrm{ot}}$, the relative scale between adversarial and OT losses becomes sensitive. The gradient magnitude of the adversarial loss backpropagated from $D$ is strongly affected by architecture, initialization, and $\lambda_{\mathrm{gp}}$. While Adam's adaptive scaling usually absorbs these differences, it forces $\lambda_{\mathrm{ot}}$ to be re-searched for every model size. The solution is to rewrite $D(G(z))$ as $D(\phi(G(z)))$, where $\phi$ is an identity mapping in the forward pass but normalizes the backward gradient $\partial\mathcal{L}_{\mathrm{adv}}^G/\partial G(z)$ using its EMA-tracked norm divided by $\sqrt{n}$. This essentially applies the second-moment logic of Adam to the backward path, allowing a single $\lambda_{\mathrm{ot}}$ to work across B/2, XL/2, and 112-layer models.

**3. Arbitrary-step Training + Deep Recursive Single-step Models: Trading Capacity for NFE**

The framework supports transport between any source/target timesteps. During training, $s\sim\mathcal{U}(0,1)$ and $t\sim\mathcal{U}(0,s)$ are sampled, and $G$ receives $(x_s,s,t)$ in a residual form: $G(x_s,s,t)=x_s-(s-t)\,g(x_s,s,t)$ (similar to velocity prediction). The discriminator $D$ depends only on $(x_t,t)$ and must not be conditioned on the source sample to avoid mathematical inconsistencies that cause training divergence. Unlike consistency methods that propagate constraints along a flow, $G$ learns the target distribution directly through $D$. To leverage the capacity advantages of multi-step models for single-step inference, the authors employ extremely deep single-step models. This is achieved by repeating Transformer blocks and reusing hidden states with a lightweight "repetition ID embedding" to distinguish iterations, maintaining end-to-end single-step training without intermediate projection errors.

### Loss & Training
The discriminator loss is $\mathcal{L}_{\mathrm{AF}}^D = \mathcal{L}_{\mathrm{adv}}^D + \lambda_{\mathrm{gp}}(\mathcal{L}_{r_1}^D + \mathcal{L}_{r_2}^D) + \lambda_{\mathrm{cp}}\mathcal{L}_{\mathrm{cp}}^D$, where R1/R2 utilize finite difference with $\epsilon=0.01$ instead of second-order derivatives (calculated on 25% of the batch). The generator loss is $\mathcal{L}_{\mathrm{AF}}^G = \mathcal{L}_{\mathrm{adv}}^G + \lambda_{\mathrm{ot}}\mathcal{L}_{\mathrm{ot}}^G$. Training uses AdamW, $\beta_1=0, \beta_2=0.9$, lr $1\times10^{-4}$, batch size 256, and EMA 0.9999. Model sizes (B/M/L/XL, patch=2) follow MeanFlow. $G$ and $D$ are of the same size with independent dataloaders. Guidance is implemented via $\mathcal{L}_{\mathrm{cg}}^G=-\mathbb{E}[C(\text{interp}(G(z,c),z',t'),t',c)]$, requiring gradient accumulation over timesteps to replicate CFG behavior.

## Key Experimental Results

### Main Results
Class-conditional generation on ImageNet-256 ($32\times32\times4$ VAE latent), evaluated via FID-50k against the full training set.

| Model | NFE | Params / Depth | FID-50k | Note |
| :--- | :--- | :--- | :--- | :--- |
| AF B/2 (**Ours**) | 1 | 28 layers | Near sCM XL/2 | Capacity preserved for 1-step |
| AF XL/2 (**Ours**) | 1 | 28 layers | **2.38** | New 1NFE SOTA |
| AF XL/2 (**Ours**, Deep Recur.) | 1 | 56 layers | **2.08** | Beats 28-layer 2NFE baseline |
| AF XL/2 (**Ours**, Deep Recur.) | 1 | 112 layers | **1.94** | Beats 28-layer 4NFE baseline |
| sCM / iMM / MeanFlow | 1 | Same size | Higher than **Ours**| Consistency family |
| R3GAN / GAT | 1 | Conv / Non-std | Weak/Incomp. | Requires frozen features or custom arch |

### Ablation Study

| Configuration | Observation | Interpretation |
| :--- | :--- | :--- |
| No $\mathcal{L}_{\mathrm{ot}}$ | Training diverges | OT regularization is necessary for stable DiT adversarial training. |
| Small $\lambda_{\mathrm{ot}}$ | Local minima traps | Insufficient to constrain the transport map; acts as a standard GAN. |
| Large $\lambda_{\mathrm{ot}}$ | $G(z)\approx z$ | Distribution matching is sacrificed for identity mapping. |
| Decay $\lambda_{\mathrm{ot}}$ vs. Fixed | Decay is superior | Early steps constrain transport; later steps refine the distribution. |
| No Grad Norm | $\lambda_{\mathrm{ot}}$ needs re-tuning | EMA normalization enables universal scaling (B → XL). |
| $D(\cdot, z)$ (Source cond.)| Divergence | Mathematically unsatisfiable due to independent $x,z$ sampling. |
| Simple guidance $C(G(z,c),c)$ | No effect | Classifier gradient vanishes at boundaries; requires timestep-cumulated gradients. |

### Key Findings
- Adversarial training on standard DiT can be trained from scratch to achieve 1NFE SOTA on ImageNet without distilled teachers or feature networks; OT regularization is the critical enabling factor.
- In guidance-free settings, **Ours** outperforms flow matching. This is attributed to the fact that $L_2$ is not a manifold metric and forward KL encourages mode coverage (producing OOD samples), whereas GAN discriminators act as perceptual metrics robust to outliers.
- The success of deep recursive single-step models suggests that the effective depth of the model is the bottleneck for single-step fidelity, rather than the "number of steps" themselves.

## Highlights & Insights
- Identifying GAN instability as an "under-determined objective" problem and using Brenier's theorem to provide a unique anchor is a clean, provable, and effective perspective, significantly simpler than the "consistency propagation" logic of the consistency family.
- Timestep-conditioned classifier guidance ($C(x_{t'}, t', c)$) successfully simulates CFG behavior by accumulating gradients along the flow, allowing single-step adversarial models to benefit from CFG-style control.
- EMA-based gradient normalization on the backward path is an underrated engineering trick that decouples loss weights from model scale by normalizing the adversarial gradient to a standard scale.
- Deep recursive single-step training conceptually challenges the bias that "flow must be multi-step," providing a new design principle: trading model depth for NFE.

## Limitations & Future Work
- Datasets are limited to ImageNet-256; large-scale text-to-image or video validation has not been performed (though citations suggest scalability).
- The $\lambda_{\mathrm{ot}}$ decay schedule still requires manual design; while gradient normalization makes it robust across sizes, the optimal schedule shape needs more research.
- Theoretical analysis is missing for regions where transport map uniqueness might fail (e.g., at multimodal distribution boundaries).
- Training extremely deep models (112 layers) remains demanding in terms of batch size and hardware, relying on small learning rates and low OT decay floors.

## Related Work & Insights
- **vs. Consistency Family (CM / sCM / MeanFlow / Shortcut)**: They propagate consistency along the flow and require training across all timesteps. **Ours** trains directly on the target timestep, preserving capacity and avoiding error accumulation.
- **vs. GAN Revival (R3GAN / GAT)**: They rely on specific convolutional designs or frozen feature networks. **Ours** uses standard DiT with only a minor change (adding a [CLS] token to $D$).
- **vs. Distillation (Salimans & Ho / Liu et al.)**: **Ours** does not require a teacher model and can be trained from scratch.
- **vs. Distillation + Adversarial Fine-tuning**: This work proves that adversarial training is sufficient as the primary training stage, removing the need for a two-stage process.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Brenier anchoring + backward gradient normalization is an elegant solution to a long-standing instability problem in DiT GANs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematic comparisons across sizes and NFEs on ImageNet, though lacking large-scale T2I.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear progression from symptom analysis to mathematical motivation and implementation tricks.
- **Value**: ⭐⭐⭐⭐⭐ Directly challenges the necessity of distillation/consistency for few-step generation, opening a new path for large-scale generative models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] E²PO: Embedding-perturbed Exploration Preference Optimization for Flow Models](embedding-perturbed_exploration_preference_optimization_for_flow_models.md)
- [\[ICLR 2026\] TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows](../../ICLR2026/image_generation/twinflow_realizing_one-step_generation_on_large_models_with_self-adversarial_flo.md)
- [\[CVPR 2025\] Instant Adversarial Purification with Adversarial Consistency Distillation](../../CVPR2025/image_generation/instant_adversarial_purification_with_adversarial_consistency_distillation.md)
- [\[ICML 2026\] Stable Velocity: A Variance Perspective on Flow Matching](stable_velocity_a_variance_perspective_on_flow_matching.md)
- [\[ICML 2026\] The Coupling Within: Flow Matching via Distilled Normalizing Flows](the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)

</div>

<!-- RELATED:END -->
