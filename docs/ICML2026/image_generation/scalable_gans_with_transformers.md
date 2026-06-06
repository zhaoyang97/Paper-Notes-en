---
title: >-
  [Paper Note] Scalable GANs with Transformers
description: >-
  [ICML2026][Image Generation][GAN Scalability] This paper proposes GAT (Generative Adversarial Transformers), a scalable GAN framework built with pure Transformer generators and discriminators in the VAE latent space. By…
tags:
  - "ICML2026"
  - "Image Generation"
  - "GAN Scalability"
  - "Transformer Generator"
  - "VAE Latent Space"
  - "Single-step Generation"
  - "ImageNet Class-conditional Generation"
date: 2026-05-08
content_hash: 7cd452c9e584aa5d
---

# Scalable GANs with Transformers

**Conference**: ICML2026  
**arXiv**: [2509.24935](https://arxiv.org/abs/2509.24935)  
**Code**: https://hse1032.github.io/GAT (Available, Project Page)  
**Area**: Image Generation / GAN / Transformer  
**Keywords**: GAN Scalability, Transformer Generator, VAE Latent Space, Single-step Generation, ImageNet Class-conditional Generation

## TL;DR
This paper proposes GAT (Generative Adversarial Transformers), a scalable GAN framework built with pure Transformer generators and discriminators in the VAE latent space. By activating early generator layers through Multi-level Noise-perturbed image Guidance (MNG) and stabilizing large-scale training with width-aware learning rate scaling, GAT-XL/2 achieves a state-of-the-art single-step FID of 2.18 on ImageNet-256 class-conditional generation in only 60 epochs, using $4\times$ fewer epochs than 1-NFE diffusion/flow baselines of comparable scale.

## Background & Motivation

**Background**: Recent leaps in generative models have largely been built on "scalability"—performance rises almost monotonically as model capacity, data volume, and compute are increased. Diffusion (DiT, SiT) and autoregressive (VAR, MAR) routes have repeatedly validated this scaling law: training with pure Transformer backbones in the VAE latent space allows for stable scaling from small models to billions of parameters.

**Limitations of Prior Work**: GANs are the only major line where scalability has not been systematically discussed. Existing "Large GAN" works (GigaGAN, StyleGAN-XL, R3GAN) are often meticulously tuned for specific tasks with high-capacity models, which does not count as evidence that "GANs can scale." Yet GANs possess advantages that diffusion models lack—single-step inference and a low-dimensional latent space with semantic manipulability—making the scaling of GANs valuable.

**Key Challenge**: Directly applying mature scalable recipes (VAE latent space + pure Transformer) to GANs reveals two failure modes in naive scaling: (1) **Early generator layers essentially do not work**—PCA visualizations show almost no changes in features within the first few blocks, and ablating early blocks has minimal LPIPS impact on final images, meaning most compute from capacity expansion is wasted; (2) **Training diverges** when scaling from S to XL using the same hyperparameters (especially learning rate)—GANs are inherently sensitive to learning rates, and as Transformers widen, the magnitude of output change per step increases linearly with channel counts, causing large models to collapse under the original lr.

**Goal**: Investigate whether GANs can scale using the "VAE latent space + pure Transformer" architecture and provide minimal remedial solutions for the aforementioned obstacles.

**Key Insight**: The authors decouple GAN scaling into independent "architectural" and "optimization" problems. Architecturally, auxiliary supervision is used to "wake up" idle early layers; optimistically, a simple lr scaling formula aligns effective update magnitudes across different model sizes to avoid per-scale tuning.

**Core Idea**: Each intermediate layer of the generator is forced to produce images for discriminator scrutiny via **Multi-level Noise-perturbed image Guidance (MNG)**, mandating a coarse-to-fine refinement division of labor. Simultaneously, a width-aware lr rule $\eta_{\text{adapt}} = \eta_{\text{base}} \cdot C_{\text{base}} / C_{\text{model}}$ ensures stable convergence from S to XL using a single set of base hyperparameters.

## Method

### Overall Architecture
The overall design of GAT can be summarized in one sentence: **a GAN where both the generator and discriminator are pure ViTs, operating on the $32\times 32$ latent space of SD-VAE**.

- Input: Random latent code $z \sim p_z$ ($d_z=64$) + class label $c$.
- Generator $G(z,c)$: Maps $(z,c)$ via a lightweight MLP to a style vector $w$, which produces per-channel modulation parameters $(\gamma, \alpha)$ injected into each ViT block through adaptive RMSNorm + LayerScale. An unpatchify linear head restores the token sequence to a VAE latent map.
- Discriminator $D(I,c)$: Another ViT where each block outputs with LayerScale. A `[cls]` token passes through a linear head to output logits, using a projection discriminator for class conditioning.
- Training Objective: Approximated relativistic pairing loss + dual R1/R2 gradient penalties (following R3GAN's aR1/aR2) + REPA representation alignment (discriminator aligns with DINOv2 features only).
- Output: A $256\times 256$ image (after VAE decoding) in a single forward pass.

GAT-S/B/L/XL configurations follow DiT. With a default patch size $p=2$, meaningful comparisons emerge at 50K iterations (batch 512, approx. 20 epochs); the GAT-XL/2 configuration achieves SOTA at 60 epochs.

### Key Designs

1. **Pure Transformer GAN Architecture on VAE Latent Space**:
    - Function: Provides a "minimal architectural shift" scalable GAN backbone where both G and D are standard ViTs without convolutional or multi-scale structures.
    - Mechanism: The generator uses unpatchify as a linear RGB decoding head with output dimensions growing with $p^2$. ViT blocks insert adaptive RMSNorm + LayerScale driven by style vector $w$, where $\gamma, \alpha$ are initialized near zero to ensure early stability. The discriminator prepends a `[cls]` token to the patch sequence, with the final `[cls]` projecting to real/fake logits. All modulations are intentionally "lightweight feature modulations" to keep the backbone close to original ViTs, inheriting proven scalability across width, depth, data, and compute.
    - Design Motivation: Previous Transformer-GAN works (TransGAN, HiT, etc.) introduced numerous non-ViT modifications for training stability, which undermined Transformer scaling advantages. The authors sought to prove pure Transformer GANs scale in latent space first, keeping architectural changes minimal and leaving remaining issues to the orthogonal MNG and lr scaling modules.

2. **Multi-level Noise-perturbed image Guidance (MNG)**:
    - Function: Solves the idle early generator layer problem by inserting auxiliary outputs at intermediate depths and having the discriminator scrutinize these intermediate images across multiple noise levels, forcing each layer to make substantial contributions.
    - Mechanism: The generator is divided into $K$ stages, each accumulating an intermediate image $\hat{x}_k$ via residual connections, denoted as $G(z,c) = [\hat{x}_1, \hat{x}_2, \dots, \hat{x}_K]$. Predefined Gaussian noise is added to each $\hat{x}_k$: $\mathcal{E}(\hat{x}_k) = \alpha_k \hat{x}_k + \sqrt{1-\alpha_k^2}\,\epsilon$, where $\alpha_1 < \alpha_2 < \dots < \alpha_K = 1$ follows a monotonic exponential schedule. Thus, shallower outputs receive stronger noise while deeper ones approach cleanliness. The discriminator receives the sequence of $\mathcal{E}(\hat{x}_k)$, while real images are similarly perturbed at each noise level. This forces early layers to match coarse structures under heavy noise while deep layers handle fine details; noise perturbation (unlike MSG-GAN's resizing) prevents the discriminator from using "cross-scale consistency" shortcuts.
    - Design Motivation: Empirical evidence showed that vanilla GAT early block PCA features were stagnant and LPIPS impact was minimal, wasting expanded capacity. Directly applying MSG-GAN's multi-scale supervision proved detrimental as it allowed the discriminator to exploit cross-scale alignment shortcuts. Multi-noise levels on a single image size provide per-layer gradients without shortcuts and minimal computational overhead.

3. **Width-aware Learning Rate Scaling (width-aware lr rule)**:
    - Function: Enables stable training from S to XL using a single set of base hyperparameters (adjusting only lr by channel count), avoiding per-scale manual tuning.
    - Mechanism: Observations show that as ViT inputs per layer are normalized to unit variance, the expected squared norm of inputs is proportional to the number of channels $C$. Consequently, output changes per parameter update are also proportional to $C$. To maintain consistent output update magnitudes across widths, the lr must be inversely proportional to the channel count: $\eta_{\text{adapt}} = \eta_{\text{base}} \cdot C_{\text{base}} / C_{\text{model}}$, where $\eta_{\text{base}}$ is the tuned lr for the base model ($C_{\text{base}}$). All other hyperparameters (batch size, optimizer, loss weights) remain constant.
    - Design Motivation: While DiT-like diffusion models can share hyperparameters across scales, GANs often diverge. This rule was strictly validated via ablation: training GAT-S with GAT-B's $\eta_{\text{adapt}}$ leads to divergence, while training GAT-B with GAT-S's $\eta_{\text{adapt}}$ results in slow convergence. This rule is orthogonal to the $\sqrt{}$-scaling rule for large batches—combining both allowed for $1/4$ the iterations with $4\times$ batch size.

### Loss & Training
The discriminator loss is an approximated relativistic pairing loss plus dual gradient penalties and REPA alignment: $\mathcal{L}_D = \mathcal{L}_D^{\text{adv}} + \lambda_{\text{aGP}}(\mathcal{L}_{\text{aR1}} + \mathcal{L}_{\text{aR2}}) + \lambda_{\text{REPA}} \mathcal{L}_{\text{REPA}}$, where $\mathcal{L}_{\text{aR}}$ approximates the true penalty as $\frac{1}{\sigma^2}\|D(\mathcal{E}(x),c) - D(\mathcal{E}(x+\epsilon'),c)\|^2$ for efficiency. $\mathcal{L}_{\text{REPA}} = \frac{1}{N+1}\sum_i \text{sim}(P(h_i), \hat{h}_i)$ aligns the discriminator's `[cls]` and patch tokens with a frozen DINOv2 teacher. The generator optimizes only $\mathcal{L}_G^{\text{adv}}$. All $x$ and $G(z,c)$ are MNG-processed multi-level noise-perturbed versions.

## Key Experimental Results

### Main Results

SOTA comparison for ImageNet-256 class-conditional single-step generation (FID-50K):

| Type | Method | Params | NFE | Epoch | FID |
|--------|------|------|------|---------|------|
| 2-NFE flow | MeanFlow-XL/2 | 676M | 2 | 240 | 2.93 |
| 1-NFE flow | MeanFlow-XL/2 | 676M | 1 | 240 | 3.43 |
| 1-NFE flow | Shortcut-XL/2 | 675M | 1 | 250 | 10.60 |
| 1-NFE GAN | BigGAN | 112M | 1 | - | 6.95 |
| 1-NFE GAN | GigaGAN | 569M | 1 | 480 | 3.45 |
| 1-NFE GAN | StyleGAN-XL† | 166M | 1 | - | 2.30 |
| **1-NFE GAN** | **GAT-XL/2** | **602M** | **1** | **60** | **2.18** |

† StyleGAN-XL uses an ImageNet pre-trained discriminator; FID is biased lower than actual image quality.
GAT-XL/2 improves FID from MeanFlow's 3.43 to 2.18 while using 1/4 of the training epochs.

Scalability curves (Fig. 3): FID-50K decreases monotonically with (a) model size (S→XL); (b) smaller patch sizes; (c)(d) correlation of $-0.95$ between FID and inference GFLOPs, fitting a power law: $\text{FID}(C) \approx 3.52 \times 10^5 \cdot C^{-0.456}$.

Extra metrics (Tab. 3): 60-epoch GAT-XL/2 reduces CLIP-FID from StyleGAN-XL's 2.62 to 1.86 and achieves higher Recall (0.572 vs 0.530), suggesting improvements are not just "overfitting Inception features."

### Ablation Study

| Config | FID Trend | Description |
|------|---------|------|
| Full GAT (MNG-exp) | Best | Default configuration |
| w/o MNG | Significantly Worse | Layer inactivation and performance collapse |
| MSG (Resize multi-scale) | Worst | Cross-scale consistency shortcut suppresses quality |
| MNG-lin (Linear schedule) | Inferior to exp | Exponential schedule is superior |
| LR Mismatch (S w/ B's lr / B w/ S's lr) | Severe Degradation | GAT-S slow convergence; GAT-B diverges |
| w/o REPA | Noticeable Drop | Aligning VFM on D alone significantly boosts G |

Decoupled G/D scaling (Fig. 6a): Scaling the discriminator alone provides significantly higher gains than scaling the generator alone. CKNNA metrics show D's alignment with DINOv2-g is higher on fake data than real data, implying generation quality is bounded by D's representation quality.

### Key Findings
- **MNG's contribution comes from "multi-noise on single image" rather than "multi-scale images"**: MSG-GAN style multi-scale supervision performed worst on GAT, as cross-scale consistency shortcuts limit G. MNG provides per-layer gradients without shortcuts.
- **GAN lr scaling depends heavily on width**: DiT's "one-size-fits-all" hyperparameter convenience does not hold for GANs; explicit scaling via $\eta \propto 1/C_{\text{model}}$ is necessary for stability.
- **Discriminator representation is the true bottleneck**: Decoupled scaling shows larger gains from D, and REPA alignment on D significantly boosts G. This points towards focusing on discriminator representations for future GAN research.
- **Scalability is monotonic and follows a power law**: FID-50K fits $\text{FID}(C) \approx 3.52 \times 10^5 \cdot C^{-0.456}$ against training GFLOPs, similar to diffusion/AR models.

## Highlights & Insights
- **"Harvesting Diffusion dividends for GANs"**: GAT integrates recent scaling successes (VAE latent, pure ViT, REPA) into GANs while retaining single-step inference and semantic latent spaces.
- **MNG as a "Noised Version" of MSG-GAN**: Replacing multi-scale image hierarchies with single-image multi-noise levels retains intermediate supervision while filtering out shortcut side effects.
- **Width-aware LR rule has both theory and practicality**: $\eta \propto 1/C$ stems from the observation that input norms $\propto C$, meaning updates $\propto \eta \cdot C$. This directly enables scaling in lr-sensitive GAN training.
- **"Scaling D is more cost-effective than scaling G"**: Contrary to traditional focus on G capacity, empirical evidence shows G is limited by the gradient quality provided by D.

## Limitations & Future Work
- **Performance Gap**: While SOTA among single-step models, GAT-XL/2 (FID 2.18) still trails 1.5B AR models or multi-step diffusion (SiT+REPA FID 1.35).
- **VAE Bottleneck**: Reliance on SD-VAE as a frozen tokenizer limits quality to the VAE's reconstruction ceiling; stronger VAEs should improve FID.
- **Task Scaling**: Experiments are focused on ImageNet; text-to-image scaling remains the ultimate battlefield for verifying scalability.
- **MNG Schedule**: The exponential $\alpha_k$ schedule is handcrafted; systematic searches might reveal better couplings with depth.

## Related Work & Insights
- **vs StyleGAN-XL / GigaGAN**: Prior large GANs relied on convolutions, progressive growth, or massive T2I data. GAT follows a pure ViT + VAE latent + systematic scaling recipe, proving "correct recipes" outweigh "brute force epochs."
- **vs DiT / SiT**: Shares VAE latent + pure ViT backbone but uses GAN objectives for 1-NFE inference. GAT introduces MNG and lr scaling to handle GAN-specific instability that DiT does not face.
- **vs R3GAN**: Inherits the relativistic pairing loss + dual gradient penalty objective but swaps convolutions for pure Transformers, proving stability in the ViT-GAN context.
- **vs MeanFlow / Shortcut**: 1-NFE flow/diffusion distillation routes. GAT trains directly without a multi-step teacher, avoiding teacher quality ceilings and achieving superior FID (2.18 vs 3.43).

## Rating
- Novelty: ⭐⭐⭐⭐ MNG is an elegant variant of MSG-GAN; width-aware lr aligns with prior scaling logic, but integrating these into a systematic GAN scaling law is a first.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive scaling curves across scales/patches, power law fitting, decoupled G/D analysis, CKNNA, and ImageNet-512 validation.
- Writing Quality: ⭐⭐⭐⭐ Clear logic; "two failure modes + two solutions" narrative is well-supported by evidence.
- Value: ⭐⭐⭐⭐⭐ Provides the first reproducible recipe for scaling GANs, demonstrating that single-step generation can be high-quality and scalable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Flatten Graphs as Sequences: Transformers are Scalable Graph Generators](../../NeurIPS2025/image_generation/flatten_graphs_as_sequences_transformers_are_scalable_graph_generators.md)
- [\[ICML 2026\] Krause Synchronization Transformers](krause_synchronization_transformers.md)
- [\[ICML 2026\] DiScoFormer: Plug-In Density and Score Estimation with Transformers](discoformer_plug-in_density_and_score_estimation_with_transformers.md)
- [\[ICML 2026\] Diagnosing and Correcting Concept Omission in Multimodal Diffusion Transformers](diagnosing_and_correcting_concept_omission_in_multimodal_diffusion_transformers.md)
- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](../../ICLR2026/image_generation/speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)

</div>

<!-- RELATED:END -->
