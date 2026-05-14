---
title: >-
  [Paper Note] One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers
description: >-
  [CVPR2026][Image Generation][Diffusion Transformer] This paper proposes ELIT (Elastic Latent Interface Transformer), which inserts variable-length latent interfaces and lightweight Read/Write cross-attention layers into…
tags:
  - "CVPR2026"
  - "Image Generation"
  - "Diffusion Transformer"
  - "elastic inference"
  - "latent interface"
  - "adaptive computation"
  - "multi-budget model"
  - "cross-attention"
date: 2026-05-08
content_hash: 79985b582be5b457
---

# One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers

**Conference**: CVPR2026
**arXiv**: [2603.12245](https://arxiv.org/abs/2603.12245)
**Code**: [snap-research/elit](https://snap-research.github.io/elit)
**Area**: Image Generation
**Keywords**: Diffusion Transformer, elastic inference, latent interface, adaptive computation, multi-budget model, cross-attention

## TL;DR

This paper proposes ELIT (Elastic Latent Interface Transformer), which inserts variable-length latent interfaces and lightweight Read/Write cross-attention layers into DiT, enabling a single model to dynamically adjust its computational budget at inference time while non-uniformly allocating computation to more difficult image regions, achieving up to 53% FID reduction on ImageNet 512px.

## Background & Motivation

1. **Rigid computation in DiT**: The per-step FLOPs of Diffusion Transformers are fixed as a function of input resolution, offering no flexibility to trade off latency against quality.
2. **Uniform computation allocation wastes resources**: DiT distributes computation uniformly across all spatial tokens, despite significant variation in generation difficulty across image regions (e.g., simple backgrounds vs. complex textures).
3. **Zero-padding experiment**: The authors pad images with zero-valued patches to increase token count; FID of DiT-B/2-Synth does not improve, demonstrating that DiT cannot redirect surplus computation toward informative regions.
4. **Limitations of prior methods**: Masking approaches (MaskDiT, MDTv2) accelerate training but still require full tokens at inference; training-free acceleration methods do not improve training efficiency; adaptive generators (FlexiDiT) still allocate computation uniformly or incur excessive complexity.
5. **Variable-length representations limited to autoencoders**: Methods such as FlexTok and TiTok learn variable-length representations at the encoder stage but do not extend elasticity into the generative model itself.
6. **RIN/FIT depart from DiT architecture**: Latent-token methods (RIN, FIT) enable non-uniform computation allocation but substantially alter the architectural design, hindering adoption within the mainstream DiT ecosystem.

## Method

### Overall Architecture

ELIT introduces a three-stage structure into the standard DiT architecture. At inference time, the number of latent tokens $K$ serves as a user-controllable knob that directly determines per-step FLOPs:

- **Spatial Head** ($B_{\text{in}}$ blocks): Processes patchified spatial tokens and extracts preliminary features.
- **Read layer**: A lightweight cross-attention operation that reads information from spatial tokens into the variable-length latent interface $l \in \mathbb{R}^{K \times d}$, prioritizing high-loss (difficult) regions.
- **Latent Core** ($B_{\text{core}}$ standard transformer blocks): Performs the majority of computation in the latent domain.
- **Write layer**: Broadcasts updated latent information back to spatial tokens.
- **Spatial Tail** ($B_{\text{out}}$ blocks): Recovers spatial detail and produces the velocity prediction.

### Key Designs

1. **Read/Write cross-attention**: Read executes cross-attention with latent tokens as queries and spatial tokens as keys/values, followed by an MLP; Write is fully symmetric. Pre-norm with adaLN-Zero modulation preserves timestep awareness, and QK normalization enhances training stability.
2. **Grouped Cross-Attention**: Spatial tokens are partitioned into $G$ non-overlapping groups (e.g., a 4×4 grid), with $J = K/G$ latent tokens assigned to each group; cross-attention is restricted within groups, reducing complexity from $\mathcal{O}(NK)$ to $\mathcal{O}(NK/G)$.
3. **Tail Dropping**: During training, the number of retained latent tokens $\tilde{J}$ is sampled each iteration from $\text{Uniform}\{J_{\min}, \ldots, J_{\max}\}$, and tail tokens are dropped. Head tokens are trained more frequently, naturally inducing an importance ordering—earlier tokens capture global structure while later tokens refine details.
4. **Cheap Classifier-Free Guidance (CCFG)**: Exploiting the multi-budget property, the main term uses the full budget $\tilde{J}$ while the guidance term uses a low budget $\tilde{J}_w$ without class conditioning, combining the advantages of AutoGuidance and CFG. This reduces inference FLOPs by approximately 33% with superior quality.

### Loss & Training

The standard Rectified Flow objective is used directly, without any auxiliary losses:

$$\mathcal{L}_{\text{RF}} = \mathbb{E}_{t, \mathbf{X}_1, \mathbf{X}_0} \| \mathcal{G}(\mathbf{X}_t, t) - (\mathbf{X}_1 - \mathbf{X}_0) \|_2^2$$

where $t$ is sampled from a logit-normal distribution. During multi-budget training, the batch size is increased from 256 to 384 to compensate for compute saved in low-budget iterations, keeping training FLOPs comparable. Models are trained for 500k steps (images) / 200k steps (video) with learning rate $10^{-4}$, 10k warmup steps, gradient clipping of 1.0, and EMA decay $\beta=0.9999$.

## Key Experimental Results

### Main Results on ImageNet-1K (Table 1)

| Model | Resolution | FID↓ (–G/+G) | FDD↓ (–G/+G) | IS↑ (–G/+G) |
|-------|------------|---------------|---------------|--------------|
| DiT-XL | 512px | 18.8 / 9.5 | 339.2 / 233.6 | 53.0 / 86.4 |
| ELIT-DiT MB | 512px | **10.1** / **4.5** | **164.1** / **98.2** | **88.8** / **147.0** |
| U-ViT-XL | 512px | 11.6 / 5.3 | 202.7 / 125.9 | 72.5 / 117.2 |
| ELIT-U-ViT MB | 512px | **7.7** / **3.8** | **135.8** / **83.1** | **98.0** / **159.3** |
| HDiT-XL | 512px | 13.0 / 6.0 | 260.3 / 170.5 | 69.4 / 114.2 |
| ELIT-HDiT MB | 512px | **9.6** / **4.6** | **171.2** / **106.8** | **94.7** / **154.6** |

At 512px, ELIT-MB achieves FID improvements of **53%, 28%, and 23%** over DiT, U-ViT, and HDiT, respectively.

### Video Generation (Kinetics-700, Table 2)

| Model | FID↓ (–G/+G) | FDD↓ (–G/+G) | FVD↓ (–G/+G) |
|-------|---------------|---------------|---------------|
| DiT-XL | 14.0 / 11.3 | 371.5 / 309.1 | 135.9 / 100.5 |
| ELIT-DiT | **13.3** / **10.7** | **277.4** / **222.0** | **116.5** / **90.5** |

### Ablation Study

- **Group size**: A 4×4 grouping is optimal at 256px; 8×8 is optimal at 512px; performance degrades with 1×1 (per-token) or full-image grouping.
- **Block allocation**: The optimal configuration for DiT-XL is 4-20-4 (head-core-tail), placing approximately 71% of blocks in the latent core; for DiT-B the optimum is 3-6-3 (~67%).
- **Tail dropping vs. random dropping**: Tail dropping (importance ordering) achieves FID 26.6, outperforming random dropping at 27.0; the gap widens at 25% token retention (36.3 vs. 38.6).
- **Convergence acceleration**: ELIT-DiT achieves 3.3× and 4.0× convergence speedup at 256px and 512px, respectively.
- **Large-scale validation**: Fine-tuning the 20B-parameter Qwen-Image (MM-DiT) for 120k steps, ELIT reduces FLOPs by up to 63% (~2.7× speedup) while the DPG-Bench average score drops only marginally from 90.45 to 88.02.
- **Compatibility with TeaCache**: ELIT is composable with the training-free acceleration method TeaCache, yielding additional speedup comparable to that obtained on plain DiT.
- **Scaling behavior**: ELIT consistently outperforms DiT across model sizes from DiT-S/4 to DiT-XL/2, with larger models showing greater gains and smaller relative overhead from the added Read/Write layers.

## Highlights & Insights

- **Minimal design, plug-and-play**: Only two cross-attention layers (Read/Write) are added; the Rectified Flow training objective and DiT architecture remain unchanged, making ELIT applicable to DiT, U-ViT, HDiT, and MM-DiT.
- **Single model, multiple budgets at inference**: Controlling the number of latent tokens enables smooth quality–compute trade-offs, with up to 60 distinct budget points available at 512px.
- **Adaptive computation allocation**: Read attention automatically focuses on high-loss regions, achieving non-uniform computation distribution, as clearly demonstrated by the zero-padding experiment.
- **AutoGuidance for free**: The multi-budget property naturally provides a weak-model variant; CCFG reduces inference cost by ~33% with superior quality, requiring no additional training or manual model degradation.
- **Substantial convergence speedup**: ELIT-DiT converges 3.3–4.0× faster, significantly outperforming the baseline under equal training FLOPs.
- **Comprehensive experimental coverage**: Experiments span images (256px/512px), video (Kinetics-700), large-scale MM-DiT (20B Qwen-Image), multiple architectures, and thorough ablations.
- **Compelling motivating experiments**: The zero-padding comparison elegantly exposes DiT's uniform computation problem, and attention map visualizations intuitively illustrate ELIT's computation redistribution.

## Limitations & Future Work

1. **Large-scale training from scratch unverified**: The Qwen-Image experiments use a fine-tuning and distillation setup; benefits of training a 20B-scale model from scratch remain unclear and may involve different convergence challenges.
2. **CCFG saturation**: CCFG tends to saturate images more quickly than standard CFG, potentially causing color oversaturation in certain scenarios.
3. **Open-domain text-to-image evaluation absent**: Experiments are primarily conducted under ImageNet class-conditional settings; text-to-image performance remains to be validated.
4. **Per-step budget scheduling unexplored**: Different sampling steps may benefit from different computational budgets (e.g., less computation in early steps), but a fixed per-step budget is used throughout; this is left for future work.
5. **Fixed grouping strategy**: Groups are defined as regular grids; content-adaptive grouping strategies are not explored.
6. **Parameter overhead from Read/Write layers**: DiT-XL grows from 675M to 698M parameters (+3.4%), which may be non-negligible for very large models.
7. **Comparison with advanced token compression methods absent**: Methods such as Token Merging and SparseDiT may be complementary to ELIT but are not empirically evaluated.

## Related Work & Insights

| Category | Representative Works | Distinction from ELIT |
|----------|---------------------|----------------------|
| Adaptive generators | FlexiDiT, Supernetwork | Uniform computation allocation or high router complexity; ELIT achieves non-uniform allocation via latent interfaces |
| Token-dropping for training acceleration | MaskDiT, MDTv2, TREAD | Full tokens must be restored at inference, precluding inference speedup; ELIT supports variable token counts at inference |
| Training-free acceleration | Token Merging, SparseDiT | Do not improve training efficiency; ELIT also accelerates training by 3–4× |
| Latent-token interfaces | RIN, FIT | Depart from DiT architecture and require specialized optimizers; ELIT is plug-and-play |
| Variable-length autoencoders | FlexTok, TiTok | Elasticity is limited to the encoder; ELIT extends elasticity into the generative model |

The core advantage of ELIT lies in its principle of "minimal modification, maximum compatibility"—it preserves the training objective, introduces no auxiliary losses, and requires no specialized optimizers, achieving inference elasticity and adaptive computation allocation through two cross-attention layers and a tail-dropping training strategy. This makes ELIT substantially easier to integrate into existing DiT ecosystems than any of the above alternatives.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Integrates variable-length latent interfaces into the DiT ecosystem with minimal intervention; tail dropping as an implicit importance-ordering mechanism is elegant and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four architectures × two resolutions × video × 20B large model × comprehensive ablations; coverage is exceptionally broad.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated (zero-padding experiment is highly convincing), structure is well-organized, and figures are informative.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses a fundamental bottleneck in DiT inference efficiency; the design is sufficiently simple for broad adoption, and multi-budget inference has strong practical utility.
- **Overall**: ⭐⭐⭐⭐⭐ — Solid work from Snap Research and Rice University, combining theoretical insight (uniform computation waste) with engineering value (plug-and-play acceleration); CVPR oral caliber.

> **Recommended reading**: The zero-padding experiment in Section 3.2 and the attention visualizations in Figure 2 are the most compelling parts of the paper, clearly demonstrating DiT's computational inefficiency and ELIT's solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Accelerating Diffusion Model Training under Minimal Budgets: A Condensation-Based Perspective](accelerating_diffusion_model_training_under_minimal_budgets_a_condensation-based.md)
- [\[ICLR 2026\] Latent Diffusion Model without Variational Autoencoder](../../ICLR2026/image_generation/latent_diffusion_model_without_variational_autoencoder.md)
- [\[CVPR 2026\] Circuit Mechanisms for Spatial Relation Generation in Diffusion Transformers](circuit_mechanisms_for_spatial_relation_generation_in_diffusion_models.md)
- [\[CVPR 2026\] Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers](just-in-time_training-free_spatial_acceleration_for_diffusion_transformers.md)
- [\[CVPR 2026\] EdgeDiT: Hardware-Aware Diffusion Transformers for Efficient On-Device Image Generation](edgedit_hardware-aware_diffusion_transformers_for_efficient_on-device_image_gene.md)

</div>

<!-- RELATED:END -->
