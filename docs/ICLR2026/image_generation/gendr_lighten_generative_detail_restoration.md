---
title: >-
  [Paper Note] GenDR: Lighten Generative Detail Restoration
description: >-
  [ICLR 2026][Image Generation][Single-step super-resolution] GenDR is proposed as a lightweight single-step diffusion super-resolution model for generative detail restoration. It identifies the fundamental divergence between T2I and SR objectives (T2I requires multi-step + 4-channel vs. SR requires fewer steps + 16-channel) → builds a customized SD2.1-VAE16 foundation model (0.9B, extending the latent space via REPA representation alignment without increasing model size) → introduces CiD/CiDA consistency score identity distillation (integrating SR-specific priors into score distillation + adversarial learning + representation alignment) → delivers a minimal pipeline containing only UNet + VAE, achieving 77ms inference while surpassing existing SOTA on all quality and efficiency metrics.
tags:
  - ICLR 2026
  - Image Generation
  - Single-step super-resolution
  - latent space expansion
  - score distillation
  - VAE 16-channel
  - consistency distillation
date: 2026-05-08
content_hash: c8aa56f8850de273
---

# GenDR: Lighten Generative Detail Restoration

**Conference**: ICLR 2026
**arXiv**: [2503.06790](https://arxiv.org/abs/2503.06790)
**Code**: None
**Area**: Image Generation
**Keywords**: Single-step super-resolution, latent space expansion, score distillation, VAE 16-channel, consistency distillation

## TL;DR
GenDR is proposed as a lightweight single-step diffusion super-resolution model for generative detail restoration. It identifies the fundamental divergence between T2I and SR objectives (T2I requires multi-step + 4-channel vs. SR requires fewer steps + 16-channel) → builds a customized SD2.1-VAE16 foundation model (0.9B, extending the latent space via REPA representation alignment without increasing model size) → introduces CiD/CiDA consistency score identity distillation (integrating SR-specific priors into score distillation + adversarial learning + representation alignment) → delivers a minimal pipeline containing only UNet + VAE, achieving 77ms inference while surpassing existing SOTA on all quality and efficiency metrics.

## Background & Motivation

**State of the Field**: Diffusion model-based real-world super-resolution (SR) has achieved remarkable progress, substantially outperforming GAN-based methods in quality, yet suffers from slow inference speed and bottlenecks in detail fidelity.

**Root Cause**: T2I and SR tasks exhibit a fundamental objective divergence — T2I generates complete images from noise and requires multi-step inference with a low-dimensional latent space (4-channel VAE to reduce generation difficulty), whereas SR only needs to supplement high-frequency details with fewer diffusion steps but requires a larger latent space (16-channel VAE) to preserve input information.

**Limitations of Prior Work**: Accelerating inference (e.g., OSEDiff single-step distillation) leads to significant quality degradation, while improving quality (e.g., DreamClear using PixArt-α + ControlNet) introduces enormous computational overhead, resulting in a quality–efficiency dilemma.

**Oversized Models**: Existing 16-channel VAE diffusion models (e.g., FLUX 12B, SD3.5) are excessively large for SR tasks — single-step 4× SR with FLUX requires >40 GB VRAM and 1.4s runtime, which is 5.3× and 11.4× that of SD2.1, respectively.

**Distillation Method Deficiencies**: Existing score distillation methods (VSD/SiD) are designed for T2I; directly applying them to SR leads to quality and content inconsistencies due to training distribution mismatch and over-reliance on imperfect score functions.

**Starting Point**: SR tasks require a customized foundation model (16-channel + appropriate scale of 0.9B) + a customized distillation method (CiD incorporating SR priors) + a minimalist inference pipeline.

## Method

### Key Design 1: SD2.1-VAE16 — Customized 16-Channel Latent Space Foundation Model

- **Function**: Builds a 0.9B foundation diffusion model suitable for SR tasks, based on the SD2.1 UNet and an open-source 16-channel VAE.
- **Mechanism**: Full-parameter training is conducted via a representation alignment (REPA) strategy, inserting an MLP projection head after the first downsampling block of the UNet to align intermediate UNet features $\mathbf{h}_t = f_\theta(\mathbf{z}_t)$ with representations $\mathbf{h}_\mathcal{E} = \mathcal{E}(\mathbf{x}_h)$ from a pretrained DINOv2 encoder:

$$\mathcal{L}^{(\text{repa})} = -\mathbb{E}_{\mathbf{x}_h, t}\left[\frac{1}{N}\sum_{n=1}^{N}\text{sim}\left(\mathbf{h}_\mathcal{E}[n], h(\mathbf{h}_t[n])\right)\right]$$

- **Design Motivation**: Although the 4-channel VAE suits T2I by reducing generation difficulty, it discards fine details and structural information for SR due to irreversible compression loss. The 16-channel VAE provides greater information capacity. Directly using 16-channel DiT models such as FLUX is prohibitively large; thus, building upon the lightweight SD2.1 achieves the optimal balance.

### Key Design 2: CiD — Consistency Score Identity Distillation

- **Function**: Distills multi-step diffusion into a single step while integrating SR task-specific priors to ensure training stability and output consistency.
- **Mechanism**: Two key modifications are made upon SiD: (1) the HR target image $\mathbf{z}_h$ is used to train the "real" score network $\phi$, aligning its output distribution with the high-fidelity image manifold; (2) $\mathbf{z}_h$ replaces the generated result $\mathbf{z}_g$ as the identity transformation, alleviating instability caused by fluctuations in generation quality. The final CiD loss is:

$$\mathcal{L}_\theta^{(\text{cid})} = \mathcal{L}_\theta^{(3)} - \xi \mathcal{L}_\theta^{(1)}$$

where $\mathcal{L}_\theta^{(3)}$ applies CFG-enhanced guidance with $\mathbf{z}_h$ as the target, $\mathcal{L}_\theta^{(1)}$ is the original SiD loss, and $\xi$ is an empirical weight.

- **Design Motivation**: Directly applying T2I-oriented VSD/SiD to SR introduces training distribution mismatch (T2I aligns text embeddings vs. SR aligns image embeddings), leading to quality and content inconsistencies. By optimizing the "real" score network with HR ground truth and introducing the identity transformation, SR priors are incorporated into the distillation process.

### Key Design 3: CiDA — Integrating Adversarial Learning and Representation Alignment

- **Function**: Introduces adversarial learning and representation alignment on top of CiD to further enhance perceptual quality and accelerate training.
- **Mechanism**: The pretrained UNet $\phi$ serves as a feature extractor with a discriminator head $h$ for adversarial training, alongside REPA regularization:

$$\mathcal{L}_\theta^{(\text{cida})} = \lambda_1 \mathcal{L}_\theta^{(\text{cid})} + \lambda_2 \mathcal{L}_\theta^{(\text{adv})} + \lambda_3 \mathcal{L}_\theta^{(\text{repa})}$$

Implementation employs LoRA adaptation (rank=64, alpha=128) and a model-sharing strategy (sharing the base model for the score network and discriminator feature extraction), substantially reducing memory and computation.

- **Design Motivation**: Pure distillation tends to produce an AI-generated "artificial" appearance; adversarial learning enforces details that follow the real data distribution. REPA regularizes in the high-level semantic space to prevent structural bias while accelerating convergence.

### Key Design 4: Minimalist Inference Pipeline

- **Function**: Constructs a minimalist inference pipeline consisting solely of VAE + UNet.
- **Mechanism**: The scheduler is removed (fixed $\bar{\alpha}_t = \bar{\beta}_t = 0.5$), the text encoder and tokenizer are removed, and precomputed fixed prompt embeddings are used as substitutes. This achieves 77ms per 512² image on an A100.
- **Design Motivation**: Single-step inference does not require multi-step scheduler scheduling; fixed prompt embeddings provide universal quality descriptions for SR tasks without affecting IQA performance (MUSIQ drops only 0.17, while saving approximately 30% of parameters and 15ms of inference time).

## Key Experimental Results

### Table 1: Quantitative Comparison on Synthetic Dataset ImageNet-Test (×4 SR)

| Method | Steps | PSNR↑ | NIQE↓ | LIQE↑ | ClipIQA↑ | MUSIQ↑ | Q-Align↑ |
|------|------|-------|-------|-------|----------|--------|----------|
| Real-ESRGAN | GAN | 26.62 | 4.49 | 3.84 | 0.509 | 64.81 | 3.423 |
| DiffBIR-50 | 50 | 25.45 | 4.93 | 4.64 | 0.749 | 73.04 | 4.323 |
| DreamClear-50 | 50 | 24.76 | 5.38 | 4.43 | 0.765 | 70.08 | 4.092 |
| OSEDiff-1 | 1 | 24.82 | 4.28 | 4.56 | 0.678 | 71.74 | 4.067 |
| InvSR-1 | 1 | 23.81 | 4.39 | 4.56 | 0.711 | 72.38 | 3.987 |
| **GenDR-1** | **1** | 24.14 | **4.13** | **4.81** | 0.740 | **74.68** | **4.361** |

### Table 2: Quantitative Comparison on Real-World Dataset RealSet80

| Method | Inference Time | NIQE↓ | LIQE↑ | ClipIQA↑ | MUSIQ↑ | Q-Align↑ |
|------|----------|-------|-------|----------|--------|----------|
| StableSR-50 | 3731ms | 3.40 | 3.85 | 0.740 | 67.58 | 4.087 |
| SeeSR-50 | 6359ms | 4.37 | 4.28 | 0.712 | 69.74 | 4.306 |
| DreamClear-50 | 6892ms | 3.73 | 3.96 | 0.724 | 67.22 | 4.121 |
| OSEDiff-1 | 103ms | 3.98 | 4.13 | 0.704 | 69.19 | 4.306 |
| InvSR-1 | 115ms | 4.03 | 4.29 | 0.727 | 69.79 | 4.301 |
| **GenDR-1** | **77ms** | **3.98** | **4.52** | **0.742** | **71.57** | **4.453** |

### Ablation Study: Distillation Strategies (RealSet80)

| Base Model | Distillation | LIQE↑ | ClipIQA↑ | MUSIQ↑ | Q-Align↑ |
|----------|----------|-------|----------|--------|----------|
| SD2.1-VAE4 | VSD | 4.13 | 0.704 | 69.19 | 4.306 |
| SD2.1-VAE4 | CiDA | 4.32 | 0.723 | 70.13 | 4.386 |
| SD2.1-VAE16 | VSD | 4.12 | 0.691 | 68.82 | 4.373 |
| SD2.1-VAE16 | SiD | 4.25 | 0.702 | 69.33 | 4.391 |
| SD2.1-VAE16 | CiD | 4.44 | 0.715 | 70.61 | 4.428 |
| SD2.1-VAE16 | **CiDA** | **4.52** | **0.742** | **71.57** | **4.453** |

## Key Findings

1. **The objective divergence between T2I and SR is the root cause**: T2I must generate all content from noise → multi-step + 4-channel; SR only supplements high-frequency details → fewer steps + 16-channel. Directly repurposing T2I models for SR is a suboptimal solution.
2. **The 16-channel VAE is critical for SR**: Even on the 0.9B small model, VAE16 preserves more detail and structural information than VAE4. VAE16 shows a slight drop on T2I tasks (GenEval −0.02, FID +14.44) but is significantly superior for SR tasks.
3. **CiDA delivers consistent incremental gains**: VSD → SiD → CiD → CiDA improves Q-Align from 4.373 → 4.391 → 4.428 → 4.453, with each step making a clear contribution (CiD contributes 0.05, adversarial contributes 0.03).
4. **Fixed prompt embeddings do not sacrifice quality**: Compared to dynamically generated prompts from DAPE/Qwen2.5VL, fixed embeddings reduce MUSIQ by only 0.17, while decreasing inference time from 113ms/3.18s to 77ms and parameter count from 1775M/8.3B to 933M.
5. **Pareto-optimal speed–quality trade-off**: GenDR achieves the best performance on all NR-IQA metrics at 77ms (fastest) with 933M parameters (second smallest), accelerating inference by 89.5× over DreamClear with half the parameters.

## Highlights & Insights

- **Insightful problem identification**: The paper is the first to systematically analyze the objective divergence between T2I and SR tasks (in terms of step requirements and latent space dimensionality), providing a theoretical basis for SR-customized diffusion models.
- **Systematic solution**: Comprehensive optimization is achieved across three levels — foundation model (VAE16) → distillation method (CiD/CiDA) → inference pipeline (minimalist design).
- **Outstanding efficiency**: Single-step inference at 77ms with 933M parameters, nearly 90× faster than multi-step methods and 25–33% faster than OSEDiff/InvSR.
- **CiDA training efficiency**: LoRA + model-sharing strategy enables efficient joint training of three UNets.

## Limitations & Future Work

- **Larger-channel VAEs are unexplored**: While the effectiveness of 16 channels is validated, larger latent spaces such as 32/64 channels are not investigated, as training an entire SD model at such scales incurs prohibitive cost.
- **High VRAM requirements for CiDA**: Despite LoRA and DeepSpeed optimizations, CiDA still demands substantial GPU memory, making it difficult to scale to DiT models such as FLUX/SD3.5.
- **PSNR is not competitive**: GenDR leads on perceptual quality metrics (LIQE/MUSIQ/Q-Align) but achieves lower PSNR than GAN-based methods and some multi-step approaches, indicating a trade-off in pixel-level fidelity.

## Related Work & Insights

| Dimension | OSEDiff (Wu et al., 2024b) | GenDR (Ours) |
|------|---------------------------|-------------|
| Base Model | SD2.1 (4-channel VAE, 0.9B) | SD2.1-VAE16 (16-channel VAE, 0.9B) |
| Distillation | Direct VSD + L1/MSE regularization | CiDA: SR priors + adversarial + REPA |
| Inference Time | 103ms | 77ms |
| Q-Align | 4.306 | 4.453 |

| Dimension | DreamClear (Ai et al., 2025) | GenDR (Ours) |
|------|------------------------------|-------------|
| Base Model | PixArt-α (2.2B) | SD2.1-VAE16 (0.9B) |
| Inference Steps | 50 steps | 1 step |
| Auxiliary Modules | 2× ControlNet + MLLM | None (fixed embeddings) |
| Inference Time | 6892ms (3× A100) | 77ms (1× A100) |
| MUSIQ | 67.22 | 71.57 |

## Rating (1–5)

- **Novelty**: 4 — The insight into T2I/SR objective divergence is novel, and the integration of SR priors into score distillation via CiD constitutes an original contribution; however, the overall framework remains an improved composition of existing components (SiD/REPA/LoRA).
- **Technical Depth**: 4 — The mathematical derivation of CiD/CiDA is rigorous, the progression from VSD → SiD → CiD is logically coherent, and each design decision is validated through ablation studies.
- **Experimental Thoroughness**: 4 — Coverage includes both synthetic and real-world datasets, 13 IQA metrics, user studies, MLLM-based evaluation, and detailed ablations (distillation strategies/VAE channels/prompt strategies); downstream task evaluation is nonetheless absent.
- **Writing Quality**: 4 — Motivation is articulated clearly (Fig. 2 provides intuitive visualization) and the methodological derivation proceeds in a well-structured manner, though the dense notation requires careful reading.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Eliminating VAE for Fast and High-Resolution Generative Detail Restoration](eliminating_vae_for_fast_and_high-resolution_generative_detail_restoration.md)
- [\[ICLR 2026\] Bridging Degradation Discrimination and Generation for Universal Image Restoration](bridging_degradation_discrimination_and_generation_for_universal_image_restorati.md)
- [\[ICLR 2026\] LVTINO: LAtent Video consisTency INverse sOlver for High Definition Video Restoration](lvtino_latent_video_consistency_inverse_solver_for_high_definition_video_restora.md)
- [\[AAAI 2026\] Multi-Metric Preference Alignment for Generative Speech Restoration](../../AAAI2026/image_generation/multi-metric_preference_alignment_for_generative_speech_restoration.md)
- [\[ICLR 2026\] GenCP: Towards Generative Modeling Paradigm of Coupled Physics](gencp_towards_generative_modeling_paradigm_of_coupled_physics.md)

<!-- RELATED:END -->
