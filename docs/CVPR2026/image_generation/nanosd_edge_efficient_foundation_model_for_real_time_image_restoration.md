---
title: >-
  [Paper Note] NanoSD: Edge Efficient Foundation Model for Real Time Image Restoration
description: >-
  [CVPR 2026][Image Generation][Diffusion model distillation] This paper proposes NanoSD, a family of Pareto-optimal lightweight diffusion foundation models (130M–315M parameters…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Diffusion model distillation"
  - "edge deployment"
  - "image restoration"
  - "super-resolution"
  - "model compression"
  - "multi-objective optimization"
  - "Stable Diffusion"
date: 2026-05-08
content_hash: 1b0b65ed20ab3efd
---

# NanoSD: Edge Efficient Foundation Model for Real Time Image Restoration

**Conference**: CVPR 2026
**arXiv**: [2601.09823](https://arxiv.org/abs/2601.09823)  
**Code**: To be confirmed  
**Area**: 3D Vision
**Keywords**: Diffusion model distillation, edge deployment, image restoration, super-resolution, model compression, multi-objective optimization, Stable Diffusion

## TL;DR

This paper proposes NanoSD, a family of Pareto-optimal lightweight diffusion foundation models (130M–315M parameters, as fast as 12 ms inference) built upon SD 1.5 through hardware-aware U-Net decomposition, block-wise feature distillation, and multi-objective Bayesian optimization. NanoSD serves as a drop-in backbone that achieves state-of-the-art performance across multiple tasks including super-resolution, face restoration, deblurring, and monocular depth estimation.

## Background & Motivation

**Conflict between diffusion model restoration capability and deployment constraints**: Latent diffusion models such as SD 1.5 possess powerful generative priors that are highly valuable for image restoration; however, their full pipeline (U-Net + VAE) is computationally prohibitive for real-time inference on edge devices.

**Limitations of Prior Work**: Existing edge-efficient methods (AdcSR, TinySR, PocketSR, etc.) primarily target super-resolution using limited distillation datasets and fail to fully exploit the rich priors embedded in pretrained text-to-image (T2I) models, resulting in suboptimal architectures or poor perceptual detail.

**Theoretical FLOPs ≠ actual latency**: NPUs are optimized for specific operator patterns (e.g., GEMM); reducing FLOPs/GMACs alone does not guarantee proportional latency reduction, necessitating a hardware-aware perspective on architectural design.

**Lack of a unified conditioning mechanism**: Different restoration tasks require distinct conditioning strategies (LoRA, ControlNet, visual prompts, etc.), and existing lightweight models cannot flexibly accommodate these control plugins.

**Importance of preserving the latent space**: Most prior methods compress the denoising U-Net or shorten the diffusion trajectory, thereby disrupting the underlying latent manifold and limiting cross-task generalization.

**Lack of end-to-end pipeline co-optimization**: The majority of prior work compresses only the U-Net while neglecting the VAE encoder–decoder, leaving the overall pipeline unnecessarily heavyweight.

## Method

### Overall Architecture

NanoSD is built upon SD 1.5 and follows a five-stage pipeline: *decompose → distill → search → assemble → fine-tune*:

1. Perform hardware-aware decomposition of the SD 1.5 U-Net by removing the lowest-contributing deep encoder/decoder/middle blocks, and construct shape-compatible module variants for the retained 6 stages.
2. Apply block-wise feature distillation for each variant, aligning outputs to the corresponding SD 1.5 teacher blocks.
3. Encode module selections as discrete vectors and search for Pareto-optimal U-Net configurations via multi-objective Bayesian optimization.
4. Freeze the selected U-Net and distill the accompanying VAE encoder–decoder.
5. Perform end-to-end fine-tuning to correct accumulated errors, yielding the final NanoSD models.

### Key Designs

**Hardware-aware U-Net decomposition (Sec. 3.1)**: Encoder-4, the middle block, and decoder-4 of SD 1.5 are removed as the lowest contributors, retaining 3 encoder stages and 3 decoder stages. For each stage, the original block structure (e.g., R-A-R-A) is examined and variants that strictly preserve input–output tensor shapes are constructed (e.g., residual-only, reduced-attention), ensuring that any combination requires no additional adapters. The resulting search space comprises $4\times4\times4\times8\times8\times8 = 32{,}768$ candidate architectures.

**Block-wise feature distillation (Sec. 3.2)**: Each candidate variant at each stage is independently trained using an L2 feature-matching loss against the corresponding SD 1.5 teacher block:
$$\mathcal{L}_{\text{distill}}^{(i,j)} = \|O_S - O_T\|_2^2$$
The 6 stages yield 30 distilled proxy blocks in total (3+3+3+7+7+7). The process is highly parallelizable and incurs minimal computational overhead.

**Multi-objective Bayesian optimization (Sec. 3.4)**: A teacher-aligned FID (taFID) metric is defined to quantify distributional deviation from SD 1.5 outputs. Two bi-objective optimization problems are formulated, pairing taFID with device latency and parameter count, respectively. Discrete search variables are relaxed to a continuous space $\mathbf{x} \in [0,1]^6$, with Gaussian processes modeling both objectives. Candidate configurations are sampled by maximizing the Expected Hypervolume Improvement (EHVI). This procedure yields 7 Pareto-optimal architectures; Model 2 (NanoSD-Prime) is selected as the representative: 315M parameters, 27 ms latency, taFID = 10.

### Loss & Training

- **Block-level distillation**: L2 feature-matching loss.
- **VAE distillation**: Standard feature-matching loss (U-Net frozen; VAE encoder–decoder distilled).
- **End-to-end fine-tuning**: Standard diffusion denoising loss.

## Key Experimental Results

### Main Results

**Super-resolution (DIV-2K Val)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ | NIQE↓ | MUSIQ↑ | MACs(G) | Para.(M) |
|--------|-------|-------|--------|------|-------|--------|---------|----------|
| Edge-SD-SR | 24.10 | 0.617 | 0.249 | 25.37 | - | 69.58 | - | 169 |
| AdcSR | 23.74 | 0.602 | 0.285 | 25.52 | 4.36 | 68.00 | 496 | 456 |
| TinySR | - | 0.572 | 0.279 | 22.94 | 4.15 | 69.90 | 427 | 341 |
| **Nano-S3Diff** | 23.13 | 0.573 | **0.278** | **22.34** | **4.09** | **70.44** | **285** | **318** |
| **Nano-OSEDiff** | **24.29** | **0.628** | 0.296 | 27.46 | 4.92 | 66.41 | 340 | 448 |

**Face restoration (CelebA-Test)**:

| Method | LPIPS↓ | NIQE↓ | MUSIQ↑ | FID↓ | LMD↓ | MACs(G) | Para.(M) |
|--------|--------|-------|--------|------|------|---------|----------|
| OSDFace | 0.336 | 3.884 | 75.64 | 45.41 | 5.286 | 2465 | 1887 |
| **Nano-OSDFace** | 0.341 | 3.913 | **76.01** | 45.92 | **5.172** | **479** | **415** |

Nano-OSDFace achieves superior MUSIQ and LMD scores compared to the original OSDFace while reducing MACs by approximately 5× and parameter count by approximately 4.5×.

### Ablation Study

- **Pareto frontier analysis**: The 7 NanoSD variants span a latency range of 12–41 ms and a parameter range of 130M–315M. Manually tuned models and Segmind TinySD both lie far from the Pareto frontier, demonstrating that hand-crafted simplification cannot effectively preserve the generative prior.
- **Latency vs. parameter count discrepancy**: Model 5 achieves the lowest latency (12 ms / 170M parameters) while Model 7 has the fewest parameters (27 ms / 130M), validating the paper's central claim that parameter count and latency are not positively correlated.
- **Multi-task generality**: The same NanoSD backbone is successfully integrated into six frameworks — OSEDiff, S3Diff, OSDFace, DiffBIR, Diff-Plugin, and Marigold — covering super-resolution, face restoration, deblurring/dehazing/deraining/desnowing, and monocular depth estimation.

### Key Findings

1. Block-wise distillation combined with Bayesian search efficiently explores a search space of 32K architectures without full-network training.
2. NanoSD-Prime (Model 2) achieves generation quality nearly on par with SD 1.5 (taFID = 10) at 27 ms NPU latency.
3. On depth estimation, Nano-Marigold achieves AbsRel = 7.2 and $\delta_1$ = 94.6 on NYUv2, a manageable gap relative to Marigold (5.5 / 96.4).

## Highlights & Insights

- This is the first work to co-compress the full SD 1.5 pipeline (U-Net + VAE) rather than the denoising network alone.
- The "divide-and-conquer" strategy of block-wise distillation followed by combinatorial search is highly efficient: 30 proxy blocks can be assembled into 32K architectures.
- Rigorous hardware-aware design ensures all variants maintain tensor shape compatibility without requiring adapters.
- A genuinely multi-task foundation model: the same backbone is compatible with multiple conditioning plugins including LoRA and ControlNet.
- Practical deployment validation: 27 ms measured on a Qualcomm NPU with 8-bit weights and 16-bit activations.

## Limitations & Future Work

- Distillation is based solely on SD 1.5; the compression potential of newer architectures such as SDXL or SD3 remains unexplored.
- taFID as a search metric measures only distributional deviation from the teacher and does not directly correlate with downstream task performance.
- A non-trivial performance gap remains between Nano-Marigold and full-scale Marigold on depth estimation (AbsRel 7.2 vs. 5.5).
- All latency measurements are conducted on Qualcomm NPUs; applicability to other hardware platforms (e.g., Apple ANE, MediaTek APU) is not validated.
- VAE distillation details are insufficiently described in the main text, impeding full reproducibility.

## Related Work & Insights

- **Diffusion-based restoration**: StableSR, DiffBIR, Diff-Plugin, and SeeSR exploit T2I priors but are computationally expensive.
- **Single-step diffusion acceleration**: SinSR (bidirectional distillation), OSEDiff (variational score distillation), S3Diff (degradation-aware LoRA).
- **Architecture compression**: SnapFusion (module contribution analysis), MobileDiff (Transformer relocation), SnapGen (depthwise separable convolutions).
- **Edge-efficient SR**: AdcSR (adversarial diffusion compression), Edge-SD-SR (LR conditioning mechanism), TinySR (deep U-Net pruning), PocketSR (multi-layer feature distillation).
- **Segmind TinySD**: A hand-simplified SD 1.5 model that lies far from the Pareto frontier compared to the NanoSD family.

## Rating

- Novelty: ⭐⭐⭐⭐ — The full-pipeline co-compression scheme combining block-wise distillation and multi-objective Bayesian search is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers 7 restoration tasks, multiple datasets, 6 integration frameworks, and hardware benchmarks.
- Writing Quality: ⭐⭐⭐⭐ — Architecture diagrams are clear and the Pareto analysis is thorough, though VAE distillation details are insufficient.
- Value: ⭐⭐⭐⭐⭐ — Provides a practical general-purpose foundation model solution for deploying diffusion models on edge devices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CoD: A Diffusion Foundation Model for Image Compression](cod_a_diffusion_foundation_model_for_image_compression.md)
- [\[CVPR 2026\] StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars](streamavatar_streaming_diffusion_models_for_real-time_interactive_human_avatars.md)
- [\[CVPR 2026\] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)
- [\[CVPR 2026\] Quantization with Unified Adaptive Distillation to enable multi-LoRA based one-for-all Generative Vision Models on edge](quantization_with_unified_adaptive_distillation_to_enable_multi-lora_based_one-f.md)
- [\[CVPR 2026\] HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images](adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)

</div>

<!-- RELATED:END -->
