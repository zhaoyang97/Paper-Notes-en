---
title: >-
  [Paper Note] QuantSparse: Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification
description: >-
  [ICLR 2026][Video Generation][video-generation] This paper proposes QuantSparse, the first framework to jointly integrate model quantization and attention sparsification for video diffusion Transformer compression. By introducing Multi-Scale Salient Attention Distillation (MSAD) and Second-Order Sparse Attention Reparameterization (SSAR), QuantSparse addresses the "amplified attention shift" problem caused by naive combination of the two techniques. On HunyuanVideo-13B with W4A8 and 15% attention density, it achieves 3.68× storage compression and 1.88× inference speedup with nearly lossless generation quality.
tags:
  - ICLR 2026
  - Video Generation
  - video-generation
  - model-compression
  - quantization
  - sparse-attention
  - diffusion-transformer
date: 2026-05-08
content_hash: 272777af0f2c4b46
---

# QuantSparse: Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification

**Conference**: ICLR 2026
**arXiv**: [2509.23681](https://arxiv.org/abs/2509.23681)
**Code**: [GitHub](https://github.com/wlfeng0509/QuantSparse)
**Area**: Video Generation
**Keywords**: video-generation, model-compression, quantization, sparse-attention, diffusion-transformer

## TL;DR

This paper proposes QuantSparse, the first framework to jointly integrate model quantization and attention sparsification for video diffusion Transformer compression. By introducing Multi-Scale Salient Attention Distillation (MSAD) and Second-Order Sparse Attention Reparameterization (SSAR), QuantSparse addresses the "amplified attention shift" problem caused by naive combination of the two techniques. On HunyuanVideo-13B with W4A8 and 15% attention density, it achieves 3.68× storage compression and 1.88× inference speedup with nearly lossless generation quality.

## Background & Motivation

1. **High computational cost of video diffusion models**: State-of-the-art models such as Wan2.1-14B require over 20 GB GPU memory and nearly one hour of inference time to generate a single high-definition video, severely limiting practical deployment, especially in resource-constrained settings.

2. **Quantization and sparsification are two complementary compression directions**: Quantization reduces storage and computation via low-bit integer representations, while sparse attention reduces complexity by pruning redundant attention computations. The two are orthogonal and theoretically offer additive benefits.

3. **Severe degradation when either method is pushed to its limit**: Extremely low-bit quantization (e.g., binarization) causes representational collapse, while extreme sparsification discards critical contextual information; both lead to severe quality degradation when applied independently.

4. **Naive combination yields worse results**: Experiments reveal that simply combining quantization and sparsification introduces "amplified attention shift"—after sparsification removes low-magnitude attention weights, the systematic perturbation introduced by quantization on the remaining attention products is amplified, with the two sources of error mutually reinforcing and severely degrading fine-grained dependency modeling in video generation.

5. **Existing methods are developed in isolation**: Quantization methods (Q-VDiT, ViDiT-Q) and sparsification methods (SparseVideoGen, Jenga) have evolved independently; no prior work has systematically explored strategies for their joint integration.

6. **Memory bottleneck in attention distillation**: For models such as HunyuanVideo with sequence length $L > 10^4$, storing the full attention matrix requires $O(L^2)$ memory, making direct attention distillation infeasible.

## Method

### Overall Architecture

QuantSparse comprises two core modules: Multi-Scale Salient Attention Distillation (MSAD) during the **calibration phase**, and Second-Order Sparse Attention Reparameterization (SSAR) during the **inference phase**.

### Problem Formulation: Amplified Attention Shift

Quantization injects noise $\epsilon$ into the QK dot product; its interaction with the sparse mask $\mathbf{M}$ produces a compound shift:

$$\Delta_{\text{total}} = \Delta_{\text{sparse}} + \Delta_{\text{quant}} + O(\|\epsilon\|_F \cdot \|\mathbf{M}\|_0)$$

The third cross term is the root cause of naive combination failure—the information loss from sparsification and the quantization noise mutually reinforce each other.

### Module 1: Multi-Scale Salient Attention Distillation (MSAD)

MSAD aligns the attention distribution of the quantized model in a memory-efficient manner via dual-scale global–local distillation.

**Global guidance**: Exploiting the spatial locality of video data, Q and K are downsampled via average pooling with stride $s$, and the global attention distillation loss is computed at reduced resolution $\tilde{L} = L/s^2$, with complexity only $1/s^2$ of full attention:

$$\mathcal{L}_{\text{global}} = \text{MSE}(\mathbf{A}_{\text{global}}^{\text{FP}} \| \mathbf{A}_{\text{global}}^{\text{quant}})$$

**Local guidance**: Attention distributions are found to be highly skewed—fewer than 10% of tokens account for the vast majority of attention mass. Only the top-$k$ salient queries are selected for full-resolution local distillation, focusing on high-impact regions at minimal cost:

$$\mathcal{L}_{\text{local}} = \text{MSE}(\mathbf{A}_{\text{local}}^{\text{FP}} \| \mathbf{A}_{\text{local}}^{\text{quant}})$$

**Joint optimization**: $\mathcal{L}_{\text{distill}} = \mathcal{L}_{\text{quant}} + \lambda_{\text{global}} \mathcal{L}_{\text{global}} + \lambda_{\text{local}} \mathcal{L}_{\text{local}}$

### Module 2: Second-Order Sparse Attention Reparameterization (SSAR)

SSAR addresses the information loss of sparse attention during inference.

**Instability of first-order residuals**: The first-order residual is defined as $\Delta^{(t)} = \mathbf{A}_{\text{full}}^{(t)} - \mathbf{A}_{\text{sparse}}^{(t)}$. Prior work assumes this residual is constant across timesteps, but quantization noise $\epsilon^{(t)}$ varies across timesteps, violating this assumption.

**Temporal stability of second-order residuals**: The key insight is that the second-order residual $\hat{\Delta}^{(t)} = \Delta^{(t)} - \Delta^{(t-1)}$ exhibits far smaller temporal variation than the first-order residual, since the quantization noise distributions of adjacent timesteps are similar and approximately cancel after differencing.

**SVD projection for denoising**: SVD decomposition is applied to the second-order residual, projecting it onto the top $r$ principal components to further suppress temporal variance:

$$\tilde{\Delta}_{\text{quant}} = \mathbf{S}_{:,:r} \mathbf{U}_{:r,:r} \mathbf{V}_{:,:r}^\top$$

During inference, the cache is refreshed at fixed intervals (every 5 steps), and the second-order correction term efficiently approximates the full attention output without additional memory overhead.

## Key Experimental Results

### Experimental Setup

- **Models**: HunyuanVideo-13B, Wan2.1-1.3B, Wan2.1-14B
- **Quantization settings**: W6A6, W4A8; channel-wise weight quantization + dynamic per-token activation quantization
- **Baselines**: Quantization methods (PTQ4DiT, Q-DiT, SmoothQuant, QuaRot, ViDiT-Q, Q-VDiT); sparsification methods (DiTFastAttn, Jenga, SparseVideoGen); and their combinations

### Main Results

**Table 1: HunyuanVideo-13B Main Results (W4A8)**

| Method | Density | VQA↑ | PSNR↑ | SSIM↑ | LPIPS↓ | Speedup |
|--------|---------|------|-------|-------|--------|---------|
| Full Prec. | 100% | 81.23 | - | - | - | 1.00× |
| Q-VDiT | 100% | 67.95 | 16.85 | 0.605 | 0.461 | 1.09× |
| Q-VDiT+SVG | 15% | 76.30 | 16.66 | 0.591 | 0.460 | 1.84× |
| **QuantSparse** | **15%** | **81.19** | **20.88** | **0.678** | **0.273** | **1.88×** |

At 15% attention density, QuantSparse achieves a VQA score of 81.19 (close to the full-precision baseline of 81.23), a PSNR substantially higher than Q-VDiT+SVG (20.88 vs. 16.66), along with 1.88× speedup and 3.68× storage compression.

### Ablation Study

**Table 2: Module Contribution Ablation (Wan2.1-14B, W4A8, 25% density)**

| Module | VQA↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|-------|-------|--------|
| No distillation | 81.92 | 14.35 | 0.486 | 0.425 |
| + Global guidance | 85.26 | 16.01 | 0.547 | 0.349 |
| + Local guidance | 86.95 | 16.82 | 0.561 | 0.325 |
| + MSAD (global+local) | **91.98** | **18.72** | **0.630** | **0.240** |
| No cache | 68.00 | 14.16 | 0.470 | 0.445 |
| + First-order residual | 70.82 | 17.08 | 0.572 | 0.285 |
| + Second-order residual | 89.73 | 18.68 | 0.616 | 0.258 |
| + SSAR (second-order+SVD) | **91.98** | **18.72** | **0.630** | **0.240** |

MSAD improves PSNR from 14.35 to 18.72 (+4.37), while SSAR improves it from 14.16 to 18.72 (+4.56); the two modules contribute comparably and complementarily.

### Efficiency Analysis

| Configuration | Model Storage | VRAM | DiT Time | Speedup |
|---------------|--------------|------|----------|---------|
| Full Prec. | 23.88 GB | 35.79 GB | 1264 s | 1.00× |
| QuantSparse W4A8 15% | **6.49 GB** (↓3.68×) | 27.02 GB (↓1.32×) | **671 s** | **1.88×** |

## Highlights & Insights

- **First systematic integration of quantization and sparsification**: Provides a mathematical analysis of "amplified attention shift" and a unified solution, filling the gap in jointly applying these two orthogonal compression techniques.
- **Memory-efficient attention distillation**: MSAD cleverly circumvents the $O(L^2)$ memory bottleneck through global downsampling combined with local salient token selection.
- **Key insight of second-order residuals**: The observation that first-order residuals are unstable under quantization while second-order residuals remain stable is an elegant mathematical finding, further improved by SVD projection for denoising.
- **Nearly lossless aggressive compression**: Quality close to full precision is maintained at 15% attention density and W4A8, far surpassing all baselines.

## Limitations & Future Work

- **Calibration phase cost**: MSAD requires running both the full-precision and quantized models simultaneously during PTQ calibration, imposing non-trivial memory and compute demands at that stage.
- **Manual cache-refresh interval**: The cache-refresh interval of 5 is empirically determined; different models and resolutions may require re-tuning.
- **Overhead of SVD decomposition**: Although the paper claims "negligible overhead," the practical cost of SVD decomposition on extremely long sequences or very large models warrants further validation.
- **Limitations of evaluation metrics**: The study primarily relies on reference-based metrics (PSNR/SSIM) and no-reference metrics (VQA/CLIPSIM), lacking large-scale human subjective evaluation.

## Related Work & Insights

### vs. Q-VDiT (Feng et al., 2025) — Current SOTA Quantization Method

Q-VDiT introduces temporal distillation for quantization calibration and represents the prior state of the art in video DiT quantization. However, Q-VDiT focuses solely on quantization without sparsification, achieving only 16.85 PSNR on HunyuanVideo at W4A8. Even naively combining Q-VDiT with the SVG sparsification method yields only 16.66 PSNR (a slight regression), confirming that naive combination is ineffective. QuantSparse substantially outperforms this with 20.88 PSNR, demonstrating the necessity of co-design.

### vs. SparseVideoGen (Xi et al., 2025) — Static Sparse Attention

SVG employs predefined spatio-temporal sparse masks to reduce attention computation and performs well at full precision. However, when combined with quantization (QuaRot+SVG on HunyuanVideo at W4A8, 15% density, VQA drops to 41.40), performance degrades severely. QuantSparse addresses the attention shift arising from quantization–sparsification interaction through MSAD and SSAR, achieving near-lossless quality at the same compression ratio.

### vs. DiTFastAttn (Yuan et al., 2024) — Cache-Based First-Order Residual

DiTFastAttn exploits the cross-timestep stability of first-order residuals for attention approximation. QuantSparse's SSAR demonstrates theoretically (Proposition 3.2) that first-order residuals are no longer stable under quantization, while second-order residuals possess temporal stability (Proposition 3.3)—a more rigorous generalization that substantially outperforms DiTFastAttn under the W4A8 setting.

## Rating

- ⭐⭐⭐⭐⭐ **Novelty**: First to co-design quantization and sparsification with rigorous theoretical analysis and two cleverly designed core modules.
- ⭐⭐⭐⭐⭐ **Experimental Thoroughness**: Covers three models from 1.3B to 14B, two quantization settings, multiple baselines and combinations, and detailed ablation studies.
- ⭐⭐⭐⭐ **Writing Quality**: Mathematical derivations are clear and figures are informative, though notation is dense and some derivation details are relegated to the appendix.
- ⭐⭐⭐⭐⭐ **Value**: 3.68× storage compression, 1.88× inference speedup, and near-lossless quality offer direct practical value for video generation deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](lumos-1_on_autoregressive_video_generation_with_discrete_diffusion_from_a_unifie.md)
- [\[ICLR 2026\] JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization](javisdit_joint_audio-video_diffusion_transformer_with_hierarchical_spatio-tempor.md)
- [\[CVPR 2026\] When to Lock Attention: Training-Free KV Control in Video Diffusion](../../CVPR2026/video_generation/when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)
- [\[CVPR 2026\] MoVieDrive: Urban Scene Synthesis with Multi-Modal Multi-View Video Diffusion Transformer](../../CVPR2026/video_generation/moviedrive_multimodal_multiview_video_diffusion.md)
- [\[ICCV 2025\] Prompt-A-Video: Prompt Your Video Diffusion Model via Preference-Aligned LLM](../../ICCV2025/video_generation/prompt-a-video_prompt_your_video_diffusion_model_via_preference-aligned_llm.md)

</div>

<!-- RELATED:END -->
