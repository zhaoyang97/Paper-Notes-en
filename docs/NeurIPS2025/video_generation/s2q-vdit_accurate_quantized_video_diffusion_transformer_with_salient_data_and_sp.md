---
title: >-
  [Paper Note] S²Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
description: >-
  [NeurIPS 2025][Video Generation][Post-training quantization] To address the high calibration variance and optimization difficulty caused by extremely long token sequences in video diffusion Transformers, this paper proposes the S²Q-VDiT framework. By combining Hessian-aware salient data selection and attention-guided sparse token distillation, it achieves lossless quantization under W4A6 settings for the first time, yielding 3.9× model compression and 1.3× inference speedup.
tags:
  - NeurIPS 2025
  - Video Generation
  - Post-training quantization
  - video diffusion models
  - calibration data selection
  - sparse attention
  - model compression
date: 2026-05-08
content_hash: 3ec85a366c3971e7
---

# S²Q-VDiT: Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation

**Conference**: NeurIPS 2025
**arXiv**: [2508.04016](https://arxiv.org/abs/2508.04016)
**Code**: [GitHub](https://github.com/wlfeng0509/s2q-vdit)
**Area**: Diffusion Models / Video Generation / Model Compression
**Keywords**: Post-training quantization, video diffusion models, calibration data selection, sparse attention, model compression

## TL;DR

To address the high calibration variance and optimization difficulty caused by extremely long token sequences in video diffusion Transformers, this paper proposes the S²Q-VDiT framework. By combining Hessian-aware salient data selection and attention-guided sparse token distillation, it achieves lossless quantization under W4A6 settings for the first time, yielding 3.9× model compression and 1.3× inference speedup.

## Background & Motivation

Video diffusion Transformers (e.g., CogVideoX, HunyuanVideo) have become the dominant paradigm for video generation, yet their billions of parameters impose substantial computational and memory overhead. Post-training quantization (PTQ) is an efficient model compression technique, but directly applying it to video diffusion models (V-DMs) leads to severe performance degradation.

The authors identify two quantization challenges unique to V-DMs:

**High calibration data variance**: The joint spatial-temporal modeling in V-DMs produces extremely long token sequences (e.g., $s \times 49$ tokens for a 6-second video). Under the same computational budget, V-DMs can only use tens of calibration samples (compared to thousands for image models). With such limited data, different calibration sample selections lead to drastic differences in quantization performance.

**Token learning difficulty**: Full spatial-temporal attention in V-DMs exhibits sparse patterns—only a small fraction of tokens significantly influence the final output. However, existing PTQ methods align full-precision and quantized outputs uniformly across all tokens, which is inefficient for long-sequence scenarios.

Prior work (Q-DiT, ViDiT-Q) primarily improves quantizer design, whereas this paper takes a novel perspective targeting **calibration data quality** and **optimization strategy**.

## Method

### Overall Architecture

S²Q-VDiT consists of two core modules: (1) Hessian-aware Salient Data Selection (SDS), which constructs a high-quality dataset for quantization calibration; and (2) Attention-guided Sparse Token Distillation (STD), which performs importance-weighted learning during block-wise optimization.

### Key Designs

1. **Hessian-aware Salient Data Selection (SDS)**: Calibration sample importance is assessed along two dimensions. **Diffusion saliency** $C_{\text{diff}} = \|x_t - x_{t-1}\|^2 / \|x_t\|^2$ measures feature differences between adjacent timesteps; larger values indicate richer denoising information at that timestep. **Quantization sensitivity** $C_{\text{quant}} = \|x_t^\top x_t\|_2$ is derived from the Levenberg-Marquardt approximation of the Hessian matrix $H^X = \mathbb{E}[2X^\top X]$; larger values indicate greater sensitivity to quantization perturbations. Both metrics are min-max normalized and multiplied to obtain a unified score $C_{\text{sample}} = \bar{C}_{\text{diff}} \cdot \bar{C}_{\text{quant}}$. The AM-GM inequality ensures that a high total score requires both dimensions to be high simultaneously, preventing bias toward samples that are extreme on only one dimension.

2. **Attention-guided Sparse Token Distillation (STD)**: Exploiting the inherent sparsity of full spatial-temporal attention in V-DMs, the method applies importance weights to the quantization loss across tokens: $\mathcal{L}_{\text{quant}} = \frac{1}{n}\sum_{j=1}^n \lambda_j \|\theta^f(x_{j,:}) - \theta^q(x_{j,:})\|^2$. The weight $\lambda_j$ is computed from multi-head attention maps by summing over all heads and query positions to obtain $S_j = \sum_{h,i} A_{h,i,j}$, then min-max normalizing to the range $[\lambda_{\min}, \lambda_{\max}]$. This focuses the optimization on a small set of critical tokens that strongly influence the output while relaxing constraints on less important ones, improving convergence quality under limited calibration data.

### Loss & Training

- A block-wise PTQ optimization strategy is adopted, calibrating quantization parameters separately for each Transformer block.
- Weights use per-channel symmetric quantization; activations use per-token dynamic quantization.
- The calibration set size is fixed at 40 samples, balancing performance and calibration time.
- STD uses $\lambda_{\min} = 0.5$ and $\lambda_{\max} = 1.0$ in the main experiments.

## Key Experimental Results

### Main Results

W4A6 quantization results on the VBench evaluation benchmark:

| Model | Method | Image Quality (IQ) | Aesthetic Quality (AQ) | Dynamic Degree (DD) | Scene Consistency (ScC) | Overall Consistency (OC) |
|------|------|:--------:|:-------:|:------:|:---------:|:---------:|
| CogVideoX-2B | FP | 58.69 | 55.25 | 50.00 | 33.79 | 25.91 |
| CogVideoX-2B | ViDiT-Q | 51.94 | 48.06 | 33.33 | 22.17 | 23.69 |
| CogVideoX-2B | **S²Q-VDiT** | **55.49** | **53.74** | **40.28** | **32.70** | **25.19** |
| HunyuanVideo | FP | 62.30 | 62.49 | 56.94 | 33.36 | 26.85 |
| HunyuanVideo | ViDiT-Q | 52.21 | 58.38 | 41.67 | 23.69 | 26.15 |
| HunyuanVideo | **S²Q-VDiT** | **58.83** | **59.62** | **48.61** | **33.65** | **26.91** |

W4A4 ultra-low-bit quantization results (CogVideoX-2B):

| Method | IQ | AQ | DD | ScC | OC |
|------|:--:|:--:|:--:|:---:|:--:|
| FP | 58.69 | 55.25 | 50.00 | 33.79 | 25.91 |
| ViDiT-Q | 45.56 | 42.03 | 12.50 | 11.91 | 19.61 |
| **S²Q-VDiT** | **53.71** | **52.31** | **36.11** | **34.23** | **24.90** |

### Ablation Study

| Component / Configuration | Key Metric Change | Notes |
|------------|-----------|------|
| ATOP (single prompt, all timesteps) | IQ ≈ 42–45 | Random baseline, poor quality |
| RTFP (random timesteps) | IQ ≈ 48–50 | Better than ATOP but unstable |
| **SDS (Ours)** | **IQ ≈ 54–56** | Significantly outperforms all heuristics |
| w/o STD | Baseline | Uniform token weighting |
| STD ($\lambda_{\min}=0.5$) | +2–3 IQ | Token weighting improves consistency |
| Calibration data 20→40 | IQ +2, OC +0.5 | 40 samples offers best cost-effectiveness |
| Calibration data 40→80 | Negligible gain | Diminishing marginal returns |

### Key Findings

- Near-lossless quantization is achieved on V-DMs at three scales (2B/5B/13B) under W4A6.
- Scene consistency of CogVideoX-5B after quantization (46.66) even surpasses the full-precision baseline (45.28).
- W4A4 represents the first exploration of 4-bit activation quantization; S²Q-VDiT still retains approximately 95% of model performance.
- Calibration overhead is minimal: only approximately 0.2 additional hours and 2 GB of GPU memory compared to PTQ4DiT.
- Achieves 3.94× model storage compression, 1.56× inference memory reduction, and 1.28× inference speedup.

## Highlights & Insights

- Addressing quantization from the perspective of data quality and optimization strategy is an underexplored yet effective direction.
- The SDS design is elegant: it simultaneously considers two orthogonal dimensions—information richness in the diffusion process and quantization sensitivity—and employs the AM-GM inequality to ensure joint optimality.
- STD leverages existing attention map information (with zero additional computation) to guide token importance weighting, making it practical and efficient.
- This work represents the first systematic exploration of W4A4 quantization settings for V-DMs.

## Limitations & Future Work

- The assumption of sparse attention patterns may not generalize to all V-DM architectures.
- SDS requires pre-computing saliency scores for all candidate samples, introducing additional upfront overhead.
- Validation is currently limited to two model families: CogVideoX and HunyuanVideo.
- Performance under more aggressive quantization settings (e.g., W2A4) remains unexplored.
- The $\lambda_{\min}$ hyperparameter in STD requires tuning, though experiments suggest low sensitivity.

## Related Work & Insights

- Complementary to Q-DiT and ViDiT-Q: those works improve quantizer design, while this paper improves data selection and optimization strategy.
- The diffusion saliency in SDS is inspired by the observation in timestep distillation and caching literature that "skipping consecutive timesteps has limited impact."
- The Hessian approximation draws on optimal weight search theory from LLM quantization works such as GPTQ.
- The proposed data selection strategy may inspire similar approaches for image diffusion model quantization and other model compression scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Novel perspective on V-DM quantization; both technical contributions are well-motivated
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multi-model, multi-bit, comprehensive ablation, and full efficiency analysis
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with coherent observation→method→experiment logic
- **Value**: ⭐⭐⭐⭐ High practical deployment value; 3.9× compression with lossless performance is highly significant for real-world video generation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VORTA: Efficient Video Diffusion via Routing Sparse Attention](vorta_efficient_video_diffusion_via_routing_sparse_attention.md)
- [\[NeurIPS 2025\] VSA: Faster Video Diffusion with Trainable Sparse Attention](vsa_faster_video_diffusion_with_trainable_sparse_attention.md)
- [\[NeurIPS 2025\] Training-Free Efficient Video Generation via Dynamic Token Carving](training-free_efficient_video_generation_via_dynamic_token_carving.md)
- [\[NeurIPS 2025\] Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation](radial_attention_onlog_n_sparse_attention_with_energy_decay_for_long_video_gener.md)
- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](../../ICCV2025/video_generation/vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)

</div>

<!-- RELATED:END -->
