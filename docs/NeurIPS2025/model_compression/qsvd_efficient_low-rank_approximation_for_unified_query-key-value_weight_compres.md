---
title: >-
  [Paper Note] QSVD: Efficient Low-Rank Approximation for Unified Query-Key-Value Weight Compression
description: >-
  [NeurIPS 2025][Model Compression][VLM compression] This paper proposes QSVD, which performs SVD on the joint QKV weight matrix and shares a single down-projection matrix across Q, K…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "VLM compression"
  - "SVD"
  - "KV cache"
  - "quantization"
  - "low-rank approximation"
date: 2026-05-08
content_hash: c7a3e1e293eda125
---

# QSVD: Efficient Low-Rank Approximation for Unified Query-Key-Value Weight Compression

**Conference**: NeurIPS 2025  
**arXiv**: [2510.16292](https://arxiv.org/abs/2510.16292)  
**Code**: [https://github.com/SAI-Lab-NYU/QSVD](https://github.com/SAI-Lab-NYU/QSVD)  
**Area**: Multimodal VLM / Model Compression  
**Keywords**: VLM compression, SVD, KV cache, quantization, low-rank approximation

## TL;DR

This paper proposes QSVD, which performs SVD on the joint QKV weight matrix and shares a single down-projection matrix across Q, K, and V to reduce KV cache size and computational overhead. Combined with importance-score-based adaptive rank allocation and a quantization scheme compatible with low-rank decomposition, QSVD achieves over 10% accuracy improvement on VLMs at lower hardware cost.

## Background & Motivation

Vision-language models (VLMs) have demonstrated strong performance on tasks such as image captioning and visual question answering, yet they face substantial computational challenges—joint processing of high-dimensional visual and textual data demands intensive computation, and autoregressive token generation creates memory-bandwidth bottlenecks.

**Core Pain Points**:

**Large KV cache footprint**: In Multi-Head Attention, Key and Value matrices grow linearly with sequence length, becoming the primary bottleneck for inference throughput.

**Insufficient efficiency of existing SVD compression**: Conventional methods apply SVD independently to Q, K, and V, yielding three separate down-projection matrices with redundant parameters and computation.

**Poor compatibility between quantization and SVD**: The intermediate representation $C_{qkv}$ produced by SVD decomposition exhibits severe channel-wise outliers, hindering low-precision quantization.

**Key Challenge**: How can weight parameter count, KV cache size, and FLOPs all be reduced simultaneously while preserving VLM accuracy?

**Key Insight**: Inspired by DeepSeek-v3's Multi-Head Latent Attention, QSVD applies SVD to the **joint QKV weight matrix**, allowing Q, K, and V to share a single down-projection matrix. Only one low-dimensional intermediate representation needs to be cached to reconstruct K and V. This is further combined with adaptive rank allocation and an SVD-compatible quantization scheme.

## Method

### Overall Architecture

QSVD comprises three core components: (1) joint QKV SVD compression, (2) cross-layer rank allocation based on importance scores, and (3) post-training quantization adapted for low-rank VLMs.

### Key Designs

1. **Joint QKV SVD Decomposition**:

    - Concatenate $W_Q, W_K, W_V \in \mathbb{R}^{E \times E}$ into $W_{\text{concat}} \in \mathbb{R}^{E \times 3E}$
    - Apply low-rank SVD to the concatenated matrix: $W_{\text{concat}} \approx W_r^d \times \Sigma_r \times W_r^u$
    - Split into a shared down-projection $W_{qkv}^d \in \mathbb{R}^{E \times r}$ and three independent up-projections $W_q^u, W_k^u, W_v^u \in \mathbb{R}^{r \times E}$
    - At inference, only $C_{qkv} = X \cdot W_{qkv}^d$ (of size $r \times L$) needs to be cached; K and V are reconstructed from $C_{qkv}$ on demand
    - Comparison: conventional separate SVD requires $6rE$ parameters and $2rL$ cache; QSVD requires only $4rE$ parameters and $rL$ cache

2. **Cross-Layer Rank Allocation via Importance Scores**:

    - The impact of truncating each singular value $\sigma_i$ on training loss is estimated via a first-order expansion: $\Delta L_{\sigma_i} \approx \langle \Delta W_{\sigma_i}, G_W \rangle_F$
    - Importance score: $\hat{I}_{\sigma_i} = \frac{1}{N}\sum_{n=1}^N \sigma_i^2 [U^T G_W^{(n)} V]_{(i,i)}^2$
    - Key optimization: a mathematical reformulation avoids constructing the full $\Delta W_{\sigma_i}$ matrix, reducing memory from $O(E^3)$ to $O(E^2)$
    - Singular values across all layers are globally ranked, and the top-k most important are retained, achieving optimal cross-layer rank allocation

3. **SVD-Compatible Quantization Scheme**:

    - After SVD decomposition, $C_{qkv} = X W_r^d \Sigma_r^\beta$ exhibits severe channel-wise outliers due to the large dynamic range of $\Sigma_r^\beta$
    - Two orthogonal matrices $H_1, H_2$ are introduced: $Y = (XH_1^\top)(H_1 W_{qkv}^d H_2^\top)(H_2 W_{qkv}^u)$
    - Core innovation: $\beta$ is treated as a learnable parameter and optimized on a calibration set to minimize quantization error $\min_\beta \sum_d \|Y_d - Y_d'\|^2$
    - $\beta$ controls how singular values are distributed between the up- and down-projections, directly affecting the outlier distribution in $C_{qkv}$

### Loss & Training

No training is required. All operations constitute post-training compression (PTQ), requiring only 256 calibration samples (drawn from the ScienceQA training set) for importance score computation and $\beta$ optimization.

## Key Experimental Results

### Main Results

**SVD Compression Accuracy Comparison (FP16, ScienceQA-IMG)**

| Method | SmolVLM 2B (R2=37.5%) | LLaVA-Next 7B (R2=22.5%) | LLaVA-v1.5 13B (R2=22.5%) |
|------|----------------------|--------------------------|--------------------------|
| ASVD | 53.84% | 50.72% | 64.70% |
| SVD-LLM | 65.89% | 65.94% | 71.44% |
| **QSVD-noQ** | **83.78%** | **69.91%** | **71.79%** |
| FP16 Baseline | 84.53% | 69.51% | 71.78% |

**Joint Quantization + SVD Compression Comparison (LLaVA-v1.5 7B)**

| Method | W8A8 Acc. | W8A4 Acc. | W4A4 Acc. | R2 |
|------|---------|---------|---------|-----|
| DuQuant | 66.53% | 57.36% | 52.56% | 50%/25%/25% |
| QVLM | 64.65% | 55.24% | 51.12% | 50%/25%/25% |
| QASVD | 52.95% | 41.92% | 12.61% | 50%/25%/25% |
| **QSVD** | **67.57%** | **65.61%** | **55.16%** | **18.75%/9.38%/9.38%** |

### Ablation Study

| Configuration | ScienceQA | VizWiz | Note |
|------|-----------|--------|------|
| Full QSVD (W8A4) | 65.61% | 52.18% | Baseline |
| w/o $\beta$ optimization | Significant drop | Significant drop | $\beta$ is critical for outlier control |
| Uniform rank allocation (replacing importance scores) | Drop | Drop | Adaptive cross-layer rank allocation is effective |
| Separate SVD (same hardware cost) | 50.72% | 47.78% | Joint SVD substantially outperforms separate SVD |

### Key Findings

- On SmolVLM 2B at R2=37.5% (KV cache reduced to 37.5% of original), QSVD degrades accuracy only from 84.53% to 83.78%—nearly lossless
- Under joint W8A4 quantization + SVD compression, QSVD achieves 65.61% accuracy (ScienceQA) with **only 9.38% KV cache**, while DuQuant achieves 57.36% at 50% KV cache
- QASVD (ASVD + QuaRot) collapses under W4A4 (12.61%), whereas QSVD maintains 55.16%—demonstrating the critical role of $\beta$ optimization for quantization compatibility
- Results are consistent across model scales (2B to 13B): QSVD significantly outperforms baselines on all five evaluated VLMs
- Reductions in KV cache size translate directly into inference speedup

## Highlights & Insights

- **Shared down-projection for QKV is the core innovation**: inspired by DeepSeek MLA but applied as a post-training compression technique, requiring no retraining
- **Computational efficiency of the importance score** is elegantly achieved: using $\hat{I}_{\sigma_i} = \frac{1}{N}\sum \sigma_i^2 [U^T G_W V]_{(i,i)}^2$ avoids $O(E^3)$ memory
- **Introduction of the $\beta$ parameter** resolves the fundamental incompatibility between SVD and quantization—controlling the allocation of singular values between up- and down-projections directly shapes the outlier distribution of intermediate activations
- The overall design is highly modular: SVD and quantization can be applied independently or jointly

## Limitations & Future Work

- SVD is applied only to QKV weights in self-attention layers; FFN layers (which typically account for a larger share of parameters) are not compressed
- Dependence on a calibration dataset (ScienceQA training samples) may yield suboptimal results on other tasks
- Evaluation is limited to classification/VQA tasks; generative tasks (e.g., image captioning) are not assessed
- Comparison with training-time compression methods such as LoRA is absent
- Empirical GPU speedup measurements are limited; results are primarily reported in terms of theoretical FLOPs and cache reduction

## Related Work & Insights

- Relationship to DeepSeek MLA: MLA learns low-rank projections during training, whereas QSVD achieves a similar effect at inference time via SVD, making it applicable to existing pretrained models
- Complementary to KV cache compression works such as Palu and ASVD—QSVD achieves higher accuracy at lower hardware cost
- The $\beta$ optimization strategy is generalizable to other scenarios requiring a balance between low-rank decomposition and quantization

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of joint QKV SVD, learnable $\beta$, and importance-score-based rank allocation is novel and effective
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 5 VLMs, 3 datasets, and multiple quantization configurations, though generative task evaluation is missing
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear, efficiency analysis is thorough, and figures are intuitive
- Value: ⭐⭐⭐⭐ Offers direct practical value for VLM deployment with a concise and efficient approach

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Accurate and Efficient Low-Rank Model Merging in Core Space](accurate_and_efficient_low-rank_model_merging_in_core_space.md)
- [\[NeurIPS 2025\] Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation](beyond_higher_rank_token-wise_input-output_projections_for_efficient_low-rank_ad.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] A*-Thought: Efficient Reasoning via Bidirectional Compression for Low-Resource Settings](a-thought_efficient_reasoning_via_bidirectional_compression_for_low-resource_set.md)
- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](../../ICLR2026/model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)

</div>

<!-- RELATED:END -->
