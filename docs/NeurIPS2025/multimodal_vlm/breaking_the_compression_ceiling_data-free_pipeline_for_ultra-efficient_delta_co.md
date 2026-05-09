---
title: >-
  [Paper Note] Breaking the Compression Ceiling: Data-Free Pipeline for Ultra-Efficient Delta Compression
description: >-
  [NeurIPS 2025][Multimodal VLM][delta compression] This paper proposes UltraDelta — the first data-free delta weight compression pipeline — which achieves compression ratios up to 224× across LLM/NLP/vision/multimodal models without performance degradation and even surpasses fine-tuned models, via three components: variance-guided mixed sparsity allocation, distribution-aware compression, and trace-norm-guided rescaling.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - delta compression
  - model pruning
  - quantization
  - data-free compression
  - multi-task deployment
date: 2026-05-08
content_hash: a19444eac2ea4ae6
---

# Breaking the Compression Ceiling: Data-Free Pipeline for Ultra-Efficient Delta Compression

**Conference**: NeurIPS 2025
**arXiv**: [2505.13563](https://arxiv.org/abs/2505.13563)
**Code**: [xiaohuiwang000/UltraDelta](https://github.com/xiaohuiwang000/UltraDelta)
**Area**: Multimodal VLM
**Keywords**: delta compression, model pruning, quantization, data-free compression, multi-task deployment

## TL;DR

This paper proposes UltraDelta — the first data-free delta weight compression pipeline — which achieves compression ratios up to 224× across LLM/NLP/vision/multimodal models without performance degradation and even surpasses fine-tuned models, via three components: variance-guided mixed sparsity allocation, distribution-aware compression, and trace-norm-guided rescaling.

## Background & Motivation

- **Storage bottleneck in multi-model deployment**: Under the fine-tuning paradigm, each downstream task requires a full model copy, imposing substantial storage overhead in multi-task deployment scenarios. Delta compression — storing a single pretrained model plus compressed delta weights — offers an effective remedy.
- **Compression ceiling of existing methods**: Pruning-based methods (DARE, Magnitude Pruning) suffer drastic performance degradation at high sparsity; quantization-based methods (BitDelta, Delta-CoMe) are constrained by 1-bit precision and cannot compress further.
- **Neglect of inter-layer heterogeneity**: Existing methods apply a uniform sparsity rate across all layers, ignoring the varying contribution of different layers to model performance, resulting in critical information loss.
- **Disruption of intra-layer distributions**: Aggressive quantization or pruning distorts the shape of intra-layer weight distributions, which is critical to downstream performance.
- **Instability under extreme compression**: At extreme sparsity, the standard rescaling factor $1/(1-s)$ is insufficient to maintain model stability, and existing methods rely on calibration data for adjustment, limiting practical applicability.
- **Absence of a unified data-free solution**: No prior method simultaneously achieves data-free operation, ultra-high compression ratio, and strong performance — an unbroken three-way trade-off.

## Method

### Overall Architecture

UltraDelta is a **hybrid (pruning + quantization)** data-free delta compression pipeline that addresses inter-layer, intra-layer, and global dimensions independently to minimize redundancy, maximize information retention, and enhance stability. The pipeline proceeds as follows: ① compute delta weights $\Delta\theta = \theta_{ft} - \theta_{pre}$ → ② assign layer-wise sparsity rates based on variance → ③ apply uniform quantization followed by value-grouped pruning → ④ trace-norm-guided global rescaling → ⑤ reconstruction at inference.

### Key Design 1: Variance-Based Mixed Sparsity Allocation (MSA, Inter-Layer)

- **Function**: Partitions all layers into low/medium/high variance groups based on the variance of their delta weights, assigning lower sparsity rates to high-variance layers to preserve more information.
- **Mechanism**: Theoretical derivation shows that layer variance is positively correlated with information entropy — higher variance implies higher information content, requiring more bits to maintain a given distortion under lossy compression: $R(D) = \frac{1}{2}\log(\sigma^2/D)$. High-variance layers are therefore assigned lower sparsity rates.
- **Design Motivation**: Existing uniform sparsity strategies treat all layers equally, causing over-pruning of information-dense layers. MSA uses variance as a data-free proxy for adaptive allocation, requiring no calibration data.

### Key Design 2: Distribution-Aware Compression (DAC, Intra-Layer)

- **Function**: First applies low-bit (4-bit) uniform quantization to map parameters to discrete values; then groups parameters by quantized value and independently applies random pruning within each group.
- **Mechanism**: Random pruning within value-grouped subsets preserves the relative proportion of each quantized value, thereby retaining the shape of the intra-layer weight distribution. Compared to Magnitude Pruning — which preferentially removes small values and severely skews the distribution — DAC exhibits superior distribution preservation.
- **Design Motivation**: Prior work demonstrates that preserving the shape of weight distributions is critical to post-compression performance. DAC combines quantization and pruning synergistically: quantization reduces bit-width while grouped pruning further reduces parameter count, together enabling ultra-high compression ratios.

### Key Design 3: Trace-Norm-Guided Rescaling (TNGR, Global)

- **Function**: Introduces an additional factor $\gamma$ on top of the standard rescaling factor $1/(1-s)$, yielding a final rescaling of $\gamma/(1-s)$, where $\gamma \in [0.5, 1.0]$ is inversely proportional to the trace norm of the delta weights.
- **Mechanism**: Theoretical analysis shows that the activation error variance is $\text{Var}(\varepsilon) = \frac{\gamma^2 s}{1-s} \odot a^2$, which explodes at high sparsity. Introducing $\gamma < 1$ effectively suppresses this error. Empirically, delta weights with larger trace norms require smaller $\gamma$ and exhibit greater sensitivity to it.
- **Design Motivation**: DARE's standard rescaling leads to performance instability under extreme sparsity (≥95%). TNGR employs the trace norm as a data-free adaptive estimation metric, eliminating the need to search for the optimal rescaling factor via data.

## Loss & Training

UltraDelta is a **pure post-processing method** that involves no training or fine-tuning and is entirely data-free. The compression pipeline relies solely on statistical properties of the delta weights themselves (variance, trace norm). At inference, the full model is reconstructed via $\theta^{final} = \theta_{pre} + \frac{\gamma}{1-s} \cdot \hat{\Delta\theta}^*$. The compressed sparse delta weights are stored using Golomb coding of zero-run lengths for efficient encoding.

## Key Experimental Results

### Table 1: LLaMA-2 Series LLMs (Average over 3 Tasks)

| Method | CR (7B) | CR (13B) | Avg (7B) | Avg (13B) |
|:---|:---:|:---:|:---:|:---:|
| Fine-tuned | 1× | 1× | 45.37 | 50.94 |
| BitDelta | 16× | 16× | 41.89 | 48.42 |
| Delta-CoMe | 16× | 16× | 42.52 | 48.71 |
| DARE (same CR) | 31.7× | 50.1× | 33.97 | 40.23 |
| MP (same CR) | 31.7× | 50.1× | 27.02 | 15.36 |
| **UltraDelta** | **32.9×** | **50.9×** | **45.57** | **52.05** |

UltraDelta **surpasses the fine-tuned model** at 32.9×/50.9× compression (45.57 vs. 45.37, 52.05 vs. 50.94), suggesting that compression may introduce a regularization benefit.

### Table 2: Extreme Compression on T5-base and ViT-L/14

| Model | Method | CR | Avg Performance |
|:---|:---|:---:|:---:|
| T5-base (8 NLP tasks) | Fine-tuned | 1× | 86.37 |
| | BitDelta | 16× | 84.68 |
| | DARE (same CR) | 220.5× | 84.30 |
| | **UltraDelta** | **224.6×** | **86.74** |
| ViT-L/14 (8 vision tasks) | Fine-tuned | 1× | 94.4 |
| | BitDelta | 16× | 94.1 |
| | DARE (same CR) | 127.6× | 89.1 |
| | **UltraDelta** | **132.5×** | **94.4** |

UltraDelta exceeds the fine-tuned model at 224.6× compression on T5-base; on ViT-L/14, it achieves **fully lossless** performance at 132.5×.

### Table 3: Ablation Study (ViT-B/32, 8 Tasks)

| Configuration | CR | Avg Accuracy |
|:---|:---:|:---:|
| DARE (97% sparsity) | 23.7× | 89.7 |
| + DAC | 50.9× (↑27.2×) | 89.7 |
| + DAC + MSA | 50.9× | 90.3 (↑0.6) |
| + DAC + MSA + TNGR | 50.9× | 90.7 (↑0.4) |

DAC doubles the compression ratio while maintaining performance; MSA and TNGR each contribute consistent performance gains.

## Highlights & Insights

- **Compression surpassing the original model**: The phenomenon of "compressed > fine-tuned" observed on LLaMA-2 and T5-base is highly thought-provoking; the authors hypothesize a regularization effect induced by compression.
- **Theory-driven design**: MSA is grounded in information theory (variance–entropy correlation and rate-distortion theory); TNGR is derived from activation error variance analysis — neither relies on pure empirical tuning.
- **Strict data-free constraint**: The pipeline requires no calibration or validation data whatsoever, relying solely on intrinsic statistics of the delta weights (variance, trace norm), making it more practical than methods such as BitDelta and DAREx.
- **Cross-modal consistency**: The same pipeline generalizes effectively across LLMs, NLP, vision, and multimodal models, demonstrating strong versatility.

## Limitations & Future Work

- **Coarse three-group design**: MSA partitions layers into three equal groups (low/medium/high), with no adaptive mechanism for determining the granularity or number of groups.
- **Heuristic $\gamma$ assignment**: The inverse relationship between $\gamma$ and the trace norm in TNGR is heuristic and lacks rigorous optimality guarantees.
- **Linear layers only**: For fair comparison, all methods compress only the linear layers within Transformers; embedding and normalization layers are left unprocessed.
- **No inference acceleration analysis**: The work focuses exclusively on storage compression without discussing the practical inference speedup achievable from the resulting sparse and quantized representations.

## Related Work & Insights

- **DARE**: The seminal work on random pruning with $1/(1-s)$ rescaling, upon which UltraDelta builds comprehensive improvements (distribution-aware pruning, adaptive sparsity, improved rescaling).
- **BitDelta / Delta-CoMe**: Representative works in the 1-bit quantization line, with a compression ratio ceiling of 16×, substantially surpassed by UltraDelta.
- **DeltaZip**: A pioneer in hybrid compression, but with insufficient information retention. UltraDelta's DAC avoids distribution distortion through value-grouped pruning.
- **Insight**: The compressibility of delta weights far exceeds prior expectations. The key lies in understanding and exploiting the statistical structure of delta weights (variance distribution, trace norm characteristics) rather than applying naive pruning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First data-free ultra-high-compression delta pipeline; each of the three components is theoretically motivated; the MSA+DAC+TNGR combination is genuinely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers four model categories (LLMs at 7B/13B and newer architectures, NLP, vision, multimodal) with ablation and distribution analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, balancing theoretical derivations with empirical analysis, supported by rich figures and tables.
- **Value**: ⭐⭐⭐⭐ — Directly practical for multi-model deployment; the data-free constraint enables plug-and-play applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Label-Free Cross-Task LoRA Merging with Null-Space Compression](../../CVPR2026/multimodal_vlm/label-free_cross-task_lora_merging_with_null-space_compression.md)
- [\[CVPR 2026\] ApET: Approximation-Error Guided Token Compression for Efficient VLMs](../../CVPR2026/multimodal_vlm/apet_approximation-error_guided_token_compression_for_efficient_vlms.md)
- [\[NeurIPS 2025\] CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization](coido_efficient_data_selection_for_visual_instruction_tuning_via_coupled_importa.md)
- [\[ICCV 2025\] AirCache: Activating Inter-Modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference](../../ICCV2025/multimodal_vlm/aircache_activating_inter-modal_relevancy_kv_cache_compression_for_efficient_lar.md)
- [\[ICCV 2025\] Dynamic-VLM: Simple Dynamic Visual Token Compression for VideoLLM](../../ICCV2025/multimodal_vlm/dynamic-vlm_simple_dynamic_visual_token_compression_for_videollm.md)

</div>

<!-- RELATED:END -->
