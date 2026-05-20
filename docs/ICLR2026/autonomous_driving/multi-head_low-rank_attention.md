---
title: >-
  [Paper Note] Multi-Head Low-Rank Attention (MLRA)
description: >-
  [ICLR 2026][Autonomous Driving][KV Cache] This paper proposes Multi-Head Low-Rank Attention (MLRA), which decomposes the single latent head in MLA into multiple independently shardable latent heads and sums the attention…
tags:
  - "ICLR 2026"
  - "Autonomous Driving"
  - "KV Cache"
  - "Tensor Parallelism"
  - "Low-Rank Attention"
  - "Decoding Efficiency"
  - "Multi-Head Latent Attention"
date: 2026-05-08
content_hash: 41fd1a5c06b2f9e8
---

# Multi-Head Low-Rank Attention (MLRA)

**Conference**: ICLR 2026
**arXiv**: [2603.02188](https://arxiv.org/abs/2603.02188)  
**Code**: [GitHub](https://github.com/SongtaoLiu0823/MLRA) / [HuggingFace](https://huggingface.co/Soughing/MLRA)  
**Area**: Autonomous Driving
**Keywords**: KV Cache, Tensor Parallelism, Low-Rank Attention, Decoding Efficiency, Multi-Head Latent Attention

## TL;DR

This paper proposes Multi-Head Low-Rank Attention (MLRA), which decomposes the single latent head in MLA into multiple independently shardable latent heads and sums the attention outputs across branches, enabling native 4-way tensor parallelism. The method achieves 2.8× decoding speedup while maintaining state-of-the-art performance.

## Background & Motivation

**Long-context inference bottleneck**: During LLM decoding, each step requires repeatedly transferring the KV cache between HBM and SRAM, making data movement—rather than computation—the dominant source of latency in long-context inference.

**Strengths and limitations of MLA**: DeepSeek's Multi-Head Latent Attention (MLA) compresses the KV cache into a single latent head (only $4.5 d_h$ per token), substantially reducing total KV cache size. However, the single latent head is **not shardable**—under tensor-parallel (TP) decoding, each device is forced to redundantly load the full KV cache.

**Performance stagnation of MLA under TP**: Regardless of the TP degree, the per-device KV cache load in MLA remains fixed at $4.5 d_h$, completely negating the weight-sharding benefits of tensor parallelism.

**Partial solution by GLA-2**: GLA-2 bisects the latent head into two smaller latent heads, reducing the per-device load to $2.5 d_h$ under 2-way TP, but further reduction is not achievable for TP > 2.

**Importance of arithmetic intensity**: The key metric for decoding efficiency is arithmetic intensity (FLOPs/byte), which requires maintaining high computational density while reducing memory loads, thereby shifting the workload from memory-bound to compute-bound regimes.

**Variance mismatch**: In MLA, RoPE keys and NoPE keys exhibit a significant variance mismatch ($\text{Var}(K^{\text{RoPE}}) / \text{Var}(K^{\text{NoPE}}) \approx d/d_c$), which is particularly pronounced when the latent dimension is much smaller than the hidden dimension.

## Method

### Overall Architecture

The core idea of MLRA is to **explicitly decompose the non-shardable single latent head in MLA into multiple independent latent heads**. Each latent head independently performs an up-projection to produce NoPE KVs, and the attention outputs across branches are summed. This design natively supports 4-way tensor parallelism.

MLRA provides two variants:
- **MLRA-2**: 2 latent heads, each serving half of the attention heads, yielding a 2-branch summed output.
- **MLRA-4**: 4 latent heads, each serving all attention heads via up-projection, yielding a 4-branch summed output.

### Key Designs

#### 1. Block Decomposition

**Function**: Partitions the KV latent matrix $C^{KV} \in \mathbb{R}^{n \times d_c}$ into 4 channel-wise blocks $C_{:,(b)}^{KV}$, and correspondingly partitions the up-projection matrices $W^{UK}$ and $W^{UV}$ into 4 row-wise sub-blocks.

**Design Motivation**: In MLA, the NoPE key/value for each head is mathematically equivalent to the sum of 4 sub-block products (Eq. 2). Moving this summation from the KV computation to the attention output level enables each sub-block to compute attention independently.

**Mechanism** (MLRA-4 as an example):
$$O_{:,i,:} = \sum_{b=0}^{3} \text{Softmax}\left(\tau Q_{:,i,:}^{\text{NoPE}} (C_{:,(b)}^{KV} W_{(b),(i)}^{UK})^\top + \tau Q_{:,i,:}^{\text{RoPE}} (K^{\text{RoPE}})^\top \right) (C_{:,(b)}^{KV} W_{(b),(i)}^{UV})$$

Each branch $b$ depends only on $C_{:,(b)}^{KV}$ (of size $d_h$) and can be assigned to separate TP devices for independent computation.

#### 2. Grouped Mapping in MLRA-2

**Function**: Adopts the grouping strategy from GLA-2, bisecting the latent head so that the first half of attention heads uses the first latent group and the second half uses the second.

**Design Motivation**: Provides a lightweight alternative that balances model capacity and efficiency within a 2-branch summation.

**Mechanism**: A group mapping function $\gamma(i)$ determines which latent group the $i$-th attention head uses; the output of each head is the sum of attention outputs from 2 branches.

#### 3. Variance Calibration

**Function**: Applies scaling factors to the query and KV latent states, and normalizes the multi-branch attention outputs.

**Design Motivation**: Theoretical analysis shows that the variance of NoPE keys is $d_c \sigma_w^2$ while that of RoPE keys is $d \sigma_w^2$; branch splitting further exacerbates this mismatch. Multi-branch summation also alters the variance of the output.

**Mechanism**:
- Latent state scaling: $C^Q \leftarrow \sqrt{d/d_c'} \cdot C^Q$, $C^{KV} \leftarrow \sqrt{4d/d_c} \cdot C^{KV}$
- Output scaling: MLRA-2 outputs are divided by $\sqrt{2}$; MLRA-4 outputs are divided by $2$.

#### 4. TP-Friendly Decoding

**Function**: Enables native 4-way TP decoding, with each device loading only $1.5 d_h$ of KV cache.

**Design Motivation**: The 4 latent blocks can be naturally distributed across 4 TP devices; each device loads only one latent block ($d_h$) plus the shared RoPE key ($0.5 d_h$).

**Mechanism**: Adopts MLA's weight absorption technique to absorb up-projections into the query side, then independently executes MQA-style decoding on each device, followed by an all-reduce summation.

### Loss & Training

- **Initialization**: Output projection parameters use zero initialization (outperforming standard $\mathcal{N}(0, 0.02)$); remaining parameters use standard initialization.
- **Optimizer**: AdamW with $(\beta_1, \beta_2)=(0.9, 0.95)$, weight decay 0.1, gradient clipping 1.0.
- **Learning rate**: Peak $1.6 \times 10^{-4}$, linear warmup for the first 2000 steps, followed by cosine annealing to 10%.
- **Training scale**: 2.9B parameters, FineWeb-Edu-100B dataset, 98.3B tokens, context length 2048, 8 × H100 GPUs.
- **Optional gating**: Adding a gating mechanism before the attention output projection can further reduce perplexity.

## Key Experimental Results

### Main Results

**Validation perplexity (average over 7 datasets, lower is better)**:

| Method | Wikipedia | C4 | Pile | RefinedWeb | Cosmopedia | FineWeb | FineWeb-Edu | **Avg.** |
|--------|-----------|----|------|------------|------------|---------|-------------|----------|
| MHA | 14.624 | 16.575 | 12.929 | 18.698 | 9.102 | 15.656 | 9.434 | 13.860 |
| GQA | 15.057 | 16.628 | 13.758 | 18.885 | 9.504 | 15.713 | 9.427 | 14.139 |
| MLA | 14.567 | 16.345 | 12.965 | 18.523 | 8.966 | 15.440 | 9.284 | 13.727 |
| GLA-2 | 14.605 | 16.323 | 13.225 | 18.509 | 9.118 | 15.424 | 9.249 | 13.779 |
| **MLRA-4** | **14.407** | **16.286** | **13.124** | **18.398** | **8.937** | **15.361** | **9.193** | **13.672** |

**Zero-shot commonsense reasoning accuracy (%)**:

| Method | ARC-E | ARC-C | OBQA | BoolQ | HellaSwag | Winogrande | PIQA | **Avg.** |
|--------|-------|-------|------|-------|-----------|------------|------|----------|
| MHA | 69.11 | 39.16 | 40.80 | 62.26 | 60.82 | 57.62 | 74.86 | 57.81 |
| GQA | 67.13 | 39.42 | 42.00 | 63.39 | 61.29 | 56.91 | 75.08 | 57.89 |
| MLA | 68.22 | 39.16 | 42.60 | 64.10 | 61.39 | 60.06 | 75.68 | 58.75 |
| **MLRA-4** | 67.63 | 41.38 | **43.00** | 61.74 | **62.16** | **61.48** | 74.48 | **58.84** |

### Ablation Study

**Effect of gating on perplexity (average over 7 datasets)**:

| Method | w/o Gating | w/ Gating | Gain |
|--------|-----------|-----------|------|
| GQA | 14.139 | 13.806 | -0.333 |
| MLA | 13.727 | 13.642 | -0.085 |
| GLA-2 | 13.779 | 13.701 | -0.078 |
| MLRA-2 | 13.804 | 13.651 | -0.153 |
| **MLRA-4** | **13.672** | **13.621** | -0.051 |

Additional ablation findings:
- **Zero initialization vs. $\mathcal{N}(0, 0.02)$**: Zero initialization consistently outperforms random initialization across all models.
- **Variance scaling**: MLA and GLA-2 benefit substantially; MLRA-2 benefits marginally, as the branch decomposition naturally mitigates variance mismatch.
- **Doubling attention heads**: Doubling the number of heads in GQA/MLA/GLA-2 while keeping total parameters fixed does not improve and in fact degrades performance.

### Key Findings

1. **MLRA-4 is universally optimal**: It achieves the best perplexity (13.672) and zero-shot reasoning accuracy (58.84%) among all baselines, including MLA.
2. **2.8× decoding speedup**: MLRA-4 achieves a stable 2.8× speedup over MLA in long-context decoding ranging from 128K to 2M tokens.
3. **1.05–1.26× over GQA**: MLRA-4 outperforms GQA in long-context decoding, with the gap widening as context length increases.
4. **TP=4 suffices to reach $1.5 d_h$**: GQA and GTA require 8-way TP to achieve a comparable per-device KV load, whereas MLRA requires only 4-way TP.
5. **Gating further improves results**: With gating, MLRA-4 perplexity drops to 13.621, remaining best-in-class.

## Highlights & Insights

- **Elegant mathematical motivation**: Starting from the block decomposition of MLA, the method moves the summation from the KV level to the attention output level—mathematically concise and intuitively clear.
- **Rigorous variance analysis**: The paper theoretically derives the variance of each component and provides explicit calibration strategies, avoiding empirical hyperparameter search.
- **Strong practical utility**: MLRA shares the same total KV cache size as MLA ($4.5 d_h$ per token); the only difference is shardability during distributed decoding, making migration from existing MLA systems low-cost.
- **Arithmetic intensity analysis**: MLRA-4 achieves an arithmetic intensity of approximately $2h$ (same as MLA), demonstrating that reduced memory loads do not sacrifice computational efficiency.
- **Complete engineering implementation**: An MLRA-4 kernel is implemented on top of FlashAttention-3 and validated on a real H100 cluster.

## Limitations & Future Work

1. **Validated only at 2.9B scale**: No experiments at 7B or larger scales; it remains to be verified whether MLRA's advantages hold for larger models.
2. **Single pretraining dataset**: Only FineWeb-Edu-100B is used; generalization to multilingual or code-mixed data is not validated.
3. **Limitations of Assumption 1**: The variance analysis assumes i.i.d. weight distributions independent of inputs, an assumption that does not hold strictly during training.
4. **Fixed 4-way decomposition**: MLRA-4 is tied to 4-way TP; 2-way or 8-way TP scenarios require MLRA-2 or further extension, respectively.
5. **Approximation in multi-branch summation**: Moving the summation outside the softmax fundamentally changes the attention distribution; while empirically effective, this is not theoretically equivalent to the original formulation.
6. **Instruction tuning and alignment not evaluated**: The study focuses solely on pretraining and does not cover SFT or RLHF stages.

## Related Work & Insights

- **MLA (DeepSeek-V2/V3)**: The direct predecessor of MLRA, which compresses KV cache via latent compression but does not support TP sharding.
- **GQA**: Grouped Query Attention achieves efficiency gains by reducing the number of KV heads, but KV cache still grows linearly with head count.
- **GLA-2 (Zadouri et al., 2025)**: The first work to attempt splitting MLA's latent head, but supports only 2-way TP.
- **TPA (Zhang et al., 2025)**: Tensor Product Attention constructs KVs as linear combinations of shared heads, with limited TP support.
- **FlashMLA / FlashAttention-3**: Efficient attention kernels; MLRA's kernel is implemented on top of FlashAttention-3.
- **LongCat (2025)**: First to observe the RoPE key variance mismatch issue; MLRA adopts and extends its scaling strategy.
- **Inspiration**: The paradigm of "lifting an internal summation to independent multi-branch computation" may generalize to other settings requiring latent compression combined with TP, such as KV cache quantization and sparse attention.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Precise insight; elegantly resolves the non-shardability of MLA through branch decomposition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple ablations, multi-dataset evaluation, and decoding speed/throughput benchmarks, though limited to the 2.9B scale.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Rigorous mathematical derivations, clear notation, and a complete logical chain from background to methodology.
- **Value**: ⭐⭐⭐⭐ — Directly addresses a practical pain point in MLA deployment, with significant utility for the DeepSeek model family and large-scale inference systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking](../../AAAI2026/autonomous_driving/comptrack_information_bottleneckguided_lowrank_dynamic_token_compres.md)
- [\[AAAI 2026\] Drive As You Like: Strategy-Level Motion Planning Based on A Multi-Head Diffusion Model](../../AAAI2026/autonomous_driving/drive_as_you_like_strategy-level_motion_planning_based_on_a_multi-head_diffusion.md)
- [\[ICCV 2025\] SRefiner: Soft-Braid Attention for Multi-Agent Trajectory Refinement](../../ICCV2025/autonomous_driving/srefiner_soft-braid_attention_for_multi-agent_trajectory_refinement.md)
- [\[ICLR 2026\] SMART-R1: Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning](advancing_multi-agent_traffic_simulation_via_r1-style_reinforcement_fine-tuning.md)
- [\[CVPR 2026\] TopoMaskV3: 3D Mask Head with Dense Offset and Height Predictions for Road Topology Understanding](../../CVPR2026/autonomous_driving/topomaskv3_3d_mask_head_with_dense_offset_and_height_predictions_for_road_topolo.md)

</div>

<!-- RELATED:END -->
