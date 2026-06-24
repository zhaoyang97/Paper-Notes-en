---
title: >-
  [Paper Note] MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference
description: >-
  [Multimodal Efficiency] This paper proposes MadaKV, a modality-aware KV cache eviction strategy. Through two core components—Modality Preference Adaptation (MPA) and Hierarchical Compression Compensation (HCC)—MadaKV significantly reduces KV cache memory consumption (by 80-95%) and decoding latency (1.3x to 1.5x speedup) while maintaining performance on multimodal long-context tasks.
tags:
  - "Multimodal Efficiency"
date: 2026-05-08
content_hash: 397a7d1083290d34
---

# MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference

| Attribute | Content |
|------|------|
| Title | MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference |
| Conference | ACL2025 |
| arXiv | [2506.15724](https://arxiv.org/abs/2506.15724) |
| Code | - |
| Area | Multimodal VLM / Efficient Inference |
| Keywords | KV Cache Eviction, Multimodal LLM, Modality Preference, Long-Context, Inference Efficiency |

## TL;DR

This paper proposes MadaKV, a modality-aware KV cache eviction strategy. Through two core components—Modality Preference Adaptation (MPA) and Hierarchical Compression Compensation (HCC)—MadaKV significantly reduces KV cache memory consumption (by 80-95%) and decoding latency (1.3x to 1.5x speedup) while maintaining performance on multimodal long-context tasks.

## Background & Motivation

- **KV Cache Issue**: In autoregressive generation, KV cache stores the Key and Value of all historical tokens to avoid redundant computation. However, the memory overhead scales dramatically with sequence length.
- **Limitations of Prior Work**: Existing KV cache eviction methods such as StreamingLLM, H2O, and SnapKV are designed for single-modality (pure text) scenarios. They lack modality awareness, leading to suboptimal performance in multimodal contexts.
- **Multimodal Specificity**:
    - Varying information density between modalities: Text tokens represent semantic concepts concisely, whereas visual tokens require a massive number of elements to represent fine-grained spatial information.
    - Varying modality preferences across different attention heads.
    - Varying importance of modalities across different tasks (e.g., text search focuses more on textual tokens, whereas image retrieval focuses on visual tokens).
- **Limitations of LOOK-M**: Although customized for multimodal models, LOOK-M empirically fixes modality priority (prioritizing the eviction of visual tokens), ignoring the dynamic variation in modality importance.

## Method

### Overall Architecture

MadaKV is a plug-and-play modality-adaptive KV cache compression strategy consisting of two core components:

### 1. Modality Preference Adaptation (MPA)

**Modality Preference Measurement**: Token importance is calculated using proxy tokens ($\mathcal{P}$, representing the last few tokens of the prompt):

$$\psi(i) = \sum_{j \in \mathcal{P}} \alpha_{j \to i}$$

The total preference weights for visual and textual tokens are then computed respectively:

$$w_v = \sum_{i \in X_v} \psi(i), \quad w_t = \sum_{i \in X_t} \psi(i)$$

**Modality Budget Allocation**: The cache budget for each modality within each attention head is proportionally allocated based on preference weights:

$$\varphi_v^{l,h} = \frac{w_v}{w_v + w_t} \varphi^l, \quad \varphi_t^{l,h} = \frac{w_t}{w_v + w_t} \varphi^l$$

Instead of treating all tokens uniformly, each attention head independently performs KV cache eviction at the modality level.

### 2. Hierarchical Compression Compensation (HCC)

**Design Motivation**: Attention patterns vary significantly across layers—shallow layers show scattered attention, deep layers exhibit concentrated attention on a few tokens, and final layers tend to distribute attention uniformly again. Thus, different layers require distinct cache budgets.

**Sparsity Calculation**: For each attention head, the minimum number of tokens required to cover a proportion $\theta$ of the total importance is found:

$$k_v^{l,h} = \min\{|\mathcal{C}_v| \mid \sum_{i \in \mathcal{C}_v} \psi(i) \geq \theta w_v\}$$

**Inter-layer Compensation**: The budget compensation value for the current layer is cumulative of the difference between the actual demand and the allocated budget:

$$K^l = \sum_{h=1}^{H}(k_v^{l,h} + k_t^{l,h} - \varphi^l)$$

The budget for the next layer is adjusted accordingly:

$$\varphi^{l+1} = \varphi^l - \frac{K^l}{L - l}$$

A positive compensation value indicates overspending in the current layer, requiring subsequent layers to share the deduction; a negative value means budget is saved and can be reserved for layers that need it more.

### Key Findings

Experiments on LLaVA-v1.5-7B with MileBench yield observations across four dimensions:

| Dimension | Observation |
|------|------|
| Token Level | Only 20% of tokens capture approximately 90% of the attention scores; textual attention is highly concentrated, while visual attention is heavily dispersed. |
| Attention Head Level | Heading behaviors differ, showing distinct preferences for modalities as reflected by the variance in attention score distributions allocated to different modalities. |
| Layer Level | Attention is uniformly distributed in the initial layers, concentrated in the middle layers, and uniform again in the final layers—requiring the preservation of more KV cache in the first and last layers. |
| Task Level | Textual tokens are more critical in text search tasks, whereas visual tokens play a dominant role in image retrieval tasks. |

## Experiments

### Main Results: Performance Comparison on MileBench

On LLaVA-v1.5-7B (at a 20% cache budget), MadaKV achieves an average accuracy of 28.22%, closely matching the 28.59% of Full Cache:

| Method | TN | IEdit | MMCoQA | STD | ALFRED | Avg |
|------|-----|-------|--------|-----|--------|------|
| Full Cache | 9.68 | 7.98 | 33.50 | 16.32 | 15.18 | 28.59 |
| StreamingLLM | 3.12 | 3.59 | 26.00 | 11.77 | 3.73 | 20.91 |
| H2O | 2.50 | 5.51 | 28.00 | 15.73 | 14.86 | 25.80 |
| SnapKV | 3.27 | 6.03 | 29.00 | 14.82 | 14.40 | 26.43 |
| LOOK-M | 3.34 | 6.51 | 29.50 | 15.79 | 13.96 | 26.47 |
| **MadaKV** | **9.38** | **6.97** | **31.00** | **15.85** | **15.06** | **28.22** |

In the TN (Text Needle) task, MadaKV exhibits the most prominent advantage (9.38 vs. 3.34 for LOOK-M) because the task requires locating textual information amidst substantial visual distractions.

### Qwen2.5-VL-7B Results

MadaKV remains highly effective on Qwen2.5-VL-7B, yielding an average performance of 62.74% compared to 63.34% of Full Cache, showing a degradation of only 0.6 percentage points.

### Effect of Different Cache Budgets

- Under a 50% cache budget, most methods perform closely to Full Cache.
- MadaKV consistently outperforms baseline methods across all budget thresholds.
- On the TN task, the performance of MadaKV with a 20% budget is on par with LOOK-M utilizing a 60% budget.
- The advantage of MadaKV is especially pronounced when the budget is extremely low (<10%).

### Efficiency Analysis

| Configuration | Decoding Latency | KV Cache VRAM |
|------|------|-----------|
| Full Cache | 27.85 ms/token | 1.63 GiB |
| MadaKV (20%) | 19.57 ms/token | 0.41 GiB |
| MadaKV (5%) | 17.16 ms/token | 0.16 GiB |

Under a 20% budget, decoding speed increases by 1.42x, and VRAM footprint is reduced by 75%.

### Ablation Study

| MPA | HCC | TN | IEdit | ALFRED |
|-----|-----|----|-------|--------|
| ✘ | ✘ | 2.47 | 3.55 | 14.32 |
| ✔ | ✘ | 6.58 | 5.72 | 14.86 |
| ✘ | ✔ | 5.51 | 5.19 | 14.61 |
| ✔ | ✔ | **9.38** | **6.97** | **15.06** |

The two components are complementary. MPA contributes more significantly (especially bringing a 4.11 improvement on the TN task), and HCC further boosts performance by 3.07.

## Highlights & Insights

1. **Crucial Innovation in Modality Awareness**: Unlike existing approaches that treat all tokens uniformly, MadaKV is the first to make KV cache eviction "aware" of modality information, handling tokens differentially based on the modality preference of each attention head.
2. **Plug-and-play Design**: It requires no model fine-tuning and can be directly applied to any Transformer-based MLLM.
3. **Inter-layer Coordination Mechanism**: HCC achieves global optimization through cross-layer budget compensation, avoiding error propagation cascade caused by excessive compression at a single layer.
4. **Comprehensive Experimental Evaluation**: The study covers various models (LLaVA-v1.5-7B/13B, Qwen2.5-VL-7B), task types, and budget levels.
5. **Clever Use of Proxy Tokens**: Prompt-end tokens are selected as proxies to evaluate token importance since they typically represent task-relevant queries.

## Limitations & Future Work

1. Validated only on 7B/13B scale models; larger models like 34B/70B have not been evaluated.
2. Only covers visual and textual modalities; not yet extended to video, audio, etc.
3. Has not been validated in ultra-long-context (e.g., 100K+ tokens) scenarios.
4. The selection strategy for proxy tokens in MPA is relatively coarse (simply choosing the last few tokens), which may not be the optimal design.

## Related Work & Insights

- **KV Cache Optimization**: Quantization techniques (FP16 $\to$ INT8), eviction policies (StreamingLLM, H2O, SnapKV, PyramidKV).
- **Multimodal KV Cache**: LOOK-M's prioritization of evicting visual tokens, lightweight analysis of FastGen and H2O.
- **Efficient Attention**: Sparse attention (Longformer), low-rank approximation.
- **MLLM Inference Optimization**: Activation checkpointing, offloading, dynamic memory management.

## Rating ⭐⭐⭐⭐

**Pros**: Clear problem definition, solid quantitative observations (four-dimensional attention analysis), intuitive and effective method, comprehensive experimental coverage, and significant performance impact (80-95% memory reduction, 1.3-1.5x speedup).

**Cons**: The core idea is relatively straightforward (allocating budgets proportionally based on modality attention scores); the multimodal long-context capability is evaluated only on a single benchmark (MileBench), leaving generalizability to be further explored.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling](../../ICCV2025/vlm_efficiency/matvlm_hybrid_mamba-transformer_for_efficient_vision-language_modeling.md)
- [\[AAAI 2026\] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](../../AAAI2026/vlm_efficiency/rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)
- [\[ICML 2025\] MMInference: Accelerating Pre-filling for Long-Context VLMs via Modality-Aware Permutation Sparse Attention](../../ICML2025/vlm_efficiency/mminference_accelerating_pre-filling_for_long-context_vlms_via_modality-aware_pe.md)
- [\[NeurIPS 2025\] PrefixKV: Adaptive Prefix KV Cache is What Vision Instruction-Following Models Need for Efficient Generation](../../NeurIPS2025/vlm_efficiency/prefixkv_adaptive_prefix_kv_cache_is_what_vision_instruction.md)
- [\[ICCV 2025\] AirCache: Activating Inter-modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference](../../ICCV2025/vlm_efficiency/aircache_activating_inter_modal_relevancy_kv_cache_compression_for_efficient_large_vision_language_model.md)

</div>

<!-- RELATED:END -->
