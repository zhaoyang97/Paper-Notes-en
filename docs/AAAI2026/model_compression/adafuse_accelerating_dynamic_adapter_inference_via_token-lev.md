---
title: >-
  [Paper Note] AdaFuse: Accelerating Dynamic Adapter Inference via Token-Level Pre-Gating and Fused Kernel Optimization
description: >-
  [AAAI 2026][Model Compression][Dynamic Adapters] To address the severe inference latency overhead (250%–950%) of dynamic MoE-LoRA adapters, this paper proposes a token-level pre-gating architecture that performs a single global routing decision at the first layer. Combined with a custom SGMM fused CUDA kernel that merges all activated LoRA adapters into the backbone in one shot, the approach reduces decoding latency by 2.4× while preserving model accuracy.
tags:
  - AAAI 2026
  - Model Compression
  - Dynamic Adapters
  - MoE-LoRA
  - CUDA Kernel Optimization
  - Inference Acceleration
  - Token-Level Routing
date: 2026-05-08
content_hash: 61a8bb9b63582a16
---

# AdaFuse: Accelerating Dynamic Adapter Inference via Token-Level Pre-Gating and Fused Kernel Optimization

**Conference**: AAAI 2026
**arXiv**: [2603.11873v1](https://arxiv.org/abs/2603.11873v1)
**Code**: N/A
**Area**: Model Compression / Efficient LLM Inference
**Keywords**: Dynamic Adapters, MoE-LoRA, CUDA Kernel Optimization, Inference Acceleration, Token-Level Routing

## TL;DR
To address the severe inference latency overhead (250%–950%) of dynamic MoE-LoRA adapters, this paper proposes a token-level pre-gating architecture that performs a single global routing decision at the first layer. Combined with a custom SGMM fused CUDA kernel that merges all activated LoRA adapters into the backbone in one shot, the approach reduces decoding latency by 2.4× while preserving model accuracy.

## Background & Motivation

The combination of MoE and LoRA (i.e., dynamic adapters) has become a popular paradigm for enhancing multi-task capabilities of LLMs—by conditionally activating different LoRA experts to increase model capacity. However, the authors identify a widely overlooked practical issue: although dynamic adapters introduce only 1%–5% additional parameters and less than 1% extra FLOPs, inference latency increases dramatically by 250%–950%. Taking MoRAL as an example, decoding latency spikes from 2.4 ms/token to 8.6 ms/token.

The root cause is not computational cost, but **fragmented CUDA kernel invocations**. Each LoRA adapter requires two CUDA kernel calls (down projection and up projection), and multi-layer dynamic routing causes the number of kernel launches to scale linearly with the number of layers. Since CUDA kernel launch overhead does not scale linearly with matrix size, the small matrices in LoRA lead to a situation where "launch overhead far exceeds actual computation."

Naively applying layer-by-layer merging to existing methods is also insufficient—MoRAL with a simple merging strategy still incurs 4.5 ms/token latency (88% higher than the backbone alone), because merging itself requires additional CUDA kernel calls.

## Core Problem
**How can the expressive capacity of dynamic adapters be retained while eliminating the inference latency overhead caused by fragmented CUDA kernel invocations?** This is fundamentally an algorithm–system co-design problem: existing layer-wise or block-wise routing structures are inherently unfriendly to kernel-level optimization.

## Method

### Overall Architecture
The core philosophy of AdaFuse is "decide once, apply everywhere." Each input token passes through a Top-2 router only at the first expanded linear layer; the resulting expert activation weights are then directly applied to all subsequent layers. This makes each token's execution path fully determined before inference, transforming "dynamic" routing into "per-token static" routing, which opens the door for system-level optimization.

The inference pipeline consists of three steps: (1) the router determines which two LoRA experts are activated for the current token and their weights; (2) the SGMM kernel fuses all activated LoRA parameters into every layer of the backbone in a single pass; (3) the fused backbone performs a forward pass identical to the original model.

### Key Designs

1. **Token-Level Pre-Gating**: Unlike existing methods that make independent routing decisions at each layer or block, AdaFuse places a single Top-2 router only at the first linear layer. This design is motivated by the observation that semantically similar tokens tend to activate consistent expert patterns across different layers. During fine-tuning, the formulation is $y^l = f^l(x^l) + \sum_{i=1}^{N} G^1(x^1)_i \cdot E_i^l(x^l)$, where $G^1$ depends only on the first-layer input. This "one-shot decision" architecture reduces routing computation from $O(L)$ to $O(1)$, and more critically, enables adapters to be pre-merged.

2. **Fused Adapter Switching**: When adjacent tokens activate different experts, the previous adapter must be "un-merged" before the new one is merged. AdaFuse unifies the un-merging of the previous token's adapter (with negative weights) and the merging of the current token's adapter (with positive weights) into a single operation: $\text{Fused\_LoRA\_DOWN} = \text{concat}[-(\text{LoRA\_DOWN})_{t-1}, (\text{LoRA\_DOWN})_t]$. This requires only one CUDA kernel call to complete adapter switching across all layers.

3. **SGMM CUDA Kernel (Segmented Gather Matrix Multiplication)**: Adapted from the SGMV concept in Punica, this custom CUDA kernel is specifically designed for fused adapter switching. SGMM packs the LoRA merging operations across all layers into a single batched GEMM, using a tiling strategy to distribute merging tasks across different thread blocks for parallel execution. A prefetch buffer mechanism hides memory load latency, and in-place addition reduces memory overhead.

### Loss & Training
The training procedure follows standard MoE-LoRA instruction fine-tuning. The only architectural difference is that the router is placed solely at the first layer, with all layers sharing the same set of routing weights. No special optimization is applied during the prefill phase, as LLM generation latency is predominantly determined by the decoding phase.

## Key Experimental Results

| Dataset / Metric | Metric | AdaFuse | Best Baseline | Notes |
|-----------------|--------|---------|--------------|-------|
| General tasks (avg. 5, Llama2-7B) | Accuracy | 60.12% | 60.45% (PESC) | Comparable; best on MMLU and TruthfulQA |
| Domain tasks (avg. 3, Llama2-7B) | Accuracy | 83.60% | 84.20% (MoLA) | Near-competitive; best on CommonsenseQA (79.03%) |
| Domain tasks (avg. 3, Mistral-7B) | Accuracy | 87.24% | 87.06% (PESC) | Marginally better |
| Decoding latency (Llama2-7B) | ms/token | 3.1 (+29%) | 8.5 (+254%, PESC) | **2.7× faster than the fastest baseline** |
| Memory usage (Llama2-7B) | GB | 13.8 (+7%) | 13.1 (+2%, PESC) | Slightly higher but acceptable |

### Ablation Study

- **SGMM kernel is critical**: Replacing SGMM with simple layer-by-layer merging increases latency from 3.1 ms to 4.2 ms (+35%), confirming the necessity of kernel-level optimization.
- **Pre-gating vs. layer-wise routing**: Switching to layer-wise routing (similar to MoRAL) causes latency to return to 8.6 ms—pre-gating is the key to acceleration.
- **Top-k selection**: Top-2 achieves the best trade-off (Top-1 degrades accuracy; Top-4 increases latency).
- **LoRA rank sensitivity**: Increasing rank from 64 to 128 yields a marginal accuracy gain with nearly no latency change—the SGMM kernel is insensitive to rank.
- **Number of experts**: $N=8$ achieves the best balance between accuracy and overhead.
- **Pre-gating + simple merging vs. MoRAL + simple merging**: 4.2 ms vs. 4.5 ms, indicating that the token-level pre-gating architecture itself contributes to speedup, though the larger gain comes from SGMM.
- **MOLA latency is extremely high (25.3 ms/token)**: Due to MOLA's block-wise routing, MoE computation is performed at every layer, resulting in the highest number of kernel invocations.

## Highlights & Insights

- **Precise problem diagnosis**: Identifying that the bottleneck of dynamic adapters lies not in FLOPs but in CUDA kernel launch overhead is an incisive observation with significant value to the field.
- **Algorithm–system co-design**: Rather than patching the system level, the approach redesigns the algorithm structure to be system-friendly. The principle of "first make the algorithm optimizable, then make the system efficient" is worth emulating.
- **Sign-concatenation trick for fused switching**: Unifying "un-merge old adapter + merge new adapter" into a single operation is both elegant and effective.
- **2.4× speedup with negligible accuracy loss**: Reducing dynamic adapter latency overhead from 250%+ to only 29% demonstrates high practical value.

## Limitations & Future Work

- **Pre-gating assumption requires further validation**: Whether the assumption that "routing patterns are consistent across layers" holds for larger models or more complex tasks is not analyzed in the paper.
- **Only decoding phase is optimized**: The prefill phase is not optimized; acceleration effects for long-input scenarios (e.g., RAG, long documents) are not evaluated.
- **Limited model scale**: Validation is conducted only on 7B models; scalability of SGMM to larger models (13B, 70B) is unknown.
- **Slight memory increase**: Simultaneously maintaining original weights and LoRA parameters for dynamic merging/un-merging leaves room for memory efficiency improvements.
- **No open-source code**: Reproducibility is limited, especially for system-level implementations such as the SGMM kernel.

## Related Work & Insights

- **vs. MoRAL/MOLA (layer-wise routing)**: These methods make independent routing decisions at each layer, causing the number of CUDA calls to scale linearly with depth. AdaFuse fundamentally eliminates this problem through a single global routing decision. Accuracy is comparable, but latency is substantially better.
- **vs. PESC (block-wise routing)**: PESC is more efficient than layer-wise methods but still requires inter-block routing switches. AdaFuse is 2.7× faster than PESC in latency, with only a marginal 0.3% accuracy drop on general tasks, while matching or exceeding PESC on specific tasks.
- **vs. static LoRA**: Static LoRA can be merged directly into the backbone after training, incurring zero inference overhead. AdaFuse aims to approach this upper bound; its current latency overhead is only 29%, at the cost of one routing computation and SGMM switching.

The design principle of "making algorithms system-friendly" is transferable to MoE adapters in vision Transformers (e.g., domain-specific adapter switching in VLMs) and broader PEFT serving scenarios. The pre-gating concept can be extended to multimodal models where different modalities use different adapters—for instance, visual tokens and language tokens in VLMs may require different LoRA experts. More generally, "decide-once, apply-everywhere" is a universal latency optimization paradigm applicable to any architecture that currently relies on per-layer or per-block dynamic decisions.

## Limitations & Future Work

- **Tested only on 7B models**: Effectiveness and speedup ratios on larger models (13B/70B) remain unverified.
- **Prefill phase not optimized**: Only the decoding phase is accelerated; prefill latency is on par with existing methods.
- **Limitations of the global routing assumption**: Assuming all layers should activate the same experts may not hold for certain tasks (e.g., lower layers may require syntactic experts while higher layers require semantic experts).
- **Limited number of experts**: Experiments cover $N=4$–$16$ experts; scalability to larger-scale MoE settings is unexplored.
- **Training cost insufficiently discussed**: The cost of fine-tuning LoRA experts and the router is not thoroughly analyzed.

## Rating

- Novelty: ⭐⭐⭐⭐ — Clear algorithm–system co-design rationale; token-level pre-gating combined with fused kernels is genuinely innovative.
- Experimental Thoroughness: ⭐⭐⭐ — Validation is limited to 7B models; larger-scale verification is absent.
- Writing Quality: ⭐⭐⭐⭐ — Thorough problem analysis with rich latency profiling figures and tables.
- Value: ⭐⭐⭐⭐ — Addresses a critical bottleneck for deploying dynamic adapters in production LLM serving systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LoRA on the Go: Instance-level Dynamic LoRA Selection and Merging](../../ACL2026/model_compression/lora_on_the_go_instance-level_dynamic_lora_selection_and_merging.md)
- [\[ACL 2026\] DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing](../../ACL2026/model_compression/dash-kv_accelerating_long-context_llm_inference_via_asymmetric_kv_cache_hashing.md)
- [\[ICLR 2026\] TiTok: Transfer Token-level Knowledge via Contrastive Excess to Transplant LoRA](../../ICLR2026/model_compression/titok_transfer_token-level_knowledge_via_contrastive_excess_to_transplant_lora.md)
- [\[AAAI 2026\] Earth-Adapter: Bridge Geospatial Domain Gaps with Mixture of Frequency Adaptation](earth-adapter_bridge_the_geospatial_domain_gaps_with_mixture_of_frequency_adapta.md)
- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](../../ACL2026/model_compression/adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)

</div>

<!-- RELATED:END -->
