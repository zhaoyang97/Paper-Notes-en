---
title: >-
  [Paper Note] SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation
description: >-
  [ACL 2025 (Oral)][Model Compression][KV Cache Compression] This paper proposes the SCOPE framework, which separately optimizes Key-Value (KV) cache compression strategies for the prefill and decoding stages in long-context generation tasks. Specifically, the prefill stage preserves the full cache to maintain understanding capability, while the decoding stage utilizes a sliding window to select heavy hitters, further optimizing memory and transmission efficiency through adapti…
tags:
  - "ACL 2025 (Oral)"
  - "Model Compression"
  - "KV Cache Compression"
  - "Long-context Generation"
  - "Attention Mechanism"
  - "Inference Optimization"
  - "Memory Efficiency"
date: 2026-05-08
content_hash: 611177845f2fadcc
---

# SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation

**Conference**: ACL 2025 (Oral)  
**arXiv**: [2412.13649](https://arxiv.org/abs/2412.13649)  
**Code**: [https://github.com/Linking-ai/SCOPE](https://github.com/Linking-ai/SCOPE)  
**Area**: Model Compression  
**Keywords**: KV Cache Compression, Long-context Generation, Attention Mechanism, Inference Optimization, Memory Efficiency  

## TL;DR

This paper proposes the SCOPE framework, which separately optimizes Key-Value (KV) cache compression strategies for the prefill and decoding stages in long-context generation tasks. Specifically, the prefill stage preserves the full cache to maintain understanding capability, while the decoding stage utilizes a sliding window to select heavy hitters, further optimizing memory and transmission efficiency through adaptive and discontinuous strategies.

## Background & Motivation

**Background**: Large language models (LLMs) need to maintain a Key-Value (KV) cache during inference to store attention information of historical tokens. As the input context and generation length increase, the memory footprint of the KV cache becomes the core bottleneck of inference efficiency. Many KV cache compression methods exist, such as SnapKV, PyramidKV, H2O, and StreamingLLM.

**Limitations of Prior Work**: Almost all existing methods focus on compression during the prefill stage (i.e., the stage of processing input), while neglecting optimization during the decoding stage (i.e., the stage of generating output). This issue is particularly prominent in long-output generation tasks (e.g., mathematical reasoning, code generation) which require generating very long output sequences, leading to continuous growth of the KV cache during the decoding stage.

**Key Challenge**: The authors identify two key phenomena: (1) Excessive compression in the prefill stage damages the model's ability to understand the full context, especially in complex reasoning tasks where the model needs complete context information to plan reasoning paths. (2) In long-output reasoning tasks, the distribution of heavy hitters (tokens with high attention scores) shifts as the generation length increases, causing fixed heavy-hitter selection to fail in the later phase of decoding.

**Goal**: To design a KV cache compression framework that simultaneously optimizes both the prefill and decoding stages, significantly reducing memory consumption while maintaining inference quality.

**Key Insight**: By visualizing attention heatmaps, the authors clearly demonstrate the heavy hitter shift phenomenon — tokens attended to in early decoding differ significantly from those attended to in late decoding. This implies the need for a dynamic heavy hitter selection mechanism to adapt to changes in attention distribution during decoding.

**Core Idea**: Keep the prefill uncompressed (preserving full context understanding) and use a sliding window to dynamically track heavy hitters during decoding, processing the two stages separately.

## Method

### Overall Architecture

SCOPE is a stage-level KV cache compression framework. In the prefill stage, it retains the complete KV cache to maintain the model's full understanding of the input. In the decoding stage, it introduces a sliding window-based strategy to dynamically select the most critical heavy hitters, while utilizing adaptive and discontinuous strategies to further reduce memory footprint and transmission overhead. SCOPE is designed as a plug-and-play component that can be stack-integrated with existing prefill-only compression methods (such as SnapKV and PyramidKV).

### Key Designs

1. **Prefill Stage Full Preservation Strategy (Full Preservation)**:

    - Function: Retain the complete KV cache during the prefill stage.
    - Mechanism: Unlike existing methods, SCOPE advocates for no compression in the prefill stage. The rationale is that reasoning tasks (e.g., mathematical reasoning) require a global understanding of the complete input context to formulate correct reasoning strategies. Experiments show that compression in the prefill stage has a disproportionate negative impact on reasoning accuracy. However, SCOPE can also be stacked with other prefill compression methods, in which case SCOPE solely manages the optimization of the decoding stage.
    - Design Motivation: The authors experimentally find that preserving the full prefill KV cache on LongGenBench yields substantially better results than compressed counterparts, especially on tasks requiring multi-step reasoning such as GSM8K+.

2. **Sliding Window Selection (Sliding Window Selection)**:

    - Function: Dynamically select the most relevant KV cache during the decoding stage.
    - Mechanism: Maintain a fixed-size sliding window to track recent heavy hitters. Once a new token is generated, heavy hitter scores are recalculated based on the latest attention distribution, and the KV pairs retained in the window are updated. The sliding window mechanism allows heavy hitter selection to closely follow shifts in attention distribution. As the reasoning chain deepens, the tokens the model focuses on naturally change, and the sliding window ensures this change is correctly tracked.
    - Design Motivation: The heavy hitter shift phenomenon is observed in experiments: in the early phase of reasoning, the model focuses on key information in the problem description, while in the later phase, it focuses more on intermediate reasoning steps. A fixed set of heavy hitters cannot adapt to this shift.

3. **Adaptive and Discontinuous Optimization Strategies (Adaptive & Discontinuous)**:

    - Function: Further optimize memory usage and data transmission efficiency.
    - Mechanism: The **adaptive strategy** dynamically adjusts the retention budget based on actual current cache usage — allocating more budget when the heavy hitter concentration of certain attention heads is low, and reducing budget when concentration is high. The **discontinuous strategy** allows retaining non-contiguous KV cache fragments instead of requiring sequence-wise continuous tokens. This enables retaining only the truly important tokens while skipping irrelevant parts, reducing memory fragmentation, though requiring a special gather operation.
    - Design Motivation: Uniform budget allocation lacks flexibility (distribution of important tokens varies significantly across different layers and heads), and forcing contiguous storage wastes budget on unimportant tokens.

### Loss & Training

SCOPE is a training-free inference acceleration method that does not involve model training or fine-tuning. All strategies are applied during inference.

## Key Experimental Results

### Main Results

Evaluated on LongGenBench (under both 4K and 8K settings) using Llama3.1-8B-Instruct. Tasks include GSM8K+, MMLU+, and CSQA+.

| Method | Stage Optimization | GSM8K+ (4K) | MMLU+ (4K) | CSQA+ (4K) | Average |
|------|---------|-------------|------------|------------|------|
| Full KV (No compression) | — | Baseline | Baseline | Baseline | Baseline |
| H2O | prefill | Significant drop | Moderate drop | Mild drop | Decrease |
| SnapKV | prefill | Moderate drop | Mild drop | Mild drop | Moderate drop |
| StreamingLLM | prefill | Drastic drop | Drastic drop | Drastic drop | Drastic drop |
| **SCOPE (slide)** | decoding | Close to baseline | Close to baseline | Close to baseline | **Best** |
| **SCOPE (adaptive)** | decoding | Close to baseline | Close to baseline | Close to baseline | **Best** |
| SnapKV + SCOPE | prefill + decoding | Better than SnapKV | Better than SnapKV | Better than SnapKV | Significant improvement |

### Ablation Study

| Configuration | GSM8K+ Acc | Explanation |
|------|-----------|------|
| Full KV (No compression) | Highest | Upper bound |
| SCOPE (slide only) | Close to Full KV | Basic sliding strategy is highly effective |
| SCOPE (slide + adaptive) | Slightly better than slide | Adaptive allocation is more precise |
| SCOPE (slide + discontinuous) | Slightly better than slide | Allowing non-contiguous retention is more flexible |
| SCOPE (full) | Optimal | Combination of three strategies |
| H2O decoding only | Significantly worse than SCOPE | Fixed heavy hitter is inferior to sliding |
| Different window sizes | Scaling up with window size | 256-512 is the optimal cost-performance range |

### Key Findings

- **SCOPE as a plug-in is highly effective**: Stacking SCOPE onto prefill methods like SnapKV yields performance significantly superior to using prefill methods alone, validating the necessity of optimizing the decoding stage.
- **Heavy hitter shift indeed exists**: Visualized through attention heatmaps, the tokens focused on in different decoding stages vary greatly, confirming the rationality of the sliding window design.
- **Reasoning tasks like GSM8K+ are most affected by KV compression**: Tasks requiring multi-step reasoning are highly sensitive to KV cache completeness, where the advantages of SCOPE are most pronounced.
- **Memory efficiency**: SCOPE significantly reduces KV cache memory footprint in the decoding stage with negligible accuracy loss.

## Highlights & Insights

- **Clear two-stage divide-and-conquer strategy**: Decoupling the KV cache management of prefill and decoding, and applying optimal strategies to each, is an elegant and effective approach. Very few prior works have paid attention to decoding-stage compression.
- **Insightful discovery of heavy hitter shift**: This is valuable not only for KV compression but also for understanding the attention mechanism, indicating that the model's informational needs change dynamically during reasoning.
- **Plug-and-play design**: SCOPE can stack with any existing prefill compression method, lowering migration costs. This "incremental" design paradigm is highly referable — not to replace existing methods, but to complement what they overlooked.

## Limitations & Future Work

- Evaluation is mainly based on the single benchmark LongGenBench; its generalization to other long-context tasks remains to be verified.
- The gather operation in the Discontinuous strategy might introduce extra computational overhead; the speedup ratio on different hardware needs to be empirically tested.
- The choice of sliding window size requires manual hyperparameter tuning; fully automated budget allocation is not yet implemented.
- Future work could combine quantization (such as KV cache quantization) with SCOPE for joint optimization to further compress memory.
- For very long outputs (e.g., >16K tokens), verification is needed to confirm whether the sliding window strategy remains effective.

## Related Work & Insights

- **vs SnapKV / PyramidKV**: These methods only optimize heavy hitter selection in the prefill stage, ignoring the decoding stage. SCOPE supplements this missing link, and the two can be stacked together.
- **vs H2O**: While H2O also has an eviction strategy in the decoding stage, it uses a fixed accumulated attention score to select heavy hitters. SCOPE's sliding window design better adapts to the heavy hitter shift phenomenon.
- **vs StreamingLLM**: StreamingLLM adopts an extreme strategy of keeping only sink tokens and the most recent tokens, which incurs large losses on reasoning tasks. SCOPE is more fine-grained in retaining important tokens.

## Rating

- Novelty: ⭐⭐⭐⭐ The discovery of the two-stage divide-and-conquer approach and the heavy hitter shift is the core contribution, though individual techniques (sliding window, adaptive, etc.) are not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Experiments on LongGenBench cover multiple tasks and baselines, and plug-in experiments are convincing. However, more benchmarks and physical latency/throughput data are lacking.
- Writing Quality: ⭐⭐⭐⭐ The two observations in the motivation section are well-articulated, and the method description is clear.
- Value: ⭐⭐⭐⭐⭐ As an ACL 2025 Oral paper, it fills the blank in decoding-stage KV compression and possesses high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](../../NeurIPS2025/model_compression/chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [\[ICLR 2026\] FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension](../../ICLR2026/model_compression/freqkv_key-value_compression_in_frequency_domain_for_context_window_extension.md)
- [\[NeurIPS 2025\] Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry for Long-Context LLMs](../../NeurIPS2025/model_compression/homogeneous_keys_heterogeneous_values_exploiting_local_kv_cache_asymmetry_for_lo.md)
- [\[ICML 2025\] RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](../../ICML2025/model_compression/rocketkv_accelerating_long-context_llm_inference_via_two-stage_kv_cache_compress.md)
- [\[NeurIPS 2025\] KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments](../../NeurIPS2025/model_compression/keydiff_key_similarity-based_kv_cache_eviction_for_long-context_llm_inference_in.md)

</div>

<!-- RELATED:END -->
