---
title: >-
  [Paper Note] Not All Bits Are Equal: Scale-Dependent Memory Optimization Strategies for Reasoning Models
description: >-
  [ICLR 2026][LLM Efficiency][Reasoning Models] Through 1700+ groups of experiments, this paper systematically demonstrates that the conclusion "4-bit quantization is memory-optimal" for non-reasoning models fails for reasoning models. The memory-optimal strategy is determined by the model's **effective size** (parameter count × bit-width), with a critical point at "8-bit 4B". Small models should spend memory on larger weights, while large models should spend it on longer gener…
tags:
  - "ICLR 2026"
  - "LLM Efficiency"
  - "Reasoning Models"
  - "KV cache compression"
  - "weight quantization"
  - "test-time scaling"
  - "memory-accuracy trade-off"
  - "Pareto frontier"
date: 2026-05-08
content_hash: ea3f6e60c6b29546
---

# Not All Bits Are Equal: Scale-Dependent Memory Optimization Strategies for Reasoning Models

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=b6qQmQ2F13](https://openreview.net/forum?id=b6qQmQ2F13)  
**Code**: [https://github.com/krafton-ai/not-all-bits-are-equal](https://github.com/krafton-ai/not-all-bits-are-equal)  
**Area**: LLM Inference Efficiency / Memory Optimization  
**Keywords**: Reasoning Models, KV cache compression, weight quantization, test-time scaling, memory-accuracy trade-off, Pareto frontier  

## TL;DR
Through 1700+ groups of experiments, this paper systematically demonstrates that the conclusion "4-bit quantization is memory-optimal" for non-reasoning models fails for reasoning models. The memory-optimal strategy is determined by the model's **effective size** (parameter count × bit-width), with a critical point at "8-bit 4B". Small models should spend memory on larger weights, while large models should spend it on longer generation or more parallel sampling.

## Background & Motivation

**Background**: Previous research on the memory-performance trade-off for non-reasoning LLMs almost exclusively focused on **compressing weights**, since weight memory was far larger than the KV cache. This led to the nearly universal conclusion that "4-bit quantization is the cross-scale memory-optimal choice" (Dettmers & Zettlemoyer 2023).

**Limitations of Prior Work**: Modern reasoning models significantly increase the number of generated tokens. The KV cache grows linearly with generation length and can become the memory bottleneck. The paper provides a counter-intuitive figure: Qwen3-4B with 4-bit weights occupies only 2.49 GB, but its KV cache for 32k tokens requires 4.42 GB ($\approx 1.8 \times$ weights). In batch inference, weight overhead is further amortized, making the aggregated KV cache the dominant factor.

**Key Challenge**: When the KV cache becomes the primary memory component, does the quantization experience established for non-reasoning models still hold? Reasoning models introduce three new dimensions that significantly impact the memory-accuracy trade-off: **generation length, parallel sampling, and KV cache compression**. However, how these jointly trade off with traditional model size and weight precision has not been systematically studied.

**Goal**: Under a **fixed memory budget**, answer "how to trade off between five knobs—model size, weight precision, token budget, parallel sampling group size, and KV cache compression—to maximize reasoning performance."

**Core Idea**: **[Empirical Laws + Critical Point]** The memory-optimal strategy is not scale-agnostic but is dominated by the model's **effective size** (effective size = parameter count × bit-width, proportional to weight memory). There exists a critical threshold of "8-bit 4B ($\approx 4.2$ GB)," across which strategies undergo a qualitative change.

## Method

### Overall Architecture
This paper does not propose a new method but conducts a **controlled empirical study**. It decomposes the memory overhead of reasoning models into weights and KV cache: $M = M_{\text{weights}}(N, P_W) + M_{\text{kv}}(N, \pi_{kv}, T, G)$. It then systematically scans five factors—parameter count $N$, weight precision $P_W \in \{4, 8, 16\}$, token budget $T$ (2k–30k), group size $G$ (1–16, majority voting), and KV compression strategy $\pi_{kv}$ (eviction / quantization). The Pareto frontiers for "accuracy vs. total memory" are plotted across four benchmarks to derive deployment guidelines. The primary models used are the Qwen3 family (0.6B–32B), with validation on DeepSeek-R1-Distill and OpenReasoning-Nemotron, covering over 1700 scenarios.

### Key Designs

**1. Effective size as a unified metric: Combining "parameter count" and "bit-width".** The paper explicitly distinguishes two concepts: `model size` refers to parameter count $N$, while `effective size` / `scale` refers to weight memory $M_{\text{weights}} \approx N \cdot P_W$. All scale-dependent conclusions are expressed relative to the threshold of "8-bit 4B ($\approx 4.2$ GB) effective size" rather than just parameter count. This allows different configurations like 4B-16bit, 8B-8bit, and 14B-4bit to be compared on the same axis, providing a calculable criterion for the strategy split between "small vs. large models."

**2. Memory allocation law under serial scaling: Small models add weights, large models add tokens.** Fix $G=1$ and keep full KV cache, using **budget forcing** to control generation length $T$ from 2k to 30k tokens. Conclusion: For models with an effective size **below** 8-bit 4B, it is more cost-effective to invest saved memory into larger weights than into longer generation. For example, 1.7B-8bit with 6k tokens outperforms 0.6B-8bit with 18k tokens, and 4B-4bit with 10k tokens exceeds 1.7B-8bit with 18k tokens; furthermore, larger effective size configurations have **lower end-to-end latency** (latency is dominated by token budget), making them strictly dominant. For models **reaching or exceeding** 8-bit 4B, the opposite holds—extending generation until saturation is the more memory-efficient way to increase accuracy (token budgets stabilize above 20k for budgets > 10 GB).

**3. Optimal weight precision depends on task nature; 4-bit is no longer universal.** The paper finds "4-bit optimal" only holds for **knowledge-intensive** tasks (GPQA-Diamond), where capacity matters for storing knowledge. However, for **mathematical reasoning / code generation** (AIME25, LiveCodeBench, MATH500), 4-bit is consistently memory-inefficient: 8B-8bit consistently outperforms 14B-4bit, and 32B-4bit is strictly outperformed by both 14B-8bit and 8B-16bit—directly contradicting Dettmers & Zettlemoyer (2023). The intuition is that mathematical reasoning relies on numerical precision in weights, which aggressive 4-bit quantization disrupts.

**4. Parallel scaling is only cost-effective for large models, and optimal group size grows with budget.** When introducing parallel sampling $G > 1$ (where KV cache expands proportionally to $G$ in exchange for majority voting accuracy), results are also scale-dependent. For effective sizes below 8-bit 4B, all parallel scaling configurations fall below the serial Pareto frontier. Above the threshold, parallel scaling pushes the frontier, and the **optimal group size $G$ increases monotonically with budget** ($4 \le G < 8$ is optimal in the 16.4–28.9 GB range; $G \ge 8$ for budgets > 28.9 GB). Furthermore, using an external PRM for Best-of-N (ActPRM-X, 7B/13.28 GB fixed overhead) is almost always memory-inefficient compared to self-contained majority voting.

**5. KV cache compression is a necessity; Eviction vs. Quantization choice depends on effective size.** Quantizing weights alone is insufficient for memory optimality. Across all weight precisions, KV eviction (R-KV) and KV quantization (HQQ backend, 2/4/8-bit) push the Pareto frontier beyond the uncompressed baseline, especially in the < 10 GB low-memory region. Their characteristics differ: quantization reduces memory per token (shifting the curve left with accuracy loss), while eviction sets a KV memory ceiling (shifting the curve vertically). Decision rule: KV eviction provides a better trade-off for effective sizes **smaller** than 8-bit 4B, while the two are evenly matched for larger models.

## Key Experimental Results

### Experimental Scale & Setting

| Dimension | Values |
|------|------|
| Models | Qwen3 0.6B–32B (Main), DeepSeek-R1-Distill, OpenReasoning-Nemotron |
| Tasks | AIME25, MATH500 (Math), GPQA-Diamond (Knowledge), LiveCodeBench (Code) |
| Weight Quantization | GPTQ 4/8-bit (+ AWQ, FP8 for validation) |
| Token Budget | 2k–30k (Step 4k, budget forcing) |
| Parallel Sampling | Majority voting, $G$ up to 16 |
| KV Compression | Eviction: R-KV / StreamingLLM; Quantization: HQQ |
| Total Scenarios | 1700+ |
| Evaluation | Average of 32 generations per instance, temperature 0.6 |

### Qwen3 Weight vs. KV cache Memory Comparison (Excerpt from Table 1, GB)

| Model | 4-bit Weight | 8-bit Weight | 16-bit Weight | KV@30k | KV@30k×16 |
|------|-----------|-----------|------------|--------|-----------|
| Qwen3-0.6B | 0.50 | 0.71 | 1.40 | 3.20 | 51.27 |
| Qwen3-4B | 2.49 | 4.19 | 7.49 | 4.12 | 65.91 |
| Qwen3-8B | 5.68 | 8.94 | 15.26 | 4.12 | 65.91 |
| Qwen3-32B | 18.01 | 32.66 | 61.02 | 7.32 | 117.19 |

> Key Observation: The 30k KV cache for small models (3–4 GB) already approaches or exceeds their weight memory; at $G=16$, the KV cache (50–117 GB) completely dwarfs the weights, confirming that "KV cache is the bottleneck for reasoning models."

### Key Findings
- **Finding 1**: The allocation strategy between weights and KV cache is scale-dependent—for effective size < 8-bit 4B, increase weights; for $\ge$ 8-bit 4B, increase tokens until saturation.
- **Finding 2**: 4-bit is broadly optimal for knowledge-intensive tasks; math/code require higher precision, with 8-bit being optimal for small models and 8/16-bit both viable for large models.
- **Finding 3**: Parallel scaling is only cost-effective for effective sizes $\ge$ 8-bit 4B, and the optimal $G$ increases with the budget.
- **Finding 4**: Quantizing weights alone is insufficient; compressing the KV cache pushes the frontier across all precisions.
- **Finding 5**: KV eviction outperforms KV quantization when the effective size is < 8-bit 4B.

## Highlights & Insights
- **Falsifying a widely accepted "universal law" and replacing it with an actionable critical point**: Moving from "4-bit is always optimal" to "depend on whether the effective size crosses 8-bit 4B" is the greatest value of this empirical paper.
- **The "effective size" metric is clever**: By using parameter count × bit-width to represent weight memory, heterogeneous configurations are unified on a single axis, enabling cross-configuration comparison and transfer.
- **Counter-intuitive conclusion on extending small model generation**: Often viewed as a way to "trade latency for memory," results show it neither saves memory nor is faster (latency is dominated by token budget), making it strictly dominated by configurations with larger effective sizes.
- **Joint multi-knob optimization instead of single-point optimization**: By placing model size, precision, length, sampling count, and KV compression into a single Pareto framework, it provides a "lookup table by budget and task type" rather than a single trick.

## Limitations & Future Work
- **Essentially an empirical study without a universal prescription**: The authors state that conclusions do not provide specific configurations for every task/model but offer "principles." Whether the "8-bit 4B" critical point holds strictly for models with larger architectural differences (MoE, different attention variants) remains to be verified.
- **Task coverage limited to math, code, and knowledge**: Memory dynamics might differ for long documents, multi-turn agents, or tool-use scenarios.
- **Latency/Throughput only analyzed in the appendix**: The main text uses memory as the sole optimization target. Real-world deployment is also constrained by latency, throughput, and batch scheduling.
- **Fixed KV compression methods**: Eviction used R-KV and quantization used HQQ. More aggressive or newer KV compression methods might shift the critical point.

## Related Work & Insights
- **Weight Quantization**: GPTQ, AWQ, FP8—the paper uses these for weight compression to verify that conclusions are not dependent on a specific scheme.
- **KV cache Compression**: Eviction-based (StreamingLLM window + attention sinks, R-KV redundancy-aware retention) and Quantization-based (HQQ online per-channel quantization).
- **Test-time scaling**: Budget forcing (Muennighoff et al. 2025) for serial extension, majority voting/self-consistency (Wang et al. 2022) for parallel expansion, and PRM/Best-of-N for external verification.
- **Insights**: (1) Any "universal optimal configuration" conclusion should be re-evaluated based on deployment scenarios (reasoning vs. non-reasoning, task type, batching); (2) When deploying reasoning models, first estimate if the "effective size" falls above or below the critical point to decide whether to invest budget in weights or test-time compute; (3) Be cautious with aggressive 4-bit quantization for precision-sensitive tasks like math and code.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Does not propose a new method, but falsifies a widely accepted universal law and replaces it with an actionable scale-dependent threshold.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 1700+ scenarios, 3 model families, 4 task categories, and 5 knobs scanned jointly, with robustness validated across quantization schemes.
- **Writing Quality**: ⭐⭐⭐⭐ — Five Findings are clearly summarized, charts are organized around the Pareto frontier, and conclusions can be directly applied as deployment guidelines.
- **Value**: ⭐⭐⭐⭐ — Provides a practical guide for reasoning model deployment based on budget and task, with direct guidance for industrial memory budget allocation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models](not_all_models_suit_expert_offloading_on_local_routing_consistency_of_mixture-of.md)
- [\[ICLR 2026\] ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models](thinkv_thought-adaptive_kv_cache_compression_for_efficient_reasoning_models.md)
- [\[ICLR 2026\] Mixture-of-Experts Can Surpass Dense LLMs Under Strictly Equal Resource](mixture-of-experts_can_surpass_dense_llms_under_strictly_equal_resource.md)
- [\[ICLR 2026\] Attention Is All You Need for KV Cache in Diffusion LLMs](attention_is_all_you_need_for_kv_cache_in_diffusion_llms.md)
- [\[ICLR 2026\] Sparse Attention Adaptation for Long Reasoning](sparse_attention_adaptation_for_long_reasoning.md)

</div>

<!-- RELATED:END -->
