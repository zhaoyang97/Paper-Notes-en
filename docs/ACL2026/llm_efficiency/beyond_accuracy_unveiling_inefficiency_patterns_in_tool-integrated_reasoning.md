---
title: >-
  [Paper Note] Beyond Accuracy: Unveiling Inefficiency Patterns in Tool-Integrated Reasoning
description: >-
  [ACL 2026][LLM Efficiency][KV-Cache] This paper proposes PTE (Prefill Token Equivalents), a hardware-aware efficiency metric for tool-integrated reasoning that unifies the costs of internal reasoning and external tool usage. Through large-scale experiments, it reveals four TIR inefficiency patterns: confirmatory tool usage, tool mixing, lack of tool prior
tags:
  - ACL 2026
  - LLM Efficiency
  - KV-Cache
date: 2026-05-08
content_hash: fa8c8dd005a63ecd
---
# Beyond Accuracy: Unveiling Inefficiency Patterns in Tool-Integrated Reasoning

**Conference**: ACL 2026  
**arXiv**: [2604.05404](https://arxiv.org/abs/2604.05404)  
**Code**: [https://github.com/sqs-ustc/tool-reasoning-framework-PTE](https://github.com/sqs-ustc/tool-reasoning-framework-PTE)  
**Area**: Others  
**Keywords**: Tool-Integrated Reasoning, Efficiency Metrics, KV-Cache, Prefill-Decode Asymmetry, Reasoning Patterns

## TL;DR
This paper proposes PTE (Prefill Token Equivalents), a hardware-aware efficiency metric for tool-integrated reasoning that unifies the costs of internal reasoning and external tool usage. Through large-scale experiments, it reveals four TIR inefficiency patterns: confirmatory tool usage, tool mixing, lack of tool priors, and tool format collapse.

## Background & Motivation

**Background**: LLMs demonstrate strong capabilities on complex tasks through Tool-Integrated Reasoning (TIR)—alternating between reasoning and external tool calls. Existing TIR benchmarks primarily focus on accuracy, while efficiency evaluation relies on simple token counts or the number of tool calls.

**Limitations of Prior Work**: Existing efficiency metrics fail to capture the actual model inference latency. The core issues are: (1) Tool calls cause KV-Cache eviction, requiring subsequent re-computation; (2) long and unfiltered tool returns inflate the context length, causing the HBM transfer overhead for each decoding step to increase linearly as the context grows. Token counts fail to reflect the cost asymmetry between the compute-intensive prefill phase and the memory-intensive decoding phase.

**Key Challenge**: In terms of token counts, the initial stages appear to have the highest consumption ("front-loading" effect), but in terms of actual hardware cost, the later steps are more expensive (context accumulation effect). Existing metrics cannot reveal this counter-intuitive cost distribution.

**Goal**: To design a unified, physical first-principles-based TIR efficiency metric and systematically identify inefficiency patterns within TIR.

**Key Insight**: Starting from the physical reality of Transformer inference—the prefill phase is compute-bound (limited by FLOPs) while the decoding phase is memory-bound (limited by HBM bandwidth). The costs of these two phases are inherently different.

**Core Idea**: Convert the memory operation costs of the decoding phase into equivalent prefill token counts (PTE) to measure the real hardware costs of internal reasoning and external tool usage on a unified scale.

## Method

### Overall Architecture
The core of PTE is to compress the real hardware cost of an entire tool-integrated reasoning trajectory into a single unified scalar. Instead of counting tokens, it splits each round of reasoning into two parts—the prefill phase (compute-intensive, limited by FLOPs) and the decoding phase (memory-intensive, limited by HBM bandwidth)—and uses a conversion coefficient $\gamma$ to translate the latter into "equivalent prefill tokens." This allows internal thinking and external tool calls to be measured using the same scale. For $k$ rounds of reasoning, the total cost is defined as $PTE = \sum_{i=1}^{k}(D_{prefill_i} + \gamma \cdot L_{seq_i} \cdot D_{decode_i})$, where the input is the round-by-round tokens and context length of a complete trajectory, and the output is an efficiency value highly aligned with measured latency.

### Key Designs

**1. PTE Conversion Coefficient: Unifying Heterogeneous Costs**

Token counts are distorted because they assume every token is equivalent, ignoring that prefill and decoding follow entirely different cost models on hardware. PTE defines the conversion coefficient as the ratio of equivalent compute cost of decoding to prefill cost: $\gamma = \frac{2 \cdot n_{layers} \cdot d_{model} \cdot HOI}{N_{params}}$, where $HOI$ is Hardware Operational Intensity (FLOPs/Byte), incorporating the GPU's compute power and bandwidth characteristics into the coefficient. Crucially, the decoding cost is not just multiplied by the number of generated tokens, but also by the cumulative sequence length $L_{seq}$—because every decoding step requires moving the entire KV-Cache from memory, an overhead that increases linearly with context length. This explains the counter-intuitive phenomenon where later steps become more expensive due to context accumulation, despite having fewer tokens.

**2. Four TIR Inefficiency Patterns: Categorization and Attribution**

Using this unified scale, the paper categorizes recurring wastes in TIR into four types to explain the origins of inefficiency. First is **confirmatory tool usage**, where the model has already reasoned the answer internally but still calls a tool to verify it, incurring high initial token costs for no reason. Second is **tool mixing**, where the model switches between multiple toolsets (e.g., Search, Python) within a single chain; while seemingly flexible, the PTE cost is extremely high without yielding accuracy gains. Third is **lack of tool priors**, where models not trained for tool use (e.g., forgetting a `print` statement resulting in no output) actually perform worse when tools are enabled. Fourth is **tool format collapse**, where the model only recognizes specific call formats from training and fails to trigger tools if names are slightly modified. These four types illustrate why "using more tools $\neq$ using tools better."

**3. Cross-Hardware Robustness Verification: Proving Efficiency as an Inherent Attribute**

A concern is whether $\gamma$ depends too heavily on specific hardware. The paper calculates PTE across five hardware platforms (H100/H200/A100/RTX4090/V100). Although $\gamma$ scaling factors vary significantly (from 0.18x to 1.0x) and absolute values differ, the efficiency rankings between models remain highly consistent—with Spearman rank correlation consistently exceeding 0.95. This indicates that PTE captures the intrinsic efficiency characteristics of model reasoning behavior rather than hardware accidents, supporting its credibility as a general metric.

## Key Experimental Results

### Main Results

| Benchmark | Best Model | PTE Difference | Key Findings |
| :--- | :--- | :--- | :--- |
| MATH500 | Similar accuracy across models | >10x | Similar accuracy but massive PTE variance |
| AIME24 | ~70% cluster | >10x | Thinking patterns yield high returns on difficult tasks |
| AIME25 | Qwen3-235B-Thinking +16.7% | 1.8x PTE | Thinking patterns are cost-effective for high difficulty |
| SimpleQA | Qwen3-235B-Thinking -3.4% | 4.2x PTE | Significant "overthinking" in simple tasks |

### Correlation Analysis: PTE vs. Token Count

| Metric | Correlation with Latency | p-value |
| :--- | :--- | :--- |
| PTE | r=0.9253 | <10⁻⁴ |
| Token Count | r=-0.3750 | 0.2558 |

### Key Findings
- PTE is highly positively correlated with actual latency (r=0.925), whereas token count shows almost no correlation (r=-0.375).
- The PTE of incorrect trajectories is consistently higher than that of correct ones—simply using more tools does not improve answer quality.
- Thinking patterns are a double-edged sword: highly cost-effective for difficult tasks (AIME25 +16.7% accuracy / 1.8x PTE) but severely wasteful for simple tasks (SimpleQA -3.4% accuracy / 4.2x PTE).

## Highlights & Insights
- **The design philosophy of PTE** is elegant—starting from physical first principles and using a single coefficient to unify two distinct cost modes. This is more scientific than heuristic token counting.
- **The insight that "higher accuracy relates to lower PTE"** is counter-intuitive yet profound—suggesting that efficient reasoning and correct reasoning are often the same thing, while inefficient reasoning is frequently accompanied by uncertainty and redundancy.
- **The classification of four inefficiency patterns** provides a clear direction for the optimization of TIR systems.

## Limitations & Future Work
- PTE assumes complete KV-Cache eviction; actual deployments might have partial cache reuse.
- The study only evaluates open-source models; the internal efficiency of closed-source API models cannot be measured.
- No specific optimization methods were proposed for the four inefficiency patterns; the work remains primarily at the diagnostic level.

## Related Work & Insights
- **vs. Traditional Token Counting**: PTE explicitly models prefill-decode asymmetry, improving the correlation coefficient with latency from -0.375 to 0.925.
- **vs. Serper Metrics**: Serper focuses on information search efficiency but does not model hardware costs; PTE provides physical significance.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to define TIR efficiency metrics from a hardware physics perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 benchmarks + multiple models + cross-hardware validation + industrial scenario validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Complete logic from first-principles derivation to experimental verification.
- Value: ⭐⭐⭐⭐⭐ PTE has the potential to become a standard metric for TIR efficiency evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning](tandem_riding_together_with_large_and_small_language_models_for_efficient_reason.md)
- [\[ICML 2026\] Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills](../../ICML2026/llm_efficiency/skill-based_mixture-of-experts_adaptive_routing_for_heterogeneous_reasoning_via_.md)
- [\[ACL 2025\] FUEL: Unveiling Environmental Impacts of Large Language Model Serving: A Functional Unit View](../../ACL2025/llm_efficiency/fuel_unveiling_environmental_impacts_of_llm_serving.md)
- [\[ICML 2026\] Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](../../ICML2026/llm_efficiency/beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)
- [\[ACL 2025\] LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks](../../ACL2025/llm_efficiency/longbench_v2_towards_deeper_understanding_and_reasoning_on_realistic_long-contex.md)

</div>

<!-- RELATED:END -->
