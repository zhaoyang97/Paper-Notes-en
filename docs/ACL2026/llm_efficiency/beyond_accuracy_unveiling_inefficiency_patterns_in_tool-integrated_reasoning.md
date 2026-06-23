---
title: >-
  [Paper Note] Beyond Accuracy: Unveiling Inefficiency Patterns in Tool-Integrated Reasoning
description: >-
  [ACL 2026][LLM Efficiency][KV-Cache] This paper proposes PTE (Prefill Token Equivalents), a hardware-aware efficiency metric for tool-integrated reasoning (TIR) that unifies the costs of internal reasoning and external tool usage. Through large-scale experiments, it reveals four inefficiency patterns in TIR: confirmatory tool use, tool mixing, lack of too
tags:
  - ACL 2026
  - LLM Efficiency
  - KV-Cache
date: 2026-05-08
content_hash: 669dd55c0e91269a
---
# Beyond Accuracy: Unveiling Inefficiency Patterns in Tool-Integrated Reasoning

**Conference**: ACL 2026  
**arXiv**: [2604.05404](https://arxiv.org/abs/2604.05404)  
**Code**: [https://github.com/sqs-ustc/tool-reasoning-framework-PTE](https://github.com/sqs-ustc/tool-reasoning-framework-PTE)  
**Area**: Others  
**Keywords**: Tool-integrated reasoning, efficiency metrics, KV-Cache, prefill-decode asymmetry, reasoning patterns

## TL;DR
This paper proposes PTE (Prefill Token Equivalents), a hardware-aware efficiency metric for tool-integrated reasoning (TIR) that unifies the costs of internal reasoning and external tool usage. Through large-scale experiments, it reveals four inefficiency patterns in TIR: confirmatory tool use, tool mixing, lack of tool priors, and tool format collapse.

## Background & Motivation

**Background**: LLMs have demonstrated strong capabilities in complex tasks through tool-integrated reasoning (TIR)—alternating between internal reasoning and external tool calls. Existing TIR benchmarks primarily focus on accuracy, while efficiency evaluation relies on simple token counts or the number of tool calls.

**Limitations of Prior Work**: Existing efficiency metrics fail to capture the real inference latency of models. The core problem lies in: (1) Tool calls causing KV-Cache eviction, requiring subsequent re-computation; (2) Long and unfiltered tool returns bloating the context length, causing HBM transfer overhead for each decoding step to increase linearly with the context. Token counting cannot reflect the cost asymmetry between the compute-bound prefilling phase and the memory-bound decoding phase.

**Key Challenge**: In terms of token counts, initial phases seem to consume the most ("front-loading" effect), but in terms of actual hardware cost, later steps are more expensive due to context accumulation. Existing metrics fail to reveal this counter-intuitive cost distribution.

**Goal**: To design a unified, physical first-principles-based TIR efficiency metric and systematically identify inefficiency patterns in TIR.

**Key Insight**: Starting from the physical reality of Transformer inference—the prefilling phase is compute-bound (limited by FLOPs), while the decoding phase is memory-bound (limited by HBM bandwidth). The costs of the two are fundamentally different.

**Core Idea**: Convert the memory operation costs of the decoding phase into equivalent prefill token counts (PTE) to measure the real hardware cost of internal reasoning and external tool use on a unified scale.

## Method

### Overall Architecture
The core of PTE is to compress the real hardware cost of an entire tool-integrated reasoning trajectory into a unified scalar. Instead of counting tokens, it splits each reasoning round into two halves—the prefilling phase (compute-bound, limited by FLOPs) and the decoding phase (memory-bound, limited by HBM bandwidth). A conversion coefficient $\gamma$ is then used to translate the latter into "equivalent prefill tokens," allowing internal thinking and external tool calls to be measured by the same yardstick. For $k$ rounds of reasoning, the total cost is expressed as $PTE = \sum_{i=1}^{k}(D_{prefill_i} + \gamma \cdot L_{seq_i} \cdot D_{decode_i})$, where the input is the per-round token count and context length of a complete trajectory, and the output is an efficiency value highly aligned with measured latency.

### Key Designs

**1. PTE Conversion Coefficient: Unifying Heterogeneous Costs**

Token counting is distorted because it assumes every token is equivalent, ignoring that prefilling and decoding follow two distinct cost models on hardware. PTE defines the conversion coefficient as the ratio of equivalent compute cost of decoding to the prefill cost: $\gamma = \frac{2 \cdot n_{layers} \cdot d_{model} \cdot HOI}{N_{params}}$, where $HOI$ is the hardware operational intensity (FLOPs/Byte), directly incorporating GPU compute/bandwidth characteristics into the coefficient. A more critical aspect lies in the summation: the decoding cost is not just multiplied by the number of generated tokens, but also by the cumulative sequence length $L_{seq}$—since every decoding step requires moving the entire KV-Cache from memory, and this overhead grows linearly with context length. This precisely explains the counter-intuitive phenomenon: while token counts suggest the early "front-loading" is most expensive, real hardware costs show later steps are costlier due to context accumulation.

**2. Four TIR Inefficiency Patterns: Classification and Attribution**

With a unified scale, the paper categorizes recurring wastes in TIR into four types, explaining their origins. The first is **confirmatory tool use**—the model has already reasoned out the answer internally but still calls a tool to verify it, incurring high initial token costs for nothing. The second is **tool mixing**, where the model switches between multiple toolsets like Search and Python in a single chain; while seemingly flexible, the PTE cost is extremely high without proportional gains in accuracy. The third is **lack of tool priors**, where the model has not been trained for tool use (e.g., forgetting to write `print` resulting in no output), making tools a drag on performance. The fourth is **tool format collapse**, where the model only recognizes formats seen during training; slight changes in tool names prevent correct triggering. These four categories illustrate that "more tool usage $\neq$ better usage."

**3. Cross-Hardware Robustness Verification**

One concern is that $\gamma$ depends on specific hardware, which might make PTE an accidental product of a specific GPU. The paper calculates PTE on H100, H200, A100, RTX4090, and V100. Although the scaling factor of $\gamma$ varies from 0.18x to 1.0x between different cards and absolute values differ significantly, the efficiency rankings between models remain highly consistent—with Spearman rank correlation stably exceeding 0.95. This indicates that PTE captures intrinsic efficiency characteristics of model reasoning behavior rather than hardware accidents, supporting its credibility as a general metric.

While PTE is an evaluation metric rather than a training objective, the paper notes it can be directly integrated into RL reward signals as an efficiency penalty, guiding models to maintain accuracy while learning to avoid unnecessary tool calls.

## Key Experimental Results

### Main Results

| Benchmark | Best Model | PTE Difference | Key Findings |
|-----------|------------|----------------|--------------|
| MATH500 | Multi-model accuracy similar | >10x | Similar accuracy but massive PTE variance |
| AIME24 | ~70% cluster | >10x | Thinking mode has high returns on hard tasks |
| AIME25 | Qwen3-235B-Thinking +16.7% | 1.8x PTE | Thinking mode is worth the cost for high difficulty |
| SimpleQA | Qwen3-235B-Thinking -3.4% | 4.2x PTE | Thinking mode leads to severe "overthinking" on simple tasks |

### Correlation Analysis: PTE vs. Token Count

| Metric | Correlation with Latency | p-value |
|--------|--------------------------|---------|
| PTE | r=0.9253 | <10⁻⁴ |
| Token Count | r=-0.3750 | 0.2558 |

### Key Findings
- PTE is highly positively correlated with actual latency (r=0.925), while token count shows almost no correlation (r=-0.375).
- The PTE of incorrect trajectories is consistently higher than that of correct ones—simply using more tools does not improve answer quality.
- Thinking mode is a double-edged sword: it is highly cost-effective for difficult tasks (AIME25 +16.7%/1.8x) but causes severe waste on simple tasks (SimpleQA -3.4%/4.2x).

## Highlights & Insights
- **The design philosophy of PTE** is elegant—starting from physical first principles, it unifies two distinct cost models with a single coefficient. This is far more scientific than heuristic token counting.
- **The discovery that "higher accuracy leads to lower PTE"** is counter-intuitive but profound—it suggests that efficient reasoning and correct reasoning are often the same thing, while inefficient reasoning is often accompanied by uncertainty and redundancy.
- **The classification of four inefficiency patterns** provides a clear direction for optimizing TIR systems.

## Limitations & Future Work
- PTE assumes complete KV-Cache eviction, whereas actual deployments might involve partial cache reuse.
- The evaluation is limited to open-source models; the internal efficiency of closed-source API models cannot be measured.
- The paper does not propose specific optimization methods for the four inefficiency patterns, remaining primarily at the diagnostic level.

## Related Work & Insights
- **vs. Traditional Token Counting**: PTE explicitly models the prefill-decode asymmetry, improving the correlation coefficient with latency from -0.375 to 0.925.
- **vs. Serper Metric**: Serper focuses on information search efficiency but does not model hardware costs; PTE provides physical significance.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to define TIR efficiency metrics from a hardware physical perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 benchmarks + multiple models + cross-hardware validation + industrial scenario validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Complete logic from first-principles derivation to experimental verification.
- Value: ⭐⭐⭐⭐⭐ PTE is expected to become a standard metric for TIR efficiency evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning](tandem_riding_together_with_large_and_small_language_models_for_efficient_reason.md)
- [\[ICML 2026\] Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills](../../ICML2026/llm_efficiency/skill-based_mixture-of-experts_adaptive_routing_for_heterogeneous_reasoning_via_.md)
- [\[ACL 2025\] FUEL: Unveiling Environmental Impacts of Large Language Model Serving: A Functional Unit View](../../ACL2025/llm_efficiency/fuel_unveiling_environmental_impacts_of_llm_serving.md)
- [\[ICML 2026\] Q-Delta: Beyond Key–Value Associative State Evolution](../../ICML2026/llm_efficiency/q-delta_beyond_key-value_associative_state_evolution.md)
- [\[ACL 2025\] LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks](../../ACL2025/llm_efficiency/longbench_v2_towards_deeper_understanding_and_reasoning_on_realistic_long-contex.md)

</div>

<!-- RELATED:END -->
