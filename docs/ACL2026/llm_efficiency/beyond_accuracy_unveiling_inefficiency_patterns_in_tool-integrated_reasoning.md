---
title: >-
  [Paper Note] Beyond Accuracy: Unveiling Inefficiency Patterns in Tool-Integrated Reasoning
description: >-
  [ACL 2026][LLM Efficiency][Tool-integrated reasoning] The paper proposes PTE (Prefill Token Equivalents), a hardware-aware efficiency metric for tool-integrated reasoning (TIR) that unifies the costs of internal reasonin…
tags:
  - "ACL 2026"
  - "LLM Efficiency"
  - "Tool-integrated reasoning"
  - "efficiency metrics"
  - "KV-Cache"
  - "prefill-decode asymmetry"
  - "reasoning patterns"
date: 2026-05-08
content_hash: c1e8f07dc06987c2
---

# Beyond Accuracy: Unveiling Inefficiency Patterns in Tool-Integrated Reasoning

**Conference**: ACL 2026  
**arXiv**: [2604.05404](https://arxiv.org/abs/2604.05404)  
**Code**: [https://github.com/sqs-ustc/tool-reasoning-framework-PTE](https://github.com/sqs-ustc/tool-reasoning-framework-PTE)  
**Area**: Others  
**Keywords**: Tool-integrated reasoning, efficiency metrics, KV-Cache, prefill-decode asymmetry, reasoning patterns

## TL;DR
The paper proposes PTE (Prefill Token Equivalents), a hardware-aware efficiency metric for tool-integrated reasoning (TIR) that unifies the costs of internal reasoning and external tool usage. Through large-scale experiments, it reveals four TIR inefficiency patterns: confirmatory tool use, tool mixing, lack of tool priors, and tool format collapse.

## Background & Motivation

**Background**: LLMs demonstrate strong capabilities through Tool-Integrated Reasoning (TIR) on complex tasks—alternating between internal reasoning and external tool calls. Existing TIR benchmarks primarily focus on accuracy, while efficiency evaluation relies on simple token counts or the number of tool calls.

**Limitations of Prior Work**: Existing efficiency metrics fail to capture true model reasoning latency. The core issues are: (1) tool calls cause KV-Cache eviction, requiring subsequent re-computation; (2) long and unfiltered tool returns inflate the context length, causing the HBM transfer overhead of each decoding step to increase linearly with context growth. Token counting cannot reflect the cost asymmetry between the compute-intensive prefill stage and the memory-intensive decoding stage.

**Key Challenge**: In terms of token counts, the early stages consume the most ("front-loading" effect), but in terms of actual hardware cost, later steps are more expensive (context accumulation effect). Existing metrics fail to reveal this counter-intuitive cost distribution.

**Goal**: Design a unified, first-principles-based TIR efficiency metric and systematically identify inefficiency patterns in TIR.

**Key Insight**: Starting from the physical reality of Transformer inference—the prefill stage is compute-bound (limited by FLOPs), while the decoding stage is memory-bound (limited by HBM bandwidth). The costs of these two stages are fundamentally different.

**Core Idea**: Convert the memory operation costs of the decoding stage into equivalent prefill token counts (PTE), providing a unified scale to measure the real hardware costs of both internal reasoning and external tool usage.

## Method

### Overall Architecture
PTE decomposes the cost of each reasoning round into prefill cost (compute-intensive) and decoding cost (memory-intensive). It utilizes a $\gamma$ coefficient to convert the decoding cost into prefill equivalent tokens. For $k$ rounds of reasoning, the total cost is defined as $PTE = \sum_{i=1}^{k}(D_{prefill_i} + \gamma \cdot L_{seq_i} \cdot D_{decode_i})$.

### Key Designs

1. **PTE Metric Design**:
    - **Function**: Provide a unified measurement for the real hardware costs of reasoning and tool usage in TIR.
    - **Mechanism**: The $\gamma$ coefficient is defined as the ratio of decoding equivalent compute cost to prefill cost: $\gamma = \frac{2 \cdot n_{layers} \cdot d_{model} \cdot HOI}{N_{params}}$, where HOI is the hardware operational intensity (FLOPs/Byte). Crucially, the decoding cost depends not only on the number of generated tokens but is also multiplied by the cumulative sequence length $L_{seq}$, as the cost of loading the KV-Cache grows linearly with context.
    - **Design Motivation**: Address the core flaws of token-counting metrics—ignoring prefill/decode cost asymmetry and the amplification effect of context length on decoding costs.

2. **Identification of Four TIR Inefficiency Patterns**:
    - **Function**: Systematically classify efficiency bottlenecks in TIR.
    - **Mechanism**: (1) Confirmatory tool use—the model reasons the answer internally before verifying it with a tool, causing unnecessary initial token consumption; (2) Tool mixing—alternating between multiple toolsets (e.g., Search + Python) in one chain, which appears flexible but incurs high PTE costs without accuracy gains; (3) Lack of tool priors—the model lacks tool-use training (e.g., forgetting `print` resulting in no output), where enabling tools actually degrades performance; (4) Tool format collapse—the model only recognizes tool call formats from training, failing if names are slightly modified.
    - **Design Motivation**: Beyond identifying inefficiency, classify and explain its root causes.

3. **Cross-Hardware Robustness Verification**:
    - **Function**: Ensure PTE validity across different hardware.
    - **Mechanism**: Validated on H100/H200/A100/RTX4090/V100; although the $\gamma$ scaling factor varies significantly (0.18x to 1.0x), the Spearman rank correlation of model efficiency rankings consistently exceeds 0.95.
    - **Design Motivation**: Prove that PTE captures intrinsic model efficiency characteristics rather than hardware-specific accidents.

### Loss & Training
PTE itself is an evaluation metric rather than a training objective, though the paper suggests it could serve as an efficiency penalty term within RL reward signals.

## Key Experimental Results

### Main Results

| Benchmark | Best Model | PTE Difference | Key Findings |
|------|---------|---------|---------|
| MATH500 | Multiple models close in accuracy | >10x | Huge PTE variance despite similar accuracy |
| AIME24 | ~70% cluster | >10x | Thinking patterns yield high returns on hard tasks |
| AIME25 | Qwen3-235B-Thinking +16.7% | 1.8x PTE | Thinking patterns are cost-effective for high difficulty |
| SimpleQA | Qwen3-235B-Thinking -3.4% | 4.2x PTE | Severe "over-thinking" in simple tasks |

### PTE vs. Token Count Correlation Analysis

| Metric | Correlation with Latency (r) | p-value |
|------|-------------|------|
| PTE | r=0.9253 | <10⁻⁴ |
| Token Count | r=-0.3750 | 0.2558 |

### Key Findings
- PTE is highly positively correlated with actual latency (r=0.925), whereas token counts show almost no correlation (r=-0.375).
- PTE for incorrect trajectories is consistently higher than for correct ones—simply using more tools does not improve answer quality.
- Thinking patterns are a double-edged sword: they are cost-effective for high-difficulty tasks (AIME25 +16.7%/1.8x) but wasteful for simple tasks (SimpleQA -3.4%/4.2x).

## Highlights & Insights
- **PTE's design philosophy** is elegant—starting from physical first principles, a single coefficient unifies two distinct cost modes. This is far more scientific than heuristic token counting.
- **The discovery that "higher accuracy leads to lower PTE"** is counter-intuitive yet profound—it suggests that efficient reasoning and correct reasoning are often the same thing, while inefficient reasoning is frequently accompanied by uncertainty and redundancy.
- **The classification of four inefficiency patterns** provides clear directions for the optimization of TIR systems.

## Limitations & Future Work
- PTE assumes complete KV-Cache eviction; actual deployments may feature partial cache reuse.
- The study only evaluates open-source models; the internal efficiency of closed-source API models remains unmeasurable.
- Specific optimization methods for the four inefficiency patterns were not proposed, as the paper primarily focuses on diagnosis.

## Related Work & Insights
- **vs. Traditional Token Counting**: PTE explicitly models prefill-decode asymmetry, improving the latency correlation coefficient from -0.375 to 0.925.
- **vs. Serper Metric**: While Serper focuses on information search efficiency, it does not model hardware costs. PTE provides a foundation in physics.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to define a TIR efficiency metric from the perspective of hardware physics.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 benchmarks + multiple models + cross-hardware validation + industrial scenario validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Complete logic flow from first-principle derivation to experimental validation.
- Value: ⭐⭐⭐⭐⭐ PTE has the potential to become a standard metric for TIR efficiency evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning](tandem_riding_together_with_large_and_small_language_models_for_efficient_reason.md)
- [\[ICML 2026\] Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills](../../ICML2026/llm_efficiency/skill-based_mixture-of-experts_adaptive_routing_for_heterogeneous_reasoning_via_.md)
- [\[ICML 2026\] Beyond Sunk Costs: Boosting LLM Pre-training Efficiency via Orthogonal Growth of Mixture-of-Experts](../../ICML2026/llm_efficiency/beyond_sunk_costs_boosting_llm_pre-training_efficiency_via_orthogonal_growth_of_.md)
- [\[NeurIPS 2025\] Unmasking COVID-19 Vulnerability in Nigeria: Mapping Risks Beyond Urban Hotspots](../../NeurIPS2025/llm_efficiency/unmasking_covid-19_vulnerability_in_nigeria_mapping_risks_beyond_urban_hotspots.md)
- [\[NeurIPS 2025\] L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models](../../NeurIPS2025/llm_efficiency/l-mtp_leap_multi-token_prediction_beyond_adjacent_context_for_large_language_mod.md)

</div>

<!-- RELATED:END -->
