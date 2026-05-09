---
title: >-
  [Paper Note] Beyond Accuracy: Unveiling Inefficiency Patterns in Tool-Integrated Reasoning
description: >-
  [ACL 2026][Tool-integrated reasoning] This paper proposes PTE (Prefill Token Equivalents), a hardware-aware efficiency metric for tool-integrated reasoning (TIR) that unifies the costs of internal reasoning and external tool use. Through large-scale experiments, the paper identifies four inefficiency patterns in TIR: confirmatory tool use, tool mixing, lack of tool priors, and tool format collapse.
tags:
  - ACL 2026
  - Tool-integrated reasoning
  - efficiency metrics
  - KV-Cache
  - prefill-decode asymmetry
  - reasoning patterns
date: 2026-05-08
content_hash: 93fc12747ba38b75
---

# Beyond Accuracy: Unveiling Inefficiency Patterns in Tool-Integrated Reasoning

**Conference**: ACL 2026
**arXiv**: [2604.05404](https://arxiv.org/abs/2604.05404)
**Code**: [https://github.com/sqs-ustc/tool-reasoning-framework-PTE](https://github.com/sqs-ustc/tool-reasoning-framework-PTE)
**Area**: Others
**Keywords**: Tool-integrated reasoning, efficiency metrics, KV-Cache, prefill-decode asymmetry, reasoning patterns

## TL;DR
This paper proposes PTE (Prefill Token Equivalents), a hardware-aware efficiency metric for tool-integrated reasoning (TIR) that unifies the costs of internal reasoning and external tool use. Through large-scale experiments, the paper identifies four inefficiency patterns in TIR: confirmatory tool use, tool mixing, lack of tool priors, and tool format collapse.

## Background & Motivation

**Background**: LLMs have demonstrated strong capabilities on complex tasks via tool-integrated reasoning (TIR), which interleaves reasoning steps with external tool calls. Existing TIR benchmarks primarily focus on accuracy, with efficiency evaluation relying on simple token counts or tool call frequencies.

**Limitations of Prior Work**: Existing efficiency metrics fail to capture actual model inference latency. The core issues are: (1) tool calls cause KV-Cache eviction, requiring recomputation in subsequent steps; (2) long, unfiltered tool outputs inflate context length, causing HBM transfer overhead per decoding step to increase linearly with context. Token counts do not reflect the cost asymmetry between the compute-intensive prefill phase and the memory-intensive decode phase.

**Key Challenge**: From a token-count perspective, costs appear highest at early steps (a "front-loading" effect), yet from an actual hardware-cost perspective, later steps are more expensive due to context accumulation. Existing metrics cannot reveal this counterintuitive cost distribution.

**Goal**: Design a unified, physics-first-principles efficiency metric for TIR and systematically identify inefficiency patterns therein.

**Key Insight**: The physical reality of Transformer inference—the prefill phase is compute-bound (FLOPs-limited) while the decode phase is memory-bound (HBM bandwidth-limited)—means the two phases have fundamentally different costs.

**Core Idea**: Convert the memory operation cost of the decode phase into an equivalent number of prefill tokens (PTE), enabling a unified scale for measuring the true hardware cost of both internal reasoning and external tool use.

## Method

### Overall Architecture
PTE decomposes the cost of each reasoning turn into a prefill cost (compute-intensive) and a decode cost (memory-intensive), converting the decode cost into prefill-equivalent token counts via a $\gamma$ coefficient. For $k$ reasoning turns, the total cost is $PTE = \sum_{i=1}^{k}(D_{prefill_i} + \gamma \cdot L_{seq_i} \cdot D_{decode_i})$.

### Key Designs

1. **PTE Metric Design**:

    - Function: Uniformly measure the true hardware cost of both reasoning and tool use in TIR.
    - Mechanism: The $\gamma$ coefficient is defined as the ratio of decode-equivalent compute cost to prefill cost: $\gamma = \frac{2 \cdot n_{layers} \cdot d_{model} \cdot HOI}{N_{params}}$, where HOI is the hardware operational intensity (FLOPs/Byte). Crucially, decode cost depends not only on the number of generated tokens but is also multiplied by the cumulative sequence length $L_{seq}$, since the cost of loading the KV-Cache grows linearly with context.
    - Design Motivation: Address the core flaw of token-count metrics—namely, their ignorance of prefill/decode cost asymmetry and the amplification effect of context length on decode cost.

2. **Identification of Four TIR Inefficiency Patterns**:

    - Function: Systematically categorize efficiency bottlenecks in TIR.
    - Mechanism: (1) *Confirmatory tool use*—the model internally reasons to an answer and then verifies it with a tool, incurring large unnecessary first-step token costs; (2) *Tool mixing*—alternating between multiple tool sets (e.g., search + Python) within a single reasoning chain, which appears flexible but incurs very high PTE with no accuracy gain; (3) *Lack of tool priors*—the model lacks tool-use training (e.g., forgetting to call `print`, resulting in no output), so enabling tools actually degrades performance; (4) *Tool format collapse*—the model only recognizes the exact tool-call format seen during training, and minor renaming prevents correct invocation.
    - Design Motivation: Not only detect inefficiency, but also categorize and explain its root causes.

3. **Cross-Hardware Robustness Validation**:

    - Function: Ensure PTE remains consistently valid across different hardware platforms.
    - Mechanism: Validation is conducted on five hardware platforms (H100/H200/A100/RTX4090/V100). Despite the $\gamma$ scaling factor varying widely from 0.18× to 1.0×, the Spearman rank correlation of model efficiency rankings consistently exceeds 0.95.
    - Design Motivation: Demonstrate that PTE captures intrinsic model efficiency characteristics rather than artifacts of specific hardware configurations.

### Loss & Training
PTE is an evaluation metric rather than a training objective, though the paper notes it can serve as an efficiency penalty term in RL reward signals.

## Key Experimental Results

### Main Results

| Benchmark | Top Model | PTE Difference | Key Finding |
|-----------|-----------|---------------|-------------|
| MATH500 | Multiple models near equal accuracy | >10× | Similar accuracy but vastly different PTE |
| AIME24 | ~70% accuracy cluster | >10× | Thinking mode yields high returns on difficult tasks |
| AIME25 | Qwen3-235B-Thinking +16.7% | 1.8× PTE | Thinking mode is cost-effective on hard tasks |
| SimpleQA | Qwen3-235B-Thinking −3.4% | 4.2× PTE | Thinking mode severely "overthinks" on simple tasks |

### PTE vs. Token Count Correlation Analysis

| Metric | Correlation with Latency | p-value |
|--------|--------------------------|---------|
| PTE | r=0.9253 | <10⁻⁴ |
| Token count | r=−0.3750 | 0.2558 |

### Key Findings
- PTE is highly correlated with actual latency (r=0.925), whereas token count shows almost no correlation (r=−0.375).
- PTE for incorrect trajectories is consistently higher than for correct ones—simply using more tools does not improve answer quality.
- Thinking mode is a double-edged sword: it is cost-effective on hard tasks (AIME25: +16.7% accuracy at 1.8× PTE) but severely wasteful on simple tasks (SimpleQA: −3.4% accuracy at 4.2× PTE).

## Highlights & Insights
- **The design philosophy of PTE** is notably elegant—deriving from physical first principles, a single coefficient unifies two fundamentally different cost regimes, making it far more principled than heuristic token counts.
- **The finding that higher accuracy correlates with lower PTE** is counterintuitive yet profound, suggesting that efficient reasoning and correct reasoning tend to be the same thing, while inefficient reasoning often accompanies uncertainty and redundancy.
- **The taxonomy of four inefficiency patterns** provides clear optimization directions for TIR system design.

## Limitations & Future Work
- PTE assumes complete KV-Cache eviction; in practice, partial cache reuse may occur.
- Only open-source models are evaluated; the internal efficiency of closed-source API models cannot be measured.
- The paper does not propose concrete optimization methods for the four identified inefficiency patterns, remaining primarily at the diagnostic level.

## Related Work & Insights
- **vs. conventional token counting**: PTE explicitly models prefill-decode asymmetry, improving the latency correlation coefficient from −0.375 to 0.925.
- **vs. Serper metric**: Serper focuses on information retrieval efficiency without modeling hardware costs; PTE provides a physically grounded interpretation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First hardware-physics-grounded definition of a TIR efficiency metric.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five benchmarks, multiple models, cross-hardware validation, and industrial scenario validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Logically complete, from first-principles derivation to experimental verification.
- Value: ⭐⭐⭐⭐⭐ PTE has the potential to become the standard efficiency metric for TIR evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] AMS-IO-Bench and AMS-IO-Agent: Benchmarking and Structured Reasoning for Analog and Mixed-Signal Integrated Circuit Input/Output Design](../../AAAI2026/others/ams-io-bench_and_ams-io-agent_benchmarking_and_structured_re.md)
- [\[AAAI 2026\] Verification-Guided Context Optimization for Tool Calling via Hierarchical LLMs-as-editors](../../AAAI2026/others/verification-guided_context_optimization_for_tool_calling_via_hierarchical_llms-.md)
- [\[AAAI 2026\] Beyond World Models: Rethinking Understanding in AI Models](../../AAAI2026/others/beyond_world_models_rethinking_understanding_in_ai_models.md)
- [\[ICLR 2026\] Beyond Linearity in Attention Projections: The Case for Nonlinear Queries](../../ICLR2026/others/beyond_linearity_in_attention_projections_the_case_for_nonlinear_queries.md)
- [\[AAAI 2026\] Towards Temporal Fusion Beyond the Field of View for Camera-based Semantic Scene Completion](../../AAAI2026/others/towards_temporal_fusion_beyond_the_field_of_view_for_camera-based_semantic_scene.md)

</div>

<!-- RELATED:END -->
