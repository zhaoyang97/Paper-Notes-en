---
title: >-
  [Paper Note] SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications
description: >-
  [NeurIPS 2025 Spotlight][LLM Agent][Speculative Decoding] Caches long token sequences via suffix trees and achieves 5.3× speedup through adaptive speculation length, targeting highly predictable repetitive inference tasks in agent scenarios.
tags:
  - "NeurIPS 2025 Spotlight"
  - "LLM Agent"
  - "Speculative Decoding"
  - "Suffix Tree"
  - "Agent Inference"
  - "Inference Acceleration"
  - "Training-Free"
date: 2026-05-08
content_hash: 2c2f7ad87ae6e4c9
---

# SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2411.04975](https://arxiv.org/abs/2411.04975)  
**Code**: [https://github.com/snowflakedb/ArcticInference](https://github.com/snowflakedb/ArcticInference)  
**Area**: LLM Agent / Inference Optimization
**Keywords**: Speculative Decoding, Suffix Tree, Agent Inference, Inference Acceleration, Training-Free

## TL;DR
Caches long token sequences via suffix trees and achieves 5.3× speedup through adaptive speculation length, targeting highly predictable repetitive inference tasks in agent scenarios.

## Background & Motivation

### State of the Field

**Background**: Speculative decoding has become a standard technique for reducing LLM inference latency, with draft model + verifier pipelines widely adopted.

**Limitations of Prior Work**:

### Root Cause

**Key Challenge**: Conventional speculative decoding is optimized for diverse requests, whereas agent workloads involve **repetitive inference** (multi-agent pipelines, self-refinement loops).

### Mechanism

**Mechanism**: Draft models must learn diverse task distributions and struggle to capture the repetitive nature of agent scenarios.

**Key Challenge**: Agent inference contains abundant long token sequences amenable to caching, which existing methods fail to exploit sufficiently.

**Key Insight**: Rather than training a draft model, use suffix trees to exactly match historical sequences and adaptively determine speculation length.

**Core Idea**: A suffix tree caches long token sequences from prompts and prior outputs, enabling training-free, extreme speculation.

## Method

### Overall Architecture
SuffixDecoding consists of two core components: (1) a **suffix tree index** that caches all prefixes from prompts and outputs, constructed in $O(n)$ and queried for the longest exact match in $O(m)$; and (2) **adaptive speculation** that dynamically adjusts speculation length based on match length and acceptance rate.

### Key Designs

1. **Suffix Tree Index**:

    - **Function**: Caches all token sequences from prompts and prior outputs.
    - **Mechanism**: Performs suffix queries on the current generation token sequence and returns subsequent tokens following the longest historical match.
    - **Design Motivation**: Approximately 70% of sequences in agent inference consist of predictable, repetitive patterns.

2. **Adaptive Speculation Length**:

    - **Function**: Dynamically adjusts the number of speculated tokens based on acceptance rate.
    - **Mechanism**: High acceptance rate > T1 → increase speculation length; low acceptance rate < T2 → decrease speculation length.
    - **Design Motivation**: Maximize speculation under high certainty; remain conservative under uncertainty.

## Key Experimental Results

### Main Results

| Method | Speedup | Notes |
|--------|---------|-------|
| EAGLE-2 | 1.9× | Model-based, requires training |
| Token Recycling | 1.9× | Training-free but limited speculation |
| Draft Model | 2.5× | Requires training |
| **SuffixDecoding** | **5.3×** | Training-free, extreme speculation |

### Workload Analysis

| Metric | SWE Workflow | Text-to-SQL | General Reasoning |
|--------|-------------|-------------|-------------------|
| Predictable sequence ratio | ~70% | ~65% | ~30% |
| Average match length | 8–12 tokens | 5–9 tokens | 1–3 tokens |
| Speculation acceptance rate | >90% | >85% | ~60% |
| Speedup | 5.3× | 4.7× | 2.1× |

### Key Findings
- Agent workloads exhibit pronounced sequence repetitiveness, clearly distinct from diverse reasoning (5.3× vs. 2.1×).
- No model training or fine-tuning required; directly applicable to any LLM.
- Memory overhead scales linearly with cache size and remains controllable.

## Highlights & Insights
- **Revival of Suffix Trees**: An innovative application of a classical data structure; training-free exact matching outperforms learned approximate speculation.
- **Advantage of Being Training-Free**: Eliminates draft model training overhead, enabling plug-and-play deployment.
- **Practical Significance of 5.3× Speedup**: Equivalent to a 5× improvement in GPU throughput.

## Limitations & Future Work
- Agent-specificity: Limited benefit for diverse reasoning workloads (2.1×).
- Cache memory pressure at large-scale deployment.
- Interaction effects with KV cache compression, quantization, and other techniques are not discussed.

## Related Work & Insights
- **vs. EAGLE-2/3**: Requires training a draft model; SuffixDecoding is training-free and faster in agent scenarios.
- **vs. Token Recycling**: Model-free but with limited speculation efficiency; SuffixDecoding's suffix tree can cache significantly longer sequences.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovative application of a classical data structure to agent inference.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on dual benchmarks: SWE-Bench and Text-to-SQL.
- Writing Quality: ⭐⭐⭐⭐ Well-motivated with a concise solution.
- Value: ⭐⭐⭐⭐⭐ Training-free, plug-and-play, 5.3× speedup.
**Code**: Unavailable

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Uncertainty Quantification in LLM Agents: Foundations, Emerging Challenges, and Opportunities](../../ACL2026/llm_agent/uncertainty_quantification_in_llm_agents_foundations_emerging_challenges_and_opp.md)
- [\[NeurIPS 2025\] What AI Speaks for Your Community: Polling AI Agents for Public Opinion on Data Center Projects](what_ai_speaks_for_your_community_polling_ai_agents_for_public_opinion_on_data_c.md)
- [\[ICLR 2026\] VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications](../../ICLR2026/llm_agent/vitabench_benchmarking_llm_agents_with_versatile_interactive_tasks_in_real-world.md)
- [\[NeurIPS 2025\] Generative AI Agents for Controllable and Protected Content Creation](generative_ai_agents_for_controllable_and_protected_content_creation.md)
- [\[ICML 2026\] MCP-Persona: Evaluating LLM Agent Capabilities in Real-World Personal Applications via Environment Simulation](../../ICML2026/llm_agent/mcp-persona_benchmarking_llm_agents_on_real-world_personal_applications_via_envi.md)

</div>

<!-- RELATED:END -->
