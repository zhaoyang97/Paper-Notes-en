---
title: >-
  [Paper Note] Multi-Agent Reasoning Improves Compute Efficiency: Pareto-Optimal Test-Time Scaling
description: >-
  [ACL 2026][Multi-Agent][Test-time computation] This paper compares self-consistency, self-refinement, multi-agent debate, and Mixture-of-Agents (MoA) under a unified compute budget. It finds that multi-agent reasoning…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Test-time computation"
  - "Multi-agent reasoning"
  - "Pareto front"
  - "Mixture-of-Agents"
  - "Compute efficiency"
date: 2026-05-08
content_hash: 16e49263b19617e2
---

# Multi-Agent Reasoning Improves Compute Efficiency: Pareto-Optimal Test-Time Scaling

**Conference**: ACL 2026  
**arXiv**: [2605.01566](https://arxiv.org/abs/2605.01566)  
**Code**: https://github.com/Multi-Agent-LLMs/lm-evaluation-harness  
**Area**: LLM Reasoning  
**Keywords**: Test-time computation, Multi-agent reasoning, Pareto front, Mixture-of-Agents, Compute efficiency

## TL;DR

This paper compares self-consistency, self-refinement, multi-agent debate, and Mixture-of-Agents (MoA) under a unified compute budget. It finds that multi-agent reasoning, specifically MoA, is more efficient on the Pareto front, improving MMLU-Pro accuracy from 64.3% to 71.4% at approximately 20x CoT budget.

## Background & Motivation

**Background**: Improving LLM reasoning capabilities no longer relies solely on training larger models; test-time computation has become a critical approach. Common methods include chain-of-thought (CoT), self-consistency (SC), multi-round self-refinement, multi-agent debate, and Mixture-of-Agents (MoA), which aggregates multiple candidate answers layer-by-layer. These methods share the common goal of spending more computation during inference to obtain more stable or powerful answers.

**Limitations of Prior Work**: Many studies only report final accuracy without comparing different methods under the same compute budget. A method calling a model dozens of times naturally outperforms a single CoT, but this does not equate to higher efficiency. Real-world deployment prioritizes which pipeline delivers higher accuracy given the same latency, compute power, and budget.

**Key Challenge**: There is a clear accuracy-cost trade-off in test-time computation. Parallel sampling increases candidate paths, while sequential refinement or debate rounds deepen reasoning, both increasing FLOPs, memory I/O, and inference latency. The core problem is not whether spending more compute is useful, but whether it should be spent on more samples, more agents, more rounds of interaction, or larger models.

**Goal**: To systematically evaluate four types of reasoning scaling strategies under the same computational mouthfeel and answer three practical questions: whether multi-agent is truly more compute-efficient than single-agent; how to balance parallel scale and sequential depth in multi-agent systems; and whether intensive test-time scaling for small models is more cost-effective than few calls to large models.

**Key Insight**: The authors look beyond just FLOPs, using an estimated runtime that accounts for both arithmetic computation and model weight memory transfer as the cost. The Pareto-front is used to identify the configuration with the highest accuracy for a given cost, avoiding biases inherent in simply comparing generation counts or FLOPs.

**Core Idea**: Treat the test-time reasoning pipeline as a tunable compute allocation problem, compare methods via the Pareto-optimal front, and summarize practical scaling rules for multi-agent reasoning from the optimal configurations.

## Method

### Overall Architecture

The experimental framework consists of a three-tier sweep. The first tier is pipeline selection: comparing self-consistency, self-refinement, debate, and MoA. The second tier is pipeline parameters: varying sample counts for SC, iteration rounds for self-refinement, agent counts/rounds for debate, and proposer counts/aggregation layers for MoA. The third tier is model size: using Llama 3.1 70B and 8B to distinguish the effects of "model capacity" versus "test-time scaling."

All methods solve multiple-choice reasoning tasks in a zero-shot CoT style. Models are prompted as reasoning experts to think step-by-step and output the final option. After each pipeline, candidates are extracted using a "Final answer of choices {choices}:" prompt, and the option with the highest log-likelihood is selected.

Two metrics are used: accuracy and compute cost. Cost is estimated as theoretical runtime rather than call counts. Generation is split into prefill and decode phases, estimating FLOP time and memory transfer time for each; the slower of the two is taken for each phase and summed. This accounts for memory bandwidth bottlenecks common in GPU inference, which FLOPs alone would underestimate for small models.

The final analysis identifies the Pareto-front across all configurations: a configuration is Pareto-optimal if no other configuration achieves higher or equal accuracy at a lower or equal cost.

### Key Designs

1.  **Unified Test-Time Computation Scaling Coordinate System**:
    *   **Function**: Compares four structurally diverse reasoning methods on a single cost-accuracy plot.
    *   **Mechanism**: Breaks down scaling into parallel and sequential dimensions. SC expands parallel CoT samples; self-refinement expands sequential correction steps; debate includes both agents and rounds; MoA includes both proposers and layers.
    *   **Design Motivation**: Prevents confusing "better method" with "more computation." A unified system forces multi-agent methods like debate and MoA to compete with SC under identical budgets, reflecting deployment scenarios.

2.  **Deployment-Oriented Compute Cost Estimation**:
    *   **Function**: Measures test-time budgets in a way that aligns with actual inference latency better than FLOPs.
    *   **Mechanism**: Estimates two types of time for prefill and decode: compute time determined by FLOPs (form of $2 \cdot P \cdot T$) and memory transfer time determined by parameter count, quantization precision, batch size, GPU count, and bandwidth. The total is $\sum \max(\text{FLOP time}, \text{memory time})$.
    *   **Design Motivation**: Small model calls might seem cheap in FLOPs but involve repeated weight loading. Including memory transfer provides a fairer trade-off analysis between "many small model rounds" and "few large model rounds."

3.  **Refining Multi-Agent Design Rules via Pareto-Front**:
    *   **Function**: Identifies truly efficient parameter combinations from a vast space.
    *   **Mechanism**: Calculations of the Pareto-front across 34 configurations and 100+ evaluations reveal patterns. In debate, optimal points come from increasing agents rather than rounds. In MoA, optimal points often appear when proposer count = layer count + 1 (e.g., 3 models/2 layers, 4 models/3 layers).
    *   **Design Motivation**: Blindly adding agents or rounds wastes budget or degrades performance. The Pareto-front extracts "directions worth scaling" into actionable configuration advice.

### Loss & Training

No new models were trained; this is a pure test-time reasoning strategy evaluation. The main experiments use 4-bit quantized Llama 3.1 70B-Instruct, with supplementary experiments on the 8B version. Generation uses temperature 0.7 and top-p 0.95. For cost control, 1000 samples were used from MMLU-Pro and BBH.

## Key Experimental Results

### Main Results

Primary results are focused on MMLU-Pro. CoT (SC with 1 sample) achieved 64.3%. Within a ~20x CoT budget, MoA's Pareto-front was strongest, followed by debate. SC saturated earlier, and self-refinement performed worse than CoT.

| Method / Configuration | MMLU-Pro Accuracy | Gain vs. CoT | Comparison vs. SC (same budget) | Key Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| CoT / SC 1 sequence | 64.3% | - | - | Basic single inference baseline |
| SC / 10 sequences | 68.7% | +4.4 pp | 0 | Parallel sampling is effective but saturates early |
| Debate / 4 agents, 2 rounds | 70.0% | +5.7 pp | +1.3 pp | Multi-agent interaction is more efficient than pure sampling |
| MoA / 5 models, 4 layers | 71.4% | +7.1 pp | +2.7 pp | Highest accuracy; dominates Pareto-front |
| Self-refinement / Multi-round | <64.3% | Negative | Lower than others | Sequential self-correction yields no reliable gain |

Difficulty analysis shows extra test-time compute is more valuable for hard problems.

| Compute Budget | Easy Accuracy | Medium Accuracy | Hard Accuracy | Observation |
| :--- | :--- | :--- | :--- | :--- |
| CoT | 94.4% | 53.0% | 8.4% | CoT solves most easy problems |
| 1-5× CoT | 95.6% | 58.6% | 13.6% | Significant gains for medium/hard |
| 5-10× CoT | 95.4% | 60.4% | 14.7% | Budget mostly helps non-easy tasks |
| 10-15× CoT | 96.0% | 62.1% | 14.2% | Hard fluctuates but stays above CoT |
| 15-20× CoT | 96.6% | 61.5% | 17.4% | Hard tasks see largest relative gain |
| Total Gain | +2.2 pp | +8.5 pp | +9.0 pp | Budget should be allocated adaptively |

### Ablation Study

The study analyzes scaling directions for multi-agent systems via parameter sweeps. Key conclusion: debate should scale agents; MoA optimal points satisfy proposers = layers + 1.

| System | Parameter Change | Recommended Trend | Explanation |
| :--- | :--- | :--- | :--- |
| Debate | Increase agents | Accuracy improves up to ~4 agents | Parallel perspectives increase diversity; too many add noise |
| Debate | Increase rounds | 2 rounds usually best | More rounds increase context cost and risk error propagation |
| MoA | Increase proposers | Best when models = layers + 1 | Sufficient candidates allow better evidence synthesis |
| MoA | Increase layers | Beneficial within ratio | Sequential aggregation has fewer side effects than debate memory |
| Model Size | 8B scaling vs 70B CoT | 70B CoT remains stronger | Capacity gap cannot be closed by low-quality small model scaling |

Allocation of model sizes in MoA (5 models/4 layers): With 70B proposers, an 8B aggregator still reaches 69.6%. However, 8B proposers with a 70B aggregator only reach 52.9%.

| MoA Config (5m, 4l) | Aggregator 8B | Aggregator 70B | Key Insight |
| :--- | :--- | :--- | :--- |
| Proposers 8B | 51.2% | 52.9% | Weak proposers produce poor evidence; strong aggregator cannot fix it |
| Proposers 70B | 69.6% | 71.4% | Quality stems from proposers; smaller aggregator causes minor drop |

### Key Findings

*   MoA is the most robust Pareto-optimal method.
*   Debate gains come from parallel agents, not more rounds.
*   Self-refinement performs poorly without external feedback.
*   Test-time compute is most valuable for hard/medium problems.
*   Small model scaling does not defeat large model CoT at the same budget.
*   In MoA, proposers are more critical than the aggregator.

## Highlights & Insights

*   **Accuracy vs. Efficiency**: The paper shifts focus from pure accuracy to the Pareto-front, which is essential for real-world deployment where budgets are finite.
*   **MoA Heuristic**: The "proposers = layers + 1" rule is a practical mnemonic for configuration.
*   **Memory Cost Awareness**: By accounting for memory bandwidth, the paper provides a more realistic assessment of the trade-off between many small calls vs. few large calls.
*   **Task Routing**: The finding that easy tasks derive little benefit suggests future "easy-CoT, hard-MoA" adaptive systems.
*   **Sequential Limits**: The failure of self-refinement highlights that sequential reasoning without external signals often just adds redundancy.

## Limitations & Future Work

*   **Sample Size**: Evaluations used 1000 samples due to cost; confidence intervals are around 0.03.
*   **Theoretical Cost**: The model excludes framework overhead, batching, and KV cache management.
*   **Model Scope**: Primarily focused on Llama 3.1 4-bit; applicability to closed-source or MoE models is unverified.
*   **Task Type**: Limited to multiple-choice reasoning.
*   **Homogeneous MoA**: Most tests used identical models for proposer/aggregator.

## Related Work & Insights

*   **vs. Self-Consistency**: SC is effective but saturates; MoA/debate are more efficient because information interaction beats simple majority voting.
*   **vs. Self-Refine**: Refine is often worse than CoT here, echoing findings that LLMs cannot reliably self-correct without external verification.
*   **vs. Multi-Agent Debate**: Debate is more efficient than SC, but rounds should be limited to avoid context bloat.
*   **vs. Mixture-of-Agents**: MoA dominates the Pareto-front; it avoids the heavy memory accumulation of debate.
*   **vs. Inference Scaling Laws**: Complements existing work by showing MoA's structure is often more budget-efficient than simple best-of-n.

## Rating

*   Novelty: ⭐⭐⭐⭐ 
*   Experimental Thoroughness: ⭐⭐⭐⭐ 
*   Writing Quality: ⭐⭐⭐⭐⭐ 
*   Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration](scaling_external_knowledge_input_beyond_context_windows_of_llms_via_multi-agent_.md)
- [\[ACL 2026\] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)
- [\[ACL 2026\] Debating the Unspoken: Role-Anchored Multi-Agent Reasoning for Half-Truth Detection](debating_the_unspoken_role-anchored_multi-agent_reasoning_for_half-truth_detecti.md)
- [\[ACL 2026\] When Identity Skews Debate: Anonymization for Bias-Reduced Multi-Agent Reasoning](when_identity_skews_debate_anonymization_for_bias-reduced_multi-agent_reasoning.md)
- [\[ACL 2026\] Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games](collaborative_multi-agent_scripts_generation_for_enhancing_imperfect-information.md)

</div>

<!-- RELATED:END -->
