---
title: >-
  [Paper Note] VeriMaAS: Automated Multi-Agent Workflows for RTL Design
description: >-
  [NeurIPS 2025][LLM Agent][RTL code generation] VeriMaAS proposes a framework for automatically composing multi-agent workflows for RTL code generation. Its core innovation is the direct integration of formal verification…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "RTL code generation"
  - "multi-agent workflow"
  - "formal verification"
  - "HDL"
  - "EDA"
date: 2026-05-08
content_hash: 7a0a6a4e9189d789
---

# VeriMaAS: Automated Multi-Agent Workflows for RTL Design

**Conference**: NeurIPS 2025
**arXiv**: [2509.20182](https://arxiv.org/abs/2509.20182)  
**Code**: [https://github.com/](https://github.com/) (available, mentioned in the paper)  
**Area**: Code Generation / Hardware Design Automation
**Keywords**: RTL code generation, multi-agent workflow, formal verification, HDL, EDA

## TL;DR

VeriMaAS proposes a framework for automatically composing multi-agent workflows for RTL code generation. Its core innovation is the direct integration of formal verification feedback from HDL tools (Yosys synthesis + OpenSTA timing analysis) into workflow orchestration, achieving a 2–12% pass@1 improvement on VeriThoughts while requiring only a few hundred samples for controller tuning—an order of magnitude fewer than full fine-tuning.

## Background & Motivation

**The RTL Code Generation Dilemma.** Existing approaches follow two main directions: (1) fine-tuning LLMs on RTL/HDL data, which demands substantial GPU resources and tens of thousands of training samples with poor generalization; (2) using frontier reasoning models (e.g., o4), which eliminates the need for fine-tuning but incurs prohibitively high inference costs. Both directions are fundamentally constrained by the scarcity of HDL-domain data relative to general-purpose programming languages.

**The Multi-Agent Workflow Paradigm.** Recent works such as MaAS and AFlow have demonstrated the advantages of automated multi-agent workflows on QA and general programming tasks. However, these methods primarily target "general knowledge" domains (e.g., Wikipedia-based QA, math competitions), and naive prompting strategies (e.g., Debate) do not transfer directly to specialized domains such as RTL design.

**Core Insight and Starting Point.** A distinctive advantage of RTL design is the availability of mature formal verification toolchains—Yosys synthesis and OpenSTA timing analysis provide precise pass/fail judgments. The core idea of VeriMaAS is to embed this verification feedback directly into the workflow orchestration process, allowing the agent controller to dynamically adjust reasoning strategies based on compilation/synthesis results, rather than relying on the LLM's own assessment of code quality.

## Method

### Overall Architecture

The VeriMaAS workflow proceeds as follows: given an RTL design task → adaptively sample agent operators to generate $K=20$ candidate Verilog designs → validate via Yosys synthesis + OpenSTA timing/power analysis → feed verification logs and error messages back to the controller → the controller decides whether to escalate to a more complex reasoning operator or terminate → return the candidate design pool.

### Key Designs

1. **Cascading Controller**:

    - *Function*: Adaptively selects reasoning operator complexity according to task difficulty.
    - *Mechanism*: Defines an operator complexity sequence I/O → CoT → ReAct → SelfRefine → Debate. At each stage, a confidence score $s_c$ is computed based on the validation failure rate of current candidate designs; if the failure rate exceeds a threshold $\tau_c$, the system escalates to the next operator.
    - *Design Motivation*: Different RTL tasks require different levels of reasoning—simple modules can be handled with zero-shot inference, while complex modules warrant multi-round reflection or debate. The cascading design avoids applying the most expensive strategy uniformly across all tasks.

2. **Formal Verification Feedback Loop**:

    - *Function*: Provides agents with precise correctness signals for designs.
    - *Mechanism*: The $K=20$ Verilog candidates at each stage are evaluated through a full EDA toolchain—Yosys for synthesis and area estimation, OpenSTA for timing and static power analysis (using the Skywater 130nm PDK). Verification logs and error messages are fed directly to the controller and subsequent reasoning steps.
    - *Design Motivation*: Conventional LLM-based code generation relies on the model's own judgment of code quality, whereas HDL verification requires precise tool feedback. The binary signals (pass/fail) from EDA tools are more reliable than the LLM's ambiguous self-evaluation.

3. **Low-Cost Controller Tuning**:

    - *Function*: Achieves controller policy learning with minimal data requirements.
    - *Mechanism*: 500 data points are randomly sampled from the VeriThoughts training set; $K=20$ candidates are generated for each point and validation failure rates are recorded. The 20th/40th/60th/80th percentiles are used as stage thresholds $\mathcal{T}$. The objective is $\max_{\mathcal{T}} \mathbb{E}[U(\mathcal{T}) - \lambda \cdot C(\mathcal{T})]$, where $U$ denotes pass@k and $C$ denotes token consumption.
    - *Design Motivation*: Compared to full fine-tuning requiring tens of thousands of samples, only a few hundred suffice to determine the thresholds, since the controller only needs to learn *when to escalate* rather than *how to write RTL code*.

### Loss & Training

The controller tuning objective is a Pareto optimization over performance and cost:

$$\max_{\mathcal{T}} \mathbb{E}_{(q,a)\sim D}[U(\mathcal{T};q,a,\mathbb{O}) - \lambda \cdot C(\mathcal{T};q,a,\mathbb{O})]$$

with $\lambda=10^{-3}$. In the PPA-aware optimization variant, the cost term is replaced by Yosys-reported area $C=\text{Area}(\mathcal{T};q,a,\mathbb{O})$, enabling joint optimization of functional correctness and physical design metrics.

## Key Experimental Results

### Main Results

| Model + Method | VeriThoughts pass@1 | VeriThoughts pass@10 | VerilogEval pass@1 | VerilogEval pass@10 |
|---|---|---|---|---|
| GPT-4o-mini (Instruct) | 80.64 | 90.87 | 50.26 | 61.02 |
| GPT-4o-mini + VeriMaAS | **83.09** (+2.45) | **92.85** (+1.98) | **52.05** (+1.79) | **64.02** (+3.00) |
| Qwen2.5-7B (Instruct) | 44.90 | 82.33 | 22.92 | 51.47 |
| RTLCoder-7B (fine-tuned) | - | - | 34.60 | 45.50 |
| Qwen2.5-7B + VeriMaAS | **56.62** (+11.72) | **86.29** (+3.96) | **29.10** (+6.18) | **56.45** (+4.98) |
| Qwen2.5-14B (Instruct) | 67.89 | 94.13 | 33.78 | 62.04 |
| VeriThoughts-14B (fine-tuned) | 78.50 | 92.10 | 43.70 | 55.14 |
| Qwen2.5-14B + VeriMaAS | **74.24** (+6.35) | **95.78** (+1.65) | **41.47** (+7.69) | **62.48** (+0.44) |
| Qwen3-14B (Reasoning) | 89.35 | 98.64 | 65.87 | 75.62 |
| Qwen3-14B + VeriMaAS | **92.16** (+2.81) | **98.75** (+0.11) | **66.96** (+1.09) | **75.71** (+0.09) |

### Ablation Study

| Configuration (o4-mini baseline) | VeriThoughts pass@1 | Token Cost (k) | Note |
|---|---|---|---|
| + CoT | 94.11 (+0.26) | 1.10 (1.09×) | Lightweight improvement |
| + ReAct | 91.96 (−1.89) | 1.70 (1.68×) | Performance degrades |
| + SelfRefine | 94.31 (+0.46) | 2.24 (2.22×) | Strong but costly |
| + VeriMaAS | 94.09 (+0.24) | 1.21 (1.20×) | Near-optimal performance at near-minimal cost |

### Key Findings
- The largest gains are observed on open-source LLMs: Qwen2.5-7B pass@1 improves from 44.90 to 56.62 (+11.72), surpassing the fine-tuned RTLCoder-7B on pass@10.
- VeriMaAS matches or exceeds fine-tuned baselines on pass@10 while also improving pass@1, indicating that the framework both improves best-candidate quality and expands the effective candidate pool.
- Compared to single-agent strategies, VeriMaAS maintains competitive performance at near-minimal token cost (only 1.20× vs. SelfRefine's 2.22×).
- PPA-aware optimization reduces area by 9–23% and latency by 4–21% with negligible pass@10 degradation.

## Highlights & Insights
- The use of formal verification as an agent feedback source has broad applicability across the hardware design AI field—EDA tools provide precise verification signals unavailable in general software programming.
- Controller tuning requires only a few hundred samples (vs. tens of thousands for fine-tuning), substantially lowering the deployment barrier.
- The cascading controller achieves a favorable performance–cost trade-off, avoiding the wasteful application of the strongest strategy to all tasks.
- PPA-aware optimization demonstrates the flexibility of the framework—optimization objectives can be switched without retraining.

## Limitations & Future Work
- This is a NeurIPS ML for Systems Workshop paper; the experimental scale is relatively limited.
- Controller thresholds are determined via simple percentile-based statistics; more sophisticated learning methods (e.g., RL) may yield further improvements.
- The workflow search space (a cascading sequence of 5 fixed operators) is relatively constrained; free composition may discover superior strategies.
- Validation is limited to RTL code generation; extension to other EDA tasks (placement and routing, timing repair) remains to be explored.

## Related Work & Insights
- **Comparison with MaAS/AFlow**: While all three fall under automated multi-agent workflow frameworks, VeriMaAS introduces domain-specific formal verification feedback, enabling the general framework to adapt to specialized hardware design tasks.
- **Comparison with fine-tuning approaches (RTLCoder/VeriThoughts)**: VeriMaAS is orthogonal to fine-tuning—it can be applied on top of any base model, including fine-tuned ones, and the two approaches are complementary.
- **Broader Insight**: The formal verification feedback paradigm can be generalized to any code generation domain with precise checking tools (e.g., SQL verification, mathematical proof checking).

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of formal verification and automated agent workflows is pioneering in the RTL domain.
- Experimental Thoroughness: ⭐⭐⭐ Workshop-scale paper, but covers multiple models and two benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear; experimental design is systematic.
- Value: ⭐⭐⭐⭐ Practically relevant for hardware design automation; low supervision cost facilitates deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MAT-Agent: Adaptive Multi-Agent Training Optimization](mat-agent_adaptive_multi-agent_training_optimization.md)
- [\[NeurIPS 2025\] Benchmarking Agentic Systems in Automated Scientific Information Extraction with ChemX](benchmarking_agentic_systems_in_automated_scientific_information_extraction_with.md)
- [\[NeurIPS 2025\] Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection](automated_composition_of_agents_a_knapsack_approach_for_agentic_component_select.md)
- [\[NeurIPS 2025\] R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](rd-agent-quant_a_multi-agent_framework_for_data-centric_factors_and_model_joint_.md)
- [\[ICML 2026\] A Minimal Agent for Automated Theorem Proving](../../ICML2026/llm_agent/a_minimal_agent_for_automated_theorem_proving.md)

</div>

<!-- RELATED:END -->
