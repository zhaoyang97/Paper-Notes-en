---
title: >-
  [Paper Note] Automated Multi-Agent Workflows for RTL Design
description: >-
  [NeurIPS 2025 (ML for Systems Workshop)][LLM Agent][multi-agent workflow] VeriMaAS is a multi-agent framework that integrates HDL formal verification feedback (Yosys + OpenSTA) into the automated workflow generation process, adaptively selecting reasoning operators (I/O → CoT → ReAct → SelfRefine → Debate) for RTL code generation tasks. With only a few hundred training samples, it achieves 5–7% higher pass@k performance than fine-tuning baselines.
tags:
  - NeurIPS 2025 (ML for Systems Workshop)
  - LLM Agent
  - multi-agent workflow
  - RTL code generation
  - formal verification
  - Verilog
  - automated workflow orchestration
date: 2026-05-08
content_hash: 640ef3292ed02f0d
---

# Automated Multi-Agent Workflows for RTL Design

**Conference**: NeurIPS 2025 (ML for Systems Workshop)
**arXiv**: [2509.20182](https://arxiv.org/abs/2509.20182)
**Code**: Available ([GitHub](https://github.com/dstamoulis/maas/tree/verimaas/verithoughts))
**Area**: LLM Agent / EDA / Hardware Design
**Keywords**: multi-agent workflow, RTL code generation, formal verification, Verilog, automated workflow orchestration

## TL;DR

VeriMaAS is a multi-agent framework that integrates HDL formal verification feedback (Yosys + OpenSTA) into the automated workflow generation process, adaptively selecting reasoning operators (I/O → CoT → ReAct → SelfRefine → Debate) for RTL code generation tasks. With only a few hundred training samples, it achieves 5–7% higher pass@k performance than fine-tuning baselines.

## Background & Motivation

**Challenges in RTL Code Generation**: As LLMs achieve breakthroughs in code generation, RTL (Register-Transfer Level) hardware design code generation has emerged as a new frontier. Compared to general-purpose programming tasks, HDL and EDA resources are relatively scarce on the internet, introducing unique challenges:

**High Fine-tuning Cost**: Existing methods [RTLCoder, VeriThoughts] rely on expensive task-specific fine-tuning, requiring substantial GPU budgets and tens of thousands of training samples.

**High Inference Cost**: Large reasoning models (e.g., o4) eliminate the need for fine-tuning but shift the computational burden to the inference stage.

**Manual Workflow Design**: Existing multi-agent workflow methods are primarily designed for QA and mathematical tasks, leaving an applicability gap for specialized domains such as RTL design.

**Core Insight**: The HDL domain offers a unique advantage — formal verification and synthesis tools (Yosys, OpenSTA) can provide precise design quality feedback. The key idea of this paper is to directly integrate feedback from these EDA tools into the workflow generation process to dynamically guide operator selection.

## Method

### Overall Architecture

The VeriMaAS pipeline proceeds as follows:

1. Given an RTL design task, the system adaptively samples a set of reasoning operators based on the input query and task difficulty.
2. Verilog candidate designs produced at each stage are evaluated via Yosys (synthesis verification) and OpenSTA (timing/power analysis).
3. Synthesis logs and error messages are fed back to the controller, which dynamically adjusts subsequent operator selection strategies.

### Key Designs

**Solution Space Definition**:

Define the operator set $O = \{\text{Zero-shot I/O, CoT, ReAct, SelfRefine, Debate}\}$. Most existing prompting schemes can be viewed as a single operator sequence within this solution space. For example:
- Always using CoT → $O = \{O_{\text{CoT}}\}$
- Self-Refine → $O = \{O_{\text{CoT}}, O_{\text{SelfRefine}}\}$

The objective is to find the optimal operator combination $O$ for each task, maximizing pass@k over $K=20$ candidate samples.

**Cascade Controller**:

The controller $C$ is the core of VeriMaAS, employing a **cascade strategy** to select operators in order of increasing complexity:

```
I/O → CoT → ReAct → SelfRefine → Debate
```

At each stage $c$, the controller computes a confidence score $s_c$:
- Run $K=20$ Verilog candidate designs through Yosys and OpenSTA.
- $s_c$ = percentage of designs failing verification/synthesis/timing/power analysis.
- If $s_c$ exceeds the stage threshold $\tau_c$, proceed to the next stage with a more complex operator.
- Otherwise, return the current candidate solutions.

**Formal Verification Integration**:

This is the fundamental distinction between this work and general-purpose multi-agent workflow methods:
- Yosys is used for synthesis and area estimation.
- OpenSTA is used for timing and static power analysis.
- Synthesis is performed using the Skywater 130nm PDK.
- The failure rate serves as a proxy for task complexity, directly driving operator escalation decisions.

### Loss & Training

**Multi-Objective Optimization**:

$$\max_T \mathbb{E}_{(q,a)\sim\mathcal{D}} \left[ U(T; q, a, O) - \lambda \cdot C(T; q, a, O) \right]$$

where:
- $U(\cdot)$ = pass@k score (utility)
- $C(\cdot)$ = average token count per query (cost)
- $\lambda = 1\text{e-}3$

**Threshold Learning**:

500 data points are randomly sampled from the VeriThoughts training set. Based on synthesis failure statistics over $K=20$ candidate designs, the 20th/40th/60th/80th percentiles are computed to serve as the stage thresholds for the five operators: $T = \{\tau_1, \ldots, \tau_C\}$.

Core advantage: This "calibration" process requires only a few hundred data points — **an order of magnitude fewer** than the tens of thousands of samples needed for full fine-tuning.

## Key Experimental Results

### Main Results

**Table 1: VeriMaAS vs. Various Baseline Models (pass@k Comparison)**

| Model | Method | VeriThoughts pass@1 | VeriThoughts pass@10 | VerilogEval pass@1 | VerilogEval pass@10 |
|------|------|:---:|:---:|:---:|:---:|
| GPT-4o-mini | Instruct | 80.64 | 90.87 | 50.26 | 61.02 |
| GPT-4o-mini + VeriMaAS | Agent | **83.09** (+2.45) | **92.85** (+1.98) | **52.05** (+1.79) | **64.02** (+3.00) |
| o4-mini | Reasoning | 93.85 | 97.88 | 75.67 | 85.13 |
| o4-mini + VeriMaAS | Agent | **94.09** (+0.24) | **98.17** (+0.29) | **76.15** (+0.48) | 84.50 (-0.63) |
| Qwen2.5-7B | Instruct | 44.90 | 82.33 | 22.92 | 51.47 |
| RTLCoder-7B | Fine-tuned | – | – | 34.60 | 45.50 |
| Qwen2.5-7B + VeriMaAS | Agent | **56.62** (+11.72) | **86.29** (+3.96) | **29.10** (+6.18) | **56.45** (+4.98) |
| Qwen2.5-14B | Instruct | 67.89 | 94.13 | 33.78 | 62.04 |
| VeriThoughts-14B | Fine-tuned | 78.50 | 92.10 | 43.70 | 55.14 |
| Qwen2.5-14B + VeriMaAS | Agent | **74.24** (+6.35) | **95.78** (+1.65) | **41.47** (+7.69) | **62.48** (+0.44) |
| Qwen3-8B | Reasoning | 84.11 | 98.82 | 58.21 | 74.64 |
| Qwen3-8B + VeriMaAS | Agent | **88.13** (+4.02) | **99.05** (+0.23) | **59.87** (+1.66) | 74.18 (-0.46) |
| Qwen3-14B | Reasoning | 89.35 | 98.64 | 65.87 | 75.62 |
| Qwen3-14B + VeriMaAS | Agent | **92.16** (+2.81) | **98.75** (+0.11) | **66.96** (+1.09) | **75.71** (+0.09) |

Key observations:
- Gains are most pronounced on open-source LLMs: Qwen2.5-7B pass@1 improves by +11.72%, surpassing the RTLCoder-7B fine-tuning baseline.
- Gains on proprietary models are smaller but consistent (o4-mini pass@1 +0.24%), demonstrating that multi-agent orchestration retains value even at high baselines.
- Minor pass@10 regressions on VerilogEval for some models may be attributable to diversity changes introduced by operator switching.

**Table 2: VeriMaAS vs. Single-Agent Prompting Strategies (with Token Cost)**

| Model | Prompting | VT pass@1 | VT pass@10 | Tokens (k) | VE pass@1 | VE pass@10 | Tokens (k) |
|------|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| o4-mini | + CoT | 94.11 (+0.26) | 97.86 | 1.10 (1.09×) | 76.06 (+0.39) | 84.35 | 1.60 (1.06×) |
| o4-mini | + ReAct | 91.96 (-1.89) | 98.04 | 1.70 (1.68×) | 74.33 (-1.34) | 84.10 | 2.14 (1.42×) |
| o4-mini | + SelfRefine | 94.31 (+0.46) | 98.57 | 2.24 (2.22×) | 75.71 (+0.04) | 84.05 | 3.23 (2.14×) |
| o4-mini | + VeriMaAS | 94.09 (+0.24) | 98.17 | **1.21 (1.20×)** | **76.15** (+0.48) | 84.50 | **1.71 (1.13×)** |
| GPT-4o-mini | + CoT | 82.25 (+1.61) | 92.05 | 0.71 (1.42×) | 51.25 (+0.99) | 62.07 | 0.77 (1.33×) |
| GPT-4o-mini | + VeriMaAS | **83.09** (+2.45) | **92.85** | 1.26 (2.52×) | 52.05 (+1.79) | **64.02** | **0.85 (1.47×)** |

VeriMaAS incurs token costs comparable to lightweight CoT and far below SelfRefine (approximately 2× overhead), while achieving superior performance.

### Ablation Study

**Table 3: Post-Synthesis Metric Changes with PPA-Aware Optimization**

| Model | VT Pass@10 | ΔArea% | ΔPower% | ΔDelay% | VE Pass@10 | ΔArea% | ΔPower% | ΔDelay% |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4o-mini | 92.46 (-0.39) | -9.18↓ | +1.6↑ | -10.32↓ | 62.93 (-1.09) | -18.83↓ | -3.26↓ | -19.47↓ |
| o4-mini | 98.06 (-0.11) | -14.86↓ | 0.00 | -15.87↓ | 84.18 (-0.32) | -12.22↓ | +1.70↑ | -3.52↓ |
| Qwen2.5-7B | 86.33 (+0.04) | -13.44↓ | -8.67↓ | -13.91↓ | 56.45 (0.00) | **-28.79↓** | +4.07↑ | -24.58↓ |
| Qwen2.5-14B | 95.72 (-0.06) | -16.8↓ | -14.57↓ | -21.39↓ | 62.33 (-0.15) | -16.17↓ | +5.22↑ | -15.53↓ |
| Qwen3-8B | 99.04 (-0.01) | -22.81↓ | -3.68↓ | -20.14↓ | 74.06 (-0.12) | -9.98↓ | -6.04↓ | -9.03↓ |
| Qwen3-14B | 98.75 (0.00) | -9.99↓ | +2.12↑ | -9.94↓ | 75.64 (-0.07) | -11.66↓ | -7.85↓ | -11.39↓ |

Key findings:
- Area and delay are broadly and substantially reduced (up to -28.79% area, -24.58% delay).
- Power exhibits trade-offs: some models show slight power increases (e.g., Qwen2.5-14B on VerilogEval, +5.22%).
- pass@10 is nearly unaffected (maximum drop of only -1.09%), demonstrating that PPA optimization does not sacrifice functional correctness.
- This confirms that the controller **can be flexibly re-optimized** toward different design objectives, whereas fine-tuning methods embed objectives into model weights.

### Key Findings

1. **Open-source models benefit most**: VeriMaAS yields substantially larger gains on Qwen2.5-7B/14B (+6–12% pass@1) than on o4-mini (+0.24%), indicating that workflow automation can effectively compensate for limited model capacity.
2. **Cost efficiency advantage**: Only approximately 500 training data points are required for threshold calibration — an order of magnitude fewer than the tens of thousands of samples needed by fine-tuning methods such as VeriThoughts.
3. **Effectiveness of the cascade strategy**: Tasks of varying complexity are automatically matched to reasoning operators of appropriate sophistication, from I/O for simple tasks to Debate for complex ones.
4. **Flexible PPA optimization**: As a proof of concept, area/delay optimization is achieved by simply modifying the cost term in the objective function, demonstrating the extensibility of the framework.

## Highlights & Insights

- **Formal verification as a natural task difficulty signal**: This is the most elegant design choice in the paper. In general QA settings, obtaining an objective signal for "answer quality" is difficult; in RTL design, however, the Yosys compilation failure rate directly reflects task complexity, providing the controller with precise feedback.
- **Training-free workflow automation**: Unlike fine-tuning methods that require gradient updates, VeriMaAS achieves workflow optimization through statistical threshold calibration, dramatically reducing the cost of domain adaptation.
- **Bridging general to specialized domains**: This paper demonstrates how general-purpose multi-agent workflow methods (MaAS, AFlow) can be adapted to the specialized domain of hardware design, with the key being the identification of domain-specific feedback signals.

## Limitations & Future Work

1. The controller currently employs a simple cascade strategy with percentile-based thresholds; future work could explore tree search or reinforcement learning policies for finer-grained workflow decisions.
2. The current implementation relies solely on open-source Yosys + OpenSTA; extending to commercial EDA tools and industrial PDKs may unlock greater PPA optimization potential.
3. The PPA optimization benchmark subset (-PPA-Tiny) is selected via pseudo-oracle from o4, introducing evaluation bias.
4. Minor pass@10 regressions on VerilogEval under some configurations suggest that operator switching strategies may reduce candidate diversity.
5. Only five fixed operators are evaluated; the possibility of operator composition or custom operator design remains unexplored.

## Related Work & Insights

- **MaAS** [Zhang et al., 2025]: The cascade controller in this paper is directly built upon the Multi-Agent as a Service framework of MaAS.
- **AFlow** [Zhang et al., 2025]: An alternative automated workflow generation method focused on general-purpose QA.
- **VeriThoughts** [Yubeaton et al., 2025]: An RTL benchmark and fine-tuning method serving as the primary comparison baseline.
- **RTLCoder** [Liu et al., 2024]: A lightweight fine-tuning approach for RTL; VeriMaAS surpasses its performance at comparable model sizes.

This paper motivates a broader direction: in any specialized domain with formal verification tools (e.g., theorem proving, circuit design, compiler optimization), verification feedback can be integrated into multi-agent workflows to enable automated reasoning strategy selection.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Incorporating formal verification feedback into automated multi-agent workflow generation is a meaningful contribution.
- **Technical Depth**: ⭐⭐⭐ — The method is relatively straightforward (cascade controller + percentile thresholds), though well-motivated.
- **Experimental Quality**: ⭐⭐⭐⭐ — Covers 6 models × 2 benchmarks, including cost analysis and PPA ablation.
- **Practicality**: ⭐⭐⭐⭐⭐ — Low training cost, plug-and-play design, and compatibility with multiple LLMs make it highly deployable.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, though the workshop paper format limits the depth of some details.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies](../../ICLR2026/llm_agent/multi-agent_design_optimizing_agents_with_better_prompts_and_topologies.md)
- [\[NeurIPS 2025\] Benchmarking Agentic Systems in Automated Scientific Information Extraction with ChemX](benchmarking_agentic_systems_in_automated_scientific_information_extraction_with.md)
- [\[NeurIPS 2025\] Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection](automated_composition_of_agents_a_knapsack_approach_for_agentic_component_select.md)
- [\[NeurIPS 2025\] MAT-Agent: Adaptive Multi-Agent Training Optimization](mat-agent_adaptive_multi-agent_training_optimization.md)
- [\[NeurIPS 2025\] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)

</div>

<!-- RELATED:END -->
