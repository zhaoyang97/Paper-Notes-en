---
title: >-
  [Paper Note] Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments
description: >-
  [ICLR 2026 (Oral)][LLM Agent][Dynamic Environment] Ours proposes the Gaia2 benchmark to evaluate LLM Agent capabilities in dynamic and asynchronous environments. It introduces real-world scenarios such as time constraints, noisy events, ambiguity resolution, and multi-agent collaboration. Combined with write action verifiers providing verifiable rewards, the benchmark can be directly used for RLVR training. Evaluations show that even the strongest model, GPT-5 (high)…
tags:
  - "ICLR 2026 (Oral)"
  - "LLM Agent"
  - "Dynamic Environment"
  - "Asynchronous Interaction"
  - "benchmark"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: 6f1df5b532ce668e
---

# Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments

**Conference**: ICLR 2026 (Oral)  
**arXiv**: [2602.11964](https://arxiv.org/abs/2602.11964)  
**Code**: Based on Agents Research Environments (ARE) platform, open source  
**Area**: LLM Agent Evaluation  
**Keywords**: LLM Agent, Dynamic Environment, Asynchronous Interaction, benchmark, Reinforcement Learning

## TL;DR

Ours proposes the Gaia2 benchmark to evaluate LLM Agent capabilities in dynamic and asynchronous environments. It introduces real-world scenarios such as time constraints, noisy events, ambiguity resolution, and multi-agent collaboration. Combined with write action verifiers providing verifiable rewards, the benchmark can be directly used for RLVR training. Evaluations show that even the strongest model, GPT-5 (high), achieves only a 42% pass@1 rate.

## Background & Motivation

Current evaluations of LLM Agents suffer from fundamental flaws: most benchmarks rely on **static** or **synchronous** environments. In these settings, the environment does not change independently of the Agent's actions—the Agent has complete temporal control, can pause or think indefinitely, and the environment state always waits for the Agent's next move.

However, real-world task environments are entirely different:
- **Time Sensitivity**: Fluctuating flight prices, inventory changes, and approaching deadlines.
- **Asynchronous Events**: New incoming messages and independent state updates.
- **Noise & Ambiguity**: Incomplete information, contradictory contexts, and requirements needing clarification.
- **Multi-party Collaboration**: Necessitating coordination with other Agents or humans.

Existing benchmarks (such as the original GAIA) only test static Q&A and tool calling, failing to evaluate Agent capabilities across these **realistic dimensions**. This leads to a significant "**sim2real gap**"—high scores on benchmarks do not predict performance in real-world deployment.

The Goal of Gaia2 is to create an evaluation platform closer to reality while maintaining quantifiability and reproducibility.

## Method

### Overall Architecture

Gaia2 builds evaluation scenarios upon the open-source Agents Research Environments (ARE) platform. Each scenario consists of a dynamic environment that evolves independently of Agent operations, a task description, and a set of fine-grained write action verifiers. In this environment that "moves forward on its own," Agents are required to perceive changes while making and executing decisions within time windows. Verifiers judge correctness at each critical action point, pushing evaluation from "is the final answer correct" to "is every action step correct."

### Key Designs

**1. Dynamic Asynchronous Environments: Breaking Temporal Control**

Traditional Agent benchmarks assume the environment waits for the Agent. Gaia2 does the opposite: prices fluctuate, inventory changes, and messages arrive asynchronously. Once a window of opportunity is missed, it disappears. Agents must make decisions within limited time windows, continuously monitor environment states, and react to unexpected events and state transitions. This shifts the focus from "planning a static optimal path" to "continuous adaptation under uncertain and changing conditions," which is where gaps are most likely exposed in real deployment.

**2. Multi-dimensional Capability Coverage: Deconstructing "Reality" into Measurable Axes**

Gaia2 deliberately designs scenarios to cover five core dimensions: time-sensitive decision-making (selecting optimal actions under time limits), noise robustness (extracting key facts from incomplete or contradictory information), ambiguity resolution (proactive clarification or selecting the most reasonable interpretation), multi-agent collaboration (exchanging information and coordinating actions), and environmental adaptation (responding to dynamic changes and revising plans). This allows evaluation to provide capability profiles decomposed by dimension, identifying whether a model is weak in "response speed" or "information noise resistance."

**3. Write Action Verifiers: Making Rewards Evaluable and Trainable**

If only the final answer is checked, the quality of intermediate decisions is lost. The technical Novelty of Gaia2 is the pre-definition of "write action" checkpoints in each scenario. Verifiers judge whether Agent operations at these critical points are correct, refining evaluation granularity to the quality of every decision step. Crucially, these step-by-step verifiable reward signals are naturally suited for Reinforcement Learning from Verifiable Rewards (RLVR), allowing the benchmark to serve both for scoring and as training signals to drive Agent self-improvement.

**4. ARE-based Scalable Architecture: Decoupling Environment and Verification Logic**

To ensure sustainability, the system is built on the open-source Agents Research Environments (ARE) framework, separating environment evolution logic from verification logic. New scenarios can be integrated via standard interfaces and are compatible with various Agent frameworks. Scenarios are drawn from consumer environments like shopping and travel planning, ensuring realistic tasks while providing a scalable research infrastructure. Evaluation primarily uses pass@1 as the main metric, supplemented by per-dimension profiles and "speed vs. API cost" trade-offs.

## Key Experimental Results

### Main Results: Overall Model Performance

| Model | pass@1 | Type | Key Characteristics |
|------|--------|------|---------|
| GPT-5 (high) | 42% | Closed-source | Strongest overall but weak in time-sensitive tasks |
| Claude-4 Sonnet | ~35-38% | Closed-source | Balance of accuracy and speed, better cost efficiency |
| Kimi-K2 | 21% | Open-source | Best among open-source models |
| Other Open-source | <20% | Open-source | Significantly lagging behind closed-source |

### Ability Dimension Analysis

| Capability Dimension | GPT-5 | Claude-4 | Kimi-K2 | Notes |
|---------|-------|----------|---------|------|
| Time-sensitive Decision | Weak | Moderate | Weak | Most challenging dimension |
| Noise Robustness | Strong | Strong | Moderate | Clear advantage for closed-source |
| Ambiguity Resolution | Strong | Moderate | Weak | Requires strong reasoning |
| Multi-Agent Collaboration | Moderate | Moderate | Weak | Weak link for all models |
| Environmental Adaptation | Moderate | Moderate | Weak | Ability to dynamically adjust plans |

### Ablation Study

| Comparison Dimension | Key Findings |
|---------|---------|
| Static vs. Dynamic Environment | Performance of all models drops significantly in dynamic environments |
| Synchronous vs. Asynchronous | Asynchronous events further widen the gap between models |
| Single-Agent vs. Multi-Agent | Multi-agent scenarios are currently the biggest bottleneck |
| No Time Limit vs. Time Limit | Time constraints have a greater impact on open-source models |

### Key Findings

1. **No model dominates all dimensions**: GPT-5 is the strongest overall but fails in time-sensitive tasks; Claude-4 performs better in cost efficiency.
2. **42% pass@1 exposes a huge gap**: Even the strongest models fail in nearly 60% of scenarios, indicating that real-world Agent tasks remain extremely challenging.
3. **The open-source vs. closed-source divide**: The 21% vs 42% gap shows that open-source models still lack sufficient capability in Agent scenarios.
4. **The "sim2real gap" is real**: Models performing similarly on static benchmarks show amplified differences in the dynamic environments of Gaia2.
5. **Potential for RLVR**: Fine-grained reward signals provided by write action verifiers pave the way for Agent training based on reinforcement learning.

## Highlights & Insights

- **Paradigm shift from "Q&A" to "Action"**: Gaia2 evaluates the ability to take correct actions in dynamic environments rather than just knowledge or reasoning.
- **Write action verifiers are a key innovation**: These allow the benchmark to serve both evaluation and training, greatly enhancing its utility.
- **Asynchrony is a neglected core challenge**: Most existing Agent systems assume synchronous interaction; Gaia2 is the first to systematically test asynchronous scenarios.
- **ICLR 2026 Oral status**: The selection for an oral presentation reflects the community's urgent need for realistic Agent evaluation.
- **Ecological value of the open-source ARE platform**: It serves not just as a benchmark, but as a sustainable research infrastructure.

## Limitations & Future Work

1. **Consumer environments may not represent all domains**: Shopping and travel scenarios differ from Agent requirements in scientific research or software development.
2. **Reproducibility challenges**: Stochasticity in dynamic environments may lead to fluctuations in results between runs.
3. **Manual design for write action verifiers**: Verifiers for each scenario require manual definition of checkpoints and correctness standards, limiting automated scaling.
4. **Insufficient tool-use testing**: While the environment is dynamic, the complexity of toolsets and API interfaces could be higher.
5. **Limited scale of multi-agent scenarios**: Currently focuses on dual-agent scenarios; larger-scale collaboration testing is yet to be developed.

## Related Work & Insights

- **Succession from GAIA (2023)**: Gaia2 introduces dynamic and asynchronous dimensions as qualitative improvements over its predecessor.
- **Distinction from WebArena and AgentBench**: These focus on static web interaction or API calls, whereas Gaia2 emphasizes the temporal evolution of the environment.
- **Complementary to SWE-bench**: The latter tests code generation, while Gaia2 tests environmental interaction and decision-making.
- **Impact on Agent training**: The RLVR-ready design makes Gaia2 a potential source of key data for training stronger Agents.
- **Insights for Agent architecture**: Designs must consider time-awareness, asynchronous event handling modules, and dynamic plan adjustment mechanisms.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Dynamic asynchronous evaluation + RLVR-ready design, leading the field)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers mainstream models, though scenario quantity is unspecified)
- Writing Quality: ⭐⭐⭐⭐ (Well-structured and clearly analyzed)
- Value: ⭐⭐⭐⭐⭐ (A major milestone in Agent evaluation, Oral acceptance is well-deserved)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Sketchtopia: A Dataset and Foundational Agents for Benchmarking Asynchronous Multimodal Communication with Iconic Feedback](../../CVPR2025/llm_agent/sketchtopia_a_dataset_and_foundational_agents_for_benchmarking_asynchronous_mult.md)
- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [\[ICLR 2026\] Real-Time Reasoning Agents in Evolving Environments](real-time_reasoning_agents_in_evolving_environments.md)
- [\[AAAI 2026\] LLMTM: Benchmarking and Optimizing LLMs for Temporal Motif Analysis in Dynamic Graphs](../../AAAI2026/llm_agent/llmtm_benchmarking_and_optimizing_llms_for_temporal_motif_analysis_in_dynamic_gr.md)
- [\[ICLR 2026\] Benchmarking LLM Tool-Use in the Wild](benchmarking_llm_tool-use_in_the_wild.md)

</div>

<!-- RELATED:END -->
