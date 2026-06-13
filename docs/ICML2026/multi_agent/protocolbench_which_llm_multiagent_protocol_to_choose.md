---
title: >-
  [Paper Note] ProtocolBench: Which LLM MultiAgent Protocol to Choose?
description: >-
  [ICML 2026][Multi-Agent][Multi-agent protocols] ProtocolBench provides the first systematic comparison of four major LLM multi-agent communication protocols (A2A, ACP, ANP, Agora) across four axes: task success…
tags:
  - "ICML 2026"
  - "Multi-Agent"
  - "Multi-agent protocols"
  - "A2A / ACP / ANP / Agora"
  - "Protocol routing"
  - "Failure recovery"
  - "End-to-end latency"
date: 2026-05-08
content_hash: 4b12cfce90cd84c9
---

# ProtocolBench: Which LLM MultiAgent Protocol to Choose?

**Conference**: ICML 2026  
**arXiv**: [2510.17149](https://arxiv.org/abs/2510.17149)  
**Code**: TBD  
**Area**: Multi-Agent / Protocol Evaluation / LLM Systems  
**Keywords**: Multi-agent protocols, A2A / ACP / ANP / Agora, Protocol routing, Failure recovery, End-to-end latency  

## TL;DR
ProtocolBench provides the first systematic comparison of four major LLM multi-agent communication protocols (A2A, ACP, ANP, Agora) across four axes: task success, end-to-end latency, message byte overhead, and failure robustness. The study finds that protocol selection results in a $36.5\%$ difference in completion time and a $3.48s$ latency gap. Furthermore, it proposes ProtocolRouter to dynamically select protocols based on scenarios/modules, reducing Fail-Storm recovery time by $18.1\%$.

## Background & Motivation

**Background**: LLM multi-agent systems (MAS) are transitioning from research prototypes to production (e.g., CAMEL, ChatDev, MetaGPT, AutoGen). Underlying communication relies on various protocols such as A2A (Google), ACP (IBM BeeAI), ANP, and Agora. Standards like MCP handle tool calling, while IoA manages dynamic discovery.

**Limitations of Prior Work**: Protocol selection is largely based on intuition, lacking standardized guidelines: (1) existing benchmarks assume fixed communication mechanisms and only evaluate task-level results (Zhu 2025, Hyun 2025), treating protocols as black boxes; (2) protocol selection simultaneously affects task success, latency, byte overhead, and failure robustness, creating tightly coupled trade-offs; (3) fair comparison is difficult—it requires fixing non-protocol factors (models, prompts, hardware, rate limits) without using abstraction layers that hide native protocol behaviors like retry, reconnect, or streaming; (4) a large evaluation space (protocol $\times$ topology $\times$ scale + dynamic failures) requires unified, lightweight logging.

**Key Challenge**: Practitioners seek an answer to "which protocol is better," but the answer is heavily scenario-dependent (one excels in GAIA, another in Streaming, and a different one in Fail-Storm). No single protocol is optimal across all scenarios. This implies a need for "scenario-based selection/synthesis" rather than picking a single "best" protocol.

**Goal**: (1) Provide a fair and reproducible protocol evaluation while maintaining native behaviors without introducing abstraction layers; (2) cover core use cases with four core axes and four scenarios; (3) go beyond evaluation by providing a router to automate protocol selection.

**Key Insight**: Utilize "thin wrappers" to wrap native protocol implementations (preserving internal retry/streaming logic) while unifying logs and metrics. Scenarios cover task quality (GAIA), latency/throughput (Streaming Queue), safety (Safety Tech), and failure recovery (Fail-Storm Recovery). The router employs a constraint-aware selection algorithm, treating protocol selection as a learnable problem.

**Core Idea**: Transform "which protocol to choose" from an implicit intuition into a quantifiable, routable engineering decision. The router does not replace the protocols but performs selection and provides cross-protocol stateless encode/decode bridges.

## Method

### Overall Architecture

**ProtocolBench**:
- Four Protocols: A2A, ACP, ANP, Agora.
- Four Scenarios: GAIA (document QA task quality), Streaming Queue (long-stream latency/throughput), Safety Tech (safety testing), Fail-Storm Recovery (recovery under multi-node failures).
- Four Metric Axes: Task success/quality, end-to-end latency/throughput, message/byte overhead, failure-time robustness.
- Implementation: Thin wrappers surrounding native protocols with shared models/prompts/hardware to ensure fairness.

**ProtocolRouter**:
- Input: User constraint requirements (e.g., "latency $< 10s$" or "recovery time $< 5s$") + runtime signals (current load, failure status).
- Output: Protocol selection per-scenario or per-module.
- Cross-protocol Message Translation: Stateless encode/decode bridges within adapters maintain application semantics while clearly indicating security boundary changes.
- Focuses on selection and composition without modifying the protocols themselves.

### Key Designs

1. **Thin Wrappers for Native Protocol Fidelity**:
    - **Function**: Avoids abstraction layers that mask native protocol behaviors like retry, reconnect, and streaming during evaluation.
    - **Mechanism**: Each protocol has a thin wrapper exposing a unified interface while utilizing the native SDK internally. A unified logging schema makes metrics comparable, and a shared scenario suite ensures all protocols run the same workload.
    - **Design Motivation**: Previous evaluations used abstraction layers that turned protocols into "generic RPCs," obscuring differentiated capabilities like A2A streaming or ANP reconnection. Since these are core to protocol design, thin wrappers ensure a fair comparison by preserving native behavior.

2. **Four Scenario $\times$ Four Metric Design**:
    - **Function**: Covers all critical dimensions affected by protocols.
    - **Mechanism**: Each scenario polarizes a specific metric—GAIA emphasizes quality, Streaming Queue emphasizes latency/throughput, Safety Tech emphasizes compliance, and Fail-Storm Recovery emphasizes robustness. While metrics are coupled, each has a "home base" scenario.
    - **Design Motivation**: Single-scenario evaluation leads to single-metric optimization, hiding trade-offs. This design forces the exposure of which protocol excels in different scenarios, providing training signals for the router.

3. **Constraint-aware ProtocolRouter**:
    - **Function**: Dynamically selects protocols based on user constraints and runtime signals.
    - **Mechanism**: Formalizes protocol selection as a constrained optimization problem. Given constraints (latency, cost, safety boundary), it selects the optimal candidate from the four protocols. It considers both per-scenario and per-module selection (different protocols for different stages). Stateless encode/decode bridges preserve application semantics.
    - **Design Motivation**: Since no single protocol is optimal everywhere, dynamic routing is necessary. Stateless bridges allow the router to generate executable outputs without breaking existing applications.

## Key Experimental Results

### GAIA Document QA (Quality Polarization)

| Protocol | Quality↑ | Success↑ |
|----------|----------|----------|
| **A2A**  | **2.51** ($+7.7\%$) | **9.29** ($+27.6\%$) |
| ACP      | $2.33$ | $7.28$ |
| ANP      | $2.21$ | $6.94$ |
| Agora    | $2.18$ | $6.81$ |

A2A significantly outperforms others on GAIA (quality $+7.7\%$, success $+27.6\%$).

### Streaming Queue (Latency Polarization)

| Protocol | Mean Latency (s)↓ | Variance↓ | Completion Time (min)↓ |
|----------|-------------------|-----------|------------------------|
| **ACP**  | **9.66**          | Lowest    | **40.28**              |
| A2A      | $11.42$           | Medium    | $47.83$                |
| ANP      | $12.18$           | Medium    | $51.20$                |
| Agora    | $13.14$           | High      | **54.97**              |

ACP has the lowest latency and variance; Agora performs worst here. The end-to-end completion gap reaches $36.5\%$ ($40.28$ vs $54.97$ min).

### Fail-Storm Recovery (Robustness Polarization)

| Protocol | Answer Retention (Post/Pre-failure) |
|----------|--------------------------------------|
| **A2A**  | **98.85%** (post $14.57$ / pre $14.74$) |
| ACP      | $92.41\%$ |
| ANP      | $86.96\%$ |
| Agora    | $81.29\%$ |

A2A exhibits the strongest robustness; Agora loses $19\%$ performance under failure.

### ProtocolRouter Gain

| Task | Best Single Protocol Baseline | **ProtocolRouter** | Δ |
|------|------------------------------|-------------------|---|
| Fail-Storm Recovery Time | A2A: $8.00s$ | **6.55s** | **−18.1%** |
| GAIA Success | A2A: $9.29$ | **9.90** | $+6.6\%$ |

The router is not a blanket-dominate solution (due to inherent metric trade-offs) but accurately improves target metrics under explicit constraints.

### Key Findings
- **No Single Optimal Protocol**: Each of the four scenarios has a "home protocol" (e.g., A2A for GAIA/Fail-Storm, ACP for Streaming), proving that protocol selection must be scenario-dependent.
- **Significant Impact of Selection**: The $36.5\%$ completion time gap, $3.48s$ latency difference, and $17.56\%$ robustness variance are major engineering disparities, not mere noise.
- **Precision Optimization via Router**: Constraint-aware selection reduces Fail-Storm recovery time by $18.1\%$, proving dynamic routing is a viable approach.
- **Per-module Opportunities**: Using different protocols for different stages (e.g., ACP for low-latency retrieval, A2A for robust recovery) shows significant potential.

## Highlights & Insights
- **First Systematic Quantification of Protocol Selection**: While previous MAS research focused on agent roles and prompts, this paper reveals that the protocol itself significantly impacts system behavior, opening the "protocol-aware MAS design" direction.
- **Engineering Rigor in Fair Evaluation**: The decision to use thin wrappers to preserve native retry/streaming behaviors is critical. Previous evaluations with abstraction layers made protocols appear "equivalent," whereas this methodology reveals true differences.
- **Scenario-axis Decoupled Design**: The $4 \times 4$ scenario/metric matrix forces the exposure of trade-offs. This evaluation framework template can be generalized to any "multi-option with multi-dimensional trade-off" problem.
- **Closed-loop from Evaluation to Action**: Beyond just reporting findings, the router provides an executable automation mechanism. Stateless bridges allow deployment in production while maintaining application semantics.

## Limitations & Future Work
- Only four mainstream protocols were evaluated; adjacent standards like MCP, IoA, or LMOS were not included.
- The four scenarios focus on typical cases; long-tail use cases (e.g., cross-organization security negotiation, dynamic agent discovery) are less covered.
- The Current Router is constraint-aware and rule-based; RL-based or learned routers could be explored.
- Cross-protocol stateless bridges might lose information for certain stateful protocol features (e.g., A2A streaming sessions).
- Evaluation of the security dimension (authentication, encryption, zero-trust) is preliminary.
- The benchmark dataset is fixed; re-testing will be required as new protocols/scenarios emerge.

## Related Work & Insights
- **vs MultiAgentBench (Zhu 2025)**: That study evaluates agent collaboration while fixing the protocol; this work specifically evaluates the protocols.
- **vs Frameworks (LangChain, LangGraph, AutoGen)**: These frameworks often hardcode communication modes; this paper reveals the differentiated impact of those modes.
- **vs Network Protocol Benchmarks (TCP/HTTP)**: While network layers have mature evaluation systems, LLM multi-agent protocols have lacked them until now.
- **Insights**: Protocol selection is a first-class engineering decision that deserves automated tooling. The scenario-axis evaluation template is applicable to any "no one-size-fits-all" tool selection problem (e.g., vector DBs, checkpoint formats).

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First systematic LLM multi-agent protocol benchmark and router; fills a critical community gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage ($4 \times 4 \times 4$), extensive data, and clear quantification of trade-offs.
- **Writing Quality**: ⭐⭐⭐⭐ Clear architecture diagrams (Figure 1). While data presentation is dense, take-home conclusions are explicit.
- **Value**: ⭐⭐⭐⭐⭐ Directly serves multi-agent engineering in production; the router is immediately adoptable; likely to influence future protocol designs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Which LLM Multi-Agent Protocol to Choose?](../../ICLR2026/multi_agent/which_llm_multi-agent_protocol_to_choose.md)
- [\[ICML 2026\] EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions](engiagent_fully_connected_coordination_of_llm_agents_for_solving_open-ended_engi.md)
- [\[ICML 2026\] OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration](omac_a_holistic_optimization_framework_for_llm-based_multi-agent_collaboration.md)
- [\[ICML 2026\] MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](maspo_joint_prompt_optimization_for_llm-based_multi-agent_systems.md)
- [\[ICML 2026\] E-mem: Multi-Agent Based Episodic Context Reconstruction for LLM Agent Memory](e-mem_multi-agent_based_episodic_context_reconstruction_for_llm_agent_memory.md)

</div>

<!-- RELATED:END -->
