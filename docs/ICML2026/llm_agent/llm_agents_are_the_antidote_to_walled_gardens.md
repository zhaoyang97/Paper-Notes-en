---
title: >-
  [Paper Note] LLM Agents Are the Antidote to Walled Gardens
description: >-
  [ICML 2026][LLM Agent][universal interoperability] This ICML 2026 position paper argues that LLM agents can "bypass" the closed API strategies of mainstream platforms through automated format conversion and human-like UI interaction to achieve "universal interoperability." This dismantles the "walled gardens" created by traditional network effects, while simultaneously
tags:
  - ICML 2026
  - LLM Agent
  - universal interoperability
  - walled gardens
  - data portability
  - agent security
date: 2026-05-08
content_hash: f1bfcacfa8173d33
---
# LLM Agents Are the Antidote to Walled Gardens

**Conference**: ICML 2026  
**arXiv**: [2506.23978](https://arxiv.org/abs/2506.23978)  
**Code**: None (Position Paper)  
**Area**: LLM Agent / Interoperability / AI Governance  
**Keywords**: LLM agent, universal interoperability, walled gardens, data portability, agent security

## TL;DR
This ICML 2026 position paper argues that LLM agents can "bypass" the closed API strategies of mainstream platforms through automated format conversion and human-like UI interaction to achieve "universal interoperability." This dismantles the "walled gardens" created by traditional network effects, while simultaneously requiring the ML community to proactively establish agent-friendly interfaces, security mechanisms, and ecosystem infrastructure to manage the resulting risks of security, legality, and new layers of lock-in.

## Background & Motivation
**Background**: While the underlying protocols of the Internet (TCP/IP, HTTP, DNS) are inherently open, the application layer is fragmented by "walled gardens"—social networks are disconnected, enterprise software uses proprietary APIs, and mobile platforms restrict developers to closed ecosystems. GDPR's data portability rights and the EU's DMA mandatory interoperability clauses are merely reactive, slow-paced local patches.

**Limitations of Prior Work**: Interoperability has long failed because technical cross-service integration is expensive and tedious (schema alignment, version compatibility, error handling, business rule encoding). Strategically, dominant platforms lack the incentive to allow users to migrate data easily (strong network effects $\rightarrow$ high switching costs $\rightarrow$ user lock-in). Legally, Terms of Service (ToS) generally prohibit automated access, and regulation lags behind platform evolution.

**Key Challenge**: User welfare and market competition require portability and interoperability, but platform commercial interests are built upon closure; traditional standardization (SOAP, REST, Semantic Web) and regulatory intervention cannot tear down these walls fast enough.

**Goal**: (1) Demonstrate why current LLM agent capabilities have fundamentally changed the cost structure of interoperability; (2) Provide a balanced analysis of the "universal interoperability" paradigm; (3) Propose three types of infrastructure that the ML community should proactively build to guide this trend toward a positive outcome.

**Key Insight**: The authors view LLM agents as a "universal adapter"—capable of dynamically adapting to any human-readable GUI or machine-readable API at runtime. This makes the strategic decision of "whether to open an API" irrelevant for platforms.

**Core Idea**: Using LLM agents to dynamically generate schema mappings, glue code, and UI script operations at runtime compresses integrations that previously took weeks of engineering into a few prompts. Rather than resisting, the community should build the scaffolding for interfaces, security, and governance before the agent ecosystem fully matures.

## Method

### Overall Architecture
As a position paper, this work does not propose an algorithm but constructs an argument chain: Background $\rightarrow$ Universal Adapter $\rightarrow$ Universal Interoperability $\rightarrow$ Call to Action $\rightarrow$ Alternative Views. It contrasts previous failures in standardization (XMPP/ActivityPub/FHIR/ISO 20022) and regulation (DMA/GDPR/ACCESS Act) with LLM agents defined as "universal adapters" that understand natural language, code, and structured formats while invoking APIs or simulating user actions. Universal interoperability emerges as the new paradigm where LLM adapters dynamically discover operations, infer schemas, and generate glue code/UI actions at runtime. The paper supports this urgency with two types of evidence: WebArena success rates jumping from 8.87% to 71.6% in 18 months (approaching the human baseline of 78.24%) and the launch of production web agents by leading labs (ChatGPT Atlas, Claude in Chrome, Gemini Computer Use, Perplexity Comet, Edge Copilot, Nova Act).

### Key Designs
The paper identifies three critical infrastructure categories for the ML community:

**1. Agent-Friendly Interfaces: Minimal Metadata to Skip Trial-and-Error**
To solve the problem of agents guessing hidden business rules through "trial-failure-prompt tuning," the authors propose non-intrusive annotations. For machine interfaces like REST, providers should supplement OpenAPI schemas with natural language rationales. For web pages, a manifest embedded in the DOM can map UI elements to endpoints (e.g., labeling a "Submit Order" button as `POST /api/order`), allowing agents to bypass UIs and call APIs directly.

**2. Security by Design: Three-Layer Runtime Security Architecture**
A three-layer enforcement architecture is proposed for autonomous data flows. The first layer consists of **signed permission documents** that specify allowed endpoints and data policies. The second layer is a **runtime policy checker** that intercepts actions violating permissions. The third layer provides **automatic rollback / kill-switches** to terminate out-of-bounds behavior. This establishes an agent-equivalent to OAuth.

**3. Ecosystem Infrastructure: Open Protocols and Governance**
To prevent the "walls" from shifting from the API layer to the agent layer, the authors advocate for participating in multi-party governance (W3C AI Agent Protocol, NANDA, etc.). Service providers should publish machine-readable changelogs (e.g., OpenAPI diffs) to allow agents to detect breaking changes. Open-source agent frameworks serve as the best defense against "agent-layer favoritism."

## Key Experimental Results

### Main Results: Universal Interoperability vs. Existing Paradigms

| Paradigm | Interface Contract | Adaptation Method | Major Weakness |
| :--- | :--- | :--- | :--- |
| Static Middleware | Pre-programmed | Manual code | High maintenance, poor reuse |
| Semantic Web | Global Ontology | Schema registries | High entry barrier, semantic drift |
| Standardized API | Pre-defined contract | Client generation | Requires uniform consensus |
| RPA / Scrapers | None | UI Scripts | Fragile to UI changes |
| **Universal Interoperability** | **Runtime Inference** | **LLM Dynamic Gen** | Security, technical debt, lock-in |

### Quantitative Evidence: Exponential Growth of Web Agent Capability

| Date | Evaluation / Event | Value or Phenomenon |
| :--- | :--- | :--- |
| 2023-03 | WebArena Success Rate | 8.87% |
| 2026-01 | WebArena Success Rate | 71.6% (Human: 78.24%) |
| 2024–2026 | Production Web Agents | 6+ Major Labs |
| 2024–2025 | Out-of-bounds Cases | robots.txt violations, spam injection |

### Key Findings
*   WebArena performance increased by nearly an order of magnitude in three years, moving "UI bypass" from research to production reality.
*   "In-the-wild" violations occurred before academic frameworks, making immediate action necessary.
*   Risks are categorized as engineering and governance challenges rather than insurmountable obstacles.

## Highlights & Insights
*   **Economic + Engineering Perspective**: The paper uses network effect theories (Katz–Shapiro) to explain LLM agents as tools for reducing "multi-homing" and "switching costs."
*   **"Bypassing" Over "Dismantling"**: Instead of forcing APIs through regulation, agents render the closure of APIs strategically meaningless by interacting with UIs.
*   **Runtime Security Blueprint**: The three-layer architecture (signed permissions $\rightarrow$ checker $\rightarrow$ rollback) provides a direct design pattern for agent gateways and frameworks like MCP or A2A.

## Limitations & Future Work
*   **Lack of Quantitative Definition**: There is no formal benchmark for "how universal" an adapter is.
*   **Feasibility of Runtime Policy Checkers**: Implementing low-latency, low-false-positive checks under industrial agent traffic remains an unsolved prototype challenge.
*   **Weak Anti-monopoly Measures**: Open-source models alone may not prevent the network effects of dominant agent frameworks.
*   **Economic Impact on Content Creators**: Insufficient discussion on the sustainability of ad-revenue models when agents consume content instead of users.

## Related Work & Insights
*   **vs. Empirical Agent Work (WebArena, Gorilla)**: Previous work asks "can they do it"; this paper asks "what happens to the industry when they do."
*   **vs. A2A / MCP**: While A2A and MCP are specific protocols, this paper warns of single-company lock-in risks and proposes multi-party governance.
*   **vs. Ironies of Automation (Bainbridge, 1983)**: Adapts classic human-reliability concerns to the agent era, where humans move from operators to supervisors with diminished intervention capacity.

## Rating
*   Novelty: ⭐⭐⭐⭐
*   Experimental Thoroughness: ⭐⭐⭐
*   Writing Quality: ⭐⭐⭐⭐⭐
*   Value: ⭐⭐⭐⭐

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Constitutional Black-Box Monitoring for Scheming in LLM Agents](constitutional_black-box_monitoring_for_scheming_in_llm_agents.md)
- [\[ICML 2026\] Agent-Omit: Adaptive Context Omission for Efficient LLM Agents](agent-omit_adaptive_context_omission_for_efficient_llm_agents.md)
- [\[ICML 2026\] ExCyTIn-Bench: Evaluating LLM Agents on Cyber Threat Investigation](excytin-bench_evaluating_llm_agents_on_cyber_threat_investigation.md)
- [\[ICML 2026\] Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use](reward_hacking_benchmark_measuring_exploits_in_llm_agents_with_tool_use.md)
- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)

</div>

<!-- RELATED:END -->
