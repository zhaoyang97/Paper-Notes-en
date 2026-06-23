---
title: >-
  [Paper Note] LLM Agents Are the Antidote to Walled Gardens
description: >-
  [ICML 2026][LLM Agent][universal interoperability] This ICML 2026 position paper argues that LLM agents can "bypass" the closed API strategies of dominant platforms through automatic format conversion and human-like UI interaction, achieving "universal interoperability." This dissolves the "walled gardens" created by traditional network effects. However, the ML communi
tags:
  - ICML 2026
  - LLM Agent
  - universal interoperability
  - walled gardens
  - data portability
  - agent security
date: 2026-05-08
content_hash: d999f2b54745c1b1
---
# LLM Agents Are the Antidote to Walled Gardens

**Conference**: ICML 2026  
**arXiv**: [2506.23978](https://arxiv.org/abs/2506.23978)  
**Code**: None (Position Paper)  
**Area**: LLM Agent / Interoperability / AI Governance  
**Keywords**: LLM agent, universal interoperability, walled gardens, data portability, agent security

## TL;DR
This ICML 2026 position paper argues that LLM agents can "bypass" the closed API strategies of dominant platforms through automatic format conversion and human-like UI interaction, achieving "universal interoperability." This dissolves the "walled gardens" created by traditional network effects. However, the ML community must proactively establish agent-friendly interfaces, security mechanisms, and ecological infrastructure to manage the resulting security, legal, and new-layer lock-in risks.

## Background & Motivation
**Background**: While the underlying internet protocols (TCP/IP, HTTP, DNS) are inherently open, the application layer is divided by a series of "walled gardens"—social networks are disconnected, enterprise software uses proprietary APIs, and mobile platforms restrict developers to closed ecosystems. GDPR's data portability rights and the EU's DMA mandatory interoperability clauses are merely reactive, slow-paced local fixes.

**Limitations of Prior Work**: Interoperability has long failed because building and maintaining cross-service integrations is technically expensive and tedious (schema alignment, version compatibility, error handling, business rule encoding). Strategically, dominant platforms lack the incentive to let users migrate data easily (strong network effects $\rightarrow$ high switching costs $\rightarrow$ user lock-in). Legally, Terms of Service (ToS) generally prohibit automated access, and regulation lags behind platform evolution.

**Key Challenge**: User welfare and market competition require portability and interoperability, but platform commercial interests are built precisely on closure; traditional standardization (SOAP, REST, Semantic Web) and regulatory intervention cannot tear down these walls fast enough.

**Goal**: (1) Demonstrate why the current capabilities of LLM agents have fundamentally altered the cost structure of interoperability; (2) Provide a balanced analysis of the "universal interoperability" paradigm; (3) Propose three types of infrastructure the ML community should build to guide this trend toward a positive outcome.

**Key Insight**: The authors view the LLM agent as a "universal adapter"—capable of dynamically adapting at runtime to any human-readable GUI or machine-readable API, making the strategic choice of whether a platform "opens its API" irrelevant.

**Core Idea**: Use LLM agents to dynamically generate schema mappings, glue code, and UI interaction scripts at runtime, compressing integrations that previously took weeks into a few prompts. This effectively negates the strategic value of withholding APIs. Rather than resisting, the community should proactively build the scaffolding for interfaces, security, and governance while the agent ecosystem is still nascent.

## Method

### Overall Architecture
As a position paper, it does not propose an algorithm but constructs an argument chain: Background $\rightarrow$ Universal Adapter $\rightarrow$ Universal Interoperability $\rightarrow$ Call to Action $\rightarrow$ Alternative Views. It contrasts past failures in standardization (XMPP/ActivityPub/FHIR/ISO 20022) and regulation (DMA/GDPR/ACCESS Act) with LLM agents defined as universal adapters—possessing both natural language/code/structured format understanding and the ability to call APIs or simulate user actions. These capabilities of automatic translation and robust UI interaction render the platform's strategic choice to open APIs effectively zero. Thus, universal interoperability—using LLM adapters to dynamically discover operations, infer schemas, and generate glue code/UI actions—becomes the new paradigm. The authors support this with two types of hard evidence: the success rate of WebArena jumping from 8.87% to 71.6% in 18 months (near the human level of 78.24%), and the production launch of web agents by leading labs (ChatGPT Atlas, Claude in Chrome, Gemini Computer Use, Perplexity Comet, Edge Copilot, Nova Act).

### Key Designs
Section 5 outlines three infrastructure categories for the ML community, answering how to build the necessary scaffolding for the agent ecosystem.

**1. Agent-Friendly Interfaces: Minimal metadata to skip trial-and-error cycles**
Current agents must guess implicit business rules behind interfaces through "trial-error-prompt adjustment" loops. The solution involves adding minimally invasive annotations to existing APIs/webpages rather than creating new standards. For machine interfaces like REST, providers should supplement OpenAPI schemas with natural language rationales. For webpages, a manifest embedded in the DOM could map buttons/forms to specific endpoints (e.g., labeling a "Submit Order" button as `POST /api/order`), allowing agents to bypass UI and call APIs directly. `llms.txt` serves as an early prototype of this direction.

**2. Security by Design: A three-layer runtime security architecture**
To handle autonomous data flows, the authors propose a three-layer enforcement architecture separating "agent autonomy" from "site controllability." Layer 1: signed permission documents, issuing verifiable permissions to each agent. Layer 2: runtime policy checkers that intercept actions against the permission document before execution. Layer 3: automatic rollback/kill-switches to terminate out-of-bounds behavior. The challenge lies in Layer 2—maintaining low latency and low false positives—which can be addressed through a mix of learned policy classifiers and symbolic checkers, integrated with sandboxes like ToolEmu or SandboxEval.

**3. Ecosystem Infrastructure: Open protocols, technical debt management, and anti-monopoly**
This targets the risk of "lock-in" shifting from the API layer to the agent/model layer. At the protocol layer, the authors support open standards like Google A2A and Anthropic MCP but warn of single-vendor lock-in, advocating for participation in multi-party groups like W3C AI Agent Protocol. For technical debt, the community should maintain open-source integration templates and machine-readable changelogs. To prevent monopolies, open-source agent frameworks and models act as the best defense against "agent-layer favoritism," requiring agents to provide auditable logs of their service selection logic.

## Key Experimental Results

### Main Comparison: Universal Interoperability vs. Prior Paradigms
Section 4 presents a comparison of paradigms:

| Paradigm | Interface Contract | Adaptation Method | Main Weakness |
|----------|-------------------|-------------------|---------------|
| Static Middleware / Custom Adapters | Pre-programmed | Manual engineering | High maintenance cost, hard to reuse |
| Semantic Web (RDF/OWL) | Global Ontology | Schema registries | High entry barrier, semantic drift |
| Standardized APIs (OpenAPI/GraphQL) | Pre-defined Contract | Auto-generated clients | Requires unified standards, limited coverage |
| RPA / Rule-based scrapers | None | UI Scripts | Fragile to UI changes, no semantic understanding |
| **Universal Interoperability** (Ours) | **Runtime Inference** | **LLM Dynamic Generation** | Security, technical debt, agent-layer lock-in |

### Quantitative Evidence: Exponential Growth of Web Agent Capabilities

| Time | Benchmark / Event | Value or Phenomenon |
|------|-------------------|---------------------|
| 2023-03 | WebArena Success Rate | 8.87% |
| 2026-01 | WebArena Success Rate | 71.6% (Human: 78.24%) |
| 2024–2026 | Production Web Agents | 6+ Labs (OpenAI, Anthropic, Google, etc.) |
| 2024–2025 | In-the-wild violations | Perplexity vs. robots.txt, Akirabot spamming 80k+ sites |

### Key Findings
- WebArena success rates increased by nearly an order of magnitude in three years, moving "UI bypassing" from research to production reality.
- In-the-wild cases have preceded academic frameworks, making action a "now-or-never" necessity.
- All identified risks are categorized as engineering and governance challenges rather than insurmountable obstacles.

## Highlights & Insights
- **Economic + Engineering Dual Perspective**: Unlike many LLM papers, this integrates Katz–Shapiro network effect theories to explain LLM agents as tools for reducing "multi-homing" and "switching costs."
- **"Walls aren't torn down; they're bypassed"**: The traditional approach relies on mandatory API opening (GDPR/DMA), whereas this identifies a technical path where agents render API closure strategically meaningless.
- **Three-Layer Security is Portable**: The signed permission → runtime checker → rollback architecture provides a tangible design blueprint for agent gateway developers.
- **Incorporating Counter-Arguments**: Section 6 treats alternative views not as opposition but as requirements to be absorbed into the solution, such as embedding consent into agent interfaces.

## Limitations & Future Work
- **Lack of Quantifiable Definition**: No specific benchmark exists for "universal interoperability" to measure "how universal" a system is.
- **Runtime Policy Checker Bottleneck**: Achieving low latency and low false positives in industrial-scale traffic remains an unproven direction.
- **Weak Anti-Monopoly Measures**: Relying on "open-source models + logs" may not be enough to counter the network effects of dominant agent frameworks.
- **Impact on Content Economics**: The discussion on how content creators sustain revenue when users bypass sites via agents is relatively thin.

## Related Work & Insights
- **vs. Empirical Agent Work (ReAct/WebArena)**: Those works answer if agents *can* do the task; this paper translates that "can" into what happens at the *industry level*.
- **vs. A2A / MCP**: These are specific protocol proposals; this paper places them in an infrastructure framework while warning of single-company lock-in.
- **vs. "Ironies of Automation" (1983)**: Adapts Bainbridge’s classic view—humans moving from operators to monitors—to argue the safety risks in Section 4.2.
- **vs. Agent-Layer Favoritism**: Incorporates recent 2025 studies on whether agents will become the new layer of monopoly.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Establishes "universal interoperability" and connects LLM agents to industrial organization theory.
- **Experimental Thoroughness**: ⭐⭐⭐ Position papers don't require benchmarks, but the use of WebArena trends and real-world cases provides a solid foundation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure and cross-disciplinary coverage; a model for position paper writing.
- **Value**: ⭐⭐⭐⭐ High utility for framework developers and policymakers; provides several research directions like policy checkers and agent-friendly schemas.

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
