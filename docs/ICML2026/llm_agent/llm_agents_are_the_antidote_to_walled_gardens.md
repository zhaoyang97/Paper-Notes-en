---
title: >-
  [Paper Note] LLM Agents Are the Antidote to Walled Gardens
description: >-
  [ICML 2026][LLM Agent][universal interoperability] This ICML 2026 position paper argues that LLM agents can "bypass" the closed API strategies of mainstream platforms through automated format conversion and human-like UI…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "universal interoperability"
  - "walled gardens"
  - "data portability"
  - "agent security"
date: 2026-05-08
content_hash: ada9539518fa56bd
---

# LLM Agents Are the Antidote to Walled Gardens

**Conference**: ICML 2026  
**arXiv**: [2506.23978](https://arxiv.org/abs/2506.23978)  
**Code**: None (Position Paper)  
**Area**: LLM Agent / Interoperability / AI Governance  
**Keywords**: LLM agent, universal interoperability, walled gardens, data portability, agent security

## TL;DR
This ICML 2026 position paper argues that LLM agents can "bypass" the closed API strategies of mainstream platforms through automated format conversion and human-like UI interaction. This facilitates "universal interoperability," dismantling the "walled gardens" created by traditional network effects. Simultaneously, it calls on the ML community to proactively establish agent-friendly interfaces, security mechanisms, and ecological infrastructure to manage the resulting risks of security, legality, and new layers of lock-in.

## Background & Motivation
**Background**: The underlying protocols of the Internet (TCP/IP, HTTP, DNS) are inherently open, yet the application layer is fragmented into "walled gardens"—social networks are disconnected, enterprise software uses private APIs, and mobile platforms restrict developers to closed ecosystems. GDPR’s data portability rights and the EU DMA’s mandatory interoperability clauses are merely reactive, slow-paced local patches.

**Limitations of Prior Work**: Interoperability has long failed because building and maintaining cross-service integrations is technically expensive and tedious (schema alignment, version compatibility, error handling, business rule encoding). Strategically, dominant platforms lack the incentive to allow users to move data easily (strong network effects $\rightarrow$ high switching costs $\rightarrow$ user lock-in). Legally, Terms of Service (ToS) generally prohibit automated access, and regulation lags behind platform evolution.

**Key Challenge**: User welfare and market competition require portability and interoperability, but platform commercial interests are built on closure. Neither traditional standardization (SOAP, REST, Semantic Web) nor regulatory intervention can tear down these walls fast enough.

**Goal**: (1) Demonstrate how current LLM agent capabilities have fundamentally changed the cost structure of interoperability; (2) Provide a balanced analysis of the "universal interoperability" paradigm; (3) Propose three types of infrastructure that the ML community should proactively build to guide this trend toward a positive outcome.

**Key Insight**: The authors view the LLM agent as a "universal adapter"—it can dynamically adapt at runtime to any human-readable GUI or machine-readable API, making the decision of whether to "open an API" no longer strategically significant for platforms.

**Core Idea**: Use LLM agents to dynamically generate schema mappings, glue code, and UI operation scripts at runtime. This compresses integration tasks that previously took weeks into a few prompts, rendering the strategic choice of "whether to open an API" meaningless. Rather than resisting, the community should build the scaffolding for interfaces, security, and governance before the agent ecosystem matures.

## Method
As a position paper, there is no traditional algorithm. The authors construct an argumentative chain of "why + how." The following restates the logical framework and extracts three operative "Key Designs" corresponding to the action plans for the ML community in §5.

### Overall Architecture
The paper unfolds in the order of "Background $\rightarrow$ Universal Adapter $\rightarrow$ Universal Interoperability $\rightarrow$ Call to Action $\rightarrow$ Alternative Views":

- **§2 Background**: Supplements the technical and economic background of interoperability, noting the limitations of past standardization (XMPP/ActivityPub/FHIR/ISO 20022) and regulation (DMA/GDPR/ACCESS Act).
- **§3 A Universal Adapter**: Defines the LLM agent as a system that "understands/generates natural language, code, and structured formats, while being able to call APIs or simulate user operations." It proposes two key capabilities—automated format translation and robust UI interaction—which together "discount" the strategic choice of opening APIs to zero.
- **§4 Universal Interoperability**: Formally defines universal interoperability (using LLM adapters to dynamically discover operations, infer schemas, and generate glue code/UI actions at runtime), comparing it with traditional paradigms.
- **§5 Call to Action**: Three things the ML community can do—agent-friendly interfaces, security by design, and ecosystem infrastructure.
- **§6 Alternative Views**: Addresses four counter-arguments (regulation-first / ontology-first / security-conservative / economic sustainability) and integrates them into the proposed solution.

### Key Designs
The three types of infrastructure in §5 represent the most "implementable" components of the paper.

1.  **Agent-Friendly Interfaces**:
    *   **Function**: Overlay minimal metadata on existing APIs/webpages so LLM agents do not have to infer implicit business rules through "trial-error-prompt tuning" cycles.
    *   **Mechanism**: For machine interfaces like REST, service providers add a natural language rationale to the OpenAPI schema (writable by non-technical staff or LLMs). The simplest form is a link to documentation; an advanced form is an LLM endpoint specifically for schema clarification. For webpages, it embeds a manifest in the DOM mapping buttons/fields to API endpoints, allowing agents to skip the UI. `llms.txt` (Howard, 2024) is an early prototype of this.
    *   **Design Motivation**: Rather than waiting for new standards, it is better to incrementally extend existing norms via "links + annotations." The research problem shifts to "how much metadata is sufficient + what is the optimal combination of static annotations and dynamic explanation services."

2.  **Security by Design (Three-layer runtime security for agents)**:
    *   **Function**: Provides end-to-end security—from permission declaration to action validation and rollback—when agents autonomously operate on critical data flows.
    *   **Mechanism**: A three-layer runtime enforcement architecture: Layer 1 is *signed permission documents* (cf. South et al., 2025), issuing verifiable documents to each agent declaring allowed endpoints and policies. Layer 2 is the *runtime policy checker*, which compares every agent action against the permission document before execution, blocking violations. Layer 3 is an *automatic rollback / kill-switch* to revert data states upon boundary violations. Challenges lie in Layer 2—achieving low latency and low false positives using a hybrid of learned policy classifiers and symbolic checkers. Sandbox tools like ToolEmu and AgentSims integrate these into CI.
    *   **Design Motivation**: Decouples "agent autonomy" from "site controllability." Similar to OAuth's role for developers, universal interoperability requires an "agent permission/rate/delegation standard" to prevent sites from resorting to blunt IP blocking.

3.  **Ecosystem Infrastructure (Open protocols, technical debt, and anti-monopoly)**:
    *   **Function**: Prevents the "walls" of interoperability from shifting from the API layer to the agent/model layer, while establishing maintainability constraints for LLM-generated integration code.
    *   **Mechanism**: At the protocol layer, support open standards like Google A2A and Anthropic MCP, but mitigate lock-in risks by participating in multi-party standards (W3C AI Agent Protocol, NANDA, etc.). For technical debt, maintain open-source "reference implementation" templates and require API providers to publish machine-readable changelogs (e.g., OpenAPI diff) for agents to scan. For anti-monopoly, open-source frameworks are the best defense against "agent-layer favoritism," where agents must auditably log the basis for service selection.
    *   **Design Motivation**: Universal interoperability’s greatest threat is "lock-in under a different name"—if agent frameworks are closed or biased, underlying API openness is moot.

## Key Experimental Results

### Main Results: Universal Interoperability vs. Existing Paradigms
The authors did not run experiments but provided a paradigm comparison in §4:

| Paradigm | Interface Contract | Adaptation Method | Main Weakness |
| :--- | :--- | :--- | :--- |
| Static Middleware / Custom Adapter | Pre-programmed | Manual engineering | High maintenance, poor reuse |
| Semantic Web (RDF/OWL) | Global ontology | Schema registries | High entry barrier, semantic drift |
| Standardized API (OpenAPI/GraphQL) | Pre-defined contract | Generated clients | Requires unified specs, limited coverage |
| RPA / Rule-based Scrapers | None | UI scripts | Fragile to UI changes, no semantics |
| **Universal Interoperability** (Ours) | **Runtime inference** | **LLM dynamic glue/UI** | Security, tech debt, agent lock-in |

### Key Findings
*   **Exponential Growth in Web Agent Capability**: WebArena success rates grew from 8.87% (March 2023) to 71.6% (January 2026), approaching human performance (78.24%). This indicates that "bypassing the UI" has moved from research to production reality.
*   **"Wild" Cases Precede Frameworks**: Incidents involving Perplexity (ignoring robots.txt) and Akirabot (spamming 80k+ sites via CAPTCHA bypass) demonstrate that unregulated universal interoperability is already happening; the question is not "if," but "in what form."
*   **Manageable Risks**: Risks are categorized as "engineering and governance challenges rather than insurmountable obstacles"—a key rhetorical stance of the paper.

## Highlights & Insights
*   **Economic + Engineering Perspective**: Unlike papers focused solely on technology, this uses Katz–Shapiro / Farrell–Saloner network effect theories to explain the LLM agent's role as "reducing multi-homing and switching costs."
*   **"Walls Aren't Torn Down; They are Outflanked"**: Traditional anti-walled-garden approaches rely on "forced API opening" (GDPR/DMA). The authors propose a technical bypass—agents can scrape UIs and mimic humans, making API closure strategically futile.
*   **Portable Security Architecture**: The *signed permission document $\rightarrow$ runtime policy checker $\rightarrow$ automatic rollback* architecture is a blueprint for anyone building agent gateways or frameworks.
*   **Constructive Counter-argumentation**: §6's "Alternative Views" are not just rebuttals but are absorbed into the proposal. For example, the "regulation-first" view is translated into "embedding consent/regulation directly into agent interfaces."

## Limitations & Future Work
*   **Lack of Quantifiable Definitions**: There is no benchmark for what constitutes "sufficiently universal." Future work should design end-to-end universal-adapter benchmarks.
*   **Feasibility of Runtime Policy Checkers**: While the direction is set, whether a high-performance, low-latency checker can work in industrial agent traffic remains to be proven.
*   **Weak Anti-monopoly Measures for Agent Layers**: Relying solely on "open-source models + transparent logs" may be insufficient to counteract the network effects of top-tier agent frameworks.
*   **Impact on Content Producer Economics**: The discussion on how content ecosystems (reliant on ad revenue) survive when users use agents instead of visiting pages is limited to a brief acknowledgment in §6.

## Related Work & Insights
*   **vs. ReAct / WebArena / Gorilla / RestGPT**: While empirical works show "how agents can do this," this paper translates "can" into "what happens at the industry level."
*   **vs. A2A (Google) / MCP (Anthropic)**: These are specific protocol proposals. This paper places them within an "ecosystem infrastructure" framework and warns of single-company lock-in.
*   **vs. Bainbridge "Ironies of Automation" (1983)**: Adapts the classic view—humans moving from operators to monitors with decreased intervention capability—to underpin the security risk analysis in §4.2.
*   **vs. Kapoor et al. (2025) / Agranat & Gal (2025)**: Integrates concurrent research on agent-layer favoritism and AI agents' impact on network effects into its risk and anti-monopoly sections.

## Rating
*   **Novelty**: ⭐⭐⭐⭐ Establishes "universal interoperability" as a framework connecting LLM agents to industrial organization theory.
*   **Experimental Thoroughness**: ⭐⭐⭐ As a position paper, it lacks its own benchmark but uses WebArena trends and real-world cases effectively. Missing prototype validation for policy checkers.
*   **Writing Quality**: ⭐⭐⭐⭐⭐ Exceptionally clear structure, interdisciplinary literature coverage, and an exemplary handling of alternative views.
*   **Value**: ⭐⭐⭐⭐ Highly significant for agent framework developers, policymakers, and strategic teams, providing clear research directions for systems and ML researchers alike.

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
