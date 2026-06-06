---
title: >-
  [Paper Note] MCP-Persona: Evaluating LLM Agent Capabilities in Real Personalized Applications via Environment Simulation
description: >-
  [ICML 2026][LLM Agent][MCP] MCP-Persona is the first LLM agent benchmark targeting real personalized MCP tools (12 servers including Slack, Rednote, Instagram, Lark…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "MCP"
  - "Personalization"
  - "agent benchmark"
  - "environment simulation"
  - "tool calling"
date: 2026-05-08
content_hash: 136747792edaf5ac
---

# MCP-Persona: Evaluating LLM Agent Capabilities in Real Personalized Applications via Environment Simulation

**Conference**: ICML 2026  
**arXiv**: [2606.02470](https://arxiv.org/abs/2606.02470)  
**Code**: https://github.com/wwh0411/MCP-Persona  
**Area**: LLM Agent / Benchmark / Tool Use  
**Keywords**: MCP, Personalization, agent benchmark, environment simulation, tool calling

## TL;DR
MCP-Persona is the first LLM agent benchmark targeting real personalized MCP tools (12 servers including Slack, Rednote, Instagram, Lark, etc.). It introduces a suite of three methods—Tool-Traverse, Context-Tree, and Persona-Gen—using LLMs to autonomously synthesize Python simulator code to avoid real-world account issues. Evaluations of 10+ SOTA agents reveal that even Claude-Sonnet-4.5 achieves only 38.66% Acc, proving that personalized tool use is a severely underestimated technical bottleneck.

## Background & Motivation

**Background**: MCP is the standard protocol for connecting LLMs to external tools and has been widely adopted by Anthropic Skills, OpenClaw, and others. However, existing tool-use benchmarks (AppWorld, PersonaBench, MCP-Universe, ToolAthlon) mostly utilize general information-seeking tools or synthetic tools, failing to evaluate personalized tools that "bind to user accounts and manipulate local states."

**Limitations of Prior Work**: Evaluating personalized MCP faces three main challenges: (1) real-world deployment requires private user data and extensive manual configuration; (2) privacy and security restrictions prevent open data sharing; (3) maintaining stable, executable simulation environments is a non-trivial technical challenge.

**Key Challenge**: Realistic evaluation requires authentic data (privacy issues), while synthetic evaluation results in a distribution gap (unreliable evaluation signals). Existing benchmarks favor the synthetic route, lacking coverage for widely used applications like Slack, Instagram, and Lark.

**Goal**: To build an evaluation platform that reflects real-world personalized tool behavior without relying on real user data, covering four major categories: social media, corporate collaboration, email, and content management.

**Key Insight**: A traverse-then-simulate paradigm. First, successes and failures of Function Calling (FC) are traversed using real accounts in a sandbox to record behaviors. Then, LLMs autonomously synthesize Python code as a simulator to ensure the distribution remains close to reality. User context is modeled using a tree hierarchy, and tasks are generated via tool chain sampling, instruction fuzzification, and manual verification.

**Core Idea**: Three tools are developed: (1) Tool-Traverse: expanded pool via seed FC and adversarial generation, recording real behaviors for LLM-driven Python simulator synthesis; (2) Context-Tree: hierarchical entity representation (User → Calendar → Event); (3) Persona-Gen: 173 tasks derived from tool chain sampling, prototyping, context injection, fuzzing, and manual verification.

## Method

### Overall Architecture

The interaction involves three components: Tools, Contexts, and Tasks. Each MCP server obtains a simulator kernel via Tool-Traverse, a user profile via Context-Tree, and 173 manually verified tasks via Persona-Gen. Agents execute tasks in the simulated environment, and performance is evaluated using Acc, SR, and Exec-Acc.

### Key Designs

1.  **Tool-Traverse: Replicating real MCP services via traverse-then-simulate**:
    *   **Function**: Enables the simulator to behave consistently with the real server (including error handling) without relying on real accounts.
    *   **Mechanism**: (a) Bootstrapping — Manually written valid seed calls $x_{\text{seed}}$ are executed on live servers to record $(t, x_{\text{seed}}, y_{\text{seed}}, \tau)$; (b) Adversarial Failure Induction — LLMs perturb seed inputs to cover four types of errors: Type Mismatch, Schema Violation, Boundary, and Semantic Conflict, recording failure responses from real servers; (c) Code-Based Simulation — Based on tool schemas, behavioral traces, and context handler APIs, LLMs autonomously synthesize a Python file $K_t$ to implement the transition $f_t: (\mathcal{C}_{\text{current}}, x) \to (\mathcal{C}_{\text{new}}, y)$, including input validation, entity checks, and complete error response logic.
    *   **Design Motivation**: Manual mocking easily misses error handling and cannot scale to 12 servers. The adversarial system covers error patterns, while LLM-as-coder reduces manual labor from writing full simulators to writing seed FCs.

2.  **Context-Tree: Hierarchical modeling of user context**:
    *   **Function**: Enables the simulator to support stateful multi-turn operations and personalized task synthesis.
    *   **Mechanism**: (a) Hierarchy Identification — Aggregating entity types, fields, and relations from the tool call pool, producing a "root-at-User" hierarchy (e.g., Lark: User → Calendar → Event) verified manually; (b) Tree Construction — Peer children of a parent entity are stored in identifier-indexed maps, with cross-type links using foreign keys; (c) Content Generation — LLMs assign generation methods to each field: Enumerate (`iplocation`), Free-Form (`channel_name`), Random (`chat_id`), or Authentic (sampled from real Rednote posts); (d) Cross-Entity Linking — Reference fields sample identifiers from previously generated content.
    *   **Design Motivation**: The tree structure matches real MCP server data structures, supporting efficient lookup and updates. Four generation methods cover different field properties, and authentic content enhances realism while replacing sensitive fields with fakes.

3.  **Persona-Gen: Two-stage task generation and manual verification**:
    *   **Function**: Produces 173 high-quality personalized tasks covering single-server and cross-server scenarios, with fuzzified instructions to simulate real users.
    *   **Mechanism**: (a) Tool Chain Sampling — Topological sampling following 5 principles (Dependency, Personalization, Deduplication, Coherence, Realism); (b) Instruction Prototyping — LLMs use typed placeholders $P$ to abstract instruction templates $S_{\text{proto}}$; (c) Context Enrichment — Entity values sampled from the context-tree replace placeholders to generate $S_{\text{inst}}$; (d) Fuzzification — Removing implicit context (e.g., a colleague's `user_id` can be inferred via a shared group) to obtain fuzzy instructions $S_{\text{fuzz}} = \mathcal{F}(S_{\text{inst}} \setminus \mathcal{C}_{\text{imp}})$; (e) Human Verification — Checking consistency and increasing difficulty (e.g., 1 post → 10 posts, pruning unnecessary context).
    *   **Design Motivation**: Purely automatic tasks are often unrealistic or too simple. A 4-step pipeline with manual verification ensures both scale and quality. Implicit context simulates the real-world challenge where user instructions are incomplete and agents must complete them from the environment.

## Key Experimental Results

### Benchmark Comparison

| Benchmark | Real-World | Personal Context | Social Media | Collab | Email | Content |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| AppWorld | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ |
| PersonaBench | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ |
| InfoMosaic-Bench | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| MCP-Universe | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| TOOLATHLON | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ |
| **MCP-Persona** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Ours is the only benchmark covering all 5 dimensions, specifically Social Media and Collaboration Platforms, which are entirely absent from other benchmarks.

### Main Results: SOTA Agents on MCP-Persona

| Model | Collab | Content | Social | Email | Lark | Rednote | Hodgepodge | Acc | SR-0.8 | Exec-Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Claude-Sonnet-4.5 | 39.94 | 19.76 | 47.04 | 43.63 | 40.81 | 42.37 | 12.50 | **38.66** | 10.40 | 41.50 |
| GPT-5 | 43.50 | 22.57 | 42.64 | 47.17 | 37.67 | 34.66 | 12.50 | 36.99 | 6.94 | 41.45 |
| Claude-Opus-4.1 | 38.79 | 13.56 | 44.79 | 9.71 | 39.67 | 34.70 | 25.00 | 34.52 | 7.05 | 36.77 |
| o4-mini | 34.38 | 21.22 | 35.61 | 53.83 | 30.43 | 25.25 | 6.25 | 30.70 | 5.78 | 34.73 |
| o3 | 26.41 | 14.55 | 32.78 | 41.08 | 34.64 | 26.05 | 37.50 | 29.79 | 5.20 | 30.27 |
| GPT-4o | 24.50 | 7.58 | 36.98 | 12.57 | 30.65 | 20.29 | 25.00 | 25.56 | 4.35 | 20.02 |

**Claude-Sonnet-4.5 achieves an overall Acc of only 38.66%**, proving personalized tools are a significant bottleneck for current LLM agents. Performance on Hodgepodge (cross-server mixtures) is particularly poor (12-25%).

### Key Findings

*   **SOTA models under 50%**: All models have Acc < 50%, far lower than the 70-80%+ seen on general tool benchmarks.
*   **Cross-server coordination is a key bottleneck**: Performance on Hodgepodge is generally between 12-25%, showing coordination across multiple services is much harder than single-server tasks.
*   **Content Management is the most difficult**: All models performed worst in Content Management (< 25%), which requires deep understanding of user history rather than simple CRUD.
*   **Massive variance in Email tasks**: o4-mini scored 53.83 and GPT-5 scored 47.17, while Claude-Opus-4.1 scored only 9.71 and GPT-4o scored 12.57, likely due to varying amounts of email data in training.
*   **Simulation Validity**: Experiments show high prediction accuracy between simulator responses and real servers, validating Tool-Traverse.

## Highlights & Insights

*   **First benchmark covering personalized MCP tools**: Evaluations for Slack, Rednote, Instagram, and Lark were previously blank; this work fills a critical gap.
*   **Traverse-then-simulate paradigm**: Combining real account traversal with LLM-generated simulator code makes benchmark construction feasible and realistic.
*   **Error mode coverage in Tool-Traverse**: Systematic adversarial generation for 4 error types ensures the simulator accurately replicates server error handling.
*   **Hybrid Context-Tree approach**: Tree structures match real-world data, and 4 content generation methods (including authentic text) balance realism and privacy.
*   **Fuzzification in Persona-Gen**: Removing implicit context simulates the core difficulty of incomplete user input, serving as a primary source of benchmark difficulty.
*   **Shocking Conclusion**: SOTA performance below 50% makes it clear that much work remains for personalized agent capabilities.

## Limitations & Future Work

*   **Small task scale (173 tasks)**: Compared to the 10,000s of items in MMLU, 173 tasks may lack statistical power in some dimensions.
*   **Limited application coverage**: 12 MCP servers are primarily EN/CN; ecosystems like LINE or KakaoTalk are not yet covered.
*   **Simulator fidelity**: While traverse-then-simulate is close to reality, some edge cases might be missed, potentially leading to agent capability over-estimates.
*   **Lack of failure analysis**: While SOTA is below 50%, there is no detailed analysis on whether failures stem from information seeking or multi-step coordination.
*   **Stricter privacy handling**: Although sensitive fields are replaced, more systematic privacy guarantees are needed for authentic content.
*   **No adversarial users**: The benchmark assumes well-intentioned tasks; real users might provide intentionally vague or adversarial instructions.
*   **Lack of agent training baseline**: The benchmark only evaluates SOTA models and does not explore whether specific fine-tuning can improve personalized tool use.

## Related Work & Insights

*   **vs AppWorld / PersonaBench**: These also attempt to evaluate personalized agents but use synthetic tools; MCP-Persona uses real MCP tools.
*   **vs ToolAthlon**: Real-world but lacks Social Media and Collaboration coverage due to account binding issues; MCP-Persona bypasses this via simulation.
*   **vs Tau-Bench**: The first tool-agent-user benchmark, but uses synthetic airline/retail tools; MCP-Persona uses a real-world distribution.
*   **vs InfoMosaic-Bench / MCP-Universe**: These cover real-world tools but are not personalized; MCP-Persona is the only one to be both real and personal.
*   **Insights**: (1) The traverse-then-simulate paradigm is applicable to any domain where real deployment requires private data but the benchmark must be open; (2) LLM-as-coder for simulator development is a key trick for scalable benchmarks; (3) The evaluation gap in personalized agent capabilities serves as a reminder not to be misled by high scores on general tool benchmarks.

## Rating

*   **Novelty**: ⭐⭐⭐⭐⭐ The traverse-then-simulate paradigm, LLM-as-coder simulators, and Context-Tree hierarchy are methodological innovations in benchmark construction.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid evaluation of 10+ SOTA agents, simulator validation, and multi-category comparison; lacks failure analysis and fine-tuning experiments.
*   **Writing Quality**: ⭐⭐⭐⭐ Clear presentation of the three-component pipeline; Figure 1 is intuitive, and Table 2 clearly shows the SOTA gap. Some prompt details for the simulator synthesis are in the appendix, affecting reproducibility slightly.
*   **Value**: ⭐⭐⭐⭐⭐ Directly reveals weaknesses in LLM agent personalization; provides a necessary evaluation foundation for the agent training community. The open-source nature and coverage of high-demand apps like Slack/Lark offer direct industrial value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools](../../ACL2026/llm_agent/mcp-flow_facilitating_llm_agents_to_master_real-world_diverse_and_scaling_mcp_to.md)
- [\[ACL 2026\] OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation](../../ACL2026/llm_agent/opera_a_dataset_of_observation_persona_rationale_and_action_for_evaluating_llms_.md)
- [\[AAAI 2026\] LieCraft: A Multi-Agent Framework for Evaluating Deceptive Capabilities in Language Models](../../AAAI2026/llm_agent/liecraft_a_multi-agent_framework_for_evaluating_deceptive_capabilities_in_langua.md)
- [\[ICML 2026\] Adaptive Querying with AI Persona Priors](adaptive_querying_with_ai_persona_priors.md)
- [\[ICLR 2026\] LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News](../../ICLR2026/llm_agent/livenewsbench_evaluating_llm_web_search_capabilities_with_freshly_curated_news.md)

</div>

<!-- RELATED:END -->
