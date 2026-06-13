---
title: >-
  [Paper Note] C-World: A Computer Use Agent Environment Creator
description: >-
  [ACL 2026][LLM/NLP][MCP Tools] The authors formalize the "agent environment" as a quadruple of Action / Task / Transition / Reward and implement it as C-World. It utilizes 5,571 real-world MCP tools…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "MCP Tools"
  - "Long-horizon tasks"
  - "World Model"
  - "State perturbation"
  - "Evaluation + Training Data Engine"
date: 2026-05-08
content_hash: cf7d550cfd99e03e
---

# C-World: A Computer Use Agent Environment Creator

**Conference**: ACL 2026  
**arXiv**: [2601.06328](https://arxiv.org/abs/2601.06328)  
**Code**: https://ziqiao-git.github.io/C-World/ (Available)  
**Area**: LLM Agent / Computer Use / Agent Environment  
**Keywords**: MCP Tools, Long-horizon tasks, World Model, State perturbation, Evaluation + Training Data Engine

## TL;DR
The authors formalize the "agent environment" as a quadruple of Action / Task / Transition / Reward and implement it as C-World. It utilizes 5,571 real-world MCP tools, automated task synthesis, state controller perturbations, and dual-signal rewards to provide high-fidelity evaluation. Furthermore, a "World Engine" is developed to simulate tool responses without live APIs, enabling scalable training. Evaluating 9 frontier LLMs reveals that "planning is generally strong while execution is generally weak." Fine-tuning with only 1,170 C-World trajectories outperforms a baseline trained on 119k samples.

## Background & Motivation

**Background**: Performance on single-turn function-calling has neared saturation (e.g., high scores on BFCL). However, real-world "computer use" tasks typically involve dozens of interaction rounds across multiple apps, containing ambiguous constraints and dynamic failures. Existing agent benchmarks (AgentBench, ToolBench, WebArena, etc.) either have a limited number of tools (<600) or consist of static problem sets, failing to support continuous training.

**Limitations of Prior Work**: (1) Small tool pools and narrow domains fail to reflect the breadth of real-world workflows; (2) Static tasks cannot "self-evolve" new constraints; (3) Lack of "perturbations"—tasks follow the happy path, preventing assessment of agent recovery or re-planning; (4) Simple evaluation signals (Pass/Fail or LLM judge) cannot decouple "planning failure" from "execution failure"; (5) Training depends entirely on live APIs, which are limited by cost, rate limits, and state instability, hindering large-scale trajectory collection.

**Key Challenge**: Enabling agents to learn in complex environments like humans requires "broad, deep, perturbable, and low-cost scalable" environments. Manually constructing such environments is prohibitively expensive, and relying on live APIs is constrained by the speed, cost, and instability of external services.

**Goal**: (1) Formalize the necessary components of an "agent environment"; (2) Construct a system capable of generating new environments on demand rather than a fixed benchmark; (3) Provide a "world model" mode independent of live APIs to enable scalable training.

**Key Insight**: Adapt the mature MDP quadruple (action space, transition, reward, task distribution) from RL to LLM agents, and model "tool call responses" as a transition function simulated by an LLM → World Engine.

**Core Idea**: Define a framework with four components $(\mathcal{A}, \mathcal{T}, \mathcal{F}, \mathcal{R})$ implemented via a dual-track of Real Mode and Synthetic Mode. All four components replace manual labor with "automated synthesis + LLM simulation."

## Method

### Overall Architecture
C-World formalizes the agent environment into four components plus a World Engine.
- $\mathcal{A}$ Action Space: Scrapes 276 MCP servers and 5,571 tools from the Smithery registry (covering 204 common apps like Gmail/GitHub/Slack), encapsulated via the MCP protocol with three-stage execution validation. At runtime, it provides `search_tools(query, k)` for agent self-retrieval via BGE-M3 + FAISS.
- $\mathcal{T}$ Task Distribution: Uses tool candidate sampling (round-robin by server to prevent single-app dominance) → generates long-horizon tasks via a check-then-revise process (including wild constraints) → applies fuzzy rewriting to prevent keyword shortcuts.
- $\mathcal{F}$ Transition: A State Controller middleware intercepts tool calls and injects three types of perturbations: tool-level (timeout/rate limit), state-level (delayed/overwritten results), and constraint-level (mid-task requirement changes).
- $\mathcal{R}$ Reward: Combines verifiable signals (schema/order/diversity) with judge-based signals (completeness/grounding/tradeoff via majority vote of three frontier LLMs).
- World Engine: Conditioned on tool response category cards (email/calendar/code-hosting, etc.), schemas, and session-level logs, the LLM generates realistic responses based on $(state, action)$, allowing for massive trajectory generation without live APIs.

Additionally, a Planner-Actor agent framework is introduced: the Planner decomposes tasks into a sub-goal graph, the Actor performs tool calls via ReAct, and the Planner verifies each step to provide feedback.

### Key Designs

1. **Unified Action Space + Tool Retrieval (Action Space)**:
    - **Function**: Converts 5,571 real MCP tools into a unified action pool that is searchable and executable.
    - **Mechanism**: Tools are collected via automated registry crawlers and manual supplements. Dedicated virtual accounts are configured for services requiring authentication. A three-stage validation (authenticated availability → successful invocation → usable responses) filters out broken tools. Finally, documentation (server identity, tool name, description, schema) is encoded by BGE-M3 and stored in FAISS, providing a `search_tools(query, k)` interface. Agents must retrieve and load tools as needed.
    - **Design Motivation**: Previous benchmarks either provided a small set of tools (unrealistic) or a large but non-functional set. Real-world scenarios require "comprehensive" and "live" tools, and agents should not use keyword shortcuts—making the retrieval interface the sole entry point.

2. **Check-then-revise Task Synthesis + Anti-shortcut (Task Distribution)**:
    - **Function**: Generates "long-horizon, multi-app, constraint-heavy, and anti-shortcut" tasks without manual labeling.
    - **Mechanism**: Samples 1–3 seed tools, uses their descriptions for queries, and recalls a larger candidate set via `search_tools`. Round-robin sampling by server ensures cross-app requirements. An LLM generates an initial query, followed by a check-then-revise loop: evaluating (i) tool coverage (effective activation of candidates) and (ii) constraint quality (diversity/coupling/long-range dependencies). Finally, fuzzy rewriting (e.g., "send the summary to the team" instead of "use slack_post_message") and the restricted `search_tools` interface force agents to decompose sub-queries themselves.
    - **Design Motivation**: Synthetic tasks are often short or rely on repetitive steps. This work uses "tool coverage" and "constraint density" metrics to force long-horizon tasks. Fuzzy rewriting addresses the implicit leakage of synthetic prompts during evaluation.

3. **Three-layer State Controller + World Engine (Transition Function)**:
    - **Function**: Enables the environment to simulate real failures (rate limits, state drift, requirement changes) and run large-scale training without live APIs.
    - **Mechanism**: The State Controller is a lightweight Python middleware within the agent runtime that intercepts MCP traffic and injects (a) tool-level, (b) state-level, and (c) constraint-level perturbations based on an "adversity budget." For fairness, total perturbation remains constant across models, but timing is randomized. The World Engine categorizes tools into "category cards" (email, calendar, etc.) containing typical response patterns and common failures. Using the schema, few-shot examples, and logs, the LLM generates responses. This enables zero-shot generalization to unseen tools in the same category. Compared to live execution on 50 tasks, it achieves Spearman $\rho=0.883$.
    - **Design Motivation**: Random noise is ineffective for learning; perturbations must be "reproducible and targeted." The World Engine is key to liberating agent training from API constraints, allowing for bulk trajectory generation and stress-testing on non-existent tools.

### Loss & Training
For training, 1,170 samples of "first-round effective actions" (abstract intent → specific tool retrieval/call) were filtered from 50 seed tasks. These were converted to ms-swift format for SFT using Hermes-style agent supervision. Comparisons were made against Toucan (119k) and ToolACE (11.3k).

## Key Experimental Results

### Main Results
Total scores (10-point scale + %) of 9 frontier LLMs in C-World Real Mode:

| Model | Overall | Completeness | Recovery% | Format% | Tool Calls |
|------|---------|-----|-----|-----|-----|
| gemini-3-pro-preview | **5.87** | 4.75 | 89.0 | 53.9 | 47.9 |
| claude-opus-4.5 | 5.42 | 4.70 | 83.7 | 51.0 | 45.2 |
| deepseek-v3.2 | 4.97 | 4.00 | **90.6** | 39.5 | 21.7 |
| grok-4 | 4.78 | 3.80 | 89.0 | **68.3** | 27.4 |
| gpt-oss-120b | 4.66 | 3.42 | 72.7 | 35.8 | 14.4 |
| gpt-5.2 | 4.43 | 3.42 | 79.3 | 12.4 | 29.2 |
| qwen3-235b-a22b | 3.53 | 2.56 | 88.1 | 31.3 | 11.2 |
| gpt-4o-mini | 3.07 | 1.13 | 50.6 | 3.3 | 51.7 |

### Ablation Study

**Training Data Efficiency**
| Model / Training Data | Samples | BFCL | MCP-Universe |
|------|-----|------|------|
| Qwen2.5-7B base | – | 19.93% | 4.40% |
| + Toucan | 119k | 27.18% | 15.28% |
| + ToolACE | 11.3k | 27.06% | 2.23% |
| **+ Ours (C-World)** | **1.2k** | **28.58%** | **15.30%** |
| Qwen3-8B base | – | 18.32% | 6.35% |
| + Toucan | 119k | 27.39% | 6.67% |
| + ToolACE | 11.3k | 29.49% | 3.29% |
| **+ Ours (C-World)** | **1.2k** | **30.05%** | **8.86%** |

**Simulation Fidelity**
| Evaluation Mode | Spearman ρ |
|------|------|
| World Engine vs Real Exec | **0.883** |
| DeepSeek-V3.2 judge vs Human | 0.759 |
| GPT-5.1 judge vs Human | 0.733 |
| Human vs Human ceiling | 0.773 |

### Key Findings
- **Execution is the bottleneck**: All models scored between 7.7 and 8.6 in Goal Decomposition (planning), but Completeness varied from 1.13 to 4.75 (a 4x gap), indicating the issue lies in "doing," not "thinking."
- **Tool call count $\neq$ Success rate**: gpt-4o-mini made the most calls (51.7) but performed worst due to cyclic repetitions. Gemini-3-Pro used similar calls (47.9) to achieve the highest score, showing that high activity must be paired with high reasoning.
- **Constraint following is a major failure mode**: Format compliance varied drastically (3.3% to 68.3%), far more than the gap in tool invocation success.
- **World Engine effectively replaces live APIs**: Model rankings showed a Spearman $\rho$ of 0.883, and the judge ensemble approached the human ceiling of 0.773.
- **Significant data efficiency**: 1,170 C-World trajectories outperformed 119k Toucan samples, proving that "long-horizon, constraint-dense, and perturbed" trajectories are more valuable than massive happy-path datasets.

## Highlights & Insights
- Applying the $(\mathcal{A}, \mathcal{T}, \mathcal{F}, \mathcal{R})$ formalism to LLM agents is a brilliant "return to basics." While previous papers treat environments as black boxes, this explicit decomposition allows for independent modification and evaluation of each component.
- The "category card" design in the World Engine is crucial for zero-shot generalization to unseen tools—it is cheaper than per-tool demonstrations and allows plausible response distributions for non-existent tools, facilitating stress tests.
- The "fixed adversity budget + randomized timing" is an elegant fairness design, avoiding bias caused by different models encountering different difficulty levels.
- The "Planner-Actor same-model + Planner per-step verification" is a lightweight solution for long-horizon tasks. Table 3 shows it helped Gemini-3-Pro climb from 5th to 1st place.
- The conclusion that 1,170 samples > 119k samples suggests that core tool learning requires "few but difficult" trajectories rather than "many but easy" ones.

## Limitations & Future Work
- The evaluation set consists of only 50 seed scenarios + 254 LongSeal questions, which does not sufficiently cover the combinatorial space of tools, servers, and constraints.
- Analysis focuses on frontier and large open-source models; sub-10B models are limited to four data points in the appendix. Whether the C-World framework can stably guide the training of small models remains to be verified.
- Although Spearman $\rho$ is high, the World Engine has systematic biases in absolute scores (e.g., DeepSeek-V3.2 synthetic Pass% is 38.9%, much lower than the real 87.5%), which may introduce bias during training supervision.
- Future Work: (1) Evolve task synthesis for automated integration of new servers/failure modes; (2) Implement confidence estimation for World Engine outputs; (3) Include more sub-10B models in the primary analysis.

## Related Work & Insights
- **vs AgentBench / WebArena**: These are static benchmarks with fewer tools (18–600 vs 5,571). C-World’s environment-level synthesis, state perturbation, and fuzzy instructions create a truly "evolvable" environment.
- **vs ToolBench / Toucan**: These focus on task synthesis but use static tasks without transition perturbations. This work’s 1.2k > 119k result challenges the "more data is better" paradigm.
- **vs StableToolBench / MCP-Bench**: These also focus on MCP tool calling, but this is the first work to formalize and demonstrate a tool response simulator (World Engine) as a viable alternative to live APIs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ An "environment creation system" rather than a "benchmark" represents a paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes 9 LLMs, dual modes, SFT comparisons, and human alignment, though the number of seed tasks (50) is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Clear formalization with independent sections for each component; excellent use of trajectories to explain perturbations.
- Value: ⭐⭐⭐⭐⭐ 5,571 tools + dual modes + open-source code makes this a highly practical infrastructure for computer-use agent training and evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] WebOperator: Action-Aware Tree Search for Autonomous Agents in Web Environment](../../ICLR2026/llm_nlp/weboperator_action-aware_tree_search_for_autonomous_agents_in_web_environment.md)
- [\[ICML 2026\] Express Your Doubts: Probabilistic World Modeling Should Not Be Based on Token logprobs](../../ICML2026/llm_nlp/express_your_doubts_--_probabilistic_world_modeling_should_not_be_based_on_token.md)
- [\[ICML 2026\] In-Context Routing (ICR): Train Once, Use Everywhere via Attention-Level Implicit ICL](../../ICML2026/llm_nlp/train_once_reuse_everywhere_generalizable_implicit_in-context_learning_by_routin.md)
- [\[NeurIPS 2025\] EnCompass: Enhancing Agent Programming with Search Over Program Execution Paths](../../NeurIPS2025/llm_nlp/encompass_enhancing_agent_programming_with_search_over_program_execution_paths.md)
- [\[ICLR 2026\] How Far Are LLMs from Professional Poker Players? Revisiting Game-Theoretic Reasoning with Agentic Tool Use](../../ICLR2026/llm_nlp/how_far_are_llms_from_professional_poker_players_revisiting_game-theoretic_reaso.md)

</div>

<!-- RELATED:END -->
