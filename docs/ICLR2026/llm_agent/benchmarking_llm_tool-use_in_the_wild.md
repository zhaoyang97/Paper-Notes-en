---
title: >-
  [Paper Note] Benchmarking LLM Tool-Use in the Wild
description: >-
  [ICLR 2026][LLM Agent][Paper Note] WildToolBench extracts three characteristics of "wild" dialogues—composite tasks, hidden intentions, and instruction switching—from real user logs to construct a multi-turn, multi-step tool-use benchmark comprising 1024 tasks across 256 scenarios. Evaluations of 57 mainstream LLMs reveal that no model exceeds 15% sessi
tags:
  - ICLR 2026
  - LLM Agent
date: 2026-05-08
content_hash: a566d57bfc3d5f14
---
# Benchmarking LLM Tool-Use in the Wild

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=yz7fL5vfpn](https://openreview.net/forum?id=yz7fL5vfpn)  
**Code**: https://github.com/yupeijei1997/WildToolBench  
**Area**: LLM Evaluation / Agent Tool-Use  
**Keywords**: Tool Calling, Multi-turn Dialogue, Agent Evaluation, Real-world User Behavior, Tool Orchestration

## TL;DR
WildToolBench extracts three characteristics of "wild" dialogues—composite tasks, hidden intentions, and instruction switching—from real user logs to construct a multi-turn, multi-step tool-use benchmark comprising 1024 tasks across 256 scenarios. Evaluations of 57 mainstream LLMs reveal that no model exceeds 15% session accuracy, indicating that current agentic capabilities are significantly weaker than leaderboard inflated scores suggest.

## Background & Motivation

**Background**: Mainstream tool-use benchmarks adopt a "multi-turn, multi-step" paradigm—where the LLM acts as an assistant engaged in multi-turn dialogues to complete coherent tasks, often requiring multiple tool calls per task. Benchmarks have evolved in difficulty from single-step Q&A styles (T-EVAL, UltraTool) to multi-turn sequential calling (BFCL-v3), and recently to using LLM-as-User simulations ($\tau$-Bench / $\tau^2$-Bench).

**Limitations of Prior Work**: These benchmarks are overly idealized. While BFCL-v3 supports multiple turns, tasks are often independent and identically distributed (i.i.d.) with complete and explicit intentions/information, which differs from real user behavior. $\tau$-Bench uses LLM-simulated users, but these "users" are too perfect, making tasks easier and leading to unstable evaluation results due to the simulation process itself. In short, high leaderboard scores do not guarantee performance in real-world scenarios.

**Key Challenge**: Interactions between real users and LLMs are inherently "wild"—trivial, messy, and flexible. The difficulty lies not in artificially complex tasks, but in the simple yet authentic nature of user behavior. The authors identify three patterns from large-scale real user logs: ① Users combine multiple simple needs into a single instruction (**Composite Tasks**), requiring orchestration rather than simple serial execution; ② User intentions are implicit and scattered across multiple turns (**Hidden Intentions**), requiring context-based inference; ③ Users naturally switch between instruction types, such as giving tasks, asking follow-ups, explaining, or chatting (**Instruction Switching**), forcing the LLM to adjust its strategy in real-time.

**Goal**: To build a tool-use benchmark that truly reflects the distribution of real user behavior by explicitly encoding the three "wild" characteristics into tasks and designing metrics that distinguish fine-grained model capabilities.

**Key Insight**: Instead of further increasing task complexity, the benchmark returns to the "compositeness, ambiguity, and variability" of real behavior. By using real-log-driven data and human annotation, the authors create a benchmark that appears commonplace to users but is highly challenging for LLMs.

## Method

### Overall Architecture

WildToolBench formalizes user-LLM interaction as a multi-turn dialogue $D = \{u_1, a_1, u_2, a_2, \dots, u_N, a_N\}$, where $u_i$ is the $i$-th user message and $a_i$ is the LLM response. Within these $N$ turns, $M$ user tasks $\{g_1, \dots, g_M\}$ are distributed. For each message, the LLM must identify task presence, categorize it, and decide on a strategy. If tools are needed, the LLM enters a multi-step execution sequence $T^j = \{a^T_1, e_1, a^T_2, e_2, \dots, a^T_S, e_S\}$, where $a^T$ is a tool call and $e$ is environmental feedback; the information is then summarized into $a_i$. From the LLM's perspective, the dialogue is a Markov Decision Process (MDP) where states include the full history and actions are token sequences forming strategies.

The benchmark categorizes tasks into four types based on the number of tool steps $S$: chitchat $g_{\text{chat}}$ ($S=0$), clarification $g_{\text{clarify}}$ ($S=0$), single-tool tasks $g_{\text{single}}$ ($S=1$), and multi-step tasks $g_{\text{multi}}$ ($S>1$). The three "wild" characteristics map to this framework: composite tasks turn $T$ into a tree rather than a chain, hidden intentions require mining latent context, and instruction switching means every message may require a strategy change.

Data construction follows a three-step pipeline: "Real logs → Multi-agent simulation → Human annotation," resulting in 256 scenarios and 1024 tasks. Each scenario consists of a multi-turn dialogue containing 4 tasks.

### Key Designs

**1. Composite Tasks & Enumerate–Match–Score Orchestration: Measuring Efficient Tool Topologies**

Real instructions often combine simple needs (e.g., "Search for popular movies → Research each → Generate slides"). This requires the LLM to recognize dependencies and construct efficient topologies to minimize Time to First Token (TTFT) by parallelizing where possible. To evaluate this, the authors use an **enumerate–match–score** approach. **Enumerate**: Human annotators label dependencies, then a depth-first topological sort enumerates all valid execution paths, forming a set of decision trees. **Match**: Each tool call from the LLM is matched incrementally against these trees. **Score**: Once a path terminates, it is compared against the minimum depth of all enumerated trees to calculate the **OP Rate (Optimal Path Rate)**. The **AP Rate (Accomplish Progress Rate)** measures the percentage of successful nodes within the valid set. Unlike WorfBench, which relies on single-path similarity, this method accounts for all valid parallel possibilities.

**2. Three Constructions of Hidden Intentions: Testing Contextual Recovery**

Approximately 80% of users in sequential tasks modify or omit context. WildToolBench constructs three types of such tasks: **Partial Information** (current message $u_i$ only contains a subset of required info, with the rest in history $\{u_1, \dots, a_{i-1}\}$); **Coreferential Reference** (using pronouns or elliptical expressions referring to entities in previous turns); and **Long-Range Dependency** (similar to partial info, but the missing data is in a distant turn where $i - j > 2$). These test context-based inference with increasing difficulty.

**3. Instruction Switching & Strategy Adaptation: Real-time Strategy Changes**

Users treat interactions as natural conversations, breaking tasks with follow-ups, explanations, or chitchat. This implies multiple "instruction switches" within a dialogue. The benchmark defines a switch as a change in task type between adjacent tasks. By carefully configuring the proportions and frequencies of $g_{\text{single}}, g_{\text{multi}}, g_{\text{chat}}, \dots$, the benchmark tests if LLMs can correctly choose between tool-calling, direct answers, and proactive clarification.

**4. Data Construction Pipeline: Human-in-the-Loop Quality Control**

To ensure realism and controllability, the pipeline involves: **Scenario Construction** (extracting seed patterns from real logs); **Task Construction** (selecting subsets from 1600+ cleaned APIs); and **Multi-agent Simulation** (using GPT-4o agents to generate initial trajectories). Critically, every step is human-verified to ensure the quality of composite tasks, contextualized intentions, and instruction switching.

### An Example

In a news-related dialogue: The user first asks for "BBC news from the day before yesterday" (Single-tool); then adds "yesterday ESPN"—this is **Partial Information**, where the LLM must infer the verb "search" from history. Then, "I want the metadata for one of them"—this is a **Coreferential Reference**, requiring the LLM to **clarify** which article is meant. Finally, "Get metadata for all remaining articles and find other articles by their authors"—this is **Long-Range Dependency + Hybrid Multi-tool**, referring back to both turn 1 and turn 2 while requiring parallel execution.

## Key Experimental Results

### Main Results

Evaluations of 57 mainstream LLMs show: **No model achieved over 15% session accuracy** (completing all 4 tasks in a dialogue), and most tasks had accuracies below 60%. Closed-source models generally outperformed open-source, and reasoning models outperformed non-reasoning counterparts.

| Model | Task Accuracy | Session Accuracy |
|------|-----------|---------------|
| Gemini-2.0-Thinking | 61.04 | **14.45** |
| Gemini-2.5-Pro | 56.25 | 14.06 |
| Claude-4-Sonnet | 56.54 | 12.50 |
| o1 | 58.79 | 12.11 |
| GLM-4.5 (Strongest Open) | 56.05 | 12.11 |
| GPT-4o | 54.88 | 11.72 |
| Kimi-K2 | 53.71 | 10.55 |

The strongest open-source models (GLM-4.5, Kimi-K2) approach the top closed-source models. However, specialized tool-use models were significantly weaker than general-purpose models, showing limited generalization.

### Analysis

| Dimension | Key Findings |
|---------|---------|
| Orchestration | Max task accuracy is only 43.75%; hybrid (serial+parallel) tasks $g^{S+P}_{\text{multi}}$ drop to 25%. Peak OP Rate is only 42.74%. |
| Hidden Intent | Long-range dependency is the hardest, with **no model exceeding 50% accuracy**. This category shows the largest performance gap between models (17.3). |
| Instruction Switching | Task accuracy decreases as switching frequency increases, with drops up to 30% in some cases. |
| Error Analysis | Bottlenecks have shifted from syntax to semantic and logical reasoning; "Wrong Name / Missing Info" and "Redundant Call" are the most common errors. |

### Key Findings

- **Leaderboard Saturation is Deceptive**: While prior benchmarks are mostly saturated, WildToolBench resets model performance to below 15%, highlighting a massive robustness gap in real-world tool use.
- **Reasoning Improves Tool-Use**: Reasoning variants consistently outperform non-reasoning ones, **refuting the conclusion from Zhou et al. (2025)** and identifying the limitations of previous evaluations.
- **"Cautious" vs. "Aggressive" Failures**: Gemini-2.0-Thinking tends to refuse (24.56%) rather than take wrong actions (Wrong Name 8.02%), whereas Grok-4 rarely refuses (3.72%) but frequently selects incorrect tools (Wrong Name 24.07%).
- **The Root of Switching Failures—Self-conditioning**: Previous responses bias subsequent decisions (e.g., if a tool was used before, the model tends to use it again), and long histories dilute attention on the current task.

## Highlights & Insights

- **Correct Focus on "Wild" Behavior**: The authors correctly identified that the real hurdle is not "complex tasks" but "wild behavior"—compositeness, ambiguity, and variability.
- **Enumerate–Match–Score Framework**: This provides a reusable paradigm for orchestration evaluation. By exhausting all legal paths and using minimum depth for optimality, it offers fine-grained OP/AP metrics superior to basic similarity scores.
- **Actionable Error Taxonomy**: Categorizing errors into Action-level (Refusal, Wrong Name, etc.) and Parameter-level allows for structural diagnosis, making it highly valuable for model developers.
- **Reliability Gap**: Although GPT-5 (proxied/representative model) maintains a localized task accuracy, its session accuracy is extremely low (5.86%), exposing the "locally usable, globally unreliable" pain point of current AI agents.

## Limitations & Future Work

- **Annotation Dependency**: Ensuring quality and alignment with real distributions requires heavy human annotation, limiting scalability.
- **Simulation Bias**: Although human-verified, the initial trajectories generated by GPT-4o might retain its behavioral biases.
- **Future Directions**: The authors are exploring combining human rubrics with automated synthetic pipelines. Potential improvements include analyzing behavior differences between expert and novice users and incorporating cost/latency weights into the OP/AP metrics.

## Related Work & Insights

- **vs. BFCL-v3**: While BFCL-v3 introduced multi-turn evaluation, intentions remain explicit. In WildToolBench, Hidden Info and Instruction Transitions are present in 100% of cases compared to 15.7% and 39.7% in BFCL-v3.
- **vs. $\tau$-Bench**: $\tau$-Bench uses LLM-simulated users which can be too perfect. WildToolBench uses human-in-the-loop annotation to solidify real behavior patterns, making it both harder and more stable.
- **vs. WorfBench / TaskBench**: These focus on planning but use single-path similarity. WildToolBench's enumeration of all legal topologies is more precise.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Excellent shift from "complex tasks" to "wild behavior."
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive multidimensional breakdown of 57 models.
- Writing Quality: ⭐⭐⭐⭐ Clear framework; metrics are well-defined but require careful reading.
- Value: ⭐⭐⭐⭐⭐ Highly practical guidance for agent evaluation; effectively exposes the "reliability gap."

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ICLR 2026\] OSWorld-MCP: Benchmarking MCP Tool Invocation in Computer-Use Agents](osworld-mcp_benchmarking_mcp_tool_invocation_in_computer-use_agents.md)
- [\[ICLR 2026\] OrchestrationBench: LLM-Driven Agentic Planning and Tool Use in Multi-Domain Scenarios](orchestrationbench_llm-driven_agentic_planning_and_tool_use_in_multi-domain_scen.md)
- [\[ICLR 2026\] MCP-Bench: Benchmarking Tool-Using LLM Agents with Complex Real-World Tasks via MCP Servers](mcp-bench_benchmarking_tool-using_llm_agents_with_complex_real-world_tasks_via_m.md)
- [\[ICLR 2026\] VitaBench: Benchmarking LLM Agents with Versatile Interactive Tasks in Real-world Applications](vitabench_benchmarking_llm_agents_with_versatile_interactive_tasks_in_real-world.md)
- [\[ICLR 2026\] Empowering LLM Tool Invocation with Tool-call Reward Model](empowering_llm_tool_invocation_with_tool-call_reward_model.md)

</div>

<!-- RELATED:END -->
