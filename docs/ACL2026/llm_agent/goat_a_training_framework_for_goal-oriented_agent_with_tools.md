---
title: >-
  [Paper Note] GOAT: A Training Framework for Goal-Oriented Agent with Tools
description: >-
  [ACL 2026][LLM Agent][Goal-oriented] GOAT introduces a pipeline that automatically constructs "dependency graphs + call-first synthetic data" from API documentation. This allows small open-source models to learn how to d…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Goal-oriented"
  - "API calling"
  - "Synthetic data"
  - "call-first generation"
  - "Dependency graph"
date: 2026-05-08
content_hash: 5380c15a99157155
---

# GOAT: A Training Framework for Goal-Oriented Agent with Tools

**Conference**: ACL 2026  
**arXiv**: [2510.12218](https://arxiv.org/abs/2510.12218)  
**Code**: https://github.com/KU-MIIL/GOAT (Available)  
**Area**: LLM Agent / Tool Use  
**Keywords**: Goal-oriented, API calling, Synthetic data, call-first generation, Dependency graph

## TL;DR
GOAT introduces a pipeline that automatically constructs "dependency graphs + call-first synthetic data" from API documentation. This allows small open-source models to learn how to decompose high-level goals into sequences of interdependent API calls without human annotation. GOAT pushes open-source models to SOTA performance on RestBench, API-Bank, and the self-constructed GOATBench, even surpassing closed-source models in specific scenarios.

## Background & Motivation

**Background**: LLMs acting as agents to invoke external tools/APIs have become a mainstream paradigm. However, existing tool-learning benchmarks mostly remain in simple settings such as "single-step API calls" or "instructions where every step is clearly defined" (e.g., ToolFormer, Gorilla, ToolLLM, API-Bank).

**Limitations of Prior Work**: Real-world scenarios are **goal-oriented**—users only provide a high-level goal (e.g., "Find high-rated movies starring actors from 'The Dark Knight' and add them to my playlist"). The agent must autonomously decompose tasks, select APIs, and use the output of one API as the input for another. Open-source models achieve nearly 0% success in such settings, while GPT-4 only manages marginally better. This performance gap stems from a **lack of goal-oriented training data**: dependencies between APIs require expensive manual annotation, making them difficult to scale.

**Key Challenge**: Existing synthetic data pipelines generally adopt an **instruction-first** approach—letting the LLM generate a query first and then infer the API call sequence. Since this is the very capability being trained, it leads to **self-reinforcement bias**: the model only synthesizes "easy cases" it already knows how to solve, failing to produce high-difficulty training signals. Consequently, synthetic data is mostly useful for distillation rather than improving the complex reasoning of open-source models.

**Goal**: (1) Automatically synthesize goal-oriented training data capturing API dependencies without human annotation; (2) Enable open-source models to reach or exceed the level of closed-source models on in-domain real-world APIs.

**Key Insight**: The authors observe that agents are typically deployed on fixed sets of APIs with **existing documentation**. These documents implicitly encode input/output specifications sufficient for deriving inter-API dependencies. Furthermore, generating a natural language query from an "already executed call sequence" (inverse task) is more accessible for LLMs than "inferring calls from a query"—the former is abstract summarization, while the latter is planning and reasoning.

**Core Idea**: Use an LLM with a three-stage filter to transform API documentation into a reliable dependency graph. Sample connected subgraphs from this graph, **execute the APIs first, and then generate the query** (call-first), leveraging the LLM's summarization capability for reliable supervision in the inverse task.

## Method

### Overall Architecture
GOAT links two stages:
1. **API Dependency Graph Construction** (Sec 3.1.1): Parse API docs → Initialize a fully connected directed multigraph → Three-stage filtering (embedding similarity → LLM semantic judgment → Real execution verification) → Obtain a graph $G=(\mathcal{V}, \mathcal{E})$ retaining only feasible dependencies. An edge $(n_i, n_j, k)$ indicates that the output of $n_i$ can fill the $k$-th parameter of $n_j$.
2. **Goal-oriented Data Synthesis** (Sec 3.1.2): Sample connected subgraphs with $\le 4$ nodes from $G$ → Topological sort → Sequentially "fill parameters → Execute → Generate sub-query" → Summarize all sub-queries into a user query $u$ and a final response $r$.
3. **Training** (Sec 3.2): Instruction-tune the LLM using LoRA on synthetic data; simultaneously fine-tune an SBERT retriever on (query, API doc) pairs using InfoNCE. Concrete parameter values are masked during training to force the model to learn structure rather than memorization.

### Key Designs

1. **Three-stage Dependency Graph Filtering**:
    - **Function**: Filters truly executable dependencies from $|\mathcal{V}|^2 \times K$ candidate edges while controlling LLM costs.
    - **Mechanism**: First, apply cheap SBERT cosine similarity filtering (low threshold $\tau$ for high recall). Second, use a single LLM call to judge semantic compatibility and generate a "justification" for the edge. Finally, perform **execution-level verification via three real API calls**—the LLM generates parameters for $n_i$ to get output $o_i$, then extracts content from $o_i$ to fill the $k$-th parameter of $n_j$ for execution $c_j$. The edge is kept only if $c_j$ succeeds. Precision/recall transitions through the stages: 0.25/0.92 → 0.59/0.42 → 0.90/0.36, forming a "funnel" pattern.
    - **Design Motivation**: Pure embedding has high recall but low precision; pure LLM is too expensive; pure execution checks are even costlier. Funnel filtering reserves expensive checks for few surviving edges. **Real execution verification** is critical—descriptions alone might miss edges that seem semantically matched but have mismatched formats or units.

2. **Call-first Data Generation**:
    - **Function**: Flips the "query then call" paradigm to "call then query" to avoid self-reinforcement bias.
    - **Mechanism**: Given a topologically sorted API sequence $(n_{k_1}, \dots, n_{k_L})$, instantiate each sequentially. For the $\ell$-th API, if a parameter has a dependency on a predecessor $m < \ell$, extract the value from $o_{k_m}$; otherwise, let the LLM generate a reasonable value based on documentation. After execution yields $o_{k_\ell}$, generate a natural language sub-query $s_{k_\ell}$. Finally, summarize all sub-queries into a user query $u$ and generate a final response $r$ using all triplets $\{(s, c, o)\}$.
    - **Design Motivation**: Transitioning from query to call is "complex planning," whereas transitioning from (call, output) to query is "summarization/abstraction"—the latter is significantly easier. Table 5 proves this: under the same Llama2-13B prompting, call-first achieves 7.0% vs. instruction-first 5.0% on TMDB, and 28.1% vs. 17.5% on Spotify.

3. **Dependency-Driven Parameter Generation + Reasoning Guidance**:
    - **Function**: Helps the LLM identify which field to extract from a previous output when filling parameters.
    - **Mechanism**: The justification text produced in the second stage of graph construction ("why the output of $n_i$ fits the $k$-th parameter of $n_j$") is reused in the synthesis stage as guidance in the parameter filling prompt.
    - **Design Motivation**: API dependencies involve both data structure and semantic matching. API outputs are often nested JSON; without this guidance, LLMs frequently extract the wrong field. Reusing generated justifications provides a zero-cost, strong prior.

### Loss & Training
- **LLM**: Standard instruction fine-tuning (next-token CE). Target sequences cover plan, API call, and final response. LoRA $r=8$, $\alpha=16$, dropout 0.05, trained for 3 epochs on a single H100.
- **Retriever**: SBERT (all-MiniLM-L6-v2) + InfoNCE, where positive samples are (query, correct API doc) pairs.
- Parameter values are masked during training to force the model to learn which APIs to call and how to retrieve data rather than memorizing values.
- **Synthesis Scale**: 8,570 entries for RestGPT-TMDB, and only 108 entries for API-Bank (which still yielded significant gains).

## Key Experimental Results

### Main Results

**RestBench** (Success% / Correct Path% / Δ Solution Length):

| Backbone | Method | TMDB SR% | TMDB CP% | Spotify SR% | Spotify CP% |
|---|---|---|---|---|---|
| text-davinci-003 (Closed) | RestGPT | 75.0 | 79.0 | 72.7 | 74.5 |
| Llama2-13B | Baseline (zero-shot) | 0.0 | 0.0 | 3.5 | 7.0 |
| Llama2-13B | **GOAT FT** | **7.0** | **13.0** | **28.1** | **28.1** |
| Vicuna-13B | RestGPT | 9.0 | 15.0 | 12.7 | 20.6 |
| Vicuna-13B | **GOAT FT** | **17.0** | **14.0** | **29.8** | **33.3** |

**API-Bank** (Plan+Retrieve+Call Subset):

| Backbone | Method | Success% | CP% | Correctness% | ROUGE |
|---|---|---|---|---|---|
| GPT-4 (Closed) | API-Bank prompting | - | - | 70.00 | 0.4808 |
| Llama-7B | Baseline | 0.0 | 0.0 | 0.00 | 0.0048 |
| Llama-7B | **GOAT FT** | **38.0** | **42.0** | **42.22** | 0.3173 |

**GOATBench** (660 instances, 4 domains, Metrics: SA/IA/SR):

| Backbone | Method | Inter SA | Inter SR | Single SA | Single SR |
|---|---|---|---|---|---|
| GPT-4.1 (Closed) | Baseline | 22.7 | 27.4 | 45.4 | 51.3 |
| Llama2-7B | ReACT + ToolLLM FT | 13.1 | 1.0 | 28.7 | 3.9 |
| Llama2-7B | **ReACT + GOAT FT** | **33.2** | **1.9** | **41.0** | **7.2** |
| Llama3-8B | Baseline | 9.7 | 7.2 | 18.2 | 7.9 |
| Llama3-8B | **GOAT FT** | **59.4** | **12.3** | **69.1** | **16.5** |

### Ablation Study

| Configuration | TMDB SR% | Spotify SR% | Description |
|---|---|---|---|
| Instruction-first | 5.0 | 17.5 | Traditional paradigm |
| **Call-first (GOAT)** | **7.0** | **28.1** | ~60% relative gain on Spotify |

Dependency Graph Filtering Precision/Recall (Tab 6):

| Stage | Precision | Recall | Description |
|---|---|---|---|
| Embedding | 0.25 | 0.92 | High-recall coarse filter |
| LLM Semantics | 0.59 | 0.42 | Mid-precision, justification reuse |
| Real Execution | 0.90 | 0.36 | High-precision final filter |

### Key Findings
- **Call-first is the decisive factor**: The +10.6 absolute point gain on Spotify demonstrates that "execution then query" is far more learnable for open-source models than "query then planning."
- **Small data drives significant gains**: With only 108 synthetic samples for API-Bank, Llama-7B improved from 0% to 38% Success, indicating signal quality far outweighs scale.
- **Robust across prompting paradigms**: GOAT improves performance under both baseline and ReACT prompting, proving it enhances core model capability rather than just specific reasoning scripts.
- **Execution validation is indispensable**: Execution-level filtering raises precision from 0.59 to 0.90, ensuring remaining edges are "gold standards" that are both semantically logical and functional.
- **Outperforming ToolLLM FT**: When synthesizing data on RapidAPI, ToolLLM (instruction-first, parallel-API) is significantly surpassed by GOAT in goal-oriented settings, validating the importance of task alignment.

## Highlights & Insights
- **Cognitive Engineering via Direction Reversal**: The authors address a critical asymmetry—LLMs find writing queries much easier than planning API calls. By reversing the data synthesis direction, they allow the LLM to summarize its own real execution results. This "leveraging LLM strengths to feed LLM weaknesses" strategy is highly generalizable to tasks like code synthesis and math where verification/summarization is easier than generation.
- **Funnel-based LLM Budget Control**: The sequence of embedding → LLM semantics → execution verification creates a template for handling "combinatorial explosion + expensive LLMs," serving as a model for extracting high-quality samples from large candidate sets.
- **Zero-cost Prior Reuse**: Justifications generated to filter edges are reused as prompts for parameter extraction, ammortizing the cost of LLM calls while providing critical domain knowledge.
- **Efficacy of Low-Resource Domain Adaptation**: The significant improvement from only 108 samples challenges the notion that instruction tuning requires tens of thousands of examples, provided the data distribution aligns with the target task.

## Limitations & Future Work
- **Weak Generalization to Unseen APIs**: GOAT is an in-domain design. The dependency graph is tied to a specific API set; new APIs require rerunning the entire pipeline.
- **Dependency on Documentation Quality**: Vague documentation or missing parameter descriptions lead to noisy edges and lower synthetic data quality.
- **Execution Constraints**: Real execution validation is difficult for paid APIs, those requiring authentication, or those with strong side effects (e.g., database writes, transfers), potentially missing these dependencies.
- **Training Cost Details**: The total token count for calling Llama3-70B during the synthesis stage is not disclosed, complicating replication cost assessments.
- **Future Directions**: (i) Cross-domain dependency graph transfer based on parameter types/schemas; (ii) Sandboxed execution for write-access APIs; (iii) Semantic clustering on the dependency graph to ensure diverse coverage of rare dependency patterns.

## Related Work & Insights
- **vs. ToolLLM**: While ToolLLM also uses RapidAPI for synthesis, it follows an "instruction-first" approach for parallel multi-API tasks. GOAT's "call-first" approach for goal-oriented tasks significantly outperforms ToolLLM FT on GOATBench (33.2 vs 13.1 SA), demonstrating the value of task alignment.
- **vs. RestGPT**: RestGPT is a strong prompting-only baseline for GPT-4. However, on Vicuna-13B, GOAT FT surpasses RestGPT-Vicuna without altering the prompting, shifting the focus from prompt engineering to data synthesis engineering.
- **vs. ToolFlow / Magnet**: These use semantic heuristics for graphs without real execution verification and do not target goal-oriented queries. GOAT is the first **execution-grounded + goal-oriented** open-source solution.
- **vs. API-Bank**: API-Bank uses non-executable Python functions and instruction-first synthesis, reaching ~20% correctness on Llama-7B. GOAT's use of real executable APIs and call-first logic pushes the same backbone to 42%.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of call-first synthesis, funnel filtering, and justification reuse creates a highly effective framework for goal-oriented tasks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive testing across 3 benchmarks, multiple backbones, and prompting paradigms, though lacking multi-seed variance and cost reporting.
- Writing Quality: ⭐⭐⭐⭐ Clear pipeline visuals and convincing precision/recall analysis for the filtering stages.
- Value: ⭐⭐⭐⭐⭐ Provides a practical, deployable solution for industrial users wanting to build agents with open-source models without manual labeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] OctoTools: An Agentic Framework with Extensible Tools for Complex Reasoning](octotools_an_agentic_framework_with_extensible_tools_for_complex_reasoning.md)
- [\[ACL 2026\] CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution](coevolve_training_llm_agents_via_agent-data_mutual_evolution.md)
- [\[ACL 2026\] ChemAmp: Amplified Chemistry Tools via Composable Agents](chemamp_amplified_chemistry_tools_via_composable_agents.md)
- [\[NeurIPS 2025\] AgentChangeBench: A Multi-Dimensional Evaluation Framework for Goal-Shift Robustness](../../NeurIPS2025/llm_agent/agentchangebench_a_multi-dimensional_evaluation_framework_for_goal-shift_robustn.md)
- [\[ACL 2026\] Supplement Generation Training for Enhancing Agentic Task Performance](supplement_generation_training_for_enhancing_agentic_task_performance.md)

</div>

<!-- RELATED:END -->
