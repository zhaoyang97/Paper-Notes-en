---
title: >-
  [Paper Note] Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection
description: >-
  [NeurIPS 2025][LLM Agent][agent composition] This paper formalizes the agent component selection problem as an online knapsack problem and proposes the Composer Agent framework, which evaluates true component capabilities via sandbox testing (rather than static semantic retrieval) and dynamically selects optimal component combinations under budget constraints using the ZCL online algorithm. The approach achieves up to a 31.6% improvement in single-agent tool selection success rate, and boosts multi-agent sub-agent selection success rate from 37% to 87%.
tags:
  - NeurIPS 2025
  - LLM Agent
  - agent composition
  - knapsack problem
  - component selection
  - online optimization
  - sandbox testing
  - multi-agent
date: 2026-05-08
content_hash: 2e33cd4346375cb7
---

# Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection

**Conference**: NeurIPS 2025
**arXiv**: [2510.16499](https://arxiv.org/abs/2510.16499)
**Authors**: Michelle Yuan, Khushbu Pahwa, Shuaichen Chang, Mustafa Kaba, Jiarong Jiang, Xiaofei Ma, Yi Zhang, Monica Sunkara (AWS Agentic AI)
**Code**: None
**Area**: LLM Agent / Component Selection / Online Optimization
**Keywords**: agent composition, knapsack problem, component selection, online optimization, sandbox testing, multi-agent

## TL;DR
This paper formalizes the agent component selection problem as an online knapsack problem and proposes the Composer Agent framework, which evaluates true component capabilities via sandbox testing (rather than static semantic retrieval) and dynamically selects optimal component combinations under budget constraints using the ZCL online algorithm. The approach achieves up to a 31.6% improvement in single-agent tool selection success rate, and boosts multi-agent sub-agent selection success rate from 37% to 87%.

## Background & Motivation
**Paradox of Choice**: The proliferation of tools, APIs, and sub-agents in the agent ecosystem confronts developers with a combinatorially explosive configuration space, rendering manual curation and metadata-based retrieval inadequate.

**Three Core Limitations of Semantic Retrieval**: (a) Component descriptions frequently misrepresent actual capabilities — for example, the description of `get_article_content` closely matches the "Web Browsing" skill, yet it is not a web search tool; (b) cost-performance trade-offs are ignored; (c) static architectures fail as requirements evolve.

**Component Dynamism**: Tool API updates and model version iterations cause static metadata to become stale, leaving "outdated" components in the library.

**Redundancy Waste**: Pure semantic retrieval tends to select functionally overlapping components (e.g., Offline Knapsack selects 10 tools for GAIA, whereas Online requires only 4 to achieve superior performance).

**Lack of Theoretical Framework**: Prior to this work, agent component selection lacked a formal problem definition and provably guaranteed algorithms.

## Problem Definition
This paper defines agent component selection as a sub-problem of ADAS (Automated Design of Agentic Systems): given a task $\tau$ (with description $x$), a component library $A$ (each component $a_i$ has cost $c_i$ and description $d_i$), and budget $B$, the objective is to find an optimal subset $S^* \subseteq A$ that maximizes the success rate $p_\tau(S)$ subject to the total cost not exceeding the budget:

$$S^* = \arg\max_{S \subseteq A} p_\tau(S) \quad \text{s.t.} \quad \sum_{a_i \in S} c_i \leq B$$

This formalization directly corresponds to the classical knapsack problem, but with three key distinctions: (1) true success rates are unknown and must be estimated via iterative testing → online knapsack; (2) components may exhibit non-additive interactions (synergy or conflict); (3) the component library evolves dynamically and must support incremental updates.

## Method: Four Composer Agent Variants

### 1. Identity Composer (Baseline)
Returns all components directly: $f(A, \tau, B) = A$. All 120 tools are loaded without any selection. Cost is extremely high ($398), and in multi-agent settings the Supervisor fails to delegate effectively, causing performance collapse (only 7% success rate in the Travel domain).

### 2. Retrieval Composer
First uses an LLM to parse the task description into up to 6 skills (each with a name, description, and importance weight), then retrieves the top-1 component per skill using an embedding model (BGE-Large-EN-v1.5). **Core Limitation**: Pure semantic matching — in the GAIA experiment, the "Web Browsing" skill was matched to `get_article_content` rather than the actual `web_search` tool, resulting in a success rate of only 0.19.

### 3. Offline Knapsack Composer
Adds budget constraints on top of retrieval: retrieves top-$K$ candidates, uses semantic similarity scores as values $v_i^{\text{OFF}} = \sum \text{SIM}(a_i, m)$, and solves a multiple-choice knapsack problem via linear programming while satisfying both budget and skill coverage constraints. **Problem**: Values remain based on semantic similarity rather than true capability, making it prone to selecting redundant tools (in the GAIA experiment, 10 tools were selected, including many useless ones).

### 4. Online Knapsack Composer (Core Contribution)
Key innovation: evaluates true component capabilities through sandbox testing prior to selection. Full pipeline:
- **Skill and Test Query Generation**: The LLM parses the task description into a skill list and generates 2–3 single-step test questions with reference plans per skill.
- **Candidate Retrieval**: The embedding model retrieves top-$K$ candidate components per skill.
- **Sandbox Testing**: Each candidate component is run on test queries in a sandbox; an LLM-as-Judge scores outcomes on three levels: helpful (1) / not helpful (0) / broken (−1).
- **Value Computation**: $v_i = \sum(\text{weight of uncovered skills} \times \text{test pass indicator})$; only contributions to skills not yet covered by already-selected components are counted.
- **ZCL Online Decision**: Computes the value-to-cost ratio $\rho_i = v_i / c_i$, compares it against a dynamic threshold $\Psi$, and admits a component only when $\rho_i \geq \Psi$.

**ZCL Threshold Formula**: $\Psi = (Ue/L)^z \cdot L/e$, where $L = 1/\max(c_a)$, $U = \sum w_j / \min(c_a)$, and $z$ is the fraction of budget consumed. The threshold rises exponentially as the budget is depleted — naturally enforcing cost control in later stages. The theoretical competitive ratio is $\ln(U/L) + 1$.

**Runtime Optimizations**: (a) Broken components are permanently flagged and skipped to avoid redundant testing; (b) already-covered skills are excluded from evaluation for new candidates, reducing redundant testing and preventing selection of functionally overlapping components.

### Optional: AvaTaR Prompt Optimization
Trajectories generated during sandbox testing are reused as feedback signals to optimize the agent system prompt via the AvaTaR framework. The optimized prompt produces more structured tool-use guidance: task decomposition, tool-to-subtask mapping, query reformulation, and error recovery strategies — at no additional data collection cost.

## Experiments

### Single-Agent Experiments (Tool Selection, Claude 3.5 Sonnet)

| Method | GAIA | SimpleQA | MedQA | # Tools | Cost |
|--------|------|----------|-------|---------|------|
| Identity (all 120 tools) | 0.47 | 0.80 | 0.92 | 122 | $398 |
| Retrieval | 0.19 | 0.24 | 0.87 | 4–6 | $12–23 |
| Offline Knapsack ($30) | 0.41 | 0.88 | 0.91 | 10 | $30 |
| Online Knapsack ($30) | 0.47 | 0.92 | 0.93 | 2–4 | $6–12 |
| Online Knapsack + AvaTaR ($30) | **0.47** | **0.92** | **0.93** | 2–4 | $6–14 |

- Online Knapsack achieves parity with Identity ($398) at a cost of $12, yielding a Pareto-optimal outcome.
- Under a $10 budget, Online Knapsack + AvaTaR improves SimpleQA performance from 0.24 to **0.82**, demonstrating the substantial impact of prompt optimization.

### Multi-Agent Experiments (Sub-Agent Selection, Library of 117 Agents)

| Method | Travel GSR | Mortgage GSR | Budget |
|--------|------------|--------------|--------|
| Identity (all 117 agents) | 0.07 | 0.07 | $117 |
| Retrieval | 0.23 | 0.37 | $5–6 |
| Offline Knapsack ($6) | 0.17 | 0.70 | $3 |
| Online Knapsack ($6) | **0.40** | **0.87** | $5 |

- Identity completely fails in the multi-agent setting (117 agents overwhelm the Supervisor's ability to delegate effectively).
- Retrieval erroneously selects semantically similar "distractor" agents that lack the necessary tools.
- Online Knapsack identifies tool-less impostor agents via sandbox testing, achieving a success rate of **87%** in the Mortgage domain.

### Cross-Model Validation
Consistent advantages are observed across Claude 3.5 Haiku, Claude 3.7 Sonnet, Llama 4 Maverick/Scout, Qwen 2.5 72B, and Llama 3.3 70B: Online Knapsack consistently outperforms both Retrieval and Offline baselines. Variance across multiple runs is negligible.

## Key Case Studies
**GAIA Tool Selection Comparison** (Table 2):
- Retrieval selects `pub_med`, `read_file`, `wolfram_alpha`, and 3 others — **no web search tool is included**.
- Offline Knapsack selects `web_search_free` plus 9 other tools (with substantial redundancy), exhausting the $30 budget.
- Online Knapsack selects only `web_search_paid`, `arxiv`, and `wikipedia` — 3 tools totaling $12 — precisely covering core requirements.

**AvaTaR Optimization Case Study** (SimpleQA: "When did Reinhold Rudenberg retire?"):
- Before optimization: 4 tool calls; `web_search_free` is rate-limited twice; the agent ultimately gives up.
- After optimization: 3 tool calls; strategically sequences `wikipedia` → `semanticscholar` → `web_search_paid`; correctly answers "1952."

## Highlights & Insights
- **Theoretical Contribution**: The first work to formalize agent component selection as a knapsack problem; the ZCL algorithm provides a theoretical competitive ratio of $\ln(U/L) + 1$.
- **Sandbox Testing Paradigm**: Evaluating true capabilities by actually executing candidate components fundamentally resolves the description-capability mismatch inherent in semantic retrieval.
- **Unified Framework**: Single-agent tool selection and multi-agent team construction are handled under the same Composer Agent framework, requiring only a swap of the component library.
- **Engineering Practicality**: The 120-tool / 117-agent scale closely mirrors real deployment scenarios; cost is reduced from $398 to $12 while maintaining equivalent performance.

## Limitations & Future Work
1. **Sandbox Latency**: The testing process requires 10–30 minutes depending on budget and number of candidates, making it unsuitable for real-time scenarios.
2. **Task Description Requirements**: Clear task descriptions must be provided by developers; support for ambiguous or exploratory requirements is limited.
3. **Independent Evaluation Assumption**: Components are tested individually; synergistic or conflicting interactions between components are acknowledged in the problem definition but not addressed.
4. **Prompt Optimization Instability**: AvaTaR exhibits performance regression in certain settings (e.g., MedQA under a $10 budget).
5. **Security Concerns**: Automated component selection may inadvertently introduce malicious tools; no security review mechanism is provided.
6. **Lack of Algorithmic Comparison**: No comparison is made against alternative optimization paradigms such as greedy methods or MDPs.

## Rating
- Novelty: ⭐⭐⭐⭐ — Modeling agent component selection as a knapsack problem is natural and well-motivated; replacing semantic matching with sandbox testing is the core innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 5 datasets × 7 models × multiple budget levels, with Pareto analysis and multi-run variance validation.
- Writing Quality: ⭐⭐⭐⭐ — The four Composer variants are presented in a clear progression; the logical chain from problem formalization to algorithm design is complete.
- Value: ⭐⭐⭐⭐ — Provides direct practical guidance for tool and sub-agent selection in agent system engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Benchmarking Agentic Systems in Automated Scientific Information Extraction with ChemX](benchmarking_agentic_systems_in_automated_scientific_information_extraction_with.md)
- [\[AAAI 2026\] A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses](../../AAAI2026/llm_agent/a_multi-agent_conversational_bandit_approach_to_online_evaluation_and_selection_.md)
- [\[NeurIPS 2025\] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)
- [\[NeurIPS 2025\] A-MEM: Agentic Memory for LLM Agents](a-mem_agentic_memory_for_llm_agents.md)
- [\[NeurIPS 2025\] Automated Multi-Agent Workflows for RTL Design](automated_multi-agent_workflows_for_rtl_design.md)

</div>

<!-- RELATED:END -->
