---
title: >-
  [Paper Note] ExCyTIn-Bench: Evaluating LLM Agents on Cyber Threat Investigation
description: >-
  [ICML 2026][LLM Agent][Threat Investigation] This paper constructs ExCyTIn-Bench, the first benchmark for evaluating end-to-end "cyber threat investigation" by LLM Agents. From 57 security log tables of real Azure tenant…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Threat Investigation"
  - "SQL Agent"
  - "Bipartite Graph QA"
  - "Azure Sentinel"
  - "ReAct"
date: 2026-05-08
content_hash: 1e848580968253f9
---

# ExCyTIn-Bench: Evaluating LLM Agents on Cyber Threat Investigation

**Conference**: ICML 2026  
**arXiv**: [2507.14201](https://arxiv.org/abs/2507.14201)  
**Code**: https://github.com/microsoft/ExCyTIn-Bench (SecRL)  
**Area**: LLM Agent / Cybersecurity / Benchmark  
**Keywords**: Threat Investigation, SQL Agent, Bipartite Graph QA, Azure Sentinel, ReAct

## TL;DR
This paper constructs ExCyTIn-Bench, the first benchmark for evaluating end-to-end "cyber threat investigation" by LLM Agents. From 57 security log tables of real Azure tenants, 7,542 SQL QA pairs with evidence chains were automatically generated using an alert-entity bipartite graph. A MySQL environment is provided for Agents to answer by querying logs and tracking multi-hop evidence. Currently, the strongest model, Claude-Opus-4.5, achieves a reward of only 0.606.

## Background & Motivation

**Background**: Cloud attacks grew by 75% from 2022 to 2023. Traditional behavioral analysis, signature matching, and anomaly detection are increasingly insufficient. SOC (Security Operations Center) analysts must manually sift through dozens of heterogeneous log tables and perform multi-hop evidence chain reasoning to locate attacks. Since LLMs have demonstrated the ability to perform multi-step observation-reasoning-action in tasks like SWE-Bench and AutoGen, applying LLM Agents to threat investigation is a natural direction.

**Limitations of Prior Work**: Existing cyber-related benchmarks (CTIBench, SECURE, SecQA, CyBench, etc.) mostly evaluate "knowledge recall" or "text understanding"—e.g., determining MITRE tactics from a CTI report or answering multiple-choice questions. **No benchmark currently requires an Agent to actively query, pivot, and link evidence from a seed alert within an environment containing dozens of log tables.** Consequently, researchers cannot systematically compare the performance gap of different models/methods in end-to-end investigation.

**Key Challenge**: Real threat investigation is an **environment-interactive, long-horizon** task requiring **domain expertise**, while existing evaluation formats (MCQ/text understanding) naturally bypass these aspects. Filling this gap requires: (1) access to real multi-stage attack data with ground truth; (2) automatic, rather than manual, generation of large-scale questions with "unique deterministic answers" and "interpretable solution paths."

**Goal**: Construct (a) a security log environment based on real multi-stage attacks, (b) a large-scale QA set with ground-truth solution paths, and (c) an executable SQL sandbox to enable Agents to perform real "log-based investigation."

**Key Insight**: The authors observed that human SOC analysts essentially "walk" on an **implicit alert-entity bipartite graph**: starting from a seed alert, they pivot to adjacent alerts through shared entities (IPs, accounts, domains, etc.), and continue to spread. This graph naturally provides the shortest path from "problem source → answer destination," which can be directly used as a template for generating multi-hop questions for LLMs.

**Core Idea**: Use 8 multi-stage attack chains from real Azure tenants as data sources. Extract all alerts and entities to form a bipartite graph. Select two alerts as start/end targets, pick the **farthest entities** on the graph as the background context and answer, and then use an LLM to write the question. This ensures the questions are non-repetitive, the ground truth is unique, and the solution path is interpretable.

## Method

### Overall Architecture
ExCyTIn-Bench consists of three components:

1.  **Data Layer**: Collects 57 Sentinel log tables (EmailEvents, SecurityAlert, SecurityIncident, etc.) from the Azure tenant "Alpine Ski House" (a fictional company used by Microsoft for security product demos). These contain 8 independent multi-stage attack chains (e.g., Manatee Tempest ransomware, BEC account takeover, SAP financial manipulation), based on real historical attack playbooks, with alert counts ranging from 7 to 7,739 and time spans from 2 hours to 5 days.
2.  **Question Generation Layer**: Automatically generates 7,542 questions based on the alert-entity bipartite graph, with 589 selected for the test set.
3.  **Environment Layer**: Loads all logs into a MySQL Docker container. Agents submit SQL queries in a ReAct-style loop → receive tabular feedback → reason → provide a final answer. Evaluation uses a partial reward based on whether "intermediate nodes on the solution path" were found.

Inputs consist of a system prompt and a security question (with a seed alert and starting entity context). The output is the final answer string from the Agent after a maximum of $N$ interaction steps, which is then scored via ground-truth entity matching.

### Key Designs

1.  **Bipartite Graph Construction & Multi-hop Question Templates**:
    *   **Function**: Automatically constructs "question-answer-solution path" triplets from raw logs without manual authoring.
    *   **Mechanism**: For each incident, define $G=(U,V,E)$, where $U$ are alert nodes, $V$ are entity nodes, and edges $E$ connect alerts to the entities appearing in their "entities" column. Select two alert nodes $u_s, u_t$ as start/end. Call `GetFarthestEntities` to select $k=2$ entities furthest from $u_t$ as context $V_s$ and 1 entity furthest from $u_s$ as answer $v_e$. An LLM writes a question centered on $(u_s, V_s, v_e, u_t)$ asking "investigate from X to Y." The shortest path automatically becomes the "gold standard solution."
    *   **Design Motivation**: Asking an LLM to write questions by reading an entire incident results in generic questions lacking unique answers. The bipartite graph structures "difficulty = path length," "answer = terminal node," and "IoC = path nodes" simultaneously, making it interpretable, reusable, and scalable to new logs.

2.  **SQL Docker Sandbox + ReAct Interaction Environment**:
    *   **Function**: Places the LLM Agent in a read-only MySQL environment to "investigate" via query actions.
    *   **Mechanism**: Following InterCode, at each step, the Agent outputs a SQL statement (Action), and the environment returns query results (Observation) until the Agent calls `submit(answer)`. The environment supports wrappers like ReAct, Best-of-N, Self-Reflection, and Expel for comparing test-time scaling strategies.
    *   **Design Motivation**: Using SQL as the action space mimics real SOC analyst workflows (KQL/SQL log queries) and converts "natural language → verifiable action" into a deterministic interface, avoiding evaluation noise from open tool calls.

3.  **Path-based Progressive Reward**:
    *   **Function**: Instead of 0/1 scoring, provides partial rewards based on how many intermediate nodes on the shortest path the Agent discovers, distinguishing between "total failure," "partial discovery," and "final answer found."
    *   **Mechanism**: Given the shortest path $\mathcal{S}=[s_1,\dots,s_n]$, the system first checks the final answer. If correct, reward = 1. Otherwise, it backtracks to each intermediate node $s_i$, using `check_step` to determine if it appeared in the Agent's query history. Cumulative reward follows an exponential decay: $r=\sum d\cdot\gamma^{|\mathcal{S}|-i}$, with $\gamma=0.4$.
    *   **Design Motivation**: Threat investigations are rarely "one-shot." Binary scoring makes all Agents look similar. Decay-based partial rewards encourage multi-hop progress while penalizing rewards for shallow nodes that might be found by random walking.

### Loss & Training
This is a benchmark paper; **no models are trained**. It only evaluates existing LLMs. During question generation, GPT-4 class models generate QA and solutions based on prompt templates, followed by manual spot checks. During evaluation, all baselines run in a unified SQL environment and are scored using the partial reward described above.

## Key Experimental Results

### Main Results

Average reward across 8 incidents and 589 test questions (higher is better):

| Model | Average Reward | Remarks |
| :--- | :--- | :--- |
| Claude-Opus-4.5 | **0.606** | Current SOTA |
| GPT-4.1 | 0.338 | Best among large chat models |
| o4-mini | ~0.39 | Reasoning models show advantages |
| GPT-4o | 0.293 | General flagship |
| Llama4-17B-Maverick | 0.290 | Best open-source |
| GPT-4.1-mini | 0.271 | Small chat model |
| Llama4-17B-Scout | 0.262 | |
| o3-mini | 0.296 | Small reasoning model |
| o1-mini | 0.222 | |
| GPT-4o-mini | 0.192 | |
| GPT-4.1-nano | 0.136 | Weakest nano |
| Phi-4-14B | 0.085 | Small models basically fail |

Performance varies significantly across incidents: Incident 38 (Fileless Attack, 25 alerts) is relatively easier (most models 0.2–0.5), while Incident 166 (SAP Financial Manipulation, 88 alerts + many cross-table joins) and Incident 39 (475-alert human-operated intrusion) are the most difficult, with rewards dropping to 0.15–0.25.

### Ablation Study

| Configuration | Average Reward | Key Finding |
| :--- | :--- | :--- |
| ReAct (default) | Baseline | Standard multi-step reasoning |
| + Self-Reflection | Slight Increase | Error self-reflection is helpful but limited |
| + Best-of-N | Increase | Compute-performance trade-off is nearly linear with N |
| + Expel (from experience replay)| Increase | Offline experience extraction is effective |
| Direct Shortest Path | Large Increase | Validates that partial reward design is reasonable |

### Key Findings
*   **Top reasoning models are far from saturated**: Even Claude-Opus-4.5 only scores 0.606, meaning ~40% of questions never reach the ground truth. "Long-chain multi-hop security investigation" remains a real challenge for frontier models.
*   **Small models are nearly unusable**: Phi-4-14B scores 0.085 and GPT-4.1-nano 0.136. This indicates a high minimum threshold for model capacity in cyber agent tasks; distillation/small model strategies from general benchmarks cannot be directly applied.
*   **Incident difficulty $\neq$ alert volume**: Incident 55 has 7,739 alerts yet scores higher (GPT-4.1 gets 0.474), whereas Incident 166 has only 88 alerts but all models perform poorly. This suggests the true bottleneck is **cross-table joins + entity ambiguity**, not log throughput.
*   **Test-time scaling is effective but limited**: Best-of-N and Reflection provide stable gains, but neither matches the improvement from switching to a stronger base model. This implies the bottleneck for cyber agents is "domain knowledge + long-range planning" rather than insufficient sampling.

## Highlights & Insights
*   **Using bipartite graphs to automatically generate multi-hop questions** is a clever design. It shifts "question creation" from "experience-based" to "graph-based sampling + LLM rewriting," ensuring unique answers and quantifiable difficulty (path length). This paradigm can be transferred to any "entity-event" domain (medical diagnosis, financial audit, IT RCA).
*   **Path-decayed partial reward** is not a new concept, but grafting it onto a "log query action space" effectively differentiates models and prevents the benchmark from either saturating or giving zero scores to everyone—a useful trick for future cyber agent evaluations.
*   **Honest difficulty selection**: Using 8 **real historical attack** playbooks (Manatee Tempest, BEC, SAP intrusion, etc.) rather than synthetic attacks minimizes the gap between the benchmark and real SOC workflows. The balance between incident variety (8) and question volume (7,542) is well-maintained.

## Limitations & Future Work
*   Single data source: Limited to one Azure tenant and the Microsoft Sentinel ecosystem. Portability to AWS GuardDuty, Splunk, or Elastic has not been verified; log schemas vary significantly between vendors.
*   Action space limited to SQL queries: Real SOC work involves EDR, packet capture, and sandbox replay. Narrowing the action space to SQL may overestimate the actual threat hunting capabilities of "SQL-fluent" LLMs.
*   Uneven difficulty across 8 incidents (reward range 0.085–0.491); a few incidents account for most of the score variance. Future work requires scaling to dozens of incidents for stable comparison.
*   Bipartite-generated questions lean towards "find the last IoC" tasks; coverage of higher-level analysis like "attacker intent determination" or "APT attribution" is insufficient.

## Related Work & Insights
*   **vs CTIBench / SECURE / SecQA**: These are closed-book MCQs or text comprehension tasks testing "how much cyber knowledge the LLM remembers." ExCyTIn is an open-book interactive task testing "whether the LLM can dig up answers in logs," which is closer to real workflows.
*   **vs CyBench (CTF)**: CyBench is an attacker-perspective CTF challenge. ExCyTIn is a defender-perspective forensics investigation. Together, they form a "red-blue" dual perspective for cyber agent evaluation.
*   **vs InterCode**: This work adopts InterCode's SQL interaction design. The insight is: **Building a general environment shell and then populating it with domain data** is much easier than building an environment from scratch.

## Rating
*   Novelty: ⭐⭐⭐⭐ First cyber investigation agent benchmark + bipartite graph generation paradigm.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Covers 12+ models + 4 prompting strategies + 8 incidents.
*   Writing Quality: ⭐⭐⭐⭐ Clear motivation, clean three-part structure; appendix provides detailed prompts and SQL examples.
*   Value: ⭐⭐⭐⭐⭐ Fills a gap in cyber agent evaluation; tasks are far from saturated and will continue to challenge frontier models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics](bioagent_bench_an_ai_agent_evaluation_suite_for_bioinformatics.md)
- [\[ICML 2026\] MCP-Persona: Evaluating LLM Agent Capabilities in Real Personalized Applications via Environment Simulation](mcp-persona_benchmarking_llm_agents_on_real-world_personal_applications_via_envi.md)
- [\[ICLR 2026\] ZeroDayBench: Evaluating LLM Agents on Unseen Zero-Day Vulnerabilities for Cyberdefense](../../ICLR2026/llm_agent/zerodaybench_evaluating_llm_agents_on_unseen_zero-day_vulnerabilities_for_cyberd.md)
- [\[NeurIPS 2025\] EU-Agent-Bench: Measuring Illegal Behavior of LLM Agents Under EU Law](../../NeurIPS2025/llm_agent/eu-agent-bench_measuring_illegal_behavior_of_llm_agents_under_eu_law.md)
- [\[ICML 2026\] Hunt Instead of Wait: Evaluating Deep Data Research on Large Language Models](hunt_instead_of_wait_evaluating_deep_data_research_on_large_language_models.md)

</div>

<!-- RELATED:END -->
