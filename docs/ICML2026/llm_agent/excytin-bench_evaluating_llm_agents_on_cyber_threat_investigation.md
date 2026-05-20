---
title: >-
  [Paper Note] ExCyTIn-Bench: Evaluating LLM Agents on Cyber Threat Investigation
description: >-
  [ICML 2026][LLM Agent][Threat Investigation] This paper introduces the first benchmark for evaluating LLM Agents in end-to-end "cyber threat investigation": ExCyTIn-Bench. From 57 real Azure tenant security log tables…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Threat Investigation"
  - "SQL Agent"
  - "Bipartite Graph QA"
  - "Azure Sentinel"
  - "ReAct"
date: 2026-05-08
content_hash: 2b65cf23155298cf
---

# ExCyTIn-Bench: Evaluating LLM Agents on Cyber Threat Investigation

**Conference**: ICML 2026  
**arXiv**: [2507.14201](https://arxiv.org/abs/2507.14201)  
**Code**: https://github.com/microsoft/ExCyTIn-Bench (SecRL)  
**Area**: LLM Agent / Cybersecurity / Benchmark  
**Keywords**: Threat Investigation, SQL Agent, Bipartite Graph QA, Azure Sentinel, ReAct

## TL;DR
This paper introduces the first benchmark for evaluating LLM Agents in end-to-end "cyber threat investigation": ExCyTIn-Bench. From 57 real Azure tenant security log tables, it automatically generates 7,542 SQL QA tasks with evidence chains using an alert-entity bipartite graph, and provides a MySQL environment for agents to answer by querying logs and multi-hop evidence tracing. The current best model, Claude-Opus-4.5, achieves only a 0.606 reward.

## Background & Motivation

**Background**: Cloud attacks increased by 75% from 2022 to 2023. Traditional behavior analysis, signature matching, and anomaly detection are increasingly ineffective. SOC (Security Operations Center) analysts must manually review dozens of heterogeneous log tables daily, performing multi-hop evidence chain reasoning to locate attacks. LLMs have demonstrated multi-step observation-reasoning-action capabilities in tasks like SWE-Bench and AutoGen, making LLM Agents a natural fit for threat investigation.

**Limitations of Prior Work**: Existing cyber benchmarks (CTIBench, SECURE, SecQA, CyBench, etc.) mostly assess "knowledge recall" or "text comprehension"—e.g., identifying MITRE tactics from CTI reports, solving CTF challenges, or answering multiple-choice questions. **No benchmark truly requires an agent to actively query, pivot, and chain evidence from a seed alert within an environment containing dozens of log tables.** This prevents systematic comparison of models/methods on end-to-end investigation.

**Key Challenge**: Real threat investigation is an **environment-interactive, long-horizon, and domain-expert** task, while current benchmarks (multiple-choice/text comprehension) inherently avoid these aspects. Addressing this gap requires: (1) obtaining real multi-stage attack data with ground-truth; (2) automatically generating large-scale questions with "unique answers" and "explainable solution paths" rather than manual curation.

**Goal**: Build (a) a security log environment based on real multi-stage attacks, (b) a large-scale QA set with ground-truth solution paths, and (c) an executable SQL sandbox, enabling agents to "investigate by querying logs."

**Key Insight**: The authors observe that human SOC analysts essentially "navigate" an **implicit alert-entity bipartite graph**: starting from a seed alert, they pivot to adjacent alerts via shared entities (IP, account, domain, etc.), and continue expanding. This graph naturally provides the shortest "source-to-target" paths, which can be directly used as templates for generating multi-hop questions for LLMs.

**Core Idea**: Using 8 real multi-stage attack chains from Azure tenants as data sources, extract all alerts and entities to form a bipartite graph, select two alerts as start/end, use the **farthest entity** on the graph as background context and answer, and have the LLM generate questions—ensuring uniqueness, ground-truth, and explainable solution paths.

## Method

### Overall Architecture
ExCyTIn-Bench consists of three components:

1. **Data Layer**: Collect 57 Sentinel log tables (e.g., EmailEvents, SecurityAlert, SecurityIncident) from the Azure tenant "Alpine Ski House" (a Microsoft demo company), injecting 8 independent multi-stage attack chains (e.g., Manatee Tempest ransomware, BEC account takeover, SAP financial manipulation). Each chain is based on real attack scenarios, with alert counts ranging from 7 to 7,739 and time spans from 2 hours to 5 days.
2. **Question Generation Layer**: Automatically generate 7,542 questions based on the alert-entity bipartite graph, with 589 as the test set.
3. **Environment Layer**: Load all logs into a MySQL Docker; agents interact in a ReAct-style loop by submitting SQL queries → receiving tabular feedback → reasoning → providing a final answer. Evaluation uses "partial reward" based on whether intermediate nodes on the solution path are found.

Input: a system prompt + a security question (with seed alert and initial entity context). Output: the agent's final answer string after up to NN interactions, scored by matching ground-truth entities.

### Key Designs

1. **Bipartite Graph Construction & Multi-hop Question Templates**:

    - **Function**: Automatically construct "question-answer-solution path" triplets from raw logs, eliminating manual question writing.
    - **Mechanism**: For each incident, define $G=(U,V,E)$, where $U$ is alert nodes, $V$ is entity nodes, and edges $E$ connect entities listed in the "entities" column of the alert table. Select two alert nodes $u_s, u_t$ as start/end, use `GetFarthestEntities` to pick $k=2$ entities farthest from $u_t$ from $u_s$ as background $V_s$, and one farthest entity $v_e$ from $u_t$ as the answer. The LLM then writes a question around $(u_s, V_s, v_e, u_t)$; the shortest path becomes the "standard solution."
    - **Design Motivation**: Having LLMs read the incident and write questions directly leads to generic, non-unique answers. The bipartite graph structures "difficulty = path length," "answer = target node," and "IoC = path nodes" in one step, making questions explainable, reusable, and extensible to new logs.

2. **SQL Docker Sandbox + ReAct Interactive Environment**:

    - **Function**: Place the LLM Agent in a read-only MySQL environment, allowing investigation via query actions.
    - **Mechanism**: Inspired by InterCode, at each step the agent outputs an SQL (action), the environment returns the query result (observation), until the agent submits `submit(answer)`. The environment supports ReAct, Best-of-N, Self-Reflection, Expel, and other wrappers for comparing test-time scaling strategies.
    - **Design Motivation**: Using SQL as the action space mirrors real SOC analyst workflows (KQL/SQL log queries) and compresses "natural language → verifiable action" into a deterministic interface, avoiding evaluation noise from open tool calls.

3. **Solution Path-based Progressive Reward**:

    - **Function**: Not a binary 0/1 score, but partial reward based on how many intermediate nodes on the shortest solution path the agent discovers, distinguishing "complete failure / partial progress / final answer found."
    - **Mechanism**: For the shortest solution path $\mathcal{S}=[s_1,\dots,s_n]$, first check if the final answer is correct (reward 1 if so); otherwise, backtrack and for each intermediate node $s_i$, use `check_step` to see if the agent's history includes it, accumulating reward with exponential decay $r=\sum d\cdot\gamma^{|\mathcal{S}|-i}$, $\gamma=0.4$.
    - **Design Motivation**: Threat investigation rarely achieves "one-shot success"; binary scoring makes all agents look similar. Decayed partial reward encourages multi-hop progress and avoids rewarding shallow nodes reached by random exploration.

### Loss & Training
This is a benchmark paper—**no models are trained**; only existing LLMs are evaluated. During question generation, GPT-4-like models generate QA + solutions via prompt templates, with manual spot checks. All baselines are evaluated in the same SQL environment, scored by the partial reward above.

## Key Experimental Results

### Main Results

Average reward (higher is better) on 8 incidents and 589 test questions:

| Model | Avg. Reward | Notes |
|-------|-------------|-------|
| Claude-Opus-4.5 | **0.606** | Current best |
| GPT-4.1 | 0.338 | Best large chat model |
| o4-mini | ~0.39 | Reasoning models have an edge |
| GPT-4o | 0.293 | General flagship |
| Llama4-17B-Maverick | 0.290 | Best open-source |
| GPT-4.1-mini | 0.271 | Small chat model |
| Llama4-17B-Scout | 0.262 | |
| o3-mini | 0.296 | Small reasoning model |
| o1-mini | 0.222 | |
| GPT-4o-mini | 0.192 | |
| GPT-4.1-nano | 0.136 | Weakest nano |
| Phi-4-14B | 0.085 | Small models barely work |

By incident, the gap is huge: incident 38 (Fileless Attack, only 25 alerts) is relatively easy (most models 0.2–0.5), while incident 166 (SAP Financial Manipulation, 88 alerts + cross-table) and incident 39 (475-alert human intrusion chain) are hardest, with most models scoring 0.15–0.25.

### Ablation Study

| Configuration | Avg. Reward | Key Findings |
|---------------|-------------|-------------|
| ReAct (default) | Baseline | Standard multi-step reasoning |
| + Self-Reflection | Slight increase | Error reflection helps, but limited |
| + Best-of-N | Increase | More compute yields near-linear gains |
| + Expel (from experience replay) | Increase | Offline experience extraction effective |
| Directly provide shortest path | Large increase | Validates partial reward design |

### Key Findings
- **Top reasoning models are far from saturated**: Even Claude-Opus-4.5 only achieves 0.606, missing ground-truth on ~40% of questions. This shows "long-chain, multi-hop security investigation" remains a real challenge for frontier models; the benchmark will not be quickly solved.
- **Small models are nearly unusable**: Phi-4-14B scores only 0.085, GPT-4.1-nano only 0.136; this indicates a high minimum capacity requirement for cyber agent tasks, and that distillation/small model approaches cannot simply borrow conclusions from general benchmarks.
- **Incident difficulty ≠ alert count**: Incident 55 has 7,739 alerts but high reward (GPT-4.1 gets 0.474), while incident 166 has only 88 alerts but all models perform poorly. The real challenge is **cross-table joins + entity ambiguity**, not log volume.
- **Test-time scaling is effective but limited**: Best-of-N / Reflection reliably improve scores, but not as much as switching to a stronger base model; this suggests the bottleneck for cyber agents is "domain knowledge + long-range planning," not "insufficient sampling."

## Highlights & Insights
- **Automatically generating multi-hop security questions via bipartite graphs** is a clever design: it shifts "question writing" from "experience-based" to "graph sampling + LLM rewriting," ensuring unique answers and quantifiable difficulty (shortest path length = difficulty). This paradigm can be transferred to any "entity-event" domain (medical diagnosis, financial audit, ops RCA).
- **Partial reward decaying along the solution path** is not new, but applying it to the "log query action space" directly differentiates models, preventing the benchmark from being saturated or all scoring zero, making it a reusable trick for future cyber agent evaluation.
- **Honest difficulty selection**: Using 8 **real historical attack** scenarios (Manatee Tempest, BEC, SAP intrusion, etc.) instead of synthetic attacks minimizes the gap with real SOC workflows. The number of incidents (8) and questions (7,542) are well balanced—covering multiple scenarios without being too large to run.

## Limitations & Future Work
- Data source is single: only one Azure tenant + Microsoft Sentinel ecosystem is used; transferability to AWS GuardDuty / Splunk / Elastic and other cloud/SIEM platforms is untested. Log schema differences across vendors are significant, so benchmark conclusions may not generalize.
- Evaluation only covers SQL query actions: Real SOC work also involves EDR, network packet capture, sandbox replay, etc. This benchmark narrows the action space to SQL, potentially overestimating the practical threat hunting ability of "SQL-capable LLMs."
- The 8 incidents have uneven difficulty (reward range 0.085–0.491), with a few incidents accounting for most score variance. More incidents are needed for stable model comparison in the future.
- Bipartite graph-generated questions tend to focus on "finding the last IoC," lacking coverage of higher-level analysis tasks like "inferring attacker intent" or "attribution to APT groups."

## Related Work & Insights
- **vs CTIBench / SECURE / SecQA**: These are all closed-book multiple-choice or text comprehension tasks, measuring "how much cyber knowledge the LLM remembers." ExCyTIn is an open-book interactive task, measuring "can the LLM dig out answers from logs," closer to real workflows.
- **vs CyBench (CTF)**: CyBench is attacker-side CTF challenges; ExCyTIn is defender-side forensics investigation. Together, they form a "red-blue dual perspective" cyber agent evaluation system.
- **vs InterCode**: This work directly reuses InterCode's SQL interactive environment design. The insight is—**first build a general environment shell, then inject domain-specific data**—which is much easier than building an environment from scratch.

## Rating
- Novelty: ⭐⭐⭐⭐ First cyber investigation agent benchmark + bipartite graph question generation paradigm; unique direction
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 12+ mainstream models + 4 prompting strategies + 8 incidents; sufficient scale
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, clean three-part benchmark structure; appendix provides detailed prompts and SQL examples
- Value: ⭐⭐⭐⭐⭐ Fills the gap in cyber agent evaluation; the task is far from saturated and can continue to challenge frontier models

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](../../ACL2026/llm_agent/silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)
- [\[ICLR 2026\] ZeroDayBench: Evaluating LLM Agents on Unseen Zero-Day Vulnerabilities for Cyberdefense](../../ICLR2026/llm_agent/zerodaybench_evaluating_llm_agents_on_unseen_zero-day_vulnerabilities_for_cyberd.md)
- [\[ACL 2025\] LegalAgentBench: Evaluating LLM Agents in Legal Domain](../../ACL2025/llm_agent/legalagentbench_evaluating_llm_agents_in_legal_domain.md)
- [\[ACL 2025\] DICE-Bench: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues](../../ACL2025/llm_agent/dice-bench_evaluating_the_tool-use_capabilities_of_large_language_models_in_mult.md)
- [\[NeurIPS 2025\] EU-Agent-Bench: Measuring Illegal Behavior of LLM Agents Under EU Law](../../NeurIPS2025/llm_agent/eu-agent-bench_measuring_illegal_behavior_of_llm_agents_under_eu_law.md)

</div>

<!-- RELATED:END -->
