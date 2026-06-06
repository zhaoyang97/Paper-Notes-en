---
title: >-
  [Paper Note] PExA: Parallel Exploration Agent for Complex Text-to-SQL
description: >-
  [ACL2026][Code Intelligence][Text-to-SQL] PExA reformulates complex Text-to-SQL as a parallel exploration problem of "generating and executing a set of semantic test cases for a natural language query." By employing thre…
tags:
  - "ACL2026"
  - "Code Intelligence"
  - "Text-to-SQL"
  - "LLM Agent"
  - "Parallel Exploration"
  - "Software Testing"
  - "Spider 2.0"
date: 2026-05-08
content_hash: c0dc54224a0f16b2
---

# PExA: Parallel Exploration Agent for Complex Text-to-SQL

**Conference**: ACL2026  
**arXiv**: [2604.22934](https://arxiv.org/abs/2604.22934)  
**Code**: Not public  
**Area**: LLM Agent / Text-to-SQL / Database Question Answering  
**Keywords**: Text-to-SQL, LLM Agent, Parallel Exploration, Software Testing, Spider 2.0

## TL;DR
PExA reformulates complex Text-to-SQL as a parallel exploration problem of "generating and executing a set of semantic test cases for a natural language query." By employing three sub-agents—Planner, Test Case Generator, and SQL Proposer—it improves execution accuracy on Spider 2.0 while maintaining latency close to strong baselines.

## Background & Motivation
**Background**: Text-to-SQL has evolved from early parsing models to LLM agents. Facing real-world complex database benchmarks like Spider 2.0, systems typically require schema linking, database compression, multi-step reasoning, execution feedback, and self-correction to handle large tables, multiple databases, nested types, and long SQL queries.

**Limitations of Prior Work**: High performance is often achieved at the cost of longer sequential reasoning or more tool calls. An agent that sequentially plans, retrieves schema, writes SQL, executes, and corrects may improve performance, but latency accumulates with each step. In interactive data analysis, users find it difficult to accept long wait times for every question.

**Key Challenge**: Complex queries indeed require exploring more database information, but exploration does not necessarily have to be serial. Existing methods treat the user query as a single long SQL generation task, which easily gets stuck in a single reasoning chain; if decomposed into parallelizable semantic requirements, evidence can be collected simultaneously.

**Goal**: The authors aim to improve the performance-latency Pareto frontier of Text-to-SQL: achieving higher execution accuracy on Spider 2.0 without linearly increasing reasoning steps.

**Key Insight**: The paper draws inspiration from software testing. A complex program is not proven correct all at once but is verified through a test suite covering different functional requirements. A complex NL query can similarly be viewed as a collection of semantic requirements. By generating a batch of simpler SQL "test cases" covering local semantics—such as filtering, joining, aggregation, and existence checks—the execution results can support the final SQL generation.

**Core Idea**: Transform Text-to-SQL from single-chain generation to test-coverage-based parallel exploration, using multiple executable test-case SQLs to probe the database simultaneously and aggregating evidence to generate the complete SQL.

## Method
The key to PExA is shifting from "thinking through the final SQL" to "verifying a set of small questions first." These small questions are not typical query decompositions because they can explore database information outside but relevant to the original query—such as which values exist in a field, whether a join returns non-empty results, or whether a filter condition is reasonable. The execution results of these test cases serve as the grounding context for the final SQL.

### Overall Architecture
The system consists of three sub-agents and two tools. The **Planner** receives the original NL question, generates a set of self-contained test cases, and decides whether to continue exploration or proceed to final generation. The **Test Case Generator** converts each NL test case into SQL, calls the **SQL Executor** to execute it, and corrects it based on error feedback. The **SQL Proposer** takes only the original question, test-case SQLs, and execution results—without the full database metadata—and synthesizes the final long SQL. The two tools are the **SQL Executor**, which returns compilation errors, empty results, or successful results, and the **Semantic Verifier**, which translates SQL back to NL and compares it with the original question to identify semantic deviations.

Parallelism is implemented in three places. The Planner generates multiple test plans in one forward pass; the Test Case Generator generates and executes independent test cases in parallel; and each test case generates multiple candidate SQLs at once, forming a single-step multi-path search. The final latency is approximately determined by the slowest branch rather than the sum of all branches.

### Key Designs
1.  **Test-coverage Query Modeling**:
    - **Function**: Decomposes a complex user question into several independently executable small SQL tests, using test coverage to approximate the original query semantics.
    - **Mechanism**: Test cases can cover local requirements like filters, joins, aggregations, existence checks, and value distributions, or probe database information surrounding the original query. Each test SQL should be self-contained, executable, and return evidence that assists in final synthesis.
    - **Design Motivation**: Ordinary sub-problem decomposition only restates explicit semantics from the original question, whereas test cases can actively explore implicit structures and intermediate results, making them better suited for real-world complex scenarios like Spider 2.0.

2.  **Tri-agent Specialization**:
    - **Function**: Deconstructs planning, local SQL execution, and final SQL synthesis to reduce the context burden on any single agent.
    - **Mechanism**: The Planner handles semantic requirements and process control; the Test Case Generator generates executable local SQL combined with lightweight schema linking and compressed schemas; the SQL Proposer aggregates test results to generate the final SQL, checked by the executor and semantic verifier.
    - **Design Motivation**: Errors in complex Text-to-SQL often stem from early planning stages or long-context reasoning. By specializing, each agent solves a narrower problem, and the final Proposer does not need to read the full database metadata.

3.  **Parallel Exploration and Single-step Multi-path Search**:
    - **Function**: Expands search width without linearly increasing wall time.
    - **Mechanism**: The planning stage generates multiple tests at once, the execution stage runs multiple test SQLs in parallel, and each test generates multiple candidate solutions in a single LLM call, with execution feedback used to quickly prune failed paths.
    - **Design Motivation**: Uncertainty in Text-to-SQL often arises from schema selection, value constraints, and join paths. Wide searching increases the probability of hitting the correct semantics, while parallelization prevents this width from translating entirely into latency.

### Loss & Training
PExA is an inference-time agent framework that does not train new models, nor does it use supervised loss or reinforcement learning objectives. The "optimization goal" is reflected in the reasoning process: maximizing test coverage and final SQL execution accuracy while constraining wall time through parallel execution. The implementation uses LangGraph to organize agents and limits the maximum iterations of each LLM agent to prevent loops. The main experiments use GPT-o3, while certain analyses compare GPT-5, Claude Sonnet-4, Claude Opus-4, and different component configurations.

## Key Experimental Results

### Main Results
Experiments were conducted on the Snow and Lite* versions of Spider 2.0. Snow contains 547 samples across 150+ databases, with approximately 800 columns per database using the Snowflake dialect; Lite* excludes BigQuery samples. Metrics include Execution Accuracy (EX), EX@4, and average wall time.

| Method | Snow EX | Snow EX@4 | Lite* EX | Lite* EX@4 | Wall Time (min) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Spider-Agent | 25.2 | 27.4 | 26.2 | 28.7 | 5.90 |
| ReFoRCE | 36.6 | 39.7 | 36.2 | 39.5 | 5.44 |
| Chat2DB | 44.1 | - | - | - | - |
| AgenticData | - | - | 44.5 | - | - |
| **PExA** | **45.7** | **49.5** | **46.6** | **49.9** | **5.55** |

| Additional Comparison | Snow EX | Snow EX@4 | Notes |
| :--- | :--- | :--- | :--- |
| **Ours** | 45.7 | 49.5 | Default schema linking |
| **Ours** w/ Gold Schema | 47.2 | 50.8 | Gold schema adds only ~1.5 pts |
| Updated Leaderboard | 70.2 | - | Reached new SOTA on Spider 2.0 |

### Ablation Study

| Configuration | EX | Relative to Full | Explanation |
| :--- | :--- | :--- | :--- |
| PExA Full | 42.9 | - | Analysis setting for single Snow run |
| w/o Plan-time parallelization | 40.0 | -2.9 | Parallel test suites in planning are critical |
| w/o Test-time parallelization | 39.9 | -3.0 | Parallel test SQL generation/execution contributes most |
| w/o Semantic Verifier | 42.3 | -0.6 | Semantic verification fixes some bias but is not the main bottleneck |
| w/o Proposer | 41.1 | -1.8 | Returning directly from test artifacts loses integration capability |

| Parallelism Setting | Exec Branch = 1 | Exec Branch = 2 | Exec Branch = Unlimit |
| :--- | :--- | :--- | :--- |
| Planning Branch = 1 | 38.4 | 39.1 | 40.0 |
| Planning Branch = 2 | 38.9 | 39.5 | 41.1 |
| Planning Branch = Unlimit | 39.9 | 41.6 | 42.9 |

| Latency Mode | Estimated Total Latency |
| :--- | :--- |
| Sequential execution (same search space) | 680 s |
| Parallel execution | 351 s |

### Key Findings
- PExA outperforms ReFoRCE on both Snow and Lite*, with Snow EX increasing from 36.6 to 45.7 and Lite* EX from 36.2 to 46.6, while maintaining a wall time of 5.55 minutes, close to ReFoRCE's 5.44 minutes.
- Ablations show that the two parallel components each contribute approximately 3 EX points, significantly more than the 0.6 points from the Semantic Verifier, indicating that performance gains primarily come from parallel search width rather than final semantic checks.
- Limiting the planning and execution branches leads to a continuous decrease in accuracy from 42.9 to 38.4; this provides a tunable parameter for system deployment, allowing for lower costs in exchange for fewer branches.
- Gold schema only provides a marginal ~1.5 point improvement, suggesting that the current primary bottleneck is not schema linking, but rather complex semantic planning and long-range logical composition.
- Error analysis indicates that failure primarily arises from semantic misunderstandings and initial planning flaws rather than SQL syntax errors. This aligns with the authors' conclusion that future work should focus on problem understanding and plan search.

## Highlights & Insights
- The software testing perspective is highly suited for complex Text-to-SQL. High-quality SQL is not written in a vacuum but requires verifying that each local hypothesis works within the database; PExA explicitly transforms this engineering intuition into an agent structure.
- PExA’s test cases are not limited to original problem decomposition but can explore non-target yet relevant database information. This is more proactive than traditional decomposition and closer to how data analysts work—querying while writing SQL in a real database.
- Parallelization is not simple "multi-sampling." It gives each exploration path a clear semantic goal, avoiding uncontrolled self-consistency that wastes tokens. Single-step multi-path search is also more suitable for low-latency scenarios than multiple rounds of serial reflection.
- The insight that gold schema has a small impact is valuable: in Spider 2.0's large schema scenarios, intuition often suggests optimizing schema linking first, but PExA demonstrates that high-level planning and semantic coverage may be more deserving of investment.

## Limitations & Future Work
- PExA relies on strong closed-source LLMs (GPT-o3), which limits cost-efficiency and reproducibility. While the paper analyzes other model mixes, a version using fully open-source models still needs verification.
- Although parallel exploration reduces wall time, it does not necessarily reduce total token usage or API costs. For production environments, dynamically selecting the number of branches, early stopping, and reusing test results are key engineering challenges.
- The quality of test cases is highly dependent on the Planner. If the initial coverage direction is wrong, subsequent parallel execution will only collect irrelevant evidence faster; error analysis confirms planning errors as the primary bottleneck.
- The Semantic Verifier relies on translating SQL back to NL, which remains susceptible to misjudgment by the LLM itself. Stronger verification might require results-set-based checks, unit tests, or symbolic constraints.
- Currently only verified on Spider 2.0; future work should transfer the model to enterprise BI, cross-database dialects, code generation, and other agent tasks requiring "executable test coverage."

## Related Work & Insights
- **vs Spider-Agent**: Spider-Agent is more like a traditional tool-augmented agent. PExA differs by explicitly organizing exploration into parallel test suites, resulting in a better accuracy-latency frontier.
- **vs ReFoRCE**: ReFoRCE emphasizes database compression and inference-time scaling. PExA also adopts lightweight schema linking but provides core value through parallel test-case SQL coverage and evidence aggregation.
- **vs Query Decomposition**: Standard decomposition only splits the explicit sub-semantics of the original question. PExA’s test cases actively explore surrounding schemas, value distributions, and intermediate results, offering broader coverage.
- **vs Self-consistency / Multi-sampling**: Multi-sampling often lacks directional control; each of PExA’s branches has a specific test objective and execution feedback, making it more of a structured search than a random multiple generation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reframing the Text-to-SQL agent with software test coverage is distinctive and naturally explains the parallelizability.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Main results on Spider 2.0, component ablations, and parallelism/latency analyses are solid, though restricted by closed-source LLM dependence and limited reproducibility.
- Writing Quality: ⭐⭐⭐⭐☆ Methodological descriptions are clear, and tables directly support claims; providing more real test-case traces would increase intuitiveness.
- Value: ⭐⭐⭐⭐⭐ Provides reusable insights for Text-to-SQL, data analysis agents, and general tool-use agents, particularly the "test-first" parallel exploration paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] R$^3$-SQL: Ranking Reward and Resampling for Text-to-SQL](r3-sql_ranking_reward_and_resampling_for_text-to-sql.md)
- [\[ACL 2026\] PV-SQL: Synergizing Database Probing and Rule-based Verification for Text-to-SQL Agents](pv-sql_synergizing_database_probing_and_rule-based_verification_for_text-to-sql_.md)
- [\[ACL 2026\] DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency](dpc_training-free_text-to-sql_candidate_selection_via_dual-paradigm_consistency.md)
- [\[ACL 2026\] Learning Adaptive Parallel Execution for Efficient Code Localization](learning_adaptive_parallel_execution_for_efficient_code_localization.md)
- [\[ACL 2026\] Bootstrapping Code Translation with Weighted Multilanguage Exploration](bootstrapping_code_translation_with_weighted_multilanguage_exploration.md)

</div>

<!-- RELATED:END -->
