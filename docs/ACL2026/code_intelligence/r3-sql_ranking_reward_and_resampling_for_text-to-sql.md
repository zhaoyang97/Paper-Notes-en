---
title: >-
  [Paper Note] R$^3$-SQL: Ranking Reward and Resampling for Text-to-SQL
description: >-
  [ACL2026][Code Intelligence][Text-to-SQL] R3-SQL targets generate-then-rank Text-to-SQL by grouping equivalent SQLs based on execution results and ranking them through a combination of pairwise/listwise and pointwise rew…
tags:
  - "ACL2026"
  - "Code Intelligence"
  - "Text-to-SQL"
  - "Candidate Ranking"
  - "Execution Result Grouping"
  - "Agentic Resampling"
  - "Position Bias"
date: 2026-05-08
content_hash: 4ffc6ec3529eeb2e
---

# R$^3$-SQL: Ranking Reward and Resampling for Text-to-SQL

**Conference**: ACL2026 Findings  
**arXiv**: [2604.25325](https://arxiv.org/abs/2604.25325)  
**Code**: Not open source  
**Area**: LLM Agent / Text-to-SQL  
**Keywords**: Text-to-SQL, Candidate Ranking, Execution Result Grouping, Agentic Resampling, Position Bias

## TL;DR
R3-SQL targets generate-then-rank Text-to-SQL by grouping equivalent SQLs based on execution results and ranking them through a combination of pairwise/listwise and pointwise rewards. It utilizes an LLM agent to detect missing correct SQLs in the candidate pool for selective resampling, achieving 75.03 EX on BIRD-dev.

## Background & Motivation
**Background**: Modern Text-to-SQL systems commonly employ a generate-then-rank paradigm: an LLM samples multiple SQL candidates, followed by a pointwise, listwise, or majority voting ranker to select the final SQL. Methods like CSC-SQL, Contextual-SQL, CHASE-SQL, and XiYan-SQL follow this approach.

**Limitations of Prior Work**: Existing rankers face two core issues. First is functional inconsistency: SQLs that are surface-level different but produce identical execution results receive different scores, potentially ranking an equivalent correct SQL lower. Second is bounded recall: if the candidate pool lacks the correct SQL, no ranker can recover it.

**Key Challenge**: The correctness of Text-to-SQL is determined by execution semantics rather than SQL string surface forms; however, common rankers still score individual SQL strings. Furthermore, the ranking stage assumes the correct candidate is already present, lacking a mechanism to detect if the generation stage missed the answer.

**Goal**: To establish a framework that simultaneously addresses ranking consistency and candidate recall, allowing the ranker to make decisions at the execution equivalence class level and actively expand the search space when the candidate pool is insufficient.

**Key Insight**: The authors decompose candidate selection into exploration and exploitation. Exploration uses agentic resampling to improve candidate pool recall; exploitation uses execution-result grouping and dual-reward ranking to improve precision.

**Core Idea**: Instead of ranking SQL strings individually, rank groups equivalent by execution results. Instead of unconditional resampling, allow an agent to selectively trigger larger resampling only when it determines the correct SQL is missing.

## Method
R3-SQL consists of three components: execution groupwise ranking, agentic resampling, and a position-consistency listwise ranker. The overall workflow generates an initial candidate pool, uses an agent to decide if resampling is needed, executes all SQLs to group candidates by result, performs pairwise/listwise preference comparisons across groups, and uses pointwise group utility as a tie-breaker.

### Overall Architecture
Input consists of a natural language question and a database schema. A base LLM generates $n$ SQL candidates $S=\{s_1,...,s_n\}$. If the agent $f(S)=0$ determines the candidate pool is insufficient, $m>n$ candidates are resampled, and a pointwise ranker selects the top-$n$ to form a replacement pool. Subsequently, SQLs are executed and grouped by result $G=\{g_1,...,g_M\}$. Each group represents a distinct semantic outcome. R3-SQL first calculates inter-group preferences $r_{list}$, then calculates group pointwise utility $r_{point}$, ranking by the tuple $(r_{list},r_{point})$ and returning the SQL with the highest pointwise rank from the winning group.

### Key Designs
1.  **Execution Result Grouping and Inter-group Preference**:
    - **Function**: Eliminates the issue of equivalent SQLs receiving inconsistent scores due to surface form differences.
    - **Mechanism**: Candidate SQLs are executed, and those with identical outputs are placed in the same group. For two groups $g_i,g_j$, a pairwise ranker compares all candidate pairs across groups to estimate $P(g_i>g_j)$. A decisive win is recorded only when the preference margin exceeds a threshold $\tau=0.05$. The group score $r_{list}(g_i)$ is the count of wins against other groups.
    - **Design Motivation**: If an execution result corresponds to multiple SQLs, surface differences within the group should not affect semantic judgment; inter-group comparison identifies "small but correct" groups better than group size voting.

2.  **Pointwise-group Utility as a Stable Anchor**:
    - **Function**: Provides a sequence-independent supplementary signal when listwise/groupwise preferences are uncertain.
    - **Mechanism**: For each group, $r_{point}(g)=w(g)\cdot u(g)$ is calculated, where $w(g)=|g|$ reflects execution consistency and $u(g)=\max_{s\in g} RR_s$ preserves the reciprocal rank signal of the strongest candidate in the group. Final sorting uses lexicographical order $(r_{list}, r_{point})$.
    - **Design Motivation**: Simple group size biases toward large groups, while simple pointwise scoring is affected by surface forms; combining them leverages relative preference, intra-group consistency, and individual candidate quality.

3.  **Agentic Resampling and Position-consistency Training**:
    - **Function**: Addresses bounded recall when the correct SQL is absent and reduces input order bias in the listwise ranker.
    - **Mechanism**: An agent audits the initial candidate pool. If it deems the correct SQL missing, the pool is discarded for a larger sampled pool pruned via a pointwise ranker. During listwise ranker training, correct/incorrect SQL pairs are fed in both original and swapped orders, using GRPO with a consistency reward $R=R_{base}+\lambda_c R_c$, where $\lambda_c=0.5$.
    - **Design Motivation**: Constant resampling increases noise and cost; position-consistency rewards ensure the ranker's judgment remains invariant to candidate order.

### Loss & Training
The ranking training of R3-SQL includes a pointwise ranker and a listwise/pairwise ranker. R3-POINT-32B is continued from Contextual-RM-32B; the R3-7B listwise ranker is trained using GRPO with an added input order consistency reward. During inference, group ranking prioritizes $r_{list}$ with $r_{point}$ as a tie-breaker, followed by a final individual SQL-level comparison for the top-2 groups to return the highest pointwise ranked candidate from the winner.

## Key Experimental Results

### Main Results

| SQL Selection Method | Ranker | BIRD-dev | Spider-test | Spider-DK | EHRSQL | ScienceBenchmark | Avg. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CSC-SQL | FMV | 71.58 | 86.64 | 76.97 | 41.04 | 56.68 | 66.58 |
| Contextual-SQL | Pointwise | 73.14 | 86.36 | 75.50 | 41.41 | 63.13 | 67.91 |
| CHASE-SQL | Listwise | 73.34 | 86.18 | 75.94 | 44.44 | 63.59 | 68.70 |
| XiYan-SQL | Listwise + FMV | 72.03 | 85.89 | 75.28 | 43.43 | 63.59 | 68.04 |
| **R3-SQL** | **Groupwise Point+List+FMV** | **75.03** | **87.19** | **77.92** | **46.30** | **66.82** | **70.65** |

### Ablation Study

| Configuration | BIRD-dev EX | Description |
| :--- | :--- | :--- |
| **R3-SQL** | **75.03** | Complete System |
| w/o Agentic Resampling | 74.25 | Decrease in candidate recall |
| w/o Pointwise Pruning | 73.92 | Increased noise in resampled pool |
| w/o Exec. Group Scoring | 73.47 | Decrease in equivalent SQL consistency |
| w/o Pointwise Ranker | 73.34 | Lack of order-independent anchor |
| w/o Listwise Ranker | 73.14 | Lack of inter-group relative preference |

### Key Findings
- R3-SQL outperforms baselines on five benchmarks, being the only method to exceed an average EX of 70 (Avg. 70.65).
- Functional inconsistency is eliminated by grouping: Contextual-SQL's score variance for identical execution results is 0.8571, while R3-SQL reduces it to 0.0000; introducing R3-7B improves BIRD-dev EX from 73.47 to 75.03.
- Agentic resampling improves candidate recall: average ranking upper bound increases from 78.80 to 82.72 (BIRD-dev from 81.23 to 84.62).
- Position-consistency reward is effective: R3-7B input consistency is 57.49%, dropping to 45.60% without the consistency reward and 37.82% without GRPO.
- Agent triggers are not blind: Trigger Resampling precision is 93.27 with 56.02 recall; Skip Resampling recall is 83.17, helping reduce unnecessary resampling.
- Computationally, R3-SQL uses 32 pointwise calls and 107 listwise calls per query, averaging 1.56 sec/query—faster and more accurate than CHASE-SQL's 1.68 sec/query.

## Highlights & Insights
- The core insight is that "SQL correctness is execution semantics, not token semantics." Ranking after grouping is more aligned with Text-to-SQL evaluation metrics.
- Agentic resampling migrates the concept of bounded recall from retrieval to candidate SQL generation: no ranker can select a non-existent correct candidate, necessitating a diagnosis of pool coverage.
- Lexicographic combination is simple but effective: inter-group preference handles relative correctness, while pointwise utility stabilizes closely ranked groups.
- "Always-resample" is inferior to agentic replacement, indicating that increasing candidate count is not a "free lunch"; noisy candidates distract the ranker, making selective triggering critical.

## Limitations & Future Work
- R3-SQL is stronger in-domain and relies on a supervised pointwise ranker; domain gaps in out-of-domain scenarios lead to marginal differences of 0.46-0.67 in the pointwise module.
- Execution result grouping depends on successful SQL execution, potentially making it fragile in scenarios with timeouts, empty results, non-deterministic functions, or restricted DB permissions.
- Agentic resampling requires additional LLM inference and more candidate generation; while more efficient than "always-resample," deployment costs remain high.
- The candidate pool replacement strategy outperformed union in experiments, but might discard rare valuable candidates; fine-grained retention/replacement strategies warrant exploration.
- Code is not open source; reproducing R3-POINT-32B, R3-7B, GRPO rewards, and agent prompts may be difficult.

## Related Work & Insights
- **vs Contextual-SQL**: Contextual-SQL scores individually via a pointwise ranker; R3-SQL merges equivalent execution results to prevent score inconsistency.
- **vs CSC-SQL / functional majority voting**: FMV relies solely on group size, which easily allows large but incorrect groups to win; R3-SQL ranks groups using listwise preference and pointwise utility.
- **vs CHASE-SQL**: CHASE-SQL relies on a listwise ranker; R3-SQL additionally handles position bias and bounded recall, achieving higher precision with slightly lower computation.
- **vs XiYan-SQL**: XiYan-SQL combines listwise ranking with FMV but lacks agentic resampling; R3-SQL incorporates active repair during the candidate generation phase.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of execution grouping, dual-reward ranking, and agentic resampling is comprehensive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers five benchmarks, seed stability, computational cost, and extensive ablation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem definition, intuitive diagrams, and well-supported experimental tables.
- **Value**: ⭐⭐⭐⭐⭐ Highly relevant for candidate selection and generation repair in practical Text-to-SQL systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PExA: Parallel Exploration Agent for Complex Text-to-SQL](pexa_parallel_exploration_agent_for_complex_text-to-sql.md)
- [\[ACL 2026\] PV-SQL: Synergizing Database Probing and Rule-based Verification for Text-to-SQL Agents](pv-sql_synergizing_database_probing_and_rule-based_verification_for_text-to-sql_.md)
- [\[ACL 2026\] DPC: Training-Free Text-to-SQL Candidate Selection via Dual-Paradigm Consistency](dpc_training-free_text-to-sql_candidate_selection_via_dual-paradigm_consistency.md)
- [\[ACL 2026\] ROSE: An Intent-Centered Evaluation Metric for NL2SQL](rose_an_intent-centered_evaluation_metric_for_nl2sql.md)
- [\[ACL 2026\] QiMeng-PRepair: Precise Code Repair via Edit-Aware Reward Optimization](qimeng-prepair_precise_code_repair_via_edit-aware_reward_optimization.md)

</div>

<!-- RELATED:END -->
