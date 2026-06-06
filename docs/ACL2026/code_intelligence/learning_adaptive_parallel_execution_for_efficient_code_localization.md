---
title: >-
  [Paper Note] Learning Adaptive Parallel Execution for Efficient Code Localization
description: >-
  [ACL2026][Code Intelligence][Code Localization] FuseSearch models parallel tool calls in code localization as a joint quality-efficiency optimization problem. Using SFT+RL…
tags:
  - "ACL2026"
  - "Code Intelligence"
  - "Code Localization"
  - "Parallel Tool Use"
  - "GRPO"
  - "Tool Efficiency"
  - "SWE-bench Verified"
date: 2026-05-08
content_hash: 34c9b72d065ab340
---

# Learning Adaptive Parallel Execution for Efficient Code Localization

**Conference**: ACL2026  
**arXiv**: [2601.19568](https://arxiv.org/abs/2601.19568)  
**Code**: No public code link found in cache  
**Area**: Code Intelligence / LLM Agent  
**Keywords**: Code Localization, Parallel Tool Use, GRPO, Tool Efficiency, SWE-bench Verified  

## TL;DR
FuseSearch models parallel tool calls in code localization as a joint quality-efficiency optimization problem. Using SFT+RL, the agent learns to adaptively adjust search width according to task stages, achieving high F1 and significantly lower time/token costs on SWE-bench Verified using a compact model.

## Background & Motivation
**Background**: Automated software development agents typically locate the files, functions, or code snippets requiring modification before patch generation. Code localization has become the primary bottleneck in the pipeline; recent results cited in the paper show that SOTA agents spend over 50% of computational resources on localization.

**Limitations of Prior Work**: Traditional agents often call tools sequentially, leading to information starvation under tight turn budgets. Enforcing a fixed number of parallel tool calls per round results in significant redundant or useless retrievals. The paper observes that 34.9% of enforced parallel tool calls are redundant, offsetting the benefits of parallelism.

**Key Challenge**: Code localization must cover sufficient context quickly within limited interaction turns. However, greater coverage increases the risk of repetitive searches or irrelevant noise. Pursuing low cost alone misses critical files, while pursuing high recall causes search costs and context noise to explode.

**Goal**: The authors aim to train a localization agent that autonomously decides "when to parallelize, how much to parallelize, and where to search," maximizing both localization F1 and information gain per tool call.

**Key Insight**: Instead of constructing complex code graphs or language-specific ASTs, the paper retains only three language-agnostic read-only tools: grep, glob, and read_file. It treats whether a tool call brings new code entities as an explicit efficiency signal.

**Core Idea**: Tool efficiency is used to measure the proportion of new information in tool calls, which is integrated with file/function F1 into SFT filtering and GRPO rewards. This allows the model to learn an adaptive parallel strategy that progresses from broad exploration to focused refinement.

## Method
FuseSearch adopts a restrained design: utilizing only three tools during inference and introducing trajectory quality and efficiency metrics only during training. It first uses a strong teacher to generate candidate search trajectories, filters trajectories that are both accurate and efficient for SFT, and finally utilizes GRPO to further optimize a reward that multiplies F1 and efficiency.

### Overall Architecture
The input is an issue description $q$. The agent generates a set of tool calls $a_t$ over $T$ discrete turns, observes the results $o_t$, and finally outputs the set of code entities $\mathcal{A}$ to be modified. Unlike sequential agents that call one tool at a time, FuseSearch can issue multiple grep/glob/read_file calls in parallel per turn. These read-only tools have no synchronous side effects, and results are aggregated into the context before the next round.

The training process consists of two stages. In the SFT stage, Kimi-K2-Instruct generates approximately 24K candidate trajectories for 6K training queries, from which approximately 6K high-quality trajectories are selected based on both file/function F1 and tool efficiency. In the RL stage, using the SFT model as the initial policy, GRPO samples multiple trajectories and calculates rewards based on localization quality and tool efficiency, encouraging the model to reduce redundant exploration without sacrificing final accuracy.

### Key Designs
1. **Minimalist Three-Tool Localization Interface**:
	- Function: Provides cross-language code localization with low infrastructure overhead.
	- Mechanism: Grep for regex content search, glob for file path matching, and read_file for reading specified files or line ranges. All tools are read-only and safe for parallel execution.
	- Design Motivation: Graph construction, AST parsing, and language servers introduce language dependencies and preprocessing overhead; minimalist tools allow the model to focus its learning capacity on search strategies.

2. **Tool Efficiency Metric**:
	- Function: Measures whether each tool call truly brings new information, rather than just looking at the number of tools or trajectory length.
	- Mechanism: Maintains a history of discovered entities $\mathcal{H}$. For the $i$-th tool returning entity set $\mathcal{E}_i$, the information gain is $g_i=|\mathcal{E}_i\setminus\mathcal{H}|/|\mathcal{E}_i|$. Total trajectory efficiency is $e=\frac{1}{k}\sum_i g_i$.
	- Design Motivation: Penalizing long trajectories cannot distinguish between "searching new areas" and "repeatedly searching old areas"; tool efficiency directly assigns low scores to redundant calls.

3. **Joint Quality-Efficiency SFT+RL Training**:
	- Function: First teaches the model parallel tool usage, then uses RL to adjust parallel width and search rhythm.
	- Mechanism: SFT retains only trajectories satisfying $F_1\geq\rho_F$ and $e\geq\rho_e$. The GRPO reward follows $R(\tau)=\alpha F_1(\tau)+\gamma(F_1(\tau)\cdot e(\tau))$, where $F_1$ is the weighted sum of file-level and function-level F1.
	- Design Motivation: If $F_1=0$, high efficiency has no practical value. The multiplicative term ensures the efficiency bonus only amplifies gains when localization quality is established, preventing the model from under-searching to achieve high efficiency.

### Loss & Training
Training data is derived from 233 high-quality GitHub repositories. Samples involving new file/function additions, excessively short issue descriptions, or no code changes were excluded, resulting in approximately 21K samples from which ground truth files, functions/methods, and line ranges were extracted. The SFT model is required to generate 2-8 tool calls per round. RL uses GRPO, sampling multiple outputs and calculating rewards based on file/function F1 and efficiency; the paper compares F1-only, $F_1+e$, and $F_1+F_1\cdot e$, ultimately selecting the multiplicative interaction term.

## Key Experimental Results

### Main Results
Evaluation was conducted on SWE-bench Verified. Following prior settings, samples where patches introduced entirely new files or functions were excluded, leaving 386/500 examples. Results demonstrate that the trained FuseSearch-4B improves both localization quality and efficiency.

| Method / Config | File F1 | Func F1 | Efficiency / Cost Results | Note |
|-------------|---------|---------|-----------------|------|
| RepoSearcher, Qwen3-4B backbone | 38.1 | 21.7 | Comparison baseline in abstract | Specialized localization agent |
| FuseSearch-4B trained | 84.7 | 56.4 | 93.6% speedup, 67.7% fewer turns, 68.9% fewer tokens | Core results |
| Qwen3-4B Base | 64.50 | 38.91 | e=59.50, T=6.12s, Tok=47.9k | Without two-stage training |
| Qwen3-4B SFT+RL | 84.65 | 56.43 | e=69.00, T=5.43s, Tok=30.9k | After two-stage training |
| Qwen3-30B-A3B SFT+RL | 83.01 | 58.62 | e=64.53, T=10.6s, Tok=43.2k | Large models also benefit |

### Ablation Study
| Config | File F1 | Func F1 | #Turn | T(s) | Tok.(k) | Note |
|------|---------|---------|-------|------|---------|------|
| Seq SFT+RL | 78.82 | 50.21 | 7.52 | 8.03 | 59.4 | 1 tool per turn |
| Par SFT+RL | 84.65 | 56.45 | 5.60 | 5.43 | 30.9 | Parallel execution is significantly better |
| SFT only | 78.86 | 47.94 | 4.96 | 9.17 | 54.8 | Learns parallelism but remains redundant |
| RL reward: F1 only | 81.84 | 54.90 | N/A | 7.28 | 39.4 | Increases quality, sub-optimal efficiency |
| RL reward: $F_1+e$ | 79.22 | 51.98 | N/A | 9.40 | 45.7 | High efficiency, lower quality |
| RL reward: $F_1+F_1\cdot e$ | 84.65 | 56.45 | N/A | 5.43 | 30.9 | Best quality, efficiency, and cost |

### Key Findings
- SFT allows the model to parallelize more aggressively and improve F1, but introduces redundancy; after RL, the model learns a "wide-then-narrow" strategy, exploring broadly early on and focusing refinement later.
- Joint filtering is more stable than filtering by F1 or efficiency alone. File F1/Func F1/e for unfiltered SFT were 75.44/43.52/55.77, which improved to 78.86/47.94/62.03 with joint filtering.
- FuseSearch accelerates downstream repair agents. Without localization, Kimi-K2 achieves a 68.4 pass rate with 41.1 turns, 312s, and 1053k tokens; with Pre-Search, it achieves a 68.1 pass rate with 31.6 turns, 223s, and 562k tokens.
- The minimalist toolset is competitive even in sequential mode, suggesting code localization does not necessarily rely on language-specific graph structures; the real gain comes from the model learning effective parallelism.

## Highlights & Insights
- Tool efficiency is the most reusable concept in this paper. It does not crudely punish "using many tools," but rather "calls without new information," which is closer to the search quality of a real agent.
- The multiplicative reward design is highly rational: efficiency is meaningless when localization fails, and only becomes a bonus when localization succeeds. This prevents the agent from learning to be "efficient by searching less."
- The paper transforms parallel tool use from an engineering capability into a learning objective. While many agent frameworks support parallel calls, models do not naturally know when to parallelize; FuseSearch explicitly trains this decision.
- The results are insightful for small-model agents: a 4B model, through task-specific training and tool efficiency rewards, can approach or replace some of the work of expensive large models in the localization phase.

## Limitations & Future Work
- The authors note that the golden patch represents only one feasible repair path and may miss other correct localizations, meaning the F1 ground truth itself is biased.
- SWE-bench Verified primarily covers Python repositories; while the tools are language-agnostic, effectiveness on static languages like Java/C++ requires validation with more training and evaluation data.
- The current benchmark focuses on issue-driven localization and does not evaluate broader code search tasks like repository QA, code understanding, or documentation generation.
- Tool efficiency relies on the definition of "whether a code entity is new"; future work could incorporate semantic novelty, invocation cost, and file importance into the efficiency metric.

## Related Work & Insights
- **vs Agentless**: Agentless uses a fixed hierarchical process from file to function to line, which is simple and stable but lacks task adaptation; FuseSearch uses a learned policy to determine search width.
- **vs LocAgent / CoSIL**: Graph navigation agents leverage structural relationships but require language-specific graph construction; FuseSearch lowers deployment barriers via grep/glob/read_file.
- **vs RepoSearcher**: RepoSearcher is also a lightweight localization agent, but mostly relies on sequential iteration; the core improvement of FuseSearch comes from parallel tool calls and efficiency rewards.
- **Insights for Future Work**: General coding agents can treat "information gain per tool call" as online feedback to train or distill retrieval strategies with less repetitive searching and lower token costs.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The tool efficiency metric and quality-efficiency multiplicative reward are practical and clearly directed.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers SWE-bench Verified, training phases, parallel modes, filtering strategies, rewards, and downstream repair; cross-language repositories are still insufficient.
- Writing Quality: ⭐⭐⭐⭐☆ Methodology definitions and ablation logic are clear; some tables are slightly dense due to high information density.
- Value: ⭐⭐⭐⭐⭐ Highly practical for reducing costs and increasing speed for code agents, especially suitable as a pre-processing step for downstream repair.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] To Diff or Not to Diff? Structure-Aware and Adaptive Output Formats for Efficient LLM-based Code Editing](to_diff_or_not_to_diff_structure-aware_and_adaptive_output_formats_for_efficient.md)
- [\[ICLR 2026\] Improving Code Localization with Repository Memory](../../ICLR2026/code_intelligence/improving_code_localization_with_repository_memory.md)
- [\[ACL 2026\] PaT: Planning-after-Trial for Efficient Test-Time Code Generation](pat_planning-after-trial_for_efficient_test-time_code_generation.md)
- [\[ACL 2026\] PExA: Parallel Exploration Agent for Complex Text-to-SQL](pexa_parallel_exploration_agent_for_complex_text-to-sql.md)
- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)

</div>

<!-- RELATED:END -->
