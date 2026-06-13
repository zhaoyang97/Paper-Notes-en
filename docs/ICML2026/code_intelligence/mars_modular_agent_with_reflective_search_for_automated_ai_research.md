---
title: >-
  [Paper Note] MARS: Modular Agent with Reflective Search for Automated AI Research
description: >-
  [ICML 2026][Code Intelligence][MLE-Bench] MARS reformulates automated AI research as a problem of "searching for the optimal solution in the software repository space." By utilizing three pillars—**Budget-Aware MCTS…
tags:
  - "ICML 2026"
  - "Code Intelligence"
  - "MLE-Bench"
  - "Modular Agent"
  - "Budget-Aware MCTS"
  - "Comparative Reflective Memory"
  - "Lesson Learning"
date: 2026-05-08
content_hash: 5cbfd12e7a96da45
---

# MARS: Modular Agent with Reflective Search for Automated AI Research

**Conference**: ICML 2026  
**arXiv**: [2602.02660](https://arxiv.org/abs/2602.02660)  
**Code**: https://github.com/jfc43/MARS (Available)  
**Area**: LLM Agent / Automated Machine Learning (AutoML) / MLE Agent  
**Keywords**: MLE-Bench, Modular Agent, Budget-Aware MCTS, Comparative Reflective Memory, Lesson Learning

## TL;DR
MARS reformulates automated AI research as a problem of "searching for the optimal solution in the software repository space." By utilizing three pillars—**Budget-Aware MCTS, a modular "Design-Decompose-Implement" pipeline, and Comparative Reflective Memory**—it achieves State-of-the-Art (SOTA) among open-source frameworks on MLE-Bench. It reaches a gold medal rate of 31.1% (using Gemini-3-Pro-Preview) and exhibits an "Aha! moment" with a 63% cross-branch lesson transfer rate.

## Background & Motivation

**Background**: LLM agents have demonstrated strength in general software engineering (fixing GitHub issues, writing tests). Recent works (AIDE, AIRA, R&D-Agent, ML-Master, InternAgent, etc.) have begun applying them to the core bottleneck of automated AI research: Machine Learning Engineering (MLE) tasks. Performance is evaluated using gold/silver/bronze medal rates on OpenAI's MLE-Bench (75 Kaggle competitions × 24h × 1×A100).

**Limitations of Prior Work**: The authors identify three structural flaws in existing MLE agents:
- **Ignoring Execution Costs**: Current search strategies (greedy, vanilla MCTS, evolutionary) focus solely on performance without considering wall-clock time. A solution that increases accuracy by 0.1% but extends training from 1h to 10h is disastrous within a 24h budget, yet standard UCT tends to favor it.
- **Fragile Monolithic Scripts**: Most agents generate a single large Python file. This compresses logic due to token limits, requires total rewrites for minor changes, and complicates debugging. It cannot handle the multi-module coupling (data-model-training loop) found in real research repositories.
- **Memory Fails at Credit Assignment**: When experimental results improve, it is unclear which specific line of code change was responsible. Verbal reflection or trajectory caching (e.g., Reflexion, MemGPT) can "remember what was done" but cannot isolate causality.

**Key Challenge**: MLE $\neq$ General Programming. The former is a probabilistic long-horizon task characterized by "expensive evaluation + opaque attribution + high architectural complexity." It requires **strategic search with budget awareness** rather than just a more intelligent single-script generator.

**Goal**: Design an agent scaffolding that simultaneously addresses three sub-questions: (1) How to explicitly trade off performance and cost during search; (2) How to enable agents to produce repo-level solutions instead of scripts; (3) How to distill the differences between "success vs. failure" into transferable causal lessons.

**Key Insight**: Formalize MLE as $s^* = \arg\max_s \mathcal{O}(s, \mathcal{E})$ s.t. $\text{Cost}(s) \le B$. The solution space is redefined from "all possible Python programs" to "all possible modular repositories $s_n = \langle \{\mathcal{M}_j\}_{j=1}^{l}, \pi_{\text{main}}\rangle$." Search, memory, and reward functions are designed around this repo-level representation.

**Core Idea**: Use **Budget-Aware MCTS** to search within the repository space, replace single-script generation with a **Design-Decompose-Implement** pipeline, and resolve credit assignment using **Comparative Reflective Memory** (contrasting current solutions with the best-known solution). These elements synergize to facilitate long-horizon "Aha! moments."

## Method

### Overall Architecture
MARS is an iterative loop. The inputs are the MLE task triplet $\mathcal{P} = (\mathcal{I}, \mathcal{E}, \mathcal{O})$ (instructions / environment / objective) and a budget $B$; the output is a modular repository that maximizes $\mathcal{O}$ within $B$.

The process consists of two phases:
1. **Task Preparation**: A multi-agent system extracts task metadata, performs Exploratory Data Analysis (EDA), generates report-guided feature engineering, and prepares train/val/test splits.
2. **MARS Loop**: Iteratively executes three collaborative modules on an MCTS tree: Module A (Resource-Aware Planning, deciding the next action) $\rightarrow$ Module B (Modular Decomposition, generating/modifying sub-modules) $\rightarrow$ Module C (Reflective Memory, distilling lessons from trajectories back to Module B). The final output is the repository corresponding to the highest-scoring leaf in the tree.

Each MCTS node represents a candidate solution $s_n = \langle \{\mathcal{M}_j\}_{j=1}^{l}, \pi_{\text{main}}\rangle$. Three expansion actions are available: **Drafting** (creating a new solution from scratch at the root), **Improvement** (modifying modules on valid nodes), and **Debugging** (fixing runtime errors on buggy nodes, up to $N_d=10$ debug iterations).

### Key Designs

1. **Modular Decomposition (Design-Decompose-Implement Pipeline)**:
    - **Function**: Transitions agent coding from "emitting a single large script" to "architecture $\rightarrow$ decomposition $\rightarrow$ implementation and per-module validation," resulting in a multi-file repository.
    - **Mechanism**: Three specialized agents act in sequence: the *Idea Generation Agent* writes a full plan in natural language; the *Modular Agent* decomposes the plan into independent functional modules $\{\mathcal{M}_j\}$ (e.g., `dataset.py`, `model.py`, `engine.py`, etc.); the *Coding Agent* implements each $\mathcal{M}_j$ sequentially. Each module is verified with independent validation scripts before the $\pi_{\text{main}}$ orchestrates the end-to-end pipeline. Modifications use **Diff-Based Editing**: providing "target file + block to replace + new code" in a standard diff format, allowing atomic multi-file updates in a single LLM inference.
    - **Design Motivation**: Distributing code avoids token output limits; focusing on small logic units reduces context noise and increases accuracy; validated modules can be cached. Table 4 shows that with modularity, the average Lines of Code (LOC) increased from 474.8 to 1103.9 and file counts from 1.0 to 6.7, proving the agent produces more complex, structured solutions.

2. **Comparative Reflective Memory (Lesson Learning)**:
    - **Function**: Addresses the credit assignment problem of "which change caused metrics to rise/fall," distilling execution trajectories into a structured, searchable lesson pool.
    - **Mechanism**: For **successful** valid solutions, a two-step process is followed: the *Empirical Analysis Agent* extracts objective findings (e.g., metric trends) from logs; the *Lesson Distillation Agent* performs **comparative reflection**, contrasting the current solution with the "best-known solution" at a code level. It outputs a lesson containing: (1) isolated causal changes, (2) comparative impact analysis, and (3) generalized rules for future iterations. For **failed** buggy solutions, an agent analyzes buggy code, error logs, and applied fixes to output a debugging lesson. A Review Agent filters the pool for redundancy. Only the $K_m = 30$ most recent lessons are kept in context, and the agent is forced to explicitly cite used lessons.
    - **Design Motivation**: Conventional memory typically summarizes "what happened," leading agents to over-generalize from noise. Diff-based comparison isolates algorithmic changes, effectively performing an automated ablation study. Results show a lesson-utilization rate of 65.8% and a **lesson-transfer rate of 63.0%** (lessons used across different tree branches), providing quantitative evidence of "Aha!" moments.

3. **Budget-Aware MCTS (Efficiency-Sensitive Reward Function)**:
    - **Function**: Integrates execution time into the reward on top of standard UCT selection, systematically biasing the search towards "fast and high-performing" nodes.
    - **Mechanism**: Node performance $M(v)$ is normalized as $G(v) = (M(v) - M_{\min}) / (M_{\max} - M_{\min})$. The budget-aware reward is defined as $R(v) = G(v) \cdot [t(v)/L(v)]^w$, where $t(v)$ is actual execution time, $L(v)$ is the time limit, and $w$ is a negative penalty weight (default $w = -0.07$). Intuition: for two solutions with the same accuracy, the faster one ($t/L$ is smaller) receives a higher reward due to the negative exponent. Expansion rules are customized for MLE: buggy nodes are marked as fully-expanded after $N_d=10$ debug steps; valid nodes are closed after generating $N_i = 2$ children; the root re-activates if $n_s$ valid nodes fail to improve the best solution.
    - **Design Motivation**: MLE tasks have strict 24h wall-clock constraints. Vanilla MCTS wastes budget on slow solutions with marginal gains. $w = -0.07$ was found to be the sweet spot; $w=0$ (vanilla) drops performance, while $w=-0.15$ biases toward trivial fast nodes. Budget-awareness improved the effective solution rate from 16.1% to 19.5%.

### Loss & Training
MARS is a training-free scaffolding using a pre-trained LLM backbone (primarily Gemini-2.5-Pro and Gemini-3-Pro-Preview). All "learning" occurs during inference via MCTS and lesson pool evolution. Key hyperparameters: $K_m=30$ (max lessons), $N_d=10$ (debug limit), $N_i=2$ (improvement branch factor), $w=-0.07$ (reward penalty). MLE-Bench default is 24h × 1×A100 (MARS+ scales to 2 parallel trees × 2×H100 × 48 vCPU).

## Key Experimental Results

### Main Results: MLE-Bench 75 Tasks, Mean ± SEM over 3 independent runs (%)

| Agent | Model | Above Median | Bronze | Silver | Gold | Any Medal |
|--------|------|------|------|------|------|------|
| AIDE | Gemini-3-Pro-Prev | 48.0 | 4.9 | 11.1 | 16.4 | 32.4 |
| AIRA-dojo | Gemini-3-Pro-Prev | 55.6 | 5.8 | 8.0 | 24.0 | 37.8 |
| ML-Master 2.0 (leaderboard) | Deepseek-V3.2-Speciale | 63.1 | 11.1 | 25.8 | 19.6 | 56.4 |
| **MARS** | Gemini-3-Pro-Prev | **65.8** | 9.3 | 15.6 | **31.1** | 56.0 |
| **MARS+** (2×H100) | Gemini-3-Pro-Prev | **74.2** | 12.4 | 16.4 | **33.8** | **62.7** |

In controlled comparisons (same LLM/env), MARS significantly outperformed AIDE and AIRA-dojo. Against the official leaderboard, MARS achieved the highest Gold medal rate (31.1%) using fewer resources. MARS+ established new SOTA across all metrics. By difficulty (Table 3), MARS outperformed AIRA-dojo across Lite, Medium, and High splits.

### Ablation Study (MLE-Bench Lite, 22 contests)

| Configuration | Key Finding | Description |
|------|---------|------|
| Full MARS | baseline | All three modules enabled. |
| w/o Modular Decomposition | Significant drop | Validates modularity's role in reducing context noise; LOC fell from 1103.9 to 474.8. |
| w/o Memory | Drastic drop | Agent learns almost nothing without the lesson pool. |
| Memory w/o Comparative Analysis | Worse than full | Causal diff analysis between current and best-known is the core of lesson quality. |
| Greedy Search | Significantly worse | Lacks exploration, focuses only on local metric optimization. |
| Vanilla MCTS ($w=0$) | Moderate drop | Presence of exploration but lacks budget awareness. |
| Budget-Aware MCTS ($w=-0.07$) | Optimal | Effective solution rate 19.5% vs 16.1% for vanilla. |

### Key Findings
- **Comparative memory is the core contribution**: Removing the comparative aspect while keeping empirical analysis results in a performance drop, proving that value comes from isolating causal changes.
- **Lessons transfer across branches**: The 63% transfer rate justifies the MCTS + memory synergy as a means of global experience sharing.
- **Budget-awareness acts as a pruning heuristic**: By favoring faster nodes for equal accuracy, the agent explores ~20% more effective solutions within the 24h limit.
- **Modular design enables architectural thinking**: Increased LOC and file counts indicate the agent is thinking at an architectural level rather than just splitting files.

## Highlights & Insights
- **Paradigm shift to "search over modular repositories"**: Instead of searching for scripts, MARS treats the repository as the unit of search. This abstraction is potentially transferable to SWE-Bench or RepoCoder.
- **Agent-led ablation studies**: Humans isolate causality via ablations; MARS internalizes this through comparative reflection.
- **Simple but effective budget trick**: $R = G \cdot (t/L)^w$ is a minor change that significantly prunes inefficient trajectories.
- **Mandatory interpretability**: Forcing the agent to cite lessons makes its behavior auditable and debuggable.

## Limitations & Future Work
- **Benchmark scope**: MLE-Bench focuses on Kaggle-like tasks; real research involving hypothesis generation and literature review is not yet covered.
- **LLM dependency**: High reasoning load for lesson distillation and modular decomposition may limit performance on smaller open-source models.
- **Static search breadth**: Fixed coefficients ($N_i, N_d$) may not adapt well to varying task difficulties.
- **Memory scaling**: The current LRU truncation of 30 lessons could be replaced with RAG-style embedding-based retrieval for better cross-task scaling.

## Related Work & Insights
- **vs. AIDE/AIRA**: MARS replaces greedy/single-script/trajectory-based approaches with MCTS/modular/comparative memory, leading to substantial gains in Gold medal rates.
- **vs. Reflexion**: Unlike binary success/failure reflection, MARS performs code-level diffs to capture the logic of improvements.
- **Insight**: Budget constraints should be treated as priors for search agents. Redefining the unit of generation to "modular components" rather than "scripts" significantly improves the ceiling for complex tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ While individual components exist, the consistent integration around cost-constrained repo-level search is a highly original system-level contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 75 tasks with multiple runs, LLMs, and thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear framework diagrams and well-defined taxonomies.
- Value: ⭐⭐⭐⭐⭐ Open-source SOTA on MLE-Bench. The design principles are highly applicable to other long-horizon agent tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] RExBench: Can coding agents autonomously implement AI research extensions?](../../ACL2026/code_intelligence/rexbench_can_coding_agents_autonomously_implement_ai_research_extensions.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](../../ACL2026/code_intelligence/mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[NeurIPS 2025\] MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research](../../NeurIPS2025/code_intelligence/mlr-bench_evaluating_ai_agents_on_open-ended_machine_learning_research.md)
- [\[NeurIPS 2025\] VeriMaAS: Automated Multi-Agent Workflows for RTL Design](../../NeurIPS2025/code_intelligence/automated_multi-agent_workflows_for_rtl_design.md)
- [\[ICML 2026\] Physics Is All You Need? A Case Study in Physicist-Supervised AI Development of Scientific Software](physics_is_all_you_need_a_case_study_in_physicist-supervised_ai_development_of_s.md)

</div>

<!-- RELATED:END -->
