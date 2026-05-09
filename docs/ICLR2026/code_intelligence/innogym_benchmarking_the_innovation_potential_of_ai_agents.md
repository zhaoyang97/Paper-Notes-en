---
title: >-
  [Paper Note] InnoGym: Benchmarking the Innovation Potential of AI Agents
description: >-
  [ICLR 2026][AI agent benchmark] This paper proposes InnoGym, the first benchmark and framework for systematically evaluating the innovation potential of AI agents. It introduces two complementary metrics—Performance Gain and Novelty—and, through 18 improvable tasks, finds that current agents exhibit a degree of innovativeness but lack the robustness to reliably translate novel ideas into performance improvements.
tags:
  - ICLR 2026
  - AI agent benchmark
  - innovation evaluation
  - performance gain
  - novelty
  - improvable tasks
date: 2026-05-08
content_hash: 29cc54ffe13e637c
---

# InnoGym: Benchmarking the Innovation Potential of AI Agents

**Conference**: ICLR 2026
**arXiv**: [2512.01822](https://arxiv.org/abs/2512.01822)
**Code**: [https://github.com/zjunlp/igym](https://github.com/zjunlp/igym)
**Area**: Code Intelligence
**Keywords**: AI agent benchmark, innovation evaluation, performance gain, novelty, improvable tasks

## TL;DR
This paper proposes InnoGym, the first benchmark and framework for systematically evaluating the innovation potential of AI agents. It introduces two complementary metrics—Performance Gain and Novelty—and, through 18 improvable tasks, finds that current agents exhibit a degree of innovativeness but lack the robustness to reliably translate novel ideas into performance improvements.

## Background & Motivation

**State of the Field**: Existing LLM and agent evaluation benchmarks (e.g., SWE-Bench, MLE-Bench, HumanEval) primarily focus on correctness—success is determined by passing test cases or matching reference answers. Such benchmarks have driven rapid progress in code generation, mathematical reasoning, and scientific discovery. MLAgentBench, DSBench, and MLGym also assess agents' ML engineering capabilities in Kaggle-style settings, but still use leaderboard performance scores as the primary criterion.

**Limitations of Prior Work**: This correctness-centric evaluation paradigm completely ignores differences in methodology. Two agents may arrive at the same correct answer via entirely different approaches, yet existing benchmarks cannot distinguish such methodological differences. Moreover, in real scientific and engineering problems, there is often no single correct answer; the key question is whether a more effective or more novel solution can be proposed.

**Root Cause**: Intelligence and innovation manifest not only in outcomes but also in methodology. Existing evaluation frameworks equate "problem-solving ability" with "the ability to produce a correct answer," and thus cannot measure an agent's creativity or methodological innovation—precisely the core capability required for AI-driven scientific discovery.

**Paper Goals**: (1) How can "innovation" be formally defined and quantified? (2) How can a benchmark be constructed that simultaneously evaluates performance improvement and methodological novelty? (3) How do current mainstream agent frameworks perform on real innovation tasks? (4) What is the relationship between performance and novelty?

**Starting Point**: Inspired by management scholar Peter Drucker's observation that "innovation is change that creates a new dimension of performance," each task is formalized as a tuple $(P, S, V, D)$, upon which Performance Gain and Novelty are defined as two orthogonal metrics, forming a two-dimensional innovation evaluation space. Tasks are also categorized into three types (Solved, Improvable, Exploratory), with a focus on improvable tasks that have clear room for improvement.

**Core Idea**: Innovation is decomposed into two dimensions—"doing better" (Performance Gain) and "doing differently" (Novelty)—to evaluate the innovation potential of agents on real engineering and scientific problems.

## Method

### Overall Architecture
InnoGym consists of two complementary components: iBench (an innovation evaluation benchmark) and iGym (a unified execution environment). iBench contains 18 high-quality improvable tasks selected from real engineering and scientific domains, each formalized as $\mathcal{T} = (P, S, V, D)$, where $P$ is the problem instance, $S$ is the solution space, $V$ is the performance measure, and $D$ is the distance function between solutions.

On the input side, agents can only access $P_{\text{visible}}$ (task description, examples, development data, and dependency environment) and a validator $C$ (which checks solution format, executability, and constraint satisfaction). The evaluator $R$, the known solution set $S_{\text{known}}$, and leaderboard data remain hidden. On the output side, after an agent submits a solution, evaluation proceeds in three steps: submission → performance evaluation (computing $V(s) = C(s) \cdot R(s)$) → novelty evaluation (computing $N(s)$).

Tasks are drawn from top academic and industrial competitions spanning 2018–2024 (NeurIPS Competitions, KDD Cup, ROADEF, GMCM, MLArchSys) as well as classic NP-hard problems, covering machine learning, operations research, systems design, mathematics, and other domains.

### Key Designs

1. **Innovation Measurement Framework (Performance Gain + Novelty)**:

    - Function: Quantifies innovation along two orthogonal dimensions.
    - Mechanism: Performance Gain $G(s) = V(s) - V^*_{\text{known}}$ measures improvement over the best known solution; Novelty $N(s) = C(s) \cdot \min_{h \in S_{\text{known}}} D(s, h)$ measures methodological divergence from the known solution set. The combination yields four innovation categories: Breakthrough Innovation (high $G$, high $N$), Performance Innovation (high $G$, low $N$), Conceptual Innovation ($G \approx 0$, high $N$), and Ineffective Exploration (low $G$, low $N$).
    - Design Motivation: A single dimension cannot capture the full picture of innovation—fine-tuning to surpass SOTA is innovation, but reaching comparable performance via an entirely new method is equally so, and the two are qualitatively distinct.

2. **Task Selection and Standardization Pipeline (iBench)**:

    - Function: Selects 18 high-quality improvable tasks from 197 candidates.
    - Mechanism: A two-stage filtering process—Stage 1 checks resource availability and computational feasibility (datasets, evaluators, leaderboards, reference solutions), reducing the pool to 72; Stage 2 verifies evaluator quality and balances domain distribution, retaining 18 tasks. Each task undergoes a six-step standardization pipeline: task specification rewriting, environment packaging, validator construction, solution set collection, evaluator normalization (ensuring absolute scores achieve Pearson $\geq 0.9$ and Kendall $\tau \geq 0.8$), and data splitting.
    - Design Motivation: Solved problems (no room for improvement) and exploratory problems (no human baseline) are excluded in order to focus on tasks with clearly defined improvement headroom.

3. **Novelty Evaluation (Agent-as-Judge)**:

    - Function: Automatically assesses the methodological novelty of solutions.
    - Mechanism: A Codex extraction prompt first distills the core strategy of each solution into a structured representation; GPT-5 then scores both the agent's solution and reference solutions along six evaluation dimensions (0–4 per dimension). The minimum distance across all reference solutions is taken and normalized to $[0, 100]$.
    - Design Motivation: Methodological differences are difficult to capture with simple code-similarity metrics and require semantic-level understanding; the Agent-as-Judge approach scales to diverse task types.
    - Limitation: Judgment quality depends on GPT-5's capabilities, and different model versions may produce inconsistent novelty scores.

### Unified Execution Environment: iGym
iGym is the unified agent execution SDK accompanying InnoGym, addressing critical shortcomings of existing SDKs (e.g., OpenHands, AutoGen, LangGraph) in long-running tasks. Key features include: (1) an Async Tool Dispatcher that allows agents to invoke multiple tools concurrently without blocking; (2) a robust recovery mechanism that supports checkpoint-based resumption across 12-hour long-running sessions; and (3) a unified abstraction layer enabling different agent frameworks (workflow-based and agent-based) to interact within the same environment. iGym ensures that performance differences across agents primarily reflect their design rather than infrastructure disparities.

### Evaluation Procedure
Each task–agent–model configuration is allowed up to 12 hours of runtime, with 3 runs performed and the best valid submission retained. Evaluation proceeds in three steps: (1) the agent generates a submission using visible data and tools; (2) validator $C(s)$ checks feasibility and evaluator $R(s)$ computes the performance score; (3) Codex extracts the core strategy and GPT-5 assesses novelty. Scores on both dimensions jointly determine the innovation category.

The main experiments use DeepSeek-v3.1 as the backbone LLM; analytical experiments additionally compare GPT-5 and Gemini-2.5-Pro. For cross-task comparability, a normalized Ratio $= G(s) / V^*(s)$ is also reported.

## Key Experimental Results

### Main Results

| Task | Leaderboard Best | MLAB Gain/Ratio/Novelty | CodeAct Gain/Ratio/Novelty | AIDE Gain/Ratio/Novelty |
|------|-----------------|------------------------|---------------------------|------------------------|
| BEETL(MI) | 76.33 | -35.66/-0.47/66.67 | No valid submission | No valid submission |
| BEETL(Sleep) | 69.23 | -14.64/-0.21/62.50 | No valid submission | -53.62/-0.77/54.17 |
| Belka | 30.62 | -19.02/-0.62/45.83 | -28.14/-0.92/45.83 | -30.01/-0.98/20.83 |
| CirclePacking | 2.635 | -0.43/-0.16/50.00 | -0.008/-0.003/25.00 | -0.25/-0.09/33.33 |
| OAG | 83.45 | -28.59/-0.34/70.83 | -30.38/-0.36/62.50 | -29.87/-0.36/70.83 |
| **Average** | **57.94** | **-24.32/-0.45/56.55** | **-41.58/-0.69/54.86** | **-42.68/-0.64/46.67** |

### Ablation Study (CirclePacking Task)

| Analysis Dimension | Key Result | Notes |
|-------------------|-----------|-------|
| Base model comparison | Gemini-2.5-Pro: 2.49, GPT-5: 2.44, DeepSeek-v3.1: 2.40 | AlphaEvolve reaches 2.65; agents amplify base model capability |
| Time budget effect | $G$ increases monotonically with time; $N$ gradually decreases | Diminishing returns: as solutions improve, methodology converges |
| Sampling temperature | Low temp → high performance, low novelty; high temp → high novelty, low performance | 0.5–0.75 is the optimal balance range |
| Prior knowledge | Starting from a Gemini-2.5-Pro solution, AIDE achieves sustained improvement | Validates that $G$ and $N$ jointly characterize innovation trajectories |

### Key Findings
- No agent surpasses human SOTA on any task; Performance Gain is consistently negative, with average Ratio ranging from $-0.45$ to $-0.69$.
- MLAB leads on both performance and novelty (average Gain $-24.32$, Novelty $56.55$), yet on complex tasks such as CDML and PTTALC, all agents fail to produce valid submissions.
- Robustness, not creativity, is the bottleneck: agents can generate novel methods but cannot reliably translate them into performance gains (e.g., RCIC and TrojanDetection exhibit high novelty alongside very low performance).
- CodeAct approaches SOTA on the mathematical optimization task CirclePacking (Ratio $= -0.003$) but generalizes poorly to other tasks.
- Base model capability significantly affects agent performance: Gemini-2.5-Pro reaches 2.49, GPT-5 reaches 2.44, and DeepSeek-v3.1 achieves only 2.40 (AlphaEvolve's 2.65 remains the highest).
- Temporal analysis reveals a pattern of diminishing returns: Performance Gain increases monotonically over time but at a decelerating rate, while declining Novelty reflects methodological convergence.

## Highlights & Insights
- Formalizing "innovation" as a two-dimensional $(G, N)$ space is an elegant design; the resulting taxonomy of Breakthrough / Performance / Conceptual Innovation is clear and well-motivated. The complex-plane representation ($G$ as modulus, $N$ as argument) further enhances visualization by distinguishing solutions with identical novelty scores but different methodological orientations.
- The systematic pipeline that filters from 197 candidate tasks down to 18 is rigorous; evaluator normalization via Pearson/Kendall tests ensures fair cross-task comparisons. The two-stage selection strategy can serve as a reference template for future benchmark construction.
- The finding that "agents are amplifiers of base models, not replacements" carries important implications for agent system design—it suggests that investing in stronger base models should take priority over developing complex agent architectures.
- The task taxonomy (Solved / Improvable / Exploratory) is conceptually clear and theoretically grounded; the decision to focus exclusively on improvable tasks by excluding solved and exploratory problems is well-justified.
- The temperature ablation on CirclePacking reveals a classic exploration–exploitation trade-off; the identified sweet spot of 0.5–0.75 has practical value for real-world agent deployment.

## Limitations & Future Work
- The main experiments cover only 10 of 18 tasks, and each configuration is run only 3 times, limiting statistical significance.
- Novelty relies on Agent-as-Judge (GPT-5 scoring), which may introduce large-model evaluation bias; different judge models may produce inconsistent novelty rankings.
- The paper lacks in-depth analysis of agent failure modes (e.g., which programming or reasoning capabilities constitute the bottleneck)—all agents fail on CDML and PTTALC, but the causes remain unexplained.
- All tasks are drawn from existing competitions and classic problems; the absence of originally designed new problems raises data contamination concerns, as LLM training data may include competition solutions.
- The 12-hour time limit may be insufficient for agents to complete complex engineering tasks, given that human competitors often invest several weeks.
- Only 3 agent frameworks are evaluated, leaving many representative systems unexamined (e.g., SWE-agent, Devin).

## Related Work & Insights
- **vs. MLE-Bench**: MLE-Bench evaluates only Kaggle rankings (i.e., Performance); this paper additionally introduces the Novelty dimension, enabling a distinction between "tuning to SOTA" and "reaching SOTA via a new method."
- **vs. InnovatorBench**: InnovatorBench assesses whether agents can reproduce innovations described in papers, but does not evaluate methodological novelty; InnoGym evaluates both performance and novelty, and focuses on open, improvable problems.
- **vs. AlphaEvolve**: AlphaEvolve is a specific innovation agent system (achieving 2.65 on CirclePacking); InnoGym is an evaluation framework—the two are complementary, as AlphaEvolve can be assessed on InnoGym.
- **vs. MLRCBench / MLGym**: These benchmarks also target ML engineering tasks but measure only performance, not methodological innovation. InnoGym draws from more diverse domains (including operations research and mathematics).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first benchmark to systematically evaluate agent innovation capability; the $(G, N)$ two-dimensional framework has theoretical depth, and the innovation taxonomy (Breakthrough / Performance / Conceptual) is original.
- Experimental Thoroughness: ⭐⭐⭐ Covers 10 tasks, 3 agent frameworks, and 3 base models, but the number of runs per configuration is small (3), some tasks yield no valid submissions, and statistical stability is limited.
- Writing Quality: ⭐⭐⭐⭐ Formal definitions are clear and rigorous; illustrations (complex plane, solution-space tree) are intuitive and creative; however, relegating iGym system details to the appendix slightly impairs readability of the main text.
- Value: ⭐⭐⭐⭐ Fills a gap in agent innovation evaluation and provides important guidance for the agent community; however, the uniformly poor agent performance limits the benchmark's discriminative power at present.

<!-- END -->

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research](../../NeurIPS2025/code_intelligence/mlr-bench_evaluating_ai_agents_on_open-ended_machine_learning_research.md)
- [\[ICLR 2026\] Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering](ambig-swe_interactive_agents_to_overcome_underspecificity_in_software_engineerin.md)
- [\[AAAI 2026\] SPAN: Benchmarking and Improving Cross-Calendar Temporal Reasoning of Large Language Models](../../AAAI2026/code_intelligence/span_benchmarking_and_improving_cross-calendar_temporal_reasoning_of_large_langu.md)
- [\[ACL 2026\] CodeWiki: Evaluating AI's Ability to Generate Holistic Documentation for Large-Scale Codebases](../../ACL2026/code_intelligence/codewiki_evaluating_ai39s_ability_to_generate_holistic_documentation_for_large-s.md)
- [\[ACL 2026\] EET: Experience-Driven Early Termination for Cost-Efficient Software Engineering Agents](../../ACL2026/code_intelligence/eet_experience-driven_early_termination_for_cost-efficient_software_engineering_.md)

<!-- RELATED:END -->
