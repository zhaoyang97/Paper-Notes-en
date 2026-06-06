---
title: >-
  [Paper Note] AgentEval: DAG-Structured Step-Level Evaluation for Agentic Workflows with Error Propagation Tracking
description: >-
  [ACL 2026][LLM Evaluation][agent evaluation] AgentEval models agent execution traces as "Evaluation DAGs," utilizing a GPT-4o judge to score nodes across 5 distinct types. It tracks root causes using a greedy parent stra…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "agent evaluation"
  - "DAG"
  - "LLM-as-judge"
  - "root cause analysis"
  - "regression detection"
date: 2026-05-08
content_hash: 8d2a1d6636a92d38
---

# AgentEval: DAG-Structured Step-Level Evaluation for Agentic Workflows with Error Propagation Tracking

**Conference**: ACL 2026  
**arXiv**: [2604.23581](https://arxiv.org/abs/2604.23581)  
**Code**: https://github.com/bettyguo/AgentEval (Available)  
**Area**: LLM Agent Evaluation / Software Engineering  
**Keywords**: agent evaluation, DAG, LLM-as-judge, root cause analysis, regression detection

## TL;DR
AgentEval models agent execution traces as "Evaluation DAGs," utilizing a GPT-4o judge to score nodes across 5 distinct types. It tracks root causes using a greedy parent strategy, incorporates a taxonomy of 21 failure categories, and integrates with CI/CD. Compared to end-to-end evaluation, it improves failure detection recall by 2.17× (0.41 → 0.89) on 450 production traces, achieves human agreement of $\kappa=0.84$, and reaches a root cause accuracy of 72% (approaching the human limit of 81%). A 4-month pilot reduced the median root cause localization time from 4.2 hours to 22 minutes.

## Background & Motivation

**Background**: Approximately 62% of organizations are experimenting with agents, and 23% are scaling deployments. However, agent evaluation relies primarily on end-to-end results (task completion) or manual trace reviews; the former masks intermediate errors, while the latter is not scalable. Gartner predicts that by 2027, 40% of agent projects will be canceled due to a lack of systematic evaluation.

**Limitations of Prior Work**: (1) End-to-end evaluation only considers the final state, failing to identify where errors originate, how they propagate, or if they cascade; (2) Static benchmarks like AgentBench, SWE-bench, and WebArena act as "capability" assessments rather than "deployment" infrastructure, making them unsuitable for continuous monitoring, regression detection, or CI/CD; (3) Observability tools like LangSmith, Phoenix, and AgentOps provide trace views but lack formal dependency modeling, error propagation tracking, and root cause attribution; (4) Process supervision literature demonstrates that evaluating intermediate steps is superior to evaluating results alone, but this has mostly been applied to training (PRMs) rather than inference-time engineering infrastructure.

**Key Challenge**: Agent workflows are inherently DAGs, where errors compound and amplify along dependency chains (the paper reports that 63% of step-level failures are propagated from upstream). However, industrial evaluation remains either end-to-end or consists of flat step-level evaluations decoupled from the DAG structure.

**Goal**: (1) Formalize agent execution as a DAG; (2) Define calibratable step-level quality metrics for each node type; (3) Provide automated attribution from failures to root causes; (4) Integrate these capabilities into CI/CD for regression detection.

**Key Insight**: Import spectrum-based fault localization concepts from software engineering into agent traces, replace hard rules with LLM-as-judge for semantic scoring, and use comparative ablation to isolate and quantify the "DAG structure" variable.

**Core Idea**: "Rather than measuring end-to-end success or flat step-level quality, use a DAG to explicitly track whether each failure is propagated from upstream or generated locally, providing actionable root causes based on dependency relationships."

## Method

### Overall Architecture
AgentEval is an OpenTelemetry-compatible sidecar evaluation service consisting of four components: (1) **Evaluation DAG Formalization**: Traces are parsed into $\mathcal{G}=(V,E,\tau,\mathcal{M})$, where node types $\tau$ are drawn from a 5-element set $\{\text{Plan, ToolSel, ParamGen, Exec, Synth}\}$, each with specific metrics; (2) **Step-level LLM-as-judge Scoring**: A GPT-4o judge assigns scores based on a 1-5 rubric, CoT reasoning, and 5 few-shot anchors; (3) **Hierarchical Failure Taxonomy**: A 3-tier system with 21 subcategories for clustering and localization; (4) **CI/CD Regression Suite**: Utilizes paired bootstrap significance testing and double-threshold alerting, integrated into GitHub Actions to block deployments.

### Key Designs

1.  **Evaluation DAG and Boundary Handling**:
    - **Function**: Provides traceable dependency contexts for every step failure, distinguishing between "local failures" and "propagated failures."
    - **Mechanism**: Nodes are evaluated sequentially following a topological sort. The context $c_i$ for node $v_i$ aggregates only the outputs of parent nodes $\mathrm{pa}(v_i)$ (excluding parent judge scores to avoid "echo chamber" effects). If $q(v_i)<\theta_{\tau(v_i)}$ indicates a failure, and multiple parents are also below the threshold, a **greedy strategy** selects the parent with the lowest score $v_j^*=\arg\min_{v_j\in\mathrm{pa}(v_i)}q(v_j)$ as the propagation source; otherwise, $v_i$ is marked as the root cause. AgentEval supports both schema-defined DAGs (expected) and trace-inferred DAGs (actual); deviations serve as quality signals. Non-DAG traces (~12%) are handled via retry loop unrolling and timestamp-based branch reconstruction.
    - **Design Motivation**: Flat step evaluation misses cascades like "context truncation in step 3 ruining everything later." While full causal inference is computationally heavy, the greedy parent strategy is an engineering-friendly compromise. In the pilot, 72% of attributions were $\le 1$ hop from the ground truth.

2.  **Type-Adaptive LLM-as-judge and Asymmetric Prompt Framework**:
    - **Function**: Enables a single judge to evaluate 5 heterogeneous step types while avoiding bias from upstream errors.
    - **Mechanism**: Each step type has unique metrics: Plan (completeness, feasibility); ToolSel (accuracy, relevance); ParamGen (type, value, completeness); Exec (success rate, validity); Synth (faithfulness, completeness, coherence). The judge (GPT-4o) and agent (Claude 3.5 Sonnet / Llama 3 70B) must belong to different model families to avoid intra-family bias. The **asymmetric prompt design** evaluates Plan nodes against the original user query, while other nodes are evaluated against the local upstream context. This ensures downstream steps are not penalized for "gracefully handling upstream bugs," but only for "amplifying" them.
    - **Design Motivation**: "Judge being weaker than the agent" remains viable because verifying a tool choice is simpler than generating it. This keeps costs low (~$0.02/trace using GPT-4o-mini). The asymmetric prompts prevent a single upstream bug from being re-calculated as $N$ root causes downstream.

3.  **Three-Tier Failure Taxonomy + Counterfactual Root Cause Validation**:
    - **Function**: Categorizes failures semantically for post-mortem analysis and validates attribution quality via counterfactual evidence.
    - **Mechanism**: A 3-tier taxonomy includes 9 L2 categories and 21 L3 subcategories (e.g., Planning/Execution/Integration → Context loss/Output hallucination → Truncation/Fabrication). **Counterfactual Validation**: For 30 failed traces, steps identified as the root cause by AgentEval were replaced with gold reference outputs. In 87% of cases, downstream scores improved (avg. +2.3 points), proving the attribution reflects causality rather than mere correlation.
    - **Design Motivation**: High detection recall does not guarantee correct attribution. Counterfactuals provide the most direct evidence: "If we fix this, does the downstream actually recover?"

### Loss & Training
AgentEval is a training-free inference-time framework. Thresholds $\theta_\tau$ are calibrated per type. Regression detection uses paired bootstrap ($p<0.05$, 10,000 resamples) and historical $2\sigma$ double thresholds. Progressive evaluation uses 10 smoke tests (<5 min) to gate the full suite (100+ traces, <1h), saving 80% in costs.

## Key Experimental Results

### Main Results
Performed on 450 production traces (CS, DA, and DP workflows, 150 each) and a manual evaluation subset of 150 traces (987 steps), compared against three baselines:

| Method | FDRec ↑ | FPR ↓ | HA ($\kappa$) ↑ | RCA ↑ |
| :--- | :--- | :--- | :--- | :--- |
| E2E Only | 0.41 | 0.08 | 0.52 | N/A |
| Flat Step | 0.67 | 0.15 | 0.71 | 0.38 |
| Rule-Based | 0.58 | 0.05 | 0.63 | 0.45 |
| **AgentEval** | **0.89** | 0.07 | **0.84** | **0.72** |

AgentEval's failure detection recall (FDRec) improved by 2.17× compared to E2E and +22 pp compared to Flat Step, quantifying the contribution of the DAG structure.

### Ablation Study
Impact of four components (metrics after removing the component):

| Configuration | FDRec ↑ | HA ↑ | RCA ↑ | Description |
| :--- | :--- | :--- | :--- | :--- |
| Full AgentEval | 0.89 | 0.84 | 0.72 | Full framework |
| −DAG Structure | 0.67 | 0.71 | 0.38 | Largest drop (−22 pp / −34 pp) |
| −LLM-as-judge | 0.62 | 0.66 | 0.51 | Second largest drop |
| −Failure Taxonomy | 0.82 | 0.79 | 0.54 | Primarily affects RCA (−18 pp) |
| −Calibration Anchors | 0.85 | 0.76 | 0.68 | Primarily affects $\kappa$ (−8 pp) |

### Key Findings
- The DAG structure is the single most critical component; the simultaneous drop in FDRec and RCA indicates that dependency modeling is essential for identifying error origins.
- The advantage of the DAG increases with workflow length: +15 pp FDRec for steps $\le 3$, increasing to +28 pp for steps $\ge 6$.
- Strong generalization across benchmarks: On $\tau$-bench and SWE-bench, FDRec remained $\ge 0.78$ without changing the taxonomy, though RCA dropped by 14-20 pp, suggesting that while detection translates well, attribution requires domain-specific taxonomy extensions.
- 4-month real-world pilot (18 engineers): Detected 23 pre-deployment regressions. The CS-Agent failure rate dropped from 31% to 18% after fixing a context truncation bug.

## Highlights & Insights
- Combining "DAG-as-evaluation-structure" with "LLM-as-judge" is a key engineering unlock—the former addresses error propagation, while the latter addresses the brittleness of rules.
- Counterfactual root cause validation (replacing faulty steps with gold data) is a highly persuasive validation paradigm for RCA, transferable to process supervision and reasoning chain debugging.
- The observation that "the judge does not need to be stronger than the agent" is counter-intuitive but practical, allowing for large-scale evaluation with cheaper judges without sacrificing detection quality.

## Limitations & Future Work
- Primarily applicable to sequential or moderately branched tool-calling agents; the advantage drops to $<5$ pp when non-DAG traces exceed 60% (e.g., complex multi-agent collaboration or long reasoning loops).
- Core experiments were performed in a single organization context; RCA remains sensitive to the domain taxonomy.
- LLM-as-judge still exhibits model-specific biases, which are mitigated but not eliminated by cross-family evaluation.
- RCA utilizes greedy heuristics rather than formal causal inference.

## Related Work & Insights
- **vs. Process Supervision (Lightman 2024)**: Both emphasize intermediate steps, but while prior work uses PRMs for RL training rewards, AgentEval implements it as inference-time deployment infrastructure and continuous monitoring.
- **vs. LangSmith/Phoenix**: These tools provide trace visualization but lack formal DAG modeling and root cause attribution. AgentEval is the first open-source framework to integrate and quantify the benefits of these three elements.
- **vs. Spectrum-based Fault Localization (Jones 2005)**: Borrowed the root cause philosophy but replaced hard rules with LLM-as-judge for semantic scoring.

## Rating
- Novelty: ⭐⭐⭐⭐ (Strong integration, though components like LLM-as-judge have precedents; systematization of DAG evaluation is the highlight.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (450 traces, 4 model judges, cross-benchmark testing, 4-month pilot, and counterfactual validation.)
- Writing Quality: ⭐⭐⭐⭐ (High data transparency regarding CI, $\kappa$, and RCA strategies.)
- Value: ⭐⭐⭐⭐⭐ (Directly addresses agent production pain points with real-world evidence and open-source code.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MultiFileTest: A Multi-File-Level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms](multifiletest_a_multi-file-level_llm_unit_test_generation_benchmark_and_impact_o.md)
- [\[ICLR 2026\] Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](../../ICLR2026/llm_evaluation/talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis.md)
- [\[ACL 2026\] Finch: Benchmarking Finance & Accounting across Spreadsheet-Centric Enterprise Workflows](finch_benchmarking_finance_amp_accounting_across_spreadsheet-centric_enterprise_.md)
- [\[ACL 2026\] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](taxpraben_a_scalable_benchmark_for_structured_evaluation_of_llms_in_chinese_real.md)
- [\[ACL 2026\] HoWToBench: Holistic Evaluation for LLM's Capability in Human-level Writing using Tree of Writing](howtobench_holistic_evaluation_for_llms_capability_in_human-level_writing_using_.md)

</div>

<!-- RELATED:END -->
