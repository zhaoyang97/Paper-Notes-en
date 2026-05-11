---
title: >-
  [Paper Note] SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents
description: >-
  [NeurIPS 2025][Code Intelligence][SWE-bench] A fully automated pipeline is developed to continuously mine real-world software engineering interaction tasks from GitHub, producing the SWE-rebench dataset of 21…
tags:
  - "NeurIPS 2025"
  - "Code Intelligence"
  - "SWE-bench"
  - "data contamination"
  - "automated pipeline"
  - "agent evaluation"
  - "software engineering"
date: 2026-05-08
content_hash: 36756c30b2167974
---

# SWE-rebench: An Automated Pipeline for Task Collection and Decontaminated Evaluation of Software Engineering Agents

**Conference**: NeurIPS 2025
**arXiv**: [2505.20411](https://arxiv.org/abs/2505.20411)
**Code**: [https://huggingface.co/datasets/nebius/SWE-rebench](https://huggingface.co/datasets/nebius/SWE-rebench)
**Area**: Agent / Code Generation
**Keywords**: SWE-bench, data contamination, automated pipeline, agent evaluation, software engineering

## TL;DR

A fully automated pipeline is developed to continuously mine real-world software engineering interaction tasks from GitHub, producing the SWE-rebench dataset of 21,000+ executable Python tasks and a decontaminated benchmark. The work reveals that several models exhibit contamination-inflated performance on SWE-bench Verified (e.g., DeepSeek-V3: 39.7% on SWE-bench vs. 21.3% on SWE-rebench).

## Background & Motivation

**Background**: LLM-based software engineering agents have shown progress on benchmarks such as SWE-bench, yet training data—particularly for interactive tasks—remains scarce and largely manually curated (e.g., SWE-Gym provides only ~2,400 tasks). On the evaluation side, SWE-bench has faced data contamination risks since its public release in late 2023.

**Limitations of Prior Work**:
   - **Training data**: RL training requires interactive and verifiable tasks, but the complexity of environment configuration in software engineering makes large-scale automated collection difficult.
   - **Evaluation contamination**: Static benchmarks become stale rapidly as new models are released; models may have been exposed to SWE-bench data during pre-training or post-training.
   - **Incomparable evaluations**: Different teams employ different scaffolds, retry mechanisms, and best-of-N strategies, making fair comparison of raw model capabilities difficult.
   - **High variance**: Stochasticity in agent trajectories renders single-run results unrepresentative.

**Key Challenge**: The SWE domain lacks large-scale verifiable interactive datasets analogous to those in mathematics, and contamination in existing benchmarks undermines the reliability of evaluation results.

**Goal**: To construct a fully automated and sustainable pipeline for mining SWE tasks, alongside a continuously updated decontaminated evaluation benchmark.

**Key Insight**: A four-stage automated pipeline—collecting PRs from GitHub Archive → LLM-based automatic environment configuration → execution validation → quality assessment. Continuous updates maintain benchmark freshness.

**Core Idea**: Fully automated mining of executable SWE tasks from GitHub, continuous updates to avoid contamination, and standardized scaffolding for fair evaluation of genuine model capabilities.

## Method

### Overall Architecture

A four-stage pipeline: (1) initial task collection—downloading and filtering PRs from GitHub Archive; (2) automated environment configuration—using an LLM (Qwen2.5-72B) to extract installation instructions from README files, Dockerfiles, and similar sources; (3) execution validation—installing environments in containers, running tests, and verifying task solvability; (4) quality assessment—using a fine-tuned LLM to automatically evaluate issue clarity, difficulty, and test correctness.

### Key Designs

1. **Fully Automated Environment Configuration (Stage 2)**

    - **Function**: An LLM automatically extracts installation instructions from repository files (README, setup.py, Dockerfile, etc.), replacing manual configuration.
    - **Mechanism**: The LLM scans repository files → extracts structured JSON installation recipes → generates up to three candidate recipes. If installation or testing fails, the LLM analyzes error logs and revises the recipe. Environments are shared across PRs grouped by major.minor version.
    - **Design Motivation**: Both SWE-bench and SWE-Gym rely on manually configured environments, limiting scale (SWE-bench covers only 12 repositories). Automation enables coverage of 3,468 repositories, with 31% successfully configured.

2. **Execution Validation (Stage 3)**

    - **Function**: Installs environments in containers, runs test patches, and verifies the "fail-then-pass" pattern.
    - **Mechanism**: Three conditions must hold—(a) at least one test fails before applying the patch; (b) all previously failing tests pass after the patch; (c) tests passing before the patch continue to pass afterward. Parallel execution is carried out on the TractoAI distributed platform.
    - **Design Motivation**: Ensures tasks are genuinely solvable and tests are valid—a fundamental requirement for RL training, as task defects could otherwise lead to incorrect penalization of agents.

3. **Automated Quality Assessment (Stage 4)**

    - **Function**: A fine-tuned Qwen 2.5-72B predicts three quality labels: issue clarity (79% accuracy), task complexity (81% accuracy), and test correctness (67% accuracy).
    - **Mechanism**: Trained on human annotations from SWE-bench Verified, taking as input the issue description, solution patch, and test patch to predict quality labels.
    - **Design Motivation**: Manual inspection of all 21K tasks is infeasible. Automated labels, while imperfect, are more precise than heuristic methods (e.g., number of modified files) and provide users with flexible filtering capability.

4. **Decontaminated Benchmark Design**

    - **Function**: 294 executable tasks (from 169 repositories), evaluating all models with a standardized ReAct scaffold.
    - **Mechanism**:
        - **Unified scaffold**: All models use the same minimal ReAct framework, identical prompts, default hyperparameters, and 128K context.
        - **Timestamp-based decontamination**: Issue/PR creation timestamps are precisely tracked to flag potentially contaminated evaluations.
        - **Multi-run statistics**: Each model is run five times; mean ± SEM and pass@5 are reported.
    - **Design Motivation**: Eliminates the influence of scaffold differences on evaluation, revealing the "bare capability" of models.

### Loss & Training

This work primarily concerns a dataset and evaluation pipeline; it does not involve agent training. The Qwen fine-tuning for quality assessment uses supervised learning on SWE-bench Verified annotations.

## Key Experimental Results

### Main Results (SWE-rebench Mar–Apr 2025 Decontaminated Subset)

| Model | Resolved% | SEM | Pass@5% |
|------|-----------|-----|---------|
| GPT-4.1 | **26.7** | 1.09 | **39.0** |
| DeepSeek-V3-0324 | 21.3 | 0.98 | 32.4 |
| DeepSeek-V3-1226 | 21.9 | 1.44 | 31.4 |
| Qwen3-235B no-think | 16.6 | 0.93 | 25.7 |
| Qwen3-32B no-think | 13.7 | 1.03 | 26.7 |
| LLaMA-4-Maverick | 12.2 | 1.69 | 27.6 |
| LLaMA-3.3-70B | 11.2 | 0.47 | 22.9 |
| Qwen2.5-72B | 9.3 | 1.26 | 19.0 |

### Contamination Comparison (SWE-bench Verified vs. SWE-rebench)

| Model | SWE-bench Verified | SWE-rebench | Gap |
|------|-------------------|-------------|------|
| DeepSeek-V3-0324 | 39.7% | 21.3% | **-18.4%** |
| Qwen2.5-72B | 24.2% | 9.3% | **-14.9%** |
| LLaMA-3.3-70B | 19.6% | 11.2% | **-8.4%** |
| Qwen2.5-Coder-32B | 11.0% | 3.2% | **-7.8%** |

### Key Findings

- **Several models may exhibit severe performance inflation on SWE-bench Verified**: DeepSeek-V3-0324 achieves 39.7% on SWE-bench but only 21.3% on the decontaminated SWE-rebench (gap of 18.4%), strongly suggesting data contamination.
- **The two DeepSeek-V3 versions perform similarly on SWE-rebench (~21%) but diverge substantially on SWE-bench (35% vs. 40%)**: The latter gap may reflect greater contamination of the March version with SWE-bench data.
- **Qwen3's thinking mode offers no advantage**: The no-think variant performs marginally better, indicating that extended reasoning does not necessarily benefit SWE tasks.
- **LLaMA-4-Maverick's pass@5 substantially exceeds its Resolved rate**: This suggests high potential but low reliability.
- **Qwen2.5-Coder exhibits poor scaffold-following behavior**: It frequently hallucinates environment responses or enters format-error loops.

## Highlights & Insights

- **The fully automated pipeline is the primary contribution**: Scaling from manually curating 12 repositories (SWE-bench) to automatically covering 3,468 repositories has profound implications for RL-based training of SWE agents.
- **Timestamp-based decontamination is a simple yet effective design**: Flagging contamination by relating issue creation dates to model release dates is imperfect (models may have encountered data before release) but provides useful transparency.
- **The value of a standardized evaluation paradigm**: A unified scaffold reveals the "bare capability" of models—many that perform well on SWE-bench benefit substantially from engineered scaffolding rather than intrinsic model strength.
- **The 21K-task dataset is highly valuable for RL training of SWE agents**: Prior to this work, SWE-Gym offered only ~2,400 tasks; SWE-rebench provides nearly a tenfold increase in scale.

## Limitations & Future Work

- Coverage is limited to Python projects (>75% Python LOC); extension to other languages has not been pursued.
- Environment configuration success rate is only 31%, discarding a substantial number of potentially valuable tasks.
- The quality assessment model has limited accuracy (67% for test correctness), potentially introducing noise.
- The decontamination method relies on timestamps and cannot detect whether a model has encountered modified versions of tasks during training.
- The benchmark contains only 294 tasks (vs. 500 in SWE-bench Verified), limiting statistical power.
- No experiments training agents on SWE-rebench data are conducted to validate its effectiveness as RL training data.

## Related Work & Insights

- **vs. SWE-bench**: SWE-bench is the manually curated gold standard; SWE-rebench is its automated alternative. The performance gap between the two quantifies contamination risk.
- **vs. SWE-Gym**: SWE-Gym provides ~2,400 tasks from a small number of repositories; SWE-rebench offers 21K+ tasks from 3,468 repositories, substantially improving scale and diversity.
- **vs. LiveCodeBench (code generation)**: SWE-rebench shares the philosophy of continuous updates to avoid contamination, but targets more complex interactive agent tasks rather than single-turn code generation.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of a fully automated pipeline and a decontaminated benchmark represents an important systems contribution, though there is no fundamental technical innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 13 models across multiple runs; the contamination analysis is compelling, though agent training experiments are absent.
- Writing Quality: ⭐⭐⭐⭐ The pipeline description is clear and the tables are information-dense.
- Value: ⭐⭐⭐⭐⭐ Provides the SWE agent community with both the large-scale training data and the trustworthy evaluation framework it urgently needs.

## Key Experimental Results

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Ambig-SWE: Interactive Agents to Overcome Underspecificity in Software Engineering](../../ICLR2026/code_intelligence/ambig-swe_interactive_agents_to_overcome_underspecificity_in_software_engineerin.md)
- [\[ACL 2026\] EET: Experience-Driven Early Termination for Cost-Efficient Software Engineering Agents](../../ACL2026/code_intelligence/eet_experience-driven_early_termination_for_cost-efficient_software_engineering_.md)
- [\[NeurIPS 2025\] VeriMaAS: Automated Multi-Agent Workflows for RTL Design](automated_multi-agent_workflows_for_rtl_design.md)
- [\[NeurIPS 2025\] FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts](flylora_boosting_task_decoupling_and_parameter_efficiency_via_implicit_rank-wise.md)
- [\[NeurIPS 2025\] MLR-Bench: Evaluating AI Agents on Open-Ended Machine Learning Research](mlr-bench_evaluating_ai_agents_on_open-ended_machine_learning_research.md)

</div>

<!-- RELATED:END -->
