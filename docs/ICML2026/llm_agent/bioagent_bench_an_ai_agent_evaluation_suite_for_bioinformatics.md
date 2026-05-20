---
title: >-
  [Paper Note] BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics
description: >-
  [ICML 2026][LLM Agent][agent evaluation] BioAgent Bench provides an end-to-end evaluation suite for "running bioinformatics pipelines with LLM agents"—10 real bioinformatics tasks × 10 frontier/open-weight models × 3 age…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "agent evaluation"
  - "bioinformatics pipeline"
  - "LLM judge"
  - "robustness perturbation testing"
date: 2026-05-08
content_hash: 41160e589540ab80
---

# BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics

**Conference**: ICML 2026  
**arXiv**: [2601.21800](https://arxiv.org/abs/2601.21800)  
**Code**: https://github.com/bioagent-bench/bioagent-bench  
**Area**: LLM Agent / Benchmark / Bioinformatics  
**Keywords**: agent evaluation, bioinformatics pipeline, LLM judge, robustness perturbation testing

## TL;DR
BioAgent Bench provides an end-to-end evaluation suite for "running bioinformatics pipelines with LLM agents"—10 real bioinformatics tasks × 10 frontier/open-weight models × 3 agent harnesses, combined with LLM judge scoring and three types of perturbation tests (corrupted/decoy/prompt-bloat). The study finds that frontier models can complete over 90% of pipelines, but robustness remains a concern.

## Background & Motivation
**Background**: LLM agents already have mature benchmarks in software engineering (SWE-bench) and general tool-use (AgentBench, ToolBench). In biomedicine, there are also BioML-bench, LAB-Bench, and BixBench. However, these benchmarks either reduce tasks to QA/code generation or focus on "data analysis" rather than "full pipeline execution."

**Limitations of Prior Work**: Real bioinformatics workflows are highly complex—requiring chaining of command-line tools, management of heterogeneous file formats, and interpretation of intermediate outputs. Evaluation is challenging because the same data can be processed by multiple valid pipelines, parameter choices greatly affect results, and many steps cannot be strictly judged as pass/fail. Directly copying SWE-bench's hard-matching evaluation is infeasible.

**Key Challenge**: (1) Real bioinformatics tasks are long (hours) and resource-intensive (tens of GB memory), while benchmarks need to be reproducible and scalable; (2) The existence of multiple solutions makes "automatic scoring" and "strict ground truth" incompatible; (3) Clinical/IP-sensitive data cannot be sent to closed-source APIs, necessitating evaluation of open-weight models, which are much weaker than frontier models.

**Goal**: (i) Build an end-to-end, pipeline-style bioinformatics task set that can run within reasonable compute budgets (<4h, <48GB); (ii) Design a scoring protocol tolerant of multiple solutions, using LLMs as judges; (iii) Add perturbation tests beyond vanilla settings to separately assess agent robustness to corrupted data, decoy files, and prompt bloat; (iv) Systematically compare 5 closed-source and 5 open-source models across 3 harnesses.

**Key Insight**: Deliberately restrict task scale to "small organisms (bacteria, viruses, fungi)" so that reference data can be bundled as input files, avoiding infrastructure issues like "agents needing to download tens of GBs of genomes," and focusing evaluation on pipeline orchestration capabilities.

**Core Idea**: Use "task prompt + input data + reference data + expected CSV/TSV output format" as a unified task specification. The LLM judge compares trace + outcome to provide step-level completion scores, supplemented by three types of perturbation tests to probe whether "high-level pipeline construction" and "low-level step reasoning" are both achieved.

## Method

### Overall Architecture
The benchmark consists of three components: (1) **Task Set**—10 end-to-end tasks covering RNA-seq, variant calling, metagenomics, transcript quantification, experimental evolution, etc. Each task includes an NL prompt, input data, reference data, and ground-truth CSV/TSV; (2) **Evaluation Harness**—agents run in a hashed sandbox directory, operating in one of Claude Code, Codex CLI, or OpenCode harnesses, able to call general Python packages or specialized bioinformatics tools, ultimately submitting each step's output and final result files to the grader; (3) **LLM Grader**—GPT-5.1 reads input/reference paths, expected outcome, agent outcome, trace (file paths only), and grading rubric, outputting `steps_completed`, `steps_to_completion`, `final_result_reached`, `results_match`, and `f1_score`. The **primary metric** is completion rate = (number of required steps passed)/(total steps).

There are four evaluation settings: multi-trial stability (repeat runs for consistency), prompt-bloat (add irrelevant content to task description), corrupted-input (manually damage input data to test agent detection), and decoy-input (add distracting files to test agent's file selection).

### Key Designs

1. **Task Curation and Scale Constraints**:

    - **Function**: Construct end-to-end pipeline tasks that run within <4h and <48GB, covering mainstream bioinformatics modalities.
    - **Mechanism**: 10 tasks span bulk/single-cell RNA-seq, comparative genomics, variant calling (bacterial evolution, GIAB NA12878, cystic fibrosis), (viral) metagenomics, transcript quantification, etc.; languages include Python/R/bash; 4 tasks (cystic-fibrosis, giab, transcript-quant, viral-metagenomics) are "verifiable" with binary pass/fail judgment. Each task specifies "end-to-end" and "structured CSV output," and deliberately selects small organisms (mouse Alzheimer model, E. coli evolution, dolphin viral metagenome, etc.) so reference data fits as input.
    - **Design Motivation**: Positioning the benchmark as "software engineering-like" rather than "bio data analysis" is to support future RL/distillation uses; scale constraints enable large-scale, reproducible evaluation (at the cost of not covering human-genome-scale workflows).

2. **LLM Judge + Multi-dimensional Scoring Protocol (Grader)**:

    - **Function**: Automated scoring in scenarios with multiple solutions, multiple steps, and abundant intermediate outputs.
    - **Mechanism**: The grader uses GPT-5.1, with input (input/reference paths, expected CSV, agent CSV, trace file path tree, grading rubric); the rubric prioritizes "pipeline completion" over numerical accuracy. Outputs five fields: `steps_completed` (steps completed), `steps_to_completion` (estimated total steps), `final_result_reached` (whether final artifact produced), `results_match` (correctness flag per rubric), `f1_score` (for giab only).
    - **Design Motivation**: Bioinformatics tasks allow multiple valid pipelines (e.g., variant calling via GATK4 HaplotypeCaller or DeepVariant), so hardcoded ground truth is infeasible; having the grader review the trace rather than just output allows partial credit for "high-level correct but output format wrong," closer to human expert judgment.

3. **Three Types of Robustness Perturbation Tests (Perturbation Suite)**:

    - **Function**: Separate "pipeline completion" from "true step-level biological reasoning."
    - **Mechanism**: (i) **Multi-trial Consistency**—run the same task 4 times, compute Jaccard for classification results (KEGG pathways, Gene IDs), Pearson for numerical results (p-value, abundance); (ii) **Prompt bloat**—add large irrelevant sections to the original prompt, observe change in completion rate $\Delta$; (iii) **Corrupted input**—manually damage FASTQ/BAM input files, ideal agent should detect and report error (✓ = detected); (iv) **Decoy input**—add extra files that should not be used, ideal agent should ignore (✗ = not misled).
    - **Design Motivation**: The core argument is that high-level pipeline construction ≠ reliable step-level reasoning. Vanilla completion rate alone overestimates agents' biological reasoning; perturbation tests are key probes for "understanding vs pattern matching."

### Loss & Training
As a benchmark, there is no training; for evaluation, GPT-5.2 in Codex CLI harness is used as the main robustness assessment model, with "high" reasoning effort enabled by default.

## Key Experimental Results

### Main Results
Average completion rate for 10 tasks in the vanilla setting (Codex CLI harness):

| Model Type | Model | Avg. Completion % |
|------------|-------|------------------|
| Closed-source frontier | Claude Opus 4.5 | **100** |
| Closed-source frontier | Gemini 3 Pro / GPT-5.2 / Sonnet 4.5 | >90 |
| Open-weight Best | GLM-4.7 | 82.5 |
| Other Open-weight | Various | as low as ~65 |

**Planning vs Execution**: Scoring only "high-level pipeline planning" (GPT-5.1, 1-5 scale) yields Pearson $r=0.61$ with end-to-end completion rate—correlated but not decisive. For example, Gemini-Pro-3 scores low on planning but high on execution, indicating open-weight models' bottleneck is more in "multi-turn agentic ability" than "domain knowledge."

### Ablation Study
Multi-trial stability (GPT-5.2, Codex CLI, 4 runs per task, Jaccard/Pearson):

| Task | Jaccard | Pearson | Notes |
|------|---------|---------|-------|
| transcript-quant | 1.000 | 1.000 | Fully deterministic |
| cystic-fibrosis | 1.000 | NA | Highly consistent |
| deseq | 0.978 | 0.995 | Nearly stable |
| viral-metagenomics | 0.667 | 1.000 | Stable numerically, variable classification |
| metagenomics | 0.395 | 0.746 | Moderate |
| alzheimer | 0.160 | 0.219 | Unstable |
| comparative-genomics | 0.004 | NA | Almost completely inconsistent |
| evolution | 0.000 | NA | Completely inconsistent |

Average Jaccard 0.43, Pearson 0.73—across 4 runs of the same agent on the same task, classification overlap is less than half.

Perturbation tests (GPT-5.2, single trial, Δ% is completion change after prompt-bloat):

| Task | Detected corrupted? | Resisted decoy? | Δ completion (%) |
|------|---------------------|-----------------|------------------|
| alzheimer-mouse | ✗ | ✗ | -12.5 |
| comparative-genomics | ✗ | ✓ | -20.0 |
| deseq | ✓ | ✗ | **-100.0** |
| evolution | ✓ | ✗ | +75.0 |
| giab | ✓ | ✗ | — |

### Key Findings
- **Frontier models do not require complex scaffolding**—Claude Opus 4.5 achieves 100% pipeline completion using bare Codex CLI, challenging the assumption that "agentic frameworks are necessary."
- **Pipeline construction ≠ step-level reasoning**—Results vary greatly across trials (comparative-genomics, evolution are almost completely inconsistent), indicating that even if agents "complete" the pipeline, their intermediate decisions (parameters, normalization, statistical assumptions) are unstable.
- **Low corrupted data detection rate**—Most agents do not detect manually corrupted input, blindly proceeding and producing erroneous results; the only exception is deseq (which errors out and drops completion by 100%), which is actually less favorable than "blindly running."
- **Poor decoy robustness**—Most agents are misled by decoy files, lacking the prior knowledge to select the correct file.
- **Open-weight models are valuable for privacy scenarios**—Although frontier closed-source models are stronger, sensitive patient data cannot be sent externally, making open-weight models necessary; this work provides the first systematic baseline for open-weight models in bioinformatics.

## Highlights & Insights
- **Delicate trade-off between task scale and evaluation feasibility**—Deliberately selecting small organisms allows reference data to be included as input files, avoiding the infrastructure cost of "agents downloading 30GB human reference," which is key for benchmark scalability.
- **LLM grader reviews trace file path trees, not file contents**—This both protects sensitive data and reduces grader token consumption, representing a pragmatic protocol design.
- **Three-way perturbation design**—Separately testing corrupted (cognitive), decoy (attention), and bloat (robustness) failures provides more diagnostic power than a single "stress test."
- **First systematic comparison of 5 closed-source and 5 open-source agents in bioinformatics**, providing the community with a directly reusable leaderboard foundation.

## Limitations & Future Work
- **Task scale is small**—Human-scale real workflows (e.g., full 30× WGS variant calling) are deliberately excluded; infrastructure steps like "finding references, downloading, staging" are skipped, limiting generalization to production scenarios.
- **LLM scoring is itself biased**—The grader is also an LLM (GPT-5.1/5.2), possibly favoring certain trace patterns; and since grader and agent are of the same generation, there is a "LLM judging LLM" circularity.
- **Perturbation tests are single trial**—Only one run per test, so statistical noise is high; for some tasks (comparative, evolution), trial-to-trial variation exceeds perturbation effect, so a 2D table of perturbation × seeds should be reported.
- **Open-weight only evaluated pass@1**; robustness was not tested on open-weight models, which is a clear shortcoming.
- **Lack of quantitative analysis of agent loop failure modes**—It is mentioned that some frontier models fall into error-correction loops or terminate prematurely, but no quantitative metrics (trace length/loop count) are provided.

## Related Work & Insights
- **vs SWE-bench (Jimenez et al., 2024)**: SWE-bench uses strict unit test pass/fail, while BioAgent Bench uses LLM judge soft scoring + step-level partial credit, better suited for multi-solution scientific workflows.
- **vs BioML-bench (Miller et al., 2025)**: BioML focuses on ML pipelines (protein engineering, single-cell, imaging, drug discovery), while BioAgent Bench focuses on bioinformatics toolchain orchestration; they are complementary.
- **vs LAB-Bench (Laurent et al., 2024)**: LAB-Bench is mainly multiple-choice "research skills" assessment, while BioAgent Bench emphasizes actual execution ability.
- **vs BixBench (Mitchener et al., 2025)**: BixBench focuses on data analysis reasoning, while BioAgent Bench emphasizes end-to-end pipeline execution + robustness perturbations.
- **Insights**: In any domain with "multiple solutions + multiple steps + abundant intermediate outputs" (quantum chemistry, earth science pipelines, robotics skill chains), the protocol of "LLM judge + task scale constraint + three-way perturbation" can be adapted to build scalable agent benchmarks.

## Rating
- Novelty: ⭐⭐⭐ Pragmatic protocol design, but task format and LLM-judge paradigm are not groundbreaking
- Experimental Thoroughness: ⭐⭐⭐ 10 tasks × 10 models × 3 harnesses is broad, but single-trial perturbation and lack of open-weight robustness are major flaws
- Writing Quality: ⭐⭐⭐⭐ Clear concepts (strict distinction between task/trial/grader/harness/suite), straightforward results section
- Value: ⭐⭐⭐⭐ Provides the first systematic answer to the feasibility of "using agents for bioinformatics," with practical deployment reference value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Watermarking LLM Agent Trajectories (ACTHOOK)](watermarking_llm_agent_trajectories.md)
- [\[ICLR 2026\] Unlearning Evaluation through Subset Statistical Independence](../../ICLR2026/llm_safety/unlearning_evaluation_through_subset_statistical_independence.md)
- [\[AAAI 2026\] From Single to Societal: Analyzing Persona-Induced Bias in Multi-Agent Interactions](../../AAAI2026/llm_safety/from_single_to_societal_analyzing_persona-induced_bias_in_multi-agent_interactio.md)
- [\[ACL 2025\] ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models](../../ACL2025/llm_safety/elba-bench_an_efficient_learning_backdoor_attacks_benchmark_for_large_language_m.md)
- [\[NeurIPS 2025\] HealthSLM-Bench: Benchmarking Small Language Models for Mobile and Wearable Healthcare Monitoring](../../NeurIPS2025/llm_safety/healthslm-bench_benchmarking_small_language_models_for_mobile_and_wearable_healt.md)

</div>

<!-- RELATED:END -->
