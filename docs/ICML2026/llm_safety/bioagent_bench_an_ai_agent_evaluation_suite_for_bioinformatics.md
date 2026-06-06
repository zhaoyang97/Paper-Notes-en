---
title: >-
  [Paper Note] BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics
description: >-
  [ICML 2026][LLM Safety][agent evaluation] BioAgent Bench introduces an end-to-end evaluation suite for executing bioinformatics pipelines with LLM agents. Featuring 10 real-world tasks…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "agent evaluation"
  - "bioinformatics pipeline"
  - "LLM judge"
  - "robustness perturbation testing"
date: 2026-05-08
content_hash: e8f70181602bfc3f
---

# BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics

**Conference**: ICML 2026  
**arXiv**: [2601.21800](https://arxiv.org/abs/2601.21800)  
**Code**: https://github.com/bioagent-bench/bioagent-bench  
**Area**: LLM Agent / Benchmark / Bioinformatics  
**Keywords**: agent evaluation, bioinformatics pipeline, LLM judge, robustness perturbation testing

## TL;DR
BioAgent Bench introduces an end-to-end evaluation suite for executing bioinformatics pipelines with LLM agents. Featuring 10 real-world tasks, it evaluates 10 frontier/open-weight models across 3 agent harnesses. Using an LLM judge for scoring alongside three classes of perturbation tests (corrupted, decoy, and prompt-bloat), the study finds that while frontier models can complete over 90% of pipelines, their robustness remains a significant concern.

## Background & Motivation
**Background**: LLM agents have established benchmarks in software engineering (SWE-bench) and general tool-use (AgentBench, ToolBench). In the biomedical domain, benchmarks like BioML-bench, LAB-Bench, and BixBench exist. However, these often reduce tasks to simplified QA or code generation rather than long-running, integrated pipeline execution.

**Limitations of Prior Work**: Real bioinformatics workflows are highly complex, requiring the chaining of command-line tools, management of heterogeneous file formats, and interpretation of intermediate outputs. Evaluation is difficult due to the existence of multiple valid pipelines for the same data, the high impact of parameter selection on results, and steps that cannot be strictly assessed via binary pass/fail criteria. Hard-match evaluation methods like those in SWE-bench are insufficient.

**Key Challenge**: (1) Real bioinformatics tasks are long-running (hours) and resource-intensive (tens of GBs), whereas benchmarks need to be reproducible and scalable. (2) The existence of multiple valid solutions creates a conflict between automatic scoring and strict ground truth. (3) Clinical and IP-sensitive data cannot be sent to closed-source APIs, necessitating the evaluation of open-weight models, which typically lag behind frontier models.

**Goal**: (i) Create a set of end-to-end bioinformatics pipeline tasks that can run within reasonable resource constraints (<4h, <48GB RAM); (ii) Design a scoring protocol using LLM judges that accommodates multiple solutions; (iii) Introduce perturbation tests to verify agent robustness against corrupted data, decoys, and prompt-bloat; (iv) Systematically compare 5 closed-source and 5 open-weight models across 3 harnesses.

**Key Insight**: By restricting the task scale to "small organisms" (bacteria, viruses, yeast), reference data can be bundled directly as input files. This bypasses infrastructure hurdles like agents needing to download tens of GBs of genomic data, allowing the evaluation to focus strictly on pipeline orchestration capabilities.

**Core Idea**: Use "task prompt + input data + reference data + expected output format" as a unified task specification. An LLM judge compares the execution trace and outcome to provide step-level completion scores, supplemented by three types of perturbation tests to determine if high-level pipeline construction and low-level step reasoning are both sound.

## Method

### Overall Architecture
The benchmark consists of three components: (1) **Task Set**: 10 end-to-end tasks covering subfields like RNA-seq, variant calling, metagenomics, and experimental evolution. Each includes NL prompts, input/reference data, and ground-truth CSV/TSVs. (2) **Evaluation Harness**: Agents operate in a hashed sandbox using one of three harnesses (Claude Code, Codex CLI, or OpenCode), calling Python packages or specialized bioinformatics tools. (3) **LLM Grader**: GPT-5.1 analyzes input/reference paths, expected outcomes, agent outcomes, traces (file paths only), and grading rubrics. It outputs `steps_completed`, `steps_to_completion`, `final_result_reached`, `results_match`, and `f1_score`. The **Primary metric** is the completion rate, defined as the ratio of necessary steps passed to total steps.

Evaluation includes four settings: multi-trial stability, prompt-bloat (irrelevant task description expansion), corrupted-input (detecting intentionally damaged files), and decoy-input (resisting distracting files).

### Key Designs

1.  **Task Curation and Scale Constraints**:
    *   Function: Constructs end-to-end pipeline tasks runnable within <4h and <48GB RAM, covering main bioinformatics modalities.
    *   Mechanism: 10 tasks spanning bulk/single-cell RNA-seq, comparative genomics, variant calling (bacterial evolution, GIAB NA12878, cystic fibrosis), viral metagenomics, etc. Languages used include Python, R, and Bash. Four tasks are binary "verifiable." Reference data is included by selecting small organisms (e.g., E. coli, mouse Alzheimer's models).
    *   Design Motivation: Positioning the benchmark as a "software engineering" style rather than a "data analysis" style enables future RL/distillation uses. Scale constraints allow for large-scale reproducible evaluation, though at the cost of excluding human-scale workflows.

2.  **Grader (LLM Judge + Multi-dimensional Scoring)**:
    *   Function: Automates scoring in scenarios with multiple solutions, multiple steps, and massive intermediate products.
    *   Mechanism: GPT-5.1 acts as the grader, using input paths, expected/agent CSVs, trace trees, and rubrics. The rubric prioritizes "pipeline completion" over exact numerical precision. It outputs five fields, including `steps_completed` and `results_match`.
    *   Design Motivation: Bioinformatics tasks allow multiple valid pipelines (e.g., GATK4 vs. DeepVariant), making hardcoded ground truth impractical. Reviewing traces rather than just final output allows partial credit for correct logic despite formatting errors.

3.  **Perturbation Suite**:
    *   Function: Decouples "pipeline completion" from "true step-level biological reasoning."
    *   Mechanism: (i) **Multi-trial Consistency**: Runs tasks 4 times, calculating Jaccard for categorical results and Pearson for numerical values. (ii) **Prompt Bloat**: Measures the change in completion rate ($\Delta$) when adding irrelevant content. (iii) **Corrupted Input**: Checks if agents identify damaged FASTQ/BAM files. (iv) **Decoy Input**: Checks if agents are misled by extraneous files.
    *   Design Motivation: The authors hypothesize that high-level pipeline construction $\neq$ reliable step-level reasoning. Completion rates alone overestimate agent capability; perturbations serve as essential probes for understanding vs. pattern matching.

### Loss & Training
As a benchmark, no training is involved. Stability and robustness assessments use GPT-5.2 in the Codex CLI harness, with "high" reasoning effort enabled.

## Key Experimental Results

### Main Results
Average completion rates for 10 tasks in the vanilla setting (Codex CLI harness):

| Model Type | Model | Avg Completion% |
| :--- | :--- | :--- |
| Closed Frontier | Claude Opus 4.5 | **100** |
| Closed Frontier | Gemini 3 Pro / GPT-5.2 / Sonnet 4.5 | >90 |
| Best Open-weight | GLM-4.7 | 82.5 |
| Other Open-weight| Various | As low as ~65 |

**Planning vs. Execution**: Scores for "high-level pipeline plans" (rated 1-5 by GPT-5.1) correlate with end-to-end completion rates with a Pearson $r=0.61$. This indicates planning is necessary but not sufficient; for instance, Gemini-Pro-3 shows higher execution strength relative to its planning score, suggesting the bottleneck for open-weight models is often agentic capability over multiple rounds rather than domain knowledge.

### Ablation Study
Multi-trial stability (GPT-5.2 in Codex CLI, 4 trials per task):

| Task | Jaccard | Pearson | Note |
| :--- | :--- | :--- | :--- |
| transcript-quant | 1.000 | 1.000 | Fully deterministic |
| cystic-fibrosis | 1.000 | NA | High consistency |
| deseq | 0.978 | 0.995 | Highly stable |
| viral-metagenomics | 0.667 | 1.000 | Numerical stability, categorical jitter |
| metagenomics | 0.395 | 0.746 | Moderate |
| alzheimer | 0.160 | 0.219 | Unstable |
| comparative-genomics | 0.004 | NA | Highly inconsistent |
| evolution | 0.000 | NA | Completely inconsistent |

The average Jaccard is 0.43 and Pearson is 0.73, meaning categorical results overlap by less than half across 4 trials of the same task.

Perturbation tests (GPT-5.2 single trial; $\Delta\%$ represents completion change after prompt-bloat):

| Task | Corrupted Detected? | Decoy Resisted? | $\Delta$ Completion (%) |
| :--- | :--- | :--- | :--- |
| alzheimer-mouse | ✗ | ✗ | -12.5 |
| comparative-genomics | ✗ | ✓ | -20.0 |
| deseq | ✓ | ✗ | **-100.0** |
| evolution | ✓ | ✗ | +75.0 |
| giab | ✓ | ✗ | — |

### Key Findings
*   **Frontier models do not require complex scaffolding**: Claude Opus 4.5 achieved 100% completion using a basic Codex CLI, challenging the assumption that specialized agentic frameworks are always necessary.
*   **Pipeline construction $\neq$ step-level reasoning**: Significant result variance across trials (e.g., comparative-genomics) indicates that even if an agent "completes" a run, its intermediate decisions (normalization, statistical assumptions) are unstable.
*   **Low detection of corrupted data**: Agents often process corrupted inputs blindly. While `deseq` detection led to a 100% completion drop, this is arguably more desirable than producing silent errors.
*   **Weak decoy robustness**: Most agents are easily misled by decoy files, lacking the judgment to select correct inputs based on prior domain knowledge.
*   **Value of open-weight models in privacy scenarios**: While frontier models are stronger, open-weight models are essential for sensitive patient data. This study provides the first systematic baseline for them in bioinformatics.

## Highlights & Insights
*   **Pragmatic tradeoff between scale and feasibility**: Selecting small organisms to bundle reference data allows the benchmark to scale by avoiding massive infrastructure overhead (e.g., downloading 30GB human references).
*   **Trace-based grading**: Evaluating the file path tree rather than full file content protects sensitive data and reduces token consumption.
*   **Tri-perturbation design**: Separating corruption (cognition), decoy (attention), and bloat (robustness) provides a more granular failure mode analysis than a single "stress test."
*   **Comparison of 10 agents**: Provides the first reproducible leaderboard for both closed and open-weight agents in the bioinformatics domain.

## Limitations & Future Work
*   **Limited task scale**: The exclusion of human-scale workflows (e.g., 30× WGS variant calling) means infrastructure tasks like "finding, downloading, and staging" references are bypassed, potentially limiting generalization to production environments.
*   **LLM grading bias**: The grader (GPT-5.1/5.2) may favor specific trace patterns and belongs to the same generation as the evaluated agents, creating a circular dependency.
*   **Single-trial perturbation tests**: Conclusions are drawn from single runs, which may contain statistical noise. Future work should report 2D tables of perturbations across multiple seeds.
*   **Lack of open-weight robustness data**: Robustness tests were primarily performed on frontier models, leaving a gap in data for open-weight models.
*   **Minimal failure mode quantification**: The analysis mentions issues like error-correction loops or premature termination but lacks quantitative metrics such as trace length or loop counts.

## Related Work & Insights
*   **vs. SWE-bench (Jimenez et al., 2024)**: SWE-bench uses strict unit test pass/fail; BioAgent Bench uses soft LLM judging + step-level partial credit, better suited for scientific workflows with multiple valid solutions.
*   **vs. BioML-bench (Miller et al., 2025)**: BioML-bench focuses on ML processes (protein engineering, imaging); BioAgent Bench focuses on bioinformatics toolchain orchestration.
*   **vs. LAB-Bench (Laurent et al., 2024)**: LAB-Bench targets research skills via multiple-choice questions; BioAgent Bench emphasizes actual execution.
*   **vs. BixBench (Mitchener et al., 2025)**: BixBench targets data analysis reasoning; BioAgent Bench emphasizes end-to-end pipeline execution and robustness.
*   **Insight**: In any field with massive intermediate products and multiple valid steps (e.g., quantum chemistry, geosciences), the protocols used here—LLM judging, scale constraints, and tri-type perturbations—provide a blueprint for creating scalable agent benchmarks.

## Rating
*   Novelty: ⭐⭐⭐ (Pragmatic protocol, but LLM-judge paradigm is established)
*   Experimental Thoroughness: ⭐⭐⭐ (Broad model/harness coverage, but robustness trials are limited)
*   Writing Quality: ⭐⭐⭐⭐ (Clear distinctions between task/trial/grader/harness)
*   Value: ⭐⭐⭐⭐ (Provides the first systematic feasibility answer for bioinformatics agents)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ACIArena: Toward Unified Evaluation for Agent Cascading Injection](../../ACL2026/llm_safety/aciarena_toward_unified_evaluation_for_agent_cascading_injection.md)
- [\[ICML 2026\] Watermarking LLM Agent Trajectories (ACTHOOK)](watermarking_llm_agent_trajectories.md)
- [\[ICML 2026\] SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety](safeharbor_hierarchical_memory-augmented_guardrail_for_llm_agent_safety.md)
- [\[ICML 2026\] Position: Retire the "Positive Backdoor" Label -- Secret Alignment Requires Strict and Systematic Evaluation](position_retire_the_positive_backdoor_label_--_secret_alignment_requires_strict_.md)
- [\[ICLR 2026\] Unlearning Evaluation through Subset Statistical Independence](../../ICLR2026/llm_safety/unlearning_evaluation_through_subset_statistical_independence.md)

</div>

<!-- RELATED:END -->
