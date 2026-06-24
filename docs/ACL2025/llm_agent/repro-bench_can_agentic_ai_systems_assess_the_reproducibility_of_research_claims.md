---
title: >-
  [Paper Note] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?
description: >-
  [ACL 2025][LLM Agent] This paper proposes REPRO-Bench, a benchmark containing 112 social science paper instances, designed to evaluate the capability of AI agents in automatically assessing the reproducibility of papers. The best existing agent achieves an accuracy of only 21.4% (lower than the random guess baseline of 25%). REPRO-Agent, developed by the authors, improves the accuracy to 36.6% (a 71% relative improvement).
tags:
  - "ACL 2025"
  - "LLM Agent"
date: 2026-05-08
content_hash: 37bd901a1d5bc858
---

# REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?

**Conference**: ACL 2025  
**Code**: None  
**Area**: LLM Agents  

## TL;DR

This paper proposes REPRO-Bench, a benchmark containing 112 social science paper instances, designed to evaluate the capability of AI agents in automatically assessing the reproducibility of papers. The best existing agent achieves an accuracy of only 21.4% (lower than the random guess baseline of 25%). REPRO-Agent, developed by the authors, improves the accuracy to 36.6% (a 71% relative improvement).

## Background & Motivation

1. **Severe reproducibility crisis in social sciences**: Large-scale replication experiments indicate that less than 40% of the papers on the SSRP platform are considered fully reproducible, and 25% of the papers contain coding errors.
2. **Extremely high cost of manual assessment**: It required 347 social scientists to replicate 110 papers, and the Psychology Reproducibility Project took over 5 years to replicate 100 studies.
3. **AI agents demonstrate potential for automation**: As LLM-driven agents excel in complex tasks, automating reproducibility assessment has become feasible.
4. **Prior benchmarks suffer from three major limitations**: (1) They focus solely on code execution without verifying consistency between results and paper claims; (2) They oversimplify real-world scenarios (by providing preprocessed contexts); (3) They lack diversity in data formats and programming languages.
5. **Unique complexity of social science papers**: They involve multiple programming languages (Stata, R, Python, MATLAB) and data formats (.dta, .csv, .rda, .xlsx), requiring cross-domain knowledge integration.
6. **Lack of critical reasoning evaluation**: Existing benchmarks do not require identifying code/data inconsistencies, which is the core of reproducibility assessment.

## Method

### REPRO-Bench Task Definition

Each task instance contains three inputs:
1. **Paper PDF**: The full social science paper.
2. **Reproduction package**: Contains data, code, and documentation.
3. **List of primary findings**: Elements verified from replication reports (tables, figures, textual claims).

Agents are required to output a reproducibility score (1 to 4) based on the grading criteria:
- **Score 1**: Primary findings are not reproducible.
- **Score 2**: Minor inconsistencies or errors exist in the code (e.g., variable coding issues) but do not affect the core conclusions.
- **Score 3**: Analytical computations are correct, but there are minor display/reporting issues (e.g., rounding errors).
- **Score 4**: Primary findings are fully reproducible.

### Data Collection

112 papers were collected from 4 sources, following a unified selection criterion $\mathcal{C}$ (social science field, valid DOI, publicly accessible reproduction package, reliable replication report, replication runtime < 2 hours):

| Source | Quantity | Characteristics |
|------|------|------|
| Mass Reproducibility (Brodeur et al.) | 92 | Primary source, where most papers are basically reproducible |
| I4R Discussion Paper Series | 11 | Contains papers where key reproducibility issues were identified |
| Retraction Watch Database | 7 | Retracted papers (containing data/analytical errors) |
| Twitter/X | 2 | Papers highlighted with issues on social media |

Balanced score distribution: 56 papers with Scores 1+2 vs. 56 papers with Scores 3+4.

### Data Statistics

- Papers average 29 pages; reproduction packages average 4.2 GB and 142 files.
- Each paper contains an average of 5 primary findings (ranging from 1 to 19).
- Programming languages: 63 Stata, 25 R, 15 multilingual, 2 Python, 1 MATLAB, 1 Julia.
- Data formats: 34 .dta, 11 .csv, 10 .rda, 51 multi-format.

The Spearman correlation coefficients between paper characteristics (page count, file count, language/format diversity, etc.) and reproducibility scores are all $|\rho| < 0.1$, indicating that these factors do not influence reproducibility.

### Agent Environment Design

- The agent starts from a directory containing `paper.pdf` and `reproduction_package/`.
- All necessary software is pre-installed (Stata, MATLAB, LaTeX).
- Operating system command-line execution and package installation are freely allowed.
- Feedback is obtained via standard output/error (stdout/stderr) streams.
- API cost limit: $4 per task.

### Evaluated Agents

| Agent | Type | Characteristics |
|-------|------|-----------------|
| AutoGPT | Generalist Agent | Long-term planning, tool selection, behavior reflection |
| CORE-Agent | Scientific Paper Agent | Specifically designed for paper replication, incorporating VLM tools |
| SWE-Agent | Software Engineering Agent | Resolves GitHub Issues, includes an ACI interface |

All three agents use gpt-4o-2024-05-13.

## Key Experimental Results

### Main Results

| Agent | Accuracy (%) | Usability Rate (%) | Average Cost ($) |
|-------|-----------|-----------|-------------|
| AutoGPT | 20.5 | 60.7 | 2.03 |
| CORE-Agent | **21.4** | 46.4 | 2.00 |
| SWE-Agent | 1.8 (Adjusted 10.7) | 1.8 (Adjusted 19.6) | 1.20 |
| REPRO-Agent | **36.6** | **92.9** | — |

The best existing agent, CORE-Agent, achieves an accuracy of only 21.4%, which is even lower than the 25% random guess baseline in a 4-choice setting. REPRO-Agent, through three major strategies (structured templates, virtual score rollback, and few-shot examples), achieves a 36.6% accuracy (a 71% relative improvement) and a 92.9% usability rate.

### Accuracy Analysis by Programming Language

- **R tasks outperform Stata tasks**: Since R is an open-source language, LLMs have better knowledge coverage for it.
- **Multilingual tasks perform worst**: Agents struggle to ensure consistent execution across multiple languages.
- **Multi-format data does not hinder performance**: Agents effectively utilize data loaders to handle various formats.

### Performance by Reproducibility Score

All agents perform best on papers with Score 4 (fully reproducible), but struggle with fine-grained judgments for Score 2 and Score 3. Agents tend to produce binary outcomes rather than deeply investigating the sources of inconsistency.

### Failure Mode Classification (Misclassification instances of Score 4 and Score 1)

| Failure Type | Description | Proportion |
|---------|------|------|
| Type 4: File localization failure | Agent fails to correctly infer the directory structure to locate data files | **Highest** |
| Type 3: Dependency installation failure | Unable to correctly install required libraries | Second highest |
| Type 2: Missed terminal output | Stata error messages are saved in log files instead of the terminal, causing agent misjudgment | Medium |
| Type 1: Incorrect result comparison | The comparison script written by the agent itself is incorrect | Lowest |

### Reasons for Unidentified Inconsistency (Score 1 misclassified as Score 4)

1. Agents do not strictly follow the complete workflow—less than 42% of the cases include both code execution/inspection and result comparison phases.
2. During code inspection, agents often read the entire file instead of focusing on key snippets, making it difficult to locate errors in long code contexts.

## Highlights & Insights

- **Filling the gap in realistic evaluation benchmarks**: This is the first benchmark that requires agents to perform end-to-end evaluation of paper reproducibility, unlike existing benchmarks that only focus on code execution.
- **Real-world complexity**: Direct use of actual social science papers and reproduction packages, involving multilingual code (Stata/R/Python/MATLAB) and diverse data formats.
- **Systematic failure analysis**: Characterizes agent failures into 4 highly actionable error types, pointing to clear directions for agent improvement.
- **REPRO-Agent validates analytical value**: Achieving a 71% relative improvement using three strategies guided by empirical analysis proves the efficacy of targeted agent enhancement.
- **Impact on social sciences**: Legal experts confirm that the benchmark captures representative patterns of social science research, which can promote better practices in code and data management.

## Limitations

- **Overall accuracy remains low**: Even REPRO-Agent's 36.6% accuracy is far from practical application, indicating that automated reproducibility assessment still requires substantial advancements.
- **Lack of task instance variations**: Multiple versions of the same paper (including intentional errors/corrections) were not introduced, preventing a fine-grained evaluation of detection capabilities.
- **Limited to the social science domain**: The benchmark has not been extended to other disciplines facing similar reproducibility challenges, such as biology.
- **Single LLM backend**: All three agents utilize gpt-4o, leaving the impact of other models or larger context windows unexplored.
- **Cost limits deep exploration**: A $4 API budget per task might be insufficient to support deeper code analysis and comprehensive result comparison.

## Rating

- **Novelty** ⭐⭐⭐⭐: The first agent benchmark for evaluating social science reproducibility, featuring distinctive task design and data sourcing.
- **Technical Depth** ⭐⭐⭐: The benchmark construction process is rigorous, though the technical improvements of REPRO-Agent itself are relatively straightforward (templates, rollback, and few-shot examples).
- **Experimental Thoroughness** ⭐⭐⭐⭐: Built upon three representative agents, detailed qualitative analysis, failure categorization, and validation of improvements, constructing a comprehensive evaluation pipeline.
- **Writing Quality** ⭐⭐⭐⭐: Clear structure, explicit data collection standards, and solid statistical analysis.
- **Value** ⭐⭐⭐⭐⭐: Directly addresses real-world issues faced by the social science community, with the benchmark being publicly available, which has significant guiding value for agent development.
- **Overall Rating** ⭐⭐⭐⭐ (4/5)

## Related Papers

- Brodeur et al. (2024): Large-scale social science reproducibility experiments
- SWE-Bench (Jimenez et al., 2024): Software engineering agent benchmark
- CORE-Bench (Siegel et al., 2024): Scientific paper replication benchmark
- AutoGPT (Gravitas, 2023): Generalist autonomous agent

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_social_science_.md)
- [\[ACL 2025\] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)
- [\[ACL 2025\] AndroidGen: Building an Android Language Agent under Data Scarcity](androidgen_agent_data_scarcity.md)
- [\[ACL 2025\] Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems](agentic_reward_modeling_integrating_human_preferences_with_verifiable_correctnes.md)
- [\[ACL 2025\] Can a Single Model Master Both Multi-turn Conversations and Tool Use? CoALM: A Unified Conversational Agentic Language Model](can_a_single_model_master_both_multi-turn_conversations_and_tool_use_coalm_a_uni.md)

</div>

<!-- RELATED:END -->
