---
title: >-
  [Paper Note] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science?
description: >-
  [ACL2025][LLM Agent][AI Agent] This work introduces REPRO-Bench, which comprises 112 reproducibility assessment tasks for social science papers. The study reveals that existing AI agents (with the highest accuracy at only 21.4%) are far from capable of automating this process. Consequently, REPRO-Agent is developed to improve the accuracy to 36.6%.
tags:
  - "ACL2025"
  - "LLM Agent"
  - "AI Agent"
  - "reproducibility assessment"
  - "social science"
  - "benchmark"
  - "code execution"
date: 2026-05-08
content_hash: 69b5ea55530b1849
---

# REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science?

**Conference**: ACL2025  
**arXiv**: [2507.18901](https://arxiv.org/abs/2507.18901)  
**Code**: [GitHub](https://github.com/uiuc-kang-lab/REPRO-Bench)  
**Area**: LLM Agent  
**Keywords**: AI Agent, reproducibility assessment, social science, benchmark, code execution

## TL;DR
This work introduces REPRO-Bench, which comprises 112 reproducibility assessment tasks for social science papers. The study reveals that existing AI agents (with the highest accuracy at only 21.4%) are far from capable of automating this process. Consequently, REPRO-Agent is developed to improve the accuracy to 36.6%.

## Background & Motivation
1. **Severe reproducibility crisis**: The social science field has long faced a "reproducibility crisis", where fewer than 40% of papers on the SSRP platform achieve full reproducibility (Level 10), and 25% contain coding errors.
2. **Enormous cost of manual assessment**: The Mass Reproduction project mobilized 347 social scientists to spend several years replicating 110 papers, illustrating unsustainable labor and time costs.
3. **Unverified capabilities of AI agents**: LLM-driven agents (such as AutoGPT, SWE-Agent, etc.) have performed outstandingly in software engineering tasks, yet whether they can handle the complex reproduction workflows specific to social sciences remains to be systematically evaluated.
4. **Overly strong assumptions of existing benchmarks**: Both SciCode and CORE-Bench assume that papers are fully reproducible, failing to evaluate an agent's ability to identify inconsistencies.
5. **Over-simplified contexts**: Pre-existing benchmarks provide pre-extracted specific steps, whereas realistic reproduction requires agents to autonomously extract information from raw PDFs and data packages.
6. **Monotonous languages and formats**: Existing benchmarks only involve a single programming language (Python/R), while social science papers frequently mix multiple languages like Stata, R, and MATLAB, alongside diverse data formats such as .dta, .csv, and .rda.

## Method

### REPRO-Bench Construction
- **Data Sources**: 112 papers are collected from 4 sources: (1) 92 from Mass Reproduction, (2) 11 from I4R Discussion Papers, (3) 7 from Retraction Watch, and (4) 2 from Twitter/X, ensuring a balanced coverage of reproducible and non-reproducible papers.
- **Scoring System**: A 4-level reproducibility rating is established: 1 = main findings are irreproducible, 2 = codes have minor inconsistencies/errors, 3 = only display-level issues exist (e.g., rounding), 4 = fully reproducible. The distribution is 20/36/8/48, with exactly 56 papers each for 1+2 and 3+4.
- **Task Definition**: The agent receives the paper PDF + reproduction package (code/data/documentation) + list of main findings, and is tasked with outputting an integer score from 1 to 4.

### Evaluated Agents
- **AutoGPT**: A general-purpose agent equipped with long-term planning, tool selection, and reflection capabilities.
- **CORE-Agent**: Specifically designed for paper reproduction, containing Visual Language Model (VLM) tools, adapted for its hard task version.
- **SWE-Agent**: A software engineering agent that utilizes the Agent-Computer Interface (ACI) to execute debugging.
- All three use GPT-4o (2024-05-13), with each task budget capped at a $4 API cost.

### Agent Environment & Evaluation Protocol
- **Environment**: The starting directory of each agent contains a paper.pdf and a reproduction_package/ subdirectory. The user prompt includes a list of the main findings to be reproduced.
- **Pre-installed Software**: Tools commonly used in social sciences, such as Stata, MATLAB, and LaTeX, are pre-installed, with version specifications detailed in the task description.
- **Evaluation Metrics**: The primary metric is accuracy (whether the generated score matches the ground truth); the auxiliary metric is applicability rate (whether the output file format and location are correct).
- **Successful Agent Workflow** (4 phases): Phase 1: Environment Understanding (listing directories, reading paper, reading README) → Phase 2: Code Review (checking for inconsistencies) → Phase 3: Script Editing & Execution → Phase 4: Comparison of Executed Results against Original Results.

### REPRO-Agent Improvement Strategies
Based on a systematic analysis of failure modes, three strategies are designed: (1) constructing structured templates based on successful cases to guide planning; (2) introducing dummy scores as a fallback mechanism to improve the applicability rate; (3) incorporating common error types as few-shot examples to enhance in-context learning.

### Benchmark Statistical Characteristics
- Papers average 29 pages, with reproduction packages averaging 4.2GB in size and 142 files in quantity.
- Each paper has an average of 5 main findings (range 1-19, standard deviation 4).
- Programming languages: Stata (63 papers), R (25 papers), multi-lingual (15 papers), and a few in MATLAB/Julia/Python.
- Data formats: .dta (34 papers), .csv (11 papers), multi-format (51 papers).
- Spearman correlation analysis ($|\rho| < 0.1$) confirms that the number of pages, number of findings, package size, and language/format diversity are uncorrelated with reproducibility scores.

## Key Experimental Results

### Table 1: Overall Performance and Cost of Agents

| Agent | Accuracy (%) | Applicability Rate (%) | Average Cost ($) |
|:--|:--:|:--:|:--:|
| AutoGPT | 20.5 | 60.7 | 2.03 |
| CORE-Agent | 21.4 | 46.4 | 2.00 |
| SWE-Agent | 1.8 (Adjusted 10.7) | 1.8 (Adjusted 19.6) | 1.20 |
| **REPRO-Agent** | **36.6** | **92.9** | - |

- The best existing agent achieves an accuracy of only 21.4%, which is even lower than random guessing (25%), highlighting the difficulty of the task.
- REPRO-Agent relative to CORE-Agent achieves a relative improvement of 71% in accuracy and 53% in applicability rate.

### Table 2: Performance Analysis across Different Dimensions

| Analytical Dimension | Key Finding |
|:--|:--|
| By Score | Agents perform significantly better on Score 4 (fully reproducible) and tend to make binary judgments |
| By Language | Accuracy on R tasks is much higher than Stata, likely because LLMs are more familiar with R |
| Multi-lingual | Performance drops in tasks involving multiple languages, as agents struggle with cross-language execution consistency |
| Data Format | Single-format and multi-format tasks perform similarly (54% vs 52%), indicating format diversity is not a bottleneck |

### Categorization of Failure Reasons
- **Type 1** (Result Comparison Error): The comparison script written by the agent is faulty, misclassifying consistent results as mismatches.
- **Type 2** (Failure to Capture Output): Stata writes error messages to log files instead of the terminal, which the agent fails to read.
- **Type 3** (Library Installation Failure): Failure to correctly install dependencies.
- **Type 4** (File Location Failure, Most Common): The complex folder structure of the reproduction package prevents the agent from correctly locating data files.

## Highlights
- This is the first end-to-end agent benchmark tailored for the **reproducibility assessment of social sciences**, with 112 task instances tightly matching real-world scenarios.
- The 4-level scoring system provides fine-grained coverage of different reproducibility levels, circumventing simple binary judgments.
- A **systematic categorization** of agent failure modes (4 types) provides clear directions for subsequent improvements.
- REPRO-Agent validates the closed-loop efficacy of "from failure analysis to improved strategy," yielding a compelling 71% relative improvement.

## Limitations & Future Work
- Each paper features only one task version, lacking variants with intentional errors or corrected code to further test the agent's robustness.
- Harder scenarios (e.g., masking experimental result data points and only providing raw data) are left uninvestigated.
- The benchmark only covers social sciences and is not yet expanded to domains like biology that face similar reproducibility challenges.
- The 36.6% accuracy of REPRO-Agent still leaves a massive gap for practical application.
- Although the extraction of main findings during the annotation process was verified by consensus, inter-annotator agreement metrics were not reported.

## Related Work
- **SciCode**: Converts paper findings into coding tasks under the assumption that the findings are fully correct, only involving Python; whereas REPRO-Bench makes no such assumption and covers multiple languages.
- **CORE-Bench**: Provides specific predefined steps and takes execution results as the ground truth; whereas REPRO-Bench requires agents to autonomously extract details from PDFs and check for consistency.
- **SWE-Bench**: Focuses on resolving GitHub issues within standardized code repositories; whereas REPRO-Bench's reproduction packages have unstandardized directory structures, demanding higher navigation capabilities from the agents.
- **Mass Reproduction (Brodeur et al., 2024)**: REPRO-Bench is directly built upon this large-scale manual reproduction effort, employing its reproduction reports as ground truth.

## Rating
- Novelty: ⭐⭐⭐⭐ — First systematic evaluation of AI agents' capabilities in social science reproducibility assessment, featuring a unique task formulation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive with 3+1 agents, multi-dimensional analysis, and thorough failure classification, though lacks comparison across more LLM backbones.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, with elaborate descriptions of data collection and grading criteria.
- Value: ⭐⭐⭐⭐ — Fills the benchmark gap in AI-assisted reproducibility evaluation, bearing practical significance for social science research workflows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)
- [\[ACL 2025\] LegalAgentBench: Evaluating LLM Agents in Legal Domain](legalagentbench_evaluating_llm_agents_in_legal_domain.md)
- [\[NeurIPS 2025\] MLRC-Bench: Can Language Agents Solve Machine Learning Research Challenges?](../../NeurIPS2025/llm_agent/mlrc-bench_can_language_agents_solve_machine_learning_research_challenges.md)
- [\[ACL 2025\] PaSa: An LLM Agent for Comprehensive Academic Paper Search](pasa_an_llm_agent_for_comprehensive_academic_paper_search.md)
- [\[ACL 2025\] GuideBench: Benchmarking Domain-Oriented Guideline Following for LLM Agents](guidebench_guideline_following.md)

</div>

<!-- RELATED:END -->
