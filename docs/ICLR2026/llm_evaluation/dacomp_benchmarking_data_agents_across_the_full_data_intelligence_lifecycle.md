---
title: >-
  [Paper Note] DAComp: Benchmarking Data Agents across the Full Data Intelligence Lifecycle
description: >-
  [ICLR 2026][LLM Evaluation][LLM-judge] DAComp is a 210-task benchmark covering the enterprise-grade "full data intelligence lifecycle." It decomposes data intelligence into a "Hard axis" for warehouse-level Data Engineering (DE) and a "Soft axis" for open-ended Data Analysis (DA). These are evaluated using executable multi-metrics and hierarchical rubric-ba
tags:
  - ICLR 2026
  - LLM Evaluation
  - LLM-judge
date: 2026-05-08
content_hash: 03d08e6f04b6ec18
---
# DAComp: Benchmarking Data Agents across the Full Data Intelligence Lifecycle

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=EtzJy9yI5J](https://openreview.net/forum?id=EtzJy9yI5J)  
**Code**: https://da-comp.github.io  
**Area**: LLM Evaluation / Data Agents / Benchmark  
**Keywords**: Data Agents, Data Engineering, Data Analysis, Warehouse-level Evaluation, LLM-judge  

## TL;DR
DAComp is a 210-task benchmark covering the enterprise-grade "full data intelligence lifecycle." It decomposes data intelligence into a "Hard axis" for warehouse-level Data Engineering (DE) and a "Soft axis" for open-ended Data Analysis (DA). These are evaluated using executable multi-metrics and hierarchical rubric-based LLM-judging, respectively. The study found that even GPT-5's strict success rate on DE is only 20%, and its DA average is below 50%, exposing critical weaknesses in current data agents regarding holistic pipeline orchestration and open-ended reasoning.

## Background & Motivation
**Background**: LLMs have demonstrated powerful reasoning and code-generation capabilities in tasks such as text-to-SQL, software engineering, and computer control. This has catalyzed "Data Agents"—agents intended to automatically transform raw data into actionable insights. Evaluating these agents requires benchmarks that accurately reflect enterprise complexity.

**Limitations of Prior Work**: Existing data-related benchmarks typically fragment complex workflows into isolated sub-tasks. One category (BIRD, Spider 2.0, DA-Code) simplifies data engineering into "generating a single SQL statement or script" with small schemas and minimal code (dozens to hundreds of lines), failing to represent the engineering effort of building multi-layer pipelines across dozens of tables and thousands of columns. Another category (DSBench, BLADE, DABStep) compresses data analysis into "deterministic Q&A with standard answers," losing the openness of planning, exploration, and synthesized insights found in real-world analysis.

**Key Challenge**: Real-world enterprise data intelligence spans two fundamentally different capabilities. On one hand, it involves systematic large-scale coding and pipeline maintenance under evolving requirements (Engineering Practicality, Hard axis). On the other hand, it requires strategic planning, iterative exploration, and synthesis of actionable recommendations for open-ended business problems (Analysis Openness, Soft axis). Existing benchmarks cover at most one axis, failing to evaluate whether an agent can complete the end-to-end data intelligence lifecycle.

**Goal**: To construct a benchmark that evaluates both the Hard and Soft axes at a level close to real enterprise complexity, covering the full chain from architectural design and pipeline implementation to system evolution and open-ended analysis, while designing reliable evaluation protocols for these non-trivial tasks.

**Key Insight**: The authors formalize the real responsibilities of data engineers and analysts into two task families: DAComp-DE and DAComp-DA. Tasks are built on industrial SaaS schemas and real analysis problems, with evaluation protocols—executable for deterministic engineering and hierarchical rubrics for open analysis—tailored to their specific natures.

**Core Idea**: Use a 210-task benchmark to fully characterize the "full data intelligence lifecycle." Warehouse-level DE is quantified with executable multi-metrics, while open-ended DA is evaluated with human-verified hierarchical rubric-based LLM-judging to precisely diagnose the bottlenecks of current data agents.

## Method

### Overall Architecture
DAComp is a benchmark rather than a single model. Its "method" lies in how it defines tasks, evaluates performance, and provides annotations. It is organized along two axes: **Hard axis = DAComp-DE (Warehouse-level Data Engineering)** and **Soft axis = DAComp-DA (Open-ended Data Analysis)**, totaling 210 tasks (DE-Arch 30 / DE-Impl 30 / DE-Evol 50 / DA 100). A high-quality Chinese adaptation, DAComp-zh, was also released.

DE models data engineering as a DAG construction process from raw sources through staging, core, and mart layers to produce semantic labels. This is formalized as $(S, C^\star) = \pi_{de}(Q_{de}, C_0, B)$, where $Q_{de}$ is the high-level requirement, $S$ is the engineering specification, $C_0$ is the initial warehouse, $B$ is the database, and $C^\star$ is the final DE warehouse. Tasks are categorized into Architecture, Implementation, and Evolution. DA models open-ended analysis as $O = \pi_{da}(Q_{da}, D)$: given semantic data $D$ and a business problem $Q_{da}$, the agent autonomously writes SQL/Python for aggregation, interprets intermediate results, and synthesizes reports including insights and visualizations, allowing for multiple valid paths without a unique gold standard.

Evaluation is branched: deterministic DE-Impl/DE-Evol use executable multi-metrics, while open-ended DA and DE-Arch use hierarchical rubric-based LLM-judging. The benchmark was refined by 8 experts through an annotation pipeline of "data collection → task design → evaluation construction."

### Key Designs

**1. Dual-axis Task Partitioning: Separating Engineering and Analysis**
Addressing the limitation that existing benchmarks only cover one axis or fragment tasks, DAComp distinguishes between the Hard and Soft axes. DAComp-DE is the first benchmark to introduce **warehouse-level** data engineering: agents must orchestrate multi-layer pipelines on industrial schemas (avg. 32 tables, 412 columns). It includes **DE-Architecture** (outputting specs $S$), **DE-Implementation** (building $C^\star$ from scratch), and **DE-Evolution** (modifying $C_0$ to $C^\star$). DAComp-DA introduces real-world open-ended analysis where agents must plan multi-step analyses and synthesize insights. Experiments confirm that engineering and analysis are independent capabilities.

**2. Executable Multi-metrics: Diagnosing Pipeline Orchestration Bottlenecks**
For deterministic tasks, the authors designed three executable metrics with increasing strictness. The **Component Score (CS)**, $\text{CS} = \sum_j w_j s_j$, evaluates each DAG node in isolation using gold upstream inputs to measure SQL generation capability. The **Cascading Failure Score (CFS)** evaluates nodes in DAG order; if an upstream dependency fails, the current node score is zeroed, measuring end-to-end data integrity. The **Success Rate (SR)**, $\text{SR} = \mathbb{I}[\forall j: s_j = 1]$, requires a flawless warehouse. The gap between CS and CFS/SR demonstrates that agents struggle with overall pipeline orchestration rather than simple code generation.

**3. Hierarchical Rubric + GSB LLM-judge: Fair Evaluation for Open-ended Analysis**
For DA and DE-Arch, a hierarchical rubric framework is used for LLM-judging. It scores six dimensions: Completeness, Accuracy, and Insightfulness via rubric; Readability, Analytical Depth, and Visualization via Good–Same–Bad (GSB). The rubric decomposes a question $Q$ into requirements and sub-requirements, enumerating multiple equivalent paths for each. The judge selects the best-matching path and aggregates scores:
$$\text{Score}_{rubric}(O, R) = \frac{\sum_{k=1}^{N} s_k}{\sum_{k=1}^{N} w_k}, \quad s_k = \Lambda(c_k, O) \in [0, w_k]$$
GSB compares the output against five baseline reports:
$$\text{Score}_{gsb}(O, O_{base}) = \frac{\max(0, |G| - |B|)}{|G| + |S| + |B|}$$
The final DA score is $\text{Score}_{da} = \alpha \cdot \text{Score}_{rubric} + (1-\alpha) \cdot \text{Score}_{gsb}$ (with $\alpha=0.6$).

**4. 8-Expert Annotation Pipeline: Reverse-engineering Industrial Assets**
Task authenticity stems from a three-stage annotation process. **Data Collection** uses open-license assets (Apache-2.0, MIT). DE collects 73 enterprise SaaS schemas, while DA uses 100 complex databases. **Task Design** involves expert voting for DA problems and professional data engineers writing DE-Evolution requirements. **Evaluation Construction** includes building hierarchical rubrics for DA and automated execution scripts for DE-Impl/Evol to capture incremental correctness.

## Key Experimental Results

### Main Results
Evaluation was conducted using the OpenHands framework with a custom DA-Agent. DE scores use CFS for Impl/Evol.

DAComp-DE (English Version, Table 3) Core Results:

| Model | DE-Arch | Impl CS | Evol SR@8 | DE Score |
|------|---------|---------|-----------|----------|
| GPT-5 | 63.93 | 39.87 | 20.00 | **43.45** |
| Gemini-2.5-Pro | 51.96 | 36.88 | 8.00 | 32.88 |
| Qwen3-Coder | 51.43 | 32.86 | 12.00 | 32.80 |
| DeepSeek-V3.1 | 52.66 | 30.73 | 10.00 | 31.41 |
| Qwen3-235B-A22B | 50.73 | 20.15 | 2.00 | 20.15 |

Even GPT-5 achieves a DE Score of only ~43% and a strict success rate of 20%. Specialized open-source models (Qwen3-Coder, DeepSeek-V3.1) are competitive with Gemini-2.5-Pro.

DAComp-DA (DA-Agent baseline, Table 5) Core Results:

| Model | Completeness | Accuracy | Visualization | DA Score |
|------|-------------|----------|---------------|----------|
| GPT-5 | 64.23 | 43.81 | 27.44 | **50.84** |
| Kimi-K2 | 52.31 | 33.56 | 14.40 | 41.89 |
| Gemini-2.5-Pro | 45.43 | 30.30 | 13.40 | 34.70 |
| DeepSeek-V3.1 | 48.74 | 32.97 | 11.45 | 34.33 |
| Qwen3-8B | 9.89 | 4.12 | 0.15 | 4.47 |

In DA tasks, most models scored below 50%. Visualization scores were universally low across all models.

### Key Findings
- **Pipeline orchestration is harder than code generation**: The significant drop from CS to SR on DE proves the bottleneck is end-to-end orchestration.
- **Engineering and analysis are independent abilities**: Model rankings differ across the two axes, validating the dual-axis design.
- **Scale is not a panacea**: Large models like Qwen3-235B-A22B struggled on Impl/Evol, suggesting warehouse-level engineering is a structural challenge.
- **Cross-lingual consistency**: Results on DAComp-zh are highly consistent with the English version.

## Highlights & Insights
- **"Full Lifecycle" design reflects enterprise reality**: Covering planning, construction, and evolution/maintenance (averaging ~1718 lines of changes across 13 files) fills a gap in previous benchmarks.
- **Three-tier metrics as diagnostic tools**: The separation of CS/CFS/SR allows researchers to pinpoint whether an agent fails at individual components or holistic orchestration.
- **Hierarchical rubric path enumeration solves fairness**: By allowing multiple correct paths and matching the best one, the evaluation accommodates methodological diversity without sacrificing rigor.
- **Complementary Rubric and GSB**: The rubric handles decomposable dimensions like Accuracy, while GSB addresses subjective overall quality like Readability.

## Limitations & Future Work
- **Potential bias in LLM-judging**: Absolute scores for open-ended DA depend on the choice of judge (Gemini-2.5-Flash), which may have its own preferences.
- **Synthetic data limits**: While schemas are industrial, synthetic data may not capture the "messiness" or long-tail distributions of real production data.
- **Weak visualization evaluation**: The low visualization scores suggest the need for more specialized chart assessment methods.
- **Task scale**: With 210 tasks, it is smaller than benchmarks like SWE-Bench; expanding to more schemas is costly due to manual reverse-engineering.

## Related Work & Insights
- **vs SWE-Bench / WebArena / OSWorld**: These focus on general agent benchmarks using execution but lack the "analysis openness" specific to data intelligence.
- **vs BIRD / Spider 2.0 / DA-Code**: These are text-to-SQL or single-script benchmarks; DAComp raises the complexity to warehouse-level DAG orchestration.
- **vs DSBench / BLADE / DABStep**: Unlike these deterministic or small-schema benchmarks, DAComp-DA supports multi-path open-ended evaluation and emphasizes autonomous visualization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First benchmark to dual-axis the full data intelligence lifecycle and introduce warehouse-level DE.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple models, frameworks, and languages with judge-consistency validation.
- Writing Quality: ⭐⭐⭐⭐ Clear task definitions and protocols, though some details are housed in the appendices.
- Value: ⭐⭐⭐⭐⭐ Provides a rigorous testbed for diagnosing enterprise-grade data agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[ICLR 2026\] Detecting Data Contamination in LLMs via In-Context Learning](detecting_data_contamination_in_llms_via_in-context_learning.md)
- [\[ICLR 2026\] ASIDE: Architectural Separation of Instructions and Data in Language Models](aside_architectural_separation_of_instructions_and_data_in_language_models.md)
- [\[ICML 2026\] Nonparametric LLM Evaluation from Preference Data](../../ICML2026/llm_evaluation/nonparametric_llm_evaluation_from_preference_data.md)
- [\[ICLR 2026\] Rethinking LLM Evaluation: Can We Evaluate LLMs with 200× Less Data?](rethinking_llm_evaluation_can_we_evaluate_llms_with_200_less_data.md)
- [\[ICLR 2026\] DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science](dare-bench_evaluating_modeling_and_instruction_fidelity_of_llms_in_data_science.md)

</div>

<!-- RELATED:END -->
