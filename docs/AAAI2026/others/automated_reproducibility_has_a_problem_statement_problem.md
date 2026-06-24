---
title: >-
  [Paper Note] Automated Reproducibility Has a Problem Statement Problem
description: >-
  [AAAI 2026][Reproducibility] This paper proposes a formalized problem definition of reproducibility grounded in the scientific method, representing empirical AI research as a hypothesis–experiment–interpretation graph structure. An LLM is used to automatically extract this structure from 20 papers, and the extracted results are validated through review by the original authors.
tags:
  - "AAAI 2026"
  - "Reproducibility"
  - "Scientific Method"
  - "Problem Formalization"
  - "LLM Automation"
  - "Empirical Study"
date: 2026-05-08
content_hash: 26bd84e96168c106
---

# Automated Reproducibility Has a Problem Statement Problem

**Conference**: AAAI 2026
**arXiv**: [2601.04226](https://arxiv.org/abs/2601.04226)  
**Code**: [Available](https://github.com/thijssnelleman/automated-reproducibility)  
**Area**: Other
**Keywords**: Reproducibility, Scientific Method, Problem Formalization, LLM Automation, Empirical Study

## TL;DR

This paper proposes a formalized problem definition of reproducibility grounded in the scientific method, representing empirical AI research as a hypothesis–experiment–interpretation graph structure. An LLM is used to automatically extract this structure from 20 papers, and the extracted results are validated through review by the original authors.

## Background & Motivation

Reproducibility is a cornerstone of the scientific method, yet independent replication demands substantial human effort. Recent work has attempted to automate this process, but each effort defines its own evaluation criteria, making cross-system comparison infeasible:

- **PaperBench**: Evaluates the replication ability of multiple LLMs; the best model achieves only a 43.4% average replication score, but rubrics are manually crafted per paper, lacking generalizability.
- **REPRO-bench**: Single-agent replication in social science; best accuracy is 36.6%; relies on code/data availability and is not generalizable across disciplines.
- **SciReplicate-Bench**: A dual-agent system (paper agent + code agent) that excels at algorithm summarization but performs poorly on implementation execution.
- **AutoReproduce**: Introduces a paper lineage algorithm but exhibits large gaps in code implementation, and introduces proprietary metrics that cannot be compared with other systems.

**Core Problem**: All prior work lacks a unified formal definition of reproducibility. Each introduces distinct evaluation metrics (rubric scores, SSRP metrics, CodeBLEU, etc.), precluding horizontal comparison across automated systems. This paper aims to propose a general problem-statement framework grounded in the scientific method.

## Method

### Overall Architecture

The reproducibility problem is modeled as a **directed graph structure**: any empirical AI study can be decomposed into a graph of the following elements:

```
Study → Hypotheses → Experiments → Outcomes → Analysis → Interpretations
```

Element definitions:
1. **Hypotheses**: Core claims of the study, either explicitly stated or derived post-hoc from research objectives.
2. **Experiments**: Comprising input datasets, methods/strategies, and the measurements produced.
3. **Analysis**: Simplified as result extraction based on defined metrics and statistical methods.
4. **Interpretations**: Support or refutation of hypotheses based on analyses across multiple experiments.

Graph flexibility: each experiment may be linked to multiple hypotheses; results may undergo multiple analyses; interpretations may draw on multiple analyses across experiments. **Interpretations are treated as relatively static**—allowing them to vary in automated settings would introduce uncontrollable uncertainty.

### Key Designs

**1. Post-hoc Hypothesis Construction**

AI papers rarely state testable hypotheses explicitly, relying instead on research questions and findings. Consequently, the LLM constructs post-hoc hypotheses from papers—from the perspective of independent replication, the expected experimental outcome is to reach the same conclusions as the original authors. This adaptation makes the framework applicable to papers that do not formally state hypotheses.

**2. LLM Automated Extraction Pipeline**

- Model: Google Gemini 2.5 Pro, temperature $t=0.0$
- Strategy: Few-shot prompting with examples indicating the sections where information is likely to appear and signal keywords
- Iterative refinement: Multiple rounds of prompt improvement on three candidate papers (dettmer2024weighted, Gundersen2025, snelleman2024edge)
- Note: Author feedback is used solely to improve prompts, not for few-shot learning

**3. Evaluation Procedure**

- Automated extraction performed on 20 papers spanning multiple AI subfields
- First authors of each paper review the LLM outputs
- Review scope: correcting wording errors, verifying hypothesis/experiment/interpretation links, and checking experimental details
- Scoring: hypotheses rated on a 7-point Likert scale; experiments and interpretations on a 5-point Likert scale

### Loss & Training

No model training is involved in this work. Core evaluation metrics:
- **Likert scale ratings**: Separate scores for hypotheses, experimental descriptions, experimental details, and result interpretations
- **Levenshtein edit distance**: Measures the extent of author corrections (character-level difference percentage)
- **Error rate statistics**: Proportion of errors across graph elements and their linking relations

## Key Experimental Results

### Main Results

**Table 1: Statistics of Evaluated Papers** (20 papers, token counts ranging from 1,291 to 11,095)

**Table 2: Method Error Rate Statistics**

| Error Type | Count | Proportion |
|---|---|---|
| Hypothesis statements requiring revision | 19 | 65.52% |
| Hypothesis edit distance (average) | 43 characters | 14.90% |
| Interpretation statements requiring revision | 9 | 24.32% |
| Interpretation edit distance (average) | 35 characters | 4.79% |
| Experiment–hypothesis links | 6 | 18.75% |
| Interpretation–hypothesis links | 0 | 0.00% |
| Interpretation–experiment links | 2 | 5.41% |
| Experiment metrics | 15 | 46.88% |
| Experiment statistical methods | 9 | 28.12% |
| Experiment strategies | 10 | 31.25% |
| Experiment results | 1103 | 69.63% |

**Overall success rate**: The method correctly captured all elements in 75% of the studies.

### Ablation Study

**Hypothesis extraction quality**:
- 6 cases failed to fully capture hypotheses, but all were at least partially correct
- Most complex case (BosEtAl25): 7 out of 9 hypotheses captured
- Although 65.52% required revision, the average correction was only 43 characters (14.90%), indicating minor modifications

**Experiment extraction quality**:
- 2 cases completely missed a specific experiment
- Numerical result errors were the highest (69.63%), primarily because visualized results (figures and charts) are difficult to extract accurately
- LLMs tend to extract from text rather than images, with non-vectorized PDF figures posing particular difficulty

### Key Findings

1. **Effect of paper length**: Longer papers (>10K tokens) are more prone to omissions, though this is not the sole factor—SkaEtAl25 (11,095 tokens) performed well.
2. **Interpretations outperform hypotheses**: The revision rate for interpretations (24.32%) is substantially lower than for hypotheses (65.52%), as interpretations more frequently quote the original text.
3. **Structured vs. visualized data**: Tabular data is extracted reliably, whereas chart data is highly unstable.
4. **Link extraction is strong**: Interpretation–hypothesis links show zero errors; interpretation–experiment links show only 5.41% error rate.

## Highlights & Insights

1. **Value of a unified framework**: This paper presents the first formalized problem definition for automated reproducibility grounded in the scientific method, enabling horizontal comparison across different systems.
2. **Quantifiability of the graph structure**: Partial replication can be quantified by counting how many nodes and edges in the graph are successfully reproduced.
3. **Large-scale author review**: Original authors of 20 papers participated in validation—a scale and level of rigor rare in this field.
4. **Applicable as a "front-end"**: The framework first extracts the problem structure, which is then passed to a code agent for replication execution, enabling effective task decomposition.

## Limitations & Future Work

1. **Poor extraction of visual results**: Accuracy for figures, box plots, and similar elements is low; multimodal processing improvements are needed.
2. **Insufficient numerical precision**: 69.63% of result data requires correction, representing a critical bottleneck for practical automated replication.
3. **Simple prompting strategy**: Only few-shot prompting is employed; more sophisticated strategies or fine-tuning could improve quality.
4. **No closed-loop integration**: The work only completes the "problem extraction" step and is not integrated with code generation or execution systems.
5. **Evaluation bias**: Author self-evaluation may lean positive; independent third-party validation is absent.
6. **Limited scale**: Validation on 20 papers is relatively small; larger-scale validation is needed to confirm generalizability.

## Related Work & Insights

- Complementary to systems such as PaperBench: this framework extracts structured problems, which a PaperBench-style agent then executes for replication.
- The graph structure is extensible: assigning replication difficulty weights to nodes would enable finer-grained scoring.
- Multi-agent division of labor: different subgraphs could be assigned in parallel to different agents.

## Rating

| Dimension | Score |
|---|---|
| Novelty | ★★★★☆ |
| Technical Depth | ★★★☆☆ |
| Experimental Thoroughness | ★★★★☆ |
| Practical Value | ★★★★☆ |
| Writing Quality | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] The Publication Choice Problem](the_publication_choice_problem.md)
- [\[NeurIPS 2025\] Put CASH on Bandits: A Max K-Armed Problem for Automated Machine Learning](../../NeurIPS2025/others/put_cash_on_bandits_a_max_k-armed_problem_for_automated_machine_learning.md)
- [\[AAAI 2026\] Judging by the Rules: Compliance-Aligned Framework for Modern Slavery Statement Monitoring](judging_by_the_rules_compliance-aligned_framework_for_modern_slavery_statement_m.md)
- [\[AAAI 2026\] Description Logics with Two Types of Definite Descriptions: Complexity, Expressiveness, and Automated Deduction](description_logics_with_two_types_of_definite_descriptions_complexity_expressive.md)
- [\[AAAI 2026\] On the Edge of Core (Non-)Emptiness: An Automated Reasoning Approach to Approval-Based Multi-Winner Voting](on_the_edge_of_core_non-emptiness_an_automated_reasoning_approach_to_approval-ba.md)

</div>

<!-- RELATED:END -->
