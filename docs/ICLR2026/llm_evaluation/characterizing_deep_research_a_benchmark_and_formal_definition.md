---
title: >-
  [Paper Note] Characterizing Deep Research: A Benchmark and Formal Definition
description: >-
  [ICLR2026][LLM Evaluation][Deep Research] This paper provides a formal definition for "Deep Research (DR)," a task frequently claimed by various models but never strictly defined. The core is identified as "high fan-out" during the search process rather than merely "outputting long reports." Accordingly, the authors constructed LIVEDRBENCH, a benchmark of 100 open-web tasks, using claim-based Precision/Recall for objective scoring. It reveals that the strongest current system…
tags:
  - "ICLR2026"
  - "LLM Evaluation"
  - "Deep Research"
  - "Formal Definition"
  - "Claim Evaluation"
  - "Problem Inversion"
  - "Information Synthesis"
date: 2026-05-08
content_hash: 30f8a3719971a206
---

# Characterizing Deep Research: A Benchmark and Formal Definition

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=5EmpOCq1Ql](https://openreview.net/forum?id=5EmpOCq1Ql)  
**Code**: https://github.com/microsoft/LiveDRBench  
**Area**: LLM Evaluation / Deep Research / Agent benchmark  
**Keywords**: Deep Research, Formal Definition, Claim Evaluation, Problem Inversion, Information Synthesis

## TL;DR
This paper provides a formal definition for "Deep Research (DR)," a task frequently claimed by various models but never strictly defined. The core is identified as "high fan-out" during the search process rather than merely "outputting long reports." Accordingly, the authors constructed LIVEDRBENCH, a benchmark of 100 open-web tasks, using claim-based Precision/Recall for objective scoring. It reveals that the strongest current system, OpenAI DR, achieves an average F1 of only 0.55, with systems generally covering only about half of the necessary search queries.

## Background & Motivation
**Background**: From late 2024 to 2025, Google, OpenAI, Perplexity, and Grok launched "Deep Research" products, claiming to replace human experts in hours-long research, report writing, and needle-in-a-haystack QA. The open-source community followed with agentic implementations like HuggingFace DR, WebThinker, and DeerFlow.

**Limitations of Prior Work**: Despite the buzz, "Deep Research" has never been clearly defined. It is vaguely understood as "time-consuming report writing" and assumed to be harder than multi-hop QA, but this is neither precise nor measurable. Furthermore, task difficulty depends on the corpus: a task like "listing all Oscar-winning movies adapted from female authors" seems like hard research, but if a single webpage already lists the answer, it collapses into a simple retrieval. The lack of a formal definition makes it impossible to objectively evaluate DR models and measure progress.

**Key Challenge**: Existing DR evaluations are either subjective (using LLM-as-a-judge for "comprehensiveness/insight/readability," e.g., DeepResearch Bench), use non-public data (FutureSearch), or use static corpora that cannot compare closed-source and open-source systems. "Report quality" as a metric conflates writing style and phrasing with actual reasoning capability.

**Goal**: (1) Provide a formal definition of DR that distinguishes it from other reasoning-intensive tasks; (2) Construct an objective, reproducible, and sustainably updated benchmark covering broad domains.

**Key Insight**: The authors argue that the true difficulty of DR lies not in "generating long reports" but in the preceding **information synthesis**—finding, processing, and combining relevant information units from the corpus. Thus, the DR output is abstracted as an intermediate representation composed of claims, decoupling the "reasoning challenge" from "surface-level report generation."

**Core Idea**: Define DR along two dimensions: "search intensity $\times$ reasoning intensity," and use a nested list of claims as an objectively scorable intermediate output representation.

## Method

### Overall Architecture
The paper focuses on "definition + benchmark" rather than training a model. The logic is: first, formalize the DR task as an extreme point in a 2D space (high search, high reasoning) and abstract its output into a set of claims synthesized on a DAG. Second, design a claim-based Precision/Recall metric for objective scoring. Third, use "problem inversion" to generate DR questions by reversing long-document reasoning problems, resulting in 100 tasks across 8 categories for LIVEDRBENCH. Finally, evaluate the DR products of OpenAI/Perplexity/Gemini and several LLM baselines, quantifying search coverage and depth via reasoning trace analysis.

### Key Designs

**1. Formal Definition of DR: Search Intensity $\times$ Reasoning Intensity, not "Output Length"**

The authors reject the common understanding of "DR = long reports" and recharacterize DR as an extreme version of multi-hop RAG. Given a corpus $C$ and query $q$, the definition requires two conditions: (a) **Search Intensity**—answering it requires processing a large number of "information units"; (b) **Reasoning Intensity**—at least one of the sub-tasks (finding, processing, or combining these units) requires non-trivial human expert reasoning. An "information unit" is defined as a paragraph-level atomic piece of info. Operational thresholds are provided: tasks taking a human expert > 10 minutes with web tools are considered reasoning-intensive; search intensity is roughly 20 information units (via $\ge 10$ queries).

This definition places tasks correctly in the task space: multi-hop QA (e.g., HotpotQA) is low in both; scientific/legal QA (e.g., CURIE, CUAD) is high in reasoning but low in search; while information synthesis tasks like those in LIVEDRBENCH reside in the high-search, high-reasoning corner.

**2. Claim-based Intermediate Representation and Precision/Recall**

To score objectively, the solution is formalized as a nested structure of ⟨query, list of claims⟩. An ideal solution must correctly answer all claims and their (recursive) sub-claims. Each claim $A_i$ receives a consistency score $s(A_i)$ relative to the ground truth (usually binary $\{0,1\}$ judged by GPT-4o). Precision and Recall are defined as:

$$\text{Prec}(A) = \frac{\sum_{A_i} w_i\, s(A_i)\, \text{Prec}(A_i)}{\sum_{A_i} 1}, \qquad \text{Rec}(A) = \frac{\sum_{A_i} w_i\, s(A_i)\, \text{Rec}(A_i)}{\sum_{A_i^*} 1}$$

Where $\text{Prec}(A_i)$ and $\text{Rec}(A_i)$ are the average consistency scores of all sub-claims (1 if atomic). A critical design choice is that **a claim's score is multiplied by its sub-claims' metrics**—meaning an incorrect sub-claim penalizes the entire claim. This rewards systems that actually search for evidence and punishes those relying on internal memory.

**3. Problem Inversion: Scaling DR Task Generation**

Creating tasks for models with full web access is difficult because an existing article might collapse the task into simple retrieval. The authors use a three-step inversion process: ① Locate information-dense long documents; ② Extract entities or concept classes with unique features; ③ Construct a question asking to identify these entities and locate supporting sources without knowing the original document. This ensures the benchmark can be refreshed periodically (to prevent contamination) using new papers or reports.

**4. LIVEDRBENCH Composition: 100 Tasks, 8 Categories**

The benchmark covers scientists, information workers, and general users: **SCIFACTS** (Materials and Geo categories for finding papers/materials by properties); **NOVELDS** (Identification and Extraction from dataset papers); **PRIORART** (Identifying key ideas from synthesized abstracts to simulate patent checks); **FLIGHTS** (Locating specific aviation accidents from high-level descriptions); **ENTITIES** (Exhaustive lists of entities satisfying specific criteria, e.g., culture/events).

### Loss & Training
Ours does not train models. Evaluation uses GPT-4o to judge claim consistency. Reasoning traces (coverage, dependency, branching) are also analyzed using GPT-4o.

## Key Experimental Results

### Main Results
Evaluated three closed-source DR products (OpenAI, Perplexity, Gemini), one open-source agent (DeepResearcher + DS-Qwen-32B), and several baselines. OpenAI DR performed best overall.

| Sub-category | OpenAI DR (F1) | Perplexity DR (F1) | Gemini DR (F1) | DeepResearcher (F1) |
|------|------|------|------|------|
| SCIFACTS Materials | 0.314 | 0.150 | 0.022 | 0.000 |
| SCIFACTS Geo | **0.721** | 0.186 | 0.316 | 0.000 |
| NOVELDS Identification | 0.667 | 0.633 | 0.400 | 0.167 |
| NOVELDS Id.&Extraction | 0.470 | 0.333 | 0.345 | 0.023 |
| NOVELDS Peer Retrieval | 0.585 | 0.311 | 0.338 | 0.042 |
| PRIORART | 0.539 | 0.419 | 0.082 | 0.199 |
| ENTITIES | 0.603 | 0.447 | 0.338 | 0.076 |
| FLIGHTS | 0.540 | 0.276 | 0.090 | 0.090 |
| **Average** | **0.555** | 0.355 | 0.263 | 0.075 |

### Ablation Study
On the difficult NOVELDS Id.&Extraction category, DR systems significantly outperformed standard LLMs with search, proving that "DR systems" provide a substantial jump in capability over simple "search + reasoning."

| Model | Precision | Recall | F1 |
|------|------|------|------|
| OpenAI DR | 0.526 | 0.448 | 0.470 |
| Perplexity DR | 0.325 | 0.349 | 0.333 |
| Gemini DR | 0.406 | 0.329 | 0.345 |
| OpenAI o4-mini (Reasoning baseline) | 0.203 | 0.146 | 0.168 |

### Key Findings
- **Coverage is only around 50%**: Analysis shows that OpenAI DR covers 66% of necessary queries, while others hover around 50%. Most systems miss half of the required search steps.
- **Breadth matters more than depth**: OpenAI DR has the highest branching factor, which correlates with its superior accuracy. Broad search coverage is a key bottleneck for DR performance.
- **The sub-claim penalty is the biggest hurdle**: Models score significantly lower when required to get both the main claim (e.g., paper title) and sub-claim (e.g., measurement property) correct simultaneously.

## Highlights & Insights
- **Operationalizing DR**: Transforms "Deep Research" from a marketing slogan into a measurable 2D definition with empirical thresholds.
- **Decoupling Evaluation**: Claim-based evaluation strips reasoning accuracy away from subjective writing styles, rewarding factual retrieval over internal LLM "hallucination" or memory.
- **Sustainable Benchmarking**: Problem inversion allows for scalable, anti-contamination benchmark updates by leveraging the continuous flow of new scientific and world data.

## Limitations & Future Work
- **Judge Bias**: Metrics rely heavily on GPT-4o as a judge, which may introduce its own biases or noise.
- **Synthesis vs. Prose**: The benchmark focuses on information synthesis and does not evaluate the quality of the final report's narrative, organization, or readability.
- **Scope**: Does not cover computer use, code execution, or external tool manipulation which are also parts of the broader "research" umbrella.

## Related Work & Insights
- **vs. DeepResearch Bench**: Moves from subjective report scoring to objective, reproducible claim-based metrics.
- **vs. BrowseComp**: Standardizes and automates the problem inversion technique to handle complex, multi-entity answers.
- **vs. Multi-hop QA**: Explicitly targets the high-search intensity corner of the task space that simple multi-hop datasets fail to reach.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to provide a formal, measurable definition and objective claim-based evaluation for DR.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive comparison across closed and open systems with deep trajectory analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear logic from definition to metric construction and evaluation.
- Value: ⭐⭐⭐⭐⭐ Sets an essential objective standard for a rapidly evolving field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DRBench: A Realistic Benchmark for Enterprise Deep Research](drbench_a_realistic_benchmark_for_enterprise_deep_research.md)
- [\[ICLR 2026\] LiveResearchBench: A Live Benchmark for User-Centric Deep Research in the Wild](liveresearchbench_a_live_benchmark_for_user-centric_deep_research_in_the_wild.md)
- [\[ICLR 2026\] An Open-Ended Benchmark and Formal Framework for Adjuvant Research with MLLM](an_open-ended_benchmark_and_formal_framework_for_adjuvant_research_with_mllm.md)
- [\[ICLR 2026\] DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents](deepresearch_bench_a_comprehensive_benchmark_for_deep_research_agents.md)
- [\[ICLR 2026\] ResearchRubrics: A Benchmark of Prompts and Rubrics For Evaluating Deep Research Agents](researchrubrics_a_benchmark_of_prompts_and_rubrics_for_evaluating_deep_research_.md)

</div>

<!-- RELATED:END -->
