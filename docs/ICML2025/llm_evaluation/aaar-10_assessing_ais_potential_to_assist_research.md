---
title: >-
  [Paper Note] AAAR-1.0: Assessing AI's Potential to Assist Research
description: >-
  [ICML 2025][LLM Evaluation][LLM Evaluation Benchmark] The AAAR-1.0 benchmark is proposed to systematically evaluate the actual capabilities of LLMs in assisting scientific research across four expert-level tasks: equation inference, experimental design, paper weakness detection, and peer review critique. The benchmark reveals significant deficiencies in current models when performing deep research tasks.
tags:
  - "ICML 2025"
  - "LLM Evaluation"
  - "LLM Evaluation Benchmark"
  - "AI-Assisted Scientific Research"
  - "Paper Peer Review"
  - "Experimental Design"
  - "Research Capability Assessment"
date: 2026-05-08
content_hash: 3c0f8597f3804d0f
---

# AAAR-1.0: Assessing AI's Potential to Assist Research

**Conference**: ICML 2025  
**arXiv**: [2410.22394](https://arxiv.org/abs/2410.22394)  
**Code**: [https://renzelou.github.io/AAAR-1.0/](https://renzelou.github.io/AAAR-1.0/)  
**Area**: Human Understanding  
**Keywords**: LLM Evaluation Benchmark, AI-Assisted Scientific Research, Paper Peer Review, Experimental Design, Research Capability Assessment

## TL;DR

The AAAR-1.0 benchmark is proposed to systematically evaluate the actual capabilities of LLMs in assisting scientific research across four expert-level tasks: equation inference, experimental design, paper weakness detection, and peer review critique. The benchmark reveals significant deficiencies in current models when performing deep research tasks.

## Background & Motivation

While LLMs have demonstrated outstanding capabilities in daily tasks (e.g., email writing, question answering, creative drafting), can they effectively assist with the core work of researchers—such as brainstorming research ideas, designing experiments, and writing/reviewing papers? Most existing efforts (e.g., AI-Scientist, LLM idea generation) focus on highly subjective, end-to-end task pipelines that are costly to evaluate and difficult to replicate. The core motivations of this work are threefold:

**Lack of systematic benchmarks for single-step research tasks**: Existing benchmarks largely focus on code implementation and model training (e.g., MLAgentBench), ignoring highly cognitive-intensive components such as idea generation, experimental planning, and paper reviewing.

**Need for automated evaluation metrics**: Tasks like idea generation rely heavily on expensive human evaluations (Si et al., 2024), which hinders large-scale comparisons.

**Higher transparency of single-step tasks**: Compared to complex task pipelines, single-step tasks have clear input/output expectations, enabling precise pinpointing of model capability boundaries.

## Method

### Overall Architecture

AAAR-1.0 deconstructs the daily activities of researchers into four independent, expert-level tasks, each with clearly defined inputs/outputs and associated automatic evaluation metrics:

| Task | Abbreviation | Input | Output | Evaluated Capability |
|------|--------------|-------|--------|----------------------|
| Equation Inference | EqInfer | Paper context + Equation | Correct/Incorrect (Binary classification) | Local contextual reasoning, symbolic understanding |
| Experimental Design | ExpDesign | Pre-experiment paper content (including figures) | List of experiments + Motive explanations | High-level experimental planning, domain knowledge |
| Paper Weakness | Weakness | Full paper (including tables and figures) | List of weaknesses | Critical analysis, deep peer reviewing |
| Review Critique | ReviewCritique | Paper + Reviews + Rebuttal | Whether each review paragraph has flaws | Meta-reviewing capability, advanced research experience |

### Key Designs

#### Task 1: EquationInference

**Four-stage data construction pipeline:**

1. **Data Crawling & Cleaning**: LaTeX source files of 1,762 published papers were crawled from ACL Anthology (2019-2023), from which 3,877 human-written positive equations were extracted using regular expressions. Choosing LaTeX source files over PDF parsing avoids noise introduced by tools like PyMuPDF.
2. **LLM Negative Synthesis**: For each positive equation, GPT-4 was used to synthesize 3 incorrect equations based on the paper context (employing high-temperature decoding to ensure diversity).
3. **LLM Filtering**: GPT-4 was utilized to identify negative cases with "context misalignment" (such as containing symbols undefined in the paper). Samples where all 3 negatives contained shortcut clues were filtered out, leaving 1,449 positive examples.
4. **Expert Review**: Five senior PhD students compiled and verified the samples using tools like TeXlive, checking whether each positive-negative pair satisfied: (a) syntactic correctness; (b) the compiled negative was indeed mathematically different from the positive. Every pair was cross-reviewed by at least 2 people, resulting in a compiled set of 1,049 positive equations (a 27.6% rejection rate).

**Design Highlight**: Structured as a binary classification task rather than multiple choice, as experiments show binary classification is more challenging for LLMs.

#### Task 2: ExperimentDesign

**High-standard annotation process:**

1. **Data Source**: Over 10k papers (cs.AI/CL/CV, 2018-2023) were crawled from arXiv, keeping only top-tier conference publications.
2. **Domain Expert Annotation**: Strict qualifications were enforced—senior PhD students, at least one top-tier publication, over 4 years of research experience, and active peer reviewers. Ten experts each annotated 10 paper drafts, defining all critical experiments and explaining the underlying motivations.
3. **Multi-round Peer Discussion**: Each expert's annotation was reviewed by another expert to verify: whether any experiments were missed, whether the summary covered critical information, and whether the explanations were reasonable. Iterative discussions were conducted until consensus was reached.
4. **Information Leakage Mitigation**: GPT-4 was used to strip sentences from the input that might leak the actual experiments (about 9.8% of sentences were removed).

A final set of 100 instances was collected. The input consists of the pre-experiment paper context (including figures), and the output is the expert-annotated list of experimental steps and their motivations.

#### Task 3: PaperWeakness

1. **Data Source**: 3,779 anonymous submissions of ICLR 2023 were crawled from OpenReview. A balanced sample of 1,000 papers (500 accepted + 500 rejected) across all 13 tracks was obtained.
2. **Weakness Extraction**: GPT-4 was used to extract weaknesses from the peer reviews, keeping the original reviewers' phrasing unmodified. Duplicate weaknesses brought up by the same reviewer were retained to capture importance weights.
3. **Input Processing**: Since LaTeX sources are often unavailable for OpenReview submissions, VILA was used to parse PDF text, and PDFFigures-2.0 was used to extract tables and figures. This resulted in 993 final instances.

#### Task 4: ReviewCritique

This task borrows the dataset from Du et al. (2024), containing 100 papers and 380 reviews. Each review is decomposed into sentence-level paragraphs (11,376 paragraphs in total). Over 40 AI research experts annotated whether each paragraph contained flaws and provided justifications. This task demands the highest level of advanced research experience among the four tasks.

### Loss & Training

Since this paper introduces an evaluation benchmark rather than a training method, its core innovation lies in the design of **automatic evaluation metrics**:

**EqInfer Evaluation**: Standard binary classification F1 score (random guessing baseline is approximately 40%).

**ExpDesign Evaluation—Entailment-based Precision/Recall**:
- **En-Precision**: For each entry in the predicted experiment list, LLM (GPT-4o) evaluates whether it is semantically entailed by the ground-truth list.
- **En-Recall**: For each entry in the ground-truth list, the LLM evaluates whether it is semantically entailed by the predicted list.
- **S-Match**: SentenceBERT is used to calculate the semantic similarity between the predicted motivation explanations and the ground-truth explanations.

**Weakness Evaluation—Multi-Reviewer Semantic Matching**:
- **S-Precision**: Measures the maximum semantic similarity of each predicted weakness against each reviewer's weakness list, averaged across all reviewers.
- **S-Recall**: Measures the maximum semantic similarity of each reviewer's weaknesses against the predicted weakness list, averaged across all reviewers.

This metric design preserves the structural information of multiple reviewer perspectives, preventing crucial diversity from being lost in simple merging.

**ReviewCritique Evaluation**: Classification F1 score to assess the accuracy of the model in identifying flaws in each review paragraph.

## Key Experimental Results

### Main Results

| Task | Metric | Best Model | Best Score | Random/Baseline | Remarks |
|------|--------|------------|------------|-----------------|---------|
| EqInfer | F1 | Top closed-source LLM | ~46% | 40% | Only slightly higher than random guessing |
| ExpDesign | En-Precision | GPT-4 series | Moderate | — | Proposes many innovative but unfeasible experiments |
| Weakness | S-Precision | GPT-4 series | Moderate | — | Weaknesses lack depth and specificity |
| ReviewCritique | F1 | Top closed-source LLM | Low | — | Struggles to identify flawed peer reviews |

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| Open-source vs. Closed-source LLMs | F1/Precision/Recall across tasks | Closed-source models generally outperform open-source ones, but the gap is narrowing |
| LaTeX source vs. PDF-parsed text | EqInfer F1 | LaTeX source as input contains less noise and richer information |
| With vs. Without visual inputs | ExpDesign metrics | Visual inputs provide marginal performance gains on experimental design |
| Different prompt strategies | Metrics across tasks | Task-specific prompts outperform generic ones |
| Before vs. After negative filtering | EqInfer difficulty | Filtering shallow shortcut clues significantly increases the task difficulty |
| Multi-reviewer vs. Single-reviewer evaluation | Weakness S-Precision | Multi-reviewer structured evaluation is fairer and more comprehensive |

### Key Findings

1. **EqInfer**: The F1 score of most LLMs is only slightly above the 40% random baseline (the highest being ~46%). This indicates that even with local contextual reasoning, LLMs remain highly deficient in mathematical equation understanding.
2. **ExpDesign**: LLM-designed experiments are more diverse and innovative than those of humans. However, a large portion of them are trivial, lack feasibility, and drift away from the original research objectives.
3. **Weakness**: Paper weaknesses identified by LLMs lack depth and specificity. Models tend to generate "one-size-fits-all" vague critiques applicable to almost any paper, lacking domain-specific insights.
4. **ReviewCritique**: LLMs struggle to effectively recognize flawed critiques in reviews, showing limited capability in assisting with meta-reviewing.

## Highlights & Insights

- **Excellent Task Deconstruction**: Deconstructing the research workflow into independent, measurable single-step tasks avoids the uncontrollability of end-to-end evaluation, providing a clear diagnostic framework for future research.
- **Rigorous Data Quality Control**: The four-stage pipeline (crawling → synthesis → filtering → expert review) guarantees high data quality for each task. The 27.6% rejection rate in EqInfer demonstrates highly rigorous quality standards.
- **Exquisitely Designed Metrics**: Entailment-based Precision/Recall for ExpDesign and multi-reviewer semantic matching for Weakness elegantly resolve the challenges of evaluating free-form text, balancing efficiency with fairness.
- **Clear Ethical Stance**: Emphasizes that LLMs should assist junior researchers by providing imperfect but insightful suggestions instead of replacing humans to lead the entire research lifecycle.

## Limitations & Future Work

1. **Limited Data Scale**: ExpDesign only has 100 instances and ReviewCritique only covers 100 papers, which may not be enough to draw robust statistical conclusions.
2. **Narrow Domain Coverage**: Primarily focused on AI/NLP/CV (ACL and ICLR papers), leaving other disciplines unrepresented.
3. **Metric Limitations**: Relying on GPT-4o as an evaluator for ExpDesign introduces the evaluator's own LLM biases; SentenceBERT semantic similarity might fail to capture fine-grained academic nuances.
4. **Lack of Deep Multimodal Evaluation**: Although ExpDesign and Weakness incorporate image inputs, the impact of visual understanding on research tasks is not fully explored.
5. **Recency Challenges**: Benchmark data source only spans up to 2023. Given the rapid iteration of LLMs, continuous updates are required to maintain evaluation efficacy.

## Related Work & Insights

- **Si et al. (2024)**: A large-scale human study showing that LLMs can generate novel but unfeasible ideas, which aligns with the findings in the ExpDesign task of this work.
- **AI-Scientist (Lu et al., 2024)**: While autonomous research agents comprise complete pipelines, this work notes that evaluating single-step intermediate outputs is equally crucial.
- **Du et al. (2024)**: Revealed that LLMs excel at summarizing paper strengths but struggle to identify weaknesses, directly inspiring the design of Weakness and ReviewCritique tasks.
- **MLAgentBench (Huang et al., 2024)**: Focuses on experiment implementation/execution, whereas this work emphasizes that high-level experimental planning prior to implementation is equally critical.
- **Insight**: The single-step task evaluation framework of AAAR can be transferred to other scientific domains and serve as intermediate diagnostic checkpoints for LLM research agents.

## Rating

| Dimension | Rating (1-5) | Description |
|-----------|--------------|-------------|
| Novelty | ⭐⭐⭐⭐ | First systematic benchmark for single-step scientific research tasks with highly novel task designs |
| Technical Depth | ⭐⭐⭐⭐ | Rigorous data collection workflows and exquisitely designed evaluation metrics |
| Experimental Thoroughness | ⭐⭐⭐ | Covers a wide range of LLMs, though some tasks suffer from limited sample sizes |
| Value | ⭐⭐⭐⭐ | Provides quantifiable evaluation tools for AI-assisted scientific research |
| Writing Quality | ⭐⭐⭐⭐ | Well-structured with highly motivated explanations |
| **Overall Rating** | **⭐⭐⭐⭐** | **An excellent benchmark paper that fills a critical gap in scientific task evaluation** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] EnIGMA: Interactive Tools Substantially Assist LM Agents in Finding Security Vulnerabilities](enigma_interactive_tools_substantially_assist_lm_agents_in_finding_security_vuln.md)
- [\[ICLR 2026\] AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite](../../ICLR2026/llm_evaluation/astabench_benchmarking_ai_agents.md)
- [\[ICLR 2026\] AirQA: A Comprehensive QA Dataset for AI Research with Instance-Level Evaluation](../../ICLR2026/llm_evaluation/airqa_a_comprehensive_qa_dataset_for_ai_research_with_instance-level_evaluation.md)
- [\[NeurIPS 2025\] The Biased Oracle: Assessing LLMs' Understandability and Empathy in Medical Diagnoses](../../NeurIPS2025/llm_evaluation/the_biased_oracle_assessing_llms_understandability_and_empathy_in_medical_diagno.md)
- [\[ICLR 2026\] DeepTRACE: Auditing Deep Research AI Systems for Tracking Reliability Across Citations and Evidence](../../ICLR2026/llm_evaluation/deeptrace_auditing_deep_research_ai_systems_for_tracking_reliability_across_cita.md)

</div>

<!-- RELATED:END -->
