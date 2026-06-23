---
title: >-
  [Paper Note] ACADREASON: Exploring the Limits of Reasoning Models with Academic Research Problems
description: >-
  [ICLR 2026][LLM Evaluation][LLM-as-Judge] AcadReason utilizes 50 research questions from top-tier journal papers across 5 high-reasoning disciplines (Computer Science, Economics, Law, Mathematics, Philosophy) to specifically test whether LLMs and Agents can acquire and reason through academic knowledge "like a researcher." The results show that most LLMs score
tags:
  - ICLR 2026
  - LLM Evaluation
  - LLM-as-Judge
  - Agent
date: 2026-05-08
content_hash: e2063f8405fc4e4c
---
# ACADREASON: Exploring the Limits of Reasoning Models with Academic Research Problems

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=vl0hQuluv4](https://openreview.net/forum?id=vl0hQuluv4)  
**Code**: https://github.com/OPPO-PersonalAI/Acadreason-benchmark  
**Area**: LLM Evaluation / Reasoning Benchmark  
**Keywords**: Academic Reasoning, Research-level Evaluation, LLM-as-Judge, Agent, Checklist Scoring

## TL;DR
AcadReason utilizes 50 research questions from top-tier journal papers across 5 high-reasoning disciplines (Computer Science, Economics, Law, Mathematics, Philosophy) to specifically test whether LLMs and Agents can acquire and reason through academic knowledge "like a researcher." The results show that most LLMs score below 20, with even GPT-5 only reaching 16 points and the strongest Agent, OAgents, peaking at 34 points, revealing a significant gap in "super-intelligent academic research" capabilities.

## Background & Motivation
**Background**: In the past two years, the focus of LLM and Agent research has shifted from "demonstrating new capabilities" to "complex reasoning and hard tasks." In the evaluation space, mainstream benchmarks either focus on math/code competitions (AIME, contest problems) or multi-domain knowledge Q&A (MMLU-Pro, GPQA, SuperGPQA).

**Limitations of Prior Work**: These benchmarks are rapidly saturating or becoming outdated. Competition-based benchmarks lack **domain breadth** by excluding science and humanities. Multi-domain academic benchmarks (like MMLU-Pro) primarily test undergraduate-level knowledge and common-sense reasoning, lacking **reasoning depth**—they test "information integration" rather than "multi-step reasoning under frontier specialized knowledge." Even research-oriented benchmarks like GAIA, PaperBench, and DeepResearch Bench have their biases: GAIA focuses on tool use and web retrieval, and PaperBench focuses on the engineering capacity to reproduce ICML papers. None purely test the ability to "understand top-tier journal theories and derive conclusions independently."

**Key Challenge**: It is difficult to achieve both breadth and depth. A benchmark either covers many fields superficially or targets pure math/code deeply. There is no rigorous evaluation that spans both STEM and humanities while requiring doctoral-level reasoning for every question.

**Goal**: To build an academic reasoning benchmark that satisfies "multi-field + high reasoning depth + timeliness (to prevent data contamination) + answerability (unique and verifiable answers)" to measure how far current LLMs and Agents are from becoming "super-intelligent research assistants."

**Key Insight**: Extract research problems directly from **purely theoretical papers** in top journals from the last three years (2023–2025). The newer and more theoretical the paper, the lower the probability that the model encountered the answer during pre-training, forcing it to reason rather than memorize. Each paper provides only one question, but the golden answer must cover the paper's core contributions, ensuring maximum workload and reasoning depth per item.

**Core Idea**: Use "real research problems from top theoretical journals" as questions, expert-customized dynamic checklists as the scoring rubric, and GPT-5-mini as the judge to construct a high-reasoning academic benchmark where even the strongest models struggle to pass.

## Method

### Overall Architecture
AcadReason is essentially a "data annotation + evaluation" pipeline resulting in 50 research-grade Q&A items across 5 high-reasoning disciplines. Each item consists of four atomic fields: Question / Hints / Checklist / Golden Answer.

The pipeline comprises three steps: **(1) High-quality Paper Collection**—430 candidates were filtered by publication date and journal tier, then refined by 10 domain experts into 50 pure theoretical papers; **(2) High-reasoning Question Extraction**—experts read the full papers, distilled core research problems into formal questions, and wrote golden answers with complete reasoning details; **(3) Checklist and Hints Extraction**—verifiable, independent scoring points (Checklists) were distilled from golden answers, and three types of hints (Background/Definition/Methodology) were organized from different paper sections. During evaluation, models answer **without access to the original paper**, and GPT-5-mini scores them against the golden answers and checklists. Models act as "researchers," using either internal knowledge or search tools.

### Key Designs

**1. Four-field Task Structure: Atomizing Research Problems for Reasoning and Scoring**

Directly asking "what does this paper do" based on a title is uncontrollable and ungradable. AcadReason structures each task into four fields. The **Question** is a self-contained research problem composed of (a) a specific problem from the paper and (b) the minimal background required for comprehension. The **Golden Answer** is a complete solution trajectory covering background, definitions, derivations, and conclusions. The **Checklist** consists of several scoring points distilled by experts, each corresponding to a key milestone (a logical step or critical fact). Unlike static checklists, these are **dynamic**—customized per question with variable lengths. **Hints** provide supplementary information categorized into background (intro/related work), definition (formulas/terms), and methodology (theoretical tools for proof).

**2. Three-stage Annotation + Strict Filtering: Forcing True Reasoning through "Top-tier + Recent + Theoretical"**

The quality of questions determines the benchmark's ceiling. Candidates are collected from top journals and filtered by experts based on whether they contain challenging reasoning and are purely theoretical. The final 50 papers must meet three criteria: ① Published in top-tier journals/conferences; ② Published in **2023–2025** (to mitigate pre-training contamination); ③ **Purely theoretical**, excluding empirical studies or surveys. The disciplines include Computer Science, Economics, Law, Mathematics, and Philosophy. Each field has 2 experts (master's/PhD level). Each paper yields only one question that covers the full scope of the paper's contribution.

**3. Multi-stage Verification + Question Answerability Verification: Closing Loopholes**

To prevent vague research questions, AcadReason includes **Question Answerability Verification**. Three domain experts perform quality control on each item based on boundary clarity, information completeness, and argumentative logic. Only items passing all three criteria are included, ensuring a multi-stage filtering pipeline.

**4. LLM-as-Judge + Dual Metrics: Proving GPT-5-mini's Reliability as a Judge**

As research answers are open-ended, exact matching is insufficient. AcadReason uses GPT-5-mini as the judge but validates it via an Inter-Rater Reliability (IRR) study with three human experts, achieving a Cohen's $\kappa = 0.861$. GPT-5-mini reached $89.55\%$ accuracy relative to human consensus on Checklist scores. The judge evaluates two metrics: (i) Exact correspondence with the golden answer (记 1 if all info is present and non-contradictory, else 0); (ii) Whether individual checklist items are satisfied.

$$R_p = \frac{\sum_{q=1}^{50} s_q}{50} \times 100, \qquad R_j = \frac{\sum_{q=1}^{50}\sum_{i=1}^{5} c_{q,i}}{250} \times 100$$

**Pass Rate** $R_p$ ($s_q \in \{0,1\}$) measures complete consistency with the standard answer ("all or nothing"). **Checklist Score** $R_j$ ($c_{q,i} \in \{0,1\}$) measures the proportion of checklist items met, offering a fine-grained evaluation of partial reasoning.

## Key Experimental Results

Over 10 LLMs and Agents were tested. Scores are presented as `Pass Rate / Checklist Score`.

### Main Results

| Category | Model | Overall (Rp/Rj) | Notes |
|------|------|------|------|
| General | GPT-5 | 16 / 40.5 | Strongest general model scores only 16 |
| General | GPT-OSS | 4 / 32.2 | |
| General | DeepSeek-V3.1 | 2 / 24.8 | Significant improvement over V3 |
| General | DeepSeek-V3 | 2 / 15.9 | |
| General | Claude-Sonnet-4 | 0 / 24.7 | Pass Rate of zero |
| General | GPT-4.1 | 0 / 21.0 | |
| Reasoning | o3 | 4 / 33.4 | Higher and more balanced Checklist scores |
| Reasoning | DeepSeek-R1 | 2 / 23.8 | Notable reasoning gain over V3 (15.9) |
| Reasoning | Qwen3 / Kimi-K2 | 6 / 20.3 | |
| Reasoning | Gemini-2.5-Pro | 2 / 22.3 | |
| Agent | **OAgents** | **34 / 65.1** | Highest overall; SOTA in most disciplines |
| Agent | Gemini-2.5-Pro-DeepResearch | 28 / 53.4 | |
| Agent | Tongyi DeepResearch | 20 / 30.9 | |
| Agent | o3-DeepResearch | 14 / 47.1 | |

**Key Finding**: **No model/framework exceeded a 40 Pass Rate**, far below 100. Strong models like GPT-4.1 and Claude-Sonnet-4 scored 0 in Pass Rate, indicating the difficulty. CS and Economics were the most difficult disciplines.

### Ablation Study: Three Types of Hints

| Model | No Hint | +background | +definition | +methodology | +ALL Hints |
|------|---------|-------------|-------------|--------------|------------|
| GPT-5 | 16/40.5 | 16/42.5 | 24/50.9 | 34/64.3 | **40/67.8** |
| GPT-OSS | 4/32.2 | 14/40.5 | 10/42.3 | 16/52.2 | 22/58.5 |
| o3 | 4/33.4 | 12/38.0 | 10/48.9 | 28/56.2 | 26/60.8 |
| DeepSeek-R1 | 2/23.8 | 4/30.6 | 6/35.7 | 8/45.3 | 20/50.4 |
| GPT-4.1 | 0/21.0 | 2/26.3 | 0/29.9 | 8/42.8 | 20/51.6 |

### Key Findings
- **Hints are significantly effective**: GPT-5's score rose from 16/40.5 to 40/67.8 with full hints, surpassing the strongest Agent framework OAgents (34/65.1). This suggests the bottleneck is often "access to frontier knowledge" rather than "reasoning capacity."
- **Methodology hints provide the largest gain**: This confirms that Ours tests deep methodology mastery rather than easily searchable background info.
- **Agents bridge the knowledge gap through retrieval**: OAgents outperformed general models because it actively supplements missing academic knowledge. URL-masking (masking original paper links) results were nearly unchanged, proving the gain comes from reasoning/general retrieval rather than finding the source paper.
- **Reasoning models > counterparts**: DeepSeek-R1 (2/23.8) > DeepSeek-V3 (2/15.9).
- **Disciplinary Differences**: Humanities (Econ/Law/Phi) rely more on external knowledge/hints; STEM (CS/Math) relies more on deep reasoning.

## Highlights & Insights
- The **"one-question-per-paper"** design maximizes depth while ensuring complete, verifiable golden answers.
- **Dynamic Checklists** are better suited for open-ended research questions than static ones, allowing for more granular scoring of complex problems.
- **Judge Reliability Verification** ($\kappa=0.861$) adds significant academic rigor compared to unverified LLM scoring.
- The **Ablation of three hint types** separates "knowledge" from "reasoning," identifying that the primary weakness of current models lies in deep methodology mastery.
- The **URL-masking experiment** effectively addresses concerns regarding whether Agents are simply "finding the source paper."

## Limitations & Future Work
- **Small Scale**: 50 questions across 5 disciplines means only 10 questions per field, leading to potential variance in discipline-specific conclusions.
- **Dependency on a Single Judge**: Relying on GPT-5-mini introduces systematic biases and makes historical comparisons difficult if the judge model is updated.
- **Limited Domain Coverage**: While multi-domain, it currently lacks physics, biology, and medicine.
- **Timeliness vs. Longevity**: The "2023–2025" window prevents current contamination but requires continuous updates as pre-training data evolves.

## Related Work & Insights
- **vs GPQA / SuperGPQA / MMLU-Pro**: These focus on undergraduate-level knowledge and are saturating; AcadReason pushes reasoning to the PhD/research level with dynamic checklists.
- **vs PaperBench**: PaperBench focuses on engineering/reproduction; AcadReason focuses purely on "understanding theory and deriving conclusions" and includes humanities.
- **vs GAIA / BrowseComp**: These focus on tool use and web search; AcadReason isolates professional academic knowledge and multi-step reasoning.
- **vs DeepResearch Bench**: AcadReason's strict construction (top-tier theory, recent papers, expert verification) makes it significantly more challenging.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovative combination of top-tier theoretical problems, dynamic checklists, and hint-based ablation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Analysis of 10+ models, multi-stage hint ablation, and judge reliability; however, the sample size is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Clear description of the pipeline and design logic with ample charts.
- Value: ⭐⭐⭐⭐ Provides a rigorous metric for evaluating "academic research capability" in frontier models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment](../../ACL2026/llm_evaluation/novbench_evaluating_large_language_models_on_academic_paper_novelty_assessment.md)
- [\[ACL 2025\] Com2: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models](../../ACL2025/llm_evaluation/com2_causal_commonsense.md)
- [\[ICLR 2026\] Towards Personalized Deep Research: Benchmarks and Evaluations](towards_personalized_deep_research_benchmarks_and_evaluations.md)
- [\[ICLR 2026\] Characterizing Deep Research: A Benchmark and Formal Definition](characterizing_deep_research_a_benchmark_and_formal_definition.md)
- [\[ICLR 2026\] ResearchRubrics: A Benchmark of Prompts and Rubrics For Evaluating Deep Research Agents](researchrubrics_a_benchmark_of_prompts_and_rubrics_for_evaluating_deep_research_.md)

</div>

<!-- RELATED:END -->
