---
title: >-
  [Paper Note] ResearchRubrics: A Benchmark of Prompts and Rubrics For Evaluating Deep Research Agents
description: >-
  [ICLR 2026][LLM Evaluation][Deep Research Agents] ResearchRubrics utilizes 2800+ hours of human effort to pair 101 real-world open-ended research prompts with 2593 expert-written, weighted, fine-grained rubrics. Using LLM-as-Judge to score agents based on these rubrics, the study evaluates mainstream Deep Research (DR) systems and finds that even the strongest agents, such as Gemini DR and OpenAI DR, fail to reach an average rubric adherence rate of 68%. Theoretical bottlenec…
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "Deep Research Agents"
  - "Human Rubrics"
  - "LLM-as-Judge"
  - "Task Complexity"
  - "Open-ended Evaluation"
date: 2026-05-08
content_hash: effb69c559dbe564
---

# ResearchRubrics: A Benchmark of Prompts and Rubrics For Evaluating Deep Research Agents

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ErnvfmSX0P](https://openreview.net/forum?id=ErnvfmSX0P)  
**Data**: https://huggingface.co/datasets/ScaleAI/researchrubrics  
**Area**: LLM Evaluation / Deep Research Agent / Benchmark  
**Keywords**: Deep Research Agents, Human Rubrics, LLM-as-Judge, Task Complexity, Open-ended Evaluation

## TL;DR
ResearchRubrics utilizes 2800+ hours of human effort to pair 101 real-world open-ended research prompts with 2593 expert-written, weighted, fine-grained rubrics. Using LLM-as-Judge to score agents based on these rubrics, the study evaluates mainstream Deep Research (DR) systems and finds that even the strongest agents, such as Gemini DR and OpenAI DR, fail to reach an average rubric adherence rate of 68%. Theoretical bottlenecks are concentrated in implicit requirement inference and multi-source information synthesis.

## Background & Motivation
**Background**: Deep Research (DR) is an emerging category of LLM agent applications. Given an open-ended question, the agent autonomously conducts multi-step web retrieval, targeted evidence collection, and cross-document synthesis to produce an evidence-backed long-form report. Companies like OpenAI, Google, and Perplexity have launched commercial DR systems and achieved high scores on benchmarks like HLE.

**Limitations of Prior Work**: DR outputs are long and complex, often lacking a single correct answer, which traditional evaluation methods cannot handle. Traditional QA benchmarks (HLE, HotpotQA, etc.) only test short answers or mechanically verifiable facts (e.g., "Which material has a bandgap of 0.9 eV and a dislocation density of $4\times10^8\,\text{cm}^{-2}$" → "Gallium Nitride"), making them unsuitable for measuring long-form synthesis quality. Recent benchmarks designed specifically for DR have flaws: some (DeepResearch Bench) use LLMs to automatically generate rubrics or reference reports, risking circular reasoning and supervisory failure; others (DeepScholar-Bench, ReportBench) narrow the scope to single technical tasks like "writing a Related Work" section, failing to cover the diverse topics real users query, from business reports to consumer decisions.

**Key Challenge**: Evaluating DR requires both **domain diversity** (realistic, messy user queries) and **expert-written fine-grained rubrics** (to avoid anchoring bias and coarse overlapping metrics from LLM self-evaluation). Existing works typically trade one for the other—either rubrics are machine-generated, or tasks are restricted to narrow, easily evaluable academic domains.

**Goal**: To create a standard benchmark of "real open-ended prompts × expert-written rubrics" that can diagnostically identify at an atomic level where DR agents fail.

**Key Insight**: The authors argue that DR task difficulty is not one-dimensional but should be characterized across three orthogonal axes: conceptual breadth, logical nesting depth, and exploratoriness (openness/under-determination). This coordinate system ensures balanced task distribution and precise analysis of agent performance across "breadth vs. depth vs. ambiguity."

**Core Idea**: By using purely human-written, weighted (including negative scores), and mandatory/optional categorized rubrics, open-ended long-form evaluation is decomposed into a large number of atomically decidable standards. These are then evaluated by an LLM-as-Judge using ternary scoring, transforming "subjective long-form quality" into a quantifiable, reproducible, and attributable rubric adherence rate.

## Method

### Overall Architecture
ResearchRubrics is essentially a **dataset and evaluation protocol** rather than a model method. It consists of three components: (1) 101 single-turn open-ended prompts covering 9 major domains; (2) 20–43 expert-written weighted rubrics per prompt, totaling 2593 rubrics; (3) an LLM-as-Judge evaluation formula that maps agent reports to "per-rubric adherence." The pipeline involves: three experts collaborating to create prompts and rubrics → labeling each task with (breadth, depth, exploration) complexity → generating reports from DR systems → LLM judge providing a ternary verdict (satisfied/partially satisfied/not satisfied) for each rubric → calculating normalized scores and decomposing failure sources. As this is a benchmark paper involving a linear manual pipeline rather than a multi-module algorithm, the focus is on "rubric design and scoring mechanics."

### Key Designs

**1. Three-Expert Data Construction Pipeline: Replacing LLM Generation with Multi-round Human Review to Eliminate Anchoring Bias**

This design addresses the issue that "LLM-generated rubrics lead to circular reasoning and anchoring bias." "Experts" are defined as individuals with solid STEM backgrounds skilled in task design and evaluation (not necessarily specialists in every prompt domain), and each works within their familiar domains. The process involves three roles: Expert 1 drafts a prompt and rubrics; Expert 2 reviews them, iterating until approval; Expert 3 performs an independent final review and fine-tuning. Prompts are inspired by user forums and Q&A sites, rewritten as research questions. Significantly, **all prompts and rubrics are human-written and human-reviewed without any LLM-generated seeds**, distinguishing this from benchmarks like LiveResearchBench where humans only "review" machine drafts.

**2. Three-Axis Task Complexity Framework: Decomposing Open-ended Difficulty into Breadth, Depth, and Exploration**

DR task difficulty is non-uniform. The authors tag each task along three orthogonal axes: **Conceptual Breadth** (Simple / Moderate / High) based on the number and dispersion of information sources (e.g., "Analyze environmental, economic, and political factors of renewable energy" is High); **Logical Nesting** (Shallow / Intermediate / Deep) based on the steps of dependent reasoning (4+ steps including "analysis → synthesis → evaluation → revision" is Deep); **Exploration** (Low / Medium / High) based on under-determination (3+ unspecified key factors requiring agent goal-setting is High). Every task is labeled as a (Breadth, Depth, Exploration) triplet, allowing for balanced distribution and dimensional attribution—for instance, revealing if models fail monotonically as logical nesting deepens while remaining less sensitive to conceptual breadth.

**3. Weighted, Mandatory/Optional, and Negative-Score Rubric System: Decomposing Quality into Atomic Standards**

To quantify subjective quality, each rubric is anchored to one of six evaluation axes: Explicit requirements, Implicit requirements, Synthesis, References, Communication, and Instruction Following. Each rubric is assigned an integer weight from $[-5, 5]$: $\pm4$ or $\pm5$ are **mandatory** standards (minimum requirements for a passing answer), while $[-3, 3]$ are **optional** standards (bonus points for distinguishing "excellent" from "sufficient"). Positive weights reward valuable attributes, while negative weights penalize common failures like factual errors, hallucination, or verbosity. Rubrics are worded such that once triggered, the negative weight is added to the sum. Weights align with a six-level human preference scale to improve human-machine scoring consistency. Separating mandatory/optional and explicitly using negative scores fills a gap in existing benchmarks.

**4. Ternary LLM-as-Judge Scoring Formula: Supporting Grayscale Judgment for Open-ended Answers**

DR answers often "partially satisfy" a rubric. The authors use an LLM judge to output a ternary verdict $m_{r_i}\in\{1, 0.5, 0\}$ (satisfied / partially satisfied / not satisfied). The final compliance score for task $k$ is the weighted sum of positive and negative weights, normalized by the sum of positive weights (theoretical maximum):

$$S_k = \frac{\sum_{r_i\in C} w_{r_i} m_{r_i}}{\sum_{r_i\in C,\, w_{r_i}>0} w_{r_i}}, \qquad m_{r_i} = \mathrm{Judge}(P_k, \mathrm{Res}, r_i) = \begin{cases} 1, & \text{satisfied} \\ 0.5, & \text{partially satisfied} \\ 0, & \text{not satisfied} \end{cases}$$

Negative rubrics follow the same logic but with negative weights. To locate failure sources, the authors define the **Category Failure Rate** $\bar{F}_c = \frac{1}{|T_c|}\sum_{t\in T_c} \frac{n_{\text{fail},c,t}}{n_{\text{fail},t}}$, representing the proportion of "unsatisfied" rubrics contributed by category $c$ across tasks where category $c$ appeared.

## Key Experimental Results

### Main Results
Evaluation of three commercial DR systems on RESEARCHRUBRICS, showing overall compliance scores (higher is better):

| System | Ternary Compliance | Binary Compliance |
|------|---------|---------|
| Gemini DR | **0.677** | **0.615** |
| OpenAI DR | 0.664 | 0.597 |
| Perplexity DR | 0.566 | 0.487 |

Core Conclusion: **No system exceeded 70% in ternary compliance**, with the strongest (Gemini DR) at 67.7% (61.5% binary). This aligns with results from LiveResearchBench (<74%) and DeepResearch Bench (<50%), suggesting fundamental architectural limitations.

Failure rates decomposed by the six rubric axes (representative values, %):

| Evaluation Axis | Gemini | ChatGPT | Perplexity |
|--------|--------|---------|-----------|
| Explicit Requirements | Low (<20) | Low | Low |
| Implicit Requirements | 49.2 | 49.0 | 48.6 |
| Synthesis | 28.9 | 28.1 | 25.7 |
| Communication | 20.0 | 19.0 | 16.5 |

**Implicit Reasoning + Information Synthesis collectively account for 45–50% of all failures**, whereas explicit fact retrieval and communication quality failures are below 20%. Agents are good at "copying explicit facts and writing beautifully" but struggle with "inferring unstated needs and integrating multiple documents into coherent arguments."

### Ablation Study
Impact of rubric design on human-machine consistency (Macro F1):

| Configuration | Gemini DR (Binary F1) | Description |
|------|---------------------|------|
| Example Detail: Low | 0.733 | Rubrics provide only basic descriptions |
| Example Detail: High | **0.760** | Rubrics include short examples; consistency +3~4% |
| LLM Augmentation: Absent | 0.760 | Original human-written rubrics |
| LLM Augmentation: Present | 0.564 | LLM automatically rewrites rubrics; consistency drops −15~20% |

Binary vs. Ternary scoring consistency: Binary reaches 0.72–0.76 Macro F1, roughly 20 points higher than ternary (0.53–0.57). Gemini-1.5-Pro is the most reliable judge (Binary 0.76).

### Key Findings
- **Implicit Reasoning is the divide between "Excellent" and "Sufficient"**: Failures in mandatory rubrics occur mostly in explicit requirements and synthesis, while failures in implicit reasoning occur mostly in **optional** rubrics—indicating systems meet basic implicit needs but miss professional-level nuances.
- **Binary scoring is more reliable**: Switching from ternary to binary scoring increases human-machine consistency by approximately 20 points, suggesting "partially satisfied" introduces ambiguity without providing stronger discriminative power.
- **LLM-augmented rubrics are detrimental**: Allowing LLMs to rewrite or expand rubrics destroys human-machine consistency by 15–20%, confirming that rubrics must be human-written. Conversely, embedding short examples (e.g., specific policy names) improves consistency by 2–4%.
- **Breadth-Accuracy Trade-off**: Gemini DR produces 111 citations with 81% accuracy, while Perplexity produces only 31 citations with 90% accuracy—pursuing coverage sacrifices precision. Both struggle with the implicit judgment of "source relevance and authority."
- **Nesting-based Difficulty Increase**: Systems handle shallow reasoning well but drop points sharply as logical nesting increases (multi-step analysis/evaluation). Conceptual breadth also impacts performance but less severely than reasoning chain length.

## Highlights & Insights
- **Purely manual rubrics are the core strength**: Amidst the "LLM-evaluating-LLM" trend, the authors invested 2800+ hours in human creation/review, proving that machine-augmented rubrics degrade consistency.
- **Transferable Complexity Framework**: The three-axis coordinates are not just for benchmarking but are diagnostic tools that clarify exactly where a model is weak.
- **Mandatory/Optional + Negative Rubrics**: A 60% score could indicate a critical failure or just a lack of polish; this separation makes that clear, which is vital for deployment decisions.
- **Binary over Ternary Reliability**: The finding that binary scoring is more reliable than ternary is counter-intuitive but supported by data, warning researchers not to sacrifice reproducibility for perceived granularity.

## Limitations & Future Work
- **Small Prompt Set (101)**: Compared to the 2593 rubrics, the task count is small and solely single-turn, missing real-world multi-turn interactive research scenarios.
- **Limited Commercial Systems Evaluated**: Only OpenAI, Gemini, and Perplexity were tested. Open-source agents and diverse retrieval architectures were not included.
- **Rubric Ceiling**: Human rubrics avoid LLM anchoring but depend on expert pre-conceptions of a "good answer," potentially misjudging truly novel or out-of-the-box responses.
- **Dynamic Information Source Issues**: DR relies on the live web; evidence found today may change tomorrow, affecting the comparability of compliance scores over time.
- **Future Directions**: Scaling to multi-turn tasks, including open-source agents, and modeling "implicit requirement inference" into the agent's planning phase.

## Related Work & Insights
- **vs. DeepResearch Bench**: Both evaluate PhD-level reports, but DeepResearch Bench uses LLM-generated rubrics and reference reports, risking circular reasoning. Ours uses human-written rubrics and proves LLM augmentation is harmful via ablation.
- **vs. ExpertLongBench**: Closest to Ours in terms of domains and expert rubrics, but it relies on high-quality existing references (CLEAR framework) and professional domains (clinical, legal). Ours includes consumer research, representing a broader scope with higher rubrics per task (avg. 26).
- **vs. LiveResearchBench / Mind2Web2**: These use real prompts but LLM-generated (human-reviewed) rubrics, which suffer from anchoring bias. Ours is human-written from the source.
- **vs. HealthBench**: Borrows the use of Macro F1 for judge validation but extends the scope from medical to general DR, achieving higher human-machine consistency (0.76 vs 0.709).

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of the three-axis framework and purely human weighted rubrics is a first for DR evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 systems × 3 judges × Binary/Ternary × multi-dimensional analysis + ablation; very solid, though system count is small.
- Writing Quality: ⭐⭐⭐⭐ Motivation, framework, and scoring formulas are clear; failure attribution is insightful.
- Value: ⭐⭐⭐⭐⭐ Open-sourcing all prompts, rubrics, and code fills a significant gap in DR evaluation for domain diversity and expert rubrics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Personalized Deep Research: Benchmarks and Evaluations](towards_personalized_deep_research_benchmarks_and_evaluations.md)
- [\[ICLR 2026\] DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents](deepresearch_bench_a_comprehensive_benchmark_for_deep_research_agents.md)
- [\[ICLR 2026\] DRBench: A Realistic Benchmark for Enterprise Deep Research](drbench_a_realistic_benchmark_for_enterprise_deep_research.md)
- [\[ICLR 2026\] LiveResearchBench: A Live Benchmark for User-Centric Deep Research in the Wild](liveresearchbench_a_live_benchmark_for_user-centric_deep_research_in_the_wild.md)
- [\[ICLR 2026\] From Reproduction to Replication: Evaluating Research Agents with Progressive Code Masking](from_reproduction_to_replication_evaluating_research_agents_with_progressive_cod.md)

</div>

<!-- RELATED:END -->
