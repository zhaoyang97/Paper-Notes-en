---
title: >-
  [Paper Note] KRISTEVA: Close Reading as a Novel Task for Benchmarking Interpretive Reasoning
description: >-
  [ACL 2025][LLM Evaluation][Close Reading] This paper proposes KRISTEVA, the first benchmark to evaluate LLM close reading capabilities, consisting of 1331 multiple-choice questions constructed from university classroom data. It covers three progressive levels of difficulty: stylistic feature extraction, contextual retrieval, and feature-context multi-hop reasoning. Nineteen SOTA LLMs still lag behind human experts in 10 out of 11 tasks.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Close Reading"
  - "Literary Understanding Benchmark"
  - "Rhetorical Device"
  - "Multi-hop Reasoning"
  - "Figurative Language Understanding"
date: 2026-05-08
content_hash: 2437039147760ca0
---

# KRISTEVA: Close Reading as a Novel Task for Benchmarking Interpretive Reasoning

**Conference**: ACL 2025  
**arXiv**: [2505.09825](https://arxiv.org/abs/2505.09825)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Close Reading, Literary Understanding Benchmark, Rhetorical Device, Multi-hop Reasoning, Figurative Language Understanding

## TL;DR

This paper proposes KRISTEVA, the first benchmark to evaluate LLM close reading capabilities, consisting of 1331 multiple-choice questions constructed from university classroom data. It covers three progressive levels of difficulty: stylistic feature extraction, contextual retrieval, and feature-context multi-hop reasoning. Nineteen SOTA LLMs still lag behind human experts in 10 out of 11 tasks.

## Background & Motivation

**Background**: Although evaluation benchmarks for LLMs emerge continuously—MMLU for multi-disciplinary knowledge, GPQA for graduate-level reasoning, MathBench for mathematics, and PutnamBench for theorem proving—scarcely any benchmarks address the literary domain, and MMLU does not even include literature as a tested subject. OpenAI's own testing also demonstrates that ChatGPT and GPT-4 lag significantly in AP English compared to other AP subjects.

**Limitations of Prior Work**: The reasoning required for literature is fundamentally different from mathematical or logical reasoning—instead of seeking a "single correct answer," it requires evaluating which of several possible interpretations is more plausible and persuasive. This represents a ground truth of "relativism" rather than "positivism." Current NLP benchmarks (1) restrict figurative language understanding (FLU) to sentence-level metaphor identification and interpretation, neglecting reasoning about the deeper function, meaning, and effect of figurative language; (2) multi-hop machine reading comprehension (MRC) has not yet addressed the highly challenging domain of literary text.

**Key Challenge**: Close reading is a cornerstone of humanities education—millions of college course essays require students to perform close reading analyses annually, yet this ability has never been used to evaluate LLMs. Close reading requires identifying stylistic features (e.g., metaphor, alliteration) $\rightarrow$ understanding their function and effect $\rightarrow$ integrating external cultural/historical context $\rightarrow$ performing multi-hop reasoning to link features and context. This is a complex, high-level reasoning process.

**Goal**: Build a benchmark to evaluate the close reading ability of LLMs that (1) covers all stages of the close reading process, (2) includes tasks requiring reasoning rather than mere factual extraction, and (3) establishes human expert baselines as a reference.

**Key Insight**: The authors operationalize 11 evaluation tasks from three of the steps of the CRIT (Critical Reader's Interpretive Toolkit) pedagogical framework—paraphrase $\rightarrow$ observe $\rightarrow$ contextualize $\rightarrow$ analyze $\rightarrow$ argue $\rightarrow$ reflect—developed by the Department of English at the University of Texas at Austin (UT Austin).

**Core Idea**: High-scoring student essays are collected from university literature course exam data. GPT-4o is utilized to extract structured stylistic features and contextual information to construct questions and answers, and o1-preview is used to generate high-quality distractors, ultimately forming a benchmark of 1331 multiple-choice questions.

## Method

### Overall Architecture

The construction pipeline of KRISTEVA includes: (1) collecting 49 de-identified CRIT exam essays from an introductory world literature course at UT Austin; (2) filtering out essays with scores below 80; (3) employing GPT-4o to extract structured answers (types, locations, elements, and purposes of stylistic features, as well as external contextual info) from the remaining high-scoring essays; (4) using o1-preview to generate three semantically distinct but structurally similar distractor options for the 7 categories of questions requiring distractors; (5) randomly shuffling correct answers and distractors to form the final multiple-choice questions.

### Key Designs

1. **Progressively Harder Task Clusters**:

    - **Function**: Gradually evaluate the close reading capabilities of LLMs from low-level extraction to high-level reasoning.
    - **Mechanism**: Level 1 (stylistic features, Q1-Q6): Q1 detects rhetorical device type, Q2 localizes the device, Q3 explains the elements of the device, Q4 reasons about the purpose of the device, Q5 ranks the importance of multiple devices, and Q6 reasons about the effect of the device. Level 2 (contextual information, Q7-Q9): Q7 retrieves relevant cultural/historical contexts from parameterized knowledge, Q8 ranks the importance of multiple contexts, and Q9 reasons about the significance of the context. Level 3 (multi-hop reasoning, Q10-Q11): Q10 pairs features with contexts, and Q11 reasons about the most plausible explanation for the feature-context connection.
    - **Design Motivation**: This corresponds to the observe $\rightarrow$ contextualize $\rightarrow$ analyze steps of the CRIT framework. Each step contains non-reasoning (extraction/retrieval) and reasoning (judgment/inference) subtasks, ensuring the evaluation covers every cognitive level of the close reading process.

2. **Classroom-Data-Based Data Source Strategy**:

    - **Function**: Provide high-quality, pedagogically grounded evaluation data.
    - **Mechanism**: Real college exam essays are used as data sources, filtered by grading professors to retain only high-scoring essays ($\ge 80$ points). For essays with revision history, the revised versions (which are typically higher quality) are used instead of the originals. Professors perform a secondary manual check to ensure the correct answer for each question is distinguishable from three plausible distractors.
    - **Design Motivation**: Unlike obtaining data from standardized tests, classroom data provides higher-quality and more voluminous text. Student essays themselves are instances of applying the CRIT framework, directly containing structured information for feature extraction and contextual analysis.

3. **o1-preview Distractor Generation**:

    - **Function**: Generate high-quality and misleading incorrect options.
    - **Mechanism**: Distractors are generated using o1-preview rather than GPT-4o or Qwen. The distractors must structurally and syntactically mimic the correct answer (to avoid identification via formatting clues) but semantically offer less convincing interpretations. After generation, option order is randomly shuffled to eliminate positional bias.
    - **Design Motivation**: Manual inspection shows that distractors generated by o1-preview are of the highest average quality and the most challenging. Since literary interpretation lacks absolute right or wrong, distractors must be "less plausible but still somewhat reasonable" options, which requires strong literary understanding from the generator model.

### Loss & Training

This paper is a benchmark paper and does not involve model training. All LLMs are evaluated under a zero-shot setting, utilizing the Language Model Evaluation Harness framework to ensure reproducibility.

## Key Experimental Results

### Main Results

| Model | Non-Reasoning Avg | Reasoning Avg | Overall |
|------|-------------|------------|------|
| Random | 25.2 | 28.5 | 25.5 |
| OLMoE-1B-7B | 50.2 | 48.2 | 49.7 |
| Llama-3.1-8B | 64.9 | 58.0 | 62.5 |
| Qwen2.5-32B | 71.4 | 63.3 | 68.5 |
| GPT-4o | 67.9 | 63.4 | 67.5 |
| o1-preview | 67.8 | 61.5 | 66.8 |
| **Phi-4 (Best)** | **72.2** | **64.3** | **69.7** |
| Human (Weighted Avg) | 70.8 | 50.0 | 65.6 |
| Human (Best Eval2) | 82.5 | 50.5 | 74.7 |

### Ablation Study

Q1 (feature type identification) is the hardest extraction task—the best model, Phi-4, achieves only 49.3%, whereas the best human achieves 66.7%.

| Task | Best LLM | Best Human | LLM Lags? |
|------|---------|---------|----------|
| Q1 Feature Type | 49.3 (Phi-4) | 66.7 (Eval2) | Yes, by 17.4% |
| Q2 Feature Location | 98.6 (Qwen-32B) | 100 (Eval1,2) | Roughly equal |
| Q5 Feature Ranking | 62.3 (Phi-4) | 75.0 (Eval1) | Yes, by 12.7% |
| Q10 Feature-Context Match | 47.0 (o1-preview) | 71.4 (Eval1) | Yes, by 24.4% |
| Q11 Feature-Context Reason | 91.0 (GPT-4o-mini) | 100 (Eval1) | Yes, by 9% |

### Key Findings

- **Phi-4 (14B) achieves the best overall performance with the smallest parameter count**: This is likely due to its high-quality textbook training data having a stronger distribution affinity with the university classroom data sources of KRISTEVA. This suggests that data quality may be more important than model scale for interpretive reasoning capabilities.
- **Reasoning models (o1-preview) show no significant advantage**: Although performing best on three out of three multi-hop reasoning tasks (Q8, Q10, Q11), it is overall inferior to Phi-4 and GPT-4o. This is consistent with recent research showing that CoT primarily improves mathematical and logical reasoning but has limited benefit for common sense and "soft reasoning."
- **Substantial variance exists between human evaluators** (standard dev of 29.3 vs 5.47 for LLMs): Evaluators from different disciplinary backgrounds (English vs Classics) excel in different types of tasks. English students perform stronger on reasoning tasks, whereas Classics students excel in extraction and contextual tasks.
- **LLMs lag behind the best human performance on 10/11 tasks**, with the sole exception of Q2 (Feature Location), where models achieve nearly 100%. The largest gap is in Q10 (Feature-Context Match), with the best model at 47.0% vs the human at 71.4%, a gap of 24.4%.

## Highlights & Insights

- **Filling the gap at the intersection of literature and NLP**: This is the first work to operationalize close reading—the most central methodology in the humanities—into an LLM evaluation benchmark. It not only tests the capabilities of LLMs but also provides the NLP community with a new reasoning paradigm: determining the "more plausible" choice rather than a "sole correct" answer.
- **Organic fusion of FLU and MRC**: KRISTEVA naturally unifies two long-independent NLP challenges—figurative language understanding and multi-hop reading comprehension—into a progressive chain of reasoning within the framework of close reading.
- **Research value of educational data**: The paper notes that humanities classrooms generate a massive volume of high-quality textual data annually (millions of close reading essays). If ethically harvested, this can serve as a valuable resource for NLP research.

## Limitations & Future Work

- **Limited data source scope**: The data comes from only one course, one professor, and an introductory-level syllabus. Different universities and literary traditions might introduce diverse analytical perspectives.
- **MCQ format disadvantage for humans**: The distractors in the multiple-choice questions (MCQs) are generated by LLMs, which might make them easier for LLMs to differentiate, whereas they may paradoxically confuse humans. Evaluators reported needing an adjustment period to get used to the MCQ format.
- **English-only texts**: Although the test texts include works in translation, all questions and answers are in English. Different linguistic literary traditions might require different evaluation dimensions.
- **Small human baseline sample size**: Comprising only 3 evaluators, representing a "human expert level" comprehensively is difficult.

## Related Work & Insights

- **vs MMLU/GPQA**: These multi-disciplinary benchmarks entirely lack the literary domain. KRISTEVA fills this gap and demonstrates the unique challenges of literary reasoning—even the strongest LLMs achieve only ~70% accuracy.
- **vs FLUTE (Chakrabarty et al., 2022)**: FLUTE evaluates figurative language understanding but is limited to sentence-level metaphor interpretation. KRISTEVA extends FLU to the discourse level and incorporates reasoning on function (Q4), significance (Q5), and effect (Q6)—which is unprecedented in FLU studies.
- **vs WenMind (Cao et al., 2024)**: WenMind evaluates LLMs' knowledge of classical Chinese literature but focuses on rote memorization rather than interpretive reasoning. KRISTEVA emphasizes the reasoning process itself.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to operationalize close reading into an LLM benchmark; the task design is pedagogically grounded and opens a brand-new evaluation dimension.
- Experimental Thoroughness: ⭐⭐⭐⭐ 19 models + human baselines, but the human baseline sample size is small, and few-shot experiments are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent academic depth, elegantly merging humanities theory with NLP evaluation methodology.
- Value: ⭐⭐⭐⭐ Significant driving force for the direction of LLM evaluation, though the application scenario remains relatively niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset](mars_benchmarking_the_metaphysical_reasoning_abilities_of_language_models_with_a.md)
- [\[ACL 2025\] FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging](financereasoning_benchmarking_financial_numerical_reasoning_more.md)
- [\[ACL 2025\] MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark](mmlu-cf_a_contamination-free_multi-task_language_understanding_benchmark.md)
- [\[ACL 2025\] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)
- [\[AAAI 2026\] Do LLMs Really Struggle at NL-FOL Translation? Revealing Their Strengths via a Novel Benchmarking Strategy](../../AAAI2026/llm_evaluation/do_llms_really_struggle_at_nl-fol_translation_revealing_their_strengths_via_a_no.md)

</div>

<!-- RELATED:END -->
