---
title: >-
  [Paper Note] MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset
description: >-
  [ACL 2025][LLM Evaluation] This paper proposes a formal definition of **Metaphysical Reasoning**, formulating reasoning under distributional changes as a three-step discrimination process. It constructs Mars (355K annotated instances), the first large-scale evaluation benchmark. Experiments demonstrate that over 20 language models perform poorly on this task, revealing a significant weakness of LLMs in understanding modifications of event components and their causal effects.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
date: 2026-05-08
content_hash: c2827f08ed47105c
---

# MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset

**Conference**: ACL 2025  
**arXiv**: [2406.02106](https://arxiv.org/abs/2406.02106)  
**Code**: [GitHub](https://github.com/HKUST-KnowComp/MARS)  
**Authors**: Weiqi Wang, Yangqiu Song (HKUST)

## TL;DR

This paper proposes a formal definition of **Metaphysical Reasoning**, formulating reasoning under distributional changes as a three-step discrimination process. It constructs Mars (355K annotated instances), the first large-scale evaluation benchmark. Experiments demonstrate that over 20 language models perform poorly on this task, revealing a significant weakness of LLMs in understanding modifications of event components and their causal effects.

## Background & Motivation

- **Core Problem**: To enable LLMs to become conscious agents with generalized reasoning capabilities, a key ability is to understand **distributional situational changes** triggered by environmental factors or other agents' actions. For example, when the weather transitions from sunny to rainy, the distribution of driver behaviors changes accordingly.
- **Limitations of Prior Work**:
  1. The scope of possible changes in events is extremely vast, making it impossible for existing knowledge bases to cover exhaustively.
  2. Reasoning under distributional changes lacks a clear formal task definition.
  3. Existing benchmarks (such as PlanBench, TRAC, etc.) only cover limited scenarios and change types, while overlooking the transitions caused by such changes.
- **Goal**: To formally define metaphysical reasoning and construct the first large-scale benchmark for systematically evaluating LLMs' reasoning capabilities under distributional changes.

## Method

### 1. Formal Definition of Event Changes

An event $e$ is represented as a function of seven categories of components: $e = f(s, v, o, t, l, n, se)$, corresponding to subject, verb, object, temporal quantifier, spatial quantifier, numerical attribute, and sub-event, respectively. A change is implemented by replacing one of these components.

For $s$, $v$, $o$, $se$, **conceptualization** is applied—progressively elevating instances to more abstract concepts; for $t$, $l$, $n$, **numerical mutation** is used—step-wise increasing numerical or spatial values. This constructs a hierarchical distribution of changes.

### 2. Three-Step Discrimination Process

| Step | Task Name | Core Problem | Input & Output |
|------|----------|--------------|----------------|
| Step 1 | Metaphysical Event Discrimination | Is the post-change event plausible in reality? | Original event $e$ + post-change event $e'$ $\rightarrow$ Binary classification |
| Step 2 | Metaphysical Inference Discrimination | Is the inference of the post-change event plausible? | Post-change event $e'$ + inferred state $i$ $\rightarrow$ Binary classification |
| Step 3 | Metaphysical Transition Reasoning | What additional change is needed to make an implausible inference plausible? | Post-change event $e'$ + metaphysical inference $i$ + additional change $c'$ $\rightarrow$ Binary classification |

### 3. Data Construction Pipeline

A pipeline combining ChatGPT generation and human annotation is utilized:

1. **Text Decomposition and Extraction**: Events are extracted from Wikitext and BookCorpus, and ChatGPT is guided via few-shot prompting to decompose the text and extract the seven component categories.
2. **Component Abstraction and Mutation**: Three conceptual or numerical mutations with progressively higher abstraction levels are generated for each component.
3. **Inference Generation**: One plausible inference and one metaphysical inference are generated for each post-change event.
4. **Transition Generation**: Additional changes that make the metaphysical inference plausible are generated.
5. **Human Annotation**: Annotations are collected via AMT with 5 votes per item, achieving an IAA of 81% and a Fleiss Kappa of 0.56; the expert verification accuracy is 93.67%.

## Key Experimental Results

### Table 1: Data scale of Mars tasks

| Task | Text Count | Event Count | Train Set | Test Set | Total | Expert Agreement |
|------|------------|-------------|-----------|-----------|-------|------------------|
| Meta. Event | 9,998 | 55,190 | 96,004 | 11,982 | 119,999 | 94.0% |
| Meta. Inference | 9,837 | 35,528 | 96,009 | 11,981 | 120,000 | 96.5% |
| Meta. Transition | 9,677 | 31,447 | 92,495 | 11,560 | 115,618 | 93.5% |

### Table 2: Main Experimental Results (Accuracy %)

| Model | Setting | Event Acc | Inference Acc | Transition Acc |
|------|------|-----------|---------------|----------------|
| DeBERTa-Large | Zero-shot | 48.27 | 47.73 | 50.73 |
| DeBERTa-Large | Fine-tuned | 64.45 | 69.57 | 72.93 |
| VERA 11B | Zero-shot | 51.82 | 60.97 | 61.31 |
| LLaMa-3-70B | Zero-shot | 57.41 | 63.40 | 60.15 |
| LLaMa-3.1-70B | Zero-shot | 59.22 | 63.61 | 61.28 |
| LLaMa-3.1-70B + RAG | Zero-shot | 61.21 | 66.38 | 61.53 |
| LLaMa-3.1-405B | Zero-shot | 60.01 | 64.52 | 61.74 |
| Gemma-2-9B | Fine-tuned | 61.23 | 69.24 | 73.30 |
| GPT-4 | Zero-shot | 53.90 | 51.20 | 49.41 |
| GPT-4 (COT) | Zero-shot | 51.28 | 51.49 | 47.62 |
| GPT-4o-mini + RAG | Zero-shot | 59.99 | 54.54 | 49.39 |

**Key Findings**:

- All models perform poorly in the zero-shot setting, with the best-performing LLM (LLaMa-3.1-405B) achieving only 60% accuracy on the Event task.
- The best fine-tuned result is around 74%, leaving substantial room for improvement.
- The GPT-4 series surprisingly underperforms open-source LLMs, possibly because negative examples are generated by ChatGPT, leading to contradictions with GPT's internal knowledge.
- Advanced prompting methods like CoT and few-shot learning yield only limited improvements.

### Table 3: Performance of Conceptual Knowledge Transfer

| Model | Training Data | Event Acc | Inference Acc | Transition Acc |
|------|----------|-----------|---------------|----------------|
| DeBERTa 435M | Mars | 64.45 | 69.57 | 72.93 |
| DeBERTa 435M | CANDLE + Mars | **64.95** | **71.85** | **74.39** |
| LLaMa-3 8B | Mars | 60.06 | 65.76 | 69.83 |
| LLaMa-3 8B | CANDLE + Mars | **60.93** | **69.13** | **74.09** |

Pre-training on CANDLE conceptual knowledge before fine-tuning on Mars consistently improves performance across all three tasks, indicating that abstract conceptual knowledge helps enhance metaphysical reasoning capabilities.

## Highlights & Insights

1. **Novel Task Formulation**: Formulates reasoning under distributional changes as a three-step discrimination process (event discrimination $\rightarrow$ inference discrimination $\rightarrow$ transition reasoning) for the first time, covering feasibility, consequences, and motivations of changes.
2. **Large-scale & High-quality Benchmark**: Comprises 355K annotated data, 3 tasks, and 7 categories of changes, with an expert verification agreement rate of >93%, which far exceeds the scale of similar benchmarks.
3. **Systematic Error Analysis**: Attributes GPT-4's errors to hallucinations (41.7%), confusion between concepts and hypernyms (36.3%), internal contradictions (17.7%), and annotation errors (4.3%), clearly revealing LLMs' failure modes.
4. **Scalable Solutions**: Demonstrates that conceptual knowledge transfer from CANDLE can improve performance. Since CANDLE is automatically constructed without human annotation, it provides a low-cost enhancement path.

## Limitations & Future Work

1. **Limited Types of Changes**: Only seven component changes are defined, omitting other mutable components such as adjectives, adverbs, and prepositional phrases.
2. **Reliance on Closed-Source Models**: The data construction pipeline relies on ChatGPT, leading to high costs and limited reproducibility.
3. **Lack of Practical Solutions**: The paper focuses on building the evaluation benchmark and does not explore systematic methods to enhance LLMs' metaphysical reasoning.
4. **Weak Spatial-Temporal and Numerical Reasoning**: Analysis shows that LLMs perform worst on reasoning about spatial, temporal, and numerical changes, and CANDLE pre-training offers no substantial benefit here.

## Related Work & Insights

- **Reasoning under Distributional Changes**: Work like Propara (Dalvi et al., 2018), TRAC (He et al., 2023b), and PlanBench (Valmeekam et al., 2023) focuses on tracking state changes and logical reasoning in limited scenarios. Mars is the first to comprehensively cover change plausibility, consequences, and transitions.
- **Conceptual Abstraction**: AbsATM (He et al., 2024) and AbsPyramid (Wang et al., 2024d) provide conceptualized data resources; conceptual knowledge from CANDLE (Wang et al., 2024b) can be transferred to enhance reasoning.
- **LLM Benchmarking**: Unlike commonsense reasoning benchmarks (such as ATOMIC and ConceptNet), Mars focuses on reasoning in out-of-distribution abstract scenarios, aligning closely with the goal of System II reasoning.

## Rating

- ⭐ Novelty: 4/5 — Novel formal definition of three-step metaphysical reasoning; first large-scale benchmark covering reasoning under distributional changes.
- ⭐ Experimental Thoroughness: 5/5 — Evaluates 20+ models under various settings (zero-shot/fine-tuned/API/RAG/CoT), and includes knowledge transfer analysis, component-level analysis, and error analysis.
- ⭐ Writing Quality: 4/5 — Well-structured and rigorous definitions, though the term "metaphysical" deviates significantly from its traditional philosophical meaning, potentially causing confusion.
- ⭐ Value: 4/5 — Reveals LLMs' critical weaknesses in abstract reasoning, but lacks practical enhancement solutions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)
- [\[ACL 2025\] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)
- [\[ACL 2025\] KRISTEVA: Close Reading as a Novel Task for Benchmarking Interpretive Reasoning](kristeva_close_reading_as_a_novel_task_for_benchmarking_interpretive_reasoning.md)
- [\[ACL 2025\] EcomScriptBench: A Multi-task Benchmark for E-commerce Script Planning via Step-wise Intention-Driven Product Association](ecomscriptbench.md)

</div>

<!-- RELATED:END -->
