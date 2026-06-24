---
title: >-
  [Paper Note] LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks
description: >-
  [ACL 2025][LLM Efficiency][Long-context Benchmark] LongBench v2 is a challenging long-context evaluation benchmark consisting of 503 hard multiple-choice questions, with context lengths ranging from 8k to 2M tokens, covering six major task types. Under a 15-minute time limit, human experts only achieve an accuracy of 53.7%, while the strongest direct-generation model (GPT-4o 2024-08) achieves only 50.1%, and the reasoning model o1-preview reaches 57.7%…
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "Long-context Benchmark"
  - "Deep Understanding"
  - "Reasoning Evaluation"
  - "Multi-task"
  - "Test-time Computation"
date: 2026-05-08
content_hash: d8e19f4290a293bc
---

# LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks

**Conference**: ACL 2025  
**arXiv**: [2412.15204](https://arxiv.org/abs/2412.15204)  
**Code**: [GitHub](https://github.com/THUDM/LongBench)  
**Area**: LLM Efficiency / Long-context Evaluation  
**Keywords**: Long-context Benchmark, Deep Understanding, Reasoning Evaluation, Multi-task, Test-time Computation

## TL;DR

LongBench v2 is a challenging long-context evaluation benchmark consisting of 503 hard multiple-choice questions, with context lengths ranging from 8k to 2M tokens, covering six major task types. Under a 15-minute time limit, human experts only achieve an accuracy of 53.7%, while the strongest direct-generation model (GPT-4o 2024-08) achieves only 50.1%, and the reasoning model o1-preview reaches 57.7%, highlighting the critical importance of test-time compute scaling for deep long-context understanding.

## Background & Motivation

**Background**: The context window of LLMs has been continuously expanding (128k $\rightarrow$ 1M $\rightarrow$ 4M tokens), but existing long-context benchmarks (such as LongBench v1, RULER, and Needle-in-a-Haystack) primarily evaluate shallow information retrieval and extraction capabilities rather than true deep understanding and reasoning.

**Limitations of Prior Work**: (1) Questions in existing benchmarks can often be solved through simple keyword search or pattern matching, failing to reflect the actual comprehension needs of long documents in real-world scenarios. (2) Insufficient difficulty leads to low discriminative power among models, with many approaching saturation on existing benchmarks. (3) Data sources and task types are monolithic, lacking coverage of realistic scenarios like codebase comprehension and structured data understanding.

**Key Challenge**: While models claim to support ultra-long contexts, their performance on tasks requiring genuine understanding and reasoning over entire long documents is far inferior to that on simple retrieval tasks. Existing benchmarks fail to expose this gap.

**Goal**: To construct a truly challenging long-context benchmark that demands deep understanding and logical reasoning from models rather than simple information retrieval.

**Key Insight**: Engage nearly 100 highly educated individuals from diverse professional backgrounds in data construction to ensure question quality, retaining only those questions that require in-depth reading and deliberate reasoning to answer.

**Core Idea**: Build a highly challenging, long-context multiple-choice benchmark through a rigorous automated and manual quality control pipeline, making it sufficient to differentiate current state-of-the-art LLMs and expose their limitations in deep understanding.

## Method

### Overall Architecture

LongBench v2 is an evaluation benchmark and does not involve training methodologies. The core contribution lies in the design, collection, and quality control of the dataset. Annotators upload long documents and propose multiple-choice questions, which are incorporated into the benchmark after automated and manual review. During evaluation, models receive the full context and the question, then output the selected option.

### Key Designs

1. **Multi-level Task Design**:

    - **Function**: Covers six major categories of tasks under realistic long-context scenarios.
    - **Mechanism**: The six major task categories include: (1) Single-document QA—deep understanding based on a single long document; (2) Multi-document QA—information integration and comparison across multiple documents; (3) Long In-Context Learning (Long ICL)—learning and applying patterns in ultra-long contexts; (4) Long Dialogue History—understanding logical threads in long conversations; (5) Codebase Comprehension—analyzing the structures and dependencies of large codebases; and (6) Long Structured Data Understanding—processing structured information such as tables and databases. Each task category demands deep reasoning beyond simple retrieval.
    - **Design Motivation**: Real-world application of long contexts extends far beyond "needle-in-a-haystack" retrieval, necessitating comprehensive coverage.

2. **Rigorous Quality Control Pipeline (Automated + Manual Review)**:

    - **Function**: Ensures sufficient difficulty and quality for each question.
    - **Mechanism**: Once proposed by annotators, questions undergo a two-tier review process: (1) automated review to check if the question can be resolved by simple text search and whether all distractors are misleading; (2) manual review to verify answer correctness and reasoning depth. Only data passing all checks receive annotation rewards; otherwise, annotators must revise and resubmit. Finally, human experts test the dataset with a 15-minute time limit, confirming that even human experts cannot solve them easily.
    - **Design Motivation**: To prevent "benchmark gaming" and ensure evaluation outcomes genuinely reflect the models' deep understanding capabilities.

3. **Multi-dimensional Evaluation Design**:

    - **Function**: Analyzes model performance on long-context tasks from multiple perspectives.
    - **Mechanism**: The 503 questions are categorized into three length intervals based on context length: Short (0-32k words), Medium (32k-128k words), and Long (128k-2M words). Meanwhile, two test settings are supported: w/o CoT (direct answering) and w/ CoT (allowing reasoning before outputting the answer). All questions are formatted as multiple-choice to ensure objective and reproducible evaluation.
    - **Design Motivation**: Length-based stratification helps analyze performance decay patterns across different context scales.

### Loss & Training

As an evaluation benchmark, there is no training process.

## Key Experimental Results

### Main Results

| Model | Context | Overall | Short | Medium | Long |
|------|--------|---------|-------|--------|------|
| Gemini-2.5-Pro 🧠 | 1M | **63.3** | 75.0 | 56.1 | 71.0 |
| o1-preview 🧠 | 128k | 57.7 | 66.8 | 52.1 | 58.1 |
| Human Expert | - | 53.7 | 100* | 25.1 | 53.7 |
| GPT-4o (2024-08) | 128k | 51.2 | 57.9 | 47.1 | 47.7 |
| Claude 3.5 Sonnet | 200k | 46.7 | 55.2 | 41.5 | 44.4 |
| Qwen2.5-72B | 128k | 43.5 | 47.9 | 40.8 | 39.8 |
| Llama 3.1 70B | 128k | 36.2 | 35.9 | 36.3 | 25.9 |
| Random Guess | - | 25.0 | 25.0 | 25.0 | 25.0 |

### Ablation Study

| Dimension | Finding |
|----------|------|
| Reasoning vs Non-reasoning Models | Reasoning models (o1-preview, DeepSeek-R1) significantly outperform direct-generation models. |
| Impact of Length | The Medium segment is the most challenging, while the Long segment is not necessarily worse (due to different task distributions). |
| w/ CoT vs w/o CoT | CoT brings significant improvements to most models (3-10 percentage points). |
| RAG Assistance | Long-context LLM + RAG shows improvements in some tasks, but not all. |

### Key Findings

- Under a 15-minute limit, human experts achieve only 53.7%, and opt out of answering on 8% of the questions, verifying the high difficulty of the benchmark.
- o1-preview (57.7%) is the first model to surpass the human baseline, indicating that test-time compute scaling is vital for long-context understanding.
- A substantial gap exists between direct-generation models (with GPT-4o best at 50.1%) and reasoning models, showing that deep understanding require extended thinking.
- Models do not necessarily score highest on the Short subset, because the Short subset filters out "easy" questions based on human performance.
- The medium-length range (32k-128k) is the most challenging for models, possibly because this length requires both global comprehension and capturing local details.
- As of April 2025, Gemini-2.5-Pro leads the leaderboard at 63.3%, showcasing a distinct advantage in supporting 1M contexts.

## Highlights & Insights

- **Test-time Compute Scaling vs. Long-context Window Expansion**: Long-context capability does not rely solely on wider windows; deep thinking during reasoning is equally crucial. This provides key insights for the developmental trajectory of long-context LLMs, suggesting parallel investments in reasoning capabilities.
- **"Humans as a surmountable baseline rather than an absolute ceiling"**: Setting a 15-minute time limit renders the human baseline a meaningful reference point rather than an unreachable ceiling. This design allows the benchmark to incentivize models to surpass human performance.
- **Engineering Practices of Data Quality Control**: The dual-review and revise-and-resubmit annotation mechanism guarantees high quality for every question, serving as a valuable reference.

## Limitations & Future Work

- The benchmark size of 503 questions is relatively limited, potentially introducing data selection bias.
- Although the multiple-choice format evaluates performance objectively, it does not fully reflect open-ended long-text comprehension capabilities.
- Annotators are primarily highly educated individuals, which may bias the questions toward academic scenarios.
- The Long subset has fewer questions, potentially limiting its statistical significance.
- The human baseline is built on a 15-minute limit; human performance may vary significantly under different time constraints.
- Future work may consider incorporating multilingual and multimodal long-context evaluations.

## Related Work & Insights

- **vs. LongBench v1**: v1 mainly tests shallow retrieval and simple extraction, whereas v2 focuses on deep understanding and reasoning, offering significantly increased difficulty.
- **vs. RULER**: RULER tests the limits of model memory and retrieval within fixed question templates, whereas v2 covers realistic application scenarios.
- **vs. Needle-in-a-Haystack**: The classic "needle-in-a-haystack" test is fundamentally a string-matching task that cannot evaluate comprehension capabilities, whereas v2 demands logical reasoning.
- **vs. $\infty$Bench**: While both target ultra-long context evaluation, v2 enforces more rigorous quality control and covers a broader range of practical task types.

## Rating

- Novelty: ⭐⭐⭐⭐ Elevating long-context evaluation from simple "retrieval" to a new dimension of "deep understanding & reasoning".
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage with evaluation on over 35 models, a human baseline, and a continuously updated leaderboard.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definitions, excellent visualizations, and a highly interactive project website.
- Value: ⭐⭐⭐⭐⭐ Has established itself as a key benchmark for evaluating long-context capabilities, featuring a continuously active leaderboard.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] FocusLLM: Precise Understanding of Long Context by Dynamic Condensing](focusllm_precise_understanding_of_long_context_by_dynamic_condensing.md)
- [\[ACL 2025\] Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models](ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)
- [\[ACL 2025\] On Many-Shot In-Context Learning for Long-Context Evaluation](on_many-shot_in-context_learning_for_long-context_evaluation.md)
- [\[ACL 2025\] LongReward: Improving Long-context Large Language Models with AI Feedback](longreward_improving_long-context_large_language_models_with_ai_feedback.md)
- [\[NeurIPS 2025\] LooGLE v2: Are LLMs Ready for Real World Long Dependency Challenges?](../../NeurIPS2025/llm_efficiency/loogle_v2_are_llms_ready_for_real_world_long_dependency_challenges.md)

</div>

<!-- RELATED:END -->
