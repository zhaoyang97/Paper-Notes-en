---
title: >-
  [Paper Note] TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues
description: >-
  [ACL2025][LLM (Other)][Temporal reasoning] The TReMu framework is proposed. By employing time-aware memorization (timeline summarization) and neuro-symbolic temporal reasoning (LLM-generated Python code execution for temporal calculations), it improves the accuracy of GPT-4o on a multi-session dialogue temporal reasoning benchmark from 29.83% to 77.67%.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "Temporal reasoning"
  - "multi-session dialogue"
  - "neuro-symbolic reasoning"
  - "memory-augmented Agent"
  - "timeline summarization"
date: 2026-05-08
content_hash: 917944cba2ed93b1
---

<!-- Generated automatically by src/gen_stubs.py -->
# TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues

**Conference**: ACL2025  
**arXiv**: [2502.01630](https://arxiv.org/abs/2502.01630)  
**Code**: To be confirmed  
**Area**: LLM/NLP  
**Keywords**: Temporal reasoning, multi-session dialogue, neuro-symbolic reasoning, memory-augmented Agent, timeline summarization

## TL;DR
The TReMu framework is proposed. By employing time-aware memorization (timeline summarization) and neuro-symbolic temporal reasoning (LLM-generated Python code execution for temporal calculations), it improves the accuracy of GPT-4o on a multi-session dialogue temporal reasoning benchmark from 29.83% to 77.67%.

## Background & Motivation

1. **Temporal reasoning is a weak point for LLMs**: Existing studies (such as TimeBench, TRAM) show that even state-of-the-art LLMs perform poorly on temporal reasoning tasks, especially when facing complex temporal relations.
2. **The temporal challenges in multi-session dialogues are underestimated**: Existing temporal reasoning benchmarks are mostly based on short texts (stories, Wikipedia) and do not fully consider the specific temporal characteristics in multi-session dialogues.
3. **Pervasive relative temporal expressions**: In multi-turn dialogues, speakers frequently use relative temporal expressions such as "last week" or "yesterday" instead of concrete dates; the model needs to infer the actual time from the dialogue context.
4. **Cross-session dependencies increase reasoning difficulty**: Events involving the same entity across different dialogue sessions need to be associated. For example, "mentioning a startup idea" and "actually starting the business a few months later" are distributed across different sessions.
5. **"Historical noise" introduced by long dialogue history**: As the number of dialogue turns increases, the accumulation of irrelevant historical information interferes with the LLM's extraction of key temporal details, reducing the signal-to-noise ratio.
6. **Lack of temporal reasoning evaluation benchmarks for multi-session dialogues**: Existing dialogue-related benchmarks (e.g., TimeDial, LoCoMo) do not explicitly cover the two key temporal features of relative time and cross-session dependency.

## Core Problem
How to improve the temporal reasoning capability of LLM-Agents in long multi-session dialogues, especially in processing relative temporal expressions and cross-session event dependencies?

## Method

### Overall Architecture
TReMu is based on the three-stage pipeline (memorization → retrieval → response) of memory-augmented LLM-Agents, with improvements made in both the memorization and reasoning stages.

### Component One: Time-aware Memorization

**Temporal Memory Writing**:
- For each dialogue session, the LLM is guided to not only generate a summary but also extract events and infer their specific occurrence dates.
- Key distinction: The time when an event is "mentioned" vs. the time when the event "occurs". For example, if a dialogue on January 28 mentions "cooking Italian food last Saturday", the memory should record that the event occurred on January 25.
- The output consists of fine-grained (time, event summary) pairs, instead of traditional holistic session summaries.

**Memory Organization**:
- All memory entries are organized in a timeline format, where events occurring at the same time are grouped together.
- Indexing is based on inferred time steps, supporting highly efficient time-based retrieval.
- Contrast with the topical summarization of MemoChat: MemoChat outputs topic summaries like "Hobbies and Daily Rituals", whereas TReMu outputs time-anchored memories such as "01/25/2020: Michelle cooked Italian food".

### Component Two: Neuro-symbolic Temporal Reasoning

- Given a temporal question and the retrieved relevant memories, the LLM is guided to generate Python code as an intermediate reasoning process.
- Utilize Python's `datetime` and `dateutil` libraries for precise temporal calculations.
- Predefined helper functions (e.g., `weekRange(t)` which returns the start and end dates of the week containing $t$) are provided to support function calling.
- `dateutil.relativedelta` can handle relative temporal calculations such as "next Friday".
- Executing the code line-by-line is equivalent to the step-by-step reasoning of CoT, but it leverages the precision of programming languages to avoid vague errors typical of natural language reasoning.
- The code execution results serve as intermediate evidence, based on which the LLM generates the final answer.

### Benchmark Construction
- Enhanced and constructed based on the LoCoMo dataset (averaging 304.9 turns, 19.3 sessions, and 9209.2 tokens per dialogue).
- Utilized a four-step pipeline with GPT-4o: event extraction → cross-session event linking → QA generation → human quality control.
- Three question types: Temporal Anchoring (264 questions, inferring the exact time of events), Temporal Precedence (102 questions, event ordering), and Temporal Interval (234 questions, temporal gaps between events).
- Contains 112 unanswerable questions, totaling 600 questions.

## Key Experimental Results

### GPT-4o Results (Table 5)

| Method | TA | TP | TI | Overall Accuracy | Unanswerable F1 |
|------|-----|-----|-----|-----------|---------|
| Standard Prompting | 18.18 | 58.82 | 30.34 | 29.83 | 20.84 |
| CoT | 67.80 | 74.51 | 49.15 | 61.67 | 43.18 |
| MemoChat | 35.23 | 43.14 | 25.21 | 32.67 | 37.02 |
| MemoChat + CoT | 51.14 | 49.02 | 26.50 | 41.67 | 38.00 |
| Timeline + CoT | 83.33 | 78.41 | 58.55 | 71.50 | 52.84 |
| **TReMu** | **84.47** | **81.37** | **68.38** | **77.67** | **64.42** |

### Cross-Model Comparison (Overall Accuracy)

| Method | GPT-4o | GPT-4o-mini | GPT-3.5-Turbo |
|------|--------|-------------|---------------|
| SP | 29.83 | 29.00 | 23.83 |
| CoT | 61.67 | 45.67 | 25.83 |
| TReMu | **77.67** | **51.17** | **33.67** |

### Key Findings
- TReMu achieves the best accuracy and F1 score across all three LLMs.
- Time-aware memorization contributes significantly: MemoChat+CoT → Timeline+CoT, where the accuracy of GPT-4o improves from 41.67 to 71.50 (+29.83).
- Neuro-symbolic reasoning brings further improvement: Timeline+CoT → TReMu, with accuracy increasing from 71.50 to 77.67 (+6.17).
- Code execution failure rates are generally very low, with GPT-4o being the lowest, demonstrating the reliability of the Python-code-based reasoning scheme.
- For GPT-4o/mini, directly using a memory mechanism is actually inferior to CoT, because these models have a sufficiently long context window; however, GPT-3.5 requires memory mechanisms due to input limitations.

## Highlights & Insights

1. **Exquisite design of time-aware memory**: Making a distinction between the "occurrence time" and "mention time" of events, and organizing memory in a timeline format, which addresses the ambiguity of relative time.
2. **Pragmatic and effective neuro-symbolic reasoning**: Leveraging the LLM's proficiency in writing Python code, the temporal calculation is outsourced to a precise symbolic executor, avoiding the vague errors of natural language reasoning.
3. **Reusable benchmark construction methodology**: The four-step pipeline (extraction → linking → QA generation → quality control) provides a template for constructing reasoning evaluation benchmarks in other fields.
4. **Clear isolation of contributions in ablation studies**: Comparable combinations of MemoChat/Timeline/CoT/symbolic reasoning clearly demonstrate the incremental gains of each component.

## Limitations & Future Work

1. **Evaluated only under multiple-choice QA settings**: Generative dialogue scenarios were not tested, and multiple-choice questions may underestimate the difficulty of temporal reasoning in practical applications.
2. **Reliance on closed-source LLMs**: Only the GPT series were tested. Open-source models (such as Llama-3-70B) fail to adapt due to context length limits and insufficient instruction-following capabilities.
3. **Limited benchmark scale**: The 600 questions are constructed from a small number of dialogues in LoCoMo, leaving diversity and generalizability yet to be verified.
4. **The memorization stage itself requires LLM reasoning**: The generation of timeline summaries relies on the temporal inference capabilities of the LLM. If the LLM itself makes an incorrect inference, it will lead to cascading errors in downstream reasoning.
5. **Potential for Python code generation errors**: Although the failure rate is low, a regeneration mechanism is still needed, and the systematic overhead of such retry strategies is not discussed.

## Related Work & Insights

### vs MemoChat (Lu et al., 2023)
MemoChat generates topical session summaries (e.g., "Hobbies and Daily Rituals"), which loses fine-grained temporal information. TReMu generates time-anchored event memories (e.g., "01/25/2020: Michelle cooked Italian food"), directly supporting temporal reasoning. Experiments show that TReMu outperforms MemoChat+CoT by 36 percentage points on GPT-4o.

### vs CoT (Wei et al., 2022)
CoT performs step-by-step reasoning in natural language, which is prone to errors in temporal calculations (such as being unable to correctly identify which date range "last week" corresponds to). TReMu replaces natural language reasoning with Python code, utilizing `dateutil.relativedelta` and customized temporal functions to achieve precise calculation, acting as a "symbolized upgrade" of CoT.

### vs Code-based Temporal QA (Li et al., 2023)
Li et al. also use code execution for temporal QA, but it is limited to short texts and does not involve memory mechanisms or multi-session scenarios, nor does it support function calling. TReMu's function calling capability (e.g., `weekRange`) extends the types of temporal reasoning that can be handled.

## Insights & Connections
- The idea of time-aware memorization can be generalized to any Agent system requiring long-term memory, not limited to dialogue scenarios.
- The three-step reasoning paradigm of "LLM generates code → symbolic execution → LLM interprets results" can be transferred to other structured reasoning tasks such as numerical and logical reasoning.
- The "extraction → linking → QA generation → quality control" pipeline in benchmark construction provides a reference for building high-quality evaluation datasets at low cost.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of time-aware memory and neuro-symbolic reasoning is novel; the design of distinguishing event occurrence time from mention time is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 3 LLMs × 6 methods, comprehensive ablations, including case studies and execution failure rate analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear problem definition, direct comparison in Tables 3 & 4 demonstrating memory differences, and overall well-structured.
- Value: ⭐⭐⭐⭐ — Temporal reasoning in multi-session dialogues is a practical demand; both the framework and benchmark make solid contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](membench_towards_more_comprehensive_evaluation_on_the_memory_of_llm-based_agents.md)
- [\[ACL 2025\] Temporal Reasoning for Timeline Summarisation in Social Media](temporal_reasoning_for_timeline_summarisation_in_social_media.md)
- [\[ACL 2025\] SynapticRAG: Enhancing Temporal Memory Retrieval in Large Language Models through Synaptic Mechanisms](synapticrag_enhancing_temporal_memory_retrieval_in_large_language_models_through.md)
- [\[ACL 2025\] Controlling Politeness in Multi-Turn Dialogues Through Pre-Phrase Augmentation](controlling_politeness_in_multi-turn_dialogues_through_pre-phrase_augmentation.md)
- [\[ACL 2025\] Disentangling Memory and Reasoning Ability in Large Language Models](disentangle_memory_reasoning.md)

</div>

<!-- RELATED:END -->
