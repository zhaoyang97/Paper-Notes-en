---
title: >-
  [Paper Note] Reasoning Gets Harder for LLMs Inside A Dialogue
description: >-
  [ACL 2026][Dialogue Systems][Paper Note] This paper introduces the Boulder dynamic benchmark, demonstrating that while LLMs perform well on isolated reasoning problems, their performance significantly degrades when the same problems are embedded in task-oriented dialogues. This is primarily attributed to multi-turn context, dialogue role constraints, and the
tags:
  - ACL 2026
  - Dialogue Systems
date: 2026-05-08
content_hash: 8437b4777c533987
---
# Reasoning Gets Harder for LLMs Inside A Dialogue

**Conference**: ACL 2026  
**arXiv**: [2603.20133](https://arxiv.org/abs/2603.20133)  
**Code**: https://github.com/ivankartac/boulder  
**Area**: Dialogue Systems / LLM Evaluation / Reasoning Capability  
**Keywords**: Task-oriented dialogue, dynamic benchmark, embedded reasoning, multi-turn interaction, tool use

## TL;DR
This paper introduces the Boulder dynamic benchmark, demonstrating that while LLMs perform well on isolated reasoning problems, their performance significantly degrades when the same problems are embedded in task-oriented dialogues. This is primarily attributed to multi-turn context, dialogue role constraints, and the burden of tool calling.

## Background & Motivation
**Background**: LLM reasoning capabilities are typically evaluated through math, code, spatial, or temporal reasoning benchmarks. These benchmarks often design problems as isolated tasks with clean inputs, clear goals, and easily verifiable answer formats.

**Limitations of Prior Work**: Real-world usage scenarios are not always isolated problems. In systems such as travel assistants, booking agents, or hotel recommenders, models must simultaneously track dialogue history, adhere to assistant roles, read tool outputs, and generate natural language responses while performing implicit reasoning. Traditional benchmarks may overestimate model stability in real interaction scenarios.

**Key Challenge**: Isolated reasoning emphasizes "calculating the correct answer," whereas task-oriented dialogue (TOD) emphasizes "responding naturally within complex interactions." When these two overlap, models allocate attention among roles, formats, tools, context, and reasoning, potentially weakening the strong reasoning abilities observed in isolated settings.

**Goal**: Construct a controllable benchmark where the same problem instance appears in both isolated prompts and task-oriented dialogue prompts to measure the performance loss caused by dialogue embedding and identify sources of loss through ablation analysis.

**Key Insight**: The authors chose travel-related task-oriented dialogues because they naturally involve arithmetic, temporal, spatial, commonsense, and structured data reasoning. Furthermore, they can be dynamically generated using databases and templates to reduce the risk of training data contamination.

**Core Idea**: Instead of just asking if an LLM "can reason," evaluate whether it "can still reason while acting as a dialogue assistant."

## Method
The core of the paper is not a new model but the Boulder evaluation framework and a series of rigorous controlled experiments revealing the impact of dialogue framing on reasoning. Each Boulder sample uses the same underlying problem instance presented in different ways: the isolated setting provides the question and JSON data directly, while the dialogue setting scatters the same information across multi-turn history, user requests, tool calls, and tool results.

### Overall Architecture
Boulder includes eight types of travel tasks covering four domains: trains, hotels, restaurants, and attractions. Each task can automatically generate new samples with verifiable ground truths. In the experiments, 100 samples were generated per task, totaling 800 test samples.

The evaluation process follows three steps: first, generate Baseline, Dialogue, and Dialogue-concise versions of the same problem; second, evaluate eight open-source or closed-source LLMs under greedy decoding; finally, use a specialized LLM parser to extract answers from natural language output, corrected for noise using manual verification and prediction-powered inference.

### Key Designs

**1. Dual-form benchmark for the same instance: Decoupling problem difficulty from dialogue presentation.**

If isolated and dialogue settings used different problems, it would be unclear whether performance drops resulted from reasoning difficulty or dialogue format. Boulder shares the same underlying database, target answer, and synonymous user questions between settings, only varying "how information is packaged." Thus, the performance gap between Baseline and Dialogue is cleanly attributed to dialogue framing without confounding variables related to problem difficulty.

**2. Dynamic generation and automatically verifiable answers: Preventing data contamination while supporting large-scale evaluation.**

LLMs might have been exposed to static public benchmarks during training. Boulder dynamically generates travel tasks based on the MultiWOZ database and custom templates, using synthetic expansion for diversity. Target answers (amounts, times, distances, Booleans, or sequences) are calculated automatically, allowing for fresh sampling while maintaining verifiability. This makes the benchmark more robust against contamination and supports large-scale evaluation across eight tasks.

**3. Ablation of dialogue factors: Decomposing performance degradation.**

To identify which factor hinders performance, the authors designed variants—reduced domains, without tools, single-turn dialogue, multi-turn baseline, baseline with dialogue role, and dialogue with reasoning instruction. By adding or removing domain complexity, tool schemas, multi-turn history, and assistant roles, the degradation can be decomposed into specific contributions. Results indicate multi-turn history and tool burdens are primary sources of loss.

### Loss & Training
This is an evaluation-focused paper and does not involve training new models. All models used a unified prompt and greedy decoding. Open-weight models were executed via Ollama or OpenRouter; closed-source models were accessed via the OpenRouter API. Metrics (accuracy, precision, and normalized MAE) were mapped to a $[0, 1]$ interval to allow for cross-task averaging.

## Key Experimental Results

### Main Results
Boulder tasks cover different reasoning types and do not force fixed output formats, making them more representative of real dialogue systems.

| Task | Domain | Reasoning Type | Extracted Value | Metric |
|------|------|----------|--------|------|
| Train ticket price | trains | Arithmetic + Commonsense | Amount | Accuracy |
| Hotel booking price | hotels | Arithmetic + Constraints | Amount | Accuracy |
| Train departure time | trains | Temporal ordering | HH:MM | Accuracy |
| Train departure frequency | trains | Temporal frequency | Minutes | MAE |
| Restaurant opening hours | restaurants | Temporal intervals | List | Precision |
| Distance between venues | hotels/restaurants | Spatial distance | Meters | MAE |
| Directional relations | attractions/restaurants | Directional relations | yes/no/unknown | Accuracy |
| Shortest walking path | attractions/hotels | Path optimization | Sequence | Accuracy |

The primary conclusion is that most models perform well in the Baseline setting but drop significantly in Dialogue. The Dialogue-concise setting is usually slightly lower than Dialogue, suggesting the issue is the dialogue framework itself rather than just short-answer constraints.

| Setting | Input Form | Reported General Trend | Key Explanation |
|------|----------|--------------------|----------|
| Baseline | Isolated question + JSON data | Average scores $\approx 0.87-0.97$; Gemini 2.5 Flash slightly $> 0.70$ | Clear task, models can fully expand reasoning |
| Dialogue | Multi-turn TOD history + tool schema/results | All models significantly lower than Baseline; larger drop for smaller models | Reasoning disrupted by role, history, and tool burden |
| Dialogue-concise | Dialogue + max 2-sentence response | Usually slightly lower than Dialogue | Explicit length limit is not the only cause |

### Ablation Study
Parser reliability is fundamental to this benchmark. The authors manually checked 5,760 parse results, reporting 95%-99% accuracy with a Cohen's $\kappa=0.94$.

| Parser Dimension | Accuracy | Model Output (Parser) | Accuracy |
|-------------|--------|-------------|--------|
| Ticket price | 96.39% | Qwen3 4B | 96.52% |
| Booking price | 98.75% | Mistral Small 24B | 98.47% |
| Departure time | 98.61% | Qwen3 30B | 98.05% |
| Departure frequency | 97.50% | Command A 111B | 98.61% |
| Opening hours | 94.86% | Qwen3 235B | 95.83% |
| Distance | 96.11% | DeepSeek V3.2 | 95.97% |
| Directional relation | 96.54% | Gemini 2.5 Flash | 93.47% |
| Shortest path | 98.06% | Claude 4.5 Sonnet | 95.97% |

Ablation of dialogue factors further explains that performance degradation is not caused by a single factor.

| Ablation Setting | Comparison | Observation | Explanation |
|----------|----------|----------|------|
| Dialogue (reduced domains) | Dialogue | Inconsistent impact across models | Simplifying domains doesn't resolve the core issue |
| Dialogue (without tools) | Dialogue | Significant improvement but still below Baseline | Tool schemas and history increase cognitive load |
| Single-turn dialogue | Dialogue | Performance increases | Multi-turn history is a major source of degradation |
| Multi-turn baseline | Baseline | Performance decreases | Multi-turn itself hurts reasoning even without TOD instructions |
| Baseline with dialogue role | Baseline | Most models decrease | Assistant roles induce shorter, more conversational answers |
| Dialogue with reasoning instruction | Dialogue | Slight improvement for some; gap remains large | "Think step-by-step" prompts are insufficient |

### Key Findings
- High scores on traditional isolated benchmarks do not translate to reasoning reliability in real interactions.
- Multi-turn interaction is the most significant negative factor; it leads to premature answers, post-hoc rationalization, or confusion regarding rules in the history.
- The TOD assistant role makes models prone to short, polite, or clarifying responses instead of fully expanded calculations.
- Tool schemas and tool-call history interfere with reasoning, likely because the model must simultaneously perform NLU, NLG, and structured data processing.
- Simple reasoning instructions provide limited relief, indicating this behavior is not easily fixed by basic prompting.

## Highlights & Insights
- The strongest aspect of the paper is the controlled comparison. Presenting the same instance in two forms provides a solid causal explanation for the "dialogue difficulty" conclusion.
- Boulder is a dynamic benchmark, which is critical for the post-2026 LLM landscape where static tasks are easily contaminated.
- Instead of simply concluding "models cannot reason," the authors highlight how the dialogue environment alters model behavior patterns, which is highly valuable for building agents.
- The evaluation pipeline—LLM parser with manual correction and PPI—is meticulously designed to avoid the subjectivity of pure LLM-as-a-judge approaches.

## Limitations & Future Work
- Tasks only cover four travel-related domains; they may not represent more complex scenarios like medical, legal, or enterprise workflows.
- The experiments focus on zero-shot single-model TOD. Real-world systems may use retrievers, planners, or multi-module pipelines, where degradation might differ.
- The authors did not systematically test few-shot prompting, fine-tuning, or specialized dialogue-reasoning training to mitigate the gap.
- Many primary results are presented as plots; while trends are clear, reproduction relies on public code and output files for precise numerical values.
- Future work could extend Boulder to cross-lingual dialogues, real user preferences, multi-agent toolchains, and longer histories.

## Related Work & Insights
- **vs TimeBench / TRAM**: These benchmarks focus on temporal reasoning but use isolated or multiple-choice formats. Boulder emphasizes reasoning embedded in NLG and dialogue history.
- **vs CoQA / TimeDial**: While these involve dialogue, they usually evaluate via reading comprehension or selection. Boulder's answers are derived from task databases and executable logic, closer to a real task-oriented assistant.
- **vs Multi-turn Instruction Following**: Existing evaluations often look at instruction adherence or preference consistency; this paper observes how multi-turn context directly damages arithmetic, temporal, and spatial reasoning.
- **Insight**: When evaluating agents, core tasks should be tested within real interaction frameworks rather than as isolated sub-tasks.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The "same instance across isolated and dialogue" controlled design addresses a major blind spot in current LLM evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes diverse models, tasks, and detailed ablations, though numerical results rely partly on figures and public logs.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure with insightful error analysis.
- Value: ⭐⭐⭐⭐⭐ Highly relevant for the development and evaluation of dialogue systems, tool-calling agents, and real-world reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems](stride-ed_a_strategy-grounded_stepwise_reasoning_framework_for_empathetic_dialog.md)
- [\[ACL 2026\] Preference Learning Unlocks LLMs' Psycho-Counseling Skills](preference_learning_unlocks_llms_psycho-counseling_skills.md)
- [\[ACL 2026\] APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI](apex-mem_agentic_semi-structured_memory_with_temporal_reasoning_for_long-term_co.md)
- [\[ICLR 2026\] ReIn: Conversational Error Recovery with Reasoning Inception](../../ICLR2026/dialogue/rein_conversational_error_recovery_with_reasoning_inception.md)
- [\[ACL 2026\] CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment](codial_interpretable_task-oriented_dialogue_systems_through_dialogue_flow_alignm.md)

</div>

<!-- RELATED:END -->
