---
title: >-
  [Paper Note] Reasoning Gets Harder for LLMs Inside A Dialogue
description: >-
  [ACL 2026][Dialogue Systems][Task-Oriented Dialogue] This paper introduces Boulder, a dynamic benchmark demonstrating that while LLMs perform well on isolated reasoning problems…
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "Task-Oriented Dialogue"
  - "Dynamic Benchmark"
  - "Embedded Reasoning"
  - "Multi-turn Interaction"
  - "Tool Calling"
date: 2026-05-08
content_hash: 93eb0da19d71a43f
---

# Reasoning Gets Harder for LLMs Inside A Dialogue

**Conference**: ACL 2026  
**arXiv**: [2603.20133](https://arxiv.org/abs/2603.20133)  
**Code**: https://github.com/ivankartac/boulder  
**Area**: Dialogue Systems / LLM Evaluation / Reasoning Ability  
**Keywords**: Task-Oriented Dialogue, Dynamic Benchmark, Embedded Reasoning, Multi-turn Interaction, Tool Calling  

## TL;DR
This paper introduces Boulder, a dynamic benchmark demonstrating that while LLMs perform well on isolated reasoning problems, their performance significantly degrades when the same problems are embedded within task-oriented dialogues. This degradation is primarily attributed to multi-turn context, dialogue role constraints, and the overhead of tool calling.

## Background & Motivation
**Background**: LLM reasoning capabilities are typically evaluated through math, code, spatial, or temporal reasoning benchmarks. These benchmarks often design problems as isolated tasks with clean inputs, clear objectives, and easily verifiable answer formats.

**Limitations of Prior Work**: Real-world usage scenarios are not always isolated problems. In systems like travel assistants, booking agents, or hotel recommendation engines, the model must simultaneously track dialogue history, adhere to an assistant persona, process tool outputs, and generate natural language responses while performing implicit reasoning. Traditional benchmarks may overestimate model stability in authentic interactive scenarios.

**Key Challenge**: There is a conflict between isolated reasoning (emphasizing "correct answers") and task-oriented dialogue (emphasizing "natural responses in complex interactions"). When integrated, models must allocate attention among roles, formats, tools, context, and reasoning, potentially weakening previously strong reasoning capabilities through the dialogue framework.

**Goal**: To construct a controlled benchmark where the same problem instance appears in both isolated prompts and task-oriented dialogue prompts. This allows for measuring the performance loss caused by dialogue embedding and identifying the sources of this loss through ablation analysis.

**Key Insight**: The authors select travel-related task-oriented dialogues because they naturally involve arithmetic, temporal, spatial, commonsense, and structured data reasoning. Furthermore, these can be dynamically generated using databases and templates to reduce the risk of training data contamination.

**Core Idea**: Instead of merely asking "can an LLM reason?", evaluate whether it "can still reason while acting as a dialogue assistant."

## Method
The core contribution is not a new model but the Boulder evaluation framework and a series of rigorous controlled experiments revealing the impact of dialogue framing on reasoning. Each sample in Boulder shares the same underlying problem instance but differs in presentation: the isolated setting provides problems and JSON data directly, while the dialogue setting disperses the same information across multi-turn history, user requests, tool calls, and tool results.

### Overall Architecture
Boulder comprises eight travel tasks covering four domains: trains, hotels, restaurants, and attractions. Each task can automatically generate new instances with verifiable ground truth. In experiments, 100 instances are generated per task, totaling 800 test cases.

The evaluation process follows three steps: first, generating Baseline, Dialogue, and Dialogue-concise inputs for the same problem; second, obtaining responses from eight open-source or closed-source LLMs under greedy decoding; and finally, using a specialized LLM parser to extract answers from natural language outputs, with manual verification and prediction-powered inference to correct parser noise.

### Key Designs
1.  **Dual-form Benchmark for Identical Instances**:
    - **Function**: Isolates the impact of "problem difficulty" from "dialogue presentation."
    - **Mechanism**: Each sample uses the same underlying database, target answer, and synonymous user query. The Baseline displays the problem and data directly, while the Dialogue presents data as part of tool results and historical conversation.
    - **Design Motivation**: If different settings used different problems, it would be impossible to determine if performance drops stemmed from reasoning difficulty or dialogue format. The dual-form design ensures a cleaner comparison.

2.  **Dynamic Generation and Automatically Verifiable Answers**:
    - **Function**: Reduces data contamination risks and supports large-scale evaluation.
    - **Mechanism**: Travel tasks are generated based on the MultiWOZ database and custom templates; some data underwent synthetic expansion for diversity. Target answers (amounts, times, distances, Boolean relations, or path sequences) are calculated automatically.
    - **Design Motivation**: LLMs likely encounter static benchmarks during training. Dynamic generation allows researchers to resample instances while maintaining verifiability.

3.  **Ablation of Dialogue Factors**:
    - **Function**: Deconstructs the sources of performance degradation.
    - **Mechanism**: Variants such as reduced domains, without tools, single-turn dialogue, multi-turn baseline, baseline with dialogue role, and dialogue with reasoning instruction were designed to incrementally remove or add domain complexity, tool schemas, multi-turn history, and persona settings.
    - **Design Motivation**: Observing "Baseline vs. Dialogue" alone does not explain the *why*; ablation clarifies the specific contributions of multi-turn interaction, tool overhead, and role bias.

### Loss & Training
This is an evaluation-focused paper; no new models were trained. All models used a unified prompt and greedy decoding. Open-weight models were run via Ollama or OpenRouter, while closed-source models were accessed via the OpenRouter API. Aggregate metrics map accuracy, precision, and normalized MAE to a $[0, 1]$ interval for cross-task averaging.

## Key Experimental Results

### Main Results
Boulder's tasks cover various reasoning types and do not force a fixed output format, making them closer to real dialogue systems.

| Task | Domain | Reasoning Type | Extracted Value | Metric |
|------|------|----------|--------|------|
| Train ticket price | trains | Arithmetic + Commonsense | Amount | Accuracy |
| Hotel booking price | hotels | Arithmetic + Room constraints | Amount | Accuracy |
| Train departure time | trains | Temporal order | HH:MM | Accuracy |
| Train departure frequency | trains | Temporal frequency | Minutes | MAE |
| Restaurant opening hours | restaurants | Temporal intervals | Restaurant list | Precision |
| Distance between venues | hotels/restaurants | Spatial distance | Meters | MAE |
| Directional relations | attractions/restaurants | Directional relationship | yes/no/unknown | Accuracy |
| Shortest walking path | attractions/hotels | Path optimization | Attraction sequence | Accuracy |

The primary conclusion is that most models perform highly in the Baseline but drop significantly in Dialogue. Dialogue-concise is usually only slightly lower than Dialogue, suggesting the issue is not just short-answer constraints but the dialogue framework itself altering model behavior.

| Setting | Input Form | Overall Trend | Key Explanation |
|------|----------|--------------------|----------|
| Baseline | Isolated question + JSON data | Most models score $\approx 0.87-0.97$; Gemini 2.5 Flash $\approx 0.70$ | Clear tasks allow explicit reasoning |
| Dialogue | Multi-turn TOD history + tool schema/results | All models score significantly lower; smaller models drop more | Reasoning is distracted by role, history, and tool overhead |
| Dialogue-concise | Dialogue + max 2 sentence response | Usually slightly lower than Dialogue | Explicit length constraints are not the sole cause |

### Ablation Study
Parser reliability is foundational for this benchmark. The authors manually inspected 5,760 parsing results, reporting 95%-99% accuracy with an inter-annotator agreement of $\kappa=0.94$.

| Parser Dimension | Accuracy | Parser Dimension | Accuracy |
|-------------|--------|-------------|--------|
| Ticket price | 96.39% | Qwen3 4B Output | 96.52% |
| Booking price | 98.75% | Mistral Small 24B Output | 98.47% |
| Departure time | 98.61% | Qwen3 30B Output | 98.05% |
| Departure frequency | 97.50% | Command A 111B Output | 98.61% |
| Opening hours | 94.86% | Qwen3 235B Output | 95.83% |
| Distance | 96.11% | DeepSeek V3.2 Output | 95.97% |
| Directional relation | 96.54% | Gemini 2.5 Flash Output | 93.47% |
| Shortest path | 98.06% | Claude 4.5 Sonnet Output | 95.97% |

Dialogue factor ablation further indicates that performance degradation is not caused by a single factor.

| Ablation Setting | Comparison Target | Observation | Explanation |
|----------|----------|----------|------|
| Dialogue with reduced domains | Dialogue | Inconsistent impact across models | Reducing domain count does not consistently solve the issue |
| Dialogue without tools | Dialogue | Significant improvement for most, yet still below Baseline | Tool schemas and history increase cognitive load |
| Single-turn dialogue | Dialogue | Performance increases | Multi-turn history is a major source of degradation |
| Multi-turn baseline | Baseline | Performance drops with multi-turn history | Multi-turn interactions hurt reasoning even without TOD instructions |
| Baseline with dialogue role | Baseline | Most models drop | Assistant roles induce shorter, more conversational responses |
| Dialogue with reasoning instruction | Dialogue | Improvement in some models, but gap remains large | Simply prompting for more reasoning is insufficient to restore capability |

### Key Findings
- High scores on traditional isolated benchmarks do not represent reasoning reliability in real interactions.
- Multi-turn interaction is the most significant negative factor, leading models to answer prematurely, rationalize post-hoc, or confuse rules within history.
- The TOD assistant role biases models toward short, polite answers or requests for clarification rather than complete computations.
- Tool schemas and tool call history interfere with reasoning, likely because the model must juggle generating natural language and understanding tool structures.
- Adding a reasoning instruction has limited effect, indicating this is not a problem that simple prompting can fully fix.

## Highlights & Insights
- The strongest aspect of the paper is the **controlled comparison**. Presenting the same instance in two forms provides a solid causal explanation for the "dialogue difficulty."
- Boulder is a **dynamic benchmark**, which is crucial for LLM evaluation post-2026. As static public problems are increasingly contaminated, dynamic generation extends benchmark lifespan.
- The authors do not simply attribute errors to a "lack of reasoning ability" but point out that the **dialogue environment changes behavioral patterns**. This is valuable for building agents and dialogue systems.
- The evaluation design (LLM parser + manual correction + PPI) is meticulous, avoiding total reliance on subjective LLM-as-a-judge metrics.

## Limitations & Future Work
- Tasks only cover four travel-related domains; while including arithmetic, spatial, and temporal reasoning, they may not represent complex scenarios like medical, legal, or enterprise workflows.
- Experiments focus on zero-shot single-model TOD. Real-world systems with RAG, planners, or multi-module pipelines might exhibit different degradation levels.
- The authors did not systematically test whether few-shot, fine-tuning, or specialized dialogue-reasoning training could mitigate the gap.
- Many results are visualized as curves; while the text provides trends, exact numerical replication depends on the released code and output files.
- Future work could extend Boulder to cross-lingual dialogues, real user preferences, multi-agent toolchains, and longer history contexts.

## Related Work & Insights
- **vs TimeBench / TRAM**: These benchmarks focus on temporal reasoning but use isolated or multiple-choice formats. Boulder emphasizes reasoning embedded in NLG and dialogue history.
- **vs CoQA / TimeDial**: While these have dialogue formats, they typically evaluate reading comprehension or selection. Boulder's answers derive from task databases and executable logic, aligning closer to task-oriented assistants.
- **vs Multi-turn Instruction Following**: Traditional multi-turn evaluations check instruction persistence or preference consistency; this work observes that multi-turn contexts directly impair core arithmetic, temporal, and spatial reasoning.
- **Insight**: When evaluating agents, core tasks should be tested within authentic interactive frameworks rather than just testing sub-task capabilities in isolation.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ The "dual-form controlled comparison" design addresses a major blind spot in current LLM evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Model and task coverage is quite complete, though numerical results rely heavily on plots.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear structure with insightful error analysis; reporting more raw data in tables would improve utility.
- **Value**: ⭐⭐⭐⭐⭐ Highly relevant for dialogue systems, tool-calling agents, and real-world reasoning assessment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems](stride-ed_a_strategy-grounded_stepwise_reasoning_framework_for_empathetic_dialog.md)
- [\[ACL 2026\] CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment](codial_interpretable_task-oriented_dialogue_systems_through_dialogue_flow_alignm.md)
- [\[ACL 2026\] Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky](disambiguation-centric_finetuning_makes_enterprise_tool-calling_llms_more_realis.md)
- [\[ACL 2026\] Preference Learning Unlocks LLMs' Psycho-Counseling Skills](preference_learning_unlocks_llms_psycho-counseling_skills.md)
- [\[ACL 2026\] APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI](apex-mem_agentic_semi-structured_memory_with_temporal_reasoning_for_long-term_co.md)

</div>

<!-- RELATED:END -->
