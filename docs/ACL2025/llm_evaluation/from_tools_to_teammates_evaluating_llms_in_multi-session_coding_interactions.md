---
title: >-
  [Paper Note] From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions
description: >-
  [ACL 2025][LLM Evaluation][multi-session dialogue] Proposes MemoryCode, a synthetic multi-session dataset to evaluate the ability of LLMs to track and execute coding instructions over long-term interactions, finding that even GPT-4o's accuracy drops by 67% when provided with the full dialogue history, revealing fundamental limitations of current LLMs in prospective memory and information integration.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "multi-session dialogue"
  - "long-term memory"
  - "coding instructions"
  - "prospective memory"
date: 2026-05-08
content_hash: 9b06ee5df72de34a
---

# From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions

**Conference**: ACL 2025  
**arXiv**: [2502.13791](https://arxiv.org/abs/2502.13791)  
**Code**: [https://github.com/Cohere-Labs-Community/MemoryCode](https://github.com/Cohere-Labs-Community/MemoryCode)  
**Area**: LLM Evaluation  
**Keywords**: multi-session dialogue, long-term memory, coding instructions, prospective memory, LLM evaluation

## TL;DR
Proposes MemoryCode, a synthetic multi-session dataset to evaluate the ability of LLMs to track and execute coding instructions over long-term interactions, finding that even GPT-4o's accuracy drops by 67% when provided with the full dialogue history, revealing fundamental limitations of current LLMs in prospective memory and information integration.

## Background & Motivation

**Background**: LLMs perform excellently in single-task solving and are widely used in work environments. However, transitioning from "tools" to "teammates" requires the ability to retain and utilize information across multiple interactions.

**Limitations of Prior Work**: Existing multi-turn/multi-session evaluations either focus on simple information retrieval (Needle-in-a-Haystack) or lack practical task requirements. Most works only test memory when prompted to retrieve, failing to test prospective memory (spontaneous retrieval).

**Key Challenge**: Can LLMs, like a new colleague, utilize information accumulated from daily work interactions for future tasks without being explicitly prompted to recall?

**Goal**: Evaluate the ability of LLMs to spontaneously retrieve and execute previously received coding guidelines during long-term, cross-session interactions.

**Key Insight**: Designing a mentor-newbie onboarding scenario, where instructions are simple (e.g., "function names must start with g_") but scattered across a large amount of irrelevant information and can be updated over time.

**Core Idea**: LLMs perform perfectly in isolated instruction scenarios, but their accuracy plunges in realistic multi-session dialogue histories—indicating that the issue lies in retrieval and integration rather than execution capability.

## Method

### Overall Architecture
Manually design seeds (instructions, distractors, personas, company names) -> Templatized recombination -> LLM-generated multi-session dialogue histories -> Evaluation at different levels: Instruction Only / Single Session / Full History.

### Key Designs

1. **Four Types of Seeds**

    - **Instructions (51 rules)**: Simple coding conventions (e.g., "function names must start with g_"), with 16 rules that can be updated up to 8 times.
    - **Distractors (80 items)**: Company policies, non-code related instructions.
    - **Personas**: 6 types of mentors x 5 types of newbies.
    - **Names**: Randomly fictionalized company and person names.
    - **Design Motivation**: The combination of seeds ensures data diversity.

2. **Controllable Dialogue History Generation**

    - Parametric templates: number of sessions (1-100), ratio of sessions containing instructions (50-70%), number of instructions per session (1-3), update ratio, etc.
    - Command R+ to generate natural dialogues.
    - **Design Motivation**: Precisely control complexity dimensions.

3. **Three-Level Evaluation**

    - **Level 1 — Instruction Only**: Direct presentation of instructions + task to evaluate execution capability.
    - **Level 2 — Single Session**: Presentation of a complete session containing instructions and distractors to evaluate intra-session retrieval.
    - **Level 3 — Full History**: Presentation of concatenated sessions to evaluate cross-session retrieval and integration.
    - **Design Motivation**: Progressively increase difficulty to dissociate "inability to execute" from "failure to retrieve".

4. **Test Function Design**

    - Use regular expressions to detect instruction adherence (e.g., whether function names start with g_).
    - Do not evaluate code functional correctness, only adherence to code specifications.
    - **Design Motivation**: Decouple retrieval capability from programming capability.

### Dataset Scale
- 360 dialogue histories, with 30 instances for each session count.
- "Short" histories (< 15 sessions, 54%), "long" histories (> 15 sessions, 46%).
- Maximum history length is 63K tokens.

## Key Experimental Results

### Main Results — Accuracy at Different Levels

| Model | Level 1 (Instruction Only) | Level 2 (Single Session) | Level 3 (Full History) | Drop |
|------|-----------------|-----------------|-------------------|------|
| GPT-4o | ~95% | ~85% | **~28%** | **-67%** |
| DeepSeek-R1 | ~93% | ~80% | ~35% | -58% |
| Claude-3.5 | ~92% | ~82% | ~30% | -62% |
| Command R+ | ~88% | ~75% | ~25% | -63% |
| Llama-3.1-70B | ~85% | ~65% | ~20% | -65% |

### Impact of Session Count on Performance

| Session Count | GPT-4o Accuracy | Small Model Accuracy |
|--------|-------------|------------|
| 1-5 | ~70% | ~50% |
| 10-20 | ~35% | ~20% |
| 50-100 | **~15%** | **~5%** |

### Ablation Study — Impact of Instruction Updates

| Configuration | Accuracy | Description |
|------|--------|------|
| No updates | Higher | Only requires retrieval |
| With updates (keep latest) | **Significant drop** | Requires integration and overriding old information |

### Key Findings
- **All models perform perfectly at Level 1**: The instructions themselves are simple, indicating it is not an execution capability issue.
- **Level 3 accuracy plunges**: GPT-4o drops from 95% to 28%, proving the bottleneck is retrieval, not execution.
- **Performance degrades with more sessions**: At 100 sessions, almost all models fail.
- **Instruction updates present an additional challenge**: Models must not only find the instruction but also identify the latest version.
- **DeepSeek-R1 (reasoning model) performs slightly better but remains poor**: Reasoning capabilities cannot compensate for retrieval deficiencies.
- **Distracting information (especially instruction-like distractors) severely impacts performance**.

## Highlights & Insights
- **Three-level evaluation design** precisely decouples "execution capability" from "retrieval capability"—the cliff-like performance drop from Level 1 to 3 clearly proves the bottleneck is retrieval.
- **Onboarding mentor scenario** closely mirrors real-world work environments—which is the typical usecase for LLMs as coding assistants; the identified limitations directly affect product design.
- **Prospective memory** (spontaneously recalling and applying past instructions without a prompt) is a fundamental human cooperative capability, yet LLMs almost entirely lack it.

## Limitations & Future Work
- Synthetic data may not fully reflect the complexity of real-world work scenarios.
- Only evaluates coding conventions, failing to test more complex knowledge (such as architectural decisions).
- Does not explore the mitigation effects of external memory mechanisms (e.g., RAG, memory modules).
- Directions for improvement: evaluating LLMs with memory modules, adaptive information compression, and progressive instruction learning.

## Related Work & Insights
- **vs Needle-in-a-Haystack**: NIAH explicitly prompts retrieval, whereas MemoryCode requires implicit/prospective memory.
- **vs MMMT-IF**: MMMT-IF tests instruction following but does not involve multi-session updates.
- **vs LoCoMo**: LoCoMo tests factual retrieval, while MemoryCode tests the application of instructions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First benchmark to test multi-session prospective memory and instruction integration.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models × multiple session lengths × three-level evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Scenario design is close to reality, with clear experimental logic.
- Value: ⭐⭐⭐⭐⭐ Provides direct insights for product design of LLMs as collaborative tools.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework](educationq_evaluating_llms_teaching_capabilities_through_multi-agent_dialogue_fr.md)
- [\[ICML 2025\] MultiCogEval: Evaluating LLMs Across Multi-Cognitive Levels](../../ICML2025/llm_evaluation/evaluating_llms_across_multi-cognitive_levels_from_medical_knowledge_mastery_to_.md)
- [\[ACL 2025\] Can External Validation Tools Improve Annotation Quality for LLM-as-a-Judge?](can_external_validation_tools_improve_annotation_quality_for_llm-as-a-judge.md)
- [\[ACL 2025\] EvoWiki: Evaluating LLMs on Evolving Knowledge](evowiki_evaluating_llms_on_evolving_knowledge.md)
- [\[ACL 2026\] SessionIntentBench: A Multi-Task Inter-Session Intention-Shift Modeling Benchmark](../../ACL2026/llm_evaluation/sessionintentbench_a_multi-task_inter-session_intention-shift_modeling_benchmark.md)

</div>

<!-- RELATED:END -->
