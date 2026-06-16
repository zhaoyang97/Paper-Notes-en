---
title: >-
  [Paper Note] StratMem-Bench: Evaluating Strategic Memory Use in Virtual Character Conversation Beyond Factual Recall
description: >-
  [ACL 2026][LLM Evaluation][strategic memory use] StratMem-Bench categorizes memories in virtual character conversations into three types: must, nice, and irr. It evaluates whether a model can actively incorporate beneficial memories and suppress irrelevant ones while ensuring factual requirements are met. The results reveal that current strong LLMs remain significant
tags:
  - ACL 2026
  - LLM Evaluation
  - strategic memory use
date: 2026-05-08
content_hash: dca60f26bb593898
---
# StratMem-Bench: Evaluating Strategic Memory Use in Virtual Character Conversation Beyond Factual Recall

**Conference**: ACL2026  
**arXiv**: [2604.26243](https://arxiv.org/abs/2604.26243)  
**Code**: https://github.com/seucoin/StratMem-Bench.git  
**Area**: LLM Evaluation / Virtual Character Conversation / Long-term Memory  
**Keywords**: strategic memory use, virtual characters, long-term conversation, memory selection, LLM evaluation

## TL;DR
StratMem-Bench categorizes memories in virtual character conversations into three types: must, nice, and irr. It evaluates whether a model can actively incorporate beneficial memories and suppress irrelevant ones while ensuring factual requirements are met. The results reveal that current strong LLMs remain significantly unstable in "supportive memory selection."

## Background & Motivation
**Background**: Long-term conversation and virtual character systems typically equip models with external memory, allowing characters to remember past experiences, user preferences, and personas. Most existing benchmarks evaluate factual recall—specifically, whether relevant facts are retrieved and reflected in the response.

**Limitations of Prior Work**: Memory usage in human conversation is not "the more, the better." Some memories are essential for answering questions (must), some simply make the response more natural, empathetic, or personalized (nice), and others, despite being in the memory bank, should not be mentioned in the current context (irr). If a benchmark only assesses factual recall, it fails to measure this selective capability.

**Key Challenge**: Virtual characters must be as proactive and thoughtful as real humans without awkwardly forcing irrelevant personal details into the response. Models need to reach a dynamic balance between proactivity and risk aversion.

**Goal**: To construct an evaluation set that explicitly distinguishes between required, supportive, and irrelevant memories, and to design metrics that measure whether a model "uses all that should be used, uses optional ones appropriately, and avoids what should not be used."

**Key Insight**: Drawing inspiration from Gricean Maxims, the paper maps "must" memories to factual correctness, "nice" memories to appropriate information quantity and social coherence, and "irr" memories to relevance violations.

**Core Idea**: Transform memory from a static fact repository into a dynamic resource in conversation, evaluating the model's functional judgment of each memory based on the current query, persona, and history.

## Method

### Overall Architecture

StratMem-Bench models strategic memory usage as a conditional response generation task: for each sample, the model is provided with a conversation history, the current user query, a character persona, and a pool of unlabeled memories. The model sees no labels and must independently determine which memories functionally contribute to the current response to generate a single-turn reply. Data is sourced from LoCoMo’s multi-session virtual character dialogues—the authors extract memory pools and personas from earlier sessions and use subsequent sessions as the current conversation history, generating a new user query. This ensures memories originate from the past while responses occur later, avoiding temporal leakage. The core of the evaluation is measuring the model's ability to "use all required, use supportive ones appropriately, and avoid irrelevant ones."

### Key Designs

**1. Instance-level labeling of three memory roles: Allowing the same memory to change roles based on the query**

Fixing a memory as "relevant" or "irrelevant" ignores the shifting goals of conversation; memory use in real characters is essentially a contextual decision. StratMem-Bench does not label based on keyword overlap but rather on the functional contribution of the memory to the current conversation goal. Each memory is categorized under the current query into three types: *must* (essential for factual correctness), *nice* (not essential for correctness but enhances personalization, empathy, or social coherence), and *irr* (irrelevant to the current goal, potentially causing tangents or abruptness). For example, "moving to a new city" is a *must* when asked for an address, a *nice* when asked how things are going, and an *irr* when asked about music preferences. Labeling was initialized by GPT-5.1 and reviewed by human experts, achieving a Fleiss' kappa of 0.81 before discussion, indicating stable agreement despite the inherent subjectivity.

**2. Strict Memory Compliance (SMC): Compressing strategic memory use into pass/fail**

Traditional average quality scores can dilute serious errors like "omitting essential memories" or "misusing irrelevant ones," obscuring bottlenecks in memory selection. SMC utilizes hard rules to characterize basic constraints: *must-only* samples require all *must* memories to be used and zero *irr* memories; *nice-only* requires at least one *nice* memory and zero *irr*; *must+nice* requires all *must* memories, at least one *nice*, and zero *irr*. By converting strategic choices into binary pass/fail determinations, hard failures in memory selection are exposed rather than masked by linguistic quality scores.

**3. MIQ, PES, and CIR Behavioral Profiling: Separating "selecting the right memory" from "using memory well"**

An aggregate score cannot explain why a model fails—some are conservative with dry responses, others are aggressive, inserting irrelevant memories. This paper uses three metrics for profiling: *MIQ* (1–5 scale) evaluates whether selected memories are integrated naturally, measuring "usage quality"; *PES* measures whether the model actively enriches responses when *nice* memories are available, characterizing proactivity; *CIR* measures the proportion of *irr* memories misused when *nice* memories are present, characterizing risk aversion. Since *PES* and *CIR* often involve a trade-off, they must be viewed together to distinguish between "too cold" and "too talkative" failure modes.

This is a benchmark paper and does not train new models. During evaluation, all models use a unified instruction template. Automatic evaluation is performed by DeepSeek-V3.2. Memory usage detection requires the evaluator to cite specific evidence from the response and utilize majority voting across three samples. This achieved a Cohen's kappa of 0.96 with human experts on 1,130 memory-response pairs, while the MIQ achieved a Cohen's kappa of 0.69 against 300 human-annotated responses.

## Key Experimental Results

### Main Results
The dataset consists of 657 samples, with *must+nice* scenarios being the majority and representing the most difficult setting for realistic character dialogue.

| Scenario | Samples | Avg. Memories | Avg. Words/Mem | Eval Meaning |
|------|--------|--------------|------------------|----------|
| must-only | 50 | 6.24 | 9.53 | Satisfy essential memory & suppress irr |
| nice-only | 132 | 9.12 | 10.09 | No hard factual need, test active enrichment |
| must+nice | 475 | 8.97 | 9.75 | Satisfy facts, enrichment, and suppression |
| Overall | 657 | 8.79 | 9.81 | Full evaluation |

| Model | SMC must-only | SMC nice-only | SMC must+nice | SMC All | MIQ All on pass |
|------|---------------|---------------|---------------|---------|-----------------|
| GPT-5.2 | 88.00 | 57.58 | 41.89 | 48.55 | 4.45 |
| GPT-5-chat | 90.00 | 46.21 | 41.68 | 46.27 | 4.56 |
| Claude Sonnet 4.5 | 90.00 | 53.03 | 46.95 | 51.45 | 4.37 |
| Gemini 3 Pro | 78.00 | 49.24 | 48.21 | 50.68 | 4.21 |
| DeepSeek-reasoner | 76.00 | 48.48 | 39.16 | 43.84 | 4.12 |
| Qwen3-235B | 92.45 | 46.56 | 42.28 | 47.18 | 4.24 |

Strong models generally achieve 76%-92% SMC on *must-only*, but performance drops significantly on *nice-only* and *must+nice*. This suggests that while models can handle "essential facts," they struggle to judge "supportive memories that could enhance the dialogue."

### Ablation Study
While there is no architectural ablation, the paper provides a decomposition of behavioral dimensions, serving as a diagnostic table for the evaluation.

| Model | must-used MIQ | nice-used MIQ | irr-used MIQ | PES All | CIR All |
|------|---------------|---------------|--------------|---------|---------|
| GPT-5.2 | 4.48 | 4.22 | 2.99 | 56.01 | 13.01 |
| GPT-5-chat | 4.55 | 4.38 | 2.81 | 51.91 | 7.91 |
| Claude Sonnet 4.5 | 4.36 | 4.18 | 3.05 | 62.48 | 15.82 |
| Gemini 3 Pro | 3.92 | 3.73 | 2.63 | 73.33 | 31.96 |
| DeepSeek-chat | 4.32 | 4.12 | 2.75 | 56.96 | 15.49 |
| Qwen3-Max | 4.14 | 4.04 | 2.64 | 57.76 | 19.77 |

### Key Findings
- Once a *must* memory is correctly selected, the MIQ is typically high, indicating the bottleneck lies in "which memory to select" rather than linguistic expression.
- *Nice* memories incur an "enrichment tax": nice-used MIQ is usually ~0.2 lower than must-used MIQ, suggesting that additional personalized information is more prone to stiff or forced integration.
- *Irr* memories cause a collapse in quality; irr-used MIQ scores are mostly between 2.6-3.1, indicating that irrelevant memories not only cause tangents but also disrupt the coherence of the character's response.
- A clear trade-off exists between PES and CIR. GPT-5-chat has the lowest CIR (~7.91) but a conservative PES; Gemini 3 Pro has the highest PES (~73.33) but its CIR reaches 31.96, showing that proactivity comes with higher risks of irrelevant memory intrusion.

## Highlights & Insights
- This paper advances long-term memory evaluation from "can it remember" to "should it be said." For real characters and personal assistants, this is closer to actual product risk than simple recall.
- The division into must, nice, and irr is highly practical. It acknowledges that dialogue quality involves not just factual correctness, but also moderate personalization and relevance control.
- SMC serves as a hard metric, MIQ as a quality metric, and PES/CIR as behavioral tendency metrics. Combining the four allows model failures to be categorized into modes like "failure to use essential memory," "lack of proactive enrichment," "misuse of irrelevant memory," or "poor integration quality."
- The finding that strong models maintain high MIQ when passing SMC suggests that improving strategic memory usage may require better memory selection or policies rather than just stronger generators.

## Limitations & Future Work
- The evaluation only covers single-turn response generation, without investigating how memory selection strategies evolve dynamically across multi-turn interactions.
- Currently limited to textual memory, excluding multi-modal character memories such as voice, images, appearance, and location.
- Data is derived from LoCoMo conversions and synthetic pipelines; while human-verified, the privacy, emotional, and social boundaries in real-world long-term user interactions are more complex.
- Dependency on LLM judges remains; while consistency with humans is high, potential biases toward specific styles or model families may exist.

## Related Work & Insights
- **vs LoCoMo / LongMemEval**: These benchmarks primarily evaluate long-range memory retrieval and factual recall; StratMem-Bench further requires models to judge the functional role of a memory in dialogue.
- **vs Personalized RAG**: Personalized RAG focuses on incorporating user preferences to improve response relevance; this work emphasizes the dynamic selection of must/nice/irr during generation and evaluates the risks of over-personalization.
- **vs Character Consistency Evaluation**: Traditional role-playing evaluations look at persona consistency; this work looks at whether characters "know when to mention the past," much like humans do.
- **Insights**: For real-world assistant systems, one could explicitly model must/nice/irr or similar labels in a memory manager and use CIR as a safety and user experience metric.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Shifts from factual recall to strategic memory use with clear problem definitions and realistic metric combinations.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multiple strong models with human consistency validation, though lacks multi-turn and real user scenario testing.
- Writing Quality: ⭐⭐⭐⭐☆ Tasks, metrics, and conclusions are well-organized; although some auto-evaluation details are in the appendix, the main text is coherent.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for virtual characters, personal assistants, and long-term memory RAG, particularly in diagnosing "too cold" vs. "too talkative" failure modes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

</div>

<!-- RELATED:END -->

## Related Papers

- [\[ACL 2026\] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation](comprehensiveness_metrics_for_automatic_evaluation_of_factual_recall_in_text_gen.md)
- [\[ACL 2026\] Evaluating Memory Capability in Continuous Lifelog Scenario](evaluating_memory_capability_in_continuous_lifelog_scenario.md)
- [\[AAAI 2026\] Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents](../../AAAI2026/llm_evaluation/beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c.md)
- [\[ACL 2026\] Stress Testing Factual Consistency Metrics for Long-Document Summarization](stress_testing_factual_consistency_metrics_for_long-document_summarization.md)
- [\[ICLR 2026\] DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science](../../ICLR2026/llm_evaluation/dare-bench_evaluating_modeling_and_instruction_fidelity_of_llms_in_data_science.md)

</div>

<!-- RELATED:END -->
