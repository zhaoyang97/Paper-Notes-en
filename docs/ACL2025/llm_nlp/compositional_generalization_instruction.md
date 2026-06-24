---
title: >-
  [Paper Note] Revisiting Compositional Generalization Capability of Large Language Models Considering Instruction Following Ability
description: >-
  [ACL 2025][LLM (Other)][Compositional Generalization] This paper proposes the Ordered CommonGen benchmark, which requires LLMs to generate sentences containing all given concepts in a specified order, thereby evaluating both compositional generalization and instruction-following capabilities. Evaluating 36 LLMs reveals that even the strongest model achieves an ordered coverage rate of only approximately 75%.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Compositional Generalization"
  - "Instruction Following"
  - "Generative Commonsense Reasoning"
  - "CommonGen"
  - "benchmark"
date: 2026-05-08
content_hash: 38511cc49a4343eb
---

# Revisiting Compositional Generalization Capability of Large Language Models Considering Instruction Following Ability

**Conference**: ACL 2025  
**arXiv**: [2506.15629](https://arxiv.org/abs/2506.15629)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Compositional Generalization, Instruction Following, Generative Commonsense Reasoning, CommonGen, benchmark

## TL;DR

This paper proposes the Ordered CommonGen benchmark, which requires LLMs to generate sentences containing all given concepts in a specified order, thereby evaluating both compositional generalization and instruction-following capabilities. Evaluating 36 LLMs reveals that even the strongest model achieves an ordered coverage rate of only approximately 75%.

## Background & Motivation

1. Generative Commonsense Reasoning (GCR) tasks (e.g., CommonGen) require models to generate natural sentences containing all given concepts, but traditional evaluations do not consider the order in which concepts occur.
2. Although LLMs exhibit certain instruction-following capabilities after instruction tuning, they still perform poorly when **strictly following constraint-based instructions** (such as using concepts in a specific order).
3. In creative text generation scenarios (such as chronological narratives, action planning, and songwriting), changes in concept order significantly affect the meaning and style of the output.
4. According to generative grammar theory, humans can combine concepts in any specified order to generate grammatically correct sentences, an ability that LLMs do not yet fully possess.
5. Existing GCR evaluation focuses only on concept coverage, failing to reflect whether models truly adhere to the order constraints in user instructions.
6. A benchmark that can **simultaneously evaluate both compositional generalization and instruction following** is needed to reveal the true capabilities and limits of LLMs.

## Method

### Overall Architecture

The proposed **Ordered CommonGen** framework introduces concept order constraints to CommonGen, requiring LLMs to generate sentences that not only include all concepts but also order them exactly as specified in the input. By performing full permutation on concept sets and evaluating with multiple templates, the framework systematically examines the compositional generalization and instruction-following capabilities of LLMs.

### Module 1: Dataset Construction

- 192 seed concept sets (each containing 4 concepts) are extracted from CommonGen-lite.
- All $4! = 24$ permutations are generated for each set, yielding a total of $192 \times 24 = 4,608$ concept sets.
- Six instruction templates of CommonGen are selected from FLAN as base templates.
- The phrase "in the specified order" is inserted into the templates to generate Ordered CommonGen templates.
- This yields a final total of $6 \times 4,608 = 27,648$ evaluation instances.

### Module 2: Multi-dimensional Evaluation Metric System

- **Concept Coverage**: Coverage w/o order (ignoring order), Coverage w/ order (considering order), and Ordered Rate (percentage of ordered coverage).
- **Sentence-level Similarity**: Pairwise-BLEU (surface n-gram overlap) and Pairwise-BLEURT (semantic similarity), where lower values indicate better diversity.
- **Corpus-level Diversity**: Distinct-2 (ratio of unique 2-grams) and Diverse Rate (ratio of unique sentences).
- **Perplexity**: Calculated using GPT2-XL to measure the naturalness of generated sentences.

### Module 3: Large-scale Model Evaluation

- **36 instruction-tuned LLMs** are evaluated, covering Llama3, Qwen2/2.5, Gemma2, Phi3, Mistral/Mixtral, OLMo2, Tülu3, GPT-3.5/4o, Gemini, etc.
- Open-source models employ greedy decoding with 4-bit quantization, while close-sourced models are queried with temperature set to 0.
- Comparing two types of templates (with and without the phrase "in the specified order") to verify instruction understanding capabilities.

### Training/Inference Details

This study is purely evaluative and does not involve model training. All evaluations are conducted in a zero-shot setting to highlight the differences in inductive reasoning capabilities. Results across the six templates are averaged to mitigate template selection bias.

## Experiments

### Table 1: Main Evaluation Results of 36 LLMs

| Model | Coverage w/o↑ | Coverage w/↑ | Ordered Rate↑ | Diverse Rate↑ |
|------|-------------|-------------|--------------|--------------|
| Llama3.1-405B | **98.91** | **74.44** (+55.41) | **75.26** (+55.46) | 98.28 |
| Llama3.3-70B | 97.25 | 66.79 (+47.34) | 68.68 (+48.22) | 94.70 |
| GPT-4o | 96.70 | 53.34 (+30.25) | 55.16 (+30.49) | 86.51 |
| Qwen2-0.5B | 53.78 | 30.84 | 57.34 | 96.60 |
| Mixtral-8x7B | 77.36 | 19.67 | 25.43 | **98.82** |

### Table 2: Performance Analysis of Different Part-of-Speech (POS) Patterns

| POS Pattern | Coverage w/o↑ | Ordered Rate↑ | Diverse Rate↑ |
|----------|-------------|--------------|--------------|
| NNNN (Noun only) | **91.13** | 44.88 | 91.97 |
| VVVV (Verb only) | 37.38 | **63.84** | **98.17** |
| VNVN | 84.83 | 54.50 | 89.58 |

### Key Findings

1. **LLMs Understand Instruction Intent**: After adding "in the specified order" to the prompts, the 'w/ order' coverage of most models significantly increases (e.g., Llama3.1-405B gains $+55.41$), demonstrating that models can comprehend and attempt to follow order constraints.
2. **But Precise Adherence Remains Limited**: Even the strongest Llama3.1-405B only achieves an Ordered Rate of approximately 75%, meaning more than 20% of the outputs fail to adhere to the designated order.
3. **Order Preferences Lead to Repetitive Outputs**: Some models (e.g., Gemma2-2B) generate identical sentences when faced with different permutations of the same concept set, reflecting biases towards frequent patterns in their training data.
4. **Verb Combinations are Harder but Demonstrate Better Order Adherence**: The VVVV pattern has a coverage rate of only 37%, but it achieves the highest Ordered Rate and Diverse Rate. This suggests that the models tend to better follow instructions when they succeed in generating sentences.
5. **Syntactic Diversity Improves while Semantic Diversity is Insufficient**: While pBLEU improves significantly, the improvement in pBLEURT is limited, showing that models prioritize achieving syntactic compositionality over semantic compositionality.

## Highlights & Insights

- It ingeniously unifies **instruction-following ability and compositional generalization capability** into a single evaluation framework, presenting a clean design that uncovers deep issues.
- The evaluation design using full permutations and multiple templates is rigorous, avoiding single-template bias, while the 27,648 instances ensure statistical reliability.
- It proposes two new assessment metrics, Ordered Rate and Diverse Rate, to precisely capture order adherence and output diversity.
- Large-scale experiments on 36 models cover major open-source and proprietary LLMs, offering widely valuable insights.
- The paper explains experimental phenomena from linguistic perspectives (generative grammar and usage-based theory), providing deep analysis.

## Limitations & Future Work

- The concept sets are fixed at 4 words, leaving longer or more complex concept combination scenarios unexplored.
- The evaluations are conducted solely in zero-shot settings, without exploring whether few-shot examples could improve order adherence.
- The evaluation is based on English; the performance of compositional generalization in other languages remains unknown.
- As a purely evaluative study, it does not propose training methods to improve LLMs' instruction following and compositional generalization capabilities.
- Metrics like Diverse Rate cannot distinguish between "meaningful diversity" and "irrelevant variations".

## Related Work & Insights

- **Generative Commonsense Reasoning**: CommonGen (Lin et al., 2020), CommonGen-lite; focusing on constrained text generation with concept coverage.
- **Instruction Following Evaluation**: FLAN (Wei et al., 2022), IFEval; measuring LLMs' ability to follow explicit constraints.
- **Compositional Generalization**: Systematic compositionality definition proposed by Lake & Baroni (2018); formal benchmarks such as SCAN and COGS.
- **Linguistic Theories**: Chomsky's generative grammar, Jackendoff's semantic compositionality, and Bybee's usage-based theory.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — A novel perspective of unifying instruction following and compositional generalization in evaluation.
- **Technical Depth**: ⭐⭐⭐ — The methodology itself is simple (full permutations + phrase insertion), and the technical contribution leans heavily towards evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive analysis with 36 models, 6 templates, and multi-dimensional metrics.
- **Value**: ⭐⭐⭐⭐ — Reveals compositional generalization bottlenecks in LLMs, providing direct guidance for future research in constrained generation and instruction tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Disentangling Memory and Reasoning Ability in Large Language Models](disentangle_memory_reasoning.md)
- [\[ACL 2025\] MDCure: A Scalable Pipeline for Multi-Document Instruction-Following](mdcure_a_scalable_pipeline_for_multi-document_instruction-following.md)
- [\[ACL 2025\] Systematic Generalization in Language Models Scales with Information Entropy](systematic_generalization_in_language_models_scales_with_information_entropy.md)
- [\[ACL 2025\] Re-TASK: Revisiting LLM Tasks from Capability, Skill, and Knowledge Perspectives](re-task_revisiting_llm_tasks_from_capability_skill_and_knowledge_perspectives.md)
- [\[ACL 2025\] Enhancing the Rule Learning Ability of Large Language Model Agent through Induction, Deduction, and Abduction](idea_enhancing_the_rule_learning_ability_of_large_language_model_agent_through_i.md)

</div>

<!-- RELATED:END -->
