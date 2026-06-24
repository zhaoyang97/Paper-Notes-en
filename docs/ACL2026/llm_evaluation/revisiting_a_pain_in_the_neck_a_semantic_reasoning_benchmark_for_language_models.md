---
title: >-
  [Paper Note] Revisiting a Pain in the Neck: A Semantic Reasoning Benchmark for Language Models
description: >-
  [ACL2026 Oral][LLM Evaluation][SEMANTICQA] This paper proposes SEMANTICQA, which unifies idioms, lexical collocations, noun compounds, and verbal multiword expressions into classification, extraction, interpretation, and sequential composition tasks. It finds that while strong LLMs perform well in open-ended interpretation, they remain significantly unstable in structured extraction, fine-grained semantic classification, and cascaded workflows.
tags:
  - "ACL2026 Oral"
  - "LLM Evaluation"
  - "SEMANTICQA"
  - "semantic phrase"
  - "multiword expression"
  - "benchmark"
  - "task composition"
date: 2026-05-08
content_hash: c08a522fe6e5d14a
---

# Revisiting a Pain in the Neck: A Semantic Reasoning Benchmark for Language Models

**Conference**: ACL2026 Oral  
**arXiv**: [2604.16593](https://arxiv.org/abs/2604.16593)  
**Code**: https://github.com/jacklanda/SemanticQA  
**Area**: LLM Evaluation / Semantic Reasoning / Phrasal Semantics  
**Keywords**: SEMANTICQA, semantic phrase, multiword expression, benchmark, task composition

## TL;DR
This paper proposes SEMANTICQA, which unifies idioms, lexical collocations, noun compounds, and verbal multiword expressions into classification, extraction, interpretation, and sequential composition tasks. It finds that while strong LLMs perform well in open-ended interpretation, they remain significantly unstable in structured extraction, fine-grained semantic classification, and cascaded workflows.

## Background & Motivation
**Background**: Large Language Models (LLMs) are frequently evaluated on benchmarks for mathematics, code, and logical reasoning. However, these tasks primarily test explicit symbolic or procedural reasoning. Phrasal semantics differs by requiring models to understand multiword expressions (MWEs) within context, such as idioms, lexical collocations, noun compounds, and verbal constructions.

**Limitations of Prior Work**: While multiword expression resources are abundant, they typically focus on a single phrase type, a single task format, or a single semantic phenomenon. A model's high performance on a specific task may result from learning a format or prompt template rather than possessing stable phrase-level semantic representations.

**Key Challenge**: Phrasal semantic ability cannot be summarized by a single score. Classification requires selecting the correct semantic relationship, extraction requires precise span localization, and interpretation requires generating contextualized paraphrases. These operations share an underlying phrasal meaning but possess entirely different output constraints and error patterns.

**Goal**: The authors aim to construct an operation-aligned benchmark that reorganizes existing MWE resources into a unified testbed. This enables systematic evaluation of model semantic stability across atomic tasks, few-shot settings, fine-grained category expansion, and sequential task compositions.

**Key Insight**: Instead of proposing a new semantic theory, the paper maps existing resources to controlled operations: classification, extraction, and interpretation. It also designs sequential task compositions, such as extracting a phrase before interpreting or classifying it.

**Core Idea**: By using unified prompts and a multi-task structure to control for format variance, SEMANTICQA measures whether a model maintains phrasal semantic consistency across different operations, rather than merely achieving surface-level scores on isolated tasks.

## Method

### Overall Architecture
The core mechanism of SEMANTICQA is to replace "adding datasets" with "decoupling operations." It reorganizes four types of phrases—idiomatic expressions (IE), lexical collocations (LC), noun compounds (NC), and verbal multiword expressions (VMWE)—into unified semantic operations: classification, extraction, and interpretation. Each sample consists of a fixed prompt template, a context sentence, and a target output to control for format differences. Evaluation does not simply average all tasks into a single leaderboard; instead, it observes whether the same model maintains consistency across operations and assesses whether few-shot learning improves grounding or if upstream extraction errors are amplified downstream. Accuracy is used for classification, sequence-level exact match for extraction, and METEOR (supplemented by ROUGE-L and BERTScore) for interpretation.

### Key Designs
**1. Operation-aligned benchmark construction: Aligning dispersed resources to a unified set of semantic operations**

Existing MWE resources often focus on one phrase type or task format. High scores may indicate only that a model has learned a specific template. This work re-maps four phrase categories to controlled operations: detection/extraction/interpretation for IE; semantic relation categorization/extraction/interpretation for LC; compositionality classification/extraction/interpretation for NC; and verbal construction extraction for VMWE. Each task employs a unified prompt structure and explicit output constraints. Consequently, while a model might "game" open interpretation via fluent paraphrasing, strict extraction and multi-class classification reveal whether it is truly grounded in phrasal structure and semantic relations.

**2. Sequential task composition: Separating atomic capabilities from workflow robustness**

Real-world phrase processing often involves cascaded scenarios where identification precedes understanding. If intermediate outputs are consumed by downstream steps, errors propagate. The paper designs two composite tasks—extraction-interpretation and extraction-classification—and reports both conditional scores (downstream performance given correct upstream extraction) and overall scores (end-to-end performance). Comparing these scores helps diagnose whether a model's failure stems from weak atomic capabilities or instability within a workflow.

**3. Oracle Schema and Category Scale Analysis: Probing the impact of explicit semantic definitions and category granularity**

The Oracle Schema adds target types and definitions to prompts for VMWE extraction (e.g., explaining the non-compositionality of verb-particle constructions) to check if models fail simply because they "do not know which category to extract." Category scale analysis expands the number of semantic categories in LC classification from 1, 2, 4, 8 to 16 to observe how accuracy declines. If performance collapses as categories refine, it suggests that in-context semantic reasoning cannot yet replace supervised learning.

### Loss & Training
SEMANTICQA is an evaluation benchmark and does not involve training new models. API-based and open-source LLMs are evaluated in zero-shot, three-shot, and five-shot settings with a sampling temperature of $0$ and top-p of $1.0$. Non-API baselines include supervised fine-tuned models like BERT and T5. Three linguistics graduate students randomly annotated 100 samples per task to serve as a reference for task difficulty (rather than an absolute upper bound). Due to varying generation formats, a pre-run is conducted, and task-specific heuristics are used to parse model outputs.

## Key Experimental Results

### Main Results

| Model / Setting | IED ACC | IEE ACCs | IEI MTR | LCC ACC | LCE ACCs | LCI MTR | NCC ACC | NCE ACCs | NCI MTR | VPE ACCs | LVE ACCs | VIE ACCs |
|-------------|---------|----------|---------|---------|----------|---------|---------|----------|---------|----------|----------|----------|
| Human | 71.0 | 87.0 | 20.5 | 47.0 | 50.0 | 16.7 | 71.0 | 73.0 | 17.2 | 85.0 | 55.0 | 78.0 |
| DeepSeek-R1 zero-shot | 71.1 | 69.4 | 12.4 | 66.6 | 31.5 | 31.8 | 60.2 | 51.3 | 31.4 | 76.8 | 26.7 | 50.5 |
| DeepSeek-R1 five-shot | 84.3 | 72.3 | 19.2 | 76.1 | 64.3 | 32.9 | 60.6 | 70.7 | 68.7 | 81.6 | 35.8 | 57.1 |
| GPT-5 zero-shot | 82.8 | 67.6 | 13.9 | 75.4 | 36.7 | 33.7 | 66.8 | 64.3 | 57.3 | 74.2 | 28.9 | 56.2 |
| GPT-5 five-shot | 85.4 | 78.7 | 22.5 | 84.3 | 68.9 | 37.4 | 67.2 | 79.0 | 75.3 | 74.7 | 38.3 | 50.5 |

### Ablation Study

| Analysis Setting | Key Metrics | Description |
|----------|---------|------|
| LC Category Expansion: DeepSeek-R1 zero-shot | Acc@2 81.7, Acc@16 35.4 | Fine-grained semantic relation classification becomes significantly harder as categories increase. |
| LC Category Expansion: GPT-5 zero-shot | Acc@2 92.2, Acc@16 56.3 | Frontier models also show significant degradation as category scale expands. |
| LC Category Expansion: GPT-5 five-shot | Acc@2 94.4, Acc@16 65.2 | Few-shot learning mitigates but does not eliminate degradation from category refinement. |
| VMWE Oracle Schema: DeepSeek-R1 zero-shot | 64.1 vs 51.6, +12.5 | Providing target type definitions significantly improves extraction performance. |
| VMWE Oracle Schema: GPT-5 five-shot | 72.6 vs 65.7, +6.9 | Strong models also benefit from an explicit schema. |

### Key Findings
- Few-shot learning is most stable and effective for interpretation tasks, but this does not necessarily equate to strict semantic grounding; gains in ROUGE-L and BERTScore may reflect exemplar-guided reconstruction.
- Extraction tasks are the most unstable as they require precise span grounding; fluent interpretation generation does not guarantee successful phrase extraction.
- In sequential extraction-interpretation, the overall MTR is significantly lower than the conditional MTR. For instance, GPT-5 under LC five-shot achieves an extraction score of 41.3 and a conditional MTR of 41.8, but an overall MTR of only 17.3, identifying upstream extraction as the primary bottleneck.
- Sequential classification also degrades markedly with an increase in the number of categories. GPT-5 in a 16-class LC setting achieves a five-shot conditional accuracy of 73.4, but an overall accuracy of only 44.8.

## Highlights & Insights
- The value of SEMANTICQA lies in "decoupling operations" rather than simply adding a new dataset. By placing the same phrasal semantics under different output constraints, it exposes local strengths and weaknesses in model capability.
- The paper is cautious regarding interpretation metrics: high BERTScore or METEOR may indicate only close paraphrasing and does not prove the model correctly grounded the phrase during extraction or classification.
- Sequential composition provides a valuable diagnostic perspective. Many real-world systems are not single-step QA but involve locating an entity/phrase before interpreting, classifying, or retrieving it; SEMANTICQA shows that current models remain unreliable in this type of workflow robustness.

## Limitations & Future Work
- The authors acknowledge that SEMANTICQA currently covers only English and four common types of phrasal phenomena, excluding long-tail types like multiword named entities or complex functional words.
- While it includes multiple task formats, future work should incorporate more complex sequential compositions and semantic retrieval paradigms.
- Given the rapid pace of model development, the benchmark requires continuous updates to include broader model families and stronger post-training models.
- Human scores were derived from 100 samples per task by three linguistics graduate students, making them suitable as difficulty references rather than absolute human ceilings.

## Related Work & Insights
- **vs Mathematics/Code/Logic Benchmarks**: Those tasks emphasize explicit symbolic steps, whereas SEMANTICQA emphasizes internal phrasal semantics, contextual disambiguation, and multiword expression grounding.
- **vs Traditional MWE Resources**: Existing resources are often fragmented by phrase type. This work reorganizes them into a unified operational framework to compare model consistency across semantic operations.
- **vs Single-task Idiom Benchmarks**: Isolated idiom detection or paraphrase tasks make it difficult to judge true understanding; SEMANTICQA uses cross-validation through extraction, classification, interpretation, and composite tasks.
- **Insight**: LLM evaluation should focus more on "consistency of the same knowledge across different operations" rather than performance on a single format.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The main innovation lies in benchmark organization and operation alignment rather than a new model, but it identifies the core problem accurately.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes multi-model evaluation, zero/few-shot settings, human references, category scaling, sequential composition, and Oracle schemas.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure and cautious discussion; the main tables are large and require focusing on task operations rather than cell-by-cell comparison.
- Value: ⭐⭐⭐⭐⭐ Highly relevant for evaluating LLM semantic understanding, MWE processing, and benchmark design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Revisiting the Reliability of Language Models in Instruction-Following](revisiting_the_reliability_of_language_models_in_instruction-following.md)
- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)
- [\[ICLR 2026\] Rethinking LLM-as-a-Judge: Representation-as-a-Judge with Small Language Models via Semantic Capacity Asymmetry](../../ICLR2026/llm_evaluation/rethinking_llm-as-a-judge_representation-as-a-judge_with_small_language_models_v.md)
- [\[ACL 2026\] Evaluating Reasoning Models for Queries with Presuppositions](evaluating_reasoning_models_for_queries_with_presuppositions.md)

</div>

<!-- RELATED:END -->
