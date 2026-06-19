---
title: >-
  [Paper Note] Revisiting a Pain in the Neck: A Semantic Reasoning Benchmark for Language Models
description: >-
  [ACL 2026][LLM Evaluation][SEMANTICQA] This paper proposes SEMANTICQA, which unifies idioms, lexical collocations, noun compounds, and verbal multiword expressions into classification, extraction, interpretation, and sequential composition tasks. It finds that while strong LLMs perform well in open-ended interpretation, they remain significantly unstable in
tags:
  - ACL 2026
  - LLM Evaluation
  - SEMANTICQA
  - semantic phrase
  - multiword expression
  - benchmark
  - task composition
date: 2026-05-08
content_hash: 8b1d9e83ea7586e7
---
# Revisiting a Pain in the Neck: A Semantic Reasoning Benchmark for Language Models

**Conference**: ACL2026 Oral  
**arXiv**: [2604.16593](https://arxiv.org/abs/2604.16593)  
**Code**: https://github.com/jacklanda/SemanticQA  
**Area**: LLM Evaluation / Semantic Reasoning / Phrase Semantics  
**Keywords**: SEMANTICQA, semantic phrase, multiword expression, benchmark, task composition

## TL;DR
This paper proposes SEMANTICQA, which unifies idioms, lexical collocations, noun compounds, and verbal multiword expressions into classification, extraction, interpretation, and sequential composition tasks. It finds that while strong LLMs perform well in open-ended interpretation, they remain significantly unstable in structured extraction, fine-grained semantic classification, and cascaded workflows.

## Background & Motivation
**Background**: Large language models are frequently evaluated on benchmarks like mathematics, code, and logical reasoning, which primarily test explicit symbolic or procedural reasoning. Phrase semantics differs by requiring models to understand multiword expressions (MWEs) in context, such as idioms, lexical collocations, noun compounds, and verbal constructions.

**Limitations of Prior Work**: Existing MWE resources are abundant but typically focus on a single phrase type, a single task format, or a single semantic phenomenon. A model performing well on one task might have merely learned a specific format or prompt template, which does not necessarily indicate stable phrase-level semantic representation.

**Key Challenge**: Phrase semantic capability cannot be summarized by a single score. Classification requires selecting correct semantic relations, extraction requires precise span localization, and interpretation requires generating contextualized paraphrases; these operations share underlying phrasal meaning but have entirely different output constraints and error profiles.

**Goal**: The authors aim to construct an operation-aligned benchmark that reorganizes existing MWE resources into a unified testbed to systematically evaluate model semantic stability across atomic tasks, few-shot settings, fine-grained category scaling, and sequential task compositions.

**Key Insight**: Instead of proposing a new semantic theory, the paper maps existing resources to controlled operations: classification, extraction, and interpretation. It also designs sequential task compositions, such as extracting a phrase before interpreting or classifying it.

**Core Idea**: By using unified prompts and a multi-task structure to control for format differences, SEMANTICQA measures whether a model maintains phrase semantic consistency across different operations, rather than checking surface scores on isolated tasks.

## Method

### Overall Architecture
The core mechanism of SEMANTICQA is to replace "adding datasets" with "decoupling operations": four types of phrases—idiomatic expressions (IE), lexical collocations (LC), noun compounds (NC), and verbal multiword expressions (VMWE)—are reorganized into unified semantic operations of classification, extraction, and interpretation. Each sample consists of a fixed prompt template, a contextual sentence, and a target output to control for format variance. Evaluation does not simply average all tasks into a leaderboard; instead, it observes whether the same model maintains semantic consistency across the three operations and whether few-shot learning truly improves grounding or if upstream extraction errors are amplified downstream. Classification uses accuracy, extraction uses sequence-level exact match, and interpretation primarily uses METEOR supplemented by ROUGE-L and BERTScore.

### Key Designs
**1. Operation-aligned benchmark construction: Aligning fragmented resources to unified semantic operations**

Existing MWE resources often target only one phrase type or task format. High scores might stem from learning format templates. This work remaps four phrase types to controlled operations: IE for detection/extraction/interpretation, LC for semantic relation categorization/extraction/interpretation, NC for compositionality classification/extraction/interpretation, and VMWE for verbal construction extraction. Every task uses a unified prompt structure and explicit output constraints. Consequently, while a model might gain points via fluent paraphrasing in open interpretation, strict extraction and multi-class classification reveal whether it is truly grounded in phrasal structures and semantic relations.

**2. Sequential task composition: Decoupling atomic capabilities from workflow robustness**

Real-world phrase processing often involves "identification followed by understanding" in cascaded scenarios. If intermediate outputs are consumed by downstream steps, errors propagate. The paper designs two composite tasks, extraction-interpretation and extraction-classification, reporting both conditional scores (downstream performance given correct upstream extraction) and overall scores (end-to-end performance). Comparing these two scores allows for a decoupled diagnosis of whether a model has adequate atomic capability but fails when integrated into a workflow.

**3. Oracle Schema and category scale analysis: Probing the impact of explicit semantic definitions and category granularity**

Oracle Schema adds target types and definitions (e.g., explaining the non-compositionality of verb-particle constructions) into the prompt for VMWE extraction to test if models fail due to a lack of explicit schema. Category scale analysis expands the number of semantic categories in LC classification from 1, 2, 4, 8 to 16 to observe how accuracy declines with refinement. If performance collapses as categories become more granular, it suggests that in-context semantic reasoning cannot yet replace supervised learning.

### Loss & Training
SEMANTICQA is a benchmark and does not involve training new models. API-based and open-source LLMs are evaluated under zero-shot, three-shot, and five-shot settings with temperature 0 and top-p 1.0. Non-API baselines include supervised fine-tuned (SFT) models like BERT and T5. Three linguistics graduate students randomly labeled 100 samples per task as a difficulty reference (rather than an absolute upper bound). Due to varying generation formats across models, pre-runs were conducted, and task-specific heuristics were used to parse model outputs.

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

| Analysis Setting | Key Metric | Description |
|----------|---------|------|
| LC Category Scale: DeepSeek-R1 zero-shot | Acc@2 81.7, Acc@16 35.4 | Increasing categories makes fine-grained semantic relation classification significantly harder |
| LC Category Scale: GPT-5 zero-shot | Acc@2 92.2, Acc@16 56.3 | Frontier models also drop significantly as category scale increases |
| LC Category Scale: GPT-5 five-shot | Acc@2 94.4, Acc@16 65.2 | Few-shot mitigates but does not eliminate degradation from category refinement |
| VMWE Oracle Schema: DeepSeek-R1 zero-shot | 64.1 vs 51.6, +12.5 | Providing target type definitions significantly improves extraction |
| VMWE Oracle Schema: GPT-5 five-shot | 72.6 vs 65.7, +6.9 | Strong models also benefit from explicit schemas |

### Key Findings
- Few-shot learning is most stable and effective for interpretation tasks, but this does not necessarily equate to strict semantic grounding; gains in ROUGE-L and BERTScore may reflect exemplar-guided reconstruction.
- Extraction tasks are the most unstable as they require precise span grounding; generating fluent interpretations does not guarantee the model can correctly extract the phrase.
- In sequential extraction-interpretation, overall MTR is significantly lower than conditional MTR. For example, GPT-5 in LC five-shot achieves 41.3 in extraction and 41.8 in conditional MTR, but the overall MTR is only 17.3, indicating that upstream extraction is the primary bottleneck.
- Sequential classification also degrades significantly with the number of categories. GPT-5 in LC 16-class five-shot has a conditional accuracy of 73.4, but an overall accuracy of only 44.8.

## Highlights & Insights
- The value of SEMANTICQA lies in "decoupling operations" rather than just adding a new dataset. By placing the same phrase semantics under different output constraints, it exposes localized strengths and weaknesses in model capabilities.
- The paper is cautious about interpretation metrics: high BERTScore or METEOR may indicate close paraphrasing but not necessarily correct grounding during extraction or classification.
- Sequential composition provides an excellent diagnostic perspective. Many real-world systems are not single-step QA but involve identifying an entity/phrase before interpreting, classifying, or retrieving it; SEMANTICQA shows that current models remain unreliable in this type of workflow robustness.

## Limitations & Future Work
- The authors acknowledge that SEMANTICQA only covers English and four common phrase categories, omitting long-tail types like multiword named entities or complex functional words.
- While it includes various task formats, future work should incorporate more complex sequential compositions and semantic retrieval evaluation paradigms.
- Models evolve rapidly; the benchmark needs continuous updates with broader model families and stronger post-training versions.
- Human scores are derived from 100 samples per task and three linguistics graduate students, making them more suitable as a difficulty reference than an absolute human ceiling.

## Related Work & Insights
- **vs. Math/Code/Logic Benchmarks**: Those tasks emphasize explicit symbolic steps, whereas SEMANTICQA emphasizes internal phrase semantics, contextual disambiguation, and MWE grounding.
- **vs. Traditional MWE Resources**: Existing resources are often fragmented by phrase type; this work reorganizes them into a unified operational framework to compare model consistency across different semantic operations.
- **vs. Single-task Idiom Benchmarks**: Isolated idiom detection or paraphrase tasks make it difficult to judge if a model truly understands the phrase; SEMANTICQA uses cross-verification through extraction, classification, interpretation, and composition tasks.
- **Insight**: LLM evaluation should focus more on "consistency of the same knowledge across different operations" rather than high scores in a single format.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The main innovation is in benchmark organization and operation alignment rather than a new model, but the problem framing is precise.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple models, zero/few-shot, human references, category scaling, sequential composition, and Oracle schemas, providing rich diagnostic dimensions.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure and cautious discussion; the main table is large and requires focusing on task operations rather than cell-by-cell comparison.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for LLM semantic understanding evaluation, MWE processing, and benchmark design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Revisiting the Reliability of Language Models in Instruction-Following](revisiting_the_reliability_of_language_models_in_instruction-following.md)
- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)
- [\[ACL 2026\] Evaluating Reasoning Models for Queries with Presuppositions](evaluating_reasoning_models_for_queries_with_presuppositions.md)
- [\[ACL 2025\] Com2: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models](../../ACL2025/llm_evaluation/com2_causal_commonsense.md)

</div>

<!-- RELATED:END -->
