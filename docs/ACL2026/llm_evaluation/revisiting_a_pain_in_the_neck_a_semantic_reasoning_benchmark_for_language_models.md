---
title: >-
  [Paper Note] Revisiting a Pain in the Neck: A Semantic Reasoning Benchmark for Language Models
description: >-
  [ACL2026][LLM Evaluation][SEMANTICQA] This paper proposes SEMANTICQA, which unifies idioms, lexical collocations, noun compounds, and verbal multiword expressions into classification, extraction, interpretation…
tags:
  - "ACL2026"
  - "LLM Evaluation"
  - "SEMANTICQA"
  - "semantic phrase"
  - "multiword expression"
  - "benchmark"
  - "task composition"
date: 2026-05-08
content_hash: 00c94236fd531aa4
---

# Revisiting a Pain in the Neck: A Semantic Reasoning Benchmark for Language Models

**Conference**: ACL2026 Oral  
**arXiv**: [2604.16593](https://arxiv.org/abs/2604.16593)  
**Code**: https://github.com/jacklanda/SemanticQA  
**Area**: LLM Evaluation / Semantic Reasoning / Phrasal Semantics  
**Keywords**: SEMANTICQA, semantic phrase, multiword expression, benchmark, task composition

## TL;DR
This paper proposes SEMANTICQA, which unifies idioms, lexical collocations, noun compounds, and verbal multiword expressions into classification, extraction, interpretation, and sequential composition tasks. It identifies that while strong LLMs appear proficient in open-ended interpretation, they remain significantly unstable in structured extraction, fine-grained semantic classification, and cascaded workflows.

## Background & Motivation
**Background**: Large Language Models (LLMs) are frequently evaluated on benchmarks for mathematics, code, and logical reasoning; however, these tasks primarily test explicit symbolic or procedural reasoning. Phrasal semantics differs by requiring models to understand multiword expressions (MWEs) within context, such as idioms, lexical collocations, noun compounds, and verbal constructions.

**Limitations of Prior Work**: Resources for multiword expressions are abundant but typically focus on a single phrase type, a single task format, or a single semantic phenomenon. A model's high performance on one task may stem from learning a specific format or prompt template rather than possessing stable phrase-level semantic representations.

**Key Challenge**: Phrasal semantic capability cannot be summarized by a single score. Classification requires selecting the correct semantic relationship, extraction requires precise span localization, and interpretation requires generating contextualized paraphrases. These operations share an underlying phrasal meaning but possess entirely different output constraints and error patterns.

**Goal**: The authors aim to construct an operation-aligned benchmark that reorganizes existing MWE resources into a unified testbed. This benchmark systematically evaluates models across atomic tasks, few-shot settings, fine-grained category scaling, and sequential task compositions.

**Key Insight**: Instead of proposing a new semantic theory, the paper maps existing resources to controlled operations: classification, extraction, and interpretation. It also designs sequential task compositions, such as extracting a phrase before classifying or interpreting it.

**Core Idea**: By using unified prompts and a multi-task structure to control for format variances, SEMANTICQA ensures that it measures whether a model maintains phrasal semantic consistency across different operations rather than merely achieving surface-level scores on isolated tasks.

## Method

### Overall Architecture
SEMANTICQA covers four categories of semantic phrases: idiomatic expressions (IE), lexical collocations (LC), noun compounds (NC), and verbal multiword expressions (VMWE). For IE, LC, and NC, the benchmark constructs classification, extraction, and interpretation tasks; for VMWE, it focuses on extraction and introduces Oracle Schema prompts that provide target types and definitions. Each sample consists of a fixed prompt template, a contextual sentence, and a target output. Accuracy is used for classification, sequence-level exact match accuracy for extraction, and METEOR (alongside ROUGE-L and BERTScore) for interpretation.

The evaluation does not simply average all tasks into a leaderboard. Instead, it observes model behavior patterns across different semantic operations: whether the same model is stable across classification, extraction, and interpretation; whether few-shot learning truly improves semantic grounding; and whether upstream extraction errors are amplified in downstream interpretation or classification.

### Key Designs
1.  **Operation-aligned benchmark construction**:
    - **Function**: Aligns dispersed MWE data resources to the same set of semantic operations, supporting comparisons across phrase types and task formats.
    - **Mechanism**: IE includes detection, extraction, and interpretation; LC includes semantic relation categorization, extraction, and interpretation; NC includes compositionality classification, extraction, and interpretation; VMWE involves verbal construction extraction. Each task utilizes a unified prompt structure and explicit output constraints.
    - **Design Motivation**: If only an open-ended interpretation task is evaluated, models might score high based on fluent paraphrasing. Rigid extraction and multi-class classification are necessary to determine if the model is genuinely grounded in phrasal structure and semantic relations.

2.  **Sequential task composition**:
    - **Function**: Simulates cascaded "recognition then understanding" scenarios in real-world phrasal semantic processing.
    - **Mechanism**: The paper designs extraction-interpretation and extraction-classification combined tasks, reporting both conditional scores and overall scores. The conditional score considers downstream performance only when upstream extraction is correct, while the overall score reflects end-to-end performance.
    - **Design Motivation**: Many models perform adequately on atomic tasks, but errors propagate once intermediate outputs are consumed by downstream steps. This design decouples atomic capability from workflow robustness.

3.  **Oracle Schema and Category Scale Analysis**:
    - **Function**: Tests the impact of explicit semantic definitions and category scales on phrasal semantic reasoning.
    - **Mechanism**: Oracle Schema adds target types and definitions to the VMWE extraction prompts (e.g., explaining the non-compositionality of verb-particle constructions). Category scale analysis in LC classification increases the number of semantic categories from 2 to 16 to observe how accuracy declines with refinement.
    - **Design Motivation**: Models may fail to extract specific expressions without an explicit semantic schema. Increasing category granularity reveals whether in-context semantic reasoning can effectively replace supervised learning.

### Loss & Training
SEMANTICQA is an evaluation benchmark and does not involve training new models. API-based and open-source LLMs are evaluated under zero-shot, three-shot, and five-shot settings with temperature set to 0 and top-p to 1.0. Non-API baselines include supervised fine-tuned models like BERT/T5. Three linguistics graduate students annotated 100 random samples per task to serve as a difficulty reference rather than an absolute upper bound. Heuristic parsers are used to process various model output formats during pre-runs.

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
| LC Scale: DeepSeek-R1 zero-shot | Acc@2 81.7, Acc@16 35.4 | Fine-grained semantic relation classification becomes significantly harder as categories increase. |
| LC Scale: GPT-5 zero-shot | Acc@2 92.2, Acc@16 56.3 | Even frontier models decline sharply as the category scale expands. |
| LC Scale: GPT-5 five-shot | Acc@2 94.4, Acc@16 65.2 | Few-shot prompting mitigates but does not eliminate degradation from category refinement. |
| VMWE Oracle Schema: DeepSeek-R1 zero-shot | 64.1 vs 51.6, +12.5 | Providing target type definitions significantly improves extraction. |
| VMWE Oracle Schema: GPT-5 five-shot | 72.6 vs 65.7, +6.9 | Strong models also benefit from explicit schemas. |

### Key Findings
- Few-shot learning is most stable and effective for interpretation tasks, but this does not necessarily equate to strict semantic grounding; improvements in ROUGE-L and BERTScore may reflect exemplar-guided reconstruction.
- Extraction tasks are the most unstable because they require precise span grounding; generating a fluent interpretation does not guarantee the model can correctly extract the phrase.
- In sequential extraction-interpretation, the overall MTR is significantly lower than the conditional MTR. For instance, with GPT-5 in LC five-shot, the extraction Acc is 41.3 and conditional MTR is 41.8, but overall MTR is 17.3, identifying upstream extraction as the primary bottleneck.
- Sequential classification also degrades significantly with increasing categories. GPT-5 achieves a conditional accuracy of 73.4 in LC 16-class five-shot, but its overall accuracy is only 44.8.

## Highlights & Insights
- The value of SEMANTICQA lies in "decoupling operations" rather than just adding a new dataset. By placing the same phrasal semantics under different output constraints, it exposes local strengths and weaknesses in model capabilities.
- The authors approach interpretation metrics with caution: high BERTScore or METEOR may indicate close paraphrasing without reflecting correct grounding during extraction or classification.
- Sequential composition provides a diagnostic perspective. Real-world systems are rarely single-step QA; they often locate entities/phrases before interpreting, categorizing, or retrieving. SEMANTICQA shows that current models remain unreliable in this workflow robustness.

## Limitations & Future Work
- The authors acknowledge that SEMANTICQA currently covers only English and four common phrasal phenomena, omitting long-tail types like multiword named entities or complex functional words.
- While it includes various task formats, future work should incorporate more complex sequential compositions and evaluative paradigms such as semantic retrieval.
- Given the rapid evolution of models, the benchmark requires continuous updates to include broader model families and stronger post-trained models.
- Human scores were derived from 100 samples per task by three linguistics students, making them suitable as difficulty references rather than absolute human ceilings.

## Related Work & Insights
- **vs. Math/Code/Logic Benchmarks**: Those tasks emphasize explicit symbolic steps, whereas SEMANTICQA focuses on internal phrasal semantics, contextual disambiguation, and multiword expression grounding.
- **vs. Traditional MWE Resources**: Existing resources are often fragmented by phrase type; this work reorganizes them into a unified operational framework to compare consistency across semantic operations.
- **vs. Single-task Idiom Benchmarks**: Isolated idiom detection or paraphrasing tasks make it difficult to judge true understanding. SEMANTICQA utilizes cross-validation via extraction, classification, interpretation, and composition.
- **Insight**: LLM evaluation should focus more on the "consistency of the same knowledge across different operations" rather than high scores in a single format.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The main innovation is in the benchmark organization and operation alignment rather than a new model, but the problem is well-defined.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple models, zero/few-shot, human references, category scaling, sequential composition, and Oracle Schema; rich diagnostic dimensions.
- Writing Quality: ⭐⭐⭐⭐☆ Structure is clear and discussions are cautious; the main tables are large and require focusing on task operations rather than cell-by-cell comparison.
- Value: ⭐⭐⭐⭐⭐ High reference value for LLM semantic evaluation, MWE processing, and benchmark design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Revisiting the Reliability of Language Models in Instruction-Following](revisiting_the_reliability_of_language_models_in_instruction-following.md)
- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)
- [\[ACL 2026\] Evaluating Reasoning Models for Queries with Presuppositions](evaluating_reasoning_models_for_queries_with_presuppositions.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] Revisiting the Reliability of Language Models in Instruction-Following](revisiting_the_reliability_of_language_models_in_instruction-following.md)
- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)
- [\[ACL 2026\] Evaluating Reasoning Models for Queries with Presuppositions](evaluating_reasoning_models_for_queries_with_presuppositions.md)
- [\[ACL 2025\] Com2: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models](../../ACL2025/llm_evaluation/com2_causal_commonsense.md)

</div>

<!-- RELATED:END -->
