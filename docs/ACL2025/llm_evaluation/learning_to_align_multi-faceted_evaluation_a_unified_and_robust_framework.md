---
title: >-
  [Paper Note] Learning to Align Multi-Faceted Evaluation: A Unified and Robust Framework
description: >-
  [ACL 2025][LLM Evaluation][multi-faceted analysis] Proposes the ARJudge evaluation framework, which adaptively generates evaluation criteria and executes text+code dual-driven analysis by fine-tuning an Analyzer, paired with a training-free Refiner for comprehensive judgment. ARJudge outperforms existing fine-tuned evaluators across multiple evaluation benchmarks, particularly achieving a performance gain of up to 11.1% on instruction-following evaluation via code-driven anal…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "multi-faceted analysis"
  - "code-driven evaluation"
  - "fine-tuned evaluator"
  - "instruction following"
date: 2026-05-08
content_hash: cb5f1fc9060009d4
---

# Learning to Align Multi-Faceted Evaluation: A Unified and Robust Framework

**Conference**: ACL 2025  
**arXiv**: [2502.18874](https://arxiv.org/abs/2502.18874)  
**Code**: None  
**Area**: Others  
**Keywords**: LLM evaluation, multi-faceted analysis, code-driven evaluation, fine-tuned evaluator, instruction following

## TL;DR

Proposes the ARJudge evaluation framework, which adaptively generates evaluation criteria and executes text+code dual-driven analysis by fine-tuning an Analyzer, paired with a training-free Refiner for comprehensive judgment. ARJudge outperforms existing fine-tuned evaluators across multiple evaluation benchmarks, particularly achieving a performance gain of up to 11.1% on instruction-following evaluation via code-driven analysis.

## Background & Motivation

Using LLMs to evaluate the output quality of other LLMs has become an important paradigm, but existing fine-tuned evaluators face two core issues:

**Insufficient Adaptability of Predefined Criteria**
   - Existing methods (e.g., Auto-J, Prometheus) use fixed, general evaluation criteria (e.g., "conciseness", "logical structure").
   - These criteria fail to cover tasks requiring specific evaluation dimensions, such as creative writing or professional domains.
   - They exhibit poor generalization capability when facing unseen new instructions.

**Instability in Evaluating Objective Constraints**
   - LLMs behave unreliably when evaluating quantitative requirements (e.g., word count limits) and structural constraints (e.g., formatting requirements).
   - They struggle to accurately judge even basic text attributes (e.g., counting word frequency).
   - Pure text analysis has inherent limitations in objective verification.

The authors argue that building a robust evaluator requires two capabilities: **adaptive generation of evaluation criteria** (what to evaluate) + **multi-faceted analysis** (how to evaluate), especially by integrating code tools to handle objective requirements.

## Method

### Overall Architecture

ARJudge consists of two components:
- **Analyzer** (Fine-tuned): Adaptively generates evaluation criteria and performs text-based as well as code-driven analysis.
- **Refiner** (Training-free): Synthesizes the multi-faceted analysis results from the Analyzer to make the final judgment.

The training data originates from a meticulously constructed **Composite Analysis Corpus**.

### Key Designs

1. **Composite Analysis Corpus Construction**

    - **Evaluation Criteria Generation**: Two types of questions
        - Type 1 (for text analysis): Given an instruction + 3 sample responses, an LLM is used to generate 3 evaluation questions.
        - Type 2 (for code analysis): Objective constraints (e.g., word count limits) are added to instructions via self-instruct, generating corresponding evaluation questions.
    - **Text Analysis Collection**: Given the instruction, two responses, and evaluation questions to the LLM, the model is required to perform comparative analysis; samples contradicting human annotations are filtered out.
    - **Code-driven Analysis Development**:
        - Generates Python verification functions for objective evaluation questions using Claude-3.5-Sonnet.
        - The input to the function is the response text, and the output is the verification result.
        - Double filtering: execution testing + reverse verification (prompting the LLM to explain the code's purpose and check its consistency with the original question).
    - Design Motivation: To organically combine "what to evaluate" and "how to evaluate" into a single training pipeline.

2. **Analyzer Training**

    - Fine-tuned based on the Qwen2.5-7B-Instruct model.
    - Approximately 25K training samples, containing three types of tasks: evaluation question generation, text-based analysis, and code generation.
    - Employs different prompt templates to distinguish between the question generation and response analysis tasks.
    - Text and code analyses share the same prompt template, with different modes triggered by starting prompts.
    - Design Motivation: Multi-task training enables the model to learn to adaptively select the appropriate analysis paradigm.

3. **Refiner Comprehensive Judgment**

    - Utilizes zero-shot inference of the same Qwen2.5-7B-Instruct model.
    - Takes all analytical results from the Analyzer as input.
    - Instructs the Refiner to revisit the instruction requirements and synthesize judgments on which response is superior.
    - Design Motivation: To preserve the general model's macro-evaluation capability and compensate for any potential biases of the Analyzer.

### Loss & Training

- Uses standard supervised fine-tuning loss (next token prediction).
- Generates analysis and judgments via greedy decoding (temperature=0).
- Only the Analyzer is fine-tuned; the Refiner uses the identical model without fine-tuning.

## Key Experimental Results

### Main Results

| Model | JudgeLM | PandaLM | Auto-J | MTBench | LLMBar | Average |
|------|---------|---------|--------|---------|--------|------|
| GPT-4o | 81.8 | 83.1 | 78.6 | 78.8 | 79.8 | 80.4 |
| Claude-3.5-Sonnet | 82.9 | 86.4 | 78.2 | 80.8 | 83.4 | 82.3 |
| Auto-J-13B | 77.9 | 77.2 | 79.7 | 75.0 | 27.8 | 67.5 |
| Prometheus2-7B | 76.5 | 76.3 | 75.1 | 74.3 | 41.5 | 68.7 |
| **ARJudge** | **81.0** | **82.4** | **78.5** | **78.3** | **68.2** | **77.7** |

ARJudge achieves the best performance among all fine-tuned evaluators, boosting accuracy by 26.7% on LLMBar compared to the best fine-tuned baseline Prometheus2.

### Ablation Study

| Setting | JudgeLM | PandaLM | Auto-J | MTBench | LLMBar |
|------|---------|---------|--------|---------|--------|
| Qwen2.5-7B (Baseline) | 80.0 | 80.7 | 73.8 | 75.2 | 52.6 |
| **ARJudge** | 81.0 | 82.4 | 78.5 | 78.3 | 68.2 |
| -w/o Fine-tuning | 73.1 | 75.6 | 68.7 | 70.0 | 62.5 |
| -w/o Fine-tuning & Multi-faceted | 74.7 | 72.2 | 65.6 | 67.8 | 63.7 |
| -w/o Refine | 81.7 | 82.8 | 79.6 | 79.1 | 63.7 |

Fine-tuning is the most critical component (performance drops globally without it), while the Refiner provides a significant boost for challenging samples (LLMBar).

### Key Findings

1. **Effectiveness of Code-Driven Analysis**: On IFEval, the consistency of code-driven analysis far exceeds that of pure text-based analysis, yielding a 100% execution success rate for generated code.
2. **Effect of Multi-Turn Analysis**: Increasing the number of analysis turns has a positive effect on conventional test sets but may amplify uncertainty on challenging datasets like LLMBar.
3. **Double-Edged Sword of Refinement**: Under a fine-tuned Analyzer, the Refiner maintains performance, whereas under a non-fine-tuned Analyzer, the Refiner actually degrades performance (failure in self-correction).
4. **Significantly Surpassing the Backbone Model**: Compared to the baseline Qwen2.5-7B level, ARJudge achieves an average improvement of 15.6%.
5. **Generalization on LLMBar**: On the adversarially designed LLMBar, the performance is close to that of DeepSeek-V3 (80.4 vs 68.2).

## Highlights & Insights

- **Unified Framework of "What and How to Evaluate"**: Rather than predefining criteria, the model adaptively generates evaluation dimensions.
- **Code as an Evaluation Tool**: Introducing code execution into the evaluation pipeline overcomes the unreliability of LLMs in objective judgment.
- **Analyzer-Refiner Division of Labor**: Combines fine-tuned expert analysis with untuned general synthesis, balancing both depth and breadth.
- **Reverse Verification Mechanism**: Employs a two-step "explanation $\rightarrow$ verification" process after code generation to ensure alignment between the code and evaluation objectives.

## Limitations & Future Work

1. Limited to pairwise comparison evaluation; does not support scoring individual responses.
2. Tool usage is restricted to Python code, without considering other validation tools such as search engines or knowledge bases.
3. The Refiner relies heavily on the LLM's own reasoning capabilities; improvements are limited if the base capacity is insufficient.
4. Training data construction depends on GPT-4o and Claude, introducing cascading bias.
5. Code generation is confined to objective constraint verification, unable to handle subjective quality evaluations.

## Related Work & Insights

- Compared to Auto-J's predefined criteria + text analysis, ARJudge introduces two additional dimensions: adaptive criteria generation and code analysis.
- Compared to Prometheus's fixed evaluation templates, ARJudge's multi-faceted analysis is more flexible.
- Insight: Code-driven analysis can be extended to more scenarios (e.g., integrating search engines for fact-checking, or calculators for mathematical reasoning).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Code-driven objective evaluation analysis offers a novel perspective, and the Analyzer-Refiner architecture has a well-structured hierarchy.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — The evaluation is comprehensive, spanning 5 datasets, various baselines, ablation studies, and case studies.
- **Writing Quality**: ⭐⭐⭐⭐ — The corpus construction process is clear and figures/tables are rich, though some sections are somewhat verbose.
- **Value**: ⭐⭐⭐⭐ — Provides a practical framework for building more reliable LLM evaluators; the code-driven analysis approach has high generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset](mars_benchmarking_the_metaphysical_reasoning_abilities_of_language_models_with_a.md)
- [\[ACL 2025\] EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework](educationq_evaluating_llms_teaching_capabilities_through_multi-agent_dialogue_fr.md)
- [\[NeurIPS 2025\] Words That Unite The World: A Unified Framework for Deciphering Central Bank Communications Globally](../../NeurIPS2025/llm_evaluation/words_that_unite_the_world_a_unified_framework_for_deciphering_central_bank_comm.md)
- [\[ACL 2025\] READoc: A Unified Benchmark for Realistic Document Structured Extraction](readoc_a_unified_benchmark_for_realistic_document_structured_extraction.md)
- [\[ACL 2025\] TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages](tumlu_a_unified_and_native_language_understanding_benchmark_for_turkic_languages.md)

</div>

<!-- RELATED:END -->
