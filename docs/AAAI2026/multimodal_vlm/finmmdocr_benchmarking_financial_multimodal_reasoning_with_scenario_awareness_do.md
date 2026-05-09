---
title: >-
  [Paper Note] FinMMDocR: Benchmarking Financial Multimodal Reasoning with Scenario Awareness, Document Understanding, and Multi-Step Computation
description: >-
  [AAAI 2026][Multimodal VLM][Financial Reasoning] This paper introduces FinMMDocR, a bilingual multimodal reasoning benchmark targeting real-world financial scenarios. It comprises 1,200 expert-annotated numerical reasoning questions spanning 12 implicit financial scenario types, 9 categories of long documents (averaging 50.8 pages), and reasoning chains averaging 11 steps. The strongest MLLM (o4-mini-high) achieves only 58% accuracy, exposing critical deficiencies of existing models in complex financial reasoning.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Financial Reasoning
  - Multimodal Benchmark
  - Document Understanding
  - Multi-Step Computation
  - RAG
date: 2026-05-08
content_hash: ac12a151c237702a
---

# FinMMDocR: Benchmarking Financial Multimodal Reasoning with Scenario Awareness, Document Understanding, and Multi-Step Computation

**Conference**: AAAI 2026
**arXiv**: [2512.24903](https://arxiv.org/abs/2512.24903)
**Code**: [https://bupt-reasoning-lab.github.io/FinMMDocR](https://bupt-reasoning-lab.github.io/FinMMDocR)
**Area**: Multimodal VLM
**Keywords**: Financial Reasoning, Multimodal Benchmark, Document Understanding, Multi-Step Computation, RAG

## TL;DR
This paper introduces FinMMDocR, a bilingual multimodal reasoning benchmark targeting real-world financial scenarios. It comprises 1,200 expert-annotated numerical reasoning questions spanning 12 implicit financial scenario types, 9 categories of long documents (averaging 50.8 pages), and reasoning chains averaging 11 steps. The strongest MLLM (o4-mini-high) achieves only 58% accuracy, exposing critical deficiencies of existing models in complex financial reasoning.

## Background & Motivation

### State of the Field
Multimodal large language models (MLLMs) and large multimodal reasoning models (LMRMs) have made significant advances in recent years, demonstrating strong performance on visual commonsense reasoning and VQA tasks. Nevertheless, their capabilities in real-world professional domains such as finance remain insufficiently evaluated.

### Limitations of Prior Work
Existing financial QA and document QA benchmarks suffer from three critical shortcomings:

**Lack of realistic financial scenarios**: Conventional benchmarks (e.g., FinQA, TAT-QA) focus solely on extracting explicitly stated information, neglecting the ability of financial analysts to form assumptions and judgments based on market context.

**Insufficient multimodal document understanding**: Some benchmarks rely on plain-text input only; multimodal benchmarks contain charts and tables that are too sparse and isolated; and long-document benchmarks lack domain-specific numerical reasoning questions.

**Neglect of precise multi-step computation**: Financial decision-making demands exact numerical computation, yet existing benchmarks overlook issues of units, percentages, and decimal precision, or permit excessively large error tolerances (e.g., 1%).

### Root Cause
In practice, financial analysts must simultaneously: (1) understand market context and implicit conditions; (2) locate and extract scattered key information from professional documents spanning dozens of pages; and (3) execute precise multi-step numerical computations. Existing benchmarks fail to evaluate all three dimensions jointly.

### Starting Point
The paper constructs a "trinity" financial multimodal reasoning benchmark that jointly assesses scenario awareness, document understanding, and multi-step computation, supplemented by bilingual coverage (Chinese and English), rigorous evaluation criteria, and comprehensive RAG analysis.

## Method

### Overall Architecture
FinMMDocR consists of 1,200 bilingual (600 Chinese + 600 English) numerical reasoning questions. Each question is paired with a realistic financial scenario, a visually rich financial document, evidence page annotations, a gold Python solution program, and a precise reference answer.

### Key Designs

#### 1. **Scenario Awareness**
- 57.9% of questions involve implicit financial scenarios (requiring inferential assumption rather than given conditions).
- Covers 12 financial scenario categories (e.g., portfolio management, financial modeling and forecasting).
- Each question involves an average of 1.9 mixed scenarios.
- **Design Motivation**: In real-world financial analysis, analysts must integrate current market context to make professional judgments, rather than simply extracting pre-stated information from documents.

#### 2. **Document Understanding**
- 837 Chinese and English financial documents across 9 categories (e.g., company research, industry research, futures and options, financial engineering).
- Average document length: 50.8 pages / 38.8k tokens.
- Includes specialized financial charts (e.g., candlestick charts).
- 65% of questions require cross-page reasoning (average of 2.4 evidence pages).
- **Design Motivation**: Locating and extracting key data points from lengthy professional documents is a core practical skill for financial analysts.

#### 3. **Multi-Step Computation**
- Average of 11 reasoning steps: 5.3 information extraction steps (1.0 textual + 4.3 visual) + 5.7 computation steps.
- Strict evaluation standard: 0.2% error tolerance with precise assessment of units, percentages, and decimal precision.
- Gold Python solutions provided for each question.
- **Design Motivation**: Financial decisions are high-stakes; computational errors can lead to significant losses, necessitating exact answers.

### Data Construction Pipeline

#### English Data (600 questions)
- 600 questions selected from DocMath-Eval_CompLong (300 testmini + 300 test).
- Supplemented with complete solution programs, reference answers, and evidence page annotations.
- Each document page rendered as an image; original plain-text input removed.

#### Chinese Data (600 questions, newly constructed)
- 385 Chinese research reports collected from authorized sources.
- Realistic financial scenarios constructed based on document content.
- Questions and solutions generated with assistance from Gemini 2.5 Pro and Claude 3.7 Sonnet.
- Rigorous quality control: annotated by 15 finance graduate students and 2 CFA experts.

#### Quality Assurance
- Cross-review conducted on candidate annotations.
- 159 of 759 initial questions rejected (21% elimination rate).
- Among the retained 600 questions, 494 required revision (82%), including 451 with corrected evidence pages, 80 with adjusted solutions, and 36 with rephrased questions.

## Key Experimental Results

### Main Results (Image-input MLLMs)

| Model | Scale | Overall ACC | w/ Scenario | w/o Scenario | Doc ≤30 pages | Doc ≥31 pages |
|-------|-------|-------------|-------------|--------------|---------------|---------------|
| OpenAI o4-mini-high | — | **58.00** | 55.72 | 62.34 | 57.02 | 58.95 |
| Doubao-1.5-thinking-pro | — | 38.17 | 39.50 | 35.41 | 43.99 | 32.51 |
| Claude 3.7 Sonnet (Thinking) | — | 37.00 | 35.60 | 39.40 | 41.96 | 32.18 |
| Qwen2.5-VL | 72B | 12.92 | 10.57 | 17.71 | 14.04 | 11.82 |
| Llama 4 Maverick | 400A17B | 2.67 | 3.65 | 0.75 | 1.86 | 3.45 |

### Ablation Study (Error Analysis on 100 failure cases of o4-mini-high)

| Error Type | Occurrences / 100 | Description |
|------------|-------------------|-------------|
| Document Understanding Error | 78 | Failure to accurately locate or extract key information |
| Knowledge & Reasoning Error | 44 | Incorrect formula selection or reasoning structure |
| Scenario Awareness Error | 33 | Misinterpretation of task intent or contextual constraints |
| Numerical Computation Error | 5 | Correct formula but insufficient computation precision |

### RAG Analysis

| Method Type | Representative Methods | Key Findings |
|-------------|------------------------|--------------|
| Text RAG | BM25, Contriever, BGE-M3 | Inferior to visual RAG |
| Visual RAG | ColQwen2.5 | **Best retrieval performance** |
| Agentic RAG | ViDoRAG, MDocAgent, and 3 others | Lower accuracy than simple ColQwen2.5, yet consumes more tokens and time |

### Key Findings
1. **No model exceeds 60% accuracy**: The strongest model, o4-mini-high, achieves only 58%; open-source models lag further behind.
2. **Reasoning-augmented models consistently outperform non-reasoning counterparts**: All top-three models are reasoning-augmented.
3. **Large disparity in visual understanding**: The performance gap among MLLMs in visual capability (~30%) is substantially larger than that in LLM text comprehension (~12%).
4. **Information extraction is the primary bottleneck**: Under the PoT setting, extraction errors have a greater impact than computation errors.
5. **Complex agents underperform simple RAG**: Longer pipelines introduce error propagation; iterative agents incur high latency with marginal gains.
6. **Multi-scenario tasks are more challenging**: Model accuracy decreases significantly as the number of scenarios per question increases.

## Highlights & Insights
- **Unified three-dimensional design**: The integrated evaluation of scenario awareness, document understanding, and multi-step computation closely mirrors real financial analyst workflows.
- **Bilingual coverage**: Breaks the English-centric dominance prevalent in financial NLP benchmarks.
- **Rigorous quality control**: An 82% data revision rate reflects exceptionally high annotation standards.
- **In-depth RAG analysis**: Systematic comparison of 6 retrieval models and 5 agentic RAG systems; the finding that "complex agents underperform simple RAG" carries significant practical value.
- **Optimal model configuration identified**: o4-mini-high is the only model whose image-input performance surpasses the text-based (OCR + LLM) configuration.

## Limitations & Future Work
- The benchmark scale of 1,200 questions is relatively small compared to other benchmarks.
- Chinese documents are sourced from "authorized channels," raising questions about reproducibility.
- The 0.2% error tolerance, while strict, may be overly demanding for certain questions involving extensive intermediate computations.
- Evaluation focuses solely on the Program-of-Thought (PoT) paradigm; other reasoning paradigms (e.g., CoT) are not sufficiently explored.
- Agentic RAG evaluation is limited to a single backbone model (Doubao-1.5-vision-pro); different backbones may yield different results.
- Human expert baselines under identical conditions are absent.

## Related Work & Insights
- **DocMath-Eval** (Zhao et al., 2024): The source of English data in this work, though the original version supports only text input.
- **FinMMR** (Tang et al., 2025) and **MME-Finance** (Gan et al., 2025): Single-image financial reasoning benchmarks with limited difficulty and scenario diversity.
- **ColPali/ColQwen2.5** (Faysse et al., 2025): Visual retrieval methods demonstrate strong performance in financial document retrieval.
- **Insight**: Future domain-specific benchmarks should not only incorporate domain knowledge but also simulate realistic professional workflows (scenario judgment → information extraction → multi-step reasoning → precise computation).

## Rating
- Novelty: ⭐⭐⭐⭐ (Unified three-dimensional design and bilingual coverage are novel, though benchmark contributions inherently have limited methodological innovation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (26 configurations + RAG analysis + error analysis; highly comprehensive)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, though substantial detail is deferred to appendices)
- Value: ⭐⭐⭐⭐⭐ (Fills a gap in financial multimodal reasoning evaluation with high practical utility)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models](../../ICLR2026/multimodal_vlm/frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models.md)
- [\[AAAI 2026\] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)
- [\[ICCV 2025\] FinMMR: Make Financial Numerical Reasoning More Multimodal, Comprehensive, and Challenging](../../ICCV2025/multimodal_vlm/finmmr_make_financial_numerical_reasoning_more_multimodal_comprehensive_and_chal.md)
- [\[ICLR 2026\] Index-Preserving Lightweight Token Pruning for Efficient Document Understanding](../../ICLR2026/multimodal_vlm/index-preserving_lightweight_token_pruning_for_efficient_document_understanding_.md)
- [\[CVPR 2026\] DocSeeker: Structured Visual Reasoning with Evidence Grounding for Long Document Understanding](../../CVPR2026/multimodal_vlm/docseeker_long_document_understanding.md)

<!-- RELATED:END -->
