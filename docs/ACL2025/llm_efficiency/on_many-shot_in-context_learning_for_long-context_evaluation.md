---
title: >-
  [Paper Note] On Many-Shot In-Context Learning for Long-Context Evaluation
description: >-
  [ACL 2025][LLM Efficiency][Many-shot ICL] This paper conducts an in-depth study of many-shot ICL for evaluating long-context language models, proposes the Sample Learning Ratio (SLR) metric to distinguish between SSL and ASL tasks, and constructs the ManyICLBench benchmark to comprehensively evaluate 12 LCLMs.
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "Many-shot ICL"
  - "Long-context Evaluation"
  - "ManyICLBench"
  - "Similar-Sample Learning"
  - "All-Sample Learning"
date: 2026-05-08
content_hash: 5ad2f35f9900653d
---

# On Many-Shot In-Context Learning for Long-Context Evaluation

**Conference**: ACL 2025  
**arXiv**: [2411.07130](https://arxiv.org/abs/2411.07130)  
**Code**: [https://github.com/launchnlp/ManyICLBench](https://github.com/launchnlp/ManyICLBench)  
**Area**: LLM Efficiency / Long-Context Evaluation  
**Keywords**: Many-shot ICL, Long-context Evaluation, ManyICLBench, Similar-Sample Learning, All-Sample Learning

## TL;DR
This paper conducts an in-depth study of many-shot ICL for evaluating long-context language models, proposes the Sample Learning Ratio (SLR) metric to distinguish between SSL and ASL tasks, and constructs the ManyICLBench benchmark to comprehensively evaluate 12 LCLMs.

## Background & Motivation
**Background**: Long-context language models (LCLMs) support contexts of 128K or even 1M tokens, but existing evaluations primarily focus on retrieval capabilities.

**Limitations of Prior Work**: Synthetic tasks such as Needle-in-a-Haystack only evaluate retrieval, lacking assessments of global context understanding.

**Key Challenge**: Existing many-shot ICL benchmarks like LongICLBench mainly employ classification tasks, but it remains unclear what capabilities these tasks actually evaluate.

**Goal**: (1) Which tasks benefit from more exemplars? (2) To what extent does each task rely on similar-sample retrieval versus all-sample learning?

**Key Insight**: Propose the Sample Learning Ratio (SLR) metric to quantify the degree of reliance of ICL tasks on retrieval versus understanding.

**Core Idea**: Many-shot ICL classification tasks are essentially retrieving similar exemplars, whereas true global understanding requires ASL tasks for proper evaluation.

## Method

### Overall Architecture
(1) Collect 12 ICL datasets (21 subtasks) → (2) Evaluate 12 LCLMs using 1k~128k context lengths → (3) Propose the SLR metric to analyze the skill requirements of each task → (4) Construct the ManyICLBench benchmark.

### Key Designs
1. **Sample Learning Ratio (SLR)**:

    - **Function**: Quantify the task's reliance on retrieving similar samples.
    - **Mechanism**: Compare performance change ratios by removing the 10% most similar and 10% least similar exemplars, respectively.
    - **Design Motivation**: SLR >> 1 indicates a strong reliance on retrieval, whereas SLR ≈ 1 indicates a need for all-sample learning.

2. **Task Categorization (SSL vs ASL)**:

    - **Function**: Categorize ICL tasks into Similar-Sample Learning (SSL) and All-Sample Learning (ASL).
    - **Mechanism**: SSL tasks (classification) primarily depend on retrieving similar exemplars; ASL tasks (math/summarization) require understanding all exemplars.
    - **Design Motivation**: Distinguish between the two types of skills to provide a more comprehensive evaluation.

3. **ManyICLBench Construction**:

    - **Function**: Curate a set of many-shot ICL benchmarks.
    - **Mechanism**: Include both SSL and ASL tasks, covering context sizes from 1k to 128k tokens.
    - **Design Motivation**: Evaluating along a single dimension is insufficient to reflect the true capabilities of LCLMs.

### Loss & Training
Purely an evaluation work with no training. Greedy decoding is used, with three random seeds for each experiment.

## Key Experimental Results

### Main Results (SSL Tasks, Macro F1 @ Different Token Counts)

| Model | 1k | 8k | 32k | 64k | 128k |
|------|-----|-----|------|------|-------|
| Qwen2-72B | 36.4 | 65.3 | 76.5 | 77.5 | 77.5 |
| Llama-3.1-70B | 38.8 | 66.1 | 76.6 | 78.5 | 65.6 |
| Gemini-1.5-Pro | 45.7 | 74.7 | 80.2 | 84.1 | 84.5 |
| GLM-4-9b | 31.6 | 57.3 | 68.3 | 72.2 | 72.9 |
| Phi-3-Mini | 30.3 | 48.1 | 57.3 | 56.8 | 48.7 |

### Task Types and Many-shot ICL Performance

| Task Type | Correlation with Context Length | Trend |
|---------|------------------|------|
| Classification | High positive correlation | Steady improvement |
| Summarization | Moderate positive correlation | Diminishing returns |
| Translation | No clear trend | Inconsistent |
| Mathematical Reasoning | Conditional benefit | Requires CoT + strong models |
| Scientific/Symbolic Reasoning | Inconsistent | Depends on task characteristics |

### Ablation Study

| Analysis | Findings |
|------|------|
| SSL SLR | Classification tasks exhibit SLR >> 1, confirming reliance on retrieval |
| ASL SLR | Mathematics/summarization exhibit SLR ≈ 1, showing no reliance on similar-sample retrieval |
| BM25 vs SentenceTransformer | Both retrievers yield consistent conclusions |

### Key Findings
- Classification tasks perform exceptionally well in SSL, but show a massive gap in ASL.
- State-of-the-art models achieve excellent performance at 64k in SSL, but suffer performance degradation starting at 16k in ASL.
- Small models (e.g., Phi-3-Mini) suffer severe degradation in long-context scenarios.

## Highlights & Insights
- The SLR metric is simple and effective, easily explainable in a single sentence.
- Introducing the dichotomy of retrieval versus understanding into ICL evaluation leads to an elegant framework design.
- The insight that many-shot ICL classification is approximately equivalent to retrieval is highly valuable to the community.

## Limitations & Future Work
- SLR based on BM25 similarity might overlook semantic-level similarity.
- Only open-source/public models were evaluated, lacking performance evaluations of the latest models like GPT-4o/Claude-3.5 on ASL.
- The impact of exemplar ordering on SSL versus ASL was not investigated.

## Related Work & Insights
- **vs LongICLBench (Li et al. 2024)**: LongICLBench only adopts classification tasks, whereas this paper demonstrates that classification primarily tests retrieval rather than global understanding.
- **vs Agarwal et al. (2024)**: They only evaluated Gemini 1.5 Pro, while this work covers 12 models.

## Supplementary Details
- The 12 models evaluated include Llama-3.1, Qwen2, Phi-3, Mistral, GLM-4, Jamba, and Gemini-1.5-Pro.
- Context lengths range from 1k to 128k, with context expanded by adding new exemplars.
- Greedy decoding is used, averaged over three random seeds.
- Mathematical tasks require CoT to benefit from more exemplars.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The SLR metric and the classification methodology of SSL/ASL are novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 12 models x 21 tasks x multiple context lengths.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic and highly informative charts/tables.
- **Value**: ⭐⭐⭐⭐ Holds practical instructional significance for the long-context evaluation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Efficient Many-Shot In-Context Learning with Dynamic Block-Sparse Attention](efficient_many-shot_in-context_learning_with_dynamic_block-sparse_attention.md)
- [\[ACL 2025\] What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs](many_shot_attacks_long_context.md)
- [\[ACL 2025\] Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models](ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)
- [\[ACL 2025\] LongReward: Improving Long-context Large Language Models with AI Feedback](longreward_improving_long-context_large_language_models_with_ai_feedback.md)
- [\[ACL 2025\] FocusLLM: Precise Understanding of Long Context by Dynamic Condensing](focusllm_precise_understanding_of_long_context_by_dynamic_condensing.md)

</div>

<!-- RELATED:END -->
