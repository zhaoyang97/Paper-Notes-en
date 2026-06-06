---
title: >-
  [Paper Note] HeurekaBench: A Benchmarking Framework for AI Co-scientist
description: >-
  [ICLR 2026][LLM Reasoning][AI co-scientist] This paper proposes HeurekaBench, a framework for constructing evaluation benchmarks grounded in real scientific workflows. It employs a multi-LLM pipeline to extract verifiabl…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "AI co-scientist"
  - "benchmark"
  - "scientific agents"
  - "single-cell biology"
  - "open-ended evaluation"
date: 2026-05-08
content_hash: 588550410881c0c1
---

# HeurekaBench: A Benchmarking Framework for AI Co-scientist

**Conference**: ICLR 2026
**arXiv**: [2601.01678](https://arxiv.org/abs/2601.01678)  
**Code**: [brbiclab.epfl.ch/projects/heurekabench](https://brbiclab.epfl.ch/projects/heurekabench)  
**Area**: LLM Reasoning
**Keywords**: AI co-scientist, benchmark, scientific agents, single-cell biology, open-ended evaluation

## TL;DR
This paper proposes HeurekaBench, a framework for constructing evaluation benchmarks grounded in real scientific workflows. It employs a multi-LLM pipeline to extract verifiable scientific insights from papers and generate open-ended research questions, enabling end-to-end assessment of AI co-scientists in data-driven scientific discovery.

## Background & Motivation
Advances in LLM reasoning have given rise to a wide range of scientific agents (e.g., CellVoyager, Biomni) designed to autonomously analyze experimental data and generate scientific insights. However, existing benchmarks suffer from fundamental limitations: most evaluate only static knowledge retrieval or single-step computational tasks (e.g., "How many miRNAs are significant at p≤0.05?"). Such instruction-following tasks fall far short of the true co-scientist role, which requires autonomous planning of analytical workflows, dataset exploration, and novel discovery. While BaisBench attempts to generate research questions, it relies on a single LLM, leading to unreliable question quality. The root cause is that existing benchmarks cannot assess open-ended, data-driven scientific discovery capabilities. This paper's starting point is to ground benchmark construction in the scientific process itself — extracting validated insights from peer-reviewed papers as ground truth for evaluation.

## Method

### Overall Architecture
HeurekaBench consists of three stages: (a) Insight Generation — extracting candidate insights from papers and semi-automatically validating them; (b) Question Generation — converting validated insights into QA pairs; (c) Question Solving — agents autonomously design multi-step analyses and produce answers, which are scored by an LLM Judge against ground truth.

### Key Designs

1. **Insight Generation Pipeline**:

    - Function: Extract reproducible scientific insights from scientific papers and their code repositories.
    - Mechanism: Four modular LLM components are designed — InsightExtractor extracts candidate insights from papers (with three structured components: abstract, experimental technique, and verbatim evidence); CodeDescriber converts code scripts into natural language descriptions; CodeMatcher pairs each insight with the most relevant code description; CodeGenerator combines scripts to produce multi-step verification workflows.
    - Design Motivation: Filtering unreliable insights via code reproducibility is more robust than relying solely on LLM-generated questions (as in BaisBench). GPT-4o is used for InsightExtractor and Claude-4-Sonnet for code-related modules.

2. **Question Generation**:

    - Function: Convert validated insights into open-ended questions (OEQs) and multiple-choice questions (MCQs).
    - Mechanism: For each insight, few-shot prompting is used to generate two QA pairs. OEQs allow multiple analytical paths to the correct answer; MCQs include high-quality distractors. Generated questions undergo two-stage filtering: (1) automatic filtering — removing questions answerable from LLM pretraining knowledge alone; (2) human review — eliminating hallucinations, duplicates, and questions based on unverified content.
    - Design Motivation: OEQs reflect the open-ended nature of real research; MCQs serve as lightweight proxies for rapid agent prototyping.

3. **Evaluation (G-Eval with Atomic Facts)**:

    - Function: GPT-4o serves as an LLM Judge to evaluate open-ended responses on a 1–5 scale.
    - Mechanism: The Judge is guided to decompose both the response and ground truth into atomic facts (conditions, trends, conclusions), then compare them item by item for completeness, partial matches, and omissions. Full marks are awarded only when all ground truth facts are present without contradiction; additional non-contradictory findings are not penalized.
    - Design Motivation: Avoid surface-level matching; reward data-driven outputs rather than factual memorization.

### Validation
The framework is instantiated in single-cell biology as sc-HeurekaBench: 41 validated insights from 22 Nature/Cell papers across 13 papers, yielding 50 OEQs and 50 MCQs. InsightExtractor achieved strong relevance on FlyBase in 44/50 cases; CodeMatcher achieved an average file correct-match rate of 74.6%.

## Key Experimental Results

### Main Results

| Agent | OEQ Correctness [1–5] | MCQ Accuracy (%) | MCQ Recall (%) | MCQ Precision (%) |
|-------|----------------------|-----------------|---------------|------------------|
| BixBench-Agent | 2.34 | 44.44 | 80.56 | 62.96 |
| CellVoyager | 2.03 | 27.78 | 38.89 | 32.41 |
| Biomni | 2.31 | **50.00** | **88.24** | **76.96** |

### Ablation Study on Planner (Biomni Agent)

| Model | Open-source | OEQ Correctness | MCQ Accuracy (%) |
|-------|-------------|----------------|-----------------|
| MedGemma-27B | ✓ | 1.53 | 20.41 |
| Qwen3-32B | ✓ | 1.47 | 40.00 |
| Qwen3-235B-thinking | ✓ | 1.85 | 46.00 |
| GPT-OSS-120B | ✓ | 2.08 | 42.00 |
| Claude-4-Sonnet | ✗ | **2.58** | 44.00 |

### Key Findings
- Biomni and BixBench-Agent outperform CellVoyager, indicating that flexible agent loops are more effective at constructing robust workflows.
- Claude-4-Sonnet as planner significantly outperforms other models (2.58 vs. 2.08), demonstrating that closed-source frontier models retain a clear advantage on co-scientist tasks.
- An end-critic (a critic introduced at the end of the agent loop) substantially improves open-source LLM performance, raising scores in the low-scoring group (scores 1–2) from 1.32 to 1.91.
- Model scale and reasoning capability (thinking mode) are critical to co-scientist performance.

## Highlights & Insights
- Grounding benchmark construction in the scientific process itself is a compelling approach — using reproducibility of paper results as the validation criterion for insights.
- The modular multi-LLM pipeline design makes the framework transferable to other scientific domains.
- The end-critic design can close the gap between open-source and closed-source models by up to 22%, serving as a lightweight yet effective improvement strategy.

## Limitations & Future Work
- The framework is currently instantiated only in single-cell biology; generalization to chemistry, physics, and other domains requires additional validation.
- sc-HeurekaBench is relatively small in scale (50 OEQs + 50 MCQs), which may be insufficient for fine-grained capability diagnostics.
- The validation process still requires substantial human involvement (running code, verifying results), leaving room for improved automation.

## Related Work & Insights
- **vs. BaisBench**: BaisBench generates questions using a single LLM without verification; HeurekaBench ensures reliability through a multi-LLM pipeline with code-based validation.
- **vs. BixBench**: BixBench primarily tests computational questions, whereas HeurekaBench evaluates open-ended scientific exploration.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel framework concept, though primarily a benchmark/systems contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional ablations are highly detailed, but dataset scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure and polished figures.
- Value: ⭐⭐⭐⭐ Provides an important evaluation framework for the AI for Science community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CoAct: Co-Active LLM Preference Learning with Human-AI Synergy](../../ACL2026/llm_reasoning/coact_co-active_llm_preference_learning_with_human-ai_synergy.md)
- [\[ICLR 2026\] GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs](geogrambench_benchmarking_the_geometric_program_reasoning_in_modern_llms.md)
- [\[ICLR 2026\] TopoBench: Benchmarking LLMs on Hard Topological Reasoning](topobench_benchmarking_llms_on_hard_topological_reasoning.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_reasoning_interv.md)
- [\[ICML 2026\] CoCoReviewBench: A Completeness- and Correctness-Oriented Benchmark for AI Reviewers](../../ICML2026/llm_reasoning/cocoreviewbench_a_completeness-_and_correctness-oriented_benchmark_for_ai_review.md)

</div>

<!-- RELATED:END -->
