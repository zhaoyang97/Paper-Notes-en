---
title: >-
  [Paper Note] YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering
description: >-
  [ACL 2025][LLM Evaluation][LLM-as-a-Judge] This work proposes the YESciEval framework, which combines a nine-dimensional fine-grained evaluation rubric and SFT+RL alignment strategies to mitigate the optimism bias of LLM judges. It builds a robust open-source LLM-as-a-Judge system for scientific question answering without requiring human annotations or closed-source models.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "LLM-as-a-Judge"
  - "Scientific Question Answering Evaluation"
  - "Optimism Bias"
  - "Adversarial Testing"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: 277b7839c3fba94c
---

# YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering

**Conference**: ACL 2025  
**arXiv**: [2505.14279](https://arxiv.org/abs/2505.14279)  
**Code**: [https://github.com/sciknoworg/YESciEval](https://github.com/sciknoworg/YESciEval)  
**Area**: LLM Evaluation  
**Keywords**: LLM-as-a-Judge, Scientific Question Answering Evaluation, Optimism Bias, Adversarial Testing, Reinforcement Learning  

## TL;DR
This work proposes the YESciEval framework, which combines a nine-dimensional fine-grained evaluation rubric and SFT+RL alignment strategies to mitigate the optimism bias of LLM judges. It builds a robust open-source LLM-as-a-Judge system for scientific question answering without requiring human annotations or closed-source models.

## Background & Motivation
**Background**: Scientific search engines (e.g., Elicit, ORKG Ask) increasingly rely on LLMs for scientific question answering (ScienceQA), but systematic evaluation of the generated answers' quality is still lacking.

**Limitations of Prior Work**: (a) n-gram metrics (e.g., BLEU/ROUGE) fail to capture domain-specific reasoning quality; (b) human evaluation is too costly to scale; (c) LLM-as-a-Judge suffers from a severe optimism bias, leaning towards high scores rather than critical evaluation.

**Key Challenge**: Reliable automatic evaluation is needed to support the iterative optimization of scientific QA, yet existing LLM evaluators exhibit unexpected vulnerability when facing heuristic adversarial attacks.

**Goal**: (a) How to define comprehensive evaluation dimensions for scientific QA? (b) How to mitigate the optimism bias of LLM judges? (c) How to construct a reliable evaluation system with zero human annotation cost?

**Key Insight**: Design a nine-dimensional evaluation rubric, construct a two-level (subtle/extreme) adversarial dataset as bias detection signals, and align evaluation behaviors using Contrastive Preference Optimization (CPO).

**Core Idea**: Utilize adversarial examples to expose the optimism bias of LLM judges, then leverage SFT+RL to enable an 8B open-source model to learn critical evaluation of scientific answers.

## Method

### Overall Architecture
A two-stage pipeline: (1) LLM generation (LLMgen) - four LLMs (Llama 8B/70B, Qwen 72B, Mistral 128B) synthesize answers from paper abstracts to form a benign dataset; (2) LLM evaluation (LLMeval) - scoring across nine dimensions (1-5 Likert scale), constructing adversarial variants (subtle/extreme), and aligning LLaMA 3.1 8B with SFT+RL to become a robust judge. A total of 48 evaluation configurations are established (4 generators x 4 evaluators x 3 data variants).

### Key Designs

1. **Nine-Dimensional Evaluation Rubric System**:

    - **Function**: Defines the complete dimensional space for scientific QA evaluation.
    - **Mechanism**: Nine dimensions across three main categories—linguistic and stylistic quality (Cohesion, Conciseness, Readability), logical and structural integrity (Coherence, Integration, Relevancy), and content accuracy and informativeness (Correctness, Completeness, Informativeness). Each dimension is equipped with a standardized 1-5 scoring rubric.
    - **Design Motivation**: Existing evaluation methods (e.g., G-Eval, FLASK) only cover partial dimensions and lack consistent definitions, necessitating a unified and comprehensive framework.

2. **Two-Level Adversarial Dataset Construction**:

    - **Function**: Designs specific heuristic text perturbations for each evaluation dimension to detect judge bias.
    - **Mechanism**: Generates subtle and extreme adversarial variants for each benign answer. Each dimension has a corresponding perturbation strategy—for example, the subtle version of Relevancy appends related domain sentences, while the extreme version injects irrelevant sports news; the subtle version of Cohesion swaps the last two sentences, while the extreme version randomly shuffles all sentences; the subtle version of Conciseness appends an LLM-generated redundant phrase at the end, while the extreme version inserts redundancy after every sentence.
    - **Design Motivation**: If the LLM evaluator cannot distinguish the quality difference between benign and adversarial samples (i.e., failing to lower the score), its evaluation is considered unreliable. Adversarial testing indirectly measures evaluation reliability without requiring human annotations.

3. **SFT + CPO Alignment**:

    - **Function**: Trains LLaMA 3.1 8B to be a robust LLM-as-a-Judge.
    - **Mechanism**: Two-step alignment—(i) SFT: QLoRA fine-tuning on benign evaluation data generated by the four LLMs to learn basic evaluation formats and dimensional understanding; (ii) RL: constructing preference pairs $(x, y_{good}, y_{bad})$ and aligning with Contrastive Preference Optimization (CPO). The CPO loss is formulated as: $\min_\theta \mathcal{L}_{prefer} - \mathbb{E}_{(x,y_{good})\sim D}[\log \pi_\theta(y_{good}|x)]$, where $\mathcal{L}_{prefer}$ represents the preference alignment term, and $\mathcal{L}_{NLL}$ penalizes the generation of low-quality evaluations.
    - **Design Motivation**: SFT alone only teaches mimicking and fails to learn "what a poor evaluation is." By using cases where high scores were incorrectly assigned to adversarial samples as $y_{bad}$, the model learns to provide critical evaluations.

### Loss & Training
QLoRA (efficient fine-tuning) is used in the SFT stage, and CPO (an extension of DPO) is used for preference alignment in the RL stage. The entire process requires zero human annotation cost—benign evaluations are generated via multi-LLM peer evaluation, and good/bad labels in adversarial evaluations are determined by rules (extreme variants should receive 1 point, and subtle variants should receive $\le 3$ points).

## Key Experimental Results

### Main Results
Average scores of benign data evaluated across the four LLMs (average over 9 dimensions, 1-5 scale):

| LLMeval \ LLMgen | Llama-8B | Llama-70B | Qwen-72B | Mistral-128B |
|------------------|----------|-----------|----------|-------------|
| Llama-8B | ~4.2 | ~4.3 | ~4.5 | ~4.4 |
| Llama-70B | ~4.1 | ~4.2 | ~4.4 | ~4.3 |
| Qwen-72B | ~4.0 | ~4.1 | ~4.3 | ~4.2 |
| Mistral-128B | ~4.1 | ~4.2 | ~4.4 | ~4.3 |

### Ablation Study
The performance of LLaMA 3.1 8B with different training strategies on adversarial samples:

| Configuration | Benign Score (Normal) | Extreme Adversarial (Should be low) | Subtle Adversarial (Should decrease moderately) |
|------|--------------|--------------------|--------------------|
| Vanilla (No Training) | High | Still High (Optimism Bias) | Almost Unchanged |
| + SFT only | High | Slightly Decreased | Almost Unchanged |
| + SFT + RL (CPO) | High | **Significantly Decreased** | **Moderately Decreased** |

### Key Findings
- **No Self-Preference**: LLM judges do not exhibit self-preference bias towards their own generated answers. Instead, all evaluators consistently prefer answers generated by Qwen (likely because Qwen is the largest model with the highest quality of generation).
- **Pervasive Optimism Bias**: Unaligned LLM evaluators still assign high scores to extreme adversarial samples, demonstrating a lack of critical evaluation capability.
- **CPO Effectively Mitigates Bias**: The 8B model after SFT+RL can correctly differentiate between benign and adversarial samples, providing more reasonable lower scores in adversarial testing.
- **BioASQ Scores Higher than ORKGSyn**: Interdisciplinary datasets present more challenges to LLM generation than specialized biomedical datasets.
- **Zero-Cost Feasibility**: The entire framework does not rely on closed-source models or human annotation, saving over €1000 compared to using GPT.
- **Precision of Adversarial Perturbation Design Per Dimension**: The perturbation strategies differ significantly across dimensions, showing that fine-grained evaluation requires precise test design.

## Highlights & Insights
- **Adversarial Testing as a Proxy for Evaluation Reliability**: Using "texts that deserve score deductions" to test whether the evaluator actually penalizes them is an ingenious indirect verification method that does not require human ground truth.
- **Reusability of the Nine-Dimensional Rubrics**: This set of evaluation rubrics is not only suitable for scientific QA but can also be extended to any generative AI evaluation scenario.
- **Cost Advantage of Pure Open-Source**: It proves that open-source LLMs coupled with smart training strategies can replace expensive GPT-4 evaluations.

## Limitations & Future Work
- **Adversarial Perturbations are Heuristically Designed**: They may not cover all types of quality issues in real-world scenarios.
- **Alignment Experiments Only Conducted on LLaMA 3.1 8B**: The alignment performance on larger models has not been verified.
- **Lack of Human Evaluation Validation**: The consistency between aligned evaluators and human judgment has not been directly measured.
- **Limited Dataset Scale**: ORKGSyn contains only 348 questions, and BioASQ contains only 73.
- **Future Directions**: Integrate human evaluation for comparative validation; scale to more scientific domains and more LLMs.

## Related Work & Insights
- **vs G-Eval**: G-Eval employs GPT-4 for evaluation but relies on closed-source models and only covers 4 dimensions; YESciEval is fully open-source, features 9 dimensions, and possesses adversarial robustness.
- **vs FLASK**: FLASK defines 12 evaluation skills but does not focus on scientific domains and lacks adversarial testing; YESciEval's adversarial testing and RL alignment act as key innovations.
- **vs JudgeLM**: JudgeLM trains judges using human preference data but is expensive; YESciEval replaces human preferences with rule-based labels from adversarial examples to achieve zero annotation cost.

## Rating
- Novelty: ⭐⭐⭐⭐ Combining adversarial testing and RL to mitigate LLM judge optimism bias is a novel approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ 48 evaluation configurations + two datasets + ablation analysis.
- Writing Quality: ⭐⭐⭐ Content-rich but somewhat verbose.
- Value: ⭐⭐⭐⭐ Significant contribution to open-source LLM-as-a-Judge and scientific QA evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Doubly-Robust LLM-as-a-Judge: Externally Valid Estimation with Imperfect Personas](../../ICLR2026/llm_evaluation/doubly-robust_llm-as-a-judge_externally_valid_estimation_with_imperfect_personas.md)
- [\[ACL 2025\] Can External Validation Tools Improve Annotation Quality for LLM-as-a-Judge?](can_external_validation_tools_improve_annotation_quality_for_llm-as-a-judge.md)
- [\[ICML 2026\] Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge](../../ICML2026/llm_evaluation/reasoning_is_not_free_robust_adaptive_cost-efficient_routing_for_llm-as-a-judge.md)
- [\[NeurIPS 2025\] DSAS: A Universal Plug-and-Play Framework for Attention Optimization in Multi-Document Question Answering](../../NeurIPS2025/llm_evaluation/dsas_a_universal_plug-and-play_framework_for_attention_optimization_in_multi-doc.md)
- [\[ACL 2025\] MisMatched: A Benchmark for Scientific Natural Language Inference](a_mismatched_benchmark_for_scientific_natural_language_inference.md)

</div>

<!-- RELATED:END -->
