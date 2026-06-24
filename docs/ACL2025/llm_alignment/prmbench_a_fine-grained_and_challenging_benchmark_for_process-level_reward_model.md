---
title: >-
  [Paper Note] PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models
description: >-
  [ACL 2025][LLM Alignment][Process-level Reward Models] This paper proposes PRMBench, a benchmark consisting of 6,216 carefully designed problems and 83,456 step-level labels, to systematically evaluate the fine-grained error detection capabilities of process-level reward models (PRMs) across three dimensions: Simplicity, Soundness, and Sensitivity. Experiments reveal significant deficiencies in 15 existing PRMs.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Process-level Reward Models"
  - "Fine-grained Evaluation"
  - "Reasoning Error Detection"
  - "Step-level Annotation"
  - "Benchmark"
date: 2026-05-08
content_hash: 4fa7d8f3c06d92fe
---

# PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models

**Conference**: ACL 2025  
**arXiv**: [2501.03124](https://arxiv.org/abs/2501.03124)  
**Code**: [PRMBench](https://github.com/PRMBench/PRMBench)  
**Area**: LLM Alignment/RLHF  
**Keywords**: Process-level Reward Models, Fine-grained Evaluation, Reasoning Error Detection, Step-level Annotation, Benchmark

## TL;DR

This paper proposes PRMBench, a benchmark consisting of 6,216 carefully designed problems and 83,456 step-level labels, to systematically evaluate the fine-grained error detection capabilities of process-level reward models (PRMs) across three dimensions: Simplicity, Soundness, and Sensitivity. Experiments reveal significant deficiencies in 15 existing PRMs.

## Background & Motivation

**Background**: Process-Level Reward Models (PRMs) are critical components for complex reasoning and decision-making tasks, providing feedback signals for each step of the reasoning process. PRMs play an increasingly important role in multi-step reasoning tasks such as mathematical reasoning and code generation, where they are used to guide search (e.g., tree search, beam search) or serve as verifiers to filter candidate solutions.

**Limitations of Prior Work**: Language models are prone to making various types of errors during reasoning, including logical errors, calculation errors, and improper premise usage. PRMs must possess the capability to detect these diverse implicit errors. However, existing benchmarks primarily focus on the binary judgment of "whether a step is correct," **lacking a systematic evaluation of multi-dimensional PRM capabilities**. For instance, can a PRM detect unnecessary redundant steps? Can it identify when a premise is subtly altered? Can it reasonably score reasoning paths that are correct but suboptimal?

**Key Challenge**: While PRMs may perform well on simple tasks (like GSM8K), the types of errors they must handle in more challenging real-world scenarios are far more complex than just "whether this step is right or wrong." Existing evaluations fail to reveal this capability gap.

**Goal**: Construct a multi-dimensional, fine-grained benchmark for evaluating PRMs that can systematically test their performance across different error types and capability dimensions, exposing their weaknesses and guiding future research directions.

**Key Insight**: Decompose the capabilities required of a PRM into three orthogonal dimensions: (1) Simplicity (whether it can detect redundant/unnecessary steps); (2) Soundness (whether it can detect logical, calculation, or premise errors); (3) Sensitivity (whether it is sensitive to minor perturbations).

**Core Idea**: Create specialized test cases for each dimension through a carefully designed problem construction pipeline, with each case containing human-verified step-level labels, thereby achieving a "stress test" for PRMs.

## Method

### Overall Architecture

The construction of PRMBench consists of three phases: (1) Seed problem collection—gathering high-quality problems from mathematical reasoning and logical reasoning domains; (2) Step-level solution generation and error injection—generating step-by-step solutions for each problem and injecting specific errors according to predefined error types; (3) Human audit and annotation—verifying the correctness of the labels for each step by human annotators. The final dataset contains 6,216 problems and 83,456 step-level labels.

### Key Designs

1. **Three-Dimensional Evaluation Framework (Simplicity, Soundness, Sensitivity)**:

    - **Function**: Comprehensively evaluate PRM capabilities across three orthogonal dimensions.
    - **Mechanism**: **Simplicity** evaluates whether the PRM can identify unnecessary steps, including redundant reasoning (superfluous steps that do not affect the conclusion) and circular reasoning. **Soundness** evaluates the core error detection capability of the PRM, covering various types such as logical errors, calculation errors, concept misuses, and premise errors. **Sensitivity** evaluates the PRM's response to minor changes—such as whether the PRM can detect minor modifications made to correct steps, or whether it remains consistent when facing steps with identical semantics but different expressions.
    - **Design Motivation**: Evaluation on a single dimension (e.g., only testing step correctness) misses key capability deficiencies of PRMs. The three-dimensional framework ensures comprehensiveness, with each dimension testing different types of model capabilities.

2. **Fine-grained Error Type Classification System**:

    - **Function**: Define and cover various error types that PRMs need to detect in practical scenarios.
    - **Mechanism**: Under the Soundness dimension, errors are further subdivided into multiple subtypes: (a) arithmetic mistakes; (b) invalid logical transitions; (c) premise modification (subtly altering a previously established condition); (d) concept misuse (applying a formula or definition incorrectly); (e) missing conditions (ignoring edge cases or constraints). Each error type has dedicated test cases.
    - **Design Motivation**: Different error types present different levels of challenge to PRMs. Logical errors may require the PRM to understand the causal structure of the reasoning chain, premise modifications require context tracking, and calculation errors demand basic numerical verification capabilities. The classification system enables diagnosing specific weaknesses of PRMs.

3. **Quality-Assured Construction Pipeline**:

    - **Function**: Ensure dataset quality and label accuracy.
    - **Mechanism**: Employs a three-step "generation-injection-verification" pipeline. First, a strong LLM is used to generate step-by-step solutions. Then, errors are injected into specific steps according to predefined templates (ensuring controllable positions and types). Finally, human annotators verify the labels step by step. Hard edge cases are resolved via majority voting. The final dataset size is 6,216 problems and 83,456 step-level labels.
    - **Design Motivation**: Constructing fully manually from scratch is too expensive, while fully automated generation cannot guarantee quality. This semi-automated pipeline balances scale and quality.

### Evaluation Metrics

Multiple metrics are employed to evaluate PRMs: step-level accuracy, error localization accuracy (whether it can locate the first erroneous step), breakdown scores for each dimension, and overall rankings.

## Key Experimental Results

### Main Results: Performance of 15 Models on PRMBench

| Model Type | Model | Simplicity | Soundness | Sensitivity | Overall |
|----------|------|-----------|-----------|------------|------|
| Closed-source Critic | GPT-4o | Relatively High | Moderate-to-High | Moderate | Top Tier |
| Closed-source Critic | Claude-3.5 | Relatively High | Moderate-to-High | Moderate | High Tier |
| Open-source PRM | Math-Shepherd | Low | Moderate-to-Low | Low | Low Tier |
| Open-source PRM | RLHFlow-PRM | Moderate | Moderate | Low | Medium Tier |
| Open-source Critic | QwQ-32B | Moderate-to-High | Moderate | Moderate | Medium-High Tier |
| Open-source Critic | Llama-3-70B | Moderate | Moderate-to-Low | Low | Medium Tier |
| Open-source PRM | Skywork-PRM | Moderate-to-Low | Moderate-to-Low | Low | Medium-Low Tier |

### Dimensional Breakdown Analysis

| Capability Dimension | Best-performing Model Type | Greatest Challenge | Average Score |
|----------|-------------------|---------|---------|
| Simplicity (Redundancy Detection) | Closed-source Critic models | Circular reasoning identification | Generally low |
| Soundness (Error Detection) | Closed-source Critic models | Premise modification errors | Moderate |
| Sensitivity (Perturbation Sensitivity) | Closed-source Critic models | Minor numerical modifications | Lowest |

### Key Findings

- **Specially trained PRMs underperform**: Most open-source PRMs perform significantly worse on PRMBench compared to general closed-source LLMs used as critics, indicating fundamental issues with existing PRM training strategies.
- **Sensitivity is the largest bottleneck**: Almost all models achieve their lowest scores on the sensitivity dimension, indicating difficulties in detecting trace modifications in reasoning steps.
- **Simplicity is neglected**: PRMs are generally poor at detecting redundant steps, prone to giving positive scores to any step that "looks correct," even if it is entirely superfluous.
- **Large variance across error types**: Models detect calculation errors significantly better than logical reasoning errors and premise modification errors.
- **Scale is not the decisive factor**: Some large-scale PRMs underperform compared to smaller models on specific dimensions, showing that training data and strategies matter more than parameter count.

## Highlights & Insights

- **Systematic nature of the three-dimensional evaluation framework**: Decomposing PRM capability into Simplicity, Soundness, and Sensitivity is a clear and inspiring framework. This decomposition is applicable not only to PRM evaluation but can also be extended to evaluate other "verifier"-class models.
- **A warning for the PRM training paradigm**: Experiments reveal a counterintuitive conclusion—general LLMs via simple critic prompts can outperform specially trained PRMs. This implies systematic biases in current PRM training data and objective functions (e.g., overfitting to binary "step correct/incorrect" labels while ignoring redundancy and sensitivity).
- **Diagnostic value of fine-grained error classification**: The ability to precisely pinpoint "which error types PRMs are weakest on" provides direct guidance for improving PRM training schemes.

## Limitations & Future Work

- Primarily focuses on the mathematical reasoning domain; evaluation of PRMs in other reasoning domains like code generation and logical reasoning remains to be expanded.
- Error injection is mainly based on predefined templates, which may not fully cover all error patterns that occur in real-world reasoning.
- Although 15 models were evaluated, the selection of open-source PRMs may not be fully comprehensive, as some very recent PRMs (such as those based on Qwen2.5-Math) were not included.
- The end-to-end performance of PRMs in actual search/verification pipelines has not been explored; hence, the relationship between benchmark scores and real-world application performance is yet to be established.
- Dynamic evaluation could be considered—using the actual effectiveness of PRMs in guiding search as a complementary evaluation dimension.

## Related Work & Insights

- **vs PRM800K / Math-Shepherd**: While these works provide training data for PRMs, they do not systematically evaluate the capabilities of the trained models. PRMBench fills this evaluation gap and demonstrates that models trained on PRM800K underperform across multiple dimensions.
- **vs ProcessBench (2412.06559)**: ProcessBench similarly evaluates error detection in reasoning processes, but focuses more on "finding the first erroneous step." PRMBench adds the Simplicity and Sensitivity dimensions to provide a more comprehensive evaluation.
- **vs ORM (Outcome Reward Model) in RLHF**: ORMs evaluate only the final outcome, whereas PRMs evaluate each step. The findings from PRMBench indicate that current step-level evaluation capabilities of PRMs are still inadequate, explaining why ORMs can sometimes be more effective in practice.

## Rating

- Novelty: ⭐⭐⭐⭐ The three-dimensional evaluation framework and fine-grained error classification are significant contributions, though the core construction pipeline of the benchmark itself is not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 15 models, multiple dimensions, 6K+ problems, 83K+ labels. The evaluation scale and depth are highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The framework definition is clear and the experimental design is logical, though some details could be more concise.
- Value: ⭐⭐⭐⭐⭐ Direct and significant driving effect on the PRM research community, revealing critical capability gaps.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning](aspo_adaptive_sentence-level_preference_optimization_for_fine-grained_multimodal.md)
- [\[ACL 2025\] Intuitive Fine-Tuning: Towards Simplifying Alignment into a Single Process](intuitive_fine_tuning_simplifying_alignment_into_single_process.md)
- [\[ACL 2025\] Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization](fine-grained_video_dubbing_duration_alignment_with_segment_supervised_preference.md)
- [\[ACL 2025\] T-REG: Preference Optimization with Token-Level Reward Regularization](t-reg_preference_optimization_with_token-level_reward_regularization.md)
- [\[ACL 2026\] Aligning Agents via Planning: A Benchmark for Trajectory-Level Reward Modeling](../../ACL2026/llm_alignment/aligning_agents_via_planning_a_benchmark_for_trajectory-level_reward_modeling.md)

</div>

<!-- RELATED:END -->
