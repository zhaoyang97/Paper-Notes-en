---
title: >-
  [Paper Note] Program Synthesis Benchmark for Visual Programming in XLogoOnline Environment
description: >-
  [ACL 2025][Code Intelligence][Program Synthesis] This work builds a program synthesis benchmark based on the XLogoOnline visual programming environment, requiring a combination of multiple skills such as spatial planning, programming, and logical reasoning. The evaluation shows that GPT-4V only solves 20% of the tasks. However, through fine-tuning on 80k+ synthetic data combined with simulator-driven curriculum learning, Llama3-8B significantly outperforms both GPT-4V and Lla…
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "Program Synthesis"
  - "Visual Programming"
  - "Benchmark"
  - "Curriculum Learning"
  - "Simulator Feedback"
date: 2026-05-08
content_hash: 08ee7092379d80bb
---

# Program Synthesis Benchmark for Visual Programming in XLogoOnline Environment

**Conference**: ACL 2025  
**arXiv**: [2406.11334](https://arxiv.org/abs/2406.11334)  
**Code**: None (to be released)  
**Area**: Text Generation / Program Synthesis  
**Keywords**: Program Synthesis, Visual Programming, Benchmark, Curriculum Learning, Simulator Feedback

## TL;DR

This work builds a program synthesis benchmark based on the XLogoOnline visual programming environment, requiring a combination of multiple skills such as spatial planning, programming, and logical reasoning. The evaluation shows that GPT-4V only solves 20% of the tasks. However, through fine-tuning on 80k+ synthetic data combined with simulator-driven curriculum learning, Llama3-8B significantly outperforms both GPT-4V and Llama3-70B.

## Background & Motivation

**Background**: Large language models and multimodal models have achieved remarkable success on single-skill benchmarks such as code generation, mathematical reasoning, and visual question answering. However, few studies evaluate these models on complex tasks that require a combination of multiple skills.

**Limitations of Prior Work**: Existing code generation benchmarks (e.g., HumanEval, MBPP) primarily test general programming capabilities, mathematics benchmarks test mathematical reasoning, and vision benchmarks test image understanding—each in isolation. However, many real-world programming tasks require the simultaneous application of multiple abilities, such as understanding visual targets, conducting spatial planning, and writing structured code. There is currently a lack of a benchmark that systematically evaluates the capability of "multi-skill combination."

**Key Challenge**: High performance of models on single skills does not imply equal competence in multi-skill combination tasks. There is a huge gap between the simple collection of individual capabilities and actual multi-skill integration.

**Goal**: (1) To construct a program synthesis benchmark requiring a multi-skill combination of spatial planning + programming + logical reasoning; (2) to evaluate the capability limits of current SOTA models; (3) to explore model performance improvement on such tasks via fine-tuning and curriculum learning.

**Key Insight**: The authors select XLogoOnline—a real visual programming environment used for computational thinking education. Each task requires students to control a turtle using the Logo language to draw a target shape, naturally demanding cooperation among multiple skills.

**Core Idea**: Construct a large-scale program synthesis benchmark using XLogoOnline, and design a curriculum learning strategy integrated with simulator feedback to enable small models to outperform large models on multi-skill tasks through targeted fine-tuning.

## Method

### Overall Architecture

The entire work is divided into three parts: (1) Benchmark Construction—collecting real tasks from the XLogoOnline platform to build a standardized evaluation set; (2) Synthetic Data Generation—programmatically generating large-scale training data (80,000+ tasks) using a simulator; (3) Fine-tuning and Curriculum Learning—first fine-tuning the model on synthetic data, then designing a difficulty-adaptive training curriculum using simulator feedback.

### Key Designs

1. **XLogoOnline Benchmark Construction**:

    - **Function**: To provide a standardized multi-skill program synthesis evaluation platform.
    - **Mechanism**: Collect programming tasks across multiple difficulty levels from the XLogoOnline education platform. Each task consists of an initial canvas state and a target image, requiring the model to generate a Logo program that guides the turtle to draw the target image. The tasks range from simple straight-line drawing to complex nested loops and conditional branches. The evaluation metric uses functional correctness—executing the generated code in the simulator and performing a pixel-level comparison between the output image and the target image.
    - **Design Motivation**: Unlike text-described programming tasks, visual programming tasks require the model to simultaneously "understand" the target image (visual understanding), "plan" the execution path (spatial planning), and "correctly write" the program (code generation). This serves as a natural test for multi-skill combinations.

2. **Large-scale Synthetic Data Generation Pipeline**:

    - **Function**: Automatically generate over 80,000 training tasks with correct solutions.
    - **Mechanism**: Utilize the XLogoOnline simulator to programmatically generate diverse training data. Specifically, Logo programs are randomly generated first, and then executed by the simulator to yield corresponding target images, thereby obtaining (target image, correct program) training pairs. By controlling the program complexity (loop depth, number of instructions, etc.), data coverage across different difficulty levels is ensured.
    - **Design Motivation**: The number of tasks from the real platform is limited and insufficient for model fine-tuning. Synthetic data can be expanded infinitely while guaranteeing annotation quality (as the program itself serves as the ground truth).

3. **Simulator-driven Curriculum Learning**:

    - **Function**: Dynamically adjust the training data distribution based on the model's current capability to achieve difficulty-adaptive training.
    - **Mechanism**: During the fine-tuning process, periodically evaluate the model's pass rate on tasks of different difficulty levels using the simulator. For simple tasks that the model can already easily solve, reduce their proportion in the training set; for medium-difficulty tasks where the model is on the "frontier of learning," increase their proportion. This forms an adaptive curriculum from easy to difficult. In practice, tasks are grouped into difficulty buckets, and sampling weights are adjusted based on the model's pass rate in each bucket.
    - **Design Motivation**: Uniform sampling of training data wastes substantial computation on tasks that the model has already mastered or cannot understand at all. The curriculum learning strategy ensures that the model is always trained within its "optimal zone of learning," improving learning efficiency.

### Loss & Training

Fine-tuning uses the standard causal language model loss function (cross-entropy) for supervised fine-tuning on synthetic data. The input consists of the task description (including a textual representation of the target image), and the output is the corresponding Logo program. Curriculum learning is implemented by dynamically adjusting the sampling probabilities of different difficulty buckets, without modifying the loss function itself.

## Key Experimental Results

### Main Results

| Model | Success Rate (%) | Parameters | Type |
|------|-----------|--------|------|
| GPT-4V | 20.0 | - | Closed-source Multimodal |
| GPT-4 (text) | 18.5 | - | Closed-source Text |
| Llama3-70B | 2.35 | 70B | Open-source Text |
| Llama3-8B (zero-shot) | 0.6 | 8B | Open-source Text |
| Llama3-8B (fine-tuned) | 43.2 | 8B | Fine-tuned |
| Llama3-8B (fine-tuned + curriculum) | **52.8** | 8B | Fine-tuned + Curriculum |

### Ablation Study

| Configuration | Success Rate (%) | Description |
|------|-----------|------|
| Fine-tuned + Curriculum | 52.8 | Full Method |
| Fine-tuned (Uniform Sampling) | 43.2 | Decreases by 9.6% without curriculum learning |
| Fine-tuned (Small Data 10K) | 28.5 | Insufficient training data |
| Fine-tuned (No Simulator Verification) | 38.7 | No simulator feedback used to select the curriculum |
| Zero-shot GPT-4V | 20.0 | Strongest zero-shot baseline |

### Key Findings

- The strongest current multimodal model, GPT-4V, achieves only a 20% success rate on this benchmark, indicating that multi-skill combination tasks remain highly challenging for existing models.
- The 2.35% success rate of Llama3-70B indicates that increasing model scale does not automatically yield multi-skill combination capabilities.
- After fine-tuning on synthetic data, Llama3-8B significantly outperforms both the 70B version and GPT-4V, highlighting the importance of domain-specific training data.
- Curriculum learning brings an additional absolute gain of 9.6%, demonstrating the value of simulator feedback during training.
- Failure analysis shows that spatial planning is the weakest link for the models—they frequently generate syntactically correct programs but exhibit erroneous path planning.

## Highlights & Insights

- **Constructing a validation loop using a simulator is highly ingenious**: The simulator is used for generating training data (ensuring quality), evaluation (functional correctness), and designing curriculum learning all at once—a single component playing a triple role. This paradigm can be transferred to any domain with a deterministic verifier.
- **Small model + good data + good strategy > large model**: The result where the 8B fine-tuned model outperforms Llama3-70B and GPT-4V strongly supports the argument that "data quality and training strategies are more critical than model scale."
- **The design concept of the multi-skill benchmark is highly referenceable**: Instead of artificially splicing multiple sub-tasks, a real-world scenario (an educational platform) requiring natural cooperation among multiple skills was selected, making the evaluation closer to reality.

## Limitations & Future Work

- The benchmark is limited to the specific domain of the Logo language; transferability to general programming remains to be validated.
- The evaluation adopts exact pixel-level matching, which might be overly strict for solutions that are visually correct but have minor pixel offsets.
- The curriculum learning strategy requires frequent simulator invocations for evaluation, which increases training costs.
- Future work could extend to more complex visual programming environments (e.g., Scratch, Blockly) or introduce natural language descriptions to replace visual targets.
- The paper does not explore the effectiveness of reasoning enhancement methods such as chain-of-thought or tree-of-thought.

## Related Work & Insights

- **vs HumanEval/MBPP**: Traditional code generation benchmarks only test programming capability without involving visual understanding and spatial planning. This benchmark is more comprehensive but narrower in domain scope.
- **vs MATH/GSM8K**: Mathematics benchmarks test reasoning capacity but do not involve code generation. The XLogoOnline task requires transforming reasoning outcomes into executable programs.
- **vs CodeContests**: Competitive programming also demands multiple abilities, but this work achieves a closed-loop of automated evaluation and data generation through the simulator.

## Rating

- Novelty: ⭐⭐⭐⭐ The first multi-skill program synthesis benchmark for visual programming; the simulator-driven curriculum learning approach is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers zero-shot and fine-tuning evaluations of multiple models, with a complete ablation study and detailed failure analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear task descriptions and systematic benchmark design.
- Value: ⭐⭐⭐⭐ Exposes the challenge that multi-skill combination tasks pose to existing models; the curriculum learning strategy is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Program Synthesis via Test-Time Transduction](../../NeurIPS2025/code_intelligence/program_synthesis_via_test-time_transduction.md)
- [\[NeurIPS 2025\] Once Upon an Input: Reasoning via Per-Instance Program Synthesis](../../NeurIPS2025/code_intelligence/once_upon_an_input_reasoning_via_per-instance_program_synthesis.md)
- [\[ACL 2025\] TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs](texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)
- [\[ICCV 2025\] TikZero: Zero-Shot Text-Guided Graphics Program Synthesis](../../ICCV2025/code_intelligence/tikzero_zero-shot_text-guided_graphics_program_synthesis.md)
- [\[ACL 2025\] FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation](feabench_repo_code_gen.md)

</div>

<!-- RELATED:END -->
