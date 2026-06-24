---
title: >-
  [Paper Note] AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment
description: >-
  [ACL 2025][Optimization][Ambiguity Detection] Proposes AmbiK, a text-only dataset dedicated to detecting ambiguous instructions in kitchen environments. It contains 1,000 pairs of ambiguous/unambiguous instructions categorized by three ambiguity types (user preference, common sense, and safety). Multiple conformal prediction-based ambiguity detection methods are evaluated, revealing that existing methods perform poorly on this benchmark.
tags:
  - "ACL 2025"
  - "Optimization"
  - "Ambiguity Detection"
  - "Robot Task Planning"
  - "Kitchen Environment"
  - "Conformal Prediction"
  - "Benchmark"
date: 2026-05-08
content_hash: 26553c993ba7ee21
---

# AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment

**Conference**: ACL 2025  
**arXiv**: [2506.04089](https://arxiv.org/abs/2506.04089)  
**Code**: [https://github.com/cog-model/AmbiK-dataset](https://github.com/cog-model/AmbiK-dataset)  
**Area**: Others  
**Keywords**: Ambiguity Detection, Robot Task Planning, Kitchen Environment, Conformal Prediction, Benchmark

## TL;DR
Proposes AmbiK, a text-only dataset dedicated to detecting ambiguous instructions in kitchen environments. It contains 1,000 pairs of ambiguous/unambiguous instructions categorized by three ambiguity types (user preference, common sense, and safety). Multiple conformal prediction-based ambiguity detection methods are evaluated, revealing that existing methods perform poorly on this benchmark.

## Background & Motivation
**Background**: LLMs are widely used in embodied AI for behavioral planning based on natural language instructions. However, real-world instructions are often ambiguous (e.g., "put the cup on the table" is ambiguous when multiple cups are present), requiring robots to determine when to ask the user for clarification.

**Limitations of Prior Work**: Existing datasets like DialFred and TEACh contain some ambiguous tasks but lack dedicated annotations. Although the KnowNo dataset is designed for ambiguity detection, its tasks are too simple (single-step instructions, few objects) and it lacks consistent categorization. These datasets cannot serve as a unified, text-only benchmark for fair comparison among different methods.

**Key Challenge**: The lack of a high-quality, fully annotated, dedicated benchmark supporting multiple ambiguity types makes it difficult to conduct fair comparisons between different ambiguity detection methods.

**Goal**: To build a text-only, ambiguous instruction benchmark tailored for kitchen scenarios, enabling unified evaluation of LLMs' ambiguity detection capabilities.

**Key Insight**: Categorize ambiguity based on "the type of knowledge required to resolve the ambiguity" (preference, common sense, safety) rather than linguistic classifications, as different types correspond to different action strategies for robots.

**Core Idea**: Construct a pairwise dataset of ambiguous and unambiguous instructions categorized by "how the robot should respond," allowing ambiguity detection methods to be compared fairly on a unified benchmark.

## Method

### Overall Architecture
AmbiK is a text-only dataset. Each data entry contains: environment description (list of objects), an ambiguous instruction, a corresponding unambiguous instruction, an ambiguity type label, clarification question-answer pairs, user intent, and a task plan.

### Key Designs

1. **Ambiguity Categorization (Based on Action Strategy)**:

    - **Function**: Categorizes ambiguity into three types: human preferences (42%), common-sense knowledge (42.5%), and safety (15.5%).
    - **Mechanism**: For preference-based ambiguities (e.g., choosing which cup), the robot should **always** ask; for common-sense ambiguities (e.g., what to toast bread with $\rightarrow$ toaster), the robot **should not** ask frequently (conforming to Grice's cooperative principle); for safety-based ambiguities (e.g., microwave-safe dishes), the robot **may** ask even if the choice seems obvious.
    - **Design Motivation**: Previous classifications were based on linguistics (referential ambiguity, spatial ambiguity, etc.). However, different linguistic types may share the same resolution strategy, making such classifications less actionable. Categorization based on action strategies directly guides robot behavior.

2. **Pairwise Construction**:

    - **Function**: Every ambiguous instruction has a linguistically minimally-different, unambiguous counterpart.
    - **Mechanism**: For example, "Please use the coffee machine to make a cup of coffee and pour it into **the cup**" (ambiguous) vs. "pour it into **the ceramic cup**" (unambiguous), where the difference lies only in key phrases.
    - **Design Motivation**: Pairwise construction allows direct measurement of whether a model can distinguish between ambiguous and unambiguous instructions, eliminating other confounding factors.

3. **Data Collection Pipeline**:

    - **Function**: Semi-automatic generation followed by human verification.
    - **Mechanism**: Manually create lists for over 750 kitchen items $\rightarrow$ randomly sample to generate 1,000 environments $\rightarrow$ use Mistral to generate unambiguous tasks $\rightarrow$ use ChatGPT to generate ambiguous counterparts and Q&A pairs $\rightarrow$ perform human annotation and auditing (inter-annotator agreement >95%).
    - **Design Motivation**: Leveraging LLM-assisted generation increases efficiency, while human verification guarantees high quality.

### Evaluation Metrics
- **Intent Coverage Rate (ICR)**: The percentage of user intent keywords covered by the model's prediction set.
- **Help Rate (HR)**: How frequently the robot requests help.
- **Correct Help Rate (CHR)**: The accuracy of requesting help based on the ambiguity type.
- **Set Size Correctness (SSC)**: The IoU between the predicted options set and the ground-truth option set.
- **Ambiguity Differentiation (AmbDif)**: The ability to distinguish between ambiguous and unambiguous instructions.

## Key Experimental Results

### Main Results
Testing five methods on GPT-3.5, GPT-4, Llama-2-7B, and Llama-3-8B:

| Method | Type | GPT-3.5 AmbDif | GPT-4 AmbDif | Llama-3 AmbDif |
|------|------|----------------|--------------|----------------|
| KnowNo | CP-based | 0.27 | 0.16 | 0.40 |
| LAP | CP-based | 0.18 | 0.15 | 0.40 |
| LofreeCP | CP-based (logit-free) | 0.28 | 0.20 | **0.44** |
| Binary | Prompt-based | 0.04 | 0.03 | 0.00 |
| No Help | Baseline | 0.00 | 0.00 | 0.00 |

### Ablation Study: AmbiK vs. KnowNo Dataset

| Dataset | KnowNo Method Help Rate | KnowNo Method Success Rate |
|--------|----------------------|------------------------|
| KnowNo (original dataset) | 0.80 | 0.79 |
| AmbiK | Very Low | Very Low |

### Key Findings
- All methods perform poorly on AmbiK—the highest AmbDif is only 0.44, indicating that existing methods struggle to distinguish between ambiguous and unambiguous tasks.
- No methods exceed 20% on SSC, indicating that CP sets are severely misaligned with actual ambiguous options.
- Robots tend to either never ask for help or always ask for help, failing to adjust flexibly based on the ambiguity type.
- Logit-free methods (Binary, LofreeCP) perform better than logit-based methods, indicating that LLM logits are unreliable as a proxy for uncertainty.
- Methods perform well on the simpler KnowNo dataset, demonstrating that AmbiK indeed provides a more challenging benchmark.

## Highlights & Insights
- **Ambiguity Categorization by Action Strategy**: Instead of classifying by linguistic features (reference, space, etc.), classifying by "what the robot should do" (preference $\rightarrow$ must ask, common sense $\rightarrow$ should not ask, safety $\rightarrow$ may ask) offers a more practical, application-oriented taxonomy.
- **Ingenious Pairwise Design**: Minimal differences between ambiguous and unambiguous instructions enable precise evaluation and direct measurement of a model's sensitivity to ambiguity.
- **Validation of Unreliable LLM Logits**: Logit-free methods outperform logit-based ones, aligning with existing literature (where RLHF leads to overconfidence).

## Limitations & Future Work
- Restricted only to kitchen environments; has not been extended to other household scenarios (garages, grocery stores, etc.).
- The text-only format lacks visual or spatial information, whereas physical robots require multimodal understanding.
- Zero-context assumption (ignoring dialogue history) limits the simulation of real interactive scenarios.
- Each ambiguous instruction has only one correct intent; multi-intent scenarios are not addressed.
- Only evaluated using few-shot prompting; fine-tuning methods have not been attempted.

## Related Work & Insights
- **vs. KnowNo**: KnowNo contains only 170 ambiguous tasks, mostly single-step simple instructions. AmbiK offers 1,000 ambiguous tasks with multi-step complex instructions, reflecting real-world scenarios more closely.
- **vs. SIF**: Ambiguity in SIF is limited to object location searches, whereas AmbiK spans preference, common-sense knowledge, and safety.
- **vs. SaGC**: "Ambiguity" in SaGC is actually task under-specification (e.g., "make something delicious"), which differs from the definition of "choices potentially leading to wrong consequences" used in this paper.
- Inspires instruction understanding and interaction design in embodied AI: models must learn "when to ask and when not to ask."

## Rating
- **Novelty**: ⭐⭐⭐⭐ The concept of categorizing ambiguity by action strategy is novel, and the pairwise construction is clean and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluates multiple methods and models across comprehensive metrics, with detailed comparisons against KnowNo.
- **Writing Quality**: ⭐⭐⭐⭐ The problem is clearly defined, the categorization taxonomy is reasonable, and the experimental analysis is in-depth.
- **Value**: ⭐⭐⭐⭐ Provides a much-needed, high-quality benchmark for ambiguity detection in embodied AI, revealing severe flaws in existing methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Composing Global Solutions to Reasoning Tasks via Algebraic Objects in Neural Nets](../../NeurIPS2025/optimization/composing_global_solutions_to_reasoning_tasks_via_algebraic_objects_in_neural_ne.md)
- [\[NeurIPS 2025\] AutoOpt: A Dataset and a Unified Framework for Automating Optimization Problem Solving](../../NeurIPS2025/optimization/autoopt_a_dataset_and_a_unified_framework_for_automating_optimization_problem_so.md)
- [\[CVPR 2025\] Towards Stable and Storage-efficient Dataset Distillation: Matching Convexified Trajectory](../../CVPR2025/optimization/towards_stable_and_storage-efficient_dataset_distillation_matching_convexified_t.md)
- [\[ICML 2025\] Understanding Sharpness Dynamics in NN Training with a Minimalist Example: The Effects of Dataset Difficulty, Depth, Stochasticity, and More](../../ICML2025/optimization/understanding_sharpness_dynamics_in_nn_training_with_a_minimalist_example_the_ef.md)
- [\[ICML 2026\] Selecting Samples on Graphs: A Unified Dataset Pruning Framework for Lossless Training Acceleration](../../ICML2026/optimization/selecting_samples_on_graphs_a_unified_dataset_pruning_framework_for_lossless_tra.md)

</div>

<!-- RELATED:END -->
