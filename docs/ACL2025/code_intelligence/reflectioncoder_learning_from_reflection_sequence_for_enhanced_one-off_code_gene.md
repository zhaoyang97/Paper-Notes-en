---
title: >-
  [Paper Note] ReflectionCoder: Learning from Reflection Sequence for Enhanced One-off Code Generation
description: >-
  [ACL 2025][Code Intelligence][Code Generation] ReflectionCoder achieves state-of-the-art (SOTA) performance in one-off code generation without requiring multi-round runtime debugging. It does this by constructing "reflection sequence" data that integrates compiler feedback, combined with two training strategies: reflection self-distillation and dynamically masked distillation.
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "Code Generation"
  - "Reflection Sequence"
  - "Knowledge Distillation"
  - "Compiler Feedback"
  - "One-off Generation"
date: 2026-05-08
content_hash: f32c8bf642c1345c
---

# ReflectionCoder: Learning from Reflection Sequence for Enhanced One-off Code Generation

**Conference**: ACL 2025  
**arXiv**: [2405.17057](https://arxiv.org/abs/2405.17057)  
**Code**: [GitHub](https://github.com/SenseLLM/ReflectionCoder)  
**Area**: Text Generation/Code Generation  
**Keywords**: Code Generation, Reflection Sequence, Knowledge Distillation, Compiler Feedback, One-off Generation  

## TL;DR
ReflectionCoder achieves state-of-the-art (SOTA) performance in one-off code generation without requiring multi-round runtime debugging. It does this by constructing "reflection sequence" data that integrates compiler feedback, combined with two training strategies: reflection self-distillation and dynamically masked distillation.

## Background & Motivation

**Background**: Code generation is one of the core applications of LLMs. Existing methods are mainly divided into two categories: (1) one-off generation, where the model directly outputs the final code; (2) iterative refinement, where the model generates code and refines it over multiple rounds based on compiler feedback. The latter typically performs better, but suffers from high inference costs and depends on compiler environments.

**Limitations of Prior Work**: The performance ceiling of one-off generation methods is limited by the quality of training data. Models learn from pure "requirement $\rightarrow$ code" data, lacking the capability to understand and correct buggy code. Conversely, although iterative refinement methods perform well, they involve complex engineering deployment and high latency, making them unsuitable for scenarios like real-time code completion.

**Key Challenge**: Compiler feedback contains rich error-correcting knowledge, but this knowledge cannot be directly utilized in one-off generation scenarios because there is no compiler at inference time, meaning the model cannot perform reflection and refinement. How to transfer the knowledge from the "reflection process" into a one-off generation model is the key challenge.

**Goal**: To design a method that allows models to learn the knowledge from compiler-feedback-driven reflection processes during training, while still performing efficient one-off code generation at inference time.

**Key Insight**: Instead of using compiler feedback at inference time, the reflection process is constructed as training data (reflection sequences) to "compress" reflection knowledge into one-off generation capabilities via distillation strategies.

**Core Idea**: Construct "attempt $\rightarrow$ error $\rightarrow$ reflection $\rightarrow$ correction" reflection sequences using compiler feedback as training data. Use reflection self-distillation to let the model learn from its own reflection process, and use dynamically masked distillation to ensure the model focuses on generating the final correct code.

## Method

### Overall Architecture
ReflectionCoder's pipeline consists of two stages: (1) **Data Construction Stage**: Given a programming problem, a teacher model (GPT-4/DeepSeek-Coder) is first used to generate initial code, obtain execution feedback from a compiler, and then refine the code based on the feedback. The entire process forms a "reflection sequence". (2) **Training Stage**: The student model is trained on a mixture of constructed reflection sequence data and raw code instruction data using twin strategies: reflection self-distillation and dynamically masked distillation. During inference, the model requires only a single forward pass to generate the final code.

### Key Designs

1. **Reflection Sequence Construction**:

    - **Function**: Construct multi-round "attempt-reflection-correction" training data containing compiler feedback.
    - **Mechanism**: Each reflection sequence contains three parts: (T) initial code attempt, (C) feedback from compiler/tests, and (E) reflection and corrected code. The format is $[T_1, C_1, E_1, T_2, C_2, E_2, \ldots]$, which may involve multiple rounds of reflection. Two datasets, ReflectionSeq-GPT and ReflectionSeq-DS, are constructed using GPT-4 and DeepSeek-Coder, respectively.
    - **Design Motivation**: By preserving the complete debugging process, the model learns the logical chain of "where error occurs, why it occurs, and how to fix it," rather than solely learning the final correct answer.

2. **Reflection Self-Distillation**:

    - **Function**: Enable the student model to learn from reflection sequences while generating only the final correct code at inference time.
    - **Mechanism**: During training, the input to the model contains the complete reflection sequences, but the loss function differentiates between block types. For the attempt and compile feedback parts (T and C blocks), the model is distilled using the logits of the teacher model. For the final correct code part (E block), a standard next-token prediction loss is directly applied. The overall loss is $\mathcal{L} = \alpha \mathcal{L}_{distill} + (1-\alpha) \mathcal{L}_{CE}$.
    - **Design Motivation**: Allow the model to comprehend the reflection process without relying on it in practice—viewing the reflection sequence during training but directly outputting the final result during inference.

3. **Dynamically Masked Distillation**:

    - **Function**: Further optimize training signals to prevent the model from attempting to generate the reflection process during inference.
    - **Mechanism**: Dynamically mask the T and C components in the reflection sequence during training, gradually increasing the masking ratio as training progresses. This allows the model to see the full reflection process in the early stage, and progressively conceals more reflection context later on, forcing the model to internalize the reflection knowledge into one-off generation capabilities. The mask ratio increases linearly from 0 to nearly 1, such that the model sees almost exclusively the final correct code at the end of training.
    - **Design Motivation**: Resolve the training-inference discrepancy. If the complete reflection sequence is always visible during training, the model might attempt to generate the reflection process at inference time, instead of directly outputting the code.

### Loss & Training
The overall training adopts a mixed data strategy, combining reflection sequence data (ReflectionSeq-GPT + ReflectionSeq-DS) with standard code instruction data (such as Evol-CodeAlpaca). Distributed training is performed using DeepSpeed ZeRO-1, with a learning rate of 5e-5, a cosine scheduler, 2 training epochs, and a global batch size of 512. Models are fine-tuned on CodeLlama and DeepSeek-Coder baselines.

## Key Experimental Results

### Main Results
The pass@1 results on HumanEval(+) and MBPP(+) benchmarks are shown below:

| Model | Params | HumanEval | HumanEval+ | MBPP | MBPP+ |
|------|--------|-----------|------------|------|-------|
| WizardCoder-CL-34B | 34B | 73.2 | 64.6 | 73.2 | 59.9 |
| MagiCoder-CL-7B | 7B | 71.3 | 64.6 | 68.4 | 56.9 |
| OpenCodeInterp-DS-6.7B | 6.7B | 76.2 | 72.0 | 73.9 | 63.7 |
| **ReflectionCoder-CL-7B** | 7B | **75.0** | **68.9** | **72.2** | **61.4** |
| **ReflectionCoder-DS-6.7B** | 6.7B | **80.5** | **74.4** | **81.5** | **69.6** |
| **ReflectionCoder-DS-33B** | 33B | **82.9** | **76.8** | **84.1** | **72.0** |

### Ablation Study
Ablation analysis under different training strategies and data combinations:

| Configuration | HumanEval | MBPP+ | Description |
|------|-----------|-------|------|
| CodeInstruct data only | 74.4 | 63.2 | Baseline, no reflection data |
| + ReflectionSeq (Standard CE) | 77.4 | 66.5 | Trained directly with reflection sequences |
| + Reflection Self-Distillation | 79.3 | 68.1 | With distillation strategy |
| + Dynamically Masked Distillation (Full Model) | 80.5 | 69.6 | Full ReflectionCoder |
| ReflectionSeq-GPT only | 78.7 | 67.8 | Only use reflection data generated by GPT |
| ReflectionSeq-DS only | 77.9 | 66.9 | Only use reflection data generated by DeepSeek |

### Key Findings
- **Reflection sequence data is the largest contributor to performance improvement**: Simply adding the reflection sequence data (i.e., using standard CE training) yields an improvement of approximately 3%, indicating that the "error $\rightarrow$ correction" process itself contains valuable supervisory signals.
- **Strong complementarity between the two distillation strategies**: Reflection self-distillation contributes to an approximate 2% improvement, and dynamically masked distillation adds another 1.5%. The combined effect outperforms using either in isolation.
- **Strong performance on the MultiPL-E multilingual benchmark**: Performance is not limited to Python; the approach significantly outperforms baselines in languages like Java, C++, and JavaScript, showing that reflection knowledge possesses cross-lingual transferability.
- Mixing both GPT and DeepSeek reflection datasets achieves the best results, suggesting that data diversity aids generalization.

## Highlights & Insights
- **Distilling the advantages of multi-round inference into one-off generation**: This concept of "viewing the entire process during training, but outputting results directly during inference" is highly elegant. It can be widely adapted to other tasks that require iterative optimization yet demand low inference latency, such as mathematical reasoning and code repair.
- **Progressive masking strategy**: The curriculum learning approach of transitioning from fully visible to fully masked dynamic scheduling gracefully addresses the training-inference gap. This design pattern can be applied to any scenario that trains on intermediate reasoning steps but aims to bypass them during inference.
- **Scalability of the data construction method**: Reflection sequences can be constructed automatically at scale without manual annotation, natively utilizing the compiler as a free verification tool.

## Limitations & Future Work
- The quality of reflection sequences directly depends on the debugging capability of the teacher model; if the teacher model cannot correct a certain class of errors, the student cannot learn it either.
- Currently validated only on function-level code generation; the effects on more complex repository-level code generation remain unknown.
- The scheduling strategy for the dynamic mask (linear growth) is relatively simple; superior curriculum learning schemes might exist.
- Using this method in non-coding domains was not explored (though mentioned as a possibility in the paper).
- **Future Directions**: Stronger code verifiers (such as formal verification tools) could replace basic compilation tests to construct higher-quality reflection sequences.

## Related Work & Insights
- **vs WizardCoder**: WizardCoder utilizes Evol-Instruct to scale instruction complexity to enhance code generation. This work focuses on the data construction perspective (reflection sequences); the two approaches could be complementary.
- **vs Self-Repair/Self-Debug**: The latter uses compiler feedback for multi-round refinement during inference, yielding strong results at high cost. This paper compresses this correction capability into one-off generation, significantly lowering inference costs.
- **vs MagiCoder**: MagiCoder enhances training data by generating high-quality programming problems from open-source code snippets. This paper introduces error correction trajectories to provide additional supervisory signals, approaching the problem from a different angle.
- The most inspiring aspect of this method is the paradigm of "distilling multi-step reasoning knowledge into single-step generation," which is highly aligned with recent trends in reasoning distillation (such as distilling CoT into direct answers).

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of reflection sequences and progressive masked distillation represents a meaningful innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple model scales, multiple benchmarks, comprehensive ablation, and MultiPL-E multilingual evaluation.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described with intuitive diagrams.
- Value: ⭐⭐⭐⭐ High practical value; the "reflect during training, direct output during inference" paradigm offers broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GiFT: Gibbs Fine-Tuning for Code Generation](gift_gibbs_fine_tuning_code_gen.md)
- [\[ACL 2025\] Rethinking Repetition Problems of LLMs in Code Generation](rethinking_repetition_problems_of_llms_in_code_generation.md)
- [\[ACL 2025\] Personality-Guided Code Generation Using Large Language Models](personality_guided_code_gen.md)
- [\[ACL 2025\] Revisit Self-Debugging with Self-Generated Tests for Code Generation](revisit_self-debugging_with_self-generated_tests_for_code_generation.md)
- [\[ACL 2025\] Tree-of-Code: A Tree-Structured Exploring Framework for End-to-End Code Generation](tree-of-code_a_tree-structured_exploring_framework_for_end-to-end_code_generatio.md)

</div>

<!-- RELATED:END -->
