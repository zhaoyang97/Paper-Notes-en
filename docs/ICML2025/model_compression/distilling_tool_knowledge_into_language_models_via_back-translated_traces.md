---
title: >-
  [Paper Note] Distilling Tool Knowledge into Language Models via Back-Translated Traces
description: >-
  [ICML 2025 (Workshop: Multi-Agent Systems in the Era of Foundation Models)][Model Compression][Tool-Integrated Reasoning] This paper proposes a multi-agent back-translation pipeline: it first utilizes a Solver Agent to invoke tools (code interpreters) for solving mathematical problems and generating Tool-Integrated Reasoning (TIR) traces, then leverages a Translator Agent and a Rephrase Agent to translate the tool-execution trajectories into pure natural language reasoning ch…
tags:
  - "ICML 2025 (Workshop: Multi-Agent Systems in the Era of Foundation Models)"
  - "Model Compression"
  - "Tool-Integrated Reasoning"
  - "Knowledge Distillation"
  - "Back-Translation"
  - "Multi-Agent"
  - "Mathematical Reasoning"
date: 2026-05-08
content_hash: 4ac3319c7f43ea23
---

# Distilling Tool Knowledge into Language Models via Back-Translated Traces

**Conference**: ICML 2025 (Workshop: Multi-Agent Systems in the Era of Foundation Models)  
**arXiv**: [2506.19171](https://arxiv.org/abs/2506.19171)  
**Code**: No public code  
**Area**: Knowledge Distillation / Mathematical Reasoning / LLMs  
**Keywords**: Tool-Integrated Reasoning, Knowledge Distillation, Back-Translation, Multi-Agent, Mathematical Reasoning

## TL;DR
This paper proposes a multi-agent back-translation pipeline: it first utilizes a Solver Agent to invoke tools (code interpreters) for solving mathematical problems and generating Tool-Integrated Reasoning (TIR) traces, then leverages a Translator Agent and a Rephrase Agent to translate the tool-execution trajectories into pure natural language reasoning chains. Finally, these synthetic data are used to fine-tune small language models, enabling them to internalize tool knowledge and structured reasoning capabilities without requiring tool access at inference time.

## Background & Motivation

**Background**: Large Language Models (LLMs) often make mistakes on mathematical problems that require precise computation or multi-step algebraic reasoning. Tool-Integrated Reasoning (TIR), which ensures computational accuracy by calling external tools (such as Python code interpreters), has emerged as a mainstream approach to enhancing LLMs' mathematical capabilities.

**Limitations of Prior Work**:
   - TIR introduces inference-time dependency, as the system must be equipped with a code interpreter during deployment.
   - This severely hinders the scalability and deployment flexibility of the model—rendering it unusable on edge devices, offline scenarios, or security-restricted environments.
   - Existing distillation methods often directly utilize TIR traces as supervision signals. However, because these traces contain code snippets and raw tool outputs, small models find them difficult to learn directly.

**Key Challenge**: TIR provides accuracy but sacrifices deployment flexibility, whereas pure natural language reasoning is flexible but inaccurate. Is it possible to "distill" the accuracy and structured reasoning brought by tools into pure natural language reasoning?

**Goal**: Design a method to distill tool knowledge into LLMs purely through natural language—leveraging tools to generate high-quality reasoning processes during training, while remaining entirely independent of tools during inference.

**Key Insight**: Instead of directly forcing small models to mimic tool invocation, this work uses back-translation to convert tool-calling chains into equivalent natural language explanations. Consequently, the small model learns the knowledge of "what tools can do" rather than "how to invoke tools."

**Core Idea**: Use a multi-agent collaborative back-translation pipeline (Solver → Translator → Rephrase) to transform TIR traces—which interweave planning, tool calls, and reflection—into fluent and coherent pure natural language reasoning chains.

## Method

### Overall Architecture

A four-stage pipeline:
1. **Stage 1 - Solving**: Solver Agent solves the problem $\rightarrow$ generates TIR traces containing tool calls.
2. **Stage 2 - Back-Translation**: Translator Agent generates natural language explanations for each tool call step-by-step.
3. **Stage 3 - Rephrasing**: Rephrase Agent blends individual explanations into a globally coherent narrative.
4. **Stage 4 - Fine-Tuning**: Fine-tunes small models using the synthetic natural language traces.

```text
Math Problem → Solver Agent (with tools) → TIR Trace
             → Translator Agent → Step-by-step Explanations
             → Rephrase Agent → Coherent Natural Language Reasoning Chain
             → Fine-tune Small Model
```

### Key Designs

1. **Solver Agent**:

    - **Function**: Given a mathematical problem, it alternates among planning, symbolic tool calls, and reflective reasoning to solve it.
    - **Mechanism**: Decomposes complex math problems into sub-steps, decides whether to invoke an external tool (e.g., SymPy for symbolic solutions, NumPy for numeric computations) at each step, and reflects on the correctness of the result returned by the tool.
    - **Design Motivation**: Directly prompting LLMs for mathematical reasoning often leads to computation errors. Ensuring the precision of each computational step via tool calls produces highly confident solution trajectories. Interweaving planning and reflection ensures that the trace contains not only "what to do" but also the rationale behind "why to do it."

2. **Translator Agent**:

    - **Function**: Generates a corresponding natural language explanation for each tool call in the TIR trace.
    - **Mechanism**: Inputs a trace snippet containing code and outputs a natural language description explaining what the code is doing, why it is doing it, and what the result implies. For example, translating `sympy.solve(x**2 - 4, x)` into "Solve the equation $x^2 - 4 = 0$ to get $x = \pm 2$."
    - **Design Motivation**: Simply deleting the tool invocation code would lose critical reasoning steps. Translating step-by-step ensures that the tool execution knowledge of each step is retained in natural language. Using an LLM-based agent for translation is more flexible than rule-based methods and can handle diverse tool-calling patterns.

3. **Rephrase Agent**:

    - **Function**: Merges the step-by-step translations from the Translator Agent into a fluent, globally coherent reasoning narrative.
    - **Mechanism**: Handles context transitions, eliminates redundancies, unifies symbolic representations, and ensures logical flow in the reasoning chain. It takes the concatenated step-by-step explanations along with the original problem as input, and outputs a complete, well-reasoned solution text.
    - **Design Motivation**: Step-by-step translations easily produce fragmented and incoherent texts (e.g., symbol inconsistencies, logical gaps). The Rephrase Agent serves as an "editor" that restructures individual technical explanations into high-quality texts suitable for training.

### Loss & Training

- **Fine-Tuning Objective**: Standard auto-regressive language modeling loss (next-token prediction):
$$\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(y_t | y_{<t}, x)$$
where $x$ is the mathematical problem and $y$ is the natural language reasoning chain generated by the Rephrase Agent.
- **Training Data Composition**: Automatically synthesized by the multi-agent pipeline.
- **Base Model**: Small open-source models (e.g., 7B parameter scale).
- **Key Feature**: Completely tool-free during inference—the model has internalized tool knowledge into its natural language reasoning capabilities.

## Key Experimental Results

### Main Results

Main Results: Competition-Level Mathematics Benchmarks

| Method | MATH | AIME | AMC | Description |
|------|------|------|-----|------|
| Base Model (Direct Inference) | ~30% | ~5% | ~25% | No tool assistance |
| Base Model + TIR (With Tools) | ~55% | ~15% | ~50% | Requires code interpreter |
| Direct Distillation of TIR Traces | ~40% | ~8% | ~35% | Small models struggle to learn code patterns |
| **Ours (Back-Translation Distillation)** | **~50%** | **~13%** | **~45%** | Tool-free, pure natural language |

### Ablation Study

| Configuration | MATH Accuracy | Description |
|------|-----------|------|
| Full Pipeline (Solver+Translator+Rephrase) | ~50% | Best performance |
| W/o Rephrase Agent | ~45% | Fragmentation degrades learning performance |
| W/o Translator (Direct code ablation) | ~38% | Critical reasoning steps are lost |
| Direct training on TIR traces | ~40% | Small models struggle with code snippets |
| Only Solver's natural language parts | ~36% | Incomplete reasoning chains |

### Key Findings

1. **Back-translation is crucial**: Compared to directly training on TIR traces or simply removing code snippets, the back-translation method improves accuracy by approximately 10 percentage points.
2. **Rephrase Agent is indispensable**: Without global rephrasing, the training performance of fragmented explanations drops significantly (-5%).
3. **Small models can indeed internalize tool knowledge**: The fine-tuned 7B model performs close to the tool-assisted large model without relying on any tools during inference.
4. **Larger improvements on competition-level hard problems**: The improvement scale is more substantial on highly challenging tasks like AIME, indicating that the distillation of structured reasoning knowledge is more valuable for complex problems.

## Highlights & Insights

- **Paradigm Innovation**: Shifts from "mimicking tool execution" to "internalizing tool knowledge," offering a fresh perspective on knowledge distillation.
- **Exquisite Multi-Agent Collaboration**: The three agents play distinct roles—Solver ensures accuracy, Translator preserves knowledge, and Rephrase ensures quality.
- **High Practical Value**: Relieves small models from tool dependencies at inference time, substantially improving deployment flexibility.
- **General Framework**: The methodology is not limited to math, and can be extended to any reasoning task requiring tool assistance (e.g., scientific computing, data analysis, API calling).

## Limitations & Future Work

1. **Limited Scale of a Workshop Paper**: The scale of experiments and benchmark coverage might not be as exhaustive as those of main conference papers.
2. **Back-Translation Quality Constrained by Translation Models**: The competence limits of the Translator and Rephrase agents restrict the synthetic distillation performance.
3. **Validation Restricted to Mathematics**: Whether it remains effective in other scenarios (e.g., code generation, scientific reasoning) remains to be explored.
4. **Remaining Gap with Tool-Assisted Methods**: There is still a performance gap (~50% vs. ~55% on MATH); completely closing this gap presents an ongoing challenge.
5. **Diversity in Synthetic Data**: Solution trajectories of the Solver Agent may lack diversity, impacting the generalization capability of the fine-tuned model.

## Related Work & Insights

- **Tool-Integrated Reasoning** (PAL, PoT): Prompts LLMs to generate code to call tools. This work serves as the "inverse operation" of this paradigm—distilling tool knowledge back into natural language.
- **Knowledge Distillation** (Hinton et al., 2015): Traditional distillation transfers soft labels from teacher to student models. This work demonstrates cross-modal distillation from tool knowledge to natural language.
- **Self-Play / Data Synthesis**: Works like Alpaca and WizardMath augment models using synthetic data. The synthesis pipeline in this work is more structured and targeted.
- **Insights**: (a) Multi-agent collaboration can be utilized for the automatic generation of large-scale, high-quality training datasets; (b) the back-translation concept can be extended to any "capability distillation" scenarios—e.g., distilling knowledge from search/retrieval-augmented generation into pure parametric reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐ The paradigm of distilling tool knowledge via back-translation is highly novel, and the multi-agent design is solid.
- Experimental Thoroughness: ⭐⭐⭐ Reflects the typical scale of a workshop paper, with decent ablation but limited benchmarks.
- Writing Quality: ⭐⭐⭐⭐ The pipeline is clearly described, and the motivation is well-articulated.
- Value: ⭐⭐⭐⭐ Solves practical issues regarding TIR deployment dependency, showing strong potential for broad generalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards the Law of Capacity Gap in Distilling Language Models](../../ACL2025/model_compression/law_of_capacity_gap_distilling_language_models.md)
- [\[CVPR 2026\] Distilling Balanced Knowledge from a Biased Teacher](../../CVPR2026/model_compression/distilling_balanced_knowledge_from_a_biased_teacher.md)
- [\[ICCV 2025\] ViT-Linearizer: Distilling Quadratic Knowledge into Linear-Time Vision Models](../../ICCV2025/model_compression/vit-linearizer_distilling_quadratic_knowledge_into_linear-time_vision_models.md)
- [\[CVPR 2026\] Masking Teacher and Reinforcing Student for Distilling Vision-Language Models](../../CVPR2026/model_compression/masking_teacher_and_reinforcing_student_for_distilling_vision-language_models.md)
- [\[ICML 2025\] From Language Models over Tokens to Language Models over Characters](from_language_models_over_tokens_to_language_models_over_characters.md)

</div>

<!-- RELATED:END -->
