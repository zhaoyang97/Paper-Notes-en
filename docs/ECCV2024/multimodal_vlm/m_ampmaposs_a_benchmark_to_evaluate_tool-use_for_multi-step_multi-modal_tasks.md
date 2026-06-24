---
title: >-
  [Paper Note] m&m's: A Benchmark to Evaluate Tool-Use for Multi-step Multi-modal Tasks
description: >-
  [ECCV 2024][Multimodal VLM][Tool use] Proposes the m&m's benchmark, which contains 4K+ multi-step multi-modal tasks and 33 executable tools, to systematically evaluate the tool-use capability of 10 LLMs across different planning strategies (multi-step vs. step-by-step), plan formats (JSON vs. code), and feedback types (parsing/validation/execution), discovering that multi-step JSON planning coupled with feedback is the currently optimal design.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Tool use"
  - "LLM planning"
  - "multi-step multi-modal"
  - "benchmark evaluation"
  - "planning strategy"
date: 2026-05-08
content_hash: 85d0cc7a3a23f8ad
---

# m&m's: A Benchmark to Evaluate Tool-Use for Multi-step Multi-modal Tasks

**Conference**: ECCV 2024  
**arXiv**: [2403.11085](https://arxiv.org/abs/2403.11085)  
**Code**: [GitHub](https://github.com/RAIVNLab/mms)  
**Area**: Multimodal VLM  
**Keywords**: Tool use, LLM planning, multi-step multi-modal, benchmark evaluation, planning strategy

## TL;DR

Proposes the m&m's benchmark, which contains 4K+ multi-step multi-modal tasks and 33 executable tools, to systematically evaluate the tool-use capability of 10 LLMs across different planning strategies (multi-step vs. step-by-step), plan formats (JSON vs. code), and feedback types (parsing/validation/execution), discovering that multi-step JSON planning coupled with feedback is the currently optimal design.

## Background & Motivation

Real-world multi-modal problems can rarely be solved by a single model and usually require combining multi-step computational pipelines of multiple models. Tool-augmented LLMs show great promise in automatically generating such plans, but they lack a standardized benchmark to evaluate the design decisions of LLMs as planners for multi-step multi-modal tasks:

**Planning Strategy**: Should LLMs generate the complete plan all at once, or step-by-step? Existing works use different strategies but have never systematically compared them.

**Plan Format**: Should Python code or structured formats like JSON be used? How do different formats impact executability?

**Feedback Mechanism**: How do parsing feedback, rule-validation feedback, and execution feedback respectively affect planning quality?

Key limitations of prior work:
- **TaskBench**: Does not provide tool implementations, uses placeholder filenames (e.g., "example.png"), and cannot be executed.
- **ToolEmu**: Uses LLMs to simulate tool execution instead of real execution, making it impossible to study execution feedback.
- **ToolBench/ToolFormer**: Do not support multi-modality or lack ground truth plans.

Key Challenge: The design space of planning agents is growing combinationally, yet no single benchmark supports a fair evaluation across all dimensions. m&m's provides the first unified benchmark supporting real multi-modal inputs, real tool execution, human-verified plans, and multiple evaluation dimensions.

## Method

### Overall Architecture

m&m's contains three core components:
1. **Benchmark Datasets**: 4427 raw tasks, 1565 human-verified tasks, and an 882-task tool-balanced subset.
2. **33 Executable Tools**: 13 ML models + 11 image processing modules + 9 public APIs.
3. **Modular Planning System**: LLM + Parser + Validator + Executor, based on the AutoGen framework.

### Key Designs

1. **Data Generation Pipeline**:

    - Function: Automatically generate diverse query-plan pairs.
    - Mechanism: A five-step process——
        - **Tool Graph Construction and Sampling**: Construct the 33 tools as a directed graph (edges represent valid connections where output-input types match), and sample subgraphs as tool sequences.
        - **Input Example Sampling**: Sample real inputs from the validation sets of 11 datasets including ImageNet, SQuAD, Visual Genome, MagicBrush, and LibriSpeech.
        - **Query Generation**: Use GPT-4 to generate natural language user queries based on tool sequences and input examples.
        - **Plan Generation**: Generate ground truth plans using a rule-based program (not GPT-4), including tool names, argument names, and argument values.
        - **Human Verification**: Three annotators verify each query-plan pair.
    - Design Motivation: Using real inputs (instead of placeholders like "example.jpg") ensures the realism of queries and the executability of plans; using a rule-based program instead of GPT-4 to generate plans eliminates the possibility of hallucinations and incorrect plans.

2. **Tool Graph Design**:

    - Function: Define valid connection relationships among the 33 tools.
    - Mechanism: Each tool is a node in a directed graph, and edges represent that the source tool's output type matches the target tool's input type. For example, image_classification $\rightarrow$ wikipedia_simple_search is valid (text label $\rightarrow$ search query).
    - Tools are categorized into three types:
        - **ML Models** (13): Text generation, image classification, object detection, VQA, etc. (from HuggingFace).
        - **Image Processing** (11): Crop, segment, count, background blur, emoji overlay, etc. (from VisProg).
        - **Public APIs** (9): Weather, geological location, mathematical facts, Wikipedia search, etc. (from RapidAPI).

3. **Modular Planning Agent**:

    - Function: Flexibly combine different planning strategies, formats, and feedback types.
    - Mechanism:
        - **Multi-step Planning**: The LLM generates the complete plan (all tool steps) at once.
        - **Step-by-step Planning**: The LLM predicts only one action at a time and predicts the next after receiving feedback.
    - Three feedback mechanisms:
        - **Parsing Feedback**: Whether the LLM output can be successfully parsed into valid JSON/code.
        - **Validation Feedback**: Whether the tool exists, whether the connections are valid, and whether the argument names are correct.
        - **Execution Feedback**: Actually execute the tool and return the output or execution error messages.
    - Design Motivation: The modular design allows each dimension to vary independently, supporting combinatorial experiments.

4. **Evaluation Metrics Design**:

    - Function: Evaluate planning quality from two dimensions: tool selection and tool invocation.
    - Mechanism: Three primary metrics——
        - **tool-F1**: F1 score of tool name prediction (evaluates tool selection).
        - **argname-F1**: F1 score of argument name prediction (evaluates tool invocation).
        - **pass rate**: The proportion of predictions with zero execution errors (evaluates executability).
    - Design Motivation: Separating the evaluation of tool selection and tool invocation avoids conflating planning errors with execution errors.

5. **Alternative Plan Generation**:

    - Function: Generate multiple valid alternative plans for each query to avoid relying solely on a single ground truth evaluation.
    - Mechanism: Generate syntactically valid (input-output type matched) and semantically valid (functionally equivalent) alternative tools for each tool, and combine alternative tools at all positions to obtain the complete set of alternative plans.
    - Influence: Considering alternative plans recovers $1\text{--}5\%$ of misjudgements in plan accuracy.

### Loss & Training

This work does not involve model training; instead, it evaluates the zero-shot planning capabilities of existing LLMs. The evaluation covers:
- 5 Open-source LLMs: Llama-2-7B/13B/70B, Mixtral-8x7B, Llama-3-70B
- 2 Code LLMs: CodeLlama-34B/70B
- 3 Commercial LLMs: GPT-3.5-turbo, GPT-4, GPT-4o
- Total: $10 \text{ models} \times 2 \text{ strategies} \times 2 \text{ formats} \times 4 \text{ feedback combinations}$

## Key Experimental Results

### Main Results: Comparison of Planning Strategies (Tool-F1)

| Model | Step-by-step Planning | Multi-step Planning | Difference |
|------|---------|---------|------|
| Llama-2-7B | $\sim 20$ | $29.78$ | $\sim +10$ |
| Llama-2-13B | $\sim 32$ | $42.27$ | $\sim +10$ |
| GPT-3.5-turbo | $\sim 59$ | $80.52$ | $+21.8$ |
| GPT-4 | $\sim 83$ | $88.46$ | $\sim +5$ |
| GPT-4o | $\sim 85$ | $89.28$ | $\sim +4$ |

### Main Results: Impact of Feedback Types on Tool Invocation

| Model | Baseline P (pass rate) | +V $\Delta$ | +E $\Delta$ | +VE $\Delta$ |
|------|---------|------|------|-------|
| Llama-2-7B | $28.23$ | $+18.14$ | $+10.32$ | $+13.72$ |
| Llama-2-13B | $38.10$ | $+29.93$ | $+32.99$ | $+23.92$ |
| Mixtral-8x7B | $75.74$ | $+10.32$ | $+8.96$ | $+10.77$ |
| GPT-3.5-turbo | $89.46$ | $+6.69$ | $+7.26$ | $+6.92$ |
| GPT-4 | $97.73$ | $+1.13$ | $-1.25$ | $+2.15$ |

### Ablation Study: JSON vs. Code Formats

| Model | JSON tool-F1 | Code tool-F1 | JSON pass rate | Code pass rate |
|------|-------------|-------------|---------------|---------------|
| Llama-2-13B | $42.27$ | $\sim 40$ | $38.10$ | $\sim 15$ |
| Mixtral-8x7B | $66.79$ | $\sim 65$ | $75.74$ | $\sim 40$ |
| GPT-3.5-turbo | $80.52$ | $\sim 79$ | $89.46$ | $\sim 55$ |
| GPT-4 | $88.46$ | $\sim 87$ | $97.73$ | $\sim 75$ |

### Key Findings

- **Multi-step planning consistently outperforms step-by-step planning**: The tool-F1 score is higher under multi-step planning for all models. Step-by-step planning models tend to terminate prematurely upon receiving positive feedback, omitting necessary tool steps.
- **Feedback significantly boosts executability but may slightly hurt tool selection**: Validation and execution feedback can boost the pass rate by $20\text{--}30\%$, but tool-F1 may drop by $<5\%$, because models sometimes mistakenly replace correct tools when fixing errors.
- **The JSON format is far superior to code generation in terms of executability**: The two formats differ negligibly in tool selection ($<3\%$), but the pass rate of code generation drops dramatically. A common error in code generation is the failure to correctly access the outputs of previous tools.
- **Validation feedback is more targeted than execution feedback**: The validator points out exact error locations, whereas the error messages returned by the executor can be vague and confusing.
- **More capable models see smaller marginal utility from feedback**: GPT-4 is already close to a perfect pass rate, showing limited improvement from feedback.

## Highlights & Insights

- **First multi-step multi-modal tool-use benchmark supporting real inputs and real execution**: Fills a key gap left by ToolEmu/TaskBench.
- **Systematic exploration of the design space**: A combinatorial experiment of $10 \text{ models} \times 2 \times 2 \times 4$ yields comprehensive empirical conclusions.
- **Use of rule-based programs to generate ground truth plans**: Eliminates plan errors caused by GPT-4 hallucination, guaranteeing data quality.
- **Alternative plan mechanism**: Recognizes that a single task can have multiple correct solutions, making evaluation fairer.
- **Formal definition of the tool graph**: Provides a rigorous problem modeling framework for subsequent tool-use research.

## Limitations & Future Work

- Only sequence-based plans are considered; dynamic or conditional branching plans (e.g., "if image classification result is A, do step X; otherwise, do step Y") are not supported.
- Sophisticated prompting strategies (such as Tree-of-Thoughts) are left unexplored; only direct and ReACT-style planning are used.
- Some tools are non-deterministic (e.g., image generation), making executive result evaluation challenging.
- Only text-only LLM planners are evaluated; multimodal LLMs (such as GPT-4V planning directly from images) are not tested.
- The scale of 33 tools is relatively small compared to ToolBench ($3451$ tools), which might not suffice to test large-scale tool-selection capabilities.
- The human verification agreement rate is $75.75\%$, indicating some degree of subjectivity in determining task correctness.

## Related Work & Insights

- **VisProg/ViperGPT** (Gupta et al. / Surís et al.): Multi-step visual reasoning using Python code formats, representing early exploration of multi-modal tool use.
- **HuggingGPT** (Shen et al., 2023): Multi-step planning without plan execution, only evaluating plan accuracy.
- **ToolFormer** (Schick et al., 2024): Allows language models to self-train for tool use, but uses natural language interfaces rather than code interfaces.
- **TaskBench** (Shen et al., 2023): The most related concurrent work, but uses placeholder inputs and GPT-4 to generate answers.
- **AutoGen** (Wu et al., 2023): Provides a multi-agent conversation framework, which is used by m&m's as the foundation of the planning system.
- Insights: **The constraint of a structured format (JSON) is, in fact, an advantage**—rigid structures reduce output formatting errors, making plans more executable. Future tool-use systems should seek a balance between flexibility and reliability.

## Rating

- Novelty: ⭐⭐⭐⭐ The first multi-modal tool-use benchmark to support full execution chains, featuring a systematic exploration of the design space.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, with a cross-experimental analysis of 10 models across multiple dimensions and 15+ evaluation metrics.
- Writing Quality: ⭐⭐⭐⭐ Clear problem statement, systematically organized experiments, and realistic, practical summaries of findings.
- Value: ⭐⭐⭐⭐ Provides a standardized evaluation platform and empirical design guidance for LLM tool-use research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](../../ICLR2026/multimodal_vlm/multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)
- [\[ECCV 2024\] ShareGPT4V: Improving Large Multi-Modal Models with Better Captions](sharegpt4v_improving_large_multi-modal_models_with_better_captions.md)
- [\[ACL 2026\] AdaTooler-V: Adaptive Tool-Use for Images and Videos](../../ACL2026/multimodal_vlm/adatooler-v_adaptive_tool-use_for_images_and_videos.md)
- [\[AAAI 2026\] VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use](../../AAAI2026/multimodal_vlm/vipact_visual-perception_enhancement_via_specialized_vlm_age.md)
- [\[ECCV 2024\] MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?](mathverse_does_your_multi-modal_llm_truly_see_the_diagrams_in_visual_math_proble.md)

</div>

<!-- RELATED:END -->
