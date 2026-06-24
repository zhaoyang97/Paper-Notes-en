---
title: >-
  [Paper Note] LLMs Can Be Easily Confused by Instructional Distractions
description: >-
  [ACL 2025][LLM (Other)][Instructional Distraction] This paper reveals that LLMs are severely misled when processing scenarios where the input itself resembles an instruction (instructional distraction). It proposes the DIM-Bench benchmark to evaluate this issue, demonstrating that mainstream LLMs, including GPT-4o, are significantly affected, and existing prompting strategies cannot fundamentally resolve it.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Instructional Distraction"
  - "Instruction-Following Robustness"
  - "DIM-Bench"
  - "Data Processing Tasks"
  - "Input Confusability"
date: 2026-05-08
content_hash: 23c2b57686cc4bd4
---

# LLMs Can Be Easily Confused by Instructional Distractions

**Conference**: ACL 2025  
**arXiv**: [2502.04362](https://arxiv.org/abs/2502.04362)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Instructional Distraction, Instruction-Following Robustness, DIM-Bench, Data Processing Tasks, Input Confusability

## TL;DR

This paper reveals that LLMs are severely misled when processing scenarios where the input itself resembles an instruction (instructional distraction). It proposes the DIM-Bench benchmark to evaluate this issue, demonstrating that mainstream LLMs, including GPT-4o, are significantly affected, and existing prompting strategies cannot fundamentally resolve it.

## Background & Motivation

1. **Background**: The instruction-following capability of LLMs is the core foundation for their application. Through instruction tuning, LLMs perform exceptionally well in zero-shot scenarios. Existing evaluation benchmarks (such as FollowBench and IFEval) mainly focus on instruction-following capabilities under dimensions like complex constraints and multi-step reasoning.

2. **Limitations of Prior Work**: In practical data processing scenarios (e.g., using LLMs for batch translation, rewriting, or proofreading), the input text itself may contain instructional content. For instance, when a user asks to translate a math problem, the LLM might attempt to solve the problem directly rather than translate it. This "instructional distraction" issue is particularly severe in batch data generation/processing, because it is impractical to manually adjust prompts step-by-step for each instance.

3. **Key Challenge**: The instruction-following capability of LLMs itself becomes a vulnerability—models are so adept at "executing instructions when they see them" that they fail to distinguish between the "user's task instructions" and "content in the input data that happens to look like instructions." Even when the prompt explicitly delimits the boundaries between Instruction and Input, the model is still led astray by the instructional content within the input.

4. **Goal**: (1) Systematically define and categorize the phenomenon of "instructional distraction"; (2) construct an evaluation benchmark; (3) evaluate the distraction resistance of mainstream LLMs; (4) explore the effectiveness of mitigation strategies.

5. **Key Insight**: Starting from the practical needs of data processing—rewriting, proofreading, translation, and style transfer—how do LLMs perform when encountering five types of "instruction-like" inputs: reasoning, code, math, bias detection, and reading comprehension?

6. **Core Idea**: Instruction-following capability is a double-edged sword—when the input data resembles an instruction, LLMs fail to prioritize correctly. This is a neglected yet systematically critical vulnerability in practical applications.

## Method

### Overall Architecture

DIM-Bench (Distractive Instruction Misunderstanding Benchmark) is a benchmark containing 2000 evaluation instances. It organizes "instructional distraction" scenarios into a two-dimensional matrix: 4 types of **instructional tasks** (rewriting/proofreading/translation/style transfer) $\times$ 5 types of **input tasks** (reasoning/code generation/math reasoning/bias detection/question answering), resulting in 20 categories with 100 instances each. It evaluates whether the output of LLMs follows the user's instructions (rather than being distracted by the implicit instructions in the input content).

### Key Designs

1. **Instructional Task Selection (4 Types)**:
    - **Function**: Select rewriting, proofreading, translation, and style transfer as the instructional tasks.
    - **Mechanism**: Design 8-10 template prompts for each task category to increase diversity. For example, the translation task covers target languages such as Chinese, Spanish, French, Arabic, Portuguese, Hindi, and Italian.
    - **Design Motivation**: These four are the most common task types in LLM data processing, and they have clear output formats (e.g., translational outputs should be in the target language, and proofreading should correct grammar), making it easy to automatically evaluate whether the correct instruction was followed.

2. **Input Task Selection (5 Sources of Distraction)**:
    - **Function**: Select reasoning (ARC dataset), code generation (Code Alpaca), math reasoning (GSM8K + MATH), bias detection (BBQ), and QA (NarrativeQA) as input data.
    - **Mechanism**: These tasks naturally contain "instructional" elements—they include clear question/instruction formats. For example, the math problem "What's the total number of cartons?" itself reads like a prompt asking the LLM to calculate the answer. The inputs for QA tasks are particularly long (averaging 743-904 tokens), which increases the distance between the instructions and the inputs.
    - **Design Motivation**: To cover different distraction intensities—QA tasks (directly containing questions) introduce the strongest distraction, while code/math tasks (containing programming/calculation requests) pose moderate-level distraction.

3. **Dual-Evaluation Method**:
    - **Function**: Combine LLM-as-a-judge evaluation and length-difference-based automatic evaluation.
    - **Mechanism**: The LLM-judge decomposes the evaluation into 2-3 binary (yes/no) sub-questions. For example, the evaluation question for translation + reasoning is: "Is the text in French? Does it preserve the multiple-choice format? Are the original options deleted?" All sub-questions must pass to be considered correct. The length difference evaluation is used for QA tasks—if the model correctly executes instructions like rewriting or translation, the output length should be close to the input; if the model answers the questions instead, the output will be much shorter than the input.
    - **Design Motivation**: Since the LLM-judge might have evaluation bias, length evaluation provides objective support. In most failure cases, outputs are concentrated in 0-200 tokens (having answered the question), whereas inputs are typically over 800 tokens.

## Key Experimental Results

### Main Results: Performance of 6 LLMs on DIM-Bench

| Model | Reasoning | Code | Math | Bias Detection | QA | Average |
|------|------|------|------|---------|-----|------|
| Llama-3.1-8B-Inst | 0.13 | 0.24 | 0.39 | 0.05 | 0.00 | 0.16 |
| Llama-3.1-70B-Inst | 0.47 | 0.64 | 0.81 | 0.29 | 0.02 | 0.45 |
| Qwen-2.5-7B-Inst | 0.65 | 0.66 | 0.79 | 0.19 | 0.03 | 0.46 |
| GPT-3.5 | 0.38 | 0.73 | 0.73 | 0.10 | 0.19 | 0.43 |
| GPT-4o-mini | 0.73 | 0.73 | 0.89 | 0.39 | 0.05 | 0.56 |
| GPT-4o | 0.61 | 0.67 | 0.82 | 0.24 | 0.02 | 0.47 |

*The table above shows the accuracy of each input task under the translation instruction (data extracted from the translation column of Table 3 in the paper).*

### Average Accuracy Across Different Instructional Task Dimensions

| Instruction Type | Average Accuracy | Explanation |
|---------|-----------|------|
| Rewriting | 0.397 | Moderate distraction resistance |
| Proofreading | 0.458 | Moderate distraction resistance |
| Translation | 0.526 | Best distraction resistance (large difference in output format) |
| Style Transfer | 0.301 | Worst (output format is highly similar to input format) |

### Effectiveness of Mitigation Strategies (Llama-3.1-70B, Translation Task)

| Method | Reasoning | Code | Math | Bias | QA |
|------|------|------|------|------|-----|
| Standard Prompt | 0.70 | 0.82 | 0.92 | 0.44 | 0.00 |
| DIRECT Prompt (explicitly asking to ignore input instructions) | 0.75 | 0.82 | 0.96 | 0.44 | **0.13** |
| CoT Prompt | 0.72 | 0.83 | 0.96 | 0.40 | 0.02 |
| Suffix Instruction (instruction placed after input) | 0.67 | 0.08 | 0.72 | 0.44 | 0.08 |

### Key Findings

- **QA inputs are the most fatal**: Almost all models achieved near-zero accuracy on QA inputs; they answered the questions upon seeing them, completely ignoring translation/rewriting instructions. Manual validation confirmed that the vast majority of failures occurred because the models directly answered the QA questions.
- **Longer inputs increase susceptibility to distraction**: Testing on QA tasks grouped by length showed that accuracy was 0.28-0.31 at an average of 362 tokens, but dropped to 0.02-0.05 when input length hit 3007 tokens. The greater the distance between the instruction and the input question, the more likely the model is to "forget" the instruction.
- **Style transfer is the most vulnerable**: Because the output format of style transfer is highly similar to the input format (both being natural language texts), the model has a harder time distinguishing between "executing the instruction" and "responding to the input".
- **Suffix instructions perform even worse**: Placing instructions after the input (suffix instruction) led to lower performance on most tasks, with code generation tumbling from 0.82 to 0.08, highlighting a strong positional effect.

## Highlights & Insights

- **Precise and practical problem definition**: Instructional distraction is not just a theoretical concern; it occurs daily during batch data processing. Anyone using LLMs for data cleaning, translation, or rewriting will encounter this problem, yet it had not been systematically studied prior to this work.
- **Ingenious benchmark design**: Organizing the benchmark as a two-dimensional matrix (4 instructions $\times$ 5 inputs) enables decoupled analysis of how instruction types and input types independently impact distraction. The length-difference evaluation leverages task properties to provide objective verification, reducing reliance on LLMs-as-a-judge.
- **Even GPT-4o fails**: The strongest model achieves an accuracy of only 0.00-0.07 on QA inputs, indicating that this is a fundamental limitation of the architecture/training paradigm rather than insufficient capability. This serves as a significant warning for applications like LLM-as-a-data-annotator.
- **Connection to Prompt Injection**: Instructional distraction can be viewed as an unintentional prompt injection—where non-malicious input data with instruction-like formatting disrupts model behavior. This offers a new perspective for security research.

## Limitations & Future Work

- **Limited coverage beyond "one-to-many" tasks**: One-to-many tasks like summarization (where output formats are highly variable) were excluded due to evaluation challenges, though the paper acknowledges they are similarly affected by instructional distraction.
- **Insufficient exploration of mitigation strategies**: The study only evaluates 3 prompting strategies and does not attempt training-level solutions (such as dedicated instruction priority training or hierarchical instruction tuning). Training schemes like the instruction hierarchy proposed by Wallace et al. (2024) might represent a more promising direction.
- **Evaluation reliance on GPT-4o as a judge**: Given that GPT-4o itself struggles on this benchmark, using it to evaluate outputs could introduce bias.
- **Lack of causal analysis for instructional distraction**: Is it a positional encoding issue in the attention mechanism, or a training bias introduced during instruction tuning? The paper lacks mechanistic analysis.
- **Directions for improvement**: Training datasets during SFT can incorporate examples containing instruction-like inputs that must be ignored; alternatively, inputs can be encapsulated as non-executable data blocks using structured generation or tool-use frameworks.

## Related Work & Insights

- **vs. Instruction Hierarchy (Wallace et al., 2024)**: The instruction hierarchy focuses on priority conflicts among multiple instructions, whereas instructional distraction features only one instruction combined with an input mistaken for an instruction. The two are complementary but differ mechanistically—the former asks "which one should be followed," while the latter centers on "is this even an instruction?"
- **vs. FollowBench / IFEval**: These benchmarks evaluate instruction-following capabilities under multiple constraints or complex settings under the assumption of clean input data. DIM-Bench reveals that even with simple instructions, "toxic" input can lead to failure—an orthogonal dimension overlooked by existing benchmarks.
- **vs. Prompt Injection Research**: Prompt injection represents intentional attacks, whereas instructional distraction occurs naturally. However, their defense mechanisms could overlap—specifically on how to enable models to "only follow system/user instructions while treating input as raw data."
- **Applicability as a test suite for LLM data pipelines**, leveraging DIM-Bench to evaluate model suitability for batch data processing tasks prior to deployment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to systematically define and evaluate the instructional distraction problem, filling an important gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated across 6 models, 20 combinations, multiple evaluation metrics, and mitigation strategies, though training-level solutions are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Well-defined problem formulation, intuitive examples, and logically organized data.
- **Value**: ⭐⭐⭐⭐ Provides direct guidance for practical LLM-based data processing, revealing a major systematic blind spot in instruction following.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs](can_llms_understand_unvoiced_speech_exploring_emg-to-text_conversion_with_llms.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] Biased LLMs Can Influence Political Decision-Making](biased_llms_can_influence_political_decision-making.md)
- [\[ACL 2025\] LLMs can Perform Multi-Dimensional Analytic Writing Assessments](llm_writing_assessment.md)
- [\[ACL 2025\] Enough Coin Flips Can Make LLMs Act Bayesian](coin_flips_bayesian.md)

</div>

<!-- RELATED:END -->
