---
title: >-
  [Paper Note] JsonTuning: Towards Generalizable, Robust, and Controllable Instruction Tuning
description: >-
  [ACL 2025][LLM Alignment][instruction tuning] This paper proposes JsonTuning—a method that replaces natural language text in instruction tuning inputs and outputs with structured JSON formats. By explicitly representing task elements, relationships, and output constraints (via JSON Schema), it consistently outperforms traditional TextTuning across 7 pre-trained models and 6 task categories, improving average performance from 26.78 to 30.88 while significantly enhancing robust…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "instruction tuning"
  - "JSON format"
  - "structured output"
  - "generalization"
  - "controllability"
date: 2026-05-08
content_hash: c01e79ff2c30db71
---

# JsonTuning: Towards Generalizable, Robust, and Controllable Instruction Tuning

**Conference**: ACL 2025  
**arXiv**: [2310.02953](https://arxiv.org/abs/2310.02953)  
**Code**: [https://github.com/gao-xiao-bai/JsonTuning](https://github.com/gao-xiao-bai/JsonTuning)  
**Area**: Alignment RLHF / Instruction Tuning  
**Keywords**: instruction tuning, JSON format, structured output, generalization, controllability

## TL;DR
This paper proposes JsonTuning—a method that replaces natural language text in instruction tuning inputs and outputs with structured JSON formats. By explicitly representing task elements, relationships, and output constraints (via JSON Schema), it consistently outperforms traditional TextTuning across 7 pre-trained models and 6 task categories, improving average performance from 26.78 to 30.88 while significantly enhancing robustness and controllability.

## Background & Motivation

**Background**: Standard instruction tuning (TextTuning) serializes the inputs and outputs of all tasks into natural language text, which the model learns through a text-to-text generation paradigm. This method aligns with the language modeling objective of LLM pre-training and represents the current mainstream approach.

**Limitations of Prior Work**:
   - **Poor Generalization**: TextTuning mixes task elements (questions, options, labels, etc.) and instructions within natural language text. Models easily memorize specific text templates instead of understanding the underlying task logic, leading to insufficient generalization on unseen tasks.
   - **Poor Robustness**: The ambiguity of natural language makes models highly sensitive to changes in prompt phrasing, label formats, or option order; minor reformulations can lead to performance degradation.
   - **Poor Controllability**: It is difficult to precisely describe or enforce specific output structures (such as nested objects, arrays, or type constraints) in natural language, which often results in model outputs that do not meet the expected format.

**Key Challenge**: The flexibility of natural language is both a strength and a weakness. This flexibility leads to ambiguous training signals, making it difficult for models to distinguish between "task logic" and "textual phrasing style," causing large performance fluctuations when encountering different expressions.

**Goal**: Focus the model on the task logic itself rather than the textual templates by introducing explicit structured representations, while also providing precise control over the output format.

**Key Insight**: JSON naturally possesses a key-value paired structure that can explicitly label the semantic role of each information segment (field name = semantic label), and JSON Schema can precisely define output constraints (types, nested structures, formatting requirements).

**Core Idea**: Replace the standard text-to-text paradigm of instruction tuning with a JSON-to-JSON (structure-to-structure) paradigm, teaching the model "what to do" instead of "how to say it."

## Method

### Overall Architecture
Convert standard instruction tuning from text-to-text to structure-to-structure:
- **Input** $S_I$: A JSON structure containing `input` (key-value pairs of task elements + `instruction` text) and `output control` (JSON Schema defining output constraints).
- **Output** $S_O$: A JSON structure containing key-value pairs of task output elements.
- **Training**: Fine-tuning with LoRA on 50K Flan 2022 samples + 10K InstructUIE structured task samples for 3 epochs.
- **Inference**: Greedy decoding, where both input and output are in JSON format.

### Key Designs

1. **Unified JSON Input Representation**:

    - **Function**: Unifies the inputs of all tasks into a JSON format where each field name clearly labels its semantic role.
    - **Mechanism**: For MCQA tasks, while TextTuning takes `"Answer the following question: Who is CEO? (A) Sundar..."` as input, JsonTuning uses `{"input": {"question": "Who is CEO?", "options": "(A) Sundar...", "instruction": "..."}, "output control": {"answer": {"type": "string"}}}`. Field names like `question` and `options` explicitly label the semantic roles of each text segment.
    - **Design Motivation**: Eliminate the confusion between task elements and instructions in natural language. The model no longer needs to "guess" which part is the question and which part is the options from continuous text, but can retrieve them directly from the structure. This reduces dependency on specific prompt templates and enhances generalization.

2. **Output Control (JSON Schema Control Information)**:

    - **Function**: Explicitly describes the expected output structure—field names, types (string/array/object), and nested structures—within the input using JSON Schema.
    - **Mechanism**: Taking language detection as an example, the control information is `{"language": {"type": "string"}, "probability scores": {"type": "object", "properties": {"French": {"type": "number"}, ...}}}`. This precisely defines which fields the output should contain and what their types are.
    - **Design Motivation**: (1) Enhance controllability: JSON Schema is more precise than natural language descriptions, and models can learn the mapping of Schema $\rightarrow$ structured output. (2) Improve generalization to new structures: By learning the composition rules of basic components (string/array/object) in Schema, the model can generate correct outputs even for complex, unseen structures during training. (3) Increase training consistency: Different tasks may require different output structures, and the control information unifies the method of "instructing the model which format to output."

3. **Structured Task Data Augmentation**:

    - **Function**: Incorporates information extraction (NER + RE) tasks from InstructUIE on top of Flan 2022, enabling the model to learn complex output structures during training.
    - **Mechanism**: The outputs in Flan 2022 are almost entirely plain text (string), lacking arrays and nested objects. By including NER/RE tasks, the model learns to generate complex JSON outputs containing arrays and objects. During evaluation, unseen Event Extraction (EE) tasks are used to test structural generalization.
    - **Design Motivation**: Without training on complex structures, the model cannot generalize to them. By introducing a modest amount of complex structural tasks during training, the model can learn the composition rules of basic structural components.

### Loss & Training
- LoRA fine-tuning with a peak learning rate of 1e-3, using AdamW + linear decay.
- Maximum sequence length is 2048 tokens.
- Multiple prompts are used for each task (e.g., 10 manually crafted prompts for NER/RE) to increase training diversity.
- JsonTuning and TextTuning are trained on exactly the same data to ensure a fair comparison, with the only difference being the data format.

## Key Experimental Results

### Main Results
Zero-shot generalization results across 7 models and 6 task categories:

| Model | TextTuning Avg | JsonTuning Avg | Gain |
|------|---------------|---------------|------|
| Falcon-7B | 12.37 | **17.64** | +5.27 |
| Mistral-7B | 30.95 | **35.74** | +4.79 |
| LLaMA-7B | 22.80 | **27.06** | +4.26 |
| LLaMA-13B | 27.79 | **31.10** | +3.31 |
| LLaMA2-7B | 26.29 | **29.19** | +2.90 |
| LLaMA2-13B | 30.27 | **33.47** | +3.20 |
| LLaMA3-8B | 37.01 | **41.96** | +4.95 |
| **Average (All)** | **26.78** | **30.88** | **+4.10** |

The performance boost in structured tasks is particularly significant: NER average increased from 37.51 to 45.28, and EE (unseen complex structural tasks) improved from near 0 to 4.83/10.17.

### Ablation Study

| Testing Dimension | TextTuning | JsonTuning | Description |
|---------|-----------|-----------|------|
| Prompt Robustness | High variance | **Low variance + High mean** | Json model exhibits small performance fluctuations across 10 prompts |
| Label Robustness (MMLU) | Large drop for unseen labels | **Maintained for unseen labels** | Performance remains stable when replacing with {W,X,Y,Z} or {$,€,£,¥} |
| EE Structural Generalization | Text receives almost score of 0 | **Json generates valid structures** | Training on NER+RE alone generalizes to more complex EE |
| Output Format Control | Unable to control precisely | **Controllable via Schema** | Case study on nested outputs of language detection + probability scores |

### Key Findings
- **Weaker models benefit more**: Falcon-7B achieved a +5.27 gain, indicating that JsonTuning is more helpful for models with weaker capabilities—the structured format reduces the difficulty of understanding task instructions.
- **Most significant improvement in structured tasks**: NER gained +7.77, and EE went from near zero to usable, demonstrating that the JSON format is naturally suited for tasks requiring structured outputs.
- **Robustness improvement is an extra benefit, not a trade-off**: JsonTuning reduces variance across prompts while improving average performance.
- **Control information is key to controllability**: Removing the output control leads to a significant performance drop in complex structural tasks.
- **Acceptable token overhead for JsonTuning**: Although the JSON format introduces additional tokens (brackets, quotes, field names) compared to plain text, the performance improvement far outweighs this overhead.

## Highlights & Insights
- **Extremely simple yet effective modification**: It requires no changes to the model architecture or the training algorithm, only the data format. Yet, it consistently improves performance across 7 models—a prime example of low-investment, high-return research.
- **Highly aligned with LLM ecosystem trends**: Major players like OpenAI and Anthropic are promoting structured output / JSON mode. This work validates the rationality of this direction from an academic perspective.
- **JSON Schema design for Output Control**: Explicitly encoding output constraints into the input enables the model to understand not just "what task to do" but also "what format to output," serving as a reusable design pattern.
- **Finding that weaker models benefit more**: This suggests that structured formats reduce the cognitive load of instruction comprehension, offering practical value for resource-constrained scenarios (e.g., small model deployment).

## Limitations & Future Work
- **Token overhead of JSON**: Extra tokens like brackets, quotes, and field names increase the input length, which might pose a challenge for models with limited context windows.
- **Manual design of JSON Schema**: Designing appropriate JSON Schemas for each task increases the cost of data preparation.
- **Only tested on LoRA fine-tuning**: The effectiveness under full-parameter fine-tuning or on larger scale models (70B+) has not been verified.
- **Limited training data scale**: Utilizing only 60K samples leaves it unknown whether the advantages of JsonTuning persist under larger data regimes.
- **Insufficient evaluation on open-ended generation**: The evaluation primarily focused on structured tasks and multiple-choice questions, leaving the performance on free-form text generation (e.g., dialogue, creative writing) insufficiently tested.

## Related Work & Insights
- **vs TextTuning (Flan, T0)**: Traditional text-to-text instruction tuning. This paper proves that simply altering the data format can yield consistent improvements, indicating that the representation format of training data has been underestimated.
- **vs Structured Output (OpenAI JSON mode)**: While the industry promotes JSON output modes through inference-time constrained decoding, this work addresses it from the training end—allowing the model to learn structured thinking during the fine-tuning phase.
- **vs InstructUIE**: InstructUIE designs specific instruction formats for IE tasks. This work generalizes structured concepts to all task categories.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea is clean and intuitive. Although the JSON format itself is not a technical innovation, systemically validating its advantages is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison across 7 models × 6 task categories × 3 dimensions (generalization, robustness, controllability).
- Writing Quality: ⭐⭐⭐⭐ Fair and clear experimental setups with intuitive case studies.
- Value: ⭐⭐⭐⭐ A plug-and-play data representation improvement that aligns with industry trends and holds strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Rethinking Table Instruction Tuning](rethinking_table_instruction_tuning.md)
- [\[ACL 2025\] Call for Rigor in Reporting Quality of Instruction Tuning Data](call_for_rigor_in_reporting_quality_of_instruction_tuning_data.md)
- [\[ACL 2025\] Measuring Data Diversity for Instruction Tuning: A Systematic Analysis and A Reliable Metric](measuring_data_diversity_for_instruction_tuning_a_systematic_analysis_and_a_reli.md)
- [\[ACL 2025\] Upcycling Instruction Tuning from Dense to Mixture-of-Experts via Parameter Merging](upcycling_instruction_tuning_from_dense_to_mixture-of-experts_via_parameter_merg.md)
- [\[ACL 2025\] Federated Data-Efficient Instruction Tuning for Large Language Models](federated_data-efficient_instruction_tuning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
