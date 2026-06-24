---
title: >-
  [Paper Note] Chart-based Reasoning: Transferring Capabilities from LLMs to VLMs
description: >-
  [ACL 2025][VLM Reasoning][Chart QA] This paper proposes a method to transfer the reasoning capabilities of LLMs to VLMs. Through improved chart representation pre-training, construction of large-scale synthetic reasoning datasets, and multi-task fine-tuning, the 5B-parameter PaLI-3 model outperforms models 10 times its size on ChartQA.
tags:
  - "ACL 2025"
  - "VLM Reasoning"
  - "Chart QA"
  - "knowledge distillation"
  - "ChartQA"
  - "reasoning enhancement"
date: 2026-05-08
content_hash: 9b842daa36ad5929
---

# Chart-based Reasoning: Transferring Capabilities from LLMs to VLMs

**Conference**: ACL 2025  
**arXiv**: [2403.12596](https://arxiv.org/abs/2403.12596)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Chart QA, VLM reasoning, knowledge distillation, ChartQA, reasoning enhancement

## TL;DR
This paper proposes a method to transfer the reasoning capabilities of LLMs to VLMs. Through improved chart representation pre-training, construction of large-scale synthetic reasoning datasets, and multi-task fine-tuning, the 5B-parameter PaLI-3 model outperforms models 10 times its size on ChartQA.

## Background & Motivation

**Background**: Vision-Language Models (VLMs) have achieved increasingly better performance on multimodal tasks, but their reasoning capabilities—especially regarding numerical computation and multi-step logic—remain limited. In contrast, the reasoning capabilities of Large Language Models (LLMs) have substantially improved through techniques like Chain-of-Thought. Chart Question Answering (Chart QA) is a typical multimodal task requiring complex reasoning: it requires extracting information from visuals and then performing numerical calculations or logical reasoning.

**Limitations of Prior Work**: Small VLMs (e.g., PaLI-3 5B) exhibit significantly weaker reasoning capabilities on tasks like ChartQA compared to large models (e.g., PaLI-X 55B), yet large models incur high inference costs. Chen et al. (2023c) noted that PaLI-3 lags behind PaLI-X on ChartQA, which is likely due to insufficient reasoning capability. Meanwhile, existing VLM training pipelines lack specialized pre-training tasks for chart understanding.

**Key Challenge**: The representation capability of small VLMs may be sufficient to understand visual elements of charts, but they lack multi-step reasoning abilities. Conversely, LLMs possess strong reasoning capabilities but cannot directly process visual input. How to combine the strengths of both?

**Goal**: Devise an efficient training recipe to transfer LLM reasoning capabilities to small VLMs, allowing them to match or exceed the performance of large models on chart question-answering tasks.

**Key Insight**: Convert charts into structured tables to serve as a bridge between LLMs and VLMs—LLMs generate reasoning traces on the tables, which are then used to train the VLM.

**Core Idea**: Enable small VLMs to inherit LLM reasoning capabilities through a three-step formulation: "improved pre-training + synthetic reasoning data + multi-task fine-tuning".

## Method

### Overall Architecture
The overall pipeline consists of three stages: (1) continued pre-training stage, using an improved chart-to-table translation task to enhance the VLM's internal representation of charts; (2) synthetic data construction stage, leveraging an LLM to generate reasoning traces based on the chart's tabular representation, constructing a training set 20 times larger than the original dataset; (3) multi-task fine-tuning stage, utilizing the multi-task framework of Hsieh et al. (2023) to simultaneously train the model on answer generation and reasoning process generation.

### Key Designs

1. **Chart Representation Pre-training**:

    - **Function**: Improve the VLM's understanding of associations between visual chart elements (colors, lines, positions) and textual content (legends, units).
    - **Mechanism**: Append a chart-to-table translation task to the pre-training stage of PaLI-3—given a chart image, the model is required to output the corresponding structured table text. Compared to the original version by Liu et al. (2023a), this approach improves the standardization of table formatting and error handling, enhancing the quality of the training data.
    - **Design Motivation**: The explicit chart-to-table translation task forces the model to learn the precise mapping from visual chart elements to structured data, providing a better internal representation for subsequent reasoning.

2. **Synthetic Reasoning Traces**:

    - **Function**: Construct training data containing intermediate reasoning steps to teach the VLM multi-step reasoning.
    - **Mechanism**: First, convert charts into tables, then input the tables and questions into an LLM (such as PaLM-2) to let the LLM generate detailed reasoning traces (including each step of information extraction, numerical calculation, and logical inference). In this manner, the original ChartQA training set is expanded by approximately 20 times. The reasoning traces describe each reasoning step in natural language, such as "First, locate the value Z in column X corresponding to row Y from the table, and then calculate..."
    - **Design Motivation**: Direct learning of chart-to-answer mapping by VLMs lacks intermediate reasoning supervision. Synthetic reasoning traces provide explicit reasoning paths, essentially "teaching the model how to think" rather than "only giving the answer".

3. **Multi-task Fine-tuning**:

    - **Function**: Simultaneously train the two tasks of answer prediction and reasoning process generation.
    - **Mechanism**: Based on the framework of Hsieh et al. (2023), two objectives are set for each training sample—directly outputting the answer, and outputting the reasoning process + answer. The two tasks share model parameters and are jointly trained using a multi-task loss $L = L_{answer} + \lambda L_{rationale}$. During inference, the model can either output only the answer (maintaining the same inference speed as PaLI-3) or output the reasoning process followed by the answer.
    - **Design Motivation**: Compared to the serial approach of distilling and then predicting, multi-task parallel training is more efficient, and the reasoning process as an auxiliary task regularizes answer prediction to avoid overfitting.

### Loss & Training
A multi-task cross-entropy loss is used, combining the answer prediction and reasoning process generation tasks with fixed weights. Additionally, program-of-thought prompting can be selectively used during the inference phase to further refine numerical reasoning results.

## Key Experimental Results

### Main Results

| Model | Parameters | ChartQA-Human | ChartQA-Aug | PlotQA | FigureQA |
|------|--------|---------------|-------------|--------|----------|
| PaLI-3-5B | 5B | 33.8 | 65.3 | - | - |
| PaLI-X-55B | 55B | 57.6 | 79.9 | - | - |
| Gemini Ultra | - | 63.3 | 80.8 | - | - |
| GPT-4V | - | 60.3 | 78.1 | - | - |
| **ChartPaLI-5B** | 5B | **64.2** | **82.5** | **73.1** | **63.2** |
| ChartPaLI-5B + PoT | 5B | **66.7** | **83.1** | - | - |

### Ablation Study

| Configuration | ChartQA-Human | ChartQA-Aug | Description |
|------|---------------|-------------|------|
| Full ChartPaLI | 64.2 | 82.5 | All components |
| w/o Continued Pre-training | 56.8 | 76.3 | Chart representation pre-training contributes +7.4 |
| w/o Synthetic Reasoning Data | 52.1 | 71.8 | Synthetic data contributes the most (+12.1) |
| w/o Multi-task Framework (Answers Only) | 60.3 | 79.6 | Multi-task improvement +3.9 |
| Original Data Volume (1x) | 58.5 | 78.2 | 20x data volume improvement +5.7 |

### Key Findings
- Synthetic reasoning data contributes the most (+12.1 points), validating that "teaching the model how to reason" is significantly more effective than "only providing answers".
- Continued chart pre-training contributes significantly (+7.4 points), indicating that robust internal representations of charts are a prerequisite for reasoning.
- The 5B-parameter ChartPaLI outperforms models like the 55B PaLI-X and Gemini Ultra, achieving an 11x increase in parameter efficiency.
- Program-of-thought yields an additional 2.5-point improvement on numerical computation-heavy questions, but offers limited help for qualitative questions.

## Highlights & Insights
- The method to transfer LLM reasoning capabilities to VLMs is highly clever: using tables as a bridging modality, it allows the LLM to generate reasoning traces within the pure text domain, which are then used to train the VLM. This approach can be generalized to other multimodal tasks requiring reasoning.
- The multi-task framework is more efficient than standard distillation, because the reasoning process and answers share underlying representations, mutually reinforcing each other.
- The 20x data augmentation strategy demonstrates that with sufficient training data, smaller models can fully match the performance of larger models.

## Limitations & Future Work
- The method relies on the specific architectures of PaLI-3 and PaLM-2; its applicability to open-source VLMs (such as LLaVA) remains to be validated.
- The intermediate chart-to-table step introduces information loss and may not be suitable for complex charts (such as those with multiple Y-axes or nested charts).
- The upper bound of synthetic reasoning trace quality is constrained by the capability of the teacher LLM; erroneous reasoning traces could potentially mislead the student model.
- End-to-end reasoning distillation methods can be explored to eliminate the explicit intermediate chart-to-table step.

## Related Work & Insights
- **vs MatCha (Liu et al., 2023)**: MatCha uses math reasoning pre-training to enhance chart understanding; this paper further introduces synthetic reasoning traces and a multi-task framework.
- **vs DePlot (Liu et al., 2023)**: DePlot focuses on chart-to-table translation; this paper adds reasoning capability transfer on top of it.
- **vs Hsieh et al. (2023) Distilling Step-by-Step**: This work is the first to apply their multi-step distillation framework to multimodal tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The pipeline design for reasoning transfer is novel, although key components are mostly combinations of existing technologies.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple datasets, extensive comparisons, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, with a particularly substantial experimental section.
- Value: ⭐⭐⭐⭐ Highly valuable reference for both chart understanding and reasoning enhancement in small models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VisRes Bench: On Evaluating the Visual Reasoning Capabilities of VLMs](../../CVPR2026/vlm_reasoning/visres_bench_on_evaluating_the_visual_reasoning_capabilities_of_vlms.md)
- [\[ACL 2025\] Judging the Judges: Can Large Vision-Language Models Fairly Evaluate Chart Comprehension and Reasoning?](judging_the_judges_can_large_vision-language_models_fairly_evaluate_chart_compre.md)
- [\[ACL 2026\] ShredBench: Evaluating the Semantic Reasoning Capabilities of Multimodal LLMs in Document Reconstruction](../../ACL2026/vlm_reasoning/shredbench_evaluating_the_semantic_reasoning_capabilities_of_multimodal_llms_in_.md)
- [\[ICCV 2025\] ChartPoint: Guiding MLLMs with Grounding Reflection for Chart Reasoning](../../ICCV2025/vlm_reasoning/chartpoint_guiding_mllms_with_grounding_reflection_for_chart_reasoning.md)
- [\[NeurIPS 2025\] SSR: Enhancing Depth Perception in VLMs via Rationale-Guided Spatial Reasoning](../../NeurIPS2025/vlm_reasoning/ssr_enhancing_depth_perception_in_vision-language_models_via_rationale-guided_sp.md)

</div>

<!-- RELATED:END -->
