---
title: >-
  [Paper Note] ChartPoint: Guiding MLLMs with Grounding Reflection for Chart Reasoning
description: >-
  [ICCV 2025][VLM Reasoning][Chart Reasoning] This paper proposes PointCoT, which integrates reflective visual grounding (bounding boxes) into the chain-of-thought for chart reasoning, enabling MLLMs to interactively verify each reasoning step against the chart's visual content. It also constructs the ChartPoint-SFT-62k dataset containing 19.2K high-quality samples, achieving a +5.04% improvement on ChartBench.
tags:
  - "ICCV 2025"
  - "VLM Reasoning"
  - "Chart Reasoning"
  - "Multimodal Large Language Models"
  - "Chain-of-Thought"
  - "Visual Grounding"
  - "Numerical Hallucination"
date: 2026-05-08
content_hash: 4cee7dd9921b4f61
---

# ChartPoint: Guiding MLLMs with Grounding Reflection for Chart Reasoning

**Conference**: ICCV 2025
**arXiv**: [2512.00305](https://arxiv.org/abs/2512.00305)  
**Code**: None  
**Area**: Chart Understanding / Multimodal Reasoning
**Keywords**: Chart Reasoning, Multimodal Large Language Models, Chain-of-Thought, Visual Grounding, Numerical Hallucination

## TL;DR

This paper proposes PointCoT, which integrates reflective visual grounding (bounding boxes) into the chain-of-thought for chart reasoning, enabling MLLMs to interactively verify each reasoning step against the chart's visual content. It also constructs the ChartPoint-SFT-62k dataset containing 19.2K high-quality samples, achieving a +5.04% improvement on ChartBench.

## Background & Motivation

Multimodal large language models (MLLMs) heavily rely on OCR-extracted textual information for chart understanding. When chart text annotations are sparse (e.g., data points lack explicit value labels), models tend to produce severe numerical hallucinations—even when reasoning steps appear plausible, the extracted numerical values contain significant errors.

The authors identify a critical observation: **MLLMs exhibit extremely weak grounding capability on chart elements and proportional relationships**. When prompted to indicate which chart region corresponds to each reasoning step, models either ignore the request or generate entirely irrelevant coordinates. This reveals that:
- Conventional CoT enhances logic-based numerical reasoning but fails to improve the model's fundamental numerical perception
- CoT generates additional reasoning tokens without enabling further interaction with the chart's visual tokens
- Models lack the "look–point–read–compute" visual reasoning logic that humans employ when reading charts

This motivates the authors to incorporate grounding reflection into the reasoning chain: the model must not only articulate each reasoning step but also output a bounding box indicating which region of the chart it is attending to, verified through re-rendered chart images.

## Method

### Overall Architecture

The PointCoT data construction pipeline consists of four stages:
1. **Step Decomposition**: An LLM generates numerical questions and CoT reasoning steps
2. **Code Editing**: An LLM modifies the plotting code to insert special characters at key positions
3. **Code Rendering**: The modified code is executed to re-render the chart
4. **Position Localization**: OCR detects the positions of special characters and extracts bounding boxes

### Key Designs

#### 1. Structured Reasoning Construction
- **Function**: Decomposes the reasoning process for chart QA into two categories of steps: "Grounding" and "Reasoning"
- **Mechanism**: Qwen2.5-72B is used as the teacher model to generate data-point-related questions and step-by-step reasoning based on the plotting code. Each sub-step is classified as:
    - **Grounding steps**: Require extracting data from the chart (e.g., locating points on coordinate axes, reading legend entries)
    - **Reasoning steps**: Perform logical inference based on information from preceding grounding steps
- **Design Motivation**: The cognitive process of chart reading is inherently structured—humans first identify key locations before performing numerical reasoning. This structure is not artificially imposed but emerges from the intrinsic logic of chart comprehension

#### 2. Point Annotation via Code Editing
- **Function**: Generates precise bounding box annotations for each grounding step
- **Mechanism**: Rather than relying on unreliable direct MLLM localization, the approach leverages the chart–code correspondence:
  1. The teacher model identifies the chart element/position corresponding to each grounding step
  2. The plotting code is modified to insert special character markers at key positions via `plt.text()`
  3. The modified code is executed to render a new chart
  4. Multiple OCR tools detect the positions of special characters and extract bounding boxes
- **Design Motivation**: LLM-based code modification achieves substantially higher success rates than direct MLLM-based chart element localization; code serves as a reliable intermediary for precise positional annotation

#### 3. Four Instruction Data Formats
- **Type 1 – Standard VQA**: Original chart + question; supervision is the answer or CoT + answer (bounding boxes excluded to prevent data leakage)
- **Type 2 – Grounding Task**: Intermediate grounding steps are incorporated into the query prompt; ground truth is the predicted bounding box
- **Type 3 – Edited Chart Reasoning**: Bounding box annotations from preceding grounding steps are overlaid onto the original chart to guide the model toward the correct regions
- **Type 4 – Reasoning Steps**: Reasoning steps are directly incorporated into the query prompt; the final supervision signal is the final answer

The resulting ChartPoint-SFT-62k dataset comprises 19.2K charts × 62.3K instruction samples.

### Loss & Training

Two-stage full-parameter fine-tuning:
- **Stage 1 – Chart Knowledge Alignment**: Trained on MMC-Instruct (410K) + ChartGemma (160K) + ChartQA (28K) + ChartBench (30K)
- **Stage 2 – Chart-Specific Annealing Tuning**: Instruction fine-tuning with the PointCoT format using ChartPoint-SFT-62k

Training details: AdamW optimizer, warmup learning rate $5 \times 10^{-5}$, weight decay 0.1, gradient clipping 1.0, effective batch size 64, bfloat16 precision, approximately 262 GPU hours (A100-40G). Coordinates are normalized to the range $[0, 999]$.

## Key Experimental Results

### Main Results

ChartQA relaxed accuracy@0.05:

| Model | Params | Human | Aug. | Avg. |
|-------|--------|-------|------|------|
| Qwen2-VL | 7B | 72.08 | 94.24 | 83.16 |
| Qwen2.5-VL | 7B | 78.96 | 93.76 | 86.36 |
| ChartMoE+PoT | 8B | 78.32 | 90.96 | 84.64 |
| **ChartPoint_Q2** | **7B** | 76.12 | **94.48** | **85.28** |
| **ChartPoint_Q2.5** | **7B** | **81.36** | 94.12 | **87.74** |

ChartBench accuracy:

| Model | Regular | Extra | Overall |
|-------|---------|-------|---------|
| Qwen2-VL | 58.36 | 59.40 | 58.90 |
| Qwen2.5-VL | 62.73 | 57.26 | 60.91 |
| ChartMoE | 56.31 | 55.58 | 51.67 |
| **ChartPoint_Q2** | **63.04** | **62.09** | **62.61** |
| **ChartPoint_Q2.5** | **66.71** | **65.03** | **65.95** |

### Ablation Study

Training strategy ablation (based on Qwen2-VL):

| Configuration | ChartQA | ChartBench | Description |
|---------------|---------|------------|-------------|
| Baseline (Qwen2-VL) | 83.16 | 58.90 | Original model |
| +Stage1 | 83.74 | 60.39 | Chart knowledge alignment |
| +Stage1+CoT | 84.11 | 60.76 | Text CoT distillation |
| **+Stage1+PointCoT** | **85.30** | **62.61** | CoT with grounding |

Coordinate format ablation:

| Format | Normalization | Human | Overall | Description |
|--------|---------------|-------|---------|-------------|
| Type A | [0–1], 4 decimal places | 73.52 | 83.68 | Continuous coordinates |
| Type B | [0–1], 3 decimal places | 74.68 | 84.42 | Reduced precision |
| **Type C** | **[0–999], integer** | **75.36** | **84.84** | Tokenizer-friendly |

### Key Findings

1. PointCoT yields substantially larger gains on ChartBench (+3.71%/+4.28%) than on ChartQA (+1.56%/+1.22%), since ChartBench contains no data-point text annotations and thus places greater demands on visual grounding
2. Text CoT distillation offers limited improvement (+0.37%), as the reasoning is generated by an LLM rather than an MLLM and does not leverage the chart's visual information
3. Improvements are more pronounced on Extra-type charts (area charts, box plots, radar charts, etc.) (+7.77%), demonstrating the generalizability of the visual reasoning framework
4. Normalizing coordinates to $[0, 999]$ integers outperforms continuous floating-point representations, as the former is more compatible with the tokenizer

## Highlights & Insights

- **Critical Diagnosis**: The paper precisely identifies the core bottleneck in MLLM chart understanding—not insufficient logical reasoning, but weak visual perception (numerical value extraction)
- **Conceptual Innovation**: Incorporating "grounding reflection" into CoT ensures that each reasoning step can be validated against visual evidence, rather than relying on purely textual inference
- **Elegant Data Construction**: The chart–code correspondence is exploited to achieve precise positional annotation via indirect code modification, circumventing the unreliability of direct MLLM-based localization
- **Rigorous Quality Control**: Success rates are tracked at each stage (96%→76%→51%→77%), with a final expert review achieving 91% acceptance across three annotators

## Limitations & Future Work

- The data construction pipeline achieves a success rate of only approximately 28% (66.8K→19.2K), limiting scalability
- Coverage is restricted to three chart types: bar charts (57.1%), line charts (33.6%), and pie charts (9.3%)
- QA pairs focus on data point reading and do not address complex numerical computation or multi-step reasoning
- The approach depends on the grounding capability of the base model (Qwen2-VL/2.5-VL) and is not applicable to models without bounding box support
- Inference-time scaling (e.g., beam search over location or reasoning steps) is left unexplored

## Related Work & Insights

- **MVoT** observes a similar phenomenon in structured scenarios such as puzzles—reasoning steps must interact with visual inputs
- **ChartMoE** provides chart–code metadata as a foundation; this work innovatively leverages such data for positional annotation
- The proposed framework can be generalized to other reasoning tasks requiring precise visual perception, such as scientific figures, maps, and engineering diagrams

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to integrate visual grounding reflection into chart CoT, diagnosing a core MLLM bottleneck
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablations and cross-model validation, though dataset type coverage is limited
- Writing Quality: ⭐⭐⭐⭐ — Pipeline description is clear, though some notational inconsistencies remain
- Value: ⭐⭐⭐⭐⭐ — Establishes a new paradigm for the chart understanding community, shifting from "textual reasoning" to "visual reasoning"

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Chart-based Reasoning: Transferring Capabilities from LLMs to VLMs](../../ACL2025/vlm_reasoning/chart-based_reasoning_transferring_capabilities_from_llms_to_vlms.md)
- [\[CVPR 2026\] Chart-FR1: Visual Focus-Driven Fine-Grained Reasoning on Dense Charts](../../CVPR2026/vlm_reasoning/chart-fr1_visual_focus-driven_fine-grained_reasoning_on_dense_charts.md)
- [\[ICLR 2026\] Reasoning in Space via Grounding in the World](../../ICLR2026/vlm_reasoning/reasoning_in_space_via_grounding_in_the_world.md)
- [\[ICML 2026\] Temporal-Aware Reasoning Optimization for Video Temporal Grounding](../../ICML2026/vlm_reasoning/temporal-aware_reasoning_optimization_for_video_temporal_grounding.md)
- [\[ICCV 2025\] Taming the Untamed: Graph-Based Knowledge Retrieval and Reasoning for MLLMs to Conquer the Unknown](taming_the_untamed_graph-based_knowledge_retrieval_and_reasoning_for_mllms_to_co.md)

</div>

<!-- RELATED:END -->
