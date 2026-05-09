---
title: >-
  [Paper Note] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs
description: >-
  [ICCV 2025][Multimodal VLM][Multi-context visual grounding] This paper introduces Multi-Context Visual Grounding as a novel task and the MC-Bench benchmark—comprising 2,000 manually annotated samples, 3 text description styles, and 20 practical skills—to evaluate 20+ MLLMs and foundation models. It reveals a substantial performance gap between current models and humans (human AP50=41.3% vs. best end-to-end model AP50=30.7%), and provides an agentic baseline combining GPT-4o and G-DINO (AP50=36.2%).
tags:
  - ICCV 2025
  - Multimodal VLM
  - Multi-context visual grounding
  - multi-image reasoning
  - instance-level evaluation
  - MLLM benchmark
  - cross-image understanding
date: 2026-05-08
content_hash: de2544c171e09fa1
---

# MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs

**Conference**: ICCV 2025
**arXiv**: [2410.12332](https://arxiv.org/abs/2410.12332)
**Code**: [github.com/xuyunqiu/MC-Bench](https://github.com/xuyunqiu/MC-Bench)
**Area**: Multimodal VLM
**Keywords**: Multi-context visual grounding, multi-image reasoning, instance-level evaluation, MLLM benchmark, cross-image understanding

## TL;DR

This paper introduces Multi-Context Visual Grounding as a novel task and the MC-Bench benchmark—comprising 2,000 manually annotated samples, 3 text description styles, and 20 practical skills—to evaluate 20+ MLLMs and foundation models. It reveals a substantial performance gap between current models and humans (human AP50=41.3% vs. best end-to-end model AP50=30.7%), and provides an agentic baseline combining GPT-4o and G-DINO (AP50=36.2%).

## Background & Motivation

### Problem Definition

Multi-context visual grounding is a novel task: given multiple images and an open-ended text description, a model must localize all instances referred to by the text across the images (as bounding boxes). This requires simultaneous cross-image reasoning, open-vocabulary language understanding, and precise instance localization.

### Limitations of Prior Work

**Existing MLLMs focus on single-image tasks**: Most MLLMs with region grounding capabilities (e.g., Shikra, Ferret, Kosmos-2) handle only a single image, ignoring cross-image context.

**Multi-image benchmarks lack instance-level annotation**: Existing multi-image MLLM benchmarks (e.g., MMBench, MMMU, BLINK, MileBench) are image-level tasks (e.g., VQA) without instance-level bounding box annotations.

**Text descriptions are overly simplistic**: Traditional visual grounding tasks (e.g., REC, OVD) rely on category names or simple referring phrases, and do not support complex reasoning, comparison, or commonsense-based descriptions.

**Video models are unsuitable for non-sequential multi-image inputs**: Multi-frame models designed for video understanding (e.g., VideoLLMs) depend on temporal continuity and cannot handle spatially or semantically related but non-sequential multi-image scenarios.

### Core Motivation

**Key insight**: Real-world images are not isolated but are interconnected through spatial, temporal, or semantic context. Instance-level intelligence across multiple images is critical for applications such as autonomous driving (understanding pedestrians from multiple camera angles), security surveillance (cross-camera target tracking), and general AI assistants (chart analysis, GUI agents). Yet no benchmark systematically evaluates MLLM capability on multi-image instance-level tasks—a significantly underexplored research direction.

## Method

### Overall Architecture

MC-Bench comprises three core contributions:
1. **Task definition**: Multi-context visual grounding—localizing instances from multiple images given open-ended text.
2. **Dataset construction**: 2,000 manually annotated samples spanning 3 text styles and 20 skills.
3. **Systematic evaluation**: Benchmarking of 20+ models, with an agentic baseline, a fine-tuning baseline, and human evaluation.

### Key Designs

#### 1. **Task Formulation: Multi-Context Visual Grounding**

- **Function**: Defines a unified multi-image instance-level vision-language task.
- **Mechanism**:
    - **Input**: Multiple images (currently image pairs) + one open-ended text description.
    - **Output**: Bounding boxes for all instances described by the text, including correct grouping information.
    - **Three text styles**:
        - **Referring**: Directly or indirectly describes instances via category, attribute, or positional information (346 samples).
        - **Comparison**: Localizes instances through cross-image comparisons (quantity, color, size, etc.) (810 samples).
        - **Reasoning**: Requires commonsense knowledge or multi-hop reasoning for localization (844 samples).
    - **One-to-many matching**: A single description may correspond to multiple groups of instances of different categories.
    - **Negative samples**: 288 samples in which the text description does not refer to any instance in the images, requiring the model to abstain from prediction.

- **Design Motivation**: The referring style tests basic cross-image perception, the comparison style tests cross-image contrastive analysis, and the reasoning style tests deep semantic reasoning—together forming a progressive evaluation of MLLM multimodal understanding.

#### 2. **Dataset Construction Pipeline**

- **Function**: Constructs a high-quality, high-diversity evaluation dataset.
- **Mechanism**:
    - **Multi-source image collection**: Images are collected from 10+ datasets (MS-COCO, BLINK, Mantis-Eval, DocVQA, etc.) and web crawling, covering natural images, charts, documents, artworks, scientific diagrams, and more (3,345 images total).
    - **Text annotation**: Annotators select semantically related image pairs and compose open-ended text descriptions linking the two images.
    - **Instance annotation**: A separate group of annotators draws bounding boxes around the described instances in positive samples.
    - **Iterative review**: Text annotators verify the consistency of bounding box annotations; inconsistent samples are re-annotated.
    - An online annotation platform is built using Label Studio.
    - **Statistics**: 3,200 bounding boxes; text length ranges from 2 to 24 words (mean 7.2); instance count ranges from 1 to 17.

- **Design Motivation**: The quality of an evaluation benchmark directly determines its value. Multi-source images ensure domain diversity, manual annotation ensures gold-standard quality, and iterative review ensures text–instance consistency.

#### 3. **Evaluation Metric Design**

- **Function**: Designs a metric system that evaluates both image-level and instance-level performance.
- **Mechanism**:
    - **Image-level (Acc)**: Whether the model correctly identifies which images contain target instances (regardless of precise location).
    - **Instance-level (AP50)**: Whether the model accurately localizes target instances (IoU ≥ 0.5).
    - For samples with multiple instance groups, the Hungarian algorithm is applied for optimal group matching.
    - Results are reported separately for referring, comparison, and reasoning styles.

- **Design Motivation**: Image-level metrics evaluate "understanding" (which image contains the target), while instance-level metrics evaluate "localization" (where exactly the target is)—the two are complementary.

#### 4. **Agentic Baseline: GPT-4o + G-DINO**

- **Function**: Constructs a divide-and-conquer agent system as an upper-bound reference.
- **Mechanism**:
    - GPT-4o acts as the reasoning agent: (1) analyzes multi-image input and identifies which images contain the target; (2) generates concise referring expressions for each target instance.
    - G-DINO acts as the localization tool: detects targets in the corresponding images using the referring expressions generated by GPT-4o.
- **Design Motivation**: End-to-end models underperform on both multi-image understanding and precise localization. By combining reasoning expertise (GPT-4o) with localization expertise (G-DINO), a stronger baseline is established.

### Loss & Training

MC-Bench is an evaluation benchmark and does not involve training. The fine-tuning baseline uses the following setup:
- Base model: Qwen2-VL-7B
- Data: 50K+ multi-context instruction tuning samples (partly from existing datasets, partly synthesized)
- Method: LoRA fine-tuning
- Metrics: Acc (image-level) and AP50 (instance-level)

## Key Experimental Results

### Main Results

**Performance of various model types on MC-Bench**:

| Model Type | Representative Model | Acc↑ | AP50↑ |
|---|---|---|---|
| Closed-source general MLLM | GPT-4o | 78.3 | 2.8 |
| Closed-source general MLLM | Gemini-1.5 Pro | 62.5 | 28.2 |
| Open-source general MLLM | Qwen2-VL-72B (seq) | 71.4 | 30.7 |
| Open-source general MLLM | Qwen2-VL-7B (seq) | 54.9 | 19.1 |
| Open-source specialist MLLM | CogVLM-Grounding-17B | 48.5 | 17.5 |
| Open-source specialist MLLM | Ferret-13B | 44.8 | 12.9 |
| Foundation model (no LLM) | G-DINO-B | 30.8 | 15.0 |
| **Agentic baseline** | **GPT-4o + G-DINO** | **77.9** | **36.2** |
| Fine-tuning baseline | Qwen2-VL-7B + IT | 57.7 | 22.6 |
| **Human** | **—** | **92.3** | **41.3** |

**Breakdown by text style (Qwen2-VL-72B seq)**:

| Style | Acc | AP50 |
|---|---|---|
| Referring | 61.6 | 33.7 |
| Comparison | 79.1 | 33.2 |
| Reasoning | 68.0 | 27.0 |

### Ablation Study

**Effect of model scale (Qwen2-VL)**:

| Model Size | Acc | AP50 | Notes |
|---|---|---|---|
| 7B (seq) | 54.9 | 19.1 | — |
| 72B (seq) | 71.4 | 30.7 | Larger models perform better, consistent with scaling laws |

**Image input format**:

| Input Format | Acc (Qwen2-VL-7B) | AP50 | Notes |
|---|---|---|---|
| Sequential input | 54.9 | 19.1 | Preserves image-level independence |
| Concatenated input | 51.4 | 17.8 | Loses image boundary information |

**Negative sample rejection**: Most models still produce predictions on negative samples (Gemini performs best: avg. 0.42 predictions/negative sample vs. human 0.19).

**Instance grouping capability**: Most models predict only one group of instances and fail to correctly group multiple target sets.

### Key Findings

1. **GPT-4o understands well but localizes poorly**: Highest Acc (78.3%) but AP50 of only 2.8%, indicating that closed-source API models have severely limited bounding box output capability.
2. **Specialist MLLMs underperform general MLLMs**: Models trained for single-image grounding generalize poorly to multi-image scenarios; the 17B CogVLM-Grounding performs comparably to the 7B Qwen-VL.
3. **Agentic baseline outperforms all end-to-end models by a large margin**: GPT-4o + G-DINO AP50 (36.2%) substantially surpasses the best end-to-end model (30.7%), demonstrating the advantage of compositional approaches.
4. **Significant human–model gap**: Human Acc 92.3% vs. best model 78.3%; AP50 41.3% vs. 36.2%, with the gap widening on reasoning-style descriptions.
5. **Instance grouping is the primary bottleneck**: Removing the grouping requirement leads to substantial AP50 improvements across nearly all models, indicating that models can localize but cannot semantically group instances.
6. **Small object detection is challenging**: MLLM performance degrades sharply on small and medium-sized targets, while the agentic baseline shows a clear advantage on small objects via G-DINO.

## Highlights & Insights

1. **Fills an important gap**: The first multi-image instance-level MLLM benchmark, addressing the absence of instance-level annotations in existing multi-image benchmarks.
2. **Progressive task design**: The three styles—referring → comparison → reasoning—form a progressive evaluation from perception to reasoning.
3. **Negative sample design**: The ability to "say no" is an overlooked but important evaluation dimension.
4. **Insight from the agentic baseline**: The division of labor between reasoning and localization substantially outperforms end-to-end models, pointing toward a promising direction for multi-image MLLM development.
5. **Discovery of the grouping problem**: Reveals a previously unnoticed weakness of MLLMs—the inability to semantically group multiple detection results.

## Limitations & Future Work

1. **Limited scale**: Only 2,000 samples; some skill categories contain fewer than 50 samples, potentially limiting statistical significance.
2. **Restricted to image pairs**: The current setting uses two images; evaluation with larger numbers of images has not been conducted.
3. **Annotation subjectivity**: Open-ended text descriptions inherently involve annotator subjectivity; despite iterative review, different annotators may produce different descriptions for the same image pair.
4. **Latest models not evaluated**: Due to submission deadlines, some recent models (e.g., GPT-4.5, Claude) were not included in the evaluation.
5. **Limited exploration of training strategies**: The fine-tuning baseline is relatively simple (LoRA + 50K data); more systematic multi-context training strategies remain to be investigated.

## Related Work & Insights

- **vs. BLINK/MileBench**: These benchmarks evaluate image-level multi-image understanding (e.g., multiple-choice questions); MC-Bench requires instance-level localization.
- **vs. RefCOCO and related grounding benchmarks**: Traditional grounding benchmarks involve single images with simple descriptions; MC-Bench involves multiple images with open-ended descriptions.
- **Implication**: Multi-image instance-level understanding may require fundamentally new model architectures—the current practice of naively concatenating or sequentially inputting multiple images is insufficient for establishing effective cross-image associations.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Pioneers the multi-context visual grounding task, filling a significant research gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation of 20+ models, with human evaluation, agentic baseline, and fine-tuning baseline, along with in-depth multi-dimensional analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, thorough analysis, rich in figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ — As a benchmark dataset, it provides important guidance for advancing MLLM research on multi-image instance-level tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?](sti-bench_are_mllms_ready_for_precise_spatial-temporal_world_understanding.md)
- [\[ICCV 2025\] ChartPoint: Guiding MLLMs with Grounding Reflection for Chart Reasoning](chartpoint_guiding_mllms_with_grounding_reflection_for_chart_reasoning.md)
- [\[ICCV 2025\] From Easy to Hard: The MIR Benchmark for Progressive Interleaved Multi-Image Reasoning](from_easy_to_hard_the_mir_benchmark_for_progressive_interleaved_multi-image_reas.md)
- [\[ICCV 2025\] ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering](reasonvqa_a_multi-hop_reasoning_benchmark_with_structural_knowledge_for_visual_q.md)
- [\[ICCV 2025\] DOGR: Towards Versatile Visual Document Grounding and Referring](dogr_towards_versatile_visual_document_grounding_and_referring.md)

</div>

<!-- RELATED:END -->
