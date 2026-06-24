---
title: >-
  [Paper Note] InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning
description: >-
  [ACL 2025][Segmentation][Part Segmentation] This work proposes InstructPart, the first real-world benchmark that combines task-oriented instructions with part-level segmentation, comprising 2,400 images, 48 object categories, 44 part categories, and 9,600 human-annotated task instructions. Evaluation shows that current VLMs are severely inadequate in instruction-driven part segmentation, while a baseline fine-tuned on LISA+DINOv2 achieves an approximate 100% performance gain.
tags:
  - "ACL 2025"
  - "Segmentation"
  - "Part Segmentation"
  - "Task-Oriented"
  - "Instruction Reasoning"
  - "VLM"
  - "Affordance"
  - "Benchmark"
date: 2026-05-08
content_hash: 5fe049d6621b5537
---

# InstructPart: Task-Oriented Part Segmentation with Instruction Reasoning

**Conference**: ACL 2025  
**arXiv**: [2505.18291](https://arxiv.org/abs/2505.18291)  
**Code**: [https://zifuwan.github.io/InstructPart/](https://zifuwan.github.io/InstructPart/)  
**Area**: Segmentation / Vision-Language Models  
**Keywords**: Part Segmentation, Task-Oriented, Instruction Reasoning, VLM, Affordance, Benchmark

## TL;DR
This work proposes InstructPart, the first real-world benchmark that combines task-oriented instructions with part-level segmentation, comprising 2,400 images, 48 object categories, 44 part categories, and 9,600 human-annotated task instructions. Evaluation shows that current VLMs are severely inadequate in instruction-driven part segmentation, while a baseline fine-tuned on LISA+DINOv2 achieves an approximate 100% performance gain.

## Background & Motivation

**Background**: Large vision-language models (VLMs) have excelled in object-level understanding tasks, such as object detection, semantic segmentation, and referring segmentation. However, most models treat objects as indivisible wholes, ignoring their constituent parts.

**Limitations of Prior Work**: (1) Existing part segmentation datasets (PartImageNet/Pascal-Part/PACO) only provide object-part annotations without task instructions—making them unable to evaluate the capability to "reason about which part to use given a task"; (2) Robot/affordance datasets (UMD/AGD20K) have limited affordance categories, or only contain point annotations, or are derived from simulation; (3) Instructions do not directly mention the part names (e.g., saying "flush the toilet" instead of "press the toilet handle"), which requires reasoning.

**Key Challenge**: Although VLMs can understand language instructions and perform segmentation, current models are almost incapable of combining both capabilities for part-grain task reasoning.

**Goal**: (1) Build a benchmark linking task instructions and part segmentation; (2) Evaluate the part reasoning capabilities of existing VLMs; (3) Provide a baseline method.

**Key Insight**: Everyday household scenarios—given a kettle image and the instruction "pour some water," the model needs to reason that it should grasp the "handle" of the kettle and output the segmentation mask of the handle.

**Core Idea**: Upgrade part segmentation from a "recognition" task to a "reasoning" task, where the model must infer target parts from implicit instructions.

## Method

### Overall Architecture
InstructPart consists of a benchmark and a simple baseline. The core contributions lie in the dataset construction and comprehensive evaluation.

### Key Designs

1. **Two Evaluation Tasks**:

    - **TRPS (Task Reasoning Part Segmentation)**: Inputs a natural language task instruction and an image, then outputs the target part mask. The instructions **do not directly mention part names** (e.g., "Flush the toilet" instead of "Press the toilet handle"), requiring reasoning.
    - **ORPS (Oracle Referring Part Segmentation)**: Inputs a part name and an image to perform direct localization. It can include additional affordance information (e.g., "handle of the cup that can be held").
    - **Design Motivation**: TRPS evaluates the dual capability of reasoning and visual grounding; ORPS evaluates visual grounding alone as a controlled variable.

2. **Dataset Construction**:

    - 2,400 real-world images covering 48 categories of household everyday objects, evenly distributed.
    - 44 part categories, 30 affordance categories, and 37 action types.
    - Each image is annotated with 4 task instructions (human-written + GPT-4 paraphrased) and hand-labeled segmentation masks.
    - Affordances of two levels: low-level (operational actions like pull/push/twist) and high-level (functional actions like turn on/pick up).
    - Annotated by 6 experts, followed by GPT-4 polishing and manual verification for quality assurance.

3. **PISA Baseline**:

    - **Function**: An improved version based on LISA (LLaVA + SAM decoder).
    - **Mechanism**: Uses a frozen DINOv2 instead of the SAM encoder for feature extraction, fuses multi-layer DINOv2 features with linear layers, and retains SAM's mask decoder (alternating TransConv and upsampling for decoding).
    - **Design Motivation**: DINOv2 outperforms SAM in extracting part-level correspondences, nearly doubling the gIoU on TRPS after fine-tuning.

### Evaluation Metrics
- gIoU: Average of IoU over all images.
- cIoU: Cumulative Intersection over Union.
- P@50: Proportion of images with IoU > 0.5.
- P@50:95: Average precision across multiple thresholds.

## Key Experimental Results

### Main Results (TRPS Task, Human-Annotating Instructions)

| Method Type | Method | gIoU↑ | cIoU↑ | P@50↑ |
|-------------|--------|-------|-------|-------|
| Open-Vocabulary Segmentation | VLPart | 0.39 | 1.16 | 0.00 |
| Open-Vocabulary Segmentation | OVSeg | 22.44 | 14.11 | 15.33 |
| Referring Segmentation | G-SAM | 29.95 | 21.45 | 25.17 |
| Reasoning Segmentation | LISA | 32.11 | 30.25 | 30.00 |
| Reasoning Segmentation | MiniGPT-v2 | 26.29 | 19.46 | 24.00 |
| **Fine-tuned baseline** | **PISA** | **~60** | **~55** | **~55** |

(PISA achieves about a 2x improvement after fine-tuning)

### Comparison between ORPS and TRPS Tasks

| Model | ORPS gIoU | TRPS gIoU | Gap |
|-------|-----------|-----------|------|
| LISA | 34.46 | 32.11 | -2.35 |
| G-SAM | 34.33 | 29.95 | -4.38 |
| Average | 22.56 | 17.49 | -5.07 |

### Key Findings
- **Current VLMs are severely deficient in part-level reasoning**: The best-performing model, LISA, achieves only 32.11 gIoU on TRPS, significantly lower than typical performance in object-level referring segmentation.
- **Reasoning is the primary bottleneck**: There is an approximate 5-point gap between ORPS (with part names given) and TRPS (with task instructions given), showing that reasoning part names from instructions is an additional challenge.
- **VLPart almost fails on TRPS**: Open-vocabulary part segmentation models can barely handle complex instructions, yielding a gIoU of only 0.39.
- **Fine-tuning data has high quality**: Fine-tuning PISA with the InstructPart training set doubles the TRPS performance, demonstrating the effectiveness of the dataset.
- **GPT-4 paraphrased instructions are not always better**: Some models perform better under human-written instructions, suggesting that instruction diversity does not necessarily lead to better evaluation.

## Highlights & Insights
- **"Task $\rightarrow$ Part" reasoning paradigm**: Fills the gap in evaluating from object-level grounding to part-level task reasoning, which holds direct value for robotic manipulation and assistive technologies.
- **Implicit part names in instructions**: Intentionally omitting part names in instructions ("pour water" instead of "grasp handle") forces the model to perform genuine reasoning, which is closer to real interaction scenarios.
- **Two-level affordance annotation**: Fine-grained annotations with low-level operational actions and high-level functional actions, providing more detailed data for affordance research.

## Limitations & Future Work
- The scale of 2,400 images is relatively small, and 48 object categories may not cover all daily scenarios.
- Only one target object is annotated in each image, leaving multi-object scenarios unconsidered.
- The PISA baseline is relatively simple (LISA+DINOv2); stronger architectures (e.g., Grounding DINO v2 + SAM2) might have more room for improvement.
- The zero-shot performance of closed-source models like GPT-4V/Gemini on TRPS remains unevaluated.
- The dataset currently only includes static images, without involving dynamic part segmentation in video scenarios.

## Related Work & Insights
- **vs LISA**: LISA performs object-level reasoning segmentation, while InstructPart pushes it to the part level. LISA's gIoU on InstructPart is only 32, demonstrating that part reasoning is much more challenging than object reasoning.
- **vs VLPart**: VLPart performs open-vocabulary part segmentation but lacks instruction reasoning, almost failing on TRPS.
- **vs AGD20K**: AGD20K contains affordances but only with point annotations and no instructions. InstructPart provides full masks and task instructions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first benchmark to connect task instructions and part segmentation with practical problem definitions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated 10 models across 3 categories with an effective baseline, though the dataset size is somewhat small.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-defined tasks.
- Value: ⭐⭐⭐⭐⭐ Direct driving value for robotic manipulation and embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] General and Task-Oriented Video Segmentation](../../ECCV2024/segmentation/general_and_task-oriented_video_segmentation.md)
- [\[NeurIPS 2025\] InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition](../../NeurIPS2025/segmentation/instructsam_a_training-free_framework_for_instruction-oriented_remote_sensing_ob.md)
- [\[ECCV 2024\] PartSTAD: 2D-to-3D Part Segmentation Task Adaptation](../../ECCV2024/segmentation/partstad_2d-to-3d_part_segmentation_task_adaptation.md)
- [\[ACL 2025\] DEF-DTS: Deductive Reasoning for Open-domain Dialogue Topic Segmentation](def-dts_deductive_reasoning_for_open-domain_dialogue_topic_segmentation.md)
- [\[ACL 2025\] Pixel-Level Reasoning Segmentation via Multi-turn Conversations](pixel-level_reasoning_segmentation_via_multi-turn_conversations.md)

</div>

<!-- RELATED:END -->
