---
title: >-
  [Paper Note] O3SLM: Open Weight, Open Data, and Open Vocabulary Sketch-Language Model
description: >-
  [AAAI 2026][Multimodal VLM][Sketch understanding] This paper constructs a large-scale sketch-image-instruction triplet dataset, SketchVCL (600K pretraining + 215K fine-tuning samples), and trains O3SLM — the first open-source large vision-language model capable of fluently understanding hand-drawn sketches across four tasks: detection, counting, retrieval, and VQA — substantially outperforming existing LVLMs on all tasks.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Sketch understanding
  - large vision-language model
  - sketch-image-text alignment
  - open vocabulary
  - instruction tuning
date: 2026-05-08
content_hash: bd346843db5b06b4
---

# O3SLM: Open Weight, Open Data, and Open Vocabulary Sketch-Language Model

**Conference**: AAAI 2026
**arXiv**: [2511.14368](https://arxiv.org/abs/2511.14368)
**Code**: [Project Page](https://vcl-iisc.github.io/O3SLM/)
**Area**: Multimodal VLM
**Keywords**: Sketch understanding, large vision-language model, sketch-image-text alignment, open vocabulary, instruction tuning

## TL;DR
This paper constructs a large-scale sketch-image-instruction triplet dataset, SketchVCL (600K pretraining + 215K fine-tuning samples), and trains O3SLM — the first open-source large vision-language model capable of fluently understanding hand-drawn sketches across four tasks: detection, counting, retrieval, and VQA — substantially outperforming existing LVLMs on all tasks.

## Background & Motivation

**Background**: Large vision-language models (LVLMs) have achieved remarkable success on tasks such as VQA and document understanding, yet they rely almost exclusively on natural images and text. Hand-drawn sketches, as an intuitive visual communication medium, can effortlessly convey spatial layouts and shape information that is difficult to express in words, and transcend language barriers, making them a more universal communication tool.

**Limitations of Prior Work**: Existing open-source LVLMs (LLaVA, Qwen-VL, DeepSeek-VL2, etc.) almost completely fail to interpret rough hand-drawn sketches. As illustrated in Figure 1, even when a model can marginally recognize certain visual cues, it cannot leverage this information for downstream tasks such as detection or reasoning. Closed-source models (GPT-4o, Gemini) exhibit rudimentary sketch comprehension but suffer from weak multimodal grounding capabilities and remain inaccessible and non-interpretable.

**Key Challenge**: The absence of large-scale, open-source joint sketch-image-text training datasets. Existing sketch datasets (QuickDraw, Sketchy, TU-Berlin, etc.) either provide only category-level sketches without paired images, or target only a single task (SBIR), and universally lack text descriptions and question-answer pairs — which are essential for training LVLMs.

**Core Problem**: Sketches are highly abstract and exhibit substantial variation (in style, cultural background, and drawing skill), creating a large domain gap with natural images. Enabling LVLMs to understand sketches requires addressing both the data scarcity and modality alignment challenges simultaneously.

**Key Insight**:
1. Constructing an automated sketch generation pipeline to produce instance-level sketches from large-scale image datasets at scale
2. Designing a two-stage training strategy: large-scale sketch-image-text alignment pretraining followed by task-specific instruction fine-tuning
3. The "Three Opens" principle: Open Weight, Open Data, and Open Vocabulary

## Method

### Overall Architecture
O3SLM adopts a streamlined architecture: CLIP ViT-L/336 as the visual backbone (encoding both sketches and natural images) → a two-layer MLP multimodal connector → Vicuna v1.5 LLM. Sketch, image, and text tokens are concatenated and fed into the LLM, which implicitly learns cross-modal alignment via self-attention. Model weights are initialized from LLaVA-1.5 to inherit its text-image alignment capability.

### Key Designs

#### 1. **SketchVCL Dataset and Automated Sketch Generation Pipeline**

**Sketch Generation Pipeline (Figure 3)**:
- For each target object instance, SAM2 is used to generate segmentation masks
- The background is masked, and the foreground is converted to a sketch via Photo2Sketch (a Pix2Pix-based method)
- Morphological gradient edge detection is applied to enhance the sketch
- Final sketch = aggregation of the Pix2Pix sketch and edge detection results

A total of 19M and 14M instance-level sketches are generated from Object365 and OpenImages, respectively.

**Dataset Composition**:

| Stage | Task | Image Dataset | Sketch Source | Size |
|-------|------|---------------|---------------|------|
| Pretraining | Detailed description + bounding box | Objects365 | SketchVCL-O365 | 300K |
| Pretraining | Detailed description + bounding box | OpenImages | SketchVCL-OI | 300K |
| Fine-tuning | Object detection | COCO | SketchMIX | 110K |
| Fine-tuning | VQA | COCO | SketchMIX | 50K |
| Fine-tuning | Counting | PixMo Count | SketchMIX | 30K |
| Fine-tuning | SBIR | Sketchy | SketchMIX | 25K |

**Design Motivation**: The Photo2Sketch approach yields higher quality than CLIP-based methods and is substantially faster than diffusion models, making it well-suited for large-scale data generation. SketchMIX aggregates multiple sketch sources (Sketchy + QuickDraw + generated sketches) to increase diversity; TU-Berlin is intentionally excluded to serve as a held-out generalization benchmark.

#### 2. **Two-Stage Training Strategy**

**Stage I: Sketch Alignment Pretraining** (600K)
The objective is to teach the model the three-way correspondence — sketch ↔ image ↔ text:
- Recognizing objects depicted in sketches
- Associating sketches with corresponding objects in natural images
- Developing fine-grained spatial understanding (required for detection)
- Preserving natural language description capability

Each image is paired with a target category label; DeepSeek-VL2 generates descriptive captions, which are further refined by LLaMA-3-8B Instruct into structured responses encompassing sketch recognition, object description, spatial relationships, and bounding box coordinates.

**Stage II: Instruction Fine-tuning** (215K)
Task-specific prefix descriptors are designed for each of the four tasks (following Molmo's approach):
- **COUNT**: Sketch-guided object counting, output as an integer
- **BBOX**: Sketch-guided object detection, output as $[x_1, y_1, x_2, y_2]$
- **VQA**: Sketch-assisted visual question answering (25K sketch QA + 25K standard QA for balance)
- **SBIR**: Sketch-based image retrieval, trained with a binary cross-entropy objective

**Design Motivation**: The two-stage separation allows the model to first establish general sketch understanding before adapting to specific tasks. Task prefixes prevent task confusion, and random prompt template sampling mitigates prompt overfitting.

#### 3. **LVLM Adaptation for SBIR**

SBIR is innovatively reformulated as a binary classification task directly trainable within the LLM framework:
$$\arg\min_\theta -\sum_{i=1}^N [y_i \log(p_\theta(\texttt{<yes>}|X_i)) + (1-y_i)\log(p_\theta(\texttt{<no>}|X_i))]$$

At inference time, images are ranked in descending order by $p_\theta(\texttt{<yes>}|X_i)$, and the Top-K results are returned.

**Design Motivation**: Conventional SBIR requires specialized metric learning architectures. Reformulating retrieval as a `<yes>`/`<no>` binary classification integrates seamlessly into the LLM training paradigm without any architectural modification.

### Loss & Training
- LoRA (rank=64) for parameter-efficient training
- 2× NVIDIA H100 GPUs
- Learning rate $2 \times 10^{-5}$ with cosine decay and 3% warmup
- Trained for 1 epoch with batch size 24
- Two model scales: 7B and 13B

## Key Experimental Results

### Main Results — Sketch-Guided Counting (Accuracy)

| Model | PixMo-Count Avg | COCO Avg |
|-------|----------------|----------|
| GPT-4o | 33.6 | 16.4 |
| Gemini 1.5 Pro | 32.5 | 17.0 |
| LLaVA-1.5-7B | 16.0 | 12.1 |
| Qwen2.5-VL-7B | 17.7 | 24.6 |
| Molmo-7B-D | 30.3 | 12.0 |
| **O3SLM-7B** | **43.5** | **31.3** |
| **O3SLM-13B** | **44.0** | **31.7** |

### Sketch-Guided Object Detection (Acc@0.5, COCO val2017)

| Model | Sketchy | QuickDraw | TU-Berlin† | SketchVCL-C |
|-------|---------|-----------|------------|-------------|
| LLaVA-1.5-7B | 29.1 | 26.9 | 29.7 | 27.4 |
| Molmo-7B-D | 25.3 | 27.9 | 27.5 | 25.3 |
| **O3SLM-7B** | **33.9** | **23.8** | **29.4** | **21.5** |
| **O3SLM-13B** | **35.6** | **28.1** | **31.5** | **24.8** |

(Note: †TU-Berlin is a held-out dataset unseen during training, used to evaluate generalization.)

### SBIR Retrieval (Sketchy Dataset)

| Model | Acc@1 | Acc@5 | Acc@10 |
|-------|-------|-------|--------|
| LLaVA-1.5-7B | 11.0 | 14.4 | 13.0 |
| **O3SLM-7B** | **65.0** | **59.2** | **39.4** |
| LLaVA-1.5-13B | 10.0 | 29.2 | 28.3 |
| **O3SLM-13B** | **55.0** | **46.4** | **32.9** |

### Ablation Study

| Configuration | Key Findings | Notes |
|---------------|-------------|-------|
| Without pretraining | SBIR drops substantially; counting less affected | Pretraining is critical for sketch-dependent tasks (retrieval) |
| Frozen multimodal connector | Significant performance degradation | 7B with tuned connector > 13B with frozen connector |
| Image-only tasks | VQAv2: 76.6 vs 80.0 (LLaVA) | Sketch training incurs <5% loss in image understanding |
| Text-guided detection | 21.0 vs 13.4 (LLaVA) | Sketch training actually improves text-guided detection |

### Key Findings
1. O3SLM substantially outperforms existing open-source LVLMs across all sketch tasks, even surpassing GPT-4o and Gemini 1.5 Pro on multiple benchmarks.
2. Strong generalization to the held-out TU-Berlin sketches demonstrates that the model has learned universal sketch understanding rather than overfitting to specific styles.
3. SBIR Acc@1 improves from 11.0% to 65.0% (5.9× gain), indicating that vanilla LLaVA is nearly incapable of sketch comprehension.
4. Tuning the multimodal connector is critical — the 7B model with a tuned connector outperforms the 13B model with a frozen connector, confirming that sketch-image alignment must be established at the projection layer.
5. The model exhibits emergent capabilities: despite being trained with sketches in isolation, it can handle fine-grained joint queries combining sketches with text.

## Highlights & Insights
1. **Bridging the gap in LVLM sketch understanding**: O3SLM is the first open-source LVLM specifically designed for sketches, releasing weights, data, and the full model.
2. **Large-scale automated sketch generation pipeline**: Over 33M instance-level sketches are generated from Object365 and OpenImages, resolving the data bottleneck.
3. **Elegant LVLM adaptation for SBIR**: Reformulating retrieval as binary classification integrates seamlessly into the LLM training framework.
4. **Emergent fine-grained understanding**: Through VQA auxiliary supervision, the model spontaneously learns to leverage textual descriptions to complement attributes that sketches cannot readily express (color, texture, etc.).
5. **Minimal degradation of existing capabilities**: Image task performance drops by less than 5%, indicating that sketch training is complementary to rather than competitive with image understanding.

## Limitations & Future Work
1. Sketches generated by Photo2Sketch may not fully replicate the diversity and noise characteristics of real hand-drawn sketches.
2. The LLaVA-1.5 architecture with CLIP ViT-L/336 resolution may be insufficient for fine-grained sketch details.
3. Training for only 1 epoch leaves open the question of whether additional epochs could yield further improvements.
4. SBIR inference requires a forward pass for every image in the gallery (10K passes), resulting in low efficiency.
5. Sketch generation capability (image → sketch) is not explored; the work focuses solely on sketch understanding.

## Related Work & Insights
- The automated data generation pipeline (SAM2 + Pix2Pix + edge detection) is generalizable to other abstract visual modalities.
- The two-stage training strategy (alignment pretraining followed by task fine-tuning) constitutes a general paradigm for integrating new modalities into LVLMs.
- The idea of reformulating retrieval as binary classification is transferable to other "matching"-type tasks.
- The importance of tuning the projection layer for modality alignment is validated (vs. fine-tuning only the LLM).

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Customizing Visual Emotion Evaluation for MLLMs: An Open-vocabulary, Multifaceted, and Scalable Approach](../../ICLR2026/multimodal_vlm/customizing_visual_emotion_evaluation_for_mllms_an_open-vocabulary_multifaceted_.md)
- [\[AAAI 2026\] SToLa: Self-Adaptive Touch-Language Framework with Tactile Commonsense Reasoning in Open-Ended Scenarios](stola_self-adaptive_touch-language_framework_with_tactile_commonsense_reasoning_.md)
- [\[ICLR 2026\] Breaking the Limits of Open-Weight CLIP: An Optimization Framework for Self-supervised Fine-tuning of CLIP](../../ICLR2026/multimodal_vlm/breaking_the_limits_of_open-weight_clip_an_optimization_framework_for_self-super.md)
- [\[CVPR 2026\] GUIDE: A Benchmark for Understanding and Assisting Users in Open-Ended GUI Tasks](../../CVPR2026/multimodal_vlm/guide_a_benchmark_for_understanding_and_assisting_users_in_open-ended_gui_tasks.md)
- [\[ICCV 2025\] On Large Multimodal Models as Open-World Image Classifiers](../../ICCV2025/multimodal_vlm/on_large_multimodal_models_as_open-world_image_classifiers.md)

</div>

<!-- RELATED:END -->
