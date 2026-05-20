---
title: >-
  [Paper Note] LMM-Det: Make Large Multimodal Models Excel in Object Detection
description: >-
  [ICCV 2025][Object Detection][Large multimodal models] This paper proposes LMM-Det, which through systematic analysis identifies low recall as the core bottleneck of large multimodal models (LMMs) in object detection. By…
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "Large multimodal models"
  - "recall"
  - "data distribution adjustment"
  - "inference optimization"
date: 2026-05-08
content_hash: c7250443d5303106
---

# LMM-Det: Make Large Multimodal Models Excel in Object Detection

**Conference**: ICCV 2025
**arXiv**: [2507.18300](https://arxiv.org/abs/2507.18300)  
**Code**: [github.com/360CVGroup/LMM-Det](https://github.com/360CVGroup/LMM-Det)  
**Area**: Object Detection
**Keywords**: Large multimodal models, object detection, recall, data distribution adjustment, inference optimization

## TL;DR

This paper proposes LMM-Det, which through systematic analysis identifies low recall as the core bottleneck of large multimodal models (LMMs) in object detection. By applying data distribution adjustment (pseudo-label augmentation) and inference optimization (per-category detection), LMM-Det improves COCO AP from 0.2 to 47.5 without any additional specialized detection modules.

## Background & Motivation

### State of the Field

Large multimodal models (LMMs) such as LLaVA and InternVL have demonstrated strong performance on image captioning, VQA, and visual grounding tasks, showcasing impressive multimodal understanding and reasoning capabilities. However, a substantial performance gap remains between LMMs and specialized detectors (e.g., RT-DETR, Salience-DETR) on standard object detection — the fundamental vision task of localizing and classifying all objects in an image.

### Limitations of Prior Work

**Existing approaches rely on additional detection modules**: Groma integrates an RPN, while VisionLLM v2 employs Grounding DINO as an external detection head. These designs are constrained by the performance of the additional modules, introduce extra latency, and fail to exploit the intrinsic detection potential of LMMs.

**Native LMM detection capability is severely limited**: LLaVA achieves only 0.2 AP on COCO in a zero-shot setting. Even after training on COCO + Object365 with higher resolution, it only reaches 38.7 AP — far below the 55+ AP achieved by specialized detectors.

**The root cause has not been systematically diagnosed**: Prior work has not analyzed the fundamental reasons for LMM failure on detection tasks.

### Root Cause & Starting Point

Through systematic visualization and distribution analysis, the authors identify **low recall as the core bottleneck**: after training, the distribution of predicted bounding boxes converges toward the training set distribution, and the incomplete annotations in COCO cause the model to terminate predictions prematurely, generating on average only approximately 7 boxes per image. Furthermore, the autoregressive prediction mechanism of LMMs is inherently ill-suited for generating large numbers of high-quality proposals.

## Method

### Overall Architecture

LMM-Det consists of a visual encoder (OWLv2-ViT), a linear projector, and a large language model (Vicuna-1.5-7B). Training is organized into four stages: Stage I aligns the vision-language modules; Stage II pre-trains detection capability on Object365; Stage III fine-tunes on COCO; and an optional Stage IV mixes LLaVA data to preserve general multimodal ability.

### Key Designs

#### 1. **Data Distribution Adjustment**
- **Function**: Augments training data with pseudo-labels to increase the number of annotated boxes per image, thereby improving model recall.
- **Mechanism**: A pre-trained specialized detector (Salience-DETR) generates pseudo-labels for each training image, which are merged with original GT annotations via NMS. The model is trained to predict both box coordinates and confidence scores (GT annotations have confidence 1; pseudo-label confidence is provided by the detector).
- **Design Motivation**: A trained LMM converges toward the training data distribution. Incomplete COCO annotations cause the model to terminate predictions early — by increasing annotation density, the model is guided to generate more candidate boxes, thereby improving recall. Note that pseudo-labels are used solely for data augmentation; LMM-Det's inference stage does not depend on any external detection module.

#### 2. **Inference Optimization**
- **Function**: Replaces one-shot prediction of all boxes with per-category sequential detection.
- **Mechanism**: Rather than prompting the LMM to output all boxes across all categories in a single pass, the model is queried independently for each category. To maintain training-inference consistency, the instruction data format is restructured to adopt a category-specific prediction strategy.
- **Design Motivation**: The fixed sampling strategy of current LMMs makes it difficult to generate a sufficient number of fine-grained proposals in a single autoregressive step. Per-category detection substantially increases the total number of proposals. Ablation experiments show that this strategy improves AP from 44.2% to 47.5% and AR@100 from 56.0% to 63.6%.

#### 3. **Token Representation Validation**
- **Function**: Investigates token representation strategies for coordinates and confidence scores.
- **Mechanism**: Compares direct token prediction against vocabulary expansion.
- **Design Motivation**: Direct token prediction requires no additional training of vocabulary embeddings, and experiments demonstrate superior detection accuracy.

### Loss & Training

Standard language modeling loss (next-token prediction):
$$\max_\theta \sum_{i=1}^L \log p_\theta(\tilde{\mathbf{y}}_i | \mathbf{x}_v, \mathbf{x}_t, \mathbf{y}_{1:i-1})$$

Training uses 595K image-text pairs and 1.86M images, conducted across 6 nodes (8×H800 per node) for a total of 176 hours.

## Key Experimental Results

### Main Results (Zero-Shot, COCO val)

| Method | Visual Encoder | LLM | Extra Det. Module | AP | AP50 | AR@100 |
|--------|--------------|-----|-------------------|------|------|--------|
| LLaVA | CLIP-L | Vicuna-7B | None | 0.2 | 0.6 | 11.2 |
| KOSMOS-2 | CLIP-L | MAGNETO | None | 7.6 | 13.7 | 18.2 |
| InternVL-2.5 | InternViT | Internlm2.5-7B | None | 11.8 | 18.4 | 27.5 |
| Groma | DINOv2 | Vicuna-7B | **Yes** | 12.8 | 17.0 | 22.5 |
| **LMM-Det** | OWLv2-L | Vicuna-7B | **None** | **24.5** | **34.7** | **46.6** |

### Main Results (Fine-tuned, COCO val)

| Method | Extra Module | AP | AP50 | AP75 | AR@100 |
|--------|-------------|------|------|------|--------|
| RT-DETR (specialized) | - | 55.3 | 73.4 | 60.0 | 74.4 |
| Salience-DETR (specialized) | - | 57.3 | 75.5 | 62.4 | 75.4 |
| Groma (DINOv2) | Yes | 43.6 | - | - | - |
| VisionLLM v2 | Yes (Grounding DINO) | 56.3↓ | 74.3 | 61.6 | - |
| LLaVA* (retrained) | None | 38.7 | 55.8 | 41.3 | 50.5 |
| **LMM-Det** | **None** | **47.5** | **66.5** | **51.1** | **63.6** |

### Ablation Study

| Configuration | AP | AP50 | AP75 | AR@100 | Note |
|--------------|------|------|------|--------|------|
| Baseline (CLIP-ViT) | 38.7 | 55.8 | 41.3 | 50.5 | LLaVA* retrained |
| + OWLv2-ViT | 42.1 | 57.8 | 45.8 | 51.3 | +3.4 AP |
| + Data Distribution Adjustment | 44.2 | 61.3 | 47.5 | 56.0 | +2.1 AP, +4.7 AR |
| + Inference Optimization | **47.5** | **66.5** | **51.1** | **63.6** | +3.3 AP, +7.6 AR |

### Multimodal Capability Retention

| Model | COCO AP | Captioning CIDEr | VQAv2 Accuracy |
|-------|---------|-----------------|---------------|
| LLaVA | 0.2 | 108.9 | 78.5 |
| LMM-Det† | 47.1 | 99.0 | 74.1 |

### Key Findings

- The fundamental cause of poor LMM detection performance is **insufficient recall** (AR@100 of only 50.5), not localization accuracy.
- The predicted box distribution converges toward the training set distribution — incomplete COCO annotations cause the model to learn to "predict fewer boxes."
- OWLv2-ViT is better suited for detection tasks than CLIP-ViT, providing superior high-resolution input support.
- LMM-Det†, upon acquiring detection capability, incurs only a minor degradation (~4%) in VQA and captioning performance.
- Single-image inference requires approximately 4 seconds, which is insufficient for real-time detection scenarios.

## Highlights & Insights

1. **Diagnosis-first methodology**: Systematically identifying the root cause of failure (low recall) before designing targeted solutions is a research paradigm worth emulating.
2. **Data distribution perspective**: LMMs learn distributional characteristics of training data (e.g., average number of boxes per image); the "distribution truncation" caused by incomplete annotations is an important but previously overlooked problem.
3. **Minimal architectural modification**: Substantial performance gains are achieved solely through data and inference strategy optimization, without introducing any additional detection modules, demonstrating the intrinsic detection capacity of LMMs.
4. **Multi-task compatibility**: The optional Stage IV demonstrates good compatibility between detection capability and general multimodal ability.

## Limitations & Future Work

1. **Slow inference**: Approximately 4 seconds per image (due to per-category prediction), far slower than real-time detectors.
2. **Remaining gap with specialized detectors**: 47.5 vs. 57.3 AP, a gap of approximately 10 points.
3. **Dependence on pseudo-label quality**: Data distribution adjustment relies on pseudo-labels generated by Salience-DETR, whose quality is bounded by the detector's performance.
4. **Evaluation limited to COCO**: Performance is not validated on larger-scale or long-tail benchmarks such as LVIS and Objects365.
5. **Absence of open-vocabulary detection evaluation**: Only fixed-category detection is assessed; the potential advantages of LMMs in open-vocabulary detection remain unexplored.

## Related Work & Insights

- Shikra and KOSMOS-2 demonstrate LMM localization capability on REC tasks, but whole-image detection (requiring enumeration of all objects) is a substantially harder problem.
- VisionLLM v2's integration of Grounding DINO actually degrades the original detection performance, suggesting that naively combining modules is suboptimal.
- The per-category inference strategy essentially decomposes "generate all at once" into multiple "conditional generation" subtasks, which is more compatible with the autoregressive generation paradigm of LMMs.

## Rating

- **Novelty**: ⭐⭐⭐ — The methods themselves (pseudo-labels + per-category detection) are relatively straightforward; the core contribution lies in the systematic diagnostic analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive coverage including zero-shot, fine-tuned, ablation, and multi-task evaluations.
- **Writing Quality**: ⭐⭐⭐⭐ — The logical chain from exploratory experiments to analysis to proposed solutions is clear and well-structured.
- **Value**: ⭐⭐⭐⭐ — First systematic demonstration that LMMs can perform object detection without additional modules, providing an important baseline and analytical framework for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Adversarial Attention Perturbations for Large Object Detection Transformers](adversarial_attention_perturbations_for_large_object_detection_transformers.md)
- [\[ICCV 2025\] Accelerate 3D Object Detection Models via Zero-Shot Attention Key Pruning](accelerate_3d_object_detection_models_via_zero-shot_attention_key_pruning.md)
- [\[ICCV 2025\] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)
- [\[ICCV 2025\] Large-scale Pre-training for Grounded Video Caption Generation](large-scale_pre-training_for_grounded_video_caption_generation.md)
- [\[AAAI 2026\] VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](../../AAAI2026/object_detection/vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)

</div>

<!-- RELATED:END -->
