---
title: >-
  [Paper Note] COUNTS: Benchmarking Object Detectors and Multimodal Large Language Models under Distribution Shifts
description: >-
  [CVPR 2025][Multimodal VLM][Distribution shift] This paper constructs COUNTS, a large-scale OOD dataset featuring 14 natural distribution shifts, over 222K samples, and more than 1.19 million bounding box annotations. It introduces two benchmarks, O(OD)² and OODG, to systematically evaluate the generalization capability of object detectors and multimodal large language models under distribution shifts, revealing that even GPT-4o only achieves a grounding accuracy of 56.7%.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Distribution shift"
  - "object detection robustness"
  - "visual grounding"
  - "OOD generalization"
  - "multimodal benchmark"
date: 2026-05-08
content_hash: c2b05f11aa2def05
---

# COUNTS: Benchmarking Object Detectors and Multimodal Large Language Models under Distribution Shifts

**Conference**: CVPR 2025  
**arXiv**: [2504.10158](https://arxiv.org/abs/2504.10158)  
**Code**: [GitHub](https://github.com/jiansheng-li/COUNTS_benchmark)  
**Area**: Multimodal VLM  
**Keywords**: Distribution shift, object detection robustness, visual grounding, OOD generalization, multimodal benchmark

## TL;DR
This paper constructs COUNTS, a large-scale OOD dataset featuring 14 natural distribution shifts, over 222K samples, and more than 1.19 million bounding box annotations. It introduces two benchmarks, O(OD)² and OODG, to systematically evaluate the generalization capability of object detectors and multimodal large language models under distribution shifts, revealing that even GPT-4o only achieves a grounding accuracy of 56.7%.

## Background & Motivation
Existing object detectors often suffer from significant performance degradation during real-world deployment due to data distribution shifts. Although out-of-distribution (OOD) generalization has been extensively studied in image classification, there is a lack of large-scale, fine-grained annotated evaluation benchmarks for more complex tasks like object detection and visual grounding. Existing detection robustness datasets suffer from distinct limitations: COCO-C utilizes synthetic corruptions (which do not reflect actual real-world scenarios); COCO-O is small-scale (only 6,782 images) and features a limited number of domains; and driving scene datasets (such as Cityscapes) represent monolithic scenarios lacking domain diversity.

More critically, while Multimodal Large Language Models (MLLMs) have advanced rapidly in recent years, their **visual grounding** capabilities under distribution shifts remain largely unexplored. Because the training data of MLLMs is opaque, traditional paradigms for defining distribution shifts are inapplicable, necessitating a new evaluation paradigm.

Key Insight: This paper addresses these issues by constructing a large-scale, fine-grained annotated dataset collected from the real world covering rich domain types. Concurrently, it proposes two evaluation benchmarks tailored for object detectors and MLLMs, systematically exposing the weaknesses of both model families under OOD conditions.

## Method

### Overall Architecture
The COUNTS project consists of three core components:
1. **COUNTS Dataset**: 14 domains, 35 categories, 222,234 real-world images, and 1,196,114 bounding box annotations.
2. **O(OD)² Benchmark**: Evaluates the OOD generalization capabilities of object detectors.
3. **OODG Benchmark**: Evaluates the OOD generalization capabilities of MLLMs in visual grounding.

### Key Designs
1. **COUNTS Dataset Construction**:

    - Three-stage pipeline: First, candidate samples are filtered from over 5 million images in public datasets like Open Images, Visual Genome, and RefCOCO; then, two independent annotators verify the domain labels (retaining only mutually agreed samples); finally, bounding boxes in the test and validation sets (23,000 images) are manually re-annotated.
    - Selection criteria for the 14 domains: Commonly encountered in reality, exhibiting a significant impact on pixel distribution, and designed to be as independent as possible.
    - Domain list: dim, painting, snow, sand, handmade, street, road, water, grass, indoor, mountain, sky, tree, occlusion.
    - All domains share the complete 35-category space, ensuring the validity of cross-domain evaluation.

2. **O(OD)² Benchmark Design**:

    - Core Idea: Six domains (sky, occlusion, grass, water, dim, handmade) are designated as target domains (representing different types of distribution shifts), while the remaining domains serve as the training set.
    - Utilizing multiple target domains simultaneously (rather than a single target domain) to reflect the uncertainty of real-world deployment scenarios more faithfully.
    - The evaluation targets include two-stage detectors (Faster R-CNN with various combinations of backbone/neck/head), one-stage detectors (RetinaNet, YOLOv9), and Transformer-based detectors (DETR, DINO, DINOv2).

3. **OODG Benchmark Design**:

    - Core Idea: Distribution shift is successfully reformulated as the discrepancy between In-Context Learning (ICL) exemplars and test query samples (rather than train-test shifts), bypassing the barrier of opaque pretraining distributions.
    - Five evaluation settings: zero-shot capability, IID in-context learning, covariate shift generalization, label shift generalization, and spurious correlation shift generalization.
    - Three test tasks: visual grounding (predict class given region), detection & grounding (predict bounding box given query), and visual-semantic mapping (match region given description).
    - The design of spurious correlation shift is particularly clever: statistical associations like "cat = dim indoor, dog = outdoor" are intentionally introduced in ICL examples to test whether the model is misled.

### Loss & Training
As this is a benchmark paper, no new training strategies are introduced. All detector experiments are implemented using the MMDetection framework, adhering to the hyperparameter configurations optimized in the original papers using RTX 4090 GPUs.

## Key Experimental Results

### Main Results——Object Detector OOD Generalization（O(OD)²）

| Detector | Val mAP | Average OOD mAP | Worst Domain | Analysis |
|--------|--------|-----------|--------|------|
| Faster R-CNN (RN-50) | 0.279 | 0.135 | Water: 0.103 | Baseline |
| Faster R-CNN (RN-101) | 0.325 | 0.148 | Water: 0.110 | Stronger backbone helps |
| DINO | 0.384 | 0.213 | Dim: 0.169 | Significant lead |
| DINOv2 | 0.389 | 0.213 | Dim: 0.165 | Best performing |
| YOLOv9 | 0.282 | 0.150 | Grass: 0.118 | One-stage performs average |

### Main Results——MLLM Visual Grounding Capabilities（OODG Zero-Shot）

| Model | Overall Accuracy | Analysis |
|------|-----------|------|
| GPT-4o | 65.9% | Strongest but still limited |
| GLaMM | 62.5% | Best among open-source models |
| Gemini-1.5-Flash | 59.1% | Moderate performance |
| Qwen2-VL | 54.4% | Significantly weaker than GPT-4o |

### Ablation Study

| Configuration | Key Metrics | Analysis |
|------|---------|------|
| Neck Comparison (FPN vs PAFPN vs NAS-FPN) | OOD mAP difference < 2% | Neck variations have limited impact on OOD generalization |
| Head Comparison (Standard vs Cascade vs SABL) | SABL > Cascade > Standard | Head design is key to improving OOD performance |
| Pretraining Data (IN-1K vs IN-21K) | +0.5% OOD mAP | Marginal gains from increasing data volume |
| Sup_timm vs Standard Supervised | +18.9% relative gain | Advanced training recipes are more effective than larger datasets |
| ICL Setting 3 (Covariate Shift) GPT-4o | Accuracy drops to 55.2% | Distribution shifts significantly disrupt ICL effectiveness |
| ICL Setting 5 (Spurious Correlation) GPT-4o | Accuracy drops to 50.3% | The model is easily misled by spurious correlations |

### Key Findings
- Improvements in IID performance do not necessarily translate to enhanced OOD generalization. Some components that improve IID performance (e.g., NAS-FPN, FreeAnchor) actually degrade OOD performance.
- For two-stage detectors, refining the head is more effective than upgrading the backbone; for one-stage detectors, both design aspects help.
- DINO/DINOv2 significantly outperform other detectors, suggesting that training strategies (self-supervised + contrastive learning) contribute more to OOD generalization than mere architectural modifications.
- Self-supervised pretraining (MoCov2) brings no OOD generalization gain, whereas advanced supervised training recipes (Sup_timm) dramatically boost generalization.
- Even under the ICL setting, MLLM visual grounding remains fragile; GPT-4o suffers from substantial accuracy drops under covariate shift and spurious correlation configurations.

## Highlights & Insights
1. First large-scale real-world dataset that concurrently supports OOD evaluation for both object detection and MLLM visual grounding.
2. Clever reformulation of distribution shift for MLLMs by leveraging the discrepancies between ICL exemplars and test inputs, bypassing the barrier of opaque pretraining data.
3. The design of the spurious correlation shift is insightful, revealing that MLLMs learn statistical biases present in context demonstrations rather than just using them for learning.
4. Although the finding that "higher IID performance does not equate to stronger OOD capability" is not entirely new, its systematic validation in detection tasks fills an important literature gap.
5. The 34-page paper (including appendix) provides massive evaluation data, offering detailed guidelines for robust detector design.

## Limitations & Future Work
- Domain definitions primarily focus on scene/background-level shifts, lacking object-level shifts (e.g., pose variation, deformation).
- The assumption of independence among the 14 domains has not been rigorously validated; strong correlation may exist between domains like 'road' and 'street'.
- The OODG benchmark currently defines distribution shifts solely within the ICL context, leaving fine-tuning distribution shifts unexplored.
- The number of object categories (35 classes) is relatively small compared to COCO (80 classes), which may limit the generalizability of some findings.
- The evaluation of open-source MLLMs is restricted to limited models (GLaMM, Qwen2-VL), missing more recent baseline models.

## Related Work & Insights
- Comparisons with COCO-O demonstrate that real-world domain diversity and substantial data scale are crucial for constructing constructive benchmarks.
- The definition of ICL shifts in OODG can be generalized to other MLLM evaluation scenarios (such as OOD evaluations for VQA and image captioning).
- The significance of detector heads inspires a new direction: designing detection heads specifically optimized for OOD robustness.
- Compared to classification OOD benchmarks like DomainBed, OOD generalization in detection tasks is still vastly understudied.

## Rating
- Novelty: ⭐⭐⭐ Tasb/dataset construction and OODG definitions are innovative; however, as a benchmark paper, technical methodology novelty is moderate.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensively covers numerous detector variations and MLLM models from multiple analytical perspectives (backbone/neck/head/pretraining).
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with rich data; however, the paper is quite long (34 pages), and some analyses could be more concise.
- Value: ⭐⭐⭐⭐ Fills the vacancy of OOD benchmarks for detection and MLLM grounding, holding substantial reference value for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] On the Out-of-Distribution Generalization of Multimodal Large Language Models](on_the_out-of-distribution_generalization_of_large_multimodal_models.md)
- [\[CVPR 2026\] ORIC: Benchmarking Object Recognition under Contextual Incongruity in Large Vision-Language Models](../../CVPR2026/multimodal_vlm/oric_benchmarking_object_recognition_under_contextual_incongruity_in_large_visio.md)
- [\[CVPR 2025\] Teaching Large Language Models to Regress Accurate Image Quality Scores Using Score Distribution](teaching_large_language_models_to_regress_accurate_image_quality_scores_using_sc.md)
- [\[CVPR 2025\] VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos](videoglamm_a_large_multimodal_model_for_pixel-level_visual_grounding_in_videos.md)
- [\[CVPR 2025\] EventGPT: Event Stream Understanding with Multimodal Large Language Models](eventgpt_event_stream_understanding_with_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
