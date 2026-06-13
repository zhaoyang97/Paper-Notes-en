---
title: >-
  [Paper Note] SteelDefectX: A Coarse-to-Fine Vision-Language Dataset and Benchmark for Generalizable Steel Surface Defect Detection
description: >-
  [CVPR 2026][Object Detection][steel surface defect detection] This paper introduces SteelDefectX, the first vision-language dataset for steel surface defect detection (7,778 images, 25 defect categories)…
tags:
  - "CVPR 2026"
  - "Object Detection"
  - "steel surface defect detection"
  - "vision-language dataset"
  - "coarse-to-fine annotation"
  - "zero-shot transfer"
  - "industrial quality inspection"
date: 2026-05-08
content_hash: 206753f2090bc55c
---

# SteelDefectX: A Coarse-to-Fine Vision-Language Dataset and Benchmark for Generalizable Steel Surface Defect Detection

**Conference**: CVPR 2026
**arXiv**: [2603.21824](https://arxiv.org/abs/2603.21824)  
**Code**: [https://github.com/Zhaosxian/SteelDefectX](https://github.com/Zhaosxian/SteelDefectX)  
**Area**: Interpretability
**Keywords**: steel surface defect detection, vision-language dataset, coarse-to-fine annotation, zero-shot transfer, industrial quality inspection

## TL;DR

This paper introduces SteelDefectX, the first vision-language dataset for steel surface defect detection (7,778 images, 25 defect categories), featuring coarse-to-fine textual annotations ranging from class-level to sample-level descriptions. A four-task benchmark is established covering pure-vision classification, vision-language classification, zero/few-shot recognition, and zero-shot transfer. Experiments demonstrate that high-quality textual annotations significantly improve model interpretability, generalization, and cross-domain transfer capability.

## Background & Motivation

**Background**: Steel surface defect detection is a critical component of quality assurance in industrial manufacturing. Existing methods primarily rely on fundamental image classification or object detection models (ResNet, ViT, etc.), achieving satisfactory classification accuracy on specific datasets. Public datasets such as NEU (6 classes, 1,800 images), GC10 (10 classes, 2,312 images), X-SDD (7 classes, 1,360 images), and S3D (5 classes, 880 images) have driven progress in this field.

**Limitations of Prior Work**: (1) Existing datasets provide only category labels or numerical annotations, lacking descriptive textual information, which limits the application of vision-language models in industrial domains; (2) Simple class-name template descriptions (e.g., "A photo of scratches") fail to capture the rich visual variation of steel defects—the same manufacturing process can produce drastically different visual patterns across different materials; (3) There is no standardized benchmark for evaluating cross-material and cross-dataset generalization.

**Key Challenge**: Vision-language models (e.g., CLIP) exhibit strong zero-shot capabilities on natural images, but perform poorly when directly applied to industrial defect data (maximum zero-shot accuracy of only 14.8%), fundamentally due to the absence of professional industrial image-text paired data.

**Goal**: (1) Construct the first steel defect vision-language dataset with professional coarse-to-fine textual annotations; (2) Establish a standardized benchmark covering diverse scenarios to evaluate vision-language models in industrial inspection; (3) Validate the improvement in generalization and transfer capability brought by high-quality textual annotations.

**Key Insight**: Industrial defect detection requires not only category labels but also semantic understanding of defect types, visual attributes, and root causes—precisely where vision-language models excel, provided high-quality image-text paired data is available.

**Core Idea**: By constructing coarse-to-fine vision-language annotations (class-level: defect type + visual attributes + causes; sample-level: shape + size + depth + location + contrast), industrial defect detection is elevated from pure-vision classification to a vision-language semantic understanding task.

## Method

### Overall Architecture

The core contribution of SteelDefectX is the dataset and benchmark rather than a novel model architecture. The overall pipeline proceeds as follows: (1) images are collected and integrated from four sources—NEU, GC10, X-SDD, and S3D—and similar sub-categories are merged to yield a unified dataset of 25 classes and 7,778 images; (2) two-level textual annotations are constructed—class-level annotations are designed by domain experts, and sample-level annotations are generated automatically via GPT-4o followed by manual refinement; (3) a four-task benchmark is established to evaluate different models and annotation levels.

### Key Designs

1. **Class-Level Annotation (Coarse-grained)**:

    - Function: Provides global semantic descriptions for each defect category.
    - Mechanism: Each category is characterized by three semantic components: (a) defect class name (e.g., "punching"); (b) representative visual attributes (e.g., "circular holes"); (c) possible industrial causes (e.g., "equipment malfunction"). Initial templates are hand-crafted by domain experts based on steel manufacturing knowledge, then refined using candidate descriptions generated by the CuPL method, and finally composed into natural language sentences.
    - Design Motivation: Class-level semantics provide consistent conceptual anchors across samples, helping vision-language models establish alignment between defect types and the semantic space.

2. **Sample-Level Annotation Pipeline (Fine-grained)**:

    - Function: Generates detailed visual descriptions for each individual sample.
    - Mechanism: A four-step pipeline is employed—(Step 1) **Candidate Generation**: GPT-4o is prompted with an open-ended prompt at a relatively high temperature (0.9) to generate 4 candidate descriptions, encouraging diversity; (Step 2) **Candidate Filtering**: Sentence-BERT is used to compute cosine similarity among descriptions, and a greedy strategy retains up to 3 diverse candidates; each candidate is then assessed with a 5-dimensional semantic coverage score—descriptions are encoded as a 5-bit vector $\mathbf{b} = [b_1,...,b_5]$ corresponding to shape, size, depth, location, and contrast, and the composite score $S(d_i) = 0.6 \cdot \frac{\|b_i\|_1}{5} + 0.4 \cdot D(d_i)$ balances coverage and diversity; (Step 3) **Candidate Supplementation**: If no candidate covers $\geq 4$ dimensions, a structured multi-question prompt is used to query each dimension individually; (Step 4) **Manual Correction**: Two annotators perform cross-validation over approximately 275 hours.
    - Design Motivation: The three-tier mechanism of automated generation, structured quality control, and manual refinement ensures annotation quality. The 5-dimensional semantic coverage framework guarantees completeness and consistency of descriptions.

3. **Four-Task Benchmark Design**:

    - Function: Systematically evaluates the value of the dataset across different scenarios.
    - Mechanism: (Task 1) Pure-vision classification—ResNet/ViT with a linear head; (Task 2) Vision-language classification—CLIP variants with Adapter fine-tuning, trained with T3 (fine-grained annotations) and tested with T0 (class-name templates); (Task 3) Zero/few-shot recognition—evaluates performance under 1/2/4/8-shot settings, comparing the effects of T0 and T3 annotations; (Task 4) Zero-shot transfer—models trained on SteelDefectX are tested on aluminum surface defects (MSD-Cls, 10 classes) and seamless steel tube defects (CGFSDS-9, 5 classes). Four annotation levels (T0→T3) with progressively increasing information are used for comparison.
    - Design Motivation: The benchmark spans from the most basic pure-vision setting to the most challenging cross-material zero-shot transfer, comprehensively covering practical industrial inspection scenarios.

### Loss & Training

Pure-vision classification: SGD with momentum 0.9, weight decay 1e-4, initial learning rate 0.1 decayed by 10× every 30 epochs, trained for 100 epochs. Vision-language classification: CLIP-Adapter framework with Adam optimizer (lr=1e-4), bidirectional cross-entropy loss, 20 epochs. A 7:3 train/test split is used throughout.

## Key Experimental Results

### Main Results

Pure-vision classification (Task 1):

| Model | Acc (%) | mAcc (%) |
|------|---------|----------|
| ShuffleNetV2 | 96.34 | 94.98 |
| ResNet-101 | 93.63 | 91.19 |
| ViT-B/16 | 44.84 | 40.31 |

Vision-language classification (Task 2, trained on T3 / tested on T0):

| Model | Backbone | Acc (%) | mAcc (%) |
|------|----------|---------|----------|
| Long-CLIP | ViT-L/14 | **93.63** | **92.56** |
| OpenCLIP | ViT-L/14 | 88.21 | 87.54 |
| CLIP | ViT-B/16 | 81.84 | 81.14 |

Zero-shot transfer (Task 4, Long-CLIP ViT-L/14):

| Annotation Level | Aluminum Acc | Seamless Steel Tube Acc |
|---------|-----------|-------------|
| Zero-shot | 8.60 | 25.11 |
| T0 (class name) | 12.90 | 28.31 |
| T1 (class-level) | 20.43 | 33.79 |
| T2 (GPT-4o) | 25.27 | 34.25 |
| T3 (manual refinement) | **29.03** | **40.18** |

### Ablation Study

Effect comparison across annotation levels (zero-shot recognition, Task 3):

| Annotation Level | SteelDefectX Zero-shot Acc |
|---------|----------------------|
| T0 (class-name template) | 7.57 |
| T1 (class-level description) | 11.27 |

Few-shot recognition as a function of the number of shots:

| Method | 1-shot | 8-shot |
|------|--------|--------|
| Long-CLIP-Adapter (T0) | ~60% | ~88% |
| Tip-Adapter-F (T0) | ~55% | ~85% |

### Key Findings

- **ViT severely underfits on small datasets**: ViT-B/16 achieves only 44.84%, far below CNN-based models (ShuffleNetV2: 96.34%), indicating that CNN inductive biases remain advantageous on small-scale datasets.
- **Annotation level monotonically improves transfer performance**: Transfer accuracy on the aluminum dataset increases consistently from 12.90% (T0) to 29.03% (T3), and the T2→T3 gain from manual refinement is also significant, demonstrating that annotation quality directly determines cross-domain transfer effectiveness.
- **Long-CLIP achieves the best performance in vision-language classification**: At 93.63% accuracy, it approaches pure-vision CNN performance (96.34%), with a smaller gap between Acc and mAcc (1.07 vs. 1.36), indicating greater robustness on long-tail categories.
- **Pretrained VLMs applied directly to industrial defects perform poorly**: CLIP achieves only 7.57% zero-shot accuracy on SteelDefectX, revealing a substantial semantic gap between the natural-image pretraining domain and industrial defect domains.
- Saliency map visualizations show that under T3 annotations, models precisely focus on defect regions, whereas under T0 annotations attention is scattered—indicating that fine-grained textual descriptions enhance spatial vision-text alignment.

## Highlights & Insights

- **5-Dimensional semantic coverage framework (shape / size / depth / location / contrast)**: Provides a reproducible, structured standard for industrial defect annotation that does not rely on subjective descriptions and can be transferred to other industrial inspection scenarios (e.g., chip defects, textile defects). This approach is more principled and controllable than free-text annotation.
- **Progressive annotation-level experimental design**: The T0→T1→T2→T3 comparative experiments clearly delineate the marginal contribution of each annotation level, offering concrete guidance on cost-benefit tradeoffs for industrial data collection—even GPT-4o-generated annotations without manual refinement (T2) yield significant improvements.
- **Validation of cross-material zero-shot transfer feasibility**: Transfer from steel to aluminum (29.03%), while modest in absolute terms, represents a 3.4× improvement over the zero-shot baseline (8.60%), demonstrating the potential of vision-language alignment for cross-material generalization.

## Limitations & Future Work

- The dataset scale remains limited (7,778 images), far smaller than natural image datasets, potentially constraining thorough training of vision-language models.
- The current benchmark supports only image-level classification and vision-language alignment, lacking pixel-level segmentation annotations—limiting applicability to object detection and segmentation tasks.
- GPT-4o-generated textual descriptions may contain hallucinations inconsistent with actual visual content; although manual correction is applied, full-scale verification is costly.
- Among the 25 defect categories, some have very few samples (e.g., crease with only 50 images), resulting in a severe long-tail distribution.
- The absolute zero-shot transfer accuracy remains low (29% / 40%), leaving a substantial gap before practical deployment.
- No systematic comparison is made with recent industrial anomaly detection methods (e.g., AnomalyGPT, WinCLIP).

## Related Work & Insights

- **vs. NEU/GC10 and other traditional datasets**: Traditional datasets provide only category labels; SteelDefectX introduces a semantic understanding dimension through coarse-to-fine textual annotations, representing a paradigm shift from "classification" to "understanding."
- **vs. MMAD (multimodal anomaly detection)**: MMAD covers diverse industrial products but restricts textual content to QA pairs, lacking professional defect attribute descriptions. The 5-dimensional semantic framework of SteelDefectX is more structured and industrially oriented.
- **vs. WinCLIP/CAM-CLIP**: These methods attempt to adapt CLIP to industrial scenarios but are constrained by text-side data quality. SteelDefectX directly addresses this bottleneck and can serve as foundational data for industrial VLM pretraining.
- The dataset construction pipeline (automated generation + semantic filtering + dimensional coverage checking + manual refinement) can serve as a general paradigm for building vision-language datasets in other vertical domains.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing the vision-language paradigm to industrial defect detection is a valuable contribution; the 5-dimensional semantic framework offers methodological merit.
- Experimental Thoroughness: ⭐⭐⭐⭐ The four-task benchmark is comprehensive and the annotation-level comparative experiments are convincing; however, comparisons with recent industrial VLM methods are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ The dataset construction pipeline is described in thorough and clear detail, with rich and informative figures and tables.
- Value: ⭐⭐⭐⭐ As the first vision-language dataset for steel defects, this work has significant impact on advancing the field, and the construction methodology is broadly applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)
- [\[CVPR 2026\] Mining Instance-Centric Vision-Language Contexts for Human-Object Interaction Detection](mining_instance-centric_vision-language_contexts_for_human-object_interaction_de.md)
- [\[CVPR 2026\] UniSpector: Towards Universal Open-set Defect Recognition via Spectral-Contrastive Visual Prompting](unispector_towards_universal_open-set_defect_recognition_via_spectral-contrastiv.md)
- [\[CVPR 2026\] Saliency-R1: Enforcing Interpretable and Faithful Vision-language Reasoning via Saliency-map Alignment Reward](saliency-r1_enforcing_interpretable_and_faithful_vision-language_reasoning_via_s.md)
- [\[ICCV 2025\] Kaputt: A Large-Scale Dataset for Visual Defect Detection](../../ICCV2025/object_detection/kaputt_a_large-scale_dataset_for_visual_defect_detection.md)

</div>

<!-- RELATED:END -->
