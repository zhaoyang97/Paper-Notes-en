---
title: >-
  [Paper Note] MEDISEG: A Medication Image Instance Segmentation Dataset for Preventing Adverse Drug Events
description: >-
  [CVPR 2026][Segmentation][Medication recognition] This work introduces MEDISEG, a medication image instance segmentation dataset (8,262 images, 32 pill classes…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Medication recognition"
  - "instance segmentation"
  - "few-shot detection"
  - "dataset"
  - "medication safety"
date: 2026-05-08
content_hash: 953be241b6a12f5e
---

# MEDISEG: A Medication Image Instance Segmentation Dataset for Preventing Adverse Drug Events

**Conference**: CVPR 2026
**arXiv**: [2603.10825](https://arxiv.org/abs/2603.10825)  
**Code**: [github.com/williamcwi/MEDISEG](https://github.com/williamcwi/MEDISEG)  
**Area**: Instance Segmentation / Medical Imaging / Dataset
**Keywords**: Medication recognition, instance segmentation, few-shot detection, dataset, medication safety

## TL;DR

This work introduces MEDISEG, a medication image instance segmentation dataset (8,262 images, 32 pill classes, with real-world occlusion/overlap scenarios). YOLOv8/v9 achieve 99.5% mAP@0.5 on the 3-class subset and 80.1% on the 32-class subset. FsDet few-shot experiments demonstrate that MEDISEG pretraining significantly outperforms CURE in occluded scenarios (1-shot: 0.406 vs. 0.131).

## Background & Motivation

**Background**: Medication errors and adverse drug events (ADEs) pose serious threats to patient safety — ADEs accounted for 8.9% of AEMT-related deaths between 1980 and 2014, with the rate continuing to rise. More than one-third of adults aged 75–85 take five or more prescription medications daily. AI-based pill recognition is a promising solution, yet progress is constrained by the quality of available datasets.

**Limitations of Prior Work**: Existing pill datasets suffer from three major shortcomings: (1) NIH Pillbox (the largest, with 133K images) was discontinued in 2021 and lacks instance segmentation annotations; (2) CURE provides partial instance segmentation annotations that are incomplete and include synthetic images; (3) all existing datasets predominantly feature single-pill images captured in controlled environments, failing to reflect real-world scenarios such as dosette boxes with multiple overlapping pills.

**Key Challenge**: In clinical and home settings, pills are routinely arranged in stacked, multi-pill configurations inside dosette boxes, requiring instance-level segmentation to distinguish individual pills — yet existing datasets consist almost exclusively of single-pill images.

**Goal**: To construct a pill image dataset that encompasses realistic multi-pill scenarios (overlap, occlusion, dosette boxes) with complete instance segmentation annotations, and to validate its contribution to few-shot generalization.

**Key Insight**: The dataset is designed from clinical needs — dosette boxes naturally produce multi-pill overlap and occlusion, while visually similar pill classes are deliberately selected to increase recognition difficulty.

**Core Idea**: Construct an instance segmentation dataset covering real-world multi-pill scenarios (occlusion/overlap/dosette boxes), and demonstrate that scene complexity — rather than sheer data volume — is the key factor for few-shot generalization.

## Method

### Overall Architecture

MEDISEG comprises two subsets: 3-Pills (3 pill classes) and 32-Pills (8,262 images, 32 classes). The pipeline proceeds as follows: capture with iPhone 12 Pro Max → crop dosette box cells to individual compartments → resize to 640×640 → manually annotate instance segmentation masks using COCO Annotator → validate with supervised YOLOv8/v9 training → evaluate with FsDet few-shot detection.

### Key Designs

1. **Real-World Acquisition Strategy**:
    - **Function**: Pills are arranged in standard 4-row × 7-column dosette boxes and photographed.
    - **Mechanism**: The compartments naturally produce pill overlap, occlusion, and background clutter. Varying lighting intensity and angle introduces realistic shadows, reflections, and highlights. Each image contains 1–13 pills, spanning single-pill to dense multi-pill scenes.
    - **Design Motivation**: To simulate real clinical and home medication scenarios rather than controlled laboratory settings. Dosette boxes are a widely used medication management tool among elderly patients.

2. **Fine-Grained Category Design**:
    - **Function**: The 32 pill classes deliberately include visually highly similar categories.
    - **Mechanism**: Pill A and Pill B share similar shapes but differ in color; Pill B and Pill C share similar colors but differ in shape — compelling the model to learn both shape and color features simultaneously. The 32-Pills subset includes multiple small white pills that are indistinguishable by coarse-grained features alone.
    - **Design Motivation**: To prevent models from relying on simple cues (e.g., color) while overlooking fine morphological differences, better reflecting the clinical challenge of recognizing large numbers of white or featureless pills.

3. **Few-Shot Evaluation Protocol**:
    - **Function**: FsDet (Faster R-CNN + ResNet + FPN) is used to conduct few-shot detection evaluation with base/novel class splits.
    - **Mechanism**: MEDISEG and CURE are used separately as base training sets; the backbone and RPN are frozen while only ROI heads are fine-tuned; evaluation is performed under 1/5/10-shot settings. A dedicated "occlusion-only" test set is constructed to assess extreme scenarios.
    - **Design Motivation**: To validate that dataset scene complexity — rather than data volume alone — contributes to few-shot representation transfer.

### Loss & Training

- YOLOv8/v9: Genetic algorithm hyperparameter search over 70 epochs (lr=0.01009, momentum=0.94, weight_decay=0.00048), best fitness=0.81253.
- Data split: 70% train / 20% val / 10% test.
- Few-shot: backbone and RPN are frozen; only ROI heads are fine-tuned with re-initialized classification layers.

## Key Experimental Results

### Main Results

| Dataset | Model | mAP@50 | mAP@50-95 | Precision | Recall |
|---------|-------|--------|-----------|-----------|--------|
| 3-Pills | YOLOv8 | 99.4% | 95.0% | 99.7% | 99.7% |
| 3-Pills | YOLOv9 | 99.5% | 96.5% | 99.6% | 99.8% |
| 32-Pills | YOLOv8 | 62.2% | 50.9% | 62.8% | 57.4% |
| 32-Pills | YOLOv9 | 80.1% | 68.4% | 81.2% | 73.7% |

### Ablation Study

| Few-Shot Setting | MEDISEG fg_cls_acc | CURE fg_cls_acc | Gain |
|------------------|--------------------|-----------------|------|
| 1-shot (occlusion set) | 0.406 | 0.131 | 3.1× |
| 5-shot (occlusion set) | 0.625 | 0.372 | 1.7× |
| 10-shot (occlusion set) | 0.740 | 0.558 | 1.3× |

| Dimension | MEDISEG | NIH Pillbox | CURE |
|-----------|---------|-------------|------|
| # Images | 8,262 | 133,774 | ~1,000 |
| # Classes | 32 | 4,392 | 196 |
| Instance segmentation | ✓ Complete | ✗ | Partial |
| Multi-pill scenes | ✓ (1–13 pills/image) | ✗ | ✗ |
| Publicly available | ✓ (CC BY 4.0) | Discontinued | ✓ |

### Key Findings

- YOLOv9 substantially outperforms YOLOv8 on 32-Pills (mAP@50: 80.1% vs. 62.2%), indicating that stronger feature fusion is critical for fine-grained recognition.
- Misclassifications primarily stem from side-view images of visually similar pills — top-view images are distinguishable, while side profiles are nearly identical.
- Few-shot performance gaps are most pronounced on the occlusion-only test set, demonstrating that multi-pill scene pretraining on MEDISEG confers greater occlusion robustness.
- The advantage of MEDISEG diminishes as the number of shots increases (3.1×→1.3×), suggesting its benefit is most critical in extremely low-data regimes.

## Highlights & Insights

- The dataset design philosophy is well-considered: deliberately constructing visually similar classes and realistic occlusion scenarios rather than simply accumulating data volume.
- The few-shot evaluation goes beyond standard test sets by constructing a dedicated occlusion-only subset for stress testing — the evaluation methodology is more instructive than the results themselves.
- An interesting finding is validated: the scene complexity of training data — rather than sheer volume — is the key factor for few-shot generalization.
- CC BY 4.0 open license + COCO format + GitHub code = high usability.

## Limitations & Future Work

- The dataset covers only 32 pill classes, whereas clinical settings involve thousands of distinct drug types.
- Images were captured exclusively with an iPhone 12, leaving cross-device generalization unvalidated.
- Variation in dynamic lighting conditions and background diversity remains limited.
- Evaluation is restricted to object detection and segmentation; performance on semantic understanding tasks has not been assessed.
- No prospective validation in real clinical environments has been conducted.

## Related Work & Insights

- **vs. NIH Pillbox**: The largest pill dataset (133K images), but discontinued and lacking instance segmentation annotations. MEDISEG is smaller in scale but offers complete annotations and multi-pill scenes.
- **vs. CURE**: The closest competing dataset, with partial segmentation annotations that are incomplete and include synthetic images. Few-shot experiments directly demonstrate the advantage of real occluded scenes.
- **Design Insight**: The concept of constructing an occlusion-only evaluation subset is worth adopting — designing targeted hard cases is more convincing than reporting numbers on standard benchmarks.

## Rating

- ⭐⭐⭐ **Novelty**: Primarily a dataset contribution with no significant methodological innovation, though the dataset design reflects careful consideration.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Multi-model, multi-subset, few-shot evaluation, and hyperparameter search yield solid validation.
- ⭐⭐⭐⭐ **Writing Quality**: Follows standard dataset paper conventions with clear structure and detailed tables.
- ⭐⭐⭐ **Value**: Practically valuable for medication safety AI, though domain scope is relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video](robotseg_a_model_and_dataset_for_segmenting_robots_in_image_and_video.md)
- [\[CVPR 2026\] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)
- [\[CVPR 2026\] Phrase-Instance Alignment for Generalized Referring Segmentation](phrase-instance_alignment_for_generalized_referring_segmentation.md)
- [\[CVPR 2026\] CA-LoRA: Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](ca-lora_concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)
- [\[CVPR 2026\] ELVIS: Enhance Low-Light for Video Instance Segmentation in the Dark](elvis_enhance_low-light_for_video_instance_segmentation_in_the_dark.md)

</div>

<!-- RELATED:END -->
