---
title: >-
  [Paper Note] RegFormer: Transferable Relational Grounding for Efficient Weakly-Supervised HOI Detection
description: >-
  [CVPR 2026][Human Understanding][Human-Object Interaction Detection] RegFormer proposes a lightweight relational grounding Transformer module that, under weak supervision with only image-level annotations…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "Human-Object Interaction Detection"
  - "Weak Supervision"
  - "Relational Grounding"
  - "Interactiveness Learning"
  - "Zero-Shot Transfer"
date: 2026-05-08
content_hash: 7d45d80245f66390
---

# RegFormer: Transferable Relational Grounding for Efficient Weakly-Supervised HOI Detection

**Conference**: CVPR 2026
**arXiv**: [2604.00507](https://arxiv.org/abs/2604.00507)  
**Code**: [https://github.com/mlvlab/RegFormer](https://github.com/mlvlab/RegFormer)  
**Area**: Human Understanding
**Keywords**: Human-Object Interaction Detection, Weak Supervision, Relational Grounding, Interactiveness Learning, Zero-Shot Transfer

## TL;DR

RegFormer proposes a lightweight relational grounding Transformer module that, under weak supervision with only image-level annotations, leverages spatially-grounded HO queries and interactiveness-aware learning to directly transfer image-level reasoning to instance-level HOI detection without additional training, achieving performance close to fully supervised methods.

## Background & Motivation

HOI detection requires localizing humans and objects and recognizing their interaction relationships. Fully supervised methods demand expensive per-pair interaction annotations. Weakly supervised methods rely solely on image-level labels (indicating which HOI triplets appear in an image), but face two critical challenges.

**Computational efficiency**: Existing methods enumerate all human-object pairs and process them individually, causing computation to scale dramatically with the number of pairs. **False positives**: Non-interacting human-object combinations generate a large number of false positives, interfering with accurate instance-level reasoning.

## Method

### Overall Architecture

**Image-level training phase**: RegFormer constructs spatially-grounded HO queries from spatial feature maps → pairwise instance encoder → interaction decoder predicts interactions. **Instance-level inference phase**: Instance constraints provided by an external detector are used to guide HO query construction and interactiveness scoring, directly transferring to instance-level HOI detection.

### Key Designs

1. **Spatially-Grounded Queries**:

    - **Function**: Construct HO query pairs from spatial feature maps that embed spatial relational cues.
    - **Mechanism**: CLIP spatial feature maps serve as the foundation; HO queries are built by aggregating features from regions relevant to human-object pairs. This enables queries to inherently encode spatial information, allowing the model to implicitly learn the spatial relationships required for interaction.
    - **Design Motivation**: Directly using instance features from a detector would tightly couple the classifier to that detector, necessitating retraining upon detector replacement.

2. **Interactiveness-Aware Learning**:

    - **Function**: Learn an interactiveness score for each human-object pair to suppress non-interacting combinations.
    - **Mechanism**: An implicit localization signal is introduced to learn whether each human-object pair is genuinely interacting. This score acts as an explicit gating mechanism at inference time, filtering out non-interacting pairs and reducing false positives.
    - **Design Motivation**: The primary source of noise in weakly supervised settings is non-interacting human-object combinations.

3. **Zero-Shot Transfer from Image-Level to Instance-Level**:

    - **Function**: Transfer image-level reasoning to instance-level detection without additional training.
    - **Mechanism**: At inference time, human/object instances from an external detector constrain HO query construction and the interactiveness scoring regions. Since spatially-grounded interaction cues are learned during training, they can be directly applied to distinguish between different instance pairs.
    - **Design Motivation**: Eliminate the need for an additional adaptation step from weak to full supervision.

### Loss & Training

Multi-label classification loss (image-level) + regularization on interactiveness scores. Training uses only image-level HOI triplet annotations.

## Key Experimental Results

### Main Results

| Method | Supervision | HICO-DET mAP | V-COCO AP | Inference Efficiency |
|--------|-------------|-------------|-----------|----------------------|
| Fully Supervised SOTA | Full | High | High | — |
| Prev. SOTA (Weak) | Weak | Medium | Medium | Slow |
| **RegFormer** | **Weak** | **Near Full Sup.** | **Near Full Sup.** | **Efficient** |

RegFormer achieves performance close to fully supervised methods under weak supervision, while substantially outperforming prior weakly supervised methods in inference efficiency.

### Ablation Study

| Configuration | mAP | False Positive Rate | Note |
|---------------|-----|---------------------|------|
| w/o Spatial Grounding | Low | High | Queries lack spatial information |
| w/o Interactiveness Score | Medium | High | More false positives |
| Full RegFormer | Best | Low | Both components complement each other |

### Key Findings

- Spatially-grounded queries enable the model to learn instance-level localization cues from image-level supervision.
- Interactiveness scoring effectively suppresses false positives, with inference time increasing only marginally as the number of instance pairs grows.
- Weakly supervised performance approaches that of fully supervised methods, substantially reducing annotation requirements.

## Highlights & Insights

- **Zero-shot transfer from weak supervision to instance-level detection**: An elegant design—training requires only image-level labels, yet inference directly supports instance-level detection without any bridging step.
- **Lightweight and efficient**: Inference time remains nearly constant as the number of instance pairs increases, addressing a critical efficiency bottleneck in weakly supervised HOI detection.
- **Detector-agnostic**: Not coupled to any specific detector; replacing the detector requires no retraining.

## Limitations & Future Work

- Performance remains dependent on the quality of the external detector.
- Generalization to rare interaction categories may be constrained by the training data distribution.
- Although image-level annotations are cheaper, manual labeling effort is still required.

## Related Work & Insights

- **vs. ML-Decoder**: ML-Decoder requires repeated cropping of paired regions, with computation scaling linearly with the number of instance pairs.
- **vs. Fully Supervised HOI (QPIC/CDN)**: RegFormer achieves comparable performance with weak annotations, significantly reducing annotation cost.

## Rating

- Novelty: ⭐⭐⭐⭐ The zero-shot transfer design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks with efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear and concise.
- Value: ⭐⭐⭐⭐ Reducing annotation requirements for HOI detection has practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Weakly Supervised Visible-Infrared Person Re-Identification via Heterogeneous Expert Collaborative Consistency Learning](../../ICCV2025/human_understanding/weakly_supervised_visible-infrared_person_re-identification_via_heterogeneous_ex.md)
- [\[ICCV 2025\] Bi-Level Optimization for Self-Supervised AI-Generated Face Detection](../../ICCV2025/human_understanding/bi-level_optimization_for_self-supervised_ai-generated_face_detection.md)
- [\[CVPR 2026\] Efficient Onboard Spacecraft Pose Estimation with Event Cameras and Neuromorphic Hardware](efficient_onboard_spacecraft_pose_estimation_with_event_cameras_and_neuromorphic_hardware.md)
- [\[CVPR 2026\] TriLite: Efficient WSOL with Universal Visual Features and Tri-Region Disentanglement](trilite_efficient_weakly_supervised_object_localization_with_universal_visual_fe.md)
- [\[CVPR 2026\] Unleashing Vision-Language Semantics for Deepfake Video Detection](unleashing_vision-language_semantics_for_deepfake_video_detection.md)

</div>

<!-- RELATED:END -->
