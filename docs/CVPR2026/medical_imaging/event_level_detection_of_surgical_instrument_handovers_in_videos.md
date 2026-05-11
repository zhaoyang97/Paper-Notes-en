---
title: >-
  [Paper Note] Event-Level Detection of Surgical Instrument Handovers in Videos
description: >-
  [CVPR 2026][Medical Imaging][surgical video] This paper proposes a spatiotemporal visual framework for detecting instrument handovers in real surgical videos. It combines ViT-based spatial feature extraction with unidire…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "surgical video"
  - "instrument handover"
  - "ViT-LSTM"
  - "multi-task"
  - "event detection"
date: 2026-05-08
content_hash: 32e557f39baf6e62
---

# Event-Level Detection of Surgical Instrument Handovers in Videos

**Conference**: CVPR 2026
**arXiv**: [2604.07577](https://arxiv.org/abs/2604.07577)
**Code**: Available
**Area**: Medical Imaging
**Keywords**: surgical video, instrument handover, ViT-LSTM, multi-task, event detection

## TL;DR

This paper proposes a spatiotemporal visual framework for detecting instrument handovers in real surgical videos. It combines ViT-based spatial feature extraction with unidirectional LSTM temporal modeling, and employs multi-task learning to jointly predict handover events and their directions, achieving an event-level detection F1 of 0.84 on kidney transplant surgical videos.

## Background & Motivation

Reliable monitoring of surgical instrument handovers is critical for maintaining procedural efficiency and patient safety. Failed handovers during surgery can lead to serious adverse events such as retained instruments. Automatically detecting handovers from intraoperative video remains highly challenging due to frequent occlusions, cluttered backgrounds, dynamic lighting, and the inherently temporal nature of handover events, which renders single-frame analysis insufficient.

Prior work SurgiGuard leverages CLIP features and graph-based reasoning for handover detection, but primarily relies on frame-level features without explicit temporal modeling. This paper introduces a spatiotemporal architecture combining ViT and LSTM, validated on real intraoperative footage rather than simulated environments.

## Method

### Overall Architecture

An 8-frame sequence is sampled from the video (stride 4, covering a 29-frame temporal window). A ViT independently extracts spatial features from each frame; after linear projection, these features are fed into a unidirectional LSTM for temporal aggregation. The shared representation is then passed to two task-specific heads.

### Key Designs

1. **ViT Spatial Feature Extraction**: A pretrained ViT backbone is employed, with the first 18 transformer layers frozen and the upper layers fine-tuned to adapt to the handover analysis task. Frame-level features are projected into a 64-dimensional embedding space.

2. **LSTM Temporal Aggregation**: A unidirectional LSTM is chosen over Transformer-based temporal models. Given the limited scale of annotated data and the sparse distribution of handover events, the strong sequential inductive bias of LSTM is better suited to modeling short interaction sequences.

3. **Multi-Task Joint Prediction**: The shared representation is fed into a binary handover detection head (sigmoid) and a direction classification head (softmax: scrub nurse receives / scrub nurse passes). Joint optimization avoids the error accumulation inherent in cascaded pipelines.

### Loss & Training

$\mathcal{L} = \lambda_{\text{det}} \cdot \mathcal{L}_{\text{det}} + \lambda_{\text{dir}} \cdot \mathcal{L}_{\text{dir}}$. $\mathcal{L}_{\text{det}}$ uses weighted BCE to handle class imbalance between positive and negative samples; $\mathcal{L}_{\text{dir}}$ uses weighted CE computed exclusively on positive samples. Sequence labels are determined by majority voting over the central 5 frames (classes: scrub nurse receives / scrub nurse passes / scrub nurse idle). Event-level evaluation extracts discrete handover events from sequence-level predictions via Gaussian smoothing followed by peak detection. During training, the first 18 ViT layers are frozen and only the upper layers are fine-tuned; frame-level features are projected to 64-dimensional embeddings before being passed to the LSTM. Data augmentation strategies are applied to reduce interference from surgical background clutter and occlusions. The dataset comprises intraoperative videos from 5 kidney transplant surgeries, totaling 484 handover events.

## Key Experimental Results

### Main Results

| Model | Detection F1 | Direction Mean F1 |
|---|---|---|
| Multi-task ViT-LSTM | **0.84** | **0.72** |
| Single-task ViT-LSTM | 0.79 | 0.63 |
| VideoMamba | 0.84 | 0.61 |

### Key Findings

- Multi-task learning outperforms single-task learning on both detection (F1 0.84 vs. 0.79) and direction classification (0.72 vs. 0.63).
- Compared to VideoMamba, detection performance is on par, but direction classification is substantially superior.
- Layer-CAM visualizations demonstrate that the model correctly attends to hand–instrument interaction regions.

## Highlights & Insights

- Validation on real kidney transplant surgical videos carries direct clinical relevance.
- Event-level evaluation (rather than frame-level) better aligns with clinical perception.
- Layer-CAM interpretability analysis enhances clinical trustworthiness.
- The unified multi-task loss avoids error accumulation from cascaded pipelines, with detection and direction classification sharing a unified spatiotemporal representation.
- The key rationale for choosing unidirectional LSTM over Transformer-based temporal models is the limited annotated data scale and sparse event distribution; LSTM's strong sequential inductive bias is better suited for short interaction sequence modeling.
- Dedicated comparison against the VideoMamba baseline reveals the impact of different temporal modeling strategies.

## Limitations & Future Work

- The dataset is small (5 surgeries, 484 handover events), and generalizability requires further validation.
- Only handovers between the scrub nurse and the primary surgeon are detected; more complex multi-person interactions are not addressed.
- No direct comparison with CLIP+graph-reasoning methods such as SurgiGuard is performed on the same dataset.
- The Gaussian smoothing parameters and peak detection thresholds for event-level evaluation require tuning for different surgical procedure types.
- The potential of bidirectional LSTM or Transformer-based temporal models on larger datasets remains unexplored.
- Auxiliary information such as instrument tracking is not leveraged to enhance handover detection.
- Data augmentation includes cropping and flipping strategies to reduce interference from surgical background clutter.
- Event-level evaluation is more meaningful for clinical deployment and avoids the overestimation problem associated with frame-level evaluation.

## Rating

- Novelty: ⭐⭐⭐ — Methodological design is relatively standard.
- Technical Depth: ⭐⭐⭐ — The ViT + LSTM + multi-task combination is straightforward.
- Experimental Thoroughness: ⭐⭐⭐ — Dataset scale is limited to 5 surgeries and 484 handover events.
- Value: ⭐⭐⭐⭐ — The surgical safety application scenario is well-defined, with strong clinical translation potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Synergistic Bleeding Region and Point Detection in Laparoscopic Surgical Videos](synergistic_bleeding_region_and_point_detection_in_laparoscopic_surgical_videos.md)
- [\[CVPR 2026\] Benchmarking Endoscopic Surgical Image Restoration and Beyond](benchmarking_endoscopic_surgical_image_restoration_and_beyond.md)
- [\[CVPR 2026\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[CVPR 2026\] LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings](lemon_a_large_endoscopic_monocular_dataset_and_foundation_model_for_perception_in.md)
- [\[CVPR 2026\] Unlocking Positive Transfer in Incrementally Learning Surgical Instruments: A Self-reflection Hierarchical Prompt Framework](unlocking_positive_transfer_in_incrementally_learning_surgical_instruments_a_sel.md)

</div>

<!-- RELATED:END -->
