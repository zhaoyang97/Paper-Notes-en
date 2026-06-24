---
title: >-
  [Paper Note] CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools
description: >-
  [CVPR 2025][Medical Imaging][Surgical Tool Tracking] This paper presents the CholecTrack20 dataset, which is the first to introduce three perspective-based trajectory definitions (intraoperative, intra-abdominal, and visibility) for laparoscopic tool tracking. It comprises 20 full surgical videos, 35K+ frames, and over 65K+ annotated tool instances. Benchmarking results indicate that current SOTA methods (<45% HOTA) fall far short of clinical demands.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Surgical Tool Tracking"
  - "Multi-Perspective Tracking"
  - "Laparoscopy"
  - "Dataset"
  - "Multi-Class Multi-Object"
date: 2026-05-08
content_hash: 3379a3ca02a51e9b
---

# CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools

**Conference**: CVPR 2025  
**arXiv**: [2312.07352](https://arxiv.org/abs/2312.07352)  
**Code**: [GitHub](https://github.com/camma-public/cholectrack20)  
**Area**: Medical Imaging / Surgical Tool Tracking  
**Keywords**: Surgical Tool Tracking, Multi-Perspective Tracking, Laparoscopy, Dataset, Multi-Class Multi-Object

## TL;DR
This paper presents the CholecTrack20 dataset, which is the first to introduce three perspective-based trajectory definitions (intraoperative, intra-abdominal, and visibility) for laparoscopic tool tracking. It comprises 20 full surgical videos, 35K+ frames, and over 65K+ annotated tool instances. Benchmarking results indicate that current SOTA methods (<45% HOTA) fall far short of clinical demands.

## Background & Motivation

**Background**: Tool tracking in surgical videos is a crucial task for computer-assisted surgery, supporting skill assessment, safety zone estimation, and human-robot collaboration. Most existing methods are trained on general tracking datasets, resulting in severe performance drops under challenging surgical conditions (e.g., bleeding, smoke, specular reflection, and tools entering/leaving the field of view).

**Limitations of Prior Work**: Existing surgical tracking datasets employ overly generalized tracking definitions—lacking clear guidelines on how to handle trajectory IDs when a tool leaves the camera's field of view or departs the abdominal cavity. Consequently, different clinical applications cannot obtain their required trajectory types, limiting the practical deployment of AI in surgery.

**Key Challenge**: The definition of a tool "trajectory" varies across clinical application scenarios—skill assessment requires full-surgery tracking (intraoperative), workflow analysis needs intra-abdominal tracking (following tools inside the body cavity), and real-time feedback necessitates visibility-based tracking (relying on the camera field of view). A single tracking definition cannot fulfill all these requirements simultaneously.

**Goal**: To define the three perspectives of surgical tool tracking and construct a high-quality annotated dataset, filling the gap in training data for surgical AI.

**Key Insight**: To define the formulation of tracking backwards from actual clinical requirements—different clinical tasks necessitate trajectories of varying granularities.

**Core Idea**: A three-perspective tracking formulation (intraoperative, intra-abdominal, and visibility) coupled with a richly annotated dataset containing spatial locations, tool classes, identity, operators, surgical phases, and visual challenges.

## Method

### Overall Architecture
Based on the raw videos from Cholec80 and CholecT50, 20 complete surgical videos are selected and sampled at 1 fps. Four trained annotators label bounding boxes, tool categories (7 classes), operators (4 classes), surgical phases (7 types), visual challenges (8 classes), and trajectory IDs under the three perspectives. All annotations undergo strict quality control.

### Key Designs

1. **Three-Perspective Trajectory Formulation**:

    - Function: To provide tailored tracking definitions for different clinical applications
    - Mechanism: (a) Intraoperative trajectory—lifetime tracking from the tool's first appearance to its final appearance in the patient, requiring re-identification across occlusions, out-of-field occurrences, and re-insertions; (b) Intra-abdominal trajectory—a single trajectory from when a tool enters the abdominal cavity until it exits; subsequent re-entry initiates a new trajectory; (c) Visibility trajectory—a continuous segment where the tool is visible in the camera's field of view forms one trajectory
    - Design Motivation: Skill assessment requires intraoperative tracking (analyzing the overall usage of tools), safety and risk prediction requires intra-abdominal tracking (analyzing operations inside the cavity), and real-time assistance demands visibility tracking (judging the currently visible state)

2. **Rich Multi-Dimensional Annotation**:

    - Function: To support the training and evaluation of various surgical AI tasks
    - Mechanism: Each tool instance is annotated with: spatial coordinates (bounding boxes), category (7 tools), trajectory IDs for the three perspectives, operator (surgeon/assistant and primary/secondary hand), surgical phase, and visual challenge types for the current frame
    - Design Motivation: Identifying tool identities relies not only on appearance but also on integrating clinical knowledge such as operators and trocar port locations

3. **Rigorous Annotation Quality Control**:

    - Function: To ensure annotation consistency and accuracy
    - Mechanism: Intra-rater agreement (Jaccard 99.4%, Cohen's Kappa 94.6%), inter-rater agreement (Jaccard 91.8%, Kappa 95.2%), and surgical experts arbitrating ambiguous cases (where 133 of 758 uncertain samples required correction)
    - Design Motivation: Annotating surgical data requires domain expertise; thus, quality control is the cornerstone of dataset reliability

### Loss & Training
As a dataset paper, this work does not introduce a specific model training strategy. Benchmarking experiments are conducted using existing tracking methods such as DeepSORT and ByteTrack.

## Key Experimental Results

### Main Results

| Method | HOTA (average across three perspectives) | Description |
|------|----------------|------|
| State-of-the-art methods | <45% | Far from clinical requirements |
| Best under visibility perspective | ~40% | Simplest tracking definition |
| Poorest under intraoperative perspective | ~30% | Demands re-identification across long-term occlusions |

### Dataset Statistics

| Metric | Value |
|------|------|
| Number of videos | 20 full surgeries |
| Total frames | 35,000+ |
| Annotated tool instances | 65,000+ |
| Tool classes | 7 categories |
| Surgical phases | 7 phases |
| Visual challenge types | 8 categories |

### Key Findings
- All existing tracking methods perform poorly in surgical scenarios (<45% HOTA), demonstrating that generic tracking technologies require surgical-specific adaptations.
- Bleeding and smoke are the most significant visual challenges causing performance degradation.
- Intra-abdominal tracking is the most challenging because it requires inferring the states of tools when they are outside the camera's field of view.
- Tool replacement and re-insertion are the primary causes of ID switches.

## Highlights & Insights
- The three-perspective tracking formulation is a highly original contribution derived from a deep understanding of clinical needs.
- The annotation scheme integrates visual cues with clinical knowledge (e.g., inferring operators from trocar ports), reflecting the domain-specific nature of surgical AI.
- The benchmark results of <45% HOTA clearly demonstrate the massive gap between current methodologies and clinical necessities.

## Limitations & Future Work
- It only includes a single surgery type, namely laparoscopic cholecystectomy.
- The 1 fps sampling rate may miss rapid tool movements.
- Seven tool classes might not cover more complex surgical procedures.
- Future work can extend this to other surgical procedures and higher temporal resolutions.

## Related Work & Insights
- **vs ATLAS Dione**: Only contains detection annotations without multi-perspective tracking.
- **vs CholecT50**: Provides tool-tissue interaction annotations but lacks tracking IDs.
- **vs MOTChallenge/DanceTrack**: General video tracking datasets that do not account for surgical-specific tool entries/departures and visual challenges.

## Rating
- Novelty: ⭐⭐⭐⭐ The three-perspective tracking definition is a significant formal contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete baseline benchmarking and quality control analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definitions and comprehensive annotation processes.
- Value: ⭐⭐⭐⭐⭐ Fills a gap in dataset resources within the surgical AI field, holding direct clinical application significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Bridging Vision and Language for Robust Context-Aware Surgical Point Tracking: The VL-SurgPT Dataset and Benchmark](../../AAAI2026/medical_imaging/bridging_vision_and_language_for_robust_context-aware_surgical_point_tracking_th.md)
- [\[NeurIPS 2025\] RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis](../../NeurIPS2025/medical_imaging/ram-w600_a_multi-task_wrist_dataset_and_benchmark_for_rheumatoid_arthritis.md)
- [\[CVPR 2025\] Thin-Shell-SfT: Fine-Grained Monocular Non-Rigid 3D Surface Tracking with Neural Deformation Fields](thin-shell-sft_fine-grained_monocular_non-rigid_3d_surface_tracking_with_neural_.md)
- [\[NeurIPS 2025\] STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology](../../NeurIPS2025/medical_imaging/starc-9_a_large-scale_dataset_for_multi-class_tissue_classification_for_crc_hist.md)
- [\[CVPR 2025\] Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline](interactive_medical_image_segmentation_a_benchmark_dataset_and_baseline.md)

</div>

<!-- RELATED:END -->
