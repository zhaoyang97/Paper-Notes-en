---
title: >-
  [Paper Note] Care-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment
description: >-
  [NeurIPS 2025][Medical Imaging][Parkinson's disease] This work introduces Care-PD — the largest multi-site anonymized 3D mesh dataset for Parkinson's disease (PD) gait analysis to date, comprising 9 cohorts…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Parkinson's disease"
  - "gait analysis"
  - "SMPL 3D mesh"
  - "multi-site dataset"
  - "motion representation learning"
date: 2026-05-08
content_hash: e7491888dab72c29
---

# Care-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment

**Conference**: NeurIPS 2025  
**arXiv**: [2510.04312](https://arxiv.org/abs/2510.04312)  
**Code**: [Website](https://neurips2025.care-pd.ca)  
**Area**: Medical Imaging  
**Keywords**: Parkinson's disease, gait analysis, SMPL 3D mesh, multi-site dataset, motion representation learning

## TL;DR
This work introduces Care-PD — the largest multi-site anonymized 3D mesh dataset for Parkinson's disease (PD) gait analysis to date, comprising 9 cohorts, 8 clinical centers, 362 subjects, and 8,477 walking bouts. It provides a systematic benchmark for UPDRS gait scoring and motion pre-training tasks, demonstrating that fine-tuning on Care-PD reduces MPJPE from 60.8 mm to 7.5 mm and improves F1 by 17 percentage points.

## Background & Motivation

**Background**: Gait assessment for PD currently relies predominantly on subjective clinician scoring (UPDRS). Although automated motion analysis can provide objective metrics, progress is constrained by small-scale, single-site datasets.

**Limitations of Prior Work**: Existing PD gait datasets are limited in scale, modality (IMU-only or video-only), standardized representation, and clinical annotation, with models suffering severe performance degradation under cross-site generalization. General-purpose motion datasets (e.g., Human3.6M, AMASS) contain only healthy subjects and lack pathological gait patterns.

**Key Challenge**: The absence of a large-scale, multi-site, standardized, and publicly available PD gait motion dataset is the fundamental bottleneck impeding the development of generalizable models.

**Goal**: To construct a unified 3D representation format (SMPL mesh) by aggregating RGB video and optical motion capture data from multiple sources, and to provide a publicly available benchmark with clinical annotations.

**Key Insight**: SMPL is adopted as a unified, privacy-preserving 3D mesh representation to normalize heterogeneous acquisition modalities (MoCap and RGB) into a common space.

**Core Idea**: Unified data format + multi-site aggregation + hierarchical evaluation protocols (within-site / cross-site / LODO / MIDA) + motion pre-training validation.

## Method

### Overall Architecture
Care-PD is a dataset and benchmark contribution rather than a novel model. The core pipeline is: (1) collecting RGB video and MoCap data from 9 cohorts; (2) preprocessing all data into 30 Hz SMPL mesh sequences with anonymization; (3) designing two categories of benchmark tasks — clinical score prediction and motion pre-training; (4) evaluating 7 motion encoders and handcrafted feature baselines under 4 generalization protocols.

### Key Designs

1. **Data Unification Pipeline**:

    - Function: Converts heterogeneous inputs (RGB video / optical MoCap) into unified SMPL 3D mesh sequences.
    - Mechanism: MoCap data undergoes quality control → joint normalization → SparseFusion-based SMPL fitting → downsampling to 30 Hz; RGB video is processed via WHAM monocular mesh recovery → visual verification → non-walking segment removal → Kabsch algorithm correction for ceiling-mounted camera tilt.
    - Design Motivation: SMPL provides a device-agnostic unified anatomical framework. Identity anonymization is achieved by releasing only texture-free mesh parameters (excluding shape parameters).

2. **Clinical Score Prediction Task**:

    - Function: Predicts UPDRS gait scores $S \in \{0,1,2,3\}$ from walking sequences $\mathbf{M}^{1:T}$.
    - Mechanism: Variable-length inputs are segmented into fixed-length clips $\{\mathbf{p}_1^{1:N}, ...\}$ according to encoder requirements. A frozen pre-trained motion encoder $\mathcal{E}$ extracts features, a lightweight classification head $\Phi$ predicts clip-level scores, and sequence-level predictions are obtained via majority voting.
    - Seven SOTA motion encoders are evaluated (POTR, MixSTE, PoseFormerV2, MotionBERT, MotionAGFormer, MotionCLIP, MoMask) alongside a Random Forest handcrafted feature baseline.

3. **Four Generalization Evaluation Protocols**:

    - Within-dataset: Leave-One-Subject-Out cross-validation.
    - Cross-dataset: Train on one dataset, test on the remaining datasets.
    - LODO (Leave One Dataset Out): Train on $D-1$ datasets, test on the held-out dataset.
    - MIDA (Multi-dataset In-domain Adaptation): LODO with additional fine-tuning of the classification head on a small amount of target-domain data.
    - Design Motivation: These protocols simulate a progression from controlled experiments to real-world deployment, progressively exposing domain transfer bottlenecks.

4. **Motion Pre-Training Task**:

    - Function: Evaluates the utility of Care-PD for two pre-training tasks — 2D-to-3D keypoint lifting and 3D reconstruction.
    - Mechanism: Four training configurations are compared — zero-shot, fine-tuned on Care-PD, fine-tuned on healthy gait, and trained from scratch.
    - Design Motivation: To empirically validate the irreplaceability of clinical data — fine-tuning on healthy gait yields substantially smaller gains than fine-tuning on pathological gait data.

### Dataset Scale
The dataset comprises 362 subjects, 8,477 walking bouts, and 18.66 hours of 3D mesh sequences, spanning 8 clinical centers across 6 countries and covering both RGB and MoCap modalities.

## Key Experimental Results

### Main Results (Impact of Motion Pre-Training on Downstream Tasks)

| Model | Training Data | Fine-tuning Data | MPJPE ↓ (mm) | PA-MPJPE ↓ | F1 ↑ |
|------|---------|---------|-------------|------------|------|
| MotionAGFormer | H3.6M | — | 60.7 | 21.4 | 48.1 |
| MotionAGFormer | H3.6M | Healthy Gait | 29.8 | 7.3 | 50.1 |
| MotionAGFormer | H3.6M | **Care-PD** | **7.5** | **2.6** | **65.1** |
| MotionAGFormer | Care-PD | — | 9.0 | 3.2 | 62.3 |
| MoMask | HumanML3D | — | 22.5 | 17.8 | 41.4 |
| MoMask | HumanML3D | Healthy Gait | 22.3 | 13.7 | 40.6 |
| MoMask | HumanML3D | **Care-PD** | **8.7** | **6.3** | **62.7** |
| MoMask | Care-PD | — | 9.6 | 7.3 | 59.8 |

Fine-tuning on Care-PD reduces MotionAGFormer's MPJPE from 60.7 mm to 7.5 mm (−87.6%) and improves F1 from 48.1 to 65.1 (+17 pp). Fine-tuning on healthy gait alone yields only a 2 pp F1 improvement, confirming that performance gains originate from pathological kinematic features.

### Evaluation Protocol Comparison

| Protocol | BMClab F1₀₋₂ | PD-GaM F1₀₋₂ | Notes |
|------|-------------|--------------|------|
| Within-dataset (LOSO) | ~0.68 | ~0.73 | Best within-site performance |
| Cross-dataset | 0.20–0.40 | 0.40+ | Drop of 0.2–0.4 across sites |
| LODO | ~0.50 | ~0.50 | Multi-dataset training helps but domain gap persists |
| MIDA | 0.74–0.78 | 0.63–0.70 | Small amount of in-domain data yields large gains |

### Key Findings
- F1 typically drops by 0.2–0.4 under cross-site evaluation, revealing severe domain shift.
- MoMask (VQ-VAE architecture) is the most robust under cross-site settings, achieving average cross-site F1 > 0.40.
- MIDA substantially outperforms LODO (e.g., from ~0.50 to ~0.78 on BMClab), underscoring the critical role of minimal in-domain supervision.
- Subgroup analyses show that models can differentiate medication on/off states (Cliff's $\Delta = 0.42$), freezing/non-freezing gait ($\Delta = 0.25$), and PD vs. healthy individuals ($\Delta = 0.50$).

## Highlights & Insights
- **Unified SMPL Representation**: Using SMPL 3D mesh to unify MoCap and video modalities simultaneously addresses cross-modal alignment and privacy protection — a strategy transferable to other clinical scenarios requiring multi-modal motion data fusion.
- **Four-Level Evaluation Protocol**: The within → cross → LODO → MIDA progression systematically exposes generalization bottlenecks, providing substantially more diagnostic information than reporting a single accuracy figure.
- **Irreplaceability of Pathological Data**: Fine-tuning on healthy gait yields negligible gains; only pathological data enables models to learn clinically relevant features — an important empirical finding.

## Limitations & Future Work
- UPDRS gait score 3 (severe impairment) is extremely rare, appearing only in the PD-GaM and 3DGait subsets, substantially impairing classifier performance.
- All motion encoders are evaluated under a frozen probe setting; end-to-end fine-tuning upper bounds remain unexplored.
- SMPL fitting introduces inherent errors, particularly in the RGB-to-SMPL pipeline (WHAM), which may fail under severe gait abnormalities.
- Longitudinal tracking — capturing disease progression in the same patient over time — is not addressed.
- Incorporating pathological gait data as a curriculum learning or data augmentation strategy into general-purpose motion encoder training warrants future exploration.

## Related Work & Insights
- **vs. Human3.6M / AMASS**: General-purpose motion datasets are large-scale but contain only healthy activities and lack pathological patterns; Care-PD is smaller but targeted, with fine-tuning gains far exceeding those from general-purpose data.
- **vs. DaphNet / mPower (IMU datasets)**: IMU datasets are portable but lack anatomical context; Care-PD's 3D mesh provides complete whole-body kinematic information.
- **vs. CellProfiler paradigm**: Analogous to the debate between handcrafted and learned features in microscopy image analysis, this work demonstrates that learned representations outperform handcrafted gait features in terms of generalizability.

## Rating
- Novelty: ⭐⭐⭐⭐ First multi-site 3D mesh PD gait dataset with a well-designed unification pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Seven encoders × four evaluation protocols × subgroup analyses + pre-training tasks — exceptionally comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Dataset description is thorough and experiments are clearly organized, though the paper is lengthy.
- Value: ⭐⭐⭐⭐⭐ Addresses the critical gap in large-scale, publicly available, standardized datasets for PD gait analysis, offering long-term community value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] From Black Box to Biomarker: Sparse Autoencoders for Interpreting Speech Models of Parkinson's Disease](from_black_box_to_biomarker_sparse_autoencoders_for_interpreting_speech_models_o.md)
- [\[NeurIPS 2025\] RAD: Towards Trustworthy Retrieval-Augmented Multi-modal Clinical Diagnosis](rad_towards_trustworthy_retrieval-augmented_multi-modal_clinical_diagnosis.md)
- [\[NeurIPS 2025\] RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis](ram-w600_a_multi-task_wrist_dataset_and_benchmark_for_rheumatoid_arthritis.md)
- [\[NeurIPS 2025\] DermaCon-IN: A Multi-concept Annotated Dermatological Image Dataset of Indian Skin Disorders](dermacon-in_a_multi-concept_annotated_dermatological_image_dataset_of_indian_ski.md)
- [\[NeurIPS 2025\] STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology](starc-9_a_large-scale_dataset_for_multi-class_tissue_classification_for_crc_hist.md)

</div>

<!-- RELATED:END -->
