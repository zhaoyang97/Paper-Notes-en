---
title: >-
  [Paper Note] Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research
description: >-
  [ICLR 2026][Medical Imaging][intracranial EEG] This paper introduces the Omni-iEEG dataset (302 patients, 178 hours of high-resolution intracranial EEG recordings), defines standardized benchmark tasks and evaluation metrics grounded in clinical priors, and demonstrates that end-to-end modeling can match or surpass traditional biomarker-based approaches for epilepsy surgical planning.
tags:
  - ICLR 2026
  - Medical Imaging
  - intracranial EEG
  - epilepsy
  - high-frequency oscillations
  - benchmark
  - dataset
date: 2026-05-08
content_hash: b1a93ae7fb9f7b53
---

# Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research

**Conference**: ICLR 2026
**arXiv**: [2602.16072](https://arxiv.org/abs/2602.16072)
**Code**: [omni-ieeg.github.io/omni-ieeg](https://omni-ieeg.github.io/omni-ieeg/)
**Area**: Medical Imaging / Neuroscience
**Keywords**: intracranial EEG, epilepsy, high-frequency oscillations, benchmark, dataset

## TL;DR

This paper introduces the Omni-iEEG dataset (302 patients, 178 hours of high-resolution intracranial EEG recordings), defines standardized benchmark tasks and evaluation metrics grounded in clinical priors, and demonstrates that end-to-end modeling can match or surpass traditional biomarker-based approaches for epilepsy surgical planning.

## Background & Motivation

**Background**: Epilepsy affects more than 50 million people worldwide, and approximately 30% of patients have drug-resistant epilepsy. Surgical resection of the Epileptogenic Zone (EZ) represents the most effective strategy for achieving seizure freedom. Intracranial EEG (iEEG) is the gold standard for EZ localization.

**Limitations of Prior Work**: Existing public iEEG datasets (Open iEEG, Zurich, HUP, SourceSink) suffer from three major issues: (1) format heterogeneity—inconsistent sampling rates, channel naming conventions, and metadata; (2) lack of standardized benchmarks—different studies adopt different evaluation protocols, rendering results incomparable; (3) scarce annotations of pathological events—expert annotations of key biomarkers such as HFOs are rarely made publicly available.

**Key Challenge**: Machine learning methods are typically validated on single-center, small-scale datasets, raising concerns about generalizability. Meanwhile, the neuroscience community lacks a unified evaluation platform to fairly assess the clinical value of different methods.

**Goal**: The paper integrates data from eight epilepsy centers, harmonizes metadata under board-certified epileptologist supervision, and constructs the Omni-iEEG dataset and benchmark. It releases 36K+ expert-annotated pathological events, defines two primary tasks and three exploratory tasks, and provides comprehensive baselines spanning biomarker-driven to fully end-to-end data-driven pipelines.

## Method

### Overall Architecture

Omni-iEEG is organized across three levels: the data level (multi-center harmonization), the task level (clinically motivated benchmark definitions), and the model level (cross-domain baseline comparisons). Starting from raw iEEG signals, pathological region localization can be achieved either through an event-driven pipeline (HFO detection followed by classification) or through an end-to-end pipeline that directly models one-minute segments.

### Key Design 1: Multi-Center Data Harmonization

Data sources include UCLA (50 patients), Children's Hospital of Michigan (135), University Hospital Zurich (20), University of Pennsylvania Hospital (58), and NIH/JHH/UMF (39), among others, totaling eight centers. Harmonization efforts include:

- **Metadata alignment**: Board-certified clinicians standardized channel naming conventions, SOZ/resection zone annotations, and surgical outcome reporting.
- **Quality control**: Non-standard channels (reference electrodes, EKG, stimulation channels) were filtered out; flat signals and excessively noisy channels were excluded.
- **Preprocessing standardization**: Source-specific recommended preprocessing pipelines were applied (e.g., bipolar montage); all benchmark signals were resampled to 1000 Hz.
- **Data splitting**: A 60%/40% patient-level split was applied, balanced across data sources, surgical outcomes, channel counts, and recording modalities.

### Key Design 2: Pathological Event Annotation Pipeline

For high-frequency oscillations (HFOs, 80–500 Hz), three mainstream detectors (STE, MNI, Hilbert) were used to generate candidate events. Four board-certified epileptologists annotated each candidate into three categories: artifact, pathological HFO co-occurring with a spike (spkHFO), and non-pathological HFO.

- Inter-rater reliability: Fleiss' $\kappa = 0.925$ (three primary annotators)
- Pairwise Cohen's $\kappa$ range: 0.88–0.94
- Total annotations: 36,177 HFO events, including 9,288 artifacts, 7,709 non-spkHFOs, and 19,180 spkHFOs

### Key Design 3: Benchmark Tasks and Evaluation Metrics

**Task 1 — Pathological Event Classification**: Three-class classification (spkHFO / non-spkHFO / artifact), evaluated using macro-averaged Precision, Recall, F1, and AUC.

**Task 2 — Pathological Brain Region Identification**: Channel-level binary classification (pathological vs. normal), with labels defined as SOZ channels versus retained channels from seizure-free patients. Patient-level surgical outcome prediction is additionally assessed via the Resection Ratio (RR):

$$RR = \sum_{c \in \text{resected}} s_c \Big/ \sum_{c \in \text{all}} s_c$$

## Key Experimental Results

### Main Results

**Task 1: Pathological Event Classification**

| Model | Precision | Recall | F1 | AUC |
|-------|-----------|--------|----|-----|
| LSTM+Attention | 0.735 | 0.736 | 0.734 | 0.911 |
| PatchTST Transformer | 0.776 | 0.769 | 0.773 | 0.931 |
| TimesNet | 0.759 | 0.773 | 0.765 | 0.922 |
| **PyHFO-Omni** | **0.803** | **0.811** | **0.806** | **0.939** |

**Task 2: Pathological Brain Region Identification**

| Model | Channel Precision | Channel Recall | Channel F1 | Channel Specificity | Channel AUC | Outcome AUC |
|-------|------------------|----------------|------------|---------------------|-------------|-------------|
| eHFO | 0.605 | 0.647 | 0.620 | 0.410 | 0.661 | 0.452 |
| PyHFO-Omni | 0.580 | 0.699 | 0.564 | 0.695 | 0.735 | **0.744** |
| SEEG-NET | 0.579 | 0.717 | 0.526 | 0.605 | 0.785 | 0.595 |
| CLAP (audio pre-trained) | 0.594 | 0.700 | 0.601 | 0.782 | 0.768 | 0.677 |
| **TimeConv-CNN** | **0.626** | **0.745** | **0.647** | **0.823** | **0.806** | 0.738 |

### Ablation Study

**Cross-Dataset Generalization (Leave-one-out HFO Classification)**

| Held-out Dataset | Precision | Recall | F1 |
|-----------------|-----------|--------|----|
| Open-iEEG | 0.696 | 0.689 | 0.623 |
| Zurich | 0.734 | 0.752 | 0.742 |
| HUP | 0.697 | 0.765 | 0.722 |
| SourceSink | 0.711 | 0.741 | 0.722 |

**Segment Length Ablation (TimeConv-CNN)**

| Segment Length | Precision | Recall | F1 | Specificity | AUC |
|---------------|-----------|--------|----|-------------|-----|
| 30 seconds | 0.577 | 0.707 | 0.544 | 0.659 | 0.773 |
| **1 minute** | **0.608** | **0.761** | **0.610** | **0.748** | **0.823** |
| 2 minutes | 0.592 | 0.747 | 0.564 | 0.668 | 0.805 |

### Key Findings

1. **End-to-end models ≈ traditional biomarkers**: TimeConv-CNN achieves a surgical outcome prediction AUC (0.738) comparable to the HFO-based PyHFO-Omni (0.744), while demonstrating notably superior channel-level AUC (0.806).
2. **Feasibility of cross-domain transfer**: The audio pre-trained model CLAP, after fine-tuning, achieves competitive performance on iEEG classification (channel AUC 0.768), suggesting that iEEG may harbor "audible" biomarker features.
3. **Failure of single-center model generalization**: Publicly available event-level models trained on single-center data exhibit significant performance degradation on the multi-center benchmark.
4. **One-minute segments are optimal**: Compared to 30-second and 2-minute segments, one-minute segments strike the best balance between information content and feature stability.

## Highlights & Insights

- **First comprehensive epilepsy iEEG benchmark**: Unifies formats, metadata, annotations, and evaluation standards, addressing longstanding reproducibility issues in the field.
- **"Audible" biomarkers**: YAMNet labels iEEG signals from SOZ channels as "helicopter" sounds, while retained channels never receive this label—a cross-modal finding with significant implications.
- **TimeConv-CNN architecture**: Applies 1D temporal convolution to compress 60,000 time-point time-frequency representations, followed by CNN to capture joint time-frequency features, enabling efficient processing of kilohertz-rate long-segment iEEG.
- **Clinically driven evaluation philosophy**: Emphasizes that a single AUC metric is insufficient; Recall (to avoid missing pathological tissue), Specificity (to avoid over-resection), and surgical outcome prediction must all be considered jointly.

## Limitations & Future Work

- spkHFO annotation retains a degree of subjectivity, despite high inter-rater agreement ($\kappa > 0.9$).
- Although the dataset spans eight centers, it remains predominantly North American, limiting demographic diversity.
- SOZ channels are substantially outnumbered by non-SOZ channels, and class imbalance remains a challenge.
- Graph-based approaches and unsupervised methods exploiting inter-channel correlations have not been explored.
- There is a risk of over-reliance on model outputs, underscoring the need for clinical expert involvement in the decision-making process.

## Related Work & Insights

- **Public iEEG datasets**: Datasets such as Open iEEG (Zhang et al., 2025) and Zurich HFO (Fedele et al., 2017) adopt heterogeneous formats; the harmonization effort in this paper provides critical infrastructure value for the field.
- **HFO biomarkers**: Gotman (2010) and Frauscher et al. (2018), among others, established the clinical value of HFOs as biomarkers for EZ localization, yet distinguishing pathological from physiological HFOs remains a challenge.
- **Audio–EEG cross-domain transfer**: The successful transfer of CLAP suggests that neural signals and acoustic signals may share underlying representational structures, warranting further exploration.

## Rating

⭐⭐⭐⭐

This paper makes solid contributions in dataset construction, benchmark design, and cross-domain analysis. With 302 multi-center patients, 36K expert annotations, a unified harmonization pipeline, and comprehensive baseline comparisons, it establishes important public infrastructure for the epilepsy iEEG community. The cross-domain transfer of CLAP and the discovery of "audible" biomarkers are particularly novel.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design](afd-instruction_a_comprehensive_antibody_instruction_dataset_with_functional_ann.md)
- [\[NeurIPS 2025\] STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology](../../NeurIPS2025/medical_imaging/starc-9_a_large-scale_dataset_for_multi-class_tissue_classification_for_crc_hist.md)
- [\[CVPR 2026\] XSeg: A Large-scale X-ray Contraband Segmentation Benchmark for Real-World Security Screening](../../CVPR2026/medical_imaging/xseg_a_large-scale_x-ray_contraband_segmentation_benchmark_for_real-world_securi.md)
- [\[CVPR 2026\] Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset](../../CVPR2026/medical_imaging/instruction-guided_lesion_segmentation_for_chest_x-rays_with_automatically_gener.md)
- [\[ICCV 2025\] GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis](../../ICCV2025/medical_imaging/gemex_a_large-scale_groundable_and_explainable_medical_vqa_benchmark_for_chest_x.md)

<!-- RELATED:END -->
