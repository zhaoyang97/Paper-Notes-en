---
title: >-
  [Paper Note] EgoEMS: A High-Fidelity Multimodal Egocentric Dataset for Cognitive Assistance in Emergency Medical Services
description: >-
  [AAAI 2026][Medical Imaging][Egocentric perspective] This paper presents the first high-fidelity multi-person multimodal egocentric EMS dataset, comprising 233 trials with 20 hours of video, annotations covering 9 interventions and 67 critical steps, and three benchmark tasks (step classification / online segmentation / CPR quality estimation) to advance the development of cognitive assistance systems for EMS.
tags:
  - AAAI 2026
  - Medical Imaging
  - Egocentric perspective
  - emergency medical services
  - multimodal fusion
  - cognitive assistance
  - activity recognition
date: 2026-05-08
content_hash: 1db14d7797ec10b3
---

# EgoEMS: A High-Fidelity Multimodal Egocentric Dataset for Cognitive Assistance in Emergency Medical Services

**Conference**: AAAI 2026
**arXiv**: [2511.09894](https://arxiv.org/abs/2511.09894)
**Code**: [Project Page](https://uva-dsa.github.io/EgoEMS)
**Area**: Medical Imaging / Dataset / Multimodal Learning
**Keywords**: Egocentric perspective, emergency medical services, multimodal fusion, cognitive assistance, activity recognition

## TL;DR
This paper presents the first high-fidelity multi-person multimodal egocentric EMS dataset, comprising 233 trials with 20 hours of video, annotations covering 9 interventions and 67 critical steps, and three benchmark tasks (step classification / online segmentation / CPR quality estimation) to advance the development of cognitive assistance systems for EMS.

## Background & Motivation
**Background**: First responders in emergency medical services (EMS) face substantial cognitive load. Advances in AI and LLMs have created opportunities for virtual cognitive assistance systems. Most existing egocentric datasets focus on everyday activities, lacking coverage of high-stakes medical domains.

**Limitations of Prior Work**: (a) Large-scale, high-fidelity annotated datasets are absent in the medical domain; (b) existing EMS data are predominantly unimodal (audio only); (c) emergency scenarios involve team collaboration, yet existing datasets are mostly single-viewpoint; (d) medical annotation is prohibitively costly.

**Key Challenge**: Authentic data are needed to train effective systems, yet real emergency care raises ethical barriers; multimodal data further multiply annotation costs.

**Goal**: Construct the first multimodal multi-person egocentric EMS dataset, establish a taxonomy aligned with national standards, and provide benchmark tasks with baselines.

**Key Insight**: Simulated scenarios are used to satisfy ethical requirements; real EMS professionals are recruited to ensure ecological validity; a hybrid manual and semi-automatic annotation pipeline is adopted to reduce cost.

**Core Idea**: Through collaboration with EMS experts, a standardized taxonomy is established, and multimodal egocentric data from 62 participants in simulated scenarios are collected to provide a solid foundation for AI-based cognitive assistance.

## Method

### Overall Architecture
Dataset composition: 233 trials × 20 hours of video × 4 modalities (video + audio + IMU + CPR ground truth) × 2,694 critical step annotations. Three scenario types are covered: cardiac arrest (76 trials), suspected cardiac event (23 trials), and stroke (41 trials).

### Key Designs

1. **EMS Taxonomy Development**:

    - Function: Establish a hierarchical taxonomy based on NREMT/NEMSIS national standards.
    - Mechanism: Analysis of 15M+ EMS records combined with expert consultation yields a three-level hierarchy of 3 protocols → 9 interventions → 67 critical steps.
    - Design Motivation: Ensures alignment between data and clinical practice, facilitating system deployability.

2. **Semi-Automatic Audio Annotation Pipeline**:

    - Function: Generate timestamped transcriptions with speaker diarization.
    - Mechanism: Zero-shot speech recognition via Gemini-2.5 achieves WER of 0.31 (vs. Whisper: 0.62–0.68) with timestamp MAE of 0.18 s.
    - Design Motivation: Reduces annotation time by approximately 90% compared to fully manual labeling.

3. **Semi-Supervised Object Annotation Pipeline**:

    - Function: Annotate bounding boxes and segmentation masks for medical instruments.
    - Mechanism: Seed images are crawled → filtered by Gemini → fine-tuned DETR → segmented by SAM2. Mean IoU = 0.76.
    - Design Motivation: Reduces annotation time from 66 hours to 1 hour (98.5% reduction).

### Loss & Training
Benchmark tasks employ multiple approaches including supervised Transformers, multimodal fusion, and zero-shot LLMs.

## Key Experimental Results

### Benchmark Task 1: Critical Step Classification

| Method | Top-1 Accuracy | Notes |
|--------|---------------|-------|
| Supervised Transformer (video) | **62.3%** | Best |
| Video + IMU fusion | 62.2% | Fusion yields no gain |
| Zero-shot Qwen-2.5 | 38.3% | LLMs show potential but insufficient |

### Benchmark Task 2: Online Critical Step Segmentation

| Method | Accuracy | Notes |
|--------|----------|-------|
| Transformer (video + IMU) | **61%** | +6% over video-only |
| Zero-shot Qwen-2.5 | 55.5% | Competitive LLM performance |
| Audio (Whisper + GPT-4o) | 38% | Responders do not always verbalize actions |

### Key Findings
- Early fusion strategies are insufficient; more sophisticated fusion methods are needed (video + IMU fusion yields negligible gain).
- Zero-shot LLMs show promise on certain tasks but fall short of supervised approaches.
- In CPR quality estimation, IMU offers a unique advantage for rate prediction (lowest RMSE), while video + IMU fusion is optimal for depth estimation (F1 = 0.83).
- EMS professionals exhibit stable CPR performance, whereas lay participants show high variability.

## Highlights & Insights
- **Pioneering Contribution**: The first multi-person multimodal egocentric EMS dataset, including CPR quality ground truth. The data collection system (GoPro + Galaxy Watch) is low-cost and reproducible.
- **Semi-Automatic Annotation Innovation**: Gemini-2.5 surpasses prior SOTA in speech recognition (WER 0.31); the DETR + SAM2 pipeline reduces annotation time by 98.5%.
- **Comprehensive Taxonomy**: Alignment with national standards (NREMT/NEMSIS) ensures practical deployability.

## Limitations & Future Work
- **Scale Constraints**: 62 participants is relatively small, and data are collected at a single geographic location (Virginia).
- **Simulation-to-Reality Gap**: Simulated scenarios cannot fully replicate the chaos and stress of real emergency care.
- **Insufficient Multi-Person Interaction Modeling**: Benchmarks primarily focus on the primary responder; team coordination modeling warrants further investigation.
- **Limited Fusion Effectiveness**: Video + IMU fusion yields negligible gain, necessitating more sophisticated strategies.

## Related Work & Insights
- **vs. Ego-Exo4D**: EgoEMS provides deeper modeling within a single high-stakes domain, whereas Ego-Exo4D covers multiple domains with less depth.
- **vs. EgoSurgery**: EgoEMS is multi-person and multimodal, and includes quality metrics, whereas EgoSurgery covers surgical video only.
- Medical AI must narrow the real-to-simulation gap and enforce rigorous privacy protection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First multimodal multi-person egocentric EMS dataset
- Experimental Thoroughness: ⭐⭐⭐⭐ Three complete benchmarks, though ablation studies are limited
- Writing Quality: ⭐⭐⭐⭐⭐ Clear ethical considerations and detailed annotation procedures
- Value: ⭐⭐⭐⭐⭐ High practical applicability with open resources

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] FIA-Edit: Frequency-Interactive Attention for Efficient and High-Fidelity Inversion-Free Text-Guided Image Editing](fia-edit_frequency-interactive_attention_for_efficient_and_high-fidelity_inversi.md)
- [\[AAAI 2026\] A Principle-Driven Adaptive Policy for Group Cognitive Stimulation Dialogue for Elderly with Cognitive Impairment](a_principle-driven_adaptive_policy_for_group_cognitive_stimu.md)
- [\[AAAI 2026\] Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering](expert-guided_prompting_and_retrieval-augmented_generation_for_emergency_medical.md)
- [\[AAAI 2026\] MAISI-v2: Accelerated 3D High-Resolution Medical Image Synthesis with Rectified Flow and Region-specific Contrastive Loss](maisi-v2_accelerated_3d_high-resolution_medical_image_synthesis_with_rectified_f.md)
- [\[CVPR 2026\] GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification](../../CVPR2026/medical_imaging/gleam_a_multimodal_imaging_dataset_and_hamm_for_gl.md)

<!-- RELATED:END -->
