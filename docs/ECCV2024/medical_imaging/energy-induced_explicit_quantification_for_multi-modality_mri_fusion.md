---
title: >-
  [Paper Note] Energy-induced Explicit Quantification for Multi-modality MRI Fusion
description: >-
  [ECCV 2024][Medical Imaging][Multi-modality MRI fusion] This paper proposes E²PA, an energy-induced explicit propagation and alignment framework. Through two modules—Energy-guided Hierarchical Fusion (EHF) and Energy-regularized Space Alignment (ESA)—it explicitly quantifies and optimizes inter-modality dependency propagation and information flow consistency in multi-modality MRI fusion, outperforming State-Of-The-Art (SOTA) methods across three public datasets.
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "Multi-modality MRI fusion"
  - "energy guidance"
  - "explicit quantification"
  - "hierarchical fusion"
  - "spatial alignment"
date: 2026-05-08
content_hash: 03beb77ee02220ac
---

# Energy-induced Explicit Quantification for Multi-modality MRI Fusion

**Conference**: ECCV 2024  
**Code**: [https://github.com/JerryQseu/EEPA](https://github.com/JerryQseu/EEPA)  
**Area**: Medical Image  
**Keywords**: Multi-modality MRI fusion, energy guidance, explicit quantification, hierarchical fusion, spatial alignment

## TL;DR

This paper proposes E²PA, an energy-induced explicit propagation and alignment framework. Through two modules—Energy-guided Hierarchical Fusion (EHF) and Energy-regularized Space Alignment (ESA)—it explicitly quantifies and optimizes inter-modality dependency propagation and information flow consistency in multi-modality MRI fusion, outperforming State-Of-The-Art (SOTA) methods across three public datasets.

## Background & Motivation

**Background**: Multi-modality magnetic resonance imaging (MRI) is essential for accurate disease diagnosis and surgical planning. Different MRI modalities (such as T1, T2, FLAIR, and T1ce) provide complementary tissue contrast information. Fusing multi-modality information delivers a more comprehensive diagnostic basis than relying on a single modality. Currently, multi-modality MRI fusion is a core task in medical image analysis.

**Limitations of Prior Work**: A key challenge in multi-modality MRI fusion is that different diseases (e.g., brain tumor segmentation, cardiac segmentation) exhibit distinct information aggregation patterns. For example, brain tumor segmentation may rely more heavily on contrast-enhanced regions in the T1ce modality, while the FLAIR modality provides edema boundary information. The fusion weights and interaction mechanisms need to adjust dynamically based on the specific task. Most existing methods rely on implicit learning (such as simple concatenation or attention mechanisms) to discover these aggregation patterns. They lack explicit modeling and quantification of the fusion process, leading to unstable fusion quality and poor generalization to unseen disease scenarios.

**Key Challenge**: The core of multi-modality fusion lies in two aspects: inter-modality dependency propagation and information flow alignment. Existing methods either focus on only one aspect or address both implicitly, leaving them unable to achieve explicit optimization. A unified framework is required to explicitly quantify and optimize both key attributes.

**Goal**: (1) How to explicitly quantify the propagation of inter-modality dependencies in multi-modality MRI fusion? (2) How to explicitly measure and optimize the consistency of information flow during the fusion process? (3) How to construct a unified fusion framework that can adapt to different diseases and modality combinations?

**Key Insight**: Taking inspiration from the concept of energy in statistical physics, the authors formulate the multi-modality fusion problem as an energy minimization process. From this perspective, the hierarchical information structure within a patient cohort can be characterized by identical energy levels (where multi-modality features of patients in the same cohort should converge to similar energy states), while the consistency of the multi-modality space can be constrained through energy minimization.

**Core Idea**: Use energy functions to explicitly quantify dependency propagation and spatial alignment in multi-modality MRI fusion, constructing a unified E²PA framework.

## Method

### Overall Architecture

The E²PA (Energy-induced Explicit Propagation and Alignment) framework consists of two core modules: the Energy-guided Hierarchical Fusion (EHF) module and the Energy-regularized Space Alignment (ESA) module. The input consists of multi-modality MRI scans (such as T1, T2, FLAIR, etc.). After passing through modality-specific feature extractors, the EHF module discovers and propagates inter-modality dependencies across hierarchical structures, while the ESA module aligns the information flow directions of different modalities in the fusion space. The system ultimately outputs the fused segmentation prediction (or other task-specific predictions).

### Key Designs

1. **Energy-guided Hierarchical Fusion (EHF)**:
    - **Function**: Reveal and optimize the propagation process of inter-modality dependencies.
    - **Mechanism**: EHF models multi-modality fusion as a hierarchical energy propagation process. The core hypothesis is that patients belonging to the same category (e.g., the same tumor type) should share the same energy levels regarding their multi-modality features within an ideal fusion space. Specifically, EHF first maps the features of each modality to a shared energy space, then progressively propagates inter-modality dependencies through a hierarchical structure (from low-level to high-level). At each level, an energy function is employed to measure the deviation between the current fusion state and the ideal state, guiding the flow of inter-modality information along the direction of gradient descent. In this way, the quantification and optimization of dependencies are made explicit—with energy values serving as metrics at every step, instead of being implicitly discovered through end-to-end training.
    - **Design Motivation**: Implicit fusion methods (such as simple feature concatenation or attention-based weighting) cannot guarantee consistent fusion performance across different patient cohorts. Introducing energy constraints helps maintain stable fusion performance over the entire patient population.

2. **Energy-regularized Space Alignment (ESA)**:
    - **Function**: Measure and optimize the consistency of information flow in multi-modality aggregation.
    - **Mechanism**: The ESA module focuses on whether the geometric structures of multi-modality features are aligned within the fusion space. Specifically, it performs space factorization on the fusion space, decomposing the fused representation into several independent subspace factors, and then applies energy minimization to constrain the degree of alignment among these factors. The goal of energy minimization is to enforce consistent directions and distributions of different modalities within the factorized subspaces, thereby ensuring consistent information flow during aggregation. This process can be understood as follows: if multi-modality fusion is viewed as an information aggregation network, ESA ensures that the direction of information flow across all network paths remains consistent, preventing conflicts or cancellations between information features.
    - **Design Motivation**: A common issue in multi-modality fusion is "modality conflict"—where information provided by different modalities may be contradictory in certain regions. ESA addresses this problem through spatial alignment, yielding more stable and reliable fusion results.

3. **Unified Aggregation Pattern**:
    - **Function**: Provide a unified fusion framework for diverse diseases and tasks.
    - **Mechanism**: Through the collaboration of EHF and ESA, E²PA learns a unified and explicit aggregation pattern. Instead of being hard-coded for a specific task, this pattern adaptively discovers the optimal mode of inter-modality interaction through energy guidance. For novel disease scenarios or modality combinations, E²PA can adapt by adjusting the parameters of the energy functions without requiring a redesign of the fusion architecture.
    - **Design Motivation**: The aggregation patterns of existing methods are typically discovered implicitly (via end-to-end training), which limits their transferability across different scenarios. The explicit energy-guided strategy provides superior interpretability and generalization capabilities.

### Loss & Training

The total loss function of E²PA consists of three components: task loss (e.g., cross-entropy loss and Dice loss for segmentation tasks), energy propagation loss from the EHF module (constraining the energy consistency of patients under the same category), and energy regularization loss from the ESA module (constraining the consistency of spatial alignment). The three components are optimized jointly via a weighted sum. Training is performed in an end-to-end manner on standard multi-modality MRI datasets.

## Key Experimental Results

### Main Results

| Dataset | Task | Modalities | Ours vs Prev. SOTA |
|:---:|:---:|:---:|:---:|
| BraTS (Brain Tumor) | Multi-modality tumor segmentation | T1/T2/FLAIR/T1ce | Outperforms SOTA |
| Cardiac MRI Dataset | Cardiac structure segmentation | Multi-modality combination | Outperforms SOTA |
| Third Public Dataset | Multi-modality segmentation | Various modality combinations | Outperforms SOTA |

### Ablation Study

| Configuration | Key Metrics | Note |
|:---:|:---:|:---:|
| EHF Only | Improved | Confirms the effectiveness of energy guidance in hierarchical fusion |
| ESA Only | Improved | Confirms the effectiveness of energy regularization in space alignment |
| EHF + ESA (Full E²PA) | Optimal | Both modules are complementary; using them together yields the best results |
| Implicit Fusion (Baseline) | Lower | Validates the necessity of explicit quantification |

### Key Findings

- The explicit energy-guided fusion strategy outperforms implicit fusion methods across all three datasets, demonstrating the value of explicit quantification.
- EHF and ESA address fusion challenges from different dimensions, making their contributions highly complementary.
- E²PA maintains its advantages across different modality combinations and tasks, demonstrating the generalization capability of the unified aggregation pattern.
- Energy values can be used to visualize the fusion process, providing a degree of interpretability.

## Highlights & Insights

- Formulating multi-modality fusion from the perspective of energy minimization is a novel and interesting viewpoint, establishing a physically interpretable theoretical foundation for MRI fusion.
- Explicitly quantifying fusion attributes (dependency propagation + space alignment) makes the fusion process analyzable and controllable, rather than a mere black box.
- The capability of the unified framework to adapt to different diseases and modality combinations reduces the engineering overhead of designing task-specific fusion strategies for clinical deployment.
- Open-sourcing the code facilitates verification and expansion by the research community.

## Limitations & Future Work

- The choice of the specific form of the energy function (how to define "identical energy") could impact performance across different tasks, necessitating a more systematic sensitivity analysis.
- Validation is currently limited to MRI data. Exploring its applicability to cross-modality fusion (such as CT-MRI) or other medical imaging modalities (e.g., ultrasound + MRI) is highly warranted.
- Robustness under missing modality scenarios (e.g., when a certain modality is unavailable)—which is highly common in clinical practice—has not been discussed.
- Determining how to choose the hierarchical fusion depth and granularity, and whether adaptive strategies can be employed.
- Whether the energy-guided approach can be generalized to non-medical multi-modality fusion scenarios (such as RGB-depth fusion).

## Related Work & Insights

- Multi-modality MRI fusion methods: from early-stage feature concatenation and mid-stage attention weighting to recent transformer-based fusion.
- Application of Energy-Based Models (EBMs) in both generative and discriminative tasks.
- Utilization of spatial alignment methods in registration tasks; this paper successfully introduces alignment into multi-modality fusion.
- Insights: The concept of explicitly quantifying fusion attributes can be transferred to other fields, such as multi-modality VLMs and multi-sensor fusion.

## Rating

- Novelty: ⭐⭐⭐⭐ Energy-guided explicit fusion offers a novel perspective with solid theoretical grounding.
- Experimental Thoroughness: ⭐⭐⭐ Validated across three datasets with ablation studies, though exact numbers are limited on the ECVA page.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and systematic methodology description.
- Value: ⭐⭐⭐⭐ Holds practical value for multi-modality medical image fusion and introduces a new theoretical framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LUMINA: A Multi-Vendor Mammography Benchmark with Energy Harmonization Protocol](../../CVPR2026/medical_imaging/lumina_a_multi-vendor_mammography_benchmark_with_energy_harmonization_protocol.md)
- [\[ECCV 2024\] GTP-4o: Modality-Prompted Heterogeneous Graph Learning for Omni-Modal Biomedical Representation](gtp-4o_modality-prompted_heterogeneous_graph_learning_for_omni-modal_biomedical_.md)
- [\[ECCV 2024\] Domesticating SAM for Breast Ultrasound Image Segmentation via Spatial-Frequency Fusion and Uncertainty Correction](domesticating_sam_for_breast_ultrasound_image_segmentation_via_spatial-frequency.md)
- [\[CVPR 2025\] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](../../CVPR2025/medical_imaging/federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)
- [\[ICCV 2025\] SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality](../../ICCV2025/medical_imaging/simmlm_a_simple_framework_for_multi-modal_learning_with_missing_modality.md)

</div>

<!-- RELATED:END -->
