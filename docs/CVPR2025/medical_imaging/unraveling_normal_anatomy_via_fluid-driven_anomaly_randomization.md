---
title: >-
  [Paper Note] Unraveling Normal Anatomy via Fluid-Driven Anomaly Randomization
description: >-
  [CVPR 2025][Medical Imaging][Normal Anatomy Reconstruction] UNA proposes a fluid-driven anomaly randomization method that online-generates infinitely diverse pathology patterns via advection-diffusion PDEs, achieving the first contrast-agnostic brain normal anatomy reconstruction model that can simultaneously process healthy and diseased CT/MRI scans.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Normal Anatomy Reconstruction"
  - "Anomaly Randomization"
  - "Fluid Dynamics"
  - "Contrast-agnostic"
  - "Stroke Detection"
date: 2026-05-08
content_hash: c05d479163f34b0a
---

# Unraveling Normal Anatomy via Fluid-Driven Anomaly Randomization

**Conference**: CVPR 2025  
**arXiv**: [2501.13370](https://arxiv.org/abs/2501.13370)  
**Code**: [GitHub](https://github.com/peirong26/UNA)  
**Area**: Medical Imaging / Brain Analysis  
**Keywords**: Normal Anatomy Reconstruction, Anomaly Randomization, Fluid Dynamics, Contrast-agnostic, Stroke Detection

## TL;DR

UNA proposes a fluid-driven anomaly randomization method that online-generates infinitely diverse pathology patterns via advection-diffusion PDEs, achieving the first contrast-agnostic brain normal anatomy reconstruction model that can simultaneously process healthy and diseased CT/MRI scans.

## Background & Motivation

Medical brain image analysis faces multiple challenges:
- Diverse MRI acquisition protocols (T1w, T2w, FLAIR, etc.) make most methods **contrast-specific**, requiring retraining for new datasets.
- Existing general-purpose models (e.g., SynthSeg, Brain-ID) are mainly designed for **healthy subjects**, showing severe performance degradation in the presence of large pathologies.
- PEPSI, the only contrast-agnostic method capable of handling pathologies, has three major limitations: (1) requires paired pathology segmentation annotations; (2) requires a pre-trained pathology segmentation model; (3) requires extra fine-tuning for anomaly detection.
- **Pathology annotation is extremely expensive**: requiring clinical experts, time-consuming, non-reproducible, and large-scale gold-standard datasets are virtually non-existent.
- Inconsistent pathology segmentation standards across different datasets further restrict data availability.
- A general brain anatomical analysis paradigm is needed that requires no paired pathology annotations, no fine-tuning, and is contrast-agnostic.

## Method

### Overall Architecture

UNA consists of three stages: (1) Fluid-driven anomaly randomization—online generation of infinitely diverse anomaly profiles from limited pathology-related annotations using advection-diffusion PDEs; (2) Appearance encoding—encoding the generated anomaly profiles onto healthy images to simulate diseased images across various modalities; (3) Contrast-agnostic learning—utilizing brain symmetry priors and self-contrastive learning to reconstruct normal anatomy from diseased images.

### Key Design 1: Fluid-Driven Anomaly Profile Randomization

**Function**: Generates an infinite number of diverse and realistic anomaly profiles from limited initial pathology annotations.

**Mechanism**: Models anomaly generation as a forward advection-diffusion PDE process: $\frac{\partial P(\mathbf{x}, t)}{\partial t} = -\nabla \times \mathbf{\Psi}(\mathbf{x}) \cdot \nabla P + \nabla \cdot (\Phi^2(\mathbf{x}) \nabla P)$. The initial condition $P_0$ is obtained from gold-standard segmentations of public stroke datasets (ATLAS, ISLES) or random Perlin noise. The velocity field $\mathbf{V} = \nabla \times \mathbf{\Psi}$ (incompressible flow) and diffusion field $D = \Phi^2$ (non-negative diffusion) are sampled via random Perlin noise. Zero Neumann boundary conditions ensure that anomalies do not exceed brain regions. The PDE is solved using RK45 adaptive time-stepping.

**Design Motivation**: The PDE framework provides continuous and controllable anomaly evolution trajectories, where boundary conditions naturally enforce realistic constraints (e.g., white matter anomalies do not bleed into outer structures), mimicking actual pathological shapes closer than simple random shape generation.

### Key Design 2: Anomaly Appearance Encoding and Random Contrast Generation

**Function**: Simulates the generated anomaly profiles into pathological appearances under different modalities (T1w, T2w, FLAIR, CT).

**Mechanism**: Random contrast and resolution healthy images are synthesized from healthy anatomical label maps via domain randomization. The generated anomaly profile $P$ is then encoded onto the healthy image: pixel values are modified based on the location and intensity of the anomaly region, simulating the appearance of pathology under different modalities (e.g., hyperintensity in T2w, hypointensity in T1w).

**Design Motivation**: Synthetic data mitigates the scarcity of real pathology annotations, while random contrast generation enables the model to handle arbitrary MRI contrasts and CT scans.

### Key Design 3: Brain Symmetry Prior and Self-Contrastive Learning

**Function**: Leverages brain hemispheric symmetry to extract subject-specific anatomical features from contralateral healthy tissues to assist reconstruction.

**Mechanism**: Brain structures exhibit approximate hemispheric symmetry. When a pathology occurs on one hemisphere, the contralateral hemisphere is typically healthy. UNA utilizes contralateral healthy regions as reference, introducing subject-specific anatomical features through self-contrastive learning. This allows the model to learn not only population-level normal anatomy but also retain individual variations.

**Design Motivation**: Population-level reconstruction fails to capture individual anatomical variations, and contralateral symmetry provides a natural self-supervised signal for personalized reconstruction.

### Loss & Training

Reconstruction loss (L1/L2 pixel-level loss) + contrastive learning loss (contralateral tissue vs. ipsilateral pathological region) + auxiliary segmentation loss.

## Key Experimental Results

### Main Results: Normal Anatomy Reconstruction of Stroke Images

| Method | CT SSIM↑ | MRI-T1w SSIM↑ | MRI-FLAIR SSIM↑ | Contrast-Agnostic | No Fine-Tuning |
|------|---------|-------------|----------------|---------|---------|
| **UNA** | **Best** | **Best** | **Best** | ✓ | ✓ |
| PEPSI | Moderate | Moderate | Moderate | ✓ | ✗ |
| Brain-ID | Poor | Poor | Poor | ✓ | ✓ |
| SynthSeg | Poor | Poor | N/A | ✓ | ✓ |

### Ablation Study: Comparison of Anomaly Randomization Methods

| Anomaly Generation Method | Reconstruction Quality | Anomaly Detection Performance |
|------------|---------|------------|
| Fluid-Driven PDE (Ours) | **Best** | **Best** |
| Random Perlin Noise | Poorer | Poorer |
| Training Without Anomalies | Worst | Worst |

### Key Findings

- UNA achieves SOTA performance on CT and multiple MRI contrasts, being the first truly contrast-agnostic and pathology-compatible method.
- Fluid-driven anomaly randomization is significantly more effective than simple random shapes, as PDE constraints generate more realistic anomaly profiles.
- The method can be directly applied to anomaly detection without any fine-tuning, demonstrating zero-shot generalization to unseen pathology types.
- The brain symmetry prior is crucial for personalized reconstruction.

## Highlights & Insights

- **Physics-inspired data augmentation**: Generating training data with fluid-dynamics PDEs is an elegant and principled approach.
- **High practicality**: A single model handles CT and multiple MRI contrasts, healthy and diseased cases, with no fine-tuning required.
- **Anomaly detection as a by-product**: Normal anatomy reconstruction directly drives anomaly detection capabilities—i.e., the discrepancy between normal and actual images reveals anomalies.

## Limitations & Future Work

- Relies on the assumption of brain symmetry, which may limit effectiveness for bilateral pathologies.
- Computational overhead from PDE solving may limit online data augmentation efficiency.
- Only validated on brain imaging; generalization to other organs remains to be explored.
- Future work could integrate large-scale pre-training to further enhance generalization capabilities.

## Related Work & Insights

- The fluid-driven anomaly randomization concept can be extended to other clinical tasks requiring data augmentation for rare cases.
- Complementary to SynthSeg's domain randomization strategy—SynthSeg randomizes appearances, while UNA randomizes pathologies.
- The paradigm of normal anatomy reconstruction $\rightarrow$ anomaly detection provides a general framework for unsupervised pathology detection.

## Rating

⭐⭐⭐⭐ — Elegantly addresses key pain points in medical image analysis: pathology annotation scarcity, modality diversity, and healthy-pathological compatibility. Fluid-driven PDE anomaly randomization is a methodological highlight. Comprehensive evaluations on CT and multiple MRI contrasts are compelling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PDD: Manifold-Prior Diverse Distillation for Medical Anomaly Detection](../../CVPR2026/medical_imaging/pdd_manifold-prior_diverse_distillation_for_medical_anomaly_detection.md)
- [\[ICML 2025\] Enhancing Statistical Validity and Power in Hybrid Controlled Trials: A Randomization Inference Approach with Conformal Selective Borrowing](../../ICML2025/medical_imaging/enhancing_statistical_validity_and_power_in_hybrid_controlled_trials_a_randomiza.md)
- [\[CVPR 2025\] Evidential learning driven Breast Tumor Segmentation with Stage-divided Vision-Language Interaction](evidential_learning_driven_breast_tumor_segmentation_with_stage-divided_vision-l.md)
- [\[ICML 2026\] Which Anatomy Matters Under Limited Labels? A Data-Efficient Anatomy-Aware Benchmark for Cardiac Pathology Prediction](../../ICML2026/medical_imaging/which_anatomy_matters_under_limited_labels_a_data-efficient_anatomy-aware_benchm.md)
- [\[NeurIPS 2025\] SMMILE: An Expert-Driven Benchmark for Multimodal Medical In-Context Learning](../../NeurIPS2025/medical_imaging/smmile_an_expert-driven_benchmark_for_multimodal_medical_in-context_learning.md)

</div>

<!-- RELATED:END -->
