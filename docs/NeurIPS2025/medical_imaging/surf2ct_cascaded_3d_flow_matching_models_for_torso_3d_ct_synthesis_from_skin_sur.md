---
title: >-
  [Paper Note] Surf2CT: Cascaded 3D Flow Matching Models for Torso 3D CT Synthesis from Skin Surface
description: >-
  [NeurIPS 2025][Medical Imaging][Flow Matching] This paper proposes Surf2CT, a cascaded 3D Flow Matching framework that, for the first time, synthesizes complete high-resolution 3D CT volumes solely from external body surface scans and demographic data (age, sex, height, weight), without requiring any internal imaging input.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - Flow Matching
  - CT Synthesis
  - Body Surface Scan
  - 3D Generation
  - Non-Invasive Imaging
date: 2026-05-08
content_hash: cc5fd0e03edd15ad
---

# Surf2CT: Cascaded 3D Flow Matching Models for Torso 3D CT Synthesis from Skin Surface

**Conference**: NeurIPS 2025
**arXiv**: [2505.22511](https://arxiv.org/abs/2505.22511)
**Code**: Not available
**Area**: Medical Imaging / CT Synthesis
**Keywords**: Flow Matching, CT Synthesis, Body Surface Scan, 3D Generation, Non-Invasive Imaging

## TL;DR

This paper proposes Surf2CT, a cascaded 3D Flow Matching framework that, for the first time, synthesizes complete high-resolution 3D CT volumes solely from external body surface scans and demographic data (age, sex, height, weight), without requiring any internal imaging input.

## Background & Motivation

CT scans provide detailed internal anatomical information but are costly and involve radiation exposure, making them unsuitable for routine screening. An ideal alternative would be to infer internal anatomy from non-invasive external body shape data—enabling home health monitoring, preventive medicine, and personalized clinical assessment. However, external body shape does not uniquely determine internal anatomy: similar surface measurements can correspond to vastly different organ sizes and compositions.

Limitations of existing methods:
- **Digital human body models** (e.g., XCAT): rely on limited template libraries, support only coarse demographic scaling, and cannot generate voxel-level CT
- **Sparse X-ray reconstruction** (e.g., X2CT-GAN, DIFR3CT): still require at least partial internal imaging input
- **Segmentation-guided CT synthesis** (e.g., Seg2Med, MAISI): require pre-existing internal segmentation maps, typically derived from existing CT/MRI

The core innovation of Surf2CT lies in completely eliminating dependence on internal imaging, generating anatomically plausible CT volumes from only 3D surface scans (even those captured with consumer-grade depth cameras) and basic demographic information.

## Method

### Overall Architecture

Surf2CT employs a three-stage cascaded generation pipeline, with each stage using an independent conditional 3D Flow Matching model:

1. **Stage 1: Surface Completion** — recovers a complete SDF (Signed Distance Function) from an incomplete surface scan
2. **Stage 2: Coarse-Resolution CT Synthesis** — generates 8 mm resolution CT from the complete SDF and demographic data
3. **Stage 3: CT Super-Resolution** — refines the output patch-by-patch to 2 mm high-resolution CT

All stages use a 3D-adapted EDM2 U-Net architecture as the backbone.

### Key Designs

1. **SDF Surface Completion (Stage 1)**: Body shape is represented as a Signed Distance Function (SDF), defined as $f(\mathbf{x}) = \pm\min_{\mathbf{y} \in \partial\Omega} \|\mathbf{x} - \mathbf{y}\|_2$, discretized on a $56 \times 56 \times 88$ grid (8 mm isotropic). A conditional Flow Matching model $G_1$ is trained to learn the velocity field that recovers the complete SDF from a partial SDF (e.g., retaining only the frontal view):

$$\frac{d\mathbf{x}(t)}{dt} = v_\theta^{(1)}(t, \mathbf{x}(t), f_{\text{partial}}, \mathbf{z}_{\text{demo}})$$

Demographic attributes (age, sex, height, weight) are concatenated with the partial SDF as constant channels along the spatial dimensions. The training objective is the standard Flow Matching loss $\mathcal{L}_1 = \mathbb{E}[\|v_\theta^{(1)}(t, \mathbf{x}_t) - (f_{\text{full}}^{\text{gt}} - \boldsymbol{\eta})\|^2]$.

2. **Coarse-Resolution CT Synthesis (Stage 2)**: On the same $56 \times 56 \times 88$ grid (corresponding to a $448 \times 448 \times 704$ mm field of view), the model learns the mapping from the complete SDF to low-resolution CT. Conditioning inputs include the complete SDF and demographic data. The model implicitly learns the association between external morphological features and internal organ location, size, and tissue density—for example, individuals with higher BMI typically exhibit greater adipose tissue volume.

3. **Patch-Based Super-Resolution (Stage 3)**: The target resolution is $224 \times 224 \times 352$ (2 mm isotropic). Direct full-volume generation at this resolution is computationally prohibitive. A patch-based strategy is adopted: $56 \times 56 \times 88$ high-resolution patches are randomly sampled, conditioned on the corresponding region of the upsampled coarse CT, sinusoidal positional encodings, and demographic information. During inference, patches are generated and fused. The Stage 3 model has only 1.89M parameters (vs. 80.68M for Stages 1/2), with reduced computation achieved by decreasing channel count and network depth.

### Loss & Training

- All three stages use the standard Flow Matching $L_2$ regression loss with conditional optimal transport probability paths
- Training data: 2,633 cases from MGH + 565 cases from AutoPET = 3,198 torso CT scans (approximately 1.13 million axial slices)
- Preprocessing: TotalSegmentator used to extract torso regions; resampled to 2 mm isotropic; HU values clipped to $[-500, 500]$
- Stages 1 and 2 each trained for approximately 35 million steps (4×A100, 72h per stage); Stage 3 trained for 15 million steps
- AdamW optimizer, initial lr $= 10^{-4}$, linear decay; gradient clipping + EMA
- Sampling: 200-step integration with the Dormand–Prince solver

## Key Experimental Results

### Main Results: Body Composition Assessment

| Metric | Original CT | Surf2CT | Difference (%) | R² |
|--------|-------------|---------|----------------|-----|
| Male Muscle Volume (mL) | 9388±1860 | 8483±1424 | -9.6% | 0.81 |
| Male Subcutaneous Fat | 8288±4189 | 9357±3992 | +12.9% | 0.86 |
| Male Visceral Fat | 5030±2472 | 4901±2406 | -2.5% | 0.74 |
| Female Muscle Volume | 5304±1152 | 4817±993 | -9.2% | 0.87 |
| Female Subcutaneous Fat | 9210±5368 | 9866±4999 | +7.1% | **0.96** |
| Female Visceral Fat | 2324±1514 | 2213±1476 | -4.8% | 0.79 |

### Organ Volume Assessment

| Organ | Original CT (mL) | Surf2CT (mL) | Difference (%) | R² |
|-------|-----------------|--------------|----------------|-----|
| Male Heart | 689.7±133.2 | 720.0±84.6 | +4.4% | 0.12 |
| Male Liver | 1695.4±380.8 | 1761.1±268.7 | +3.9% | 0.25 |
| Male Kidney | 337.4±70.1 | 329.1±56.9 | -2.5% | 0.16 |
| Female Heart | 543.9±82.9 | 557.0±92.0 | +2.4% | 0.11 |
| Female Liver | 1524.9±346.7 | 1495.7±239.0 | -1.9% | 0.33 |
| Female Kidney | 293.1±55.3 | 260.5±58.9 | **-11.1%** | 0.04 |

### Surface Completion Assessment

| Metric | Partial SDF | After Completion |
|--------|-------------|-----------------|
| Chamfer Distance (mm) | 521.78±228.09 | **2.71±1.80** |
| IoU | 0.87±0.09 | **0.98±0.02** |
| NMAE | 0.14±0.07 | **0.02±0.01** |

### Ablation Study / Additional Evaluation

| Evaluation Item | Result | Notes |
|----------------|--------|-------|
| Lung Localization Bias | -2.5 mm | Bland–Altman analysis; limits of agreement [-62.6, +57.5] mm |
| Lung Localization R² | 0.36 | Moderate correlation |
| Sex Difference Modeling | Correctly Captured | Sex-specific organ size and muscle/fat distribution differences consistent with real CT |

### Key Findings

- Surface completion is highly effective: Chamfer Distance reduced from 521.8 mm to 2.7 mm
- Body composition metrics show strong correlation (R² up to 0.67–0.96), suggesting clinical utility
- Mean organ volume errors are within ±5% for most organs, but individual-level prediction accuracy is low (R² generally < 0.35)
- Results for female subjects are overall superior to those for male subjects (possibly due to greater uncertainty from lower female representation in training data)
- The model correctly learns associations between demographics and anatomy (e.g., BMI and fat distribution)

## Highlights & Insights

- **Paradigm Innovation**: This work represents the first demonstration of synthesizing internal 3D CT from purely external data (body surface + demographics), opening a new direction for non-invasive anatomical imaging
- **Practical Value of Cascaded Design**: The three-stage coarse-to-fine cascaded strategy elegantly resolves the trade-off between resolution and computational cost, with each stage carrying a clear physical interpretation
- **Broad Clinical Imagination**: Home 3D scanning → virtual CT → health monitoring, as well as radiation-free digital twins for surgical planning
- **Potential for Anomaly Detection**: Discrepancies between synthesized and real CT may reflect hidden pathological abnormalities

## Limitations & Future Work

- Training data is predominantly composed of male oncology patients, introducing significant sex and disease-type bias
- Individual organ volume prediction R² values are low (particularly for lungs and kidneys), indicating limited constraint that external body shape imposes on internal organs
- Noise and artifacts from consumer-grade scanners are not addressed
- The surface-to-CT mapping is inherently ill-posed (one-to-many); synthesized CT volumes may not reflect the pathological state of a specific individual
- Reliability and fairness in real-world clinical decision-making have not been evaluated

## Related Work & Insights

- **XCAT**: Conventional digital human body model based on template-driven approach
- **EDM2**: Diffusion model architecture adapted in this work for 3D Flow Matching
- **TotalSegmentator**: Automatic organ segmentation tool used for evaluation and preprocessing
- **BOSS**: Statistical body model that jointly learns skin/skeleton/organ geometry
- Insight: Flow Matching holds great promise for 3D medical image generation; the cascaded strategy is generalizable to other large-scale 3D generation tasks

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Pioneering task formulation: synthesizing internal CT from purely external data
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive multi-dimensional evaluation, though training data bias is notable
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with well-organized description of the three-stage pipeline
- **Value**: ⭐⭐⭐⭐ — Opens a new paradigm, though clinical deployment remains distant

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [\[NeurIPS 2025\] Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number](prior-guided_flow_matching_for_target-aware_molecule_design_with_learnable_atom_.md)
- [\[NeurIPS 2025\] Multimodal 3D Genome Pre-training](multimodal_3d_genome_pre-training.md)
- [\[NeurIPS 2025\] 3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks](3drad_a_comprehensive_3d_radiology_medvqa_dataset_with_multi.md)
- [\[NeurIPS 2025\] PolyPose: Deformable 2D/3D Registration via Polyrigid Transformations](polypose_deformable_2d3d_registration_via_polyrigid_transformations.md)

</div>

<!-- RELATED:END -->
