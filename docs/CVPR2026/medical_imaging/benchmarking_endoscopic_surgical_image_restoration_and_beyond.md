---
title: >-
  [Paper Note] Benchmarking Endoscopic Surgical Image Restoration and Beyond
description: >-
  [CVPR 2026][Medical Imaging][Endoscopic image restoration] This work constructs SurgClean, the first multi-source real-world endoscopic surgical image restoration dataset (3,113 images covering three degradation types: smoke, fog, and liquid splash), and systematically benchmarks 22 representative image restoration methods (12 general-purpose + 10 task-specific) on it. The results reveal a significant gap between existing methods and clinical requirements, and further analyze the fundamental differences between surgical-scene and natural-scene degradations.
tags:
  - CVPR 2026
  - Medical Imaging
  - Endoscopic image restoration
  - surgical scene desmoke/defog/desplash
  - benchmark dataset
  - image quality assessment
  - clinical applications
date: 2026-05-08
content_hash: 5408b42c25d93d5a
---

# Benchmarking Endoscopic Surgical Image Restoration and Beyond

**Conference**: CVPR 2026
**arXiv**: [2505.19161](https://arxiv.org/abs/2505.19161)
**Code**: [https://github.com/PJLallen/Surgical-Image-Restoration](https://github.com/PJLallen/Surgical-Image-Restoration)
**Area**: Medical Imaging
**Keywords**: Endoscopic image restoration, surgical scene desmoke/defog/desplash, benchmark dataset, image quality assessment, clinical applications

## TL;DR

This work constructs SurgClean, the first multi-source real-world endoscopic surgical image restoration dataset (3,113 images covering three degradation types: smoke, fog, and liquid splash), and systematically benchmarks 22 representative image restoration methods (12 general-purpose + 10 task-specific) on it. The results reveal a significant gap between existing methods and clinical requirements, and further analyze the fundamental differences between surgical-scene and natural-scene degradations.

## Background & Motivation

In minimally invasive surgery, a clear operative field is critical for surgeons to accurately identify anatomical structures and avoid procedural errors. Three common visual degradation types occur during endoscopic surgery:

**Surgical Smoke**: Energy-based instruments such as electrocautery and ultrasonic scalpels generate substantial smoke during cutting and hemostasis, obscuring the operative field.

**Lens Fogging**: Temperature differentials between the body interior and exterior cause moisture condensation on the endoscope lens surface, producing uniform fogging.

**Liquid Splash**: Blood, tissue fluid, bile, and other fluids splatter onto the lens during surgery, causing localized occlusion.

These degradations seriously compromise surgical safety and efficiency, frequently forcing surgeons to pause and clean the lens.

**Limitations of Existing Datasets**:
- Most datasets rely on **synthetic data** (e.g., superimposing Gaussian smoke on clean images), which poorly approximates real degradations.
- Existing real-data sets largely **cover only a single degradation type** (primarily desmoke), lacking defog and desplash subsets.
- There is a lack of **multi-source, multi-procedure** real paired data.

**Key Challenge**: Existing image restoration algorithms perform well on natural scenes but suffer drastic performance drops when directly transferred to surgical scenes—implying fundamental differences between surgical and natural degradations and motivating the need for dedicated datasets and tailored algorithms.

## Method

### Overall Architecture

The core contribution of this paper is a dataset and benchmark rather than a novel algorithm. The work is organized into three parts:
1. **Dataset Construction**: Degraded frames are selected and annotated from surgical videos of 414 patients across two medical institutions.
2. **Benchmark Evaluation**: 22 image restoration methods are evaluated on SurgClean.
3. **Analysis Beyond Pixel-Level Restoration**: The impact of restored images on downstream tasks (depth estimation, semantic segmentation) is examined, along with the differences between surgical and natural scene degradations.

### Key Designs

1. **SurgClean Dataset Construction**:

    - Function: Degraded frames are selected from approximately 43,640 minutes of laparoscopic and thoracoscopic surgical videos from 414 patients.
    - Annotation pipeline: Four surgical interns perform initial screening → two senior surgeons conduct review → ensuring reliable per-frame degradation annotations.
    - Scale: 3,113 degraded images (2,127 smoke + 849 fog + 137 splash) at 1280×720 resolution.
    - Sources: Site A (cholecystectomy, bile duct, pancreas, spleen, and liver surgeries) and Site B (mediastinal, esophageal, and lung surgeries).
    - Design Motivation: Multi-source, multi-procedure collection ensures data diversity; sample ratios reflect the real-world frequency of each degradation type in clinical practice.

2. **Paired Label Generation and Optical Flow Alignment**:

    - Function: Provides a paired clean reference frame for each degraded frame.
    - Mechanism: A PS-frame scheme is adopted—the nearest clean frame preceding a degraded frame serves as the reference. Due to endoscope motion causing misalignment, a pretrained PWC-Net estimates optical flow to warp the reference frame into alignment:
      $\mathbf{F}_{UR \to P} = \mathcal{O}(\mathbf{UR}, \mathbf{P}), \quad \mathbf{UR}_{warp} = \mathcal{W}(\mathbf{UR}, \mathbf{F}_{UR \to P})$
    - A mask $\mathbf{M}$ suppresses regions of inaccurate optical flow during training: $\mathcal{L}_{rec} = \sum_i ||\mathbf{M}_i \odot (\mathbf{UR}_{warp,i} - \mathbf{P}_i)||_1$
    - Design Motivation: Perfectly aligned ground truth is unobtainable in real surgical settings; optical flow alignment represents a pragmatic compromise between authenticity and training feasibility.

3. **Fine-Grained Degradation Grading**:

    - Function: Smoke and fog samples are graded into 4 severity levels; splash samples are categorized into 4 substance types.
    - Smoke/Fog levels: Level 1 (mild, <1/3 of field of view) → Level 2 (moderate, 1/3–2/3) → Level 3 (severe, >2/3) → Level 4 (complete occlusion, impairing judgment).
    - Splash types: $T_{blood}$ (blood), $T_{fat}$ (fat), $T_{bile}$ (bile), $T_{fluid}$ (tissue fluid).
    - Design Motivation: Fine-grained grading enables analysis of algorithm performance across varying difficulty levels and provides a foundation for difficulty-aware training strategies.

4. **Analysis Beyond Pixel-Level Restoration**:

    - **Depth Estimation**: A depth estimator is applied to restored defog images to assess 3D structural preservation.
    - **Semantic Segmentation**: SAM and MedSAM are used to evaluate scene parsing and instrument segmentation performance.
    - **Surgical vs. Natural Scene Differences**: t-SNE visualizations reveal clear separation between surgical fog and natural fog in feature space; surgical fog exhibits locally abrupt distributions, whereas natural fog is gradual.
    - Design Motivation: The ultimate goal of surgical image restoration is not optimizing pixel-level metrics but facilitating downstream clinical tasks such as anatomical structure recognition.

### Loss & Training

All 22 compared methods are trained under a unified configuration:
- PyTorch implementation, dual NVIDIA RTX 4090 GPUs.
- Adam optimizer, random 128×128 patch cropping, batch size 2.
- 200k total iterations with learning rate halved every 100k iterations.
- All methods trained with optical-flow-aligned paired labels.

## Key Experimental Results

### Main Results

**Performance of general-purpose restoration models on SurgClean**:

| Method | Desmoke PSNR↑ | Desmoke SSIM↑ | Defog PSNR↑ | Defog SSIM↑ | Desplash PSNR↑ | Desplash SSIM↑ | Params |
|--------|--------------|--------------|------------|------------|---------------|---------------|--------|
| ConvIR | **19.43** | 0.678 | 18.87 | 0.619 | 21.33 | 0.717 | 14.83M |
| FocalNet | 19.24 | 0.679 | **19.07** | **0.628** | 21.42 | 0.717 | 3.74M |
| Restormer | 18.94 | 0.674 | 19.04 | 0.619 | 21.40 | 0.718 | 26.13M |
| MambaIR | 19.32 | 0.679 | 18.87 | 0.622 | 21.43 | 0.722 | 4.31M |
| X-Restormer | 18.03 | 0.659 | 18.60 | 0.628 | **22.32** | **0.735** | 42.52M |
| AST | 19.18 | 0.635 | 17.05 | 0.606 | 22.05 | 0.731 | 19.92M |
| RAMiT | 19.03 | 0.677 | 19.02 | 0.625 | 21.43 | 0.718 | **0.30M** |

### Ablation Study (Cross-Dataset Generalization & Downstream Tasks)

| Experimental Setting | Key Finding | Note |
|----------------------|-------------|------|
| DesmokeData → DesmokeData | Higher PSNR | DesmokeData degradations are relatively simple |
| SurgClean → SurgClean | Relatively lower PSNR | SurgClean degradations are more complex |
| DesmokeData → SurgClean | Large performance drop | Poor cross-domain generalization |
| SurgClean → DesmokeData | **Smaller performance drop** | Models trained on SurgClean generalize better |
| Depth estimation after restoration | Highest defog PSNR ≠ best depth | Pixel metrics and downstream tasks are not fully aligned |
| Semantic segmentation after restoration | MambaIRv2 achieves highest mIoU despite moderate PSNR | Trade-off between semantic preservation and pixel reconstruction |

### Key Findings

- **All methods fall far short of clinical standards**: The best desmoke PSNR is only 19.43 dB and defog 19.07 dB, with clearly visible residual degradations.
- **Task-specific methods show no clear advantage**: Dedicated desmoke/defog methods do not outperform general-purpose restoration models, indicating a large distribution gap between surgical and natural degradations.
- **Low-severity degradations are manageable; high-severity ones remain challenging**: Levels 1–2 show clear improvement, while Levels 3–4 show limited gains.
- **Pixel metrics are inconsistent with downstream task performance**: Methods with the highest restoration PSNR do not necessarily excel at depth estimation or semantic segmentation.
- **Models trained on SurgClean generalize better**: Attributed to the more diverse and complex degradation distribution in SurgClean.

## Highlights & Insights

- **First multi-type real-world surgical restoration dataset**: Fills a critical gap in real surgical data for defog and desplash tasks.
- **Comprehensive benchmark design**: 22 methods, 3 degradation types, 4 severity levels, and 5 evaluation metrics, providing a standardized platform for future research.
- **In-depth analysis of surgical vs. natural degradation differences**: t-SNE visualizations and depth estimation results reveal fundamental distinctions between the two degradation categories, pointing to directions for developing surgery-specific algorithms.
- **Closed-loop evaluation from restoration to downstream tasks**: Evaluation goes beyond pixel-level metrics to assess impacts on depth estimation and semantic segmentation, better reflecting clinical needs.

## Limitations & Future Work

- **Very limited desplash samples**: Only 137 images, insufficient to adequately train deep learning models.
- **Paired labels are not perfectly aligned**: Optical flow alignment is an approximation that may introduce artifacts under large-displacement scenarios.
- **Simultaneous multiple degradations are not considered**: In practice, smoke and splash may co-occur during surgery.
- **No new algorithm is proposed**: As a benchmark paper, the primary contributions lie in data and evaluation, with no targeted methodological innovation.
- **Evaluated methods skew toward general-purpose approaches**: Recent diffusion model–based restoration methods (e.g., DiffIR, IR-SDE) are not included.

## Related Work & Insights

- Compared to CycleGAN-DesmokeGAN (1,400 unpaired images) and Desmoke-LAP (3,000 unpaired images), SurgClean provides real paired labels and covers multiple degradation types.
- DesmokeData (961 images) has paired labels but covers only desmoke with lower degradation complexity.
- Natural-scene restoration methods (e.g., those trained on the RESIDE dehazing dataset) transfer poorly to surgical scenes, underscoring the necessity of domain-specific design.
- The lightweight RAMiT (0.3M parameters) offers an edge-deployment advantage with acceptable performance, warranting further exploration in surgical settings.

## Rating
- Novelty: ⭐⭐⭐⭐ — First real surgical restoration dataset covering three degradation types, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 22 methods, multi-dimensional evaluation, cross-dataset validation, and downstream task analysis; extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, thorough data analysis, and rich figures and tables.
- Value: ⭐⭐⭐⭐ — Provides the surgical image restoration community with a standardized platform and important baselines.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings](lemon_a_large_endoscopic_monocular_dataset_and_foundation_model_for_perception_in.md)
- [\[CVPR 2026\] Event-Level Detection of Surgical Instrument Handovers in Videos](event_level_detection_of_surgical_instrument_handovers_in_videos.md)
- [\[CVPR 2026\] Beyond Pixel Simulation: Pathology Image Generation via Diagnostic Semantic Tokens and Prototype Control](beyond_pixel_simulation_pathology_image_generation_via_diagnostic_semantic_token.md)
- [\[CVPR 2026\] Focus-to-Perceive Representation Learning: A Cognition-Inspired Hierarchical Framework for Endoscopic Video Analysis](focus-to-perceive_representation_learning_a_cognition-inspired_hierarchical_fram.md)
- [\[CVPR 2026\] Synergistic Bleeding Region and Point Detection in Laparoscopic Surgical Videos](synergistic_bleeding_region_and_point_detection_in_laparoscopic_surgical_videos.md)

<!-- RELATED:END -->
