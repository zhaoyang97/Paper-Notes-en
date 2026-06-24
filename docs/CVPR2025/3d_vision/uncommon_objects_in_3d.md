---
title: >-
  [Paper Note] UnCommon Objects in 3D
description: >-
  [CVPR 2025][3D Vision][3D Dataset] Meta introduces uCO3D—currently the largest public object-centric 3D dataset, containing high-resolution videos of over 1,000 object categories with complete 360° 3D annotations (camera poses, depth maps, point clouds, 3D Gaussian Splatting reconstructions, and text descriptions). Training on this dataset yields significantly better performance on multiple 3D learning tasks compared to MVImgNet and CO3Dv2.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Dataset"
  - "Object-Centric"
  - "Gaussian Splatting"
  - "Novel View Synthesis"
  - "3D Generation"
date: 2026-05-08
content_hash: 97a818ae01ac0a91
---

# UnCommon Objects in 3D

**Conference**: CVPR 2025  
**arXiv**: [2501.07574](https://arxiv.org/abs/2501.07574)  
**Code**: [https://github.com/facebookresearch/uco3d](https://github.com/facebookresearch/uco3d)  
**Area**: 3D Computer Vision / Dataset  
**Keywords**: 3D Dataset, Object-Centric, Gaussian Splatting, Novel View Synthesis, 3D Generation

## TL;DR
Meta introduces uCO3D—currently the largest public object-centric 3D dataset, containing high-resolution videos of over 1,000 object categories with complete 360° 3D annotations (camera poses, depth maps, point clouds, 3D Gaussian Splatting reconstructions, and text descriptions). Training on this dataset yields significantly better performance on multiple 3D learning tasks compared to MVImgNet and CO3Dv2.

## Background & Motivation

**Background**: The rapid advancement of 3D deep learning and 3D generative AI relies heavily on high-quality 3D training data. Current mainstream object-centric 3D datasets include CO3Dv2 (approx. 37K sequences, 51 categories) and MVImgNet (approx. 220K sequences, 238 categories), which primarily consist of crowdsourced multi-view videos.

**Limitations of Prior Work**: (1) Insufficient category diversity—CO3Dv2 contains only 51 daily object categories, and while MVImgNet expands this to 238, it still falls far short of covering real-world long-tail objects (such as traditional musical instruments, antiques, handicrafts, etc.); (2) Incomplete viewpoint coverage—most crowdsourced videos only capture the front or sides of objects (approx. 180°–270°), lacking complete 360° coverage, which results in missing back-view information; (3) Inconsistent quality—crowdsourced data suffers from extensive motion blur, poor focus, and cluttered backgrounds, leading to low-quality samples.

**Key Challenge**: 3D learning models (such as novel view synthesis, 3D reconstruction, and text-to-3D generation) require high diversity, high coverage, and high-quality training data. However, there is an inherent conflict between large-scale crowdsourcing and quality control: relaxing quality standards yields more data but introduces excessive noisy samples, while strict quality control limits the scale of the dataset.

**Goal**: To construct an object-centric 3D dataset that simultaneously satisfies high diversity (1000+ categories), high coverage (complete 360° coverage), and high quality (strict quality control), while providing rich annotations (camera poses, depth maps, point clouds, Gaussian Splatting, and text descriptions).

**Key Insight**: The authors ensure high-quality data while maintaining a large scale by using a carefully designed capture protocol (requiring camera operators to complete at least one full circle around the object) and a multi-level quality control pipeline (combining automated checks and manual auditing). Additionally, VGGSfM is introduced as a more advanced SfM tool to improve the quality of 3D annotations.

**Core Idea**: By utilizing a fine-grained capture protocol, multi-level quality control, and advanced 3D reconstruction tools, this work bypasses the "data scale vs. data quality" trade-off to establish a large-scale 3D dataset with comprehensive annotations.

## Method

### Overall Architecture
The construction of uCO3D is divided into four stages: (1) Data Collection—designing the capture protocol and crowdsourcing high-resolution object videos; (2) Quality Control—filtering out low-quality samples using a multi-level automated and manual review process; (3) 3D Annotation—using VGGSfM to estimate camera poses and sparse point clouds, while using monocular depth estimation to generate depth maps; (4) Value-Added Annotation—generating 3D Gaussian Splatting reconstructions and VLM-generated text descriptions for each object. The final dataset provides a unified PyTorch data loading interface.

### Key Designs

1. **360° Coverage Capture Protocol**:

    - **Function**: Ensures each object video covers a complete 360° perspective.
    - **Mechanism**: The capture protocol requires the operator to place the object on a clean background, holds the phone, and rotates around the object for at least one full circle (>360°) at a stable distance and speed. The capture app displays a real-time coverage indicator to remind users to complete missing angles. Each video contains at least 200 frames with a resolution of $\geq 1080\text{p}$. Post-capture, viewpoint coverage is automatically checked by estimating the azimuth angle range of the camera trajectory, filtering out videos with coverage of less than 300°.
    - **Design Motivation**: 360° coverage is crucial for 3D reconstruction and novel view synthesis, but it remains a primary shortcoming of existing datasets. Many sequences in CO3Dv2 cover only a 90°–180° arc, preventing models from learning the appearance of the object's back.

2. **Multi-Level Quality Control Pipeline**:

    - **Function**: Systematically filters out low-quality samples to guarantee the overall quality of the dataset.
    - **Mechanism**: The quality control pipeline consists of four levels: **(Level 1) Automated Frame-Level Check**—detecting motion blur (via Laplacian variance thresholding), abnormal exposure (via histogram analysis), and focus quality. **(Level 2) Sequence-Level Check**—verifying SfM convergence, camera trajectory continuity, and the number of reconstructed points to filter out sequences where SfM fails. **(Level 3) 3D Consistency Check**—excluding sequences with poor 3D annotation quality using reprojection errors and multi-view consistency scores. **(Level 4) Manual Audit**—expert annotators conduct a final review on samples passing automated checks, evaluating aesthetic criteria such as object-centricity and background cleanliness. Every stage records exact rejection rate statistics.
    - **Design Motivation**: A single-level quality control mechanism struggles to address all quality dimensions. The hierarchical design allows inexpensive automated checks to filter out the majority of low-quality samples first (~60%), leaving only ~40% for the expensive manual audit phase, thus balancing quality and cost.

3. **3D Gaussian Splatting Reconstruction and Text Description**:

    - **Function**: Provides 3D Gaussian Splatting (3DGS) reconstructions and linguistic descriptions for each object, enriching the annotation dimensions of the dataset.
    - **Mechanism**: For each object sequence that passes quality control, the 3DGS algorithm is used to generate high-quality 3D reconstructions. The optimization is initialized using sparse point clouds from VGGSfM and optimized via multi-view photometric consistency. Each reconstructed Gaussian Splatting model includes parameters such as position, covariance, color, and opacity. Concurrently, a VLM (Vision-Language Model) is utilized to generate text captions across multiple views of each object, retaining the most accurate description after filtering and deduplication.
    - **Design Motivation**: 3DGS reconstructions offer directly usable ground-truth reconstruction results for novel view synthesis and 3D generation, sparing downstream users from running reconstruction pipelines themselves. The text descriptions directly support generative tasks like text-to-3D.

### Loss & Training
Since uCO3D is a dataset paper, it does not involve new training losses. The paper evaluates several downstream models trained on uCO3D for validation: (1) an Instant3D-style text-to-3D model; (2) DUSt3R/MASt3R dense 3D reconstruction models; and (3) standard novel view synthesis models. All models utilize their respective standard training configurations.

## Key Experimental Results

### Dataset Comparison

| Dataset | Sequences | Categories | 360° Coverage | 3DGS Reconstruction | Text Description | Avg. Resolution |
|--------|--------|--------|-----------|-----------|----------|-----------|
| CO3Dv2 | 37K | 51 | ✗ | ✗ | ✗ | 640p |
| MVImgNet | 220K | 238 | ✗ | ✗ | ✗ | 512p |
| **uCO3D** | **~50K** | **1,000+** | **✓** | **✓** | **✓** | **1080p** |

### Downstream Task Experiments

| Task | Training Data | Metric | CO3Dv2 | MVImgNet | uCO3D |
|------|----------|------|--------|----------|-------|
| Novel View Synthesis (NVS) | All | PSNR ↑ | 24.3 | 25.1 | **27.2** |
| Novel View Synthesis (NVS) | All | SSIM ↑ | 0.832 | 0.851 | **0.889** |
| Novel View Synthesis (NVS) | All | LPIPS ↓ | 0.142 | 0.128 | **0.095** |
| Dense 3D Reconstruction | All | Chamfer-L1 ↓ | 0.058 | 0.051 | **0.039** |
| Text-to-3D | All | FID ↓ | - | - | **42.3** |
| Text-to-3D | All | CLIP Score ↑ | - | - | **0.312** |

### Ablation Study: Impact of Data Quality

| Configuration | NVS PSNR ↑ | NVS LPIPS ↓ | 3D Reconstruction Chamfer ↓ | Description |
|------|-----------|-------------|-----------------|------|
| Full uCO3D | **27.2** | **0.095** | **0.039** | Full dataset after quality control |
| w/o Level 4 Manual Audit | 26.5 | 0.108 | 0.043 | Holds some samples with low aesthetic quality |
| w/o Level 3 3D Consistency Check | 25.8 | 0.121 | 0.051 | 3D annotation noise increases |
| w/o 360° Coverage Filter | 25.1 | 0.134 | 0.055 | Lack of back-view info degrades NVS performance |
| Automated QC Only (Level 1-2) | 24.9 | 0.138 | 0.057 | Quality is close to MVImgNet |

### Key Findings
- **Data quality > data quantity**: Although the number of sequences in uCO3D is far fewer than in MVImgNet (~50K vs. 220K), it outperforms MVImgNet across all downstream tasks. This demonstrates that meticulous quality control and 360° coverage are far more valuable than simply scaling up raw data volume.
- **360° coverage is the most critical factor**: Removing the coverage filter drops the PSNR from 27.2 to 25.1 (-2.1 dB), which is the most significant drop among all ablations, demonstrating that view completeness is crucial for 3D learning.
- **Multi-level QC is indispensable**: Removing any level of quality control leads to a marked drop in performance. Together, Level 3 (3D consistency) and Level 4 (manual audit) contribute an approximate improvement of 1.4 dB in PSNR.
- uCO3D is the first real-world object dataset that can directly support text-to-3D training, as it simultaneously provides 3D reconstructions and corresponding text descriptions.

## Highlights & Insights
- **The dataset building philosophy of "establishing high quality before expanding scale"** offers a valuable reflection compared to the currently popular paradigm of "scaling first and filtering later". Controlling quality at the source of collection (e.g., using a 360° coverage indicator) is much more efficient than post-hoc filtering.
- **Value-added annotations (3DGS + captions) greatly extend the applicability of the dataset**: They not only serve traditional NVS and 3D reconstruction but also directly support cutting-edge tasks like text-to-3D generation. This paradigm of "one-time collection, multi-dimensional annotation" is worth adopting widely.
- **The introduction of VGGSfM** demonstrates that advancements in SfM tools directly enhance the annotation quality of 3D datasets. Dataset papers should focus not only on collection but also on the selection and upgrades of the annotation toolchain.

## Limitations & Future Work
- The dataset primarily focuses on rigid objects; deformable objects (e.g., cloth, liquids) and large-scale scenes (e.g., buildings, vehicles) are not covered.
- Even with 1000+ categories, the number of samples per long-tail category might be inadequate for independent training.
- The text descriptions are automatically generated by a VLM and may occasionally be inaccurate or too generic.
- The capture cost is relatively high—360° coverage requirements and multi-level QC place an extra burden on crowdsourcing.
- The mechanism for the continuous maintenance and expansion of the dataset remains unclear.

## Related Work & Insights
- **vs CO3Dv2**: CO3Dv2 is Meta's ancestral dataset (51 categories, without 360° guarantees). uCO3D represents a comprehensive upgrade in the number of categories (20x), viewpoint coverage, and richness of annotations.
- **vs MVImgNet**: MVImgNet is larger but has inconsistent quality and lacks 360° coverage. uCO3D achieves superior downstream performance with less data, manifesting the value of prioritizing quality over quantity.
- **vs Objaverse**: Objaverse provides synthetic 3D models, whereas uCO3D provides multi-view videos and 3D annotations of real-world objects, making the two complementary.

## Rating
- Novelty: ⭐⭐⭐ Essentially a dataset engineering effort rather than a methodological breakthrough, though it makes a substantial impact on scale and quality.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on three downstream tasks with comprehensive quality control ablations and fair comparisons.
- Writing Quality: ⭐⭐⭐⭐ Follows standard writing practices for dataset papers, with clear descriptions of the capture and QC pipelines.
- Value: ⭐⭐⭐⭐⭐ As an open-source, foundational dataset in the 3D domain, it holds long-term value for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PICO: Reconstructing 3D People In Contact with Objects](pico_reconstructing_3d_people_in_contact_with_objects.md)
- [\[CVPR 2025\] Gen3DEval: Using vLLMs for Automatic Evaluation of Generated 3D Objects](gen3deval_using_vllms_for_automatic_evaluation_of_generated_3d_objects.md)
- [\[CVPR 2025\] Instant3dit: Multiview Inpainting for Fast Editing of 3D Objects](instant3dit_multiview_inpainting_for_fast_editing_of_3d_objects.md)
- [\[CVPR 2025\] RigGS: Rigging of 3D Gaussians for Modeling Articulated Objects in Videos](riggs_rigging_of_3d_gaussians_for_modeling_articulated_objects_in_videos.md)
- [\[CVPR 2025\] IAAO: Interactive Affordance Learning for Articulated Objects in 3D Environments](iaao_interactive_affordance_learning_for_articulated_objects_in_3d_environments.md)

</div>

<!-- RELATED:END -->
