---
title: >-
  [Paper Note] Image Quality Assessment: From Human to Machine Preference
description: >-
  [CVPR 2025][Segmentation][Image Quality Assessment] This paper introduces Image Quality Assessment for Machine Vision System (IQA for MVS) for the first time, establishing the Machine Preference Database (MPD) which contains 2.25 million fine-grained annotations and 30,000 reference/distorted image pairs. Experiments demonstrate that existing HVS-centric IQA metrics fail to accurately characterize machine preferences, revealing fundamental differences between human and machin…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Image Quality Assessment"
  - "Machine Preference"
  - "Multimodal Large Language Models"
  - "Downstream Tasks"
  - "Database Construction"
date: 2026-05-08
content_hash: 8a17d7e2a62858e1
---

# Image Quality Assessment: From Human to Machine Preference

**Conference**: CVPR 2025  
**arXiv**: [2503.10078](https://arxiv.org/abs/2503.10078)  
**Code**: [https://github.com/lcysyzxdxc/MPD](https://github.com/lcysyzxdxc/MPD)  
**Area**: Image Segmentation  
**Keywords**: Image Quality Assessment, Machine Preference, Multimodal Large Language Models, Downstream Tasks, Database Construction

## TL;DR

This paper introduces Image Quality Assessment for Machine Vision System (IQA for MVS) for the first time, establishing the Machine Preference Database (MPD) which contains 2.25 million fine-grained annotations and 30,000 reference/distorted image pairs. Experiments demonstrate that existing HVS-centric IQA metrics fail to accurately characterize machine preferences, revealing fundamental differences between human and machine vision systems.

## Background & Motivation

**Background**: Over the past two decades, the field of IQA has developed a mature methodological framework around the Human Visual System (HVS), including hundreds of fine-grained databases and algorithms that accurately model HVS preferences. Since 2023, the number of machine-to-machine (M2M) connections has surpassed machine-to-human (M2H) connections for the first time, making machines the primary consumers of image and video data.

**Limitations of Prior Work**: There are fundamental differences in perception mechanisms between HVS and Machine Vision Systems (MVS)—HVS focuses on the similarity of texture, structure, and color, whereas MVS focuses on the consistency of downstream task results (segmentation, detection, question answering). An obvious quality degradation perceived by human eyes might not affect downstream machine tasks at all; conversely, imperceptible perturbations to human eyes might severely damage machine outputs. Currently, there is not even a database that annotates global machine preference scores for images.

**Key Challenge**: Evaluation metrics for image processing algorithms are fragmented—different compression or restoration algorithms use separate downstream tasks and models for validation, lacking a unified and comprehensive representation of machine preferences.

**Goal**: (1) Define subjective machine preferences—which downstream tasks, testing models, and evaluation metrics? (2) Establish a large-scale database for annotating machine preferences; (3) Verify whether existing HVS-centric IQA algorithms can predict machine preferences.

**Key Insight**: Referencing the ITU standards for human subjective evaluation, 15 LMMs and 15 specialized CV models are treated as "machine subjects" to collect machine Mean Opinion Scores (MOS) across 7 downstream tasks.

**Core Idea**: Score 30,000 distorted images using 30 machine models across 7 downstream tasks to construct the first machine preference database (MPD), proving that existing IQA metrics cannot accurately represent machine preferences.

## Method

### Overall Architecture

The entire workflow consists of three steps: (1) Reference Image Collection and Distortion Generation: 1,000 high-quality reference images (NSI/SCI/AIGI) are selected, and 30 types of distortions $\times$ 5 intensity levels are applied to generate 30,000 distorted images; (2) Multi-task Annotation: 15 LMMs and 15 specialized CV models are utilized to annotate the performance differences of reference/distorted image pairs across 7 tasks; (3) MOS Aggregation: The 7 tasks are grouped into 5 dimensions with 15 subjects per dimension, which are then normalized and aggregated into a MOS ranging from (0, 5).

### Key Designs

1. **Multi-task Machine Preference Definition Framework**:

    - **Function**: Standardizes the definition of "what constitutes a high-quality image from a machine's perspective"
    - **Mechanism**: Covers 7 downstream tasks categorized into LMM tasks (YoN/MCQ/VQA/CAP) and CV tasks (SEG/DET/RET). The scoring methods for each task are carefully designed: YoN uses the softmax probability difference $S_{YoN} = |\sigma(P_{dis}) - \sigma(P_{ref})|$; MCQ utilizes the cosine distance of option probability vectors; VQA employs a CLIP text encoder to compute semantic similarity; CAP uses an ensemble metric of BLEU+CIDEr+SPICE; SEG/DET uses IoU; RET uses the sum of Top-1/5/10 accuracy
    - **Design Motivation**: No single task can represent the comprehensive preferences of machines. By covering 7 tasks and grouping them into 5 assessment dimensions (with 15 subjects per dimension), this simulates the standard practice in human IQA where scores from multiple annotators are averaged to obtain the MOS

2. **Large-Scale Distorted Image Construction**:

    - **Function**: Comprehensively covers various types of distortion in real-world communication scenarios
    - **Mechanism**: Reference images comprise three major categories: Natural Scene Images (UGC/PGC), Screen Content Images (webpages/games/movies), and AI-Generated Images (6 text-to-image models). The 30 distortion types are classified into 7 categories (Blur, Luminance, Chrominance, Contrast, Noise, Compression, Spatial), each with 5 levels of intensity. To ensure fairness, the distortion levels are manually controlled to align human-perceived quality degradation at comparable levels
    - **Design Motivation**: Most existing IQA databases focus on a single image type (natural or AI-generated images) with limited distortion types. MPD ensures completeness of evaluation by comprehensively covering various image and distortion types

3. **Multi-granularity Analysis and Validation Framework**:

    - **Function**: Validates the reliability and findings of MPD from both global and individual perspectives
    - **Mechanism**: Global analyses evaluate the SRCC correlation among task scores and the distributions of MOS under 30 types of distortions. Individual analyses examine performance differences of each machine subject across various levels of intensity and content types, as well as the inter-subject consistency among machine subjects (SRCC=0.62). The inter-subject consistency among humans is also compared (SRCC=0.76), proving that "individual differences" in machine preferences are even larger than those in humans
    - **Design Motivation**: Drawing an analogy to the ITU standards for human subjective evaluation, it is essential to validate subject reliability and the rationality of score distributions

### Loss & Training

MPD itself is a database contribution and does not involve model training. However, the paper comprehensively benchmarks more than 10 existing IQA metrics (such as PSNR, SSIM, LPIPS, CLIPIQA, ARNIQA, etc.) on MPD as validation experiments.

## Key Experimental Results

### Main Results (Performance of IQA Metrics in Predicting Machine Preferences)

| IQA Metric | Severe Distortion SRCC ↑ | Mild Distortion SRCC ↑ | NSI SRCC ↑ | Human-centric Design |
|----------|----------------|----------------|------------|---------|
| PSNR | 0.387 | 0.310 | 0.412 | ✗ |
| SSIM | 0.597 | 0.267 | 0.629 | ✗ |
| LPIPS | 0.625 | 0.008 | 0.697 | ✔ |
| ARNIQA | 0.834 | 0.239 | 0.870 | ✗ |
| AHIQ | 0.806 | 0.530 | 0.845 | ✗ |
| TOPIQ-FR | 0.718 | 0.425 | 0.751 | ✔ |

Key Findings: (1) All metrics perform reasonably well under severe distortion but drop sharply under mild distortion (LPIPS drops to 0.008); (2) Metrics specifically designed for HVS (indicated with ✔) can perform even worse (e.g., LPIPS becomes almost ineffective under mild distortion).

### Ablation Study (Characterizing Machine Preferences)

| Analysis Dimension | Key Findings |
|---------|---------|
| Inter-task Correlation | VQA shows the lowest correlation with SEG/DET/RET, indicating that different tasks represent distinct preference dimensions |
| Machine Subject Consistency | Inter-machine SRCC = 0.62 < Inter-human SRCC = 0.76, indicating larger variance in machine preferences |
| Distortion Sensitivity | Machines are highly sensitive to Lens blur but almost unaffected by Mean brighten |
| Intensity Impact | Quality degrades as Lab saturation intensity increases, while it remains almost unchanged under HSV saturation |

### Key Findings

- Existing IQA metrics perform extremely poorly in predicting machine preferences under mild distortion, which is exactly the most common scenario in practical applications.
- Machine sensitivity to different distortion types is completely different from humans—while four types of Macro Block distortions appear almost identical to the human eye, the machine's perception for Block exchange is far worse than the other three.
- Fine-tuning existing IQA algorithms yields only marginal improvements on MPD, and they still fail to fully characterize machine preferences, indicating the need for fundamentally new approaches.
- The sophistication of LMMs affects their robustness: InternVL2 performs exceptionally well on YoN, whereas the traditional LLaVA1.5 is surprisingly more stable on CAP.

## Highlights & Insights

- **Forward-looking Problem Definition**: In an era where M2M communications surpass M2H, systematically defining image quality preferences for machines is a timely and critical pioneering effort. As more images are consumed by machines rather than humans, MVS-centric IQA will become a core requirement.
- **Robust Database Construction Methodology**: Designing the machine MOS collection pipeline by referencing ITU standards, covering 7 tasks and 30 models, ensures both rigor and comprehensiveness. This methodology can be generalized to video quality assessment.
- **Practical Value in Revealing HVS vs. MVS Differences**: For instance, if image compression algorithms are oriented towards machine consumption (e.g., autonomous driving, security surveillance), they should be optimized against distortion types that machines are sensitive to, rather than human preferences.

## Limitations & Future Work

- The reference images are limited to 1,000; although there are 30,000 distorted images, the diversity of image content may still be insufficient.
- LMM outputs are influenced by parameters like temperature; though set to 0, output stability still varies across different models.
- Determining the weights of the 7 tasks remains an open problem—the current equal-weighted normalization may not fit all application scenarios.
- Video quality assessment is not considered, whereas video is the primary media format consumed by machines.
- Machine models iterate extremely rapidly, and the database needs continuous maintenance to maintain its timeliness.
- There is a lack of directly proposed IQA algorithms optimized for machine preferences; the paper only validates the limitations of existing methods.

## Related Work & Insights

- **vs LIVE/TID2013/KADID**: These classic IQA databases only annotate human preferences and contain limited distortion types. MPD introduces the machine preference dimension for the first time and covers three image categories (NSI/SCI/AIGI).
- **vs KonIQ-10K/AGIQA-3K**: Although these NR-IQA databases are large in scale, they are still based on human subjective evaluation. MPD reveals the huge gap between human evaluation and machine performance.
- **vs CMC-Bench**: CMC-Bench evaluates the multimodal capabilities of LMMs, whereas MPD does the opposite—using LMMs as evaluators to annotate machine preferences for images, offering complementary perspectives.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Formulating a complete framework for IQA for Machine Vision for the first time; highly pioneering in its problem definition.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation with 2.25 million annotations, 30 models, and 10+ IQA algorithms, though it lacks proposed new algorithms.
- Writing Quality: ⭐⭐⭐⭐ The database construction workflow is clear, though some formulas and tables are overly dense.
- Value: ⭐⭐⭐⭐⭐ The database and methodology have far-reaching impacts on the IQA community, potentially driving a paradigm shift from HVS to MVS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] QMamba: On First Exploration of Vision Mamba for Image Quality Assessment](../../ICML2025/segmentation/qmamba_on_first_exploration_of_vision_mamba_for_image_quality_assessment.md)
- [\[CVPR 2025\] The Devil is in Temporal Token: High Quality Video Reasoning Segmentation](the_devil_is_in_temporal_token_high_quality_video_reasoning_segmentation.md)
- [\[CVPR 2025\] GleSAM: Segment Any-Quality Images with Generative Latent Space Enhancement](segment_any-quality_images_with_generative_latent_space_enhancement.md)
- [\[CVPR 2025\] ROS-SAM: High-Quality Interactive Segmentation for Remote Sensing Moving Object](ros-sam_high-quality_interactive_segmentation_for_remote_sensing_moving_object.md)
- [\[CVPR 2025\] MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation](mv-ssm_multi-view_state_space_modeling_for_3d_human_pose_estimation.md)

</div>

<!-- RELATED:END -->
