---
title: >-
  [Paper Note] SignAvatars: A Large-scale 3D Sign Language Holistic Motion Dataset and Benchmark
description: >-
  [ECCV 2024][3D Vision][sign language] Proposes SignAvatars, the first large-scale multi-prompt (HamNoSys/language/word) 3D sign language holistic motion dataset (70K videos, 8.34M frames, 153 signers). It designs an automatic 3D annotation pipeline with biomechanical constraints, and proposes the VQ-VAE-based SignVAE model as the first benchmark baseline for 3D Sign Language Production (SLP).
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "sign language"
  - "3D holistic motion"
  - "dataset"
  - "motion generation"
  - "SMPL-X"
date: 2026-05-08
content_hash: a1553f9d7060ca64
---

# SignAvatars: A Large-scale 3D Sign Language Holistic Motion Dataset and Benchmark

**Conference**: ECCV 2024  
**arXiv**: [2310.20436](https://arxiv.org/abs/2310.20436)  
**Code**: [Project Page](https://signavatars.github.io/)  
**Area**: LLM Evaluation  
**Keywords**: sign language, 3D holistic motion, dataset, motion generation, SMPL-X

## TL;DR

Proposes SignAvatars, the first large-scale multi-prompt (HamNoSys/language/word) 3D sign language holistic motion dataset (70K videos, 8.34M frames, 153 signers). It designs an automatic 3D annotation pipeline with biomechanical constraints, and proposes the VQ-VAE-based SignVAE model as the first benchmark baseline for 3D Sign Language Production (SLP).

## Background & Motivation

**Background**: Globally, there are 466 million deaf and hard-of-hearing individuals who communicate using over 300 sign languages. While research in NLP and computer vision is highly mature, 3D digitization research in the domain of sign language lags significantly behind.

**Limitations of Prior Work**:
   - Existing sign language datasets (e.g., Phoenix, How2Sign) are limited to 2D videos or 2D keypoint annotations, suffering from depth ambiguity issues (where different gestures may look identical in 2D).
   - The only 3D sign language dataset, SGNify, contains only 50 videos, supports only isolated gestures (one gesture per video), and only offers HamNoSys annotations.
   - 3D avatar annotation is a labor-intensive task completed entirely manually by sign language experts, which often yields unnatural results.

**Key Challenge**: The 3D digitization of sign language requires precise holistic-body (especially hand) mesh annotations. However, existing automatic methods struggle to handle the frequent hand self-occlusions and two-hand interactions in sign language scenarios, whereas manual annotation is prohibitively expensive and often leads to unnatural results.

**Goal** (1) Construct the first large-scale, multi-prompt 3D holistic sign language motion dataset; (2) design an automatic annotation pipeline capable of handling complex hand interactions; (3) establish the first benchmark for 3D Sign Language Production (SLP).

**Key Insight**: Collect sign language videos at scale from public datasets and online videos, design an automatic SMPL-X fitting pipeline that combines temporal smoothing and biomechanical constraints to acquire 3D annotations, and propose the first baseline model based on the VQ-VAE architecture to generate 3D holistic sign language motions from multi-type prompts.

**Core Idea**: Elevate massive 2D sign language videos into precise 3D holistic-body mesh annotations via an automatic annotation pipeline, establishing the first large-scale 3D sign language dataset and production benchmark supporting multiple prompts (HamNoSys, natural language, and words).

## Method

### Overall Architecture

The system consists of three core components:

1. **Data Collection and Processing**: Aggregate 70K videos from multiple public data sources (including ASL 34K, GSL 8.3K, HamNoSys 5.8K, and Word 21K).
2. **Automatic 3D Annotation Pipeline**: Hierarchical initialization $\rightarrow$ multi-objective optimization (temporal smoothing + biomechanical constraints) $\rightarrow$ output of SMPL-X parameters.
3. **3D SLP Benchmark**: The VQ-VAE-based SignVAE model, which generates 3D holistic motion from text, HamNoSys, and word prompts.

### SignAvatars Dataset

**Data Scale and Composition**:

| Subset | Video Count | Frame Count | Type | Signer Count |
|------|--------|------|------|---------|
| Word | 21K | 1.39M | Word-level | 119 |
| PJM | 2.6K | 0.21M | HamNoSys | 2 |
| DGS | 1.9K | 0.12M | HamNoSys | 8 |
| GRSL | 0.8K | 0.06M | HamNoSys | 2 |
| LSF | 0.4K | 0.03M | HamNoSys | 2 |
| ASL | 34K | 5.7M | Language Sentence-level | 11 |
| GSL | 8.3K | 0.83M | Language + gloss | 9 |
| **Total** | **70K** | **8.34M** | **Multiple** | **153** |

**Representation**: The SMPL-X parametric model is employed, representing the motion state of each frame as $M_t = (\theta_t^b, \theta_t^h, \theta_t^f, \phi, \tilde{\beta})$, where:
- $\theta_t^b \in \mathbb{R}^{23 \times 6}$: body pose (including global orientation)
- $\theta_t^h \in \mathbb{R}^{30 \times 6}$: hand pose
- $\theta_t^f \in \mathbb{R}^{6}$: jaw pose
- $\phi$: facial expression
- $\tilde{\beta}$: optimized consistent shape parameters

The MANO hand subset is also provided: $M_t^h = (\theta_t^h, \tilde{\beta})$

### Key Designs

1. **Automatic Annotation Pipeline (Multi-objective Optimization)**: Holistic-body SMPL-X fitting is conducted by minimizing an objective function containing multiple regularization terms:

    $$E(\theta, \beta, \phi) = \lambda_J L_J + \lambda_\theta L_\theta + \lambda_\alpha L_\alpha + \lambda_\beta L_\beta + \lambda_s L_{\text{smooth}} + \lambda_a L_{\text{angle}} + L_{\text{bio}}$$

   Definitions of key terms:
    - $L_J$: 2D reprojection joint loss (aligns the projected SMPL-X joints onto the image with the 2D keypoints predicted by ViTPose + MediaPipe)
    - $L_\theta$: pose prior term (derived from SMPLify-X)
    - $L_\alpha$: bending penalty (constrains extreme bending of only elbows and knees)
    - $L_\beta$: shape prior
    - $L_{\text{smooth}}$: temporal smoothness regularization, enforcing smooth frame-to-frame variations of the body and hands
    - $L_{\text{angle}}$: joint angle limit prior
    - $L_{\text{bio}}$: biomechanical constraints

   **Design Motivation**: Standard 2D-to-3D reconstruction methods (e.g., OSX) underperform in sign language scenarios due to frequent hand self-occlusions and interactions, demanding temporal information and biomechanical constraints to recover plausible gestures.

2. **Biomechanical Hand Constraints**: Hand pose estimation is highly challenging because of rapid motion, interactions, and occlusions. A three-fold constraint is designed:

    - $L_{\text{bl}}$: bone length constraint (the bone length of each finger must remain within a reasonable range)
    - $L_{\text{palm}}$: palm region optimization (constrains curvature and angular distance of the four metacarpal bones)
    - $L_{\text{ja}}$: joint angle prior (constrains joint angles within the convex hull on the flexion-extension and abduction-adduction planes)
   
   Total biomechanical loss: $L_{\text{bio}} = \lambda_{bl} L_{\text{bl}} + \lambda_{palm} L_{\text{palm}} + \lambda_{ja} L_{\text{ja}}$

   **Mechanism**: Leverages anatomical constraints of the human hand to eliminate anatomically impossible hand poses, maintaining physiological plausibility even under severe occlusion.

3. **Hierarchical Initialization**: Initialization is key to successful optimization. A multi-source fusion strategy is employed:

    - 3D Initialization: Fusion of OSX + ACR + PARE (enhancing stability under occlusion or truncation)
    - 2D Keypoints: Fusion of ViTPose-based whole-body pose estimation + MediaPipe (confidence-guided filtering)
    - Optimization proceeds in five stages: the first three optimize shape parameters to obtain a mean shape, and the remaining two stages freeze the shape.

4. **SignVAE (3D SLP Baseline Model)**: A two-stage VQ-VAE architecture:

    - **Stage 1 - Dual Codebook Training**:
        - Motion VQ-VAE: Encodes the motion sequence $M_{1:T} \rightarrow$ quantizes it into codebook indices $\rightarrow$ decodes to reconstruct, with a downsampling rate of $w=4$.
        - Linguistic VQ-VAE (PLFG): Encodes prompts such as text and HamNoSys via CLIP or custom embeddings $\rightarrow$ quantizes them into linguistic codebook indices $\rightarrow$ decodes to reconstruct.
    - **Stage 2 - Autoregressive Generation**: Fuses linguistic feature embeddings and linguistic codebook index vectors, employing an autoregressive model to predict motion codebook index sequences given the semantic codebook indices.
   
   **Design Motivation**: Establishing semantic-to-motion correspondences in the discrete codebook space yields superior performance compared to directly regressing from high-level CLIP features.

### Loss & Training

**Annotation Pipeline Loss**: Weighted multi-objective optimization (equation as shown above) performed progressively in five stages.

**Motion VQ-VAE Training**:
$$L_{m\text{-}vq} = L_{\text{recon}}(M_{1:T}, \hat{M}_{1:T}) + \|sg[F^m_{1:T}] - \hat{F^m_{1:T}}\|_2 + \beta \|F^m_{1:T} - sg[\hat{F^m_{1:T}}]\|_2$$

**SLP Autoregressive Training**: Cross-entropy loss
$$L_{\text{SLP}} = \mathbb{E}_{X \sim p(X)}[-\log p(X|c)]$$

## Key Experimental Results

### Main Results

**3D Holistic Reconstruction Accuracy (EHF Dataset)**:

| Method | PA-MPVPE Holistic | PA-MPVPE Hand | PA-MPJPE Body | PA-MPJPE Hand |
|------|-------------|-------------|-------------|-------------|
| SMPLify-X | 65.3 | 75.4 | 62.6 | 12.9 |
| PyMAF-X | 50.2 | 10.2 | 52.8 | 10.3 |
| Motion-X w/GT 3Dkpt | 19.7 | - | 23.9 | - |
| **Ours (w/ bio)** | **12.9** | **4.7** | **15.6** | **5.8** |

**3D SLP HamNoSys Subset**:

| Method | DTW-MJE Top1↑ | DTW-MJE Top3↑ | DTW-MJE Top5↑ |
|------|--------------|--------------|--------------|
| Ham2Pose (2D only) | 0.092 | 0.197 | 0.354 |
| Ham2Pose-3d | 0.253 | 0.369 | 0.511 |
| **SignVAE (Ours)** | **0.516** | **0.694** | **0.786** |

### Ablation Study

**Ablation of PLFG Module (HamNoSys holistic)**:

| Method | R-Precision Top1↑ | R-Precision Top3↑ | MM-dist↓ |
|------|-------------------|-------------------|----------|
| Ham2Pose-3d | 0.291 | 0.386 | 3.875 |
| SignDiffuse (modified MDM) | 0.285 | 0.415 | 3.866 |
| SignVAE (Base, w/o PLFG) | 0.385 | 0.613 | 3.056 |
| **SignVAE (Ours, w/ PLFG)** | **0.429** | **0.657** | **2.651** |

### Key Findings

1. **Annotation Quality Substantially Outperforms SoTA**: On the EHF dataset, the PA-MPJPE body joint error is 15.6mm, a ~35% improvement compared to 23.9mm of Motion-X (w/ GT 3Dkpt); the PA-MPVPE for the hand is only 4.7mm.
2. **Critical Role of Biomechanical Constraints**: Incorporating biomechanical constraints reduces the hand PA-MPVPE from 5.4 to 4.7, and the hand MPVPE from 12.5 to 9.7, robustly eliminating unnatural gestures.
3. **Discrete Codebook Interaction Outperforms End-to-End Regression**: SignVAE outperforms SignDiffuse (based on MDM) on R-Precision Top1 by 50% (0.429 vs 0.285), demonstrating that semantic-to-motion alignment in the discrete representation space is more effective.
4. **Significant Boost from the PLFG Module**: Compared to the Base version that directly employs CLIP features, integrating the PLFG module improves R-Precision Top1 from 0.385 to 0.429, and reduces MM-dist from 3.056 to 2.651.
5. **Word-Level Prompts Perform Best**: Among the three prompt types, the word-level prompt yields the lowest FID (0.756) and the highest R-Precision (0.475 Top1), whereas spoken language remains the most challenging.

## Highlights & Insights

1. **Filling a Critical Research Gap**: This is the first comprehensive work in 3D sign language to simultaneously deliver a large-scale dataset, an automatic annotation method, and a production benchmark, carrying substantial social impact.
2. **Ingenious Application of Biomechanical Constraints**: Incorporating anatomical knowledge of human hands (bone length limits, palm curvature, and joint angle convex hulls) as optimization constraints offers an elegant solution to combat hand self-occlusions and interactions. This can be adapted for general hand-reconstruction tasks.
3. **Multi-source Initialization Fusion Strategy**: Fusing multiple models (OSX + ACR + PARE + ViTPose + MediaPipe) coupled with confidence-guided filtering represents a practical and robust engineering pipeline.
4. **Dual-Codebook VQ-VAE Design**: Formulating semantic-to-motion correspondence at the discrete codebook level (rather than inside a continuous feature space) leverages the discrete nature of VQ-VAE to reinforce cross-modal association, an approach highly valuable for other text-to-motion tasks.
5. **Diverse Prompt Support**: A unified framework supporting three prompt modes—HamNoSys (linguistic notation), spoken language (natural language), and word—covering diverse application requirements in sign language research.

## Limitations & Future Work

1. **Lack of Mature 3D Translation-Back Evaluation Methods**: There is currently no common method to perform translation-back for 3D sign language. Existing metrics (such as DTW-MJE, FID) may not completely capture the generation quality.
2. **Substantial Gap in Spoken Language Generation**: The spoken language-level SLP has an FID of 4.359, which is significantly higher than the word-level FID of 0.756, indicating that mapping from natural language directly to sign language motions remains highly difficult.
3. **Exclusion of Lower Body in Evaluation**: Sign language primarily involves the upper body; thus, current evaluations overlook lower-body movements. However, in practical applications, standing posture and weight shift also affect naturalness.
4. **Residual Errors in Automatic Annotations**: While drastically superior to previous baselines, the automatic annotation pipeline may still yield inaccuracies under extreme self-occlusion or ultra-rapid motion.
5. **Future Directions**: Fusing 3D Sign Language Translation (SLT) and SLP to construct multimodal sign language frameworks, and developing large-scale sign language motion models tailored for AR/VR environments.

## Related Work & Insights

- **SGNify**: The only prior 3D sign language dataset (containing only 50 videos). SignAvatars scales this up by 1400x.
- **Ham2Pose**: A representative method mapping HamNoSys to 2D poses, which is adapted into a 3D version to serve as a baseline.
- **MDM (Motion Diffusion Model)**: A generic motion diffusion model, adapted to construct the SignDiffuse baseline.
- **SMPL-X / MANO**: Parametric body/hand models serving as the foundational representation of 3D sign language.
- **Insights**: This dataset establishes critical infrastructure for deaf digitized communications. The dual-codebook cross-modal alignment paradigm of VQ-VAE is highly worth exploring in a broader range of multimodal generation tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First large-scale 3D sign language holistic motion dataset, filling a critical research gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Reconstruction quality, SLP performance, and ablation studies are comprehensively discussed, though constrained by the maturity of current evaluation metrics.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation with highly detailed methodological documentation, though mathematical formulations are dense and some paragraphs are tightly packed.
- **Value**: ⭐⭐⭐⭐⭐ High technical and social value, offering critical infrastructure for digitized communication within the deaf and hard-of-hearing community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Text-Driven 3D Hand Motion Generation from Sign Language Data](../../CVPR2026/3d_vision/text-driven_3d_hand_motion_generation_from_sign_language_data.md)
- [\[ECCV 2024\] SegPoint: Segment Any Point Cloud via Large Language Model](segpoint_segment_any_point_cloud_via_large_language_model.md)
- [\[ECCV 2024\] Omni6D: Large-Vocabulary 3D Object Dataset for Category-Level 6D Object Pose Estimation](omni6d_large-vocabulary_3d_object_dataset_for_category-level_6d_object_pose_esti.md)
- [\[ECCV 2024\] Power Variable Projection for Initialization-Free Large-Scale Bundle Adjustment](power_variable_projection_for_initialization-free_large-scale_bundle_adjustment.md)
- [\[ECCV 2024\] TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos](tram_global_trajectory_and_motion_of_3d_humans_from_in-the-wild_videos.md)

</div>

<!-- RELATED:END -->
