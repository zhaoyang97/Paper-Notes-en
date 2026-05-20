---
title: >-
  [Paper Note] Controllable and Expressive One-Shot Video Head Swapping
description: >-
  [ICCV 2025][Human Understanding][head swapping] This paper proposes a diffusion-based multi-condition controllable video head swapping framework (SwapAnyHead) that achieves high-fidelity identity preservation…
tags:
  - "ICCV 2025"
  - "Human Understanding"
  - "head swapping"
  - "diffusion model"
  - "expression transfer"
  - "identity preservation"
  - "video generation"
date: 2026-05-08
content_hash: d4a2ac75d4d76022
---

# Controllable and Expressive One-Shot Video Head Swapping

**Conference**: ICCV 2025
**arXiv**: [2506.16852](https://arxiv.org/abs/2506.16852)  
**Code**: [https://humanaigc.github.io/SwapAnyHead/](https://humanaigc.github.io/SwapAnyHead/)  
**Area**: Human Understanding / Face Generation
**Keywords**: head swapping, diffusion model, expression transfer, identity preservation, video generation

## TL;DR

This paper proposes a diffusion-based multi-condition controllable video head swapping framework (SwapAnyHead) that achieves high-fidelity identity preservation, seamless background blending, and accurate cross-identity expression transfer and editing via a shape-agnostic mask strategy, a hair enhancement strategy, and an expression-aware 3DMM-driven landmark retargeting module.

## Background & Motivation

Video head swapping—seamlessly replacing the head from a source image onto a target video—holds significant application potential in film production, virtual reality, and advertising synthesis. However, existing methods face two core challenges:

- **Incomplete identity preservation**: Face swapping methods (e.g., SimSwap, DiffSwap) replace only the local facial region, ignoring head shape and hairstyle; head swapping methods (e.g., HeSer) perform poorly in hairstyle diversity and complex background handling.
- **Imprecise expression transfer**: IPAdapter/DreamBooth-based methods lack expression control; AniPortrait relies on 3DMM templates with limited expressiveness; Follow-Your-Emoji uses 3D landmarks but remains influenced by the source image's original expression, and cross-identity facial proportion differences cause expression distortion.

The root cause is the need to simultaneously preserve the complete identity of the source head (head shape + hairstyle) while achieving precise expression transfer and seamlessly integrating the background and body of the target video. The paper addresses this by reformulating head swapping as a **conditional inpainting task**, using identity features, background cues, and 3D landmarks as conditions within a unified latent diffusion paradigm.

## Method

### Overall Architecture

Built upon a Latent Diffusion Model (LDM) with extended background and expression conditioning. The framework comprises four modules:
1. **AppearanceNet**: Extracts source image identity information and injects it into the denoising network via self-attention.
2. **PoseGuider**: Extracts multi-scale motion features to establish spatial correspondence between control signals and generated images.
3. **MotionModule**: Maintains temporal consistency across frames.
4. **Image Encoder**: Replaces the text encoder, providing global reference image information via cross-attention.

The training objective is the standard diffusion loss augmented with an identity loss: $\mathcal{L} = \mathcal{L}_{LDM} + \mathcal{L}_{id}$

### Key Designs

1. **Shape-Agnostic Mask Strategy**:

    - **Function**: Eliminates shape leakage caused by head segmentation boundaries in training data.
    - **Mechanism**: The foreground mask $M_f$ is dilated and partitioned into $(k_h \times k_w)$ non-overlapping blocks; each block is uniformly set to 0 or 1 depending on whether it contains foreground pixels, thereby destroying precise segmentation boundaries. During training, the foreground is randomly scaled and recombined with the inpainted background: $M_f^{new}(i,j) = \begin{cases} 1, & \text{if any}(M_f(i,j))>0 \\ 0, & \text{if every}(M_f(i,j))=0 \end{cases}$
    - **Design Motivation**: In training data, the reference and target images are different frames from the same video (same identity), so the inpainted background implicitly encodes head shape information (perceptible to the SD network but not to the human eye), constraining generation to the original head region and limiting hairstyle diversity.

2. **Hair Enhancement Strategy**:

    - **Function**: Eliminates hairstyle prior bias in the shoulder region, enabling diverse hairstyle generation.
    - **Mechanism**: A collection of long-hair videos is gathered; during training, a long-hair image is randomly selected to obtain a hair mask $M_{hair}$, and shoulder keypoint detection generates a rectangular mask $M_{rect}$ to erode the original body region: $M_{cloth}^{new} = M_{cloth} \odot (1-M_{hair}) \odot (1-M_{rect})$
    - **Design Motivation**: When long hair rests on the shoulders, clothing segmentation leaves holes on the shoulder region that reflect the hairstyle. During training this introduces the target image's hairstyle prior, causing the model to generate long hair in those holes even when the source image has short hair.

3. **Expression-Aware Landmark Retargeting**:

    - **Function**: Enables precise cross-identity expression transfer while eliminating the influence of the source image's original expression.
    - **Mechanism**: A two-step process — (a) **Neu-Exp Module**: fits a 3DMM to the source image, zeros out the expression coefficients to obtain a neutral expression template, and computes neutral landmarks: $L_{ref}^{neu} = L_{ref} - L_{ref}^t + L_{ref}^{t,exp=0}$; (b) **Scale-aware Retargeting**: computes scale factors $s_{eye}, s_{mouth}$ based on the ratio of facial feature sizes (eyes/mouth) between the source and driving images, adaptively adjusting expression deltas.
    - **Design Motivation**: 3DMM templates effectively disentangle identity, expression, and pose but have limited expressiveness; Mediapipe 3D landmarks are highly expressive but cannot remove the source expression. Combining both approaches leverages the strengths of each. Scale-aware adaptation resolves expression exaggeration or understatement caused by cross-identity facial proportion differences.

### Loss & Training

- **Diffusion Loss** $\mathcal{L}_{LDM}$: Standard denoising objective.
- **Identity Loss** $\mathcal{L}_{id}$: Recovers the denoised image from predicted noise; computes L2 pixel loss over the head region plus cosine similarity loss on ArcFace identity embeddings.
- **Training Data**: HDTF, VoxCeleb, VFHQ, TalkingHead1KH — approximately 30K video clips and ~20K identities.
- Image resolution: 512×512; Mediapipe is used to extract 3D landmarks; LaMa is used for background inpainting.

## Key Experimental Results

### Main Results

| Method | Type | ID Similarity ↑ | Pose Error ↓ | Expression Error ↓ |
|--------|------|-----------------|--------------|-------------------|
| AniPortrait | Portrait Animation | 0.892 | 6.55 | 0.025 |
| Follow-Your-Emoji | Portrait Animation | 0.906 | 16.8 | 0.026 |
| SimSwap | Face Swap | 0.592 | 0.702 | 0.005 |
| DiffSwap | Face Swap | 0.397 | 1.144 | 0.008 |
| HeSer | Head Swap | 0.319 | 6.840 | 0.022 |
| **SwapAnyHead (Ours)** | **Head Swap** | **0.895** | **9.83** | **0.014** |

Evaluation uses 200 VFHQ videos and 200 portrait images (different identities), 48 frames per video. The proposed method substantially outperforms HeSer in ID similarity among head swapping methods (0.895 vs. 0.319) and achieves the lowest expression error among all comparable methods.

### Ablation Study

**Expression-Aware Landmark Retargeting Ablation:**

| Configuration | Expression Error ↓ | Note |
|---------------|--------------------|------|
| w/o Neu-Exp Module | 0.025 | Source expression superimposed on driving expression, e.g., residual smile |
| w/o Scale-aware Retargeting | 0.015 | Facial proportion mismatch causes closed eyes / exaggerated mouth |
| Full method | 0.014 | Accurate expression transfer |

**Qualitative Ablation of Shape-Agnostic Mask and Hair Enhancement:**
- Without Shape-Agnostic Mask: generated head is constrained by the driving video's head shape, resulting in size mismatch.
- Without Hair Enhancement: incorrect hairstyle artifacts appear on the shoulder, generating long hair even from a short-hair reference image.

### Key Findings
- Face swapping methods (SimSwap/DiffSwap) exhibit the lowest pose and expression errors because they only perform fine-grained modifications rather than regenerating the entire head.
- Portrait animation methods naturally achieve higher identity preservation as they do not need to blend a new background or body.
- Shape leakage is more severe than expected — the SD network can perceive segmentation boundary information that is invisible to the human eye.

## Highlights & Insights
- **Precise problem formulation**: The paper clearly distinguishes between "face swapping" and "head swapping," identifying the fundamental limitations of existing methods in preserving head shape and hairstyle.
- **Elegant Shape-Agnostic Mask strategy**: Destroying segmentation boundaries via block-wise discretization to prevent shape leakage is a simple yet highly effective idea.
- **3DMM + Mediapipe combination**: Leverages the disentanglement capability of 3DMM and the expressiveness of Mediapipe to overcome the limitations of either approach alone.
- Supports expression editing (beyond mere transfer), offering greater flexibility for practical film post-production workflows.

## Limitations & Future Work
- Quantitative evaluation metrics are limited; image quality metrics such as FID and LPIPS are absent.
- Body–hair interactions below the shoulder may still produce artifacts under extreme poses.
- Reliability depends on Mediapipe stability, which may degrade under extreme viewpoints or occlusion.
- Training data is predominantly frontal or semi-profile; generalization to large-angle profile views is not validated.
- Quantitative evaluation of video temporal consistency (e.g., inter-frame flickering metrics) is missing.

## Related Work & Insights
- **SimSwap / FaceShifter**: Face swapping methods that replace only the facial region without altering head shape or hairstyle.
- **HeSer**: A two-stage head swapping method with insufficient hairstyle diversity and background inpainting capability.
- **Follow-Your-Emoji**: Driven by Mediapipe 3D landmarks with high expressiveness, but suffers from residual source expression.
- **AniPortrait**: 3DMM-driven portrait animation with good disentanglement but expressiveness limited by the 3DMM.
- **HS-Diffusion**: Semantic-mixing diffusion model for head swapping, but incapable of expression transfer.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The shape-agnostic mask and hair enhancement strategies represent novel and distinctive training data augmentation approaches.
- **Experimental Thoroughness**: ⭐⭐⭐ Quantitative metrics are somewhat limited, but qualitative results and ablation studies are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Problem analysis is thorough, method motivation is clearly articulated, and illustrations are rich.
- **Value**: ⭐⭐⭐⭐ Video head swapping addresses a task with strong practical demand; the method demonstrates high applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] One-Shot Knowledge Transfer for Scalable Person Re-Identification](one-shot_knowledge_transfer_for_scalable_person_re-identification.md)
- [\[ICCV 2025\] DreamActor-M1: Holistic, Expressive and Robust Human Image Animation with Hybrid Guidance](dreamactor-m1_holistic_expressive_and_robust_human_image_animation_with_hybrid_g.md)
- [\[ICCV 2025\] ImHead: A Large-scale Implicit Morphable Model for Localized Head Modeling](imhead_a_large-scale_implicit_morphable_model_for_localized_head_modeling.md)
- [\[ICCV 2025\] GGTalker: Talking Head Synthesis with Generalizable Gaussian Priors and Identity-Specific Adaptation](ggtalker_talking_head_systhesis_with_generalizable_gaussian_priors_and_identity-.md)
- [\[ICCV 2025\] Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars](avat3r_large_animatable_gaussian_reconstruction_model_for_hi.md)

</div>

<!-- RELATED:END -->
