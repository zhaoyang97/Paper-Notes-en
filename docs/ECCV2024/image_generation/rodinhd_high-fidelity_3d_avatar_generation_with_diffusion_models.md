---
title: >-
  [Paper Note] RodinHD: High-Fidelity 3D Avatar Generation with Diffusion Models
description: >-
  [ECCV 2024][Image Generation][3D Avatar Generation] RodinHD is proposed to address the catastrophic forgetting problem of the triplane decoder and achieve high-fidelity 3D avatar generation through hierarchical portrait representation injection.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "3D Avatar Generation"
  - "High Fidelity"
  - "Diffusion Models"
  - "Triplane"
  - "Catastrophic Forgetting"
date: 2026-05-08
content_hash: 285f369ae881f248
---

# RodinHD: High-Fidelity 3D Avatar Generation with Diffusion Models

**Conference**: ECCV 2024  
**arXiv**: [2407.06938](https://arxiv.org/abs/2407.06938)  
**Code**: Yes (Project Page)  
**Area**: Image Generation  
**Keywords**: 3D Avatar Generation, High Fidelity, Diffusion Models, Triplane, Catastrophic Forgetting

## TL;DR

RodinHD is proposed to address the catastrophic forgetting problem of the triplane decoder and achieve high-fidelity 3D avatar generation through hierarchical portrait representation injection.

## Background & Motivation

### Key Challenge

**Key Challenge**: **Background**: Generating high-fidelity 3D avatars from a single portrait image is a crucial problem at the intersection of computer graphics and computer vision. High-quality 3D avatars have a broad range of applications in virtual reality, social media, gaming, and teleconferencing.

Existing methods (such as Rodin, DreamFusion, etc.) utilize diffusion models to generate 3D representations (e.g., triplanes or NeRFs), but still perform unsatisfactorily in terms of generating details, particularly delicate structures like hairstyles. The authors identify an overlooked yet critical issue — **catastrophic forgetting**.

Specifically, triplane-based 3D generation methods generally employ a shared MLP decoder to decode triplane features into color and density. When sequentially trained (fitted) on a large number of different avatars, the MLP decoder forgets previously learned knowledge, resulting in degraded rendering quality. This resembles the catastrophic forgetting problem in continual learning but has not been sufficiently recognized in the 3D generation field.

Furthermore, when utilizing the input portrait image to guide 3D generation, existing methods typically extract only global features (such as CLIP embeddings), thereby ignoring rich 2D texture details.

## Method

### Overall Architecture

RodinHD improves upon the following pipeline: (1) extracting multi-scale, hierarchical visual representations from the portrait image; (2) generating 3D avatars in the triplane space using a 3D diffusion model; (3) rendering the triplane into the final image via an improved MLP decoder. Core improvements focus on addressing the forgetting in the decoder and the injection of portrait features.

### Key Designs

1. **Data Scheduling + Weight Consolidation**:
    - Function: Addresses the catastrophic forgetting issue of the MLP decoder during sequential fitting.
    - Mechanism: (a) Data scheduling strategy: Instead of processing each avatar sequentially, a dynamic training data scheduling scheme is designed to mix training data from different avatars, preventing the model from overfitting on a single avatar and forgetting others. (b) Weight consolidation regularization (similar to EWC): While updating decoder parameters, regularization constraints are imposed on important parameters to restrict their variation range.
    - Design Motivation: Standard sequential fitting causes decoder parameters to drift on new avatars; regularization and data scheduling effectively mitigate this problem.

2. **Hierarchical Portrait Representation Injection**:
    - Function: Fully utilizes input portrait images to guide 3D generation.
    - Mechanism: A multi-scale feature pyramid (ranging from shallow low-level texture features to deep high-level semantic features) is extracted from the input portrait image, and then injected into different levels of the 3D diffusion model via multi-layer cross-attention. Shallow features provide texture details, while deep features offer structural guidance.
    - Design Motivation: A single global feature fails to convey rich details; multi-layer injection enables the 3D generation process to capture finer 2D cues.

3. **Optimized Noise Schedule for Triplanes**:
    - Function: Improves training of the diffusion model on triplane data.
    - Mechanism: The noise scheduling scheme of the diffusion process is adjusted according to the signal-to-noise ratio characteristics of triplane data. Since the data distribution of triplanes differs from natural images, the standard noise schedule might not be optimal. A more suitable noise schedule is designed by analyzing the frequency characteristics of triplanes.
    - Design Motivation: The standard noise schedule is designed for natural images; applying it directly to triplanes can lead to training instability or degraded quality.

### Loss & Training

- Diffusion training loss: Training the denoising network in the triplane space.
- Rendering loss: Rendering the triplane into a 2D image and calculating pixel-level and perceptual losses against the ground-truth image.
- Weight consolidation regularization loss: Constraining updates of important parameters in the MLP decoder.
- Training scale: Trained on 46K avatar datasets.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA (Rodin) | Gain |
|--------|------|------|----------|------|
| Synthetic Avatars | FID ↓ | Significantly Better | Rodin | Obvious Improvement |
| Real Portraits | Visual Quality | Sharper | Rodin | Rich in Details |
| Hairstyle Details | Visual Quality | Substantially Improved | Other methods | Most Outstanding Improvement |
| Generalization | In-the-wild | Good | Limited | Better Generalization |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| w/o Data Scheduling | Severe Forgetting | Good quality on new avatars but degradation on old ones |
| w/o Weight Consolidation | Partial Forgetting | Parameter drift leads to quality degradation |
| Single-level Feature Injection | Insufficiency in Details | Lacks multi-scale information |
| Standard Noise Schedule | Unstable Training | Mismatched triplane characteristics |
| Full RodinHD | Optimal | All improvements are complementary |

### Key Findings

- Catastrophic forgetting is an overlooked but crucial problem in triplane-based methods.
- The combination of data scheduling and weight consolidation effectively alleviates forgetting, significantly improving rendering sharpness.
- Hierarchical feature injection is the key to improving detail quality (especially for hairstyles).
- Trained on 46K avatars, the model generalizes well to in-the-wild portrait inputs.

## Highlights & Insights

- Identifies and resolves the overlooked issue of catastrophic forgetting in triplane generation, a finding that possesses general applicability.
- Intuitively migrates cross-domain methodologies by introducing continual learning solutions (EWC/data scheduling) to the 3D generation field.
- The design of hierarchical feature injection is intuitive and effective.
- Achieves remarkable progress in generating fine structures such as hairstyles.

## Limitations & Future Work

- The method focuses on avatar/bust generation, and its applicability to full-body 3D humans requires further verification.
- The data requirement of 46K avatars for training is substantial; acquiring such data remains a challenge.
- The resolution of the triplane representation limits the upper bound of the ultimate detail quality.
- Emerging representation methods such as 3D Gaussian Splatting could be incorporated in the future.
- Generating dynamic avatars (with expressions and actions) is an important direction for future extensions.

## Related Work & Insights

- **Rodin**: The predecessor of RodinHD, which pioneered using diffusion models to generate 3D avatar triplanes.
- **DreamFusion / Zero-1-to-3**: Representative methods for 2D-to-3D generation.
- **EWC**: Elastic Weight Consolidation, a classic regularization method in continual learning.
- Insight: Training stability in 3D generation methods deserves broader attention, indicating that continual learning techniques have valuable applications in this domain.

## Rating

- Novelty: ⭐⭐⭐⭐ Significant contribution by identifying the catastrophic forgetting issue with a practical solution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Thorough large-scale training and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ In-depth problem analysis and clear method descriptions.
- Value: ⭐⭐⭐⭐ Provides a practical boost to high-fidelity 3D avatar generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] EMDM: Efficient Motion Diffusion Model for Fast and High-Quality Motion Generation](emdm_efficient_motion_diffusion_model_for_fast_and_high-quality_motion_generatio.md)
- [\[ECCV 2024\] NeuSDFusion: A Spatial-Aware Generative Model for 3D Shape Completion, Reconstruction, and Generation](neusdfusion_a_spatial-aware_generative_model_for_3d_shape_completion_reconstruct.md)
- [\[ECCV 2024\] A High-Quality Robust Diffusion Framework for Corrupted Dataset](a_highquality_robust_diffusion_framework_for_corrupted_datas.md)
- [\[ICCV 2025\] TeRA: Rethinking Text-guided Realistic 3D Avatar Generation](../../ICCV2025/image_generation/tera_rethinking_text-guided_realistic_3d_avatar_generation.md)
- [\[ICCV 2025\] FaceCraft4D: Animated 3D Facial Avatar Generation from a Single Image](../../ICCV2025/image_generation/facecraft4d_animated_3d_facial_avatar_generation_from_a_single_image.md)

</div>

<!-- RELATED:END -->
