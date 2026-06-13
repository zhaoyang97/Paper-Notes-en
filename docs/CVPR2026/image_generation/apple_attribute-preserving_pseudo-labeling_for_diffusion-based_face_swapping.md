---
title: >-
  [Paper Note] APPLE: Attribute-Preserving Pseudo-Labeling for Diffusion-Based Face Swapping
description: >-
  [CVPR 2026][Image Generation][Face Swapping] APPLE proposes a teacher-student framework based on diffusion models. It trains a teacher model using conditional deblurring (instead of traditional conditional inpainting) to…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Face Swapping"
  - "diffusion model"
  - "Teacher-Student"
  - "Pseudo-Label"
  - "Attribute Preservation"
date: 2026-05-08
content_hash: 66c6ebec1f38a236
---

# APPLE: Attribute-Preserving Pseudo-Labeling for Diffusion-Based Face Swapping

**Conference**: CVPR 2026  
**arXiv**: [2601.15288](https://arxiv.org/abs/2601.15288)  
**Code**: [https://cvlab-kaist.github.io/APPLE](https://cvlab-kaist.github.io/APPLE)  
**Area**: Image Generation / Face Swapping  
**Keywords**: Face Swapping, diffusion model, Teacher-Student, Pseudo-Label, Attribute Preservation

## TL;DR
APPLE proposes a teacher-student framework based on diffusion models. It trains a teacher model using conditional deblurring (instead of traditional conditional inpainting) to generate attribute-aligned pseudo-labels, which are then used to train a student model. This achieves SOTA performance in attribute preservation (FID 2.18, Pose Error 1.85) while maintaining identity transfer capabilities.

## Background & Motivation

**Background**: Face swapping aims to transfer the identity of a source image to a target image while preserving the target's attributes such as pose, expression, skin tone, lighting, and makeup. This technology is widely used in content creation, privacy protection, and filmmaking.

**Limitations of Prior Work**: Early GAN-based methods (SimSwap, HiFiFace, FaceDancer, etc.) rely on conflicting objectives—identity loss and reconstruction loss—leading to unstable training and frequent copy-paste style artifacts.

**Root Cause in Diffusion Methods**: Recent diffusion methods (DiffSwap, FaceAdapter, REFace) model the task as conditional inpainting, where the target face area is masked and then reconstructed. However, **the masking operation removes identity information while simultaneously losing crucial attribute cues** (lighting, skin tone, makeup, etc.). Consequently, even with auxiliary conditional information, the model fails to faithfully preserve these attributes.

**Key Insight**: The core insight is that **the key to attribute preservation lies not in better attribute encoding, but in providing high-quality, attribute-aligned pseudo-labels as conditional inputs for the student model**. If the teacher can generate attribute-consistent pseudo-labels, the student can learn from clean images (rather than degraded masked images), achieving superior attribute preservation.

**Core Idea**: Replace conditional inpainting with conditional deblurring to train the teacher model, combined with an attribute-aware inversion scheme to generate high-quality pseudo-labels. The student model is then trained using these labels to achieve a win-win for both attribute preservation and identity transfer.

## Method

### Overall Architecture
APPLE is a **teacher-student framework** divided into three stages:
- **Teacher Training**: Train the diffusion teacher model using a conditional deblurring objective.
- **Pseudo-Label Generation**: The teacher generates attribute-aligned pseudo-labels through attribute-aware inversion.
- **Student Training**: The student is trained under a direct editing objective, conditioned on the pseudo-labels.

The base architecture utilizes FLUX.1-Krea [dev] as the diffusion backbone, PulID as the identity encoder, and OminiControl as the attribute conditional branch (LoRA rank=64).

### Key Designs

1. **Conditional Deblurring instead of Conditional Inpainting**:

    - **Traditional Approach**: Mask the target face area to zero and train the model to reconstruct the original image from the masked image.
    - **Ours**: Replace the target face area with a **blurred version** (downsampled to $8 \times 8$ and upsampled back to original size), removing high-frequency identity details while preserving low-frequency attribute cues (skin tone, lighting, pose, etc.).
    - **Design Motivation**: Masking completely eliminates attribute information, whereas blurring preserves low-frequency signals. This allows the model to infer lighting and tones, significantly improving attribute preservation. Blurring is applied only to the face area using a face parsing model, leaving the background untouched.

2. **Attribute-Aware Inversion**:

    - **Problem**: While the deblurring strategy preserves global attributes, fine-grained attributes like makeup and accessories are still difficult to recover from blurred information.
    - **Mechanism**: Leverage the characteristic that noise obtained via diffusion inversion is not perfectly Gaussian—residual semantic information from the input image remains in the inversion noise. APPLE **intentionally exploits** this residual information to anchor fine-grained attributes.
    - **Key Choice**: During inversion, only the **attribute condition** ($\varnothing, \mathcal{F}_{att}(I)$) is used, rather than the full condition. Full-condition inversion leaves residual identity information in the noise, leading to artifacts, whereas attribute-only conditions embed meaningful attribute cues while avoiding identity bias.
    - **Validation**: PCA visualization reveals that noise from attribute-conditioned inversion exhibits clear facial semantic structures; ablation studies of four conditional configurations confirm that the attribute condition is optimal.

3. **Student Model Training (Pseudo-Triplet Learning)**:

    - The teacher performs a face swap on a target image $I_{tgt}^A$ which has identity A, changing it to identity B to generate a pseudo-label $\hat{I}_{tgt}^{A \to B}$.
    - A pseudo-triplet $(I_{src}^A, \hat{I}_{tgt}^{A \to B}, I_{tgt}^A)$ is constructed.
    - The student takes the source identity features + pseudo-label attribute features as input to reconstruct the original target image.
    - **Core Advantage**: The student receives **clean pseudo-label images** rather than degraded masked images, enabling more effective learning of attribute preservation; no auxiliary networks or complex preprocessing are required during inference.

### Loss & Training
- **Overall Training Objective**: $\mathcal{L}_{total} = \mathcal{L}_{flow} + \lambda_{id} \mathcal{L}_{id}$
- **Flow Matching Loss** (Rectified Flow): $\mathcal{L}_{flow} = \mathbb{E}[\|(\epsilon - I_{tgt}) - v_t(z_t, \mathbf{id}_{src}, \mathbf{att}_{tgt})\|^2]$
- **Identity Loss**: $\mathcal{L}_{id} = 1 - \cos(\mathcal{F}_{id}(\hat{x_0}(z_t)), \mathcal{F}_{id}(I_{src}))$
- The teacher is trained for 15K steps (without identity loss) + 50K steps (with identity loss), and the student resumes training from the teacher for 15K steps.
- Effective batch size 16, using 4x A6000 GPUs.

## Key Experimental Results

### Main Results

| Method | FID↓ | ID Sim.↑ | ID Ret. Top-1↑ | Pose↓ | Expr.↓ |
|------|------|----------|---------------|-------|--------|
| SimSwap | 18.54 | 0.55 | 94.10 | 3.11 | 1.73 |
| FaceDancer | 3.80 | 0.51 | 89.70 | 2.23 | 0.74 |
| REFace | 7.22 | **0.60** | **97.60** | 3.67 | 1.08 |
| CSCS | 11.00 | **0.65** | **99.00** | 3.64 | 1.44 |
| APPLE (Teacher) | 3.68 | 0.54 | 90.40 | 2.07 | 0.70 |
| **APPLE (Student)** | **2.18** | 0.54 | 90.50 | **1.85** | **0.64** |

### Ablation Study

| Configuration | FID↓ | ID Sim.↑ | Pose↓ | Expr.↓ | Description |
|------|------|----------|-------|--------|------|
| Inpainting (Baseline) | 11.00 | 0.54 | 3.37 | 1.01 | Traditional masking scheme |
| Deblurring | 4.20 | 0.53 | 2.58 | 0.79 | Deblurring significantly improves attribute preservation |
| Deblurring + Inv. | 3.68 | 0.54 | 2.07 | 0.70 | Attribute-aware inversion provides further gains |

### Key Findings
- Switching from Inpainting to Deblurring reduced FID from 11.00 to 4.20 and Pose error from 3.37 to 2.58.
- Attribute-aware inversion further reduced Pose error from 2.58 to 2.07.
- The student model eventually outperformed the teacher (FID 2.18 vs 3.68), validating the effectiveness of the pseudo-label training strategy.
- While CSCS and REFace achieve higher identity similarity, they are heavily biased toward identity matching and exhibit poor attribute preservation (copy-paste artifacts).

## Highlights & Insights
- **Conditional deblurring is an elegant compromise**: It preserves more information than masking without introducing identity leakage, making it simple yet effective.
- **Intentionally leveraging the non-Gaussian nature of inversion noise**: This is highly ingenious—previous methods tried to eliminate residual semantics in inversion noise, whereas APPLE does the opposite, using it to preserve attributes.
- **Universal value of the Teacher-Student paradigm**: The core idea of this framework (generating good pseudo-labels to train a better model) is applicable to other conditional generation tasks.

## Limitations & Future Work
- Identity similarity metrics are slightly lower than the strongest baselines (0.54 vs 0.65), suggesting it may be insufficient in extreme identity transfer scenarios.
- Relies on the VGGFace2-HQ dataset for training; generalization to extreme cases like non-frontal views or occlusions remains to be verified.
- The quality of the teacher model's pseudo-labels is the bottleneck of the entire pipeline; further improvements to teacher quality could yield even greater benefits.

## Related Work & Insights
- DreamID also uses pseudo-datasets for training but relies on a GAN model (FaceDancer) to generate pseudo-labels, which has limited quality.
- The concept of attribute-aware inversion can be extended to other image editing tasks—using inversion noise to preserve specific attributes.
- The conditional deblurring strategy may inspire other conditional generation tasks requiring attribute preservation, such as virtual try-on or style transfer.

## Rating
- Novelty: ⭐⭐⭐⭐ Both conditional deblurring and attribute-aware inversion are novel, though the teacher-student framework itself is common.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Rigorous argumentation with multi-dimensional quantitative evaluation and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic with a complete chain of derivation for motivation and methods.
- Value: ⭐⭐⭐⭐ High practicality as attribute preservation is a core challenge in face swapping.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Preserving Source Video Realism: High-Fidelity Face Swapping for Cinematic Quality](preserving_source_video_realism_high-fidelity_face_swapping_for_cinematic_qualit.md)
- [\[CVPR 2026\] High-Fidelity Diffusion Face Swapping with ID-Constrained Facial Conditioning](high-fidelity_diffusion_face_swapping_with_id-constrained_facial_conditioning.md)
- [\[CVPR 2026\] Guiding a Diffusion Model by Swapping Its Tokens](guiding_a_diffusion_model_by_swapping_its_tokens.md)
- [\[CVPR 2026\] All-in-One Slider for Attribute Manipulation in Diffusion Models](all_in_one_slider_attribute_manipulation.md)
- [\[ICCV 2025\] NullSwap: Proactive Identity Cloaking Against Deepfake Face Swapping](../../ICCV2025/image_generation/nullswap_proactive_identity_cloaking_against_deepfake_face_swapping.md)

</div>

<!-- RELATED:END -->
