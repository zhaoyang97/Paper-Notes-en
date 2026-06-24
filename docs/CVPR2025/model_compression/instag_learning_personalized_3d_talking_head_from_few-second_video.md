---
title: >-
  [Paper Note] InsTaG: Learning Personalized 3D Talking Head from Few-Second Video
description: >-
  [CVPR 2025][Model Compression][3D talking head] InsTaG is proposed to extract a general motion prior from multi-speaker long videos via Identity-Free Pre-training, and then rapidly learn high-fidelity personalized 3D talking heads from only a 5-second video via Motion-Aligned Adaptation, achieving 82.5 FPS real-time inference.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "3D talking head"
  - "3D Gaussian Splatting"
  - "few-shot adaptation"
  - "identity-free pre-training"
  - "motion alignment"
  - "real-time rendering"
date: 2026-05-08
content_hash: 8bfd18e868650c56
---

# InsTaG: Learning Personalized 3D Talking Head from Few-Second Video

**Conference**: CVPR 2025  
**arXiv**: [2502.20387](https://arxiv.org/abs/2502.20387)  
**Code**: [Project Page](https://fictionarry.github.io/InsTaG/)  
**Area**: Model Compression  
**Keywords**: 3D talking head, 3D Gaussian Splatting, few-shot adaptation, identity-free pre-training, motion alignment, real-time rendering

## TL;DR

InsTaG is proposed to extract a general motion prior from multi-speaker long videos via Identity-Free Pre-training, and then rapidly learn high-fidelity personalized 3D talking heads from only a 5-second video via Motion-Aligned Adaptation, achieving 82.5 FPS real-time inference.

## Background & Motivation

**Background**: 3D talking head synthesis based on radiance fields (NeRF/3DGS) can generate high-fidelity personalized videos, but person-specific methods require large amounts of high-quality video data and long training times to adapt to each new identity.

**Limitations of Prior Work**:
1. **High Data Requirements**: Existing person-specific methods (e.g., ER-NeRF, TalkingGaussian) require several minutes of long video training.
2. **Long Adaptation Time**: Training a new identity from scratch takes hours.
3. **Poor Quality of Few-Shot Methods**: Existing one-shot/few-shot methods (e.g., Real3DPortrait, MimicTalk) sacrifice personalization quality and inference efficiency for generalization.

**Key Challenge**: Achieving fast adaptation to new identities requires a general motion prior; however, person-specific architectures cannot be directly pre-trained on multi-person data, as the appearance and personalized motion of different identities conflict with one another.

**Key Insight**: Completely decoupling general motion learning and personalized adaptation—utilizing temporary personalized fields to "filter out" identity information during the pre-training stage, and employing a motion aligner to adapt pre-trained motion to new identities during the adaptation stage.

## Method

### Overall Architecture

InsTaG is based on a 3DGS-based person-specific synthesizer, adopting a face-mouth decomposition architecture (two branches: face and inner mouth), where each branch contains a structure field (static Gaussians) and a motion field (predicted deformation). The core pipeline is as follows:
1. **Pre-training Stage**: Learning a Universal Motion Field (UMF) from a multi-speaker long video corpus.
2. **Adaptation Stage**: Initializing a person-specific model for a new identity using a few seconds of video, and rapidly aligning it via a Motion Aligner and a Face-Mouth Hook.

### Key Designs

**1. Identity-Free Pre-training**
- **Function**: Extracting general motion priors from multi-speaker long videos into a shared Universal Motion Field (UMF).
- **Mechanism**: Assigning a temporary Personalized Field (private structure field $\theta_P^i$ + private minor motion field $\mathcal{D}_P^i$) to each training video to store identity appearance and personalized motion. The UMF predicts identity-agnostic general deformation $\delta_U$, while the Personalized Field predicts residual motion $\delta_P^i$. The two are superimposed for rendering: $\tilde{\theta}_P^i = \theta_P^i + \delta_U + \delta_P^i$.
- **Negative Contrast Loss**: Truncated dot product loss $\mathcal{L}_C = \mathbb{I}_{trunc}(\Delta\mu_P^i \cdot \Delta\mu_P^j)$, encouraging personalized motions of different identities to be as distinct as possible, thereby maximizing the filtering of general motion into the UMF.
- **Design Motivation**: Addressing the identity conflict issue of person-specific architectures on multi-person data; the truncated design of the contrastive loss avoids excessively pushing apart motions that are already sufficiently differentiated.

**2. Motion-Aligned Adaptation**
- **Motion Aligner**: Learning coordinate offsets $\Delta\mu_A$ and motion scale factors $\tau_A$ to resolve the discrepancy in implicit facial structures and motion scale differences between the new identity and the UMF. Before querying the UMF, the offset is applied: $\Delta\mu' = \Delta\mu \times \tau_A$, where $\Delta\mu \in \mathcal{U}(\mu + \Delta\mu_A, \mathbf{C})$.
- **Face-Mouth Hook**: Taking the maximum/minimum $\Delta\mu$ from the deformation of the face branch as lip motion cues, which are input to the mouth motion field to guide the alignment of the inner mouth motion. The scale factor $\tau_M$ is controlled by the lip motion distance $\mu_{dist}$, ensuring the mouth opening synchronizes with the lips.
- **Design Motivation**: Directly driving Gaussian deformation using the pre-trained UMF (instead of guiding image generation) requires precise geometric alignment; the Face-Mouth Hook addresses the uncoordinated face-mouth motion under sparse data.

**3. Geometry Prior Regularizer**
- **Function**: Leveraging depth and normal priors provided by a geometry estimator to regularize the geometric degradation of 3DGS under sparse viewpoints.
- **Mechanism**: $\mathcal{L}_{Geo} = \lambda_D L_D(D, \check{D}) + \lambda_N \sum(1 - N_{i,j} \cdot \check{N}_{i,j})$
- **Design Motivation**: Insufficient viewpoint coverage in sparse training data leads to geometric ambiguity; monocular depth and normal estimation provide additional constraints.

### Loss & Training

- **Pre-training**: $\mathcal{L}_{pre}^i = \mathcal{L}_I(I^i, I_{GT}^i) + \lambda_C \sum_{j \neq i} \mathcal{L}_C$, $\lambda_C = 1$, 5 long videos, 150K iterations.
- **Adaptation**: $\mathcal{L}_{ada} = \mathcal{L}_I + \mathcal{L}_{Geo}$, starting with warm-up (without Motion Aligner), followed by training with the full loss, 10K iterations.
- Image loss $\mathcal{L}_I$ = L1 + D-SSIM, optimized via AdamW.

## Key Experimental Results

### Main Results (5-second Video Self-Reconstruction)

| Method | Type | PSNR(A/F)↑ | LPIPS(A/F)↓ | LMD↓ | Sync-C↑ | Training Time↓ | FPS↑ | Real-time |
|---|---|---|---|---|---|---|---|---|
| ER-NeRF | From Scratch | 28.23/25.63 | 0.040/0.031 | 3.541 | 3.074 | 2h | 30.8 | ✓ |
| TalkingGaussian | From Scratch | 28.32/26.01 | 0.041/0.028 | 3.588 | 3.556 | 31min | 114.2 | ✓ |
| MimicTalk | Adaptation | 24.69/26.27 | 0.075/0.031 | 3.489 | 6.926 | 16min | 8.2 | ✗ |
| **InsTaG** | **Adaptation** | **28.86/26.32** | **0.039/0.026** | **3.167** | 5.318 | 13min | **82.5** | ✓ |

### Comparison of Different Data Volumes

| Method | 5s PSNR↑ | 10s PSNR↑ | 20s PSNR↑ |
|---|---|---|---|
| TalkingGaussian | 28.321 | 29.130 | 29.536 |
| MimicTalk | 24.69 | - | - |
| **InsTaG** | **28.86** | **29.45** | **29.82** |

### Key Findings

1. **Outperforming Scratch-trained Methods with 5-second Video**: With a 5-second video + 13-minute adaptation, InsTaG matches or even outperforms ER-NeRF/TalkingGaussian trained from scratch for hours on full videos.
2. **Simultaneous Lip-Sync and Rendering Quality**: Achieving the optimal LMD (lip motion distance) of 3.167 alongside the highest PSNR, validating the effectiveness of the personalized motion prior.
3. **Real-time Inference**: 82.5 FPS, significantly outperforming MimicTalk (8.2 FPS), and is close to TalkingGaussian trained from scratch (114.2 FPS) while adapting much faster.
4. **Cross-Identity/Gender/Language Generalization**: Experiments validate robustness across various identities, genders, and languages.

## Highlights & Insights

- Identity-Free Pre-training elegantly achieves multi-person pre-training on a person-specific architecture; the core insight lies in utilizing temporary personalized fields to "absorb" identity-specific information.
- The truncated design of the Negative Contrast Loss is ingenious—only penalizing personalized motions with identical directions, without pushing away those that are already sufficiently distinguished.
- The Face-Mouth Hook simply and effectively resolves the lack of coordination in face-mouth decomposed architectures under sparse data conditions.
- The overall architecture is compact and consistent, seamlessly connecting pre-training and adaptation without requiring auxiliary 2D-to-3D modules.

## Limitations & Future Work

- Pre-training utilizes only 5 long videos; expanding the pre-training corpus could further enhance generalization.
- The adaptation phase still requires approximately 13 minutes, remaining a step away from "instant" adaptation.
- Modeling of body parts beyond the talking head is not addressed.
- The Sync-C (audio-visual synchronization) metric is lower than that of MimicTalk, potentially due to MimicTalk's larger backbone network.
- The face-mouth decomposition assumes a fixed structure, which may be constrained under extreme expressions such as wide-open mouths.

## Related Work & Insights

- The face-mouth decomposition from TalkingGaussian is inherited and enhanced via the hook mechanism in this work.
- Unlike MimicTalk, which injects pre-trained speech models using LoRA, InsTaG directly pre-trains the motion field, ensuring greater compactness and efficiency.
- The paradigm of motion prior decoupling and alignment is generalizable to other person-specific 3D reconstruction tasks.

## Rating

⭐⭐⭐⭐ — Clear technical pipeline. The design of Identity-Free Pre-training is novel and elegant. Experiments fully demonstrate superior performance and efficiency under extremely limited data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Logits DeConfusion with CLIP for Few-Shot Learning](logits_deconfusion_with_clip_for_few-shot_learning.md)
- [\[CVPR 2025\] Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning](tripartite_weight-space_ensemble_for_few-shot_class-incremental_learning.md)
- [\[CVPR 2025\] MuTri: Multi-view Tri-alignment for OCT to OCTA 3D Image Translation](mutri_multi-view_tri-alignment_for_oct_to_octa_3d_image_translation.md)
- [\[CVPR 2026\] Adaptive Video Distillation: Mitigating Oversaturation and Temporal Collapse in Few-Step Generation](../../CVPR2026/model_compression/adaptive_video_distillation_mitigating_oversaturation_and_temporal_collapse_in_f.md)
- [\[CVPR 2025\] Plug-and-Play Versatile Compressed Video Enhancement](plug-and-play_versatile_compressed_video_enhancement.md)

</div>

<!-- RELATED:END -->
