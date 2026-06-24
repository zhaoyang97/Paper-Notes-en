---
title: >-
  [Paper Note] MotionCharacter: Fine-Grained Motion Controllable Human Video Generation
description: >-
  [AAAI 2026][Video Generation][human video generation] This paper proposes the MotionCharacter framework, which decouples motion into two independently controllable dimensions—action type and motion intensity—to achieve fine-grained motion control and identity consistency in high-fidelity human video generation.
tags:
  - "AAAI 2026"
  - "Video Generation"
  - "human video generation"
  - "motion control"
  - "identity preservation"
  - "optical flow"
  - "diffusion models"
date: 2026-05-08
content_hash: 3e2ac3b242ed5018
---

# MotionCharacter: Fine-Grained Motion Controllable Human Video Generation

**Conference**: AAAI 2026
**arXiv**: [2411.18281](https://arxiv.org/abs/2411.18281)  
**Code**: [https://motioncharacter.github.io/](https://motioncharacter.github.io/)  
**Area**: Video Generation
**Keywords**: human video generation, motion control, identity preservation, optical flow, diffusion models

## TL;DR

This paper proposes the MotionCharacter framework, which decouples motion into two independently controllable dimensions—action type and motion intensity—to achieve fine-grained motion control and identity consistency in high-fidelity human video generation.

## Background & Motivation

### State of the Field
Personalized text-to-video (T2V) generation has made remarkable progress in recent years, with subject-driven T2V models (e.g., VideoBooth, DreamVideo, ID-Animator) capable of producing high-quality videos faithfully depicting specific individuals.

### Limitations of Prior Work

**Insufficient motion control granularity**: Existing methods can only describe actions through coarse-grained text (e.g., "open mouth") and cannot precisely control motion intensity (e.g., "slightly" vs. "widely"). This is because text captures actions in a discrete manner, while motion intensity is inherently continuous.

**Coupling of action semantics and intensity**: Action type and motion magnitude are naturally entangled in text descriptions, forcing the model to "guess" the intended magnitude and yielding unpredictable results.

**Difficulty in identity preservation**: Maintaining subject identity consistency becomes a significant challenge as motion becomes more dynamic. Existing methods face an irreconcilable trade-off between Dynamic Degree and Face Similarity—either generating near-static videos to preserve identity or sacrificing identity to achieve dynamic motion.

### Starting Point
The paper explicitly decouples motion into two independently controllable components—**action type** and **motion intensity**—specifying action type via text phrases and controlling intensity via a continuous optical-flow-based scalar. A dedicated identity preservation module is designed to address identity degradation under dynamic motion.

## Method

### Overall Architecture

Given a reference identity image $\mathcal{I}$, a text prompt $\mathcal{P}$, an action phrase $\mathcal{A}$, and a motion intensity $\mathcal{M}$, the model generates a video $\mathcal{V} = \mathcal{F}(\mathcal{I}, \mathcal{P}, \mathcal{A}, \mathcal{M})$. The framework comprises three core components: an ID Content Insertion Module, a Motion Control Module, and a composite loss function.

### Key Designs

#### 1. **ID Content Insertion Module**

**Function**: Extracts identity embeddings from the reference image and injects them into the diffusion model to ensure subject identity consistency in the generated video.

**Mechanism**:
- The face region is first cropped from the reference image to filter background interference.
- The cropped face is passed in parallel through a CLIP image encoder and an ArcFace face recognition model, yielding a broad contextual embedding $E_{clip}$ and a fine-grained identity embedding $E_{arc}$, respectively.
- The two embeddings are fused via cross-attention: $C_{id} = \text{Proj}(\text{Attn}(E_{arc}W_q', EW_k', EW_v'))$, where $E = E_{clip} + E_{arc}$.
- The identity embedding $C_{id}$ serves as an image prompt embedding and guides the diffusion model together with the text prompt embedding: $z' = \text{Attn}(Q, K^t, V^t) + \lambda \cdot \text{Attn}(Q, K^i, V^i)$.

**Design Motivation**: ArcFace provides precise facial features while CLIP provides contextual information; the two are complementary. Cross-attention allows ArcFace features to selectively attend to the most relevant contextual information in CLIP.

#### 2. **Motion Control Module**

**Function**: Enables independent control of action type and motion intensity.

**Mechanism**:

**(a) Motion intensity estimation**: The RAFT optical flow model is used to compute pixel-level motion between adjacent frames. Foreground motion regions are extracted via thresholding, and the mean foreground optical flow value is computed as the motion intensity:

$$\mathcal{M} = \frac{1}{N-1} \sum_{i=1}^{N-1} f_{i,fg}$$

where $f_{i,fg}$ is the mean optical flow value of the foreground in each frame. The scalar $\mathcal{M}$ is mapped to a motion intensity embedding $E_M$ via an MLP.

**(b) Motion condition injection**: Two parallel cross-attention modules inject the action embedding $E_A$ (obtained from the CLIP text encoder) and the motion intensity embedding $E_M$ separately:

$$Z'' = \text{Attn}(Q', K^a, V^a) + \alpha \cdot \text{Attn}(Q', K^m, V^m)$$

**Key distinction**: Unlike SVD, which uses a single motion bucket for coarse global control, the dual-branch strategy separates semantic guidance (what action) from intensity control (how much), fusing them via parallel cross-attention to achieve predictable, fine-grained control.

#### 3. **ID-Consistency Loss**

**Function**: Enforces identity preservation at the semantic level, compensating for the insensitivity of pixel-level MSE loss to high-level concepts such as identity.

**Core formula**:

$$\mathcal{L}_{id} = 1 - \frac{1}{N} \sum_{i=1}^{N} \frac{\phi(I) \cdot \phi(X_i^f)}{|\phi(I)||\phi(X_i^f)|}$$

where $\phi$ is a pretrained ArcFace backbone. This loss directly penalizes deviations in the identity feature space, ensuring that core character attributes are preserved even under complex motion.

### Loss & Training

**Region-Aware Loss**: Normalized foreground optical flow is used as a weighted mask to impose larger denoising loss weights on high-motion regions (e.g., the face):

$$\mathcal{L}_R = \frac{1}{NH'W'} \sum_i \sum_{x,y} M_{i,\text{norm}} \cdot [\epsilon_i(x,y) - \hat{\epsilon}_i(x,y)]^2$$

**Total loss**: $\mathcal{L}_{total} = \mathcal{L}_R + \lambda_{id} \cdot \mathcal{L}_{id}$

**Image–video mixed training**: Approximately 17,619 stylized portrait images are replicated into 16-frame static videos (with motion intensity of 0), providing "zero-intensity calibration" and helping the model learn a smooth transition spectrum from static to dynamic.

**Human-Motion Dataset**: A total of 106,292 video clips are collected from multiple sources including VFHQ, CelebV-Text, and CelebV-HQ, with dual-track annotations—overall descriptions and action phrases—generated automatically using a large multimodal model (LMM).

## Key Experimental Results

### Main Results

| Method | Dover Score↑ | Motion Smooth.↑ | Dynamic Degree↑ | CLIP-I↑ | CLIP-T↑ | Face Sim.↑ |
|------|-------------|-----------------|-----------------|---------|---------|-----------|
| IPA-PlusFace | 0.797 | 0.985 | 0.325 | 0.587 | 0.218 | 0.480 |
| IPA-FaceID-PlusV2 | 0.813 | 0.987 | 0.085 | 0.575 | 0.217 | **0.617** |
| ID-Animator | 0.857 | 0.979 | 0.433 | 0.607 | 0.204 | 0.546 |
| **MotionCharacter** | **0.869** | **0.998** | **0.449** | **0.633** | **0.227** | 0.609 |

Key finding: IPA-FaceID-PlusV2 achieves the highest Face Similarity (0.617) but an extremely low Dynamic Degree (0.085), demonstrating that existing methods are forced to trade off identity against motion. MotionCharacter achieves a **428%** improvement in Dynamic Degree at the cost of only a 1.3% reduction in Face Similarity.

### Ablation Study

| $\mathcal{L}_R$ | $\mathcal{L}_{id}$ | Dover↑ | Dynamic Degree↑ | Face Sim.↑ |
|:---:|:---:|--------|-----------------|-----------|
| ✗ | ✗ | 0.801 | 0.355 | 0.484 |
| ✗ | ✓ | 0.810 | 0.359 | 0.588 |
| ✓ | ✗ | 0.860 | 0.419 | 0.500 |
| ✓ | ✓ | **0.869** | **0.449** | **0.609** |

| Motion Control Module | Dover↑ | Dynamic Degree↑ | Face Sim.↑ |
|:---:|--------|-----------------|-----------|
| w/o MCM | 0.805 | 0.245 | 0.601 |
| w/ MCM | **0.869** | **0.449** | **0.609** |

### Key Findings

1. **Synergistic enhancement**: The combination of the two loss functions outperforms each individually—$\mathcal{L}_{id}$ provides a stable identity foundation that enables $\mathcal{L}_R$ to sculpt more expressive motion.
2. **Substantial impact of MCM**: The Motion Control Module improves Dynamic Degree by 83.3% (0.245→0.449) while maintaining face similarity.
3. **User study validation**: Across 3,000 ratings from 10 expert evaluators on 100 videos, MotionCharacter achieves the highest preference in all three dimensions: identity consistency, motion controllability, and video quality.

## Highlights & Insights

1. **Elegant realization of motion decoupling**: The orthogonal combination of a continuous optical flow scalar and discrete text action phrases allows users to precisely adjust motion magnitude via a slider.
2. **"Zero-intensity calibration" training strategy**: Static images are used as training samples with motion intensity of 0, anchoring the zero point of the continuous motion intensity spectrum.
3. **Dual-track annotation of the Human-Motion Dataset**: Simultaneous annotation of motion semantics and motion intensity provides a data foundation for fine-grained motion generation research.

## Limitations & Future Work

1. The method is primarily validated on facial motion control; control of fast full-body motion is not sufficiently demonstrated.
2. The motion intensity range is limited to 0–20; extreme motion beyond this range requires capping.
3. The approach relies on the accuracy of optical flow estimation, which may be unreliable in complex occlusion scenarios.
4. The dataset is predominantly single-person; motion control in multi-person interaction scenarios remains an open problem.

## Related Work & Insights

- **IP-Adapter / InstantID series**: Provides the foundation for the identity embedding injection scheme adopted in this paper.
- **Stable Video Diffusion (SVD)**: Also uses optical flow but only for coarse global control; the contrast highlights the advantage of this paper's dual-branch decoupling approach.
- **AnimateDiff**: Serves as the base T2V generation model in this work.

## Rating

- Novelty: ⭐⭐⭐⭐ — Motion decoupling and zero-intensity calibration are novel ideas
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive quantitative, qualitative, user study, and ablation experiments
- Writing Quality: ⭐⭐⭐⭐ — Clear logic and intuitive illustrations
- Value: ⭐⭐⭐⭐ — Practically valuable reference for the controllable video generation field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MotionAgent: Fine-grained Controllable Video Generation via Motion Field Agent](../../ICCV2025/video_generation/motionagent_fine-grained_controllable_video_generation_via_motion_field_agent.md)
- [\[AAAI 2026\] DreamRunner: Fine-Grained Compositional Story-to-Video Generation with Retrieval-Augmented Motion Adaptation](dreamrunner_fine-grained_compositional_story-to-video_genera.md)
- [\[ICML 2026\] DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation](../../ICML2026/video_generation/dfsattn_dynamic_fine-grained_sparse_attention_for_efficient_video_generation.md)
- [\[CVPR 2026\] LAMP: Language-Assisted Motion Planning for Controllable Video Generation](../../CVPR2026/video_generation/lamp_language-assisted_motion_planning_for_controllable_video_generation.md)
- [\[ICLR 2026\] MoSA: Motion-Coherent Human Video Generation via Structure-Appearance Decoupling](../../ICLR2026/video_generation/mosa_motion-coherent_human_video_generation_via_structure-appearance_decoupling.md)

</div>

<!-- RELATED:END -->
