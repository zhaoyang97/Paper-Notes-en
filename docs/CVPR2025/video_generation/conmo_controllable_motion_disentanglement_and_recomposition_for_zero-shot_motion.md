---
title: >-
  [Paper Note] ConMo: Controllable Motion Disentanglement and Recomposition for Zero-Shot Motion Transfer
description: >-
  [CVPR 2025][Video Generation][Motion Transfer] ConMo proposes a zero-shot motion transfer framework. By disentangling the composite motion in a reference video into independent subject motions and background (camera) motion, and then controllably recomposing these motions during target video generation, it enables various applications such as multi-subject motion transfer, semantic/shape transformation, subject removal, and camera motion simulation. It significantly outperfor…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Motion Transfer"
  - "Video Diffusion Models"
  - "Motion Disentanglement"
  - "Zero-Shot Generation"
  - "Multi-Subject Motion"
date: 2026-05-08
content_hash: 0a63294c534fcd40
---

# ConMo: Controllable Motion Disentanglement and Recomposition for Zero-Shot Motion Transfer

**Conference**: CVPR 2025  
**arXiv**: [2504.02451](https://arxiv.org/abs/2504.02451)  
**Code**: [GitHub](https://github.com/Andyplus1/ConMo)  
**Area**: Video Generation  
**Keywords**: Motion Transfer, Video Diffusion Models, Motion Disentanglement, Zero-Shot Generation, Multi-Subject Motion

## TL;DR
ConMo proposes a zero-shot motion transfer framework. By disentangling the composite motion in a reference video into independent subject motions and background (camera) motion, and then controllably recomposing these motions during target video generation, it enables various applications such as multi-subject motion transfer, semantic/shape transformation, subject removal, and camera motion simulation. It significantly outperforms existing methods in motion fidelity and text alignment.

## Background & Motivation
The advancement of text-to-video (T2V) generation has made motion transfer possible. However, current methods suffer from two key limitations: (1) they cannot handle multi-subject videos, making it difficult to transfer the independent motion of each subject separately; (2) when the shape of the target subject differs significantly from the source subject (e.g., car to motorbike), motion adaptation becomes challenging.

The Key Challenge lies in the fact that existing methods use a **holistic motion representation**, which mixes different subject motions and camera motion together. This leads to motion "crosstalk" in multi-subject scenarios, making it impossible to control each subject independently. In shape-changing scenarios, the shape constraints of the original subject are too strong, limiting the adaptation of the diffusion model to new semantics.

This paper's Key Insight: first **disentangle** the composite motion into individual subject motions and background motion, and then controllably recompose the motions through a **soft guidance** strategy. The background motion is introduced to "dilute" the shape constraints in the subject motion, providing greater flexibility for shape changes.

## Method

### Overall Architecture
ConMo consists of two phases: (1) **Reference video motion disentanglement phase** — during the DDIM inversion process, utilizing the subject masks extracted by SAM2, the inter-frame feature differences of each subject and the background are calculated separately as independent motion cues; (2) **Motion recomposition + target video generation phase** — the disentangled motion cues are injected into the denoising process via a Motion Guidance function and a Soft Guidance strategy to generate motion-consistent target videos.

### Key Designs
1. **Motion Disentanglement via LSMM (Local Spatial Marginal Mean)**:

    - Function: Extracts the independent motion representation of each subject from the DDIM inversion features of the reference video.
    - Mechanism: For subject $s_k$, use SAM2 to obtain the mask $M_{s_k}$, and calculate the Local Spatial Marginal Mean (LSMM) of frame pairs $(i, j)$ on the subject motion region $M_{s_k}^i \cup M_{s_k}^j$: $\phi(s_k, i, j, t) = \frac{1}{\sum(M_{s_k}^{i|j} \cup M_{s_k}^{j|i})} \sum f(z_t^i) \cdot (M_{s_k}^{i|j} \cup M_{s_k}^{j|i})$
    - Key improvement: Excludes the interference regions of other subjects using the set difference operation $M_{s_k}^{i|j} = M_{s_k}^i \setminus M_{s_m}^j$, preventing cross-contamination of multi-subject motions.
    - Motion representation: $\Delta_{s_k}^{(i,j)} = \phi(s_k, i, j, t) - \phi(s_k, j, i, t)$

2. **Motion Guidance for Recomposition**:

    - Function: Optimizes the noise latent variables during the target video denoising process to match the motion features of the target video with the reference motion.
    - Mechanism: Defines a guidance loss for each subject as $\mathcal{L}_{s_k} = \sum_i \sum_j \|\Delta_{s_k}^{(i,j)} - \tilde{\Delta}_{s_k}^{(i,j)}\|_2^2$, adjusting the denoising direction via gradient guidance.
    - Design Motivation: The motion of each subject can be independently recomposed, added, or removed, achieving fine-grained motion control.

3. **Soft Guidance Strategy**:

    - Function: Weakens the shape semantic constraints in the subject motion by blending background motion, enabling more flexible shape transformation.
    - Mechanism: $\Delta_{s_k^*}^{(i,j)} = \frac{\Delta_{s_k}^{(i,j)} + w_c \cdot \Delta_c^{(i,j)}}{w_c + 1}$, where $w_c$ controls the blending intensity of the background motion.
    - Design Motivation: Experiments show that using background motion alone can approximate camera motion. Introducing background motion into the subject motion can "dilute" the original shape structure constraints, leaving more space for the diffusion model to generate target subjects of different shapes.

### Loss & Training
ConMo is a zero-shot method that requires no training. During inference, gradients are calculated and noise latent variables are updated at each denoising step $t$ via the Motion Guidance function (Eq.5). Guidance losses for subject motion and background motion are used respectively to control the motion consistency of each part.

## Key Experimental Results

### Main Results

| Method | Text Alignment ↑ | Motion Fidelity ↑ | User Score - Motion Keeping ↑ | User Score - Motion Quality ↑ | User Score - Text Alignment ↑ |
|------|------|----------|------|------|------|
| Control-A-Video | 30.13 | 0.7661 | 3.43 | 2.38 | 1.42 |
| VMC | 32.56 | 0.7979 | 2.45 | 2.33 | 4.23 |
| MotionClone | 31.00 | 0.8876 | 4.20 | 3.40 | 3.01 |
| DMT | 31.46 | 0.8815 | 4.20 | 3.70 | 4.10 |
| **ConMo** | **31.96** | **0.8931** | **4.40** | **4.11** | **4.30** |

### Ablation Study

| Configuration | Text Alignment ↑ | Motion Fidelity ↑ | Description |
|------|---------|------|------|
| DMT (baseline) | 31.46 | 0.8675 | Global motion guidance |
| +Eq.1 (Local LSMM) | 31.55 | 0.8813 | Basic local motion extraction |
| +SG (Soft Guidance) | 31.89 | 0.8795 | Enhanced shape adaptation |
| +Eq.3 (Interference Exclusion) | 31.96 | 0.8931 | Full ConMo |

### Key Findings
- Fine-grained motion extraction that excludes multi-subject cross-interference (Eq.3 vs Eq.1) significantly improves motion fidelity.
- The soft guidance strategy can progressively weaken original shape constraints by increasing $w_c$, enhancing text alignment.
- Videos generated using only background motion mainly contain camera motion changes, verifying the effectiveness of motion disentanglement.
- ConMo is the only method capable of independently transferring individual subject motions in multi-subject scenarios.

## Highlights & Insights
- **Inspiration of Motion Disentanglement and Recomposition**: Decomposing composite motion into independent subject motion + camera motion is an elegant paradigm, laying the foundation for fine-grained motion control.
- **Core Insight of the Soft Guidance Strategy**: Motion features encode shape semantic information simultaneously; blending background motion can "dilute" shape constraints—this finding is insightful for understanding motion representation.
- **Broad Applicability**: The unified framework supports six applications: multi-subject transfer, semantic transformation, size editing, position editing, subject removal, and camera simulation.
- **Zero-Shot**: Plug-and-play, requiring no training.
- **Mask Exclusion Strategy**: Handles multi-subjects with overlapping trajectories using a set difference operation, which is simple yet effective.

## Limitations & Future Work
- Relies on the quality of SAM2 masks; inaccurate masks in complex occlusion scenarios will affect motion disentanglement.
- The parameter $w_c$ in soft guidance requires manual tuning, lacking an adaptive mechanism.
- The evaluation dataset is relatively small (26 videos, 56 edit pairs); large-scale evaluation is lacking.
- Motion disentanglement is based on intermediate feature differences, which may not fully separate highly coupled interaction motions (e.g., two people fighting).
- Generation quality is limited by the underlying T2V model, currently based on older models like AnimateDiff.

## Detailed Application Scenarios

ConMo unlocks various applications through the disentanglement-recomposition paradigm, which is worth elaborating:

1. **Semantic/Shape Transformation**: Controlling the similarity between the target subject and the original shape by adjusting $w_c$ (the weight of background motion in soft guidance). A larger $w_c$ weakens the original shape constraint, allowing the target subject to match the text description more freely (e.g., "car" to "motorbike" requires a larger deformation space).
2. **Position/Size Editing**: Changing the location and scale where motion occurs by translating or scaling local mask regions in the target latent. For example, translating helicopter motion from the ground to the sky, aligning the semantics ("fly") with the visual content.
3. **Subject Removal**: Replacing the motion representation $\Delta_{s_k}$ of a specific subject with the background motion $\Delta_c$, which is equivalent to filling the subject area with the background.
4. **Camera Motion Simulation**: Guiding the generation using only the background motion $\Delta_c$ can replicate the camera motion trajectory of the original video.

## Related Work & Insights
- **vs DMT**: DMT models motion using global feature differences and cannot handle multi-subject scenarios; ConMo achieves independent motion control through mask localization.
- **vs MotionClone**: MotionClone represents motion with sparse temporal attention weights, showing high motion fidelity but poor text alignment; ConMo achieves a better balance between the two.
- **vs VMC**: VMC achieves motion customization through temporal attention adaptation, but experiences semantic inconsistency during shape changes; ConMo's soft guidance strategy better addresses shape variations.
- **vs Control-A-Video**: CAV uses control signals (like Canny edges), excessively preserving the original structure and limiting shape variations; ConMo is free from structural constraints.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Proposes the motion disentanglement + recomposition motion transfer paradigm for the first time, with deep insights from the soft guidance strategy on shape adaptation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Rich qualitative comparisons and systematic ablation studies, but the dataset scale is small, with fewer quantitative metrics.
- Writing Quality: ⭐⭐⭐⭐ Clear method descriptions and rich, intuitive illustrations, though the notation system is slightly complex.
- Value: ⭐⭐⭐⭐ Opens up the direction of fine-grained motion control for multiple subjects with broad application prospects, but is constrained by the overall quality of current T2V models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DiTFlow: Video Motion Transfer with Diffusion Transformers](video_motion_transfer_with_diffusion_transformers.md)
- [\[CVPR 2025\] Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion](zero-1-to-a_zero-shot_one_image_to_animatable_head_avatars_using_video_diffusion.md)
- [\[CVPR 2025\] TokenMotion: Decoupled Motion Control via Token Disentanglement for Human-centric Video Generation](tokenmotion_decoupled_motion_control_via_token_disentanglement_for_human-centric.md)
- [\[NeurIPS 2025\] DisMo: Disentangled Motion Representations for Open-World Motion Transfer](../../NeurIPS2025/video_generation/dismo_disentangled_motion_representations_for_openworld_moti.md)
- [\[CVPR 2025\] Motion Prompting: Controlling Video Generation with Motion Trajectories](motion_prompting_controlling_video_generation_with_motion_trajectories.md)

</div>

<!-- RELATED:END -->
