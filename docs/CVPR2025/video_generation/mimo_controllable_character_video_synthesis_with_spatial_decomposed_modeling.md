---
title: >-
  [Paper Note] MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling
description: >-
  [CVPR 2025][Video Generation][Character Video Synthesis] MIMO proposes a character video synthesis framework based on spatial decomposed modeling, which decomposes 2D videos into three spatial components (human, scene, and occluder) along 3D depth. Through decoupled encoding and compositional decoding, it achieves flexible control over character identity, 3D motion, and interactive scenes, significantly outperforming prior methods on complex motions and scene interactions.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Character Video Synthesis"
  - "Spatial Decomposed Modeling"
  - "Diffusion Models"
  - "Human Motion Transfer"
  - "Scene Interaction"
date: 2026-05-08
content_hash: 794c354bc45ccd9d
---

# MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling

**Conference**: CVPR 2025  
**arXiv**: [2409.16160](https://arxiv.org/abs/2409.16160)  
**Code**: [https://menyifang.github.io/projects/MIMO/index.html](https://menyifang.github.io/projects/MIMO/index.html)  
**Area**: Video Generation / 3D Vision  
**Keywords**: Character Video Synthesis, Spatial Decomposed Modeling, Diffusion Models, Human Motion Transfer, Scene Interaction

## TL;DR
MIMO proposes a character video synthesis framework based on spatial decomposed modeling, which decomposes 2D videos into three spatial components (human, scene, and occluder) along 3D depth. Through decoupled encoding and compositional decoding, it achieves flexible control over character identity, 3D motion, and interactive scenes, significantly outperforming prior methods on complex motions and scene interactions.

## Background & Motivation

1. **Background**: Character video synthesis is a fundamental problem in computer vision and computer graphics. 3D-based methods (NeRF/3DGS) require multi-view acquisition and per-case training, whereas 2D-based methods leveraging pretrained diffusion models can already generate character animations from a single reference image.
2. **Limitations of Prior Work**: 2D methods such as Animate Anyone, MimicMotion, and Champ only focus on simple 2D movements (e.g., front-facing dancing) and perform poorly in scenarios with complex 3D motion (e.g., extreme deformations, self-occlusions) and scene interactions (occlusion relations between humans and objects).
3. **Key Challenge**: Existing methods rely on inadequate video attribute parsers in 2D feature spaces, ignoring the inherent 3D spatial hierarchy of video scenes—foreground occluders, middle-ground characters, and background scenes actually reside in different depth layers.
4. **Goal**: (1) How to encode human motion with a better 3D representation? (2) How to fully decouple identity from motion? (3) How to achieve scene-occlusion awareness during synthesis?
5. **Key Insight**: Videos are essentially composed of spatial components at different depth layers. Lifting 2D pixels to 3D space, decomposing them by depth layers, and encoding them independently yields richer control signals.
6. **Core Idea**: Utilize monocular depth estimation to decompose videos into three layers (human, scene, and occluder) based on 3D depth, and encode them into identity codes, motion codes, and scene codes respectively, which serve as compositional conditions for the diffusion decoder.

## Method

### Overall Architecture
Given an input character video clip, MIMO performs reconstruction learning through the following steps: (1) lifting pixels of each frame to 3D using a monocular depth estimator, and extracting three spatial components (human, scene, and occluder) based on depth layering; (2) further decoupling the human component into identity encoding $\mathcal{C}_{id}$ and motion encoding $\mathcal{C}_{mo}$; (3) embedding the scene and occluder components using a shared VAE encoder and concatenating them into a complete scene encoding $\mathcal{C}_{so}$; (4) injecting all encodings as conditions into a diffusion decoder for video reconstruction. During inference, users can freely combine attribute encodings from different sources to synthesize new videos.

### Key Designs

1. **Hierarchical Spatial Layer Decomposition**:

    - **Function**: Automatically decompose video frames into three spatial components: human layer, scene layer, and occluder layer.
    - **Mechanism**: First, a monocular depth estimator is used to obtain the depth map for each frame, and the main subject is extracted using a human detector, followed by acquiring the masklet with a video tracker. Then, objects with a smaller (closer) average depth than the human layer are identified as the occluder layer, while the remaining parts form the scene layer. The video for each component is obtained via $v^i = v \odot \mathcal{M}^i$.
    - **Design Motivation**: Unlike previous methods that directly learn features of the entire 2D frame, layered modeling enables the network to learn 3D-aware hierarchical composition (foreground occluders in front of the human, background behind), naturally handling occlusion relationships in human-object interaction scenarios.

2. **Structured Motion Code**:

    - **Function**: Provide a more expressive 3D motion representation than 2D skeletons.
    - **Mechanism**: Define a set of 6,890 learnable latent codes anchored to the vertices of the SMPL human model. After estimating SMPL parameters for each frame, these codes are transformed with the human pose and projected onto a 2D plane. A continuous 2D feature map $\mathcal{F}_t$ is obtained via differentiable rasterization, which is then embedded as motion encoding by a pose encoder. This establishes a stable correspondence between the same set of identifiable codes and posed 2D renderings of different frames.
    - **Design Motivation**: 2D skeletons struggle to represent occlusion relationships in 3D spatial motion, and while Champ's 3D maps (normal maps, depth maps, etc.) show improvements, they lack dense identifiers for body parts. This method provides structured, identifiable, dense motion signals, significantly improving generalization to extreme 3D motion.

3. **Canonical Appearance Transfer**:

    - **Function**: Fully decouple identity information from motion information.
    - **Mechanism**: Utilize a pretrained human retargeting model to transform a posed human image into a canonical appearance image in a standard A-pose. Then, a CLIP image encoder is used to extract global features, and a reference-net is used to extract local features, which are combined into the identity encoding $\mathcal{C}_{id}$.
    - **Design Motivation**: Previous methods randomly select a frame from the video as the appearance reference. The highly similar poses between frames inevitably lead to entanglement between appearance and motion. Transforming the pose to a canonical pose eliminates this coupling, making learning more efficient and resolving the issue of hand-foot synthesis confusion.

### Loss & Training
Training employs the standard diffusion noise prediction loss: $\mathcal{L} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, c_{id}, c_{so}, c_{mo}, t)\|^2_2]$. SD 1.5 pretrained weights are used to initialize the U-Net and reference-net, while AnimateDiff initializes the motion module. The VAE and CLIP encoders are frozen, and the U-Net, pose encoder, and reference-net are trained. Training is conducted on 8 A100-80G GPUs for approximately 50k iterations, with 24-frame videos and a batch size of 8.

## Key Experimental Results

### Main Results

The training dataset, HUD-7K, contains 5K real videos + 2K synthetic videos. The test set comprises 100 videos with diverse content, including dancing, sports, and movies.

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ |
|------|-------|-------|--------|------|
| Animate Anyone | 21.003 | 0.722 | 0.264 | 304.3 |
| Mimic-Motion | 20.688 | 0.731 | 0.343 | 289.2 |
| Champ | 21.044 | 0.724 | 0.312 | 412.5 |
| **MIMO (Ours)** | **25.210** | **0.883** | **0.125** | **221.4** |

Ours gains at least 4.16 in PSNR and 0.152 in SSIM, leading comprehensively.

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | FVD↓ |
|------|-------|-------|--------|------|
| Full model | 25.210 | 0.883 | 0.125 | 221.4 |
| w/o SDM (No Spatial Decomposition) | 22.148 | 0.762 | 0.231 | 268.5 |
| w/ 2D skeleton | 24.326 | 0.842 | 0.186 | 237.2 |
| w/ 3D maps | 24.402 | 0.844 | 0.203 | 278.1 |
| w/o CA (No Canonical Appearance) | 24.918 | 0.871 | 0.148 | 223.1 |

### Key Findings
- **Spatial Decomposed Modeling contributes the most**: Removing SDM drops PSNR by 3.06, proving that 3D-aware hierarchical encoding is core.
- **Structured motion coding outperforms 2D skeletons and 3D maps**: It brings about 0.9 and 0.8 PSNR gains respectively, particularly in extreme 3D motion and self-occlusion scenarios.
- **Canonical appearance transfer effectively alleviates hand-foot confusion**: Removing it drops PSNR by 0.3, and qualitative results show significantly increased confusion in hand and foot synthesis.
- Performance is particularly outstanding in scene interaction (human-object occlusion) and large camera motion scenarios.

## Highlights & Insights
- **The concept of 3D depth layering is highly clever**: Instead of actual 3D reconstruction, utilizing monocular depth estimation + hierarchical masks decomposes the video into semantically clear spatial components. This is a lightweight 3D-aware approach that can be transferred to any video generation task requiring scene hierarchy understanding.
- **The motion representation of SMPL vertex-anchored learnable encoding is worth adapting**: Defining learnable latent codes on the 3D human surface preserves structural information while enabling end-to-end learning, which is superior to hand-crafted 2D/3D pose representations.
- The framework achieves true multi-attribute controllable synthesis (character/motion/scene) and supports the brand-new editing task of video character replacement.

## Limitations & Future Work
- Relies on the accuracy of SMPL parameter estimation; it may fail for non-standard human bodies (extremely obese, extremely thin, or non-human characters).
- Trained only on 512-resolution videos; scalability to high resolutions remains unverified.
- The video length limitation of 24 frames restricts the coherence of long video generation.
- The detection of the occluder layer relies on a simple depth comparison rule, which may not be robust enough for complex multi-person scenes.

## Related Work & Insights
- **vs Animate Anyone/Champ**: These methods directly learn full-frame features in 2D space, controlling motion with only 2D skeletons or simple 3D maps, failing to handle scene interactions. MIMO comprehensively outperforms them through spatial decomposition and structured motion encoding.
- **vs MimicMotion**: MimicMotion improves quality using confidence-aware pose guidance but remains restricted to 2D space, lacking generalization to 3D motion.
- **vs HumanNeRF/NeuMan**: 3D methods require per-case training. MIMO achieves generalizable character synthesis using 2D diffusion models.

## Rating
- Novelty: ⭐⭐⭐⭐ Spatial decomposed modeling and structured motion encoding are both highly effective new designs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparisons and ablations are comprehensive, though the test set of 100 videos is slightly small.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and detailed illustrations.
- Value: ⭐⭐⭐⭐ Provides a powerful baseline method for controllable character video synthesis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] StreetCrafter: Street View Synthesis with Controllable Video Diffusion Models](streetcrafter_street_view_synthesis_with_controllable_video_diffusion_models.md)
- [\[CVPR 2025\] World-Consistent Video Diffusion with Explicit 3D Modeling](world-consistent_video_diffusion_with_explicit_3d_modeling.md)
- [\[CVPR 2025\] LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis](levitor_3d_trajectory_oriented_image-to-video_synthesis.md)
- [\[CVPR 2025\] BF-STVSR: B-Splines and Fourier—Best Friends for High Fidelity Spatial-Temporal Video Super-Resolution](bf-stvsr_b-splines_and_fourier---best_friends_for_high_fidelity_spatia.md)
- [\[CVPR 2025\] AnimateAnything: Consistent and Controllable Animation for Video Generation](animateanything_consistent_and_controllable_animation_for_video_generation.md)

</div>

<!-- RELATED:END -->
