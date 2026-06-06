---
title: >-
  [Paper Note] Unified Camera Positional Encoding for Controlled Video Generation
description: >-
  [CVPR 2026][Video Generation][Camera Positional Encoding] This paper proposes Unified Camera Positional Encoding (UCPE), which injects complete camera geometric information (6-DoF pose, intrinsics…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Camera Positional Encoding"
  - "Lens Distortion"
  - "Diffusion Transformer"
  - "Camera-Controlled Generation"
date: 2026-05-08
content_hash: b17fe2c9fb66405d
---

# Unified Camera Positional Encoding for Controlled Video Generation

**Conference**: CVPR 2026
**arXiv**: [2512.07237](https://arxiv.org/abs/2512.07237)  
**Code**: [https://github.com/chengzhag/UCPE](https://github.com/chengzhag/UCPE)  
**Area**: Video Generation
**Keywords**: Camera Positional Encoding, Video Generation, Lens Distortion, Diffusion Transformer, Camera-Controlled Generation

## TL;DR

This paper proposes Unified Camera Positional Encoding (UCPE), which injects complete camera geometric information (6-DoF pose, intrinsics, and lens distortion) into Transformer attention mechanisms via relative ray encoding and absolute orientation encoding. UCPE enables fine-grained video generation control across heterogeneous camera models while introducing less than 1% additional trainable parameters.

## Background & Motivation

Transformers have become the universal backbone for 3D perception, video generation, and world models, where understanding camera geometry is essential for grounding visual observations in 3D space. However, existing camera encoding methods suffer from significant limitations.

Prior approaches fall into two categories: (1) **Absolute encoding** (e.g., Plücker encoding) represents each ray as a 6D direction-plus-moment vector, but relies on a predefined world coordinate system and generalizes poorly across scenes; (2) **Relative encoding** (e.g., CaPE, GTA, PRoPE) achieves coordinate-system invariance by encoding relative camera transformations within attention, but **assumes pinhole projection** and cannot handle nonlinear lens distortions such as fisheye or panoramic optics.

In practical applications (autonomous driving, robotic perception), wide-angle, fisheye, and omnidirectional cameras are ubiquitous, and strong distortions cause pinhole-assumption-based methods to fail. Furthermore, existing text-to-video methods define camera poses only relative to the first frame, leaving the global rotation (particularly pitch and roll) ambiguous — the absolute orientation of the initial viewpoint cannot be specified or reproduced.

The core insight is to **refine encoding from the "camera level" to the "ray level"**: each image token corresponds to an individual observation ray, and relative transformations are applied in ray space rather than camera space, naturally accommodating arbitrary projection models.

## Method

### Overall Architecture

UCPE consists of two complementary components: (1) **Relative Ray Encoding** — constructs a local ray coordinate frame for each token and applies inter-ray relative geometric transformations within attention; (2) **Absolute Orientation Encoding** — anchors the gravity direction via a Latitude-Up Map (Lat-Up Map), providing absolute control over pitch and roll. Both components are injected into a pretrained diffusion Transformer (Wan model) through lightweight spatial attention adapters, preserving the pretrained prior.

### Key Designs

1. **Relative Ray Encoding**:

    - **Function**: Achieves geometrically consistent positional encoding across heterogeneous cameras in ray space.
    - **Mechanism**: Each camera is uniformly abstracted as a ray mapping function $\Phi_\psi: (u,v) \mapsto (o, d)$. For each token $t$, a local ray coordinate frame is constructed with the ray direction $d_t$ as the z-axis; the camera's downward direction determines the x- and y-axes, forming an orthogonal basis $R_t^{wr}$. Combined with translation, this yields a ray-to-world transform $T_t^{wr}$. Its inverse $T_t^{rw}$ serves as a geometric operator expanded to the feature dimension via the Kronecker product, enabling relative query-key transformations in attention: $Q_t^\top D_t D_{t'}^{-1} K_{t'}$ captures the geometric relationship between ray pairs.
    - **Design Motivation**: Existing relative encodings operate at the camera level, where all tokens within a view share the same transformation matrix, failing to express the spatially varying projections induced by nonlinear distortions. Ray-level encoding assigns each token an independent geometric operator, naturally accommodating fisheye, panoramic, and arbitrary projection models.

2. **Absolute Orientation Encoding (Lat-Up Map)**:

    - **Function**: Resolves pitch/roll ambiguity at the first frame and provides absolute camera orientation control.
    - **Mechanism**: The Latitude Map encodes the elevation angle of each ray relative to the horizontal plane: $\text{Lat}_t = \arctan2(-d_{t,y}, \sqrt{d_{t,x}^2 + d_{t,z}^2})$. The Up Map is obtained by rotating each ray by a small angle $\delta$ about its local axis, projecting back onto the image plane, and computing the normalized displacement direction. The two maps are concatenated into a 3D vector, linearly projected, and added as a bias to the token features.
    - **Design Motivation**: Relative encoding is inherently agnostic to absolute orientation. Real-world video typically exhibits a gravity-aligned "up" direction. The Lat-Up Map provides appearance cues such as sky/ground separation, while also offering partial distortion awareness for wide-angle lenses.

3. **Spatial Attention Adapter**:

    - **Function**: Efficiently injects UCPE into a pretrained diffusion Transformer.
    - **Mechanism**: Within each DiT block, the original self-attention is kept intact. A UCPE-conditioned attention branch is added in parallel, using a reduced-dimension projection of $1/C$ to limit parameters and computation. Its output is mapped back to the original dimension via a zero-initialized linear layer and added to the original attention output, ensuring that the model behavior at initialization is identical to the pretrained baseline.
    - **Design Motivation**: Directly replacing positional encodings would disrupt the pretrained prior. A LoRA-style parallel adapter introduces camera awareness with negligible overhead (<1% parameters) while preserving visual fidelity.

### Loss & Training

The adapter parameters are trained with the standard diffusion loss. To support diverse camera training and evaluation, the authors construct a large-scale video dataset covering a wide range of intrinsics, distortion configurations, and motion trajectories. The encoding mixes relative ray encoding with RoPE (each occupying half of the feature dimension) to support reasoning in both ray space and image space.

## Key Experimental Results

### Main Results

**Synthetic Dataset (Multiple Lens Types)**

| Method | Pose Accuracy (RotErr↓) | Pose Accuracy (TransErr↓) | FID↓ | Distortion Consistency |
|--------|------------------------|--------------------------|------|----------------------|
| Wan CameraCtrl | High | High | Medium | No distortion control |
| ReCamMaster | Medium | Medium | Medium | Fails to reproduce distortion |
| **UCPE** | **Lowest** | **Lowest** | **Lowest** | **Accurate reproduction** |

**RealEstate10K Dataset (Pinhole Camera)**

| Method | RotErr↓ | TransErr↓ | FVD↓ | Notes |
|--------|---------|-----------|------|-------|
| CameraCtrl | High | High | High | Severe artifacts |
| AC3D | Medium | Medium | Medium | Preserves training aesthetics but unbalanced composition |
| Wan CameraCtrl | Medium | Medium | Medium | Poor motion consistency |
| **UCPE** | **Lowest** | **Lowest** | **Lowest** | Sharper results with better motion following |

### Ablation Study

| Configuration | RotErr | TransErr | Notes |
|---------------|--------|----------|-------|
| Plücker (absolute) | High | High | Strong coordinate-system dependency |
| CaPE (relative, camera-level) | Medium | Medium | Limited by pinhole assumption |
| PRoPE (projection-level) | Medium-low | Medium-low | Still restricted to pinhole |
| **UCPE relative ray** | **Lowest** | **Lowest** | Best generalization at ray level |
| UCPE + Lat-Up Map | Lowest | Lowest | Additional pitch/roll control |

### Key Findings

- Upgrading from camera-level to ray-level encoding yields improvements even in pinhole settings, indicating that fine-grained ray-level geometric information has universal value.
- The Lat-Up Map not only resolves orientation ambiguity but also provides auxiliary distortion awareness for wide-angle lenses.
- The adapter requires fewer than 1% additional parameters, demonstrating that geometric priors can be injected into pretrained models with extreme efficiency.
- On non-pinhole cameras such as fisheye and UCM models, UCPE is the only method capable of correctly reproducing target distortions.

## Highlights & Insights

- **"Ray-level positional encoding" is an elegant unified abstraction**: Any camera model (pinhole, fisheye, panoramic) can be abstracted as a pixel-to-ray mapping. Performing relative transformations in ray space naturally accommodates all projection types without requiring model-specific handling.
- **Zero-initialized adapter design**: Zero-initializing the output projection ensures that the training starting point is equivalent to the original model, mitigating the catastrophic forgetting commonly observed when fine-tuning large pretrained models.
- **Generality beyond video generation**: UCPE can be directly applied to other Transformer-based tasks requiring camera geometric reasoning, such as multi-view reconstruction and novel view synthesis.

## Limitations & Future Work

- Validation is currently limited to video generation; experiments on multi-view reconstruction and novel view synthesis remain to be conducted.
- Non-central cameras (e.g., catadioptric systems) have distinct ray origins; while theoretically handleable, this has not been experimentally verified.
- Training data diversity still relies on synthetic data, as real-world heterogeneous camera data remains scarce.
- The Lat-Up Map assumes the existence of a gravity direction and is inapplicable to gravity-free environments such as outer space.

## Related Work & Insights

- **vs. Plücker encoding**: An absolute encoding that depends on a world coordinate system and generalizes poorly across poses; UCPE's relative ray encoding eliminates this dependency.
- **vs. PRoPE**: Performs relative encoding in projection space, but the intrinsic matrix $K$ is only valid for pinhole models; UCPE replaces the projection matrix with a ray mapping function.
- **vs. ReCamMaster/CameraCtrl**: Methods based on Plücker encoding or parameter injection cannot handle nonlinear distortions.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Ray-level positional encoding represents a fundamental advance in camera encoding, unifying heterogeneous camera geometries.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dataset, multi-baseline comparisons are comprehensive, though evaluation is limited to video generation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous and clear; method comparison figures are intuitive.
- **Value**: ⭐⭐⭐⭐⭐ A unified camera representation with broad impact across the 3D vision community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UniTalking: A Unified Audio-Video Framework for Talking Portrait Generation](unitalking_a_unified_audio-video_framework_for_talking_portrait_generation.md)
- [\[ICCV 2025\] ReCamMaster: Camera-Controlled Generative Rendering from A Single Video](../../ICCV2025/video_generation/recammaster_camera-controlled_generative_rendering_from_a_single_video.md)
- [\[CVPR 2026\] UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions](uniavgen_unified_audio_and_video_generation_with_asymmetric_cross-modal_interact.md)
- [\[CVPR 2026\] VideoCoF: Unified Video Editing with Temporal Reasoner](videocof_unified_video_editing_with_temporal_reasoner.md)
- [\[CVPR 2026\] U-Mind: A Unified Framework for Real-Time Multimodal Interaction with Audiovisual Generation](u-mind_a_unified_framework_for_real-time_multimodal_interaction_with_audiovisual.md)

</div>

<!-- RELATED:END -->
