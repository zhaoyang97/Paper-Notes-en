---
title: >-
  [Paper Note] Let Your Image Move with Your Motion! – Implicit Multi-Object Multi-Motion Transfer
description: >-
  [CVPR 2026][Video Generation][Motion Transfer] This paper proposes FlexiMMT, the first I2V framework supporting implicit multi-object multi-motion transfer. It introduces a Motion Decoupling Mask Attention (MDMA) mechani…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Motion Transfer"
  - "Multi-Object Multi-Motion"
  - "Attention Mask"
  - "Video Diffusion Model"
  - "I2V Generation"
date: 2026-05-08
content_hash: fdd98919fcd1cfa5
---

# Let Your Image Move with Your Motion! – Implicit Multi-Object Multi-Motion Transfer

**Conference**: CVPR 2026
**arXiv**: [2603.01000](https://arxiv.org/abs/2603.01000)
**Code**: [Project Page](https://ethan-li123.github.io/FlexiMMT_page/)
**Area**: Video Generation
**Keywords**: Motion Transfer, Multi-Object Multi-Motion, Attention Mask, Video Diffusion Model, I2V Generation

## TL;DR

This paper proposes FlexiMMT, the first I2V framework supporting implicit multi-object multi-motion transfer. It introduces a Motion Decoupling Mask Attention (MDMA) mechanism to constrain motion/text tokens to interact only with their corresponding object regions, and a Differential Mask Extraction Mechanism (DMEM) to derive object masks from diffusion attention maps with progressive propagation, enabling precise compositional multi-object motion transfer.

## Background & Motivation

1. **Background**: Motion transfer is an important direction in controllable video generation, aiming to capture motion dynamics from reference videos and apply them to target subjects. Existing methods fall into two categories: explicit (pose/optical flow/trajectory) and implicit (encoding motion embeddings from reference videos). Implicit methods learn motion representations from reference videos via trainable motion tokens.

2. **Limitations of Prior Work**: Nearly all existing implicit motion transfer methods handle only single-object single-motion scenarios. When multiple objects with different motion patterns are present in a scene, existing methods cannot independently assign different motions to different objects.

3. **Key Challenge**: When multiple sets of motion tokens are directly injected into 3D full-attention layers, all token interactions are globally entangled — motion tokens of one object interfere with video tokens of other objects, causing motion confusion and erroneous transfer.

4. **Goal**: To achieve independent multi-object motion transfer in I2V generation, allowing each object to move according to its designated reference video.

5. **Key Insight**: Performing motion decoupling at the attention level — constraining motion and text tokens to interact only with video tokens of their corresponding objects via object-specific masks.

6. **Core Idea**: Employ masked attention to achieve motion decoupling, ensuring each object "sees" only its own motion signal and textual description.

## Method

### Overall Architecture

FlexiMMT is built upon the CogVideoX-5B-I2V diffusion model. During training, given a single reference video, trainable motion tokens are learned to encode motion representations, with object masks applied in attention layers to constrain motion-video interactions. During inference, for an input image containing multiple objects and multiple reference videos, the respectively pre-trained motion tokens are concatenated with text and video tokens; MDMA masks ensure each set of motion tokens influences only its corresponding object region, while RMPM dynamically propagates object masks to subsequent frames.

### Key Designs

1. **Motion Decoupling Mask Attention (MDMA)**:
   - **Function**: Explicitly decouples motion signals of different objects at the attention level.
   - **Mechanism**: Constructs a mask matrix $\mathcal{M}$ comprising two categories of sub-masks: Motion-to-X (M2X) and Text-to-X (T2X). M2X ensures motion tokens interact only with video tokens of their corresponding object ($\mathcal{M}_{m \to v}$), and different motion tokens do not interfere with each other ($\mathcal{M}_{m \to m} = \mathbf{0}$). T2X ensures text tokens describing a motion attend only to their corresponding object.
   - **Design Motivation**: Global token interaction in the original MM-DiT leads to motion entanglement; the masking mechanism achieves precise motion isolation with minimal architectural modification.

2. **Differential Mask Extraction Mechanism (DMEM)**:
   - **Function**: Provides object masks for the training and inference stages respectively.
   - **Mechanism**: During training (single object), the attention map between text query $Q_y^k$ and video key $K_v$ is used to automatically extract the object region, binarized with a mean threshold. During inference (multiple objects), a semantic segmentation model extracts the first-frame mask, which is then propagated to subsequent frames via RMPM.
   - **Design Motivation**: During training, this avoids the computational overhead of external segmentation models and training-inference inconsistency. During inference, simple attention-based methods cannot distinguish multiple objects due to feature entanglement, necessitating the segmentation-plus-propagation approach.

3. **Regression-based Mask Propagation Module (RMPM)**:
   - **Function**: Accurately propagates the first-frame object mask to all frames of the video.
   - **Mechanism**: Maintains a sliding-window anchor set (first frame + neighboring frames) and propagates anchor masks to the current frame via a feature correlation matrix $\mathcal{C}_l^k$. Dynamic RMPM further optimizes this: propagation stops when the mask change across consecutive steps falls below threshold $\alpha=5\%$, reusing the stable mask thereafter.
   - **Design Motivation**: Motion transfer in diffusion models is primarily accomplished during early denoising steps, after which mask changes are negligible. Dynamic termination substantially improves inference efficiency.

### Loss & Training

- Adopts the standard noise prediction loss from CogVideoX-5B-I2V.
- Motion tokens are trained for 2,000 steps using the AdamW optimizer, with a learning rate of 3e-3 and batch size of 1.
- Video resolution: $720 \times 480$, 49 frames per clip.
- RMPM sliding window size: $W=2$.
- Experiments conducted on 6 NVIDIA A800 GPUs.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (FlexiMMT) | Prev. SOTA | Gain |
|---------|--------|-----------------|------------|------|
| 200 pairs | Trajectory Fidelity (TF) | 0.577 | 0.488 (Go-with-Flow) | +0.089 |
| 200 pairs | Flow Fidelity (FF) | 0.723 | 0.648 (Go-with-Flow) | +0.075 |
| 200 pairs | Appearance Consistency | 0.904 | 0.939 (FlexiAct) | Slightly lower, but motion more accurate |
| 200 pairs | Human Eval – Motion Fidelity | 89.475% | 6.500% (FlexiAct) | Overwhelming lead |
| 200 pairs | Human Eval – Temporal Consistency | 83.875% | 11.550% (FlexiAct) | Overwhelming lead |

### Ablation Study

| Configuration | TF↑ | FF↑ | Notes |
|---------------|-----|-----|-------|
| Full FlexiMMT | 0.577 | 0.723 | Baseline |
| w/o M2X mask | 0.381 | 0.618 | TF drops 34%; M2X is core to motion decoupling |
| w/o T2X mask | 0.461 | 0.665 | TF drops 20%; T2X also important |
| w/o training-stage mask | 0.440 | 0.656 | Cannot accurately learn reference motion |
| w/o inference-stage mask (DMEM) | 0.373 | 0.602 | Multi-motion completely entangled |
| w/o RMPM | 0.377 | 0.607 | Similar to w/o inference mask |

### Key Findings

- In human evaluation, FlexiMMT receives 89.475% of votes for motion fidelity, far exceeding all baselines (second-place FlexiAct: only 6.5%).
- CLIP-based metrics such as AC and TC are biased toward static or weakly-moving videos — methods that fail at motion may paradoxically achieve high AC/TC scores.
- The proposed Flow Fidelity (FF) metric captures motion similarity more comprehensively than Trajectory Fidelity.
- Dynamic RMPM significantly reduces inference time compared to full RMPM with no performance degradation.

## Highlights & Insights

- The first framework to address implicit multi-object multi-motion transfer, filling an important gap in the field.
- The MDMA mechanism is conceptually simple yet effective: hard-cutting cross-object signals at the attention level via mask matrices is more reliable than soft constraints.
- Using different mask extraction strategies for the training and inference stages is a pragmatic engineering choice — lightweight attention-based extraction during training, and segmentation-plus-propagation during inference.
- Motion tokens can be freely recombined and exchanged, enabling truly compositional motion transfer.

## Limitations & Future Work

- Training requires each video to contain only a single object, demanding a large collection of single-object reference videos.
- First-frame segmentation relies on an external semantic segmentation model (Grounded SAM), introducing additional dependencies.
- RMPM propagates masks via feature correlation and may fail under occlusion or rapid motion causing significant appearance changes.
- Relatively lower AC and TC scores suggest that the generated motion may introduce a certain degree of appearance drift.

## Related Work & Insights

- **vs. FlexiAct**: FlexiAct extracts motion through spatiotemporal attention features, but motion signals become severely entangled in multi-object scenarios; FlexiMMT explicitly isolates them at the attention level via MDMA.
- **vs. Go-with-the-Flow**: Uses explicit optical flow for motion control, requiring a flow estimator and imposing geometric constraints on objects; FlexiMMT is an implicit method and thus more flexible.
- **vs. MotionDirector**: A LoRA-based motion decoupling approach for T2V that cannot handle the appearance-preservation requirements of I2V.
- **Insights**: Structured constraints at the attention level (rather than purely loss-based constraints) constitute an effective paradigm for achieving disentangled control.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First implicit multi-object multi-motion I2V framework; both problem formulation and solution are innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 200-pair evaluation, automatic and human metrics, detailed ablation study; dataset scale is somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐ Rigorous formal notation and clear overall structure.
- **Value**: ⭐⭐⭐⭐ Significant advancement for controllable video generation; opens a new direction for multi-motion transfer.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Anti-I2V: Safeguarding your photos from malicious image-to-video generation](anti-i2v_safeguarding_your_photos_from_malicious_image-to-video_generation.md)
- [\[CVPR 2026\] Rethinking Position Embedding as a Context Controller for Multi-Reference and Multi-Shot Video Generation](rethinking_position_embedding_as_a_context_controller_for_multi-reference_and_mu.md)
- [\[CVPR 2026\] MoVieDrive: Urban Scene Synthesis with Multi-Modal Multi-View Video Diffusion Transformer](moviedrive_multimodal_multiview_video_diffusion.md)
- [\[CVPR 2026\] SymphoMotion: Joint Control of Camera Motion and Object Dynamics for Coherent Video Generation](symphomotion_joint_control_of_camera_motion_and_object_dynamics_for_coherent_vid.md)
- [\[ICCV 2025\] Multi-identity Human Image Animation with Structural Video Diffusion](../../ICCV2025/video_generation/multi-identity_human_image_animation_with_structural_video_diffusion.md)

</div>

<!-- RELATED:END -->
