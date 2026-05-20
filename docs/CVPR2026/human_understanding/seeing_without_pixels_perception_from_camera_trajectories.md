---
title: >-
  [Paper Note] Seeing without Pixels: Perception from Camera Trajectories
description: >-
  [CVPR 2026][Human Understanding][camera trajectory] This paper is the first to systematically elevate camera pose trajectories (6DoF pose sequences) to an independent modality for video perception. Through a contrastive…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "camera trajectory"
  - "contrastive learning"
  - "video perception"
  - "modality fusion"
  - "action understanding"
date: 2026-05-08
content_hash: d29c002351d6c36f
---

# Seeing without Pixels: Perception from Camera Trajectories

**Conference**: CVPR 2026
**arXiv**: [2511.21681](https://arxiv.org/abs/2511.21681)  
**Code**: [https://sites.google.com/view/seeing-without-pixels](https://sites.google.com/view/seeing-without-pixels)  
**Area**: Human Understanding / Multimodal Learning
**Keywords**: camera trajectory, contrastive learning, video perception, modality fusion, action understanding

## TL;DR

This paper is the first to systematically elevate camera pose trajectories (6DoF pose sequences) to an independent modality for video perception. Through a contrastive learning framework, a lightweight Transformer encoder, CamFormer, is trained to map camera trajectories into a joint embedding space aligned with text. Across 10 downstream tasks on 5 datasets, the paper demonstrates that camera trajectories serve as a lightweight and robust signal for video content understanding—even surpassing video models requiring thousands of times more computation on physical activity tasks.

## Background & Motivation

1. **Background**: Video understanding has explored a broad range of modalities—visual, audio, IMU, thermal, depth, and tactile—aligned with text via contrastive learning. Yet camera pose trajectories have consistently been overlooked as a semantic perception signal, confined to geometric tasks such as 3D reconstruction and visual odometry.
2. **Limitations of Prior Work**: Visual encoders are computationally intensive (e.g., EgoVLPv2 requires approximately 89.5 GMACs) and are limited in scenarios involving visual occlusion or out-of-view actions. Sensors such as IMUs require dedicated hardware and cannot be retrospectively obtained from existing videos.
3. **Key Challenge**: Camera trajectories are an intrinsic property of any video and can be estimated directly from it, yet they have long been considered too low in information density (only a 9D vector per frame) and too semantically ambiguous to support video content understanding.
4. **Goal**: To validate a seemingly counterintuitive hypothesis—that video content can be understood from camera motion trajectories alone, without any pixel information.
5. **Key Insight**: Human perception is active—we move in order to see, and camera trajectories serve as physical fingerprints of the cameraperson's intent. A basketball layup is accompanied by an upward tilt; lifting a tire involves a top-down sweep; walking produces rhythmic forward-and-backward oscillation—these are semantic motion signatures.
6. **Core Idea**: Contrastive learning is used to map low-dimensional camera trajectories into a textual semantic space, demonstrating that *how the camera moves* genuinely reveals *what is happening*.

## Method

### Overall Architecture

The input is a camera pose sequence $\mathbf{p} \in \mathbb{R}^{N \times 9}$ corresponding to a video clip (3D translation + 6D continuous rotation representation, relative to the sequence midpoint), paired with a text description (action narration or video caption). CamFormer encoder $f$ is trained via contrastive learning to align trajectory embeddings with the output of a frozen CLIP text encoder $g$. The learned embeddings can be directly applied to a variety of downstream tasks including retrieval, classification, and temporal analysis.

### Key Designs

1. **CamFormer Trajectory Encoder**:

    - **Function**: Encodes low-dimensional pose sequences into semantically rich embedding vectors.
    - **Mechanism**: A lightweight Transformer (4 layers, 4 heads, 256-dim FFN, dropout 0.1, only 0.3M parameters). The input 9D pose sequence is first linearly projected to $d_{in}=128$ dimensions, followed by positional encoding and Transformer blocks for temporal fusion. Temporal mean pooling is then applied, and a final linear projection maps the result to $d_{out}=512$ dimensions (matching the CLIP text dimension).
    - **Design Motivation**: Camera trajectories are inherently low-dimensional and sparse, so the encoder does not require large model capacity. At 0.3M parameters and 0.02 GMACs, it operates three orders of magnitude lighter than video encoders (150M parameters, 89.5 GMACs).

2. **Contextualized Trajectory Encoding**:

    - **Function**: Resolves semantic ambiguity in short-window trajectories.
    - **Mechanism**: A base temporal window $[t_1, t_2]$ is randomly extended by a total of $w$ seconds of context on both sides, where $w \sim \mathcal{U}(0, w_{max})$ and $w_{max}=8\text{s}$. The entire extended sequence is passed into CamFormer, but the final embedding is computed by mean-pooling only the $N$ output tokens corresponding to the original window. This injects global context into the local representation without dilution from adjacent unrelated actions.
    - **Design Motivation**: A 1-second trajectory may correspond to multiple semantics (e.g., "reaching out" could mean picking up a cup or opening a door); extended context disambiguates such cases.

3. **Contrastive Learning Training Strategy**:

    - **Function**: Learns cross-modal alignment between trajectories and text.
    - **Mechanism**: Standard bidirectional InfoNCE loss $\mathcal{L} = \mathcal{L}_{P \to T} + \mathcal{L}_{T \to P}$, where positive samples within a batch are matched (trajectory, text) pairs and all others serve as negatives. The text encoder $g$ uses a frozen CLIP model, providing fixed semantic anchors.
    - **Design Motivation**: Reusing the strong semantic space already established by CLIP allows CamFormer to focus solely on aligning trajectories to that space.

### Loss & Training

The training loss is a bidirectional InfoNCE contrastive loss with temperature hyperparameter $\tau$; the text encoder is fully frozen throughout training. For the egocentric domain, pretraining is conducted on Ego-Exo4D (221.3h); for the third-person domain, on DynPose-100K (157.5h). Pose sampling rates range from 5–30 Hz depending on the dataset.

## Key Experimental Results

### Main Results

**Egocentric Text Retrieval (5-way MCQ, Ego-Exo4D)**

| Method | Modality | GMACs | Params | Physical iv/oov | Procedural iv/oov | Overall |
|--------|----------|-------|--------|-----------------|-------------------|---------|
| CLIP | Image | 2.95 | 59M | 25.2/18.2 | 26.8/21.9 | 22.9 |
| EgoVLPv2 (Ego-Exo4D) | Video | 89.49 | 150.7M | 39.1/25.6 | 50.5/45.4 | 38.4 |
| CamFormer | Trajectory | **0.02** | **0.3M** | **56.1/46.4** | 34.3/32.7 | 44.8 |
| CamFormer⋆ | Video+Trajectory | 89.51 | 151M | 56.0/45.8 | **51.4/45.9** | **46.0** |

**Activity Classification Accuracy (Ego-Exo4D)**

| Activity | CamFormer Accuracy |
|----------|--------------------|
| Basketball | >90% |
| Rock climbing | >90% |
| Cooking | Lower (procedural activity) |

### Ablation Study

| Pose Source | Activity Cls. (scratch) | Activity Cls. (pretrained) | Gain |
|-------------|------------------------|---------------------------|------|
| MegaSaM | 53.67 | 60.83 | +7.16 |
| ViPE | 60.83 | 66.15 | +5.32 |
| π³ | 61.47 | 66.15 | +4.68 |
| Aria (hardware) | 61.83 | 71.28 | +9.45 |

### Key Findings
- **Physical vs. Procedural Activities**: CamFormer achieves >90% accuracy on large-scale physical activities such as basketball and rock climbing, significantly surpassing video-based models; however, for fine-grained procedural activities such as cooking and repair, motion signatures are weak, and trajectories are better used as a complementary signal.
- **Out-of-View Actions**: CamFormer's advantage is particularly pronounced when the action is not visible in the egocentric frame (oov)—e.g., detecting a "landing" is difficult from video frames but is unambiguously reflected in a descending trajectory.
- **Zero-Shot Cross-Dataset Generalization**: CamFormer pretrained on Ego-Exo4D, when applied directly to Nymeria, achieves 31.6% accuracy (chance = 20%), substantially outperforming video baselines on unseen categories such as *legs* and *focus attention*.
- **Estimated Poses Are Viable**: Although hardware poses from Aria yield the best results, RGB-only pose estimators (MegaSaM/ViPE/π³) also function effectively, demonstrating practical applicability.
- **Third-Person Domain Also Effective**: On DynPose-100K third-person text retrieval, CamFormer (36.2%) outperforms LMM-based baselines such as ShotVL (33.1%).

## Highlights & Insights
- The premise of *"perceiving without pixels"* is itself highly thought-provoking. A 0.3M-parameter, 0.02-GMACs micro-model outperforming a 150M-parameter, 89.5-GMACs video model on physical activities indicates that motion intent signals have been severely underestimated.
- **Contextualized encoding** is a general-purpose technique for handling low-information-density modalities—extending the input window while pooling only the target window's outputs—and is directly transferable to sparse modalities such as IMU and audio.
- The fusion strategy for **trajectory as a complementary modality** is minimal—simply averaging feature vectors—yet consistently yields gains, indicating high complementarity and near-zero redundancy between trajectory and visual features.
- Camera trajectories possess unique advantages as a modality: they can be retrospectively estimated from any video, require no dedicated hardware, are privacy-friendly (pixel-free), and incur negligible computational cost.

## Limitations & Future Work
- Trajectory signals are weak for procedural activities (cooking/repair), requiring visual information to achieve strong performance.
- Only a Transformer encoder architecture with InfoNCE loss has been explored; other architectures and training objectives (e.g., MAE self-supervision) warrant investigation.
- Pose estimation errors degrade downstream performance; high-quality hardware poses (Aria) outperform estimated poses by 5–10 points.
- Deep integration with LLMs/VLMs—such as using trajectory embeddings as additional input tokens to a VLM—has not yet been explored.

## Related Work & Insights
- **vs. PRIMUS (IMU-text contrastive learning)**: PRIMUS achieves only 23.2% on Ego-Exo4D retrieval versus CamFormer's 44.8%. Although IMU also captures motion, it operates at higher sampling rates, introduces more noise, and requires dedicated hardware.
- **vs. CLIP/EgoVLPv2 (vision-text)**: CamFormer surpasses an 89.5-GMACs video model with only 0.02 GMACs on physical activities; however, video modality retains an advantage on procedural activities, and their combination yields the best overall results.
- **vs. CameraBench/ShotVL (camera motion description generation)**: These LMM-based methods describe camera motion as a video attribute (e.g., "zoom"/"pan"), whereas CamFormer directly interprets trajectories as semantic signals—with superior performance.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic study to treat camera trajectories as an independent perception modality; entirely novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five datasets, 10 tasks, multiple pose source comparisons, full coverage of egocentric and third-person domains.
- Writing Quality: ⭐⭐⭐⭐⭐ Experimental sections organized in a Q&A format, highly engaging, with well-designed figures and tables.
- Value: ⭐⭐⭐⭐⭐ Introduces a lightweight, robust, and privacy-friendly new modality for video understanding with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Face Time Traveller: Travel Through Ages Without Losing Identity](face_time_traveller_travel_through_ages_without_losing_identity.md)
- [\[NeurIPS 2025\] Cycle-Sync: Robust Global Camera Pose Estimation through Enhanced Cycle-Consistent Synchronization](../../NeurIPS2025/human_understanding/cycle-sync_robust_global_camera_pose_estimation_through_enhanced_cycle-consisten.md)
- [\[CVPR 2026\] FSMC-Pose: Frequency and Spatial Fusion with Multiscale Self-calibration for Cattle Mounting Pose Estimation](fsmc-pose_frequency_and_spatial_fusion_with_multiscale_selfcalibration_for_cattle.md)
- [\[CVPR 2026\] 4DSurf: High-Fidelity Dynamic Scene Surface Reconstruction](textit4dsurf_high-fidelity_dynamic_scene_surface_reconstruction.md)
- [\[CVPR 2026\] How to Take a Memorable Picture? Empowering Users with Actionable Feedback](how_to_take_a_memorable_picture_empowering_users_with_actionable_feedback.md)

</div>

<!-- RELATED:END -->
