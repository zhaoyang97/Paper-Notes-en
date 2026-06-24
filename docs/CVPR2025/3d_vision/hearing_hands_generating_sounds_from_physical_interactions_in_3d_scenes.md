---
title: >-
  [Paper Note] Hearing Hands: Generating Sounds from Physical Interactions in 3D Scenes
description: >-
  [CVPR 2025][3D Vision][3D Scene Interaction] This paper proposes recording action-sound pairs of human hand interactions in 3D reconstructed scenes to train a rectified flow-based generative model. This achieves the prediction of corresponding interaction sounds from 3D hand trajectories, generating results that human evaluators cannot distinguish from real sounds in approximately 47% of cases.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Scene Interaction"
  - "Sound Generation"
  - "Hand Motion"
  - "Gaussian Splatting"
  - "Rectified Flow"
date: 2026-05-08
content_hash: c3eda4022b202f00
---

# Hearing Hands: Generating Sounds from Physical Interactions in 3D Scenes

**Conference**: CVPR 2025  
**arXiv**: [2506.09989](https://arxiv.org/abs/2506.09989)  
**Code**: [https://www.yimingdou.com/hearing_hands/](https://www.yimingdou.com/hearing_hands/) (Project Page)  
**Area**: 3D Vision  
**Keywords**: 3D Scene Interaction, Sound Generation, Hand Motion, Gaussian Splatting, Rectified Flow

## TL;DR

This paper proposes recording action-sound pairs of human hand interactions in 3D reconstructed scenes to train a rectified flow-based generative model. This achieves the prediction of corresponding interaction sounds from 3D hand trajectories, generating results that human evaluators cannot distinguish from real sounds in approximately 47% of cases.

## Background & Motivation

**Background**: Current 3D reconstruction methods (such as NeRF and Gaussian Splatting) primarily focus on the visual representation of static scenes. Some works have begun exploring scene interactivity, such as object joint motions or deforming visual dynamics, but these methods only focus on visual-level changes.

**Limitations of Prior Work**: Existing interactive 3D reconstruction neglects the key interaction modality of sound. Sound can convey physical properties that vision cannot directly express, such as whether a surface is hard or soft, smooth or rough, hollow or solid. On the other hand, video-to-audio generation methods require video inputs as conditions, preventing users from freely specifying the interactive actions they want to simulate.

**Key Challenge**: How to predict the sounds generated during operations at specific scene positions solely based on user-specified 3D hand motion trajectories, without actual video inputs? This requires the model to simultaneously understand the material properties of the scene and the temporal characteristics of the action.

**Goal**: (1) Build a sound dataset paired with 3D hand trajectories and scene visual information; (2) Train a generative model capable of producing sound conditioned on 3D hand motions; (3) Ensure the generated sounds accurately reflect material properties and action timing.

**Key Insight**: The authors observe a correlation between the visual appearance of a material and the sound produced when it is physically manipulated. By parameterizing hand actions as sequences of 3D hand poses and combining this with the visual information provided by Gaussian Splatting scene reconstruction, a mapping from action to sound can be established.

**Core Idea**: Use 3D hand trajectories and scene visual features as conditions to predict mel-spectrograms of interaction sounds via a rectified flow generative model.

## Method

### Overall Architecture

The system consists of two parts: (1) A visual neural field $F_\theta$ that reconstructs the 3D scene using Gaussian Splatting, mapping 3D points to RGB and depth; (2) An action-conditional sound generator $F_\phi$ that takes scene video $\mathbf{v}$ and hand motion $\mathbf{a}$ as inputs to generate the corresponding sound $\mathbf{s}$. During training, videos of real humans interacting with the scene by hand are first collected. The 3D hand poses are extracted and registered to the scene coordinate system. Then, scene reconstruction is used to synthesize interaction videos from different viewpoints to pair with the sounds.

### Key Designs

1. **3D Hand-Scene Data Collection and Augmentation Pipeline**:

    - **Function**: Build a 3D-registered hand motion-sound paired dataset.
    - **Mechanism**: The scene is first reconstructed using Gaussian Splatting, and then videos of human-hand interactions in the scene are recorded. HaMeR is used to detect 3D hand keypoints $\mathbf{a} \in \mathbb{R}^{2N \times 21 \times 3}$, and the interaction camera is registered to the scene coordinate system via COLMAP. Crucially, the detected 3D hands are projected onto the rendered views of the scene reconstruction to generate "simulated interaction videos" — including global views (with hand overlay) and local views (cropped and centered on the hand). This removes human body occlusions and achieves 3D-consistent data augmentation by synthesizing different viewpoints (top-down, side-view). A total of 24 scenes and 9.1 hours of interaction data were collected.
    - **Design Motivation**: Directly using the original videos would introduce human body occlusions and viewpoint limitations. Through 3D registration + re-rendering, clean visual inputs can be obtained, and multi-view augmentation can be performed.

2. **Rectified Flow-based Sound Generation Model**:

    - **Function**: Generate sound spectrograms from visual + action conditions.
    - **Mechanism**: Built upon the Frieren video-to-sound model, but with two key modifications. First, CLIP is used instead of CAVP to encode video features, as CLIP provides better spatial consistency and material understanding. Second, 3D hand motion $\mathbf{a}$ is explicitly injected into the model as an additional condition—the hand pose is encoded into the same dimension as the frame embeddings via a linear layer, normalized to a unit vector, upsampled to the same temporal frequency as the spectrogram (31.25Hz), and finally element-wise added with the visual embeddings to serve as the condition vector.
    - **Design Motivation**: The original Frieren performs poorly on synthetic interaction videos because simulated videos lack the low-level details of real videos. CLIP provides material information, and the 30Hz hand pose provides high-resolution action details, complementing each other.

3. **Dual-stream Visual Encoding (Global + Local Views)**:

    - **Function**: Capture both the overall scene layout and material details of the interaction area.
    - **Mechanism**: The global video $\mathbf{v}_g$ (scene rendering with hand overlay) and local video $\mathbf{v}_l$ (close-up cropped around the hand center) are independently encoded by CLIP. The two resulting feature vectors are concatenated and fed into the model. Video frames are downsampled to 4Hz before being fed into CLIP.
    - **Design Motivation**: The global view provides the position of the hand in the scene and the overall context, whereas the local view provides the material texture details of the contact region. Combining both allows for accurate prediction of the sound type.

### Loss & Training

The standard training objective of rectified flow matching is used. The model is trained from scratch for 40 epochs with a batch size of 128 using the Adam optimizer. The learning rate warm-starts from $10^{-5}$ to $4 \times 10^{-4}$ and then linearly decays to $3.4 \times 10^{-4}$. During inference, a 26-step sampling is conducted with a guidance scale of 4.5. A pre-trained vocoder is used to convert the spectrograms into waveforms.

## Key Experimental Results

### Main Results

| Method | STFT ↓ | Envelope ↓ | CLAP-acc All ↑ | CLAP-acc Action ↑ | CLAP-acc Material ↑ | Realness (%) |
|------|--------|-----------|----------------|-------------------|---------------------|-----------|
| RegNet | 0.62 | 0.77 | 1.08 | 42.55 | 3.52 | - |
| Frieren | 0.74 | 0.81 | 23.94 | 41.73 | 42.55 | 43.79±2.64 |
| **Ours** | **0.50** | **0.66** | **28.09** | **50.50** | **45.62** | **47.18±2.66** |

### Ablation Study

| Configuration | STFT ↓ | Envelope ↓ | CLAP-acc All ↑ | CLAP-acc Material ↑ | CLAP-acc Action ↑ |
|------|--------|-----------|----------------|---------------------|-------------------|
| Full model | 0.50 | 0.66 | 28.09 | 45.62 | 50.50 |
| w/o CLIP | 0.68 | 0.77 | 18.25 | **31.80** (-13.82) | 43.90 |
| w/o hand pose| 0.69 | 0.77 | 20.96 | 39.11 | **38.21** (-12.29) |
| w/o synthetic-view | 0.62 | 0.73 | 24.12 | 40.56 | 47.61 |

### Key Findings

- Removing CLIP features leads to the largest drop in CLAP material accuracy ($45.62 \rightarrow 31.80$), suggesting that CLIP primarily provides material information.
- Removing hand pose leads to the largest drop in CLAP action accuracy ($50.50 \rightarrow 38.21$), suggesting that hand pose primarily encodes action timing.
- In the real-vs-fake study, the proposed method's 47.18% misclassification rate is close to the 50% random baseline, indicating that the generated sounds are almost indistinguishable from real sounds.
- The advantages are particularly pronounced on rough surfaces and soft materials.

## Highlights & Insights

- **3D Scene Interaction with Single-step Sound Extraction**: Integrating sound generation with 3D reconstruction allows users to "audition" different objects in a scene by specifying hand trajectories, which is a novel AR/VR application direction. The brilliance lies in using existing hand detection and scene reconstruction technologies to build a fully automated data collection pipeline.
- **3D-consistent Data Augmentation**: Utilizing 3D reconstruction to eliminate human body occlusion and synthesizing multi-view training data. This idea of leveraging 3D geometric prior for data augmentation can be transferred to other tasks requiring viewpoint diversity.
- **Insights on CLIP Replacing CAVP**: Discovering that a general-purpose vision-language model (CLIP) outperforms a specialized audio-visual alignment model (CAVP) in material understanding implies that material understanding relies more on visual semantics than on audio-visual alignment.

## Limitations & Future Work

- Assumes that objects in the scene do not move or deform when manipulated, which is frequently violated when manipulating small objects.
- Relies on the accuracy of 3D hand detection models; detection errors will lead to inaccurate hand movements in the dataset.
- Only trained on 24 scenes, which may limit generalization to completely different scenes and materials.
- Does not account for spatial audio propagation effects (e.g., reverberation, distance attenuation); this could be integrated with acoustic reconstruction work.

## Related Work & Insights

- **vs ObjectFolder**: ObjectFolder builds object-level multimodal representations but can only handle rigid small objects and simple impact sounds, whereas this work handles scene-level reconstruction and supports complex hand movements.
- **vs Diff-Foley/Frieren**: They generate sound from video, whereas this work generates sound from 3D hand trajectories, requiring no video input and achieving cleaner material views through 3D constraints.
- **vs Tactile-augmented radiance fields**: Previous work by the same author dealt with tactile feedback, whereas this work deals with sound, and sound is not an inherent property of the surface but a function of the action.

## Rating

- Novelty: ⭐⭐⭐⭐ First to introduce sound generation into interactive 3D scene reconstruction, presenting a new problem formulation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Automated metrics combined with large-scale human evaluation; the ablation design is reasonable.
- Writing Quality: ⭐⭐⭐⭐ Clear problem motivation and highly detailed pipeline description.
- Value: ⭐⭐⭐⭐ Has direct application value in AR/VR and robotics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HOI3DGen: Generating High-Quality Human-Object-Interactions in 3D](hoi3dgen_generating_high-quality_human-object-interactions_in_3d.md)
- [\[ICCV 2025\] Bolt3D: Generating 3D Scenes in Seconds](../../ICCV2025/3d_vision/bolt3d_generating_3d_scenes_in_seconds.md)
- [\[CVPR 2025\] Generating 3D-Consistent Videos from Unposed Internet Photos](generating_3d-consistent_videos_from_unposed_internet_photos.md)
- [\[CVPR 2025\] Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions](grounding_3d_object_affordance_with_language_instructions_visual_observations_an.md)
- [\[CVPR 2025\] Material Anything: Generating Materials for Any 3D Object via Diffusion](material_anything_generating_materials_for_any_3d_object_via_diffusion.md)

</div>

<!-- RELATED:END -->
