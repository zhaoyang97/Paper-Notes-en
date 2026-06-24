---
title: >-
  [Paper Note] S³D-NeRF: Single-Shot Speech-Driven Neural Radiance Field for High Fidelity Talking Head Synthesis
description: >-
  [ECCV 2024][3D Vision][Talking Head Synthesis] This paper proposes S³D-NeRF, a NeRF-based method that leverages a hierarchical facial appearance encoder, a cross-modal facial deformation field, and a lip-sync discriminator to synthesize high-fidelity talking head videos driven by speech using only a single source image, outperforming existing single-shot methods in video quality and lip synchronization.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Talking Head Synthesis"
  - "Neural Radiance Fields"
  - "Speech-Driven"
  - "Single-Shot Driven"
  - "Lip Synchronization"
date: 2026-05-08
content_hash: 3a64b4953e1f86a1
---

# S³D-NeRF: Single-Shot Speech-Driven Neural Radiance Field for High Fidelity Talking Head Synthesis

**Conference**: ECCV 2024  
**arXiv**: [2408.09347](https://arxiv.org/abs/2408.09347)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Talking Head Synthesis, Neural Radiance Fields, Speech-Driven, Single-Shot Driven, Lip Synchronization

## TL;DR
This paper proposes S³D-NeRF, a NeRF-based method that leverages a hierarchical facial appearance encoder, a cross-modal facial deformation field, and a lip-sync discriminator to synthesize high-fidelity talking head videos driven by speech using only a single source image, outperforming existing single-shot methods in video quality and lip synchronization.

## Background & Motivation

**Background**: Talking Head Synthesis is an important task in computer vision. Current NeRF-based methods (such as AD-NeRF, ER-NeRF) perform excellently in driving talking head videos, but most are identity-specific models, requiring long training times for each new identity and typically relying on intermediate representations (such as 3DMM coefficients) rather than directly using audio signals.

**Limitations of Prior Work**: (1) Identity-specific NeRF methods suffer from poor generalization, requiring retraining for any new identity, which incurs high training costs of up to dozens of hours; (2) Existing single-shot driving methods (such as SadTalker, StyleTalk), while generalizing well, yield unsatisfactory image quality and lip accuracy; (3) Most methods indirectly lease or map audio signals through intermediate representations (3DMM, keypoints, etc.), introducing information loss.

**Key Challenge**: Directly mapping from audio to facial deformation is non-trivial — audio signals are strongly correlated only with the lip region and weakly correlated with other facial regions (eyebrows, eyes), but talking head synthesis requires coordinated movement of the entire face. This leads to a conflict between "local signals vs. global motion."

**Goal**: (1) How to learn sufficiently expressive identity appearance features from a single image? (2) How to accurately model the motion of different facial regions based on audio signals? (3) How to ensure temporal consistency in the lip region?

**Key Insight**: The authors observe that the correlation between different facial regions and speech signals varies significantly — the lip region has the strongest correlation, while the upper face is barely affected by audio. Utilizing a cross-attention mechanism to compute audio-visual correlation scores can serve as a prior to guide facial deformation prediction, achieving precise region-aware animation.

**Core Idea**: Build the first speech-driven generalizable NeRF talking head method from a single image by combining a multi-scale tri-plane representation, an audio-visual cross-attention deformation field, and a lip-sync discriminator.

## Method

### Overall Architecture
The input consists of a single-frame source image and a driving audio sequence, and the output is a high-fidelity multi-view talking head video. The overall pipeline is divided into three stages: (1) A hierarchical facial appearance encoder extracts multi-scale features from the source image and constructs a tri-plane representation; (2) A cross-modal facial deformation field predicts the displacement of 3D points based on the correlation between audio signals and visual features; (3) After volume rendering generates a coarse facial image, a super-resolution module adds details and background.

### Key Designs

1. **Hierarchical Facial Appearance Encoder**:

    - **Function**: Extracts multi-scale features from a single source image and constructs an efficient tri-plane representation to model the appearance of arbitrary speakers.
    - **Mechanism**: A Feature Pyramid Network (FPN) structure is adopted to extract feature maps $\mathbf{D}_0$ to $\mathbf{D}_3$ across four scales via a downsampling convolutional network, which are then upsampled and concatenated to obtain hierarchical features $\mathbf{F}_i$. The feature map at each scale is reshaped into three orthogonal sub-planes ($\mathbf{F}_{xy}$, $\mathbf{F}_{yz}$, $\mathbf{F}_{xz}$), forming a multi-scale tri-plane representation. During rendering, the camera transformations between the source pose and the target pose are used to locate the projection positions of 3D points on the tri-planes, and feature vectors are extracted using bilinear interpolation.
    - **Design Motivation**: A single-scale feature map cannot simultaneously capture coarse-grained structural information and fine-grained texture details. The multi-scale tri-plane representation allows features of different resolutions to complement each other, enabling accurate modeling of unseen speakers' facial appearances even with only a single image.

2. **Cross-modal Facial Deformation Field**:

    - **Function**: Predicts 3D deformations of various facial regions based on speech signals to achieve precise voice-driven animation.
    - **Mechanism**: First, multi-scale visual features are aggregated into a unified visual embedding $\mathbf{F}_{agg}$ via Slot Attention, while a 1D fully convolutional network processes the speech signal to obtain audio features $\mathbf{a}_{dri}$. Multi-head cross-attention (MHCA) is employed to compute correlation scores between the visual embedding (query) and the audio features (key/value): $\mathbf{F}_{cm} = \text{MHCA}(\mathbf{F}_{agg}, \mathbf{a}_{dri})$. This correlation score serves as upfront knowledge and is input into a U-Net structured deformation prediction module, which predicts the displacement: $\Delta\mathbf{x} = \text{Deform}(\mathbf{x}, \mathbf{a}_{dri}, \mathbf{F}_{cm})$ combining 3D positions and audio features.
    - **Design Motivation**: Naive deformation modules that directly predict full-face deformation from audio tend to yield blurry results because audio is strongly correlated only with the mouth region. The cross-attention heatmap shows the highest activation in the lower face area, validating that this region-aware design correctly distributes motion magnitudes across different regions.

3. **Lip-sync Discriminator**:

    - **Function**: Constrains the lip movements in the generated video to remain temporally synchronized with the driving audio.
    - **Mechanism**: The discriminator consists of visual and audio branches, extracting embeddings $\mathbf{e}_l$ and $\mathbf{e}_a$ from $T$ frames of the lip region and audio clips respectively, and determining synchronization via cosine similarity $\cos(\mathbf{e}_l, \mathbf{e}_a)$. Contrastive triplet loss is used during training, pre-trained on a larger-scale hybrid dataset (HDTF+LRS2). The generator's lip-sync loss is $\mathcal{L}_{sync}^{gen} = \cos(\mathbf{e}_l^{gen}, \mathbf{e}_{gt})$, with gradients backpropagated after freezing the discriminator.
    - **Design Motivation**: Videos generated by NeRF lack explicit constraints on lip temporal consistency, which can lead to frames with incorrect lip shapes. Introducing a lip-sync discriminator as external supervision provides stronger discriminative power than Wav2Lip's sync expert.

### Loss & Training
The total loss is $\mathcal{L} = \mathcal{L}_{pix} + 0.01\mathcal{L}_{per} + \mathcal{L}_{adv} + 0.5\mathcal{L}_{sync}^{gen} + 0.001\mathcal{L}_{deform}$, which includes pixel-level L2 reconstruction loss, perceptual loss, adversarial loss from the StyleGAN discriminator, lip-sync loss, and deformation regularization loss. A coarse-to-fine strategy is adopted: first, a coarse image of the inner face region is rendered via NeRF, after which a super-resolution module adds details for the outer face and background, alleviating the difficulty of NeRF modeling multiple textures simultaneously.

## Key Experimental Results

### Main Results

**Comparison with Single-shot Methods (HDTF Dataset)**:

| Method | SSIM↑ | LPIPS↓ | F-LMD↓ | M-LMD↓ | CPBD↑ | Sync↑ |
|------|-------|--------|--------|--------|-------|-------|
| Wav2Lip | 0.749 | 0.332 | 3.582 | 3.652 | 0.195 | 8.365 |
| SadTalker | 0.776 | 0.289 | 3.623 | 3.273 | 0.234 | 6.188 |
| StyleTalk | 0.678 | 0.386 | 4.621 | 3.654 | 0.192 | 6.473 |
| **S³D-NeRF** | **0.819** | **0.258** | **2.799** | **2.929** | **0.263** | 6.514 |

**Comparison with NeRF Methods (Videos Provided by AD-NeRF)**:

| Method | SSIM↑ | M-LMD↓ | Sync↑ | FPS | Fit Time |
|------|-------|--------|-------|-----|----------|
| AD-NeRF | 0.846 | 1.982 | 5.626 | 0.13 | 36h |
| ER-NeRF | 0.884 | 1.659 | 6.781 | 29 | 4.5h |
| **S³D-NeRF** | 0.852 | **1.493** | **7.118** | 9.5 | **<0.01h** |

### Ablation Study

| Configuration | SSIM↑ | LPIPS↓ | F-LMD↓ | M-LMD↓ | Sync↑ |
|------|-------|--------|--------|--------|-------|
| w/o Deform | 0.556 | 0.457 | N/A | N/A | 0.240 |
| Naive Deform (ND) | 0.744 | 0.278 | 3.475 | 3.507 | 5.201 |
| Feat-Concatenate Deform | 0.834 | 0.242 | 2.855 | 3.196 | 6.106 |
| w/o Lip Sync | 0.801 | 0.263 | 3.276 | 3.342 | 5.649 |
| **S³D-NeRF (full)** | **0.829** | **0.258** | **2.799** | **2.929** | **6.514** |

### Key Findings
- The cross-modal deformation field contributes the most: removing the deformation module makes it entirely impossible to generate talking animation, and Naive Deform yields a blurry average-face effect.
- Using cross-attention correlation scores as a prior significantly outperforms the feature direct concatenation (FCD) approach, improving the Sync score by approximately 6.7%.
- The lip-sync discriminator brings about a 15% improvement in Sync and a prominent reduction in mouth landmark distance.
- S³D-NeRF holds an absolute advantage in generalization: it requires no retraining for a new identity (Fit Time < 0.01h), whereas identity-specific NeRF methods require several hours to dozens of hours.

## Highlights & Insights
- **Cross-attention correlation score as a deformation prior**: This is a highly ingenious design — using attention heatmaps to directly reflect the degree of correlation between the audio and different facial regions. This acts as a spatial guideline indicating "where to move and where not to move" for the deformation network, solving the core challenge of audio being a local signal that is hard to drive global motion.
- **Multi-scale tri-plane representation generalizes to new identities**: Compared to parametric representations such as 3DMM, constructing a tri-plane representation directly from raw images preserves more identity details. The multi-scale design provides rich enough appearance information even from a single image.
- **The coarse-to-fine decoupled rendering strategy** can be transferred to other NeRF tasks — modeling key regions and background regions separately reduces the learning difficulty of NeRF.

## Limitations & Future Work
- Does not support background replacement — the background of the source image is fixed.
- Facial contours tend to blur when the head pose is too large.
- Rendering speed is 9.5 FPS. Although superior to AD-NeRF, it is far slower than ER-NeRF's 29 FPS.
- Does not use more efficient 3D representations such as 3D Gaussian Splatting, which could offer further potential for improvements in speed and quality.
- The training dataset (HDTF) is limited in scale; expanding the data scale or introducing pre-training might further enhance cross-identity generalization capabilities.

## Related Work & Insights
- **vs AD-NeRF/ER-NeRF**: These methods are identity-specific NeRFs, achieving extremely high image quality but requiring individual training for each identity. S³D-NeRF sacrifices a minor amount of image quality to achieve extremely strong generalization capability (Fit Time < 0.01h vs 4.5~36h).
- **vs SadTalker/StyleTalk**: These are single-shot methods based on 3DMM intermediate representations. S³D-NeRF bypasses intermediate representations to drive NeRF directly from audio, leading comprehensively in image quality (SSIM +5%) and mouth accuracy (F-LMD 23%↓).
- **vs Wav2Lip**: Wav2Lip performs exceptionally well in lip synchronization (Sync > GT), but the generated faces are very blurry (high LPIPS, low CPBD). S³D-NeRF shows a distinct advantage in clarity.

## Rating
- Novelty: ⭐⭐⭐⭐ The first single-shot speech-driven generalizable NeRF talking head method, featuring an innovative cross-attention deformation field design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Compares comprehensively against both single-shot and NeRF methods, with ablation studies covering all components.
- Writing Quality: ⭐⭐⭐⭐ Clear problem decomposition and well-structured method descriptions.
- Value: ⭐⭐⭐⭐ Achieves an excellent trade-off between generalization and quality, offering practical application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] TalkingGaussian: Structure-Persistent 3D Talking Head Synthesis via Gaussian Splatting](talkinggaussian_structure-persistent_3d_talking_head_synthesis_via_gaussian_spla.md)
- [\[ECCV 2024\] MALD-NeRF: Taming Latent Diffusion Model for Neural Radiance Field Inpainting](taming_latent_diffusion_model_for_neural_radiance_field_inpainting.md)
- [\[ECCV 2024\] Dynamic Neural Radiance Field from Defocused Monocular Video](dynamic_neural_radiance_field_from_defocused_monocular_video.md)
- [\[ECCV 2024\] Mesh2NeRF: Direct Mesh Supervision for Neural Radiance Field Representation and Generation](mesh2nerf_direct_mesh_supervision_for_neural_radiance_field_representation_and_g.md)
- [\[ECCV 2024\] Invertible Neural Warp for NeRF](invertible_neural_warp_for_nerf.md)

</div>

<!-- RELATED:END -->
