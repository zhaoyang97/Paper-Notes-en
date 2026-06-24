---
title: >-
  [Paper Note] SoundVista: Novel-View Ambient Sound Synthesis via Visual-Acoustic Binding
description: >-
  [CVPR 2025][3D Vision][Novel-view acoustic synthesis] SoundVista proposes a method for synthesizing ambient sound from sparse distributed microphone recordings at arbitrary novel views. By leveraging a Visual-Acoustic Binding (VAB) module to infer acoustic properties from panoramic RGB-D data, the method optimizes reference microphone layouts and adaptively weights the contributions of reference recordings using a Transformer, significantly outperforming existing methods in b…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Novel-view acoustic synthesis"
  - "visual-acoustic binding"
  - "binaural audio"
  - "spatial audio rendering"
  - "ambient sound field"
date: 2026-05-08
content_hash: 326ccabc326a9b86
---

# SoundVista: Novel-View Ambient Sound Synthesis via Visual-Acoustic Binding

**Conference**: CVPR 2025  
**arXiv**: [2504.05576](https://arxiv.org/abs/2504.05576)  
**Code**: None  
**Area**: 3D Vision / Audio Synthesis  
**Keywords**: Novel-view acoustic synthesis, visual-acoustic binding, binaural audio, spatial audio rendering, ambient sound field

## TL;DR

SoundVista proposes a method for synthesizing ambient sound from sparse distributed microphone recordings at arbitrary novel views. By leveraging a Visual-Acoustic Binding (VAB) module to infer acoustic properties from panoramic RGB-D data, the method optimizes reference microphone layouts and adaptively weights the contributions of reference recordings using a Transformer, significantly outperforming existing methods in both simulated and real-world scenes.

## Background & Motivation

**Background**: Along with the rapid development of visual novel-view synthesis driven by technologies like 3DGS and NeRF, novel-view acoustic synthesis (NVAS) in the audio field has lagged significantly behind. Most existing NVAS methods simplify the task by focusing on only 1 or 2 fixed sound sources (e.g., speech or musical instruments), ignoring the contribution of other environmental sounds to the natural sound field.

**Limitations of Prior Work**: (1) Traditional acoustic methods (such as RIR convolution) require precise clean signals and locations for each sound source, which are unavailable in real-world scenarios; (2) Deep learning-based NVAS methods rely on heuristic reference microphone placements (nearest neighbor, random, or fixed), which cannot adapt to diverse room layouts; (3) How to weight the contributions of multiple reference microphones remains an open question, and simple distance-based weighting is unreliable due to obstacles; (4) Existing methods generalize poorly when facing complex scenarios (e.g., multiple sound sources, large spaces, multi-room layouts).

**Key Challenge**: Synthesizing authentic ambient sound requires solving two difficult problems simultaneously: learning sound field transformations without prior knowledge of individual sound sources, and optimizing the placement and contributions of reference microphones under a limited budget.

**Goal**: (1) Learn the transfer function from reference recordings to target view audio directly without relying on sound source details; (2) Utilize visual information to infer acoustic properties to optimize reference position sampling; (3) Adaptively weight the contributions of multiple reference microphones.

**Key Insight**: The authors observe that visual data (panoramic RGB-D) contains information highly correlated with acoustic properties: depth reveals obstacles and room geometry, while texture suggests material differences. By aligning visual and acoustic features, acoustic properties can be inferred solely from vision.

**Core Idea**: Visual-Acoustic Binding (VAB) is used to align RGB-D visual features with the acoustic features of echo responses. The VAB embeddings are then utilized to optimize reference locations and adaptively weight reference contributions, avoiding dependency on explicit source information.

## Method

### Overall Architecture

SoundVista consists of four modules: (1) The VAB module pre-trains a visual encoder to align panoramic RGB-D features with RT60 acoustic parameters; (2) The reference location sampler utilizes VAB embeddings to cluster candidate locations and selects representative locations for putting the reference microphones; (3) The reference integration Transformer calculates attention weights based on the VAB embeddings of target and reference locations, determining the contribution scale of each reference recording; (4) The spatial audio renderer generates binaural audio for the target viewpoint using the weighted reference recordings and conditional information.

### Key Designs

1. **Visual-Acoustic Binding (VAB) Module**:

    - **Function**: Infers the acoustic properties of locations from panoramic RGB-D images without physical acoustic measurements.
    - **Mechanism**: A large amount of paired panoramic RGB-D and Room Impulse Response (RIR) data is gathered in the SoundSpaces simulator, with RT60 (reverberation time) extracted as the acoustic representation. A ResNet-18 visual encoder $\phi(\cdot)$ is trained to predict RT60. RT60 measures the time required for sound energy to decay by 60dB, and its variations reflect obstacles and surface material differences. A joint RGB+Depth input yields the best performance, reducing the prediction error by more than 50% compared to using spatial location information alone.
    - **Design Motivation**: Acquiring acoustic parameters (RIR) in real-world scenarios is difficult and expensive, whereas RGB-D data is easy to obtain. Through pre-trained alignment, acoustic properties can be inferred at inference time using only visual inputs.

2. **Reference Location Sampler**:

    - **Function**: Automatically selects optimal reference microphone placements within a limited budget (N microphones).
    - **Mechanism**: VAB embeddings are extracted for all candidate locations in the scene (panoramic images can be rendered using NeRF/3DGS without physical shooting). The VAB embeddings are combined with spatial location coordinates for clustering. Each cluster represents an acoustic zone—a region with similar acoustic properties that is not obviously partitioned by obstacles. The centroid of each cluster is selected as a reference location.
    - **Design Motivation**: Ideal placement should cover different acoustic zones of a scene. VAB embeddings inherently pre-encode acoustic property-related information, yielding more accurate results than clustering based strictly on Euclidean distance.

3. **Reference Integration Transformer**:

    - **Function**: Handles a variable number of reference inputs and adaptively calculates the contribution weight of each reference according to the target location.
    - **Mechanism**: Each reference is treated as an element in a sequence. The Query is formed by concatenating the target location VAB embedding $g_k$ and a learnable latent embedding $\mathbf{e}$: $g_k^e = [g_k \| \mathbf{e}]$. The Key/Value pairs are formed by concatenating the reference VAB embeddings and relative position vectors: $g_i^r = [g_i \| r_{ki}]$. The attention weights $a_{ki} = \frac{g_k^e \cdot g_i^{r\top}}{\sqrt{C}}$ are normalized via Softmax to serve as reference contribution weights. Crucially, these weights are independent of the audio content.
    - **Design Motivation**: Weighting based purely on distance is unreliable in the presence of obstacles, while weighting based on audio content is susceptible to degradation on out-of-distribution audio content. VAB embeddings provide a scene-aware yet content-independent weighting scheme.

### Loss & Training

The loss function is a weighted combination of three terms: (1) Waveform loss: MSE between the predicted and target waveforms, establishing amplitude and phase accuracy; (2) Binaural ILD loss: Energy difference between the left and right channels to ensure accurate spatial effects; (3) Multi-resolution spectral magnitude loss: Calculates spectral magnitude differences across multiple FFT/hop resolutions, including log-scaled spectral magnitude loss to handle high variance.

The renderer employs a stacked U-Net architecture with a decoupled design utilizing global conditions $c_g$ (location effects) and local conditions $c_l$ (orientation effects). During the pre-training phase, the binauralization capability is learned by fixing target locations and varying only the head orientation.

## Key Experimental Results

### Main Results (Soundspace-Ambient benchmark, seen scenes)

| Method | No. References | STFT ↓ | MAG ↓ | ENV ↓ | LRE ↓ |
|------|-------|--------|-------|-------|-------|
| AV-NeRF | 1 | 9.424 | 0.426 | 0.195 | 1.922 |
| ViGAS | 1 | 3.740 | 0.361 | 0.154 | 2.040 |
| **SoundVista** | 1 | **2.526** | **0.291** | **0.132** | **1.408** |
| BEE | 4 | 4.098 | 0.365 | 0.162 | 2.083 |
| **SoundVista** | 4 | **2.444** | **0.289** | **0.130** | **1.390** |

Compared with the best baseline ViGAS: STFT decreases by 32.5%, MAG by 19.4%, and LRE by 31%.

### Ablation Study (VAB RT60 Prediction)

| Input Modality | Error w/o finetuning | Error w/ finetuning |
|---------|-----------------|-----------------|
| Position Only | Highest | High |
| Depth | Decreased >50% | Medium |
| RGB+Depth | Suboptimal | **Lowest** |

### Key Findings

- SoundVista with only 1 reference outperforms the BEE method using 4 references, thanks to VAB-guided intelligent reference selection and weighting.
- Contribution weights are decoupled from audio content, meaning they are robust to testing-time content distribution shifts—a major disadvantage of Few-shotRIR and BEE.
- Performance with the top-4 references matches that of using all references, demonstrating that distant references contribute minimally.
- On the N2S real-world scene benchmark, SoundVista surpasses the best baseline ViGAS by 7.6% in binaural layout representation (LRE).
- Depth information contributes the most to VAB, with the combination of RGB+Depth achieving optimal performance.

## Highlights & Insights

- **Visual-Acoustic Bridge**: Utilizing RGB-D information to infer acoustic properties is a clever cross-modal transfer that bypasses expensive physical acoustic testing.
- **Content-Independent Reference Weighting**: Decoupling weighting from audio content prevents performance degradation caused by testing-time out-of-distribution sounds.
- **Practical Reference Position Optimization**: Clustering based on VAB embeddings automatically adapts to scenes of varying sizes and complexities.

## Limitations & Future Work

- The VAB module is pre-trained in the SoundSpaces simulator, and real-world transfer may be subject to the sim-to-real gap.
- References are currently assumed to be ambisonic microphones, which might be constrained by hardware availability in practice.
- Scenes with 10 sound sources already present challenges; more extreme numbers of sound sources (e.g., dozens of sound sources in traffic scenes) remain to be tested.
- Future work could combine 3DGS scene reconstruction to achieve end-to-end joint audio-visual modeling.

## Related Work & Insights

- Represents another successful application of CLIP-style modality alignment in the audio-visual domain.
- AV-RIR's visual-RIR binding is close work; SoundVista generalizes this to compile a complete sound field at the ambient level.
- The strategy of reference position optimization can be generalized to other multi-sensor deployment tasks.

## Rating

- **Novelty**: 8/10 — The VAB-driven reference sampling and adaptive weighting scheme are highly novel.
- **Experimental Thoroughness**: 8/10 — Thoroughly validated across both simulated and real-world scenes with comprehensive ablation studies.
- **Writing Quality**: 8/10 — Clearly defined problems with detailed pipeline illustrations.
- **Value**: 8/10 — Provides practical advancements for immersive audio-visual experiences and the spatial audio domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Novel View Synthesis with Pixel-Space Diffusion Models](novel_view_synthesis_with_pixel-space_diffusion_models.md)
- [\[CVPR 2025\] MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](zero-shot_novel_view_and_depth_synthesis_with_multi-view_geometric_diffusion.md)
- [\[CVPR 2025\] MOVIS: Enhancing Multi-Object Novel View Synthesis for Indoor Scenes](movis_enhancing_multi-object_novel_view_synthesis_for_indoor_scenes.md)
- [\[CVPR 2025\] CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis](comapgs_covisibility_map-based_gaussian_splatting_for_sparse_novel_view_synthesi.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](../../ICCV2025/3d_vision/self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)

</div>

<!-- RELATED:END -->
