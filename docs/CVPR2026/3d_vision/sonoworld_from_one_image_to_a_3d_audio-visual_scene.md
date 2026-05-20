---
title: >-
  [Paper Note] SonoWorld: From One Image to a 3D Audio-Visual Scene
description: >-
  [CVPR 2026][3D Vision][3D audio-visual scene] SonoWorld is proposed as a training-free framework that generates an explorable 3D audio-visual scene from a single image. The pipeline expands the input image into a 360° pa…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D audio-visual scene"
  - "spatial audio generation"
  - "panorama reconstruction"
  - "Ambisonics encoding"
  - "single-image generation"
date: 2026-05-08
content_hash: 446093f8b6288c98
---

# SonoWorld: From One Image to a 3D Audio-Visual Scene

**Conference**: CVPR 2026
**arXiv**: [2603.28757](https://arxiv.org/abs/2603.28757)  
**Code**: [https://humathe.github.io/sonoworld/](https://humathe.github.io/sonoworld/)  
**Area**: 3D Vision / Audio-Visual Scene Generation
**Keywords**: 3D audio-visual scene, spatial audio generation, panorama reconstruction, Ambisonics encoding, single-image generation

## TL;DR

SonoWorld is proposed as a training-free framework that generates an explorable 3D audio-visual scene from a single image. The pipeline expands the input image into a 360° panorama and reconstructs it as a 3D Gaussian scene, places sound-source anchors via VLM-driven semantic grounding, and renders spatial audio through Ambisonics encoding, achieving geometric and semantic alignment between the visual and auditory modalities.

## Background & Motivation

**Background**: Visual scene generation has advanced substantially in recent years—from panoramic approaches such as WorldGen to 3D Gaussian Splatting techniques—enabling freely navigable 3D worlds from a single image. However, all such systems produce "silent worlds" that can be seen but not heard.

**Limitations of Prior Work**: Genuine immersive experiences are inherently multi-sensory. In a garden, for instance, waterfall sounds should originate upstream and grow louder as the listener approaches, birdsong should descend from the canopy, and insect sounds should shift as the head rotates. Without semantically correct audio carrying distance and directional cues, even a photorealistic visual world remains perceptually incomplete. Existing audio generation methods either produce mono audio, are limited to single objects or fixed viewpoints, or cannot handle scene-level audio comprising diverse source types such as point sources (e.g., birdsong), surface sources (e.g., a river), and ambient sounds (e.g., wind).

**Key Challenge**: Scene-level spatial audio generation must simultaneously address three problems: (1) composing heterogeneous source types whose behaviors differ fundamentally—point, surface, and ambient sources; (2) reasoning purely from visual context about what objects emit sound, how they sound, and at what loudness; and (3) anchoring all sounds to plausible 3D positions inferred from the image while producing perceptually convincing spatial effects.

**Goal**: The paper defines a novel task, Image2AVScene—simultaneously generating an interactive 3D visual scene and a spatially and semantically aligned sound field from a single image—and proposes the first complete framework for this task.

**Key Insight**: An equirectangular panorama representation is adopted to unify the coordinate systems of vision and audio, with a VLM providing semantic understanding to bridge the two modalities.

**Core Idea**: A training-free pipeline—panorama outpainting → 3DGS reconstruction → VLM-driven 360° semantic grounding → Ambisonics encoding—enables the generation of a freely navigable 3D audio-visual scene from a single image.

## Method

### Overall Architecture

Given an RGB image as input, the system outputs a 3D visual scene $\mathbf{V}$ (represented as 3D Gaussian Splatting) and a spatial audio field $\mathbf{A}$ (represented as Ambisonics). The pipeline comprises four stages: (1) visual scene generation—camera calibration → panorama outpainting → 3D reconstruction; (2) 360° semantic grounding—VLM-based sounding category extraction → open-vocabulary segmentation → panoramic mask refinement → back-projection to 3D; (3) Ambisonics encoding—text-to-audio generation → equalization → spatial encoding; (4) free-viewpoint rendering—HRTF decoding to binaural audio.

### Key Designs

1. **Panoramic Visual Scene Generation**

    - **Function**: Expands a single image into a 360° panorama and lifts it into a 3D scene.
    - **Mechanism**: GeoCalib is first applied for single-image camera calibration to obtain the elevation angle and field of view $(φ, f) = \text{Calib}(I)$. The image is then projected into an equirectangular panorama via Gaussian-pyramid anti-aliasing sampling, and the WorldGen outpainting model completes the full 360° field of view. The panorama is subsequently lifted into a 3D Gaussian Splatting scene using either HunyuanWorld (open-source) or Marble (commercial).
    - **Design Motivation**: The panoramic representation inherently covers the full 360° field of view and provides a unified coordinate system. Elevation correction addresses the vertical distortion caused by prior methods that assume horizontal camera orientation.

2. **360° Semantic Grounding**

    - **Function**: Localizes all potentially sounding entities and their spatial extents within the 3D scene.
    - **Mechanism**: A VLM (GPT-4o or LLaVA-Next-34B) first reasons from the input image to infer the sounding category set $\mathcal{C}$ along with associated attributes (source type, text prompt, equalization parameters). Because open-vocabulary segmentation (OVS) models are trained on perspective images, the panorama is tiled into overlapping FoV patches and segmented individually with X-Decoder, then reprojected to panoramic coordinates. Concurrently, SAM2 segments the full panorama to obtain class-agnostic but geometrically accurate regions; OVS results then vote over SAM2 regions, combining SAM2's global geometric consistency with X-Decoder's semantic precision. Finally, the depth map is used to back-project masks into 3D, yielding sound-source anchors $\mathcal{P}$.
    - **Design Motivation**: Tile-based OVS produces boundary discontinuities and incomplete regions upon stitching, while SAM2 is geometrically precise but class-agnostic; the two are complementary, and the voting fusion strategy resolves both accuracy and consistency issues in panoramic semantic segmentation.

3. **Ambisonics Encoding and Rendering**

    - **Function**: Converts semantically grounded sound sources into spatial audio that can be rendered at any listener position and orientation.
    - **Mechanism**: MMAudio generates a waveform $a_{i,\text{raw}}$ for each source from its text prompt; after equalization $a_i(t) = 10^{v_i/20} a_{i,\text{raw}}(t)$, Ambisonics coefficients are encoded according to source type. Point sources are approximated by their centroid: $\mathbf{A}_\text{point} = \sum_i a_i \sigma(\|d_i\|) \mathbf{y}_L(...)$; surface sources average contributions over the entire point cloud to create a diffuse sound field; ambient sounds encode only the omnidirectional component $\mathbf{A}_\text{global} = a_\text{global}[1, 0, ..., 0]^\top$. Distance attenuation is modeled as $\sigma(d)=e^{-\alpha d}/d$. The entire rendering pipeline is differentiable over the audio buffer.
    - **Design Motivation**: Different source types behave fundamentally differently—birdsong is a point source requiring precise directional cues, a river is a surface source producing a diffuse field, and wind is an ambient sound independent of direction. Classifying and processing each type within the unified Ambisonics framework handles this heterogeneity. The differentiable property additionally enables the framework to be extended to acoustic learning and source separation tasks.

### Loss & Training

SonoWorld is a training-free framework that requires no training. It is composed entirely of pretrained models (VLM, outpainting model, 3D reconstruction model, audio generation model). The differentiable rendering pipeline is used for optimization in downstream tasks such as one-shot room acoustic learning.

## Key Experimental Results

### Main Results

Evaluated on the authors' SonoScene360 dataset (68 clips across 6 real-world scenes):

| Method | ΔAngular↓ | CC↑ | AUC↑ | D-CLAPT↑ | D-CLAPR↑ |
|---|---|---|---|---|---|
| MMAudio | — | — | — | 0.322 | 33.8% |
| SEE-2-SOUND | 1.397 | 0.194 | 0.603 | 0.156 | 22.1% |
| OmniAudio | 1.449 | 0.148 | 0.588 | 0.104 | 39.7% |
| Ours (Open-source) | 0.975 | 0.491 | 0.753 | 0.413 | 52.9% |
| **Ours (Proprietary)** | **0.728** | **0.658** | **0.838** | **0.457** | **67.6%** |

The DOA error is reduced by 47%, CC improves by over 239%, and semantic metrics improve by over 117%.

### Ablation Study

One-shot room acoustic learning:

| Method | ΔAngular↓ | MAG↓ | ENV↓ |
|---|---|---|---|
| NAF | 1.76 | 3.96 | 3.60 |
| AV-NeRF | 1.58 | 4.58 | 1.89 |
| **Ours** | **0.22** | **3.46** | **1.22** |

### Key Findings

- The method achieves audio callback latency below 1 ms on an Apple M3 Pro laptop, well below the 5.3 ms real-time requirement.
- In a user study with 50 participants across 12 scenes, SonoWorld received the highest preference rate in all comparisons.
- The open-source version (HunyuanWorld + LLaVA-Next) significantly outperforms baselines that use commercial model outputs.
- The siren scene reveals a limitation regarding moving sound sources—static image input cannot capture source motion.

## Highlights & Insights

- **First task definition and complete solution for Image2AVScene**: Visual scene generation and spatial audio generation are unified within a single framework.
- **Unifying power of the panoramic representation**: The panorama not only provides a complete 360° field of view but is also naturally aligned with the Ambisonics coordinate system, making it a key architectural choice underlying the method's success.
- **Complementary fusion of VLM + SAM2**: OVS provides semantics but lacks global consistency; SAM2 is globally consistent but class-agnostic; the voting fusion strategy exploits their complementarity elegantly.
- **Generality of the differentiable rendering pipeline**: The same framework extends straightforwardly to acoustic learning and source separation.
- **Training-free design**: The entire system is an ingenious composition of existing pretrained models, ensuring strong engineering practicality.

## Limitations & Future Work

- Moving sound sources cannot be handled, as the input is a static image.
- First-order Ambisonics (FOA) offers limited spatial resolution; higher-order representations would improve this but require an exponentially growing number of channels.
- Sound generation quality depends on MMAudio, which may produce suboptimal results for rare source categories.
- Complex acoustic phenomena such as room reverberation and multipath effects are not modeled.
- The quality of the generated visual scene is bounded by the capabilities of the outpainting and 3D reconstruction models.

## Related Work & Insights

- **WonderWorld / WorldGen**: Foundational work on panorama-to-3D scene generation.
- **MMAudio**: A video-to-audio generation model, used here for per-source audio synthesis.
- **X-Decoder + SAM2**: The combination of open-vocabulary segmentation with panoramic refinement via SAM2 is a design pattern worth adopting in future work.
- The framework's approach can be extended to 4D dynamic scenes and acoustic perception in embodied intelligence.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Pioneering work that is the first to define and address the task of generating 3D audio-visual scenes from a single image.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Includes a self-constructed dataset, comprehensive metrics, a user study, and extended applications, though the number of evaluated real-world scenes is limited (6 scenes).
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem definition is clear, method description is complete, and mathematical derivations are rigorous.
- **Value**: ⭐⭐⭐⭐⭐ — Opens a new direction of multi-sensory scene generation for VR/AR and embodied intelligence.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Pano3DComposer: Feed-Forward Compositional 3D Scene Generation from Single Panoramic Image](pano3dcomposer_feed-forward_compositional_3d_scene_generation_from_single_panora.md)
- [\[CVPR 2026\] Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation](ada3drift_adaptive_trainingtime_drifting_for_onest.md)
- [\[CVPR 2026\] AffordMatcher: Affordance Learning in 3D Scenes from Visual Signifiers](affordmatcher_affordance_learning_in_3d_scenes_from_visual_signifiers.md)
- [\[CVPR 2026\] Text–Image Conditioned 3D Generation](text-image_conditioned_3d_generation.md)
- [\[CVPR 2026\] MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second](movies_motion-aware_4d_dynamic_view_synthesis_in_one_second.md)

</div>

<!-- RELATED:END -->
