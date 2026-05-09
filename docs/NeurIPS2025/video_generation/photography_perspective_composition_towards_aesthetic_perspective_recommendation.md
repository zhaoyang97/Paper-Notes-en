---
title: >-
  [Paper Note] Photography Perspective Composition: Towards Aesthetic Perspective Recommendation
description: >-
  [NeurIPS 2025][Video Generation][Photography Composition] This paper proposes a novel "Photography Perspective Composition" (PPC) paradigm that goes beyond traditional cropping-based approaches. It constructs a perspective transformation dataset via 3D reconstruction, generates recommended viewpoints through Image-to-Video generation, aligns with human preferences via RLHF, and evaluates perspective quality using a PQA model.
tags:
  - NeurIPS 2025
  - Video Generation
  - Photography Composition
  - Perspective Transformation
  - Aesthetic Assessment
  - RLHF
date: 2026-05-08
content_hash: 924240bb3c9e0d91
---

# Photography Perspective Composition: Towards Aesthetic Perspective Recommendation

**Conference**: NeurIPS 2025
**arXiv**: [2505.20655](https://arxiv.org/abs/2505.20655)
**Code**: [Project Page](https://vivocameraresearch.github.io/ppc)
**Area**: Video Generation
**Keywords**: Photography Composition, Perspective Transformation, Video Generation, Aesthetic Assessment, RLHF

## TL;DR

This paper proposes a novel "Photography Perspective Composition" (PPC) paradigm that goes beyond traditional cropping-based approaches. It constructs a perspective transformation dataset via 3D reconstruction, generates recommended viewpoints through Image-to-Video generation, aligns with human preferences via RLHF, and evaluates perspective quality using a PQA model.

## Background & Motivation

1. **Background**: Photography composition methods are primarily based on 2D cropping (free cropping, subject-aware cropping, ratio-aware cropping), with existing datasets such as GAICD, CPC, and FCDB.

2. **Limitations of Prior Work**: Cropping-based methods operate solely within the 2D image plane and cannot improve compositions where the spatial arrangement of subjects is fundamentally suboptimal. Professional photographers perform "3D recomposition" by adjusting shooting angles, yet this direction remains unexplored in computational photography.

3. **Key Challenge**: Three major challenges exist: (1) the absence of perspective transformation datasets; (2) compositional aesthetics form a partial order rather than a total order; and (3) the lack of evaluation criteria for perspective quality.

4. **Goal**: To provide an end-to-end solution covering dataset construction, recommendation methodology, and evaluation models for perspective composition recommendation.

5. **Key Insight**: Existing professional photography images are leveraged alongside 3D reconstruction to inversely generate "good-to-poor" perspective transformation videos, which are then reversed to obtain "poor-to-good" training data.

6. **Core Idea**: An I2V model is used to generate transformation videos from suboptimal to aesthetically improved viewpoints for composition recommendation, rather than directly outputting a single image.

## Method

### Overall Architecture

Three major modules: (1) Automatic PPC dataset construction → (2) PPC video generation + RLHF → (3) PQA perspective quality assessment model.

### Key Designs

**1. Automatic Construction of the Perspective Transformation Dataset**

- **Function**: Generate training data with perspective transformations from professional photography images.
- **Mechanism**: Professional photographs serve as "good composition" inputs. ViewCrafter is used for 3D reconstruction to generate "good-to-poor" videos along random camera trajectories, which are then reversed to yield "poor-to-good" training data. A PQA model automatically filters samples with poor reconstruction quality (distortion, static frames, blur), replacing manual screening.
- **Design Motivation**: Real-world POV photography videos are scarce and difficult to obtain; the inverse generation strategy cleverly exploits the abundance of professional photography resources.

**2. I2V-Based Perspective Recommendation**

- **Function**: Given a suboptimal viewpoint, generate a transformation video leading to an aesthetically enhanced viewpoint.
- **Mechanism**: The problem is formulated as an Image-to-Video task using open-source I2V models such as CogVideoX, HunYuan, and WAN. No additional text prompts or camera trajectories are required. The last frame of the generated video serves as the recommended viewpoint. Feature matching projects a guidance bounding box onto the original image, with the box shape changing as the user moves to provide real-time guidance. DPO (Direct Preference Optimization) is introduced to align with human preferences.
- **Design Motivation**: The video format naturally accommodates before-and-after comparison (handling partial rather than total ordering) while providing intuitive visual guidance.

**3. PQA Perspective Quality Assessment Model**

- **Function**: Automatically evaluate the quality of perspective transformation videos.
- **Mechanism**: A two-stage training strategy based on Qwen2-VL-2B. Stage 1: unpaired videos (5K videos generating 15K pairs) to learn basic quality discrimination. Stage 2: paired videos (pairwise comparisons of three model outputs for the same input) to learn fine-grained compositional aesthetics. Three evaluation dimensions: Visual Quality (VQ), Motion Quality (MQ), and Compositional Aesthetics (CA). BTT (Bradley-Terry with Ties) loss is employed.
- **Design Motivation**: Fine-tuning VLMs requires large amounts of data, yet expert composition annotations are scarce. The two-stage strategy first establishes a foundation using easily obtainable quality data, then refines with a small amount of expert annotations.

### Loss & Training

- PPC model: I2V base training + Flow-DPO loss for human preference alignment.
- PQA model: BTT (Bradley-Terry with Ties) loss, with VQ/MQ/CA dimensions decoupled via dedicated special tokens.
- A five-level rating scheme (A–E) is used for data filtering.

## Key Experimental Results

### Main Results

**Comparison of I2V Models for Perspective Transformation Generation**

| Model | CMM ↑ | FVD ↓ | VQ ↑ | MQ ↑ | CA ↑ |
|-------|-------|-------|------|------|------|
| CogVideoX 1.5 5B | 0.550 | 303 | 0.707 | 0.731 | 0.720 |
| HunYuan I2V | 0.493 | **264** | **0.722** | **0.750** | 0.707 |
| **Wan2.1 14B** | **0.599** | 345 | 0.720 | 0.745 | 0.707 |

**Effect of RLHF**

| Setting | CMM ↑ | FVD ↓ | VQ ↑ | MQ ↑ | CA ↑ |
|---------|-------|-------|------|------|------|
| w/o RLHF | 0.493 | **264.8** | 0.722 | 0.750 | 0.707 |
| w/ RLHF | **0.501** | 270.2 | **0.748** | **0.777** | **0.734** |

### Ablation Study

| Experiment | Condition | CMM ↑ / FVD ↓ |
|------------|-----------|--------------|
| Data proportion | 20% / 40% / 80% / 100% | 0.501/460, **0.599/345**, 0.524/362, 0.567/359 |
| Rotation angle | 10° / **20°** / 30° / Mix | 0.441/397, **0.559/337**, 0.398/444, **0.599/345** |
| PQA pair count | 1 / 5 / 10 / 100 | CA acc: 0.588 / 0.789 / **0.810** / 0.810 |
| PQA training stage | Single-stage / **Two-stage** | CA acc: 0.491 / **0.810** |

### Key Findings

- 40% of the data is sufficient to achieve optimal performance; additional data yields no further improvement.
- Performance degrades significantly at a rotation angle of 30°, where the discrepancy between the original and transformed viewpoints becomes too large.
- Mixed-angle training data achieves the best performance, indicating that diversity outweighs precise angle control.
- The two-stage training of PQA is critical: single-stage training yields a CA accuracy of only ~49%, equivalent to random chance.
- The PPC model exhibits compositional consistency: given multiple suboptimal viewpoints of the same scene, the outputs converge toward a consistent aesthetically enhanced viewpoint.

## Highlights & Insights

- Establishes a novel "perspective composition" paradigm, elevating the problem from 2D cropping to 3D viewpoint adjustment.
- Elegant data construction strategy: inverse generation combined with automatic filtering, eliminating the need for real-world POV photography videos.
- Video-based recommendation rather than image-based recommendation—elegantly handles the partial ordering problem while providing pedagogical value.
- The two-stage training strategy of PQA addresses the scarcity of expert annotation data.

## Limitations & Future Work

- Generation quality degrades notably at large transformation angles due to limitations of the underlying 3D reconstruction model.
- The current approach supports only short-range perspective transformations; large-scale viewpoint changes produce unsatisfactory results.
- The PQA model is based on a 2B-parameter VLM, which may limit its assessment capacity.
- The simple homography-based guidance bounding box may lack precision in complex scenes.

## Related Work & Insights

- This work is the first to combine 3D scene reconstruction with photographic compositional aesthetics.
- The single-image 3D reconstruction capability of ViewCrafter is a core technical dependency.
- Inspiration: the proposed framework could be extended to video photography composition, drone aerial path planning, and related applications.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Pioneering introduction of the perspective composition paradigm with a novel dataset construction approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated across multiple scenarios including single-subject, multi-subject, landscape, and drone settings, with thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with rich illustrations.
- **Value**: ⭐⭐⭐⭐ Groundbreaking for computational photography, though practical utility is constrained by 3D reconstruction quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](../../ICLR2026/video_generation/lumos-1_on_autoregressive_video_generation_with_discrete_diffusion_from_a_unifie.md)
- [\[CVPR 2026\] CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video](../../CVPR2026/video_generation/cubecomposer_spatio-temporal_autoregressive_4k_360_video_generation_from_perspec.md)
- [\[NeurIPS 2025\] Safe-Sora: Safe Text-to-Video Generation via Graphical Watermarking](safesora_safe_texttovideo_generation_via_graphical_watermark.md)
- [\[NeurIPS 2025\] LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation](lemica_lexicographic_minimax_path_caching_for_efficient_diffusion-based_video_ge.md)
- [\[NeurIPS 2025\] Force Prompting: Video Generation Models Can Learn and Generalize Physics-based Control Signals](force_prompting_video_generation_models_can_learn_and_generalize_physics-based_c.md)

</div>

<!-- RELATED:END -->
