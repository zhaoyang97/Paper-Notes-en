---
title: >-
  [Paper Note] ORIDa: Object-Centric Real-World Image Composition Dataset
description: >-
  [CVPR 2025][Image Generation][Image Composition] ORIDa constructs the first large-scale, real-shot, and publicly available object composition dataset containing over 30,000 images of 200 unique objects (including fact-counterfactual pairs and multi-position variations). It validates the dataset's efficacy on object removal and insertion tasks via fine-tuning on StableDiffusion-Inpaint.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image Composition"
  - "Object Insertion"
  - "Dataset"
  - "Real-World Data"
  - "Diffusion Models"
date: 2026-05-08
content_hash: 5f16679b2fe97cfb
---

# ORIDa: Object-Centric Real-World Image Composition Dataset

**Conference**: CVPR 2025  
**arXiv**: [2506.08964](https://arxiv.org/abs/2506.08964)  
**Code**: [https://hello-jinwoo.github.io/orida](https://hello-jinwoo.github.io/orida)  
**Area**: Diffusion Models / Image Editing  
**Keywords**: Image Composition, Object Insertion, Dataset, Real-World Data, Diffusion Models

## TL;DR

ORIDa constructs the first large-scale, real-shot, and publicly available object composition dataset containing over 30,000 images of 200 unique objects (including fact-counterfactual pairs and multi-position variations). It validates the dataset's efficacy on object removal and insertion tasks via fine-tuning on StableDiffusion-Inpaint.

## Background & Motivation

**Background**: Object compositing is the task of placing and blending an object into a target scene, involving challenges such as identity preservation, color harmonization, shadow generation, and geometric alignment. Current methods are categorized into training-free approaches (e.g., FreeCompose) and training-based approaches (e.g., ObjectStitch, ObjectDrop).

**Limitations of Prior Work**: (1) Training-free methods underperform in details, particularly scene harmonization and object identity preservation; (2) Training-based methods mostly rely on synthetic data, which lacks the complexity and diversity of real-world scenes; (3) The most relevant dataset, ObjectDrop, although captured in the real world, contains only 2,500 object pairs and is not publicly available, with only 1 scene and 1 position per object.

**Key Challenge**: The lack of high-quality real-world image composition datasets severely restricts the development of object composition models. Synthetic data fails to capture complex real-world "object-to-scene interaction" effects, such as lighting, shadows, and reflections.

**Goal**: To build a large-scale, publicly available, real-shot dataset that meets the following criteria: (1) contains a sufficient number of unique objects; (2) shows each object in multiple different scenes; (3) provides fact (object present) and counterfactual (background-only) image pairs; (4) includes multiple position variations of the object in each scene.

**Key Insight**: Utilizing a meticulously designed data acquisition pipeline (fixed camera parameters, tripod and remote shutter, strict filtering) ensures that the presence or absence of the object is the sole variable within a scene, thereby obtaining high-quality fact-counterfactual data pairs.

**Core Idea**: Constructing the ORIDa dataset (200 objects, 30,000+ images, 50+ scenes per object, 4 positions per scene), supporting ISP augmentation through the provision of raw DNG files, and demonstrating that high-quality object composition models can be trained solely on real-world data without synthetic data.

## Method

### Overall Architecture

The ORIDa dataset consists of two components: (1) **Fact-Counterfactual (F-CF) Collection**—each group has 5 images: 1 background-only plus 4 object images in different positions, totaling 5,699 groups; (2) **Fact-Only (F-Only) Images**—single images of objects in specific scenes, totaling 5,035 images. RAW DNG files were captured using 5 different smartphones in PRO mode. StableDiffusion-Inpaint is fine-tuned on this dataset for object removal and insertion tasks.

### Key Designs

1. **Data Acquisition and Quality Control**:

    - Function: To ensure that scene changes in the dataset are solely caused by the presence or absence of the object.
    - Mechanism: During F-CF collection, the camera is fixed (using tripods and remote controls) along with key parameters (shutter speed, ISO, white balance, focus) to continuously shoot 5 images. A strict filtering process identifies three types of undesirable cases: (a) accidental background changes (such as illumination variations or pedestrians); (b) out-of-focus images; (c) incorrect object poses. Out of 7,000 F-CF groups, 5,699 were selected, and 5,035 were retained from 5,500 F-Only images.
    - Design Motivation: Data quality is core to the value of a dataset. Only by strictly controlling variables can the bidirectional effects of object-to-scene influence (shadows, reflections, etc.) and scene-to-object influence (color shifts, etc.) be accurately captured.

2. **Rich Annotation System**:

    - Function: To provide hierarchical annotations supporting downstream research.
    - Mechanism: Four types of annotations are provided: (1) object descriptive texts (generated by GPT-4o and Gemini 1.5 Pro); (2) object points (manually annotated); (3) bounding boxes; (4) segmentation masks (generated by SAM2). Additionally, the raw DNG files support 5 types of ISP augmentations (original, high/low color temperature, high/low vibrancy) for training color harmonization.
    - Design Motivation: Object composition involves multiple sub-problems (identity preservation, shadow generation, color harmonization), which require rich annotations to support evaluation and training across different dimensions.

3. **SD-Inpaint Based Object Removal/Insertion Models**:

    - Function: To verify that effective object composition models can be trained solely using ORIDa real-world data.
    - Mechanism: Object removal: fine-tuning SD-Inpaint (9-channel input) directly for only 5,000 steps (320k samples), which is far fewer than the 6.4 million samples used in ObjectDrop. Object insertion: trained for 500k steps using raw images from ORIDa and COCO (without synthetic data). During inference, skip residual connections (from DemoFusion) are incorporated to maintain object identity.
    - Design Motivation: To demonstrate that high-quality real data can replace massive synthetic data. ObjectDrop requires a batch size of 512 and two-stage training with synthetic data, whereas ORIDa's model achieves superior performance with significantly less data.

### Loss & Training

- **Optimizer**: Adam, lr=5e-5, cosine scheduler
- **Training Scale**: Object removal: 5,000 steps / batch size 64; Object insertion: 500k steps / batch size 64
- **Hardware**: 4x NVIDIA A100-PCIE (40GB), object insertion training takes approximately 150 hours

## Key Experimental Results

### Main Results — Object Removal

| Method | PSNR↑ | DINO↑ | CLIP↑ | LPIPS↓ |
|------|-------|-------|-------|--------|
| SD-Inpaint | 21.76 | 0.845 | 0.903 | 0.108 |
| **SD-Ours_r** | **25.60** | **0.902** | **0.938** | **0.088** |

User study (5-point scale): SD-Ours_r scores 4.23, significantly outperforming SD-Inpaint (2.78), LaMa (2.63), and MGIE (1.96).

### Object Insertion User Study

| Dimension | SD-Ours_i Preference Rate |
|------|-----------------|
| Object Identity Preservation | 66% |
| Shadow Generation | 79% |
| Color Harmonization | 71% |
| Overall Quality | 67% |

### Ablation Study — Data Scale

| Data Ratio | Effect Description |
|----------|---------|
| 25% | Inaccurate shadow generation, artifacts in object appearance |
| 50% | Improvement observed but inconsistencies persist |
| 100% | Best performance, accurate shadows, and well-preserved object identity |

### Key Findings

- SD-Ours trained purely on real-world data (without synthetic data) significantly outperforms baselines in both object removal and insertion.
- The F-CF data in ORIDa enables the model to learn accurate shadow removal and generation, which is difficult to achieve with purely synthetic data.
- Data scale ablation confirms that the amount of data is crucial for high-quality composition.
- ISP augmentation of raw DNG files effectively enhances the color harmonization capability.

## Highlights & Insights

- **Dataset Level**: The first object composition dataset that is large-scale, real-shot, publicly available, and features multiple objects, multiple scenes, and multiple positions.
- **Bidirectional Effects**: Simultaneously captures both "object-to-scene influence" (shadows, reflections) and "scene-to-object influence" (color variations), which synthetic data cannot simulate.
- **Practical Engineering Value**: Proves that in the field of object composition, meticulously captured real data is more effective than vast amounts of synthetic data.
- **RAW DNG Support**: Provides unprocessed raw files, opening up new possibilities for ISP-level data augmentation.

## Limitations & Future Work

- The number of objects (200) is relatively limited; the diversity of covered object categories and materials needs to be expanded.
- The insertion results exhibit some blurriness, stemming from the inherent limitations of the pretrained SD-Inpaint model.
- Object pose variations are intentionally restricted and do not cover composition scenarios under different viewpoints/poses.
- Future work can extend this to video composition, object insertion in dynamic scenes, and other directions.

## Related Work & Insights

- **ObjectDrop**: The most relevant prior work, but it is not publicly available, features only 1 scene/position per object, and relies on auxiliary training with synthetic data.
- **Paint-by-Example**: Trained on synthetic data, underperforming in maintaining object identity.
- **AnyDoor/ObjectStitch**: State-of-the-art object composition methods that still fall short in shadow generation and color adaptation.

## Rating

- **Novelty**: 7/10 — The core contribution lies in the dataset construction rather than methodological innovation; the model is only used for validation fine-tuning.
- **Experimental Thoroughness**: 8/10 — Multi-dimensional evaluations are provided, including quantitative metrics, user studies, ablation experiments, and data analysis.
- **Writing Quality**: 8/10 — The dataset construction process is clearly described with detailed statistical analysis.
- **Value**: 8/10 — Fills the gap in real-world object composition datasets and makes a significant contribution to the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GLASS: Guided Latent Slot Diffusion for Object-Centric Learning](glass_guided_latent_slot_diffusion_for_object-centric_learning.md)
- [\[CVPR 2025\] UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics](unireal_universal_image_generation_and_editing_via_learning_real-world_dynamics.md)
- [\[CVPR 2025\] CTRL-O: Language-Controllable Object-Centric Visual Representation Learning](ctrl-o_language-controllable_object-centric_visual_representation_learning.md)
- [\[CVPR 2026\] MICo-150K: A Comprehensive Dataset Advancing Multi-Image Composition](../../CVPR2026/image_generation/mico-150k_a_comprehensive_dataset_advancing_multi-image_composition.md)
- [\[CVPR 2025\] Self-Supervised ControlNet with Spatio-Temporal Mamba for Real-World Video Super-Resolution](self-supervised_controlnet_with_spatio-temporal_mamba_for_real-world_video_super.md)

</div>

<!-- RELATED:END -->
