---
title: >-
  [Paper Note] SAP: Segment Any 4K Panorama
description: >-
  [CVPR 2025][Segmentation][Panoramic Segmentation] This work reformulates 360° panoramic segmentation as a perspective video segmentation problem. By decomposing the panorama into a sequence of overlapping patches along a zigzag trajectory and fine-tuning the memory module of SAM2, combined with large-scale training on 183K synthetic 4K panoramas, it achieves a zero-shot panoramic segmentation improvement of +17.2 mIoU.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Panoramic Segmentation"
  - "SAM2"
  - "Equirectangular Projection"
  - "4K Segmentation"
  - "Video Segmentation Paradigm"
  - "Data Synthesis"
date: 2026-05-08
content_hash: 9139e8f8c8aac456
---

# SAP: Segment Any 4K Panorama

**Conference**: CVPR 2025  
**arXiv**: [2603.12759](https://arxiv.org/abs/2603.12759)  
**Code**: None  
**Area**: Semantic Segmentation / Panoramic Images  
**Keywords**: Panoramic Segmentation, SAM2, Equirectangular Projection, 4K Segmentation, Video Segmentation Paradigm, Data Synthesis

## TL;DR

This work reformulates 360° panoramic segmentation as a perspective video segmentation problem. By decomposing the panorama into a sequence of overlapping patches along a zigzag trajectory and fine-tuning the memory module of SAM2, combined with large-scale training on 183K synthetic 4K panoramas, it achieves a zero-shot panoramic segmentation improvement of +17.2 mIoU.

## Background & Motivation

### Limitations of Prior Work

**Background**: **Widespread application of panoramas**: 360° panoramic images are widely used in autonomous driving, robotic navigation, and VR/AR, but their Equirectangular Projection (ERP) format introduces severe geometric distortion.

**SAM2 Degradation on Panoramas**: While SAM2 performs exceptionally well on standard perspective images, its performance drops significantly when directly applied to ERP panoramas—objects in polar regions are severely distorted, and segmentation boundaries become inaccurate.

**Boundary Discontinuity**: The left and right boundaries of an ERP image are continuous on a sphere, but their 2D projection creates discontinuous seams, leading to mis-segmentation of boundary-crossing objects.

**4K Resolution Challenge**: Panoramas are usually in 4K (4096×2048) or higher resolution, which SAM2 cannot directly process. Simple downsampling loses considerable detail.

**Scarcity of Training Data**: There is currently a lack of large-scale, high-quality annotated datasets for panoramic segmentation.

**Core Idea**: Repurpose panoramic segmentation into perspective video segmentation—decomposing the panorama into a sequence of overlapping perspective patches along a specific trajectory to leverage SAM2's video segmentation capabilities.

## Method

### Overall Architecture

SAP consists of three core components:

1. **Zigzag Trajectory Decomposition**: Decomposes 4K panoramas along a zigzag path into a series of overlapping perspective patches, simulating a sequence of video frames.
2. **SAM2 Adaptation**: Freezes the main body of SAM2 and only fine-tunes the memory attention module to adapt to panoramic scenes.
3. **InfiniGen Data Synthesis**: Utilizes generative models to synthesize 183K 4K panoramas (6.4M masks in total) to address the lack of training data.

### Key Designs

#### Key Design 1: Zigzag Trajectory Decomposition

- Samples the ERP panorama along the horizontal and vertical directions of a zigzag path into N overlapping perspective patches.
- Each patch corresponds to a viewpoint on the sphere without ERP distortion.
- Sufficient overlap between patches ensures the continuity of objects across patches.
- The trajectory order is designed to ensure that spatially adjacent patches remain adjacent in the sequence, allowing SAM2's memory module to model spatio-temporal consistency.
- **Key Point**: Trajectory-aligned training yields an additional +10.7 mIoU compared to naive scanning.

#### Key Design 2: Efficient SAM2 Fine-Tuning

- Freezes the image encoder and mask decoder of SAM2.
- Only fine-tunes the memory attention module to learn spatio-temporal dependencies under panoramic patch sequences.
- Treats each patch as a frame in a video, where spatial relations between patches are analogous to temporal relations in videos.
- Parameter-efficient—only requires training a small number of parameters in the memory module.

#### Key Design 3: InfiniGen Large-Scale 4K Panorama Synthesis

- Uses diffusion models to automatically synthesize 183K 4K panoramas.
- Automatically generates semantic segmentation masks (6.4M masks) for each panorama.
- The synthetic data covers diverse scenes (indoor/outdoor/natural/urban), improving generalization.
- Tackles the core bottleneck of high annotation costs and data scarcity in panoramic segmentation.

## Key Experimental Results

### Main Results

| Model | Zero-shot mIoU | Large Object mIoU | Parameter Count |
|------|-----------|------------|--------|
| SAM2-tiny | 51.6 | — | — |
| SAM2-large | 58.6 | — | — |
| **SAP-tiny** | **75.8** | — | — |
| **SAP-large** | **75.8+** | +30.6 vs SAM2 | — |

- Zero-shot mIoU improvement: **+17.2** (outperforming the largest SAM2 model).
- The improvement for large objects is the most significant: **+30.6**.

### Ablation Study

| Configuration | mIoU | Details |
|------|------|------|
| Naive scanning (raster scan) | ~65.1 | Patches arranged in raster order |
| Zigzag trajectory (w/o aligned training) | ~69.5 | Trajectory outperforms raster |
| **Zigzag + Trajectory-Aligned Training** | **75.8** | +10.7 vs naive scanning |
| Without synthetic data | ~68 | Synthetic data contributes ~7-8 mIoU |
| Full SAM2 fine-tuning | ~74 | Memory-only fine-tuning is actually better |

### Key Findings

- Trajectory-aligned training along the zigzag path is the most significant contributor (+10.7 mIoU).
- SAP-tiny outperforms SAM2-large, demonstrating that structural adaptation is more crucial than scaling up the model.
- The performance gain for large objects is significantly greater than for small objects (+30.6 vs. +17.2 average), showing that panoramic distortion affects large objects most severely.
- The scale and diversity of synthetic data are vital for generalization.

## Highlights & Insights

1. **Elegant Problem Reformulation**: By reformulating panoramic segmentation as perspective video segmentation, the method cleverly reuses SAM2's powerful temporal modeling capabilities.
2. **Importance of Trajectory Design**: The zigzag trajectory maintains spatial proximity, enabling the memory module to effectively utilize context.
3. **Data Engine**: Synthesizing 183K panoramas with automatic annotation using InfiniGen is a practical solution to overcome the data bottleneck.
4. **Efficiency Advantage**: Fine-tuning only the memory module offers low training costs, making it suitable for practical deployment.

## Limitations & Future Work

- A domain gap may exist between synthetic data and real-world scenes, especially regarding complex lighting and fine-grained textures.
- The number of patches and overlap ratio for the zigzag trajectory are hyperparameters that require tuning.
- Currently only supports semantic segmentation and does not extend to instance/panoptic segmentation.
- Not validated on non-ERP panoramic formats (e.g., cubemaps).

## Related Work & Insights

- **SAM/SAM2**: This work performs minimal adaptation on SAM2, validating the effectiveness of the paradigm: foundation models + domain adaptation.
- **Trans4PASS**: Traditional panoramic segmentation methods typically design specialized deformable convolutions/attention to address ERP distortion. This work demonstrates that "decomposing into perspective views" is simpler and more effective.
- **Insights**: The video segmentation paradigm can be generalized to other tasks requiring the processing of high-resolution or non-standard projection images (e.g., satellite images, fisheye images).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The panoramic-to-video reformulation is novel and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Thorough ablation studies and comprehensive experiments on synthetic data.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and intuitive methodology description.
- **Value**: ⭐⭐⭐⭐ — Wide range of application scenarios for 4K panoramic segmentation.
- **Comprehensive Recommendation**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Segment Any Motion in Videos](segment_any_motion_in_videos.md)
- [\[CVPR 2025\] GleSAM: Segment Any-Quality Images with Generative Latent Space Enhancement](segment_any-quality_images_with_generative_latent_space_enhancement.md)
- [\[CVPR 2025\] SAM2-LOVE: Segment Anything Model 2 in Language-Aided Audio-Visual Scenes](sam2-love_segment_anything_model_2_in_language-aided_audio-visual_scenes.md)
- [\[CVPR 2025\] FineCaption: Compositional Image Captioning Focusing on Wherever You Want at Any Granularity](finecaption_compositional_image_captioning_focusing_on_wherever_you_want_at_any_.md)
- [\[ICCV 2025\] OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation](../../ICCV2025/segmentation/omnisam_omnidirectional_segment_anything_model_for_uda_in_panoramic_semantic_seg.md)

</div>

<!-- RELATED:END -->
