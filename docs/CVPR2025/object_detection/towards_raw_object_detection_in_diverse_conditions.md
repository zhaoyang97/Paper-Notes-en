---
title: >-
  [Paper Note] Towards RAW Object Detection in Diverse Conditions
description: >-
  [CVPR 2025][Object Detection][RAW Image Detection] This paper proposes the AODRaw dataset (7,785 high-resolution real-world RAW images, 62 categories, 9 illumination/weather conditions) along with a RAW-domain pre-training and cross-domain distillation scheme, achieving superior RAW object detection performance under diverse adverse conditions without requiring an ISP module.
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "RAW Image Detection"
  - "Adverse Condition Perception"
  - "Cross-Domain Distillation"
  - "Dataset"
date: 2026-05-08
content_hash: 6cbf64c7db571873
---

# Towards RAW Object Detection in Diverse Conditions

**Conference**: CVPR 2025  
**arXiv**: [2411.15678](https://arxiv.org/abs/2411.15678)  
**Code**: [GitHub](https://github.com/lzyhha/AODRaw)  
**Area**: Object Detection  
**Keywords**: RAW Image Detection, Adverse Condition Perception, Cross-Domain Distillation, Dataset

## TL;DR

This paper proposes the AODRaw dataset (7,785 high-resolution real-world RAW images, 62 categories, 9 illumination/weather conditions) along with a RAW-domain pre-training and cross-domain distillation scheme, achieving superior RAW object detection performance under diverse adverse conditions without requiring an ISP module.

## Background & Motivation

- Existing object detection methods mainly process sRGB images, which are compressed from RAW data via an ISP, potentially losing critical details under complex illumination and adverse weather conditions.
- RAW images preserve a higher bit depth (e.g., 16-bit) and contain richer distinguishable information, showing distinct advantages under challenging environments such as low light, fog, and rain.
- Existing RAW object detection datasets suffer from severe limitations: they have few categories (e.g., LOD has only 8, RAOD has only 6) and cover simplistic scenarios (chiefly daytime/low-light), lacking diverse real-world adverse weather conditions.
- Conventional RAW detection approaches rely on a neural ISP to convert RAW inputs to the sRGB domain, inducing extra computational overhead and failing to fully exploit authentic RAW characteristics.
- Directly transferring sRGB pre-trained models to the RAW domain yields sub-optimal performance due to the domain gap.
- An extensive, multi-condition RAW detection benchmark as well as an efficient detection scheme without ISP adapters is highly needed.

## Method

### Overall Architecture

The contributions are twofold: (1) constructing the large-scale, multi-condition AODRaw dataset; (2) proposing a cross-domain distillation-based RAW pre-training scheme to train the backbone directly in the RAW domain, thereby eliminating the sRGB-RAW domain gap. The overall workflow consists of generating synthetic RAW data (ImageNet-RAW) from ImageNet-1K using an unprocessing method, pre-training the backbone on this synthetic data, and then fine-tuning the detector on the real-world RAW dataset. During pre-training, academic knowledge distillation from an sRGB pre-trained teacher model helper facilitates better representation learning in the RAW model.

### Key Designs

**1. AODRaw Dataset Construction**

- **Function**: Provides a large-scale, multi-condition, high-resolution real-world RAW detection benchmark.
- **Mechanism**: The dataset contains 7,785 RAW images at $6000 \times 4000$ resolution with 135,601 annotated instances covering 62 categories. It spans 2 illumination levels (daytime, low-light) and 3 weather states (clear, rainy, foggy) to form 9 combined conditions featuring both indoor and outdoor environments. The images average 17.4 annotated instances each in the COCO format.
- **Design Motivation**: Existing RAW detection datasets are overly restricted in terms of category count, condition variety, and scale, which hinders a thorough evaluation of RAW detection under authentic adverse conditions.

**2. Synthetic ImageNet-RAW Pre-training**

- **Function**: Eliminates the domain gap between sRGB pre-training and RAW fine-tuning.
- **Mechanism**: sRGB images from ImageNet-1K are reversely converted into 16-bit RAW format utilizing an unprocessing method while simulating sensor noise. This unprocessing pipeline is embedded randomly as data augmentation, adjusting brightness and noise levels dynamically at each iteration to bolster generalization.
- **Design Motivation**: Constructing a real RAW dataset of ImageNet scale is prohibitively expensive; hence, a synthetic strategy translates established databases into RAW data for low-cost pre-training.

**3. Cross-Domain Knowledge Distillation**

- **Function**: Helps the RAW-domain model learn higher-quality feature representations.
- **Mechanism**: An off-the-shelf sRGB pre-trained model serves as the teacher network, guiding the RAW student model via feature distillation. Because camera noise in raw sensors impedes direct representation learning, distillation effectively bridges this learning gap.
- **Design Motivation**: Empirical results show that learning rich features via RAW pre-training is more difficult due to noise degradation. Utilizing established sRGB pre-training knowledge can successfully ease the optimization difficulty.

### Loss & Training

- The detector uses standard detection objectives (e.g., multi-stage classification and regression losses in Cascade R-CNN).
- Distillation loss: a feature alignment loss matching the teacher (sRGB pre-trained) and student (RAW pre-trained) models.
- Training spans 48 epochs with a batch size of 16 (100 epochs for Deformable DETR).
- RAW images are demosaiced from Bayer $1 \times H \times W$ format into $3 \times H \times W$ patterns, and gamma correction is applied to accelerate convergence.
- Evaluation runs under two settings: downsampling the inputs to $2000 \times 1333$, or cropping them into $1280 \times 1280$ patches (with an overlap of 300).

## Key Experimental Results

### Main Results

| Method | Backbone | Pre-train $\rightarrow$ Fine-tune | AP | AP_normal | AP_low | AP_rain | AP_fog |
|------|------|------------|------|-----------|--------|---------|--------|
| Cascade RCNN | ConvNeXt-T | sRGB $\rightarrow$ sRGB | 34.0 | 37.0 | 31.5 | 32.9 | 27.2 |
| Cascade RCNN | ConvNeXt-T | sRGB $\rightarrow$ RAW | 33.7 | 36.8 | 31.3 | 31.3 | 27.2 |
| Cascade RCNN | ConvNeXt-T | RAW $\rightarrow$ RAW | **34.8** | **37.7** | **32.1** | **36.1** | **28.4** |
| RAOD | ConvNeXt-T | sRGB+ISP $\rightarrow$ RAW | 34.4 | 37.3 | 32.4 | 37.7 | 29.4 |

### Ablation Study

| Train Domain | Eval Domain | AP | AP50 | AP75 |
|--------|--------|------|------|------|
| sRGB | sRGB | 34.0 | 52.7 | 36.3 |
| sRGB | RAW | 33.7 | 52.0 | 35.9 |
| RAW | RAW | 34.8 | 53.3 | 36.7 |

Cross-domain testing shows a noticeable performance drop, validating the existence of the sRGB-RAW domain gap.

### Key Findings

1. The Cascade R-CNN incorporating RAW-domain pre-training and distillation achieves an AP of 34.8%, surpassing the sRGB baseline of 34.0% without relying on any ISP modules.
2. The advantage of RAW-based detection is highly pronounced under severe weather: AP_rain climbs from 32.9% to 36.1% (+3.2%), far exceeding the gains achieved under normal conditions.
3. Modeling with sRGB pre-trained weights and fine-tuning on RAW data performs slightly worse than fine-tuning on sRGB (33.7 vs 34.0), verifying the underlying domain gap.
4. The dataset exhibits a high density of small objects and a long-tailed category distribution, presenting a stiffer detection challenge.

## Highlights & Insights

- Constructs the first large-scale real-world RAW detection dataset spanning 9 distinct illumination/weather combinations, successfully addressing a crucial data gap.
- PROPOSES a clean and efficient "synthetic RAW pre-training + distillation" paradigm, which outperforms neural ISP-based alternatives without introducing additional ISP modules.
- Systematically unmasks the domain gap between sRGB and RAW, which is shown to be particularly prominent under adverse environmental conditions.
- Features a robust experimental layout assessing both sRGB and RAW detection architectures.

## Limitations & Future Work

- Differences remain between synthetic RAW data and real-world RAW, which may restrict the pre-training effect.
- The dataset scale (7,785 images) remains relatively small for large-scale detection; future expansion should be pursued.
- This work focuses on the classification and bounding box regression tasks; the capabilities of RAW data in fine-grained tasks such as instance or panoptic segmentation have yet to be explored.
- The distillation protocol is fairly straightforward; more advanced distillation concepts (e.g., feature pyramid-level distillation) could be investigated.
- RAW domain discrepancy across different camera models and sensors was not yet evaluated.

## Related Work & Insights

- **RAW-Adapter**: Utilizes a trainable ISP to bridge the gap between sRGB pre-training and RAW fine-tuning; this paper proves that direct RAW pre-training offers a more elegant solution.
- **Unprocessing**: Synthesizing RAW data by reversely mapping sRGB images back to the RAW format serves as the cornerstone of the pre-training strategy in this work.
- **LOD/RAOD**: Early RAW detection datasets with limited category volumes and scenario coverages; AODRaw represents a massive expansion of these formats.
- Insight: RAW inputs might yield similar gains under adverse environmental conditions for other dense prediction tasks (e.g., semantic segmentation, depth estimation).

## Rating

- **Novelty**: ⭐⭐⭐ — The dataset construction is highly valuable; the proposed strategy (synthetic pre-training + distillation) is primarily an application of existing concepts.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Highly comprehensive benchmarking that addresses multiple detectors, backbones, and optimization configurations.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured and accompanied by clear, thorough analysis.
- **Value**: ⭐⭐⭐⭐ — The AODRaw dataset fills an important gap in the field and provides a solid foundation for further research on adverse condition perception.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SimROD: A Simple Baseline for Raw Object Detection with Global and Local Enhancements](../../AAAI2026/object_detection/simrod_a_simple_baseline_for_raw_object_detection_with_global_and_local_enhancem.md)
- [\[CVPR 2025\] MulSen-AD: Multi-Sensor Object Anomaly Detection](mulsen_ad_multi_sensor_anomaly_detection.md)
- [\[CVPR 2025\] Test-Time Backdoor Detection for Object Detection Models](test-time_backdoor_detection_for_object_detection_models.md)
- [\[CVPR 2026\] SpiralDiff: Spiral Diffusion with LoRA for RGB-to-RAW Conversion Across Cameras](../../CVPR2026/object_detection/spiraldiff_spiral_diffusion_with_lora_for_rgb-to-raw_conversion_across_cameras.md)
- [\[CVPR 2026\] See What We Cannot See: A Geo-guided Reasoning Benchmark for Object Counting under Adverse Earth Observation Conditions](../../CVPR2026/object_detection/see_what_we_cannot_see_a_geo-guided_reasoning_benchmark_for_object_counting_unde.md)

</div>

<!-- RELATED:END -->
