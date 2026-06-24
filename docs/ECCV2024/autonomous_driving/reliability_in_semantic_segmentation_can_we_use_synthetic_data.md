---
title: >-
  [Paper Note] Reliability in Semantic Segmentation: Can We Use Synthetic Data?
description: >-
  [ECCV 2024][Autonomous Driving][Semantic Segmentation] This work presents the first systematic utilization of Stable Diffusion to generate synthetic OOD data for a comprehensive reliability assessment of semantic segmentation models, encompassing robustness evaluation under covariate shift, OOD object detection, and model calibration. It demonstrates that evaluation results on synthetic data correlate highly with those on real OOD data.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Semantic Segmentation"
  - "Synthetic Data"
  - "Reliability Assessment"
  - "OOD Detection"
  - "Stable Diffusion"
date: 2026-05-08
content_hash: f1374858be1415b5
---

# Reliability in Semantic Segmentation: Can We Use Synthetic Data?

**Conference**: ECCV 2024  
**arXiv**: [2312.09231](https://arxiv.org/abs/2312.09231)  
**Code**: Yes  
**Area**: Autonomous Driving  
**Keywords**: Semantic Segmentation, Synthetic Data, Reliability Assessment, OOD Detection, Stable Diffusion

## TL;DR

This work presents the first systematic utilization of Stable Diffusion to generate synthetic OOD data for a comprehensive reliability assessment of semantic segmentation models, encompassing robustness evaluation under covariate shift, OOD object detection, and model calibration. It demonstrates that evaluation results on synthetic data correlate highly with those on real OOD data.

## Background & Motivation

In safety-critical applications such as autonomous driving, evaluating the robustness of perception models against covariate shift and their ability to detect out-of-distribution (OOD) inputs is of paramount importance. However, collecting and annotating real OOD data is extremely challenging and expensive—scenarios like extreme weather (heavy snow, dense fog) or rare conditions (floods, fires) are difficult to collect systematically.

Existing robustness assessments primarily rely on: (1) real shift datasets like ACDC (which have limited coverage); and (2) synthetic perturbations such as adding noise or blur (which do not correlate with robustness to real-world shifts). While Taori et al. criticized synthetic robustness benchmarks for being disconnected from real shifts, this work points out that with the rapid advancement of generative models, it is now possible to generate sufficiently realistic synthetic data for meaningful virtual evaluation.

The Core Problem of this paper is: **Can synthetic data replace real OOD data to evaluate the reliability of semantic segmentation models?**

## Method

### Overall Architecture

The framework consists of two pipelines, both based on a pre-trained Stable Diffusion 1.5 model:

1. **Covariate Shift Generation**: ControlNet is fine-tuned on Cityscapes using semantic masks to control the generation. Images of driving scenes across different domains (night, rain, snow, fog, India) are generated zero-shot via text prompts.
2. **OOD Object Inpainting**: Utilizing the inpainting capability of SD, 42 categories of objects not belonging to Cityscapes (e.g., babies, benches, billboards) are inserted zero-shot into Cityscapes images. Grounded SAM is then used to extract the masks of these inserted objects.

### Key Designs

**Covariate Shift Data Generation**:
- ControlNet is fine-tuned on the Cityscapes training set (for only 2,100 steps), conditioned on semantic masks, with captions extracted by CLIP-interrogator as text inputs.
- During inference, OOD domain descriptions are appended to the captions (e.g., `[caption, in night]`). Semantic masks from the Cityscapes validation set are used as conditions to generate target-domain images zero-shot.
- The generated images automatically inherit the mask annotations, eliminating the need for manual labeling.

**OOD Object Inpainting Pipeline**:
- Insertion positions and sizes are chosen randomly, and the region is cropped from the image and upscaled to 512×512.
- SD inpainting is used with the object name as the prompt to generate the object, employing an inner-outer dual-region strategy to maintain background consistency.
- Grounded SAM is used to extract the object mask, followed by a noise-and-denoise refinement step to blend edges.
- Two sets are constructed: an automatic set of 23,040 images and a curated set of 656 manually selected images.

**Evaluation Protocol**:
- 40 open-source semantic segmentation models pre-trained solely on Cityscapes are gathered, covering various architectures (from ConvNets to Transformers) and scales.
- Pearson Correlation Coefficient (PCC) is utilized to measure the correlation between evaluations performed on synthetic data and those on real OOD data.

### Loss & Training

This work does not involve training the segmentation models; indeed, it evaluates existing models. ControlNet is trained using the standard reconstruction loss. For model calibration, temperature scaling—a simple and efficient post-processing calibration method—is utilized to optimize the temperature parameter on the synthetic OOD data.

## Key Experimental Results

### Main Results

Pearson correlation coefficient between synthetic and real evaluations under covariate shift:

| Generation Method | OOD Knowledge? | OOD Data? | Night | Rain | Snow | Fog | India |
|---|---|---|---|---|---|---|---|
| GAN-based TSIT | No | **Yes** | 0.83 | 0.84 | 0.81 | - | - |
| Physics-based Fog Sim | **Yes** | No | - | - | - | 0.82 | - |
| **Ours (SD1.5)** | **No**| **No** | **0.85** | **0.86** | **0.85** | 0.77 | 0.71 |
| Ours (SDXL) | No | No | 0.84 | 0.90 | 0.82 | **0.89** | **0.93** |

OOD object detection improvement experiments (on SMIYC RoadAnomaly21):

| Method | AUROC↑ | AUPR↑ | FPR95↓ |
|---|---|---|---|
| RbA (Swin-B) baseline | 95.6 | 78.4 | 11.8 |
| + COCO Data | 97.8 | 85.3 | 8.5 |
| + **Ours (curated)** | **97.2** | **84.9** | **8.1** |
| + Ours (all) | 97.3 | 84.8 | 8.2 |
| RbA (Swin-L) baseline | 96.4 | 79.6 | 15.0 |
| + COCO Data | 98.2 | 88.7 | 8.2 |
| + **Ours (curated)** | **97.2** | **88.0** | **7.9** |
| + **Ours (all)** | **98.1** | **88.6** | 8.3 |

### Ablation Study

Calibration experiments—Success rate of calibration using synthetic data (proportion of models with improved ECE):

| OOD Domain | Domain Distance | Synthetic Calibration Success Rate |
|---|---|---|
| India | Small | 72.5% |
| Fog | Medium | >90% |
| Rain | Medium | >90% |
| Snow | Large | >90% |
| Night | Large | >90% |

Key finding: Stable and reliable robustness evaluation results can be obtained with only ~500 synthetic images.

### Key Findings

1. **The larger the domain gap, the more pronounced the advantages of synthetic data**: In domains with small shifts such as Fog / India, the original Cityscapes validation set can already predict OOD performance reasonably well. However, in large-shift domains like Night / Snow, the PCC of synthetic data far exceeds that of Cityscapes (e.g., Night: $PCC_{Syn}$ is more than 2x higher than $PCC_{CS}$).
2. **Cityscapes mIoU does not predict night-time performance**: A high Cityscapes mIoU does not imply high night-time robustness, whereas synthetic night-time mIoU strongly correlates with real night-time mIoU.
3. **High synthetic-to-real correlation in OOD detection**: The curated synthetic set consistently achieves a PCC of around 0.8 across various anomaly metrics, while the fully automated set also provides acceptable results.
4. **Synthetic data can effectively train OOD detectors**: The RbA model trained on synthetic data achieves performance comparable to the variant augmented with real COCO data.
5. **Consistent architectural trends in model rankings**: Transformer- and ConvNeXt-based backbones exhibit stronger robustness on both synthetic and real OOD data.

## Highlights & Insights

- **Zero-shot generation paradigm**: ControlNet only needs to be fine-tuned on in-domain data, after which test data for arbitrary OOD domains can be generated zero-shot via text prompts, offering extreme scalability.
- **Immense practical value**: Systematic collection of real data for extreme scenarios like floods or fires is virtually impossible, but they can be generated effortlessly using text prompts.
- **Dual value in both evaluation and training**: Synthetic data serves not only to evaluate model robustness (evaluation side) but can also be utilized for calibration and OOD detection training (training side).
- **Differing requirements for data quality**: Evaluation requires high-quality synthetic data (where the curated set excels), whereas training OOD detectors does not—even flawed synthetic data proves to be effective.
- The generated synthetic data has been integrated into the official BRAVO benchmark.

## Limitations & Future Work

- SD 1.5 shows relatively lower correlation in the Fog and India domains (0.77 and 0.71). Stronger generative models (e.g., SDXL) can yield significant improvements, despite being more computationally expensive.
- Temperature scaling calibration does not always guarantee ECE improvement, a limitation that persists even when using real data.
- The inpainting quality of OOD objects still has room for improvement; some generations exhibit discrepancies in color saturation or result in incomplete objects.
- This study focuses solely on semantic segmentation; whether the findings generalize to other tasks, such as object detection or depth estimation, remains to be validated.
- The diversity of generated data is constrained by the design of text prompts; more systematic prompting strategies warrant further investigation.

## Related Work & Insights

- Echoes but deepens the work of RELIS (Jorge et al.): while RELIS aggregates all weather conditions for comprehensive analysis, this work conducts a domain-specific analysis, discovering that tiny domain gaps and large domain gaps are fundamentally different.
- The mask-to-image generation capability of ControlNet allows semantic annotations to be obtained for free. This approach can be applied to other scenarios requiring out-of-domain labeled data.
- Evaluation using synthetic data can serve as the first step in a complete validation pipeline, filtering out non-robust model prototypes to reduce overall operational costs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic evaluation of segmentation reliability using generative models
- **Technical Quality**: ⭐⭐⭐⭐ — Large-scale evaluation of 40 models with rigorous statistical analysis
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers three aspects: robustness, OOD detection, and calibration
- **Value**: ⭐⭐⭐⭐⭐ — Can be directly incorporated into safety-critical system validation pipelines
- **Overall Recommendation**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ClimaOoD: Improving Anomaly Segmentation via Physically Realistic Synthetic Data](../../CVPR2026/autonomous_driving/climaood_improving_anomaly_segmentation_via_physically_realistic_synthetic_data.md)
- [\[ECCV 2024\] Rethinking Data Augmentation for Robust LiDAR Semantic Segmentation in Adverse Weather](rethinking_data_augmentation_for_robust_lidar_semantic_segmentation_in_adverse_w.md)
- [\[ECCV 2024\] ItTakesTwo: Leveraging Peer Representations for Semi-supervised LiDAR Semantic Segmentation](ittakestwo_leveraging_peer_representations_for_semi-supervised_lidar_semantic_se.md)
- [\[ICCV 2025\] Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving](../../ICCV2025/autonomous_driving/unraveling_the_effects_of_synthetic_data_on_end-to-end_autonomous_driving.md)
- [\[ECCV 2024\] SFPNet: Sparse Focal Point Network for Semantic Segmentation on General LiDAR Point Clouds](sfpnet_sparse_focal_point_network_for_semantic_segmentation_on_general_lidar_poi.md)

</div>

<!-- RELATED:END -->
