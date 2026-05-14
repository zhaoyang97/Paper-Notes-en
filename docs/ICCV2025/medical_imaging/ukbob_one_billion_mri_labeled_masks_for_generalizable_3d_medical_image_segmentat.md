---
title: >-
  [Paper Note] UKBOB: One Billion MRI Labeled Masks for Generalizable 3D Medical Image Segmentation
description: >-
  [ICCV 2025][Medical Imaging][Medical image segmentation] This paper introduces UKBOB—the largest annotated medical image segmentation dataset to date (51,761 MRI 3D volumes, 72 organ classes…
tags:
  - "ICCV 2025"
  - "Medical Imaging"
  - "Medical image segmentation"
  - "UK Biobank"
  - "large-scale dataset"
  - "automated annotation quality control"
  - "test-time adaptation"
date: 2026-05-08
content_hash: 74d52aad105bb865
---

# UKBOB: One Billion MRI Labeled Masks for Generalizable 3D Medical Image Segmentation

**Conference**: ICCV 2025
**arXiv**: [2504.06908](https://arxiv.org/abs/2504.06908)
**Code**: [https://emmanuelleb985.github.io/ukbob](https://emmanuelleb985.github.io/ukbob)
**Area**: Medical Image Segmentation / Foundation Models / Datasets
**Keywords**: Medical image segmentation, UK Biobank, large-scale dataset, automated annotation quality control, test-time adaptation

## TL;DR

This paper introduces UKBOB—the largest annotated medical image segmentation dataset to date (51,761 MRI 3D volumes, 72 organ classes, 1.37 billion 2D segmentation masks)—and proposes a Specialized Organ Label Filter (SOLF) for cleaning automated annotations and an Entropy Test-Time Adaptation (ETTA) method for handling domain shift under noisy labels. The resulting Swin-BOB foundation model achieves state-of-the-art performance on the BRATS and BTCV benchmarks.

## Background & Motivation

**Background**: Large-scale annotated datasets (e.g., ImageNet, LAION) are foundational to the success of computer vision foundation models. However, the medical imaging domain lacks large-scale annotated datasets due to privacy regulations, high annotation costs, and logistical complexity. Existing datasets such as BRATS (1,470 samples / 3 classes) and BTCV (50 samples / 12 classes) remain limited in scale.

**Limitations of Prior Work**: (1) Existing medical datasets either lack diversity or are too small to train generalizable foundation models; (2) automated annotation can address the scale problem but introduces noisy labels; (3) train-test domain shift (across different scanners, acquisition protocols, and modalities) causes performance degradation.

**Key Challenge**: Medical imaging urgently requires large-scale annotated data to train foundation models, yet manually annotating 17.9 million images is infeasible, while automated annotation inevitably introduces noise.

**Goal**: (a) Construct the largest-scale medical image segmentation annotation dataset; (b) design a label quality control mechanism; (c) address domain shift at test time for models trained on noisy labels.

**Key Insight**: The UK Biobank's 51,761 whole-body MRI scans are leveraged as the data source, with TotalVibeSegmentator used for automated annotation, followed by a statistically-grounded geometric filter to clean the labels.

**Core Idea**: Construct an ultra-large-scale medical image dataset via automated annotation combined with statistical filtering, then apply entropy-driven test-time adaptation to handle residual noise, yielding a highly generalizable 3D medical segmentation foundation model.

## Method

### Overall Architecture

Pipeline: UK Biobank MRI data → TotalVibeSegmentator automated annotation of 72 organ classes → SOLF statistical filtering to remove noisy labels → Swin-UNetr training to obtain the Swin-BOB foundation model → downstream fine-tuning (BTCV, BRATS, etc.) → ETTA test-time adaptation for improved robustness.

### Key Designs

1. **UKBOB Dataset Construction**:

    - **Function**: Generate segmentation annotations for 72 organ classes from 51,761 neck-to-knee whole-body MRI scans in UK Biobank (4 sequences: fat-only, water-only, in-phase, out-of-phase).
    - **Mechanism**: TotalVibeSegmentator is used for automated annotation, producing 17.9M 2D images and 1.37 billion 2D segmentation masks.
    - **Scale Comparison**: Approximately 237× more masks than the previously largest dataset, TotalSegmentator (1,204 samples / 104 classes / 5.8M masks).
    - **Manual Validation Set UKBOB-manual**: 300 MRI scans, 11 abdominal organs, 3,000 2D images, serving as a quality verification benchmark for automated annotations.

2. **Specialized Organ Label Filter (SOLF)**:

    - **Function**: A statistical filter based on organ geometric properties, designed to remove erroneous labels from automated annotations.
    - **Mechanism**: For each organ class $c$, three geometric features are computed:
        - **Normalized Volume**: $v_c = V_c / V_{\text{body}}$ (organ volume relative to whole-body volume)
        - **Sphericity**: $\Phi_c = \pi^{1/3}(6V_c)^{2/3}/A_c$ (degree to which the shape approximates a sphere)
        - **Eccentricity**: $E_c = \sqrt{1-\lambda_{\min}/\lambda_{\max}}$ (elongation of the shape)
    - **Filtering Rule**: For each feature, samples in the extreme $\epsilon$ percentiles are excluded. A label is flagged as inaccurate when at least two of the three features fall outside their normal range.
    - **Design Motivation**: Human organ morphology follows geometrically regular patterns shaped by evolution. A genuinely abnormal patient is unlikely to deviate simultaneously on two or more independent geometric features; therefore, multi-feature joint anomalies are more likely to indicate annotation errors.
    - **vs. IQR Filtering**: SOLF exploits patient-level whole-body statistics (normalized volume) and organ-specific geometric features, yielding more precise filtering than simple single-statistic IQR approaches.

3. **Entropy Test-Time Adaptation (ETTA)**:

    - **Function**: Optimizes batch normalization parameters at test time to improve segmentation robustness.
    - **Mechanism**: Given a test sample $\mathbf{x}$, the entropy loss over predicted probabilities is computed as:
    $$\mathcal{L}_{\text{ent}} = -\frac{1}{N}\sum_{i=1}^N \sum_{c=1}^C p_{i,c} \log p_{i,c}$$
    Only the BN parameters $\theta_{\text{BN}}$ are updated while all other parameters are frozen:
    $$\theta_{\text{BN}}^* = \arg\min_{\theta_{\text{BN}}} \mathcal{L}_{\text{ent}}(f_{\theta_{\text{fixed}}, \theta_{\text{BN}}}(\mathbf{x}))$$
    - **Design Motivation**: When training data contains noisy labels, standard TTA methods may lack robustness. ETTA adapts to test samples by encouraging low-entropy (high-confidence) predictions while updating only a small number of BN parameters to remain efficient.
    - **Architecture-Agnostic**: Applicable to any segmentation network containing batch normalization layers.

4. **Swin-BOB Foundation Model**:

    - **Function**: Swin-UNetr pre-trained on filtered UKBOB, serving as a 3D medical segmentation foundation model.
    - **Training Configuration**: $96^3$ crop size, batch size 8, AdamW optimizer, 3,000 epochs, 2× A6000 GPUs.
    - **Loss Function**: Binary Cross-Entropy + Dice Similarity Coefficient.
    - **Downstream Fine-tuning**: Same configuration, with warm-up reduced to 50 epochs for a total of 500 epochs.

### Loss & Training

Both pre-training and fine-tuning use Binary Cross-Entropy + DSC. ETTA employs entropy loss at test time, updating only BN parameters.

## Key Experimental Results

### Main Results

**BTCV Abdominal CT Segmentation (12 classes, standard benchmark)**:

| Model | Dice Score |
|-------|-----------|
| UNetr | 0.856 |
| Swin-UNetr | 0.869 |
| nnUNet | 0.802 |
| MedSegDiff | 0.879 |
| **Swin-BOB (Ours)** | **0.892** |
| MedSegDiff-V2 (10-fold ens.) | 0.895 |
| **Swin-BOB (10-fold ens.)** | **0.897** |

**BRATS Brain Tumor MRI Segmentation (3 classes)**:

| Model | Dice Score | Hausdorff Distance |
|-------|-----------|-------------------|
| SegResNet | 0.890 | 8.650 |
| Swin-UNetr | 0.886 | 9.016 |
| **Swin-BOB (Ours)** | **0.894** | **8.650** |

### Ablation Study

**SOLF Filtering Effect (Zero-Shot Generalization to BTCV)**:

| Filter Configuration | BTCV Mean Dice | AMOS Mean Dice |
|---------------------|---------------|---------------|
| No filtering | 0.856 | 0.818 |
| + Volume filtering | 0.875 | 0.832 |
| + **Full SOLF** | **0.882** | **0.840** |

**ETTA Effect (Multiple Models × Multiple Datasets)**:

| Configuration | BTCV Dice | AMOS Dice | BRATS Dice |
|--------------|-----------|-----------|-----------|
| Swin-BOB | 0.883 | 0.847 | 0.882 |
| Swin-BOB + TTA [baseline] | 0.883 | 0.857 | 0.887 |
| **Swin-BOB + ETTA** | **0.892** | **0.864** | **0.894** |

**SOLF Threshold $\varepsilon$ Ablation**:

| $\varepsilon$ | BTCV Dice |
|--------------|-----------|
| 0 (no filtering) | 0.792 |
| 1 | 0.884 |
| **2** | **0.892** |
| 3 | 0.766 |
| 4–5 | 0.745 |

### Key Findings

- **SOLF significantly improves label quality**: Zero-shot BTCV Dice improves from 0.856 to 0.882 (+2.6%), substantially outperforming simple IQR filtering.
- **$\varepsilon = 2$ is the optimal filtering threshold**: A too-lenient threshold ($\varepsilon = 0$) retains too much noise, while a too-strict threshold ($\varepsilon \geq 3$) discards valid data.
- **ETTA consistently improves all models**: Outperforms standard TTA across all 9 combinations of 3 networks × 3 datasets.
- **Data scaling law**: Increasing UKBOB usage from 10% to 100% yields consistent Dice Score improvements on BTCV and BRATS, validating the value of large-scale pre-training.
- **Cross-modal generalization**: Swin-BOB pre-trained on MRI achieves strong zero-shot transfer to CT data (BTCV), indicating that the learned representations generalize across modalities.
- **Manual annotation validation**: Unfiltered UKBOB annotations achieve Dice scores of 0.873 (abdomen) and 0.811 (spine) against manual annotations; after SOLF filtering, these improve to 0.891 and 0.867.

## Highlights & Insights

- **Unprecedented scale**: 1.37 billion 2D masks, 237× larger than the previous largest dataset (TotalSegmentator). This is the first "billion-scale" annotated dataset in medical imaging, potentially catalyzing an "ImageNet moment" for the field.
- **Elegant SOLF design**: Quality control is grounded in evolutionarily determined geometric regularities of organs (volume, sphericity, eccentricity). Multi-feature joint decision-making enhances robustness—a single anomalous feature may reflect genuine pathology, whereas simultaneous anomalies in two or more features are highly likely to indicate annotation errors.
- **Architecture-agnostic ETTA**: Operating solely on BN layer parameters, ETTA functions as a plug-and-play module applicable to any BN-containing segmentation network.
- **Zero-shot MRI → CT generalization**: Demonstrates that anatomical structural knowledge learned from large-scale MRI pre-training transfers across modalities, challenging the conventional wisdom that separate MRI and CT models must be trained independently.

## Limitations & Future Work

- The dataset is limited to neck-to-knee MRI scans and does not cover the head (for brain tasks beyond BRATS) or the extremities.
- Residual noise may remain in automated annotations even after SOLF filtering, particularly for small organs such as the duodenum.
- ETTA requires parameter optimization at test time, increasing inference latency.
- The study is based solely on the Swin-UNetr architecture; more modern architectures (e.g., nnU-Net V2, Universal Model) are not explored.
- UK Biobank participants are predominantly white British, potentially introducing population bias into the dataset.
- Segmentation accuracy for some of the 72 organ classes, particularly small structures, may be limited (per-class detailed results are not reported in the paper).

## Related Work & Insights

- **vs. TotalSegmentator [2023]**: A 104-class CT organ segmentation dataset with only 1,204 samples. Although UKBOB covers fewer classes (72), it contains 43× more samples and operates in the MRI modality.
- **vs. AbdomenAtlas [2024]**: 20,460 CT abdominal scans with 25 classes. UKBOB is larger in scale, covers the whole body, and is in the MRI modality.
- **vs. TotalVibeSegmentator [2025]**: The automated annotation tool used to construct UKBOB, originally trained on only 85+16 samples. UKBOB transforms it into a practically useful resource for foundation model training through large-scale annotation and quality filtering.
- **vs. TENT [2020]**: The standard test-time entropy minimization approach. ETTA builds upon TENT with optimizations tailored for the noisy-label pre-training setting, outperforming standard TTA in 6 out of 9 experimental configurations.
- The UKBOB construction paradigm (large-scale automated annotation + statistical filtering + manual validation subset) is transferable to other medical imaging modalities (CT, X-ray, ultrasound, etc.).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — A breakthrough contribution in dataset scale; SOLF and ETTA are technically incremental but well-motivated and carefully designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Annotation validation, zero-shot generalization, multi-benchmark evaluation, ablation studies, data scaling analysis, and feature visualization are all comprehensively covered.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure; the full pipeline from dataset construction and quality control to model training and evaluation is well presented.
- **Value**: ⭐⭐⭐⭐⭐ — A milestone-level dataset and foundation model for medical imaging with far-reaching impact on the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Progressive Test Time Energy Adaptation for Medical Image Segmentation](progressive_test_time_energy_adaptation_for_medical_image_segmentation.md)
- [\[ICCV 2025\] IDF: Iterative Dynamic Filtering Networks for Generalizable Image Denoising](idf_iterative_dynamic_filtering_networks_for_generalizable_image_denoising.md)
- [\[ICCV 2025\] MRGen: Segmentation Data Engine for Underrepresented MRI Modalities](mrgen_segmentation_data_engine_for_underrepresented_mri_modalities.md)
- [\[CVPR 2026\] MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation](../../CVPR2026/medical_imaging/medclipseg_probabilistic_vision-language_adaptation_for_data-efficient_and_gener.md)
- [\[ICCV 2025\] DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup](dictas_a_framework_for_class-generalizable_few-shot_anomaly_segmentation_via_dic.md)

</div>

<!-- RELATED:END -->
