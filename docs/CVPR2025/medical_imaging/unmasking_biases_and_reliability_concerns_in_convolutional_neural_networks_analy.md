---
title: >-
  [Paper Note] Unmasking Biases and Reliability Concerns in Convolutional Neural Networks Analysis of Cancer Pathology Images
description: >-
  [CVPR 2025][Medical Imaging][CNN] By training four CNNs (ResNet50, DenseNet121, InceptionV3, VGG16) on $20 \times 20$ pixel background patches cropped from 13 cancer pathology benchmark datasets (containing no clinical diagnostic information), this work discovers that classification accuracy far exceeds random guessing (up to 93%). This systematically reveals that CNNs in cancer pathology analysis may rely on dataset collection biases (such as staining protocols and scanner d…
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "CNN"
  - "dataset bias"
  - "cancer pathology"
  - "shortcut learning"
  - "robustness evaluation"
date: 2026-05-08
content_hash: 8568b3b32918983c
---

# Unmasking Biases and Reliability Concerns in Convolutional Neural Networks Analysis of Cancer Pathology Images

**Conference**: CVPR 2025  
**arXiv**: [2603.12445](https://arxiv.org/abs/2603.12445)  
**Code**: To be confirmed  
**Area**: Medical Imaging  
**Keywords**: CNN, dataset bias, cancer pathology, shortcut learning, robustness evaluation

## TL;DR

By training four CNNs (ResNet50, DenseNet121, InceptionV3, VGG16) on $20 \times 20$ pixel background patches cropped from 13 cancer pathology benchmark datasets (containing no clinical diagnostic information), this work discovers that classification accuracy far exceeds random guessing (up to 93%). This systematically reveals that CNNs in cancer pathology analysis may rely on dataset collection biases (such as staining protocols and scanner differences) rather than genuine pathological features for decision-making.

## Background & Motivation

**Background**: CNNs have become the dominant approach for automated diagnostic analysis of cancer pathology images, achieving high classification accuracy across various cancer benchmark datasets, such as melanoma, colorectal cancer, and lung cancer. Due to the black-box nature of CNNs, researchers mainly rely on empirical metrics like accuracy and F1-score to evaluate their performance.

**Limitations of Prior Work**: CNNs have been proven to be extremely sensitive to weak signals invisible to the human eye. Factors such as scanner location differences, CCD sensor temperature, and different technicians' scanning habits can leave subtle traces in images. This so-called "shortcut learning" issue means that models may not have learned disease-related features but instead exploited systematic biases introduced during data collection.

**Key Challenge**: The current ML evaluation paradigm (train/val/test split + accuracy/F1) cannot distinguish whether "a model has truly learned pathological features" or "the model is merely exploiting dataset biases." In high-risk applications like cancer pathology, this evaluation blind spot can lead to severe consequences, as bias-dependent models may completely fail in real-world clinical settings.

**Goal**: (1) Systematically verify whether non-medical biases sufficient to be exploited by CNNs exist across various widely-used benchmark datasets; (2) compare the sensitivity of different CNN architectures to these biases; and (3) provide more cautious recommendations for evaluating CNNs in cancer pathology.

**Key Insight**: If a $20 \times 20$ pixel background region containing no medical information is cropped from an image, a CNN should theoretically perform only at the level of random guessing (50% for binary classification). An actual accuracy far exceeding this level proves the existence of non-medical biases in the dataset.

**Core Idea**: Use a "background cropping test" as a litmus test to conduct a large-scale evaluation of dataset bias in CNNs across 13 cancer pathology benchmark datasets.

## Method

### Overall Architecture

The inputs are 13 public cancer pathology benchmark datasets (including 4 subsets of MedMNIST+, 4 magnification factors of BreakHis, 4 versions of ISIC 2016-2019, and the breast IDC dataset). The output is a comparison of classification accuracy on raw images versus background-cropped images for each dataset-architecture combination. The severity of bias is quantified by comparing the gap (or lack thereof) between the two.

### Key Designs

1. **Background Cropping Strategy**:

    - **Function**: Crop small $20 \times 20$ pixel patches from five fixed locations (top-left, top-right, center, bottom-left, bottom-right) of each raw image.
    - **Mechanism**: The $20 \times 20$ size is too small to contain any meaningful tissue structures or lesion features; in particular, patches cropped from the corners consist almost entirely of background. Standard PIL library functions are used for automatic cropping to avoid human bias.
    - **Design Motivation**: To construct a "zero-information" baseline. If CNNs can still perform classification on these background patches, the presence of bias is undeniable.

2. **Multi-Architecture Comparative Analysis**:

    - **Function**: Utilize four of the most widely used CNN architectures: ResNet50, DenseNet121, InceptionV3, and VGG16.
    - **Mechanism**: All models are fine-tuned using transfer learning from ImageNet pre-trained weights, optimized with Adam (learning rate of 0.0001), 32 batch size, and trained for 5 epochs. The exact same training configuration is applied to all four architectures to ensure a fair comparison.
    - **Design Motivation**: Different architectures have varying receptive field sizes and feature extraction behaviors, potentially leading to different levels of sensitivity to bias signals. This comparison helps reveal which designs are more prone to shortcut learning.

3. **Multi-Dataset Coverage and Unified Experimental Protocol**:

    - **Function**: Cover 13 highly-cited benchmark datasets across multiple modalities, including dermoscopy, ultrasound, CT, and microscopic H&E staining.
    - **Mechanism**: Each dataset is standardly formulated as a binary classification task (cancer presence vs. absence) and partitioned into an 80/10/10 split. The identical training and test procedures are applied to both raw images and the five cropped datasets.
    - **Design Motivation**: Consistent findings across datasets and modalities are more convincing than a discovery on a single dataset, effectively ruling out exceptional cases.

### Loss & Training

Standard cross-entropy loss (softmax + binary classification) is employed, with Adam optimizer ($lr=0.0001$) and ImageNet pre-trained initialization. Training converges within only 5 epochs (benefiting from transfer learning). A complete model is independently trained for each dataset/cropped location combination.

## Key Experimental Results

### Main Results

| Dataset | Modality | Best Raw Image Accuracy | Max Background-Cropped Accuracy | Random Baseline |
|--------|------|-------------------|-------------------|---------|
| BreastMNIST | Ultrasound | ~88.46% | ~75.64% | 50% |
| DermaMNIST | Dermoscopy | ~95.01% | ~93.42% | 50% |
| NoduleMNIST | CT | ~87.10% | ~85.81% | 50% |
| PathMNIST | H&E Tissue | ~98.72% | ~90.07% | 50% |
| BreakHis 40× | H&E Tissue | ~97.69% | ~88% | 50% |
| ISIC-2017 | Dermoscopy | ~80.67% | ~80.67% | 50% |
| ISIC-2018 | Dermoscopy | ~88.29% | ~76.57% | 50% |
| ISIC-2019 | Dermoscopy | ~70.93% | ~63.75% | 50% |

### Ablation Study

| Configuration | DermaMNIST Accuracy | Description |
|------|------------------|------|
| Raw Image (DenseNet121) | 95.01% | Best raw image performance |
| Four Corner Crops (All Architectures) | ~93.42% | Consistent bias signals with almost no performance loss |
| Center Crop (Architecture Dependent) | 93.5-94.5% | Slightly higher than corners, potentially containing minor lesion information |
| After Class Balancing (PathMNIST) | >80% | Bias persists after controlling for class imbalance |
| ISIC-2019 Crops | 54-63% | Dataset with relatively weak bias |

### Key Findings

- **DermaMNIST exhibits the most severe bias**: The classification accuracy of the cropped images from the four corners is almost on par with the raw images (~93.42% vs. ~95%), meaning the CNNs rely almost entirely on non-medical features. This is likely due to the dataset spanning over 20 years and originating from various acquisition devices.
- **VGG16 is the most sensitive to bias**: Across multiple datasets, VGG16's accuracy on cropped images is closest to, or occasionally even exceeds, that on raw images (e.g., on ISIC-2017, cropped accuracy of ~80.5% > raw accuracy of ~79.83%). This may be related to its larger receptive field and fully connected layers.
- **Class balancing does not eliminate bias**: Even when standardizing the number of positive and negative samples to be equal, the accuracy of CNNs on background crops remains way above 50%, ruling out class imbalance as a confounding explanation.
- **Center vs. Corners**: For the CT dataset (NoduleMNIST), the center crop achieves the highest accuracy (potentially due to residual nodule information), whereas for dermoscopy datasets, the difference between corner and center crops is minimal (indicating bias signals are uniformly distributed across the entire image).

## Highlights & Insights

- **The "background cropping" experimental design is remarkably simple yet highly convincing**: It does not require any explainability tools (e.g., GradCAM). Instead, it quantitatively proves the existence of bias merely by controlling the input content. This "method of elimination" approach can be extended to other domains.
- **Value of a large-scale systematic study**: Findings from a single dataset can easily be dismissed as anomalies. However, the comprehensive combination of 13 datasets $\times$ 4 architectures $\times$ 6 inputs provides compelling, unavoidable evidence.
- **Cautionary tale for the transfer learning paradigm**: Even with ImageNet pre-training, CNNs still rapidly adapt to dataset-specific bias features. This indicates that "good initialization" from pre-trained weights does not automatically prevent shortcut learning.

## Limitations & Future Work

- **Only CNN architectures are analyzed**: Modern architectures such as Vision Transformers (ViTs) and self-supervised learning (SSL) pre-trained models are not covered; their sensitivity to bias may differ.
- **Depth of bias source analysis is limited**: Although the presence of bias is proved, the work does not precisely locate whether it stems from staining differences, JPEG compression artifacts, or scanner metadata. Combining this with frequency-domain analysis or style transfer experiments would locate the source of bias more precisely.
- **Lack of mitigation strategies**: The study only diagnoses the issue without proposing technical solutions to remove the biases. Future work could consider strategies such as data augmentation (color jittering, random cropping), adversarial training, or domain generalization.
- **Rationality of the $20 \times 20$ crop size**: For high-resolution Whole Slide Images (WSIs), $20 \times 20$ pixels indeed contain no medical information. However, for low-resolution datasets (e.g., MedMNIST with $28 \times 28$ original images), $20 \times 20$ covers most of the image area.

## Related Work & Insights

- **vs. Torralba & Efros (2011) classic dataset bias study**: While that pioneering work revealed dataset bias in natural images, this study extends the same analysis to the medical imaging domain, demonstrating that the problem might be even more severe (since medical image acquisition processes are more standardized, which paradoxically makes them more prone to systematic biases).
- **vs. DeGrave et al. (2021) chest X-ray bias study**: DeGrave et al. discovered that CNNs rely on hospital tokens and acquisition sites in X-ray images rather than pulmonary pathology for decision-making. This study extends similar findings to more cancer types and datasets, providing more comprehensive evidence.
- **vs. Explainability methods (GradCAM, etc.)**: Post-hoc explainability tools require interpretation and can sometimes be misleading. The "background cropping" method used in this study offers more direct evidence, and the two approaches can be used complementarily.

## Rating

- Novelty: ⭐⭐⭐ The method itself is simple; the core idea is extending known background tests to a large-scale systematic study.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 13 datasets $\times$ 4 architectures, with rigorous experimental designs such as class balancing ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear, systematic, and rich in tables and figures.
- Value: ⭐⭐⭐⭐ High warning value for the medical AI community, though the lack of mitigation strategies slightly reduces its practical guidance significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Reanimating Images using Neural Representations of Dynamic Stimuli](reanimating_images_using_neural_representations_of_dynamic_stimuli.md)
- [\[AAAI 2026\] PanFoMa: A Lightweight Foundation Model and Benchmark for Pan-Cancer Pathology Image Analysis](../../AAAI2026/medical_imaging/panfoma_a_lightweight_foundation_model_and_benchmark_for_pan-cancer.md)
- [\[ICCV 2025\] SIC: Similarity-Based Interpretable Image Classification with Neural Networks](../../ICCV2025/medical_imaging/sic_similarity-based_interpretable_image_classification_with_neural_networks.md)
- [\[ICCV 2025\] AcZeroTS: Active Learning for Zero-shot Tissue Segmentation in Pathology Images](../../ICCV2025/medical_imaging/aczerots_active_learning_for_zeroshot_tissue_segmentation_in.md)
- [\[CVPR 2025\] T-FAKE: Synthesizing Thermal Images for Facial Landmarking](t-fake_synthesizing_thermal_images_for_facial_landmarking.md)

</div>

<!-- RELATED:END -->
