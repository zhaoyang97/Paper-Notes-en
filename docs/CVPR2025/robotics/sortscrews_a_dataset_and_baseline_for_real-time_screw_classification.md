---
title: >-
  [Paper Note] SortScrews: A Dataset and Baseline for Real-time Screw Classification
description: >-
  [CVPR 2025][Robotics][Screw Classification] This paper proposes the SortScrews dataset—an industrial classification dataset containing 560 RGB images of size $512 \times 512$ across 6 screw categories, accompanied by a reusable data acquisition pipeline. Transfer learning models EfficientNet-B0 and ResNet-18 are established as baselines, with ResNet-18 achieving a validation accuracy of 96.4% on this dataset.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Screw Classification"
  - "Industrial Dataset"
  - "Transfer Learning"
  - "EfficientNet"
  - "ResNet"
  - "Automated Sorting"
date: 2026-05-08
content_hash: c897de9bc60f02d0
---

# SortScrews: A Dataset and Baseline for Real-time Screw Classification

**Conference**: CVPR 2025  
**arXiv**: [2603.13027](https://arxiv.org/abs/2603.13027)  
**Code**: [https://github.com/ATATC/SortScrews](https://github.com/ATATC/SortScrews)  
**Area**: Industrial Vision / Image Classification / Dataset  
**Keywords**: Screw Classification, Industrial Dataset, Transfer Learning, EfficientNet, ResNet, Automated Sorting

## TL;DR
This paper proposes the SortScrews dataset—an industrial classification dataset containing 560 RGB images of size $512 \times 512$ across 6 screw categories, accompanied by a reusable data acquisition pipeline. Transfer learning models EfficientNet-B0 and ResNet-18 are established as baselines, with ResNet-18 achieving a validation accuracy of 96.4% on this dataset.

## Background & Motivation
Automatic identification of screw types is crucial for industrial automation, robotic assembly, and inventory management. However, publicly available screw classification datasets are extremely scarce, especially for single-object, controlled scenarios common in automated sorting systems. Existing industrial vision datasets (such as MVTec AD) focus primarily on defect detection rather than part type recognition. Screw categories often exhibit only subtle geometric variations (head shape, length, thread patterns), posing a challenge for visual classification systems. SortScrews aims to address this gap by providing a compact, standardized dataset and a reusable acquisition pipeline.

## Core Problem
How to construct a standardized screw classification dataset under limited data constraints, and verify whether lightweight CNN models can achieve effective fine-grained industrial part classification under controlled acquisition conditions?

## Method

### Data Acquisition System
- **Hardware**: iCAN C55N QHD 2K webcam + wooden stand + printed positioning guide (for perspective calibration)
- **Acquisition Pipeline**: Each image contains a single screw instance placed within the calibration area indicated by the guide. Four acquisition settings introduce minor illumination and perspective variations to simulate small-scale environmental fluctuations in industrial settings.
- **Accompanying Scripts**: Reusable data acquisition scripts are provided, enabling users to rapidly construct similar datasets using inexpensive cameras.

### Dataset Specifications
- **Size**: 560 RGB images with $512 \times 512$ resolution
- **Categories**: 6 screw types + 1 background class, with 80 images per class (balanced dataset)
- **Screw Types**: Flat head 1.5cm, round head 2.5cm, flat head 3.0cm, flat head 3.5cm, flat head 6.0cm, round head 7.5cm
- **Validation Set**: 28 images, uniformly sampled from each category
- **Annotation Format**: CSV file containing image filenames and category IDs

### Baseline Models
Two classic CNN architectures are employed using ImageNet pre-trained weights for transfer learning:

1. **EfficientNet-B0**: An efficient architecture based on compound scaling (joint scaling of depth, width, and resolution), with the final classification layer replaced to match the number of dataset classes.
2. **ResNet-18**: A lightweight architecture based on residual connections, with the final classification layer replaced similarly.

### Training Setup
- Optimizer: AdamW ($lr=1\times10^{-3}$, $weight\ decay=1\times10^{-4}$)
- Input Resolution: $224 \times 224$ (resized from $512 \times 512$)
- Batch size: 16
- Loss Function: Cross-entropy
- Epochs: 100
- The backbone networks are frozen by default to stabilize training on small data, which is configurable in the training framework.
- Training Framework: Trainer provided by MIP Candy (Fu and Chen, 2026)
- Hardware: 2023 MacBook Pro (M3, 16GB), Metal-accelerated, epoch time approximately 2-3 seconds
- Evaluation Metric: Since the dataset classes are balanced, classification accuracy is directly utilized as the primary evaluation metric.

## Key Experimental Results

### Classification Performance

| Model | Validation Accuracy | Inference Time |
|------|-----------|---------|
| EfficientNet-B0 | 86.2% | 10.99 ± 4.86 ms |
| ResNet-18 | **96.4%** | 19.88 ± 4.08 ms |

### GPU Inference Speed (NVIDIA CUDA)

| Model | Mean | Median | Throughput |
|------|------|--------|--------|
| ResNet-18 | 6.42 ms | 6.15 ms | 155.8 fps |
| EfficientNet-B0 | 17.95 ms | 17.92 ms | 55.7 fps |

### Error Analysis
- **ResNet-18**: Only one sample was misclassified (Class 4 $\rightarrow$ Class 2), demonstrating highly stable performance.
- **EfficientNet-B0**: Errors are concentrated between Class 2 (round head 2.5cm) and Class 4 (flat head 3.5cm), as well as confusion with Class 0 (background). This indicates difficulty in distinguishing screws with similar lengths but different head shapes.
- **Positional Bias**: Models learned an unexpected bias regarding the location of the screw in the image, likely stemming from sparse semantic supervision. Introducing positional supervision, such as bounding boxes, could mitigate this issue.
- These errors reveal the core difficulty of fine-grained industrial part recognition: subtle geometric differences are challenging to reliably distinguish without large volumes of data or multi-perspective support.

## Highlights & Insights
- **Open-source Dataset and Acquisition Pipeline**: The release includes both the dataset and reusable acquisition scripts, lowering the barrier for other researchers to build similar industrial datasets.
- **Efficient Learning under Controlled Acquisition**: Achieving 96.4% accuracy with only 560 images and lightweight models validates the value of standardized acquisition environments for small-data learning.
- **Detailed Failure Analysis**: Misclassification patterns across categories are systematically analyzed via confusion matrices, revealing meaningful phenomena such as positional bias.
- **Real-time Deployability**: ResNet-18 achieves 155.8 fps on GPU, satisfying real-time sorting requirements.

## Limitations & Future Work
- The dataset scale is extremely small (560 images), falling far short of large-scale vision benchmarks, and its generalization capability has not been fully verified.
- It contains only 6 screw types, whereas actual industrial scenarios involve far more varieties.
- Single-perspective acquisition lacks multi-view information, leaving robustness to occlusions and orientation variations questionable.
- It lacks depth or 3D information, relying solely on RGB images.
- The validation set contains only 28 images, limiting statistical reliability.
- It has not been tested in real-world industrial environments such as actual conveyor belts.
- The performance of ResNet-18 surpassing EfficientNet-B0 warrants deeper analysis—modern architectures may not necessarily perform better on small-data, fine-grained tasks.
- Future expansion directions proposed by the authors include: additional screw types, multi-perspective images, conveyor-belt acquisition environments, and depth/3D information.

## Related Work & Insights
- **MVTec AD**: An industrial anomaly detection dataset focused on defects rather than part classification, which is complementary to SortScrews but has a different objective.
- **ImageNet**: SortScrews leverages ImageNet pre-trained weights for transfer learning, but direct evaluation is not applicable to industrial parts.
- **General Fine-Grained Classification** (e.g., CUB-200 Birds, Stanford Cars): While sharing the focus on fine-grained recognition, the controlled acquisition conditions of SortScrews significantly reduce background complexity, allowing effective learning from a small dataset.

## Insights & Connections
- Controlled acquisition combined with transfer learning represents a practical paradigm for small-data classification in industrial scenarios, which can be extended to other small components (such as nuts, washers, and rivets).
- The finding of positional bias suggests a need to introduce more spatial randomness during data acquisition or to incorporate spatial augmentations during training.
- The strategy of open-sourcing the acquisition pipeline is highly informative—reducing dataset construction costs carries greater industrial value than solely increasing model complexity.
- The effectiveness of freezing the backbone network and only fine-tuning the classification head on extremely small datasets provides a valuable reference for resource-constrained industrial deployments.

## Rating
- Novelty: ⭐⭐⭐ The method itself (transfer learning + standard CNNs) contains no core architectural innovation; the primary contribution lies in the dataset.
- Experimental Thoroughness: ⭐⭐⭐⭐ Baseline comparisons and error analyses are provided, but the validation set is excessively small, and comparisons with more modern models are lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear and well-structured, although the technical depth is limited.
- Value: ⭐⭐⭐⭐ It fills a gap in screw classification datasets and provides open-source acquisition tools of practical utility, though the small dataset scale limits its broader impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] TinyNav: End-to-End TinyML for Real-Time Autonomous Navigation on Microcontrollers](tinynav_end-to-end_tinyml_for_real-time_autonomous_navigation_on_microcontroller.md)
- [\[CVPR 2025\] GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities](gigahands_a_massive_annotated_dataset_of_bimanual_hand_activities.md)
- [\[NeurIPS 2025\] LUMIA: A Handheld Vision-to-Music System for Real-Time, Embodied Composition](../../NeurIPS2025/robotics/lumia_a_handheld_vision-to-music_system_for_real-time_embodied_composition.md)
- [\[ICLR 2026\] Real-Time Robot Execution with Masked Action Chunking](../../ICLR2026/robotics/real-time_robot_execution_with_masked_action_chunking.md)
- [\[ACL 2025\] CHEER-Ekman: Fine-grained Embodied Emotion Classification](../../ACL2025/robotics/cheer-ekman_fine-grained_embodied_emotion_classification.md)

</div>

<!-- RELATED:END -->
