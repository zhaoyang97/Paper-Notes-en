---
title: >-
  [Paper Note] Alberta Wells Dataset: Pinpointing Oil and Gas Wells from Satellite Imagery
description: >-
  [ICML 2025][Segmentation][Remote sensing] This paper introduces the first large-scale oil and gas well detection benchmark, the Alberta Wells Dataset (containing over 213k well locations and 188k+ satellite imagery patches). The localization of abandoned, suspended, and active oil and gas wells is formulated as binary segmentation and object detection tasks, and various CNN and Transformer baseline models are evaluated.
tags:
  - "ICML 2025"
  - "Segmentation"
  - "Remote sensing"
  - "oil and gas well detection"
  - "binary segmentation"
  - "object detection"
  - "satellite imagery"
date: 2026-05-08
content_hash: cb7ed66e4f522ced
---

# Alberta Wells Dataset: Pinpointing Oil and Gas Wells from Satellite Imagery

**Conference**: ICML 2025  
**arXiv**: [2410.09032](https://arxiv.org/abs/2410.09032)  
**Code**: [GitHub](https://github.com/RolnickLab/Alberta_Wells_Dataset)  
**Area**: Segmentation  
**Keywords**: Remote sensing, oil and gas well detection, binary segmentation, object detection, satellite imagery

## TL;DR

This paper introduces the first large-scale oil and gas well detection benchmark, the Alberta Wells Dataset (containing over 213k well locations and 188k+ satellite imagery patches). The localization of abandoned, suspended, and active oil and gas wells is formulated as binary segmentation and object detection tasks, and various CNN and Transformer baseline models are evaluated.

## Background & Motivation

Millions of abandoned oil and gas wells worldwide are leaking methane (a potent greenhouse gas) into the atmosphere and toxic compounds into groundwater. Canada alone has approximately 370,000 abandoned wells, emitting the equivalent of about 500,000 tonnes of $CO_2$ annually, while the US has around 4 million, emitting over 5 million tonnes of $CO_2$ equivalent per year. Plugging these wells can mitigate the damage, but the locations of a vast number of abandoned wells remain unknown—for instance, up to 90% of abandoned wells in Pennsylvania are estimated to be undocumented.

Remote sensing combined with machine learning presents opportunities for locating abandoned wells at scale. However, existing datasets are extremely small (ranging from 500 to 12,000 wells), limited in geographical coverage, and typically only contain active wells, making them unsuitable for detecting abandoned or suspended wells. This work aims to bridge this gap by presenting the first truly large-scale, publicly available benchmark dataset covering multiple well statuses.

## Method

### Overall Architecture

The core contribution of this work is the **Alberta Wells Dataset (AWD)**—a large-scale benchmark dataset for oil and gas well identification, along with baseline evaluations. The overall workflow consists of three main components:

1. **Data Collection and Quality Control**: Well location metadata is retrieved from the Alberta Energy Regulator (AER), filtered, deduplicated, and categorized through domain-expert validation.
2. **Satellite Imagery Acquisition and Label Generation**: Using Planet Labs 4-band (RGB + Near-Infrared) high-resolution satellite imagery, segmentation masks and COCO-format detection labels are generated for each patch.
3. **Baseline Model Evaluation**: Various deep learning models are evaluated under both binary segmentation and object detection settings.

### Key Designs

#### Data Collection and Quality Control Flow

The raw AER ST37 data contains approximately 637k metadata records and 512k geographic coordinate records, but suffers from significant redundancy and inaccurate status labels. The cleaning pipeline is as follows:

- **Deduplication**: Metadata is deduplicated by license number (keeping the latest update); shapefiles are deduplicated by license date.
- **Fusion and Filtering**: The two sources are merged, and domain experts devise rules to categorize wells into three classes:
    - **Active**: 107,139 wells, with statuses such as Flowing/Pumping/Gas Lift.
    - **Suspended**: 55,007 wells, with suspension status.
    - **Abandoned**: 54,947 wells, with Abandoned/Junked and Abandoned statuses.
- **Coordinate Deduplication**: Duplicate coordinates are resolved by retaining only the record with the most recent spud (drilling start) date.
- **Boundary Validation**: Wells are verified to ensure they fall within the provincial boundaries of Alberta.
- Approximately 217k valid records remain after final filtering.

#### Patch Generation

Alberta is partitioned into non-overlapping square image patches, with each patch having a side length of 1050m (approx. 1.1025 $km^2$ in area). The number of well-containing and well-free patches is balanced to be roughly equal. The final dataset consists of 188,688 patches, where 94,344 contain wells, covering a total of 213,447 wells.

#### Clustering-Based Dataset Splitting Algorithm

To ensure geographical diversity across the train, validation, and test splits, a two-level K-Means clustering algorithm (Algorithm 1) is proposed:

- **Step 1**: K-Means clustering (with $M=300$ clusters) is performed on the centroid coordinates of all well-containing patches, forming local area clusters $k_1$.
- **Step 2**: A second round of K-Means clustering (with $N=30$ superclusters) is applied to the centroids of $k_1$ clusters to form superclusters $k_2$ representing cities or major geographical regions.
- **Step 3**: Within each supercluster $k_2$, the two $k_1$ clusters containing the fewest wells are assigned to the validation and test sets, respectively, while the remaining ones are allocated to the training set.
- **Step 4**: Well-free patches are assigned to corresponding clusters based on convex hull radius.
- **Step 5**: Imbalance Correction: If there is an excess of well-free patches, they are subsampled proportionally; otherwise, additional well-free patches are sampled from unallocated ones.

This methodology ensures that each data split incorporates samples from diverse geographical areas while preventing spatial data leakage.

#### Satellite Imagery Acquisition and Annotation

- **Imagery Source**: PlanetScope 4-band (RGB + Near-Infrared), PSB.SD instrument, with a resolution of ~3m/pixel.
- **Rationale for Planet Labs**: Daily revisit rates ensure consistency; multispectral bands (NIR is beneficial for detecting ground depressions); and global coverage.
- **Segmentation Annotation**: Wells are annotated with a circular region of 90m in diameter (ranging from 70–120m in practice) to generate binary and multi-class segmentation masks.
- **Detection Annotation**: Bounding boxes are defined using the same spatial scale in COCO format.
- **Data Augmentation**: Random resizing to $256 \times 256$, horizontal/vertical flips (each with $p=0.25$), and channel normalization.

### Loss & Training

**Segmentation Models**:
- CNN Baselines (U-Net, PAN, DeepLabV3+): Trained with BCELogits loss, ResNet50 backbone, batch size of 128, cosine annealing learning rate scheduler, AdamW optimizer, for 50 epochs.
- Transformer Baselines (Segformer, UperNet): Trained with Dice loss and polynomial learning rate decay; Segformer uses a mit-b0-ade backbone (batch size 128); UperNet uses ConvNeXt/Swin backbones (batch size 64).

**Detection Models**:
- RetinaNet/SSD Lite: Batch size of 512; Faster R-CNN/FCOS: Batch size of 256.
- DETR: Batch size of 64.
- All detection models use a ResNet50 backbone (except SSD Lite which uses MobileNet), optimized via AdamW with cosine annealing for 120 epochs.

## Key Experimental Results

### Main Results

**Table 4: Binary Segmentation Results**

| Model | Backbone | Params | IoU | F1 | Precision | Recall |
|------|----------|--------|-----|------|-----------|--------|
| U-Net | ResNet50 | 32.52M | 58.0±0.5 | 61.9±0.8 | **90.2±2.2** | 62.3±1.6 |
| U-Net | ResNeXt50 | 32M | 58.2±0.2 | 62.1±0.3 | 88.2±3.5 | 63.6±1.7 |
| U-Net | SE_ResNet50 | 35.06M | 58.9±0.7 | 62.9±0.7 | 88.8±1.6 | 64.4±1.4 |
| U-Net | EfficientNetB6 | 43.83M | **60.4±0.3** | **64.8±0.4** | 87.8±0.4 | 66.3±0.3 |
| PAN | ResNet50 | 24.26M | 57.8±0.8 | 61.5±0.9 | 89.3±1.2 | 61.5±0.9 |
| DeepLabV3+ | ResNet50 | 26.68M | 56.8±0.7 | 60.6±0.7 | 89.4±1.3 | 61.8±1.1 |
| Segformer | mit-b0-ade | 3.72M | 57.6±0.5 | 61.3±0.6 | 82.6±2.9 | 69.2±2.1 |
| UperNet | ConvNeXt-S | 128.29M | 59.4±0.1 | 63.5±0.1 | 81.5±0.5 | 71.5±0.4 |
| UperNet | ConvNeXt-B | 146.27M | 59.7±0.3 | 63.8±0.2 | 81.1±0.7 | 72.2±0.2 |
| UperNet | Swin-S | 81.15M | 59.9±0.7 | 64.2±0.7 | 80.6±0.5 | **73.1±0.1** |

**Table 5: Object Detection Results**

| Model | Backbone | Params | IoU@0.1 | IoU@0.3 | IoU@0.5 | mAP@50 | mAP@50:95 |
|------|----------|--------|---------|---------|---------|--------|-----------|
| RetinaNet | ResNet50 | 18.87M | 24.58 | 43.07 | 59.79 | 0.18 | 0.63 |
| Faster R-CNN | ResNet50 | 41.09M | **36.79** | 46.95 | 61.29 | 5.20 | 19.12 |
| FCOS | ResNet50 | 31.85M | 34.79 | 48.51 | 62.66 | 9.67 | 30.46 |
| SSD Lite | MobileNet | 3.71M | 33.91 | **50.30** | **65.07** | 9.76 | 25.14 |
| DETR | ResNet50 | 41.47M | 41.78 | 51.15 | 63.17 | **15.22** | **38.45** |

### Ablation Study

**Impact of NIR Band on Segmentation (U-Net + ResNet50)**

| Configuration | IoU | F1 | Precision | Recall |
|------|-----|------|-----------|--------|
| RGB+NIR | **58.0±0.5** | **61.9±0.8** | **90.2±2.2** | 62.3±1.6 |
| RGB only | 56.6±0.4 | 60.5±0.4 | 87.0±1.4 | **62.5±0.1** |

**Impact of NIR Band on Detection (FCOS + ResNet50)**

| Configuration | IoU@0.1 | IoU@0.3 | IoU@0.5 | mAP@50 | mAP@50:95 |
|------|---------|---------|---------|--------|-----------|
| RGB+NIR | **34.79** | **48.51** | **62.66** | **9.67** | **30.46** |
| RGB only | 32.39 | 46.80 | 61.23 | 5.70 | 20.00 |

**Impact of Well Types in Training Data (U-Net + ResNet50)**

| Metric | Active Only | All Types | Description |
|------|---------|---------|------|
| IoU | 0.502 | **0.576** | +14.7% |
| F1 | 0.503 | **0.614** | +22.1% |
| Precision | **0.998** | 0.913 | Training on active wells only yields extremely high precision but very low recall |
| Recall | 0.502 | **0.614** | Significant boost in recall |

### Key Findings

1. **Segmentation Outperforms Detection**: Overall, performance in the segmentation task exceeds that of object detection, indicating that semantic segmentation might be a more suitable formulation for real-world well localization and identification.
2. **U-Net + EfficientNetB6 Achieves Superior Segmentation**: Standard U-Net paired with an EfficientNetB6 backbone delivers the best segmentation metrics (IoU 60.4%, F1 64.8%), likely owing to its larger receptive field.
3. **UperNet + Swin Scores the Highest Recall** (73.1%): This model is highly suitable for monitoring applications where minimizing false negatives (missed detections) is critical.
4. **DETR Shows Comprehensive Detection Performance**: It achieves the best detection results with an mAP@50 of 15.22 and mAP@50:95 of 38.45, highlighting its powerful global context modeling capability.
5. **NIR Band Significantly Improves Performance**: Incorporating the Near-Infrared channel yields consistent performance gains in both segmentation and detection, notably boosting mAP@50:95 from 20.0 to 30.5.
6. **Joint Training on Diverse Well Statuses Is Crucial**: Training models exclusively on active wells leads to a failure in successfully detecting abandoned or suspended wells.

## Highlights & Insights

- **Unprecedented Dataset Scale**: Contains 213k wells and 94k well-containing patches, surpassing the previous largest benchmark (12,490 wells) by over an order of magnitude.
- **Elegant Geographical Splitting Algorithm**: The two-level K-Means clustering system guarantees that the train and test splits are geographically decoupled, preventing spatial info leaks while preserving diversity.
- **Direct Practical Application**: Abandoned wells represent a major, highly uncertain source of methane emissions; this dataset directly contributes to climate change mitigation efforts.
- **Value of Multispectral Bands**: Empirical evidence proves that the Near-Infrared band is crucial for highlighting sub-pixel spatial clues such as soil and ground depression patterns.
- **Problem Modeling Insight**: By concurrently framing well identification as detection and segmentation, the authors reveal that segmentation is much better suited for addressing the task.

## Limitations & Future Work

1. **Annotation Noise**: The dataset relies on official AER records; undocumented wells may still exist, causing potential false negative annotations (unlabeled wells).
2. **Geographical Limitation**: The dataset is restricted to Alberta, Canada; the zero/few-shot generalization capabilities to other regions remain unexplored.
3. **Performance Degradation in Dense Regions**: Most patches contain only 1–5 wells; detection performance drops significantly in rare, highly dense well areas.
4. **Faint Visual Cues for Abandoned/Suspended Wells**: Dense vegetation cover and infrastructural degradation often obscure these wells, making them inherently more challenging to detect.
5. **Substantial Room for Improvement**: The best segmentation IoU is only 60.4%, and the top detection mAP@50 yields just 15.2%.
6. **Underutilization of Multi-Class Annotations**: Although multi-class status annotations (active, suspended, abandoned) are provided, multi-class segmentation experiments were not conducted.
7. **Promising Future Directions**: Fine-tuning foundation models like Segment Anything Model (SAM); temporal (multi-temporal) analysis; self-supervised pre-training; and semi-supervised learning strategies to leverage unannotated areas.

## Related Work & Insights

- While remote sensing integrated with ML has been extensively explored for climate action tasks (e.g., land-use classification, crop mapping, forest monitoring), this work successfully extends its coverage to oil and gas well detection.
- Previous related datasets targeting oil and gas infrastructure (e.g., NEPU-OWOD, Well Pad Dataset) are severely limited in scale and only contain active wells.
- The proposed geographical clustering-based split scheme represents a valuable paradigm that can be generalized to the creation of other remote-sensing benchmarks.
- **Takeaway**: For detecting tiny targets (wells span only approximately 30 pixels in the image) via Earth observation, incorporating multispectral imagery and selecting the right formulation (segmentation vs. detection) are highly crucial.

## Rating

| Dimension | Score | Description |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | First large-scale well detection benchmark; the dataset construction and spatial splitting algorithm are highly novel. |
| Technical Depth | ⭐⭐⭐ | The primary focus is benchmark construction, while the modeling section is restricted to standard baseline evaluations. |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Comprehensive experiments covering multiple segmentation and detection models, NIR band ablation, and well-type ablation. |
| Value | ⭐⭐⭐⭐⭐ | Directly serves real-world climate mitigation needs, with a publicly accessible dataset and open-sourced code. |
| Writing Quality | ⭐⭐⭐⭐ | Well-structured, complete with a comprehensive Datasheet and detailed appendices. |
| Overall Rating | ⭐⭐⭐⭐ | An outstanding benchmark paper with high real-world and societal impact. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Harnessing Massive Satellite Imagery with Efficient Masked Image Modeling](../../ICCV2025/segmentation/harnessing_massive_satellite_imagery_with_efficient_masked_image_modeling.md)
- [\[ICML 2025\] Using Multiple Input Modalities Can Improve Data-Efficiency and O.O.D. Generalization for ML with Satellite Imagery](using_multiple_input_modalities_can_improve_data-efficiency_and_ood_generalizati.md)
- [\[AAAI 2026\] Generalizable Slum Detection from Satellite Imagery with Mixture-of-Experts](../../AAAI2026/segmentation/generalizable_slum_detection_from_satellite_imagery_with_mixture-of-experts.md)
- [\[NeurIPS 2025\] Self-supervised Synthetic Pretraining for Inference of Stellar Mass Embedded in Dense Gas](../../NeurIPS2025/segmentation/self-supervised_synthetic_pretraining_for_inference_of_stellar_mass_embedded_in_.md)
- [\[NeurIPS 2025\] GTPBD: A Fine-Grained Global Terraced Parcel and Boundary Dataset](../../NeurIPS2025/segmentation/gtpbd_a_fine-grained_global_terraced_parcel_and_boundary_dataset.md)

</div>

<!-- RELATED:END -->
