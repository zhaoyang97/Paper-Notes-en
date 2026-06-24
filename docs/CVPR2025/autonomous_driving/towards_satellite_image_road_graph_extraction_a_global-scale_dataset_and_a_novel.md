---
title: >-
  [Paper Note] Towards Satellite Image Road Graph Extraction: A Global-Scale Dataset and A Novel Method
description: >-
  [CVPR 2025][Autonomous Driving][Road Graph Extraction] This paper constructs Global-Scale, a large-scale global satellite road graph extraction dataset (approximately 20 times larger than the largest existing public dataset), and proposes the SAM-Road++ method. Under the proposed method, a node-guided resampling strategy is designed to resolve the training-inference mismatch, while an "extended-line" strategy mitigates road fragmentation caused by occlusions…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Road Graph Extraction"
  - "Satellite Imagery"
  - "Global-Scale Dataset"
  - "Graph Connection Prediction"
  - "SAM"
date: 2026-05-08
content_hash: 845de602e18e6ef8
---

# Towards Satellite Image Road Graph Extraction: A Global-Scale Dataset and A Novel Method

**Conference**: CVPR 2025  
**arXiv**: [2411.16733](https://arxiv.org/abs/2411.16733)  
**Code**: [https://github.com/earth-insights/samroadplus](https://github.com/earth-insights/samroadplus)  
**Area**: Autonomous Driving/Remote Sensing  
**Keywords**: Road Graph Extraction, Satellite Imagery, Global-Scale Dataset, Graph Connection Prediction, SAM

## TL;DR

This paper constructs Global-Scale, a large-scale global satellite road graph extraction dataset (approximately 20 times larger than the largest existing public dataset), and proposes the SAM-Road++ method. Under the proposed method, a node-guided resampling strategy is designed to resolve the training-inference mismatch, while an "extended-line" strategy mitigates road fragmentation caused by occlusions, achieving SOTA performance across multiple datasets.

## Background & Motivation

**Background**: Road graph extraction is a critical task in autonomous driving and navigation systems. Existing end-to-end methods are generally categorized into iterative methods (e.g., RNGDet series, which generate road graphs point-by-point) and global methods (e.g., SAM-Road, which predict road graphs globally). While iterative methods suffer from error accumulation and high computational costs, global methods are more efficient but still have design limitations.

**Limitations of Prior Work**: As a representative global method, SAM-Road trains its connection classifier using ground-truth (GT) nodes, but utilizes nodes extracted from predicted segmentation masks as inputs during inference. This leads to a fundamental mismatch between the input distributions of the training and inference stages, thereby limiting the performance of the classifier during inference. Furthermore, occlusions such as trees and building shadows in satellite images make road connectivity determination highly difficult, representing an overlooked bottleneck.

**Key Challenge**: The decoupled design of the segmentation phase and the connection prediction phase in global methods causes training-inference inconsistency. From a data perspective, existing annotated road graph datasets (e.g., City-Scale with only 180 images, and SpaceNet with only $400 \times 400$ small images) are too small in scale and lack scenario diversity, leading to insufficient model generalization and unreliable evaluation.

**Goal**: (1) Construct a large-scale, diverse global-scale benchmark for road graph extraction; (2) resolve the training-inference mismatch; and (3) mitigate road fragmentation in occluded scenarios.

**Key Insight**: The authors observe that SAM-Road's classifier is trained on GT nodes but inferred on predicted nodes. Aligning the training stage node distribution more closely with the inference stage distribution could improve consistency. For the occlusion issue, the authors hypothesize that if road segments exist on both ends of a straight line, there is a high probability that a road exists in the occluded segment between them.

**Core Idea**: Utilize node-guided resampling to replace GT node coordinates during training with the highest-probability coordinates obtained from the predicted masks, and employ an-extended line strategy to provide additional contextual information for the classifier to identify roads under occlusion.

## Method

### Overall Architecture

SAM-Road++ is built upon a SAM encoder-decoder backbone. Given an input satellite image, it first generates road segmentation masks and keypoint masks. During training, node pairs are obtained from GT and predicted masks through node-guided resampling. During inference, nodes are selected from the masks using NMS. The connection classifier predicts the existence of a road between nodes based on node features and extended-line information. The total loss is sum of binary cross-entropy losses for $\mathcal{L}_{mask}$ (road segmentation) and $\mathcal{L}_{topo}$ (topological connection).

### Key Designs

1. **Node-guided Resampling**:

    - **Function**: Align the input distributions of the classifier between the training and inference stages.
    - **Mechanism**: During training, $N$ source nodes are first sampled from the GT, and target nodes and their connection relationships within a distance $R$ of each source node are determined. Then, keeping the source nodes fixed, the point with the highest probability on the predicted mask within a radius $r$ centered around each target node is selected as the new target node. In this way, the resampled nodes retain the GT connection information, while their positions are closer to the nodes selected by NMS during inference. To ensure diversity, nodes with rarer degree attributes are preferentially sampled.
    - **Design Motivation**: Directly applying the inference procedure (NMS) during training makes it impossible to obtain the connection labels needed for supervision. Therefore, a trade-off is adopted: preserving the GT topology, but aligning the coordinate endpoints to the predicted mask, while fully leveraging the segmentation insights from the first stage.

2. **Extended-line Strategy**:

    - **Function**: Provide additional contextual information under occlusion scenarios for the connection classifier.
    - **Mechanism**: For a pair of nodes whose connectivity needs to be determined, besides extracting their respective node features, mask values are uniformly sampled $n=15$ times on the extensions at both ends of the line segment connecting the two nodes, and $m=20$ times along the connection line between the two nodes. The extended line length is set to 8 pixels and the width to 3 pixels to simulate a road. These additional sampled values serve as extra inputs to the classifier.
    - **Design Motivation**: This is based on the road continuity assumption: if roads clearly exist in the extension directions corresponding to two adjacent nodes, there is a high likelihood of a connecting road, even if the intermediate segment is occluded by trees or building shadows. This enables the model to "see" the road information before and after the occluded region.

3. **Global-Scale Dataset**:

    - **Function**: Provide a large-scale and diverse global-scale road graph extraction benchmark.
    - **Mechanism**: Latitude and longitude coordinates across various terrains (urban, rural, mountainous, etc.) on global continents (except Antarctica) are manually selected from Google Earth. Satellite images of size $2048 \times 2048$ with a spatial resolution of 1m/pixel are retrieved via the Google Static Map API, and corresponding road graph annotations are obtained from OpenStreetMap. The dataset comprises 3,468 images (2,375 for training, 339 for validation, 624 for in-domain testing, and 130 for out-of-domain testing from unseen cities).
    - **Design Motivation**: Existing datasets only cover urban areas and are limited in scale, which cannot support robust model evaluation and generalization training. The configuration of an out-of-domain (OOD) test set enables the evaluation of model prediction capabilities on unseen regions.

### Loss & Training

The total loss is formulated as $\mathcal{L} = \mathcal{L}_{mask} + \mathcal{L}_{topo}$, where both components are binary cross-entropy losses. $\mathcal{L}_{mask}$ supervises road segmentation, and $\mathcal{L}_{topo}$ supervises topological connections between nodes. The Adam optimizer is utilized with a learning rate of 0.001. The model is trained for 150 epochs on Global-Scale, and on other datasets until the validation metrics stabilize.

## Key Experimental Results

### Main Results

| Dataset | Method | F1 | Precision | Recall | APLS |
|--------|------|-----|-----------|--------|------|
| City-Scale | SAM-Road | 77.23 | 90.47 | 67.69 | 68.37 |
| City-Scale | **SAM-Road++** | **80.01** | 88.39 | **73.39** | 68.34 |
| City-Scale | SAM-Road++* | **80.66** | 89.08 | **74.07** | **69.55** |
| SpaceNet | SAM-Road | 80.52 | 93.03 | 70.97 | 71.64 |
| SpaceNet | **SAM-Road++** | **81.57** | **93.68** | 72.23 | **73.44** |
| Global-Scale (ID) | SAM-Road | 59.80 | **91.93** | 45.64 | 59.08 |
| Global-Scale (ID) | **SAM-Road++** | **62.33** | 88.95 | **49.27** | **62.19** |
| Global-Scale (OOD) | SAM-Road | 46.64 | 84.54 | 33.81 | 40.51 |
| Global-Scale (OOD) | **SAM-Road++** | **48.34** | 82.21 | **36.04** | **43.17** |

*indicates fine-tuning after pretraining on Global-Scale.

### Ablation Study

| Extended-line | Node-guided Resampling | APLS | F1 |
|:---:|:---:|------|------|
| ✗ | ✗ | 71.64 | 80.52 |
| ✗ | ✓ | 71.90 | 81.77 |
| ✓ | ✗ | 73.22 | 80.89 |
| ✓ | ✓ | **73.44** | **81.57** |

### Key Findings

- Node-guided resampling contributes more to the F1 score (+1.25), indicating that it effectively improves the overall accuracy of topological predictions. The extended-line strategy contributes more to APLS (+1.58), showing that it helps the model predict more accurate road path lengths.
- Pretraining on the Global-Scale dataset followed by fine-tuning on City-Scale and SpaceNet further boosts performance, validating the significance of large-scale datasets.
- The performance of all models on Global-Scale is inferior to that on the smaller datasets, proving that this dataset is indeed more challenging.
- While performance drops significantly on the OOD test set, SAM-Road++ still substantially outperforms other methods, demonstrating stronger generalization capabilities.

## Highlights & Insights

- **Generality of the Training-Inference Alignment Concept**: Node-guided resampling essentially simulates the inference environment during the training stage. This paradigm can be generalized to any two-stage pipeline with training-inference discrepancies (e.g., detection-then-classification frameworks).
- **Simplicity and Efficiency of the Extended-line Strategy**: Without introducing extra models or complex architectures, the occlusion challenge is mitigated solely by sampling mask information along the extended directions of connecting lines. This design methodology leveraging road geometric priors (continuity) is highly informative.
- The design of the OOD test set aligns evaluation closer to real-world application scenarios, establishing a more reliable benchmark for the remote sensing community.

## Limitations & Future Work

- Precision is slightly lower than the original SAM-Road in certain scenarios, as resampled nodes may deviate from the GT, leading to predictions of non-existent roads.
- The City-Scale test set consists of only 27 images; abnormal performance on a single image can excessively impact metrics, leading to unstable evaluations.
- Dataset annotations depend on OpenStreetMap, which may be incomplete in economically less-developed regions.
- Future efforts could explore more adaptive resampling caught-radii $r$ and extended line lengths, as well as extending the method to fine-grained road attribute extraction (e.g., number of lanes, road types).

## Related Work & Insights

- **vs SAM-Road**: SAM-Road first introduced SAM to the field of road graph extraction to achieve global prediction, but its two-stage decoupled design caused a training-inference mismatch. SAM-Road++ couples the two stages via the resampling strategy while maintaining unaffected inference efficiency.
- **vs RNGDet++**: While iterative methods offer high accuracy, they suffer from slow speed and error accumulation. As a global method, SAM-Road++ completely outperforms them in F1 score and achieves higher efficiency.
- **vs Sat2Graph**: Early global methods relied on graph-encoding tensors and complex post-processing. SAM-Road++ simplifies this pipeline and yields better results.

## Rating

- Novelty: ⭐⭐⭐⭐ Although the training-inference alignment and extended-line strategies are clear, they are not disruptive innovations; the dataset contribution is the highlight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive experiments with three datasets, in-domain/OOD testing, and complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and rich illustrations.
- Value: ⭐⭐⭐⭐ Both the dataset and the proposed method have practical utility, making a significant contribution to the remote sensing community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LiSu: A Dataset and Method for LiDAR Surface Normal Estimation](lisu_a_dataset_and_method_for_lidar_surface_normal_estimation.md)
- [\[CVPR 2025\] ClimbingCap: Multi-Modal Dataset and Method for Rock Climbing in World Coordinate](climbingcap_multi-modal_dataset_and_method_for_rock_climbing_in_world_.md)
- [\[CVPR 2026\] SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving](../../CVPR2026/autonomous_driving/searchad_large-scale_rare_image_retrieval_dataset_for_autonomous_driving.md)
- [\[CVPR 2025\] GLane3D: Detecting Lanes with Graph of 3D Keypoints](glane3d_detecting_lanes_with_graph_of_3d_keypoints.md)
- [\[NeurIPS 2025\] ChronoGraph: A Real-World Graph-Based Multivariate Time Series Dataset](../../NeurIPS2025/autonomous_driving/chronograph_a_real-world_graph-based_multivariate_time_series_dataset.md)

</div>

<!-- RELATED:END -->
