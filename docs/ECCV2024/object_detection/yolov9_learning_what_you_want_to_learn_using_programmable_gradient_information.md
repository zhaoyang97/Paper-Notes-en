---
title: >-
  [Paper Note] YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information
description: >-
  [ECCV 2024][Object Detection][YOLO] YOLOv9 proposes Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN) to address the information bottleneck problem in deep networks. It comprehensively outperforms existing real-time object detectors on MS COCO with fewer parameters and computation, surpassing methods pre-trained on large-scale datasets while training from scratch.
tags:
  - "ECCV 2024"
  - "Object Detection"
  - "YOLO"
  - "Programmable Gradient Information"
  - "Information Bottleneck"
  - "Lightweight Network"
  - "Real-time Object Detection"
date: 2026-05-08
content_hash: 30d804584f85f9c7
---

# YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information

**Conference**: ECCV 2024  
**arXiv**: [2402.13616](https://arxiv.org/abs/2402.13616)  
**Code**: [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)  
**Area**: Object Detection  
**Keywords**: YOLO, Programmable Gradient Information, Information Bottleneck, Lightweight Network, Real-time Object Detection

## TL;DR

YOLOv9 proposes Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN) to address the information bottleneck problem in deep networks. It comprehensively outperforms existing real-time object detectors on MS COCO with fewer parameters and computation, surpassing methods pre-trained on large-scale datasets while training from scratch.

## Background & Motivation

**Background**: Deep learning has achieved performance far exceeding traditional methods in fields such as computer vision. Researchers mainly focus on more powerful network architectures (CNNs, Transformers, Mamba) and more appropriate objective functions (loss functions, label assignment, auxiliary supervision). The YOLO series remains the most widely adopted solution for real-time object detection.

**Limitations of Prior Work**: When input data undergoes layer-by-layer feature extraction and spatial transformation, a significant amount of information is lost (the information bottleneck phenomenon). Consequently, the loss function cannot generate reliable gradients to update model weights, causing the model to establish incorrect associations between data and targets.

**Key Challenge**: Existing methods addressing information loss have respective drawbacks: (1) Reversible architectures (e.g., RevCol) preserve information but significantly increase inference overhead (extra connections increase inference time by 20% or even double it); (2) Reconstruction losses in masked modeling may conflict with object detection losses; (3) Deep supervision causes error accumulation and performs poorly on shallow and lightweight models.

**Goal**: How to provide complete and reliable gradient information for deep networks without increasing inference overhead, so that models ranging from lightweight to large-scale can all benefit from auxiliary supervision?

**Key Insight**: Grounded in the information bottleneck principle and reversible function theory, the reversible architecture is placed on an auxiliary branch instead of the main branch. It is utilized only during training to generate reliable gradients and is discarded during inference to avoid extra computational overhead.

**Core Idea**: Program gradient information propagation across different semantic levels through an auxiliary reversible branch to allow the main branch to obtain reliable gradient updates, while designing GELAN with conventional convolutions to achieve parameter utilization efficiency that outperforms depthwise separable convolutions.

## Method

### Overall Architecture

YOLOv9 consists of two core innovations: **PGI (Programmable Gradient Information)** and **GELAN (Generalized Efficient Layer Aggregation Network)**. PGI is an auxiliary supervision framework comprising three components: the main branch, the auxiliary reversible branch, and multi-level auxiliary information. GELAN is a novel network architecture combining the design philosophies of CSPNet and ELAN. During inference, only the main branch (GELAN) is utilized, introducing no extra computational overhead.

### Key Designs

1. **Auxiliary Reversible Branch**:

    - **Function**: Introduces a reversible architecture as an auxiliary branch during the training phase to generate reliable gradients for updating the main branch parameters.
    - **Mechanism**: According to the information bottleneck principle, the mutual information of data $X$ decreases after transformation: $I(X,X) \geq I(X,f_\theta(X)) \geq I(X,g_\phi(f_\theta(X)))$. A reversible function $r$ satisfies $X = v_\zeta(r_\psi(X))$, which prevents information loss: $I(X,X) = I(X,r_\psi(X))$. By placing the reversible branch in an auxiliary position, it can be directly removed during inference.
    - **Design Motivation**: The deep features of the main branch lose crucial information due to the information bottleneck. The auxiliary reversible branch provides gradients with complete information to "correct" the parameter learning of the main branch. The implementation constructs an ICN (Information Complete Network) using DHLC (DynamicDet) connections.
    - **Advantages**: It introduces no inference overhead and is applicable to shallow networks (whereas traditional reversible architectures perform worse on shallow networks).

2. **Multi-level Auxiliary Information**:

    - **Function**: Inserts an integration network between the feature pyramid levels of the auxiliary supervision to aggregate gradient information from different prediction heads.
    - **Mechanism**: In traditional deep supervision, different feature pyramids are guided to learn objects of different scales (e.g., shallow layers learn small objects), which leads to the loss of information at other scales (the information fracture problem). Multi-level auxiliary information ensures that each pyramid level receives information about all target objects.
    - **Design Motivation**: To resolve the problems of error accumulation and information fracture in deep supervision. Various integration networks such as FPN or PAN can be used to plan the required semantic levels, enabling network architectures of different scales to benefit.

3. **GELAN (Generalized Efficient Layer Aggregation Network)**:

    - **Function**: Dynamically combines the cross-stage partial connections of CSPNet and the efficient layer aggregation design of ELAN to construct a generic network architecture supporting arbitrary computational blocks.
    - **Mechanism**: Generalizes the design of ELAN, which originally only supported stacked convolutional layers, to support arbitrary computational blocks (Res block, Dark block, CSP block, etc.), allowing users to freely choose suitable blocks based on inference devices.
    - **Design Motivation**: Designed based on gradient path planning to ensure efficient gradient propagation. Experiments demonstrate that using the CSP block yields the best performance (fewer parameters, lower computational cost, and higher AP). GELAN is insensitive to depth settings, and the variation between ELAN depth and CSP depth is linear, ensuring stable performance without specific tuning.

### Loss & Training

- The training settings fully follow YOLOv7 AF, training from scratch for 500 epochs.
- A linear learning rate warmup is applied during the first 3 epochs.
- Mosaic data augmentation is turned off in the last 15 epochs.
- The auxiliary loss of PGI completely inherits the auxiliary head settings of YOLOv7.
- The lead-head guided (LHG) label assignment strategy is adopted for PGI's auxiliary supervision, yielding better results.
- An anchor-free prediction head is used.

## Key Experimental Results

### Main Results

Comprehensive comparison with existing SOTA real-time object detectors on the MS COCO 2017 val set (all trained from scratch):

| Model | Parameters (M) | FLOPs (G) | AP$_{50:95}$ (%) | Comparison with Same-level SOTA |
|---|---|---|---|---|
| YOLO MS-S (Prev. SOTA - Lightweight) | 8.1 | 31.2 | 46.2 | - |
| **YOLOv9-S (Ours)** | **7.1** | **26.4** | **46.8** | +0.6, Params -12%, FLOPs -15% |
| YOLO MS (Prev. SOTA - Medium) | 22.2 | 80.2 | 51.0 | - |
| **YOLOv9-M (Ours)** | **20.0** | **76.3** | **51.4** | +0.4, Params -10%, FLOPs -5% |
| YOLOv7 AF (Prev. SOTA - General) | 43.6 | 130.5 | 53.0 | - |
| **YOLOv9-C (Ours)** | **25.3** | **102.1** | **53.0** | Same AP, Params -42%, FLOPs -22% |
| YOLOv8-X (Prev. SOTA - Large) | 68.2 | 257.8 | 53.9 | - |
| **YOLOv9-E (Ours)** | **57.3** | **189.0** | **55.6** | +1.7, Params -16%, FLOPs -27% |

Comparison with ImageNet pre-trained methods: YOLOv9 trained from scratch outperforms RT-DETR (pre-trained on ImageNet), achieving the same accuracy with only 66% of the parameters of RT-DETR-X.

### Ablation Study

**GELAN block ablation (Table 2)**:

| Block Type | Params | FLOPs | AP$_{50:95}$ | Description |
|---|---|---|---|---|
| Conv (Original ELAN) | 6.2M | 23.5G | 44.8% | Baseline |
| Res block | 5.4M | 21.0G | 44.3% | Fewest parameters but slight accuracy drop |
| Dark block | 5.7M | 21.8G | 44.5% | Moderate parameters and accuracy |
| **CSP block** | **5.9M** | **22.4G** | **45.5%** | Fewer parameters with 0.7% AP improvement |

**Impact of PGI on models of different scales (Table 5)**:

| Model | Baseline AP | + Deep Supervision | + PGI | Description |
|---|---|---|---|---|
| GELAN-S (Lightweight) | 46.7% | 46.5% (-0.2) | 46.8% (+0.1) | DS is harmful to lightweight models, PGI is effective |
| GELAN-M (Medium) | 51.1% | 51.2% (+0.1) | 51.4% (+0.3) | DS is unstable, PGI consistently improves |
| GELAN-C (General) | 52.5% | 52.5% (=) | 53.0% (+0.5) | DS shows no gain, PGI significantly improves |
| GELAN-E (Large) | 55.0% | 55.3% (+0.3) | 55.6% (+0.6) | DS is only effective on large models, PGI outperforms |

### Key Findings

1. **Visualization of information preservation**: Forward propagation visualization with randomly initialized weights shows that PlainNet loses target position information after 50 layers and ResNet becomes blurry after 100 layers, whereas GELAN can still clearly identify target boundaries even at 200 layers.
2. **Visualization of PGI gradient quality**: After only one epoch of bias warm-up, PGI can accurately focus attention on target regions, whereas GELAN without PGI exhibits divergence at target boundaries.
3. **CSP block is the optimal computational block for GELAN**: It reduces parameter count while improving accuracy by 0.7%.
4. **Deep supervision is harmful to lightweight models**: DS leads to a 0.2% AP drop on GELAN-S, whereas PGI yields positive gains across all model scales.
5. **GELAN is insensitive to depth**: Parameters, computational cost, and accuracy scale linearly with depth, eliminating the need for meticulous hyperparameter tuning.
6. **Conventional convolutions outperform depthwise separable convolutions**: By using conventional convolutions, GELAN exceeds the parameter utilization efficiency of depthwise convolution-based YOLO MS.

## Highlights & Insights

- **Theory-driven practical innovation**: Starting from theoretical analysis of the information bottleneck principle and reversible functions, this work deduces that information loss (rather than the traditionally believed gradient vanishing/saturation) is the root cause of training difficulties in deep networks.
- **Zero inference overhead for auxiliary branches**: The reversible branch exists only during training and is completely removed during inference, achieving the best of both worlds.
- **Breaking the limitations of deep supervision**: Traditional deep supervision is applicable only to extremely deep networks, whereas PGI allows lightweight models to benefit from auxiliary supervision as well.
- **Training from scratch outperforms pre-training**: Without relying on ImageNet pre-training, YOLOv9 trained from scratch outperforms methods using large-scale pre-training datasets, such as RT-DETR.
- **Modular design philosophy**: GELAN allows arbitrary replacement of computational blocks, adapting to the needs of different inference devices.

## Limitations & Future Work

- The design of the auxiliary reversible branch (ICN) increases video memory and computational overhead during training; although there is no impact on inference, the training cost remains high.
- The paper only validates the method on MS COCO object detection, without showcasing its generalization capability to other tasks (such as instance segmentation, keypoint detection, etc.).
- Although GELAN's ELAN/CSP depth is insensitive, it lacks a mechanism to automatically search for optimal configurations, relying instead on manual settings.
- A comparison with NAS (Neural Architecture Search) methods is missing; hence, it is unclear whether GELAN is optimal within the search space.
- The auxiliary branch design of PGI is deeply coupled with the specific detection framework, and transferring it to other tasks may require substantial redesign.

## Related Work & Insights

- **Information Bottleneck Theory**: The information bottleneck principle proposed by Tishby et al. provides a theoretical foundation for understanding information loss in deep networks.
- **Reversible Architecture (RevCol, RevNet)**: These architectures guarantee no information loss but suffer from high inference overhead. This work addresses the trade-off by placing the reversible architecture in an auxiliary branch.
- **ELAN / CSPNet**: GELAN is directly built upon the gradient path planning concepts of these two networks.
- **DynamicDet / CBNet**: DHLC connection methods are utilized to construct the auxiliary reversible branch (ICN).
- **Deep Supervision**: PGI is essentially a generalized version of deep supervision, which resolves the problem of error accumulation through multi-level auxiliary information.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The concept of PGI introducing an auxiliary reversible branch based on the information bottleneck theory is novel; GELAN represents a clever combination of mature techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ The ablation studies are extremely detailed (e.g., block types, depths, PGI components, and different model scales), supported by compelling visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ The theoretical analysis is clear, and the logical progression from the problem to the solution is coherent.
- **Value**: ⭐⭐⭐⭐⭐ Given the high significance of the YOLO series, the design concepts of PGI and GELAN are poised to have a profound impact on future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Can OOD Object Detectors Learn from Foundation Models?](can_ood_object_detectors_learn_from_foundation_models.md)
- [\[CVPR 2025\] Boosting Domain Incremental Learning: Selecting the Optimal Parameters Is All You Need](../../CVPR2025/object_detection/boosting_domain_incremental_learning_selecting_the_optimal_parameters_is_all_you.md)
- [\[ECCV 2024\] Bridge Past and Future: Overcoming Information Asymmetry in Incremental Object Detection](bridge_past_and_future_overcoming_information_asymmetry_in_incremental_object_de.md)
- [\[ICML 2025\] Outlier Gradient Analysis: Efficiently Identifying Detrimental Training Samples for Deep Learning Models](../../ICML2025/object_detection/outlier_gradient_analysis_efficiently_identifying_detrimental_training_samples_f.md)
- [\[ECCV 2024\] AugDETR: Improving Multi-scale Learning for Detection Transformer](augdetr_improving_multi-scale_learning_for_detection_transformer.md)

</div>

<!-- RELATED:END -->
