---
title: >-
  [Paper Note] Uncertainty-Aware Gradient Stabilization for Small Object Detection
description: >-
  [ICCV 2025][Object Detection][Small object detection] This paper identifies gradient instability caused by steep loss curvature in traditional localization methods when applied to small objects…
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "Small object detection"
  - "gradient stability"
  - "uncertainty-aware"
  - "classification-based localization"
  - "adversarial perturbation"
date: 2026-05-08
content_hash: b4e63b1995616fd0
---

# Uncertainty-Aware Gradient Stabilization for Small Object Detection

**Conference**: ICCV 2025  
**arXiv**: [2303.01803](https://arxiv.org/abs/2303.01803)  
**Code**: None  
**Area**: Object Detection  
**Keywords**: Small object detection, gradient stability, uncertainty-aware, classification-based localization, adversarial perturbation

## TL;DR

This paper identifies gradient instability caused by steep loss curvature in traditional localization methods when applied to small objects, and proposes UGS (Uncertainty-aware Gradient Stabilization), a framework comprising three components — classification-based localization, uncertainty minimization, and uncertainty-guided refinement — to stabilize gradients and significantly improve small object detection performance.

## Background & Motivation

Small object detection has long been a core challenge in computer vision. Taking Cascade R-CNN as an example, it achieves 45.5% and 55.2% AP on medium and large objects on COCO test-dev, respectively, yet only 23.7% on small objects — a substantial gap.

Existing approaches to small object detection primarily address the problem from the following perspectives:
- **Feature enhancement**: Increasing feature map resolution and fusing contextual information
- **Data augmentation**: Oversampling small object regions
- **Scale-aware training**: Multi-scale processing
- **Super-resolution**: Reconstructing high-resolution representations

This paper, however, analyzes the difficulty of small object detection from a novel and orthogonal perspective — **gradient stability**.

**Core Finding**: Through Hessian matrix analysis, the authors demonstrate that conventional norm-based ($\mathcal{L}_2$) and IoU-based localization losses exhibit steeper loss curvature for small objects:

- For $\mathcal{L}_2$ loss, the Hessian with respect to center coordinates is $\mathbf{H}_x = \frac{2}{w_a^2}$, which is inversely proportional to the square of the anchor size — smaller anchors lead to larger Lipschitz constants.
- For IoU loss, the gradient is inversely proportional to the target width $w$, and the Hessian is inversely proportional to $w^3$ — small objects thus incur larger gradients and steeper curvature.

Such steep loss curvature leads to unstable updates during optimization, causing oscillation or divergence near minima and resulting in convergence difficulties for small objects. This finding is further corroborated by experimental visualization: by the 12th training epoch, gradients for medium and large objects have converged, while many small objects still exhibit significant gradient responses.

## Method

### Overall Architecture

The UGS framework consists of three mutually cooperative components: (1) a classification-based localization objective that produces bounded and confidence-driven gradients; (2) an uncertainty minimization loss that explicitly models and reduces prediction uncertainty; and (3) an uncertainty-guided refinement module that identifies and refines high-uncertainty regions via adversarial perturbations. UGS can be integrated as a plug-and-play module into various detectors.

### Key Designs

1. **Classification-based Localization with IN Labels**:

    - **Function**: Converts the continuous regression problem into a classification problem, bounding the gradients.
    - **Mechanism**: The continuous regression range $[-\alpha, \alpha]$ is discretized into $n+1$ grid points. A target value $T$ is mapped to two adjacent grid points to form a two-hot soft label, optimized via cross-entropy loss:
    $\mathcal{L}_{CE} = -\mathbf{p}_{i_l}^* \log \mathbf{p}_{i_l} - \mathbf{p}_{i_r}^* \log \mathbf{p}_{i_r}$
      The key improvement is the introduction of **Interval Non-uniform (IN) labels** via exponential grid spacing:
    $\mathbf{y}_i^{IN} = \text{sign}(\mathbf{y}_i) \cdot \frac{\alpha}{e^{\alpha\beta}-1}(e^{\beta|\mathbf{y}_i|}-1)$
      The parameter $\beta$ controls grid density — a larger $\beta$ places denser grids near zero, balancing the localization target distribution for small objects.
    - **Design Motivation**: The gradient of the classification loss, $|\mathbf{p}_i - \mathbf{p}_i^*|$, is bounded in $[0,1]$, making it confidence-driven and avoiding the gradient explosion induced by regression losses varying with object scale. IN labels address the class imbalance problem arising from small object regression targets being concentrated in a limited range, where soft label probability mass covers only a small number of grid bins.

2. **Uncertainty Minimization (UM)**:

    - **Function**: Explicitly reduces prediction uncertainty via entropy minimization.
    - **Mechanism**: The information entropy of the predicted distribution $\mathbf{p}$ is computed and minimized:
    $\mathcal{L}_{UM} = \mathcal{H}(\mathbf{p}) = -\sum_{i=0}^{n} \mathbf{p}_i \log \mathbf{p}_i$
    - **Design Motivation**: Due to insufficient feature representation, small objects are prone to high prediction uncertainty (i.e., flat predicted distributions). Minimizing entropy suppresses the divergence of predicted distributions, reduces coordinate prediction variance, and provides bounded gradients for stable optimization.

3. **Uncertainty-guided Refinement (UR)**:

    - **Function**: Identifies high-uncertainty regions via adversarial perturbations and applies targeted refinement.
    - **Mechanism**: A min-max optimization objective is formulated in the FPN feature space. Adversarial perturbations are generated along the gradient direction of $\mathcal{L}_{UM}$:
    $\epsilon_i^* \approx \rho \cdot \frac{\nabla_{\mathbf{P}_i} \mathcal{L}_{UM}(\mathbf{P}_i)}{\|\nabla_{\mathbf{P}_i} \mathcal{L}_{UM}(\mathbf{P}_i)\|_2}$
      The perturbation direction points toward regions where $\mathcal{L}_{UM}$ is most sensitive to activation changes, i.e., high-uncertainty regions.
    - **Design Motivation**: Adversarial perturbations amplify the refinement effort on high-uncertainty regions while preserving stable updates in high-confidence regions, thereby enhancing overall feature robustness. Experiments demonstrate that this module learns to handle occluded objects and noise-like targets.

### Loss & Training

The overall localization loss is:
$$\mathcal{L}_{localization} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{UM} + \gamma \sum_{i=1}^{N} \mathcal{L}_i^{ur}(\mathbf{P}_i + \epsilon_i^*)$$

with optimal hyperparameters $\lambda=0.5$, $\gamma=0.1$, and perturbation magnitude $\rho=0.5$. UGS replaces the original detector's localization loss (e.g., $\mathcal{L}_2$ or Smooth-$\mathcal{L}_1$), while the classification loss remains unchanged.

## Key Experimental Results

### Main Results

| Dataset | Method | AP | AP50 | APs |
|-------|------|-----|------|-----|
| VisDrone | FCOS | 19.9 | 37.7 | 11.4 |
| VisDrone | FCOS + UGS | 22.4 (+2.5) | 39.7 | 13.0 |
| VisDrone | Faster R-CNN | 21.3 | 36.4 | 12.8 |
| VisDrone | Faster R-CNN + UGS | 24.2 (+2.9) | 41.3 | 15.8 |
| VisDrone | GFL V1 | 28.4 | 50.0 | 15.9 |
| VisDrone | GFL V1 + UGS | **31.2 (+2.8)** | **53.0** | **19.2** |
| VisDrone | DINO-5scale | 35.5 | 58.0 | 22.4 |
| VisDrone | DINO-5scale + UGS | **38.1 (+2.6)** | **61.9** | **24.2** |

DINO-5scale + UGS surpasses the previous SOTA method DQ-DETR (37.0 AP). On the SODA-A rotated small object detection dataset, UGS improves Rotated RetinaNet by 4.5% AP.

### Ablation Study

| Configuration | AP | AP50 | APs | Notes |
|------|-----|------|-----|------|
| $\mathcal{L}_2$ (Baseline) | 21.3 | 36.4 | 12.8 | Original regression baseline |
| $\mathcal{L}_{CE}$ | 22.1 | 37.1 | 13.2 | Classification-based localization |
| $\mathcal{L}_{CE}$ + IN | 22.5 | 38.2 | 13.4 | + Interval Non-uniform labels |
| $\mathcal{L}_{CE}$ + $\lambda\mathcal{L}_{UM}$ ($\lambda$=0.5) | 22.9 | 38.4 | 13.6 | + Uncertainty minimization |
| $\mathcal{L}_{CE}$ + $\lambda\mathcal{L}_{UM}$ + $\gamma\mathcal{L}^{ur}$ | **23.5** | **39.0** | **14.1** | Full UGS |

Each component contributes consistent gains; the full UGS improves over the $\mathcal{L}_2$ baseline by 2.2% AP.

### Key Findings

1. UGS reduces gradient variance for small objects by 2.9× compared to Smooth-$\mathcal{L}_1$ loss, validating the effectiveness of gradient stabilization.
2. UGS is also effective on general detection datasets (COCO, VOC): R-50 Faster R-CNN improves by 3.8% AP on VOC and 1.4% APs on COCO.
3. Training overhead is modest: a 15% increase in time, 0.6% in computation, and 13% in memory.
4. UGS is compatible with YOLO architectures: TPH-YOLOv5 at $1536^2$ resolution improves by 2.5% AP, reaching 41.7% AP on VisDrone.

## Highlights & Insights

- **Rigorous theoretical analysis**: Starting from the Hessian matrix, the paper formally derives the theoretical basis for gradient instability in small objects — a novel and compelling perspective.
- **Orthogonality to detector design**: As a drop-in replacement for localization losses, UGS can be plug-and-play applied to anchor-based, anchor-free, two-stage, DETR-based, and YOLO-series detectors.
- **Uncertainty-guided adversarial refinement** is an elegant design: using the gradient direction of the uncertainty loss to generate adversarial perturbations enables the model to proactively learn to handle high-uncertainty regions.

## Limitations & Future Work

1. The hyperparameters $\alpha$, $\beta$, and $n$ in IN labels require manual tuning; no adaptive strategy is provided.
2. Validation is limited to 2D detection; gradient stability for small objects in 3D detection or instance segmentation remains unexplored.
3. The UR module introduces additional forward-backward propagation steps, which may be limiting in certain real-time scenarios.
4. Joint application with recent super-resolution and feature enhancement methods has not been explored, leaving potential complementarity unexamined.

## Related Work & Insights

- The classification-based localization approach builds upon GFL V1, extending it with IN labels and uncertainty mechanisms.
- In the context of uncertainty estimation, KL-Loss and GFL V1 are restricted to specific frameworks, whereas UGS offers broader compatibility.
- The strategy of using adversarial perturbations for feature refinement (inspired by works such as the SAM optimizer) opens promising new research directions.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The gradient stability perspective for analyzing small object detection is genuinely novel, with complete theoretical derivation and experimental validation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four datasets, 6+ detectors, comprehensive ablations, gradient variance analysis, and training overhead analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear and visualizations are intuitive.
- **Value**: ⭐⭐⭐⭐⭐ The method is highly generalizable and can serve as a plug-and-play enhancement module for small object detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CQ-DINO: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection](../../NeurIPS2025/object_detection/cq-dino_mitigating_gradient_dilution_via_category_queries_for_vast_vocabulary_ob.md)
- [\[AAAI 2026\] Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection](../../AAAI2026/object_detection/temporal_object-aware_vision_transformer_for_few-shot_video_object_detection.md)
- [\[CVPR 2026\] Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images](../../CVPR2026/object_detection/small_target_detection_based_on_mask-enhanced_attention_fusion_of_visible_and_in.md)
- [\[ICCV 2025\] Adversarial Attention Perturbations for Large Object Detection Transformers](adversarial_attention_perturbations_for_large_object_detection_transformers.md)
- [\[ICCV 2025\] SFUOD: Source-Free Unknown Object Detection](sfuod_source-free_unknown_object_detection.md)

</div>

<!-- RELATED:END -->
