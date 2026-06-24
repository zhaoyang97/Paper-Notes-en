---
title: >-
  [Paper Note] RSAR: Restricted State Angle Resolver and Rotated SAR Benchmark
description: >-
  [CVPR 2025][Object Detection][Rotated Object Detection] This paper revisits angle decoders in rotated object detection from a unified perspective of dimensional mapping, reveals the prediction bias caused by ignoring the unit circle constraint in existing methods, proposes the Unit Cycle Resolver (UCR), and leverages UCR to construct RSAR, currently the largest multi-class rotated SAR object detection dataset.
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Rotated Object Detection"
  - "SAR Images"
  - "Boundary Discontinuity of Angle"
  - "Unit Circle Constraint"
  - "Weakly Supervised"
date: 2026-05-08
content_hash: f9d24f919db8c488
---

# RSAR: Restricted State Angle Resolver and Rotated SAR Benchmark

**Conference**: CVPR 2025  
**arXiv**: [2501.04440](https://arxiv.org/abs/2501.04440)  
**Code**: [https://github.com/zhasion/RSAR](https://github.com/zhasion/RSAR)  
**Area**: Object Detection / Remote Sensing  
**Keywords**: Rotated Object Detection, SAR Images, Boundary Discontinuity of Angle, Unit Circle Constraint, Weakly Supervised

## TL;DR
This paper revisits angle decoders in rotated object detection from a unified perspective of dimensional mapping, reveals the prediction bias caused by ignoring the unit circle constraint in existing methods, proposes the Unit Cycle Resolver (UCR), and leverages UCR to construct RSAR, currently the largest multi-class rotated SAR object detection dataset.

## Background & Motivation

**Background**: Rotated object detection represents bounding boxes as $(cx, cy, w, h, \theta)$ and is widely applied in remote sensing, 3D detection, and scene text detection. While significant progress has been made in the optical remote sensing domain (driven by datasets like DOTA), rotated object detection in the SAR (Synthetic Aperture Radar) domain has progressed slowly.

**Limitations of Prior Work**: (1) The SAR domain lacks large-scale rotated annotation datasets, and annotation is highly costly and time-consuming; (2) The core challenge of rotated object detection—the boundary discontinuity of angle—remains unresolved. Existing angle decoding schemes (PSC via phase-shift coding, ACM via complex exponential functions), although mapping one-dimensional angles to multi-dimensional coding spaces to resolve boundary discontinuity, ignore the unit circle constraint that the coding states must satisfy.

**Key Challenge**: Existing angle encoding methods independently predict the coding values of each dimension, and the prediction results can deviate from the unit circle. This leads to a many-to-one mapping, where different coding states can correspond to the same angle through linear scaling, unnecessarily complicating the optimization space and introducing prediction bias.

**Goal**: (1) Propose an improved angle decoder to enhance angle prediction accuracy, particularly for weakly supervised rotated detection; (2) Efficiently build a large-scale SAR rotated object detection dataset using the improved method.

**Key Insight**: The authors revisit PSC and ACM from a unified perspective of dimensional mapping—one-dimensional mapping suffers from boundary discontinuity, two-dimensional mapping corresponds to ACM ($\cos\theta + j\sin\theta$), and three-dimensional mapping corresponds to PSC. This unified perspective clearly reveals the common flaw of "lacking unit circle constraints."

**Core Idea**: By adding a simple unit circle constraint loss $\mathcal{L}_{uc}$, the predicted angle coding state is ensured to meet the unit circle (2D) or ellipse (3D) constraint conditions, eliminating the optimization difficulties caused by many-to-one mapping.

## Method

### Overall Architecture
UCR is a plug-and-play angle decoding module applicable to the angle prediction head of any rotated object detector. Taking the weakly supervised method H2RBox-v2 as an example, where the model learns to predict rotated bounding boxes from horizontal bounding box annotations, UCR replaces its original angle decoder. During training, classification, regression, and unit circle constraints are jointly optimized using the total loss $\mathcal{L} = \mathcal{L}_{cls} + \lambda_{reg}\mathcal{L}_{reg} + \lambda_{uc}\mathcal{L}_{uc}$.

### Key Designs

1. **Unified Perspective of Dimensional Mapping**:

    - Function: Unify PSC and ACM, two seemingly different angle decoding methods, into the same mathematical framework to reveal their common flaws.
    - Mechanism: The root cause of the angle boundary discontinuity problem is that the values at both ends of the angle range in one-dimensional space should be equal but are far apart. The solution is dimensional mapping: two-dimensional mapping $m_1 = \cos\theta, m_2 = \sin\theta$ (ACM); three-dimensional mapping satisfies $\sum_{i=1}^3 m_i^2 = 3/2$ and $\sum_{i=1}^3 m_i = 0$ (PSC is an effective solution). Key finding: regardless of the mapping, the coding states must lie on the unit circle/ellipse, but existing methods fail to guarantee this constraint due to independent dimensional predictions.
    - Design Motivation: The unified perspective not only clarifies the mathematical essence of existing methods but, more importantly, exposes the common defect of "independent predictions leading to deviations from the unit circle."

2. **Unit Circle Constraint Loss**:

    - Function: Constrain the predicted angle coding state of the model to satisfy unit circle conditions, eliminating the optimization difficulties caused by many-to-one mapping.
    - Mechanism: $\mathcal{L}_{uc} = |n/2 - \sum_{i=1}^n m_i^2| + \sigma(n) |\sum_{i=1}^n m_i|$, where $n$ is the mapping dimension. For 2D ($n=2$), $\sigma(2)=0$, and the loss simplifies to $|1 - m_1^2 - m_2^2|$, which constrains the coding state to the unit circle. For 3D ($n=3$), $\sigma(3)=1$, introducing an additional constraint that the sum of the coding values is zero.
    - Design Motivation: The constraint loss restricts the solution space of the coding state from the entire $n$-dimensional space to the unit circle/ellipse, eliminating the many-to-one mapping caused by linear scaling, simplifying the optimization target, and improving angle prediction accuracy.

3. **Invalid Region Mechanism**:

    - Function: Address the high randomness of coding values in the early stages of training to improve training stability.
    - Mechanism: An invalid region is defined as $\sum_{i=1}^n m_i^2 < m_{invalid}$. When the predicted coding state is near the center of the unit circle (with an extremely small magnitude), only the unit circle constraint loss is applied, while the angle regression loss is bypassed. Since the mapping from coding values near the center to angles is unstable, the constraint loss is used first to push predictions toward the unit circle boundary before performing angle regression.
    - Design Motivation: At the beginning of training, the predicted coding values are close to zero (due to random initialization), and forcing angle regression at this point would lead to gradient instability. The invalid region allows the model to first learn "within what range to predict" and then "what value to predict."

### Loss & Training
The total loss is $\mathcal{L} = \mathcal{L}_{cls} + \lambda_{reg}\mathcal{L}_{reg} + \lambda_{uc}\mathcal{L}_{uc}$. UCR is plug-and-play and can be applied to both fully and weakly supervised detectors. When constructing the RSAR dataset, the UCR-enhanced H2RBox-v2 generates rotated pseudo-labels, which are then manually calibrated to obtain the final annotations.

## Key Experimental Results

### Main Results
Weakly supervised rotated detection performance on RSAR, DOTA-v1.0, and HRSC:

| Method | Angle Decoder | Dimension | Supervision | RSAR AP50 | DOTA AP50 | HRSC AP50 |
|------|----------|------|------|----------|----------|----------|
| FCOS (R-50) | - | - | Rotated Box | 66.66 | 71.44 | 89.26 |
| H2RBox-v2 | ACM | 2D | Horizontal Box | 65.34 | 72.37 | 89.58 |
| H2RBox-v2 | PSC | 3D | Horizontal Box | 65.16 | 72.31 | 89.30 |
| H2RBox-v2 | **UCR** | **2D** | **Horizontal Box** | **69.21** | **73.22** | **89.73** |
| H2RBox-v2 | **UCR** | **3D** | **Horizontal Box** | **68.33** | **73.99** | **89.74** |

### Ablation Study

| Configuration | RSAR mAP | DOTA mAP | Description |
|------|---------|---------|------|
| ACM (Unconstrained) | 30.64 | 41.05 | Baseline |
| PSC (Unconstrained) | 30.91 | 40.69 | Baseline |
| UCR 2D | 32.25 | 42.65 | +1.6/+1.6 |
| UCR 3D | **32.64** | **43.10** | +1.7/+2.1 |

### Key Findings
- UCR significantly improves angle prediction accuracy in weakly supervised settings, with AP50 on RSAR rising from 65.34 to 69.21 (2D UCR), even outperforming fully supervised FCOS (66.66).
- On DOTA-v1.0, weakly supervised UCR (73.99 AP50) outperforms fully supervised FCOS (71.44 AP50), demonstrating the crucial role of angle decoder design.
- The 3D mapping UCR slightly outperforms the 2D mapping UCR, but the difference is marginal, and the 2D mapping is simpler to compute.
- RSAR contains 95,842 images and 183,534 annotated instances, rendering it currently the largest multi-class rotated SAR detection dataset.

## Highlights & Insights
- **Theoretical Contribution of the Unified Perspective**: Unifying PSC and ACM into a dimensional mapping framework represents an elegant research paradigm ("unify first, then discover flaws"), revealing a simple yet overlooked constraint.
- **Minimalist Yet Effective Improvements**: Merely adding a single constraint loss term significantly improves angle prediction accuracy, introducing minimal invasiveness to the overall framework.
- **Efficient Strategy for Dataset Construction**: Generating pseudo-labels with an improved weakly supervised model followed by human calibration is far more efficient than manual annotation from scratch, presenting a highly successful case of AI-assisted annotation.

## Limitations & Future Work
- The constraint in UCR is a soft constraint (implemented via a loss function) and cannot guarantee that predictions fall strictly on the unit circle. Hard constraints (such as normalization operations) could be explored in future work.
- Certain categories (e.g., aircraft) in the RSAR dataset were excluded due to orientation ambiguity, indicating that the dataset coverage needs further expansion.
- This approach has only been validated on rotated bounding box detection; its performance on other tasks requiring angle predictions, such as instance segmentation, remains to be evaluated.
- The invalid region threshold $m_{invalid}$ is a hyperparameter that requires manual tuning.

## Related Work & Insights
- **vs ACM**: ACM encodes angles using complex exponential functions (2D mapping), but predicting cos and sin independently does not guarantee they remain on the unit circle. UCR resolves this issue by adding a constraint loss.
- **vs PSC**: PSC employs phase-shift coding (3D mapping) and similarly ignores constraints. The unified perspective of UCR reveals that they share the same essence and defects.
- **vs GWD/KLD**: Methods based on Gaussian distributions alleviate boundary problems from different aspects but do not fundamentally solve them. UCR directly constrains the coding space, presenting a more straightforward solution.

## Rating
- Novelty: ⭐⭐⭐⭐ The unified perspective and unit circle constraint are concise and strong contributions, although the improvement itself is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across three datasets (RSAR, DOTA, and HRSC), and introduces a brand-new RSAR dataset.
- Writing Quality: ⭐⭐⭐⭐ The explanation of the unified perspective is clear, with intuitive diagrams aiding comprehension.
- Value: ⭐⭐⭐⭐ The RSAR dataset substantially promotes the field of rotated SAR detection, and the simple, effective UCR can be widely utilized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VCBench: A Streaming Counting Benchmark for Spatial-Temporal State Maintenance in Long Videos](vcbench_a_streaming_counting_benchmark_for_spatial-temporal_state_maintenance_in.md)
- [\[CVPR 2025\] Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset](object_detection_using_event_camera_a_moe_heat_conduction_based_detector_and_a_n.md)
- [\[CVPR 2026\] Fourier Angle Alignment for Oriented Object Detection in Remote Sensing](../../CVPR2026/object_detection/fourier_angle_alignment_for_oriented_object_detection_in_remote_sensing.md)
- [\[NeurIPS 2025\] BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes](../../NeurIPS2025/object_detection/burstdeflicker_a_benchmark_dataset_for_flicker_removal_in_dynamic_scenes.md)
- [\[NeurIPS 2025\] Spatio-Temporal Graphs Beyond Grids: Benchmark for Maritime Anomaly Detection](../../NeurIPS2025/object_detection/spatio-temporal_graphs_beyond_grids_benchmark_for_maritime_anomaly_detection.md)

</div>

<!-- RELATED:END -->
