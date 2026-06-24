---
title: >-
  [Paper Note] SFUOD: Source-Free Unknown Object Detection
description: >-
  [ICCV 2025][Object Detection][Source-free domain adaptation] This paper introduces a novel Source-Free Unknown Object Detection (SFUOD) setting and proposes the CollaPAUL framework, which simultaneously detects known and unknown objects without access to source data by combining collaborative tuning to fuse source- and target-domain knowledge with a principal-axis-based pseudo-label assignment strategy for unknown objects.
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "Source-free domain adaptation"
  - "unknown object detection"
  - "mean teacher"
  - "collaborative tuning"
  - "principal axis pseudo-labels"
date: 2026-05-08
content_hash: 94956b006f68ec8b
---

# SFUOD: Source-Free Unknown Object Detection

**Conference**: ICCV 2025
**arXiv**: [2507.17373](https://arxiv.org/abs/2507.17373)  
**Code**: [https://github.com/SFUOD](https://github.com/SFUOD) (to be released)  
**Area**: Object Detection
**Keywords**: Source-free domain adaptation, unknown object detection, mean teacher, collaborative tuning, principal axis pseudo-labels

## TL;DR

This paper introduces a novel Source-Free Unknown Object Detection (SFUOD) setting and proposes the CollaPAUL framework, which simultaneously detects known and unknown objects without access to source data by combining collaborative tuning to fuse source- and target-domain knowledge with a principal-axis-based pseudo-label assignment strategy for unknown objects.

## Background & Motivation

**Source-Free Object Detection (SFOD)** aims to adapt a pre-trained detector to an unlabeled target domain without accessing source data, thereby addressing data privacy concerns. However, existing SFOD methods assume a closed-set scenario in which the source and target domains share the same category set.

In real-world applications such as autonomous driving, detectors must recognize **unknown objects** not defined in the source domain. For instance, a model trained exclusively on annotated vehicles (Car, Truck, Bus) must also detect pedestrians, cyclists, and other novel categories at inference time to prevent safety incidents.

Existing SFOD methods (e.g., DRU, PET) face two major challenges under the SFUOD setting:

**Knowledge confusion**: Source-domain knowledge conflicts with unknown objects, causing known objects to be misclassified as unknown and vice versa.

**Pseudo-label failure**: The teacher model, having never observed unknown objects in the source domain, cannot generate reliable pseudo-labels for them.

Empirical validation confirms that directly applying SFOD methods to the SFUOD setting yields low known mAP and near-zero unknown recall.

## Method

### Overall Architecture

CollaPAUL is built upon the Mean Teacher framework and comprises two core components:
1. **Collaborative Tuning**: Fuses source- and target-domain knowledge to mitigate knowledge confusion.
2. **Principal Axis Unknown Labeling (PAUL)**: Assigns pseudo-labels to unknown objects.

### Key Designs

**1. Collaborative Tuning**

An auxiliary target encoder, independent of the source-pretrained student encoder, is introduced to extract target-domain-specific knowledge:

- **Truncated SVD Reconstruction**: SVD decomposition is applied to backbone features, retaining only the top-$r$ principal components for reconstruction to reveal the implicit representation of the target domain.
- **Cross-Domain Attention**: Collaborative layers are inserted between decoder layers with the following configuration:
    - Query: source-domain features $f_s$
    - Key/Value: concatenation of source-domain features $f_s$ and target-domain features $f_t$
    - Softmax attention adaptively fuses knowledge from both domains.

Collaborative layers are inserted after the first $L=3$ decoder layers of DETR, enabling the decoder to learn enriched fused representations through iterative propagation.

**2. Principal Axis Unknown Labeling (PAUL)**

Core assumption: known and unknown objects share the property of *objectness*, distinguishing them from non-object proposals.

Procedure:
1. Assign pseudo-labels to known objects using a confidence threshold of 0.3.
2. Apply PCA to the features of known proposals to extract the principal axis $P$.
3. Project both known and remaining proposals onto the principal axis: $\bar{f} = f \cdot P^T$.
4. Compute objectness scores as the cosine similarity between remaining proposals and known proposals along the principal axis.
5. Set threshold $\delta$ as the mean objectness score of known proposals and generate objectness mask $M_\text{obj}$.
6. Combine with confidence mask $M_\text{conf}$ via OR operation to produce the final unknown mask $M_\text{unk}$.
7. Selected proposals are labeled as the "unknown" class.

### Loss & Training

- Detection loss: classification loss + $L_1$ regression loss + GIoU loss
- Teacher model updated via EMA with $\alpha = 0.99$
- Base model: Deformable-DETR + ResNet-50
- AdamW optimizer; trained on 4 × RTX 3090 GPUs

## Key Experimental Results

### Main Results

Weather adaptation benchmark (Cityscapes → Foggy Cityscapes):

| Method | Car | Truck | Bus | Known mAP | U-Recall | H-Score |
|--------|-----|-------|-----|-----------|----------|---------|
| Source only | 43.20 | 12.05 | 24.43 | 26.56 | 0.00 | 0.00 |
| Mean Teacher | 50.20 | 0.00 | 0.54 | 16.91 | 6.02 | 8.88 |
| DRU | 41.14 | 9.65 | 18.12 | 22.97 | 3.60 | 6.22 |
| **CollaPAUL** | **52.10** | **16.49** | **28.37** | **32.32** | **10.59** | **15.95** |

Cross-scene benchmark (Cityscapes → BDD100K): Known mAP 28.21, U-Recall 8.57, H-Score 13.15, achieving comprehensive superiority across all metrics.

### Ablation Study

Contribution of individual components (weather adaptation):

| Collab | PAUL | Known mAP | U-Recall | H-Score |
|--------|------|-----------|----------|---------|
| ✗ | ✗ | 22.97 | 3.60 | 6.22 |
| ✓ | ✗ | 30.63 | 3.56 | 6.38 |
| ✗ | ✓ | 25.40 | 6.46 | 10.30 |
| ✓ | ✓ | **32.32** | **10.59** | **15.95** |

Ablation on the number of collaborative layers: $L=3$ achieves the best performance (H-Score 15.95); $L=1/2$ is insufficient while $L=4/5$ leads to overfitting.

PAUL vs. alternative unknown labeling methods: PAUL (H-Score 15.95) $\gg$ Attention-driven (7.12) $\gg$ Confidence-based (6.38).

### Key Findings

- Collaborative tuning primarily improves known mAP (+7.66%), while PAUL mainly boosts unknown recall (+2.86%); their combination yields synergistic gains.
- Cross-domain attention substantially outperforms simple prefix-tuning (H-Score: 15.95 vs. 12.57).
- Principal-axis-based objectness estimation is critical for identifying unknown objects, far surpassing conventional confidence-based approaches.
- Combining the objectness mask and the confidence mask yields the best overall performance.

## Highlights & Insights

1. **Definition of the SFUOD setting**: Bridges the gap between source-free domain adaptation and open-set detection, with clear practical relevance.
2. **Elegant principal-axis objectness estimation**: Leverages the principal component space of known objects to identify unknowns—a concise and principled design.
3. **Collaborative tuning resolves knowledge confusion**: An independent target encoder preserves source-domain knowledge while learning target-domain representations.
4. **Complete benchmark construction**: Two SFUOD benchmarks are provided to facilitate future research.

## Limitations & Future Work

- The SFUOD formulation assigns all unknown objects to a single "unknown" category without fine-grained classification.
- The number of retained principal components $r$ in truncated SVD reconstruction requires tuning.
- Evaluation is limited to Cityscapes-based benchmarks; large-scale and diverse datasets remain untested.
- The parameter count of collaborative layers grows with detector scale.

## Related Work & Insights

- Compared to SOMA (AOOD), SFUOD requires no source data, making it more practical.
- Compared to OWOD (Open World Object Detection), SFUOD does not require incremental learning.
- The principal-axis projection idea is generalizable to other settings requiring estimation of inter-class commonality.
- The cross-domain attention design in collaborative tuning is applicable to other domain adaptation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Defines an important new setting with creative method design)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive ablations, but limited number of benchmarks)
- Writing Quality: ⭐⭐⭐⭐ (Clear problem formulation and complete method description)
- Value: ⭐⭐⭐⭐ (Strong application potential of the new setting, though performance has room for improvement)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection](../../ICLR2026/object_detection/cgsa_class-guided_slot-aware_adaptation_for_source-free_object_detection.md)
- [\[AAAI 2026\] Beyond Boundaries: Leveraging Vision Foundation Models for Source-Free Object Detection](../../AAAI2026/object_detection/beyond_boundaries_leveraging_vision_foundation_models_for_so.md)
- [\[CVPR 2026\] Foundation Model Priors Enhance Object Focus in Feature Space for Source-Free Object Detection](../../CVPR2026/object_detection/foundation_model_priors_enhance_object_focus_in_feature_space_for_source-free_ob.md)
- [\[CVPR 2026\] Detecting Unknown Objects via Energy-Based Separation for Open World Object Detection](../../CVPR2026/object_detection/detecting_unknown_objects_via_energy-based_separation.md)
- [\[ICCV 2025\] DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion](distil_data-free_inversion_of_suspicious_trojan_inputs_via_latent_diffusion.md)

</div>

<!-- RELATED:END -->
