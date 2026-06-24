---
title: >-
  [Paper Note] MarvelOVD: Marrying Object Recognition and Vision-Language Models for Robust Open-Vocabulary Object Detection
description: >-
  [ECCV2024][Multimodal VLM][Open-Vocabulary Object Detection] MarvelOVD is proposed to integrate the detector's context-awareness and background recognition capabilities into the pseudo-label generation and training pipeline of VLMs. By purifying noisy pseudo-labels online and adaptively reweighting training boxes, this framework significantly outperforms existing methods on COCO and LVIS.
tags:
  - "ECCV2024"
  - "Multimodal VLM"
  - "Open-Vocabulary Object Detection"
  - "Pseudo-label Learning"
  - "CLIP"
  - "Online Mining"
  - "Adaptive Reweighting"
date: 2026-05-08
content_hash: 4c14f6cc26b6215e
---

# MarvelOVD: Marrying Object Recognition and Vision-Language Models for Robust Open-Vocabulary Object Detection

**Conference**: ECCV2024  
**arXiv**: [2407.21465](https://arxiv.org/abs/2407.21465)  
**Code**: [https://github.com/wkfdb/MarvelOVD](https://github.com/wkfdb/MarvelOVD)  
**Area**: Multimodal VLM  
**Keywords**: Open-Vocabulary Object Detection, Pseudo-label Learning, CLIP, Online Mining, Adaptive Reweighting

## TL;DR

MarvelOVD is proposed to integrate the detector's context-awareness and background recognition capabilities into the pseudo-label generation and training pipeline of VLMs. By purifying noisy pseudo-labels online and adaptively reweighting training boxes, this framework significantly outperforms existing methods on COCO and LVIS.

## Background & Motivation

Open-vocabulary object detection (OVD) aims to train detectors using only labeled base classes while detecting novel classes during inference. A dominant paradigm is deploying VLMs (e.g., CLIP) to generate pseudo-labels for novel classes to train the detector. However, VLMs suffer from severe **domain shift** on cropped local regions (proposals), resulting in highly noisy pseudo-labels.

The paper provides an in-depth analysis of the sources of noise in CLIP pseudo-labels and finds:

- **Low misclassification rate** (only 3.3%): CLIP achieves high classification accuracy on boxes that actually contain novel objects.
- **Extremely high noise rate** (76.6%): CLIP fails to distinguish "boxes containing no valid objects" (e.g., local parts like a dog's leg or a person's arm).

There are two root causes:

**Lack of contextual information**: CLIP is pre-trained on full images and cannot leverage context outside the cropped areas (e.g., a person's arm is misclassified as a "tie" due to the lack of full-body context).

**Lack of "background" concept**: CLIP's inference lacks a "background" class, forcing it to predict one of the target classes even when the input is irrelevant (e.g., a dog's leg is forcedly classified as a "cow").

**Key Insight**: Detectors naturally possess context-aware feature extraction capabilities through RoI Align and include a "background" class during training. Consequently, the detector can serve as a powerful complement to VLMs, assisting in filtering out noisy pseudo-labels.

## Method

### Overall Architecture

MarvelOVD comprises three core modules:

1. **Candidate Pseudo-label Assignment** (Offline): Uses CLIP to predict all proposals and retains candidate boxes with a low threshold (0.5).
2. **Online Pseudo-label Mining** (Online Mining): Dynamically filters high-quality pseudo-labels during training by incorporating the detector's novelty assessment.
3. **Adaptive Training Optimization**: Stratified label assignment + adaptive proposal reweighting.

### Key Designs

#### 1. Online Pseudo-label Mining

**Burn-in Phase**: The detector is initially trained for the first $\omega=500$ steps using high-confidence (>0.8) CLIP pseudo-labels to acquire an initial capability to distinguish between novel, base, and background classes.

**Online Mining Phase**: After the burn-in phase, weakly-augmented images are fed into the detector to compute the novelty score for each candidate box:

$$z_i = \frac{\sum_{k \in \mathcal{C}^N} \exp(\mathbf{r}_i \cdot \mathbf{c}_k)}{\sum_{j \in \mathcal{C}^B \cup \mathcal{C}^N \cup \{c_{bg}\}} \exp(\mathbf{r}_i \cdot \mathbf{c}_j)}$$

Then, max-norm normalization is applied to obtain $s_i^{det}$, and the final confidence is a weighted fusion of CLIP and the detector:

$$s_i = \lambda \cdot s_i^{CLIP} + (1-\lambda) \cdot s_i^{det}$$

$\lambda=0.5$ and a threshold of $\delta=0.9$ are used to filter final pseudo-labels. As training progresses and the detector strengthens, pseudo-label quality continuously improves.

#### 2. Stratified Label Assignment

Directly using novel pseudo-labels causes **base-novel conflict** due to IoU overlap with base annotations, degrading base class detection performance. The solution is:

- **Step 1**: Perform IoU matching using base annotations first to assign base labels.
- **Step 2**: Match the remaining boxes (labeled as background in the first step) with novel pseudo-labels.

This guarantees that the learning of base classes is not interfered with by pseudo-labels.

#### 3. Adaptive Proposal Reweighting

Imprecise localization of pseudo-labels leads to highly varying overlap between matched training proposals and ground-truth objects. Traditional methods that assign equal loss weights to all matched proposals perform poorly. This work independently computes weights for each novel training proposal:

$$w_i = \lambda' \cdot s_i + (1-\lambda') \cdot r_i$$

where $s_i$ is the pseudo-label confidence, and $r_i = 1 - b_i$ (with $b_i$ being the background score). It is observed that the background score on weakly-augmented images is negatively correlated with the overlap between the training proposal and the ground-truth object; thus, proposals with lower background scores receive higher weights.

### Loss & Training

$$\mathcal{L} = \frac{1}{N}\left(\sum_{i=1}^{n^{base}} l(b_i^{base}, \mathcal{G}^{base}) + \gamma \sum_{i=1}^{n^{novel}} w_i \cdot l(b_i^{novel}, \mathcal{G}^{novel})\right)$$

where $\gamma=2$ serves as the global weight for the novel loss, and $w_i$ is the adaptive, independent weight.

## Key Experimental Results

### Main Results: COCO OVD Comparison

| Method | Extra Data | AP50_Novel | AP50_Base | AP50_All |
|------|---------|------------|-----------|----------|
| OV-RCNN | Transfer Learning | 22.8 | 46.0 | 39.9 |
| ViLD | CLIP Distillation | 27.6 | 59.5 | 51.3 |
| RegionCLIP | Image-Text Pairs + Pre-training | 31.4 | 57.1 | 50.4 |
| VL-PLM (baseline) | CLIP Pseudo-labels | 32.3 | 54.0 | 48.3 |
| BARON | CLIP Distillation | 34.0 | 60.4 | 53.5 |
| OADP | CLIP Distillation + Pseudo-labels | 35.6 | 55.8 | 50.5 |
| Rasheed et al. | Image-Text Pairs + Image-level Labels | 36.6 | 54.0 | 49.4 |
| SAS-Det | CNN-CLIP RoI Features | 37.4 | 58.0 | 53.0 |
| **MarvelOVD** | **CLIP Pseudo-labels** | **38.9** | **56.5** | **51.9** |

Compared to the baseline VL-PLM, the novel AP is improved by **+6.6**, without requiring any extra data or pre-training.

### Main Results: LVIS OVD Comparison

| Method | AP_r (novel) | AP_c| AP_f | AP |
|------|-------------|------|------|-----|
| VLDet | 22.4 | - | - | 34.4 |
| Detic | 24.6 | 32.5 | 35.6 | 32.4 |
| Rasheed et al. | 25.2 | 33.4 | 35.8 | 32.9 |
| **MarvelOVD** | **26.0** | **34.2** | **36.9** | **34.2** |

Optimal performance is consistently achieved under a large vocabulary space (1203 classes).

### Ablation Study: Step-by-step Contribution of Components

| Configuration | AP50_Novel | AP50_Base | AP50_All |
|------|------------|-----------|----------|
| VL-PLM baseline | 32.7 | 54.0 | 48.5 |
| + Weak-Strong Augmentation | 34.2 | 53.9 | 49.1 |
| + Stratified Label Assignment | 34.4 | **56.4↑** | 50.5 |
| + Online Pseudo-label Mining | **37.8↑** | 56.5 | 51.3 |
| + Adaptive Reweighting | **38.9↑** | 56.6 | 51.8 |

- Stratified Label Assignment: base AP recovers from 53.9 to 56.4 (fully supervised level), while novel AP remains unaffected.
- Online Mining: novel AP improves substantially by +3.4, serving as the core contribution.
- Adaptive Reweighting: further improves by +1.1.

### Ablation on Threshold and Burn-in Steps

| Threshold δ | 0.8 | 0.85 | 0.9 | 0.95 |
|--------|-----|------|-----|------|
| AP50_Novel | 37.0 | 38.2 | **38.9** | 38.4 |

| Burn-in Steps ω | 0.5k | 1k | 2k | 5k |
|----------------|------|-----|-----|-----|
| AP50_Novel | **38.9** | 38.7 | 38.7 | 38.5 |

The number of burn-in steps has minimal impact (pseudo-label quality converges once the model stabilizes); a threshold of 0.9 is optimal.

### Key Findings

1. The core issue of CLIP pseudo-labels is not misclassification (3.3%) but the inability to filter out noisy boxes (76.6%).
2. The detector's background score is highly negatively correlated with the proposal quality, serving as the best reliability metric (AP50_Novel=39.8 vs. 37.6 with IoU).
3. $\lambda$ and $\lambda'$ perform robustly within the [0.3, 0.7] range, while extreme values (relying solely on the detector or solely on CLIP) lead to performance degradation.
4. Pseudo-label quality continuously improves along with training, establishing a positive feedback loop.

## Highlights & Insights

1. **In-depth Problem Analysis**: The paper precisely quantifies two root causes of CLIP pseudo-label noise (lack of context + absence of background concept) instead of broadly attributing it to "domain shift".
2. **Online Complementary Mechanism**: The detector and VLM act complementarily—CLIP classifies accurately but cannot recognize noise, whereas the detector recognizes noise but has weaker classification capabilities. Weighted fusion yields significant benefits.
3. **Positive Feedback Loop**: Detector strengthens $\rightarrow$ cleaner pseudo-labels $\rightarrow$ detector strengthens further. This loop is completed online without requiring offline iterations.
4. **Stratified Label Assignment**: Uncovers and resolves the overlooked base-novel conflict issue, restoring base AP to the supervised baseline level single-handedly.
5. **No Extra Data Required**: Does not rely on image-text pairs, image-level classification data, or extra pre-training, exploiting only the latent novel objects already present in the training set.

## Limitations & Future Work

1. It relies on a pre-trained class-agnostic proposal generator; if the proposals fail to cover certain areas, novel objects cannot be detected.
2. The burn-in phase still requires high-threshold CLIP pseudo-labels, demonstrating a baseline dependency on the initial quality of CLIP.
3. Evaluation is restricted to two-stage detectors (e.g., Mask R-CNN, CenterNet2), leaving extension to end-to-end detectors like DETR unverified.
4. The experiments only employ ViT-B/32 CLIP; more powerful VLMs (e.g., ViT-L) might alter the complementary relationship between the detector and the VLM.
5. Novel categories must be specified before training, making it inapplicable to open-world scenarios where novel classes are completely unknown beforehand.

## Related Work & Insights

- **VL-PLM**: The direct baseline of this work, which generates pseudo-labels offline using only CLIP, suffering from heavy noise.
- **Detic**: Leverages ImageNet-21K image-level labels to expand the detector vocabulary, but requires extensive extra classification data.
- **BARON**: Performs knowledge distillation from CLIP; although the approach differs, it shares the same focus on transferring knowledge from VLMs to detectors.
- **Semi-supervised Detection**: Concepts of weak-strong augmentations and teacher-student paradigms remain effective in OVD pseudo-label learning.
- **Insight**: The background-awareness skill of the detector itself is a valuable yet overlooked signal, which could be extended to other dense prediction tasks leveraging VLMs (e.g., open-vocabulary segmentation).

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of using the detector as a complement to denoise VLM predictions is novel and intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive ablation and deep analysis across both COCO and LVIS datasets.
- Writing Quality: ⭐⭐⭐⭐ — Clear problem analysis, intuitive illustrations, and coherent logic.
- Value: ⭐⭐⭐⭐ — An important improvement in the direction of pseudo-label-based OVD; the method is simple, effective, and reproducible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Boosting Vision-Language Models Towards Cross-Domain Incremental Object Detection](../../CVPR2026/multimodal_vlm/boosting_vision-language_models_towards_cross-domain_incremental_object_detectio.md)
- [\[CVPR 2026\] SynCLIP: Synonym-Coherent Language-Image Pretraining for Robust Open-Vocabulary Dense Perception](../../CVPR2026/multimodal_vlm/synclip_synonym-coherent_language-image_pretraining_for_robust_open-vocabulary_d.md)
- [\[CVPR 2026\] ORIC: Benchmarking Object Recognition under Contextual Incongruity in Large Vision-Language Models](../../CVPR2026/multimodal_vlm/oric_benchmarking_object_recognition_under_contextual_incongruity_in_large_visio.md)
- [\[CVPR 2026\] Mechanisms of Object Localization in Vision-Language Models](../../CVPR2026/multimodal_vlm/mechanisms_of_object_localization_in_vision-language_models.md)
- [\[ECCV 2024\] Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models](vary_scaling_up_the_vision_vocabulary_for_large_visionlanguag.md)

</div>

<!-- RELATED:END -->
