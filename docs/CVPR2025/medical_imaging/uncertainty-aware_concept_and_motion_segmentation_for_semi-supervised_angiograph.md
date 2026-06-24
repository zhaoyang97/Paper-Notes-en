---
title: >-
  [Paper Note] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos
description: >-
  [CVPR2025][Medical Imaging][Semi-supervised learning] This paper proposes the SMART framework, which utilizes a SAM3-based teacher-student architecture combined with text concept prompts, confidence-aware consistency regularization, and dual-stream temporal consistency to achieve semi-supervised vessel segmentation in X-ray coronary angiography videos.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "Semi-supervised learning"
  - "video segmentation"
  - "coronary angiography"
  - "SAM3"
  - "optical flow"
  - "uncertainty"
date: 2026-05-08
content_hash: 201203a792b11d5a
---

# Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos

**Conference**: CVPR2025  
**arXiv**: [2603.00881](https://arxiv.org/abs/2603.00881)  
**Code**: [GitHub](https://github.com/qimingfan10/SMART)  
**Area**: Medical Imaging  
**Keywords**: Semi-supervised learning, video segmentation, coronary angiography, SAM3, optical flow, uncertainty

## TL;DR

This paper proposes the SMART framework, which utilizes a SAM3-based teacher-student architecture combined with text concept prompts, confidence-aware consistency regularization, and dual-stream temporal consistency to achieve semi-supervised vessel segmentation in X-ray coronary angiography videos.

## Background & Motivation

Coronary artery disease (CAD) is the leading cause of death globally, and X-ray coronary angiography (XCA) is the clinical gold standard. Accurate segmentation of coronary arteries is the foundation of automated diagnosis, but the annotation cost is extremely high (requiring frame-by-frame pixel-level annotation).

Limitations of Prior Work:

**Difficulties in direct application of SAM series**: SAM/SAM2 rely on geometric prompts (points, bounding boxes), which limits their generalization ability across different clinical institutions.

**Neglect of temporal information**: Static image methods fail to exploit the temporal dynamics of XCA videos.

**Unreliable pseudo-labels**: Due to low contrast and low signal-to-noise ratio in coronary images, the outputs from the teacher model are highly noisy.

**Advantages of SAM3**: SAM3 introduces concept prompts (semantic text descriptions), which avoids reliance on geometric priors.

## Method

### Phase 1: Text-Driven Segmentation Fine-Tuning
The teacher SAM3 is fine-tuned on the labeled dataset D_l. The SAM3 architecture is retained, and only the parameters related to text prompts in the image encoder, text encoder, and detector are fine-tuned. The optimization is performed using a joint Dice + BCE loss.

### Phase 2: Motion-Aware Semi-Supervised Learning

#### Confidence-Aware Consistency Regularization (CCR)
Addressing the core challenge of unreliable teacher outputs:
1. Inject N=8 independent Gaussian noise perturbations into each frame to obtain N teacher predictions.
2. Calculate the average prediction $\bar{P}$ as the reliable pseudo-label.
3. Compute the uncertainty weight $\mathcal{U}$ (variance of the N predictions).
4. The confidence-aware consistency loss exerts stronger supervision in regions with high uncertainty, pushing the model to improve predictions in uncertain areas.

$$\mathcal{L}_{conf} = \frac{\sum \mathcal{D}(x,y) \cdot \mathcal{U}(x,y)}{\sum \mathcal{U}(x,y) + N\eta} + \frac{\beta}{N} \sum \mathcal{U}(x,y)$$

#### Dual-Stream Temporal Consistency (DSTC)
Leveraging optical flow to model temporal dynamics of vessels:
1. Estimate forward and backward optical flow using pre-trained SEA-RAFT.
2. **Motion consistency loss L_opti**: Ensure pixel-level alignment of predictions between adjacent frames using mask warping.
3. **Flow coherence loss L_coh**: Penalize the deviation of boundary points from the primary motion of the vessel structure, helping to distinguish between foreground and background.

### Total Loss
$$\mathcal{L}_{all} = \lambda_{Dice}\mathcal{L}_{Dice} + \lambda_{Bce}\mathcal{L}_{Bce} + \lambda_{conf}\mathcal{L}_{conf} + \lambda_{opti}\mathcal{L}_{opti} + \lambda_{coh}\mathcal{L}_{coh}$$

Only the student model is used during inference.

## Key Experimental Results

Evaluated on XCAV (111 videos/59 patients) and CAVSA (1061 videos/121 patients) using only 16 labeled videos:

| Method | XCAV DSC | XCAV clDice | CAVSA DSC | CAVSA clDice |
|------|----------|-------------|-----------|--------------|
| UNet (Supervised) | 70.80 | 69.24 | 64.19 | 70.27 |
| SAM3 (Direct) | 42.73 | 34.51 | 30.82 | 30.14 |
| CPC-SAM | 77.90 | 79.15 | 77.90 | 78.28 |
| Denver | 73.30 | 70.40 | 76.53 | 79.17 |
| **SMART** | **84.39** | **83.01** | **91.00** | **97.73** |

**Significant Improvements**:
- XCAV: DSC is 6.49% higher than CPC-SAM.
- CAVSA: Using only 1.5% of the labeled data, the DSC is improved by 13.1%.

**Ablation Study** (Impact of key components on XCAV/CAVSA):

| Configuration | XCAV DSC | XCAV clDice | CAVSA DSC | CAVSA clDice |
|------|---------|-------------|-----------|-------------|
| TPT+CCR (w/o DSTC) | 82.38 | 79.84 | 78.87 | 81.17 |
| TPT+DSTC (w/o CCR) | 76.71 | 79.86 | 25.82 | 32.65 |
| CCR+DSTC (w/o TPT) | 76.24 | 78.53 | 47.77 | 50.37 |
| Full SMART | 84.39 | 83.01 | 91.00 | 97.73 |

When CCR is removed, the CAVSA DSC drops sharply to 25.82%, demonstrating that regularization of unreliable teacher outputs is indispensable. Experiments on the number of noise perturbations indicate that N=8 is the optimal choice (DSC of 84.39 vs 83.59 with N=2).

## Highlights & Insights

1. **Ingenious Application of SAM3 Concept Prompts**: Replacing geometric prompts with textual semantic descriptions avoids the dependence of point/box prompts on shape priors, achieving significantly better cross-institution generalization than point/box prompting schemes.
2. **Elegant Design of Confidence-Aware Regularization**: A counter-intuitive design of uncertainty weighting—applying stronger supervision on more uncertain regions to push the model to improve on its weak spots, rather than simply ignoring uncertain areas.
3. **Dual-Stream Optical Flow Consistency**: Bidirectional (forward + backward) flow mitigates confirmation bias in unidirectional flow, with motion consistency and flow coherence ensuring pixel alignment and foreground/background distinction respectively.
4. **Strong Performance with Minimal Annotations**: Achieves SOTA performance with only 16 labeled videos (having only 1-2 frames annotated per video).
5. **Cross-Domain Generalization on CADICA**: Qualitatively demonstrates robust cross-domain segmentation capabilities on an unlabeled, third-party dataset.
6. **Open-Source Code**: The full code is released, ensuring high reproducibility.

## Limitations & Future Work

1. The teacher model is frozen (not updated) during semi-supervised training, preventing continuous improvement of pseudo-label quality from unlabeled data. This potentially sacrifices further performance gains compared to schemes using updatable teachers.
2. Optical flow estimation relies on pre-trained SEA-RAFT. The quality of the optical flow directly affects the effectiveness of temporal consistency, and no sensitivity analysis was conducted on optical flow accuracy.
3. Validated only on coronary angiography scenarios without extending to other medical video segmentation tasks (such as endoscopy or ultrasound videos).
4. Only qualitative visualization is performed on the CADICA dataset without quantitative metrics; hence, the statistical significance of cross-domain generalization remains unknown.
5. Inference speed and model size are not reported; SAM3 as a foundation model incurs a relatively large computational overhead.
6. Training is conducted for only 6k iterations with a batch size of 4, reflecting a small scale, and performance when scaled to larger data remains unknown.
7. Dual-stream temporal consistency assumes that the vessel topology remains invariant in XCA videos, which might not hold true in longer sequences.

## Rating
- Novelty: 4/5 — The combination of SAM3 concept prompts, uncertainty awareness, and dual-stream optical flow is novel and practical.
- Experimental Thoroughness: 4/5 — Evaluated on three datasets with detailed ablation studies and comparisons against multiple baselines, yielding highly convincing results.
- Writing Quality: 3/5 — Generally clear, but some mathematical symbols are inconsistent; open-sourced code provides extra credit.
- Value: 4/5 — Strong performance under extremely limited annotations holds practical significance for clinical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)
- [\[CVPR 2025\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[CVPR 2025\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [\[CVPR 2025\] Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](addressing_data_scarcity_in_3d_trauma_detection_through_self-supervised_and_semi.md)
- [\[CVPR 2025\] Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation](sam_dpo_semi_supervised.md)

</div>

<!-- RELATED:END -->
