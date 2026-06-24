---
title: >-
  [Paper Note] Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding
description: >-
  [CVPR2025][Medical Imaging][3D object detection] A two-stage label-efficient learning framework is proposed: first, a 3D U-Net encoder is pre-trained via self-supervised Masked Image Modeling on 1,206 unlabeled CT scans; then, combined with VDETR + Vertex RPE and Mean Teacher semi-supervised learning, it achieves a 3D abdominal trauma detection mAP@0.50 of 45.30% (+115%) using only 144 labeled cases.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "3D object detection"
  - "Self-Supervised Learning"
  - "Semi-Supervised Learning"
  - "Trauma Detection"
  - "VDETR"
date: 2026-05-08
content_hash: bed0a3222869e50d
---

# Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding

**Conference**: CVPR2025  
**arXiv**: [2603.12514](https://arxiv.org/abs/2603.12514)  
**Code**: [github.com/shivasmic/3d-trauma-detection-ssl](https://github.com/shivasmic/3d-trauma-detection-ssl)  
**Area**: Medical Imaging  
**Keywords**: 3D object detection, Self-Supervised Learning, Semi-Supervised Learning, Trauma Detection, VDETR

## TL;DR

A two-stage label-efficient learning framework is proposed: first, a 3D U-Net encoder is pre-trained via self-supervised Masked Image Modeling on 1,206 unlabeled CT scans; then, combined with VDETR + Vertex RPE and Mean Teacher semi-supervised learning, it achieves a 3D abdominal trauma detection mAP@0.50 of 45.30% (+115%) using only 144 labeled cases.

## Background & Motivation

**Clinical Urgency of Abdominal Trauma Detection**: Manual slice-by-slice analysis of CT scans in emergency departments is time-consuming and prone to inter-observer variability. Automated detection can significantly accelerate clinical decision-making.

**Extreme Scarcity of Annotation**: In the RSNA abdominal trauma dataset, only 206 out of 4,711 cases (4.4%) have segmentation annotations, making traditional fully supervised methods infeasible.

**Specific Challenges in 3D Detection**: Abdominal organs exhibit highly irregular shapes, and 2-D center distances are insufficient to describe 3D bounding box relationships. Moreover, general-purpose 3D feature extractors transfer poorly.

**Complementary Potential of Self-Supervised + Semi-Supervised Learning**: Self-supervised pre-training learns anatomical priors from unlabeled data, while semi-supervised learning stabilizes training by utilizing large amounts of unlabeled data.

**Locality Advantage of VDETR**: The Vertex RPE in V-DETR provides explicit geometric interior-exterior relationships by calculating the relative position encoding of 8 vertices.

**Limitations of Prior Work**: Direct adaptation of 2D detection to 3D exhibits poor performance, while models like VoteNet and FCAF3D rely heavily on extensive annotations.

## Method

### Overall Architecture

A three-stage pipeline is designed: (1) A 3D U-Net is pre-trained via patch-based MIM on all 1,206 CT scans; (2) The VDETR detector is trained in two phases (frozen $\rightarrow$ unfrozen encoder); (3) Mean Teacher semi-supervised learning utilizes 2,000 unlabeled cases.

### Key Designs

- **Masked Image Modeling**: Divides a 128³ patch into 8³ sub-blocks, randomly masks 75% of them, and then reconstructs the raw intensity values of the masked regions using the 3D U-Net.
- **3D Vertex RPE**: For each query and voxel position, it calculates the offset vectors to the 8 vertices of the predicted box, generating an attention bias via an MLP: $\mathbf{A} = \text{softmax}(\mathbf{QK}^T + \mathbf{R})$, where $\mathbf{R} = \sum_{i=1}^{8}\text{MLP}_i(F(\Delta\mathbf{P}_i))$.
- **Two-stage Training**: Phase I (0-20 epochs) freezes the encoder to train only the VDETR decoder. Phase II (20-100 epochs) unfreezes the encoder for joint fine-tuning with the encoder learning rate set to 1/10 of the decoder's (1e-5 vs 1e-4) and applies a 3-epoch warmup to prevent catastrophic forgetting.
- **Mean Teacher Semi-Supervised Learning**: Weak augmentation (Gaussian noise σ=0.01, intensity shift ±2%) generates teacher pseudo-labels, while strong augmentation (σ=0.05, shift ±10%, elastic deformation) generates student predictions. The consistency loss includes center MSE, size MSE, and classification KL divergence.
- **Classification Branch**: Frozen encoder $\rightarrow$ GAP $\rightarrow$ 2-layer FC (256$\rightarrow$128$\rightarrow$7) with only 33,799 trainable parameters. A weighted BCE loss where $w_i^{\text{pos}} = N_i^{\text{neg}} / N_i^{\text{pos}}$ (e.g., bowel injury weight 4.45) is used to handle extreme class imbalance.

### Loss & Training

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{supervised}} + \lambda(t) \times (\mathcal{L}_{\text{center}} + \mathcal{L}_{\text{size}} + \mathcal{L}_{\text{cls}})$$

where $\lambda(t)$ linearly increases from 0 to 0.3 (epochs 20-60). Weighted BCE loss is used in the classification task to handle class imbalance.

## Key Experimental Results

### 3D Detection Validation Set Ablation

| Method | Best Epoch | mAP@0.50 | mAP@0.75 |
|------|-----------|----------|----------|
| VDETR (without SSL) | 5 | 26.36% | 6.82% |
| **VDETR + SSL** | **99** | **56.57%** | **45.12%** |
| Gain | — | +115% | +562% |

### 3D Detection Test Set

| Metric | Without SSL | With SSL | Gain |
|------|--------|--------|------|
| mAP@0.50 | 23.03% | 45.30% | +97% |
| mAP@0.75 | 16.67% | 28.72% | +72% |

### Classification Task (Frozen Encoder Linear Probe)

| Category | Test Accuracy |
|------|---------|
| Bowel Injury | 97.5% |
| High-grade Liver Injury | 98.3% |
| Overall Average | 94.07% |

### Key Findings

- Without semi-supervised learning, the detection performance peaks at epoch 5 and then collapses (overfitting); adding semi-supervised learning stabilizes training and ensures convergence.
- Features learned from self-supervised pre-training are of extremely high quality: the frozen encoder linear probe achieves 94.07% accuracy at epoch 0 with no further improvement during subsequent training.
- Surprisingly, semi-supervised learning does not benefit the classification task (75.4% vs baseline 77.7%), likely due to pseudo-label noise.

## Highlights & Insights

1. **Extreme Data Efficiency Validation**: Achieved a usable 3D detector with only 144 labeled and 2,000 unlabeled cases.
2. **Synergistic Effect of Self-Supervision $\rightarrow$ Semi-Supervision**: Pre-training provides a stable feature representation, while semi-supervised learning provides implicit regularization; their combination avoids training collapse.
3. **3D Medical Adaptation of Vertex RPE**: Successfully adapted the SOTA method from 3D detection in natural scenes to the medical domain.
4. **Complete System Design**: Open-sourced the entire pipeline including data preprocessing $\rightarrow$ self-supervision $\rightarrow$ detection + classification.
5. **Rigorous Self-Supervised Quality Validation**: Feature quality of the pre-trained model is doubly proven by a reconstruction PSNR of 19.39dB and a linear probe accuracy of 76%.

## Limitations & Future Work

1. The evaluation set for the detector contains only 32 cases, limiting statistical significance.
2. The AUC for classification is only 51.4% (due to probability calibration issues); although it does not affect binary predictions, it indicates that the model's confidence is unreliable.
3. Validation is limited to the RSNA abdominal trauma dataset, with lack of generalization to other 3D medical detection tasks.
4. Compared to the RSNA 2023 competition winner (98% AUC, multi-model ensemble), a significant gap remains for this single-model approach.
5. High computational overhead: full 3D voxel processing requires an A100 GPU, with a batch size of only 1-2.

## Related Work & Insights

- **3D Detection**: Fully convolutional methods like VoteNet and FCAF3D rely heavily on dense annotations; 3DETR and GroupFree are adapted to 3D but lack localized learning.
- **V-DETR**: Solves the locality problem in 3D through an 8-vertex RPE; this work is the first to apply it to 3D medical detection.
- **Self-Supervised Learning**: MAE is effective in natural images; Eckstein et al. validated the value of pre-training for 3D medical detection.
- **Semi-Supervised Detection**: Mean Teacher and Unbiased Teacher are effective in 2D detection; this work extends them to 3D medical scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ (Individual components are existing methods, but their integration is logical and highly systematic.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Ablation studies are convincing, but the test set is too small; lacks comparison with other 3D detectors.)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, but a bit too long with redundant details.)
- Value: ⭐⭐⭐⭐⭐ (3D medical detection under label scarcity is a real pain point, and the framework holds significant reference value.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] An OpenMind for 3D Medical Vision Self-supervised Learning](../../ICCV2025/medical_imaging/an_openmind_for_3d_medical_vision_selfsupervised_learning.md)
- [\[CVPR 2025\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)
- [\[CVPR 2025\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)
- [\[NeurIPS 2025\] Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum](../../NeurIPS2025/medical_imaging/ditch_the_denoiser_emergence_of_noise_robustness_in_self-supervised_learning_fro.md)
- [\[NeurIPS 2025\] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](../../NeurIPS2025/medical_imaging/self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)

</div>

<!-- RELATED:END -->
