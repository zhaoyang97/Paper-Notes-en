---
title: >-
  [Paper Note] Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding
description: >-
  [CVPR 2026][Medical Imaging][Self-supervised learning] This paper proposes a two-stage label-efficient framework: a patch-based MIM self-supervised pretraining of a 3D U-Net encoder on 1,206 unlabeled CT volumes…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Self-supervised learning"
  - "semi-supervised learning"
  - "Masked Image Modeling"
  - "3D object detection"
  - "VDETR"
  - "Vertex Relative Position Encoding"
  - "abdominal CT"
  - "trauma detection"
date: 2026-05-08
content_hash: 277e3b134cc3555f
---

# Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding

**Conference**: CVPR 2026
**arXiv**: [2603.12514](https://arxiv.org/abs/2603.12514)
**Code**: [GitHub](https://github.com/shivasmic/3d-trauma-detection-ssl)
**Area**: Medical Imaging / 3D Trauma Detection
**Keywords**: Self-supervised learning, semi-supervised learning, Masked Image Modeling, 3D object detection, VDETR, Vertex Relative Position Encoding, abdominal CT, trauma detection

## TL;DR
This paper proposes a two-stage label-efficient framework: a patch-based MIM self-supervised pretraining of a 3D U-Net encoder on 1,206 unlabeled CT volumes, followed by VDETR with 3D vertex relative position encoding for 3D lesion detection, augmented by Mean Teacher semi-supervised consistency regularization over 2,000 additional unlabeled volumes. Using only 144 annotated samples, the framework achieves 56.57% val mAP@0.50, a 115% improvement over fully supervised training.

## Background & Motivation
**Urgent clinical need for abdominal CT trauma detection**: Emergency settings require rapid and accurate detection of internal injuries, yet manual analysis of 3D medical volumes is time-consuming and subject to inter-observer variability.

**Severe scarcity of annotated data**: Among 4,711 sequences in the RSNA Abdominal Trauma dataset, only 206 (4.4%) carry segmentation annotations, rendering conventional fully supervised methods impractical.

**Loss of 3D spatial relationships in 2D slice-wise analysis**: Treating CT volumes as independent 2D slices fails to capture the complex volumetric spatial structures present in the data.

**Inadequacy of centroid-based positional metrics for irregular organs**: Standard DETR-style position encodings compute distances from centroids to pixels, providing insufficient geometric description for irregularly shaped organs and lesion regions.

**Poor transfer of features pretrained on natural domains**: 3D feature extractors pretrained on natural images or videos transfer poorly to medical imaging data characterized by Hounsfield Unit values and distinctive intensity distributions.

**Underexplored integration of SSL, semi-supervised learning, and Transformer-based detection in 3D medical imaging**: A systematic combination of these three paradigms remains an open gap in the literature.

## Method

### Overall Architecture
Input: raw DICOM CT sequences → preprocessing and standardization to $512\times336\times336$ voxels (anisotropic spacing $2.0\times1.0\times1.0$ mm) → Stage 1: patch-based MIM self-supervised pretraining of 3D U-Net encoder → Stage 2: frozen/unfrozen encoder + VDETR decoder for 3D detection + Mean Teacher semi-supervised training → Output: 3D bounding boxes + classification labels.

### Key Design 1: Patch-based Masked Image Modeling for Self-Supervised Pretraining
- **Function**: Extracts $128^3$ patches from 1,206 CT volumes (206 annotated + 1,000 unannotated), subdivides each patch into $8^3$ sub-blocks, randomly masks 75% of sub-blocks, and trains a 3D U-Net to reconstruct the masked regions.
- **Mechanism**: Following the MAE paradigm, the reconstruction objective forces the encoder to learn meaningful anatomical structure patterns and spatial relationships without any manual annotation.
- **Design Motivation**: Medical data annotation is prohibitively expensive (only 4.4% labeled), whereas unlabeled data is abundant. Patch-level operations substantially reduce computational cost ($128^3$ vs. $512\times336\times336$), while multi-patch sampling ensures adequate anatomical coverage. After 50 epochs of training, the encoder weights are frozen and serve as a fixed feature extraction backbone for downstream tasks.

### Key Design 2: VDETR + 3D Vertex Relative Position Encoding
- **Function**: The pretrained encoder outputs $32\times21\times21\times256$ feature maps; 4,096 tokens are sampled and fed into the VDETR decoder, which computes the geometric relationship between each voxel and the 8 vertices of the predicted bounding box via 3D RPE.
- **Mechanism**: For each query $q$ and voxel position, the offset vectors to all 8 vertices of the predicted box are computed as $\Delta\mathbf{P}_i \in \mathbb{R}^{K \times N \times 3}$. After nonlinear transformation and MLP projection, a positional bias $\mathbf{R} = \sum_{i=1}^{8}\mathbf{P}_i$ is generated and added to the standard attention scores: $\mathbf{A} = \text{softmax}(\mathbf{QK}^T + \mathbf{R})$.
- **Design Motivation**: Medical organ and lesion shapes are highly irregular; a single centroid distance cannot determine whether a voxel lies inside, outside, or on the boundary of a target. The 8-corner encoding provides complete geometric containment/exclusion information, enabling the model to learn correct locality inductive biases even from limited training data.

### Key Design 3: Two-Stage Training + Mean Teacher Semi-Supervised Learning
- **Function**: Phase I (epochs 0–20) freezes the encoder and trains only the decoder; Phase II (epochs 20–100) unfreezes the encoder for joint fine-tuning (learning rate 10× lower than the decoder), while introducing Mean Teacher semi-supervised training over 2,000 additional unlabeled volumes.
- **Mechanism**: The Teacher model generates pseudo-labels using weak augmentation (Gaussian noise $\sigma=0.01$, intensity shift $\pm2\%$); the Student model is trained with strong augmentation ($\sigma=0.05$, shift $\pm10\%$, blur, elastic deformation), and a consistency loss enforces prediction agreement between the two.
- **Design Motivation**: Phase I prevents randomly initialized decoder gradients from corrupting pretrained features. The differential learning rates in Phase II (encoder $1\times10^{-5}$ vs. decoder $1\times10^{-4}$) mitigate catastrophic forgetting. Semi-supervised training is activated only at epoch 20 (with $\lambda$ linearly increasing from 0 to 0.3) to avoid training collapse caused by low-quality pseudo-labels when the decoder has not yet converged.

### Key Design 4: Multi-Label Injury Classification (Downstream Task II)
- **Function**: The frozen encoder's bottleneck features ($32\times21\times21\times256$) are passed through global average pooling → two FC layers ($256\rightarrow128\rightarrow7$) → 7 independent binary classifiers.
- **Mechanism**: Linear probe evaluation — only the 33,799-parameter classification head is trained (vs. the encoder's 5.6M parameters), directly assessing the discriminative capacity of the self-supervised representations.
- **Design Motivation**: Severe class imbalance (e.g., bowel injury has only 18% positive rate) is addressed by a weighted BCE loss $w_i^{pos} = N_i^{neg}/N_i^{pos}$, imposing heavier penalties on false negatives for rare classes.

## Loss & Training
**Total detection loss**:

$$\mathcal{L}_{total} = \mathcal{L}_{supervised} + \lambda(t) \times (\mathcal{L}_{center} + \mathcal{L}_{size} + \mathcal{L}_{cls})$$

The consistency loss comprises three components: center MSE, size MSE, and classification KL divergence (temperature $T=2.0$); $\lambda(t)$ increases linearly from 0 to 0.3 over epochs 20–60.

**Classification loss**: Weighted Binary Cross-Entropy $\mathcal{L}_{cls} = \frac{1}{7}\sum_{i=1}^{7}\mathcal{L}_{BCE}^i$, with positive sample weights such as $w_{bowel\ injury}^{pos}=4.45$.

## Key Experimental Results

### Table 1: Detection Performance Comparison (Validation Set)

| Metric | VDETR (w/o semi-sup.) | VDETR + SSL | Gain |
|--------|-----------------------|-------------|------|
| Best Epoch | 5 | 99 | — |
| mAP@0.10 | 27.27% | 56.57% | +107% |
| mAP@0.25 | 27.27% | 56.57% | +107% |
| mAP@0.50 | 26.36% | **56.57%** | **+115%** |
| mAP@0.75 | 6.82% | 45.12% | **+562%** |

**Key Findings**: Without semi-supervised learning, the model peaks at epoch 5 and then collapses catastrophically (dropping to ~8%), demonstrating that 144 annotated samples alone are entirely insufficient for stable training. The addition of semi-supervised learning yields stable convergence.

### Table 2: Detection Performance Comparison (Test Set, 32 volumes)

| Metric | VDETR (w/o semi-sup.) | VDETR + SSL | Gain |
|--------|-----------------------|-------------|------|
| mAP@0.10 | 23.03% | 45.30% | +97% |
| mAP@0.25 | 23.03% | 45.30% | +97% |
| mAP@0.50 | 23.03% | **45.30%** | **+97%** |
| mAP@0.75 | 16.67% | 28.72% | +72% |

### Table 3: Classification Ablation Study

| Method | Encoder | Test Acc | Test AUC |
|--------|---------|----------|----------|
| Fine-tune + augmentation (144 samples) | Unfrozen | 77.7% | 57.7% |
| Fine-tune + augmentation + SSL (144 samples) | Unfrozen | 75.4% | 57.3% |
| Fine-tune + augmentation + Focal Loss | Unfrozen | 75.9% | 56.0% |
| Linear probe (2,244 samples) | **Frozen** | **94.07%** | 51.4% |

**Key Findings**: Semi-supervised learning degrades classification performance (pseudo-label noise); expanding the labeled set from 144 to 2,244 samples combined with a frozen encoder linear probe achieves 94.07%, confirming that high-quality labels outweigh pseudo-labels.

### Table 4: Per-Class Classification Performance on Test Set (482 volumes)

| Injury Category | Test Acc | Test AUC |
|----------------|----------|----------|
| Bowel healthy | 97.5% | 0.577 |
| Bowel injury | 97.5% | 0.584 |
| Liver healthy | 87.6% | 0.500 |
| Liver high-grade | **98.3%** | 0.429 |
| Kidney high-grade | 96.1% | 0.470 |
| Spleen healthy | 87.1% | 0.518 |
| Extravasation | 94.4% | 0.521 |
| **Overall** | **94.07%** | 0.514 |

## Highlights & Insights
- **Systematic integration of self-supervised and semi-supervised learning**: The two-stage design is conceptually clean — MIM pretraining establishes a strong feature foundation, while Mean Teacher semi-supervised training addresses label scarcity in the detection phase. This pipeline is directly reusable for other medical detection scenarios with scarce annotations.
- **Stability improvement from semi-supervised training is the most salient contribution**: The transition from catastrophic collapse at epoch 5 to stable convergence over 100 epochs, with a 562% gain in mAP@0.75, demonstrates that consistency regularization provides regularization benefits far beyond mere performance improvement.
- **Medical adaptation of 3D RPE**: Introducing V-DETR's 8-corner position encoding into 3D medical image detection offers a fundamentally more expressive geometric description of irregular organ shapes compared to centroid-based distances.
- **Linear probe achieves 94.07% at epoch 0**: This demonstrates that self-supervised pretraining yields immediately transferable features requiring no fine-tuning.
- **Code is publicly available** and the complete pipeline is fully reproducible.

## Limitations & Future Work
- **Absolute detection performance leaves room for improvement**: A test mAP@0.50 of 45.30% remains far from clinical deployment readiness, and mAP@0.75 of only 28.72% indicates insufficient localization precision.
- **Classification AUC is very low (51.4%)**: Despite high accuracy (94.07%), the model exhibits severe probability miscalibration, with sigmoid confidence scores poorly aligned with true probabilities. The authors attribute this to calibration issues but do not address it in the paper.
- **Limited data scale**: Only 206 annotated and 1,000 unlabeled volumes are used for pretraining, which is modest by contemporary large-scale pretraining standards.
- **Semi-supervised training is ineffective or detrimental for classification**: The gain from expanding labeled data from 144 to 2,244 samples (+16.37%) far exceeds the effect of semi-supervised learning (which actually decreases performance by 2.3%), indicating poor generalization of the semi-supervised strategy to the classification task.
- **Evaluation on a single dataset (RSNA)**: Cross-dataset and cross-domain generalization are not assessed.
- **No direct comparison with other 3D medical detection methods (e.g., nnDetection)**: Head-to-head comparisons with domain-specific state-of-the-art methods are absent.
- → The framework is extensible to multi-organ detection, CT-MRI cross-modal transfer, and larger-scale pretraining data.

## Related Work & Insights
- **vs. MAE (He 2022)**: MAE targets 2D natural images; this work extends the patch-based MIM paradigm to 3D medical volumes and demonstrates that reconstruction objectives remain effective in the CT domain (PSNR 19.39 dB, linear probe 76%).
- **vs. V-DETR (2024)**: V-DETR achieves state-of-the-art results on the indoor scene dataset ScanNetV2; this work is the first to introduce 3D RPE into medical image detection. The core contribution lies not in RPE itself but in its systematic integration with self-supervised and semi-supervised learning.
- **vs. Eckstein et al. (2024) on 3D medical object detection pretraining**: That work demonstrates the importance of pretraining for 3D medical detection; this paper further integrates semi-supervised learning into the pipeline.
- **vs. Mean Teacher (Tarvainen 2017)**: The classic semi-supervised framework is adapted from 2D image classification to 3D volumetric detection, with the addition of three-branch consistency losses over center, size, and classification predictions.
- **vs. RSNA 2023 competition winning solution**: The competition winner achieved 98% AUC via a two-stage pipeline with model ensembling; this work achieves 94.07% accuracy with a single model and frozen encoder, using 29% less data at substantially lower complexity.

## Rating
- **Novelty**: ⭐⭐⭐ — Individual components (MIM, V-DETR, Mean Teacher) are not novel; the contribution lies in their systematic integration and the design of a complete pipeline for label-scarce scenarios.
- **Experimental Thoroughness**: ⭐⭐⭐ — Ablation studies cover SSL, semi-supervised learning, classification, and detection, but cross-dataset validation and direct comparisons with domain-specific SOTA are missing; the test set is small (32 volumes).
- **Writing Quality**: ⭐⭐⭐⭐ — The paper is well-structured, the design motivations for the two-stage training strategy are clearly articulated, and mathematical derivations are complete.
- **Value**: ⭐⭐⭐ — The integration paradigm of self-supervised and semi-supervised learning under label-scarce conditions is transferable; the application of 3D RPE to medical detection provides useful reference.
- **Overall Value**: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Semi-Supervised Diseased Detection from Speech Dialogues with Multi-Level Data Modeling](../../ACL2026/medical_imaging/semi-supervised_diseased_detection_from_speech_dialogues_with_multi-level_data_m.md)
- [\[CVPR 2026\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semisupervised_framework.md)
- [\[ICCV 2025\] An OpenMind for 3D Medical Vision Self-supervised Learning](../../ICCV2025/medical_imaging/an_openmind_for_3d_medical_vision_selfsupervised_learning.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization](neuroseg_meets_dinov3_transferring_2d_self-supervised_visual_priors_to_3d_neuron.md)

</div>

<!-- RELATED:END -->
