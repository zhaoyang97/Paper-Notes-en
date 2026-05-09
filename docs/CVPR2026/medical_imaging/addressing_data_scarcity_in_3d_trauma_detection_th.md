---
title: >-
  [Paper Note] Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding
description: >-
  [CVPR 2026][Medical Imaging][Abdominal trauma detection] Under extreme annotation scarcity—only 206 labeled cases (144 for training)—this work combines patch-based MIM pretraining of a 3D U-Net, a VDETR detector with 3D vertex RPE, and Mean Teacher semi-supervised consistency regularization over 2,000 unlabeled volumes. The approach improves 3D abdominal trauma detection mAP@0.50 from 26.36% to 56.57% on the validation set (+115%), while a frozen encoder with a lightweight classification head achieves 94.07% accuracy on 7-class injury classification.
tags:
  - CVPR 2026
  - Medical Imaging
  - Abdominal trauma detection
  - MIM pretraining
  - semi-supervised learning
  - VDETR
  - 3D vertex relative position encoding
date: 2026-05-08
content_hash: ee2f35b7580020bf
---

# Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding

**Conference**: CVPR 2026
**arXiv**: [2603.12514](https://arxiv.org/abs/2603.12514)
**Code**: [GitHub](https://github.com/shivasmic/3d-trauma-detection-ssl)
**Area**: Medical Imaging / 3D Object Detection / Self-Supervised Learning
**Keywords**: Abdominal trauma detection, MIM pretraining, semi-supervised learning, VDETR, 3D vertex relative position encoding

## TL;DR

Under extreme annotation scarcity—only 206 labeled cases (144 for training)—this work combines patch-based MIM pretraining of a 3D U-Net, a VDETR detector with 3D vertex RPE, and Mean Teacher semi-supervised consistency regularization over 2,000 unlabeled volumes. The approach improves 3D abdominal trauma detection mAP@0.50 from 26.36% to 56.57% on the validation set (+115%), while a frozen encoder with a lightweight classification head achieves 94.07% accuracy on 7-class injury classification.

## Background & Motivation

**Background**: Abdominal CT trauma detection is critical in emergency radiology, requiring rapid and accurate identification and localization of internal injuries. The RSNA 2023 challenge advanced the field; the winning solution achieved 98% AUC via multi-stage pipelines and ensemble strategies, but relied on large-scale annotated data.

**Limitations of Prior Work**: Annotation is prohibitively expensive—only 206 of 4,711 sequences in the RSNA dataset carry segmentation labels (4.4%). Conventional 2D slice analysis discards 3D spatial relationships; direct 3D convolution over full-resolution volumes (512×336×336) is computationally intractable; and centroid-based detection methods cannot represent the complex geometry of irregular organs and lesions.

**Key Challenge**: 3D deep learning detection demands large annotated datasets, yet medical image annotation costs are extremely high (only 4.4% labeled). General-purpose 3D pretraining (natural video or synthetic data) transfers poorly to medical imaging due to fundamentally different HU distributions and anatomical patterns.

**Goal**: To achieve reliable abdominal trauma detection and localization under extreme annotation scarcity with only hundreds of labeled 3D CT volumes.

**Key Insight**: A two-stage learning scheme—first, self-supervised pretraining on all available unlabeled CTs to acquire anatomical priors; then semi-supervised detection fine-tuning combining the small labeled set with large unlabeled data.

**Core Idea**: MIM pretraining provides anatomical priors; 3D vertex RPE models complex geometry; Mean Teacher semi-supervised learning exploits unlabeled data. The three components act synergistically to address extreme annotation scarcity.

## Method

### Overall Architecture

The framework consists of two stages. Stage 1 pretrains a 3D U-Net encoder via patch-based MIM (75% masking ratio) on 1,206 unlabeled CTs (50 epochs, MSE loss). Stage 2 connects the pretrained encoder to a VDETR decoder with 3D vertex RPE for detection, while applying Mean Teacher semi-supervised consistency regularization over 2,000 unlabeled volumes. A separate classification branch uses the frozen encoder with a lightweight head (33,799 parameters) for 7-class injury classification.

### Key Designs

1. **Patch-based MIM Self-Supervised Pretraining**:
    - **Function**: Extracts 128×128×128 patches from each CT volume, divides them into 8×8×8 sub-blocks, randomly masks 75%, and trains a 3D U-Net to reconstruct the masked regions.
    - **Mechanism**: A standard encoder-decoder U-Net architecture where the encoder progressively downsamples via 3D convolutions and pooling, and the decoder reconstructs via transposed convolutions. Trained with MSE loss for 50 epochs using Adam optimizer. Multiple patches are sampled per volume each epoch to ensure full anatomical coverage. After training, the encoder is frozen as a fixed feature extractor.
    - **Design Motivation**: The high 75% masking ratio forces the network to learn meaningful anatomical patterns and spatial relationships rather than simple interpolation. The patch-based strategy avoids the memory bottleneck of processing full-resolution volumes.

2. **VDETR + 3D Vertex Relative Position Encoding (3DV-RPE)**:
    - **Function**: Samples 4,096 tokens from the encoder's 32×21×21 feature maps (256-dim) as input to the Transformer decoder, augmenting attention with 8-corner vertex position encodings.
    - **Mechanism**: For each query $q$ and voxel position $\mathbf{p}_v$, offset vectors to all 8 vertices of the predicted bounding box are computed as $\Delta\mathbf{P}_i \in \mathbb{R}^{K \times N \times 3}$, converted via MLP to attention bias $\mathbf{R} = \sum_{i=1}^{8} \mathbf{P}_i$, and added to standard attention: $\mathbf{A} = \text{softmax}(\mathbf{QK}^T + \mathbf{R})$.
    - **Design Motivation**: Conventional center-distance metrics fail to characterize irregular organs—a voxel equidistant from a box center may lie inside, outside, or on its boundary. Eight-corner RPE provides complete geometric relationship information, enabling effective spatial inductive biases even with limited data.

3. **Two-Phase Training + Semi-Supervised Consistency Regularization**:
    - **Function**: Phase I (epochs 0–20) freezes the encoder and trains the decoder; Phase II (epochs 20–100) unfreezes the encoder for joint fine-tuning while incorporating semi-supervised learning over 2,000 unlabeled volumes.
    - **Mechanism**: Weak augmentations ($\sigma=0.01$, $\pm 2\%$) are applied to unlabeled volumes to generate teacher pseudo-labels; strong augmentations ($\sigma=0.05$, $\pm 10\%$, blur, elastic) generate student predictions. The consistency loss has three components: $\mathcal{L}_{center}$ (MSE) + $\mathcal{L}_{size}$ (MSE) + $\mathcal{L}_{cls}$ (KL, $T=2.0$). The weight $\lambda(t)$ ramps linearly from 0 to 0.3 between epochs 20 and 60.
    - **Design Motivation**: Phase I prevents randomly initialized decoder gradients from corrupting pretrained features. Phase II employs a 3-epoch encoder warmup with gradual unfreezing (lr=1e-5, one-tenth of the decoder lr) to prevent catastrophic forgetting. Semi-supervised training activates only after epoch 20 to avoid instability.

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}_{supervised} + \lambda(t) \times (\mathcal{L}_{center} + \mathcal{L}_{size} + \mathcal{L}_{cls})$$

The classification task uses weighted BCEWithLogits (positive-class weight $w_i^{pos} = N_i^{neg}/N_i^{pos}$; e.g., bowel injury $w^{pos}=4.45$). The classification head has only 33,799 trainable parameters and is trained with AdamW and cosine scheduling for 50 epochs. Data augmentation includes Gaussian noise, intensity shift/scale, and gamma correction.

## Key Experimental Results

### Main Results

| Task | Metric | w/o SSL | w/ SSL | Gain |
|------|--------|---------|--------|------|
| Val detection | mAP@0.50 | 26.36% | 56.57% | +115% |
| Val detection | mAP@0.75 | 6.82% | 45.12% | +562% |
| Test detection | mAP@0.50 | 23.03% | 45.30% | +97% |
| Test detection | mAP@0.75 | 16.67% | 28.72% | +72% |
| Classification (frozen encoder) | 7-class mean Acc | — | 94.07% | — |
| Classification (frozen encoder) | bowel AUC | — | 0.975 | — |

### Ablation Study

| Training Strategy | Encoder | Test Acc | Test AUC | Notes |
|-------------------|---------|----------|----------|-------|
| Fine-tune + augmentation | Unfrozen | 77.7% | 57.7% | 144-sample baseline |
| Fine-tune + augmentation + SSL | Unfrozen | 75.4% | 57.3% | Pseudo-label noise hurts |
| Fine-tune + augmentation + Focal | Unfrozen | 75.9% | 56.0% | Focal loss shows no notable gain |
| Linear probe (full data) | Frozen | **94.07%** | 51.4% | 2,244 samples; frozen encoder is optimal |

### Key Findings

- Purely supervised training peaks at epoch 5 then collapses catastrophically to ~8%—a typical manifestation of training instability under extreme annotation scarcity.
- Semi-supervised consistency regularization completely eliminates catastrophic collapse and achieves stable convergence.
- The 562% gain in mAP@0.75 indicates that consistency regularization improves not only detection recall but also localization precision substantially.
- The frozen encoder reaches its optimal 94.07% classification accuracy at epoch 0, with no further improvement during training—demonstrating that the self-supervised features are already maximally discriminative and constituting strong evidence of pretraining quality.
- Classification with 2,244 samples (94.07%) far outperforms 144 samples with pseudo-labels (75.4%), confirming that annotation quality is more important than pseudo-label quantity.

## Highlights & Insights

- The 115% mAP improvement and the qualitative transition from training collapse to stable convergence provide compelling evidence of SSL's value in extreme data scarcity scenarios.
- 3D vertex RPE elegantly addresses the fundamental limitation of center-distance metrics in representing irregular organs.
- Achieving 94% classification accuracy with a frozen encoder is the strongest evidence of pretraining quality—the learned features are already maximally discriminative.
- Design choices such as two-phase training, differentiated learning rates, and delayed semi-supervised activation reflect a deep understanding of training dynamics.

## Limitations & Future Work

- Validation is limited to abdominal trauma CT; generalizability to other anatomical regions or pathology types remains unknown.
- Classification AUC is only 51.4% (confidence calibration issue); temperature scaling post-processing is needed but was not implemented.
- mAP@0.75 of 28.72% indicates considerable room for improvement in localization precision under strict IoU.
- A performance gap remains relative to the RSNA 2023 winner (98% AUC, multi-stage + ensemble), though this work focuses on low-annotation methodology.
- The effect of using the full 4,505 unlabeled cases has not been explored.

## Related Work & Insights

- **vs. V-DETR**: This work is the first to apply V-DETR's 8-corner position encoding to 3D medical image detection, achieving significant gains in combination with domain-specific pretraining.
- **vs. MAE**: Masked reconstruction is extended from 2D natural images to 3D patch-based medical settings; the 75% masking ratio is consistent with the original paper.
- **vs. Eckstein et al.**: Prior work demonstrated benefits of pretraining for 3D medical detection without semi-supervision; this paper integrates both into a complete framework.
- **vs. RSNA 2023 winner**: The latter relies on large annotated data and complex ensembles; this work demonstrates that low-annotation SSL approaches can approach comparable performance.
- **Insights**: A successful case study of the "pretrain → few-shot fine-tune" paradigm for medical imaging; semi-supervised learning may contribute more critically to training stability than to accuracy gains per se.

## Rating

- **Novelty**: ⭐⭐⭐ — Individual components (MIM / VDETR / semi-supervised) are established; the contribution lies in their first systematic integration and validation in an extremely annotation-scarce 3D medical setting.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive ablations across dual tasks (detection + classification); thorough analysis of training dynamics (collapse-to-stability visualization).
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, sufficient experimental detail, open-source code, and good reproducibility.
- **Value**: ⭐⭐⭐ — Practically informative for low-annotation medical imaging scenarios; the methodological framework is transferable to other 3D detection tasks.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [\[CVPR 2026\] NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization](neuroseg_meets_dinov3_transferring_2d_self-supervised_visual_priors_to_3d_neuron.md)
- [\[CVPR 2026\] Uncertainty-Aware Concept and Motion Segmentation for Semi-Supervised Angiography Videos](uncertainty-aware_concept_and_motion_segmentation_for_semi-supervised_angiograph.md)

<!-- RELATED:END -->
