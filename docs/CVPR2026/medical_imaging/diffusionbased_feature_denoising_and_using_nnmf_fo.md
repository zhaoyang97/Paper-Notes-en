---
title: >-
  [Paper Note] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification
description: >-
  [CVPR 2026][Medical Imaging][Brain Tumor Classification] A four-stage pipeline is proposed consisting of NNMF feature extraction → statistical feature selection → lightweight CNN classification → feature-space diffusion purification. The method maintains 85.1% clean accuracy while substantially improving robust accuracy under AutoAttack ($L_\infty$, $\epsilon=0.10$) from a baseline of 0.47% to 59.5%.
tags:
  - CVPR 2026
  - Medical Imaging
  - Brain Tumor Classification
  - NNMF
  - Diffusion Defense
  - AutoAttack
  - Feature-Space Denoising
date: 2026-05-08
content_hash: addab9d09f626a9d
---

# Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification

**Conference**: CVPR 2026
**arXiv**: [2603.13182](https://arxiv.org/abs/2603.13182)
**Code**: None
**Area**: Medical Image Classification / Adversarial Robustness
**Keywords**: Brain Tumor Classification, NNMF, Diffusion Defense, AutoAttack, Feature-Space Denoising

## TL;DR

A four-stage pipeline is proposed consisting of NNMF feature extraction → statistical feature selection → lightweight CNN classification → feature-space diffusion purification. The method maintains 85.1% clean accuracy while substantially improving robust accuracy under AutoAttack ($L_\infty$, $\epsilon=0.10$) from a baseline of 0.47% to 59.5%.

## Background & Motivation

**State of the Field**: Deep learning has achieved high accuracy in brain tumor MRI classification (CNN-based methods exceeding 97%), yet models remain extremely vulnerable to adversarial perturbations—imperceptible input modifications that can completely destroy classification performance. AutoAttack has become a standard tool for evaluating robustness as a unified attack benchmark.

**Limitations of Prior Work**: Most adversarial defense research focuses on the image space (pixel-level denoising or adversarial training), which incurs high computational cost and may degrade clean accuracy. NNMF provides interpretable parts-based non-negative representations naturally suited for non-negative data such as MRI, yet its potential in combination with diffusion-based defenses remains unexplored.

**Root Cause**: Clinical AI must simultaneously achieve high accuracy and high robustness, but these objectives typically trade off against each other. Existing end-to-end CNNs nearly completely collapse under AutoAttack (accuracy dropping to near 0%), motivating the need for a new defense paradigm.

**Paper Goals**: To provide substantial robustness against strong attacks such as AutoAttack for brain tumor MRI classification without significantly sacrificing clean accuracy.

**Starting Point**: Shifting the defense from pixel space to feature space—first extracting compact, interpretable features via NNMF, then performing diffusion forward noising and learned denoising purification in that feature space, leveraging both dimensionality reduction and diffusion purification to eliminate adversarial perturbations.

**Core Idea**: After NNMF dimensionality reduction to an interpretable low-rank feature space, diffusion purification is applied within that space to neutralize adversarial perturbations.

## Method

### Overall Architecture

A four-stage pipeline: (1) MRI preprocessing → NNMF decomposition to extract basis components (rank=15); (2) multi-criterion statistical selection of Top-M features via AUC / Cohen's d / p-value; (3) training a lightweight CNN classifier on the selected features; (4) at inference, applying forward diffusion noising → learned denoiser reconstruction → purified features fed to the classifier.

### Key Designs

1. **NNMF Feature Extraction and Statistical Selection**:
    - Function: MRI images are converted to grayscale, resized to 128×128, normalized, and vectorized to form a non-negative matrix $V \in \mathbb{R}^{K \times N}_+$, which is decomposed as $V \approx WH$ (rank=15).
    - Mechanism: A KL divergence objective with multiplicative update rules is used to optimize the decomposition. The basis matrix $W$ is learned on the training set; validation/test sets obtain feature vectors by projecting onto the fixed $W$ via non-negative least squares, followed by L2 normalization for consistent feature scaling. Each of the 15 components is then evaluated on three complementary statistical criteria: AUC (discriminative ability), Cohen's d (effect size), and Welch's t-test p-value (statistical significance), from which the Top-M features by aggregate ranking are selected.
    - Design Motivation: The non-negativity constraint in NNMF yields parts-based interpretable representations—each basis component corresponds to a recognizable anatomical pattern (e.g., skull boundary, tissue distribution). Multi-criterion statistical selection jointly considers discriminative power, effect size, and statistical reliability.

2. **Feature-Space Diffusion Purification**:
    - Function: A defense pipeline of forward diffusion noising followed by learned denoiser reconstruction is executed in NNMF feature space rather than pixel space.
    - Mechanism: A linear noise schedule is defined to progressively add Gaussian noise to clean features $x_0$, generating $x_t$. A regression-based denoising network is trained with inputs consisting of the noisy feature $x_t$ concatenated with sinusoidal positional encodings of timestep $t$, and outputs the denoised estimate $\hat{x}_0$, supervised with MSE loss. At inference, a selected timestep (e.g., $t=41$) is used to add noise, after which the denoiser restores the features. To stabilize the defense against the stochasticity of the noising process, Expectation over Transformation (EOT, K=8 samples averaged) is applied.
    - Design Motivation: Adversarial perturbations primarily act in pixel space; after NNMF dimensionality reduction, perturbations are compressed into a low-rank space. Applying diffusion purification in this space further eliminates residual perturbation effects at substantially lower computational cost than pixel-space denoising.

### Loss & Training

- NNMF optimization: KL divergence $C(V|WH)$ with multiplicative update rules for iterative optimization of $W$ and $H$.
- CNN classifier training: Standard cross-entropy loss on L2-normalized NNMF features.
- Denoiser training: MSE loss $\|\hat{x}_0 - x_0\|^2$ on (noisy feature, clean feature) pairs.
- Robustness evaluation: AutoAttack ($L_\infty$, $\epsilon=0.10$) comprising APGD-CE and Square Attack components.
- Implementation: MATLAB (NNMF/CNN/diffusion) + Python (PyTorch/AutoAttack), with ONNX format used to bridge the models.

## Key Experimental Results

### Main Results

Dataset: Kaggle Brain Tumor MRI, approximately 2,200 images, binary classification (normal vs. tumor), 70/20/10 split.

| Model Configuration | Accuracy | ROC-AUC | MCC | Brier Score↓ | Log-Loss↓ |
|---|---|---|---|---|---|
| Clean Baseline | 86.05% | 0.9105 | 0.7178 | 0.1461 | 0.4751 |
| Clean Defended | 85.12% | 0.8967 | 0.6988 | 0.1555 | 0.4963 |
| Robust Baseline (AA) | 0.47% | 0.0075 | -0.9906 | 0.4702 | 1.1629 |
| Robust Defended (AA) | **59.53%** | **0.7485** | 0.1703 | 0.2150 | 0.6182 |

### Ablation Study

| Analysis Dimension | Key Metric | Description |
|---|---|---|
| NNMF basis component visualization | 15 basis images at rank=15 | Each component captures complementary anatomical patterns such as skull boundary, tissue distribution, and local density |
| Class-mean activation heatmap | Tumor class shows systematically higher activation on specific components | Confirms the class-discriminative nature of NNMF features |
| Denoising reconstruction quality | $\|\hat{x}_0-x_0\| < \|x_t-x_0\|$ | The denoiser effectively reduces reconstruction error; all points fall below the identity line |
| Clean accuracy loss | 86.05%→85.12% (−0.93 pp) | Diffusion purification has negligible impact on clean data accuracy |
| Probability calibration improvement | Brier 0.4702→0.2150 | The defense substantially improves probability calibration quality under adversarial conditions |
| Computational efficiency | GPU 116.6s vs. CPU 201.5s | 1.73× speedup; overall overhead is acceptable |

### Key Findings

- Without defense, baseline accuracy collapses from 86.05% to 0.47% under AutoAttack, with MCC dropping to −0.99 (fully inverted predictions), confirming the severity of adversarial vulnerability.
- The diffusion defense restores robust accuracy to 59.53% while clean accuracy drops by only 0.93 percentage points—an extremely favorable accuracy–robustness trade-off.
- Brier Score decreases from 0.4702 to 0.2150, indicating that the defense not only recovers classification performance but also substantially improves probability calibration.
- The parts-based representation from NNMF provides interpretability unavailable with pixel-level methods—each component visually corresponds to specific anatomical structures.

## Highlights & Insights

- Performing diffusion defense in feature space rather than pixel space is a novel perspective—it reduces computational cost and decouples the defense from the downstream classifier.
- The parts-based representation of NNMF inherently provides interpretability, with each visualized component corresponding to a concrete anatomical structure.
- The accuracy–robustness trade-off is highly favorable: less than 1% clean accuracy loss yields a dramatic improvement in robust accuracy from 0.5% to 59.5%.
- Multi-dimensional evaluation (Accuracy / AUC / MCC / Brier / LogLoss) is more comprehensive and reliable than reporting accuracy alone.

## Limitations & Future Work

- The dataset comprises only approximately 2,200 images in a simple binary classification setting; the scale is too small. Patient-wise splitting is not confirmed, raising the risk of slice-level data leakage.
- The choice of NNMF rank=15 lacks systematic ablation; it is unclear whether this is optimal.
- Evaluation is conducted at a single attack strength $\epsilon=0.10$ only; robustness curves across varying $\epsilon$ values are not explored.
- The selection of diffusion timestep $t=41$ appears ad hoc, with no systematic analysis of the timestep–robustness/accuracy trade-off.
- The MATLAB+Python hybrid pipeline hinders practicality and reproducibility; an end-to-end PyTorch implementation would be preferable.
- Writing quality is below standard, with numerous grammatical and expression issues (likely a non-native English manuscript), affecting the overall rigor of the paper.

## Related Work & Insights

- **vs. End-to-end CNN Classification (Hossain et al.)**: The latter achieves 97.87% accuracy with a 5-layer CNN but provides no robustness guarantee; the proposed approach strategically sacrifices a small amount of accuracy in exchange for substantial adversarial defense.
- **vs. Classification-Denoising Networks (Thiry & Guth)**: The latter jointly learns classification and denoising objectives; the proposed approach adopts a modular, decoupled design in which each component can be independently replaced or improved.
- **vs. NMF-CNN (Chan et al.)**: The prior work applies NMF to enhance CNN in acoustic event detection; this paper is the first to combine NMF, CNN, and diffusion defense for adversarial robustness in medical imaging.
- **vs. AutoAttack (Croce & Hein)**: The standard AA evaluation protocol (APGD-CE + Square) is strictly followed, avoiding spurious robustness claims that arise from using only weak attacks.
- **Insights**: The feature-space defense paradigm is transferable to other medical image classification tasks; the interpretable intermediate representation provided by NNMF can contribute to more transparent clinical AI systems.

## Rating

- Novelty: ⭐⭐⭐ The combination of NNMF and diffusion defense in feature space is creative, though each individual component is not original.
- Experimental Thoroughness: ⭐⭐⭐ Evaluation metrics are comprehensive (6 metrics), but the dataset is too small and systematic ablations across multiple attack strengths are absent.
- Writing Quality: ⭐⭐ Numerous grammatical and expression issues; some passages have poor readability, undermining the paper's credibility.
- Value: ⭐⭐⭐ The feature-space defense paradigm is a meaningful reference, but the limited experimental scale and rigor constrain the reliability of the conclusions.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation](pgr-net_prior-guided_roi_reasoning_network_for_brain_tumor_mri_segmentation.md)
- [\[CVPR 2026\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)
- [\[CVPR 2026\] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modalityspecific_encoders_and_partially.md)
- [\[CVPR 2026\] RelativeFlow: Taming Medical Image Denoising Learning with Noisy Reference](relativeflow_taming_medical_image_denoising_learning_with_noisy_reference.md)
- [\[CVPR 2026\] CRFT: Consistent-Recurrent Feature Flow Transformer for Cross-Modal Image Registration](crft_consistent-recurrent_feature_flow_transformer_for_cross-modal_image_registr.md)

<!-- RELATED:END -->
