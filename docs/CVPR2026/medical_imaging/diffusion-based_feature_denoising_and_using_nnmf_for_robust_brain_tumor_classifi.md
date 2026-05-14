---
title: >-
  [Paper Note] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification
description: >-
  [CVPR 2026][Medical Imaging][Brain tumor classification] This paper proposes an NNMF+CNN+diffusion defense framework for brain tumor MRI classification. MRI images are first decomposed into compact…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Brain tumor classification"
  - "Non-negative Matrix Factorization (NNMF)"
  - "Adversarial robustness"
  - "Diffusion denoising defense"
  - "AutoAttack"
date: 2026-05-08
content_hash: a21060837aa0b759
---

# Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification

**Conference**: CVPR 2026
**arXiv**: [2603.13182](https://arxiv.org/abs/2603.13182)
**Code**: To be confirmed
**Area**: Medical Imaging
**Keywords**: Brain tumor classification, Non-negative Matrix Factorization (NNMF), Adversarial robustness, Diffusion denoising defense, AutoAttack

## TL;DR

This paper proposes an NNMF+CNN+diffusion defense framework for brain tumor MRI classification. MRI images are first decomposed into compact, interpretable low-rank features via NNMF; the most discriminative components are selected using AUC, Cohen's d, and p-value statistical criteria; a lightweight CNN then performs classification. At inference time, a feature-space purification module combining forward diffusion noise injection and a learned denoiser is introduced. Under AutoAttack ($L_\infty$, $\epsilon=0.10$), robust accuracy improves from 0.47% to 59.53%.

## Background & Motivation

**Background**: Deep learning has achieved high accuracy in brain tumor MRI classification, with CNN-based methods reaching 97%+ classification precision. However, these models operate directly in high-dimensional image space, incurring large parameter counts and limited interpretability.

**Limitations of Prior Work**: DNNs are highly sensitive to adversarial perturbations—imperceptible input modifications can cause drastic drops in classification accuracy. In safety-critical applications such as medical diagnosis, this fragility is unacceptable. Existing adversarial defenses (e.g., adversarial training, input transformations) typically operate in the raw image space, imposing high computational cost and potentially sacrificing clean accuracy.

**Key Challenge**: Adversarial perturbations in high-dimensional image space are difficult to filter effectively due to the large attack surface. Dimensionality-reduced representations such as PCA reduce dimensions but lack interpretability and non-negativity constraints, making them unsuitable for medical images whose pixel values are naturally non-negative.

**Goal**: (1) How to obtain compact, interpretable, and medically appropriate feature representations? (2) How to implement effective adversarial defense in feature space rather than image space? (3) How to substantially improve robust accuracy without significant loss of clean accuracy?

**Key Insight**: NNMF is naturally suited for parts-based decomposition of non-negative data, compressing MRI images into a small number of interpretable basis components. Applying diffusion-denoising defense in a low-dimensional feature space is more efficient than in the original image space—forward diffusion injects structured noise to overwhelm adversarial perturbations, and a learned denoiser then recovers clean features.

**Core Idea**: NNMF compresses MRI images into interpretable low-rank features, within which a diffusion noise-injection and denoising mechanism provides lightweight yet effective adversarial defense.

## Method

### Overall Architecture

The framework consists of four stages: (1) data preprocessing → (2) NNMF feature extraction and statistical selection → (3) lightweight CNN classifier training → (4) feature-space diffusion defense. The inference pipeline is: input image → NNMF projection → L2 normalization → forward diffusion noise injection → denoiser recovery → CNN classification.

### Key Designs

#### 1. NNMF Feature Extraction

- **Function**: Decompose MRI images into a small number of non-negative basis components and their coefficients.
- **Mechanism**: Each image is converted to grayscale, resized to $128 \times 128$, normalized to $[0,1]$, and flattened into a column vector, forming a non-negative data matrix $V \in \mathbb{R}_+^{K \times N}$. A KL divergence objective with multiplicative update rules decomposes this into $V \approx WH$, where $W \in \mathbb{R}_+^{K \times R}$ contains $R=15$ basis components and $H \in \mathbb{R}_+^{R \times N}$ is the coefficient matrix. The update rules are: $W \leftarrow W \otimes \frac{(V ./ (WH)) H^T}{\mathbf{1} H^T}$, $H \leftarrow H \otimes \frac{W^T (V ./ (WH))}{W^T \mathbf{1}}$
- **Design Motivation**: The non-negativity constraint of NNMF yields a parts-based representation—each basis component corresponds to meaningful anatomical structures such as skull boundaries and tissue distributions—offering greater interpretability than PCA. Validation and test sets are projected via non-negative least squares with fixed basis $W$ to ensure consistency.

#### 2. Statistical Feature Selection

- **Function**: Select the Top-M most discriminative components from the 15 NNMF components.
- **Mechanism**: Three metrics are computed for each component: (a) **ROC-AUC**—measures the ability of a single component to distinguish tumor from normal; (b) **Cohen's d**—effect size reflecting the separation between class distributions; (c) **Welch's t-test p-value**—statistical significance test. Components are ranked by the combination of these three metrics, and the Top-M features are selected; feature vectors are then L2-normalized.
- **Design Motivation**: Not all NNMF components are discriminative; redundant components introduce noise and enlarge the attack surface. Multi-criterion evaluation ensures that selected features exhibit both strong class separation and statistical reliability.

#### 3. Lightweight CNN Classifier

- **Function**: Train a small CNN on the selected NNMF features for binary classification (tumor vs. normal).
- **Mechanism**: Input is the L2-normalized Top-M dimensional NNMF feature vector, which passes through convolutional layers, max pooling, and fully connected layers. The model is optimized on the training set with the validation set monitored to prevent overfitting.
- **Design Motivation**: Training directly on low-dimensional NNMF features avoids the computational overhead of high-dimensional image space, while CNNs capture nonlinear relationships among features. Clean test accuracy of approximately 85.1% confirms that NNMF features retain sufficient discriminative information.

#### 4. Feature-Space Diffusion Defense

- **Function**: At inference time, apply forward diffusion noise to the NNMF feature vector and recover clean features with a learned denoiser.
- **Mechanism**:

  **Forward diffusion**: A linear noise schedule is defined; Gaussian noise is injected into clean features $x_0$ at a randomly sampled time step $t$ to obtain $x_t$. Higher $t$ values correspond to stronger noise injection; the paper uses $t=41$ steps.

  **Denoiser training**: A regression network is trained, taking as input the noisy features $x_t$ and a time step embedding (sinusoidal encoding), and outputting the recovered clean features $\hat{x}_0$. Training uses mean squared error (MSE) loss.

  **Inference defense**: Input features are first subjected to forward diffusion up to step $t$, then recovered by the denoiser before being passed to the classifier. Due to the stochastic nature of diffusion, Expectation over Transformation (EOT) is applied, averaging predictions over $K=8$ random samples.

- **Design Motivation**: Adversarial perturbations are overwhelmed by structured Gaussian noise during forward diffusion; the denoiser learns a mapping that recovers the clean data distribution rather than preserving adversarial perturbations. Operating in a low-dimensional feature space rather than high-dimensional image space renders the diffusion-denoising process highly lightweight.

### Loss & Training

- **NNMF optimization**: KL divergence + multiplicative update rules, rank $R=15$.
- **CNN classifier**: Cross-entropy loss; training-validation split used for monitoring.
- **Denoiser**: MSE loss $\|\hat{x}_0 - x_0\|_2^2$; sinusoidal time step encoding.
- **Adversarial evaluation**: AutoAttack ($L_\infty$, $\epsilon=0.10$), including APGD-CE and Square Attack; defense side uses EOT ($K=8$).
- **Overall pipeline**: NNMF + CNN + diffusion training implemented in MATLAB; models exported to ONNX format and evaluated under AutoAttack in PyTorch.

## Key Experimental Results

### Main Results

Comprehensive performance comparison under clean and AutoAttack conditions:

| Model | Accuracy | Precision | Recall | F1 | MCC | ROC-AUC | Brier Score↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Clean Baseline | 0.861 | 0.855 | 0.898 | 0.876 | 0.718 | 0.911 | 0.146 |
| Clean Defended | 0.851 | 0.853 | 0.881 | 0.867 | 0.699 | 0.897 | 0.156 |
| Robust Baseline | **0.005** | 0.000 | 0.000 | 0.000 | -0.991 | 0.008 | 0.470 |
| Robust Defended | **0.595** | 0.612 | 0.720 | 0.662 | 0.170 | 0.749 | 0.215 |

Key comparison: The baseline nearly completely collapses under AutoAttack (accuracy 0.47%, MCC≈−1), whereas the diffusion defense restores robust accuracy to 59.53%. Clean accuracy decreases only marginally from 86.05% to 85.12% (loss <1%).

### Ablation Study

| Configuration | Clean Acc | Robust Acc | Notes |
|------|:---:|:---:|------|
| No NNMF (raw image + CNN) | High | Extremely low | Large attack surface in high-dimensional space |
| NNMF + CNN (no diffusion defense) | 0.861 | 0.005 | NNMF dimensionality reduction alone does not provide robustness |
| NNMF + CNN + diffusion defense | 0.851 | **0.595** | Diffusion defense is the core source of robustness |

Validation set CNN accuracy is approximately 83% and test set accuracy approximately 85.1%; the small validation-test gap indicates no overfitting.

### Key Findings

1. **Diffusion defense is the core driver of robustness improvement**: Without defense, AutoAttack reduces accuracy to 0.47% (MCC≈−1, equivalent to fully inverted predictions); with diffusion defense, accuracy recovers to 59.53%.
2. **Minimal clean accuracy loss**: The defense module introduces less than 1% degradation in clean accuracy, confirming that the denoiser successfully preserves discriminative information.
3. **NNMF rank=15 suffices to retain discriminative information**: Only 15 basis components support 85%+ clean accuracy, indicating that the effective information dimensionality of brain tumor MRI classification is low.
4. **EOT averaging enhances defense stability**: Averaging over multiple stochastic diffusion samples yields more robust predictions.
5. **Probability calibration metrics also improve significantly**: After defense, Brier Score decreases from 0.470 to 0.215 and Log-Loss from 1.163 to 0.618, indicating that the defense not only improves classification accuracy but also enhances the reliability of predicted probabilities.

## Highlights & Insights

1. **Diffusion defense in feature space rather than image space**: Performing diffusion-denoising in the low-dimensional NNMF-compressed feature space requires far less computation than image-space defense, and adversarial perturbations are more readily overwhelmed by noise in low-dimensional space. This approach is transferable to any pipeline that applies dimensionality reduction prior to classification.
2. **Dual role of NNMF**: It serves simultaneously as an interpretable feature extractor—each basis component corresponding to meaningful anatomical structures—and as a pre-defense dimensionality reducer that shrinks the dimensions manipulable by an adversary, inherently increasing attack difficulty.
3. **Statistically driven feature selection**: Using three criteria—AUC, Cohen's d, and p-value—to screen components is more principled than selection based solely on variance or reconstruction error, ensuring that selected features are both discriminative and statistically significant.
4. **Engineering design of a MATLAB–Python hybrid pipeline**: Leveraging MATLAB's matrix factorization capabilities for NNMF and PyTorch for adversarial evaluation, bridged via ONNX—this pragmatic engineering strategy merits reference.

## Limitations & Future Work

1. **Small dataset and lack of patient-level splitting**: The dataset contains only approximately 2,200 MRI slices with no patient ID information; slice-level leakage between training and test sets from the same patient may exist, potentially inflating reported performance.
2. **Binary classification only (tumor vs. normal)**: The method is not extended to multi-class brain tumor typing (e.g., glioma vs. meningioma vs. pituitary tumor), which better reflects real clinical needs.
3. **Modest clean accuracy**: A classification accuracy of 85% is not state-of-the-art in medical imaging, possibly due to an overly narrow information bottleneck at NNMF rank=15; higher or adaptive rank selection may improve performance.
4. **Fixed diffusion step count as a hyperparameter**: The choice of $t=41$ steps lacks systematic ablation; different attack strengths may require different noise injection magnitudes.
5. **Only $L_\infty$ attacks evaluated**: $L_2$, $L_1$, and more advanced attacks (e.g., C&W attack) are not considered, leaving the generalization robustness of the defense insufficiently validated.
6. **Simple denoiser architecture**: A basic regression network is used without exploring stronger denoising architectures (e.g., U-Net style or Transformer-based), which may limit recovery quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation](pgr-net_prior-guided_roi_reasoning_network_for_brain_tumor_mri_segmentation.md)
- [\[CVPR 2026\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)
- [\[CVPR 2026\] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)
- [\[ICLR 2026\] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](../../ICLR2026/medical_imaging/compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)
- [\[CVPR 2026\] RelativeFlow: Taming Medical Image Denoising Learning with Noisy Reference](relativeflow_taming_medical_image_denoising_learning_with_noisy_reference.md)

</div>

<!-- RELATED:END -->
