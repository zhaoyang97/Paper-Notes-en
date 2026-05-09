---
title: >-
  [Paper Note] Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning
description: >-
  [CVPR 2026][Medical Imaging][Fairness] A fairness-aware framework based on attention MIL and gradient reversal layers (GRL) is proposed for multi-class lung disease diagnosis from chest CT volumes, eliminating gender bias while preserving diagnostic accuracy.
tags:
  - CVPR 2026
  - Medical Imaging
  - Fairness
  - Lung Disease Diagnosis
  - CT Classification
  - Multiple Instance Learning
  - Adversarial Training
date: 2026-05-08
content_hash: 6ae4b2577106b708
---

# Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning

**Conference**: CVPR 2026
**arXiv**: [2603.12988](https://arxiv.org/abs/2603.12988)
**Code**: [GitHub](https://github.com/ADE-17/cvpr-fair-chest-ct)
**Area**: Medical Imaging
**Keywords**: Fairness, Lung Disease Diagnosis, CT Classification, Multiple Instance Learning, Adversarial Training

## TL;DR

A fairness-aware framework based on attention MIL and gradient reversal layers (GRL) is proposed for multi-class lung disease diagnosis from chest CT volumes, eliminating gender bias while preserving diagnostic accuracy.

## Background & Motivation

Automated deep learning analysis of chest CT carries significant clinical value for lung cancer screening and COVID-19 detection; however, models may encode and amplify demographic disparities present in training data, leading to systematic unfairness toward minority groups. The Fair Disease Diagnosis Challenge held at the CVPR 2026 PHAROS-AIF-MIH Workshop requires classifying CT scans into four categories (Healthy, COVID-19, Adenocarcinoma, Squamous Cell Carcinoma), evaluated by the gender-stratified macro F1 mean:

$$P = \frac{1}{2}(\text{MacroF1}_{\text{male}} + \text{MacroF1}_{\text{female}})$$

This metric explicitly penalizes gender-unequal predictions. The paper addresses three core challenges: (1) sparse pathological signals in CT volumes (only a minority of 200+ slices contain lesions); (2) severe demographic imbalance (only 18 female vs. 91 male squamous cell carcinoma cases); and (3) gender potentially serving as a latent shortcut feature exploited by the model.

## Method

### Overall Architecture

An attention MIL model built on a ConvNeXt backbone, augmented with a GRL adversarial branch to enforce gender fairness. The pipeline comprises four stages: slice feature extraction, attention pooling, disease classification, and adversarial gender classification.

### Key Designs

1. **Attention MIL Pooling**: Each CT volume is treated as a bag of slices. A ConvNeXt-Base encoder extracts a $D$-dimensional embedding $h_i = f_{\text{enc}}(x_i)$ per slice; a two-layer MLP attention network then assigns importance weights $w_i = \frac{\exp(s_i)}{\sum_j \exp(s_j)}$ to each slice, and a weighted aggregation yields the scan-level representation $H = \sum_i w_i h_i$. Zero-padded positions are masked via a binary mask. The core motivation is to enable the model to automatically learn the importance of diagnostically relevant slices without slice-level annotations.

2. **Gradient Reversal Layer (GRL) Adversarial Debiasing**: A gender classifier is appended to the scan embedding $H$, with the GRL reversing and scaling gradients during backpropagation: $z_{\text{gen}} = c(\mathcal{R}_\lambda(H))$. The training objective is $\mathcal{L} = \mathcal{L}_{\text{disease}} + \lambda_{\text{adv}} \mathcal{L}_{\text{gender}}$. The design motivation is that even without explicit gender inputs, the backbone may encode gender information from CT acquisition parameters and body morphology as spurious correlations.

3. **Subgroup Oversampling and Stratified CV**: A `WeightedRandomSampler` substantially increases the sampling weight of female squamous cell carcinoma cases (Female G), ensuring the rarest subgroup is represented in every batch. Five-fold cross-validation is stratified on the joint (class, gender) key, guaranteeing representation of all eight subgroups in each fold.

### Loss & Training

- **Focal Loss + Label Smoothing**: $\mathcal{L}_{\text{disease}} = -\alpha(1-p_t)^\gamma \log \tilde{p}_t$, with $\gamma=2$, $\alpha=0.25$, smoothing $\varepsilon=0.1$, concentrating gradients on hard samples and scarce subgroups.
- **Two-Stage Fine-Tuning**: Epochs 1–5 freeze the backbone (LR $10^{-3}$); from Epoch 6 onward the backbone is unfrozen (backbone LR $10^{-5}$, heads LR $10^{-4}$) with cosine annealing.
- **Gradient Accumulation**: $K=4$ steps, yielding an effective batch size of 16 volumes.
- At most $M=32$ slices per volume; random sampling during training and uniform sampling during inference.

## Key Experimental Results

### Dataset

889 3D chest CT scans (734 train / 155 validation), four classes: Adenocarcinoma (300), COVID-19 (240), Healthy (240), Squamous Cell Carcinoma (109). The overall gender distribution is relatively balanced (481 male / 408 female), but only 18 female vs. 91 male SCC cases; volume depth varies widely (20–800+ slices per scan).

### Main Results

| Dataset | Metric | Ours | Best Single Fold | Notes |
|---------|--------|------|------------------|-------|
| 889 CT scans (734 train) | Competition Score P | 0.685 ± 0.030 | 0.759 (Fold 1) | 5-fold mean |
| Same | Male macro-F1 | 0.679 ± 0.068 | 0.754 | Gender gap reduced after GRL |
| Same | Female macro-F1 | 0.691 ± 0.030 | 0.722 | Female F1 slightly higher than male |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Mean Pooling (Baseline) | Low | Tumor signal diluted by healthy slices |
| + Max Pooling | Improved | Restores detection of sparse tumor slices |
| + Attention-MIL | Further improved | Learns to ignore blank lung regions |
| + Subgroup Oversampling | Significant gain in Female F1 | Prevents minority class collapse |
| + GRL | P = 0.685 | Closes fairness gap; male and female performance equalized |

### Key Findings

- GRL successfully narrows the gender fairness gap: female macro-F1 (0.691) slightly exceeds male (0.679).
- Squamous cell carcinoma (SCC) remains the most challenging class (F1 = 0.366 ± 0.083), limited by data scarcity and clinical overlap.
- OOF threshold optimization maintains strict gender fairness (Male F1 0.679 vs. Female F1 0.688).

## Highlights & Insights

- The fairness problem is decomposed into three distinct failure modes (sparse signals, demographic imbalance, latent shortcut features), each addressed by a dedicated module.
- GRL adversarial training offers an elegant solution for eliminating gender bias in feature space, more thorough than simple data balancing.
- OOF threshold optimization avoids overfitting to the validation set and serves as a practical post-hoc fairness calibration technique.

## Limitations & Future Work

- Female SCC data remains extremely scarce (only 18 cases); oversampling cannot fully compensate.
- Generative data augmentation (e.g., diffusion-based CT synthesis for scarce subgroups) is unexplored.
- The fairness constraint could be extended to a stronger fairness-constrained optimization formulation.
- Attention visualization and clinical interpretability are not discussed in depth.
- The 5-fold ensemble at inference incurs high computational cost, requiring each test sample to pass through all five models.

## Related Work & Insights

- The GRL domain adaptation method of Ganin & Lempitsky (2015) is elegantly transferred to the fairness setting.
- The Attention-MIL framework of Ilse et al. (2018) is well-suited to handling sparse signals in CT volumes.
- The combination of Focal Loss (Lin et al., 2017) and Label Smoothing is friendly to scarce subgroups.
- The inference-time combination of 5-fold ensemble, TTA, and OOF threshold optimization yields robust challenge performance.

## Rating

- **Novelty**: ⭐⭐⭐ All components are established methods; however, their integration for fair diagnostic classification is valuable.
- **Experimental Thoroughness**: ⭐⭐⭐ 5-fold CV and ablation are reasonably complete, though the dataset is small and ablations are primarily qualitative.
- **Writing Quality**: ⭐⭐⭐⭐ Problem decomposition is clear, motivation is well-articulated, and the challenge paper format is well-structured.
- **Value**: ⭐⭐⭐ A practical framework for fairness in medical AI, though it primarily constitutes a challenge solution.

## Additional Notes

This paper presents a competition entry for the CVPR 2026 PHAROS-AIF-MIH Workshop Challenge, achieving a robust competition score of 0.685. The overall methodology is engineering-oriented; nonetheless, the multi-level fairness constraint design — oversampling at the data level, GRL at the feature level, and OOF thresholding at the decision level — provides a reusable toolkit for fairness in medical AI.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Robust Fair Disease Diagnosis in CT Images](robust_fair_disease_diagnosis_in_ct_images.md)
- [\[CVPR 2026\] MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](milpf_multiple_instance_learning_on_precomputed_fe.md)
- [\[CVPR 2026\] Every Error has Its Magnitude: Asymmetric Mistake Severity Training for Multiclass Multiple Instance Learning](every_error_has_its_magnitude_asymmetric_mistake_severity_training_for_multiclas.md)
- [\[CVPR 2026\] EMAD: Evidence-Centric Grounded Multimodal Diagnosis for Alzheimer's Disease](emad_evidence-centric_grounded_multimodal_diagnosis_for_alzheimers_disease.md)
- [\[CVPR 2026\] Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images](cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_.md)

<!-- RELATED:END -->
