---
title: >-
  [Paper Note] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning
description: >-
  [CVPR2025][Medical Imaging][glioblastoma] This work proposes RICE-NET, a multimodal 3D deep learning model that integrates longitudinal MRI data with radiotherapy radiation dose (RD) maps to distinguish between post-operative radiation-induced contrast enhancement (RICE) and tumor recurrence in glioblastoma, achieving an F1-score of $0.92$ on an independent test set.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "glioblastoma"
  - "radiation-induced contrast enhancement"
  - "tumor recurrence"
  - "multimodal classification"
  - "3D ResNet"
date: 2026-05-08
content_hash: 73640aae4db5801c
---

# Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning

**Conference**: CVPR2025  
**arXiv**: [2603.11827](https://arxiv.org/abs/2603.11827)  
**Code**: To be confirmed  
**Area**: Medical Imaging  
**Keywords**: glioblastoma, radiation-induced contrast enhancement, tumor recurrence, multimodal classification, 3D ResNet

## TL;DR

This work proposes RICE-NET, a multimodal 3D deep learning model that integrates longitudinal MRI data with radiotherapy radiation dose (RD) maps to distinguish between post-operative radiation-induced contrast enhancement (RICE) and tumor recurrence in glioblastoma, achieving an F1-score of $0.92$ on an independent test set.

## Background & Motivation

- Glioblastoma (GBM) patients require radiotherapy after surgical resection to eliminate residual tumor cells, but radiotherapy can also damage normal brain tissue.
- New contrast-enhancing lesions appearing in post-operative follow-up images present a major diagnostic challenge: distinguishing between tumor recurrence and radiation-induced contrast enhancement (RICE), both of which appear highly similar on MRI.
- Current clinical workflows rely on complex and time-consuming evaluations by multidisciplinary tumor boards, requiring reviews of pre- and post-operative scans, multiple follow-up images, and radiotherapy plans.
- Existing methods heavily rely on clinically scarce diffusion MRI or neglect radiotherapy dose maps, despite the latter receiving increasing attention in tumor boards.
- Most prior studies overlook the longitudinal evolutionary information of the images.

## Method

### Data and Preprocessing
- Data source: 92 GBM patients from Heidelberg University Hospital, with a training/validation set of 80 patients (48 tumor recurrence + 32 RICE) and an independent test set of 12 patients (7 recurrence + 5 RICE).
- Three 3D input volumes per patient:
  1. **Post-operative MRI (MRI post-OP)**: T1-weighted contrast-enhanced MRI at post-op baseline, used for radiotherapy planning.
  2. **Event MRI (MRI event)**: T1-weighted contrast-enhanced MRI when the new contrast-enhancing lesion is detected.
  3. **Radiotherapy Dose Map (RD map)**: 3D spatial distribution of the cumulative radiation dose.
- Preprocessing workflow: Isotropic resampling $\rightarrow$ ANTs registration $\rightarrow$ HD-BET skull stripping $\rightarrow$ Z-score normalization $\rightarrow$ cropping to $224 \times 224 \times 224$ voxels.
- The ground truth was confirmed by biopsy results.

### Network Architecture
- A 3D ResNet-18 implemented based on the MONAI framework, extending the original 2D ResNet to three dimensions to process volumetric data.
- Architecture: Initial 3D convolutional layer $\rightarrow$ 4 residual blocks (3D BatchNorm + ReLU) $\rightarrow$ global average pooling $\rightarrow$ fully connected classification layer.
- Residual connections ensure more stable gradient flow and convergence, which is particularly crucial for small medical datasets.
- Multimodal fusion strategy: Channel-wise concatenation, where multiple $224 \times 224 \times 224$ volumes are stacked along the first dimension.
- Separate and independent models were trained for different modal combination experiments (instead of using shared weights with selective inputs).
- ResNet-18 was selected instead of deeper networks to balance expressiveness and computational efficiency, mitigating overfitting risks in the small-sample scenario of 92 patients.

### Loss & Training
- Training lasted for 800 epochs with 5-fold cross-validation (folds were fixed at the patient level and remained consistent across all experiments).
- Adam optimizer + cross-entropy loss function.
- A weighted random sampler was employed to ensure balanced training over the two classes (addressing the imbalance of 48 recurrence vs. 32 RICE cases).
- The evaluation metric chosen was the macro F1-score: taking the unweighted average of the F1-scores of both classes, which is more robust under class imbalance.
- Data augmentation: elastic deformation, rotation, scaling, Gaussian noise, brightness, and gamma adjustments.
- Evaluation of the test set utilized a majority voting ensemble strategy of the 5-fold cross-validated models.

### Interpretability Analysis
- Occlusion sensitivity maps were employed: systematically occluding small 3D cubic regions and observing changes in the output probability.
- Occlusion was performed synchronously across all registered volumes to identify the regions that are most influential for the classification.

## Key Experimental Results

### Ablation Study (F1-score)

| Input Modality | Validation F1 | Test F1 |
|----------|----------|----------|
| MRI post-OP Only | $0.70$ | — |
| MRI event Only | $0.58$ | — |
| RD map Only | **$0.78$** | — |
| MRI post-OP + MRI event | — | — |
| MRI post-OP + RD | $0.828$ | — |
| MRI event + RD | **$0.83$** | — |
| All Three (RICE-NET) | $0.804$ | **$0.916$** |

### Key Findings
- **The radiotherapy dose map is the most informative single-modality input** (F1 = $0.78$ vs. MRI post-OP $0.70$ vs. MRI event $0.58$).
- Integrating MRI with RD further improves performance, validating the complementarity of modalities.
- The ensembled cross-validation model achieved an F1-score of $0.916$ on the independent test set (via majority voting).
- In experiments using only MRI, the gap between validation and test F1-scores was approximately $0.35$, reflecting the statistical uncertainty of small datasets.
- Occlusion analysis demonstrates that the model's focus areas are highly correlated with high-dose regions, while also attending to contrast-enhancing lesions.

## Highlights & Insights

1. **First End-to-End Classification Fusing Radiotherapy Dose Maps**: Radiotherapy planning is utilized as an explicit input, validating its importance as the strongest single-modality signal.
2. **Longitudinal MRI Modeling**: Post-operative baseline and event-timepoint images are utilized simultaneously to capture lesion evolutionary information.
3. **Systematic Ablation**: Full ablation across 7 modal combinations quantifies the diagnostic contribution of each modality.
4. **Clinical Interpretability**: Occlusion sensitivity maps align with clinical areas of interest, facilitating assisted decision-making.
5. **Use of Routine T1 MRI**: No reliance on scarce diffusion MRI, enhancing clinical applicability.

## Limitations & Future Work

1. **Extremely Small Sample Size**: Only 92 patients (80 for training, 12 for testing), lacking sufficient statistical reliability and showing prominent gaps between validation and test performances.
2. **Lack of Unaffected Control Group**: The dataset only contains recurrence and RICE categories, missing non-lesion controls.
3. **Simple Channel Fusion**: Channel-wise concatenation might not capture complex interaction patterns between the MRI and dose maps.
4. **Single-center Data**: All data originate from Heidelberg University Hospital; thus cross-center generalizability remains unknown.
5. **Exclusion of Clinical Variables**: Patient age, treatment regimens, and other clinical metadata are not integrated.
6. **All Modal Combination Yields Lower Validation F1 than Some Partial Combinations** ($0.804$ vs. $0.83$), indicating possible overfitting or modal conflict.

## Related Work & Insights

| Method | Characteristics | Comparison with Ours |
|------|------|-----------|
| Bernhardt et al. | DEGRO guidelines, clinical workflow based on diffusion MRI | RICE-NET uses routine T1 MRI, which is more easily accessible |
| Wang et al. | DTI + DSC-MRI to differentiate pseudoprogression | Relies on diffusion imaging, possessing low clinical accessibility |
| Eichkorn et al. | Analyzing association between RICE and ischemic stroke risk factors | Non-deep learning method, whereas RICE-NET provides automated classification |
| Standard clinical workflow | Multidisciplinary evaluation by tumor boards | RICE-NET can assist in accelerating decision-making, though clinical validation is still required |

## Rating

- Novelty: ⭐⭐⭐⭐ — First to utilize radiotherapy dose maps as a deep learning input for RICE vs. recurrence classification.
- Experimental Thoroughness: ⭐⭐⭐ — Comprehensive ablation design but the sample size is too small, lacking statistical significance.
- Writing Quality: ⭐⭐⭐⭐ — Clear problem motivation and adequate description of clinical background.
- Value: ⭐⭐⭐ — Crucial clinical problem but requires large-scale multi-center validation to confirm practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)
- [\[CVPR 2025\] Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation](federated_modality-specific_encoders_and_partially_personalized_fusion_decoder_f.md)
- [\[CVPR 2025\] Automated Detection of Malignant Lesions in the Ovary Using Deep Learning Models and XAI](automated_detection_of_malignant_lesions_in_the_ovary_using_deep_learning_models.md)
- [\[CVPR 2025\] Evidential learning driven Breast Tumor Segmentation with Stage-divided Vision-Language Interaction](evidential_learning_driven_breast_tumor_segmentation_with_stage-divided_vision-l.md)
- [\[CVPR 2025\] Reinforcing the Weakest Links: Modernizing SIENA with Targeted Deep Learning Integration](reinforcing_the_weakest_links_modernizing_siena_with_targeted_deep_learning_inte.md)

</div>

<!-- RELATED:END -->
