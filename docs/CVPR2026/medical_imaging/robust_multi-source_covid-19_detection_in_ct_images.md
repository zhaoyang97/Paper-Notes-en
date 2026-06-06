---
title: >-
  [Paper Note] Robust Multi-Source Covid-19 Detection in CT Images
description: >-
  [CVPR 2026][Medical Imaging][COVID-19 detection] This paper proposes a multi-task learning framework that jointly trains a COVID-19 diagnosis head and a source hospital identification head (supervised by a logit-adjusted…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "COVID-19 detection"
  - "multi-source domain"
  - "multi-task learning"
  - "logit-adjusted loss"
  - "CT images"
date: 2026-05-08
content_hash: 314cd610a2c09962
---

# Robust Multi-Source Covid-19 Detection in CT Images

**Conference**: CVPR 2026
**arXiv**: [2604.03320](https://arxiv.org/abs/2604.03320)  
**Code**: [https://github.com/Purdue-M2/-multisource-covid-ct](https://github.com/Purdue-M2/-multisource-covid-ct)  
**Area**: Medical Imaging
**Keywords**: COVID-19 detection, multi-source domain, multi-task learning, logit-adjusted loss, CT images

## TL;DR

This paper proposes a multi-task learning framework that jointly trains a COVID-19 diagnosis head and a source hospital identification head (supervised by a logit-adjusted loss) on a shared EfficientNet-B7 backbone, encouraging the feature extractor to learn institution-invariant representations. The method achieves an F1 of 0.9098 on a multi-source CT dataset.

## Background & Motivation

1. **Background**: Deep learning has demonstrated strong performance in COVID-19 CT detection; however, most existing methods assume that training and test data originate from the same institution. In multi-center settings, domain shifts introduced by different scanners, imaging protocols, and patient populations substantially degrade model performance.

2. **Limitations of Prior Work**: (1) Existing methods optimize a single COVID vs. non-COVID objective without awareness of data provenance, causing features to be biased toward the center contributing the most data. (2) Sample counts are unevenly distributed across centers (approximately 330 cases each for three centers and only 234 for one), further exacerbating this bias.

3. **Key Challenge**: In single-task training, the backbone is free to exploit institution-specific spurious features (e.g., reconstruction kernels, intensity distributions) to minimize the loss — features that completely fail to transfer across centers.

4. **Goal**: To encourage the shared feature extractor to learn disease representations that are invariant across institutions, while preventing implicit bias induced by imbalanced source distributions.

5. **Key Insight**: Source hospital identification is introduced as an auxiliary task, compelling the backbone to simultaneously "understand" source-level differences, thereby learning disentangled representations that distinguish disease from provenance.

6. **Core Idea**: A source classification head supervised by a logit-adjusted loss is appended alongside the disease detection head, forcing the backbone encoder to learn source-invariant features.

## Method

### Overall Architecture

**Input**: Each CT scan is preprocessed via lung extraction and KDS (kernel density slice) sampling to select 8 representative slices. The 8 slices are independently encoded by the shared EfficientNet-B7 backbone into 2560-dimensional feature vectors, which are then element-wise mean-pooled into a single scan-level representation. Two classification heads output the COVID-19 diagnosis probability (binary, sigmoid) and the source hospital prediction (four-class, softmax), respectively. Only the disease detection head is used at inference time.

### Key Designs

1. **KDS Slice Sampling Preprocessing**:

    - **Function**: Standardizes CT scans of varying lengths into a fixed 8-slice representation.
    - **Mechanism**: The lung area of each valid slice is computed, and a Gaussian kernel density estimate (Scott's rule bandwidth) is fitted. The cumulative distribution function is divided into 8 equal percentile intervals, and the slice closest to the midpoint of each interval is selected. Repeated selections in short scans are padded with the last slice.
    - **Design Motivation**: Compared to uniform sampling, KDS adapts to anatomical structure — regions where the lung cross-section changes rapidly (hilum, lung base) are sampled more densely, while the more uniform apical region is sampled less, preserving the most diagnostically informative content.

2. **Dual-Head Multi-Task Architecture**:

    - **Function**: Separates disease signals from institution-specific features.
    - **Mechanism**: The COVID-19 detection head and the source identification head share the same 2560-dimensional scan-level representation. The presence of the source head prevents the backbone from relying on source-specific features to minimize the disease loss, as such features would simultaneously be exploited by the source head, causing mutual gradient constraint. This constitutes an implicit form of domain regularization. The source head is discarded at inference time.
    - **Design Motivation**: In single-task training, the backbone is free to encode hospital identity as a shortcut. With the source identification task added, the backbone must serve both objectives simultaneously and naturally tends to learn features that are useful for both without being biased toward any particular source.

3. **Logit-Adjusted Source Classification Loss**:

    - **Function**: Corrects gradient bias caused by imbalanced source distributions.
    - **Mechanism**: Prior to the softmax in the standard cross-entropy loss, a bias term $\log(p(d))$ is added to the logit of each source: $\ell_{\text{LA}} = -\log \frac{e^{f_d(x) + \log(p(d))}}{\sum_{d'} e^{f_{d'}(x) + \log(p(d'))}}$, where $p(d)$ is the prior frequency of source $d$ in the training set. This lowers the raw confidence required to predict rare sources, ensuring that minority sources consistently receive meaningful gradient signals.
    - **Design Motivation**: Under standard cross-entropy, the source head is biased toward majority sources (~330 cases each for three centers vs. 234 for one), and this bias propagates through the shared backbone. Logit adjustment eliminates this bias by equalizing gradient contributions.

### Loss & Training

Total loss: $\ell = \ell_{\text{ce}}(h(f(x)), y) + \gamma \cdot \ell_{\text{LA}}$. The hyperparameter $\gamma$ is searched over {0.1, 0.2, 0.5, 1.0}, with the optimal value $\gamma=0.5$. Adam optimizer, learning rate $1 \times 10^{-4}$, weight decay $5 \times 10^{-4}$, batch size 10, mixed-precision training. Data augmentation includes random flipping, translation/scale/rotation, hue-saturation jitter, brightness-contrast adjustment, and coarse dropout. Training runs for 8 epochs, and the checkpoint with the highest validation F1 is retained. Single A100 GPU.

## Key Experimental Results

### Main Results

| Configuration | γ | F1 ↑ | AUC ↑ | Sensitivity | Specificity | Final Score ↑ |
|------|---|------|-------|--------|--------|--------------|
| Baseline (BCE) | - | 0.8915 | 0.9627 | 0.8984 | 0.9167 | 0.8008 |
| Multi-task (CE) best | 0.1 | 0.8889 | 0.9683 | 0.9062 | 0.9056 | 0.7996 |
| **Multi-task + LA** | **0.5** | **0.9098** | 0.9647 | 0.9062 | **0.9389** | **0.8194** |

Multi-task + LA improves the Final Score by 1.86 percentage points over the single-task baseline and by 1.98 percentage points over the standard CE multi-task variant.

### Ablation Study (Per-Source F1)

| Method | Src 0 | Src 1 | Src 2 | Src 3 | Final Score |
|------|-------|-------|-------|-------|-------------|
| Baseline | 0.9221 | 0.8776 | 0.9269 | 0.4767 | 0.8008 |
| MT CE (γ=0.1) | 0.8888 | 0.9000 | 0.9389 | 0.4706 | 0.7996 |
| **MT + LA (γ=0.5)** | **0.9555** | 0.8888 | **0.9756** | 0.4578 | **0.8194** |

### Key Findings

- **Standard CE multi-task training is ineffective**: Across all $\gamma$ values, CE-based multi-task yields a Final Score equal to or lower than the single-task baseline, indicating that multi-task training without logit adjustment introduces source bias into the backbone.
- **LA gains are source-specific**: Source 2 improves by 4.87 percentage points (→0.9756) and Source 0 by 3.34 percentage points (→0.9555), demonstrating that logit adjustment redistributes gradient contributions across centers.
- **Low Source 3 scores are a structural artifact**: All 45 Source 3 cases in the validation set are non-COVID, making the COVID F1 score zero by definition and dragging down the mean. The non-COVID F1 for Source 3 reaches 0.9888 under MT+LA.
- **Non-monotonic sensitivity to $\gamma$**: $\gamma=0.5$ is optimal; $\gamma=0.2$ performs worse than the baseline (gradient interference without sufficient signal), while $\gamma=1.0$ allows the source objective to overwhelm the disease objective.

## Highlights & Insights

- **Source classification as implicit domain regularization**: Without requiring explicit domain alignment objectives (e.g., MMD, adversarial training), the framework promotes source-invariant representation learning through an auxiliary classification task alone — a concise and effective approach.
- **Elegant application of logit adjustment**: The logit-adjusted loss of Menon et al. is transferred from the class imbalance setting to the source imbalance setting, with theoretical guarantees of Fisher consistency for balanced error.
- **KDS slice sampling as a transferable technique**: This adaptive sampling strategy based on kernel density estimation generalizes to any scenario requiring fixed-length representations extracted from variable-length sequences.

## Limitations & Future Work

- **Methodological simplicity**: The approach is essentially EfficientNet-B7 with an auxiliary 4-way classification head and a logit-adjusted loss, offering limited technical novelty.
- Source labels are assumed to be available at training time, making the method inapplicable in fully anonymized settings.
- The dataset comprises only 4 sources and ~1,200 training samples; the small scale raises questions about the generalizability of the conclusions.
- The validation set design is flawed — Source 3 contains no COVID-positive samples, introducing a structural ceiling on the metrics.
- No comparison is made against proper domain adaptation methods (DANN, CORAL) or domain generalization methods (DRO, SWAD).
- Unsupervised domain discovery could be considered as an alternative to explicit source labels.

## Related Work & Insights

- **vs. Hsu et al. (SSFL+KDS)**: This paper reuses the same preprocessing pipeline and augments it with a multi-task head, demonstrating the importance of source awareness for cross-center generalization.
- **vs. domain adaptation methods**: Rather than performing explicit domain alignment, this paper achieves similar effects indirectly via an auxiliary task — simpler but with weaker theoretical guarantees.
- **vs. Li et al. (3D + weighted CE)**: Li et al. use 3D volumetric data with class-weighted loss, while this paper adopts 2D slice sampling with source-weighted loss; the two directions are complementary.

## Rating

- **Novelty**: ⭐⭐ Multi-task learning and logit-adjusted loss are both established techniques combined in a straightforward manner, with limited methodological innovation.
- **Experimental Thoroughness**: ⭐⭐⭐ Ablation studies are adequate, but the dataset is too small (~1,500 samples) and comparisons with domain adaptation baselines are absent.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated, and the experimental analysis is detailed, particularly the per-source breakdown and $\gamma$ sensitivity analysis.
- **Value**: ⭐⭐ The method is overly simple, the application scenario (COVID-19 CT) is no longer at the research frontier, and practical impact is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Robust Fair Disease Diagnosis in CT Images](robust_fair_disease_diagnosis_in_ct_images.md)
- [\[CVPR 2026\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semisupervised_framework.md)
- [\[CVPR 2026\] Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)
- [\[CVPR 2026\] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)
- [\[CVPR 2026\] CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)

</div>

<!-- RELATED:END -->
