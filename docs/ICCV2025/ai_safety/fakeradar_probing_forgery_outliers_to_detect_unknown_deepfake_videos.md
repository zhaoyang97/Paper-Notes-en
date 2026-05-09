---
title: >-
  [Paper Note] FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos
description: >-
  [ICCV 2025][AI Safety][Deepfake Detection] This paper proposes FakeRadar, a deepfake video detection framework that actively generates outlier samples simulating unknown forgeries in the feature space via Forgery Outlier Probing, and designs an Outlier-Guided Tri-Training strategy with three-class optimization (Real/Fake/Outlier). FakeRadar significantly outperforms existing methods on cross-dataset and cross-manipulation evaluations.
tags:
  - ICCV 2025
  - AI Safety
  - Deepfake Detection
  - Cross-Domain Generalization
  - Outlier Probing
  - Contrastive Learning
  - CLIP
date: 2026-05-08
content_hash: 33643035c3ef9687
---

# FakeRadar: Probing Forgery Outliers to Detect Unknown Deepfake Videos

**Conference**: ICCV 2025
**arXiv**: [2512.14601](https://arxiv.org/abs/2512.14601)
**Code**: N/A
**Area**: AI Security
**Keywords**: Deepfake Detection, Cross-Domain Generalization, Outlier Probing, Contrastive Learning, CLIP

## TL;DR
This paper proposes FakeRadar, a deepfake video detection framework that actively generates outlier samples simulating unknown forgeries in the feature space via Forgery Outlier Probing, and designs an Outlier-Guided Tri-Training strategy with three-class optimization (Real/Fake/Outlier). FakeRadar significantly outperforms existing methods on cross-dataset and cross-manipulation evaluations.

## Background & Motivation
- **Background**: Deepfake technology has advanced rapidly, producing highly realistic face-manipulated videos, prompting a large body of detection work at both image and video levels.
- **Limitations of Prior Work**: Existing methods rely on specific forgery artifacts (boundary inconsistencies, blinking anomalies, texture irregularities, etc.) and are effective only against known forgery types, with severely degraded generalization when facing novel generative techniques (e.g., diffusion models).
- **Key Challenge**: Real faces form compact, dense distributions in feature space, while forged faces form sparse, scattered clusters due to diverse manipulation types — classifiers trained on known forgery patterns cannot cover the space of unknown ones.
- **Key Insight**: The paradigm shifts from "passively learning known forgery patterns" to "actively probing unknown forgery regions," analogous to a radar system scanning for unknown targets via spectrum analysis.
- **Core Idea**: Leveraging the deep feature priors of CLIP pre-trained models, the method "pre-explores" regions in feature space where unknown forgeries may emerge through dynamic sub-cluster modeling and cluster-conditioned outlier generation.

## Method

### Overall Architecture
FakeRadar builds upon a frozen CLIP ViT-B/16 backbone with ST-Adapters inserted for parameter-efficient fine-tuning. It comprises two core components: (1) Forgery Outlier Probing (FOP), which generates outlier samples simulating unknown forgeries in feature space; and (2) Outlier-Guided Tri-Training (OGTT), which jointly optimizes the model via three-class classification (Real/Fake/Outlier), merging Fake and Outlier into a single "forgery" class at inference.

### Key Designs
1. **Forgery Outlier Probing (FOP)**:

    - **Function**: Models the feature-space distribution of training samples at fine granularity and generates outlier samples near sub-cluster boundaries to simulate unknown forgeries.
    - **Mechanism**:
        - **Dynamic Sub-Cluster Modeling**: Treats real faces and each of the four forgery types as independent classes. A main clustering network learns soft assignments over a GMM distribution, aligned with GMM responsibilities via KL divergence loss $\mathcal{L}_{main} = \sum_i KL(\mathbf{r}_i \| \mathbf{r}_i^E)$. A sub-cluster network attempts to split each cluster into two sub-clusters, using the Hastings ratio $H_s$ to decide split/merge operations.
        - **Cluster-Conditioned Outlier Generation**: Outliers $\mathcal{V}_k$ are sampled from the $\varepsilon$-likelihood region of sub-cluster distributions, conditioned on their probability density under the sub-cluster Gaussian being below threshold $\varepsilon$, placing them near cluster boundaries.
    - **Design Motivation**: A fixed $K=5$ is insufficient to capture the intra-class multimodal structure of forgery types; dynamic split/merge enables more precise distribution modeling, and boundary-region outliers better simulate "novel forgery shifts."

2. **Outlier-Guided Tri-Training (OGTT)**:

    - **Function**: Jointly optimizes the backbone and a three-class classifier to explicitly distinguish Real, Fake, and Outlier categories.
    - **Mechanism**:
        - **Outlier-Driven Contrastive Loss**: An InfoNCE-based contrastive loss $\mathcal{L}_{con}$ maximizes similarity between samples and their assigned sub-cluster centers while minimizing similarity to other sub-cluster centers and outliers.
        - **Outlier-Conditioned Cross-Entropy Loss**: A three-class cross-entropy $\mathcal{L}_{cls} = -\sum_{c} y_c \log p_c$ ensures clear decision boundaries among all three classes, particularly preventing Outlier samples from being misclassified as Real.
        - Total loss: $\mathcal{L}_{total} = \mathcal{L}_{con} + \lambda \mathcal{L}_{cls}$, with $\lambda=0.5$.
    - **Design Motivation**: Three-class training allows the model to independently label unknown forgeries during training, while merging Fake and Outlier at inference yields a binary prediction; contrastive loss enhances inter-class separation and cross-entropy ensures clear decision boundaries.

3. **Model Adaptation & Inference**:

    - **Function**: Inserts ST-Adapters into the frozen CLIP backbone for parameter-efficient fine-tuning.
    - **Mechanism**: $\text{ST-Adapter}(x) = x + \text{ReLU}(\text{Conv3D}(xW_{down}))W_{up}$, where 3D convolutions capture spatio-temporal features with minimal additional parameters.
    - **Design Motivation**: Preserves rich semantic features from CLIP pre-training while adapting to spatio-temporal patterns in deepfake detection.

### Loss & Training
- Total loss: contrastive loss + $0.5 \times$ three-class cross-entropy.
- Each mini-batch contains 16 training samples and 16 outlier samples.
- Adam optimizer with cosine learning rate schedule, initial learning rate $1\text{e-}4$, trained for 60 epochs.
- Each video is sampled into 4 temporal clips of 12 consecutive frames; face images are resized to $224\times224$.

## Key Experimental Results

### Main Results (Cross-Dataset Evaluation, AUC%)

| Method | Backbone | FF++ | CDFv2 | DFDCP | DFDC | DFD |
|------|------|------|-------|-------|------|-----|
| LSDA | EfficientNet | - | 91.1 | 81.2 | 77.0 | 95.6 |
| TALL | Swin Trans. | 99.9 | 90.8 | - | 76.8 | - |
| AltFreezing | 3D ResNet | 99.7 | 89.5 | - | - | 93.7 |
| **FakeRadar** | **ViT-B/16** | **99.1** | **91.7** | **88.5** | **84.1** | **96.2** |

### Ablation Study

| Configuration | CDFv2 | DFDC | DFDCP | DFD | Avg. |
|------|-------|------|-------|-----|------|
| FakeRadar (full) | 91.6 | 84.1 | 88.5 | 96.2 | 90.1 |
| w/o ODCL | 89.9 | 81.2 | 85.6 | 94.9 | 87.9 |
| w/o OCCE | 88.6 | 80.9 | 87.7 | 94.7 | 88.0 |
| w/o FOP | 88.8 | 80.2 | 86.7 | 94.5 | 87.6 |
| Binary classification only | 88.4 | 78.4 | 85.1 | 94.3 | 86.7 |
| No fine-tuning | 88.2 | 78.3 | 84.8 | 94.2 | 86.4 |

**Model Variants** (Cross-Dataset AUC%):

| Variant | FF++ | CDFv2 | DFDCP | DFDC | DFD |
|------|------|-------|-------|------|-----|
| Frozen (no fine-tuning) | 55.2 | 60.0 | 59.0 | 55.2 | 57.4 |
| Supervised (binary) | 98.2 | 88.2 | 84.8 | 78.3 | 94.2 |
| Proposed (full) | 99.1 | 91.7 | 88.5 | 84.1 | 96.2 |

### Key Findings
- FakeRadar surpasses UCF and LTTD on DFDC by 3.6% and 3.7% AUC respectively, demonstrating substantial cross-domain generalization gains.
- In cross-manipulation evaluations (training on one manipulation type, testing on the remaining three), FakeRadar achieves the best average AUC among all methods, outperforming DCL by 7.41% in the F2F training scenario.
- Dynamic sub-cluster modeling shows substantial fluctuation in cluster count during early training (epochs 0–8), stabilizing around 3 sub-clusters after approximately epoch 10.
- The error correction rate of the tri-classifier on DFDC (whose forgery types differ from the training set) reaches approximately 40% at epoch 10, significantly higher than the ~5% observed on the FF++ test set.
- Dynamic $K$ vs. fixed $K=5$: the full model achieves 90.1% vs. 87.4% for fixed $K$, a gap of 2.7%.

## Highlights & Insights
- The paradigm shift to "active probing" is conceptually compelling: rather than passively learning known patterns, the model actively explores unknown regions in feature space.
- Outlier generation is performed in feature space rather than pixel space, circumventing the difficulty of synthesizing realistic forged images.
- The design of three-class (Real/Fake/Outlier) training combined with binary (Real/Fake) inference is both elegant and effective.
- The dynamic sub-cluster split/merge process self-adapts during training, eliminating the need to manually specify the final cluster count.
- t-SNE visualizations clearly demonstrate that FakeRadar learns more compact feature distributions with sharper decision boundaries.

## Limitations & Future Work
- Training is conducted solely on FaceForensics++ (HQ); the effect of incorporating more diverse or recent forgery types in training data remains unexplored.
- The $\varepsilon$ threshold for outlier sampling is a hyperparameter, and its sensitivity across different datasets is not thoroughly discussed.
- At inference, Outlier predictions are naively merged into the Fake class without leveraging outlier confidence scores for more nuanced decisions.
- Only the ViT-B/16 backbone is evaluated; whether larger models (e.g., ViT-L) could further improve generalization remains an open question.
- Adversarial robustness evaluation (e.g., against intentionally evasive adversarial examples) is absent.

## Related Work & Insights
- The Virtual Outlier Synthesis (VOS) concept is effectively transferred to the deepfake detection domain.
- The ST-Adapter parameter-efficient fine-tuning strategy successfully preserves CLIP's general-purpose features while adapting to the downstream task.
- The dynamic split/merge sub-cluster modeling (inspired by the Dirichlet process) is a promising approach for distribution modeling in other settings.
- This work is orthogonal to pixel-space data augmentation methods such as SBI — synthesizing in feature space vs. image space — and the two approaches could potentially be combined.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The paradigm shift from passive detection to active probing, feature-space outlier generation, and tri-training strategy together constitute a highly original contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation spanning cross-dataset, cross-manipulation, ablation, model variant, feature visualization, and sub-cluster evolution analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated and the "radar probing" analogy is intuitive, though some notation could be further simplified.
- **Value**: ⭐⭐⭐⭐⭐ Practically significant — provides a generalizable detection approach that does not rely on specific forgery artifacts, directly addressing the challenge of continuously emerging novel generative techniques.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Vulnerability-Aware Spatio-Temporal Learning for Generalizable Deepfake Video Detection](vulnerability-aware_spatio-temporal_learning_for_generalizable_deepfake_video_de.md)
- [\[AAAI 2026\] Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception](../../AAAI2026/ai_safety/detect_all-type_deepfake_audio_wavelet_prompt_tuning_for_enhanced_auditory_perce.md)
- [\[CVPR 2026\] Tutor-Student Reinforcement Learning: A Dynamic Curriculum for Robust Deepfake Detection](../../CVPR2026/ai_safety/tutor-student_reinforcement_learning_a_dynamic_curriculum_for_robust_deepfake_de.md)
- [\[AAAI 2026\] Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection](../../AAAI2026/ai_safety/fine-grained_dino_tuning_with_dual_supervision_for_face_forgery_detection.md)
- [\[NeurIPS 2025\] Not All Deepfakes Are Created Equal: Triaging Audio Forgeries for Robust Deepfake Singer Identification](../../NeurIPS2025/ai_safety/not_all_deepfakes_are_created_equal_triaging_audio_forgeries_for_robust_deepfake.md)

<!-- RELATED:END -->
