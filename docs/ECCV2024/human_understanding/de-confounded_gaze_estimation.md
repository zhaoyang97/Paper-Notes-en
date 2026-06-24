---
title: >-
  [Paper Note] De-confounded Gaze Estimation
description: >-
  [ECCV 2024][Human Understanding][Gaze Estimation] This paper proposes a causal intervention-based gaze estimation framework, FSCI, which decouples gaze-related features from irrelevant features (such as identity and illumination) via feature separation. By utilizing a dynamic confounder bank to perform causal intervention on irrelevant features, FSCI achieves a 36.2% improvement over the baseline and an 11.5% improvement over the SOTA under cross-domain settings.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Gaze Estimation"
  - "Causal Inference"
  - "Cross-Domain Generalization"
  - "Feature Separation"
  - "Confounder"
date: 2026-05-08
content_hash: 2f545691afa49002
---

# De-confounded Gaze Estimation

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Human Understanding  
**Keywords**: Gaze Estimation, Causal Inference, Cross-Domain Generalization, Feature Separation, Confounder

## TL;DR

This paper proposes a causal intervention-based gaze estimation framework, FSCI, which decouples gaze-related features from irrelevant features (such as identity and illumination) via feature separation. By utilizing a dynamic confounder bank to perform causal intervention on irrelevant features, FSCI achieves a 36.2% improvement over the baseline and an 11.5% improvement over the SOTA under cross-domain settings.

## Background & Motivation

**Background**: Deep learning-based gaze estimation methods have achieved highly mature performance within the same domain, but their performance degrades significantly in cross-domain scenarios (different data collection environments, different populations). Currently, mainstream methods primarily mitigate this issue through domain adaptation or feature alignment.

**Limitations of Prior Work**: Gaze estimation models inevitably learn gaze-irrelevant information during training, such as facial identity features and illumination conditions. These "confounders" cause severe performance degradation during domain transfer, as models may erroneously rely on these domain-specific features to predict gaze directions.

**Key Challenge**: Gaze features are highly entangled with irrelevant features (such as identity and illumination) in the feature space, making it difficult for traditional methods to separate these two types of features effectively. Even with domain adaptation, the model might still exploit confounders as "shortcuts" for prediction.

**Goal**: How to train a gaze estimation model robust to confounders (identity, illumination, etc.) without access to target domain data? Specifically, this is decomposed into: (1) How to effectively separate gaze features from irrelevant features? (2) How to eliminate the impact of irrelevant features on gaze prediction through causal intervention?

**Key Insight**: Starting from the perspective of causal inference, the authors treat confounders as confounding variables in a causal graph. Through causal intervention (do-calculus), the backdoor path from confounders to prediction results can be cut off, enabling the model to make predictions solely based on gaze features.

**Core Idea**: Achieving causal intervention through feature separation combined with a dynamic confounder bank, thereby eliminating the influence of confounders such as identity and illumination on gaze estimation to achieve cross-domain generalization.

## Method

### Overall Architecture

The overall workflow of the FSCI (Feature-Separation-based Causal Intervention) framework is as follows: given an input face/eye image, features are first extracted through a shared backbone network, and then decomposed into gaze-related features and gaze-irrelevant features by a feature separation module. During the training phase, a Dynamic Confounder Bank is utilized to perform causal intervention on the gaze-irrelevant features. By averaging various irrelevant features stored in the bank, the model "sees" diverse combinations of potential confounders during training, thereby weakening its dependence on any specific confounder. Ultimately, the model predicts gaze angles using only clean gaze features.

### Key Designs

1. **Feature Separation Module**:

    - **Function**: Decomposing the feature representation of the input image into gaze-related features $f_g$ and gaze-irrelevant features $f_c$.
    - **Mechanism**: Two branch networks are used to learn gaze features and irrelevant features respectively from the features extracted by the shared backbone. Adversarial training ensures the mutual independence of these two types of features—features from the gaze branch should not contain identity information, and features from the irrelevant branch should not contain gaze information. Specifically, gaze features are learned under the supervision of gaze angle prediction, while irrelevant features are learned through auxiliary tasks such as identity classification, with an orthogonality constraint introduced to ensure the two types of features are as independent as possible.
    - **Design Motivation**: Precise causal intervention on irrelevant features can only be performed if gaze features and irrelevant features are effectively separated. If the two types of features are entangled, intervening in irrelevant features will also affect gaze features.

2. **Dynamic Confounder Bank**:

    - **Function**: Storing and dynamically updating gaze-irrelevant features from different samples, serving as the basis for causal intervention.
    - **Mechanism**: A queue-based feature bank is maintained, continuously updating its content with irrelevant features extracted from the current batch during training. During causal intervention, all irrelevant features in the bank are averaged to obtain a "de-confounded" irrelevant feature representation $\bar{f}_c = \frac{1}{N}\sum_{i=1}^{N} f_c^i$, which is then combined with the gaze features for prediction. Consequently, during training, the model "sees" the average effect of all possible confounders for each sample, thereby eliminating dependency on any specific confounder.
    - **Design Motivation**: Direct implementation of causal intervention $P(Y|do(X))$ is practically unfeasible as it requires traversing all possible confounders. The dynamic bank provides an effective approximation scheme—approximating the backdoor adjustment formula by maintaining a sufficiently large and continuously updated pool of confounder samples.

3. **Causal Intervention Training**:

    - **Function**: Implementing causal intervention during training so that the final model's predictions are unaffected by confounders.
    - **Mechanism**: During the forward pass of training, the gaze features of the current sample are fused with the average features from the confounder bank before predicting the gaze angle. The loss function includes the gaze angle prediction loss, the feature separation orthogonality loss, and an auxiliary identity classification loss. During inference, the average features of the bank are also used to replace the irrelevant features of the current sample, ensuring that predictions are unaffected by individual-specific confounders.
    - **Design Motivation**: Traditional methods merely attempt to make the model ignore confounders, but the model may still exploit this information through implicit paths. Causal intervention mathematically cuts off the backdoor path, offering stronger guarantees.

### Loss & Training

The total loss function consists of three components: (1) gaze angle regression loss $\mathcal{L}_{gaze}$, which uses L1 loss to supervise the difference between predicted and ground-truth angles; (2) orthogonality loss $\mathcal{L}_{orth}$, which constrains the orthogonality between gaze features and irrelevant features; and (3) identity classification loss $\mathcal{L}_{id}$, which ensures that irrelevant features indeed capture identity details. Training is divided into a warmup stage (training feature separation only) and a causal intervention stage (incorporating the dynamic confounder bank for end-to-end training).

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours FSCI | Prev. SOTA | Gain |
|---|---|---|---|---|
| ETH-XGaze → MPIIGaze | Angular Error (°) | 6.2 | ~7.0 | ~11.5% |
| ETH-XGaze → EyeDiap | Angular Error (°) | 7.1 | ~8.0 | ~11.3% |
| Cross-Domain Average | Angular Error (°) | - | Baseline | Up to 36.2% |

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| Without Feature Separation | Baseline Performance | Confounders severely affect cross-domain performance |
| Feature Separation without Causal Intervention | Moderate Improvement | Separation is helpful but insufficient |
| Full FSCI | Best Performance | Feature separation and causal intervention work synergistically |
| Static Confounder Bank vs Dynamic Bank | Dynamic is Better | Dynamic updates cover more variations of confounders |

### Key Findings

- Both feature separation and causal intervention are indispensable: applying feature separation alone without causal intervention yields limited effectiveness; causal intervention must be built upon effective feature separation.
- The size of the dynamic confounder bank affects performance; an excessively small bank is insufficient to cover the distribution space of the confounders.
- FSCI performs robustly under various cross-domain settings without requiring access to any target domain data.
- Identity and illumination are the two primary types of confounders, and eliminating their influence can significantly improve cross-domain performance.

## Highlights & Insights

- Introducing causal inference to the field of gaze estimation offers a novel perspective, providing a theoretical foundation for solving cross-domain generalization issues.
- The design of the dynamic confounder bank cleverly translates the theoretical requirements of causal intervention into a feasible engineering solution.
- The source-only training scheme, which requires no target domain data, makes the method highly practical for real-world deployment.
- The framework is generic, and the paradigm of feature separation combined with causal intervention can be extended to other vision tasks affected by confounders.

## Limitations & Future Work

- The quality of feature separation highly depends on the design of auxiliary tasks, and ensuring complete separation remains an open question.
- The size and update strategy of the dynamic bank require manual tuning, lacking an adaptive mechanism.
- The paper only considers identity and illumination as confounders, whereas more types of confounders may exist in real-world scenarios.
- Causal intervention assumes independence among confounders, but interactions among different confounders may actually exist in practice.
- Integrating contrastive learning can be explored to enhance the efficacy of feature separation.

## Related Work & Insights

- **PureGaze**: Achieves domain-invariant gaze features through self-supervised contrastive learning, but does not consider confounders from a causal perspective.
- **RUDA**: Performs cross-domain transfer for gaze estimation via unsupervised domain adaptation, but requires target domain data.
- **Applications of Causal Inference in Vision**: Works such as CaaM and CIRL apply causal inference to tasks like image classification; this paper extends it to a regression task (gaze angle estimation).
- Inspiration: The concept of causal intervention can be applied to other regression tasks requiring cross-domain generalization, such as head pose estimation and human pose estimation.

## Rating

- Novelty: ⭐⭐⭐⭐ The application of causal inference to gaze estimation is relatively novel, with a clear theoretical motivation.
- Experimental Thoroughness: ⭐⭐⭐ The cross-domain experimental setup is reasonable, but the datasets and comparison methods could be more comprehensive.
- Writing Quality: ⭐⭐⭐ The description of causal inference is clear, but several experimental details are not fully elaborated.
- Value: ⭐⭐⭐⭐ Provides a theoretically grounded new framework for cross-domain gaze estimation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] 3DGazeNet: Generalizing 3D Gaze Estimation with Weak-Supervision from Synthetic Views](3dgazenet_generalizing_3d_gaze_estimation_with_weak-supervision_from_synthetic_v.md)
- [\[NeurIPS 2025\] OmniGaze: Reward-inspired Generalizable Gaze Estimation in the Wild](../../NeurIPS2025/human_understanding/omnigaze_reward-inspired_generalizable_gaze_estimation_in_the_wild.md)
- [\[CVPR 2025\] Enhancing 3D Gaze Estimation in the Wild Using Weak Supervision with Gaze Following Labels](../../CVPR2025/human_understanding/enhancing_3d_gaze_estimation_in_the_wild_using_weak_supervision_with_gaze_follow.md)
- [\[CVPR 2026\] Render-to-Adapt: Unsupervised Personal Adaptation for Gaze Estimation](../../CVPR2026/human_understanding/render-to-adapt_unsupervised_personal_adaptation_for_gaze_estimation.md)
- [\[ICCV 2025\] Multi-view Gaze Target Estimation](../../ICCV2025/human_understanding/multi-view_gaze_target_estimation.md)

</div>

<!-- RELATED:END -->
