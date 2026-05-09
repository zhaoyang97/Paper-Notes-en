---
title: >-
  [Paper Note] MaxSup: Overcoming Representation Collapse in Label Smoothing
description: >-
  [NeurIPS 2025][Label Smoothing] By decomposing the loss function of Label Smoothing (LS), this paper identifies an "error amplification term" that exacerbates misclassification, leading to intra-class feature collapse. The proposed Max Suppression (MaxSup) method redirects the penalty target from the ground-truth logit to the top-1 logit, eliminating the error amplification effect while preserving beneficial regularization.
tags:
  - NeurIPS 2025
  - Label Smoothing
  - Regularization
  - Representation Collapse
  - Logit Penalty
  - Overconfidence
date: 2026-05-08
content_hash: 661e41ae725826e8
---

# MaxSup: Overcoming Representation Collapse in Label Smoothing

**Conference**: NeurIPS 2025  
**arXiv**: [2502.15798](https://arxiv.org/abs/2502.15798)  
**Code**: [GitHub](https://github.com/ZhouYuxuanYX/Maximum-Suppression-Regularization)  
**Area**: Deep Learning Regularization / Image Classification  
**Keywords**: Label Smoothing, Regularization, Representation Collapse, Logit Penalty, Overconfidence

## TL;DR

By decomposing the loss function of Label Smoothing (LS), this paper identifies an "error amplification term" that exacerbates misclassification, leading to intra-class feature collapse. The proposed Max Suppression (MaxSup) method redirects the penalty target from the ground-truth logit to the top-1 logit, eliminating the error amplification effect while preserving beneficial regularization.

## Background & Motivation

Label Smoothing (LS) is a widely adopted regularization technique in deep learning that redistributes a portion of the one-hot label probability uniformly across all classes to reduce model overconfidence. LS has demonstrated improvements in accuracy and calibration on tasks such as image recognition and machine translation.

However, recent studies have revealed two critical issues with LS: (1) LS paradoxically increases overconfidence on misclassified samples; and (2) LS compresses feature representations into excessively tight clusters, diluting intra-class diversity. The precise cause of the latter has not been previously established.

The core contribution of this paper lies in a logit-level decomposition of the LS loss, which identifies the root cause: LS penalizes the ground-truth logit $z_{gt}$ rather than the maximum predicted logit $z_{max}$. When the prediction is correct ($z_{gt} = z_{max}$), LS functions normally; when the prediction is incorrect ($z_{gt} \neq z_{max}$), LS further suppresses the already non-maximal $z_{gt}$, widening the gap between the correct and incorrect logits and creating a vicious cycle.

## Method

### Overall Architecture

The design of MaxSup is straightforward: since the problem with LS stems from penalizing the wrong logit ($z_{gt}$ instead of $z_{max}$), the penalty target is simply redirected to $z_{max}$. This ensures a consistent regularization signal regardless of whether the prediction is correct.

### Key Designs

1. **Loss Decomposition of Label Smoothing**: Standard LS replaces the one-hot label $\mathbf{y}$ with soft labels $s_k = (1-\alpha)y_k + \frac{\alpha}{K}$. The authors decompose the LS cross-entropy loss into standard CE plus an additional regularization term $L_{LS}$. At the logit level, $L_{LS}$ is expressed as:

$$L_{LS} = \alpha\left(z_{gt} - \frac{1}{K}\sum_{k=1}^{K}z_k\right)$$

which penalizes the gap between the ground-truth logit and the mean logit. A key derivation further decomposes this into two components:

$$L_{LS} = \underbrace{\frac{\alpha}{K}\sum_{z_m < z_{gt}}(z_{gt} - z_m)}_{\text{Regularization term (beneficial)}} + \underbrace{\frac{\alpha}{K}\sum_{z_n > z_{gt}}(z_{gt} - z_n)}_{\text{Error amplification term (harmful)}}$$

When the prediction is correct, $z_{gt}$ is the maximum logit and the error amplification term vanishes, allowing LS to properly reduce overconfidence. When the prediction is incorrect, the error amplification term becomes negative, further suppressing $z_{gt}$ and reinforcing the erroneous prediction.

2. **Max Suppression Regularization**: MaxSup replaces the penalty target from $z_{gt}$ to $z_{max}$:

$$L_{MaxSup} = \alpha\left(z_{max} - \frac{1}{K}\sum_{k=1}^{K}z_k\right)$$

This minimal modification ensures that when the prediction is correct ($z_{max} = z_{gt}$), MaxSup is equivalent to LS; when the prediction is incorrect, MaxSup penalizes the erroneous top logit rather than suppressing the already lagging $z_{gt}$, thereby eliminating the error amplification effect.

3. **Gradient Analysis**: The gradient of MaxSup takes a clear form:

$$\frac{\partial L_{MaxSup}}{\partial z_k} = \begin{cases} \alpha(1 - \frac{1}{K}), & \text{if } k = \arg\max(\mathbf{q}) \\ -\frac{\alpha}{K}, & \text{otherwise} \end{cases}$$

The top-1 logit is reduced by $\alpha(1-1/K)$, while all other logits (including $z_{gt}$) receive a slight upward adjustment of $\alpha/K$. During misclassification, $z_{gt}$ is exempt from penalization and instead receives a marginal boost, facilitating error correction.

### Loss & Training

The final training loss consists of standard CE plus the MaxSup regularization term. Implementation requires only replacing the ground-truth position in the LS soft label with the model's top-1 predicted position. The authors employ a linearly increasing $\alpha$ schedule to improve training stability. The modification incurs negligible computational overhead and can be directly integrated into existing training pipelines.

## Key Experimental Results

### Main Results — ImageNet-1K Classification

| Model | Baseline | Label Smoothing | MaxSup | Gain (MaxSup vs. LS) |
|-------|----------|----------------|--------|----------------------|
| ResNet-18 | 69.09% | 69.54% | **69.96%** | +0.42 |
| ResNet-50 | 76.41% | 76.91% | **77.69%** | +0.78 |
| ResNet-101 | 75.96% | 77.37% | **78.18%** | +0.81 |
| MobileNetV2 | 71.40% | 71.61% | **72.08%** | +0.47 |
| DeiT-Small | 74.39% | 76.08% | **76.49%** | +0.41 |

MaxSup consistently outperforms LS and other variants (OLS, Zipf-LS) across all CNN and Transformer architectures.

### Ablation Study — Effect of LS Components

| Configuration | DeiT-Small Accuracy | Note |
|---------------|---------------------|------|
| Baseline (no regularization) | 74.21% | Reference |
| + Label Smoothing (full) | 75.91% | Regularization + error amplification |
| + Regularization term only | 75.98% | Slight improvement after removing error amplification |
| + Error amplification term only | 73.63% | Confirms error amplification is harmful (below baseline) |
| + Error amplification $\alpha(z_{gt}-z_{max})$ | 73.69% | Same conclusion |
| + MaxSup | **76.12%** | Best result, validating the design |

### Key Findings

- **Feature Quality Analysis**: MaxSup preserves larger intra-class distances $\bar{d}_{within}$ (0.300 vs. 0.254 for LS) while maintaining good inter-class separability $R^2$ (0.497 vs. 0.461 for LS), alleviating representation collapse.
- **Transfer Learning**: Linear probe accuracy on CIFAR-10 — MaxSup 0.810 vs. LS 0.746 vs. Logit Penalty 0.724; MaxSup nearly retains the baseline transfer capability of 0.814, whereas LS suffers substantial degradation.
- **Semantic Segmentation**: On ADE20K with UperNet + DeiT-Small, MaxSup pretraining achieves mIoU 42.8% vs. LS 42.4% vs. Baseline 42.1%.
- **Fine-Grained Classification**: On CUB-200, MaxSup achieves 82.53% vs. LS 81.96%; on Stanford Cars, 92.25% vs. 91.64%.
- **Long-Tailed Distribution**: On CIFAR-10-LT (imbalance ratio 50), MaxSup achieves 81.4% vs. LS 80.5% vs. Focal Loss 76.8%.
- **Grad-CAM Visualization** shows that MaxSup produces more focused attention on target regions, whereas LS tends to attend to irrelevant background areas.

## Highlights & Insights

- The elegance of the analysis is notable: by decomposing the LS loss into two terms in logit space, the paper reveals the dual nature of LS — beneficial when predictions are correct, harmful when they are not.
- The modification introduced by MaxSup is extremely simple (replacing $z_{gt}$ with $z_{max}$), requires no additional hyperparameters, and incurs virtually no computational cost, making it highly suitable for large-scale adoption.
- The analysis linking intra-class diversity preservation to transfer learning performance offers a new perspective for understanding feature regularization methods.

## Limitations & Future Work

- The authors acknowledge that the effect of MaxSup in **knowledge distillation** settings remains unexplored (prior work has noted that LS-trained teacher models may degrade distillation quality).
- In robustness evaluations (CIFAR-10-C), MaxSup yields slightly higher error than the CE baseline, indicating that it does not uniformly dominate on out-of-distribution data.
- In long-tailed classification, MaxSup, like LS, does not address the few-shot (minority class) problem.
- The theoretical analysis assumes that the logit ranking changes dynamically during training, but the proportion of training steps where $z_{gt} \neq z_{max}$ across different training phases is not analyzed.

## Related Work & Insights

- **Distinction from Logit Penalty**: Logit Penalty penalizes the $\ell_2$ norm of all logits (global shrinkage), whereas MaxSup penalizes only the top-1 logit (local suppression), thereby preserving greater intra-class diversity.
- **Distinction from OLS and Zipf-LS**: The latter two adjust the construction of soft labels, while MaxSup changes the penalty target position, more directly addressing the root cause of the problem.
- **Implications for Neural Collapse Research**: LS drives features toward excessive collapse, while MaxSup partially resists this tendency.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The loss decomposition analysis is insightful, though the resulting method modification is minimal.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers diverse architectures (CNN/ViT) and tasks (classification, segmentation, transfer learning, fine-grained recognition, long-tailed learning, robustness).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical derivations are clear, with rigorous logical flow from problem identification to solution design.
- **Value**: ⭐⭐⭐⭐ As a plug-and-play replacement for LS, MaxSup is simple, effective, and highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ADPretrain: Advancing Industrial Anomaly Detection via Anomaly Representation Pretraining](adpretrain_advancing_industrial_anomaly_detection_via_anomaly_representation_pre.md)
- [\[NeurIPS 2025\] The Persistence of Neural Collapse Despite Low-Rank Bias](the_persistence_of_neural_collapse_despite_low-rank_bias.md)
- [\[NeurIPS 2025\] Neural Collapse in Cumulative Link Models for Ordinal Regression: An Analysis with Unconstrained Feature Model](neural_collapse_in_cumulative_link_models_for_ordinal_regression_an_analysis_wit.md)
- [\[NeurIPS 2025\] A Generalized Label Shift Perspective for Cross-Domain Gaze Estimation](a_generalized_label_shift_perspective_for_crossdomain_gaze_e.md)
- [\[NeurIPS 2025\] Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry](reliable_active_learning_from_unreliable_labels_via_neural_collapse_geometry.md)

</div>

<!-- RELATED:END -->
