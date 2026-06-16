---
title: >-
  [Paper Note] Enhancing Out-of-Distribution Detection with Extended Logit Normalization
description: >-
  [CVPR 2026][AI Safety][Paper Note] This paper identifies that LogitNorm leads to two types of feature collapse (dimensional collapse and origin collapse) during training. It proposes a hyperparameter-free Extended Logit Normalization (ELogitNorm), which replaces the distance-to-origin scaling factor with the distance from features to decision boundaries
tags:
  - CVPR 2026
  - AI Safety
date: 2026-05-08
content_hash: 8875496a0979f25b
---
# Enhancing Out-of-Distribution Detection with Extended Logit Normalization

**Conference**: CVPR 2026  
**arXiv**: [2504.11434](https://arxiv.org/abs/2504.11434)  
**Code**: [https://github.com/limchaos/ElogitNorm](https://github.com/limchaos/ElogitNorm)  
**Area**: AI Safety  
**Keywords**: OOD Detection, Logit Normalization, Feature Collapse, Decision Boundary, Model Calibration

## TL;DR
This paper identifies that LogitNorm leads to two types of feature collapse (dimensional collapse and origin collapse) during training. It proposes a hyperparameter-free Extended Logit Normalization (ELogitNorm), which replaces the distance-to-origin scaling factor with the distance from features to decision boundaries. This significantly enhances the performance of various post-hoc OOD detection methods and improves confidence calibration without compromising classification accuracy.

## Background & Motivation
Out-of-Distribution (OOD) detection is critical for the safe deployment of machine learning models. Existing research focuses either on designing post-hoc scoring functions (MSP, KNN, SCALE, etc.) or modifying training losses to improve OOD discriminability. LogitNorm is a representative training-time method that alleviates overconfidence by normalizing the logit vectors.

However, LogitNorm suffers from three limitations: (1) it causes **feature collapse**, where feature variance concentrates in few directions and OOD samples cluster near the origin; (2) it sacrifices classification accuracy for OOD performance; and (3) it is only effective for limited scoring functions, sometimes even degrading performance when combined with certain post-hoc methods.

The core insight of this paper is that the normalization factor $\tau\|\mathbf{f}\|$ in LogitNorm is essentially equivalent to scaling by the distance to the origin $\|\mathbf{z}\|$ (since $\|\mathbf{f}\| \approx \bar{\sigma}\|\mathbf{z}\| + \eta$), which encourages features to collapse toward the origin. A more principled approach is to use the **distance from features to decision boundaries** $\mathcal{D}(\mathbf{z})$ as the scaling factor—samples near boundaries exhibit higher uncertainty, while those far from boundaries are more reliably classified.

## Method

### Overall Architecture
This paper addresses the "side effects" of LogitNorm as a training-time OOD method: while it mitigates overconfidence, it degrades the feature space. The overall architecture of ELogitNorm is lightweight—it maintains the network architecture (e.g., ResNet-18/50) and only replaces the standard cross-entropy loss $\mathcal{L}_{CE}$ with $\mathcal{L}_{ELogitNorm}$. The key difference lies in the logit scaling factor: instead of using the logit norm $\tau\|\mathbf{f}\|$ as in LogitNorm, this method utilizes the distance to the decision boundary $\mathcal{D}(\mathbf{z})$. Once trained, the healthier feature distribution can be seamlessly integrated with any post-hoc OOD scoring method (MSP, KNN, SCALE, etc.) without task-specific customization.

### Key Designs

**1. Feature Collapse Diagnosis: Identifying the issues in LogitNorm**

The motivation for ELogitNorm is built on the observation that LogitNorm crushes the feature space into two types of collapse while suppressing overconfidence. First is **dimensional collapse**: Singular Value Decomposition (SVD) of features trained with LogitNorm reveals many near-zero singular values, indicating that effective feature dimensions are heavily compressed. Second is **origin collapse**: while OOD samples naturally tend to cluster near the origin, LogitNorm further pushes them toward it. The paper explains this via Proposition 1: the logit norm is approximately proportional to the feature norm, $\sigma_{min}\|\mathbf{z}\| - \|\mathbf{b}\| \leq \|\mathbf{f}\| \leq \sigma_{max}\|\mathbf{z}\| + \|\mathbf{b}\|$. Thus, using $\|\mathbf{f}\|$ as a scaling factor implicitly constrains the "distance to origin $\|\mathbf{z}\|$," which is the root cause of the collapse.

**2. Decision Boundary Distance Scaling: Changing the scaling "anchor" from the origin to the boundary**

Since scaling by "distance to origin" is problematic, the paper proposes a more discriminative anchor: the distance from features to the decision boundaries of competing classes. The intuition is clear—samples near decision boundaries are inherently uncertain and prone to misclassification, while samples far from them are more reliable. Let $f_{max}$ be the predicted class index; the scaling factor is defined as the average point-to-plane distance from the feature to all other class boundaries:

$$\mathcal{D}(\mathbf{z}) = \frac{1}{c-1}\sum_{i \neq f_{max}} \frac{|(\mathbf{w}_{f_{max}} - \mathbf{w}_i)^T\mathbf{z} + (b_{f_{max}} - b_i)|}{\|\mathbf{w}_{f_{max}} - \mathbf{w}_i\|_2}$$

The training loss is the cross-entropy with logits scaled by this distance instead of a temperature: $\mathcal{L}_{ELogitNorm} = -\log \frac{e^{f_y/\mathcal{D}(\mathbf{z})}}{\sum_i e^{f_i/\mathcal{D}(\mathbf{z})}}$. Consequently, "fuzzy" samples near the boundary (small $\mathcal{D}(\mathbf{z})$) receive larger effective logits and stronger gradient signals, forcing them away from the boundary, rather than indiscriminately pulling all samples toward the origin.

**3. Minimum Scaling Factor Space Analysis: Geometric explanation for avoiding collapse**

The paper provides a geometric explanation in Proposition 2 by comparing the size of the set of features where the two scaling factors reach their minimum values. The scaling factor of LogitNorm is minimized at the origin, which is a **0-dimensional point**, forcing optimization to shrink features toward this single point and causing collapse. In contrast, the scaling factor of ELogitNorm is minimized at the intersection of all decision boundaries, which is an $m-c+1$ dimensional affine subspace (e.g., 503 dimensions for ResNet-18 on CIFAR-10 vs. 0 dimensions for LogitNorm). This jump in dimensionality allows features sufficient "freedom" to spread out, which is why ELogitNorm avoids dimensional collapse and maintains a more uniform distribution.

### Loss & Training
The primary loss function is $\mathcal{L}_{ELogitNorm}$, which has **no additional hyperparameters**. This is a practical advantage over LogitNorm, which requires tuning the temperature $\tau$ on a validation set, whereas the boundary distance $\mathcal{D}(\mathbf{z})$ is adaptively calculated from weights and features. Other training settings follow standard cross-entropy: ResNet-18 on CIFAR for 100 epochs, SGD, lr=0.1, momentum=0.9, and weight decay $5 \times 10^{-4}$.

## Key Experimental Results

### Main Results

| Dataset (ID) | Scoring Method | Metric | Cross-Entropy | LogitNorm | Ours | Gain |
|-----------|---------|------|---------------|-----------|------------|------|
| CIFAR-10 | SCALE | far-OOD AUROC | 86.46 | — | **96.94** | +10.48 |
| CIFAR-10 | SCALE | far-OOD FPR95 | 67.49 | — | **13.18** | -54.31 |
| CIFAR-10 | MSP | far-OOD AUROC | 90.73 | 96.74 | **96.68** | +5.95 |
| ImageNet-1K | MSP | far-OOD AUROC | 85.23 | 91.54 | **93.19** | +7.96 |
| ImageNet-1K | MSP | far-OOD FPR95 | 51.45 | 31.32 | **27.74** | -23.71 |
| ImageNet-200 | KNN | far-OOD AUROC | 93.16 | — | **96.08** | +2.92 |

### Ablation Study

| Configuration | ECE(%) ↓ | Description |
|------|---------|------|
| Cross-Entropy + Original logit | 3.3 | Baseline calibration |
| LogitNorm + $\mathbf{f}/(\tau\|\mathbf{f}\|)$ | 4.1 | Best config for LogitNorm |
| ELogitNorm + $\mathbf{f}/\mathcal{D}(\mathbf{z})$ | **1.8** | Optimal calibration, lowest ECE |
| LogitNorm Accuracy (CIFAR-10) | 94.83 | Lower than Cross-Entropy (95.10) |
| Ours Accuracy (CIFAR-10) | **95.11** | Comparable to or better than CE |
| Ours Accuracy (ImageNet-200) | **87.12** | Outperforms Cross-Entropy (86.58) |

### Key Findings
- ELogitNorm shows the most significant improvement in far-OOD scenarios, reducing the FPR95 of the SCALE method from 67.49% to 13.18%.
- Unlike LogitNorm, ELogitNorm is compatible with all post-hoc methods (LogitNorm+ReAct causes severe degradation).
- Singular value spectrum analysis confirms that ELogitNorm achieves a more uniform feature distribution and prevents dimensional collapse.
- The hyperparameter-free design makes the method easier to deploy without requiring a validation set for temperature tuning.

## Highlights & Insights
- The diagnostic perspective on feature collapse is novel: linking LogitNorm's normalization factor to the distance to the origin in feature space reveals the implicit collapse mechanism.
- Proposition 2 provides an elegant geometric explanation for why distance to the decision boundary is superior to distance to the origin.
- The hyperparameter-free design is a significant practical advantage: while LogitNorm requires tuning $\tau$, ELogitNorm is fully adaptive.

## Limitations & Future Work
- Gains in near-OOD scenarios are relatively limited, which the authors acknowledge as a shared challenge for all training-time methods.
- Calculating the decision boundary distance involves all $c-1$ planes, which may increase overhead when the number of classes is very large (e.g., 1000 classes in ImageNet-1K), though the authors claim an efficient implementation.
- The method has not been validated on Transformer architectures like ViT.

## Related Work & Insights
- Compared to methods specifically designed for KNN scoring such as CIDER and NPOS, ELogitNorm achieves better results more simply (ImageNet-200 far-OOD AUROC: 96.08 vs 94.83/90.66).
- The concept of decision-boundary-aware distance can be extended to other areas such as uncertainty estimation and domain adaptation.
- The unified perspective of adaptive temperature scaling ($s = \tau\|\mathbf{f}\|$ vs $s = \mathcal{D}(\mathbf{z})$) provides a framework for designing improved calibration losses.

## Rating
- Novelty: ⭐⭐⭐⭐ The diagnosis of feature collapse and the motivation for decision boundary scaling are strong, though the core technical change is relatively small.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation using the OpenOOD framework across 4 ID datasets, 6 post-hoc methods, 3 repetitions, and assessment of both calibration and accuracy.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical analysis and clear illustrations, though some formula repetitions are slightly verbose.
- Value: ⭐⭐⭐⭐ Highly practical for the OOD detection community; the hyperparameter-free design lowers the barrier to entry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RankOOD: Class Ranking-based Out-of-Distribution Detection](rankood_-_class_ranking-based_out-of-distribution_detection.md)
- [\[CVPR 2026\] Sparsity as a Key: Unlocking New Insights from Latent Structures for Out-of-Distribution Detection](sparsity_as_a_key_unlocking_new_insights_from_latent_structures_for_out-of-distr.md)
- [\[CVPR 2026\] Bypassing the Transport Plan: Dynamic Reweighting for Out-of-Distribution Detection with Optimal Transport](bypassing_the_transport_plan_dynamic_reweighting_for_out-of-distribution_detecti.md)
- [\[NeurIPS 2025\] Revisiting Logit Distributions for Reliable Out-of-Distribution Detection](../../NeurIPS2025/ai_safety/revisiting_logit_distributions_for_reliable_out-of-distribution_detection.md)
- [\[CVPR 2026\] Learning Latent Concepts for Detecting Out-of-Distribution Objects](learning_latent_concepts_for_detecting_out-of-distribution_objects.md)

</div>

<!-- RELATED:END -->
