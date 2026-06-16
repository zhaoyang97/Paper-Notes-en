---
title: >-
  [Paper Note] Confusion-Aware Spectral Regularizer for Long-Tailed Recognition
description: >-
  [CVPR 2026][Others][Paper Note] This paper demonstrates that "worst-class error" in long-tailed scenarios is tightly upper-bounded by the spectral norm of the frequency-weighted confusion matrix. Consequently, it proposes CAR, a regularizer that directly minimizes this spectral norm using a differentiable confusion matrix proxy and an EMA estimator.
tags:
  - CVPR 2026
  - Others
date: 2026-05-08
content_hash: 7bc8309e3f697284
---
# Confusion-Aware Spectral Regularizer for Long-Tailed Recognition

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zhu_Confusion-Aware_Spectral_Regularizer_for_Long-Tailed_Recognition_CVPR_2026_paper.html)  
**Code**: https://github.com/misswayguy/CAR  
**Area**: Long-Tailed Recognition  
**Keywords**: Long-Tailed Recognition, Worst-class Generalization, Confusion Matrix Spectral Norm, PAC-Bayes Upper Bound, Differentiable Regularizer  

## TL;DR
This paper demonstrates that "worst-class error" in long-tailed scenarios is tightly upper-bounded by the spectral norm of the frequency-weighted confusion matrix. Consequently, it proposes CAR, a regularizer that directly minimizes this spectral norm using a differentiable confusion matrix proxy and an EMA estimator. CAR improves worst-class accuracy by 6%~10% and overall accuracy by 2.4%~4.8% over previous SOTA on benchmarks like ImageNet-LT, CIFAR100-LT, and iNaturalist.

## Background & Motivation
**Background**: Mainstream approaches to long-tailed recognition follow three lines: data-level resampling (over/under-sampling), loss-level reweighting and logit adjustment (CB, Focal, LDAM, logit adjustment), and model-level decoupled learning (separating feature and classifier training). These methods aim to compensate for the suppression of tail classes (few samples) by head classes (many samples).

**Limitations of Prior Work**: The authors observe two long-neglected "gaps" using ViT-Small on ImageNet-LT: ① The **test** accuracy of the worst-performing class is significantly lower than the overall test accuracy; ② The test accuracy of the worst class is far lower than its own **training** accuracy (many SOTA methods achieve >90% training accuracy for the worst class, yet test accuracy is near 0%). Thus, the model fits the training samples of the worst classes well but fails to generalize. Existing methods focusing on "overall accuracy" and "sample frequency" are largely ineffective against this generalization gap.

**Key Challenge**: The core difficulty in long-tailed problems is not merely the "lack of tail samples," but the "poor training-to-test generalization of the worst classes." Methods like frequency reweighting or resampling adjust the distribution of empirical risk but do not directly constrain the **inter-class confusion structure**, which is the direct source of generalization collapse in the worst classes.

**Goal**: (1) Theoretically derive a metric that characterizes worst-class generalization; (2) Design a differentiable, batch-computable regularizer to directly optimize this metric during training.

**Key Insight**: The authors shift the perspective from "sample frequency" to the "confusion spectrum." They introduce a frequency-weighted measure of worst-class error and derive, based on the PAC-Bayes framework, that the worst-class error is tightly upper-bounded by the **spectral norm of the frequency-weighted confusion matrix** plus a model complexity term. Since the error is controlled by the spectral norm, it is minimized directly during training.

**Core Idea**: Introducing a regularizer that minimizes the spectral norm of the frequency-weighted confusion matrix to specifically improve worst-class generalization, replacing traditional frequency reweighting.

## Method
CAR does not modify the network architecture or sampling strategy; it adds a regularizer alongside standard cross-entropy. The method consists of a theoretical quantity and two engineering components for optimization. The theoretical quantity is the **frequency-weighted worst-class error**, dominated by the spectral norm $\|C^f_{S,\gamma}\Lambda\|_2$. Since this matrix is non-differentiable and cannot be computed over the full dataset at each step, a **differentiable confusion matrix proxy** and an **EMA confusion estimator** are introduced. The final training objective is "Cross-Entropy + $\alpha \cdot$ Spectral Regularization."

### Key Designs

**1. Frequency-Weighted Worst-Class Error and Spectral Norm Bound: Converting "Worst-Class Generalization" into an Optimizable Scalar**

This is the theoretical foundation. The authors define the (off-diagonal) confusion matrix $C^f_D$, where $c_{ij}=P(\hat y(x)=i\mid y=j)$ and the diagonal is set to 0. The column sum $\sum_i c_{ij}$ represents the conditional error rate of class $j$. To "amplify" rare classes, a class-level weight $\lambda_j=(m_j+r_0)^{-1/2}$ is introduced ($m_j$ is the relative frequency, $r_0>0$ is a smoothing factor), forming a diagonal matrix $\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_K)$. The **frequency-weighted worst-class error** is defined as

$$\mathrm{WCE}(f)=\|C^f_D\Lambda\|_1=\max_j \lambda_j\sum_i c_{ij}.$$

Classes with fewer samples have larger $\lambda_j$, causing their conditional errors to be weighted more heavily. Based on PAC-Bayes, the authors prove that any class error $e_j$ is bounded by this weighted worst-class error, which is further decomposed as:

$$e_j\le \frac{1}{\lambda_j}\|C^f_D\Lambda\|_1\le \underbrace{\frac{\nu}{\lambda_j}\|C^f_{S,\gamma}\Lambda\|_2}_{\text{Empirical Spectral Norm}}+\underbrace{\mathcal{E}(f,S,\gamma,\delta)}_{\text{Complexity Term}}.$$

Here, $C^f_{S,\gamma}$ is the empirical confusion matrix with margin $\gamma$, and $\nu$ is a constant. This bound is tight for the worst class. **The unique contribution of this work is identifying that the spectral norm of the confusion matrix itself is a controllable and complementary term** to the weight spectral norm. Minimizing $\|C^f_{S,\gamma}\Lambda\|_2$ directly tightens the worst-class error bound.

**2. Differentiable Confusion Matrix Proxy: Replacing Indicators with Soft Gates and Soft Argmax**

The term $R(f)=\|C^f_{S,\gamma}\Lambda\|_2$ cannot be optimized directly because its elements

$$\hat c^\gamma_{ij}=\frac{1}{m_j}\sum_{q:y_q=j}\mathbf{1}\!\left[f_w(x_q)[y_q]\le\gamma+f_w(x_q)[i]\right]\cdot\mathbf{1}\!\left[\arg\max_{i'\ne y_q}f_w(x_q)[i']=i\right]$$

involve non-differentiable indicator functions. The first indicator checks if the score of class $i$ approaches the ground truth $j$, and the second checks if $i$ is the strongest competitor. The authors use smooth functions to create a differentiable proxy:

$$\tilde c_{ij}=\frac{1}{m_j}\sum_{q:y_q=j}\underbrace{\sigma\!\big(\gamma+f_w(x_q)[i]-f_w(x_q)[j]\big)}_{\text{Soft Margin Gate}}\times\underbrace{S\!\big(f_w(x_q)-f_w(x_q)[j]\big)[i]}_{\text{Soft Argmax over non-}j}.$$

$\sigma$ is the sigmoid function, and $S$ is the softmax function. This differentiable proxy $\tilde c_{ij}$ allows the spectral norm regularization to backpropagate to the network parameters $w$.

**3. EMA Confusion Estimator: Stabilizing High-Variance Batch Estimates via Momentum**

Computing $C^f_{S,\gamma}$ on the full training set at each step is infeasible. Instead of using a high-variance batch-level confusion matrix, the authors maintain an Exponential Moving Average (EMA) of the differentiable batch estimates:

$$\hat C_t=\beta\,\hat C_{t-1}+(1-\beta)\,\tilde C^f_{B_t,\gamma},\quad \hat C_0=0.$$

A key trick is that **only the current term $\tilde C^f_{B_t,\gamma}$ carries gradients for $w$, while the historical term $\hat C_{t-1}$ is treated as a constant (stop-gradient)**. This reduces estimation variance without breaking differentiability. The final objective is:

$$L(f)=\frac{1}{m}\sum_{q=1}^{m}\mathrm{CE}\big(f(x_q),y_q\big)+\alpha\,\big\|\hat C_t\Lambda\big\|_2,$$

where $\alpha>0$ controls regularization strength.

### Loss & Training
The final loss is $L(f)=\mathrm{CE}+\alpha\|\hat C_t\Lambda\|_2$. The backbone used is ViT-Small (validated on Tiny/Base/Large, ResNet, and Swin) with AdamW and a batch size of 128. Training lasts 200 epochs for CIFAR100-LT/ImageNet-LT and 300 epochs for iNaturalist. Pre-training fine-tuning is conducted for 100 epochs. Key hyperparameters include EMA momentum $\beta=0.5$, smoothing factor $r_0=0.2$, and regularization weight $\alpha\approx0.5$.

## Key Experimental Results

### Main Results (Training from scratch, ViT-Small, Top-1 %)
CAR independently outperforms existing methods and achieves state-of-the-art results when combined with ConCutMix.

| Method | ImageNet-LT Tail | ImageNet-LT Overall | CIFAR100-LT Overall | iNaturalist Overall |
|------|------|------|------|------|
| CE | 16.30 | 46.51 | 41.40 | 59.56 |
| LDAM-DRW (NeurIPS'19) | 25.90 | 50.39 | 45.40 | 65.57 |
| GML (CVPR'23) | 32.17 | 55.24 | 50.23 | 70.85 |
| LOS (ICLR'25, Prev. SOTA) | 32.73 | 56.20 | 50.85 | 71.01 |
| **CAR (Ours)** | 35.77 | 57.48 | 51.85 | 71.56 |
| **CAR + ConCutMix** | **38.07** | **60.07** | **55.68** | **73.38** |

Compared to the previous SOTA (LOS), the overall accuracy increases by 2.37%~4.83%, and tail accuracy increases by 3.28%~7.98%.

### Worst-class Generalization (ViT-Small, Worst-class Accuracy / WR=Test/Training)
CAR significantly improves the test accuracy and generalization ratio ($WR$) of the worst classes.

| Method | ImageNet-LT Test(%) | ImageNet-LT WR | CIFAR100-LT Test(%) | CIFAR100-LT WR |
|------|------|------|------|------|
| SAFA (ECCV'22) | 10 | 0.11 | 8 | 0.09 |
| LOS (ICLR'25) | 10 | 0.11 | 8 | 0.09 |
| **CAR (Ours)** | 18 | 0.19 | 14 | 0.15 |
| **CAR + ConCutMix** | **22** | **0.23** | **18** | **0.19** |

### Ablation Study (ViT-Small, Overall Top-1 %)
Both components—frequency weighting ($\Lambda$) and the EMA estimator—contribute to performance gains.

| Configuration | ImageNet-LT | CIFAR100-LT | Description |
|------|------|------|------|
| w/o $\Lambda$ | 54.39 | 49.62 | Without weighting, gradients bias toward head classes. |
| w/ $\Lambda$ | 57.48 | 51.85 | Frequency-aware weighting balances the gradient. |
| w/o EMA | 55.77 | 50.20 | Batch estimation has high variance, causing instability. |
| w/ EMA | 57.48 | 51.85 | Momentum updates smooth the confusion estimation. |

### Key Findings
- **Worst-class generalization is a primary bottleneck**: Previous SOTA training accuracy was high (>90%) while test accuracy was low (~10%). CAR improves test accuracy to 14%~22%, proving it addresses "generalization" rather than "fitting."
- **Components are indispensable**: Removing $\Lambda$ drops performance by 3.09 points on ImageNet-LT; removing EMA drops it by 1.71 points.
- **Orthogonal to data augmentation**: CAR achieves consistent gains when combined with ReMix, MetaSAug, CMO, SAFA, and ConCutMix.
- **Backbone-agnostic**: Consistent improvements are observed across ViT-Tiny/Base/Large, ResNet, and Swin.
- **Visualization evidence**: The confusion matrices under CAR show significantly lower off-diagonal responses compared to BALMS.

## Highlights & Insights
- **Formalizing worst-class generalization**: The paper identifies a neglected diagnostic phenomenon and converts it into an optimizable theoretical quantity via the PAC-Bayes framework.
- **Spectrum perspective**: It argues that the spectral norm of the confusion matrix is a controllable term complementary to traditional weight spectral normalization.
- **Clean differentiable proxy**: The decomposition of the non-differentiable indicator into a soft margin gate and soft argmax is an elegant engineering solution.
- **EMA with stop-gradient**: This approach for estimating global batch statistics is highly effective for reducing variance while maintaining differentiability.

## Limitations & Future Work
- **Theoretical assumptions**: The PAC-Bayes bound depends on specific assumptions (ReLU, feedforward networks), and the synergy between the confusion spectral norm and the weight spectral norm is primarily empirical.
- **Absolute accuracy**: Worst-class test accuracy remains relatively low (18%~22%), suggesting that long-tailed generalization is far from solved.
- **Hyperparameter tuning**: While reported as insensitive, optimal values for $\alpha, \gamma, r_0$, and $\beta$ were largely derived from CIFAR100-LT.
- **Cold start**: The EMA estimator starts at $\hat C_0=0$, which may lead to biased estimates in early training phases.
- **Task scope**: The experiments are limited to image classification; effectiveness in detection or segmentation remains to be seen.

## Related Work & Insights
- **Comparison with Resampling/Reweighting**: These methods adjust sample importance or logit bias; CAR instead constrains the inter-class confusion structure.
- **Comparison with Decoupled Learning**: Decoupled methods balance features and classifiers in stages; CAR is a single-stage, end-to-end regularizer that is compatible with them.
- **Comparison with Spectral Normalization**: While both use spectral norms, spectral normalization controls network weight complexity, whereas CAR controls the confusion spectral norm to minimize worst-class error.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Ties worst-class generalization to confusion matrix spectra and provides a differentiable optimization path.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across benchmarks, backbones, and augmentations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and logic, although some minor notation issues exist.
- Value: ⭐⭐⭐⭐ A plug-and-play, backbone-agnostic regularizer with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AdaPrior: Bayesian-Inspired Adaptive Prior Correction for Long-Tailed Continual Learning](adaprior_bayesian-inspired_adaptive_prior_correction_for_long-tailed_continual_l.md)
- [\[CVPR 2026\] Spectral Mixture-of-Experts for Continual Learning](spectral_mixture-of-experts_for_continual_learning.md)
- [\[CVPR 2025\] TAET: Two-Stage Adversarial Equalization Training on Long-Tailed Distributions](../../CVPR2025/others/taet_two-stage_adversarial_equalization_training_on_long-tailed_distributions.md)
- [\[CVPR 2026\] Multi-Hierarchical Contrastive Spectral Fusion for Multi-View Clustering](multi-hierarchical_contrastive_spectral_fusion_for_multi-view_clustering.md)
- [\[CVPR 2026\] Adaptive Data Augmentation with Multi-armed Bandit: Sample-Efficient Embedding Calibration for Implicit Pattern Recognition](adaptive_data_augmentation_with_multi-armed_bandit_sample-efficient_embedding_ca.md)

</div>

<!-- RELATED:END -->
