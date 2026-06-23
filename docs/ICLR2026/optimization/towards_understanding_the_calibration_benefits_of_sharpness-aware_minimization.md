---
title: >-
  [Paper Note] Towards Understanding the Calibration Benefits of Sharpness-Aware Minimization
description: >-
  [ICLR 2026][Optimization & Theory][CSAM] This paper theoretically proves that the effectiveness of Sharpness-Aware Minimization (SAM) in mitigating "overconfidence" in deep networks stems from its implicit regularization of the negative entropy of the predictive distribution (equivalent to implicit maximum entropy). Based on this, it proposes an improved vers
tags:
  - ICLR 2026
  - Optimization & Theory
  - CSAM
date: 2026-05-08
content_hash: 6e20e0d6ee33fc29
---
# Towards Understanding the Calibration Benefits of Sharpness-Aware Minimization

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=c0ERcCz6lD](https://openreview.net/forum?id=c0ERcCz6lD)  
**Code**: None  
**Area**: Optimization Theory / Model Calibration / SAM  
**Keywords**: Sharpness-Aware Minimization, Model Calibration, Maximum Entropy Regularization, Overconfidence, CSAM

## TL;DR
This paper theoretically proves that the effectiveness of Sharpness-Aware Minimization (SAM) in mitigating "overconfidence" in deep networks stems from its implicit regularization of the negative entropy of the predictive distribution (equivalent to implicit maximum entropy). Based on this, it proposes an improved version, CSAM, which specifically suppresses overconfident samples and achieves lower calibration errors than SAM and various calibration methods across multiple datasets (including ImageNet-1K).

## Background & Motivation

**Background**: Modern deep networks (ResNet, DenseNet, ViT, etc.) are increasingly used in safety-critical scenarios (autonomous driving, medical diagnosis), but they are widely known to be "poorly calibrated"—model confidence (maximum softmax value) is often much higher than true accuracy, known as **overconfidence**. An ideally calibrated model should satisfy the property: "If I say I am 80% sure, then 80% of these samples should be correct." Concurrently, SAM (Foret et al., 2021), an optimizer that pushes solutions toward flat regions of the loss surface, has gained attention for significantly improving generalization. It first climbs in the gradient direction by a perturbation radius $\rho$ to reach $\tilde\theta = \theta + \rho\,\nabla L/\|\nabla L\|$, and then performs a descent step at that perturbed point.

**Limitations of Prior Work**: Some previous works (Zheng et al., 2021; Möllenhoff & Khan, 2023) observed the phenomenon that "models trained with SAM are better calibrated," but **none have formally explained "why."** Meanwhile, specialized calibration methods have drawbacks: explicit confidence penalties like focal loss or label smoothing can hurt accuracy and limit the improvement space for post-hoc methods (temperature scaling); post-hoc methods (temperature scaling, isotonic regression) require an additional validation set and are merely remedial.

**Key Challenge**: Overconfidence mainly stems from overfitting and over-parameterization—networks push the probability of the true label toward 1 in late-stage training. Existing calibration methods either sacrifice accuracy or treat the symptoms rather than the cause; while SAM enhances both accuracy and calibration, this "side effect" lacks a mechanistic explanation and thus cannot be further utilized or enhanced.

**Goal**: (1) Provide theoretical proof of SAM's calibration benefits; (2) Characterize whether these benefits persist under out-of-distribution (OOD) shifts; (3) Design a stronger calibration optimizer based on the theory.

**Key Insight**: The authors compare the true label confidence $p_y$ and $\tilde p_y$ before and after the SAM perturbation. Intuitively, since SAM evaluates at a perturbation point (a neighborhood point with worse loss), the true label probability should be suppressed, which precisely corresponds to "preventing confidence from blindly rushing to 1."

**Core Idea**: Prove that minimizing the perturbed loss $\ell_{\tilde\theta}$ is equivalent to adding a **maximum entropy regularization** to the original loss (sharing roots with focal loss), thereby attributing SAM's calibration gain to "implicit entropy regularization"; then, amplify this regularization on overconfident samples to derive CSAM.

## Method

### Overall Architecture
The paper follows a "theory-first, method-follows" pipeline: first using two lemmas to prove that SAM perturbations exponentially suppress true label confidence, then using two theorems to translate this "confidence suppression" into "implicit maximum entropy regularization," and finally modifying the SAM outer loss to obtain CSAM based on the observation that this regularization is most active in late-stage training. The entire pipeline revolves around the derivation of the ratio of true label probabilities before and after perturbation.

Let $p_y = [f_\theta(x)]_y$ be the true label $y$ confidence under weights $\theta$, and $\tilde p_y = [f_{\tilde\theta}(x)]_y$ be the confidence under perturbed weights $\tilde\theta$. For cross-entropy loss, the single-sample loss is $\ell_\theta(z) = -\log p_y$. Calibration is measured by Expected Calibration Error (ECE): samples are grouped into $M$ bins based on top confidence, and the weighted average of the difference between average confidence and average accuracy is computed:

$$\widehat{\text{ECE}} = \sum_{i=1}^{M}\frac{|B_i|}{n}\,\big|\,\text{acc}(B_i) - \text{conf}(B_i)\,\big|.$$

The method answers why SAM-trained models keep $\text{conf}(B_i)$ close to $\text{acc}(B_i)$.

### Key Designs

**1. Perturbation exponentially suppresses true label confidence: SAM's source of "anti-overconfidence"**

Regarding why SAM prevents overconfidence, the authors prove (Lemma 1, 1-SAM case): under the condition of non-zero gradients and a bounded Hessian minimum eigenvalue $\kappa_{\min}(\nabla^2\ell_{\theta'}(z)) \ge -\|\nabla_\theta\ell(z)\|/\rho$ (holding for the interpolation point $\theta' = (1-t)\theta + t\tilde\theta,\ t\in[0,1]$), the true label confidence at the perturbed point is multiplicatively compressed:

$$\tilde p_y \le e^{-\rho\|\nabla_\theta\ell(z)\|/2}\,p_y.$$

That is, $\tilde p_y$ **decays exponentially** with perturbation radius $\rho$ and gradient norm $\|\nabla_\theta\ell(z)\|$. Samples that are more "confident with large gradients" are suppressed most heavily. Lemma 2 extends this to m-SAM using the geometric mean of probabilities.

**2. SAM is equivalent to implicit maximum entropy regularization: Shared roots with focal loss without losing accuracy**

Theorem 1 shows for 1-SAM: Let $\lambda = (1-\tilde p_y)/(1-p_y)$, then

$$\ell_{\tilde\theta}(z) \ge \ell_\theta(z) - \lambda H(p_y) + H(\tilde p_y),$$

where $H(p) = -p\log p - (1-p)\log(1-p)$ is the binary entropy. Since $\lambda > 1$, minimizing $\ell_{\tilde\theta}$ **prioritizes maximizing $H(p_y)$** (coefficient $\lambda$ is larger than the coefficient 1 for suppressing $H(\tilde p_y)$): when $p_y$ is near 1, it is pulled down, and when near 0, it is pushed up, performing maximum entropy regularization. Theorem 2 generalizes this to m-SAM. The penalty is **stronger in late-stage training** (when $\tilde p_y$ is high), explaining why SAM excels in calibrating architectures prone to overconfidence without sacrificing accuracy.

**3. CSAM: Amplifying entropy regularization on overconfident samples**

The authors observe that SAM mainly starts punishing the distribution in late training when $\tilde p_y$ is already high. CSAM is proposed to make overconfident samples "appear" even more confident to trigger stronger entropy penalties by modifying the outer (descent step) loss:

$$\tilde\ell_{\tilde\theta}(z) = \begin{cases} -\log\tilde p_y, & \tilde p_y \le 1/2,\\ -(1+\tilde p_y)^{-\gamma}\log\tilde p_y, & \text{otherwise,}\end{cases}$$

where $0\le\gamma\le 2$ is a hyperparameter. Theorem 3 proves that for $\tilde p_y > 1/2$,

$$\tilde\ell_{\tilde\theta}(z) \ge \ell_\theta(z) - \lambda H(p_y) + (1-\gamma/2)H(\tilde p_y).$$

Compared to Theorem 1, the implicit penalty on $H(p_y)$ is further amplified by $(1-\gamma/2)$. CSAM applies extra force only on overconfident samples ($\tilde p_y > 1/2$) and maintains standard SAM for others.

### Loss & Training
Cross-entropy (CE) is the default loss. CSAM modifies only the per-sample loss in the SAM outer descent step with $\gamma\in\{0.5, 1.0, 2.0\}$. Perturbation radius $\rho$ follows standard settings: 0.05 for CIFAR-10 and ImageNet ResNet, 0.2 for CIFAR-100 and ImageNet ViT. Base optimizers are SGD (0.9 momentum) or AdamW with cosine learning rate decay.

## Key Experimental Results

### Main Results

ImageNet-1K (ID metrics + ImageNet-C OOD metrics; TCE is ECE after temperature scaling):

| Model | Method | Test Acc ↑ | ECE ↓ | TCE ↓ | AdaECE ↓ |
|------|------|-----------|-------|-------|----------|
| ResNet-50 | SGD | 76.97 | 3.39 | 1.80 | 3.31 |
| ResNet-50 | SAM | 77.32 | 1.52 | 1.54 | 1.44 |
| ResNet-50 | **CSAM** | **77.95** | **1.18** | **1.09** | **1.19** |
| ViT-S/16 | AdamW | 71.35 | 9.72 | 3.66 | 9.72 |
| ViT-S/16 | SAM | 75.42 | 1.76 | 1.66 | 1.73 |
| ViT-S/16 | **CSAM** | **75.91** | **1.58** | **1.34** | **1.54** |

Comparison with various calibration methods on CIFAR-10 (WideResNet-28-10):

| Method | Test Acc ↑ | ECE ↓ | AdaECE ↓ | TCE ↓ |
|------|-----------|-------|----------|-------|
| CE | 95.83 | 2.36 | 2.04 | 1.06 |
| Focal Loss | 95.91 | 1.16 | 1.42 | 1.01 |
| AdaFocal | 95.78 | 0.91 | 0.65 | 0.97 |
| MIMO | 95.96 | 0.88 | 0.73 | 0.74 |
| bSAM | 96.45 | 1.82 | 1.78 | 0.70 |
| SAM | 96.91 | 0.86 | 0.84 | 0.52 |
| **CSAM** | **96.97** | **0.50** | **0.48** | **0.47** |

### Ablation Study

OOD (ResNet-18 / CIFAR-10 training, transferred to SVHN, CIFAR-10/100-C):

| Optimizer | Config | ECE ↓ | Note |
|--------|------|-------|------|
| SGD | Vanilla | 5.76 | Worst overconfidence |
| SAM | Vanilla | 3.24 | ~1.8x better than SGD |
| CSAM | Vanilla | 2.55 | Further reduction |
| SGD | Ensemble | 1.84 | Ensemble reduces ECE |
| SAM | Ensemble | 1.09 | Ensemble effective for SAM |
| CSAM | Ensemble | **0.86** | Best results |

### Key Findings
- **SAM's uncalibrated ECE is often lower than SGD's ECE after temperature scaling**: This suggests SAM inherently produces reliable predictions.
- **Calibration benefits persist under OOD**: SGD's ECE is roughly double that of SAM under OOD, while SAM also generalizes better.
- **CSAM achieves the lowest calibration error among all baselines**: Methods like focal loss often sacrifice accuracy, whereas CSAM maintains or slightly improves accuracy while minimizing ECE.
- **Architectures more prone to overconfidence benefit more from SAM**: ViT's ECE drops from ~10% to ~1.5%, catching up to ResNet levels.

## Highlights & Insights
- **Reduction of "optimizer side effects" to a clean regularization term**: Deriving the relationship between perturbation and entropy is elegant.
- **Improvements rooted in theoretical gaps**: Finding that entropy penalty is late-stage leads to the design of CSAM's weighted penalty at $\tilde p_y > 1/2$.
- **The role of gradient norm**: Lemma 1 reveals that larger gradient norms make SAM more effective, matching empirical observations in ViT+AdamW settings.

## Limitations & Future Work
- **Theory depends on boundedness assumptions**: The bound on the Hessian's minimum eigenvalue may not hold for all interpolation points.
- **Coverage only for Cross-Entropy**: Analyses are primarily based on CE loss.
- **Introduction of hyperparameter $\gamma$**: Although restricted to $[0, 2]$, it still requires selection.
- **Future directions**: Replacing assumptions with trajectory-level conditions and exploring adaptive $\gamma$ tuning.

## Related Work & Insights
- **vs Focal Loss / AdaFocal**: These explicitly penalize confident samples but often at the cost of accuracy; SAM/CSAM performs implicit entropy regularization while maintaining generalization.
- **vs bSAM (Bayesian SAM)**: bSAM introduces many hyperparameters and is harder to tune; CSAM is simpler and more effective for calibration.
- **vs Existing SAM Theories**: Most focus on generalization and Hessian spectra; this work fills the gap by connecting SAM with **model calibration** through entropy regularization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First theoretical explanation of SAM's calibration through entropy regularization.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive ID/OOD baselines, though focused on CE loss.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivations and strong correspondence between theory and figures.
- Value: ⭐⭐⭐⭐ Provides a "calibration during training" solution for safety-critical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[ICLR 2026\] MASAM: Multimodal Adaptive Sharpness-Aware Minimization for Heterogeneous Data Fusion](masam_multimodal_adaptive_sharpness-aware_minimization_for_heterogeneous_data_fu.md)
- [\[ICLR 2026\] Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization](minor_first_major_last_a_depth-induced_implicit_bias_of_sharpness-aware_minimiza.md)
- [\[ICML 2026\] Stability Analysis of Sharpness-Aware Minimization](../../ICML2026/optimization/stability_analysis_of_sharpness-aware_minimization.md)
- [\[ICLR 2026\] Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models](bi-lora_efficient_sharpness-aware_minimization_for_fine-tuning_large-scale_model.md)
- [\[ICML 2025\] Tilted Sharpness-Aware Minimization](../../ICML2025/optimization/tilted_sharpness-aware_minimization.md)

</div>

<!-- RELATED:END -->
