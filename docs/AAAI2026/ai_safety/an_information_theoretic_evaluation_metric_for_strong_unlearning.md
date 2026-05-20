---
title: >-
  [Paper Note] An Information Theoretic Evaluation Metric for Strong Unlearning
description: >-
  [AAAI 2026][AI Safety][Machine Unlearning] This paper exposes a fundamental flaw in existing black-box unlearning evaluation metrics (MIA, JSD…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Machine Unlearning"
  - "Mutual Information"
  - "White-Box Evaluation"
  - "IDI Metric"
  - "COLA Method"
date: 2026-05-08
content_hash: 3e431baec5717825
---

# An Information Theoretic Evaluation Metric for Strong Unlearning

**Conference**: AAAI 2026
**arXiv**: [2405.17878](https://arxiv.org/abs/2405.17878)  
**Code**: To be confirmed  
**Area**: AI Safety / Machine Unlearning
**Keywords**: Machine Unlearning, Mutual Information, White-Box Evaluation, IDI Metric, COLA Method

## TL;DR

This paper exposes a fundamental flaw in existing black-box unlearning evaluation metrics (MIA, JSD, etc.)—modifying only the final classification head is sufficient to satisfy all black-box metrics while intermediate layers fully retain information about the forget set. The paper proposes IDI, a white-box metric that quantifies unlearning effectiveness by estimating, via InfoNCE, the mutual information between each layer's representations and the forget labels. It further proposes COLA, an unlearning method that achieves IDI scores approaching Retrain on CIFAR-10/100 and ImageNet-1K.

## Background & Motivation

- **Background**: Machine Unlearning (MU) aims to remove the influence of specific data from trained models, satisfying regulatory requirements such as the "right to be forgotten." Ideal strong unlearning requires that the unlearned model be indistinguishable from one retrained from scratch without the forget data. However, full retraining is prohibitively expensive, shifting research focus toward approximate unlearning methods.
- **Limitations of Prior Work**: Existing evaluation relies primarily on black-box metrics—Membership Inference Attacks (MIA) and accuracy comparisons—which examine only model outputs and fail to capture residual forget-data information in intermediate layers. More critically, no reliable white-box metric exists to verify strong unlearning.
- **Key Challenge**: The paper reveals the fundamental problem through a simple experiment: Head Distillation (HD) modifies only the final classification head (freezing all encoder layers) and uses distillation to match the output distribution of a retrained model, achieving strong performance on all black-box metrics (with the best MIA score). Yet the encoder is identical to the original model, retaining 100% of the forget data's information. This demonstrates that black-box metrics cannot assess strong unlearning.
- **Key Insight**: Motivated by the information bottleneck principle (deeper layers in DNNs compress information progressively), the paper quantifies residual information via mutual information between each layer's features and the forget labels—if unlearning succeeds, intermediate-layer mutual information should approach that of the retrained model.

## Method

### Overall Architecture

The paper makes two core contributions: (1) **IDI (Information Difference Index)**, a white-box metric that estimates layer-wise mutual information between model features and forget labels, computing the proportion of information removed relative to the original model; and (2) **COLA (COLapse-and-Align)**, an unlearning method that first collapses forget-set features to make them indistinguishable, then aligns retain-set features to recover task performance.

### Key Designs

1. **Mutual Information Estimation for IDI**
    - **Function**: Estimate the mutual information $I(\mathbf{Z}_\ell; Y)$ between layer-$\ell$ features $\mathbf{Z}_\ell$ and forget labels $Y$.
    - **Mechanism**: An InfoNCE lower-bound estimator is used, defining critic functions $f_{\nu_\ell}$ and $g_{\eta_\ell}$ per layer. $f_{\nu_\ell}$ reuses the model's layers $\ell{+}1$ through $L$ plus a projection head as a feature extractor; $g_{\eta_\ell}$ models the binary label as two trainable vectors. MI estimates are obtained by maximizing the InfoNCE objective.
    - **Design Motivation**: The model-specific critic design (reusing subsequent layers) makes MI estimation architecture-agnostic—the same estimation pipeline applies to ResNet, ViT, and other architectures without redesign.

2. **IDI Score Computation**
    - The information difference for the unlearned model $\theta_u$ is: $ID(\theta_u) = \sum_{\ell=1}^{L} \max(0, I_{\theta_u}(\mathbf{Z}_\ell; Y) - I_{\theta_r}(\mathbf{Z}_\ell; Y))$
    - $ID(\theta_o)$ is similarly computed for the original model $\theta_o$.
    - $\text{IDI} = ID(\theta_u) / ID(\theta_o)$, with range $[0, 1]$; values closer to 0 indicate more thorough unlearning.
    - The Retrain baseline has a theoretical IDI of 0; the original model has IDI of 1.

3. **Head Distillation Experiment Exposing Black-Box Metric Failures**
    - The encoder is frozen; only the final classification head is retrained.
    - KL-divergence distillation aligns the output distribution to a pseudo-retrained model with the forget-class logit set to negative infinity.
    - Result: MIA and JSD black-box metrics rank among the best, yet IDI = 1.000 (information fully retained).
    - Further validation: freezing the HD encoder and retraining the classification head with only 2% of training data recovers 82%+ forget-class accuracy (vs. 41% for Retrain), directly demonstrating information retention.

4. **COLA Unlearning Method**
    - **Function**: Eliminate forget-set information at the feature level.
    - **Collapse phase**: Collapse forget-set encoder features so that forget-set sample representations become indistinguishable from retain-set representations.
    - **Align phase**: Align retain-set features to recover task performance.
    - On CIFAR-10/ResNet-18: IDI = 0.010 (vs. Retrain's 0.0); MIA = 12.64 (close to Retrain's 10.64).

### Loss & Training

- InfoNCE critic networks for IDI computation are trained independently (SGD optimizer) using the forget set ($Y=1$) and retain set ($Y=0$).
- COLA operates at the feature level without full retraining.
- Experiments cover CIFAR-10/100 and ImageNet-1K with ResNet-18/50 and ViT architectures.
- Unlearning scenarios include single-class forgetting, multi-class forgetting (5/10 classes), and random data forgetting.

## Key Experimental Results

### Main Results (CIFAR-10, Single-Class Forgetting, ResNet-18)

| Method | UA↑ | TA | MIA (closer to Retrain is better) | IDI (↓, 0 is optimal) | RTE |
|--------|-----|----|------------------------------------|------------------------|-----|
| Retrain | 100.0 | 95.64 | 10.64 | 0.0 | 154.56 min |
| HD | 100.0 | 95.22 | 2.05 | **1.000** | 0.10 min |
| SALUN | 100.0 | 95.42 | 0.01 | 0.936 | 3.54 min |
| RL | 99.93 | **95.66** | 0.0 | 0.830 | 3.09 min |
| SCRUB | 100.0 | 95.37 | 19.73 | **−0.056** | 3.49 min |
| **COLA** | 100.0 | 95.36 | **12.64** | **0.010** | 4.91 min |

### Ablation Study — Layer-wise MI Visualization

| Unlearning Method | Shallow-Layer MI | Deep-Layer MI | IDI |
|-------------------|-----------------|---------------|-----|
| Retrain | Low MI (information dispersed) | Low MI | 0.0 (baseline) |
| Original / HD | High MI (fully retained) | High MI | 1.0 (not unlearned) |
| GA | Low MI (close to Retrain) | Low MI | 0.334 (relatively good) |
| SALUN | High MI (close to Original) | High MI | 0.936 (barely unlearned) |
| COLA | Low MI (close to Retrain) | Low MI | **0.010** (best) |

### Key Findings

- **HD achieves "perfect" black-box scores yet IDI = 1.0**: The most compelling evidence that black-box metrics are unreliable—examining outputs alone cannot determine whether a model has truly unlearned.
- **SALUN achieves strong black-box performance yet IDI = 0.936**: SALUN's unlearning occurs primarily at the output level; intermediate layers almost entirely retain forget-data information.
- **2% of data suffices to recover 82% accuracy from a "forgotten" encoder**: For methods such as SALUN and RL, freezing the encoder and retraining with minimal data reconstructs forget-class capability—a serious risk for security and regulatory compliance.
- **COLA's IDI = 0.010 approaches Retrain's 0.0**: The feature-level collapse strategy effectively eliminates information residue in intermediate layers.
- **IDI exhibits consistent behavior across architectures (ResNet/ViT) and datasets (CIFAR/ImageNet)**: The metric generalizes well.

## Highlights & Insights

- **The "HD experiment" is the paper's most insightful contribution**: A minimal intervention—modifying only the classification head—exposes a systemic flaw in the field's evaluation framework. When black-box metrics can be so easily deceived, all unlearning claims grounded in such metrics become suspect.
- **Mutual information as a natural unlearning evaluation metric**: MI directly measures "how much information about the forget data the model still retains"—precisely the definition of strong unlearning. Black-box metrics are merely incomplete proxies for this objective.
- **Implications of COLA**: Achieving thorough feature-level unlearning requires more than gradient ascent (GA) or saliency-based masking (SALUN); explicit collapse of feature representations is necessary.

## Limitations & Future Work

- IDI computation requires training independent critic networks per layer, incurring substantial overhead (results averaged over 5 trials per setting).
- InfoNCE as a lower-bound MI estimator may not be tight, particularly when true MI is high.
- The collapse operation in COLA may be less effective in random data forgetting (non-class-level forgetting) scenarios.
- The analysis is limited to classification tasks; unlearning evaluation for generative models (e.g., diffusion models, LLMs) is not addressed.

## Related Work & Insights

- **vs. Black-box MIA metrics**: MIA only detects whether model outputs expose membership information; it cannot detect information residue in intermediate layers. IDI directly measures layer-wise mutual information, offering a more fundamental assessment.
- **vs. SALUN / L1-sparse and other SOTA unlearning methods**: These methods perform well on black-box metrics, but IDI reveals that their encoders substantially retain forget-data information—an important cautionary finding for the unlearning community.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The HD experiment exposing black-box metric failures is a significant finding; IDI fills the gap in white-box evaluation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 14 unlearning methods × 3 datasets × 2 architectures, covering single-class, multi-class, and random forgetting scenarios.
- **Writing Quality**: ⭐⭐⭐⭐ — The logical chain from problem identification to metric design to method proposal is clear.
- **Value**: ⭐⭐⭐⭐⭐ — Significant impact on the evaluation paradigm of the entire machine unlearning field; all subsequent unlearning methods should report IDI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] InfoDecom: Decomposing Information for Defending Against Privacy Leakage in Split Inference](infodecom_decomposing_information_for_defending_against_privacy_leakage_in_split.md)
- [\[ICLR 2026\] Why Do Unlearnable Examples Work: A Novel Perspective of Mutual Information](../../ICLR2026/ai_safety/why_do_unlearnable_examples_work_a_novel_perspective_of_mutual_information.md)
- [\[AAAI 2026\] Easy to Learn, Yet Hard to Forget: Towards Robust Unlearning Under Bias](easy_to_learn_yet_hard_to_forget_towards_robust_unlearning_under_bias.md)
- [\[NeurIPS 2025\] Efficient Verified Machine Unlearning for Distillation](../../NeurIPS2025/ai_safety/efficient_verified_machine_unlearning_for_distillation.md)
- [\[ACL 2026\] When Bigger Isn't Better: A Comprehensive Fairness Evaluation of Political Bias in Multi-News Summarisation](../../ACL2026/ai_safety/when_bigger_isn39t_better_a_comprehensive_fairness_evaluation_of_political_bias_.md)

</div>

<!-- RELATED:END -->
