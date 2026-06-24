---
title: >-
  [Paper Note] Effortless Active Labeling for Long-Term Test-Time Adaptation
description: >-
  [CVPR 2025][Test-Time Adaptation] This work proposes EATTA, an approach that labels only one most valuable sample per batch (instead of multiple) based on feature perturbation sensitivity during long-term test-time adaptation (TTA). Combined with a gradient norm debiasing strategy to balance the gradients of supervised and unsupervised losses, EATTA achieves an average error rate of 50.9% on ImageNet-C with an extremely low annotation cost, outperforming SimATTA with three ti…
tags:
  - "CVPR 2025"
  - "Test-Time Adaptation"
  - "Active Learning"
  - "Pseudo-Labeling"
  - "Gradient Normalization"
  - "Long-Term Adaptation"
date: 2026-05-08
content_hash: 77cdd1beaff0a049
---

# Effortless Active Labeling for Long-Term Test-Time Adaptation

**Conference**: CVPR 2025  
**arXiv**: [2503.14564](https://arxiv.org/abs/2503.14564)  
**Code**: [https://github.com/flash1803/EATTA](https://github.com/flash1803/EATTA)  
**Area**: Others  
**Keywords**: Test-Time Adaptation, Active Learning, Pseudo-Labeling, Gradient Normalization, Long-Term Adaptation

## TL;DR
This work proposes EATTA, an approach that labels only one most valuable sample per batch (instead of multiple) based on feature perturbation sensitivity during long-term test-time adaptation (TTA). Combined with a gradient norm debiasing strategy to balance the gradients of supervised and unsupervised losses, EATTA achieves an average error rate of 50.9% on ImageNet-C with an extremely low annotation cost, outperforming SimATTA with three times the labeling budget by 3.9%.

## Background & Motivation

**Background**: Test-time adaptation (TTA) fine-tunes pre-trained models on unlabeled test data during the inference phase to adapt to distribution shifts. Long-term TTA suffers from continuous performance degradation due to the accumulation of pseudo-label noise. Recent active TTA (ATTA) methods introduce a small amount of human annotations to correct pseudo-labels, but they require labeling multiple samples per batch.

**Limitations of Prior Work**: ATTA methods such as SimATTA and HILTTA require labeling 3 samples per batch, leading to a linear growth in total annotation cost as the number of batches increases, which heavily burdens the annotators. Moreover, existing methods select high-entropy samples for labeling, but high-entropy samples are not necessarily the most valuable ones for single-step optimization.

**Key Challenge**: Adequate annotations are needed to correct cumulative pseudo-labeling errors under a highly restricted budget—how can optimal adaptation performance be achieved with the minimum annotation cost (only 1 label per batch)?

**Goal**: To design an active TTA method with extremely low annotation costs, capable of effectively resisting pseudo-label degradation in long-term adaptation with only 1 annotated sample per batch.

**Key Insight**: Feature perturbation sensitivity is proposed as a sample utility measure. Samples residing at the distribution boundaries of the source and target domains are most sensitive to small perturbations, making them the most valuable for single-step optimization.

**Core Idea**: Feature perturbation sensitivity instead of entropy is adopted to select the single most valuable sample per batch for labeling, paired with gradient norm-based debiasing to dynamically balance the gradient contributions of supervised and unsupervised losses.

## Method

### Overall Architecture
When each test batch arrives: (1) evaluate the annotation utility of each sample using feature perturbation sensitivity, and request the label for the single highest-value sample; (2) integrate a class-balancing mechanism to avoid repeatedly labeling samples of the same class; (3) dynamically adjust the weights of the supervised loss and unsupervised (entropy) loss using gradient norm-based debiasing; (4) update the model using the annotated sample and pseudo-labeled samples.

### Key Designs

1. **Feature Perturbation Sensitivity Selection Strategy**:

    - **Function**: Select the single most valuable sample per batch for model optimization.
    - **Mechanism**: Apply small Gaussian noise $\epsilon$ to the features of each sample, and compare the predictive probability differences on the pseudo-labeled class before and after perturbation: $\text{diff}(x_i) = |\phi(h(f(x_i)))_{\hat{y}_i} - \phi(h(f(x_i)+\epsilon))_{\hat{y}_i}$. Samples with the largest differences lie close to the decision boundary and contribute the most to optimization when labeled. Meanwhile, maintain a class registry of the recent $K$ labeled samples to avoid repeatedly annotating the same classes.
    - **Design Motivation**: More precise than high-entropy selection—while high-entropy samples might be pure noise (far from both distributions), perturbation-sensitive samples lie precisely at the distribution intersection, delivering the highest labeling value.

2. **Gradient Norm-based Debiasing**:

    - **Function**: Balance the contributions of supervised and unsupervised losses to the model update.
    - **Mechanism**: Compute the $L_2$ norm of the gradients of the supervised and unsupervised losses, and use the other's norm to weigh oneself: $\gamma_1 = 2 \|\nabla L_{unsup}\| / (\|\nabla L_{sup}\| + \|\nabla L_{unsup}\|)$ (with $\gamma_2$ symmetrically defined). Smooth the weights using exponential moving average (EMA) to maintain long-term stability. Consequently, when annotations are scarce and the supervised gradient is small, its weight automatically increases, and vice versa.
    - **Design Motivation**: Since EATTA only has 1 labeled sample per batch, the batch's supervised gradient is far smaller than the unsupervised counterpart. Simply adding them together would overshadow the annotation signal. Gradient normalization ensures equal contributions from both signals.

### Loss & Training
The total loss is $L = \gamma_1 L_{sup} + \gamma_2 L_{unsup}$, where $L_{sup}$ is the cross-entropy on the labeled sample(s), and $L_{unsup}$ is the entropy minimization over all samples. The affine parameters of the batch normalization (BN) layers are trainable. Optionally, a buffer can be maintained to store historical labeled samples for replay.

## Key Experimental Results

### Main Results (ImageNet-C, Continual TTA)

| Method | Labels/Batch | Average Error Rate↓ |
|------|----------|-----------|
| TENT | 0 | 70.9% |
| CoTTA | 0 | 69.8% |
| SAR | 0 | 60.7% |
| SimATTA | 3 | 54.8% |
| HILTTA | 3 | 53.7% |
| EATTA | 1 | 53.8% |
| EATTA | 3 | 51.9% |
| EATTA (BFS=300) | 3 | **50.9%** |

### Ablation Study

| Configuration | Average Error Rate | Description |
|------|-----------|------|
| Perturbation sensitivity selection | 53.8% | Full strategy |
| High-entropy selection replacement | ~55% | Conventional selection is inferior to perturbation |
| w/o Gradient debiasing | ~55.5% | Supervised signal is overshadowed |
| w/o Class balancing | ~54.5% | Repeatedly labeling the same class wastes budget |
| Labeling 1 sample every 5 batches | ~55% | Effective even under extremely low budget |

### Key Findings
- EATTA with only 1 label per batch (53.8%) closely approaches the performance of HILTTA with 3 labels per batch (53.7%), improving annotation efficiency threefold.
- Gradient norm-based debiasing is a critical component—performance drops by ~2% without debiasing, as the scarce annotation signal is overwhelmed by the massive pseudo-label gradients.
- Feature perturbation sensitivity selection is more effective than traditional high-entropy selection, validating the hypothesis that "decision-boundary samples hold the highest value".
- Even when only 1 sample is labeled every 5 batches, the performance remains significantly superior to fully unsupervised methods.

## Highlights & Insights
- **The "one is enough" active labeling strategy**: This breaks the conventional mindset that ATTA requires substantial annotations, demonstrating that a single carefully selected labeled sample can achieve the effect of three randomly selected ones.
- **Elegant design of perturbation sensitivity**: It requires only one forward pass plus an additional noise-perturbed forward pass to evaluate sample utility, yielding extremely low computational overhead.
- **Generality of gradient normalization**: This multi-loss balancing technique is not limited to TTA but is applicable to any training scenario combining supervised and unsupervised learning.

## Limitations & Future Work
- The perturbation noise magnitude $\epsilon$ is a hyperparameter; different degrees of distribution shift may require different values of $\epsilon$.
- The choice of $K$ in the class-balancing mechanism impacts performance; an excessively large $K$ restricts the selection space.
- Validation is limited to classification tasks; downstream tasks such as object detection or segmentation have not been explored.
- The method assumes that annotations can be acquired in real time during inference, which may suffer from delays in actual deployments.

## Related Work & Insights
- **vs SimATTA/HILTTA**: EATTA achieves comparable or superior performance with only 1/3 of the annotation cost, core to which is its more precise sample selection strategy.
- **vs TENT/CoTTA**: Unsupervised methods suffer from continuous performance degradation during long-term adaptation; EATTA demonstrates that an extremely small amount of annotations is sufficient to resist this degradation.
- **vs Active Learning**: Traditional active learning selects the most uncertain samples, but in TTA such samples can often be pure noise. Perturbation sensitivity is better suited to single-step optimization scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined scheme of perturbation sensitivity selection and gradient norm debiasing is novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation including continual TTA on ImageNet-C, various annotation budgets, and thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and concise methodological descriptions.
- Value: ⭐⭐⭐⭐ Holds significant value for TTA deployments on edge devices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Test-Time Adaptation by Causal Trimming](../../NeurIPS2025/others/test-time_adaptation_by_causal_trimming.md)
- [\[ICLR 2026\] Exposing Mixture and Annotating Confusion for Active Universal Test-Time Adaptation](../../ICLR2026/others/exposing_mixture_and_annotating_confusion_for_active_universal_test-time_adaptat.md)
- [\[ICML 2025\] Ranked Entropy Minimization for Continual Test-Time Adaptation](../../ICML2025/others/ranked_entropy_minimization_for_continual_test-time_adaptation.md)
- [\[CVPR 2026\] Neural Collapse in Test-Time Adaptation](../../CVPR2026/others/neural_collapse_in_test-time_adaptation.md)
- [\[ICML 2025\] Beyond Entropy: Region Confidence Proxy for Wild Test-Time Adaptation](../../ICML2025/others/beyond_entropy_region_confidence_proxy_for_wild_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
