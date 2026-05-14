---
title: >-
  [Paper Note] Learning from Synthetic Data via Provenance-Based Input Gradient Guidance
description: >-
  [CVPR 2026][Social Computing][learning from synthetic data] This paper proposes leveraging provenance information—automatically obtained during the synthetic data generation process—as auxiliary supervision signals. By a…
tags:
  - "CVPR 2026"
  - "Social Computing"
  - "learning from synthetic data"
  - "input gradient guidance"
  - "spurious correlation suppression"
  - "data augmentation"
  - "provenance information"
date: 2026-05-08
content_hash: bc9b48e81dbdef5e
---

# Learning from Synthetic Data via Provenance-Based Input Gradient Guidance

**Conference**: CVPR 2026
**arXiv**: [2604.02946](https://arxiv.org/abs/2604.02946)
**Code**: None
**Area**: Deep Learning Methods
**Keywords**: learning from synthetic data, input gradient guidance, spurious correlation suppression, data augmentation, provenance information

## TL;DR

This paper proposes leveraging provenance information—automatically obtained during the synthetic data generation process—as auxiliary supervision signals. By applying input gradient guidance (suppressing input gradients in non-target regions), the method directly encourages models to learn discriminative representations focused on target regions. Effectiveness is validated across multiple tasks and modalities, including weakly supervised localization, spatio-temporal action detection, and image classification.

## Background & Motivation

Synthetic data (data augmentation, generative model editing, etc.) has become an important means of improving model robustness in deep learning training. Existing synthetic learning methods (e.g., CutMix, Mixup, diffusion-based image editing) indirectly improve robustness by diversifying training sample distributions, but suffer from fundamental limitations:

1. **Lack of explicit guidance**: Models receive only supervision labels and must independently determine which regions of the input space truly contribute to classification, making them prone to learning spurious correlations (e.g., backgrounds, co-occurring objects).
2. **Synthetic bias problem**: Artifacts and biases introduced by data augmentation and generative models may themselves be erroneously learned by models, preventing accuracy from scaling linearly with data volume.
3. **Robustness as a byproduct**: Robustness gains in existing methods are merely indirect effects of increased training samples, rather than direct learning of discriminative features of target objects.

Core insight: The synthesis process naturally records provenance information—i.e., which pixels originate from which target (e.g., CutMix synthesis masks, differences between pre- and post-edited images)—yet this freely available information has never been exploited to explicitly constrain model learning behavior.

## Method

### Overall Architecture

The framework consists of three components: (1) synthesizing training data and extracting provenance information $\mathbf{I}$; (2) standard downstream task learning (classification loss $L_{cls}$); and (3) provenance-based input gradient regularization (provenance loss $L_{PG}$). The total loss is $L_{total} = L_{cls} + \alpha L_{PG}$, where $\alpha$ controls regularization strength.

### Key Designs

1. **Provenance Extraction**:
    - Function: Automatically obtain annotation information indicating which target each element of the synthetic data originates from.
    - Mechanism: Three synthesis strategies are handled separately. (a) Image mixing (CutMix, etc.): The synthesis mask $M$ is directly used as provenance—$\mathbf{I}_A = M$ (regions from image A) and $\mathbf{I}_B = 1-M$ (regions from image B). (b) Skeleton sequence mixing: A spatio-temporal binary mask $M \in \{0,1\}^{P \times F \times E}$ marks the source of each skeleton feature. (c) Generative model image editing: A difference map $D(u,v)$ between the original and edited images is computed, binarized via Otsu thresholding to yield a mask, where $\mathbf{I}(u,v)=1$ denotes unedited target regions.
    - Design Motivation: Provenance information is a natural byproduct of the synthesis process, requiring no additional annotation cost, and can precisely identify target vs. non-target regions.

2. **Input Gradient Guidance**:
    - Function: Directly constrain the model's dependence on different input regions for its predictions.
    - Mechanism: The gradient of the model output (logit) with respect to the input, $\nabla_{\tilde{x}} f_y(\tilde{x})$, is computed, and provenance information is used to suppress gradients in non-target regions. For soft labels (image mixing): $L_{PG} = \|(1-M) \odot \nabla_{\tilde{x}} f_A(\tilde{x}) + M \odot \nabla_{\tilde{x}} f_B(\tilde{x})\|_2^2$, i.e., predictions for class A should not depend on regions originating from image B, and vice versa. For hard labels (generative model editing): $L_{PG} = \|(1-M) \odot \nabla_{\tilde{x}} f_y(\tilde{x})\|_2^2$, i.e., the logit for class $y$ should not depend on edited regions.
    - Design Motivation: Input gradients reflect the sensitivity of model predictions to individual input elements. By suppressing gradients in non-target regions, the model is forced to rely on target regions for discrimination, directly eliminating spurious correlations.

3. **Generalizability Across Tasks and Modalities**:
    - Function: The method is applicable to any synthetic learning framework in which non-target regions can be identified.
    - Mechanism: Whenever the synthesis process can identify provenance information, the provenance loss can be introduced. The paper demonstrates applicability across three synthesis strategies (image mixing: CutMix, ResizeMix, PuzzleMix; skeleton sequence mixing: BatchMix; generative model editing: ALIA) and three tasks (weakly supervised object localization, weakly supervised spatio-temporal action detection, and image classification).
    - Design Motivation: The method is independent of specific synthesis approaches and DNN architectures, constituting a general framework enhancement for learning pipelines.

### Loss & Training

The total loss is $L_{total} = L_{cls} + \alpha L_{PG}$. Experiments show that $\alpha$ in the range $[0.01, 0.09]$ consistently improves performance. Since the provenance loss involves second-order differentiation (gradients of input gradients), PyTorch autograd and AMP are used for acceleration, with the loss computation performed in FP32 to avoid numerical instability.

## Key Experimental Results

### Main Results

| Task / Dataset | Metric | Baseline | + Synthesis Method | + Ours | Gain |
|---|---|---|---|---|---|
| Weakly Supervised Localization / CUB (VGG16) | MaxBoxAccV2 Mean | — | 62.3 (CutMix) | **65.1** | +2.8 |
| Weakly Supervised Localization / CUB (VGG16) | MaxBoxAccV2 Mean | — | 57.6 (ResizeMix) | **62.2** | +4.6 |
| Weakly Supervised Localization / CUB (SAT) | MaxBoxAccV2 Mean | 91.4 | 91.5 (CutMix) | **92.1** | +0.6 |
| Spatio-Temporal Action Detection / UCF101-24 | AP@0.5 | 37.4 (SKP) | 38.0 (BatchMix) | **39.7** | +1.7 |
| Image Classification / Waterbirds | Top-1 Acc | 62.2 | 71.4 (ALIA) | **80.7** | +9.3 |
| Image Classification / iWildCam | Top-1 Acc | 75.0 | 83.5 (ALIA) | **84.4** | +0.9 |
| Image Classification / CUB | Top-1 Acc | 70.8 | 71.7 (ALIA) | **72.0** | +0.3 |

### Ablation Study

| Configuration | CUB Localization Acc (%) | Note |
|---|---|---|
| Random mask | 60.5 | Random mask used as surrogate provenance |
| Unmasked (full region) | 61.1 | No target/non-target distinction |
| Ours (true provenance) | **65.1** | True provenance from synthesis process |

Ablation on provenance quality (image classification): Applying dilation/erosion of ±10%/±30% to foreground masks leads to limited performance degradation (CUB: 72.1→71.5), demonstrating robustness to provenance accuracy.

### Key Findings

- **Largest gain on Waterbirds (+9.3pp)**: Waterbirds is specifically designed as a spurious correlation benchmark with strong background-class correlations; provenance guidance directly suppresses background dependence, yielding the most pronounced improvement.
- On VGG16, CutMix + Ours improves IoU from 67.3% to 74.6% at $\delta=0.5$, but decreases from 28.6% to 23.1% at $\delta=0.7$, suggesting the model tends to produce larger bounding boxes under stricter localization thresholds.
- Regarding training efficiency, although second-order differentiation increases per-epoch time (140s→150s), faster convergence (50 epochs→15 epochs) reduces total training time from 1.9h to 0.6h.
- The ablation comparison between random masks and no masks clearly demonstrates that the accuracy of provenance information—rather than gradient regularization per se—is the key driver of improvement.

## Highlights & Insights

- The **"free lunch" supervision signal** is the paper's most significant contribution. Provenance information is naturally produced during synthesis but has been consistently overlooked; this paper is the first to systematically exploit it as auxiliary supervision. The idea is simple yet far-reaching—any training pipeline using data augmentation can adopt it at zero additional cost.
- The paradigm shift **from indirect to direct** is compelling: prior synthetic learning methods improve robustness indirectly by enriching sample distributions, whereas this work uses gradient regularization to explicitly tell the model "where to look," yielding more direct and stronger effects.
- The method demonstrates strong generalizability—effective across modalities (images, skeleton sequences), tasks (localization, detection, classification), and synthesis strategies (mixing, generative model editing).
- The convergence acceleration is a notable surprise: although second-order differentiation theoretically increases computation, the accelerated convergence actually reduces total training time.

## Limitations & Future Work

- The granularity of provenance information is constrained by the synthesis method—CutMix provides only rectangular masks, and generative model editing relies on Otsu binarization of difference maps, which may lack sufficient precision.
- Provenance extraction for generative model editing depends on pixel-wise comparison between original and edited images; when the generative model modifies the appearance of the target object itself, erroneous attribution may occur.
- Experiments use moderate-scale models and datasets (VGG16, ResNet-50); effectiveness on large-scale pretrained models remains unverified.
- Future work could explore combining provenance information with attention mechanisms, or extending the approach to self-supervised learning by exploiting correspondences between pre- and post-augmentation views.

## Related Work & Insights

- **vs. CutMix/ResizeMix/PuzzleMix**: These methods perform data-level mixing augmentation only; the proposed method introduces explicit gradient-level guidance on top of this, constituting a two-stage improvement of "augmentation + guidance."
- **vs. Right for the Right Reasons (Ross et al.)**: That work also manipulates input gradients but requires human-annotated saliency regions; the proposed method automatically obtains provenance information from the synthesis process without additional annotation.
- **vs. ALIA**: ALIA uses diffusion models to edit training images for increased diversity; the proposed method introduces provenance loss on top of ALIA, achieving a +9.3pp gain on Waterbirds.
- The provenance guidance concept may inspire improvements in positive/negative sample construction for contrastive learning.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of treating synthesis-process provenance as a free supervision signal is novel and practical, though the technical means (input gradient regularization) are relatively established.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across three synthesis strategies × three tasks × multiple datasets, with detailed ablation studies covering provenance quality, hyperparameter sensitivity, and training efficiency.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, method derivation is rigorous, and notation is consistent throughout.
- Value: ⭐⭐⭐⭐ Strong generalizability, simple implementation, and zero additional annotation cost make the method well-suited for broad adoption in data augmentation training pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Gradient Extrapolation for Debiased Representation Learning](../../ICCV2025/social_computing/gradient_extrapolation_for_debiased_representation_learning.md)
- [\[ACL 2026\] ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection](../../ACL2026/social_computing/toxitrace_gradient-aligned_training_for_explainable_chinese_toxicity_detection.md)
- [\[CVPR 2026\] Revisiting Unknowns: Towards Effective and Efficient Open-Set Active Learning](revisiting_unknowns_towards_effective_and_efficient_open-set_active_learning.md)
- [\[ACL 2026\] On the Step Length Confounding in LLM Reasoning Data Selection](../../ACL2026/social_computing/on_the_step_length_confounding_in_llm_reasoning_data_selection.md)
- [\[ICLR 2026\] Functional Embeddings Enable Aggregation of Multi-Area SEEG Data for Robust BCI](../../ICLR2026/social_computing/functional_embeddings_enable_aggregation_of_multi-area_seeg_data_for_robust_bci.md)

</div>

<!-- RELATED:END -->
