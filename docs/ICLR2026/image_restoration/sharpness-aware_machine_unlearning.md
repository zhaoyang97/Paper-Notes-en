---
title: >-
  [Paper Note] Sharpness-Aware Machine Unlearning
description: >-
  [ICLR 2026][Image Restoration][machine unlearning] This paper systematically analyzes the theoretical properties of SAM in the machine unlearning setting through a signal-noise decomposition framework. It finds that SAM abandons its denoising capability on the forget set while retaining it on the retain set. Motivated by this finding, the paper proposes the Sharp MinMax algorithm, which partitions the model into two components subject to sharpness minimization (retain) and sharpness maximization (forget) respectively, achieving state-of-the-art unlearning performance.
tags:
  - ICLR 2026
  - Image Restoration
  - machine unlearning
  - Sharpness-Aware Minimization
  - SAM
  - Signal-Noise Decomposition
  - Sharp MinMax
date: 2026-05-08
content_hash: b6eb881b71afd419
---

# Sharpness-Aware Machine Unlearning

**Conference**: ICLR 2026
**arXiv**: [2506.13715](https://arxiv.org/abs/2506.13715)
**Code**: None
**Area**: Image Restoration
**Keywords**: machine unlearning, Sharpness-Aware Minimization, SAM, Signal-Noise Decomposition, Sharp MinMax

## TL;DR

This paper systematically analyzes the theoretical properties of SAM in the machine unlearning setting through a signal-noise decomposition framework. It finds that SAM abandons its denoising capability on the forget set while retaining it on the retain set. Motivated by this finding, the paper proposes the Sharp MinMax algorithm, which partitions the model into two components subject to sharpness minimization (retain) and sharpness maximization (forget) respectively, achieving state-of-the-art unlearning performance.

## Background & Motivation

Machine Unlearning aims to efficiently remove the influence of specific training data from a model without retraining from scratch. Existing methods such as influence-function-based updates (Influence Unlearning), sparse fine-tuning (L1-Sparse), and gradient ascent (NegGrad) have made progress, yet they lack deep theoretical understanding of the unlearning process and rely heavily on extensive hyperparameter tuning with unpredictable behavior in practice.

The key challenge is that when a model simultaneously receives retain signals and forget signals, the two classes of signals interfere with and may cancel each other during training. In particular, **achieving a reliable balance between retain accuracy and completeness of forgetting** has long lacked principled theoretical guidance.

Sharpness-Aware Minimization (SAM) has been shown to find flatter loss minima, effectively suppressing noise memorization and improving generalization. A natural hypothesis arises: does an optimizer that suppresses memorization also excel at forgetting? This paper investigates this question through rigorous theoretical and empirical analysis.

## Method

### Overall Architecture

This paper builds on a signal-noise decomposition framework over a two-layer CNN, decomposing model weight updates into a signal learning coefficient $\kappa$ and a noise learning coefficient $\zeta$, and systematically analyzes the behavioral differences between SGD and SAM under the NegGrad unlearning scheme.

Specifically, each patch of an input image either contains a class signal $y_i \varphi$ or a noise vector $\xi_i$. Model weights are decomposed into signal and noise components, and the training objective is to increase $\kappa$ (signal learning) while controlling $\zeta$ (noise memorization).

### Key Designs

1. **SAM Denoising Failure Lemma (Lemma 3.1)**: Under the NegGrad unlearning scheme, the perturbation term $\hat{\epsilon}$ of SAM deactivates noise neurons on the retain set $\mathcal{R}$ (preserving denoising properties), while noise neurons on the forget set $\mathcal{F}$ remain active. This means SAM degenerates to SGD-like behavior on the forget set—overfitting to the forget set to a degree comparable to SGD. The root cause is that NegGrad performs gradient ascent on the forget set, reversing the effect of the SAM perturbation.

2. **Differentiated Test Error Bounds (Theorem 3.2 & 3.3)**: Under NegGrad with SGD, benign overfitting is achievable when the signal strength satisfies $\|\varphi\|_2 \geq C_1 d^{1/4} n^{-1/4} P \sigma_p$; otherwise, harmful overfitting leads to test error $\geq 0.1$. SAM's advantage is that, even under weaker signal conditions $\|\varphi\|_2 \geq \Omega(1)$, it can still guarantee low test error as long as the retain set signal is sufficient, owing to the preservation of SAM's denoising property on the retain set.

3. **Signal Margin and $\alpha$ Threshold (Lemma 3.4)**: SAM's signal learning rate on the retain set is $\Theta(\|\varphi\|_2^2)$ times that of SGD, allowing SAM to tolerate a smaller retain weight $\alpha$ (i.e., stronger emphasis on forgetting). In the benign overfitting regime, the gap in required $\alpha$ between SGD and SAM is of order $O(\sqrt{d/n})$. This finding has practical implications: **the choice of $\alpha$ depends not only on the relative sizes of the retain and forget sets, but also on signal strength and data dimensionality**.

4. **Sharp MinMax Algorithm**: Motivated by the theoretical insight that SAM has an advantage on the retain set but degenerates on the forget set, the authors propose partitioning the model into two components:

    - **Retain model**: trained with SAM (sharpness minimization) to leverage its denoising property for maintaining generalization.
    - **Forget model**: trained with sharpness maximization, deliberately driving the model toward sharper loss landscapes on the forget set, thereby achieving more thorough forgetting via "memorize then forget."

   Model partitioning is based on gradient magnitude ranking: parameter gradient magnitudes with respect to the forget set are computed, and parameters with high magnitudes are assigned to the forget model.

### Loss & Training

- **NegGrad dual-objective loss**: $\mathcal{L}_{\text{NegGrad}} = \alpha \cdot \mathcal{L}(\mathcal{R}) - (1-\alpha) \cdot \mathcal{L}(\mathcal{F})$, combining gradient descent on the retain set with gradient ascent on the forget set.
- **Sharp MinMax**: the retain component uses SAM's $\min_W \mathcal{L} + [\max_{\hat{\epsilon}} \mathcal{L}(W+\hat{\epsilon}) - \mathcal{L}(W)]$; the forget component uses sharpness maximization $\min_W \mathcal{L} - [\max_{\hat{\epsilon}} \mathcal{L}(W+\hat{\epsilon}) - \mathcal{L}(W)]$.
- **Forgetting difficulty quantification**: The memorization score $\text{mem}(\mathcal{A}, \mathcal{S}, i)$ from Feldman & Zhang (2020) is used to measure per-sample forgetting difficulty; high-memorization samples are harder to unlearn.

## Key Experimental Results

### Experimental Setup
- Datasets: CIFAR-100, ImageNet-1K (main experiments); CIFAR-10, Tiny-ImageNet (supplementary experiments)
- Model: ResNet-50
- Forget set size $|\mathcal{F}| \approx 5\%|\mathcal{S}|$, partitioned by memorization score into $\mathcal{F}_{\text{high}}, \mathcal{F}_{\text{mid}}, \mathcal{F}_{\text{low}}$
- Evaluation metric: ToW (Tug-of-War), jointly measuring retain accuracy, forget accuracy, and test accuracy
- Baselines: NegGrad, RL, SalUn, L1-Sparse, SCRUB

### Main Results

| Method | ImageNet AVG ToW | CIFAR-100 AVG ToW | Notes |
|--------|-----------------|-------------------|-------|
| NegGrad+SGD | 83.83 | 81.80 | Baseline |
| NegGrad+SAM 0.1 | 83.68 | 72.78 | SAM as $\mathcal{U}$ |
| NegGrad+ASAM 1.0 | 84.84 | 83.11 | Best NegGrad variant |
| Sharp MinMax+ASAM 1.0 | ~87.90 | ~87.90 | **New SOTA** |

### Ablation Study

| Configuration | Key Metric | Notes |
|--------------|------------|-------|
| Decreasing $\alpha$ | SAM shows stronger robustness to degradation | SGD collapses first; ASAM 1.0 is most robust |
| MIA accuracy | SAM consistently reduces MIA accuracy | Forget set samples are harder to identify via membership inference |
| Feature entanglement $E_{Wp}$ | SAM < SGD | Better separation of retain/forget features after unlearning with SAM |

### Key Findings

1. **SAM consistently improves all unlearning methods**: Whether used as a pretraining or unlearning optimizer, SAM improves the ToW metric across the board.
2. **Overfitting can be beneficial for unlearning**: In strict sample-level unlearning scenarios such as privacy and copyright protection, deliberately overfitting the model to the forget set is actually more effective—challenging the conventional wisdom that overfitting is always harmful.
3. **SGD sometimes achieves lower forget accuracy**: Although SAM achieves higher ToW, SGD can sometimes reach lower forget-set accuracy, corroborating the theoretical prediction that SGD overfits more deeply to the forget set.
4. **Loss landscape visualization**: SAM-pretrained models exhibit flatter landscapes, yet interestingly, SGD models become flatter after unlearning, suggesting a potential implicit regularization effect.

## Highlights & Insights

- **Strong theoretical contribution**: This is the first work to rigorously analyze SAM's behavior in machine unlearning under a signal-noise decomposition framework, proving the "selective failure" of SAM's denoising property—a counterintuitive yet important finding.
- **Connecting optimization and unlearning**: The work deeply integrates sharpness-aware optimization with machine unlearning and provides theoretical guidance for choosing $\alpha$, eliminating reliance on purely heuristic tuning.
- **Elegant Sharp MinMax design**: Leveraging the insight that "overfitting = better forgetting," the model is partitioned into two complementary components that simultaneously maintain generalization and enhance forgetting.
- **Wasserstein entanglement metric**: An optimal-transport-based feature entanglement measure $E_{Wp}$ is proposed, which better discriminates irregularly shaped feature distributions compared to the variance-based entanglement $E_{\text{Var}}$.

## Limitations & Future Work

1. **Incomplete theory for weak signal regimes**: SAM's behavior when the retain signal strength is $O(1)$ is not fully characterized, and harmful overfitting may occur in this regime.
2. **Interaction between $\alpha$ and model partition ratio**: The interaction between the two hyperparameters—retain weight $\alpha$ and the proportion of the forget model—is not theoretically analyzed.
3. **Two-layer CNN assumption**: The theoretical analysis is based on a two-layer CNN; extension to deeper networks requires additional work.
4. **"Regularization" effect of SGD after unlearning**: The paper observes that the loss landscape of SGD flattens after unlearning but provides no explanation for this phenomenon.
5. **Computational overhead**: SAM itself requires two forward/backward passes, and Sharp MinMax incurs additional cost from model partitioning.

## Related Work & Insights

- **Relation to SalUn**: SalUn also performs parameter-selective unlearning but uses random label flipping; Sharp MinMax replaces this with sharpness maximization, which is better theoretically grounded.
- **Relation to SCRUB**: SCRUB combines knowledge distillation with NegGrad; SAM can serve as a direct plug-in to improve its performance.
- **Implications for privacy unlearning**: The finding that "overfitting benefits forgetting" has important implications for designing unlearning algorithms that satisfy differential privacy constraints.
- **Implications for LLM unlearning**: The proposed framework may extend to unlearning in large language models (e.g., knowledge editing, concept erasure), especially given the widespread adoption of SAM in LLM fine-tuning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size](adablock-dllm_semantic-aware_diffusion_llm_inference_via_adaptive_block_size.md)
- [\[CVPR 2026\] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](../../CVPR2026/image_restoration/drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)
- [\[CVPR 2026\] MAD-Avatar: Motion-Aware Animatable Gaussian Avatars Deblurring](../../CVPR2026/image_restoration/motionaware_animatable_gaussian_avatars_deblurring.md)
- [\[ICCV 2025\] Outlier-Aware Post-Training Quantization for Image Super-Resolution](../../ICCV2025/image_restoration/outlier-aware_post-training_quantization_for_image_super-resolution.md)
- [\[NeurIPS 2025\] Luminance-Aware Statistical Quantization: Unsupervised Hierarchical Learning for Illumination Enhancement](../../NeurIPS2025/image_restoration/luminance-aware_statistical_quantization_unsupervised_hierarchical_learning_for_.md)

</div>

<!-- RELATED:END -->
