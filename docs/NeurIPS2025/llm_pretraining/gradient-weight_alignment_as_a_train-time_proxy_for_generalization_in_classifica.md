---
title: >-
  [Paper Note] Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks
description: >-
  [NeurIPS 2025][LLM Pretraining][generalization] This paper proposes Gradient-Weight Alignment (GWA), which quantifies the directional consistency (cosine similarity) between the gradient of each training sample and the m…
tags:
  - "NeurIPS 2025"
  - "LLM Pretraining"
  - "generalization"
  - "gradient-weight alignment"
  - "early stopping"
  - "training dynamics"
  - "sample influence"
date: 2026-05-08
content_hash: 0bfde85fc66e980b
---

# Gradient-Weight Alignment as a Train-Time Proxy for Generalization in Classification Tasks

**Conference**: NeurIPS 2025
**arXiv**: [2510.25480](https://arxiv.org/abs/2510.25480)
**Code**: [hlzl/gwa](https://github.com/hlzl/gwa)
**Area**: LLM Pre-training
**Keywords**: generalization, gradient-weight alignment, early stopping, training dynamics, sample influence

## TL;DR

This paper proposes Gradient-Weight Alignment (GWA), which quantifies the directional consistency (cosine similarity) between the gradient of each training sample and the model weights. During training, GWA accurately predicts generalization performance, identifies the optimal early stopping point, and localizes influential training samples—all without requiring a validation set.

## Background & Motivation

The standard approach to evaluating model generalization in deep learning relies on a hold-out validation set, but this paradigm suffers from several fundamental drawbacks:

**Data waste**: The validation set is partitioned from training data, imposing a particularly high cost in data-scarce settings.

**Fragile i.i.d. assumption**: The validation set assumes the same distribution as training data, failing to reflect domain shift in real-world deployment.

**Lack of sample-level attribution**: Validation sets provide only aggregate performance metrics and cannot attribute generalization behavior to specific training samples.

**Existing alternatives each have inherent limitations**:
- Loss curvature methods (Hessian) require second-order derivatives, incurring prohibitive computational cost and numerical instability during training.
- Influence functions are strictly post-hoc and cannot be used for online monitoring.
- Pairwise gradient alignment requires storing all sample gradients, making memory requirements unscalable.
- LabelWave's prediction-change metric fails to detect overfitting under label noise.
- Gradient Disparity (GD) breaks down entirely on large-scale data (e.g., ImageNet).

The core scientific question is: **Can information available solely during training effectively evaluate model generalization and diagnose potential issues, thereby fully replacing the validation set?**

The theoretical foundation draws from the directional convergence theory of Ji & Telgarsky (2020): when training with cross-entropy on ideally separable data, model weights not only converge directionally but the gradient direction ultimately aligns with the weight direction ($\mathbb{E}_i[\gamma(x_i, \mathbf{w}_T)] \to 1$). This paper extends that theoretical result to realistic noisy data, proposing gradient-weight alignment as a real-time proxy for generalization.

## Method

### 1. Per-Sample Alignment Score

For each training sample $(x_i, y_i)$, the alignment score is defined as the cosine similarity between the negative gradient and the model weights:

$$\gamma(x_i, \mathbf{w}_T) = \cos\text{sim}(\mathbf{g}_T(x_i), \mathbf{w}_T) = \frac{\mathbf{g}_T(x_i) \cdot \mathbf{w}_T}{\|\mathbf{g}_T(x_i)\| \|\mathbf{w}_T\|}$$

where $\mathbf{g}_T(x_i) = -\nabla_\mathbf{w}\mathcal{L}(\mathbf{w}_T, x_i)$ is the negative gradient of sample $x_i$ at epoch $T$. The core intuition is:

- **High alignment ($\gamma \to 1$)**: The learning direction of this sample is consistent with the overall optimization direction of the model, representing effective generalization learning.
- **Low/negative alignment ($\gamma < 0$)**: The sample gradient conflicts with the weight direction, potentially indicating noisy labels, outliers, or overfitting signals.

### 2. GWA Definition (Distribution-Level Aggregate Metric)

The distribution $\mathcal{A}_T$ of per-sample alignment scores is aggregated into a single scalar—the **excess-kurtosis-corrected expectation**:

$$\text{GWA}_T = \frac{\mathbb{E}_i[\mathcal{A}_T]}{\text{Kurt}_i[\mathcal{A}_T] + \beta} = \frac{M_T^{(1)}}{M_T^{(4)} / (M_T^{(2)})^2 - 3 + \beta}$$

Component interpretations:

- **Numerator**: Mean of the alignment distribution $M_T^{(1)}$, reflecting overall learning efficiency.
- **Denominator**: Excess kurtosis plus offset constant $\beta = 1.2$, penalizing heavy-tailed distributions.
- **Motivation for kurtosis**: Grounded in Feldman (2020)'s long-tail theory—rare and atypical samples exert disproportionate influence on the model. High kurtosis indicates the presence of many anomalously influential samples, signaling an unhealthy learning process.

The choice of $\beta = 1.2$ ensures minimal kurtosis influence when the distribution resembles a truncated Gaussian ($\text{Kurt} \approx 0$), while significantly suppressing GWA when the distribution is heavy-tailed (e.g., Laplace).

### 3. Scalable Estimator

Computing full-network gradients directly is prohibitively expensive. The paper proposes two key optimizations to make GWA practically applicable:

**Optimization 1: Use only the last-layer (linear head) gradients.**

The classifier's core objective is to learn a linearly separable latent representation, and the last layer provides the most direct task signal. Gradients can be computed efficiently in closed form without backpropagation:

$$\mathbf{g}_T(x_i) = -z_i \cdot (\hat{y}_i - y_i)^\top$$

where $z_i$ is the latent representation, $\hat{y}_i$ is the softmax logit vector, and $y_i$ is the one-hot target.

**Optimization 2: Online epoch-level estimation.**

Rather than sweeping over the full dataset at fixed checkpoints, per-sample alignment scores are accumulated incrementally over all mini-batches within an epoch, and the first four moments of the distribution are estimated as:

$$\hat{M}_T^{(k)} = \frac{1}{N}\sum_{t=0}^{K-1}\sum_{x_i \in \mathcal{B}_{T,t}} \left(\gamma(x_i, \mathbf{w}_{T,t}) - \hat{M}_T^{(1)}\right)^k$$

**Computational overhead**: On ViT/S-16 + ImageNet-1k, GWA adds only ~2.5 seconds per epoch (~0.003 GFLOPs vs. 4.6 GFLOPs for a forward pass) with no increase in peak GPU memory (25.11 GB unchanged)—far cheaper than evaluating a 1% validation set (16 seconds).

### 4. Early Stopping Strategy

- **Training from scratch**: Skip the warm-up period (first 10% of training steps), then select the epoch with the maximum GWA value as the early stopping point.
- **Fine-tuning**: Pre-trained models start with high alignment, and GWA initially decreases as the model adapts to new data before rising again. Strategy: first identify the initial minimum, then select the epoch with maximum GWA thereafter.

## Key Experimental Results

### Table 1: Early Stopping Performance Comparison (Top-1 Test Accuracy %)

Results on ViT/S-16 across different early stopping strategies (average over 3 runs):

| Early Stopping Strategy | CIFAR-10 | CIFAR-10-N (9%) | CIFAR-10-N (17%) | ImageNet Val | ImageNet V2 | ImageNet ReaL |
|------------------------|----------|-----------------|------------------|-------------|-------------|---------------|
| Val Set (10%) | 81.10 | 78.31 | 75.23 | 73.01 | 60.01 | 79.68 |
| Val Set (1%) | 79.99 | 78.70 | 74.75 | 73.46 | 60.52 | 80.14 |
| LabelWave | 81.00 | 78.37 | 75.02 | 73.02 | 60.05 | 79.66 |
| GD | 79.22 | 77.56 | 74.66 | 67.22 | 54.59 | 74.25 |
| **GWA** | **81.57** | **78.93** | **75.70** | **73.28** | **60.53** | **79.95** |

Key findings:
- GWA outperforms the 10% validation set by 0.4% and LabelWave by 0.67% on CIFAR-10/CIFAR-10-N.
- GD collapses entirely on ImageNet (approximately 6% below baseline), with its early stopping criterion either triggering too early or never triggering.
- GWA on ViT even surpasses the 99/1% validation split while requiring no validation set whatsoever.

### Table 2: OOD Robustness Comparison (ViT/S-16 Test Accuracy %)

Performance of models selected by different early stopping criteria on corruption benchmarks:

| Model Selection Strategy | CIFAR-C Blur | CIFAR-C Digital | CIFAR-C Noise | CIFAR-C Weather | ImgNet-C Blur | ImgNet-C Digital | ImgNet-C Noise | ImgNet-C Weather |
|--------------------------|-------------|----------------|--------------|----------------|--------------|-----------------|---------------|-----------------|
| Val Set (10%) | 81.19 | 79.42 | 77.08 | 79.25 | 55.78 | 64.23 | 62.43 | 60.06 |
| Val Set (1%) | −0.88 | −1.09 | −0.68 | −1.04 | +0.59 | +0.44 | +0.43 | +0.57 |
| **GWA** | **+0.52** | **+0.53** | **+0.60** | **+0.56** | **+0.57** | **+0.61** | **+0.93** | **+0.60** |

Key findings:
- Models selected by GWA achieve an average improvement of 0.55% on CIFAR-C and 0.67% on ImageNet-C.
- This demonstrates that the training dynamics captured by GWA extend beyond in-domain performance to improve OOD robustness.
- In contrast, the 1% validation set actually degrades performance on CIFAR-C.

### Fine-tuning Supplement (Table 3)

Results for ViT/B-16 fine-tuned from ImageNet-21k: GWA surpasses the 10% validation set baseline on ImageNet Val (84.15), V2 (74.32), and ReaL (89.05), while achieving performance comparable to the 1% validation set on iNat18 (73.73) and Places365 (58.78).

## Highlights & Insights

1. **Elegantly simple core innovation**: Using the model weights themselves as the reference vector for gradient alignment reduces $O(N^2)$ pairwise gradient alignment to $O(N)$ gradient-weight alignment—both principled and efficient.
2. **Seamless theory-to-practice pipeline**: Directional convergence theory → cosine similarity definition → kurtosis correction → closed-form gradient computation → online estimator; each step is grounded in clear theoretical motivation.
3. **Dual utility**: GWA serves simultaneously as an early stopping criterion (replacing the validation set) and a data quality diagnostic tool (detecting mislabeled data).
4. **Sample-level attribution as a byproduct**: Negatively aligned samples in CIFAR-10-N are almost exclusively mislabeled instances, while highly aligned samples progress visually from simple to complex yet still representative examples—directly validating the simplicity bias hypothesis.
5. **OOD robustness gains**: GWA is not only effective in-domain; models it selects are also more robust on corruption benchmarks, confirming that it captures genuine generalization signals.
6. **Near-zero overhead**: Only 2.5 additional seconds per epoch with no memory increase—faster than evaluating a 1% validation set (16 seconds).

## Limitations & Future Work

1. **Validated only for classification with cross-entropy**: Coverage does not extend to detection, segmentation, generation, contrastive learning, or other loss formulations.
2. **Only last-layer gradients are used**: While sufficient for classification, potentially critical information may be lost for tasks that depend on multi-layer features (e.g., dense prediction).
3. **$\beta$ is fixed at 1.2**: This constant is set based on the excess kurtosis of a uniform distribution; whether adjustment is needed for different tasks or data distributions remains unexplored.
4. **Large-scale experiments are limited**: ImageNet-1k is the largest experiment; larger-scale settings (full ImageNet-21k training or LLM fine-tuning) are untested.
5. **Heuristic fine-tuning early stopping**: The strategy of first finding the minimum then the subsequent maximum requires further robustness validation across more diverse fine-tuning settings.
6. **Extension to self-supervised/autoregressive losses not yet realized**: The authors mention this direction in the conclusion but provide no preliminary results.

## Related Work & Insights

- **Generalization measures**: Classical theoretical approaches such as PAC-Bayes bounds, loss curvature (Hessian), and margin-based generalization bounds are computationally expensive and of limited practical utility.
- **Sample influence**: Influence functions (Koh & Liang, 2017) and TracIn (Pruthi et al., 2020) perform post-hoc attribution and cannot be used online.
- **Gradient coherence analysis**: Stiffness (Fort et al., 2020), Gradient Confusion (Sankararaman et al., 2020), and Coherent Gradients (Chatterjee, 2020) study pairwise gradient alignment but are unscalable in memory and computation.
- **Validation set alternatives**: LabelWave (Yuan et al., 2024) performs early stopping based on prediction changes but fails under noise; Gradient Disparity (Forouzesh & Thiran) is unreliable on large-scale data.
- **Directional convergence theory**: Ji & Telgarsky (2020) prove directional convergence of weights and gradients under idealized conditions; this paper extends those results to realistic noisy settings.
- **Simplicity bias**: Arpit et al. (2017), Rahaman et al. (2019), and others find that models learn simple features before complex ones; GWA's sample-level analysis provides direct visual evidence for this phenomenon.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Using the model weights as the reference vector for gradient alignment offers a clean and effective new perspective that translates a theoretical result into a practical tool.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple architectures (ViT/ConvNeXt), datasets (CIFAR/ImageNet/iNat18/Places365), and scenarios (noise/corruption/fine-tuning).
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical motivation is clear, derivations are rigorous, experiments are well-organized, and figures are intuitively designed.
- **Value**: ⭐⭐⭐⭐ Directly applicable to training pipelines, particularly suited for data-scarce and noise-prone settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning to Flow from Generative Pretext Tasks for Neural Architecture Encoding](learning_to_flow_from_generative_pretext_tasks_for_neural_architecture_encoding.md)
- [\[NeurIPS 2025\] Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking](flatness_is_necessary_neural_collapse_is_not_rethinking_generalization_via_grokk.md)
- [\[NeurIPS 2025\] Generalization Bounds for Rank-sparse Neural Networks](generalization_bounds_for_rank-sparse_neural_networks.md)
- [\[NeurIPS 2025\] Leveraging Importance Sampling to Detach Alignment Modules from Large Language Models](leveraging_importance_sampling_to_detach_alignment_modules_from_large_language_m.md)
- [\[NeurIPS 2025\] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training](power_lines_scaling_laws_for_weight_decay_and_batch_size_in_llm_pre-training.md)

</div>

<!-- RELATED:END -->
