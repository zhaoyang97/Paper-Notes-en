---
title: >-
  [Paper Note] Rethinking Long-tailed Dataset Distillation: A Uni-Level Framework with Unbiased Recovery and Relabeling
description: >-
  [AAAI 2026][Model Compression][Dataset Distillation] This paper proposes the first uni-level dataset distillation framework for long-tailed distributions. Through three core strategies — expert model debiasing…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "Dataset Distillation"
  - "Long-tailed Distribution"
  - "Uni-level Optimization"
  - "BN Statistics Calibration"
  - "Unbiased Recovery"
date: 2026-05-08
content_hash: e1b82fdf627a61d7
---

# Rethinking Long-tailed Dataset Distillation: A Uni-Level Framework with Unbiased Recovery and Relabeling

**Conference**: AAAI 2026
**arXiv**: [2511.18858](https://arxiv.org/abs/2511.18858)  
**Code**: None  
**Area**: Model Compression
**Keywords**: Dataset Distillation, Long-tailed Distribution, Uni-level Optimization, BN Statistics Calibration, Unbiased Recovery

## TL;DR
This paper proposes the first uni-level dataset distillation framework for long-tailed distributions. Through three core strategies — expert model debiasing, fair BN statistics calibration, and confidence-guided initialization — the method achieves +15.6% on CIFAR-100-LT and +11.8% on Tiny-ImageNet-LT, comprehensively outperforming DAMED.

## Background & Motivation

Dataset Distillation (DD) aims to synthesize a compact yet representative dataset such that models trained on it approximate the performance of models trained on the full dataset, which is critical for resource-constrained scenarios.

**Core challenges under long-tailed distributions**: Class imbalance is pervasive in real-world settings (a few head classes with abundant samples, many tail classes with scarce samples). However, existing DD methods (DREAM, DATM, EDC, etc.) assume uniform data distributions, leading to:

- **Synthetic sets dominated by head classes**: tail classes are under-represented
- **Biased BN statistics**: imbalanced class frequencies corrupt the mean and variance estimates of batch normalization
- **Training instability**: medium-frequency classes receive unstable or insufficient gradient feedback

**Limitations of DAMED (the only prior work)**: DAMED is the sole existing work explicitly targeting long-tailed DD, but it suffers from three fundamental issues:

- **Tail class under-representation**: relies on feature-extracting experts trained on long-tailed data without debiasing, yielding poor tail class representation quality
- **Unintended trade-offs in trajectory matching**: in bi-level optimization, medium-frequency classes are affected by unstable gradients
- **High computational overhead**: bi-level trajectory optimization requires substantial GPU memory and time

**Key Challenge**: Trajectory matching methods are inherently in conflict with debiasing strategies — debiasing operations (e.g., reweighting, logit adjustment) alter the expert's optimization trajectory, undermining the premise of trajectory matching, while post-hoc debiasing is infeasible.

**Key Insight**: The paper abandons the trajectory matching paradigm in favor of a uni-level statistical alignment framework. Unbiased distillation is achieved through two complementary components — unbiased synthetic image recovery (via an Observer model) and unbiased soft-label relabeling (via a Teacher model) — supported by three dedicated strategies.

## Method

### Overall Architecture

The pipeline consists of a preparation phase and a distillation phase:
1. Train a debiased Observer model (for BN statistics alignment) and a Teacher model (for soft label generation)
2. Freeze the Observer model and perform fair BN statistics calibration over the entire training set
3. Apply a confidence-guided multi-round initialization strategy using the Teacher model to generate the initial synthetic image set
4. Recover synthetic images via BN statistics alignment, then relabel with soft labels using the Teacher model
5. Train a student model on the distilled set to evaluate quality

### Key Designs

#### 1. **Expert Model Debiasing (Observer + Teacher)**
- **Function**: Eliminate class imbalance bias in both the Observer and Teacher models
- **Mixed consistency loss** (for robustness enhancement):
$$\mathcal{L}_{robust} = -\sum_{i=1}^{2} \cos(\mathbf{z}_i, \text{sg}(\mathbf{p}_{\bar{i}}))$$
  Symmetric alignment is performed on two mixed-label augmented views, where $\text{sg}(\cdot)$ denotes the stop-gradient operator to ensure unidirectional alignment.
- **Class debiasing loss** (for rebalanced class supervision):
$$\mathcal{L}_{debias} = \alpha \sum_{k=0}^{C-1} \frac{-(r_k)^{-q} y_k \log p_k}{\sum_j (r_j)^{-q}} - \beta \sum_{k=0}^{C-1} y_k \log p_k$$
  where $\alpha = (t/T)^2$ and $\beta = 1-\alpha$, with a dynamic schedule that progressively shifts focus toward minority classes.
- **Design Motivation**: A biased Observer leads to biased BN statistics, which in turn produces biased synthetic images; a biased Teacher produces inaccurate soft labels, causing semantic guidance to fail. Debiasing must occur at the source.

#### 2. **Fair BN Statistics Calibration**
- **Function**: Eliminate both intra-class bias and inter-class bias in BN statistics
- **Dynamic momentum calibration**: The Observer model parameters are frozen, and a single forward pass over the entire training set is performed. For each BN layer and each class, statistics are updated using a dynamically adjusted momentum:
$$\mu_{l,t}^c = (1 - \alpha_t^c) \cdot \mu_{l,t-1}^c + \alpha_t^c \cdot \hat{\mu}_{l,t}^c, \quad \alpha_t^c = \frac{B_t^c}{N_{t-1}^c + B_t^c}$$
  where $B_t^c$ is the number of class-$c$ samples in the current batch and $N_{t-1}^c$ is the cumulative count from previous steps.
- **Global equalized averaging**:
$$\mu_l(\mathcal{D}; \theta_R) = \frac{1}{C} \sum_{c=0}^{C-1} \mu_{l,T}^c(\mathcal{D}; \theta_R)$$
- **Design Motivation**: Standard BN uses exponential moving averages with fixed momentum, causing recent batches to dominate the statistics while early batches are forgotten. Under long-tailed settings, every tail class sample carries high representational value and must contribute equally. The dynamic momentum ensures each sample contributes equally to the final statistics.

#### 3. **Confidence-Guided Multi-Round Initialization**
- **Function**: Provide diverse and high-quality initialization for the synthetic dataset
- **Mechanism**: Multiple augmented variants (e.g., random crops) are generated for each real image and scored using the Teacher model's negative cross-entropy. A multi-round selection strategy is applied: in each round, each image contributes at most one unused highest-confidence augmentation, ensuring sample-level diversity.
- **Tail class handling**: When real tail class samples are insufficient, zero-initialized placeholders are inserted to maintain structural consistency across classes.
- **Design Motivation**: Random initialization leads to poor convergence; directly sampling real images is infeasible for tail classes due to scarcity. Multi-round selection combined with confidence-guided scoring balances quality and diversity.

### Loss & Training

- Expert model training: $\mathcal{L} = \gamma_1 \mathcal{L}_{robust} + \gamma_2 \mathcal{L}_{debias}$
- Statistical alignment loss: $\mathcal{L}(\mathcal{S}) = \sum_{l=1}^{L} \mathbf{D}_l^\mu(\mathcal{S}, \mathcal{D}; \theta_R) + \mathbf{D}_l^\sigma(\mathcal{S}, \mathcal{D}; \theta_R)$
- Student model training (evaluation): $\mathcal{L}_{match} = \kappa_1 \cdot \mathcal{L}_{CE}(s(x_s^i), y_s^i) + \kappa_2 \cdot \|\tilde{\mathbf{y}}_s^i - s(x_s^i)\|_2^2$
- Student network: depth-3 ConvNet for CIFAR, depth-4 ConvNet for Tiny-ImageNet, ResNet-50 additionally for ImageNet-LT
- Evaluation training for 1000 epochs; experiments repeated 5 times; primarily conducted on a single RTX 3090

## Key Experimental Results

### Main Results

| Dataset | IF | IPC | DAMED | Ours | Gain |
|--------|-----|-----|-------|------|------|
| CIFAR-10-LT | 100 | 10 | 53.4% | **62.7%** | +9.3% |
| CIFAR-10-LT | 100 | 50 | 64.0% | **68.8%** | +4.8% |
| CIFAR-100-LT | 10 | 10 | 31.5% | **47.1%** | **+15.6%** |
| CIFAR-100-LT | 50 | 10 | 29.8% | **42.1%** | +12.3% |
| Tiny-ImageNet-LT | 10 | 10 | 26.0% | **37.8%** | **+11.8%** |
| ImageNet-LT | 5 | 10 | 20.8% | **24.7%** | +3.9% |
| ImageNet-LT | 10 | 10 | 20.3% | **23.5%** | +3.2% |

### Extreme Setting (IPC=1, one synthetic image per class)

| Dataset | DAMED | Ours | Gain |
|--------|-------|------|------|
| CIFAR-10-LT (IF100) | 24.1% | **44.8%** | +20.7% |
| CIFAR-100-LT (IF50) | 7.8% | **31.8%** | +24.0% |
| Tiny-ImageNet-LT (IF100) | 6.0% | **20.1%** | +14.1% |

### Ablation Study (CIFAR-100-LT, IF=50)

| Configuration | IPC=10 | IPC=20 | IPC=50 |
|------|--------|--------|--------|
| w/o model debiasing | 31.7 | 32.3 | 32.8 |
| w/o statistics calibration | 40.9 | 41.8 | 42.1 |
| w/o adaptive initialization | 40.8 | - | - |
| **Full method** | **42.1** | **43.4** | **44.2** |

### Computational Efficiency Comparison (runtime)

| Method | CIFAR-10-LT (IF100) | CIFAR-100-LT (IF50) |
|------|-----|-----|
| DAMED: expert training | 31388s | 26269s |
| DAMED: distillation synthesis | 30141s | 29328s |
| Ours: expert training | **2395s** | **2183s** |
| Ours: distillation synthesis | **118s** | **273s** |

Total runtime is less than 1/20 of DAMED.

### Key Findings

1. **Model debiasing is the most critical component**: its removal causes the largest performance drop (42.1→31.7), as a biased expert directly limits the performance ceiling of distillation.
2. **Greater advantage under extreme imbalance**: at IF=256 (ImageNet-LT, ResNet-50), the proposed method achieves 48.2% vs. DAMED's 17.2%, a +31% improvement.
3. **Strong cross-architecture generalization**: the method significantly outperforms DAMED on ConvNet-3/VGG-11/ResNet-18/AlexNet, with smaller performance variance across architectures.
4. **Substantial tail class accuracy gains**: class-wise analysis shows the proposed method outperforms DAMED on both head and tail classes, with especially notable improvements on tail classes.
5. **Constant memory footprint**: GPU memory remains constant across different IPC settings (~3.1 GB), whereas DAMED's memory scales linearly with IPC.

## Highlights & Insights

1. **Paradigm shift**: from bi-level trajectory matching to uni-level statistical alignment, fundamentally resolving the conflict between debiasing and trajectory matching.
2. **Mathematical elegance of dynamic momentum**: the simple formula $\alpha_t^c = B_t^c / (N_{t-1}^c + B_t^c)$ naturally ensures equal contribution from each sample regardless of processing order, while simultaneously eliminating both intra-class and inter-class bias.
3. **20× speedup with constant memory**: the dramatic improvement in computational efficiency renders the method highly practical.
4. **Extreme IPC=1 scenario**: the method still substantially outperforms baselines when only one synthetic image per class is available, demonstrating the value of the debiasing strategy under extreme information scarcity.
5. **Surpassing balanced dataset distillation**: at IF=256, the method even outperforms certain methods distilled on the full balanced ImageNet-1K.

## Limitations & Future Work

1. Validation is currently limited to image classification; the framework can be extended to detection, segmentation, and other tasks.
2. The Teacher and Observer use the same debiasing strategy; differentiated debiasing schemes are worth exploring.
3. Confidence-guided initialization depends on the quality of the augmentation strategy employed.
4. The framework can be extended to federated learning or multi-domain dataset distillation scenarios.
5. Generative model-assisted initialization has not been explored.

## Related Work & Insights

- **DAMED (CVPR 2025)**: The only prior long-tailed DD work; uses trajectory matching with frequency-aware shifts, but inherits representational bias from biased experts.
- **EDC (NeurIPS 2024)**: A uni-level dataset distillation method effective in balanced settings, but lacks debiasing strategies.
- **RDED (CVPR 2024)**: A uni-level method based on real image augmentation; suffers from tail class sampling difficulties under imbalance.
- **SRe2L (CVPR 2023)**: Reduces memory overhead by replacing trajectory matching with feature matching.
- **UniMix**: A class-aware Mixup augmentation strategy that provides inspiration for tail class augmentation in long-tailed recognition.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rectifying Soft-Label Entangled Bias in Long-Tailed Dataset Distillation](../../NeurIPS2025/model_compression/rectifying_soft-label_entangled_bias_in_long-tailed_dataset_distillation.md)
- [\[AAAI 2026\] TGDD: Trajectory Guided Dataset Distillation with Balanced Distribution](tgdd_trajectory_guided_dataset_distillation_with_balanced_distribution.md)
- [\[ICLR 2026\] Dataset Color Quantization: A Training-Oriented Framework for Dataset-Level Compression](../../ICLR2026/model_compression/dataset_color_quantization_a_training-oriented_framework_for_dataset-level_compr.md)
- [\[AAAI 2026\] EEG-DLite: Dataset Distillation for Efficient Large EEG Model Training](eeg-dlite_dataset_distillation_for_efficient_large_eeg_model_training.md)
- [\[AAAI 2026\] Post Training Quantization for Efficient Dataset Condensation](post_training_quantization_for_efficient_dataset_condensation.md)

</div>

<!-- RELATED:END -->
