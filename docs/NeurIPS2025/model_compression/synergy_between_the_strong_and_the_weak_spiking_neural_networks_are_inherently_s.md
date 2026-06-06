---
title: >-
  [Paper Note] Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing
description: >-
  [NeurIPS 2025][Model Compression][Spiking Neural Networks] This paper identifies that SNNs can be naturally decomposed into multiple sub-models along the temporal dimension. By comparing output confidence across timestep…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Spiking Neural Networks"
  - "Self-Distillation"
  - "Knowledge Distillation"
  - "Temporal Dimension"
  - "Strong2Weak"
date: 2026-05-08
content_hash: 4f094195234fe38f
---

# Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing

**Conference**: NeurIPS 2025
**arXiv**: [2510.07924](https://arxiv.org/abs/2510.07924)  
**Code**: None  
**Area**: Model Compression / Spiking Neural Networks
**Keywords**: Spiking Neural Networks, Self-Distillation, Knowledge Distillation, Temporal Dimension, Strong2Weak

## TL;DR

This paper identifies that SNNs can be naturally decomposed into multiple sub-models along the temporal dimension. By comparing output confidence across timestep sub-models to identify "strong" and "weak" instances, the paper proposes two self-distillation schemes — Strong2Weak and Weak2Strong — that significantly improve SNN performance without any external teacher model, achieving gains of up to 5.36% on neuromorphic datasets.

## Background & Motivation

Spiking Neural Networks (SNNs) transmit information via binary spikes and require only accumulation operations rather than multiply-accumulate operations, resulting in extremely low power consumption. They serve as energy-efficient alternatives to ANNs. Combined with neuromorphic chips (e.g., the Tianjic chip consuming only 0.7 mW), SNNs can complete visual tasks with ultra-low latency and power.

**Key Challenge**: Due to binary information representation, a performance gap between SNNs and ANNs persists. Knowledge distillation can improve SNN performance, but existing methods suffer from:
1. Reliance on large external teacher models (ANNs or larger SNNs), introducing additional pretraining overhead.
2. TSSD, while performing self-distillation, extends training timesteps and adds weak classifiers, substantially increasing training cost.
3. TKS depends on ground-truth labels to assess output correctness, failing to exploit dark knowledge from incorrect outputs, limiting both efficiency and performance.

**Key Insight**: The temporal nature of SNNs allows them to be naturally decomposed into multiple sub-models along the time dimension — SNN instances at each timestep produce different outputs due to differences in initial membrane potential and input current. This inherent diversity provides a natural condition for distillation learning.

## Method

### Overall Architecture

An SNN $f(\theta)$ running for $T$ timesteps is decomposed into $T$ sub-models $\{f(\theta;1), f(\theta;2), \cdots, f(\theta;T)\}$. These sub-models share the same architecture and parameters but produce different outputs due to the membrane potential dynamics of spiking neurons. Strong and weak sub-models are identified by evaluating output confidence, followed by self-distillation.

### Key Designs

1. **Identify the Strong and the Weak**:

    - The softmax output probability distribution of each sub-model is computed: $p(t) = \text{softmax}(o(t))$
    - Output confidence is defined as the maximum probability: $con(t) = \max(p(t))$
    - The sub-model with the highest confidence is designated "strong"; the one with the lowest is "weak."
    - Confidence is averaged over in-batch samples rather than determined per sample, making the process simple and label-free.

2. **Strong2Weak Distillation**:

    - The strongest sub-model $t_s$ serves as teacher; the weakest sub-model $t_w$ serves as student.
    - Logits are softened with temperature $\alpha=2$, and knowledge is transferred via KL divergence:
    - $\mathcal{L}_{S2W} = \alpha^2 KL(p(t_s) || p(t_w))$
    - Total loss: $\mathcal{L} = \mathcal{L}_{CE}(O, Y) + \lambda_{S2W} \mathcal{L}_{S2W}$ ($\lambda=1$)
    - Core Idea: Improve the weakest link by having weak sub-models learn from stronger ones.

3. **Weak2Strong Distillation**:

    - The reverse direction: the weak sub-model acts as teacher and the strong sub-model as student.
    - $\mathcal{L}_{W2S} = \alpha^2 KL(p(t_w) || p(t_s))$
    - Design Motivation: Weak models may contain latent dark knowledge — overly dominant models may overfit by ignoring fine-grained details, while weak models provide complementary information or regularization.
    - This embodies the philosophy of mutual improvement through teaching.

### Flexible Implementation Variants

- **Ensemble Teacher**: The average output of $T-1$ high/low-confidence sub-models serves as the teacher.
- **Ensemble Student**: A single strongest/weakest sub-model guides the remaining $T-1$ sub-models.
- **Simultaneous Distillation**: S2W and W2S are applied concurrently.
- **Cascaded Distillation**: Distillation proceeds level by level according to confidence ranking.

### Loss & Training

- Rectangular surrogate gradients are used for backpropagation to address the non-differentiability of spikes.
- KL divergence is the default distillation loss; MSE and Logit Standardization are also supported.
- Training timesteps $T$ match inference timesteps, introducing no additional overhead.

## Key Experimental Results

### Main Results (Multi-Dataset Comparison)

| Dataset | Architecture | Vanilla SNN | Strong2Weak | Weak2Strong |
|--------|------|-------------|-------------|-------------|
| CIFAR10 | VGG-9 | 94.21% | 94.79% (+0.58) | 94.70% (+0.49) |
| CIFAR100 | MS-ResNet18 | 76.33% | 78.25% (+1.92) | 77.98% (+1.65) |
| CIFAR10-DVS | VGG-9 | 73.97% | 78.93% (+4.96) | **79.33% (+5.36)** |
| DVS-Gesture | VGG-9 | 87.85% | **91.43% (+3.58)** | 91.20% (+3.35) |

### ImageNet Comparison

| Method | Architecture | T | Accuracy |
|------|------|---|--------|
| STAA-SNN (CVPR'25) | ResNet34 | 4 | 70.40% |
| MPS (ICLR'25) | SEW-ResNet34 | 4 | 69.03% |
| TKS | SEW-ResNet34 | 4 | 69.60% |
| **Strong2Weak** | SEW-ResNet34 | 4 | **70.53%** |
| Weak2Strong | SEW-ResNet34 | 4 | 69.87% |

### Low-Timestep Inference (CIFAR10-DVS)

| Method | T=1 | T=2 | T=3 | T=4 | T=5 |
|------|-----|-----|-----|-----|-----|
| Vanilla SNN | 10.00% | 60.10% | 69.50% | 73.30% | 74.10% |
| MPS | 66.60% | 74.30% | 75.50% | 75.70% | 76.60% |
| Weak2Strong | **73.40%** | **76.50%** | **77.50%** | **78.80%** | **79.70%** |

### Key Findings
- Gains on neuromorphic datasets substantially exceed those on static image datasets (CIFAR10-DVS +5.36% vs. CIFAR10 +0.58%), as neuromorphic data contains richer temporal features.
- Weak2Strong outperforms Strong2Weak on most neuromorphic tasks, demonstrating the significant complementary effect of dark knowledge.
- Gains at low timestep inference are substantial (T=1: from 10% to 73.4%), enabling adaptation to different latency constraints without retraining.
- Self-distillation also improves adversarial robustness: accuracy under FGSM attack increases from 19.00% to 21.23%.

## Highlights & Insights

- **Minimal yet effective design**: No additional modules or teacher models are introduced; the approach fully exploits the inherent temporal properties of SNNs.
- **Philosophical significance of Weak2Strong**: The effectiveness of having weak models teach strong ones highlights the underappreciated role of dark knowledge in distillation.
- **Confidence-based assessment without label dependency**: Strong and weak sub-models are identified without ground-truth labels, enabling application in unsupervised settings.
- **Low-latency inference capability**: Models trained with more timesteps can be deployed with fewer timesteps while maintaining competitive performance.

## Limitations & Future Work

- Applying S2W and W2S simultaneously does not yield significantly better results, as excessive similarity reduces sub-model diversity.
- How to balance diversity and similarity among sub-models remains an open question.
- The paper's claim that SNNs are "Inherently Superior in Temporal Processing" is supported primarily by classification experiments.
- Validation on more complex temporal tasks (e.g., time-series forecasting, speech recognition) is absent.
- Gains on static image datasets are limited, as inter-timestep variation is smaller in that setting.

## Related Work & Insights

- **vs. TKS**: TKS relies on labels to distinguish correct from incorrect outputs for distillation; the proposed method is label-free and can exploit dark knowledge.
- **vs. TSSD**: TSSD extends training timesteps and adds weak classifiers, increasing overhead; the proposed method introduces zero additional cost.
- **vs. Traditional Knowledge Distillation**: Conventional methods require large external teacher models; the proposed approach is purely self-distillation.
- **vs. ANN Self-Distillation (BYOT)**: ANNs require multiple additional output heads; SNNs are naturally decomposed via timesteps.

## Rating

- Novelty: ⭐⭐⭐⭐ Clever exploitation of SNN temporal properties for self-distillation; the Weak2Strong perspective is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers static and neuromorphic datasets, multiple architectures, and analyses of low-latency inference and robustness.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative progression from decomposition to identification to distillation.
- Value: ⭐⭐⭐⭐ A zero-overhead SNN training improvement that can be directly integrated into existing SNN training pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Spiking Brain Compression: Post-Training Second-Order Compression for Spiking Neural Networks](spiking_brain_compression_post-training_second-order_compression_for_spiking_neu.md)
- [\[NeurIPS 2025\] QuadEnhancer: Leveraging Quadratic Transformations to Enhance Deep Neural Networks](quadenhancer_leveraging_quadratic_transformations_to_enhance_deep_neural_network.md)
- [\[NeurIPS 2025\] Global Minimizers of ℓp-Regularized Objectives Yield the Sparsest ReLU Neural Networks](global_minimizers_of_ellp-regularized_objectives_yield_the_sparsest_relu_neural_.md)
- [\[NeurIPS 2025\] S2M-Former: Spiking Symmetric Mixing Branchformer for Brain Auditory Attention Detection](s2m-former_spiking_symmetric_mixing_branchformer_for_brain_auditory_attention_de.md)
- [\[NeurIPS 2025\] Disentangling Latent Shifts of In-Context Learning with Weak Supervision](disentangling_latent_shifts_of_in-context_learning_with_weak_supervision.md)

</div>

<!-- RELATED:END -->
