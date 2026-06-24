---
title: >-
  [Paper Note] Scheduling Weight Transitions for Quantization-Aware Training
description: >-
  [ICCV 2025][Model Compression][Quantization-aware training] This paper identifies that conventional learning rate scheduling fails to control the effective step size of quantized weights in quantization-aware training (QAT), and proposes a Transition Rate (TR) scheduling technique that explicitly governs the number of discrete weight transitions via a Transition-Adaptive Learning Rate (TALR), substantially improving low-bit quantized model performance.
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "Quantization-aware training"
  - "transition rate scheduling"
  - "adaptive learning rate"
  - "network quantization"
  - "low-bit precision"
date: 2026-05-08
content_hash: 6d592c0c30829eaa
---

# Scheduling Weight Transitions for Quantization-Aware Training

**Conference**: ICCV 2025
**arXiv**: [2404.19248](https://arxiv.org/abs/2404.19248)  
**Code**: [https://cvlab.yonsei.ac.kr/projects/TRS/](https://cvlab.yonsei.ac.kr/projects/TRS/)  
**Area**: Model Compression
**Keywords**: Quantization-aware training, transition rate scheduling, adaptive learning rate, network quantization, low-bit precision

## TL;DR

This paper identifies that conventional learning rate scheduling fails to control the effective step size of quantized weights in quantization-aware training (QAT), and proposes a Transition Rate (TR) scheduling technique that explicitly governs the number of discrete weight transitions via a Transition-Adaptive Learning Rate (TALR), substantially improving low-bit quantized model performance.

## Background & Motivation

QAT learns low-bit weights by simulating quantization during training. The core mechanism maintains full-precision *latent weights* that are discretized by a quantizer to produce quantized weights for the forward pass. The optimizer updates the latent weights, and the quantized weights change only when the latent weights cross a quantizer **transition point**.

Conventional practice applies a hand-crafted learning rate (LR) schedule directly to update the latent weights. While this effectively controls the magnitude of parameter changes in full-precision training, the authors demonstrate that **this assumption breaks down in QAT**:

**The effective step size of quantized weights is weakly correlated with LR**: even with a small LR, if latent weights are concentrated near transition points, a large number of weights will still flip, causing violent fluctuations in the effective step size.

**Latent weights gravitate toward transition points in the later stages of training**: this is the fundamental mechanism behind the well-known quantized weight oscillation problem [Nagel et al., 2022].

**Coarse-to-fine optimization cannot be achieved**: LR decay guarantees convergence in full-precision training, but the drastic changes of quantized weights undermine this property in QAT.

A **scheduler specifically designed for QAT** that directly governs the actual changes in quantized weights is therefore needed.

## Method

### Overall Architecture

The core idea is to schedule a **target transition rate** (TR) for quantized weights rather than scheduling the learning rate of latent weights, and to update the latent weights with the adaptive TALR so that the actual TR tracks the target value.

### Key Designs

1. **Transition Rate (TR)**
   Defined as the fraction of quantized weights that undergo a discrete flip in a single update step:
    $k^t = \frac{\sum_{i=1}^{N} \mathbb{I}[w_d^t(i) \neq w_d^{t-1}(i)]}{N}$
   where $w_d$ denotes the discrete weights (integer values output by the round/signum function). The authors show that the effective step size of each quantized weight is approximately $|\Delta w_q^t| \approx \delta^t \cdot \mathbb{I}[w_d^t \neq w_d^{t-1}]$, i.e., the step size is either zero or a fixed value $\delta^t$, so **the mean effective step size is primarily determined by the number of transitions**.

2. **Running TR Estimation**
   The current TR is smoothed via exponential moving average:
    $K^t = m \cdot K^{t-1} + (1-m) \cdot k^t$
   with momentum $m = 0.99$ to reduce the influence of outliers.

3. **Transition-Adaptive Learning Rate (TALR)**
   The learning rate is adaptively adjusted based on the discrepancy between the running TR and the target TR:
    $U^t = \max(0, U^{t-1} + \eta(R^t - K^t))$
   When $K^t < R^t$, TALR increases to push more latent weights across transition points; otherwise it decreases. The latent weights are then updated as $\mathbf{w}^{t+1} = \mathbf{w}^t - U^t \mathbf{g}^t$.

4. **Target TR Scheduling**
   The target TR $R^t$ is decayed from an initial value to zero using a standard scheduler (e.g., cosine annealing). The initial value is set to $\lambda \sqrt{b_w}$ (where $b_w$ is the weight bit-width), ensuring that higher bit-widths begin with a larger TR.

### Loss & Training

- The training objective is identical to standard QAT (cross-entropy or distillation loss); only the optimizer's learning rate is replaced by TALR.
- Additional computational overhead consists solely of element-wise comparisons and scalar operations, increasing training time by approximately 2%.
- The method is compatible with multiple optimizers (SGD, Adam, AdamW) and various quantization schemes (binary, multi-bit).
- Learnable scale parameters of weight quantizers are fixed during TR scheduling to prevent transition point shifts from interfering with TR control.

## Key Experimental Results

### Main Results

**ImageNet Classification Top-1 Accuracy**

| Model | Bit-width (W/A) | SGD | SGD_T (Ours) | Adam | Adam_T (Ours) |
|-------|----------------|-----|-------------|------|--------------|
| MobileNetV2 | 2/2 | 46.9 | **53.6** (+6.7) | 49.6 | **53.8** (+4.2) |
| MobileNetV2 | 3/3 | 65.6 | **67.0** (+1.4) | 66.5 | **67.3** (+0.8) |
| MobileNetV2 | 4/4 | 69.9 | **70.5** (+0.6) | 70.0 | **70.8** (+0.8) |
| ResNet-18 | 1/1 | 55.3 | **55.8** (+0.5) | 56.1 | **56.3** (+0.2) |
| ResNet-18 | 2/2 | 66.8 | **66.9** (+0.1) | 66.7 | **67.2** (+0.5) |
| DeiT-T | 2/2 | - | - | 54.6 | **57.4** (+2.8) |
| DeiT-S | 2/2 | - | - | 68.4 | **71.8** (+3.4) |

Gains are largest for lightweight models under aggressive quantization (MobileNetV2 2-bit: +6.7%), demonstrating that TR scheduling is especially critical in extreme quantization regimes.

### Ablation Study

**Effect of TR Factor $\lambda$ on Performance (ResNet-20, CIFAR-100, 2-bit)**

| TR Factor | 1e-3 | 2e-3 | 3e-3 | 4e-3 | 5e-3 | 6e-3 | 7e-3 | 8e-3 |
|-----------|------|------|------|------|------|------|------|------|
| Accuracy | 62.5 | 64.2 | 64.3 | 65.3 | **65.5** | 65.1 | 63.1 | 63.6 |

The optimal TR factor lies in the range of 4e-3 to 6e-3. Sensitivity to this hyperparameter is moderate (approximately 3% spread), and all values outperform the SGD baseline (64.1).

### Key Findings

- TALR decays sharply in the later stages of training, as latent weights cluster near transition points and require extremely small updates to maintain a low TR.
- Under a step decay scheduler, the conventional LR scheme suffers severe performance degradation (ResNet-20: 64.1 → 61.3), whereas TR scheduling experiences only a marginal drop (65.5 → 64.9), demonstrating superior robustness.
- Consistent gains are also observed on object detection (MS COCO, RetinaNet; AP improvement of 0.27–0.31), validating the generalizability of the method.

## Highlights & Insights

- The paper precisely diagnoses the root cause of LR failure in QAT: the step size of quantized weights is determined by transitions, which are jointly governed by both LR and the distribution of latent weights, making LR scheduling alone insufficient.
- TALR implicitly accounts for the distribution of latent weights, constituting an elegant adaptive strategy.
- The method is extremely lightweight (only element-wise comparisons and scalar operations are added) yet consistently improves performance across diverse architectures, optimizers, and bit-widths.

## Limitations & Future Work

- The initial TR factor $\lambda$ still requires manual tuning, shifting the hyperparameter burden from LR space to TR space.
- For multi-bit quantization, the larger number of transition points and potentially differing TR dynamics across layers may render a globally unified schedule suboptimal.
- Validation is limited to image classification and object detection; applicability to NLP sequence models (e.g., LLM quantization) remains unexplored.
- Experiments are conducted exclusively on uniform quantization schemes; compatibility with non-uniform quantization methods (e.g., GPTQ, AWQ) warrants further investigation.

## Related Work & Insights

- [Nagel et al., 2022] identified the quantized weight oscillation problem and mitigated it through weight freezing or regularization; this paper addresses the issue from a more fundamental perspective (directly controlling the number of transitions), avoiding the potential training degradation associated with weight freezing.
- The TR scheduling concept can be viewed as "learning rate scheduling in quantization space," extending the well-established coarse-to-fine paradigm from full-precision training to discrete parameter optimization.
- The approach has potential implications for LLM quantization (e.g., follow-up work on QLoRA), where weight oscillation during quantized training of large models is equally severe.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First proposal of transition rate scheduling for QAT; insightful analysis)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multiple architectures, optimizers, bit-widths, and tasks; comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐⭐ (Rigorous argumentation; seamless flow from observation to analysis to solution)
- Value: ⭐⭐⭐⭐ (Lightweight and general; plug-and-play improvement for QAT)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Compute-Optimal Quantization-Aware Training](../../ICLR2026/model_compression/compute-optimal_quantization-aware_training.md)
- [\[ACL 2025\] EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](../../ACL2025/model_compression/efficientqat.md)
- [\[ICML 2025\] BoA: Attention-aware Post-training Quantization without Backpropagation](../../ICML2025/model_compression/boa_attention-aware_post-training_quantization_without_backpropagation.md)
- [\[ICLR 2026\] Towards Quantization-Aware Training for Ultra-Low-Bit Reasoning LLMs](../../ICLR2026/model_compression/towards_quantization-aware_training_for_ultra-low-bit_reasoning_llms.md)
- [\[ICML 2026\] WinQ: Accelerating Quantization-Aware Training of Language Models Around Saddle Points](../../ICML2026/model_compression/winq_accelerating_quantization-aware_training_of_language_models_around_saddle_p.md)

</div>

<!-- RELATED:END -->
