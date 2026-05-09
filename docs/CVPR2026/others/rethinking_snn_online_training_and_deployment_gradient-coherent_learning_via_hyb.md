---
title: >-
  [Paper Note] Rethinking SNN Online Training and Deployment: Gradient-Coherent Learning via Hybrid-Driven LIF Model
description: >-
  [CVPR 2026][Spiking Neural Networks] This paper proposes the Hybrid-Driven LIF (HD-LIF) model family, which achieves gradient separability and alignment by adopting distinct spike computation mechanisms in the sub- and supra-threshold regions. This approach resolves the fundamental forward–backward propagation inconsistency in SNN online training, while simultaneously optimizing training accuracy, memory complexity, and inference power consumption across all stages.
tags:
  - CVPR 2026
  - Spiking Neural Networks
  - Online Training
  - Gradient Coherence
  - LIF Model
  - Low-Power Deployment
date: 2026-05-08
content_hash: c3087cba65b89cec
---

# Rethinking SNN Online Training and Deployment: Gradient-Coherent Learning via Hybrid-Driven LIF Model

**Conference**: CVPR 2026
**arXiv**: [2410.07547](https://arxiv.org/abs/2410.07547)
**Code**: [Available](https://github.com/haozecheng/HD-LIF)
**Area**: Other (Spiking Neural Networks / Efficient Training)
**Keywords**: Spiking Neural Networks, Online Training, Gradient Coherence, LIF Model, Low-Power Deployment

## TL;DR

This paper proposes the Hybrid-Driven LIF (HD-LIF) model family, which achieves gradient separability and alignment by adopting distinct spike computation mechanisms in the sub- and supra-threshold regions. This approach resolves the fundamental forward–backward propagation inconsistency in SNN online training, while simultaneously optimizing training accuracy, memory complexity, and inference power consumption across all stages.

## Background & Motivation

When training Spiking Neural Networks (SNNs) with Spatio-Temporal Back-Propagation (STBP), GPU memory grows **linearly** with the number of time steps, severely limiting applicability to complex scenarios. Online training keeps memory constant by truncating temporal gradient dependencies, but suffers from two fundamental drawbacks:

**Gradient Inconsistency**: Surrogate gradient functions depend on membrane potential values (e.g., the triangular function $\frac{\partial s}{\partial m} = \frac{1}{\gamma^2}\max(\gamma - |m - \theta|, 0)$), causing the temporal gradient contribution weight $\epsilon^l[i,t]$ to be a function of the membrane potential and thus non-separable. After truncating temporal gradients in online training, the forward and backward passes become inconsistent.

**No Inference Advantage**: Existing online learning methods only optimize training memory; the resulting SNNs offer no additional advantage over STBP-trained models at inference time.

## Method

### Overall Architecture

The HD-LIF model adopts different mechanisms below and above the firing threshold $\theta^l$: below the threshold, the standard LIF membrane potential dynamics are retained; above the threshold, a Precise-Positioning Reset (P2-Reset) is applied, resetting the residual potential to $\theta^l$ and transmitting a spike of corresponding magnitude:

$$\mathbf{s}_t^{l,*} = \begin{cases} \mathbf{m}_t^l - \theta_t^l, & \mathbf{m}_t^l \geq \theta_t^l \\ 0, & \text{otherwise} \end{cases}$$

The output spike is further compressed to low bit-width via a quantization function $\mathbf{Q}(\cdot, s, n, \tau)$. Synaptic weights adopt 1-bit ($\{-1, +1\}$) or 1.5-bit ($\{0, \pm 1\}$) learning modes.

### Key Designs

1. **Gradient Separability (Theorem 4.2)**: The core theoretical contribution of HD-LIF. Since $\frac{\partial s_t^{l,*}}{\partial m_t^l}$ is a constant (0 or 1) in the sub- and supra-threshold regions respectively, the surrogate gradient does not depend on the membrane potential value. This makes the temporal gradient contribution weight $\epsilon^l[i,t] = \chi^l[i,i] \prod_{j=t+1}^{i} \chi^l[j,j-1]$, where $\chi^l[i,i] \in \{0,1\}$ and $\chi^l[j,j-1] \in \{0, \lambda_j^l\}$ are constants drawn from finite sets. Consequently, the online training gradient can be seamlessly derived from the STBP gradient:
    $\left(\frac{\partial \mathcal{L}}{\partial m_t^l}\right)_{\text{Online}} = \frac{\chi^l[t,t]}{\chi^l[t,t] + \sum_{i=t+1}^{T} \chi^l[i,i] \prod_{j=t+1}^{i} \chi^l[j,j-1]} \left(\frac{\partial \mathcal{L}}{\partial m_t^l}\right)_{\text{STBP}}$

2. **Parallel HD-LIF**: Simplifies firing to $\mathbf{s}_t^{l,*} := (\mathbf{I}_t^l \geq \theta_t^l)$, removing leakage and charging dynamics. The NOPs of the neuron layer consist solely of $T$ ADD operations, significantly reducing inference overhead. These neurons are mixed into the network at a fixed ratio to balance accuracy and efficiency.

3. **Mem-BN HD-LIF**: Introduces temporal batch normalization along the membrane potential dimension, with learnable parameters $\alpha_t^l$ and $\beta_t^l$ to control the degree of normalization. A key property is that these parameters can be re-parameterized and folded into the membrane-related parameters, introducing no additional computation at inference:
    $\hat{\lambda}_t^l = \alpha_t^{l,*} \lambda_t^l, \quad \hat{\mathbf{I}}_t^l = \alpha_t^{l,*} \mathbf{I}_t^l - \beta_t^{l,*}$
   When $\alpha_t^l=1, \beta_t^l=0$, the model degenerates to vanilla HD-LIF, guaranteeing a performance lower bound.

4. **SECA Efficient Channel Attention**: A lightweight attention module with $O(K)$ parameters and $O(KC)$ computation, where spike trains share SECA weights along the temporal dimension. Two variants are proposed: $\text{SECA}_\text{I}$ (standard) and $\text{SECA}_\text{II}$ (incorporating pre- and post-synaptic input currents to compensate for the limited feature extraction capacity of compressed weights).

### Loss & Training

- Online training: at each training batch, one time step is randomly selected for gradient updates (following SLTT), making GPU memory independent of the number of time steps.
- Synaptic weights are compressed to 1-bit or 1.5-bit; 1.5-bit further reduces synaptic operations and power consumption by promoting weight sparsity.
- Membrane leakage parameters $\lambda_t^l$ and thresholds $\theta_t^l$ are set as learnable parameters to enhance adaptive gradient control.

## Key Experimental Results

### Main Results

| Dataset | Method | Backbone | Params (MB) | T | Accuracy (%) | Gain |
|--------|------|------|---------|---|----------|------|
| CIFAR-10 | GLIF (STBP) | ResNet-18 | 44.66 | 4 | 94.67 | — |
| CIFAR-10 | **HD-LIF (Ours)** | ResNet-18 | **2.82** | 4 | **95.59** | +0.92 |
| CIFAR-100 | SLTT (Online) | ResNet-18 | 44.84 | 6 | 74.38 | — |
| CIFAR-100 | **HD-LIF (Ours)** | ResNet-18 | **3.00** | 4 | **78.45** | +4.07 |
| ImageNet-1k | SLTT (Online) | ResNet-34 | 87.12 | 6 | 66.19 | — |
| ImageNet-1k | **HD-LIF (Ours)** | ResNet-34 | **10.06** | 4 | **69.77** | +3.58 |
| DVS-CIFAR10 | NDOT (Online) | VGG-SNN | 37.05 | 10 | 77.50 | — |
| DVS-CIFAR10 | **HD-LIF (Ours)** | VGG-SNN | **2.49** | 10 | **83.00** | +5.50 |

### Ablation Study

| Model Configuration | GPU Mem (GB) | Params (MB) | Accuracy (%) | NOPs (M) | Power (mJ) | Notes |
|---------|-----------|---------|----------|---------|---------|------|
| LIF (Baseline) | 1.50 | 44.84 | 71.75 | 6.59 | 0.25 | Standard online training |
| HD-LIF | 1.68 | 4.40 | 80.16 | 6.59 | 0.26 | +8.41%, 10× param compression |
| HD-LIF + 4-bit quant. | 1.92 | 4.40 | 79.62 | 6.59 | **0.03** | Power reduced to 0.03 mJ |
| HD-LIF + Parallel (50%) | 1.44 | 4.40 | 78.82 | **4.62** | 0.23 | 30% NOPs reduction |
| HD-LIF + 4-bit + Parallel | 1.70 | 4.40 | **78.61** | **4.62** | **0.02** | Best overall: 10×↓ params, 11×↓ power |

Effect of SECA attention module (CIFAR-100, ResNet-18): 78.45% → 79.33% (+0.88%), with negligible parameter overhead.

### Key Findings

- HD-LIF online training **surpasses STBP-trained models** in inference accuracy for the first time (CIFAR-10: 95.59% vs. GLIF 94.67%), overturning the conventional belief that online training necessarily sacrifices accuracy.
- On CIFAR-100, HD-LIF achieves a +4.07% accuracy gain with only 3.00 MB parameters, compared to SLTT's 44.84 MB (~15× compression).
- On static data, HD-LIF approaches near-SOTA performance within the first time step (analogous to an ANN); on neuromorphic data, it accumulates information over time steps (analogous to a conventional SNN), demonstrating the dual nature of the hybrid-driven mechanism.

## Highlights & Insights

1. **Fundamentally resolves the gradient inconsistency in online training**: Rather than mitigating the issue through approximations or regularization, the paper redesigns the spike mechanism so that gradients are naturally separable — a theoretically elegant solution.
2. **All-stage optimization**: A single framework simultaneously optimizes training memory, inference accuracy, parameter count, NOPs, and power consumption, representing a holistic rather than a single-point breakthrough.
3. **Re-parameterizable design of Mem-BN**: Auxiliary normalization is introduced during training and folded into membrane parameters at zero inference cost — highly practical from an engineering perspective.
4. **10× parameter compression with accuracy gains**: Through 1-bit/1.5-bit weight compression and the information transmission capacity of HD-LIF, the method achieves seemingly contradictory objectives simultaneously.

## Limitations & Future Work

1. Training speed is slightly slower than vanilla LIF (37.93s vs. 20.52s per epoch), as the increased number of learnable parameters introduces additional overhead.
2. Validation is currently limited to ResNet and VGG backbones; applicability to Transformer-based SNN architectures remains unexplored.
3. The effectiveness of the 1-bit/1.5-bit weight compression scheme on more complex tasks (e.g., object detection, semantic segmentation) has not been verified.
4. The mixing ratio of Parallel HD-LIF (50%) is set manually, lacking an adaptive selection strategy.

## Related Work & Insights

- **Relation to SLTT/OTTT**: These methods belong to the same SNN online training paradigm but only reduce memory by truncating gradients or selective back-propagation, without addressing gradient inconsistency. HD-LIF resolves the issue at the model level.
- **Relation to GLIF**: GLIF introduces rich neuronal dynamics but still relies on STBP, with memory growing linearly with time steps. HD-LIF achieves both rich dynamics and constant-level memory.
- **Insights**: (1) The idea of "embedding gradient friendliness into the firing mechanism" can be generalized to the design of other spike models. (2) The re-parameterization strategy allows auxiliary structures introduced during training to vanish at zero inference cost — a technique worth applying in broader contexts.

## Rating

- Novelty: ⭐⭐⭐⭐ The hybrid-driven mechanism of HD-LIF and the gradient separability theorem constitute important theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five datasets, multi-dimensional metrics, detailed ablations, and comparisons across different configurations.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivations and a progressively structured model family design, though the dense notation requires careful cross-referencing.
- Value: ⭐⭐⭐⭐ First to break the accuracy ceiling of online training, with significant implications for practical SNN deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ViT3: Unlocking Test-Time Training in Vision](vit3_unlocking_test_time_training_in_vision.md)
- [\[CVPR 2026\] IrisFP: Adversarial-Example-based Model Fingerprinting with Enhanced Uniqueness and Robustness](irisfp_adversarial-example-based_model_fingerprinting_with_enhanced_uniqueness_a.md)
- [\[CVPR 2026\] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)
- [\[CVPR 2026\] TeamHOI: Learning a Unified Policy for Cooperative Human-Object Interactions with Any Team Size](teamhoi_learning_a_unified_policy_for_cooperative_human-object_interactions_with.md)
- [\[CVPR 2026\] Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](rooftop_wind_field_reconstruction_using_sparse_sen.md)

</div>

<!-- RELATED:END -->
