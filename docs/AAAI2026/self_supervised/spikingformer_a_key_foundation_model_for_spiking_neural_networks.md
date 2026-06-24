---
title: >-
  [Paper Note] Spikingformer: A Key Foundation Model for Spiking Neural Networks
description: >-
  [AAAI 2026 Oral][Self-Supervised Learning][Spiking Neural Networks] This paper proposes Spikingformer, which integrates MS Residual with Self-Attention in a spike-driven manner to address the non-spike computation introduced by SEW Residual in Spikformer, while preserving global modeling capability.
tags:
  - "AAAI 2026 Oral"
  - "Self-Supervised Learning"
  - "Spiking Neural Networks"
  - "Transformer"
  - "spike-driven"
  - "residual connection"
  - "energy-efficient AI"
date: 2026-05-08
content_hash: afff07aae0045a53
---

# Spikingformer: A Key Foundation Model for Spiking Neural Networks

**Conference**: AAAI 2026 Oral  
**arXiv**: [2304.11954](https://arxiv.org/abs/2304.11954)  
**Code**: [GitHub](https://github.com/TheBrainLab/Spikingformer)  
**Area**: Self-Supervised
**Keywords**: Spiking Neural Networks, Transformer, spike-driven, residual connection, energy-efficient AI

## TL;DR

This paper proposes Spikingformer, which integrates MS Residual with Self-Attention in a spike-driven manner to address the non-spike computation introduced by SEW Residual in Spikformer, while preserving global modeling capability.

## Background & Motivation

### State of the Field

**Background**: The core advantage of SNNs lies in event-driven spike computation: replacing energy-intensive multiply-accumulate operations (MAC, 4.6pJ) with low-power accumulate operations (AC, 0.9pJ). However, existing SNN backbones exhibit a fundamental conflict:

### Limitations of Prior Work

**Limitations of Prior Work**: SEW ResNet / Spikformer employ SEW Residual connections, whose output range after residual addition is $\{0,1,2,...,16\}$, causing the subsequent convolutional layer to perform integer–floating-point multiplications, thereby breaking the spike-driven property.

### Root Cause

**Key Challenge**: SD-Transformer uses MS Residual with linear attention, preserving spike-driven computation but sacrificing global modeling capability.

### Solution

**Goal**: How can one maintain **global self-attention modeling** capability while ensuring **purely spike-driven** computation (accumulate-only operations) throughout the entire network?

## Method

### Overall Architecture

Spikingformer = Spiking Tokenizer + $L$ Spiking Transformer Blocks + Classification Head

### Key Designs

**MS Residual (Core Modification)**: The SN layer is placed before ConvBN:
$$O_l = \text{ConvBN}_l(\text{SN}_l(O_{l-1})) + O_{l-1}$$

Compared to SEW Residual: $O_l = \text{SN}_l(\text{ConvBN}_l(O_{l-1})) + O_{l-1}$

In MS Residual, the SN layer ensures that the data fed into ConvBN are pure spikes (0/1), so ConvBN performs only floating-point additions. The floating-point values after residual addition are subsequently converted back to spikes by another SN layer before the next ConvBN.

**Pre-activation SSA (PSSA)**: The positions of SN layers within Spikformer's SSA are rearranged so that SN is applied before ConvBN:
$$Q = \text{SN}_Q(\text{ConvBN}_Q(\text{SN}(X))), \quad K, V \text{ analogously}$$
$$\text{Attention}(Q,K,V) = \text{ConvBN}(\text{SN}(QK^TV \cdot s))$$

where $Q, K, V \in \{0,1\}^{T \times N \times D}$ are pure spike tensors and $s$ is a scaling factor.

**Spiking Tokenizer**: A multi-stage ConvBN-SN structure supporting optional MaxPooling downsampling.

**Spikingformer†**: An enhanced variant employing CML (ConvBN-MaxPool-LIF) downsampling to improve gradient backpropagation.

### Energy Consumption

$$E_{SNN} = E_{AC} \times \Big(\sum_{i=2}^N SOP_{Conv}^i + \sum_{j=1}^M SOP_{SSA}^j\Big) + E_{MAC} \times FLOP_{Conv}^1$$

where $SOP^l = fr \times T \times FLOPs^l$ (fr denotes firing rate). Only the first layer, which encodes non-spike inputs, requires MAC operations.

## Key Experimental Results

### Main Results

| Model | Params | Time Steps | ImageNet Top-1 | Energy (mJ) |
|-------|--------|------------|----------------|-------------|
| Spikformer-8-768 | 66.34M | 4 | 74.81% | 32.07 |
| SD-Transformer-8-768 | 66.34M | 4 | 77.07% | 6.09 |
| Spikingformer-8-768 | 66.34M | 4 | 75.85% | 13.68 |
| **Spikingformer†-8-768** | 66.34M | 4 | **77.64%** | 16.30 |
| ANN Transformer-8-512 | 29.68M | 1 | 80.80% | 38.34 |

- CIFAR-10: 95.95% (Spikingformer†-9.32M, T=4)
- CIFAR-100: 80.37% (Spikingformer†-9.32M, T=4)
- DVS128 Gesture: 98.6% (T=16)
- Comprehensively evaluated across 13 datasets

## Highlights & Insights

- Systematically analyzes the spike-driven behavior of SEW vs. MS Residual, revealing that the non-spike output range of SEW Residual grows linearly with network depth.
- The only SNN backbone that simultaneously achieves spike-driven computation and global attention.
- Energy consumption is approximately 42% of that of ANN Transformer (16.30 vs. 38.34 mJ), with the accuracy gap narrowed to 3.16%.
- The CML downsampling variant, Spikingformer†, further improves performance.

## Limitations & Future Work

- An accuracy gap of approximately 3% remains compared to ANN Transformer (77.64 vs. 80.80).
- Energy estimates are based on theoretical 45nm assumptions; deployment effectiveness on actual neuromorphic hardware has not been validated.
- The $QK^TV$ operation in global attention still requires a floating-point scaling factor $s$, making it not fully spike-driven.
- The classification head's AvgPooling-FC still involves floating-point operations.

## Related Work & Insights

| Method | Spike-Driven | Global Attention | ImageNet Acc |
|--------|-------------|-----------------|-------------|
| SEW ResNet-152 | ✗ | ✗ | 69.26% |
| MS-ResNet-104 | ✓ | ✗ | 76.02% |
| Spikformer | ✗ | ✓ | 74.81% |
| SD-Transformer | ✓ | ✗ | 77.07% |
| **Spikingformer†** | **✓** | **✓** | **77.64%** |

## Insights

- The concise design of "repositioning the SN layer" suffices to eliminate non-spike computation, underscoring the importance of operator ordering within network architectures.
- The work provides a significant experimental benchmark for the SNN community (13 datasets), promoting standardized evaluation.
- The combination of spike-driven computation and global modeling paves the way for deploying SNNs in a broader range of general-purpose tasks.

## Rating

⭐⭐⭐⭐ — Solid engineering contribution with clean and effective design and comprehensive experiments, though the core innovation (repositioning the SN layer) is relatively straightforward.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Robust Spiking Neural Networks by Temporal Mutual Information](../../CVPR2026/self_supervised/robust_spiking_neural_networks_by_temporal_mutual_information.md)
- [\[CVPR 2026\] On the Role of Temporal Granularity in the Robustness of Spiking Neural Networks](../../CVPR2026/self_supervised/on_the_role_of_temporal_granularity_in_the_robustness_of_spiking_neural_networks.md)
- [\[CVPR 2026\] Reconstructing Spiking Neural Networks Using a Single Neuron with Autapses](../../CVPR2026/self_supervised/reconstructing_spiking_neural_networks_using_a_single_neuron_with_autapses.md)
- [\[ICLR 2026\] PredNext: Explicit Cross-View Temporal Prediction for Unsupervised Learning in Spiking Neural Networks](../../ICLR2026/self_supervised/prednext_explicit_cross-view_temporal_prediction_for_unsupervised_learning_in_sp.md)
- [\[ICML 2026\] How 'Neural' is a Neural Foundation Model?](../../ICML2026/self_supervised/how_neural_is_a_neural_foundation_model.md)

</div>

<!-- RELATED:END -->
