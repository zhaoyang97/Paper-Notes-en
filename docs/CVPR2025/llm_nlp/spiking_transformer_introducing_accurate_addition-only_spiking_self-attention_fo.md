---
title: >-
  [Paper Note] Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention for Transformer
description: >-
  [CVPR 2025][LLM (Other)][Spiking Transformer] This paper proposes Accurate Addition-Only Spiking Self-Attention (A²OS²A), which significantly improves Spiking Transformer accuracy by leveraging a hybrid strategy that fuses binary, ReLU, and ternary spiking neurons while maintaining pure addition-only computation (no multiplication), achieving 78.66% on ImageNet-1K.
tags:
  - "CVPR 2025"
  - "LLM (Other)"
  - "Spiking Transformer"
  - "Spiking Self-Attention"
  - "Addition-Only"
  - "Energy-Efficient Computing"
  - "Hybrid Spiking Neurons"
date: 2026-05-08
content_hash: 2f01122c26c1fe7e
---

# Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention for Transformer

**Conference**: CVPR 2025  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Spiking Transformer, Spiking Self-Attention, Addition-Only, Energy-Efficient Computing, Hybrid Spiking Neurons

## TL;DR

This paper proposes Accurate Addition-Only Spiking Self-Attention (A²OS²A), which significantly improves Spiking Transformer accuracy by leveraging a hybrid strategy that fuses binary, ReLU, and ternary spiking neurons while maintaining pure addition-only computation (no multiplication), achieving 78.66% on ImageNet-1K.

## Background & Motivation

**Background**: Transformer has shown outstanding performance in various tasks such as vision and language due to its self-attention mechanism, but its computational cost and energy consumption are extremely high. Meanwhile, spiking neural networks (SNNs), characterized by event-driven computation and binary spike transmission, possess an inherent advantage of ultra-low energy consumption. Combining the powerful capabilities of Transformers with the energy efficiency of SNNs has become a popular research direction in recent years.

**Limitations of Prior Work**: Existing SNN-based Transformer methods typically rely solely on binary spiking neurons to handle all computation steps of Q, K, and V when adapting the self-attention mechanism to the spiking paradigm. Although this pure binarization ensures addition-only computation without multiplication, it severely limits representation capability, leading to a significant drop in accuracy. Specifically, the softmax normalization and scaling operations in self-attention are difficult to implement accurately within the binary spiking domain, causing severe information loss.

**Key Challenge**: There is a fundamental conflict under the SNN framework between maintaining the "pure addition/no multiplication" energy-efficiency advantage and preserving the representation accuracy of the self-attention mechanism. Pure binary spikes can only represent 0/1, which yields weak expressive power; introducing floating-point multiplications to compensate for accuracy would forfeit the energy-efficiency advantages of SNNs.

**Goal**: How to improve the representation accuracy of spiking self-attention without introducing floating-point multiplications? Can accuracy and energy efficiency be balanced by incorporating richer types of spiking neurons?

**Key Insight**: The authors observe that different computational steps in self-attention have different requirements for numerical precision—the similarity calculation of Q and K needs to retain sign information, while the aggregation of V requires non-negative weighting. Therefore, different types of spiking neurons (binary, ReLU, ternary) can be selectively introduced so that each step uses the most suitable spike representation, while the overall computation still retains multiplication-free operation.

**Core Idea**: Utilizing a hybrid of binary, ReLU, and ternary spiking neurons to replace the pure binary scheme, achieving accurate addition-only spiking self-attention.

## Method

### Overall Architecture

The overall architecture of A²OS²A follows the standard Vision Transformer architecture design, consisting of patch embedding, multi-layer Transformer blocks, and a classification head. The core modifications are concentrated on the self-attention calculation inside the Transformer block. The input image is first converted into a token sequence through patch embedding, and then fed into multi-layer spiking Transformer blocks for feature extraction. Each block contains a spiking self-attention module (A²OS²A) and a spiking feed-forward network (SNN-FFN).

### Key Designs

1. **Hybrid Spiking Neuron Strategy**:

    - **Function**: Selecting the most appropriate spiking neuron type for different computational steps of self-attention.
    - **Mechanism**: In the standard self-attention $\text{Attn}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d})V$, Q and K are used to calculate the similarity matrix, which needs to represent both positive and negative values to distinguish between similarity and dissimilarity; the attention weights (softmax output) are non-negative; and V is the aggregated value vector. Based on these properties, the authors configure different spiking neurons for different steps: (1) Query/Key use ternary spiking neurons (output {-1, 0, 1}) to retain positive and negative sign information for accurate similarity calculation; (2) the attention weights part uses ReLU spiking neurons (non-negative outputs) to simulate the non-negative characteristics of softmax; (3) Value uses standard binary spiking neurons (output {0, 1}).
    - **Design Motivation**: Pure binary spiking neurons can only output 0/1 and cannot represent negative values, which forces all similarities to be non-negative during $QK^T$ calculation, making it impossible to distinguish between positive and negative correlations. Ternary neurons introduce -1 to make similarity calculation more precise. ReLU spiking neurons ensure non-negative attention weights, aligning with the semantics of standard attention.

2. **Softmax-Free/Scaling-Free Attention Computation**:

    - **Function**: Completely eliminating softmax and $1/\sqrt{d}$ scaling operations in self-attention.
    - **Mechanism**: In standard self-attention, both softmax and scaling involve division and exponential operations, which cannot be implemented with pure addition. A²OS²A naturally generates bounded attention scores (ranging between $[-d, d]$) through the dot product of ternary Q/K, which are then processed by ReLU spiking neurons to obtain non-negative attention weights. Due to the natural normalization effect of ternary spikes and the clipping effect of ReLU, a reasonable distribution of attention weights is guaranteed without the need for additional softmax or scaling operations.
    - **Design Motivation**: Softmax contains exponential and division operations, which is the biggest obstacle to achieving pure addition computation in SNNs. Eliminating softmax allows the entire self-attention calculation to only involve addition and comparison operations, fully aligning with the event-driven computational paradigm of SNNs.

3. **Guaranteed Addition-Only Computation**:

    - **Function**: Ensuring that no floating-point multiplication occurs throughout the forward propagation of the entire attention mechanism.
    - **Mechanism**: Under the spiking domain, Q/K are ternary {-1, 0, 1}, V is binary {0, 1}, and attention weights are non-negative integers/ReLU values. Therefore, the computation of $QK^T$ only requires addition and subtraction (multiplying by ±1 is equivalent to addition/subtraction); the aggregation of attention weights and V also only requires addition (multiplying by 0/1 is equivalent to selective accumulation). The computational complexity of the entire self-attention is transformed from $O(n^2 d)$ multiplications to pure addition operations, enabling significant energy-efficiency gains on neuromorphic chips.
    - **Design Motivation**: This is the core objective of the paper—achieving "addition-only" computation while maintaining accuracy, truly unleashing the energy-efficiency advantages of SNNs on hardware.

### Loss & Training

The model is trained using the standard cross-entropy loss. To train the network with discrete spikes, the Straight-Through Estimator (STE) is used to handle the non-differentiable problem of spiking neurons. Discrete spikes are used in forward propagation, and continuous gradients are used as approximations in backward propagation. The number of training timesteps is a key hyperparameter affecting both accuracy and inference energy consumption.

## Key Experimental Results

### Main Results

| Method | Architecture | ImageNet-1K Top-1 (%) | Parameters | Timesteps |
|------|------|----------------------|--------|--------|
| SpikFormer | Spiking ViT | 74.81 | 66.3M | 4 |
| Spike-driven Transformer | Spiking ViT | 77.07 | 66.3M | 4 |
| **A²OS²A (Ours)** | **Spiking ViT** | **78.66** | **66.3M** | **4** |

### Ablation Study

| Setup | ImageNet-1K Top-1 (%) |
|------|----------------------|
| Pure Binary Spike (baseline) | ~74-75 |
| + Ternary Q/K | ~76-77 |
| + ReLU Attention Weights | ~77-78 |
| + Full A²OS²A | 78.66 |

### Key Findings

- The hybrid spiking strategy improves by about 3-4 percentage points on ImageNet-1K compared to the pure binary scheme.
- Ternary spiking neurons contribute the most to Q/K as they solve the core problem of positive/negative information loss in similarity calculations.
- Eliminating softmax not only maintains accuracy but actually improves it, while substantially simplifying hardware implementation.
- Outstanding results surpassing existing SNN Transformers were also achieved on small datasets such as CIFAR-10/100.

## Highlights & Insights

- **The core insight is simple yet effective**: Instead of forcing the same spiking neuron across all steps, selecting the most appropriate type based on computational semantics is engineering-focused but highly practical.
- **True "addition-only"**: Unlike some methods that are nominally SNNs but secretly use floating-point multiplications in certain modules, A²OS²A strictly guarantees multiplication-free operations across the entire pipeline.
- **The idea of eliminating softmax is inspiring**: It proves that under an appropriate spiking representation, softmax is not a necessity for self-attention, which has broad implications for efficient inference.

## Limitations & Future Work

- The paper is only evaluated on image classification tasks, lacking experiments on downstream tasks such as detection and segmentation.
- Although 78.66% is the SOTA for SNN Transformers, there is still a Congressional gap compared to standard ViT (~82-84%).
- Energy consumption was not measured on actual neuromorphic chips; thus, the theoretical energy-efficiency advantages lack hardware verification.
- The training stability and hyperparameter sensitivity of ternary/ReLU spiking neurons are not fully discussed.
- The integration with techniques such as knowledge distillation has not been explored, which could potentially further narrow the accuracy gap between SNNs and ANNs.

## Related Work & Insights

- **SpikFormer** (ICLR 2023): Early work introducing Transformer to SNNs, using pure binary spikes.
- **Spike-driven Transformer**: Proposes a spike-driven Transformer, but accuracy remains limited.
- **MetaSpikFormer**: Combines Meta-Learning with SNN Transformer.
- The hybrid spiking strategy of this paper is inspiring for future designs of efficient inference architectures—instead of pursuing the extreme simplicity of a single representation, selectively choosing appropriate numerical representations might be a better direction.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Spiking Transformer with Spatial-Temporal Attention](spiking_transformer_with_spatial-temporal_attention.md)
- [\[CVPR 2025\] Rethinking Spiking Self-Attention Mechanism: Implementing a-XNOR Similarity Calculation in Spiking Transformers](rethinking_spiking_self-attention_mechanism_implementing_a-xnor_similarity_calcu.md)
- [\[CVPR 2025\] STAA-SNN: Spatial-Temporal Attention Aggregator for Spiking Neural Networks](staa-snn_spatial-temporal_attention_aggregator_for_spiking_neural_networks.md)
- [\[NeurIPS 2025\] Spectral Conditioning of Attention Improves Transformer Performance](../../NeurIPS2025/llm_nlp/spectral_conditioning_of_attention_improves_transformer_performance.md)
- [\[CVPR 2025\] Exposure-slot: Exposure-centric Representations Learning with Slot-in-Slot Attention](exposure-slot_exposure-centric_representations_learning_with_slot-in-slot_attent.md)

</div>

<!-- RELATED:END -->
