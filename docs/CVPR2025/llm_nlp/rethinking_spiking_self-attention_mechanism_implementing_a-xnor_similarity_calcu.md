---
title: >-
  [Paper Note] Rethinking Spiking Self-Attention Mechanism: Implementing a-XNOR Similarity Calculation in Spiking Transformers
description: >-
  [CVPR 2025][LLM (Other)][Spiking Neural Networks] This paper provides an in-depth analysis of the fundamental reasons why the dot product fails as a similarity metric in spiking query-key pairs due to a large number of "non-spiking events." It proposes the a-XNOR similarity metric specifically designed for spike sequences, redefining the correlation of non-spiking pairs as a specific value $a$. This approach significantly improves performance across various spiking Transforme…
tags:
  - "CVPR 2025"
  - "LLM (Other)"
  - "Spiking Neural Networks"
  - "Spiking Transformer"
  - "Self-Attention Mechanism"
  - "XNOR Similarity"
  - "Energy-Efficient Computing"
date: 2026-05-08
content_hash: c0dd0e6e20e2eb12
---

# Rethinking Spiking Self-Attention Mechanism: Implementing a-XNOR Similarity Calculation in Spiking Transformers

**Conference**: CVPR 2025  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Spiking Neural Networks, Spiking Transformer, Self-Attention Mechanism, XNOR Similarity, Energy-Efficient Computing

## TL;DR

This paper provides an in-depth analysis of the fundamental reasons why the dot product fails as a similarity metric in spiking query-key pairs due to a large number of "non-spiking events." It proposes the a-XNOR similarity metric specifically designed for spike sequences, redefining the correlation of non-spiking pairs as a specific value $a$. This approach significantly improves performance across various spiking Transformer architectures and datasets.

## Background & Motivation

**Background**: Transformers have significantly elevated the performance ceiling in various tasks due to their global receptive fields and parallelization capabilities. Researchers have begun integrating Transformers into Spiking Neural Networks (SNNs), attempting to combine the strong expressive capability of Transformers with the ultra-low power consumption benefits of SNNs. Spiking Transformers have demonstrated potential in tasks such as image classification and object detection.

**Limitations of Prior Work**: There remains a significant performance gap between existing spiking Transformers and their artificial neural network (ANN) counterparts. In traditional Transformers, self-attention measures the similarity between queries and keys via a softmax-normalized dot product. However, in SNNs, both queries and keys are binary spike sequences (consisting only of 0 and 1), making the direct use of the dot product fundamentally problematic.

**Key Challenge**: Spike sequences are highly sparse (most timesteps are 0, i.e., "non-spiking" states), and the dot product between two spike sequences is dominated by a large number of (0,0) pairs. In the standard dot product, $0 \times 0 = 0$, meaning that the information of both sequences "not spiking simultaneously" is completely ignored. In reality, "simultaneous non-spiking" also carries information about the similarity between the two sequences (analogous to true negatives in binary classification). The dot product fails to capture this information, leading to severe distortion in similarity measurement.

**Goal**: (1) Mathematically and rigorously analyze the reasons why the dot product fails in spiking self-attention. (2) Design a new similarity metric to replace the dot product, enabling proper handling of the sparse binary nature of spike sequences.

**Key Insight**: The authors design a new similarity metric based on the XNOR logical operation. The XNOR operation outputs 1 when two bits are identical and 0 when they differ, which is naturally suited for measuring the matching degree of binary spike pairs. Building on this, a learnable parameter $a$ is introduced to distinguish the importance of "simultaneous spiking" and "simultaneous non-spiking" matches.

**Core Idea**: Replace the dot product with a-XNOR to calculate the similarity of spiking $\mathbf{Q}$ and $\mathbf{K}$, redefining the contribution of non-spiking (0,0) pairs as a parameter $a$ instead of 0, thereby resolving the failure of the dot product on sparse spike sequences.

## Method

### Overall Architecture

In the self-attention layer of standard spiking Transformers, the original dot product similarity calculation is replaced by the a-XNOR similarity calculation. This method is plug-and-play and can be directly applied to various existing spiking Transformer architectures without modifying their overall structure. Input spike sequence $\rightarrow$ linear transformation to obtain spiking $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ $\rightarrow$ use a-XNOR instead of dot product to calculate the similarity between $\mathbf{Q}$ and $\mathbf{K}$ $\rightarrow$ weighted sum with $\mathbf{V}$ $\rightarrow$ output.

### Key Designs

1. **Mathematical Failure Analysis of Dot Product**:

    - **Function**: Provides the theoretical foundation explaining why a new similarity metric is needed.
    - **Mechanism**: For two binary spike vectors $\mathbf{q}, \mathbf{k} \in \{0,1\}^d$, the dot product is $\mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^d q_i \cdot k_i$. Due to spike sparsity, most dimensions have $q_i=0$ or $k_i=0$, with only a very small number of dimensions being 1 simultaneously. Consequently, the dot product values are small, lack discriminative power, and carry insufficient information. More critically, (0,0) pairs constitute the vast majority in practice, yet their contribution to the dot product is 0—this ignores the correlation information encoded by "both neurons being simultaneously silent."
    - **Design Motivation**: Mathematically and rigorously demonstrates why the standard dot product is unsuitable for spike sequences.

2. **a-XNOR Similarity Metric**:

    - **Function**: Replaces the dot product to provide a more effective similarity calculation for spike sequences.
    - **Mechanism**: Constructs similarity based on the XNOR logical operation. Standard XNOR outputs 1 when two bits are identical ((1,1) $\rightarrow$ 1, (0,0) $\rightarrow$ 1) and 0 when they differ. Building upon this, a parameter $a$ is introduced to distinguish between the two "identical" cases: when both positions have spikes (1,1), the contribution is 1 (co-activation is a strong correlation signal); when both positions have no spikes (0,0), the contribution is $a$ (where $a$ is a learnable parameter, $0 < a < 1$). The similarity formula is $\text{sim}(\mathbf{q}, \mathbf{k}) = \sum_{i} [q_i \cdot k_i + a \cdot (1-q_i)(1-k_i)]$. The introduction of $a$ serves two purposes: (1) it acknowledges that (0,0) pairs also convey correlation information ($a>0$); (2) it differentiates the importance of (1,1) and (0,0) ($a<1$), as a spike occurrence carries more information than its absence in sparse spiking regimes.
    - **Design Motivation**: XNOR naturally measures matching degrees, and the introduction of the parameter $a$ allows "simultaneous silence" and "simultaneous activation" to have different weights, aligned with the information theory principle that "rare events carry more information."

3. **Plug-and-Play Integration**:

    - **Function**: Ensures the broad applicability of the proposed method.
    - **Mechanism**: The a-XNOR similarity directly replaces the dot product calculation in various spiking Transformer architectures without requiring modifications to the network structure, training strategies, or other hyperparameters. The parameter $a$ can be set as a global constant or as a layer-wise/head-wise learnable parameter. In terms of hardware implementation, the XNOR operation can be efficiently implemented using bitwise operations, which is highly friendly to neuromorphic chips.
    - **Design Motivation**: The universality of the method is key to its practical value.

### Loss & Training

The same training configurations as each baseline spiking Transformer are used (cross-entropy loss, identical optimizers, and hyperparameters), replacing only the similarity calculation in the attention layers. The parameter $a$ is automatically optimized via backpropagation. During training, the surrogate gradient method is employed to address the non-differentiability of spiking functions.

## Key Experimental Results

### Main Results

Tested on both static and neuromorphic datasets across multiple spiking Transformer architectures:

| Architecture | Dataset | Original Accuracy | +a-XNOR | Gain |
|--------------|---------|-------------------|---------|------|
| Spikformer | CIFAR-100 | Baseline | Significant Improvement | +Significant |
| Spike-driven Trans. | CIFAR-100 | Baseline | Improvement | +Noticeable |
| Spikformer | ImageNet | Baseline | Improvement | +Substantial |
| Various Architectures | DVS128-Gesture | Baseline | Consistent Improvement | +Stable |

### Ablation Study

| $a$ Setting | Performance |
|-------------|-------------|
| $a=0$ (Degenerates to standard dot product) | Baseline performance |
| $a=\text{fixed small value}$ | Noticeable improvement |
| $a=\text{learnable parameter}$ | Optimal |
| $a=1$ (Treating 1-1 and 0-0 with equal weight) | Inferior to $a<1$ |

### Key Findings

- a-XNOR brings consistent positive improvements across all tested architectures, demonstrating the universality of the identified problem and the proposed solution.
- The optimal $a$ value typically falls between 0 and 1, validating the hypothesis that "spikes carry more information."
- Learnable $a$ tends to converge to a small positive value.
- In layers with higher sparsity, the improvement of a-XNOR is more pronounced.

## Highlights & Insights

- **In-depth Problem Analysis**: The paper rigorously analyzes why the dot product fails on spike sequences mathematically, making the motivation highly compelling.
- **Elegant Solution**: The concept of a-XNOR is concise, intuitive, and simple to implement, yet it addresses a fundamental issue.
- **Information-Theoretic Perspective**: Assigning different weights to (0,0) and (1,1) pairs echoes the principle that "rare events carry more information."
- **Highly Practical Plug-and-Play Property**: It can directly boost the performance of various existing spiking Transformer models.
- **Hardware Friendly**: The XNOR operation can be efficiently implemented via bitwise operations, making it highly suitable for deployment on neuromorphic chips.

## Limitations & Future Work

- The parameter $a$ is a scalar that treats all dimensions equally. Future work could explore dimension-wise or channel-wise values for $a$.
- The spiking properties of $\mathbf{V}$ are not specially handled.
- Validation is primarily conducted on classification tasks; this could be extended to dense tasks such as detection and segmentation.
- Theoretical analysis could be more quantitative, such as deriving an analytical relationship between the optimal $a$ value and the spike firing rate.

## Related Work & Insights

- **Spikformer**: The first work to introduce the Vision Transformer into SNNs.
- **Spike-driven Transformer**: Proposes spike-driven self-attention.
- **Insights**: The approach of rethinking and improving from the perspective of fundamental operations can be generalized to other SNN components.

## Rating

- Novelty: ⭐⭐⭐⭐ (The analysis of dot product failure in spiking attention and the proposal of a-XNOR are both highly insightful)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive validation across multiple architectures and datasets)
- Writing Quality: ⭐⭐⭐⭐ (Clear problem analysis and rigorous logic)
- Value: ⭐⭐⭐⭐ (Significant potential to drive progress in the spiking Transformer domain)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention for Transformer](spiking_transformer_introducing_accurate_addition-only_spiking_self-attention_fo.md)
- [\[CVPR 2025\] Spiking Transformer with Spatial-Temporal Attention](spiking_transformer_with_spatial-temporal_attention.md)
- [\[CVPR 2025\] STAA-SNN: Spatial-Temporal Attention Aggregator for Spiking Neural Networks](staa-snn_spatial-temporal_attention_aggregator_for_spiking_neural_networks.md)
- [\[CVPR 2025\] Exposure-slot: Exposure-centric Representations Learning with Slot-in-Slot Attention](exposure-slot_exposure-centric_representations_learning_with_slot-in-slot_attent.md)
- [\[ICLR 2026\] Rethinking Code Similarity for Automated Algorithm Design with LLMs](../../ICLR2026/llm_nlp/rethinking_code_similarity_for_automated_algorithm_design_with_llms.md)

</div>

<!-- RELATED:END -->
