---
title: >-
  [Paper Note] Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations
description: >-
  [ICCV 2025][Model Compression][PEFT] This paper proposes LieRA, which leverages Lie group theory to generalize matrix-level PEFT methods (e.g., LoRA) to high-dimensional parameter spaces (e.g.…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "PEFT"
  - "LoRA"
  - "Lie Group"
  - "High-Dimensional Parameter Space"
  - "Convolutional Kernel Fine-Tuning"
date: 2026-05-08
content_hash: ebc37777e2a6d307
---

# Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations

**Conference**: ICCV 2025
**arXiv**: [2504.00851](https://arxiv.org/abs/2504.00851)  
**Code**: [https://github.com/Chongjie-Si/Subspace-Tuning](https://github.com/Chongjie-Si/Subspace-Tuning)  
**Area**: Parameter-Efficient Fine-Tuning / Model Compression
**Keywords**: PEFT, LoRA, Lie Group, High-Dimensional Parameter Space, Convolutional Kernel Fine-Tuning

## TL;DR

This paper proposes LieRA, which leverages Lie group theory to generalize matrix-level PEFT methods (e.g., LoRA) to high-dimensional parameter spaces (e.g., convolutional kernels). By representing perturbations in the Lie algebra and mapping them back to the Lie group via the exponential map, LieRA achieves efficient fine-tuning while preserving the structural properties of the parameter space.

## Background & Motivation

- **Background**: Existing PEFT methods (LoRA and its variants) are primarily designed for two-dimensional matrices (linear layers) and face structural degradation when applied to high-dimensional parameter spaces (e.g., four-dimensional convolutional kernels). Naively reshaping LoRA's low-rank updates into convolutional kernel shapes destroys spatial locality—adjacent elements in the matrix may correspond to spatially distant positions in the kernel after reshaping.
- **Limitations of Prior Work**: Many vision foundation models (e.g., ConvNeXt, Stable Diffusion) rely heavily on convolution operations, yet no unified approach exists for fine-tuning such models without disrupting high-dimensional parameter structures.
- **Key Challenge**: Rather than designing ad hoc strategies for each type of high-dimensional parameter, a principled unified framework for generalizing existing matrix-level PEFT methods to higher-dimensional spaces is needed.

## Method

### Overall Architecture

Parameters are treated as elements of a Lie group, and updates are modeled as perturbations in the corresponding Lie algebra. The exponential map projects perturbations back onto the Lie group, ensuring smooth updates that preserve parameter space structure. A first-order Taylor approximation is applied to simplify computation, making the framework practically efficient.

### Key Designs

1. **Lie Group Construction**:

    - The set of convolutional kernel parameters $G = \{\mathcal{W} \in \mathbb{R}^{C_{in} \times C_{out} \times k \times k} | W_{c,i,j,l} \neq 0\}$ is treated as a Lie group.
    - The group operation is defined as element-wise multiplication (Hadamard product) $\odot$.
    - The identity element is the all-ones tensor $\mathcal{I}$, and the inverse is the element-wise reciprocal.
    - $G \cong \prod_{c,i,j,l} (\mathbb{R} \setminus \{0\})$ is a Cartesian product of one-dimensional Lie groups, naturally endowed with a smooth manifold structure.
    - The corresponding Lie algebra $\mathfrak{g}$ is isomorphic to $\mathbb{R}^{C_{out} \times C_{in} \times k \times k}$, forming a linear vector space.

2. **Multiplicative Parameter Update**:

    - Conventional additive update: $\mathcal{W} \rightarrow \mathcal{W} + \Delta\mathcal{W}$ (structure-breaking).
    - LieRA multiplicative update: $\mathcal{W} \rightarrow \mathcal{W} \odot \exp(\Delta\mathcal{W})$.
    - The multiplicative update scales each element proportionally, preserving relative structure and spatial locality within the kernel.
    - Since $G$ is closed under the group operation, updated parameters remain in $G$, maintaining manifold structure.

3. **First-Order Taylor Approximation**:

    - Since $\Delta\mathcal{W}$ is small, $\exp(\Delta\mathcal{W}) \approx \mathcal{I} + \Delta\mathcal{W}$.
    - The update rule simplifies to: $\mathcal{W} \odot \exp(\Delta\mathcal{W}) \approx \mathcal{W} + \mathcal{W} \odot \Delta\mathcal{W}$.
    - This approximation substantially reduces computational overhead with negligible performance loss.

### Theoretical Analysis: Rank Capacity

- LoRA rank capacity: $\mathcal{R}(\mathbf{AB}) = r$, constrained by the low rank $r$.
- LieRA rank capacity: $\mathcal{R}(\mathbf{W} \odot \mathbf{AB}) = \min(n, m)$ (full rank), since pre-trained weights are typically approximately full-rank and the Hadamard product preserves high rank.
- Full-rank capacity grants LieRA stronger expressiveness and greater flexibility for task adaptation.

### Loss & Training

- LieRA adopts the same training strategy as LoRA, replacing only the additive update with the multiplicative update.
- The framework is compatible with all matrix-level PEFT methods (LoRA, DoRA, PISSA, etc.) as a general wrapper.

## Key Experimental Results

### Main Results

| Method | #Param | VTAB-1k Avg | COCO Det mAP | COCO Seg mAP |
|--------|--------|-------------|--------------|--------------|
| Full FT | 102.05M | 78.2 | 49.0 | 43.4 |
| LoRA r=8 | 7.30M | 74.2 | - | - |
| **LieRA r=8** | **7.30M** | **75.5** | - | - |
| LoRA r=16 | 14.48M | 74.1 | 35.5 | 33.6 |
| **LieRA r=16** | **14.48M** | **75.5** | **39.1** | **37.0** |
| LoRA r=32 | 34.54M | - | 35.9 | 34.4 |
| **LieRA r=32** | **34.54M** | - | **40.5** | **38.2** |

NLP Tasks (LLaMA3-8B Commonsense Reasoning):

| Method | Params(%) | BoolQ | PIQA | HellaS. | ARC-c | Avg. |
|--------|-----------|-------|------|---------|-------|------|
| LoRA r=16 | 0.35% | 72.3 | 86.7 | 93.5 | 75.7 | 82.8 |
| **LieRA r=16** | **0.35%** | **74.7** | **87.9** | **95.6** | **79.9** | **85.1** |
| LoRA r=32 | 0.70% | 70.8 | 85.2 | 91.7 | 71.2 | 80.8 |
| **LieRA r=32** | **0.70%** | **74.3** | **88.7** | **95.4** | **80.3** | **85.3** |

### Ablation Study

Effect of Taylor Approximation (ConvNeXt-V2-B, r=16):

| Method | VTAB Avg | COCO Avg | Training Time | GPU |
|--------|----------|----------|---------------|-----|
| LieRA w/ TA | 75.5 | 42.3 | 50.17 min | 9.97 GB |
| LieRA w/o TA | 75.7 | 42.7 | 76.28 min | 14.74 GB |

Coupling with Other PEFT Methods:

| Method | VTAB Avg | COCO Avg | Overall Avg |
|--------|----------|----------|-------------|
| PISSA r=16 | 74.7 | 38.2 | 56.5 |
| PISSA+LieRA | 75.7 | 42.4 | 59.1 |
| DoRA r=16 | 74.7 | 38.4 | 56.6 |
| DoRA+LieRA | 75.5 | 42.5 | 59.0 |

### Key Findings

- LieRA yields substantial gains on CV tasks (convolutional layer fine-tuning), outperforming LoRA by 3–5 mAP on COCO detection.
- LieRA is also effective on NLP tasks (linear layer fine-tuning), achieving an average improvement of 2.3–4.5% on LLaMA3-8B.
- The first-order Taylor approximation incurs negligible performance loss while significantly reducing resource consumption (34% reduction in COCO training time, 32% reduction in GPU memory).
- LieRA serves as a general framework that can augment other PEFT methods such as DoRA and PISSA.

## Highlights & Insights

- **Elegant Mathematical Framework**: Lie group/Lie algebra theory provides a unified treatment of parameter spaces across different dimensionalities—theoretically principled and remarkably simple in implementation (only requiring addition to be replaced by element-wise multiplication).
- **Strong Generality**: LieRA is not a new standalone PEFT method but a framework capable of enhancing any matrix-level PEFT approach.
- **Theoretical Advantage in Rank Capacity**: The Hadamard product achieves full-rank capacity, granting greater expressiveness than pure low-rank updates in theory.
- **Implementation Simplicity**: The core modification is a single line of code, changing $W + \Delta W$ to $W + W \odot \Delta W$.

## Limitations & Future Work

- Validation is currently limited to four-dimensional tensors (convolutional kernels); extension to higher-dimensional parameters remains unexplored.
- The Lie group construction requires additional handling for weights containing zero values (e.g., sparse networks).
- The Taylor approximation may be insufficiently accurate when $\Delta W$ is large.
- The advantage of multiplicative updates is less pronounced for linear layers than for convolutional layers.

## Related Work & Insights

- The proposed framework could be applied to other novel architectures with high-dimensional parameters (e.g., the selective mechanism parameters in Mamba).
- The Lie group perspective may inspire PEFT methods grounded in other mathematical structures (e.g., fiber bundles).
- The full-rank capacity insight could be incorporated into the design of other low-rank methods.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The Lie group perspective is distinctive, elevating the fine-tuning problem to the level of differential geometry.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-task, multi-model validation across CV and NLP with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear, though the dense notation requires some background.
- **Value**: ⭐⭐⭐⭐⭐ Plug-and-play with a single-line code change.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TR-PTS: Task-Relevant Parameter and Token Selection for Efficient Tuning](tr-pts_task-relevant_parameter_and_token_selection_for_efficient_tuning.md)
- [\[ICLR 2026\] Memba: Membrane-driven Parameter-Efficient Fine-Tuning for Mamba](../../ICLR2026/model_compression/memba_membrane-driven_parameter-efficient_fine-tuning_for_mamba.md)
- [\[ACL 2026\] Rethinking Parameter Sharing for LLM Fine-Tuning with Multiple LoRAs](../../ACL2026/model_compression/rethinking_parameter_sharing_for_llm_fine-tuning_with_multiple_loras.md)
- [\[AAAI 2026\] Renormalization Group Guided Tensor Network Structure Search](../../AAAI2026/model_compression/renormalization_group_guided_tensor_network_structure_search.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](../../NeurIPS2025/model_compression/reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)

</div>

<!-- RELATED:END -->
