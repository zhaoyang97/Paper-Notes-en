---
title: >-
  [Paper Note] TR-PTS: Task-Relevant Parameter and Token Selection for Efficient Tuning
description: >-
  [ICCV 2025][Model Compression][PEFT] This paper proposes TR-PTS, a framework that performs task-driven layer-wise parameter selection via the Fisher Information Matrix and dynamically filters/merges tokens using CLS attention scores. By tuning only 0.34%–0.60% of parameters, TR-PTS surpasses full fine-tuning by 3.40% on FGVC and 10.35% on VTAB.
tags:
  - ICCV 2025
  - Model Compression
  - PEFT
  - Vision Transformer
  - Token Selection
  - Fisher Information Matrix
  - Parameter Selection
date: 2026-05-08
content_hash: 9b39909c360eb5b8
---

# TR-PTS: Task-Relevant Parameter and Token Selection for Efficient Tuning

**Conference**: ICCV 2025
**arXiv**: [2507.22872](https://arxiv.org/abs/2507.22872)
**Code**: [https://github.com/synbol/TR-PTS](https://github.com/synbol/TR-PTS)
**Area**: Model Compression / Parameter-Efficient Fine-Tuning
**Keywords**: PEFT, Vision Transformer, Token Selection, Fisher Information Matrix, Parameter Selection

## TL;DR

This paper proposes TR-PTS, a framework that performs task-driven layer-wise parameter selection via the Fisher Information Matrix and dynamically filters/merges tokens using CLS attention scores. By tuning only 0.34%–0.60% of parameters, TR-PTS surpasses full fine-tuning by 3.40% on FGVC and 10.35% on VTAB.

## Background & Motivation

Large-scale pretrained ViTs demonstrate strong performance on downstream vision tasks, but full fine-tuning is prohibitively expensive. Existing parameter-efficient fine-tuning (PEFT) methods suffer from three key limitations:

**Inference overhead**: Methods such as VPT introduce additional learnable modules, increasing computational cost at inference time.

**Lack of task awareness**: Most methods apply a uniform tuning strategy across tasks, ignoring the varying importance of different layers and parameters for specific tasks.

**Decoupled parameter and token optimization**: Existing work treats parameter selection and token processing independently, despite the fact that token informativeness is inherently task-dependent.

The authors observe that **different tasks rely on different subsets of tokens for final prediction** (confirmed by visualizations in Figure 2), motivating a unified framework that simultaneously performs task-relevant parameter selection and token refinement.

## Method

### Overall Architecture

TR-PTS consists of two synergistic modules: Task-Relevant Parameter Selection and Task-Relevant Token Selection, which are jointly optimized to mutually reinforce each other.

### Key Designs

1. **Task-Relevant Parameter Selection (FIM-based Layer-wise Parameter Allocation)**:

    - The Fisher Information Matrix (FIM) is used to quantify the task sensitivity of each parameter, approximated via the squared gradient of the cross-entropy loss: $\mathcal{F}(\theta) \approx \mathbb{E}_{(x,y)\sim D}\left[\left(\frac{\partial \mathcal{L}_{CE}}{\partial \theta}\right)^2\right]$
    - After selecting the top-M% parameters by FIM score, the per-layer contribution weight $w_l$ is computed and normalized to determine the number of trainable connections per neuron: $C_l = \max(1, \frac{w_l}{\min(w)} \cdot C_{\min})$
    - Within each layer, the top-$C_l$ connections per neuron are selected by FIM score as trainable parameters.
    - **Design Motivation**: Compared to gradient magnitudes used in GPS, FIM is less susceptible to optimization noise and more accurately reflects parameter importance; layer-wise allocation ensures at least one active connection per layer, preventing local network deactivation.

2. **Task-Relevant Token Selection (CLS Attention-based Dynamic Token Filtering and Merging)**:

    - The attention scores $a_i$ from the CLS token to each image token are used to measure token importance.
    - The top-$\lfloor\rho N\rfloor$ tokens with the highest attention scores are retained at selection rate $\rho$.
    - Rather than discarding unselected tokens, they are aggregated into a single merged token via attention-weighted averaging: $x_{\text{merged}} = \frac{\sum_{i\in\mathcal{I}} a_i x_i}{\sum_{i\in\mathcal{I}} a_i}$
    - The refined sequence is then: $X_{\text{refined}} = \{x_{\text{CLS}}, X_{\text{selected}}, x_{\text{merged}}\}$
    - **Design Motivation**: This combines the benefits of token pruning (reduced computation) and token merging (preserved global information).

3. **Parameter–Token Co-selection Strategy**:

    - **Key finding**: Layers with sparser parameters tend to encode less informative tokens.
    - Token reduction is therefore prioritized at parameter-sparse layers ("sparse insertion" strategy).
    - A binary mask $M$ controls gradient updates: $\Theta^{(t+1)} = \Theta^{(t)} - \eta(M \odot \nabla_\Theta \mathcal{L})$

### Loss & Training

- Standard cross-entropy loss is used for training.
- Adam optimizer with cosine learning rate decay, trained for 100 epochs.
- Backbone: ViT-B/16 pretrained on ImageNet-21K.

## Key Experimental Results

### Main Results

**VTAB-1k Benchmark (19 visual classification tasks)**:

| Method | Natural (mean) | Specialized (mean) | Structured (mean) | Overall (mean) | Params (%) |
|---|---|---|---|---|---|
| Full Fine-tuning | - | - | - | 65.57 | 100.00 |
| LoRA | - | - | - | 72.63 | 0.90 |
| GPS | - | - | - | 75.18 | 0.25 |
| **TR-PTS** | - | - | - | **75.92** | **0.34** |

**FGVC Benchmark (5 fine-grained classification datasets)**:

| Method | CUB-200 | NABirds | Flowers | Dogs | Cars | Mean | Params (%) |
|---|---|---|---|---|---|---|---|
| Full | 87.3 | 82.7 | 98.8 | 89.4 | 84.5 | 88.54 | 100.00 |
| GPS | 89.9 | 86.7 | 99.7 | 92.2 | 90.4 | 91.78 | 0.77 |
| **TR-PTS** | **90.0** | **87.1** | 99.6 | **92.4** | **90.6** | **91.94** | **0.60** |

### Ablation Study

**Component contributions (VTAB-1k subset)**:

| Token Selection | Parameter Selection | dSprites/loc | Flower102 | Sun397 |
|---|---|---|---|---|
| ✗ | ✗ | 12.5 | 97.0 | 51.0 |
| ✓ | ✗ | 14.8 | 98.8 | 51.2 |
| ✗ | ✓ | 85.1 | 99.4 | 54.2 |
| ✓ | ✓ | **87.7** | **99.5** | **54.5** |

**Token selection placement strategy comparison**:

| Strategy | Selection Rate | Sun397 | Flower102 | Loc | Camelyon |
|---|---|---|---|---|---|
| Dense | 0.95 | 53.5 | 99.3 | 85.2 | 87.3 |
| Random | 0.95 | 54.0 | 99.3 | 85.9 | 87.9 |
| **Sparse** | **0.95** | **54.5** | **99.4** | **87.7** | **88.1** |

### Key Findings

- TR-PTS achieves the lowest FLOPs and inference memory consumption among all evaluated PEFT methods.
- The distribution of FIM-critical parameters across layers varies significantly by task: Flower102 concentrates on Blocks 8/10, Sun397 on Block 0, while Patch/Camelyon exhibit a uniform distribution.
- The overlap between parameter selection sets across tasks is low (e.g., only 0.17 between Sun397 and Patch/Camelyon), validating the necessity of task-adaptive selection.
- Token visualizations show that shallower layers retain more tokens, while deeper layers progressively focus on foreground objects.

## Highlights & Insights

- The **joint optimization of parameters and tokens** is a novel contribution; the discovery that "parameter-sparse layers correspond to high token redundancy" motivates a principled co-selection strategy.
- Experiments span 24 datasets with comprehensive and consistently strong results.
- No additional parameters are introduced; there is no extra overhead during either training or inference.
- FIM provides a more stable measure of parameter importance compared to gradient magnitudes.

## Limitations & Future Work

- Validation is limited to classification tasks; extension to dense prediction tasks such as detection and segmentation remains unexplored.
- The token selection rate $\rho$ and minimum connection count $C_{\min}$ are hyperparameters requiring manual tuning.
- FIM computation requires both forward and backward passes, increasing initialization-stage cost.
- Adaptive per-layer token selection rates are not explored; a fixed $\rho$ is applied uniformly across layers.

## Related Work & Insights

- GPS is the closest predecessor, performing parameter selection via gradient magnitudes without token compression.
- ToMe (Token Merging) proposes similarity-based token merging but does not account for task relevance.
- The finding that "token compression should be applied at parameter-sparse layers" may inspire future work: in other model compression settings, identifying computationally sparse regions for more aggressive resource reduction is a promising direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The joint parameter–token selection framework and FIM-based layer-wise allocation strategy are creative contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 24 datasets, multi-dimensional ablations, and computational cost analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with rich figures and tables.
- **Value**: ⭐⭐⭐⭐ — Practically useful and a valuable reference for the PEFT community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations](generalized_tensor-based_parameter-efficient_fine-tuning_via_lie_group_transform.md)
- [\[CVPR 2026\] FAAR: Efficient Frequency-Aware Multi-Task Fine-Tuning via Automatic Rank Selection](../../CVPR2026/model_compression/faar_efficient_frequency-aware_multi-task_fine-tuning_via_automatic_rank_selecti.md)
- [\[ICLR 2026\] Memba: Membrane-driven Parameter-Efficient Fine-Tuning for Mamba](../../ICLR2026/model_compression/memba_membrane-driven_parameter-efficient_fine-tuning_for_mamba.md)
- [\[NeurIPS 2025\] Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs](../../NeurIPS2025/model_compression/recurrent_attention-based_token_selection_for_efficient_streaming_video-llms.md)
- [\[ICCV 2025\] Task Vector Quantization for Memory-Efficient Model Merging](task_vector_quantization_for_memory-efficient_model_merging.md)

</div>

<!-- RELATED:END -->
