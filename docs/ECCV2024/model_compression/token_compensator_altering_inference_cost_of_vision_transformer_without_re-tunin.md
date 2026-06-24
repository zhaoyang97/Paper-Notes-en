---
title: >-
  [Paper Note] Token Compensator: Altering Inference Cost of Vision Transformer without Re-Tuning
description: >-
  [ECCV 2024][Model Compression][Token Compression] This paper proposes ToCom (Token Compensator), a lightweight plug-in for model arithmetic frameworks. Acquired via rapid parameter-efficient self-distillation, ToCom can be directly inserted into any pre-trained downstream models during inference to compensate for the performance loss caused by token compression rate mismatch, without requiring re-training.
tags:
  - "ECCV 2024"
  - "Model Compression"
  - "Token Compression"
  - "Vision Transformer"
  - "Model Arithmetic"
  - "Knowledge Distillation"
  - "Inference Efficiency"
date: 2026-05-08
content_hash: ae2415bfee16b0ac
---

# Token Compensator: Altering Inference Cost of Vision Transformer without Re-Tuning

**Conference**: ECCV 2024  
**arXiv**: [2408.06798](https://arxiv.org/abs/2408.06798)  
**Code**: [Yes](https://github.com/JieShibo/ToCom)  
**Area**: Model Compression  
**Keywords**: Token Compression, Vision Transformer, Model Arithmetic, Knowledge Distillation, Inference Efficiency

## TL;DR

This paper proposes ToCom (Token Compensator), a lightweight plug-in for model arithmetic frameworks. Acquired via rapid parameter-efficient self-distillation, ToCom can be directly inserted into any pre-trained downstream models during inference to compensate for the performance loss caused by token compression rate mismatch, without requiring re-training.

## Background & Motivation

Token compression is an important technique for accelerating the training and inference of Vision Transformers (ViTs), primarily through two approaches:

**Token Pruning**: Removing unimportant tokens

**Token Merging** (e.g., ToMe): Merging similar tokens

However, existing token compression methods face a severe practical issue: **a mismatch between the compression rates during training and inference leads to significant performance degradation**. Specifically:

- If a model is trained with a compression rate of $r=8$ but needs $r=12$ during inference (larger compression for acceleration), performance drops significantly.
- Conversely, if $r=4$ is used during inference (reducing compression for higher accuracy), the performance still cannot recover to that of $r=0$.
- This limits the flexible application of token compression to off-the-shelf, pre-trained models.

**Core Problem**: How to allow users to freely adjust the compression rate during inference without needing to re-train the model for each specific rate?

## Method

### Overall Architecture

ToCom is based on the concept of "Model Arithmetic":

| Stage | Operation | Description |
|------|------|------|
| Pre-training Stage | Perform fast self-distillation on pre-trained models | Trained on ImageNet for only ~10 epochs |
| Plug-in Acquisition | Obtain the ToCom compensator set $P$ | $P = \{P_{0\rightarrow1}, P_{1\rightarrow2}, \dots, P_{15\rightarrow16}\}$ |
| Inference Stage | Insert ToCom into off-the-shelf models | Compensates for the performance loss due to compression rate mismatch |

The core idea is that the discrepancy between models with different compression rates can be captured by a lightweight "compensator". By learning this compensation relationship on a pre-trained model, it can be transferred to any downstream model.

### Key Designs

**1. Parameter-Efficient Self-Distillation**

The training pipeline of ToCom:
- Select a pre-trained model $\hat{\mathcal{M}}$ (e.g., DeiT-B)
- Randomly sample two different compression rates $m$ and $n$
- Apply ToMe (Token Merging) to $\hat{\mathcal{M}}$ with compression rates $r=m$ and $r=n$ respectively
- Learn the compensation mapping from one compression rate to another via a knowledge distillation loss

**2. Compositional Compensator Chain**

ToCom consists of a sequence of compensators between adjacent compression rates: $\mathcal{P}_{i\rightarrow(i+1)}$. When compensating from a compression rate $m$ to $n$:
- If $m < n$ (increasing compression): Cascade and add ($\oplus$) the compensators $\mathcal{P}_{m\rightarrow(m+1)}, \dots, \mathcal{P}_{(n-1)\rightarrow n}$
- If $m > n$ (decreasing compression): Apply the compensators in reverse and subtract ($\ominus$)

This design enables ToCom to handle compensation between any pair of compression rates.

**3. Relationship with Knowledge Distillation**

The training loss function is defined as:
$$\mathcal{L} = \mathcal{L}_{KD}\left(\hat{\mathcal{M}}_m \oplus \left(\bigoplus_{i=m}^{n-1} \mathcal{P}_{i\rightarrow(i+1)}\right), \hat{\mathcal{M}}_n\right)$$

This forces the output of the source model with the applied compensators to be as close as possible to the output of the target model.

### Loss & Training

- **Loss Function**: Standard knowledge distillation loss (KL divergence)
- **Training Data**: Only the pre-training dataset (ImageNet-1K)
- **Training Time**: Only 10 epochs for DeiT-B
- **Optimizer**: AdamW, learning rate 1e-3, weight decay 0.05
- **Batch Size**: 1024, trained on 8×V100 GPUs
- Compression rate range $r \in \{0, 1, 2, \dots, 16\}$, randomly sampled during training

## Key Experimental Results

### Main Results

Performance improvements on 20+ downstream tasks (DeiT-B, trained with source $r$, inferred with target $r$, +ToCom):

| Benchmark | Degradation w/o ToCom | Max Recovery with ToCom | Number of Downstream Tasks |
|----------|----------------|-------------------|-----------|
| CIFAR-100 | Significant drop | Restores up to 2.3% | 1 |
| FGVC (Fine-Grained Visual Classification) | Significant drop | Restores up to 1.5% | 4 |
| VTAB-1k | Significant drop | Restores up to 2.0% | 19 |

### ADE20k Semantic Segmentation

| Setting | mIoU | GFLOPs |
|------|------|--------|
| r=0 (No Compression) | 48.7 | 106.2 |
| r=8 | 48.0 | 91.8 |
| r=8 + ToCom | **48.3** | 91.8 |
| r=12 | 46.4 | 84.5 |
| r=12 + ToCom | **47.2** | 84.5 |
| r=16 | 41.3 | 77.3 |
| r=16 + ToCom | **43.4** | 77.3 |

### Key Findings

- ToCom can significantly recover performance losses caused by token compression without introducing any additional computational overhead (GFLOPs remain unchanged).
- The improvement of ToCom is more pronounced at higher compression rates (e.g., +2.1 mIoU on ADE20k under $r=16$).
- ToCom is versatile: learned once on a pre-trained model, it can be directly inserted into models for various downstream tasks.
- Supports various ViT architectures: DeiT-B, DeiT-S, ViT-B (MAE).
- Affordable training cost: requires only 10 epochs of self-distillation on ImageNet.
- Extendable to dense prediction tasks (e.g., semantic segmentation) by merging before FFN layers and unmerging afterwards.

## Highlights & Insights

1. **Plug-and-play**: ToCom can be directly inserted into any pre-trained downstream models without re-training, significantly improving practicality.
2. **Elegant abstraction of model arithmetic**: Modeling the discrepancy between compression rates as additive/subtractive compensators is mathematically simple and intuitive.
3. **Decoupled training and inference**: For the first time, the mismatch between training and inference compression rates in token compression is solved systematically.
4. **Economical training**: Compared to re-training a model for each compression rate, ToCom only requires one lightweight self-distillation.
5. **Extension to dense tasks**: Adaptable to semantic segmentation via a merge-before-FFN and unmerge-after-FFN strategy.

## Limitations & Future Work

- The compensation performance has an upper bound, making it unable to fully recover to the non-compressed performance level.
- Currently based only on ToMe (Token Merging), its generalizability to other token compression methods (e.g., pruning) remains unverified.
- The compression rate range $r \in \{0, \dots, 16\}$ is predefined, and finer granularity might require more compensators.
- Combining multiple compensators might introduce cumulative errors.
- Validated only on classification and segmentation tasks, leaving other downstream tasks like object detection to be explored.
- Integrating an adaptive compression rate selection mechanism could be considered.

## Related Work & Insights

- **ToMe (Token Merging)**: The baseline token compression method for this work, which accelerates ViT by merging similar tokens.
- **Parameter-Efficient Fine-Tuning (PEFT)**: The success of methods like AdaptFormer on VTAB-1k provides a downstream adaptation paradigm for this work.
- **Knowledge Distillation**: The training of ToCom is inherently self-distillation between different compression rates.
- **Model Arithmetic**: Formalizing model differences as addition/subtraction operations, likely inspired by works such as task vectors.
- This work provides an elegant solution for dynamic inference efficiency adjustment, which relates to the direction of elastic inference.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Theoretical Depth | 3.5 |
| Experimental Thoroughness | 4.5 |
| Practicality | 4.5 |
| Writing Quality | 4 |
| Overall | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ThinkingViT: Matryoshka Thinking Vision Transformer for Elastic Inference](../../CVPR2026/model_compression/thinkingvit_matryoshka_thinking_vision_transformer_for_elastic_inference.md)
- [\[ECCV 2024\] Isomorphic Pruning for Vision Models](isomorphic_pruning_for_vision_models.md)
- [\[CVPR 2025\] BHViT: Binarized Hybrid Vision Transformer](../../CVPR2025/model_compression/bhvit_binarized_hybrid_vision_transformer.md)
- [\[CVPR 2026\] LoPrune: Efficient Data Pruning for LoRA-Based Fine-Tuning of Vision Transformer](../../CVPR2026/model_compression/loprune_efficient_data_pruning_for_lora-based_fine-tuning_of_vision_transformer.md)
- [\[ECCV 2024\] PaPr: Training-Free One-Step Patch Pruning with Lightweight ConvNets for Faster Inference](papr_training-free_one-step_patch_pruning_with_lightweight_convnets_for_faster_i.md)

</div>

<!-- RELATED:END -->
