---
title: >-
  [Paper Note] Adventurer: Optimizing Vision Mamba Architecture Designs for Efficiency
description: >-
  [CVPR 2025][Causal Inference][Vision Mamba] The Adventurer series of vision models is proposed, which adapts image inputs to a unidirectional causal scanning framework through two simple designs: "Heading Average token" and "Inter-layer Flipping". This allows the Mamba architecture to achieve 4-6x the training speed of existing Vision Mamba models on vision tasks, while maintaining comparable or even superior accuracy to ViT.
tags:
  - "CVPR 2025"
  - "Causal Inference"
  - "Vision Mamba"
  - "Causal Image Modeling"
  - "Linear Complexity"
  - "State Space Models"
  - "Efficient Vision Architectures"
date: 2026-05-08
content_hash: ac86e6ade08d0d13
---

# Adventurer: Optimizing Vision Mamba Architecture Designs for Efficiency

**Conference**: CVPR 2025  
**arXiv**: [2410.07599](https://arxiv.org/abs/2410.07599)  
**Code**: [https://github.com/wangf3014/Adventurer](https://github.com/wangf3014/Adventurer)  
**Area**: Model Architecture / Vision Backbone Networks  
**Keywords**: Vision Mamba, Causal Image Modeling, Linear Complexity, State Space Models, Efficient Vision Architectures

## TL;DR
The Adventurer series of vision models is proposed, which adapts image inputs to a unidirectional causal scanning framework through two simple designs: "Heading Average token" and "Inter-layer Flipping". This allows the Mamba architecture to achieve 4-6x the training speed of existing Vision Mamba models on vision tasks, while maintaining comparable or even superior accuracy to ViT.

## Background & Motivation

**Background**: Vision Transformer (ViT) is currently the mainstream vision backbone, but the quadratic complexity of its self-attention mechanism poses severe computational and memory bottlenecks when processing high-resolution, fine-grained images. State Space Models (SSMs) like Mamba have been introduced to the computer vision field due to their linear complexity, but existing Vision Mamba architectures (such as Vim and VMamba) usually require multi-directional scanning to compensate for the information imbalance in causal modeling.

**Limitations of Prior Work**: Although multi-directional scanning does not increase the parameter count, it exponentially increases the actual computational cost and inference latency, making the practical training speed of Vision Mamba even slower than ViT. For example, the training throughput of Vim-Base is only around 200 images/s, which is significantly lower than DeiT-Base's 861 images/s.

**Key Challenge**: The information imbalance issue in causal modeling—tokens at the end of the sequence can aggregate information from all preceding tokens, whereas tokens at the beginning of the sequence lack global context, leading to severe discrepancies in representation quality. Existing solutions address this issue through multi-directional scanning, but at the cost of efficiency.

**Goal**: How to resolve the information imbalance in causal vision modeling while using only unidirectional scanning, thereby truly unleashing the linear complexity advantages of Mamba.

**Key Insight**: The authors draw inspiration from the saccade mechanism of human eyes—the human eye can only focus on a small area at a time and understands complex scenes through rapid saccades, which naturally aligns with the unidirectional scanning approach of causal modeling.

**Core Idea**: Provide a global information starting point using a Heading Average token and eliminate positional bias with inter-layer flipping, achieving the effect of multi-directional scanning with only unidirectional scanning.

## Method

### Overall Architecture
Adventurer follows the standard ViT architecture: image patching $\rightarrow$ patch embedding $\rightarrow$ position embedding $\rightarrow$ $L$ causal blocks $\rightarrow$ class token classification. Each block consists of a causal token mixer (Mamba-2 by default) and a channel mixer (SwiGLU MLP). The key differences lie in two modifications: (1) inserting a globally average-pooled token before the input of each layer to serve as the start of the sequence; (2) flipping the order of the patch token sequence between every two layers. The class token is placed at the end of the sequence.

### Key Designs

1. **Heading Average**:

    - **Function**: Provides a global context starting point for causal sequences, addressing the issue of information scarcity for tokens at the beginning of the sequence.
    - **Mechanism**: Inserts an average token $x_{\text{AVG}} = \frac{1}{n+1}\sum_j x_j$ at the beginning of the input of each layer, compressing the global information of all patch tokens. During causal scanning, every token in the sequence can at least "see" the global context through this heading token. Once processing for a layer is complete, the output of this token is discarded and recomputed in the next layer, which ensures that each layer receives the latest global information.
    - **Design Motivation**: Directly addresses the fundamental issue that tokens at the beginning of causal modeling cannot obtain subsequent information. Experiments compare various alternatives (copying the cls token, a learnable new token, multiple fine-grained tokens), demonstrating that global averaging achieves the best results and is the simplest.

2. **Inter-Layer Flipping**:

    - **Function**: Eliminates information imbalance caused by positional differences, enabling the model to learn direction-invariant features.
    - **Mechanism**: Reverses the sequence of patch tokens between every two blocks (while keeping the position of the cls token unchanged). Consequently, tokens initially at the beginning of the sequence (with less information) move to the end (with the richest information) in the next layer, and vice versa. The alternating scanning directions allow each token to alternately acquire contextual information from different directions across different layers.
    - **Design Motivation**: Compared to multi-directional scanning (2-way, 4-way), inter-layer flipping introduces zero computational overhead (the flipping operation itself has almost free cost) while achieving similar performance. Experiments show that flipping contributes more than the heading average (+0.9% vs +0.5%), likely because flipping additionally facilitates learning direction-invariant features.

3. **Mamba-2 Token Mixer + SwiGLU Channel Mixer**:

    - **Function**: Provides efficient sequence modeling and feature channel mixing.
    - **Mechanism**: The token mixer adopts Mamba-2 (the latest version of structured SSMs) to replace self-attention, with an expansion ratio of 2x and feature dimensions designed as multiples of 256 to fully exploit hardware parallelism. The channel mixer utilizes a SwiGLU MLP with a hidden layer dimension scaled to 2.5x of the input (instead of the standard 4x in standard MLPs), reducing computation while enhancing representation capacity via gating mechanisms.
    - **Design Motivation**: Ablation studies show that pure Mamba layers (without channel mixers) are feasible but suffer from a 1.3x slower speed and slightly lower accuracy. SwiGLU MLP outperforms the standard MLP (+0.1% to 0.2%) and features linear-layer operations that are more hardware-friendly.

### Loss & Training
A multi-stage training strategy is adopted: 300 epochs of pre-training at 128x128 $\rightarrow$ 100 epochs of training at 224x224 $\rightarrow$ 20 epochs of fine-tuning at 224x224 (with stronger data augmentation and higher drop path rate). This is equivalent to ~230 epochs of training at 224x224, outperforming the commonly adopted 300-epoch paradigm.

## Key Experimental Results

### Main Results

| Model | Token Mixer | Input | Params | Throughput (img/s) | ImageNet Acc |
|------|------------|------|--------|---------------|-------------|
| DeiT-Small | Self-Attn | 224 | 22M | 1924 | 79.8% |
| Vim-Small | Mamba | 224 | 26M | 395 | 80.5% |
| MambaReg-S | Mamba | 224 | 28M | 391 | 81.4% |
| **Adventurer-Small** | Mamba | 224 | 44M | **1405** | **81.8%** |
| DeiT-Base | Self-Attn | 224 | 86M | 861 | 81.8% |
| Vim-Base* | Mamba | 224 | 98M | ~200 | ~81.9% |
| **Adventurer-Base** | Mamba | 224 | 99M | **856** | **82.6%** |
| DeiT-Base | Self-Attn | 384 | 86M | 201 | 83.1% |
| **Adventurer-Base** | Mamba | 448 | 99M | **216** | **84.3%** |

Downstream tasks (ADE20k semantic segmentation / COCO object detection): Adventurer-Base achieves 46.6% mIoU on ADE20k (superior under comparable speed) and 48.4% AP^b on COCO.

### Ablation Study

| Configuration | Tiny Acc | Small Acc | Base Acc |
|------|----------|-----------|----------|
| Naive Causal (w/o HA/ILF) | - | 80.3% | - |
| + Heading Average | - | 80.8% (+0.5) | - |
| + Inter-Layer Flipping | - | 81.2% (+0.9) | - |
| + Both (Full Model) | 78.2% | **81.8% (+1.5)** | 82.6% |
| DeiT Causal → +Both | 78.8→79.9 | - | - |
| DeiT Standard (Non-causal) | 79.9 | - | - |

Channel mixer ablation: pure Mamba layers 81.6%, + standard MLP 81.7%, + SwiGLU 81.8% (Small).

### Key Findings
- The contribution of Inter-Layer Flipping (ILF) is approximately twice that of Heading Average (HA) (+0.9 vs +0.5), with the two being highly complementary.
- Combined with HA+ILF, causal DeiT strictly matches the accuracy of standard DeiT, proving that causal modeling does not sacrifice representation capacity.
- At a high resolution of 1280x1280, Adventurer-Base is 11.7x faster and 14.0x more memory-efficient than ViT-Base.
- As patch size decreases (making sequence length longer), the accuracy of Adventurer improves steadily, and its speed advantage becomes increasingly pronounced.

## Highlights & Insights
- **Unidirectional Scan Equivalent to Multi-Directional Scan**: Replacing costly multi-directional scans (which double computational cost) with zero-cost inter-layer flipping represents an elegant synergy of engineering and theory. This idea can be transferred to any sequence model requiring bidirectional information flow.
- **Causal Modeling = Standard ViT Accuracy**: Ablation studies strictly prove that after introducing simple mechanisms, causal models lose absolutely no representative capacity, suggesting that about half of the computation in standard ViT's fully-visible attention is redundant.
- **Killer Advantage in High-Resolution Scenarios**: As the input resolution increases, ViT with quadratic complexity slows down drastically, whereas Adventurer's linear complexity allows it to run at 5x speed on sequences with 3000+ tokens.

## Limitations & Future Work
- The model was trained only on ImageNet-1k, and the effects of large-scale pre-training (e.g., ImageNet-21K or self-supervised pre-training) have not been explored.
- Simple learnable matrices are still used for positional encodings, without optimizations tailored for causal scenarios (such as RoPE or other rotary position encodings).
- Only classification, segmentation, and detection tasks are currently validated, while generative tasks or multimodal scenarios have not been covered.
- The 2.5x expansion ratio of SwiGLU MLP is an empirical choice without a systematic hyperparameter search.

## Related Work & Insights
- **vs Vim**: Vim uses bidirectional scans (forward + backward), while Adventurer achieves similar performance using unidirectional scans with inter-layer flipping, yet running 4-5x faster.
- **vs VMamba**: VMamba combines Mamba with 2D convolutions for multi-directional scanning, featuring a more complex structure, whereas Adventurer is more concise and efficient.
- **vs MambaReg**: MambaReg introduces register tokens with multi-directional scanning, achieving slightly higher accuracy but at a 3-4x training cost.
- Directly valuable for high-resolution visual understanding applications (such as medical images and remote sensing imagery).

## Rating
- Novelty: ⭐⭐⭐⭐ The core innovations (HA+ILF) are simple yet effective, with clear presentation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across classification/segmentation/detection, with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ The metaphor of torch-bearing adventurers is vivid, and the structure is clear.
- Value: ⭐⭐⭐⭐ Highly valuable for high-resolution vision tasks, though requiring more large-scale validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Learning Chain of Counterfactual Thought for Bias-Robust Vision-Language Reasoning](../../ECCV2024/causal_inference/learning_chain_of_counterfactual_thought_for_bias-robust_vision-language_reasoni.md)
- [\[CVPR 2025\] Image Quality Assessment: Investigating Causal Perceptual Effects with Abductive Counterfactual Inference](image_quality_assessment_investigating_causal_perceptual_effects_with_abductive_.md)
- [\[CVPR 2025\] FG-VCE: Towards Fine-Grained Interpretability — Counterfactual Explanations for Misclassification with Saliency Partition](towards_fine-grained_interpretability_counterfactual_explanations_for_misclassif.md)
- [\[CVPR 2025\] Joint Scheduling of Causal Prompts and Tasks for Multi-Task Learning](joint_scheduling_of_causal_prompts_and_tasks_for_multi-task_learning.md)
- [\[ACL 2025\] Counterfactual Explanations for Aspect-Based Sentiment Analysis](../../ACL2025/causal_inference/counterfactual_explanations_for_aspect-based_sentiment_analysis.md)

</div>

<!-- RELATED:END -->
