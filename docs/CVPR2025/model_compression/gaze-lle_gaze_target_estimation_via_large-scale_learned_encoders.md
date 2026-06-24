---
title: >-
  [Paper Note] Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders
description: >-
  [CVPR 2025][Model Compression][Gaze target estimation] Gaze-LLE is proposed, a minimalist gaze target estimation framework based on a frozen DINOv2 encoder. With only ~2.8M trainable parameters (1-2 orders of magnitude fewer than previous methods), no auxiliary depth/pose models, and no independent head encoder, it achieves state-of-the-art (SOTA) performance (AUC 0.958) on benchmarks such as GazeFollow and VideoAttentionTarget, using only person position prompting and a ligh…
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Gaze target estimation"
  - "DINOv2"
  - "foundation models"
  - "position prompting"
  - "lightweight decoder"
date: 2026-05-08
content_hash: 02ea6a9cb461671b
---

# Gaze-LLE: Gaze Target Estimation via Large-Scale Learned Encoders

**Conference**: CVPR 2025  
**arXiv**: [2412.09586](https://arxiv.org/abs/2412.09586)  
**Code**: [github.com/fkryan/gazelle](https://github.com/fkryan/gazelle)  
**Area**: Model Compression / Gaze Estimation  
**Keywords**: Gaze target estimation, DINOv2, foundation models, position prompting, lightweight decoder

## TL;DR

Gaze-LLE is proposed, a minimalist gaze target estimation framework based on a frozen DINOv2 encoder. With only ~2.8M trainable parameters (1-2 orders of magnitude fewer than previous methods), no auxiliary depth/pose models, and no independent head encoder, it achieves state-of-the-art (SOTA) performance (AUC 0.958) on benchmarks such as GazeFollow and VideoAttentionTarget, using only person position prompting and a lightweight transformer decoder.

## Background & Motivation

**Background**: Gaze target estimation predicts where a person is looking in a scene and is a key component for understanding human behavior. Previous methods all employ multi-branch architectures consisting of an independent head encoder and a scene encoder, combined with auxiliary models for depth, pose, or object detection, leading to complex feature fusion.

**Limitations of Prior Work**: (1) Multi-branch architectures are complex to train, requiring meticulously designed fusion mechanisms and multi-task loss functions; (2) The number of trainable parameters is large (commonly 30-100M+); (3) Convergence is slow, typically requiring dozens of GPU hours. Although foundation models like DINOv2 perform exceptionally well in dense prediction tasks such as depth estimation, directly replacing backbones in existing gaze architectures paradoxically degrades performance.

**Key Challenge**: While DINOv2 features are powerful, existing gaze architectures cannot exploit them effectively. This is because the multi-branch design requires head location information to be injected before encoding (RGB + head channel), which is incompatible with the input format when DINOv2 is frozen.

**Key Insight**: Three essential design decisions: (1) injecting the head position after the encoder rather than before; (2) decoding with a transformer instead of a CNN to obtain global information propagation; (3) eliminating the independent head branch since DINOv2 already encodes sufficient head orientation information.

**Core Idea**: Frozen DINOv2 + position-prompted lightweight decoder = a minimalist and SOTA gaze estimation framework.

## Method

### Overall Architecture

The input RGB image is processed by a frozen DINOv2 encoder to extract scene token features ($d_\mathcal{F} \times H \times W$), which are then linearly projected to $d_\text{model}$. A learnable head position embedding (head prompting) is added to the feature map at the corresponding head position, after which the features are fed into a 3-layer transformer encoder for update. Finally, they are upsampled and decoded into a heatmap along with an optional in/out classification.

### Key Designs

1. **Head Position Prompting**: The head bounding box is downsampled into a binary mask $M$. A learnable embedding $p_\text{head}$ is added to the tokens at corresponding positions: $S = x_\mathcal{F} + (M * p_\text{head})$. Crucially, this is injected *after* the encoder (late integration), leaving the features of the frozen encoder undisturbed by head information.

2. **Lightweight Transformer Decoder**: Comprising only 3 standard transformer encoder layers and 2D sinusoidal position encoding, it leverages the global information propagation ability of self-attention. This allows gaze targets far from the head to be captured, whereas CNN decoders typically fail due to receptive field limitations.

3. **No Head Branch**: Experiments demonstrate that when a transformer decoder is used, an extra cropped head branch offers almost no performance improvement (AUC 0.954 vs 0.953). This is because DINOv2's global features already encode head orientation, which can be automatically extracted by the global attention of the transformer.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_\text{hm} + \lambda \mathcal{L}_\text{in/out}$$

Where $\mathcal{L}_\text{hm}$ is the pixel-level BCE loss with a Gaussian heatmap ($\sigma=3$) as the ground truth, and $\mathcal{L}_\text{in/out}$ is the BCE classification loss. The DINOv2 backbone is completely frozen, and only the decoder with 2.8M parameters is trained. SOTA performance is reached in under 1.5 GPU hours of training.

## Key Experimental Results

| Method | Trainable Params | Input | GazeFollow AUC↑ | Avg L2↓ | Min L2↓ |
|------|-----------|------|-----------------|---------|---------|
| Chong et al. | ~61M | I | 0.921 | 0.137 | 0.077 |
| Gupta et al. | 35M | I+D+P | 0.943 | 0.114 | 0.056 |
| Tafasca et al. | 105M | I | 0.944 | 0.113 | 0.057 |
| **Gaze-LLE (ViT-B)** | **2.8M** | **I** | **0.956** | **0.104** | **0.045** |
| **Gaze-LLE (ViT-L)** | **2.9M** | **I** | **0.958** | **0.099** | **0.041** |

### Ablation Study

| Head Integration Location | Decoder | Branch | AUC | Avg L2 |
|-------------|--------|------|-----|--------|
| Before encoder (early) | CNN | H+S | 0.854 | 0.254 |
| Before encoder (early) | Transformer | H+S | 0.904 | 0.178 |
| After encoder (late) | CNN | H+S | 0.932 | 0.155 |
| After encoder (late) | Transformer | H+S | 0.954 | 0.113 |
| **After encoder (late)** | **Transformer** | **S Only** | **0.953** | **0.114** |

### Key Findings

- Directly replacing backbones with DINOv2 in existing architectures degraded performance (AUC 0.921 $\rightarrow$ 0.908); it must be paired with the new decoder design.
- Late head integration improves AUC by ~0.05 compared to early integration—a key decision for frozen encoders.
- Transformer decoder vs. CNN decoder: AUC 0.953 vs. 0.916 (the performance gap is even larger without a head branch).
- Demonstrates strong zero-shot generalization across datasets, performing well on ChildPlay and GOO-Real without fine-tuning.

## Highlights & Insights

- **Extreme Simplification**: Simplifies complex multi-branch architectures into a single encoder plus a lightweight decoder, reducing the parameter count by 10 to 40 times.
- **Astonishing Training Efficiency**: Achieves SOTA in under 1.5 GPU hours, whereas previous methods required dozens of hours.
- **Implication of Head Prompting**: When freezing foundation models, task-conditioning information should be injected *after* feature extraction.
- **First demonstration** that gaze estimation can dispense with auxiliary signals like depth or pose, as DINOv2 implicitly encodes this information.

## Limitations & Future Work

- Reliability depends on the quality of pre-trained DINOv2; larger and stronger foundation models are expected to yield further improvements.
- Currently handles only single-person gaze; multi-person scenarios require multiple forward decoding passes.
- Video scenes are processed frame-by-frame without utilizing temporal information.
- The head bounding box is provided by an external detector (not end-to-end).
- Robustness to extreme occlusions or low-resolution heads has not been fully explored.

### VideoAttentionTarget Results

| Method | AUC↑ | L2↓ | AP in/out↑ |
|------|------|-----|--------|
| Chong et al. | 0.860 | 0.134 | 0.853 |
| Miao et al. | 0.917 | 0.109 | 0.908 |
| **Gaze-LLE (ViT-L)** | **0.937** | **0.103** | **0.903** |

## Related Work

- **Multi-branch gaze estimation**: Recasens, Chong, Fang, Gupta, Tafasca, etc.—scene + head + auxiliary models.
- **Vision foundation models**: DINOv2, CLIP—large-scale self-supervised pre-training.
- **Dense prediction with foundation models**: Depth Anything and others demonstrate that frozen models can perform dense prediction.
- **Social gaze analysis**: Shared attention, mutual gaze, etc.—which can benefit from shared scene representations.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to demonstrate that a frozen foundation model + lightweight decoder can achieve gaze estimation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely detailed analysis of the design space across 4 benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ Exceptionally clear analysis and presentation of design decisions.
- Value: ⭐⭐⭐⭐⭐ Identifies the correct paradigm for gaze estimation in the era of foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learned Image Compression with Dictionary-based Entropy Model](learned_image_compression_with_dictionary-based_entropy_model.md)
- [\[ICLR 2026\] S2R-HDR: A Large-Scale Rendered Dataset for HDR Fusion](../../ICLR2026/model_compression/s2r-hdr_a_large-scale_rendered_dataset_for_hdr_fusion.md)
- [\[CVPR 2025\] LALIC: Linear Attention Modeling for Learned Image Compression](linear_attention_modeling_for_learned_image_compression.md)
- [\[CVPR 2025\] QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge](quartdepth_post-training_quantization_for_real-time_depth_estimation_on_the_edge.md)
- [\[NeurIPS 2025\] C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models](../../NeurIPS2025/model_compression/c-lora_contextual_low-rank_adaptation_for_uncertainty_estimation_in_large_langua.md)

</div>

<!-- RELATED:END -->
