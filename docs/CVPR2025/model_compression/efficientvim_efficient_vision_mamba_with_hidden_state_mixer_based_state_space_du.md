---
title: >-
  [Paper Note] EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality
description: >-
  [CVPR 2025][Model Compression][Vision Mamba] EfficientViM is proposed, which shifts the channel mixing operations in the SSD layer from the token space ($O(LD^2)$) to the compressed hidden state space ($O(ND^2)$, $N \ll L$). This achieves a 2x to 4x faster inference speed compared to existing Vision Mamba models while maintaining competitive accuracy (77.9% with 11,952 img/s for the M3 model on ImageNet-1K).
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Vision Mamba"
  - "SSM"
  - "Lightweight Models"
  - "Hidden State Mixer"
  - "Efficient Inference"
date: 2026-05-08
content_hash: d116fa0d7777cf0a
---

# EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality

**Conference**: CVPR 2025  
**arXiv**: [2411.15241](https://arxiv.org/abs/2411.15241)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Vision Mamba, SSM, Lightweight Models, Hidden State Mixer, Efficient Inference

## TL;DR
EfficientViM is proposed, which shifts the channel mixing operations in the SSD layer from the token space ($O(LD^2)$) to the compressed hidden state space ($O(ND^2)$, $N \ll L$). This achieves a 2x to 4x faster inference speed compared to existing Vision Mamba models while maintaining competitive accuracy (77.9% with 11,952 img/s for the M3 model on ImageNet-1K).

## Background & Motivation

**Background**: State Space Models (SSMs/Mamba) replace the quadratic complexity of attention with linear sequence length complexity. Vision Mamba (ViM, VSSD, etc.) applies Mamba to vision tasks, but the actual inference speed remains slower than lightweight CNNs/Transformers.

**Limitations of Prior Work**: The computational bottleneck of standard SSD layers does not lie in the sequence scanning itself ($O(LND)$), but in the linear projection operations on full-length sequences. The complexity of operations such as generating $\mathbf{x}$, gating $\mathbf{z}$, and output projection is $O(LD^2)$. Although compressed hidden states ($N \ll L$) are used internally in SSD, channel mixing and gating are still performed in the full token space.

**Key Challenge**: The core advantage of SSMs lies in handling long sequences ($L$-dimensional) by compressing hidden states ($N$-dimensional). However, the most time-consuming operations in existing implementations do not leverage this compression.

**Goal**: How to make Vision Mamba truly attain its theoretically expected efficiency in practice.

**Key Insight**: Shifting linear projection and gating operations from the token space to the hidden state space, changing the dominant term from $LD^2$ to $ND^2$.

**Core Idea**: Shift the most time-consuming channel mixing operations in SSD from the $L$-dimensional token space to the $N$-dimensional hidden state space ($N \ll L$), thereby significantly accelerating Vision Mamba inference.

## Method

### Overall Architecture
EfficientViM is a three-stage hierarchical architecture, where the stem consists of 4 layers of stride-2 convolutions, and each stage is composed of HSM-SSD + FFN blocks, with downsampling between stages. During inference, it uses the compressed hidden state space for channel mixing, combined with Multi-Stage Hidden State Fusion (MSF) to enhance representation.

### Key Designs

1. **Hidden State Mixer SSD (HSM-SSD)**:

    - Function: Shift channel mixing operations from the token space to the hidden state space.
    - Mechanism: The output in standard SSD, $\mathbf{x}_{out} = \text{Linear}(\mathbf{y} \odot \sigma(\mathbf{z}))$, operates in the $L \times D$ space. HSM-SSD first projects the input into the hidden state $\mathbf{h}_{in} \in \mathbb{R}^{N \times D}$, performs gating and projection in this space, and then projects it back to the token space via the C matrix. The dominant complexity is reduced from $O(LD^2)$ to $O(ND^2)$.
    - Design Motivation: The hidden state is the core compressed representation of SSM ($N$ is typically much smaller than $L$). Performing channel mixing in this space keeps information loss controllable while significantly boosting speed.

2. **Multi-Stage Hidden State Fusion (MSF)**:

    - Function: Utilize the hidden states of each stage to enrich the final classification features.
    - Mechanism: Average the hidden states of each stage $\hat{h}^{(s)} = \frac{1}{N}\sum_i h_i^{(s)}$, project them to classification logits $z^{(s)}$, and fuses the logits of each stage weighted by learned softmax weights $\hat{\beta}^{(s)}$. No extra inference overhead is introduced (as hidden states are byproducts of SSM).
    - Design Motivation: Hidden states contain global information but are typically discarded. MSF allows hidden states from both lower and higher stages to participate in classification, enhancing representation hierarchy.

3. **Single-Head Design with State Importance Weights**:

    - Function: Avoid memory-bound reshape operations in multi-head attention.
    - Mechanism: Use a single head but introduce state-level importance weights $\mathbf{A} \in \mathbb{R}^{L \times N}$, allowing each hidden state dimension to have different importance for different tokens. This provides a multi-head-like capability to capture diversity while avoiding memory layout overhead.
    - Design Motivation: Multi-head implementations require frequent tensor reshaping, which becomes an inference bottleneck in lightweight models.

### Loss & Training
Standard ImageNet training recipe. 300/450 epoch training. The architecture defines 4 variants (M1-M4) covering different speed-accuracy trade-offs with parameters ranging from 6.7M to 19.6M.

## Key Experimental Results

### Main Results (ImageNet-1K)

| Model | Throughput (img/s) | Top-1 (%) | Params | FLOPs |
|------|-------------------|-----------|--------|-------|
| EfficientViM-M1 | 20,731 | 72.9% | 6.7M | 239M |
| EfficientViM-M2 | 17,005 | 75.8% | 13.9M | 355M |
| EfficientViM-M3 | 11,952 | 77.9% | 16.6M | 656M |
| EfficientViM-M4 | 8,170 | 79.6% | 19.6M | 1111M |

Comparison vs Vision Mamba: EfficientViM-M3 is 0.7% higher than ViM-Ti (77.2%) and 2.7x faster, and 3.8% higher than VSSD-Nano (74.1%) and 4.0x faster.

### Ablation Study

| Configuration | Top-1 | Throughput | Description |
|------|-------|-----------|------|
| Standard SSD (baseline) | ~75.4% | ~10k | NC-SSD baseline |
| +HSM (Hidden State Mixer) | ~75.2% | ~17k | Slight accuracy drop but speed +70% |
| +MSF (Multi-Stage Fusion) | 75.8% | 17,005 | Recovers accuracy with no speed loss |
| Multi-head $\rightarrow$ Single-head + weights | +0.3% | +15% | Single-head is faster and more accurate |

### Key Findings
- HSM is the core of the speed improvement (70% speedup) with negligible accuracy loss (<0.2%), which MSF completely recovers.
- EfficientViM achieves the best speed-accuracy trade-off across all model scales.
- It is 90% faster than MobileNetV3-L 0.75 with comparable accuracy, and 4x faster than MobileViTV2 0.75 with 0.2% higher accuracy.
- A hidden state dimension of $N=16\text{-}64$ is already sufficient to capture global information; further increasing $N$ yields diminishing returns.

## Highlights & Insights
- **Accurately Locating the Bottleneck**: Finding that the actual bottleneck of SSMs is not the sequence scan but the channel projection; this insight is the foundation of the method's effectiveness.
- **Dual Utilization of Hidden States**: HSM utilizes hidden states for efficient channel mixing, while MSF uses hidden states as auxiliary classification features—fully exploiting the value of SSM "byproducts".
- **Practicality of the Design Philosophy**: Instead of chasing theoretical novelty, it achieves true applicability of Mamba on mobile devices via detailed engineering optimizations (single-head, hidden state space operations).

## Limitations & Future Work
- The approximation of HSM might suffer from larger accuracy drops in tasks requiring fine token-level information (e.g., dense prediction).
- Validated only on classification tasks; downstream tasks like detection/segmentation are not reported.
- The choice of hidden state dimension $N$ still requires manual tuning.

## Related Work & Insights
- **vs ViM/PlainMamba**: These methods directly apply Mamba to vision but do not optimize practical inference efficiency. EfficientViM achieves substantial speedup by redesigning the computational pipeline.
- **vs SHViT**: SHViT uses single-head attention for efficiency; EfficientViM offers higher accuracy at comparable speeds (M1: 72.9% vs 72.8%).
- **vs EfficientNet/MobileNet**: Traditional lightweight CNNs still have a speed advantage on small-scale models, but EfficientViM starts to outperform them at medium scales.

## Rating
- Novelty: ⭐⭐⭐⭐ The HSM approach is simple and effective, and the idea of "shifting the operation space" is inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison on ImageNet, detailed ablation, but lacks downstream tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear bottleneck analysis, easy-to-follow method derivation.
- Value: ⭐⭐⭐⭐ Makes Vision Mamba truly applicable in lightweight deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Mamba-Adaptor: State Space Model Adaptor for Visual Recognition](mamba-adaptor_state_space_model_adaptor_for_visual_recognition.md)
- [\[ACL 2025\] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](../../ACL2025/model_compression/state_offset_tuning_ssm_peft.md)
- [\[ICLR 2026\] SSDi8: Accurate and Efficient 8-bit Quantization for State Space Duality](../../ICLR2026/model_compression/ssdi8_accurate_and_efficient_8-bit_quantization_for_state_space_duality.md)
- [\[CVPR 2025\] MambaIC: State Space Models for High-Performance Learned Image Compression](mambaic_state_space_models_for_high-performance_learned_image_compression.md)
- [\[CVPR 2025\] MobileMamba: Lightweight Multi-Receptive Visual Mamba Network](mobilemamba_lightweight_multi-receptive_visual_mamba_network.md)

</div>

<!-- RELATED:END -->
