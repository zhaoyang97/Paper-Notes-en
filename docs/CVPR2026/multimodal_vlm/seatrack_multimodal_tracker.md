---
title: >-
  [Paper Note] SEATrack: Simple, Efficient, and Adaptive Multimodal Tracker
description: >-
  [CVPR 2026][Multimodal VLM][Multimodal tracking] This paper proposes SEATrack, a multimodal tracker that achieves dynamic cross-modal attention map alignment via AMG-LoRA and efficient global relation modeling via HMoE, attaining a state-of-the-art performance–efficiency trade-off on RGB-T/D/E tracking with minimal trainable parameters.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Multimodal tracking
  - parameter-efficient fine-tuning
  - attention alignment
  - mixture of experts
  - LoRA
date: 2026-05-08
content_hash: 47a00c307eb261d2
---

# SEATrack: Simple, Efficient, and Adaptive Multimodal Tracker

**Conference**: CVPR 2026
**arXiv**: [2604.12502](https://arxiv.org/abs/2604.12502)
**Code**: Available
**Area**: Object Tracking / Multimodal
**Keywords**: Multimodal tracking, parameter-efficient fine-tuning, attention alignment, mixture of experts, LoRA

## TL;DR

This paper proposes SEATrack, a multimodal tracker that achieves dynamic cross-modal attention map alignment via AMG-LoRA and efficient global relation modeling via HMoE, attaining a state-of-the-art performance–efficiency trade-off on RGB-T/D/E tracking with minimal trainable parameters.

## Background & Motivation

**State of the Field**: Multimodal tracking fuses RGB with complementary modalities such as thermal infrared, depth, and event data to enable all-weather robust tracking. The PEFT paradigm has gradually replaced full fine-tuning to mitigate catastrophic forgetting.

**Limitations of Prior Work**: The number of trainable parameters in PEFT-based methods has inflated 16-fold from early approaches to the latest SOTA, fundamentally undermining the efficiency rationale of PEFT. Meanwhile, the domain gap in dual-stream architectures causes conflicting attention maps across modalities, hampering joint representation learning.

**Root Cause**: A performance–efficiency dilemma — more parameters yield better performance but erode the core value of PEFT.

**Paper Goals**: (1) Break the performance–efficiency trade-off via cross-modal attention alignment; (2) Design an efficient global relation modeling module to replace attention-based fusion.

**Starting Point**: Multimodal inputs are spatiotemporally aligned, so the attention maps for intra-modal target matching should in principle be consistent — this consistency can be exploited for mutual cross-modal guidance.

**Core Idea**: AMG-LoRA uses the matching information from one modality to guide the matching process of the other, enabling bidirectional dynamic alignment.

## Method

### Overall Architecture

A dual-stream ViT architecture freezes the backbone of a pretrained RGB tracker. AMG-LoRA (attention alignment) and HMoE (cross-modal fusion) are inserted every 2 layers. Search-region features from both modalities are aggregated via element-wise addition and fed into the prediction head for target localization.

### Key Designs

1. **AMG-LoRA (Adaptive Mutual-Guidance Low-Rank Adaptation)**:

   - **Function**: Simultaneously performs domain adaptation and dynamic cross-modal attention map alignment.
   - **Mechanism**: (i) LoRA adapts the K/V projection matrices of the attention layers for domain adaptation; (ii) Inspired by Classifier-Free Guidance, cross-modal alignment is reformulated as a multi-branch trade-off. The alignment formula is: $\textbf{attn}_{rgb} = \tilde{\textbf{attn}}_{rgb} + w_X(\tilde{\textbf{attn}}_X - \tilde{\textbf{attn}}_{rgb})$, where $w_X$ is a learnable scaling factor.
   - **Design Motivation**: Target saliency varies across modalities with changing scenes, requiring dynamic rather than static alignment to prevent negative transfer from unreliable modalities. Only 0.14M parameters yield PR improvements of 18.3%/7.2%/6.1%.

2. **HMoE (Hierarchical Mixture of Experts)**:

   - **Function**: Efficient global relation modeling that replaces the quadratic complexity of attention.
   - **Mechanism**: Unlike existing MoE methods that aggregate only at the expert level, HMoE enables fine-grained interactions from sub-token to token level. Low-rank linear layers serve as expert functions, with hierarchical soft routing implemented via learnable gating matrices.
   - **Design Motivation**: Attention-based fusion is expressive but incurs quadratic complexity; local fusion is efficient but lacks a global receptive field. HMoE is approximately 35% faster than its attention-based counterpart while maintaining comparable performance.

3. **Dual-Stream Design with Shared LoRA**:

   - **Function**: Establishes joint representation learning across the two streams.
   - **Mechanism**: The RGB and X modality streams share the same LoRA bypass, promoting cross-modal feature alignment. At inference time, the LoRA matrices can be merged into the original weights, incurring no additional latency.
   - **Design Motivation**: Shared parameters reduce parameter count while fostering cross-modal consistency in domain adaptation.

### Loss & Training

Standard tracking losses (classification + regression) are employed. The AMG scaling factor is initialized to 1 and adapts automatically to scenes during training.

## Key Experimental Results

### Main Results

| Method | Trainable Params | LasHeR PR↑ | DepthTrack PR↑ | VisEvent PR↑ |
|--------|-----------------|-----------|---------------|-------------|
| ProTrack | 0.3M | 52.1 | 58.3 | 65.2 |
| Un-Track | 4.8M | 65.4 | 63.8 | 69.1 |
| SDSTrack | 2.1M | 68.2 | 65.5 | 71.3 |
| **SEATrack** | **0.8M** | **70.4** | **65.5** | **71.3** |

### Ablation Study

| Configuration | LasHeR PR | Params | Notes |
|--------------|----------|--------|-------|
| Baseline (frozen ViT) | 52.1 | 0M | No adaptation |
| + LoRA | 60.8 | 0.12M | Domain adaptation only |
| + AMG-LoRA | 70.4 | 0.14M | Domain adaptation + alignment |
| + HMoE | 70.4 | 0.8M | Full model |
| Attention replacing HMoE | 70.2 | 1.6M | 35% slower |

### Key Findings

- AMG-LoRA adds only 0.02M parameters (from 0.12M to 0.14M) yet yields nearly 10% PR improvement.
- HMoE achieves performance comparable to attention-based fusion while being 35% faster.
- CFG-inspired dynamic alignment outperforms static alignment (fixed $w=1$) by 3–5 percentage points.

## Highlights & Insights

- Borrowing Classifier-Free Guidance for cross-modal alignment in tracking is an elegant analogy: modality reliability is treated as the "conditioned" vs. "unconditional" branch.
- The insight that "cross-modal attention alignment is the key to breaking the performance–efficiency dilemma" is generalizable to other multimodal tasks.

## Limitations & Future Work

- Validation is limited to the tracking task; effectiveness on detection or segmentation remains untested.
- The number of experts and heads in HMoE requires manual tuning.
- Scenarios involving more than two modalities are not considered.
- AMG could be extended to broader types of attention alignment.

## Related Work & Insights

- **vs. SDSTrack**: SDSTrack reuses frozen attention layers for global interaction but at high complexity; SEATrack replaces this with HMoE.
- **vs. ProTrack**: ProTrack pioneered the prompt-tuning paradigm but has limited expressiveness; SEATrack's AMG-LoRA is more effective.

## Rating

- Novelty: ⭐⭐⭐⭐ Both AMG-LoRA and HMoE exhibit original designs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across RGB-T/D/E tasks.
- Writing Quality: ⭐⭐⭐⭐ Motivation and design logic are clearly presented.
- Value: ⭐⭐⭐⭐ Offers meaningful reference for multimodal PEFT research.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition](adaptvision_efficient_vision-language_models_via_adaptive_visual_acquisition.md)
- [\[ICCV 2025\] LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models](../../ICCV2025/multimodal_vlm/llava-prumerge_adaptive_token_reduction_for_efficient_large_multimodal_models.md)
- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](quant_experts_token-aware_adaptive_error_reconstruction_with_mixture_of_experts_.md)
- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)
- [\[CVPR 2026\] AVR: Adaptive VLM Routing for Computer Use Agents](adaptive_vision-language_model_routing_for_computer_use_agents.md)

<!-- RELATED:END -->
