---
title: >-
  [Paper Note] LightAVSeg: Lightweight Audio-Visual Segmentation
description: >-
  [ICML 2026][Segmentation][AVS] LightAVSeg decouples "semantic selection (what)" and "spatial localization (where)", replacing $\mathcal{O}(N^2)$ cross-modal attention with global channel modulation. This enables the AVS…
tags:
  - "ICML 2026"
  - "Segmentation"
  - "AVS"
  - "Channel Modulation"
  - "Linear Complexity"
  - "SeaFormer"
  - "Mobile CPU"
date: 2026-05-08
content_hash: fb2dc512a9cea7be
---

# LightAVSeg: Lightweight Audio-Visual Segmentation

**Conference**: ICML 2026  
**arXiv**: [2605.08805](https://arxiv.org/abs/2605.08805)  
**Code**: None  
**Area**: Audio-Visual Segmentation / Mobile Inference / Cross-Modal Interaction  
**Keywords**: AVS, Channel Modulation, Linear Complexity, SeaFormer, Mobile CPU

## TL;DR
LightAVSeg decouples "semantic selection (what)" and "spatial localization (where)", replacing $\mathcal{O}(N^2)$ cross-modal attention with global channel modulation. This enables the AVS model to achieve 50.4 mIoU (MS3) with only 20.5M parameters and 163.4 ms on Snapdragon 8 Elite, about $8\times$ faster than AVSegFormer-R50.

## Background & Motivation

**Background**: Audio-Visual Segmentation (AVS) aims to localize sounding objects in videos at the pixel level. Mainstream methods (AVSegFormer / SelM) use Transformer-based cross-modal attention for dense token fusion, achieving over 70 mIoU, but at the cost of heavy models (151M parameters) and extremely high latency (1271 ms/frame on Snapdragon 8 Elite).

**Limitations of Prior Work**: Deploying AVS on AR/VR/mobile devices is very challenging. Existing "lightweight" efforts mostly swap the backbone (e.g., ResNet-50 to MobileNetV2), but the cross-modal interaction module remains the bottleneck—its complexity is $\mathcal{O}(N^2)$ with $N \propto H \times W$, exploding with higher resolutions.

**Key Challenge**: Attention mechanisms are actually overkill for AVS—the audio essentially provides global semantic information ("who is sounding"), while visual features already encode sufficient spatial structure. Building dense pixel-to-pixel affinity matrices is a waste of computation.

**Goal**: Design a mobile-friendly AVS framework that reduces cross-modal interaction complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$, while maintaining accuracy comparable to R50-based competitors; also, avoid spurious cross-modal associations that lightweight models tend to learn.

**Key Insight**: The authors observe—"audio is for what, vision is for where". By structurally decoupling these, audio only needs to modulate relevant visual channels via channel-wise modulation, without participating in any dense spatial computation.

**Core Idea**: Use an iterative Reciprocal Audio-Visual Encoder to update the audio state at each stage via global visual descriptors, then inject it as a channel bias into visual features; maintain a recursive audio path in the Cross-Modal Fusion Decoder during upsampling; enforce pixel-level alignment during training with Multi-Scale Audio-Visual Alignment Loss, which is discarded during inference.

## Method

### Overall Architecture
Inputs are video frames $x_v \in \mathbb{R}^{T \times 3 \times H \times W}$ and raw audio waveform $x_a$. The visual stream uses SeaFormer-Large to extract a multi-scale feature pyramid $\{V_i\}_{i=1}^N$, while the audio stream applies STFT to obtain log-mel spectrograms, then encodes them with MobileNetV2 into an initial global state $A_0 \in \mathbb{R}^{T \times C_a \times 1 \times 1}$. The Reciprocal Encoder updates both the audio state $A_i$ and visual features $\widetilde{V}_i^{enc}$ at each stage. The Decoder maintains a recursive audio path $A_i^\ast \to \hat{A}_i$ during upsampling, injecting it into the visual decoding features $\widetilde{V}_i^{dec}$. Training uses $\mathcal{L} = \mathcal{L}_{\text{seg}} + \lambda \mathcal{L}_{\text{msa}}$, where $\mathcal{L}_{\text{seg}}$ is Dice + BCE, and $\lambda = 0.5$.

### Key Designs

1. **Reciprocal Audio-Visual Encoder (Semantic Selection)**:

    - **Function**: At each visual stage, uses a global visual descriptor to gate the audio state update, making the audio increasingly scene-specific with network depth, and in turn modulates the visual features.
    - **Mechanism**: Applies $1{\times}1$ max pooling to current visual features $V_i$ to obtain a global descriptor $V_i^{1\times1}$ (intentionally discarding spatial info to enforce semantic selection only); then updates $A_i = \text{Conv}_{1\times1}(A_{i-1}) \odot \sigma_h(\text{Conv}_{1\times1}(V_i^{1\times1}))$ using h-sigmoid gating; finally, injects audio as a channel bias into vision via spatial broadcasting: $\widetilde{V}_i^{enc} = V_i + \mathcal{B}(A_i)$.
    - **Design Motivation**: Traditional cross-modal attention builds an $N \times N$ affinity matrix at $\mathcal{O}(N^2)$ cost; here, only pointwise projection + broadcasting is used, strictly $\mathcal{O}(N)$. This "audio for what / vision for where" decoupling saves computation and suppresses visually irrelevant sources (background noise) early.

2. **Cross-Modal Fusion Decoder (Spatial Localization + Recursive Audio Path)**:

    - **Function**: Continuously injects audio guidance during upsampling to prevent the global semantic consistency established in the encoder from being diluted by resolution changes.
    - **Mechanism**: Maintains an audio path parallel to visual decoding; at each stage, fuses previous decoded audio $A_{i-1}$ and corresponding encoder audio $A_i$ via ReLU and $1{\times}1$ convolution to obtain $A_i^\ast$; then gates with the visual global descriptor to get $\hat{A}_i = A_i^\ast \odot \sigma_h(\text{Conv}_{1\times1}(\widetilde{V}_i^{enc1\times1}))$; finally, injects as a global channel bias into visual decoding features: $\widetilde{V}_i^{dec} = \widetilde{V}_i^{enc} + \mathcal{B}(\text{Conv}_{1\times1}(\hat{A}_i))$.
    - **Design Motivation**: Visual feature spatial scales change drastically during upsampling; if audio guidance is injected only once in the encoder, it loses effect at higher resolutions. The recursive path ensures every layer "hears" the current task-relevant audio context.

3. **Multi-Scale Audio-Visual Alignment Loss ($\mathcal{L}_{\text{msa}}$)**:

    - **Function**: Uses foreground masks to explicitly supervise the "audio-visual channel similarity map" at each scale, mitigating spurious cross-modal associations in lightweight models.
    - **Mechanism**: For each scale $i$, $\widetilde{V}_i^{dec}$ and $\hat{A}_i$ are $\ell_2$ normalized along the channel dimension, then spatial cosine similarity map $\text{sim}_i = \langle \bar{v}_i, \bar{a}_i \rangle$ is computed, sharpened with temperature $\tau=0.1$ and passed through sigmoid to get $s_i$; upsampled to ground-truth size and aligned with the foreground mask $M$ using BCE: $\mathcal{L}_{\text{msa}} = \frac{1}{S} \sum_i \text{BCE}(\hat{s}_i, M)$.
    - **Design Motivation**: BCE is chosen over KL divergence due to lower variance and better stability on bounded $[0,1]$ scores. Multi-scale supervision enforces coarse semantic alignment at shallow layers and fine boundary refinement at deeper layers, achieving "coarse-to-fine" progressive refinement. This branch can be discarded during inference for zero extra cost.

### Loss & Training
Total loss is $\mathcal{L} = \mathcal{L}_{\text{seg}} + 0.5 \mathcal{L}_{\text{msa}}$, where $\mathcal{L}_{\text{seg}} = \mathcal{L}_{\text{dice}} + \mathcal{L}_{\text{bce}}$. Inputs are $224 \times 224$, trained with AdamW (lr $10^{-4}$, batch 8) for 60 epochs; visual backbone SeaFormer-Large is pretrained, audio backbone MobileNetV2 is pretrained on AudioSet and frozen. Mobile deployment uses the TNN framework for latency measurement.

## Key Experimental Results

### Main Results

| Method | Backbone | Params | Mobile (ms) | S4 $\mathcal{M}_J$ | MS3 $\mathcal{M}_J$ |
|--------|----------|--------|-------------|--------------------|---------------------|
| AVSegFormer | R50+VGGish | 151.1M | 1271.4 | 76.5 | 49.5 |
| SelM | R50+VGGish | 117.6M | 1003.8 | 76.6 | 54.5 |
| AVSBench (Sea+MNetV2) | Sea+MNetV2 | 30.2M | 237.1 | 47.9 | 35.2 |
| AVSegFormer (Sea+MNetV2) | Sea+MNetV2 | 51.0M | 432.6 | 53.8 | 40.7 |
| **LightAVSeg (Ours)** | Sea+MNetV2 | **20.5M** | **163.4** | **75.6** | **50.4** |

### Ablation Study

| Configuration | MS3 $\mathcal{M}_J$ | Notes |
|---------------|---------------------|-------|
| ResNet-50 (R50) backbone | 52.9 | Upper bound accuracy, but 675 ms mobile latency is unacceptable |
| SeaFormer-Tiny | 30.6 | Extremely fast at 22.1 ms but poor accuracy |
| SeaFormer-Base | 44.2 | 80.2 ms / 44.2 mIoU, moderate |
| **SeaFormer-Large (selected)** | **50.4** | **163.4 ms, optimal accuracy-latency tradeoff** |
| Only $\mathcal{L}_{\text{seg}}$ | 49.3 | Baseline |
| $+\mathcal{L}_{\text{AVM}}$ | 49.2 | Adding AVSBench's KL is almost ineffective |
| $+\mathcal{L}_{\text{mix}}$ | 48.8 | Actually degrades performance |
| $+\mathcal{L}_{\text{msa}}$ (Ours) | **50.4** | BCE-based alignment is stable and effective |

### Key Findings
- On MS3 multi-source scenarios, LightAVSeg (50.4) even surpasses the heavy AVSegFormer-R50 (49.5). The authors attribute this to global channel modulation being better at suppressing multi-source noise than dense attention, consistent with the observation that lightweight models are less prone to spurious attention.
- Simply swapping to a lightweight backbone (AVSegFormer-Sea) only achieves 40.7 MS3 mIoU, indicating the interaction module is the real bottleneck—this is the core empirical takeaway.
- Multi-scale supervision via $\mathcal{L}_{\text{msa}}$ leads to "coarse-to-fine" evolution: shallow layers perform global semantic selection, deeper layers progressively refine boundaries. This structure aligns with deep supervision but with a more focused objective.

## Highlights & Insights
- Explicitly decomposing cross-modal interaction into "semantic selection (what) + spatial localization (where)" is a highly portable design principle—any task where "modality A provides global context / modality B carries spatial structure" can adopt this, e.g., RGB-D segmentation, text-guided segmentation.
- Injecting audio as a "global channel bias" into vision is both parameter-free and expressive, reaffirming the efficiency and effectiveness of channel modulation in cross-modal scenarios.
- The training-only alignment loss, which is discarded during inference, exemplifies the "do the heavy lifting during training, zero cost at inference" paradigm, transferable to any deployment scenario with strict latency budgets.

## Limitations & Future Work
- The audio stream uses MobileNetV2 to extract a global vector from spectrograms, retaining no temporal details; this may be insufficient for rapidly changing multi-source scenarios (e.g., dialogues, instrument alternation).
- $\mathcal{L}_{\text{msa}}$ assumes "global audio corresponds to the entire foreground", lacking fine-grained alignment for cases where "different sources occupy different foreground regions".
- Only tested at $224 \times 224$; no data provided for real-world deployment at higher resolutions (e.g., 1080p video frames).
- Mobile latency is measured only on Snapdragon 8 Elite, with no coverage for mid/low-end chips (e.g., MediaTek, Qualcomm 7 series).

## Related Work & Insights
- **vs AVSegFormer / SelM**: These use cross-attention for dense pixel fusion at $\mathcal{O}(N^2)$; this work uses channel modulation + broadcasting at $\mathcal{O}(N)$, even slightly outperforming the heavy R50 version on MS3.
- **vs SeaFormer / TopFormer**: These are pure-vision mobile segmentation works; this paper adapts their "squeeze-enhanced attention" to cross-modal scenarios and introduces audio-visual decoupling.
- **vs AVSBench (KL alignment)**: AVSBench uses KL divergence for modality alignment; this work demonstrates that BCE is more stable on bounded scores and directly aligns "high response regions vs foreground mask".

## Rating
- Novelty: ⭐⭐⭐⭐ "what/where decoupling + channel modulation" achieves linear complexity for AVS cross-modal interaction
- Experimental Thoroughness: ⭐⭐⭐⭐ S4/MS3/AVSS benchmarks + comprehensive backbone/loss ablation + real mobile latency
- Writing Quality: ⭐⭐⭐⭐ Clear framework diagrams and formulas, concise writing
- Value: ⭐⭐⭐⭐⭐ Directly serves real-world mobile AR/video editing, a key step for AVS on-device deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Implicit Counterfactual Learning for Audio-Visual Segmentation](../../ICCV2025/segmentation/implicit_counterfactual_learning_for_audio-visual_segmentation.md)
- [\[CVPR 2026\] SouPLe: Enhancing Audio-Visual Localization and Segmentation with Learnable Prompt Contexts](../../CVPR2026/segmentation/souple_enhancing_audio-visual_localization_and_segmentation_with_learnable_promp.md)
- [\[ICCV 2025\] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](../../ICCV2025/segmentation/towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)
- [\[ICCV 2025\] TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models](../../ICCV2025/segmentation/tavis_text-bridged_audio-visual_segmentation_with_foundation_models.md)
- [\[ICML 2026\] UGround: Towards Unified Visual Grounding with Unrolled Transformers](uground_towards_unified_visual_grounding_with_unrolled_transformers.md)

</div>

<!-- RELATED:END -->
