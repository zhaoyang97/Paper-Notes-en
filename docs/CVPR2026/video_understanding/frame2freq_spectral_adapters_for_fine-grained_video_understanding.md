---
title: >-
  [Paper Note] Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding
description: >-
  [CVPR2026][Video Understanding][Frequency-Domain Adapters] This paper proposes Frame2Freq—the first family of PEFT adapters that performs temporal modeling in the frequency domain. By transforming frozen VFM frame embedd…
tags:
  - "CVPR2026"
  - "Video Understanding"
  - "Frequency-Domain Adapters"
  - "Parameter-Efficient Fine-Tuning"
  - "Image-to-Video Transfer"
  - "Fine-Grained Action Recognition"
  - "Fast Fourier Transform"
  - "Vision Foundation Model"
date: 2026-05-08
content_hash: 3bce8c5d73194e66
---

# Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding

**Conference**: CVPR2026  
**arXiv**: [2602.18977](https://arxiv.org/abs/2602.18977)  
**Code**: [th-nesh/Frame2Freq](https://github.com/th-nesh/Frame2Freq)  
**Area**: Video Understanding  
**Keywords**: Frequency-Domain Adapters, Parameter-Efficient Fine-Tuning, Image-to-Video Transfer, Fine-Grained Action Recognition, Fast Fourier Transform, Vision Foundation Model

## TL;DR

This paper proposes Frame2Freq—the first family of PEFT adapters that performs temporal modeling in the frequency domain. By transforming frozen VFM frame embeddings into the spectral space via FFT and learning frequency band-level filtering, Frame2Freq surpasses fully fine-tuned models on five fine-grained action recognition benchmarks with fewer than 10% trainable parameters.

## Background & Motivation

1. **Limitations of Prior Work in transferring image-pretrained backbones to video**: Existing temporal adapters (convolution/attention-based) capture only static image cues and extremely rapid flickering changes, neglecting mid-frequency motion signals that carry the key discriminative information for fine-grained actions (e.g., "opening a bottle" vs. "closing a bottle").
2. **Spectral discriminability analysis reveals systematic bias**: Inspired by ANOVA, the authors design a Frequency Discriminability Analysis that quantitatively demonstrates how conventional adapters such as ST-Adapter concentrate discriminative energy at the low- and high-frequency extremes, severely underutilizing the mid-frequency range.
3. **Fine-grained actions exhibit naturally salient spectral signatures**: Applying 3D FFT to videos from Diving48 reveals that different somersault counts and body postures produce markedly distinct spectral patterns (more somersaults → higher high-frequency energy; tuck vs. pike → different directional components), which are difficult to observe in RGB space.
4. **Urgent need to distinguish symmetric action pairs**: Datasets such as Drive&Act and IKEA-ASM contain a large proportion of near-symmetric action pairs (e.g., "pick up" vs. "put down") that cannot be distinguished by spatial appearance alone, requiring precise capture of motion phase differences.
5. **Prohibitive cost of full fine-tuning**: VFMs contain hundreds of millions of parameters, making full fine-tuning impractical. Existing PEFT methods (AIM, DualPath, ST-Adapter) all operate in the time domain and do not exploit frequency-domain structure.
6. **Generalization challenges on domain-specific small datasets**: Driving surveillance, furniture assembly, and human-robot interaction datasets contain only a few thousand samples, demanding adapters that achieve strong generalization with a small parameter budget.

## Method

### Overall Architecture

Lightweight Frame2Freq adapters are inserted after each Transformer block of a frozen ViT backbone (CLIP/DINOv2). Given patch embeddings $X \in \mathbb{R}^{T \times N \times D}$ extracted from $T$ input frames, each adapter follows a bottleneck structure $\text{FC}_{down} \to \text{spectral/temporal branch} \to \text{FC}_{up}$ to produce temporally enhanced features, which are added back to the backbone output via a residual connection. Frame-wise CLS tokens are aggregated and passed to a linear classification head.

### Frame2Freq-ST (Short-Time Spectral Adapter)

- A **STFT** (Hann window) is applied to the projected embeddings along the temporal axis, yielding a joint time-frequency representation $\tilde{X} \in \mathbb{C}^{B \times N \times F \times T' \times C_a}$.
- Two depthwise separable 3D convolutions—$\text{Conv}_{temp}$ along the temporal axis and $\text{Conv}_{freq}$ along the frequency axis—refine the representation to capture short-term transitions and adjacent frequency-band relationships.
- iSTFT transforms the output back to the time domain, followed by $\text{FC}_{up}$ to restore dimensionality. Total trainable parameters: **3.5M**.
- Best suited for domain-specific datasets with homogeneous action scales (Drive&Act, IKEA-ASM).

### Frame2Freq-MS (Multi-Scale Spectral Adapter)

- After projection, the channel dimension is **split in half** into a spectral branch $X_{freq}$ and a temporal branch $X_{temp}$.
- Spectral branch: FFT is applied to $X_{freq}$ at $K$ different window sizes $\{w_k\} = [T, T/2, T/4]$; features at each scale are refined by a shared depthwise convolution $\text{Conv}_{freq}$, averaged across scales, and transformed back via iFFT.
- Temporal branch: $X_{temp}$ is processed by a $(3\times1\times1)$ convolution $\text{Conv}_{temp}$ to capture short-range temporal continuity.
- The two branches are concatenated and projected back via $\text{FC}_{up}$. Total trainable parameters: **7.3M**.
- Best suited for complex datasets with large variation in motion frequency (Diving48, SSv2).

### Loss & Training

Standard cross-entropy classification loss is used without any auxiliary loss. Models are trained for 60 epochs with uniform sampling of 16 or 32 frames.

## Key Experimental Results

### Main Results

| Dataset | Method | Backbone | Trainable Params | Top-1 Acc |
|---------|--------|----------|-----------------|-----------|
| Diving48 | ST-Adapter | ViT-B/16 CLIP | 7M | 90.4% |
| Diving48 | **Frame2Freq-MS** | ViT-B/16 CLIP | 7.3M | **92.2%** (+1.8) |
| Diving48 | ORViT (full FT) | ViT-B/16 | 160M | 88.0% |
| SSv2 | ST-Adapter | ViT-B/16 CLIP | 14M | 69.5% |
| SSv2 | **Frame2Freq-MS** | ViT-B/16 CLIP | 14M | **70.4%** (+0.9) |
| SSv2 | **Frame2Freq-MS** | ViT-L/14 CLIP | 19M | **72.1%** |
| Drive&Act | ST-Adapter | DINOv2 | 7.1M | 75.2% |
| Drive&Act | **Frame2Freq-ST** | DINOv2 | 3.5M | **82.0%** (+6.8) |
| IKEA-ASM | ST-Adapter | DINOv2 | 7.1M | 70.5% |
| IKEA-ASM | **Frame2Freq-ST** | DINOv2 | 3.5M | **78.1%** (+7.6) |
| HRI-30 | ST-Adapter | DINOv2 | 7.1M | 85.5% |
| HRI-30 | **Frame2Freq-MS** | DINOv2 | 7.3M | **89.8%** (+4.3) |

The advantage is particularly pronounced on symmetric action pairs: +10.5% on the Drive&Act symmetric subset (66.4→77.1) and +11.8% on the IKEA-ASM symmetric subset (68.5→80.3).

### Ablation Study

| Ablation | Setting | SSv2 | Diving48 |
|----------|---------|------|----------|
| Frequency branch only | — | 67.5 | 90.9 |
| Temporal branch only | — | 69.1 | 90.4 |
| **Freq + Temporal (Frame2Freq)** | — | **69.7** | **92.2** |
| Multi-scale window $[T]$ | Single scale | 69.0 | 91.5 |
| Multi-scale window $[T,T/2,T/4]$ | Three scales | **69.7** | **92.2** |
| Multi-scale window $[T,T/2,T/4,T/8]$ | Four scales | 69.4 | 91.0 |
| Adapters in layers 1–4 only | Shallow | 55.8 | 67.6 |
| Adapters in all layers 1–12 | Full depth | **69.7** | **92.2** |

- The combination of spectral and temporal branches yields the best complementarity; three-scale windows are optimal, and adding a finer scale ($T/8$) leads to saturation and degradation.
- Simple mean/concat fusion outperforms gated and learnable fusion strategies, suggesting the two branches are naturally complementary.

## Highlights & Insights

- **First frequency-domain PEFT adapter**: This is the first work to apply FFT/STFT for image-to-video temporal adaptation of frozen VFMs, opening a new research direction.
- **Rigorous theoretical motivation**: The ANOVA-inspired Frequency Discriminability Analysis quantitatively exposes the spectral bias of existing adapters, providing a principled basis for the proposed design.
- **Two flexible variants**: Frame2Freq-ST (3.5M parameters) suits single-scale domain-specific data, while Frame2Freq-MS (7.3M) handles complex multi-scale scenarios, offering practitioners a practical choice.
- **Exceptional parameter efficiency**: Frame2Freq surpasses fully fine-tuned models on 4 out of 5 datasets with fewer than 10% of their trainable parameters.
- **Breakthrough on symmetric action recognition**: Gains exceeding +10% on the most challenging symmetric action pairs.

## Limitations & Future Work

- The gain on SSv2 is marginal (+0.9%), suggesting limited advantage of frequency-domain modeling for coarse-grained label settings.
- Frame2Freq-ST achieves only 75.1% on Diving48 (single-scale modeling struggles with compound multi-component motions), requiring prior knowledge to select between the two variants.
- Only standard cross-entropy loss is employed; frequency-domain contrastive losses or frequency band-level supervision signals remain unexplored.
- Experiments are limited to ViT-B/16 and ViT-L/14 backbones and have not been extended to larger models (e.g., ViT-G).
- The STFT window size and multi-scale configuration $[T, T/2, T/4]$ are manually specified rather than learned adaptively.
- Richer time-frequency analysis tools such as wavelet transforms and multi-resolution filters have not been explored (the authors also note this as a future direction in their conclusion).

## Related Work & Insights

- **vs. ST-Adapter**: Frame2Freq is directly built on the ST-Adapter framework, replacing or augmenting the temporal depthwise convolution with an FFT branch, yielding consistent improvements across all benchmarks (+0.9%–+7.6%).
- **vs. AIM / DualPath**: Both are PEFT methods that operate solely in the time domain; they trail Frame2Freq-MS by approximately 3.5% on Diving48.
- **vs. DTF-Transformer**: DTF also employs 1D FFT filters for video temporal modeling but requires full fine-tuning (88M parameters); Frame2Freq achieves comparable or superior performance with only 7.3M parameters.
- **vs. VFPT**: The only prior PEFT method utilizing the frequency domain, but restricted to spatial adaptation; Frame2Freq is the first to extend frequency-domain processing to the temporal dimension.
- **vs. full fine-tuning (ORViT, Uniformerv2)**: Frame2Freq-MS exceeds ORViT by 4.2% on Diving48 and matches Uniformerv2 on SSv2 with less than one-tenth of the parameters.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Frequency-domain PEFT adapters constitute a genuinely new paradigm, supported by rigorous spectral discriminability analysis
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Five datasets, two backbones, few-shot evaluation, and comprehensive multi-dimensional ablations
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear and analysis is thorough, though notation is occasionally redundant
- Value: ⭐⭐⭐⭐⭐ — Opens a new frequency-domain pathway for VFM video adaptation with immediate practical value for fine-grained action recognition

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Text-guided Fine-Grained Video Anomaly Understanding](text-guided_fine-grained_video_anomaly_understanding.md)
- [\[CVPR 2026\] Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos](mistake_attribution_fine-grained_mistake_understanding_in_egocentric_videos.md)
- [\[CVPR 2026\] UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models](ufvideo_towards_unified_fine-grained_video_cooperative_understanding_with_large_.md)
- [\[AAAI 2026\] FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding](../../AAAI2026/video_understanding/finevau_a_novel_human-aligned_benchmark_for_fine-grained_video_anomaly_understan.md)
- [\[ICLR 2026\] Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding](../../ICLR2026/video_understanding/lets_split_up_zero-shot_classifier_edits_for_fine-grained_video_understanding.md)

</div>

<!-- RELATED:END -->
