---
title: >-
  [Paper Note] OpenFS: Multi-Hand-Capable Fingerspelling Recognition with Implicit Signing-Hand Detection and Frame-Wise Letter-Conditioned Synthesis
description: >-
  [CVPR2026][Human Understanding][Fingerspelling Recognition] This paper proposes OpenFS, a framework that achieves multi-hand fingerspelling recognition with implicit signing-hand detection via dual-level positional encoding, a signing-hand focusing loss, and a monotonic alignment loss. A frame-wise letter-conditioned diffusion generator is further designed to synthesize OOV training data. OpenFS achieves state-of-the-art performance on three benchmarks (ChicagoFSWild / ChicagoFSWildPlus / FSNeo) with inference speed over 100× faster than PoseNet.
tags:
  - CVPR2026
  - Human Understanding
  - Fingerspelling Recognition
  - Sign Language Understanding
  - Implicit Signing-Hand Detection
  - Monotonic Alignment Loss
  - Diffusion-based Generation
  - OOV Generalization
date: 2026-05-08
content_hash: 390b2d97e2ba2db5
---

# OpenFS: Multi-Hand-Capable Fingerspelling Recognition with Implicit Signing-Hand Detection and Frame-Wise Letter-Conditioned Synthesis

**Conference**: CVPR2026
**arXiv**: [2602.22949](https://arxiv.org/abs/2602.22949)
**Code**: [JunukCha/OpenFS](https://github.com/JunukCha/OpenFS)
**Area**: Human Body Understanding
**Keywords**: Fingerspelling Recognition, Sign Language Understanding, Implicit Signing-Hand Detection, Monotonic Alignment Loss, Diffusion-based Generation, OOV Generalization

## TL;DR

This paper proposes OpenFS, a framework that achieves multi-hand fingerspelling recognition with implicit signing-hand detection via dual-level positional encoding, a signing-hand focusing loss, and a monotonic alignment loss. A frame-wise letter-conditioned diffusion generator is further designed to synthesize OOV training data. OpenFS achieves state-of-the-art performance on three benchmarks (ChicagoFSWild / ChicagoFSWildPlus / FSNeo) with inference speed over 100× faster than PoseNet.

## Background & Motivation

**Fingerspelling as an essential complement to sign language**: Sign languages cannot easily create unique signs for every proper noun or neologism; fingerspelling (letter-by-letter spelling) is therefore indispensable for expressing technical terms, names, and new words, making its accurate recognition a critical bridge between the Deaf and hearing communities.

**Signing-hand ambiguity**: Existing methods rely on optical flow or hand motion magnitude for explicit signing-hand detection, but the non-signing hand sometimes exhibits larger motion, leading to detection errors, recognition failures, and training instability.

**Peaky behavior of CTC loss**: Methods that adopt CTC loss tend to make sparse predictions concentrated on a small number of frames (peaky behavior), providing insufficient supervision to the encoder and hindering the learning of discriminative hand pose representations.

**Underestimated OOV problem**: New words and neologisms emerge continuously, making generalization to unseen vocabulary critical; yet manually collecting fingerspelling data for new words is costly and requires expert signers.

**Existing generative methods are unsuitable for fingerspelling**: Text-to-motion models based on CLIP capture word-level semantics as a global condition and cannot model the fine-grained finger-joint movements and inter-letter transitions required by fingerspelling.

**Lack of OOV evaluation benchmarks**: No standardized benchmark previously existed for evaluating OOV fingerspelling recognition, making systematic assessment of model generalization impossible.

## Method

### Overall Architecture

OpenFS consists of three core components:

- **Multi-hand fingerspelling recognizer**: A Transformer encoder–decoder architecture. The encoder receives normalized 2D single- or multi-hand pose sequences extracted by MediaPipe, embeds them via an MLP, adds dual-level positional encodings, and feeds them into Transformer encoder layers. The decoder receives letter sequences (with `<start>` and `<end>` tokens) and predicts the next letter via cross-attention.
- **Frame-wise letter-conditioned generator**: A Transformer encoder combined with a diffusion mechanism. Noisy hand poses and frame-wise letter embeddings are concatenated frame-by-frame and fed into the encoder for iterative denoising to generate realistic fingerspelling pose sequences.
- **FSNeo benchmark**: The generator is used to synthesize 1,635 unique words × 5 sequences = 8,175 samples for neologisms categorized by NEO-BENCH (lexical, morphological, and semantic neologisms).

### Key Designs

**1. Dual-Level Positional Encoding**

- **Hand identity encoding $\tau$**: All tokens belonging to the same hand share the same encoding, distinguishing different hands (left/right and different individuals).
- **Temporal positional encoding $\eta$**: Different hands in the same frame share the same temporal encoding, while different frames use different values, preserving temporal alignment and ordering.
- Both encodings follow sinusoidal formulas and are added to the pose token embeddings before being fed into the encoder.

**2. Signing-Hand Focusing Loss (SF Loss) $\mathcal{L}_{SF}$**

- Average attention maps are extracted from each decoder cross-attention layer and aggregated into a hand-level attention distribution by hand identity.
- Minimizing the entropy of this distribution encourages the decoder to focus on the dominant signing hand, achieving implicit signing-hand detection.

**3. Monotonic Alignment Loss (MA Loss) $\mathcal{L}_{MA}$**

- A cumulative cross-attention map is constructed; finite differences are computed along the letter dimension, where positive values indicate that a subsequent letter attends more strongly to earlier frames than the preceding letter (violating temporal order).
- Penalizing these positive deviations enforces monotonically increasing temporal attention, serving as a replacement for CTC loss.

**4. Coarse-to-Fine Frame-Wise Letter Annotation**

- **Coarse annotation**: The cross-attention matrix of the trained recognizer is used; frames whose attention weights exceed a threshold are assigned to the corresponding letter, and conflicting frames are labeled as blank $\phi$.
- **Fine annotation**: The recognizer is frozen and a frame-wise annotation refiner (taking encoder features as input and predicting per-frame letters) is trained, with the blank class weight set to 0.1 to suppress its dominance.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{CE} + \lambda_{SF}\mathcal{L}_{SF} + \lambda_{MA}\mathcal{L}_{MA}$$

where $\lambda_{SF} = 0.8$ and $\lambda_{MA} = 1.0$. The generator is trained with MSE loss.

## Key Experimental Results

### Main Results

Letter accuracy comparison on ChicagoFSWild (CFSW), ChicagoFSWildPlus (CFSWP), and FSNeo:

| Method | CFSW | CFSWP | FSNeo |
|--------|------|-------|-------|
| Shi et al. (2018) | 57.5 | 58.3 | - |
| Shi et al. (2019) | 61.2 | 62.3 | - |
| FSS-Net | 52.5 | 64.4 | - |
| PoseNet | 61.6 | 61.0 | 61.2 |
| **Ours** | **75.4** | **70.5** | **80.5** |
| PoseNet† | 69.2 | 69.4 | 94.9 |
| **Ours†** | **77.7** | **74.6** | **97.6** |

† denotes use of additional synthetic training data.

Inference speed comparison (868 CFSW samples, A40 GPU):

| Method | Batch Size | Latency (s) ↓ | Throughput ↑ | Letters/s ↑ | FPS ↑ |
|--------|-----------|--------------|-------------|------------|-------|
| PoseNet | 1 | 4,282 | 0.2 | 1 | 6 |
| **Ours** | 1 | 39 | 22.0 | 106 | 962 |
| **Ours** | 32 | 6 | 149.8 | 725 | 6,356 |

### Ablation Study

Ablation of positional encoding and auxiliary losses (letter accuracy on CFSW):

| Configuration | Acc. |
|---------------|------|
| Standard PE + no auxiliary loss | 73.2 |
| Standard PE + auxiliary loss | 73.1 |
| Dual-level PE + no auxiliary loss | 74.8 |
| Dual-level PE + auxiliary loss (full model) | **75.4** |

Comparison of generator conditioning strategies (letter accuracy on generated sequences):

| Conditioning Strategy | PoseNet Recognition | Ours Recognition |
|-----------------------|--------------------|--------------------|
| WC (word-level, CLIP) | 19.9 | 23.3 |
| LC (letter-level) | 26.4 | 40.2 |
| FWLC (frame-wise letter, Ours) | **63.5** | **82.3** |

### Key Findings

- The auxiliary losses (SF + MA) yield synergistic gains only when combined with dual-level positional encoding (73.2→73.1 without dual PE vs. 74.8→75.4 with dual PE).
- Implicit signing-hand detection achieves 99.9% accuracy with only one failure (which is also ambiguous to humans), far exceeding PoseNet's 90.4%.
- Synthetic data improves not only OpenFS but also PoseNet substantially (CFSW +7.6, FSNeo +33.7), validating the generality of the generator.
- Frame-wise letter conditioning (FWLC) significantly outperforms word-level (WC) and letter-level (LC) conditioning, as fingerspelling requires precise per-frame letter–pose correspondence.

## Highlights & Insights

- **Implicit signing-hand detection** replaces explicit detection; it is naturally realized through cross-attention via the SF loss with 99.9% accuracy, eliminating recognition failures caused by detection errors.
- **MA loss replaces CTC loss**, addressing peaky behavior through cross-attention regularization and yielding semantically richer encoder representations.
- **End-to-end with no post-processing**, achieving 962 FPS (single sample) to 6,356 FPS (batch), over 100× faster than PoseNet.
- **A complete recognition–generation closed loop**: the recognizer's cross-attention generates frame-wise labels → trains the generator → synthetic data augments the recognizer, forming a positive feedback cycle.
- The paper introduces FSNeo, the first OOV fingerspelling evaluation benchmark, filling a notable gap in the field.

## Limitations & Future Work

- Validation is limited to American Sign Language (ASL) fingerspelling; applicability to other sign language systems (e.g., two-handed British Sign Language fingerspelling) remains unknown.
- The approach relies on MediaPipe for hand pose extraction; failures of the pose estimator propagate to recognition (end-to-end RGB approaches may be more robust in some scenarios).
- The diffusion generator requires 50 denoising steps; generation speed is not reported and may be unsuitable for real-time data augmentation.
- FSNeo is composed entirely of synthetic data, and a distribution gap with real-world OOV fingerspelling scenarios may exist.
- Ablation experiments are conducted only on CFSW, without fully validating component contributions on CFSWP and FSNeo.

## Related Work & Insights

- **Fingerspelling recognition**: Shi et al. (2018/2019) established the ChicagoFSWild dataset series using CNN+LSTM with visual attention; PoseNet employs a Transformer encoder–decoder with re-ranking using single-hand pose input; FSS-Net focuses on fingerspelling detection for search and retrieval; HandReader is a multimodal framework fusing RGB and pose. This paper advances the field by addressing implicit signing-hand detection and replacing CTC loss.
- **Fingerspelling/motion generation**: Text-to-motion models such as MDM use global CLIP conditioning, which is unsuitable for fingerspelling that requires letter-level fine-grained control; sign language generation research focuses on full-body motion but emphasizes the semantic expressiveness of hand joints. This paper proposes a frame-wise letter-conditioned diffusion generator specifically designed for the per-frame letter–pose correspondence inherent in fingerspelling.

## Rating

- Novelty: ⭐⭐⭐⭐ (Implicit signing-hand detection + MA loss replacing CTC + frame-wise conditioned generator — three coupled innovations forming a coherent system)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three datasets, speed comparison, detailed ablations, generalization of synthetic data to other methods, and qualitative analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, rich and intuitive figures, complete motivation–method–experiment logical chain)
- Value: ⭐⭐⭐⭐ (Open-source code and data, new benchmark, deployment-friendly real-time speed, practical significance for the Deaf community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OMG-Bench: A New Challenging Benchmark for Skeleton-based Online Micro Hand Gesture Recognition](omg-bench_a_new_challenging_benchmark_for_skeleton-based_online_micro_hand_gestu.md)
- [\[CVPR 2026\] Sketch2Colab: Sketch-Conditioned Multi-Human Animation via Controllable Flow Distillation](sketch2colab_sketch-conditioned_multi-human_animation_via_controllable_flow_dist.md)
- [\[CVPR 2026\] MMGait: Towards Multi-Modal Gait Recognition](mmgait_multi_modal_gait_recognition.md)
- [\[CVPR 2026\] HandDreamer: Zero-Shot Text to 3D Hand Model Generation](handdreamer_zero_shot_text_to_3d_hand_model_generation.md)
- [\[CVPR 2026\] A2P: From 2D Alignment to 3D Plausibility for Occlusion-Robust Two-Hand Reconstruction](from_2d_alignment_to_3d_plausibility_unifying_hete.md)

</div>

<!-- RELATED:END -->
