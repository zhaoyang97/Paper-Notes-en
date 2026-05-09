---
title: >-
  [Paper Note] DeformTrace: A Deformable State Space Model with Relay Tokens for Temporal Forgery Localization
description: >-
  [AAAI2026][Audio & Speech][Temporal Forgery Localization] This paper proposes DeformTrace, which introduces a deformable dynamic receptive field mechanism and relay token scheme into state space models, combining Transformer-level global modeling with SSM-level efficient inference to achieve state-of-the-art accuracy and substantial efficiency gains in temporal forgery localization.
tags:
  - AAAI2026
  - "Audio & Speech"
  - Temporal Forgery Localization
  - State Space Model
  - Deformable Mechanism
  - Relay Token
  - Deepfake Detection
date: 2026-05-08
content_hash: 0fd69d2f2d406875
---

# DeformTrace: A Deformable State Space Model with Relay Tokens for Temporal Forgery Localization

**Conference**: AAAI2026  
**arXiv**: [2603.04882](https://arxiv.org/abs/2603.04882)  
**Code**: To be confirmed  
**Area**: Audio & Speech  
**Keywords**: Temporal Forgery Localization, State Space Model, Deformable Mechanism, Relay Token, Deepfake Detection

## TL;DR
This paper proposes DeformTrace, which introduces a deformable dynamic receptive field mechanism and relay token scheme into state space models, combining Transformer-level global modeling with SSM-level efficient inference to achieve state-of-the-art accuracy and substantial efficiency gains in temporal forgery localization.

## Background & Motivation
Temporal Forgery Localization (TFL) aims to precisely identify tampered temporal segments in video and audio, providing finer-grained interpretability than binary forgery detection, and is critical for security forensics. Existing TFL methods (e.g., BA-TFD+, UMMAFormer, DiMoDif) primarily rely on CNNs or multi-scale Transformers, which achieve gradually improving accuracy but suffer from high computational overhead and slow inference.

State space models (SSMs), particularly the Mamba family, have demonstrated advantages of linear complexity and faster inference for long-sequence modeling. However, direct application to TFL faces three core challenges:

1. **Boundary ambiguity**: Forgery boundaries are less well-defined than in action detection; standard SSM's fixed state updates introduce a temporal smoothing effect that degrades localization precision.
2. **Forgery sparsity**: The majority of frames are authentic, and SSM's recurrent updates are dominated by non-forged patterns, weakening sensitivity to sparse forgeries.
3. **Long-range decay**: Although SSMs are efficient on long sequences, information decays exponentially with distance, limiting the capture of long-range contextual dependencies.

## Core Problem
How to design a TFL architecture that exploits the linear-complexity advantage of SSMs while overcoming their inherent limitations in boundary ambiguity, forgery sparsity, and long-range decay?

## Method

### Overall Architecture
DeformTrace builds upon the query-based architecture of TadTR and comprises three main modules:

- **Multi-scale audio-visual feature extraction**: A frozen Raven pre-trained encoder extracts visual and audio features, which are fused via concatenation and linear projection, then processed through $L-1$ downsampling layers to produce $L=6$ levels of multi-scale features.
- **Deformable encoder**: Consists of Deformable Self-SSM (DS-SSM), Deformable Self-Attention, and FFN.
- **Deformable decoder**: Consists of Deformable Cross-SSM (DC-SSM), Multi-Head Self-Attention, Deformable Cross-Attention, and FFN, with $M=3$ layers.

### Key Designs

#### Deformable Self-SSM (DS-SSM) — Addressing Boundary Ambiguity
This work is the first to introduce a deformable dynamic receptive field mechanism into temporal state space models. The core idea is:

1. For each feature $f_n^l$, a normalized temporal reference point $p_n^l$ is computed, mapping the feature index onto the video timeline.
2. An MLP predicts an offset matrix $o \in \mathbb{R}^{L \times N_s}$ (cross-scale offsets), which is added to the reference points to form deformable sampling locations.
3. Features are sampled from multi-scale representations via bilinear interpolation and then aggregated by an MLP.

Unlike deformable Mamba variants in the image domain (which require patch partitioning and token reordering), DS-SSM exploits the inherent temporal continuity of video/audio, eliminating redundant operations and significantly reducing computational cost.

#### Relay Token Mechanism — Addressing Long-Range Decay
In SSMs, token interactions depend on powers of the control matrix $\bar{A}^k$; since the elements of $\bar{A}$ are less than 1, information decays exponentially with distance. Inspired by relay nodes in wireless communications:

- $N_r=8$ learnable global relay tokens are uniformly inserted into the input sequence.
- The sequence is partitioned into $N_r+1$ subspaces; local states efficiently pass information to relay tokens, which then broadcast aggregated information to other subspaces.
- This forms a sparse sequence-to-token cross-subspace information flow.

Two auxiliary losses accompany this mechanism:

- **Enhancement loss** $\mathcal{L}_{enh}$: Encourages each relay token to better aggregate information from its neighboring subsequence (via cosine similarity).
- **Collaboration loss** $\mathcal{L}_{coop}$: Encourages diversity among different relay tokens to reduce redundancy (by minimizing mutual information).

#### Deformable Cross-SSM (DC-SSM) — Addressing Forgery Sparsity
Cross-sequence interaction is introduced into deformable state space modeling:

1. Each query $q_j^m$ uses its anchor proposal $(t_j^m, d_j^m)$ as a reference.
2. An MLP predicts multi-scale offsets to compute deformable sampling points.
3. Features are sampled from encoder outputs, concatenated with a learnable null token, and processed via forward SSM updates.
4. Leveraging SSM's aggregation property, only the last appended token is retained as output.

This mechanism partitions the global state space into query-specific subspaces, reducing the accumulation of non-forged information and enhancing sensitivity to sparse forgeries.

### Loss & Training
$$\mathcal{L} = \mathcal{L}_{match} + \mathcal{L}_{cls} + \lambda_1 \cdot \mathcal{L}_{enh} + \lambda_2 \cdot \mathcal{L}_{coop}$$
where $\lambda_1=0.5$ and $\lambda_2=0.2$. $\mathcal{L}_{match}$ adopts a DETR-style Hungarian matching loss.

## Key Experimental Results

### Datasets
- **LAV-DF**: 78K/31K/26K train/val/test, average video length 8.6 seconds.
- **AV-Deepfake1M**: 746K/57K/343K; larger scale and finer-grained forgeries.

### Main Results (LAV-DF)

| Method | mAP@0.5 | mAP@0.75 | mAP@0.95 | mAP Avg | mAR Avg |
|--------|---------|----------|----------|---------|---------|
| UMMAFormer | 98.8 | 95.5 | 37.6 | 77.3 | 92.3 |
| DiMoDif | 95.5 | 87.9 | 20.6 | 67.8 | 91.9 |
| FullFormer (pure Transformer baseline) | 94.6 | 85.7 | 29.4 | 69.9 | 87.3 |
| **DeformTrace** | **97.1** | **90.7** | **38.1** | **75.3** | **92.9** |

### Main Results (AV-Deepfake1M)

| Method | mAP Avg | mAR Avg | AUC |
|--------|---------|---------|-----|
| UMMAFormer | 22.2 | 42.8 | - |
| DiMoDif | 49.3 | 79.6 | - |
| FullFormer | 40.4 | 66.8 | - |
| **DeformTrace** | **52.9** | **81.8** | **99.2** |

DeformTrace surpasses the second-best DiMoDif by an average of 3.6% mAP and 2.2% mAR on AV-Deepfake1M.

### Efficiency Comparison

| Method | Trainable Params | FLOPs (G) | Inference Time (ms) |
|--------|-----------------|-----------|---------------------|
| UMMAFormer | 49.72M | 1563.9 | 857 |
| BA-TFD+ | 152.9M | 218.2 | 681 |
| **DeformTrace** | **20.8M** | **212.4** | **104** |

DeformTrace is **8.2×** faster than UMMAFormer in inference and reduces trainable parameters by **58%**.

### Ablation Study (AV-Deepfake1M)
- Vanilla SSM baseline: mAP 41.2, mAR 68.7
- +DS-SSM +DC-SSM: mAP 49.8 (+8.6)
- +Relay Token with all losses: mAP 52.9 (+11.7), mAR 81.8 (+13.1)
- $N_r=8$ relay tokens is optimal; excessive tokens lead to over-segmentation of subspaces.

## Highlights & Insights
1. **Three "firsts"**: First to introduce deformable receptive fields into temporal SSMs; first to propose relay tokens to explicitly mitigate SSM long-range decay; first to introduce cross-sequence interaction into state space modeling.
2. **Efficiency and accuracy simultaneously**: DeformTrace achieves higher accuracy with 1/7 of the Transformer inference time and 1/2.4 of the trainable parameters.
3. **Strong robustness**: Outperforms existing methods under 10 types of audio-visual perturbations (compression, noise, blur, etc.).
4. **Elegant relay token design**: Inspired by relay nodes in communications; auxiliary losses ensure both information aggregation and token diversity.

## Limitations & Future Work
1. DC-SSM is currently limited to interactions between queries and feature sequences; the authors suggest it could be extended to broader scenarios such as audio-visual correspondence learning.
2. The feature extractor (Raven) is frozen; end-to-end fine-tuning and its impact on performance remain unexplored.
3. The number of relay tokens is related to video duration, and adaptive adjustment strategies may be needed for practical deployment.
4. Experiments are validated only on talking-face forgery detection scenarios; more general video tampering settings have not been explored.

## Related Work & Insights
- **vs UMMAFormer/DiMoDif**: These methods rely on attention mechanisms and feature pyramids with high computational cost. DeformTrace replaces part of the attention modules with SSMs to achieve higher accuracy at linear complexity.
- **vs deformable Mamba variants (image domain)**: Image-domain methods require patch partitioning and token reordering; DeformTrace simplifies the design by exploiting temporal continuity.
- **vs LongMamba**: LongMamba addresses long-range decay via fixed-threshold channel classification and token filtering with limited adaptability; the relay token mechanism is more flexible and fully learnable.
- **vs TadTR/TE-TAD**: These query-based TAD methods achieve insufficient accuracy on TFL; DeformTrace significantly improves performance by incorporating SSM modules on top of this framework.

The relay token mechanism is a general paradigm applicable to any SSM architecture requiring long-range dependency modeling. The cross-sequence interaction design in DC-SSM provides a novel SSM-based approach for multimodal fusion beyond forgery detection. The idea of migrating deformable mechanisms from attention/convolution to SSMs is worth exploring in other temporal tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ (Three first-time innovations, though overall represents a combinatorial advance over existing techniques)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Two large-scale datasets + complete ablation + robustness tests + efficiency analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, intuitive figures, complete formulations)
- Value: ⭐⭐⭐⭐ (Strong practical utility in TFL, significant efficiency gains, generalizable components)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] GEM-TFL: Bridging Weak and Full Supervision for Forgery Localization](../../CVPR2026/audio_speech/gem-tfl_bridging_weak_and_full_supervision_for_forgery_localization_through_em-g.md)
- [\[AAAI 2026\] USE: A Unified Model for Universal Sound Separation and Extraction](use_a_unified_model_for_universal_sound_separation_and_extraction.md)
- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)
- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)
- [\[AAAI 2026\] Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition](let_the_model_learn_to_feel_mode-guided_tonality_injection_f.md)

<!-- RELATED:END -->
