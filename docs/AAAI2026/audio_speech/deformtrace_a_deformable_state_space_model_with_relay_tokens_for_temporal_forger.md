---
title: >-
  [Paper Note] DeformTrace: A Deformable State Space Model with Relay Tokens for Temporal Forgery Localization
description: >-
  [AAAI2026][Audio & Speech][Temporal Forgery Localization] DeformTrace is proposed, introducing deformable dynamic receptive fields and a relay token mechanism into the State Space Model (SSM). By combining the global modeling capabilities of Transformers with the efficient inference of SSMs, it achieves SOTA accuracy and significant efficiency improvements in temporal forgery localization.
tags:
  - "AAAI2026"
  - "Audio & Speech"
  - "Temporal Forgery Localization"
  - "State Space Model"
  - "Deformable Mechanism"
  - "Relay Token"
  - "Deepfake Detection"
date: 2026-05-08
content_hash: 019e20b16d52b39a
---

# DeformTrace: A Deformable State Space Model with Relay Tokens for Temporal Forgery Localization

**Conference**: AAAI2026  
**arXiv**: [2603.04882](https://arxiv.org/abs/2603.04882)  
**Code**: To be confirmed  
**Area**: Audio & Speech  
**Keywords**: Temporal Forgery Localization, State Space Model, Deformable Mechanism, Relay Token, Deepfake Detection

## TL;DR
DeformTrace is proposed, introducing deformable dynamic receptive fields and a relay token mechanism into the State Space Model (SSM). By combining the global modeling capabilities of Transformers with the efficient inference of SSMs, it achieves SOTA accuracy and significant efficiency improvements in temporal forgery localization.

## Background & Motivation
Temporal Forgery Localization (TFL) aims to accurately identify the specific time spans that have been tampered with in video and audio, providing finer-grained explainability than binary deepfake detection and proving crucial for security forensics. Existing TFL methods (e.g., BA-TFD+, UMMAFormer, DiMoDif) primarily rely on CNNs or multi-scale Transformers. Although they have gradually improved accuracy, they still suffer from heavy computational overhead and slow inference speeds.

State Space Models (SSMs), particularly the Mamba series, exhibit linear complexity and faster inference advantages in long-sequence modeling. However, their direct application to TFL encounters three core challenges:

1. **Boundary Ambiguity**: Forgery boundaries are not as clearly defined as action boundaries in action detection. The fixed-state updates of standard SSMs create a temporal smoothing effect, which degrades localization precision.
2. **Forgery Sparsity**: The majority of frames are real, meaning the recursive updates of SSMs are dominated by non-forged patterns, which weakens their sensitivity to sparse forgeries.
3. **Long-range Decay**: Although SSMs are efficient for long sequences, information decays exponentially over distance, limiting their capability to capture long-range contexts.

## Core Problem
How can one design a TFL architecture that leverages the linear complexity advantage of SSMs while overcoming their inherent limitations in boundary ambiguity, forgery sparsity, and long-range decay?

## Method

### Overall Architecture
DeformTrace is based on TadTR's query-based architecture and consists of three main modules:

- **Multi-scale Audio-Visual Feature Extraction**: Frozen Raven pre-trained encoders are used to extract visual and audio features. These are concatenated and linearly projected to obtain fused features, which are then passed through $L-1$ downsampling layers to generate $L=6$ levels of multi-scale features.
- **Deformable Encoder**: Contains Deformable Self-SSM (DS-SSM), Deformable Self-Attention, and FFN.
- **Deformable Decoder**: Contains Deformable Cross-SSM (DC-SSM), Multi-Head Self-Attention, Deformable Cross-Attention, and FFN with $M=3$ layers.

### Key Designs

#### Deformable Self-SSM (DS-SSM) — Addressing Boundary Ambiguity
This introduces the deformable dynamic receptive field mechanism to temporal state space models for the first time. The mechanism operates as follows:

1. A normalized temporal reference point $p_n^l$ is calculated for each feature $f_n^l$, mapping the feature index onto the video timeline.
2. An MLP is used to predict the offset matrix $o \in \mathbb{R}^{L \times N_s}$ (cross-scale offset), which is added to the reference points to form deformable sampling points.
3. Bilinear interpolation is used to sample from multi-scale features, which are then aggregated by an MLP.

Unlike deformable Mamba variants in the image domain (which require patch partition and token sorting), DS-SSM exploits the inherent temporal continuity of video/audio, omitting redundant operations and significantly reducing computational overhead.

#### Relay Token Mechanism — Addressing Long-range Decay
In SSMs, interactions between tokens depend on the powers of the state transition matrix $\bar{A}^k$. Since the elements of $\bar{A}$ are less than 1, information decays exponentially over distance. Drawing inspiration from relay nodes in wireless communication:

- $N_r=8$ learnable global relay tokens are uniformly inserted into the input sequence.
- The sequence is partitioned into $N_r+1$ subspaces, where local states efficiently transfer information to the relay tokens, which then broadcast the aggregated information to other subspaces.
- This forms a sparse sequence-to-token cross-sequence information flow.

Two auxiliary losses are designed to accompany this:

- **Enhancement Loss** $\mathcal{L}_{enh}$: Encourages each relay token to better aggregate information from adjacent subsequences (via cosine similarity).
- **Cooperation Loss** $\mathcal{L}_{coop}$: Encourages diversity among different relay tokens to reduce redundancy (via minimizing mutual information).

#### Deformable Cross-SSM (DC-SSM) — Addressing Forgery Sparsity
This mechanism introduces cross-sequence interaction into deformable state space modeling:

1. Each query $q_j^m$ uses its anchor proposal $(t_j^m, d_j^m)$ as a reference.
2. An MLP predicts multi-scale offsets to compute deformable sampling points.
3. Features are sampled from the encoder output, concatenated with a learnable empty token, and then updated via forward SSM.
4. Leveraging the aggregation property of SSM, only the last appended token is retained as the output.

This design partitions the global state space into query-specific subspaces, reducing the accumulation of non-forgery information and enhancing sensitivity to sparse forgeries.

### Loss & Training
$$\mathcal{L} = \mathcal{L}_{match} + \mathcal{L}_{cls} + \lambda_1 \cdot \mathcal{L}_{enh} + \lambda_2 \cdot \mathcal{L}_{coop}$$
where $\lambda_1=0.5$ and $\lambda_2=0.2$. $\mathcal{L}_{match}$ adopts the DETR-style Hungarian matching loss.

## Key Experimental Results

### Datasets
- **LAV-DF**: 78K/31K/26K for training/validation/testing, with an average video length of 8.6 seconds.
- **AV-Deepfake1M**: 746K/57K/343K, featuring larger-scale and finer-grained forgeries.

### Main Results (LAV-DF)

| Method | mAP@0.5 | mAP@0.75 | mAP@0.95 | mAP Avg | mAR Avg |
|------|---------|----------|----------|---------|---------|
| UMMAFormer | 98.8 | 95.5 | 37.6 | 77.3 | 92.3 |
| DiMoDif | 95.5 | 87.9 | 20.6 | 67.8 | 91.9 |
| FullFormer (Pure Transformer Baseline) | 94.6 | 85.7 | 29.4 | 69.9 | 87.3 |
| **DeformTrace** | **97.1** | **90.7** | **38.1** | **75.3** | **92.9** |

### Main Results (AV-Deepfake1M)

| Method | mAP Avg | mAR Avg | AUC |
|------|---------|---------|-----|
| UMMAFormer | 22.2 | 42.8 | - |
| DiMoDif | 49.3 | 79.6 | - |
| FullFormer | 40.4 | 66.8 | - |
| **DeformTrace** | **52.9** | **81.8** | **99.2** |

It outperforms the second-best method, DiMoDif, on AV-Deepfake1M by an average of 3.6% mAP and 2.2% mAR.

### Efficiency Comparison

| Method | Trainable Params | FLOPs(G) | Inference Time (ms) |
|------|-----------|----------|-------------|
| UMMAFormer | 49.72M | 1563.9 | 857 |
| BA-TFD+ | 152.9M | 218.2 | 681 |
| **DeformTrace** | **20.8M** | **212.4** | **104** |

The inference speed is **8.2 times** faster than UMMAFormer, with a **58%** reduction in trainable parameters.

### Ablation Study (AV-Deepfake1M)
- Vanilla SSM baseline: mAP 41.2, mAR 68.7
- +DS-SSM +DC-SSM: mAP 49.8 (+8.6)
- +Relay Token and all losses: mAP 52.9 (+11.7), mAR 81.8 (+13.1)
- The number of relay tokens $N_r=8$ is optimal; too many tokens lead to over-segmentation of the subspace.

## Highlights & Insights
1. **Three "Firsts"**: The first to introduce deformable receptive fields into temporal SSMs; the first to propose a relay token mechanism to explicitly mitigate long-range decay in SSMs; and the first to introduce cross-sequence interactions into state-space modeling.
2. **Balancing Efficiency and Accuracy**: Higher accuracy is achieved using only 1/7 of the inference time and 1/2.4 of the trainable parameters compared to Transformers.
3. **Strong Robustness**: Outperforms existing methods under 10 types of audio-visual distortions (such as compression, noise, blur, etc.).
4. **Ingenious Relay Token Design**: Inspired by the relay concept in communication fields, auxiliary losses ensure both information aggregation and diversity.

## Limitations & Future Work
1. DC-SSM is currently only used for interactions between queries and feature sequences. The authors mention it can be extended to broader scenarios such as audio-visual correspondence learning.
2. The feature extractor (Raven) is frozen, and the impact of end-to-end fine-tuning on performance has not been explored.
3. The number of relay tokens correlates with video duration, and adaptive adjustment strategies may be required for practical deployment.
4. Experiments are only validated on forgery detection in talking face scenarios, without involving more general video tampering scenarios.

## Related Work & Insights
- **vs UMMAFormer/DiMoDif**: These methods rely heavily on attention and feature pyramids, which are computationally expensive. DeformTrace replaces some attention modules with SSMs, achieving higher accuracy with linear complexity.
- **vs Deformable Mamba Variants (Image Domain)**: Image-domain methods require patch partition and token sorting, whereas DeformTrace simplifies the design by exploiting temporal continuity.
- **vs LongMamba**: LongMamba addresses long-range decay through fixed-threshold channel classification and token filtering, which has limited adaptability; the relay token mechanism is more flexible and learnable.
- **vs TadTR/TE-TAD**: These query-based TAD methods lack sufficient accuracy for TFL, whereas DeformTrace significantly improves performance by integrating SSM modules on top of them.

### Insights & Connections
1. The relay token mechanism is a general framework that can be applied to any SSM architecture requiring long-range dependency modeling.
2. The cross-sequence interaction design of DC-SSM provides a new SSM paradigm for multi-modal fusion, which is not limited to forgery detection.
3. The idea of transferring deformable mechanisms from attention/convolution fields to SSMs is worth exploring in other temporal tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ (Three "first" innovations, but the overall framework is a combinatorial innovation of existing techniques)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Two large-scale datasets + comprehensive ablation + robustness testing + efficiency analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, intuitive diagrams, complete formulations)
- Value: ⭐⭐⭐⭐ (High practicality in the TFL domain, significant efficiency gains, and general-purpose components offer valuable reference)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GEM-TFL: Bridging Weak and Full Supervision for Forgery Localization](../../CVPR2026/audio_speech/gem-tfl_bridging_weak_and_full_supervision_for_forgery_localization_through_em-g.md)
- [\[ICLR 2026\] MambaVoiceCloning: Efficient and Expressive Text-to-Speech via State-Space Modeling and Diffusion Control](../../ICLR2026/audio_speech/mambavoicecloning_efficient_and_expressive_text-to-speech_via_state-space_modeli.md)
- [\[ICLR 2026\] Speech World Model: Causal State–Action Planning with Explicit Reasoning for Speech](../../ICLR2026/audio_speech/speech_world_model_causal_stateaction_planning_with_explicit_reasoning_for_speec.md)
- [\[AAAI 2026\] A Text-Routed Sparse Mixture-of-Experts Model with Explanation and Temporal Alignment for Multi-Modal Sentiment Analysis](text-routed_sparse_mixture-of-experts_model_with_explanation_and_temporal_alignm.md)
- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)

</div>

<!-- RELATED:END -->
