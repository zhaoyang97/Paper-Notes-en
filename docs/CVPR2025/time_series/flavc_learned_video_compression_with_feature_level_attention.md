---
title: >-
  [Paper Note] FLAVC: Learned Video Compression with Feature Level Attention
description: >-
  [CVPR 2025][Time Series][Learned Video Compression] This work proposes FLAVC, which introduces a Feature-level Attention (FLA) module into the learned video compression (LVC) framework. By converting high-level local patch embeddings into one-dimensional batch-wise vectors and replacing traditional attention weights with a global context matrix, FLA achieves full-frame-level global perception. Combined with a Dense Overlapping Patcher and a hybrid Transformer-CNN encoder…
tags:
  - "CVPR 2025"
  - "Time Series"
  - "Learned Video Compression"
  - "Feature-level Attention"
  - "Transformer"
  - "Global Context Matrix"
  - "Dense Patcher"
date: 2026-05-08
content_hash: 5b8f9ac5bf8ae8eb
---

# FLAVC: Learned Video Compression with Feature Level Attention

**Conference**: CVPR 2025  
**Code**: [https://github.com/Z-CV-code/FLAVC](https://github.com/Z-CV-code/FLAVC)  
**Area**: Video Compression  
**Keywords**: Learned Video Compression, Feature-level Attention, Transformer, Global Context Matrix, Dense Patcher

## TL;DR
This work proposes FLAVC, which introduces a Feature-level Attention (FLA) module into the learned video compression (LVC) framework. By converting high-level local patch embeddings into one-dimensional batch-wise vectors and replacing traditional attention weights with a global context matrix, FLA achieves full-frame-level global perception. Combined with a Dense Overlapping Patcher and a hybrid Transformer-CNN encoder, FLAVC achieves state-of-the-art rate-distortion performance across four video compression datasets.

## Background & Motivation

**Background**: Learned Video Compression (LVC) reduces spatio-temporal redundancy in video sequences using deep learning methods. Recent advances primarily focus on shifting compression operations from the pixel domain to the feature domain, achieving efficient coding by combining motion estimation and compensation modules (MEMC) with CNN-based context extraction. Representative approaches such as DCVC and CANF-VC have already surpassed traditional coding standards (H.265/HEVC) in rate-distortion performance.

**Limitations of Prior Work**: (1) Existing feature-domain methods heavily rely on the motion estimation module; when motion estimation is inaccurate (e.g., due to occlusions or non-rigid deformations), compensation quality drops dramatically. (2) CNN-based context models are constrained by local receptive fields, failing to capture long-range dependencies across the entire frame, which leads to a lack of global perception in fast-motion scenarios. (3) The motion vectors themselves must be encoded and transmitted, generating additional bit rate overhead.

**Key Challenge**: Motion-based compensation frameworks are inherently limited by the accuracy of motion estimation and the overhead of motion coding. In complex motion scenarios, this paradigm of "motion estimation $\rightarrow$ compensation $\dots$ residual coding" encounters a severe bottleneck.

**Goal**: To design a full-frame-level global perception mechanism that is independent of motion signatures, enabling efficient spatio-temporal redundancy elimination directly in the feature domain.

**Key Insight**: Leveraging the global attention mechanism of Transformers, while addressing the excessive computational cost of traditional self-attention on high-resolution features. The authors design an ingenious feature-level attention mechanism that compresses patch embeddings into one-dimensional vectors to construct a global context matrix, significantly reducing computational complexity.

**Core Idea**: To bypass motion estimation and directly perceive full-frame context via Feature-level Attention, achieving efficient global modeling by replacing traditional attention weights with a global context matrix.

## Method

### Overall Architecture
Encoding pipeline of FLAVC: Features of the current frame are extracted via a hybrid Transformer-CNN encoder to obtain multi-scale representations $\rightarrow$ the FLA module utilizes the reference frame features to construct a global context matrix $\dots$ interaction with current frame features yields conditional predictions $\rightarrow$ the entropy encoder encodes latent representations based on the conditional distribution $\rightarrow$ bitstream transmission. Symmetrical decoding operations are performed at the decoder to recover features and reconstruct video frames.

### Key Designs

1. **Feature-level Attention (FLA) Module**:

    - **Function**: To achieve full-frame-level global perception independent of motion estimation.
    - **Mechanism**: The high-level feature map of the reference frame is first segmented into local patch embeddings. Each patch embedding is then converted into a one-dimensional batch-wise vector through linear projection. All patch vectors are aggregated to construct the global context matrix $\mathbf{G} \in \mathbb{R}^{B \times D}$. The features of the current frame are similarly encoded into query vectors, and the attention output is obtained through matrix multiplication with the global context matrix. The critical shift here is that instead of computing traditional $\text{softmax}(QK^T/\sqrt{d})V$ attention, the global context matrix directly replaces the attention weight matrix, reducing the complexity from $O(N^2)$ to $O(ND)$.
    - **Design Motivation**: Traditional self-attention exhibits quadratic computational complexity with respect to full-resolution feature maps, making it unsuitable for video compression. FLA substantially reduces computation while maintaining global perception by compressing the spatial dimensions into one-dimensional vectors.

2. **Dense Overlapping Patcher (DP)**:

    - **Function**: To preserve local detail features during the patchification process.
    - **Mechanism**: Traditional non-overlapping patch partition loses local information at patch boundaries. DP employs overlapping sliding windows for patch extraction, establishing a 50% overlap region between adjacent patches. The overlapping parts are naturally fused during embedding projection, enabling the global context matrix to incorporate more complete local details.
    - **Design Motivation**: Video compression demands high-precision pixel-level reconstruction and cannot tolerate information loss at patch boundaries. DP ensures full preservation of local features at the cost of a moderate increase in computation.

3. **Transformer-CNN Hybrid Encoder**:

    - **Function**: To alleviate the spatial feature bottleneck without increasing the latent representation size.
    - **Mechanism**: The encoder consists of alternately stacked Transformer blocks and CNN blocks. The CNN blocks are responsible for local feature extraction and spatial downsampling, while the Transformer blocks conduct global modeling on the downsampled low-resolution features. This hybrid design allows the Transformer to process only smaller feature maps while the CNN handles high-resolution details. The size of the final latent representation remains the same as that of a pure CNN encoder, introducing no extra transmission bit rate.
    - **Design Motivation**: Pure-Transformer encoders suffer from explosive computational complexity at high resolutions, whereas pure-CNN encoders lack global perception capabilities. The hybrid design achieves an optimal balance between efficiency and representation capacity.

### Loss & Training
The Rate-Distortion Optimization loss is employed: $\mathcal{L} = R + \lambda D$, where $R$ denotes the coding bit rate and $D$ represents the distortion metric (MSE or MS-SSIM). $\lambda$ governs the rate-distortion trade-off. Training is based on the Vimeo-90K dataset and structured on the NeuralCompression and TCM frameworks.

## Key Experimental Results

### Main Results

| Dataset | Method | BD-rate savings vs H.265 | BD-rate savings vs H.266 |
|--------|------|--------------------------|--------------------------|
| UVG | DCVC-HEM | -28.3% | -5.2% |
| UVG | CANF-VC | -31.5% | -8.7% |
| UVG | **FLAVC (Ours)** | **-38.2%** | **-15.6%** |
| MCL-JCV | DCVC-HEM | -25.1% | -3.8% |
| MCL-JCV | **FLAVC (Ours)** | **-34.7%** | **-12.3%** |
| HEVC-B | **FLAVC (Ours)** | **-36.5%** | **-13.8%** |
| HEVC-C | **FLAVC (Ours)** | **-32.1%** | **-10.2%** |

### Ablation Study

| Configuration | UVG BD-rate vs H.265 | Description |
|------|---------------------|------|
| Full FLAVC | -38.2% | Full model |
| w/o FLA (only MEMC) | -29.8% | Degenerates to traditional motion compensation framework |
| w/o DP (standard patch) | -35.4% | Without dense overlapping patcher |
| w/o Hybrid Encoder (pure CNN) | -31.6% | Without Transformer global modeling |
| FLAVC-Light (scaled-down) | -33.5% | Computation dramatically reduced |

### Key Findings
- The FLA module is the most critical component (contributing approximately -8.4% BD-rate), validating the significance of global perception in video compression.
- FLAVC consistently outperforms the traditional H.266/VVC coding standard across all four datasets, achieving 10-16% BD-rate savings.
- Dense Patcher contributes around -2.8% BD-rate, offering more pronounced gains in videos with dense textures and sharp edges.
- FLAVC-Light (the scaled-down version) retains a -33.5% BD-rate saving while reducing computational complexity by approximately 60%, making it highly suitable for practical deployment.
- In fast-motion scenarios (such as green/high-motion sequences in UVG), the advantages of FLA are more prominent, as global perception continues to function effectively even when motion estimation fails.
- Cited 5 times (as of April 2026).

## Highlights & Insights
- **Global Perception Bypassing Motion Estimation**: FLA captures inter-frame correlations without relying on explicit motion estimation, marking a significant breakthrough in design paradigm. This mechanism has the potential to entirely replace the traditional "motion estimation + compensation" paradigm in the future.
- **One-Dimensional Compression Efficiency Trick**: Compressing patch embeddings into one-dimensional vectors to construct a global matrix elegantly reduces the computational complexity of attention from quadratic to linear, a trick that can be transferred to other efficiency-critical scenarios.
- **Practical Value of FLAVC-Light**: Demonstrates that the proposed method scales effectively to smaller variants while remaining competitive, offering high engineering viability for practical applications.

## Limitations & Future Work
- The latency of the current version may not satisfy real-time video communication requirements, necessitating further optimization of inference speed.
- The global context matrix in FLA is constructed frame-by-frame without utilizing temporal information spanning multiple frames; exploring the accumulation of temporal context is a potential future direction.
- It can be integrated with traditional coding standards (H.266/VVC) to form hybrid frameworks that exploit established rate control mechanisms.
- Although the repository is open-sourced, the training scripts are not yet released (as of April 2026, containing only the README and framework diagrams).

## Related Work & Insights
- **vs DCVC-HEM**: DCVC-HEM employs multi-scale motion compensation, operating in the feature domain but still relying on motion estimation. The FLA module in FLAVC offers a viable alternative to motion estimation.
- **vs CANF-VC**: CANF-VC utilizes conditionally augmented normalizing flows for probabilistic modeling, but its context extraction remains confined to the local receptive fields of CNNs. FLAVC delivers a fundamental advancement via global perception capability.
- **vs TCM (Transformer-CNN Mixed)**: FLAVC is built upon the TCM framework, with its core contributions being the introduction of FLA and DP.
- **vs H.266/VVC**: Traditional standards retain advantages in terms of low latency and hardware friendliness, yet their compression efficiency has been visibly surpassed by learned approaches including FLAVC.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of replacing motion estimation with feature-level global attention is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Conducts comprehensive comparisons across four datasets against both traditional coding standards and learned methods.
- Writing Quality: ⭐⭐⭐⭐ Clear methodological descriptions and intuitive architectural diagrams.
- Value: ⭐⭐⭐⭐ Advances learned video compression toward the global perception paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DejaVid: Encoder-Agnostic Learned Temporal Matching for Video Classification](dejavid_encoder-agnostic_learned_temporal_matching_for_video_classification.md)
- [\[NeurIPS 2025\] AttentionPredictor: Temporal Patterns Matter for KV Cache Compression](../../NeurIPS2025/time_series/attentionpredictor_temporal_patterns_matter_for_kv_cache_com.md)
- [\[ICML 2025\] Customizing the Inductive Biases of Softmax Attention using Structured Matrices](../../ICML2025/time_series/customizing_the_inductive_biases_of_softmax_attention_using_structured_matrices.md)
- [\[NeurIPS 2025\] Learning Time-Scale Invariant Population-Level Neural Representations](../../NeurIPS2025/time_series/learning_time-scale_invariant_population-level_neural_representations.md)
- [\[ICLR 2026\] Online Time Series Prediction Using Feature Adjustment](../../ICLR2026/time_series/online_time_series_prediction_using_feature_adjustment.md)

</div>

<!-- RELATED:END -->
