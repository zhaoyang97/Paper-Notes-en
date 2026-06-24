---
title: >-
  [Paper Note] DejaVid: Encoder-Agnostic Learned Temporal Matching for Video Classification
description: >-
  [CVPR 2025][Time Series][Video Classification] This paper proposes DejaVid, an encoder-agnostic, lightweight approach for enhancing video classification. Instead of representing a video with a single embedding, DejaVid represents it as a variable-length Temporal Sequence of Embeddings (TSE). By learning importance weights for each time step and feature dimension, combined with an improved differentiable DTW algorithm for temporal alignment classification…
tags:
  - "CVPR 2025"
  - "Time Series"
  - "Video Classification"
  - "Dynamic Time Warping"
  - "Temporal Sequence of Embeddings"
  - "Encoder-Agnostic"
  - "Lightweight Post-Processing"
date: 2026-05-08
content_hash: a410103a1d0e2bf6
---

# DejaVid: Encoder-Agnostic Learned Temporal Matching for Video Classification

**Conference**: CVPR 2025  
**arXiv**: [2506.12585](https://arxiv.org/abs/2506.12585)  
**Code**: [https://github.com/darrylho/DejaVid](https://github.com/darrylho/DejaVid)  
**Area**: Time Series / Video Understanding  
**Keywords**: Video Classification, Dynamic Time Warping, Temporal Sequence of Embeddings, Encoder-Agnostic, Lightweight Post-Processing

## TL;DR
This paper proposes DejaVid, an encoder-agnostic, lightweight approach for enhancing video classification. Instead of representing a video with a single embedding, DejaVid represents it as a variable-length Temporal Sequence of Embeddings (TSE). By learning importance weights for each time step and feature dimension, combined with an improved differentiable DTW algorithm for temporal alignment classification, it achieves SOTA results of 77.2% on SSV2 and 89.1% on K400 with an increase of only <1.8% parameters.

## Background & Motivation

**Background**: Large video Transformers (such as VideoMAE V2-g with 1B parameters) have achieved excellent results in action recognition, but their approach to handling variable-length videos is straightforward yet crude—extracting embeddings from multiple temporal segments and spatial crops and then averaging them.

**Limitations of Prior Work**: The averaging operation loses three types of critical temporal information: (1) variation in video duration—different videos have different lengths; (2) temporal order of events—reversing the actions of opening and closing a door yields completely different semantics; (3) temporal variation in feature importance—in a basketball shooting video, early frames emphasize player positioning whereas later frames focus on whether the ball enters the hoop. Existing improvement schemes (e.g., inserting temporal layers into Transformers) require retraining large models, which is computationally prohibitive.

**Key Challenge**: Large models perform well but have weak temporal modeling capabilities, whereas specialized temporal methods require invasive architecture modifications and retraining, making it difficult to achieve both simultaneously.

**Goal**: How to enhance temporal modeling capabilities using a lightweight post-processor without modifying or retraining large-scale pre-trained encoders.

**Key Insight**: Representing videos as a Temporal Sequence of Embeddings (TSE) instead of a single embedding vector, using the DTW algorithm for temporal alignment, and learning importance weights across the temporal and feature dimensions.

**Core Idea**: Utilizing sliding windows to encode videos into variable-length TSEs, and performing classification by aligning and matching them with the centroid TSE of each class via weighted DTW.

## Method

### Overall Architecture
Pre-training phase: Sliding windows are applied to videos to extract $T \times N_f$ Temporal Sequences of Embeddings (TSEs), and the centroid TSE for each class is computed using the DBA algorithm. Training phase: The class centroid TSEs and the temporal-feature weight tensor $U$ are learned. Inference phase: The weighted DTW distances from the input TSE to each class centroid are computed, followed by a softmin operation to obtain classification probabilities.

### Key Designs

1. **Temporal Sequence of Embeddings (TSE)**:

    - **Function**: Preserving the temporal order and variable-length characteristics of videos.
    - **Mechanism**: A sliding window is applied to the video, and each window is fed into a frozen encoder to obtain an embedding vector. The embeddings of all windows are concatenated into a $T \times N_f$ TSE (where $T$ varies with video length). Compared to traditional multi-segment averaging, TSE naturally preserves temporal order and supports variable lengths. The centroid TSE is initialized using the DBA (DTW Barycenter Averaging) algorithm and subsequently optimized as learnable parameters.
    - **Design Motivation**: A single embedding cannot differentiate between videos that have identical frame sets but different temporal orders (e.g., opening a door vs. closing a door).

2. **Time-Weighted DTW Distance**:

    - **Function**: Modeling the temporal variation of feature importance.
    - **Mechanism**: Classic DTW computes point-to-point distances using the Manhattan distance. This paper introduces learnable weights $u_{i,k}$, such that $dist_w(u_i, a_i, b_j) = \sum_k u_{i,k} |a_{i,k} - b_{j,k}|$. The weight tensor $U \in \mathbb{R}_{>0}^{N_c \times T_c \times N_f}$ has the same shape as the centroid TSE, representing the importance of each feature at each time step for each class. Positivity is guaranteed by storing $\log(U)$ and applying $\exp$ during forward propagation. The diagonal transition of DTW is removed to stabilize the model (fixing the path length to $n+m-1$).
    - **Design Motivation**: In a basketball shooting video, the feature of "the ball entering the hoop" is more important in the later stages. The default uniform weighting of DTW cannot capture such temporal variations.

3. **Neural Network Reformulation of DTW**:

    - **Function**: Efficiently reformulating DTW dynamic programming into a backpropagation-capable neural network.
    - **Mechanism**: The 2D grid of DTW is reorganized diagonally—the $l$-th diagonal can be executed in parallel once the $(l-1)$-th diagonal is completed. Each diagonal is equivalent to a min-pooling layer with kernel=2 and stride=1, plus an additive skip-connection. The entire DTW process is thus transformed into a serial stack of min-pooling layers, which can be directly optimized using standard backpropagation. A custom CUDA kernel is developed to accelerate the computation by two orders of magnitude.
    - **Design Motivation**: The serial path of traditional DTW, which is $O(nm)$, is too long. The reformulation reduces the critical path to $O(n+m)$.

### Loss & Training
The weighted DTW distances from the training TSEs to all class centroids are computed, and a softmin operation yields the probability distribution, which is trained with cross-entropy loss. Only the centroid TSE $C$ and weight tensor $U$ are optimized, while the encoder is completely frozen. The model is trained using the AdamW optimizer for 36 epochs with a cosine learning rate schedule. In each epoch, a copy of the current centroids/weights is frozen to compute the warping path, while the unfrozen original parameters are used for backpropagation.

## Key Experimental Results

### Main Results

| Method | Params | K400 Top-1 | SSV2 Top-1 | HMDB51 Top-1 |
|------|--------|-----------|-----------|-------------|
| VideoMAE V2-g (baseline) | 1013M | 88.4% | 76.7% | 88.1% |
| InternVideo2-6B | 6B | 92.1% | 77.5% | - |
| **DejaVid (frozen weights)** | **+5.8M** | **89.1%** | 77.1% | 88.3% |
| **DejaVid (full learning)** | **+11.6M** | 88.9% | **77.2%** | **88.6%** |

Compared to VideoMAE V2-g: K400 +0.7%, SSV2 +0.5%, and HMDB51 +0.5%. The additional parameter count is less than 1.8% of the encoder, and training takes less than 3 hours.

### Ablation Study

| Configuration | SSV2 Top-1 | K400 Top-1 | HMDB51 Top-1 |
|------|-----------|-----------|-------------|
| Centroid Learning Only (Frozen Weights) | 77.1% | 89.1% | 88.3% |
| Weight Learning Only (Frozen Centroids) | 77.0% | 88.4% | 88.5% |
| Full Learning | **77.2%** | 88.9% | **88.6%** |
| Temporal Oversampling (w/o DejaVid) | 76.2% | 88.1% | 88.0% |

### Key Findings
- Centroid learning contributes the most (an average of +0.43%), while weight learning is more beneficial when data is scarce (HMDB51 +0.4%).
- Pure temporal oversampling (increasing the number of sampled frames but still averaging them) does not yield performance gains, proving that DejaVid's improvement comes from temporal alignment rather than the amount of information.
- Removing the diagonal transition of DTW is crucial for model stability—fixing the path length eliminates unfair comparisons among different paths.
- Updating the frozen copy used for warping path computation every epoch instead of every batch also aids convergence stability.

## Highlights & Insights
- **Encoder-Agnostic Plug-and-Play Enhancement**: Without modifying any architecture or retraining any weights, significant improvements are achieved on a 1B-parameter SOTA model. This post-processing paradigm can be generalized to any video encoder.
- **Neural Network Reformulation of DTW**: Expressing a classic dynamic programming algorithm as a stack of min-pooling + skip-connection increases parallelism and naturally supports backpropagation. This reformulation concept can be transferred to other DP algorithms.
- **Modeling Temporal Variation of Feature Importance**: Learning the weight for each time step and feature dimension is an insightful design—different actions indeed focus on different information at different stages of time.

## Limitations & Future Work
- Validated only on VideoMAE V2-g (as it is the only SOTA that released pre-trained weights); the claim of being encoder-agnostic requires validation with more models.
- Weight learning is prone to overfitting on large datasets (on K400, full learning performs worse than frozen weights), requiring regularization.
- TSE generation requires multiple encoder forward passes per video (33 times for K400), which incurs high inference costs.
- The centroid TSE length is fixed at 8, a choice that lacks theoretical justification.

## Related Work & Insights
- **vs ILA/ATM**: These methods insert temporal layers between Transformer blocks, requiring retraining. DejaVid is fully post-processing and zero-intrusive.
- **vs Soft-DTW/D3TW**: Previous differentiable DTW works did not model variations in temporal-feature dimension importance. DejaVid scales DTW to high-dimensional scenarios by learning weights.
- **vs SlowFast**: SlowFast uses a dual-path encoder to fuse different frame rates, which is computationally expensive. DejaVid uses a single encoder + sliding window + DTW, making it much lighter.

## Rating
- Novelty: ⭐⭐⭐⭐ The neural network reformulation of DTW and time weighting are highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Verified on three datasets, but utilizes only one encoder.
- Writing Quality: ⭐⭐⭐⭐ Clear and highly systematic.
- Value: ⭐⭐⭐⭐ The plug-and-play generalizability has substantial practical engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] FLAVC: Learned Video Compression with Feature Level Attention](flavc_learned_video_compression_with_feature_level_attention.md)
- [\[NeurIPS 2025\] TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting](../../NeurIPS2025/time_series/timeperceiver_an_encoder-decoder_framework_for_generalized_time-series_forecasti.md)
- [\[NeurIPS 2025\] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](../../NeurIPS2025/time_series/multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)
- [\[ACL 2025\] LETS-C: Leveraging Text Embedding for Time Series Classification](../../ACL2025/time_series/lets-c_leveraging_text_embedding_for_time_series_classification.md)
- [\[AAAI 2026\] SELDON: Supernova Explosions Learned by Deep ODE Networks](../../AAAI2026/time_series/seldon_supernova_explosions_learned_by_deep_ode_networks.md)

</div>

<!-- RELATED:END -->
