---
title: >-
  [Paper Note] VideoMamba: Spatio-Temporal Selective State Space Model
description: >-
  [ECCV 2024][Video Understanding][State Space Model] This paper proposes VideoMamba (KAIST version), a pure Mamba-based video recognition model. By designing a Spatio-Temporal Forward and Backward SSM, it effectively handles the complex interaction between non-sequential spatial information and sequential temporal information in videos, achieving competitive performance with Transformers while maintaining linear complexity.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "State Space Model"
  - "Mamba"
  - "Spatio-temporal Modeling"
  - "Video Recognition"
  - "Bidirectional Scanning"
date: 2026-05-08
content_hash: e151fa65edebfa83
---

# VideoMamba: Spatio-Temporal Selective State Space Model

**Conference**: ECCV 2024  
**arXiv**: [2407.08476](https://arxiv.org/abs/2407.08476)  
**Code**: [https://github.com/jinyjelly/VideoMamba](https://github.com/jinyjelly/VideoMamba)  
**Area**: Video Understanding  
**Keywords**: State Space Model, Mamba, Spatio-temporal Modeling, Video Recognition, Bidirectional Scanning

## TL;DR

This paper proposes VideoMamba (KAIST version), a pure Mamba-based video recognition model. By designing a Spatio-Temporal Forward and Backward SSM, it effectively handles the complex interaction between non-sequential spatial information and sequential temporal information in videos, achieving competitive performance with Transformers while maintaining linear complexity.

## Background & Motivation

**Background**: Transformers have achieved remarkable performance in video recognition, but the quadratic complexity of self-attention becomes a severe bottleneck when processing multi-frame videos, especially in resource-constrained environments.

**Limitations of Prior Work**: Traditional CNNs employ 3D convolutions or spatio-temporal decomposed convolutions, which are relatively efficient but limited in capturing long-range dependencies. In contrast, pure Transformer architectures can model long-range dependencies, but their quadratic complexity scales unacceptably with sequence length.

**Key Challenge**: Spatially, video information is non-sequential (e.g., the position and pose of a person in a frame), whereas temporally, it is sequential (e.g., human actions evolving over frames). The key challenge is how to effectively express such spatio-temporal interactions in a 1D sequence model.

**Goal**: This study aims to explore the adaptation of pure Mamba architectures to video recognition tasks, particularly focusing on addressing the spatio-temporal bidirectional scanning directions for video tokens.

**Key Insight**: The investigation starts from the choice of the backward scanning direction, systematically studying the impact of three strategies on model performance: spatial reversal, temporal reversal, and spatio-temporal reversal.

**Core Idea**: Through a Spatio-Temporal Forward and Backward SSM, the backward scan undergoes full spatio-temporal reversal for all tokens, enabling the forward and backward token sequences to complement each other.

## Method

### Overall Architecture

VideoMamba (KAIST version) utilizes a pure Mamba encoder architecture. The input video $V \in \mathbb{R}^{T \times H \times W \times C}$ is first mapped to $n_t \cdot n_h \cdot n_w$ video tokens $z_i \in \mathbb{R}^d$ ($d=384$) via a Video Tokenizer (a 3D convolution with a tubelet size of $s_t \times s_h \times s_w = 2 \times 16 \times 16$). After adding positional embeddings and prepending a class token, the sequence is fed into $L=24$ layers of VideoMamba encoders. Finally, the class token is normalized and passed through a single-layer MLP to output the classification result.

### Key Designs

1. **Video Tokenizer**: It extracts tokens from non-overlapping tubelets using a 3D convolution. The key initialization strategy is to inflate a pre-trained 2D convolution into a 3D convolution—expanding the weight tensor along the temporal axis and averaging them:

    $n_t = \lfloor T/s_t \rfloor, \quad n_h = \lfloor H/s_h \rfloor, \quad n_w = \lfloor W/s_w \rfloor$

   This inflation strategy allows the model to effectively leverage ImageNet pre-trained weights.

2. **Positional Embedding**: Although SSMs theoretically do not require positional embeddings (as their recurrent nature implicitly captures sequential order), this paper systematically compares various positional embedding schemes considering the spatio-temporal nature of videos. Ablation studies confirm that **Temporal Expanding** initialization is optimal, which copies the image pre-trained positional embeddings $P_{image} \in \mathbb{R}^{n_h \cdot n_w \times d}$ across the temporal axis $n_t$ times:

   | Positional Embedding Method | SSV2 | HMDB |
   |------------|------|------|
   | No Positional Embedding | 63.2% | 48.7% |
   | Sinusoidal | 63.3% | 47.5% |
   | Learned (Random Init) | 63.4% | 47.9% |
   | Learned (Spatial Interpolation) | 63.6% | 49.4% |
   | Learned (Embedding Dimension Interpolation) | 63.6% | 51.5% |
   | **Learned (Temporal Expanding)** | **63.7%** | **58.9%** |

   Temporal expanding outperforms the second-best method by 7.4% on HMDB, demonstrating that inheriting spatial positional information from image models and properly expanding it to the temporal dimension is crucial.

3. **Spatio-Temporal Forward and Backward SSM**: This is the core contribution of this work. To handle the interaction between non-sequential spatial and sequential temporal information in videos, three backward scanning directions are proposed:

    - **Spatio-temporal reversal**: Completely reverses the order of all $n_t \cdot n_h \cdot n_w$ tokens, which is equivalent to concatenating video frames vertically into a long image and then reversing it. The forward and backward token orders are fully complementary.
    - **Spatial reversal**: Reverses only the $n_h \cdot n_w$ tokens within each frame while keeping the temporal sequence unchanged. This maintains a clear temporal flow.
    - **Temporal reversal**: Keeps the in-frame spatial token order unchanged, reversing only the temporal sequence of frames. This represents an event in reverse without altering the spatial integrity of the frames.

   "Spatio-temporal reversal" achieves the best performance (SSV2: 64.7%, HMDB: 55.2%) because the forward and backward scans offer maximum complementarity of token sequences. Spatial reversal performs the worst, as the relative positions of most tokens remain unchanged in the forward and backward paths.

4. **Delta Parameter Analysis**: The $\Delta$ parameter in Mamba acts as a gating mechanism—a large $\Delta$ indicates forgetting the hidden states to focus on the current input, while a small $\Delta$ indicates ignoring the current input. Visualization analyses show that:

    - Shallow layers: $\Delta$ values are generally high, allowing the model to grasp the overall scene first.
    - Deep layers: $\Delta$ values decrease and focus on key motion regions (e.g., hands in cycling, the athlete in diving), successfully filtering out static backgrounds.
   
   This proves that VideoMamba achieves highly effective spatio-temporal reasoning via $\Delta$.

5. **Temporal Consistency Dependency Analysis**: The model's dependency on temporal order is verified by shuffling the input frames:

   | Shuffle Strategy | HMDB Top-1 |
   |---------|-----------|
   | Interleaved (maximum perturbation) | 51.3% |
   | Pairwise | 53.5% |
   | Block-wise | 56.5% |
   | **Sequential (original order)** | **58.9%** |

   The more severe the temporal perturbation, the more pronounced the performance drop, verifying that the model heavily relies on temporal order for reasoning.

### Loss & Training

- **Optimizer**: AdamW, learning rate 3e-4, using a cosine decay schedule with linear warmup.
- **Training Strategy**: Trained on K400 for 30 epochs, SSV2 for 35 epochs, and HMDB for 50 epochs with a batch size of 64.
- **Data Augmentation**: Label Smoothing, RandAugment, Random Erasing.
- **Initialization**: ImageNet-1K pre-trained weights to initialize the backbone.
- **Inference**: Multi-crop (view) inference averaging the scores.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (VideoMamba) | Comparing Method | Description |
|--------|------|------|------|------|
| HMDB51 (IN-1K Initialization) | Top-1 | 59.3% (32f) | VideoSwin-T 54.4% | +4.9%, fewer parameters |
| HMDB51 (K400 Initialization) | Top-1 | 68.6% (16f) | Mamba-ND 59.0% | +9.6% |
| HMDB51 (K400 Initialization) | Top-1 | 75.7% (32f) | VideoSwin-T(K400) 69.9% | +5.8% |
| SSV2 (IN-1K Initialization) | Top-1 | 64.2% (32f) | VideoSwin-T 52.3% | +11.9% |
| K400 (IN-1K Initialization) | Top-1 | 77.7% (32f) | VideoSwin-T 78.8% | Comparable, lower GFLOPs |

### Efficiency Comparison

| Model | GFLOPs | Parameters | GPU Memory |
|------|--------|-------|------|
| VideoMamba (16f) | 34 G | 26.3M | Significantly lower than Transformer |
| VideoSwin-T (32f) | 88 G | 27.8M | - |
| VideoMAE-S (16f) | 57 G | 22.0M | - |
| TimeSformer (8f) | 196 G | 121.4M | - |

### Ablation Study

| Configuration | SSV2 | HMDB | Description |
|------|------|------|------|
| Spatial reversal | 61.9% | 43.3% | Worst, insufficient complementarity |
| Temporal reversal | 63.3% | 52.9% | Medium |
| **Spatio-temporal reversal** | **64.7%** | **55.2%** | **Optimal, full complementarity** |
| Frames 8f → 16f → 32f | 61.0→63.7→64.2 | 52.7→58.9→59.3 | Continuous improvement with more frames |
| Embedding dim 192 → 384 | 54.6→63.7 | 56.5→68.6 | Significant improvement with larger dimensions |

### Key Findings

- Spatio-temporal reversal is the optimal backward scanning strategy; the complementarity of forward and backward token sequences is crucial.
- Positional embeddings are essential for video SSMs (introducing learnable positional embeddings improves HMDB performance by 10.2%), and the initialization method has a substantial impact.
- VideoMamba truly depends on temporal order for reasoning rather than merely treating the video as a collection of static images.
- The visualization of the $\Delta$ parameter reveals a progressive reasoning mode of SSMs from shallow-layer global perception to deep-layer local focus.
- While using only 39% of the GFLOPs of VideoSwin-T, VideoMamba outperforms it by 11.9% on SSV2.

## Highlights & Insights

1. **Systematic Scanning Direction Analysis**: The comparison among three backward scanning strategies provides clear design guidelines—complete spatio-temporal reversal is optimal because it maximizes the complementarity of bidirectional scanning.
2. **$\Delta$ Visualization Analysis**: An in-depth analysis of the $\Delta$ parameter behavior in video SSMs is presented for the first time, revealing a hierarchical reasoning pattern moving from global understanding to local focusing.
3. **Systematic Exploration of Positional Embeddings**: This study provides a comprehensive benchmark and guidance for designing positional embeddings in SSM-based video models.
4. **Temporal Consistency Experiments**: The model's reliance on temporal sequential order, rather than simple appearance recognition, is rigorously validated through frame shuffling experiments.

## Limitations & Future Work

1. Performance on K400 (77.7%) is slightly lower than VideoSwin-T (78.8%), suggesting that pure SSMs might perform worse than local attention on scene-dominant tasks.
2. Only ImageNet-1K pre-training is utilized, leaving the effects of larger-scale pre-training (e.g., IN-21K) unexplored.
3. The fixed tubelet size ($2 \times 16 \times 16$) may not be optimal for all video resolutions and frame rates.
4. Hybrid architectures combining SSM and attention mechanisms have not been explored, which could potentially yield further improvements.
5. The model size is relatively small (only 26M parameters); the scalability of larger models remains to be investigated.

## Related Work & Insights

- **Relationship with OpenGVLab VideoMamba**: There are two papers sharing the same name. This work comes from KAIST, focusing on systematic scanning direction studies and $\Delta$ analysis; the OpenGVLab version focuses on scalability and long-context video understanding.
- **Vision Mamba (Vim)**: This work scales the bidirectional scanning of Vim into the spatio-temporal dimensions.
- **S4ND**: An earlier attempt to employ SSMs in videos, but its performance is limited due to the lack of an input-dependent selection mechanism.
- **Insights**: The analogy of $\Delta$ to attention weights might inspire new interpretability tools; the importance of complementarity in bidirectional scanning can be generalized to other sequence modeling tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The design of Spatio-temporal Forward-Backward SSM is clear, and the systematic comparison of the three backward scans is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablation studies (scanning directions, positional embeddings, frame count, dimension), with deep Delta analysis.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, clearly defined problems, and intuitive visualization analyses.
- Value: ⭐⭐⭐⭐ Provides systematic guidance for designing video SSMs, although its overall impact might be diluted by the work under the same name from OpenGVLab.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] VideoMamba: State Space Model for Efficient Video Understanding](videomamba_state_space_model_for_efficient_video_understanding.md)
- [\[NeurIPS 2025\] PASS: Path-Selective State Space Model for Event-Based Recognition](../../NeurIPS2025/video_understanding/pass_path-selective_state_space_model_for_event-based_recognition.md)
- [\[ECCV 2024\] Spatio-Temporal Proximity-Aware Dual-Path Model for Panoramic Activity Recognition](spatio-temporal_proximity-aware_dual-path_model_for_panoramic_activity_recogniti.md)
- [\[ECCV 2024\] PiTe: Pixel-Temporal Alignment for Large Video-Language Model](pite_pixel-temporal_alignment_for_large_video-language_model.md)
- [\[CVPR 2025\] MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking](../../CVPR2025/video_understanding/mambavlt_time-evolving_multimodal_state_space_model_for_vision-language_tracking.md)

</div>

<!-- RELATED:END -->
