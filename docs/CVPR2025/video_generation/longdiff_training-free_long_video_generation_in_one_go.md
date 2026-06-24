---
title: >-
  [Paper Note] LongDiff: Training-Free Long Video Generation in One Go
description: >-
  [CVPR 2025][Video Generation][Long Video Generation] Through theoretical analysis, LongDiff reveals two key challenges when short-video models generate long videos: temporal position blurring and information dilution. It proposes two simple temporal attention modification strategies, Position Mapping (GROUP+SHIFT) and Informative Frame Selection (IFS), enabling short-video models to generate high-quality long videos in one go without training.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Long Video Generation"
  - "Training-Free Inference"
  - "Position Mapping"
  - "Key Frame Selection"
  - "Temporal Attention"
date: 2026-05-08
content_hash: ec9e0b0df625f3c3
---

# LongDiff: Training-Free Long Video Generation in One Go

**Conference**: CVPR 2025  
**arXiv**: [2503.18150](https://arxiv.org/abs/2503.18150)  
**Code**: None  
**Area**: Diffusion Models / Video Generation  
**Keywords**: Long Video Generation, Training-Free Inference, Position Mapping, Key Frame Selection, Temporal Attention

## TL;DR

Through theoretical analysis, LongDiff reveals two key challenges when short-video models generate long videos: temporal position blurring and information dilution. It proposes two simple temporal attention modification strategies, Position Mapping (GROUP+SHIFT) and Informative Frame Selection (IFS), enabling short-video models to generate high-quality long videos in one go without training.

## Background & Motivation

**Background**: Existing video diffusion models (e.g., LaVie, VideoCrafter) are primarily designed and trained to generate short videos under 16 frames. Directly applying these models to long video generation leads to temporal inconsistency and loss of visual details.

**Limitations of Prior Work**: Training-based long video approaches (autoregressive, hierarchical methods, etc.) require substantial computational resources and scarce long video datasets. Existing training-free methods either rely on sliding windows (e.g., FreeNoise), which restrict long-distance frame interactions and thus degrade global consistency, or blend features in the frequency domain (e.g., FreeLong), yielding limited improvements.

**Key Challenge**: Relative Position Encoding (RPE) in short-video models fails in long sequences—the model cannot distinguish a large number of distinct relative positions, resulting in disordered frame sequences (position blurring). Meanwhile, the lower bound of entropy in temporal attention for long sequences increases as the number of frames grows, leading to reduced effective options and information per frame (information dilution).

**Goal**: (1) How to resolve temporal position blurring in long sequences without training? (2) How to avoid information dilution of visual details in long sequences?

**Key Insight**: Starting from pseudodimension theory and information entropy theory, the analysis reveals that these two challenges can be mitigated through subtle modifications to the temporal Transformer—specifically, by reducing the number of positions that need to be distinguished and restricting the range of frames for information propagation.

**Core Idea**: Map a large number of relative positions to a manageable scale while maintaining distinguishability via the GROUP+SHIFT operation, and restrict each frame to interact only with neighboring frames and key frames via IFS, thereby generating high-quality long videos in one go.

## Method

### Overall Architecture

LongDiff is a training-free method that applies only minor modifications to the temporal attention layers of existing short-video models (e.g., LaVie, VideoCrafter). At each denoising step, when the video latent state passes through the temporal Transformer layers, the relative position matrix is first remapped via Position Mapping, and then the temporal association range of each frame is constrained via the IFS Mask. The input is an extended Gaussian noise sequence (e.g., 128 frames), and the output is a long video generated in one go.

### Key Designs

1. **Position Mapping (PM): GROUP + SHIFT**:

    - **Function**: Resolves the temporal position blurring problem, allowing the model to accurately distinguish the relative order of frames in long sequences.
    - **Mechanism**: Operates in two steps. **GROUP**: Maps $2N-1$ original relative positions to $2G-1$ group indices using the formula $p_g = \lceil p / \lceil(N-1)/(G-1)\rceil \rceil$, compressing the large range of positions into a manageable scale for the model. **SHIFT**: Iteratively shifts the grouped position matrix, downward for lower triangular elements and horizontally for upper triangular elements at each step to maintain anti-symmetry. After $M=S-1$ shifts, each position accumulates a unique "assignment record." Finally, temporal attention is computed separately for the $M+1$ position matrices, and the average of the softmax attention is used as the final output.
    - **Design Motivation**: Theorem 1 proves that the model's ability to distinguish positions is bounded by the supremum of the attention logit, making it impossible to effectively distinguish many positions in long sequences. Simple clipping (clip) discards long-distance position information, while interpolation actually introduces more positions to distinguish. GROUP reduces the total number of positions, and SHIFT restores intra-group distinguishability; their combination balances both global and local context.

2. **Informative Frame Selection (IFS)**:

    - **Function**: Solves the information dilution problem, maintaining visual details in long videos.
    - **Mechanism**: First, the input feature $F \in \mathbb{R}^{N \times C \times hw}$ of each temporal Transformer layer is converted into a pseudo-video $V$ through max/avg/min pooling along the channel dimension, normalized to $[0, 255]$. The pseudo-video is then uniformly divided into $n$ segments, and the frame with the highest combined score of image entropy and frame difference in each segment is selected as a keyframe. Finally, an IFS Mask is constructed so that each frame only attends to its neighboring $L$ frames and all keyframes: $\text{Mask}_{ij} = 1$ when $|i-j| \leq L$ or $j$ is a keyframe.
    - **Design Motivation**: Theorem 2 proves that the lower bound of information entropy is $\ln N - 2B$, wherein the effective information per frame decreases as the frame count $N$ increases. Although fixed window limits preserve details, they hinder long-distance interaction. IFS utilizes keyframes as a global information summary, maintaining global consistency while limiting the amount of information propagated.

3. **Theoretical Foundation (Theorems 1 & 2)**:

    - **Function**: Provides a theoretical foundation for the proposed method design.
    - **Mechanism**: Theorem 1, based on pseudodimension analysis, proves that distinguishing $g(N)$ position groups requires the upper bound of attention logit to satisfy $(g(N)/2)^{1/2r} \cdot \epsilon/4e$; in long sequences, over 60% of frame pairs fail to meet this condition. Theorem 2, based on information entropy analysis, proves that the lower bound of attention weight entropy scales with $N$, causing information dilution.
    - **Design Motivation**: Theoretical analysis pinpointed the exact problems—the limited representation capacity of position encoding and the information dissipation of fully connected attention—providing targeted guidance for the designs of PM and IFS.

### Loss & Training

LongDiff is a completely training-free, inference-time method that does not involve any training or loss function. PM and IFS directly modify the computation process of temporal attention. The $M$ SHIFT operations can be computed in parallel, introducing only a minor deduction in inference speed.

## Key Experimental Results

### Main Results

Generating 128-frame videos on LaVie:

| Method | SC ↑ | BC ↑ | MS ↑ | TF ↑ | IQ ↑ | OC ↑ |
|------|------|------|------|------|------|------|
| Direct | 88.95 | 93.23 | 92.77 | 91.44 | 64.76 | 22.34 |
| FreeNoise | 92.30 | 95.87 | 96.32 | 94.94 | 67.14 | 24.42 |
| FreeLong | 95.16 | 96.80 | 96.85 | 96.04 | 67.55 | 24.56 |
| **LongDiff** | **98.10** | **98.23** | **97.46** | **96.84** | **68.83** | **25.24** |

On VideoCrafter-512:

| Method | SC ↑ | BC ↑ | MS ↑ | TF ↑ | IQ ↑ | OC ↑ |
|------|------|------|------|------|------|------|
| FreeNoise | 91.43 | 93.48 | 93.33 | 91.88 | 68.39 | 22.69 |
| FreeLong | 90.84 | 92.37 | 89.11 | 88.46 | 66.62 | 21.85 |
| **LongDiff** | **93.69** | **95.59** | **94.59** | **93.35** | **70.03** | **23.17** |

### Ablation Study

| Configuration | SC ↑ | BC ↑ | MS ↑ | TF ↑ | IQ ↑ | OC ↑ |
|------|------|------|------|------|------|------|
| w/o PM | 91.85 | 94.79 | 95.12 | 93.26 | 65.73 | 22.58 |
| w/o IFS | 94.43 | 96.37 | 93.65 | 92.85 | 65.45 | 23.46 |
| Clip (Alternative to PM) | 91.02 | 94.18 | 94.77 | 92.87 | 65.12 | 22.43 |
| Interpolation | 92.52 | 95.29 | 95.25 | 94.76 | 66.43 | 23.01 |
| Group only (w/o SHIFT) | 94.49 | 96.63 | 96.74 | 95.79 | 67.45 | 24.62 |
| **Full LongDiff** | **98.10** | **98.23** | **97.46** | **96.84** | **68.83** | **25.24** |

### Key Findings

- PM and IFS are both indispensable: removing PM drops SC from 98.10 to 91.85 (a substantial decline in temporal consistency), while removing IFS drops IQ from 68.83 to 65.45 (loss of visual details).
- Importance of the SHIFT operation: the performance of GROUP alone is far inferior to GROUP+SHIFT, demonstrating that the restoration of intra-group position distinguishability is crucial.
- Consistent and comprehensive lead is achieved on two models with distinct architectures (LaVie/RoPE, VideoCrafter/RPE), showing the strong generalization capability of the method.
- Compared to simple position handling such as clipping and interpolation, PM exhibits a highly significant advantage.

## Highlights & Insights

- **Theory-driven Method Design**: Pinpoints the problems from two theoretical tools, pseudodimension and information entropy, and subsequently designs targeted solutions. This paradigm of "theoretical analysis prior to method design" is highly exemplary and represents a rare theory-driven contribution in the video generation domain.
- **Ingenious Combination of GROUP+SHIFT**: GROUP scales down the total number of positions to fit the model's limited expressive capability, while SHIFT restores distinguishability by accumulating "assignment records," which is eventually leveraged through the average of multiple attention calculations. This strategy can be transferred to any scenario involving long-sequence position encodings (e.g., position extrapolation in long-text generation).
- **Completely Training-Free & Plug-and-Play**: As an inference-time method, LongDiff can be directly applied to any short-video model utilizing relative position encoding, requiring no extra data or training resources, making it highly practical.

## Limitations & Future Work

- Only verified on models with 3D U-Net architectures; has not yet been evaluated on newer DiT architecture models.
- A "long video" of 128 frames (approx. 8 seconds) remains relatively short for practical applications; its efficacy on even longer sequences warrants further validation.
- Keyframe detection is based on image entropy and frame differences, which is relatively simple and may be inaccurate in complex scenarios.
- The SHIFT operation requires $M+1$ attention calculations; although parallelizable, it still increases the computational overhead.

## Related Work & Insights

- **vs. FreeNoise**: FreeNoise employs sliding-window temporal attention, restricting interactions between distant frames. LongDiff maintains global information exchange through the keyframe mechanism of IFS, achieving a comprehensive lead in all metrics.
- **vs. FreeLong**: FreeLong mixes spatial and temporal features in the frequency domain. LongDiff addresses the issue from two fundamental perspectives: position encoding in temporal attention and information propagation, yielding better results and a more solid theoretical foundation.
- **Connection to NTK-aware Scaling (LLM)**: The Position Mapping idea of LongDiff shares similarities with position encoding extrapolation methods in the LLM domain (such as NTK-aware RoPE scaling); both address the position encoding challenge of how to generalize "short sequences trained models to long sequences."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Theory-driven + ingenious GROUP/SHIFT design, with an innovative keyframe selection in IFS.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison and detailed ablation on two models, but lacks validation on longer sequences and DiT models.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivations and excellent method visualization.
- Value: ⭐⭐⭐⭐⭐ Training-free, plug-and-play, with both stellar theoretical insights and practical effectiveness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] One-Minute Video Generation with Test-Time Training](one-minute_video_generation_with_test-time_training.md)
- [\[ICML 2025\] Diffusion Adversarial Post-Training for One-Step Video Generation](../../ICML2025/video_generation/diffusion_adversarial_post-training_for_one-step_video_generation.md)
- [\[CVPR 2025\] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](streamingt2v_consistent_dynamic_and_extendable_long_video_generation_from_text.md)
- [\[CVPR 2025\] MovieBench: A Hierarchical Movie Level Dataset for Long Video Generation](moviebench_a_hierarchical_movie_level_dataset_for_long_video_generation.md)
- [\[CVPR 2025\] Presto: Long Video Diffusion Generation with Segmented Cross-Attention and Content-Rich Video Data Curation](long_video_diffusion_generation_with_segmented_cross-attention_and_content-rich_.md)

</div>

<!-- RELATED:END -->
