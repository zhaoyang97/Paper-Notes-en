---
title: >-
  [Paper Note] Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval
description: >-
  [CVPR 2025][Video Understanding][Video-Text Retrieval] The AVIGATE framework is proposed to selectively fuse audio and visual information through a gated attention mechanism (filtering out useless audio noise) and to design an adaptive margin contrastive loss to handle ambiguous positive-negative relationships between videos and texts, achieving state-of-the-art (SOTA) performance on multiple video-text retrieval benchmarks.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Video-Text Retrieval"
  - "Audio Guidance"
  - "Gated Attention"
  - "Contrastive Learning"
  - "Multi-Granularity Alignment"
date: 2026-05-08
content_hash: 71b7defc51d90bdc
---

# Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval

**Conference**: CVPR 2025  
**arXiv**: [2504.02397](https://arxiv.org/abs/2504.02397)  
**Code**: [https://github.com/](https://github.com/) (Project page: [http://cvlab.postech.ac.kr/research/AVIGATE](http://cvlab.postech.ac.kr/research/AVIGATE))  
**Area**: Video Understanding / Multimodal  
**Keywords**: Video-Text Retrieval, Audio Guidance, Gated Attention, Contrastive Learning, Multi-Granularity Alignment

## TL;DR

The AVIGATE framework is proposed to selectively fuse audio and visual information through a gated attention mechanism (filtering out useless audio noise) and to design an adaptive margin contrastive loss to handle ambiguous positive-negative relationships between videos and texts, achieving state-of-the-art (SOTA) performance on multiple video-text retrieval benchmarks.

## Background & Motivation

**Background**: Video-text retrieval primarily relies on visual and textual features for cross-modal alignment. Numerous methods construct multi-granularity matching schemes (such as CLIP4Clip, X-Pool, UATVR, etc.) based on CLIP pre-trained features to continuously improve retrieval accuracy.

**Limitations of Prior Work**: Audio signals in videos (such as speaker identity, background sounds, emotional cues, etc.) are ignored by most approaches. A few methods leveraging audio (e.g., ECLIPSE, TEFAL) blindly assume that audio is always helpful. However, in reality, the audio of many videos consists of irrelevant background music or noise, and blind fusion can instead degrade the quality of video representations. Furthermore, TEFAL requires joint processing of text and video/audio, making it highly inefficient as the entire database must be reprocessed for every new query during retrieval.

**Key Challenge**: While the audio modality is indeed valuable for video understanding (e.g., conversation content, environmental sounds), the audio in all videos is not always useful—there is a distinction between "informative audio vs. noisy audio," and existing methods lack dynamic assessment of audio quality.

**Goal**: (1) To dynamically assess whether audio is useful and selectively fuse it; (2) to better handle semantically close negative sample pairs in contrastive learning.

**Key Insight**: The authors observe that the correlation between visual and audio modalities varies drastically across different videos. Thus, a gating mechanism can be designed to "automatically adjust the contribution weight of audio"—amplifying its impact when the audio is informative and closing its channel when the audio is noisy.

**Core Idea**: Use a gated attention mechanism to dynamically filter audio noise, and employ an adaptive margin-based contrastive loss to perceive semantic proximity between negative samples, achieving efficient and accurate video-text retrieval.

## Method

### Overall Architecture

AVIGATE uses three independent encoders to handle the three modalities: a CLIP image encoder to extract frame-level embeddings, a CLIP text encoder to extract text embeddings, and an AST (Audio Spectrogram Transformer) to extract audio embeddings. The audio embeddings are compressed to a fixed number of tokens via an audio resampler, then input to a Gated Fusion Transformer along with the frame embeddings to output the final video representation. Finally, the similarity is calculated with text embeddings using a multi-granularity alignment scheme.

### Key Designs

1. **Audio Resampler**:

    - **Function**: Compresses the dense audio embeddings output by AST into a fixed length of $M$ embeddings.
    - **Mechanism**: Uses a query-based Transformer where $M$ learnable query embeddings extract information from the raw audio embeddings through a cross-attention mechanism. This design preserves key audio features while heavily reducing the computational cost of subsequent fusion. AST parameters are frozen during training.
    - **Design Motivation**: Because the audio sampling rate is much higher than the video frame rate, direct fusion of all audio tokens yields excessive computational overhead, necessitating an information compression layer.

2. **Gated Fusion Transformer**:

    - **Function**: Selectively fuses audio embeddings with frame embeddings, dynamically determining the degree of audio contribution.
    - **Mechanism**: Consists of $L$ layers of Gated Fusion Blocks. Each layer contains a "fusion process" and a "refinement process." In the fusion process, frame embeddings act as queries, while audio embeddings act as keys/values in multi-head cross-attention. The output is multiplied by a gating score $g_{\text{mha}}$ and added via residual connection; it is then processed by an FFN and multiplied by $g_{\text{ffn}}$. In the refinement process, self-attention is used to further enhance inter-frame relationships. Gating scores are generated by a Gating Function: audio and frame embeddings are average-pooled separately, concatenated, and passed through two independent MLPs followed by tanh activation to yield two scalar gating values. High gating scores emphasize the audio contribution, while low scores shield the visual content from noise interference.
    - **Design Motivation**: Unlike the static fusion in ECLIPSE/TEFAL, the gating mechanism enables the model to learn "when to listen to audio and when to ignore it," resolving the core issue of noisy audio interference.

3. **Adaptive Margin-based Contrastive Loss**:

    - **Function**: Dynamically configures different margins for each pair of negative samples during contrastive learning.
    - **Mechanism**: For each negative pair $(V_i, T_j)$, the intra-modal visual similarity $c_{ij}^v$ and intra-modal textual similarity $c_{ij}^t$ are calculated. The adaptive margin is then set as $m_{ij} = \min(\lambda(1 - (c_{ij}^v + c_{ij}^t)/2), \delta)$. For negative samples that are semantically less similar, the margin is greater, forcing the model to push them far apart; for semantically similar negative samples, the margin is smaller, preventing generalization degradation caused by over-squeezing.
    - **Design Motivation**: Traditional contrastive losses treat all negative samples equally. However, in practice, many "negative samples" share semantic correlations (such as two videos describing similar scenes). A fixed margin forces these semantically related samples to be pushed far apart, which harms generalization.

### Loss & Training

The final loss is a bidirectional contrastive loss (video-to-text + text-to-video), where an adaptive margin $m_{ij}$ is added to the negative sample similarity in both directions. The similarity score is computed via a multi-granularity alignment scheme: global alignment (cosine similarity between average-pooled frame embeddings and text) + local alignment (aggregation of per-frame similarity with text using log-sum-exp).

## Key Experimental Results

### Main Results

| Dataset | Metric | AVIGATE | Prev. SOTA (UATVR) | Gain |
|--------|------|---------|-----------|------|
| MSR-VTT (ViT-B/16) | T2V R@1 | **52.1** | 50.8 | +1.3 |
| MSR-VTT (ViT-B/16) | V2T R@1 | **51.2** | 48.1 | +3.1 |
| MSR-VTT (ViT-B/16) | RSum | **429.0** | 422.4 | +6.6 |
| MSR-VTT (ViT-B/32) | T2V R@1 | **50.2** | 47.5 | +2.7 |
| MSR-VTT (ViT-B/32) | V2T R@1 | **49.7** | 46.9 | +2.8 |

### Ablation Study

| Configuration | RSum (MSR-VTT) | Description |
|------|----------------|------|
| Full model (AVIGATE) | **429.0** | Full model |
| w/o audio | ~422 | Removes audio fusion, degrading to a vision-text only model |
| w/o gating (blind fusion) | ~420 | Removes gating; blind fusion of audio degrades performance instead |
| w/o adaptive margin | ~425 | Uses fixed-margin contrastive loss, degrading performance |

### Key Findings

- The gating mechanism is key: blind fusion of audio performs worse than the vision-only model; the gating mechanism ensures that audio fusion brings positive gains.
- The adaptive margin outperforms both fixed margin and no margin, showing a more pronounced advantage on datasets with highly semantically similar samples.
- AVIGATE maintains high retrieval efficiency—video and text are encoded independently, avoiding the need to re-run the entire database for each query as in TEFAL.

## Highlights & Insights

- **The design of Gated Fusion is highly ingenious**: Adjusting the audio impact using a scalar gating value activated by tanh simply and effectively solves the problem of determining "when audio is useful." This intuition can be transferred to any multimodal fusion scenario where "a certain modality is not always useful."
- **Core insight of the adaptive margin**: Intra-modal similarity can serve as a proxy signal for cross-modal semantic correlation—visually similar videos tend to have semantically similar corresponding text. This prior provides a solid rationale for the margin design.
- **The combination of multi-granularity alignment + independent encoding** ensures a good balance between retrieval efficiency and accuracy, which is crucial for real-world deployment.

## Limitations & Future Work

- Validated only on retrieval tasks, without testing the effectiveness of audio gated fusion on other downstream tasks such as video captioning or video QA.
- The gating score is a global scalar, lacking the ability to make fine-grained selections across different temporal segments or frequency bands of the audio.
- The audio encoder (AST) is frozen during training, which may prevent full alignment of audio representations with the CLIP space.
- Deeper direct audio-text interactions have not been explored (in this work, audio is only fused with vision and is not directly matched with text).

## Related Work & Insights

- **vs. ECLIPSE**: ECLIPSE uses cross-attention to fuse audio-visual modalities but lacks a gating mechanism to filter noisy audio. The gating design in this work is the core difference.
- **vs. TEFAL**: TEFAL requires joint video-text processing to generate representations, resulting in low retrieval efficiency; AVIGATE encodes video and text independently, yielding higher efficiency.
- **vs. UATVR**: UATVR handles uncertainty through distribution matching but does not use audio; AVIGATE builds on UATVR by introducing the audio modality and achieves further improvements.

## Rating

- Novelty: ⭐⭐⭐⭐ Both gated fusion and adaptive margin have innovations, but the overall framework is somewhat incremental.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation across multiple benchmarks and ablation studies is relatively comprehensive, but lacks visualization analysis of the gating scores.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and intuitive diagrams.
- Value: ⭐⭐⭐⭐ The gated fusion concept has practical value and is transferable to other multimodal scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SEAL: SEmantic Attention Learning for Long Video Representation](seal_semantic_attention_learning_for_long_video_representation.md)
- [\[ICLR 2026\] OmniCVR: A Benchmark for Omni-Composed Video Retrieval with Vision, Audio, and Text](../../ICLR2026/video_understanding/omnicvr_a_benchmark_for_omni-composed_video_retrieval_with_vision_audio_and_text.md)
- [\[CVPR 2025\] Heterogeneous Skeleton-Based Action Representation Learning](heterogeneous_skeleton-based_action_representation_learning.md)
- [\[ECCV 2024\] Text-Guided Video Masked Autoencoder](../../ECCV2024/video_understanding/text-guided_video_masked_autoencoder.md)
- [\[CVPR 2025\] H-MoRe: Learning Human-centric Motion Representation for Action Analysis](h-more_learning_human-centric_motion_representation_for_action_analysis.md)

</div>

<!-- RELATED:END -->
