---
title: >-
  [Paper Note] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval
description: >-
  [Audio & Speech] This paper proposes SAVE, a speech-aware video representation learning method that introduces a dedicated speech branch (Whisper ASR + CLIP text encoder) and a soft-ALBEF visual-audio early alignment strategy, achieving comprehensive state-of-the-art performance across five video-text retrieval benchmarks.
tags:
  - "Audio & Speech"
date: 2026-05-08
content_hash: 6fbae214ca89cd90
---

# SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval

| Info | Content |
|------|------|
| **Conference** | CVPR 2026 |
| **arXiv** | [2603.08224](https://arxiv.org/abs/2603.08224) |
| **Area** | Human Understanding |
| **Keywords** | video-text retrieval, speech awareness, audio-visual fusion, soft-ALBEF, multimodal learning |

## TL;DR

This paper proposes SAVE, a speech-aware video representation learning method that introduces a dedicated speech branch (Whisper ASR + CLIP text encoder) and a soft-ALBEF visual-audio early alignment strategy, achieving comprehensive state-of-the-art performance across five video-text retrieval benchmarks.

## Background & Motivation

Video-text retrieval (VTR) methods commonly adopt CLIP as the backbone; however, since CLIP provides only image and text encoders, existing approaches naturally neglect the audio track of videos. Recent audio-visual methods (EclipSE, TEFAL, AVIGATE) incorporate audio encoders but suffer from two critical issues:

**Audio encoders fail to represent speech content effectively**: Existing audio encoders (ResNet-18, AST) are trained on environmental sound datasets and encode speech semantics poorly. The authors demonstrate this through an experiment showing that speech samples of different categories are completely intermixed in AST's feature space and thus indistinguishable.

**Lack of alignment prior to visual-audio fusion**: Visual features (CLIP image encoder) and audio features (AST) are never pre-aligned before fusion, which limits the effectiveness of direct fusion. Although ALBEF (align before fuse) has proven successful in vision-language pre-training, video-audio pairs often lack semantic correspondence (e.g., background music unrelated to video content), making direct application of hard ALBEF prone to introducing spurious correlations.

## Method

### Overall Architecture: Three-Branch Network

SAVE extends AVIGATE's dual-branch design (visual + audio) into a three-branch architecture:

1. **Visual branch**: CLIP ViT-B/32 extracts frame features $\{v_i\}$
2. **Audio branch**: AST (frozen) extracts audio tokens, which are resampled and fused with visual tokens via Gated-Fusion to produce $\{\hat{a}_i\}$
3. **Speech branch (new)**: Whisper large-v3 → ASR text → CLIP text encoder → speech tokens $\{s_i\}$, further processed via Gated-Fusion to yield $\{\hat{s}_i\}$

The final speech-aware video representation is: $\{\tilde{v}_i\} = \{v_i\} + (\{\hat{a}_i\} + \{\hat{s}_i\})/2$

**Design Motivation**: The visual branch is treated as the primary signal (higher weight), while speech and audio are weighted equally (lacking prior knowledge). This simple fusion encourages Gated-Fusion to learn truly informative signals.

### Soft-ALBEF Early Alignment

The key innovation is using ImageBind to compute a video-audio affinity matrix $M_0$ as soft labels, replacing the hard labels used in ALBEF.

$$\ell_{\text{pearson}} = \frac{1}{b}\sum_{i=1}^{b} d_p(\sigma(M_0[i,\cdot]), \sigma(M_1[i,\cdot])) + \frac{1}{b}\sum_{j=1}^{b} d_p(\sigma(M_0[\cdot,j]), \sigma(M_1[\cdot,j]))$$

where $M_1$ is the affinity matrix produced by the current network, and $d_p$ denotes the Pearson distance. Pearson distance is preferred over MSE/Huber due to its invariance to scale and shift, allowing the network to focus on learning relative ranking structure.

### Handling Missing Data

- No audio track: Mel filterbank is set to zero
- ASR failure: An empty string is used; the tokenizer pads it to a zero vector

### Training Details

- The Pearson distance loss serves as an auxiliary objective combined with equal weight alongside AVIGATE's adaptive-margin contrastive loss
- Backbone (CLIP) fine-tuning learning rate: 1e-7; other modules: 1e-4 (to prevent catastrophic forgetting)
- 8× RTX 3090 GPUs

## Key Experimental Results

### Main Results: Text-to-Video Retrieval SumR

| Method | MSRVTT-9k | MSRVTT-7k | VATEX | Charades | LSMDC | mR1 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| CLIP4Clip | 197.5 | 150.1 | 248.5 | 107.6 | 112.7 | 35.1 |
| PIG | 203.0 | 157.1 | 252.1 | - | - | - |
| AVIGATE | 207.7 | 162.7 | 249.3 | 110.6 | 125.7 | 37.9 |
| **SAVE** | **216.2** | **165.8** | **255.5** | **121.4** | **128.3** | **39.6** |

SumR gains of SAVE over AVIGATE: MSRVTT-9k +8.5, VATEX +6.2, Charades +10.8.

### Group Analysis (MSRVTT-9k)

| Group | SAVE vs. AVIGATE SumR Difference |
|------|:---:|
| Visually relevant (499 cases) | Positive gain |
| Sound-relevant (226 cases) | +11.5 |
| Speech-relevant (171 cases) | +12.9 |
| Sound + speech relevant (104 cases) | **+16.4** |

### Efficiency Analysis

| Method | Computational Complexity | Inference Time | SumR |
|------|:---:|:---:|:---:|
| TEFAL | $O(n_{\mathcal{A}} n_{\mathcal{T}} + n_{\mathcal{V}} n_{\mathcal{T}})$ | 140.57ms | 209.2 |
| AVIGATE | $O(n_{\mathcal{A}} + n_{\mathcal{V}} + n_{\mathcal{T}})$ | 9.90ms | 207.7 |
| **SAVE** | $O(n_{\mathcal{S}} + n_{\mathcal{A}} + n_{\mathcal{V}} + n_{\mathcal{T}})$ | **9.90ms** | **216.2** |

SAVE maintains the same inference latency as AVIGATE (9.90ms), since video features can be extracted offline.

### Ablation Study: Speech Branch vs. Audio Branch

- Without the speech branch: SumR −4.3
- Without the audio branch: SumR −8.7
- Both branches contribute; the audio branch has a larger impact because sound-relevant queries are more prevalent in the datasets.

## Highlights & Insights

1. **Precise problem identification**: A toy experiment directly demonstrates AST's clustering failure in the speech feature space, providing highly convincing motivation.
2. **Elegant speech branch design**: The Whisper ASR → CLIP text encoder pipeline cleverly leverages CLIP's text-visual alignment capability to encode speech content.
3. **Strong generalizability of soft-ALBEF**: Using ImageBind to provide noise-tolerant soft supervision addresses the fundamental issue of missing semantic correspondence in visual-audio pairs.
4. **Zero additional inference cost**: All newly introduced computations can be performed offline.
5. **Remarkable gains on Charades**: Even though only 13.5% of videos contain ASR text, SumR still improves by 10.8, demonstrating that soft-ALBEF effectively leverages the audio modality.

## Limitations & Future Work

- Validated only on short video clips; ASR transcripts in long videos (e.g., e-commerce livestreams) tend to be longer and noisier.
- Dependent on Whisper's ASR quality; performance may vary for non-English languages.
- Uses ViT-B/32 without exploring larger backbones due to GPU budget constraints.
- Employing ImageBind for soft-ALBEF introduces additional offline computational cost.
- Limited improvement headroom for entirely silent videos.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LeVo: High-Quality Song Generation with Multi-Preference Alignment](../../NeurIPS2025/audio_speech/levo_high-quality_song_generation_with_multi-processing_refined_supervision.md)
- [\[CVPR 2026\] OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text](omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](../../NeurIPS2025/audio_speech/node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)
- [\[ICLR 2026\] EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning](../../ICLR2026/audio_speech/emotionthinker_prosody-aware_reinforcement_learning_for_explainable_speech_emoti.md)
- [\[ACL 2026\] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](../../ACL2026/audio_speech/learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)

</div>

<!-- RELATED:END -->
