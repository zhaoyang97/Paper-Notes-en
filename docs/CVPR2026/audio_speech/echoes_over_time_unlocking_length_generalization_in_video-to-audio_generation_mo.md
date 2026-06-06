---
title: >-
  [Paper Note] Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models
description: >-
  [CVPR 2026][Audio & Speech][Video-to-Audio generation] This paper proposes MMHNet, a Multimodal Hierarchical Network based on a hierarchical architecture and non-causal Mamba-2…
tags:
  - "CVPR 2026"
  - "Audio & Speech"
  - "Video-to-Audio generation"
  - "long-sequence generation"
  - "hierarchical network"
  - "Mamba"
  - "multimodal alignment"
date: 2026-05-08
content_hash: f30f1068c7dc3028
---

# Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models

**Conference**: CVPR 2026
**arXiv**: [2602.20981](https://arxiv.org/abs/2602.20981)  
**Code**: None (Project page: [https://echoesovertime.github.io](https://echoesovertime.github.io))  
**Area**: Speech/Audio
**Keywords**: Video-to-Audio generation, long-sequence generation, hierarchical network, Mamba, multimodal alignment

## TL;DR
This paper proposes MMHNet, a Multimodal Hierarchical Network based on a hierarchical architecture and non-causal Mamba-2, achieving length generalization by training on short clips (8 seconds) while generating high-quality, well-aligned audio for long videos (5+ minutes). MMHNet substantially outperforms existing methods on the UnAV100 and LongVale benchmarks.

## Background & Motivation
Video-to-Audio (V2A) generation aims to synthesize semantically and temporally aligned audio for silent videos, with significant applications in filmmaking and game development. Existing V2A methods (e.g., MMAudio, Diff-Foley) are primarily optimized for short audio generation (8–10 seconds) and fail to generalize effectively to long video scenarios.

The **root cause** lies in three compounding issues: (1) long audio-video training data is scarce, with public datasets typically capped at one minute; (2) Transformer architectures rely on positional encodings (e.g., RoPE), causing sharp performance degradation when inference sequence length exceeds training length; (3) naive segmentation-and-concatenation approaches result in fragmented audio, unnatural transitions, and degraded audio quality.

The paper identifies that **explicit positional encodings** are the fundamental bottleneck — effective when training length is fixed, but a limiting factor for length generalization. Experiments show that removing positional encodings from MMAudio causes the generated sounds to become homogeneous, while retaining them leads to quality degradation on long sequences (FD_PANN drops by 3–4 points). The **core idea** is therefore to replace Transformer attention modules with **Mamba-2, which requires no positional encodings**, combined with **hierarchical token routing** for efficient long-sequence processing.

## Method

### Overall Architecture
MMHNet extends MMAudio's multimodal DiT architecture, comprising multimodal blocks (processing joint audio, visual, and text information) and single-modal blocks (processing audio only). A conditional velocity field is modeled in the compressed latent space via flow matching, and audio is generated using an ODE solver. Key innovations include: (1) replacing attention modules with non-causal Mamba-2; (2) introducing a hierarchical framework with temporal routing and multimodal routing; and (3) dynamic chunking and upsampling for token compression and reconstruction.

### Key Designs
1. **Non-Causal Mamba-2 as the Core Network**: Replaces Transformer attention modules, thereby eliminating dependence on positional encodings entirely. Causal Mamba-2 models the mask matrix via cumulative products, which introduces modulation decay in long sequences. Non-causal Mamba-2 instead defines the mask as the inverse of the transformation matrix, avoiding decay from cumulative products and enabling bidirectional information flow. This allows the model to handle arbitrary-length sequences at inference without architectural modification. Compared to causal Mamba, the non-causal variant allows a global hidden state to simultaneously fuse all modalities without constraint from scan order, making it better suited for multimodal fusion under offline video conditions.

2. **Temporal Routing Layer**: Audio and video events contain substantial redundancy (e.g., similar frames and sound events within the same temporal segment). Temporal routing identifies change boundaries by computing cosine similarity between adjacent tokens. Tokens with high similarity are masked (indicating redundancy), while those with low similarity are retained (indicating temporal boundaries or event transitions). This effectively filters redundant temporal information and reduces computational complexity.

3. **Multimodal Routing Layer (MM Routing)**: Selects tokens highly correlated with a reference modality for forward propagation. Only tokens with similarity $\geq 0.5$ are selected for processing. For example, audio-visual synchronization features from Synchformer can be aligned with text conditioning. This improves alignment efficiency by attending only to cross-modal highly relevant tokens.

4. **Hierarchical Chunking and Upsampling**: The downsampler compresses encoder outputs into fewer vectors by selecting tokens at boundary positions identified by boundary indicators. Processed tokens are restored to the original resolution by an upsampler, using the Straight-Through Estimator (STE) to allow gradients to flow through the selection operation. Earlier layers operate in the compressed space for multimodal alignment, while later layers process fine-grained details in the original space.

### Loss & Training
The model is trained with a conditional flow matching objective on the VGGSound dataset using 8-second clips, and directly generalizes to arbitrary lengths at inference. The small model (S) uses $N=5$ multimodal blocks and $N'=4$ single-modal blocks (157M parameters); the large model (L) uses $N=10$ and $N'=7$ (1.09B parameters).

## Key Experimental Results

### Main Results

| Dataset | Metric | MMHNet-S | MMHNet-L | MMAudio-L | LoVA | HunyuanVideo-Foley |
|---------|--------|----------|----------|-----------|------|-------------------|
| UnAV100 | FD_PANNs ↓ | **5.87** | 5.29 | 9.01 | 7.50 | 10.28 |
| UnAV100 | IB-Score ↑ | **36.82** | 36.27 | 30.71 | 24.62 | 32.90 |
| UnAV100 | DeSync ↓ | 0.439 | **0.410** | 0.593 | 1.232 | 0.757 |
| LongVale | FD_PANNs ↓ | 10.10 | **10.03** | 16.12 | 21.81 | 28.00 |
| LongVale | IB-Score ↑ | **30.62** | 30.00 | 21.60 | 17.04 | 18.75 |
| LongVale | DeSync ↓ | **0.438** | 0.465 | 0.678 | 1.233 | 1.082 |

### Ablation Study

| Configuration | FD_PANNs ↓ | IB-Score ↑ | DeSync ↓ | Notes |
|---------------|-----------|-----------|---------|-------|
| Transformer (no positional encoding) | 9.00 | 28.41 | 0.638 | Baseline; loses temporal structure |
| Causal Mamba-2 | 9.18 | 33.32 | 0.497 | Constrained by scan direction |
| **Non-Causal Mamba-2** | **5.87** | **36.82** | **0.439** | Bidirectional information flow; best overall |
| Non-hierarchical (UnAV100) | 6.31 | 35.00 | 0.621 | No token compression |
| **Hierarchical** (UnAV100) | **5.87** | **36.82** | **0.439** | Routing compression yields consistent gains |

### Key Findings
- Non-causal Mamba-2 significantly outperforms both the Transformer and causal Mamba-2 across all metrics, particularly in long-video multimodal alignment (IB-Score improves by 8+ points).
- Hierarchical token routing yields consistent improvements, with more pronounced gains on LongVale (IB-Score from 26.34 to 30.62).
- The token selection threshold of 0.5 is optimal; an overly high threshold (0.7) causes catastrophic failure.
- The autoregressive method (V-AURA) performs worst on length generalization, validating that error accumulation from step-by-step prediction is a fundamental limitation.
- On VGGSound (same-length train and test), MMHNet achieves performance on par with MMAudio, demonstrating that length generalization does not sacrifice short-clip quality.

## Highlights & Insights
- **Train-short, test-long paradigm**: Training exclusively on 8-second clips enables generation of high-quality audio exceeding 5 minutes.
- **Non-causal Mamba-2 as a replacement for positional encodings**: The approach is transferable to other sequence generation tasks requiring length generalization.
- **Hierarchical routing for token compression**: Temporal and multimodal routing selectively retains informative tokens, simultaneously reducing computational cost and improving alignment quality.
- **Evaluation methodology innovation**: Multi-segment chunked evaluation is adopted for long audio, circumventing the inability of pretrained classifiers to process full-length long audio.

## Limitations & Future Work
- Generation quality depends on the quality of pretrained conditioning features (CLIP, Synchformer).
- Validation is limited to audio-video scenarios; applicability to other long-sequence multimodal generation tasks remains to be explored.
- The fixed routing threshold (0.5) could be further optimized with an adaptive thresholding mechanism.

## Related Work & Insights
- **vs. MMAudio**: MMHNet directly extends MMAudio by replacing the Transformer with Mamba-2 to achieve length generalization.
- **vs. LoVA**: LoVA represents the prior SOTA in the long V2A domain, but exhibits notable quality degradation beyond one minute.
- **vs. V-AURA**: Although autoregressive methods can theoretically handle arbitrary lengths, error accumulation results in the worst empirical performance.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic study of length generalization in V2A; the combination of non-causal Mamba and hierarchical routing is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Two long-video benchmarks plus VGGSound; multi-dimensional ablations and cross-length analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Pilot experiments provide clear motivation; architectural descriptions are thorough.
- **Value**: ⭐⭐⭐⭐ Addresses a practical bottleneck in V2A length generalization with direct applications to film and game audio synthesis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text](omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text.md)
- [\[CVPR 2026\] Unlocking Strong Supervision: A Data-Centric Study of General-Purpose Audio Pre-Training Methods](unlocking_strong_supervision_a_data-centric_study_of_general-purpose_audio_pre-t.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](../../NeurIPS2025/audio_speech/node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)
- [\[ACL 2026\] MARQUIS: A Three-Stage Pipeline for Video Retrieval-Augmented Generation](../../ACL2026/audio_speech/marquis_a_three-stage_pipeline_for_video_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
