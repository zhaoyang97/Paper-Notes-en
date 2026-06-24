---
title: >-
  [Paper Note] Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues
description: >-
  [ACL 2025][Multimodal VLM][Nonverbal communication] This paper proposes VENUS—the first large-scale multimodal dialogue dataset (89,459 dialogues, 14,910 hours) containing temporally aligned text, 3D facial expressions, and body language annotations. Based on this dataset, the authors develop the MARS multimodal language model, which discretizes nonverbal cues using VQ-VAEs to unify them with text in a single autoregressive framework, enabling joint understanding and generati…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Nonverbal communication"
  - "multimodal dialogue"
  - "facial expressions"
  - "body language"
  - "vector quantization"
date: 2026-05-08
content_hash: 2a68951281cc902d
---

# Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues

**Conference**: ACL 2025  
**arXiv**: [2506.00958](https://arxiv.org/abs/2506.00958)  
**Area**: Multimodal VLM  
**Keywords**: Nonverbal communication, multimodal dialogue, facial expressions, body language, vector quantization  

## TL;DR

This paper proposes VENUS—the first large-scale multimodal dialogue dataset (89,459 dialogues, 14,910 hours) containing temporally aligned text, 3D facial expressions, and body language annotations. Based on this dataset, the authors develop the MARS multimodal language model, which discretizes nonverbal cues using VQ-VAEs to unify them with text in a single autoregressive framework, enabling joint understanding and generation of text and nonverbal actions in dialogues.

## Background & Motivation

- Human dialogue is a complex interaction of verbal and nonverbal signals—facial expressions, gestures, and body language convey emotions and intentions.
- For example, "Do you know what time it is?" paired with different expressions conveys entirely different meanings (neutral = inquiring vs. frowning + crossed arms = blaming).
- Existing LLMs are limited to text, neglecting nonverbal communication; a few works focus only on facial expressions while ignoring body language.
- **Key Challenge**: The lack of large-scale training datasets that simultaneously annotate text, facial expressions, and body language.
- Existing datasets are either small in scale or lack nonverbal annotations (e.g., YTD-18M has videos but no 3D annotations; BEAT/EMAGE have 3D but lack dialogue scenarios).

## Method

### Overall Architecture

**The VENUS Dataset Construction Pipeline** (automatically extracted from YouTube podcast videos):

1. **Data Collection & Filtering**:
    - Download YouTube podcast videos (869 channels, 27,128 videos).
    - Filter via thumbnail face detection (F1) → remove the first 1 minute (P1) → segment into 10-minute clips (P1 & F2, FPS=25).

2. **Audio Processing**:
    - PyAnnote voice segmentation: retain only videos containing exactly two speakers (F3).
    - WhisperX language detection (retaining only English, F4) + temporally aligned speech transcription (P2).

3. **Visual Speaker Identification**:
    - Light-ASD active speaker detection (P3) + person detector to crop speaker images.
    - MobileNet feature extraction + cosine similarity to align speaker identities across frames (P4).

4. **Nonverbal Cue Extraction**:
    - EMOCA-v2 to extract facial expression parameters (FLAME, 156-dim → 53-dim: 50 expression + 3 jaw).
    - OSX to extract whole-body parameters (SMPL-X, 179-dim → 117-dim: 27 upper-body + 45×2 hands).
    - Savitzky-Golay smoothing to ensure temporal continuity.

**MARS Model**:
- **VQ-VAE Quantization**: Train Face VQ-VAE and Body VQ-VAE separately to discretize continuous nonverbal parameter sequences into codebook tokens.
- **Unified Autoregressive Modeling**: Input sequences are formed by interleaving text tokens, face tokens, and body tokens chronologically, using a Transformer for unified next-token prediction.

### Key Designs

- **Decoupled VQ-VAE**: Face and body components use independent encoders, quantizers, and decoders to capture motion patterns at different granularities respectively.
- **Loss Function**: Commitment loss (codebook learning) + grouped reconstruction loss (L1 loss on expressions/jaw/upper-body/hands respectively) + motion velocity loss (to maintain temporal continuity).
- **EMA Codebook Update**: Exponential Moving Average is employed to stabilize training.
- **Hierarchical Token Prediction**: At the same time step, the model first predicts the text token, and then conditionally predicts the face and body token code indices.

## Key Experimental Results

### Main Results

**VENUS Dataset Statistics**:

| Metric | Value |
|------|------|
| Dialogues | 89,459 |
| Turns | 1,114,328 |
| Total Duration | 14,910 hours |
| Sentences | 7,118,654 |
| Unique Words | 527,270 |
| Avg. Turns/Dialogue | 21 |
| Avg. Nonverbal Frames/Turn | 547 |
| Total Nonverbal Expressions | **1 Billion** |

**Comparison with Existing Datasets**:

| Dataset | Dialogues | Turns | Duration (h) | Text | Video | Nonverbal |
|--------|--------|--------|---------|------|------|--------|
| IEMOCAP | 151 | 7,333 | 12 | ✓ | ✓ | ✗ |
| YTD-18M | 18M | 54M | 30K | ✓ | ✓ | ✗ |
| BEAT | — | — | 76 | ✓ | ✗ | ✓ |
| EMAGE | — | — | 60 | ✓ | ✗ | ✓ |
| **VENUS** | **89,459** | **1,114,328** | **14,910** | **✓** | **✓** | **✓** |

→ VENUS is the **first large-scale dialogue dataset to simultaneously incorporate text, video, and nonverbal 3D annotations**.

**VQ-VAE Reconstruction Quality (Comparison with SOTA)**:

| Method | Face VMSE↓ | Face LVD↓ | Face Diversity↑ | Body VMSE↓ | Body LVD↓ |
|------|-----------|-----------|-----------------|-----------|-----------|
| Ng et al. (2023) | 0.5787 | 0.4422 | 7.5866 | 2.6424 | 0.1268 |
| Guo et al. (2024) | 0.5474 | 0.4160 | 7.7693 | 2.0608 | 0.0994 |
| **Ours** | **0.5106** | **0.4020** | **7.8430** | **1.9946** | **0.0962** |

→ Outperforms prior methods comprehensively in facial and body reconstruction while maintaining higher diversity.

### Key Findings

1. VENUS exhibits a rich distribution of nonverbal expressions; t-SNE visualization demonstrates that facial expressions naturally cluster into meaningful emotional patterns even without explicit emotion labels.
2. Body language also forms distinct clusters corresponding to common conversational gestures (e.g., nodding, hand gestures for emphasis).
3. Each dialogue averages 21 turns and 547 frames of nonverbal expressions, supporting multi-turn long-dialogue modeling.
4. L1 reconstruction loss outperforms smooth L1 and L2 losses (confirmed by ablation experiments).
5. Using independent codebooks for facial expressions and body language outperforms joint modeling.

## Highlights & Insights

- **Filling a Critical Gap**: The first large-scale dialogue dataset aligning text, 3D facial expressions, and 3D body language, making nonverbal communication modeling possible.
- **Scalable Pipeline**: A fully automated data construction pipeline that can continuously harvest new data from YouTube.
- **Unified Modeling Concept**: Discretizes nonverbal cues and shares an autoregressive framework with text, which is elegant and concise.
- **Practical Application Prospects**: An infrastructure-level contribution to scenarios like virtual characters, digital humans, social robots, and immersive conversational AI.
- **Thorough Analysis**: Multi-dimensional analysis of dataset quality (distribution visualization, ablation studies, reconstruction metrics) enhances credibility.

## Limitations & Future Work

- Only covers English podcast dialogues, ignoring nonverbal expression differences across languages and cultures.
- 3D parameters rely on pseudo-labels (predictions from EMOCA-v2 and OSX), which introduces automatic extraction errors.
- Focuses solely on dyadic (two-person) dialogue scenarios; complex interactions in multi-party dialogues are not covered.
- The scale and architecture of the MARS model are preliminary, without fully leveraging large-scale pre-trained LLMs.
- The perceptual quality of the generated nonverbal actions lacks validation from large-scale user studies.

## Related Work & Insights

- **Multimodal LLMs**: LLaVA (Liu et al., 2024), Qwen-VL (Bai et al., 2023), MiniGPT-4 (Chen et al., 2023), VideoChat (Li et al., 2023)
- **Video Grounded Dialogue Learning**: Champagne/YTD-18M (Han et al., 2023), MultiDialog (Park et al., 2024)
- **Human Motion Synthesis**: MotionGPT (Wu et al., 2024); EMAGE (Liu et al., 2024a) based on 3D gesture generation
- **VQ-VAE**: Van Den Oord et al. (2017); Razavi et al. (2019) hierarchical VQ-VAE

## Rating

- **Novelty**: ★★★★★ — The first large-scale dialogue dataset with tri-modal alignment of text, face, and body; a pioneering contribution.
- **Experimental Thoroughness**: ★★★★☆ — Thorough VQ-VAE ablations, though MARS evaluation is preliminary but proves concept feasibility.
- **Value**: ★★★★★ — The open-sourced dataset and code provide infrastructure for nonverbal communication research.
- **Writing Quality**: ★★★★☆ — The pipeline is clearly described, though the density of mathematical formulations might slightly affect readability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AkaCE: A Multimodal Multi-party Dataset for Emotion Recognition in Movie Dialogues](akan_cinematic_emotions_ace_a_multimodal_multi-party_dataset_for_emotion_recogni.md)
- [\[ACL 2025\] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)
- [\[CVPR 2025\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](../../CVPR2025/multimodal_vlm/sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)
- [\[ACL 2025\] mOSCAR: A Large-scale Multilingual and Multimodal Document-level Corpus](moscar_a_large-scale_multilingual_and_multimodal_document-level_corpus.md)
- [\[ACL 2025\] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art](godbench_a_benchmark_for_multimodal_large_language_models_in_video_comment_art.md)

</div>

<!-- RELATED:END -->
