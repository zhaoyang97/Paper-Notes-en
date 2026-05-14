---
title: >-
  [Paper Note] EmoVid: A Multimodal Emotion Video Dataset for Emotion-Centric Video Understanding and Generation
description: >-
  [AAAI 2026][Video Understanding][emotion video dataset] This paper presents EmoVid, the first large-scale multimodal emotion video dataset targeting artistic and non-photorealistic content (22,758 video clips)…
tags:
  - "AAAI 2026"
  - "Video Understanding"
  - "emotion video dataset"
  - "video generation"
  - "emotion annotation"
  - "text-to-video"
  - "affective computing"
date: 2026-05-08
content_hash: 92a2cde836dcae5f
---

# EmoVid: A Multimodal Emotion Video Dataset for Emotion-Centric Video Understanding and Generation

**Conference**: AAAI 2026
**arXiv**: [2511.11002](https://arxiv.org/abs/2511.11002)
**Code**: [https://zane-zyqiu.github.io/EmoVid](https://zane-zyqiu.github.io/EmoVid) (Project Page)
**Area**: Video Understanding
**Keywords**: emotion video dataset, video generation, emotion annotation, text-to-video, affective computing

## TL;DR
This paper presents EmoVid, the first large-scale multimodal emotion video dataset targeting artistic and non-photorealistic content (22,758 video clips), spanning three content types—animation, film, and emoji stickers—and demonstrates the effectiveness of emotion-conditioned video generation by fine-tuning the Wan2.1 model, achieving significant improvements over baselines on emotion accuracy metrics.

## Background & Motivation

Video is a powerful medium for storytelling and expression, with emotion playing a central role in audience engagement. While recent video generation models have made substantial progress in visual coherence and motion, **attention to emotional expressiveness remains very limited**. In creative applications such as animated character generation, meme creation, and film editing, emotional expressiveness is critical yet remains underexplored.

Existing video emotion datasets suffer from the following limitations:

**Limited scale**: Datasets such as CAER (12h), MELD (1.4h), and DEAP (2h) are relatively small.

**Narrow content coverage**: Nearly all existing datasets focus on facial expressions in real-world scenarios, lacking stylized and non-photorealistic content.

**Incomplete modalities**: Many datasets lack audio or textual descriptions.

**Unsuitability for generative tasks**: Existing datasets are primarily designed for emotion recognition and lack sufficient visual diversity.

**Key Challenge**: Emotion is critically important in video creation, yet no benchmark dataset or evaluation protocol suited to creative scenarios exists. EmoVid fills this gap by covering three artistic video types: animation, film clips, and emoji stickers.

## Method

### Overall Architecture
EmoVid's construction and application consists of three components:
1. Dataset construction (collection, annotation, and attribute extraction)
2. Data analysis (emotion patterns, color–emotion associations, temporal dynamics)
3. Benchmark evaluation and model fine-tuning (T2V and I2V tasks)

### Key Designs

1. **Multi-source Data Collection**:

    - **Function**: Videos are collected from three sources—the MagicAnime dataset (2,807 animated facial clips), Condensed Movies (13,255 film clips segmented via PySceneDetect, retaining clips of 4–30 seconds), and the Tenor API (6,696 GIF emoji stickers).
    - **Mechanism**: Multiple artistic styles and content types are covered, including American, Chinese, and Japanese animation, film scenes, and internet memes.
    - **Design Motivation**: To ensure diversity in emotional expression and cross-domain generalizability.

2. **Human–Machine Collaborative Annotation**:

    - **Function**: The Mikels eight-category emotion model (amusement, awe, contentment, excitement, anger, disgust, fear, sadness) is adopted, combining manual annotation with automatic VLM-based labeling.
    - **Mechanism**: 20% of the data is manually annotated (each video by 3 annotators, retained only when at least 2 agree); the NVILA-Lite-2B model is then fine-tuned on this subset to annotate the remaining 80%.
    - **Quality Validation**: A randomly sampled 1% validation set is used to compute Cohen's kappa between three human annotators and the VLM; the discrepancy is < 4%, indicating comparable annotation quality.
    - **Design Motivation**: To balance annotation precision and resource consumption.

3. **Multi-dimensional Attribute Annotation and Analysis**:

    - **Function**: Color attributes (colorfulness, brightness, hue) are extracted for each clip, and textual descriptions are generated using NVILA-8B.
    - **Key Findings**:
        - Videos with positive emotions (positive valence) tend to be brighter and more colorful.
        - High-arousal emotions are darker but more colorful.
        - The Markov emotion transition matrix for film clips reveals strong self-persistence (fear: 0.53, anger: 0.46).
        - Intra-valence transitions are far more frequent than cross-valence transitions.
    - **Design Motivation**: To provide color and temporal priors that can be leveraged for emotion-aware video generation.

### Loss & Training
- The Wan2.1 model is fine-tuned using the DiffSynth Studio framework.
- LoRA configuration: rank=32, lr=1e-4, epochs=3, batch_size=1.
- Training data is balanced: 2,727 animation + 8,000 film + 6,616 sticker clips.

## Key Experimental Results

### Main Results

**T2V Task**:

| Method | FVD↓ | CLIP↑ | Flicker↓ | EA-2cls↑ | EA-8cls↑ |
|--------|------|-------|----------|----------|----------|
| VideoCrafter-V2 | 610.1 | 0.3012 | 0.0184 | 80.42 | 42.50 |
| HunyuanVideo | 552.6 | 0.2776 | 0.0116 | 76.87 | 40.41 |
| CogVideoX | 584.0 | 0.3013 | 0.0213 | 82.91 | 44.58 |
| WanVideo (before) | 594.3 | 0.2982 | 0.0091 | 84.17 | 44.16 |
| **WanVideo (after)** | **573.7** | **0.3021** | 0.0143 | **88.33** | **48.33** |

**I2V Task**:

| Method | FVD↓ | SD↑ | Flicker↓ | EA-2cls↑ | EA-8cls↑ |
|--------|------|-----|----------|----------|----------|
| DynamiCrafter512 | 512.3 | 0.7288 | 0.0280 | 90.41 | 71.25 |
| CogVideoX | 528.4 | 0.7214 | 0.0331 | 90.83 | 70.83 |
| WanVideo (before) | 517.9 | 0.7146 | 0.0325 | 91.25 | 71.30 |
| **WanVideo (after)** | **517.8** | 0.7193 | 0.0324 | **94.58** | **76.25** |

### Ablation Study

| Configuration | EA-8cls (T2V) | EA-8cls (I2V) | Note |
|---------------|--------------|--------------|------|
| WanVideo original | 44.16 | 71.30 | Untuned baseline |
| WanVideo + EmoVid fine-tuning | **48.33** | **76.25** | Significant improvement in emotion accuracy |
| Gain | +4.17 | +4.95 | Validates EmoVid's effectiveness |
| Over strongest competitor | +3.75 (vs CogVideoX) | +5.42 (vs DynamiCrafter) | Outperforms all baselines |

### Key Findings
- The fine-tuned model maintains or slightly improves general visual metrics (FVD, CLIP) while achieving **substantial gains in emotion accuracy (EA)**.
- Emotion accuracy improvement is more pronounced in the I2V task (+4.95) than in T2V (+4.17).
- Three emotion trajectory patterns are identified in film clips: "hold, intra-valence drift, and arousal leap."
- Negative emotions exhibit a chain escalation pattern: sadness → fear/anger.
- Qualitative results show that the fine-tuned model more precisely captures emotional intent, including more appropriate facial expressions and emotionally consistent motion patterns.

## Highlights & Insights
- **First emotion video benchmark for artistic content**: Fills the data gap in affective computing for stylized video.
- **Multimodal completeness**: Includes video, audio, textual descriptions, color attributes, and emotion labels.
- **Temporal emotion analysis**: The Markov transition matrix reveals regularities in emotion evolution within films.
- **Practical applications**: Emoji stickers generated via LoRA fine-tuning are directly deployable in social media contexts.
- **Standardized evaluation protocol**: EA-2cls and EA-8cls metrics provide standardized assessment for emotion-aware generation.

## Limitations & Future Work
- Each clip is assumed to convey a single emotion; in practice, emotions may be complex and composite.
- The audio modality remains underutilized; future work could develop unified video–audio–text multimodal models.
- Unbalanced emotion distribution (e.g., fewer amusement and awe samples) may affect model performance on minority classes.
- Emoji stickers lack audio due to the GIF format, resulting in incomplete modality coverage.
- Fine-tuning slightly increases the flicker metric (0.0091→0.0143), indicating a minor degradation in visual stability.

## Related Work & Insights
- This work bridges affective computing and video generation, opening a new research direction in affective video computing.
- Quantitative analysis of color–emotion associations can provide prior guidance for controllable video generation.
- The Mikels eight-category emotion model, well-validated in the image domain, is shown to be applicable to the video domain.
- The human–machine collaborative annotation paradigm (20% manual + 80% VLM) offers a practical framework for large-scale annotation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First emotion video dataset targeting artistic content; a uniquely positioned contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across T2V and I2V tasks; ablation study is relatively straightforward.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with thorough data analysis.
- **Value**: ⭐⭐⭐⭐ — The dataset, benchmark, and analytical insights offer lasting contributions to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MambaMia: State-Space Hierarchical Compression for Hour-Long Video Understanding in Large Multimodal Models](state-space_hierarchical_compression_with_gated_attention_an.md)
- [\[CVPR 2026\] OpenMarcie: Dataset for Multimodal Action Recognition in Industrial Environments](../../CVPR2026/video_understanding/openmarcie_dataset_for_multimodal_action_recognition_in_industrial_environments.md)
- [\[NeurIPS 2025\] SmartWilds: Multimodal Wildlife Monitoring Dataset](../../NeurIPS2025/video_understanding/smartwilds_multimodal_wildlife_monitoring_dataset.md)
- [\[NeurIPS 2025\] egoEMOTION: Egocentric Vision and Physiological Signals for Emotion and Personality Recognition in Real-World Tasks](../../NeurIPS2025/video_understanding/egoemotion_egocentric_vision_and_physiological_signals_for_emotion_and_personali.md)
- [\[AAAI 2026\] LiViBench: An Omnimodal Benchmark for Interactive Livestream Video Understanding](livibench_an_omnimodal_benchmark_for_interactive_livestream_video_understanding.md)

</div>

<!-- RELATED:END -->
