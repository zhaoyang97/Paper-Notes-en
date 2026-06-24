---
title: >-
  [Paper Note] ImpliHateVid: Implicit Hate Speech Detection in Videos
description: >-
  [ACL 2025][Social Computing][Implicit hate speech] The task of implicit hate speech detection in videos is proposed for the first time. The ImpliHateVid dataset containing 2,009 videos is constructed, and a two-stage contrastive learning framework is designed to integrate text, image, and audio tri-modal features.
tags:
  - "ACL 2025"
  - "Social Computing"
  - "Implicit hate speech"
  - "video content moderation"
  - "multimodal"
  - "contrastive learning"
  - "dataset"
date: 2026-05-08
content_hash: 3025dc7a1b9b451e
---

# ImpliHateVid: Implicit Hate Speech Detection in Videos

**Conference**: ACL 2025  
**arXiv**: [2508.06570](https://arxiv.org/abs/2508.06570)  
**Code**: [GitHub](https://github.com/videohatespeech/Implicit_Video_Hate)  
**Area**: Social Computing  
**Keywords**: Implicit hate speech, video content moderation, multimodal, contrastive learning, dataset

## TL;DR

The task of implicit hate speech detection in videos is proposed for the first time. The ImpliHateVid dataset containing 2,009 videos is constructed, and a two-stage contrastive learning framework is designed to integrate text, image, and audio tri-modal features.

## Background & Motivation

**Background**: Hate speech detection research mainly focuses on text (tweets, comments) and images (memes), while video hate detection is in its infancy (e.g., HateMM) and primarily addresses explicit hate.

**Limitations of Prior Work**: Implicit hate speech indirectly conveys prejudice through coded language, implied meanings, and contextual cues. While seemingly harmless on the surface, it propagates harm, making it challenging for existing methods to capture.

**Key Challenge**: Videos dominate digital communication, yet there is a lack of datasets and detection methods specifically targeting implicit hate in videos.

**Goal**: Construct the first video implicit hate speech detection dataset and propose an effective multimodal detection method.

**Key Insight**: Hate videos are collected from low-moderation platforms such as BitChute and Odysee, and multimodal information is fused through two-stage contrastive learning.

**Core Idea**: Through two-stage contrastive learning (intra-modal $\rightarrow$ cross-modal) combined with sentiment and caption features, multimodal cues of implicit hate in videos are comprehensively captured.

## Method

### Overall Architecture

Preprocessing (extracting audio/text/visual frames) $\rightarrow$ ImageBind feature extraction (1024-dimensional) $\rightarrow$ Stage 1 intra-modal contrastive learning $\rightarrow$ Stage 2 cross-modal contrastive learning $\rightarrow$ fusion classification.

### Key Designs

1. **Two-stage contrastive learning**: Stage 1: Train three modal feature encoders (audio/text/image), concatenate the tri-modal features, project them to a shared space through a projection head, and optimize with supervised contrastive loss. Stage 2: Train cross-modal encoders (IT/IA/TA) to further align cross-modal representations.
2. **Auxiliary features**: Sentiment features (NRCLex sentiment lexicon + VADER sentiment score) and caption features (OFA-generated image captions $\rightarrow$ BERT encoding) complement the primary modal representations.
3. **Data labeling process**: Supervised by 1 professor and 1 PhD student, annotated by 4 undergraduate students. Videos are annotated in weekly batches of 50, with no more than 20 videos per day and 10-15 minute rest intervals to protect annotators' mental health.

### Loss & Training

Total loss = Stage 1 loss + Stage 2 loss + supervised contrastive loss of sentiment/captions. The supervised contrastive loss forces similar samples closer and dissimilar samples further apart.

## Key Experimental Results

### Main Results (Binary Classification: Hate/Non-Hate)

| Method | ImpliHateVid Acc | ImpliHateVid F1 | HateMM Acc | HateMM F1 |
|------|-----------------|-----------------|------------|-----------|
| BERT (text) | 0.691 | 0.688 | 0.735 | 0.664 |
| ViT (image) | 0.766 | 0.768 | 0.748 | 0.672 |
| GPT-4 (video) | 0.499 | 0.666 | 0.401 | 0.572 |
| MulT | 0.835 | 0.835 | 0.657 | 0.521 |
| CSID | 0.815 | 0.815 | 0.732 | 0.714 |
| **Ours** | **0.875** | **0.877** | **0.976** | **0.976** |

### Ablation Study (Three-class Classification: Macro-F1 of Non-Hate/Implicit/Explicit)

| Method | Macro-F1 |
|------|----------|
| BERT | 0.591 |
| ViT | 0.588 |
| GPT-4o (text) | 0.308 |
| CSID (best multimodal baseline) | 0.742 |
| Ours | **0.803** |

### Key Findings

- Large multimodal models like GPT-4 and LLaVA perform poorly on video hate detection (close to random).
- Multimodal methods significantly outperform unimodal ones, especially in implicit hate speech detection.
- The performance improvement is even more significant on HateMM (Acc 0.976), indicating strong generalization of the method.
- Implicit hate speech detection remains more challenging than explicit hate detection.

## Highlights & Insights

- For the first time, hate speech detection is extended from explicit to implicit hate in videos.
- The design of two-stage contrastive learning is reasonable: learning individual modal representations first, followed by cross-modal alignment.
- The attention paid to annotators' mental health during the annotation process is worth referencing by other similar works.
- The failure of large multimodal LLMs on such tasks is worth pondering.

## Limitations & Future Work

- The dataset is limited in scale, containing only 2,009 videos.
- Consists only of English content.
- The binary and three-class classification settings are coarse and could be scaled to more fine-grained hate types.
- Annotation sources (BitChute/Odysee) may have content biases.

## Related Work & Insights

- Complementary to video hate detection works such as HateMM and MultiHateClip.
- The effectiveness of contrastive learning in multimodal hate detection can be generalized to other harmful content detection tasks.
- Provides a new technical direction for social media content moderation.

## Technical Details

- ImageBind extracts 1024-dimensional features, which are mapped to a shared embedding space by a projection head.
- Sentiment feature dimension: NRCLex $d_e$ dimensions + VADER 1 dimension $\rightarrow$ concatenated as $f_{ES} \in \mathbb{R}^{d_e+1}$.
- Caption generation: OFA model $\rightarrow$ BERT encoded as $f_C \in \mathbb{R}^{d_c}$.
- Videos are uniformly sampled for 100 frames, with padding applied if insufficient.
- Dataset balance: approximately 50% each for hate/non-hate, and approximately 25% each for implicit/explicit hate.
- The average number of transcribed words for non-hate videos (175) is about twice that of hate videos (80-85).
- Annotator protection measures: $\le 20$ videos per day, 10-15 minute breaks after each video, regularly scheduled mental health checkups.

## Rating

- Novelty: ⭐⭐⭐⭐ The first video implicit hate speech detection dataset, establishing a meaningful task definition.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple baseline comparisons and cross-dataset validation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and detailed method description.
- Value: ⭐⭐⭐⭐ Fills the gap in video implicit hate speech detection, offering high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] RV-HATE: Reinforced Multi-Module Voting for Implicit Hate Speech Detection](../../ACL2026/social_computing/rv-hate_reinforced_multi-module_voting_for_implicit_hate_speech_detection.md)
- [\[ACL 2025\] HateDay: Insights from a Global Hate Speech Dataset Representative of a Day on Twitter](hateday_global_hate_speech.md)
- [\[ACL 2025\] STATE ToxiCN: A Benchmark for Span-level Target-Aware Toxicity Extraction in Chinese Hate Speech Detection](state_toxicn_a_benchmark_for_span-level_target-aware_toxicity_extraction_in_chin.md)
- [\[ACL 2025\] Silencing Empowerment, Allowing Bigotry: Auditing the Moderation of Hate Speech on Twitch](silencing_empowerment_allowing_bigotry_auditing_the_moderation_of_hate_speech_on.md)
- [\[ACL 2025\] MDiT-Bench: Evaluating the Dual-Implicit Toxicity in Large Multimodal Models](mdit-bench_evaluating_the_dual-implicit_toxicity_in_large_multimodal_models.md)

</div>

<!-- RELATED:END -->
