---
title: >-
  [Paper Note] Understanding Co-speech Gestures in-the-wild
description: >-
  [ICCV 2025][Audio & Speech][Co-speech gestures] This paper proposes JEGAL — a joint gesture-audio-language tri-modal embedding space that learns co-speech gesture representations under weak supervision via a global phras…
tags:
  - "ICCV 2025"
  - "Audio & Speech"
  - "Co-speech gestures"
  - "tri-modal representation learning"
  - "gesture retrieval"
  - "gesture word spotting"
  - "active speaker detection"
date: 2026-05-08
content_hash: 4720a85b977bbed3
---

# Understanding Co-speech Gestures in-the-wild

**Conference**: ICCV 2025
**arXiv**: [2503.22668](https://arxiv.org/abs/2503.22668)
**Code**: [Project Page](https://www.robots.ox.ac.uk/~vgg/research/jegal)
**Area**: Audio & Speech
**Keywords**: Co-speech gestures, tri-modal representation learning, gesture retrieval, gesture word spotting, active speaker detection

## TL;DR

This paper proposes JEGAL — a joint gesture-audio-language tri-modal embedding space that learns co-speech gesture representations under weak supervision via a global phrase-level contrastive loss and a local gesture-word coupling loss. Three new gesture understanding tasks and benchmarks are introduced, and the method outperforms a range of baselines including large vision-language models.

## Background & Motivation

Humans gesture while speaking — gesture is an integral component of human communication. Co-speech gestures span a rich spectrum: from beat gestures (rhythmic hand movements that emphasize specific words) to iconic gestures (gestures that convey semantic content, e.g., spreading one's arms to indicate "enormous"). Non-verbal communication accounts for 55% of total communication, yet machines' understanding of gesture semantics remains severely limited.

Learning associations between gestures and speech/text is exceptionally challenging for several reasons:

**Sparse and ambiguous cross-modal correlations**: Typically only a handful of words are clearly expressed through gesture, and the same utterance can manifest vastly different gestures across contexts or individuals.

**High inter-personal and cross-cultural variability**: Gestures depend on the speaker's emotion, cultural background, and social setting.

**Semantically vacuous gestures**: Beat gestures are aligned only with prosodic rhythm and carry no semantic content, making direct mapping to specific words infeasible.

Limitations of prior work:
- GestSync learns gesture representations via audio-visual synchronization, but captures only low-level temporal correlations rather than high-level semantics.
- GestureDiffuCLIP learns a gesture-text joint embedding but lacks word-level correspondences.
- Large vision-language models (e.g., CLIP, LanguageBind) are not designed for gesture understanding and have limited capacity to handle long videos and gesture-specific features.

## Method

### Overall Architecture

JEGAL (Joint Embedding space for Gestures, Audio, and Language) learns tri-modal representations through two complementary contrastive learning objectives:

1. **Global phrase-level contrastive loss**: Encourages the model to learn holistic semantic correspondences between gesture clips and speech/text segments.
2. **Local gesture-word coupling loss**: Encourages the model to discover frame-level associations between gesture clips and specific words.

### Key Designs

1. **Three modal encoders + fusion module**

   - **Gesture encoder $\mathbb{G}$**: A 3D convolutional layer (temporal receptive field of 5 frames in the first layer) followed by a Transformer encoder, producing frame-level features $\mathbf{g}^T \in \mathbb{R}^{T \times d}$. The backbone is initialized from GestSync and frozen; only the Transformer head is trained. Facial regions are masked to prevent lip-motion information leakage.
   - **Text encoder $\mathbb{L}$**: Multilingual RoBERTa XLM-Base extracts subword features, which are encoded and projected by a Transformer head to yield $\mathbf{l}^w \in \mathbb{R}^{W \times d/2}$.
   - **Speech encoder $\mathbb{S}$**: A 2D-CNN encodes mel-spectrograms, outputting $\mathbf{s} \in \mathbb{R}^{T' \times d/2}$.
   - **Fusion module**: Subword text embeddings are aggregated into word-level embeddings $\mathbf{l}^w$; speech features are aggregated into word-level features $\mathbf{s}^w$ according to word boundaries; these are concatenated along the feature dimension to obtain the joint word-level representation $\mathbf{c}^w \in \mathbb{R}^{W \times d}$.

   During training, speech or text input is randomly zeroed out with 50% probability (modality dropout), encouraging the model to learn from both modalities in a balanced manner and enabling single-modality inference at test time.

2. **Gesture-word alignment mechanism**

   Word boundaries are aligned to speech but not necessarily to gestures — a gesture may be longer, shorter, or temporally offset relative to the corresponding word. To address this, an attention-based pooling mechanism is designed: the temporal window of each word is expanded by $p=10$ frames on each side, and the word embedding $c^{w_i}$ is used to compute an attention-weighted aggregation over gesture frames within the expanded window:

   $g^{w_i} = \sum_{j=S}^{E} \frac{\exp(\gamma \cdot g^{T_j} \cdot c^{w_i})}{\sum_{j=S}^{E} \exp(\gamma \cdot g^{T_j} \cdot c^{w_i})} \cdot g^{T_j}$

   This enables the model to flexibly locate the true temporal extent of a gesture beyond the speech boundaries.

3. **Dual training objectives**

   **Global phrase-level contrastive loss**: Frame-level gesture features and word-level speech-text features are average-pooled to obtain global embeddings $\mathbf{g}$ and $\mathbf{c}$, and an InfoNCE loss is applied:

   $\mathcal{L}_{seq} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\gamma \cdot \cos(g_i, c_i))}{\sum_{j=1}^{N} \exp(\gamma \cdot \cos(g_i, c_j))}$

   **Local gesture-word coupling loss**: For each speech-text word $c^{w_i}$, the most similar gesture frame $g^{w_j}$ is identified, and a coupling score is computed as $\lambda(g^w, c^w) = \frac{1}{W}\sum_{i=1}^{W}\max_{j}\cos(g^{w_i}, c^{w_j})$. The core assumption is that matched gesture–speech-text pairs exhibit stronger word-level coupling. A contrastive loss in the InfoNCE form is applied analogously.

   Final loss: $\mathbb{L} = \beta \cdot \mathcal{L}_{seq} + (1-\beta) \cdot \mathcal{L}_{couple}$

### Loss & Training

- Data: PATS (25 speakers, 162 hours) + a subset of MultiVSR (6,934 speakers, 556 hours), totalling approximately 720 hours from 7,000+ speakers.
- Preprocessing: Resampled to 25 FPS / 16 kHz; word-aligned transcriptions generated with WhisperX; low-gesture-activity samples filtered based on L2 distance of body keypoints.
- Optimizer: AdamW, lr=5e-5, weight decay=1e-4, betas=(0.9, 0.98).
- Gesture head: 6-layer Transformer; text head: 3 layers; hidden dim=512, FFN=2048, 8 attention heads.

## Key Experimental Results

### Main Results

**Cross-modal retrieval (AVS-Ret benchmark, 500 diverse gesture clips)**:

| Method | Modality | S→G R@5↑ | S→G R@10↑ | S→G MR↓ | G→S R@5↑ | G→S R@10↑ | G→S MR↓ |
|--------|----------|----------|-----------|---------|----------|-----------|---------|
| GestSync (FT) | Audio | 10.0 | 18.2 | 70.5 | 11.6 | 16.6 | 82.5 |
| Clip4Clip (FT) | Text | 8.0 | 12.6 | 132.0 | 3.6 | 7.0 | 125.0 |
| **JEGAL** | **T+A** | **18.8** | **30.8** | **31.0** | **18.2** | **20.2** | **24.5** |

JEGAL substantially outperforms all baselines; multimodal fusion (T+A) markedly surpasses unimodal variants.

**Gesture word spotting (AVS-Spot benchmark, 500 annotated clips)**:

| Method | Modality | Accuracy↑ |
|--------|----------|----------|
| GestSync (FT) | Audio | 21.04 |
| GestureDiffuCLIP (FT) | Text | 19.50 |
| JEGAL (Text) | Text | 61.00 |
| **JEGAL (T+A)** | **T+A** | **63.60** |

JEGAL decisively outperforms all baselines on word spotting (63.6% vs. 21.04%), with the primary advantage attributable to the local gesture-word coupling loss.

**Active speaker detection (AVS-Asd benchmark)**:

| Method | 2-person↑ | 4-person↑ | 6-person↑ |
|--------|-----------|-----------|-----------|
| GestSync (FT) | 81.2 | 64.8 | 54.4 |
| **JEGAL (T+A)** | **76.8** | **57.8** | **48.0** |

GestSync achieves the best performance on this task due to its strong frame-level synchronization supervision; JEGAL ranks second.

### Ablation Study

| Loss Configuration | Retrieval R@5↑ | Retrieval MR↓ | Spotting Acc↑ | ASD Acc↑ |
|-------------------|---------------|--------------|--------------|----------|
| Global contrastive only | 12.20 | 45 | 20.83 | 44.2 |
| Word coupling only | 8.50 | 76 | 52.46 | 14.8 |
| **Global + word coupling** | **18.80** | **31** | **63.60** | **48.0** |

| Fusion Strategy | Retrieval R@5↑ | Spotting Acc↑ | ASD Acc↑ |
|----------------|---------------|--------------|----------|
| Independent pairwise contrastive (text) | 9.39 | 34.31 | 29.6 |
| Independent pairwise contrastive (audio) | 9.80 | 23.67 | 31.4 |
| Late fusion (average) | 17.00 | 56.04 | 41.2 |
| **Late fusion (concatenation)** | **18.80** | **63.60** | **48.0** |

### Key Findings

- **Speech and text capture complementary gesture signals**: Text is more effective at word-level semantic correspondence (spotting 61.0% vs. 41.8%), whereas speech is more sensitive to stressed/emphasized words (stressed-word spotting gap of 39.4% vs. non-stressed-word gap of 14.8%).
- **Both losses are indispensable**: The global contrastive loss is critical for retrieval and ASD, while the word coupling loss is critical for spotting; their combination achieves the best performance across all tasks.
- Concatenation fusion outperforms average fusion and independent pairwise contrastive training — the model needs to integrate multiple information streams within a shared embedding space.

## Highlights & Insights

- **Word-level learning under weak supervision**: Despite having only phrase-level supervision (with no annotation of which words are gesturally expressed), the max-coupling strategy in the coupling loss successfully learns word-level correspondences.
- **Definition of three new tasks and benchmarks**: A systematic evaluation framework for co-speech gesture understanding (retrieval / spotting / ASD) is established, providing a foundation for advancing the field.
- **Face masking design**: Masking facial regions prevents lip-motion information leakage, ensuring that the learned representations encode pure gesture signals.

## Limitations & Future Work

- Training data is predominantly English; cross-lingual and cross-cultural generalization of gesture understanding remains unvalidated.
- The gesture encoder operates on RGB video, incurring high computational cost; future work could explore lightweight inputs based on 2D/3D keypoints.
- Only hand gestures are considered; other non-verbal signals such as head movements and facial expressions are not incorporated.
- On the active speaker detection task, JEGAL still lags behind GestSync, indicating room for improvement in frame-level synchronization alignment.

## Related Work & Insights

- A fundamental distinction from sign language understanding: in sign language, gestures are the primary communication channel (with text as a translation), whereas co-speech gestures supplement spoken language (co-occurring with words but not translating them), necessitating entirely different modeling approaches.
- The max-coupling strategy in the coupling loss is conceptually analogous to Multiple Instance Learning (MIL), which learns instance-level information from bag-level labels only.
- The learned representations have potential applications in digital human gesture generation, language learning assistance, and privacy-preserving speaker detection.

## Rating

- **Novelty**: ⭐⭐⭐⭐ A systematic tri-modal framework for co-speech gesture understanding with an elegantly designed word coupling loss
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three new benchmark tasks with comprehensive ablation studies and insightful analysis of speech vs. text contributions
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear motivation, excellent figures, and thorough analysis
- **Value**: ⭐⭐⭐⭐ Opens a systematic research direction for co-speech gesture understanding; the three benchmarks provide lasting value to the community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Instance-Specific Test-Time Training for Speech Editing in the Wild](../../NeurIPS2025/audio_speech/instance-specific_test-time_training_for_speech_editing_in_the_wild.md)
- [\[ACL 2026\] Computational Narrative Understanding for Expressive Text-to-Speech](../../ACL2026/audio_speech/computational_narrative_understanding_for_expressive_text-to-speech.md)
- [\[AAAI 2026\] DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling](../../AAAI2026/audio_speech/dualspeechlm_towards_unified_speech_understanding_and_generation_via_dual_speech.md)
- [\[ICCV 2025\] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)
- [\[AAAI 2026\] HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding](../../AAAI2026/audio_speech/hpsu_a_benchmark_for_human-level_perception_in_real-world_spoken_speech_understa.md)

</div>

<!-- RELATED:END -->
