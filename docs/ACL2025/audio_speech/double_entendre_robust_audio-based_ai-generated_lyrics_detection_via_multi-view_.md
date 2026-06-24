---
title: >-
  [Paper Note] Double Entendre: Robust Audio-Based AI-Generated Lyrics Detection via Multi-View Fusion
description: >-
  [ACL 2025][Audio & Speech][AI-generated lyrics detection] This paper proposes DE-detect, an audio-only multi-view late fusion pipeline. By combining the textual features of automatically transcribed lyrics with lyric-related acoustic features extracted by a speech model, it achieves robust detection of AI-generated lyrics, outperforming single-modality methods in both in-domain and out-of-domain scenarios.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "AI-generated lyrics detection"
  - "multimodal fusion"
  - "speech embeddings"
  - "robust detection"
  - "multi-view fusion"
date: 2026-05-08
content_hash: 891ad84a55e4ce1c
---

# Double Entendre: Robust Audio-Based AI-Generated Lyrics Detection via Multi-View Fusion

**Conference**: ACL 2025  
**arXiv**: [2506.15981](https://arxiv.org/abs/2506.15981)  
**Code**: [https://github.com/deezer/robust-AI-lyrics-detection](https://github.com/deezer/robust-AI-lyrics-detection)  
**Area**: Speech  
**Keywords**: AI-generated lyrics detection, multimodal fusion, speech embeddings, robust detection, multi-view fusion

## TL;DR

This paper proposes DE-detect, an audio-only multi-view late fusion pipeline. By combining the textual features of automatically transcribed lyrics with lyric-related acoustic features extracted by a speech model, it achieves robust detection of AI-generated lyrics, outperforming single-modality methods in both in-domain and out-of-domain scenarios.

## Background & Motivation

**Background**: AI music generation tools (e.g., Suno, Udio) are revolutionizing the music industry, but also pose significant challenges to copyright protection and content moderation. Existing AI-generated music (AIGM) detection methods are primarily divided into audio-based and lyrics-based categories.

**Limitations of Prior Work**: (1) Audio-based detectors achieve over 99% in-domain accuracy, but generalize poorly to new generators and are highly sensitive to audio perturbations like pitch shifts and noise. (2) Lyrics-based detectors require clean, formatted lyric text, but in real-world deployment, only audio is available, and lyrics metadata is typically inaccessible.

**Key Challenge**: Lyrics detection relies on unavailable clean text, while audio-based detection is overly sensitive to low-level acoustic artifacts. Neither approach operates reliably in real-world scenarios.

**Goal**: To design a robust AIGM detection system that takes only audio as input, leveraging both lyrical semantic information and lyric-related acoustic cues.

**Key Insight**: Treat the audio as dual modalities simultaneously: transcribing it via ASR to obtain lyric text (the "what") and capturing lyric-related acoustic features (the "how", such as prosody and intonation) via a speech model, followed by late fusion.

**Core Idea**: Combine automatically transcribed lyrics (semantic content) and speech embeddings (acoustic cues) using multi-view late fusion to achieve robust, audio-only AI lyrics detection.

## Method

### Overall Architecture

DE-detect is a modular late fusion pipeline, with the overall workflow as follows:

1. **Input**: Audio-only signal
2. **Text Branch** (upper channel): The ASR model (Whisper large-v2) transcribes the audio into lyrics $\rightarrow$ the text embedding model (LLM2Vec + Llama3 8B) generates lyrical semantic representations
3. **Speech Branch** (lower channel): The speech model (XEUS) directly extracts lyric-related acoustic features (prosody, intonation, speaker characteristics, etc.) from the audio
4. **Late Fusion**: Features from both branches are linearly projected to 128 dimensions $\rightarrow$ concatenated $\rightarrow$ fed into an MLP classifier to determine real/fake

### Key Designs

#### Text Branch

- **Function**: Automatically transcribing audio into lyric text and then extracting semantic features
- **Mechanism**: Whisper large-v2 is used for ASR transcription, followed by LLM2Vec (based on Llama3 8B) to extract contextualized semantic embeddings of the entire set of lyrics
- **Design Motivation**: Addressing the issue of unavailable lyrics. Although transcribed lyrics contain errors (WER around 20-40%), the text-based detector is robust to these errors. Experiments show that Whisper large-v2 performs best on the detection task, suggesting that a lower WER does not necessarily translate to better detection performance

#### Speech Branch

- **Function**: Extracting lyric-related acoustic information from audio to capture AI-generated prosody and vocal features
- **Mechanism**: The XEUS speech model is used to extract features, followed by mean-pooling to obtain a vector representation
- **Design Motivation**: Transcription only captures "what" is said, while speech embeddings capture "how" it is said, including prosody, intonation, and speaker characteristics. XEUS exhibits the best performance (92.2% recall) as its pre-training data includes singing voices. Experiments also show that XEUS performs near random (50.5%) when distinguishing between real and partly-fake audio, indicating its features do not rely on acoustic artifacts

#### Late Fusion Design

- **Function**: Fusing features from both the text and speech branches for final classification
- **Mechanism**: Features from both branches are linearly projected to 128 dimensions, concatenated, and fed into an MLP trained with binary cross-entropy loss
- **Design Motivation**: The advantages of modular late fusion include: (1) each component can be updated independently; (2) it preserves the strengths of individual components (like multilingual capabilities); (3) it is robust to component changes. This is crucial in the fast-evolving landscape of AIGM

### Loss & Training

- The MLP classifier is trained using **Binary Cross-Entropy Loss**.
- Training Data: Based on the lyric dataset from Labrak et al. (2025), which contains 3,655 real lyrics and 3,535 AI-generated lyrics (from 3 LLMs), covering 9 languages and 6 music genres.
- Audio for AI lyrics is generated using Suno v3.5, while real lyrics use their original audio.
- The final dataset consists of 7,190 songs, balanced between real and fake.

## Key Experimental Results

### Main Results

| Model | Recall (en) | Recall (all) | AUROC (en) | AUROC (all) |
|------|-------------|--------------|------------|-------------|
| GT Lyrics (LLM2Vec) † | 91.3 | 94.3 | 99.0 | 97.3 |
| CNN (Spectrogram) ‡ | 97.5 | 97.4 | 99.9 | 99.8 |
| XEUS | 89.1 | 92.2 | 94.5 | 97.0 |
| Llama3 8B (LLM2Vec) | 90.6 | 90.7 | 97.6 | 94.8 |
| **DE-detect** | **93.9** | **94.9** | **98.2** | **98.5** |

DE-detect achieves a multilingual macro-average recall of 94.9% and an AUROC of 98.5%, outperforming the baseline that uses clean ground-truth lyrics (94.3%), and is only slightly lower than the CNN spectrogram method in-domain.

### Ablation Study

**Out-of-domain Robustness Evaluation (Audio Perturbations + Udio Generalization)**:

| Model | Stretch | Pitch | EQ | Noise | Reverb | Udio |
|------|---------|-------|----|-------|--------|------|
| CNN | 98.1 | 59.0 | 79.4 | 77.4 | 80.7 | 56.9 |
| XEUS | 92.5 | 92.3 | 92.3 | 92.4 | 92.4 | 85.9 |
| Llama3 8B | 90.0 | 89.7 | 89.6 | 89.3 | 89.6 | 85.9 |
| **DE-detect** | **94.1** | **93.9** | **94.0** | **93.9** | **94.1** | **87.9** |

CNN plunges to 59.0% under Pitch attack, and is only 56.9% on Udio generalization. In contrast, DE-detect maintains 93.9%–94.1% performance across all perturbations, and achieves 87.9% on Udio generalization.

**Partly-Fake Experiments** (verifying if the model relies on acoustic artifacts):

| Model | Real vs. Partly-Fake | Fake vs. Partly-Fake |
|------|---------------------|---------------------|
| XEUS | 50.5 (≈random) | 92.0 |
| Llama3 8B | 64.9 | 90.0 |

XEUS performs near random when distinguishing between real and partly-fake audio, proving that it does not rely on acoustic artifacts but rather focuses on lyrical content.

### Key Findings

1. **Transcription quality is not the deciding factor**: A lower WER does not necessarily yield better detection performance; Whisper large-v2 does not have the lowest WER, yet performs the best.
2. **Speech embeddings do not rely on acoustic artifacts**: XEUS performs near random in the real vs. partly-fake experiment, suggesting its features primarily reflect lyrical content rather than generator artifacts.
3. **Multi-view fusion provides consistent benefits**: DE-detect outperforms single-modality methods by 1.5–2% in recall across all out-of-domain scenarios.
4. **CNN methods degrade severely out-of-domain**: They become practically unusable under pitch shifts (59.0%) and on Udio generalization (56.9%).

## Highlights & Insights

1. **Highly Practical**: The entire pipeline requires only audio input and has zero dependency on lyrics metadata, making it ideal for industrial deployment.
2. **Clear Intuition for Multi-View Fusion**: The logic of fusing "what" (transcribed lyrics semantics) and "how" (speech acoustic features) is simple and elegant.
3. **Future-Proof Modular Design**: Each component can be upgraded independently to adapt to the rapidly evolving AIGM ecosystem.
4. **Elaborately Designed Partly-Fake Experiments**: Through controlled variables, it cleverly verifies that the model is indeed detecting lyrics rather than acoustic artifacts.

## Limitations & Future Work

1. Combined training data is mainly based on audio generated by Suno v3.5, exhibiting bias toward other generators (such as Udio).
2. Robustness evaluation does not cover scenarios with superimposed multi-source attacks (e.g., simultaneous pitch shifts and noise).
3. The dataset size is relatively small (~7k songs). Larger and more diverse datasets are needed in the future.
4. Dual-use risks exist, as attackers could exploit weaknesses to bypass detection.

## Related Work & Insights

- **Labrak et al. (2025)**: Proposed a lyrics detection dataset and textual baselines, but relied on clean ground-truth lyrics.
- **Afchar et al. (2024)**: CNN spectrogram methods are highly efficient in-domain but generalize poorly, revealing the inherent limitations of detecting acoustic artifacts.
- **XEUS (Chen et al., 2024b)**: A powerful multilingual speech model whose training data contains singing voices, providing a key component for the speech branch of this work.
- **Insight**: In AI-generated content detection, fusing multiple complementary views is more robust than relying on a single modality, and modular design is key to addressing rapidly evolving generation technologies.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to apply dedicated speech embeddings to AI lyrics detection in the music domain, offering an innovative multi-view fusion approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Very comprehensive evaluation across four dimensions: in-domain, out-of-domain, perturbations, and partly-fake setups.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logic and intuitive chart designs.
- **Value**: ⭐⭐⭐⭐ — Highly practical with direct significance to AI content governance in the music industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MusicDET: Zero-Shot AI-Generated Music Detection](../../ICML2026/audio_speech/musicdet_zero-shot_ai-generated_music_detection.md)
- [\[ICML 2026\] Probing Token Spaces under Generator Shift in AI-Generated Music Detection](../../ICML2026/audio_speech/probing_token_spaces_under_generator_shift_in_ai-generated_music_detection.md)
- [\[ACL 2025\] On the Robust Approximation of ASR Metrics](on_the_robust_approximation_of_asr_metrics.md)
- [\[ACL 2025\] T2A-Feedback: Improving Basic Capabilities of Text-to-Audio Generation via Fine-grained AI Feedback](t2a_feedback_audio_gen.md)
- [\[ACL 2025\] Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking](mitigating_confounding_in_speech-based_dementia_detection_through_weight_masking.md)

</div>

<!-- RELATED:END -->
