---
title: >-
  [Paper Note] Bridging the Language Gap: Synthetic Voice Diversity via Latent Mixup for Equitable Speech Recognition
description: >-
  [ICML 2025][Audio & Speech][ASR] This paper proposes LatentVoiceMix, which performs mixup interpolation in the latent space of the speaker style encoder of the voice conversion model Diff-HierVC to generate synthetic speech data with novel voice characteristics for augmenting ASR training. This approach achieves superior WER improvements on the low-resource language Wolof compared to waveform augmentation, spectrogram augmentation, and standard voice conversion.
tags:
  - "ICML 2025"
  - "Audio & Speech"
  - "ASR"
  - "Low-Resource Languages"
  - "Mixup"
  - "Voice Conversion"
  - "Fairness"
date: 2026-05-08
content_hash: 1a121e9fdd9cf649
---

# Bridging the Language Gap: Synthetic Voice Diversity via Latent Mixup for Equitable Speech Recognition

**Conference**: ICML 2025  
**arXiv**: [2511.20534](https://arxiv.org/abs/2511.20534)  
**Code**: None  
**Area**: Speech Recognition / Data Augmentation  
**Keywords**: ASR, Low-Resource Languages, Mixup, Voice Conversion, Fairness

## TL;DR

This paper proposes LatentVoiceMix, which performs mixup interpolation in the latent space of the speaker style encoder of the voice conversion model Diff-HierVC to generate synthetic speech data with novel voice characteristics for augmenting ASR training. This approach achieves superior WER improvements on the low-resource language Wolof compared to waveform augmentation, spectrogram augmentation, and standard voice conversion.

## Background & Motivation

**Background**: Modern ASR systems perform exceptionally well on high-resource languages like English, primarily due to abundant training data. Although there are over 7,000 languages globally, the vast majority are low-resource. Data collection for these languages is difficult and costly, leading to significant language bias in ASR performance.

**Limitations of Prior Work**: Traditional data augmentation methods (such as noise addition, speed perturbation, SpecAugment) improve robustness but do not explicitly enhance the diversity of speaker characteristics in the dataset. While voice conversion-based augmentation (e.g., CycleGAN-VC, StarGAN-VC) can increase speaker diversity, artifacts in the generated audio limit its downstream effectiveness. MixRep performs mixup in the encoder activation layers but is restricted to English.

**Key Challenge**: Low-resource languages require more diverse training data to bridge the performance gap, yet it is neither feasible to collect massive new datasets nor straightforward to generate genuinely diverse speaker characteristics through simple signal-level perturbations.

**Goal**: To generate synthetic speech that preserves the original linguistic content while presenting novel and realistic speaker characteristics, without requiring additional data collection.

**Key Insight**: The authors observe that voice conversion models (such as Diff-HierVC) disentangle audio into independent representations of linguistic content and speaker voice style. This allows for mixup interpolation within the latent space of speaker style vectors to generate new voices within the convex hull of existing speakers, which ensures both realism and increased diversity.

**Core Idea**: To perform a Beta-distribution-weighted convex combination of speaker style vectors in the latent space of the voice conversion model's style encoder, generating novel voices for data augmentation in low-resource ASR.

## Method

### Overall Architecture

The pipeline of LatentVoiceMix: Input is a low-resource language audio dataset $\rightarrow$ denoising $\rightarrow$ extracting and storing a 255-dimensional speaker voice style vector for each audio $\rightarrow$ selecting source audio (providing linguistic content) + randomly selecting target and mixup styles $\rightarrow$ performing convex combination in the latent space $\rightarrow$ generating synthetic audio via Diff-HierVC $\rightarrow$ post-denoising $\rightarrow$ inheriting the original transcripts.

### Key Designs

1. **Speaker Style Extraction and Storage**:

    - **Function**: Extracts a fixed-length speaker style representation for each audio clip in the corpus.
    - **Mechanism**: Uses the style encoder of Diff-HierVC to encode each audio segment into a 255-dimensional vector. This vector captures time-invariant voice features (pitch, timbre, speaking style, etc.) independent of linguistic content, and is systematically stored in the file system for future reuse.
    - **Design Motivation**: Decouples style extraction from the synthesis process, allowing the mixup operation to be performed efficiently in vector space rather than on raw waveforms.

2. **Latent Space Mixup Strategy**:

    - **Function**: Generates new speaker voices in the latent space of the style encoder.
    - **Mechanism**: Randomly selects two style vectors $\mathbf{t}_{\text{target}}$ and $\mathbf{t}_{\text{mixup}}$ that differ from the source speaker, and computes their convex combination as $\mathbf{t}_{\text{mixed}} = \lambda \mathbf{t}_{\text{target}} + (1-\lambda) \mathbf{t}_{\text{mixup}}$, where $\lambda \sim \text{Beta}(0.5, 0.5)$.
    - **Design Motivation**: The Beta(0.5, 0.5) distribution is U-shaped, favoring extreme values close to 0 or 1. This biases the mixed style toward one of the source speakers, thereby maintaining naturalness while increasing diversity. Interpolating within the convex hull ensures that the generated styles do not deviate too far from the true speaker distribution.

3. **Post-processing Denoising**:

    - **Function**: Denoises the synthetic audio.
    - **Mechanism**: Uses the noisereduce package to perform final denoising on the generated synthetic audio, removing residual artifacts introduced during the voice conversion process.
    - **Design Motivation**: Ablation experiments show that removing the post-denoising step increases the WER from 0.202 to 0.214, indicating that post-processing makes a significant contribution to final performance.

### Loss & Training

This paper does not modify the training loss of the ASR model, but instead introduces improvements at the data augmentation level. The augmented data is directly applied to the standard ASR training pipeline (training NeMo from scratch for 50 epochs or fine-tuning Whisper-tiny for 4 epochs).

## Key Experimental Results

### Main Results (AN4 Small English Dataset)

| Augmentation Method | WER ↓ |
|---------|-------|
| No Augmentation | 0.785 |
| Waveform Augmentation (+33%) | 0.436 |
| Voice Conversion Augmentation (+33%) | 0.424 |
| **Mixup Augmentation (+33%)** | **0.339** |

### Whisper Fine-tuning Wolof Comparison

| Augmentation Method | WER ↓ | SpeechMOS |
|---------|-------|-----------|
| No Augmentation | 0.283 | 2.661 |
| Spectrogram Augmentation (SpecAugment) | 0.242 | n/a |
| Waveform Augmentation | 0.217 | 2.117 |
| Voice Conversion Augmentation | 0.215 | 2.710 |
| **Mixup Augmentation (Ours)** | **0.202** | 2.243 |

### Ablation Study

| Configuration | WER ↓ |
|------|-------|
| No Post-denoising, Source=Target (8h) | 0.235 |
| No Post-denoising, Source=Target (16h) | 0.221 |
| Mixup w/ 3 Styles (16h) | 0.221 |
| No Post-denoising (16h) | 0.214 |
| **Full Mixup (16h)** | **0.202** |

### Key Findings
- Among all augmentation strategies, latent space mixup consistently achieves the lowest WER, with a particularly pronounced advantage on the low-resource language (Wolof).
- In multilingual experiments, after augmenting Wolof data from 8 hours to 24 hours, the Wolof-English WER gap was reduced from 0.234 to 0.175, while English performance also improved slightly ($0.562 \rightarrow 0.550$).
- PCA analysis shows that the style distribution generated by mixup is closer to that of the original speakers, whereas waveform augmentation generates styles with larger variance that deviate from the true distribution.

## Highlights & Insights
- **Performing mixup in the latent space of the style encoder rather than at the waveform/spectrogram level** is the core innovation of this paper—it preserves linguistic content while embedding diversity in a semantically rich space, which is more effective than perturbations at the raw signal level.
- The choice of Beta(0.5, 0.5) is clever—its U-shaped distribution steers most mixed results closer to one of the two source speakers rather than meaningless intermediate states, which helps in generating more natural voices.
- The paper addresses the low-resource language challenge from the perspective of equity, closely linking technical contribution with social impact.

## Limitations & Future Work
- Validated only on a single low-resource language (Wolof), lacking generalization proof on more languages (such as other African or Southeast Asian languages).
- The Diff-HierVC model itself was primarily trained on English data, and the reliability of its style disentanglement on non-English languages is not sufficiently discussed.
- The data scale is relatively small (only 16 hours for Wolof), and whether the marginal returns of mixup diminish at a larger scale remains unknown.
- SpeechMOS indicates that the quality of mixup-generated audio (2.243) is lower than both the original data (2.661) and voice conversion (2.710), suggesting room for improvement in synthesis quality.
- Lacks comparison with more modern data augmentation methods (such as Codec-based speech synthesis or TTS-based augmentation).

## Related Work & Insights
- **vs SpecAugment**: SpecAugment applies masking at the spectrogram level, which enhances transformation robustness rather than speaker diversity. The proposed method achieves a WER of 0.202 vs. 0.242 for SpecAugment in Whisper fine-tuning, demonstrating a clear gap.
- **vs MixRep**: MixRep performs mixup in the activation layers of the ASR encoder and has only been validated in low-resource English scenarios. This work operates in the style space of the voice conversion model, which more directly increases speaker diversity.
- **vs Voice Conversion Augmentation (e.g., StarGAN-VC)**: Traditional voice conversion directly converts styles to targeted real speakers. In contrast, this work creates "non-existent" novel speakers via mixup, expanding the speaker style space more effectively.

## Rating
- Novelty: ⭐⭐⭐⭐ First to perform mixup in the latent space of a voice conversion style encoder for ASR augmentation.
- Experimental Thoroughness: ⭐⭐⭐ Thorough ablations but limited language coverage, lacking comparisons with more modern baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear description of the methodology, with well-organized flowcharts and experimental tables.
- Value: ⭐⭐⭐ Simple and practical method, but the application scenario is somewhat narrow, requiring validation on more languages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](../../ACL2026/audio_speech/closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[NeurIPS 2025\] Adapting Speech Language Model to Singing Voice Synthesis](../../NeurIPS2025/audio_speech/adapting_speech_language_model_to_singing_voice_synthesis.md)
- [\[NeurIPS 2025\] Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space](../../NeurIPS2025/audio_speech/efficient_speech_language_modeling_via_energy_distance_in_continuous_latent_spac.md)
- [\[ICLR 2026\] Latent Speech-Text Transformer](../../ICLR2026/audio_speech/latent_speech_text_transformer.md)
- [\[ICML 2025\] Do Not Mimic My Voice: Speaker Identity Unlearning for Zero-Shot Text-to-Speech](do_not_mimic_my_voice_speaker_identity_unlearning_for_zero-shot_text-to-speech.md)

</div>

<!-- RELATED:END -->
