---
title: >-
  [Paper Note] In-the-wild Audio Spatialization with Flexible Text-guided Localization
description: >-
  [ACL 2025][Audio & Speech][audio spatialization] This paper proposes the TAS (Text-guided Audio Spatialization) framework, which utilizes flexible text prompts (e.g., 3D spatial location descriptions or relative positions between sound sources) to guide a latent diffusion model in converting monaural audio into binaural audio. A SpatialTAS dataset containing 376K samples is constructed. This method outperforms existing approaches on both simulated and real-recorded data…
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "audio spatialization"
  - "binaural audio"
  - "text-guided"
  - "latent diffusion"
  - "spatial reasoning"
date: 2026-05-08
content_hash: d9960568266bf52f
---

# In-the-wild Audio Spatialization with Flexible Text-guided Localization

**Conference**: ACL 2025  
**arXiv**: [2506.00927](https://arxiv.org/abs/2506.00927)  
**Code**: [GitHub](https://github.com/Alice01010101/TASU)  
**Area**: Audio & Speech  
**Keywords**: audio spatialization, binaural audio, text-guided, latent diffusion, spatial reasoning

## TL;DR
This paper proposes the TAS (Text-guided Audio Spatialization) framework, which utilizes flexible text prompts (e.g., 3D spatial location descriptions or relative positions between sound sources) to guide a latent diffusion model in converting monaural audio into binaural audio. A SpatialTAS dataset containing 376K samples is constructed. This method outperforms existing approaches on both simulated and real-recorded data, and a spatial semantic consistency evaluation model is developed based on Llama-3.1-8B.

## Background & Motivation
**Background**: Audio spatialization maps monaural audio to binaural audio, providing spatial perception for VR/AR and embodied AI. Existing methods mainly rely on visual frame guidance.

**Limitations of Prior Work**: (a) Visual-guided methods are limited by the camera field of view (FOV) and cannot handle sound sources outside the visual field; (b) they lack flexible interactive control, meaning users cannot selectively specify the spatial locations of specific sound sources; (c) high-quality, large-scale stereo data is scarce.

**Key Challenge**: Complex multi-object interactive environments require flexible and controllable spatialization methods, but existing methods either depend on complete visual frames or lack selective control.

**Goal**: To flexibly control audio spatialization using text descriptions (instead of visual frames), supporting 3D position specifications and descriptions of relative relationships between sound sources.

**Key Insight**: Modeling the binaural difference (left-right channel difference) instead of the complete binaural audio reduces the modeling difficulty; a latent diffusion model is used to generate in the mel-spectrogram latent space.

**Core Idea**: To use text descriptions of sound source spatial positions and train a latent diffusion model to learn channel differences, achieving flexible and controllable audio spatialization.

## Method

### Overall Architecture
Given the input monaural audio $A_{\text{mono}} = A_l + A_r$ and text spatial prompts $T_{\text{prompts}}$, the model learns the channel difference $A_{lr} = A_l - A_r$. During inference, the binaural audio is reconstructed using $\hat{A}_l = (A_{\text{mono}} + A_{lr})/2$ and $\hat{A}_r = (A_{\text{mono}} - A_{lr})/2$.

### Key Designs
1. **Channel Difference Learning + Latent Diffusion**:

    - **Function**: Instead of directly generating binaural audio, it learns the latent representation of the difference between the left and right channels.
    - **Mechanism**: A VAE encodes the mel-spectrogram of $A_{lr}$ into a latent space $\rightarrow$ a conditional diffusion model denoises conditioned on text + audio embeddings $\rightarrow$ the VAE decodes and a HiFi-GAN vocoder reconstructs the waveform.
    - **Design Motivation**: Learning channel differences is simpler than learning full binaural audio, and the latent space is computationally more efficient than the waveform space.

2. **Text-Spatial Consistency Enhancement**:

    - **Function**: Fine-tune the text encoder to learn spatial discriminative ability by using flipped channel audio ($A_{rl} = A_r - A_l$) as negative samples.
    - **Mechanism**: A classifier $P$ determines whether the audio difference matches the text description, and a BCE loss $\mathcal{L}_{loc}$ trains the text encoder to capture spatial orientation information.
    - **Design Motivation**: Pre-trained text encoders (such as FLAN-T5) lack spatial-audio alignment training, and channel flipping provides simple yet effective contrastive signals.

3. **LLM Spatial Understanding Evaluation**:

    - **Function**: Fine-tune Llama-3.1-8B as a spatial audio reasoning evaluator to assess the spatial semantic correctness of the generated binaural audio.
    - **Mechanism**: Feed ground-truth and generated binaural audio into the evaluation model to answer spatial questions. A smaller gap in prediction accuracy indicates higher spatial fidelity.

### Loss & Training
Total loss = diffusion noise prediction loss $\mathcal{L}_\theta$ + spatial consistency BCE loss $\mathcal{L}_{loc}$, using Classifier-Free Guidance ($\gamma=2.5$).

## Key Experimental Results

### Main Results (SpatialTAS Test Set)

| Method | FD↓ | FAD↓ | DOA↓ | DE↓ | Direction↓ | Distance↓ |
|------|-----|------|------|-----|-----------|-----------|
| Mono-Mono | 9.03 | 3.67 | 19.66 | 18.12 | 12.79 | 15.33 |
| PseudoBinaural | 7.23 | 2.81 | 6.39 | 4.00 | 10.36 | 12.91 |
| TAS (ours) | **4.93** | **1.44** | **3.07** | **2.45** | **6.99** | **8.16** |

### Ablation Study

| Configuration | FD↓ | DOA↓ | Direction↓ |
|------|-----|------|-----------|
| Full model | 4.93 | 3.07 | 6.99 |
| w/o text | 6.77 | 5.87 | 9.25 |
| w/o Flipper | 5.08 | 4.14 | 8.63 |

### Key Findings
- Text guidance is significantly superior to unconditional generation: omitting text leads to substantial degradation in all metrics.
- The flipped channel enhancement contributes significantly to spatial perception metrics (DOA, Direction).
- Great generalization is achieved on real-recorded data (FAIR-Play, YouTube-Binaural), approaching or outperforming visual-guided methods on traditional metrics such as STFT and ENV.
- Relative position descriptions (e.g., "A is on the left of B") are more difficult to learn than absolute ones but are more practical.

## Highlights & Insights
- Replacing visual frames with text as a spatialization guidance condition represents a major direction shift—it is free from FOV limitations and supports selective control.
- Channel difference modeling is simple and elegant, avoiding the complexity of directly modeling full binaural audio.
- Utilizing LLMs for spatial audio evaluation is a novel paradigm that fills the gap in traditional audio quality metrics which fail to assess spatial correctness.

## Limitations & Future Work
- The training data is simulated (SpatialSoundQA); reverberation and noise in real-world environments might not be fully covered.
- Text descriptions still need to be manually provided by users or generated from visual frames using GPT-4o, meaning a fully end-to-end workflow is not yet realized.
- There is a 10-second audio length limitation, leaving long-duration audio scenarios uncovered.
- Only English text descriptions are evaluated.

## Related Work & Insights
- **vs Visual-Guided Methods (e.g., Mono2Binaural)**: Free from FOV limitations but requires additional text descriptions as inputs.
- **vs Li et al. (2024b)**: This was the first text-guided method but was only annotated on the small-scale FAIR-Play, whereas TAS constructs a large-scale dataset of 376K samples.
- **vs Waveform-space Diffusion**: Latent diffusion is computationally more efficient and produces better generation quality.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of text-guided spatialization and channel-difference latent diffusion is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on both simulated and real data, using both generation and understanding metrics, along with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear descriptions of methods and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Practical application value for VR/AR and embodied AI audio systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Understanding Co-speech Gestures in-the-wild](../../ICCV2025/audio_speech/understanding_co-speech_gestures_in-the-wild.md)
- [\[CVPR 2025\] Towards Open-Vocabulary Audio-Visual Event Localization](../../CVPR2025/audio_speech/towards_open-vocabulary_audio-visual_event_localization.md)
- [\[ACL 2025\] SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models](speechweave_diverse_multilingual_synthetic_text_audio_data_generation_pipeline_f.md)
- [\[ACL 2026\] ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling](../../ACL2026/audio_speech/controlaudio_tackling_text-guided_timing-indicated_and_intelligible_audio_genera.md)
- [\[CVPR 2025\] Object-aware Sound Source Localization via Audio-Visual Scene Understanding](../../CVPR2025/audio_speech/object-aware_sound_source_localization_via_audio-visual_scene_understanding.md)

</div>

<!-- RELATED:END -->
