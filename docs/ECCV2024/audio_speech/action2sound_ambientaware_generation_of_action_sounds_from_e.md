---
title: >-
  [Paper Note] Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos
description: >-
  [ECCV 2024][Audio & Speech][Video-to-Audio Generation] AV-LDM is proposed to implicitly disentangle foreground action sounds and background ambient sounds by introducing audio from different time segments of the same video as an ambient sound condition during training. Combined with retrieval-augmented generation (RAG) to select appropriate ambient sound conditions during inference, it significantly outperforms existing methods on Ego4D and EPIC-KITCHENS.
tags:
  - "ECCV 2024"
  - "Audio & Speech"
  - "Video-to-Audio Generation"
  - "Ambient Sound Disentanglement"
  - "Egocentric Video"
  - "Latent Diffusion Models"
  - "Retrieval-Augmented Generation"
date: 2026-05-08
content_hash: 9ca7579bfdbc5c81
---

# Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos

**Conference**: ECCV 2024  
**arXiv**: [2406.09272](https://arxiv.org/abs/2406.09272)  
**Code**: [https://github.com/ChanganVR/action2sound](https://github.com/ChanganVR/action2sound)  
**Area**: Video Understanding / Audio Generation  
**Keywords**: Video-to-Audio Generation, Ambient Sound Disentanglement, Egocentric Video, Latent Diffusion Models, Retrieval-Augmented Generation  

## TL;DR
AV-LDM is proposed to implicitly disentangle foreground action sounds and background ambient sounds by introducing audio from different time segments of the same video as an ambient sound condition during training. Combined with retrieval-augmented generation (RAG) to select appropriate ambient sound conditions during inference, it significantly outperforms existing methods on Ego4D and EPIC-KITCHENS.

## Background & Motivation
Video-to-audio generation is an important task with application scenarios including film foley, VR game sound effects, and post-dubbing for text-to-video generation. Existing methods (e.g., Spec-VQGAN, Diff-Foley, REGNET) implicitly assume a complete correspondence between video and audio. However, in real-world in-the-wild videos, a large amount of sound originates from off-screen (e.g., air conditioner humming, distant conversations, traffic noise), which is irrelevant or weakly correlated with the visual content. These **background ambient sounds** often have much higher energy in the spectrogram than brief action sounds, causing the model to be dominated by ambient noise during learning, leading to either uncontrolled ambient noise or complete neglect of true action sounds during inference. Although prior work REGNET attempted to generate only visually relevant sounds via a bottleneck design, it relies on a predefined classification taxonomy and cannot generalize to in-the-wild data.

## Core Problem
**How to enable the model to distinguish and separately generate foreground action sounds and background ambient sounds using only naturally mixed audio (without foreground/background separation annotations) in-the-wild training data?** The core difficulties of this problem lie in: (1) action sounds are usually transient and weak in energy, while ambient sounds are continuous and high in energy, causing the training loss to be dominated by ambient noise; (2) many ambient sounds are unrelated to the visuals (e.g., recording equipment noise, off-screen speech), and forcing visual-to-audio inference leads to hallucinations; (3) blind source separation for in-the-wild scenarios remains an unsolved open problem, and direct application of noise reduction algorithms yields poor results.

## Method

### Overall Architecture
AV-LDM is an audio-visual latent diffusion model. The input is a silent video, and the output is the corresponding audio waveform. The overall pipeline consists of three stages:
1. **Audio-Visual Representation Learning** (Pre-training): Timesformer and AST are used as video and audio encoders, respectively. The audio-visual embedding space is aligned using InfoNCE contrastive learning, while training the AV-Sim scoring model for retrieval.
2. **AV-LDM Training**: Target audio is converted into a mel-spectrogram and encoded into the latent space via a pre-trained VAE. Video features and **neighboring audio conditions** are injected into the UNet diffusion model through cross-attention, which is trained with standard denoising objectives.
3. **Inference**: Given a silent video, the AV-Sim model retrieves the most semantically similar audio from the training set as the ambient sound condition (or a low-ambient sound condition is used to achieve action-focused generation). The final waveform is generated via LDM sampling, VAE decoding, and a HiFi-GAN vocoder.

### Key Designs

1. **Neighboring Audio Condition for Ambient Disentanglement**: This is the core insight of this work. It leverages the temporal stationarity of ambient sounds (where ambient sounds from different time intervals in the same long video are highly similar). During training, a temporally adjacent but non-overlapping audio clip $A_n$ is randomly sampled from the same long video as an additional condition. Since $A_n$ and the target audio $A$ share ambient sound characteristics but contain different action sounds, the model only needs to "borrow" ambient information from $A_n$, while learning action sound cues from the video. This elegantly avoids explicit source separation and allows the model to learn disentanglement implicitly. Regarding concerns that $A_n$ might contain duplicate action sounds: statistical analysis based on (verb, noun) classification reveals that the probability of semantic repetition is only 9%, and temporal alignment is virtually impossible; therefore, the model cannot "cheat."

2. **Retrieval-Augmented Generation (RAG)**: While neighboring audio can be extracted from the same video during training, this condition is unavailable at inference. The paper designs two inference modes:

    - **Joint Action-Ambient Generation**: The pre-trained AV-Sim model is used to retrieve the audio from the training set that is visually most similar to the input video as $A_n$, capturing scene-relevant ambient sounds (e.g., outdoor bird singing).
    - **Action-Focused Generation**: An audio clip with a low ambient noise level is used as $A_n$, guiding the model to generate action sounds only while minimizing ambient noise. Using an all-zero vector yields poor results (as it deviates too far from the training distribution), whereas using a low-ambient audio clip works well.

3. **Audio-Visual Condition Injection Mechanism**: Video features $c_v \in \mathbb{R}^{16 \times 768}$ (extracted by Timesformer) and audio condition features $c_a \in \mathbb{R}^{24 \times 768}$ (encoded by VAE and projected via MLP) are concatenated and combined with learnable positional encodings, serving as the key/value for the UNet cross-attention layers. During training, $c_v$ is replaced with zero with a probability of 0.2 (supporting classifier-free guidance), but $c_a$ is always kept, as even a small dropout rate degrades performance.

### Loss & Training
- **Diffusion Denoising Objective**: $\mathcal{L} = \mathbb{E}_{t, z_0, \epsilon_t} \| \epsilon_t - \epsilon_\theta(x_t, t, c_v, c_a) \|^2$
- **Classifier-Free Guidance**: During inference, $\tilde{\epsilon}_t = \omega \cdot \epsilon_\theta(z_t, t, c_v, c_a) + (1-\omega) \cdot \epsilon_\theta(z_t, t, \varnothing, \varnothing)$, with a guidance scale $\omega = 6.5$.
- **Contrastive Learning Pre-training**: InfoNCE loss aligns audio-visual embeddings, with the video encoder initialized using EgoVLP pre-trained weights.
- VAE and LDM weights are initialized using pre-trained Stable Diffusion weights; the VAE is frozen, and the LDM is trained for 8 epochs with a batch size of 720.
- The HiFi-GAN vocoder is trained from scratch on mixed data from Ego4D + EPIC-KITCHENS + AudioSet.
- **Ego4D-Sounds Dataset Construction**: From 2113 hours of Ego4D videos with audio, after a 4-stage filtering process (removing silence, social scenarios, speech/music only, and low-energy segments), 1.2 million 3-second audio-visual action clips are obtained.

## Key Experimental Results

### Ego4D-Sounds Test Set

| Method | FAD ↓ | AV-Sync (%) ↑ | CLAP ↑ |
|------|-------|---------------|--------|
| Ground Truth | 0.0000 | 77.69 | 0.2698 |
| Retrieval | 1.8353 | 11.84 | 0.0335 |
| REGNET | 8.3800 | 3.90 | 0.0099 |
| Spec-VQGAN | 3.9017 | 7.12 | 0.0140 |
| Diff-Foley | 3.5608 | 5.98 | 0.0346 |
| **AV-LDM (Ours)** | **0.9999** | **45.74** | **0.1435** |

### EPIC-KITCHENS

| Method | FAD ↓ | AV-Sync (%) ↑ |
|------|-------|---------------|
| Retrieval | 1.9618 | 13.84 |
| Diff-Foley | 3.4649 | 14.19 |
| Ours w/o cond | 1.4731 | 50.42 |
| **AV-LDM (Ours)** | **1.3200** | **59.26** |

### Human Evaluation

| Method | Action Sound Quality ↑ | Minimal Ambient Noise ↑ |
|------|------------|------------|
| Retrieval | 12.5% | 12.5% |
| Diff-Foley | 47.5% | 12.5% |
| AV-LDM w/o cond | 55.0% | 17.5% |
| AV-LDM (action-focused) | 60.0% | **97.5%** |
| **AV-LDM (action-ambient)** | **72.5%** | 22.5% |

### Ablation Study
- **Removing Neighboring Audio Condition (w/o cond)**: FAD increases from 0.9999 to 1.4681, proving the core role of ambient sound condition training.
- **Adding Denoising Preprocessing (w/o cond + denoiser)**: AV-Sync plummets to 1.09%, and CLAP drops to almost zero, indicating that existing denoising algorithms are completely unsuitable for in-the-wild data.
- **Random Retrieval Condition (w/ random test cond)**: FAD is 1.0635 vs. 0.9999 with retrieval condition, and AV-Sync is 28.74% vs. 45.74%, validating the effectiveness of RAG retrieval.
- **Removing Pre-trained Vocoder (w/o vocoder)**: FAD increases to 4.9282, indicating that Griffin-Lim spectrogram inversion yields much lower quality than HiFi-GAN.
- **Robustness to Retrieval Pool Size**: Scaling down the pool from the full dataset to only 100 samples barely increases FAD from 1.01 to 1.12, showing that the model is robust to the pool size.
- **Frame Rate Ablation**: 5 FPS is sufficient; higher frame rates (8/10 FPS) bring no significant improvement.

## Highlights & Insights
- **Extremely Elegant Disentanglement Concept**: Instead of explicit source separation, it achieves foreground/background sound disentanglement simply by using "neighboring audio from different time segments of the same video" as a conditioning input. This insight is model-agnostic and can be transferred to any conditional generation architecture.
- **Emergent Ability of Controllable Generation**: The model has never seen clean action sound annotations during training. However, by adjusting the ambient noise level of the conditioning audio, it can smoothly control the energy level of the generated ambient sound—showing an emergent capability.
- **Large-scale Dataset Contribution**: The paper constructs Ego4D-Sounds (1.2M segments), which is an order of magnitude larger than existing datasets and includes action narrative text annotations.
- **Transfer from Stable Diffusion**: It cleverly reuses the VAE and UNet weights of SD to process audio spectrograms (by duplicating mel-spectrograms into 3 channels), significantly accelerating training.
- **Cross-domain Generalization**: Zero-shot testing on VR cooking game videos demonstrates the model's potential for cross-domain generalization.

## Limitations & Future Work
- **Mono 16kHz**: Audio quality is limited. High-fidelity audio generation (e.g., 44.1kHz stereo) remains to be explored.
- **Fixed 3-Second Duration**: Incapable of handling longer action sequences or continuous action streams.
- **Ambient Sound Stationarity Assumption**: In scenarios with rapid scene changes (e.g., walking through different rooms), the ambient sound in neighboring audio may vary significantly, violating the assumption.
- **Domain Gap in VR Game Scenes**: While preliminary results are promising, the visual feature differences between synthetic scenes and real egocentric videos are large, requiring further domain adaptation.
- **Evaluation Metric Limitations**: The upper bound of AV-Sync accuracy is only 77.69% (ground truth), indicating that the metric itself is noisy; the action-focused mode lacks objective evaluation (due to the absence of isolated ground truth).
- **Unexplored Text Conditioning**: Although Ego4D contains action narratives, the model does not utilize text as a generation condition, which could potentially enhance semantic alignment.

## Related Work & Insights
- **vs. Diff-Foley**: Both use the LDM framework, but Diff-Foley does not address ambient noise, resulting in poor generation quality on in-the-wild data. Additionally, Diff-Foley's video features are not tailored for egocentric videos. AV-LDM achieves a 72% lower FAD and a 7x higher AV-Sync on Ego4D-Sounds compared to Diff-Foley.
- **vs. REGNET**: REGNET also attempts to generate only visually relevant sounds but heavily relies on predefined sound categories and bottleneck designs, failing completely on in-the-wild data (FAD 8.38, AV-Sync only 3.9%). AV-LDM's implicit disentanglement is much more generalizable.
- **vs. CondFoleyGen**: CondFoleyGen also uses extra video conditions to control generated sound features, but with a different purpose (modifying sound attributes vs. foreground-background disentanglement) and does not handle the ambient noise issue in-the-wild.
- **"Neighboring Condition" as a General Disentanglement Paradigm**: The idea of using temporally adjacent samples as conditions to disentangle continuous signals from transient signals can be extended to scenarios like video dehazing (continuous haze vs. transient objects) and medical signal denoising (baseline wander vs. transient events).
- **Application of RAG in Multimodal Generation**: Retrieval augmentation is not only useful for text generation but is also highly effective for conditional generation in modalities like audio and video.
- **Ego4D Ecosystem**: Ego4D-Sounds extends the application scope of Ego4D. Generative tasks (not just understanding) are emerging as a new direction in egocentric video research.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of using neighboring audio conditions for disentanglement is simple and effective, though the base architecture (LDM + cross attention) follows a standard design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely complete, featuring multi-dataset evaluation, comprehensive ablation studies, human evaluation, controllability validation, and cross-domain demos.
- Writing Quality: ⭐⭐⭐⭐⭐ The problem definition is clear, the motivation is natural and smooth, and the diagrams are intuitive and easy to understand.
- Value: ⭐⭐⭐⭐ It pioneers the direction of ambient-aware audio generation and makes a substantial dataset contribution, though the application scenario is somewhat niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Listen to Look into the Future: Audio-Visual Egocentric Gaze Anticipation](listen_to_look_into_the_future_audio-visual_egocentric_gaze_anticipation.md)
- [\[ECCV 2024\] Latent-INR: A Flexible Framework for Implicit Representations of Videos with Discriminative Semantics](latent-inr_a_flexible_framework_for_implicit_representations_of_videos_with_disc.md)
- [\[ICML 2025\] Teaching Physical Awareness to LLMs through Sounds](../../ICML2025/audio_speech/teaching_physical_awareness_to_llms_through_sounds.md)
- [\[ICLR 2026\] Aurelius: Relation Aware Text-to-Audio Generation At Scale](../../ICLR2026/audio_speech/aurelius_relation_aware_text-to-audio_generation_at_scale.md)
- [\[ICLR 2026\] WearVox: An Egocentric Multichannel Voice Assistant Benchmark for Wearables](../../ICLR2026/audio_speech/wearvox_an_egocentric_multichannel_voice_assistant_benchmark_for_wearables.md)

</div>

<!-- RELATED:END -->
