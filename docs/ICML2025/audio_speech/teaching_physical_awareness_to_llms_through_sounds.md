---
title: >-
  [Paper Note] Teaching Physical Awareness to LLMs through Sounds
description: >-
  [ICML 2025][Audio & Speech][Physical awareness] Proposes the ACORN framework, which teaches LLMs to understand physical world phenomena from sound. It generates large-scale training data through a physics-based acoustic channel simulator, coupled with an audio encoder that captures both magnitude and phase information.
tags:
  - "ICML 2025"
  - "Audio & Speech"
  - "Physical awareness"
  - "Acoustic channel"
  - "LLM audio understanding"
  - "Doppler effect"
  - "Acoustic simulation"
date: 2026-05-08
content_hash: 37320a9f01324aa7
---

# Teaching Physical Awareness to LLMs through Sounds

**Conference**: ICML 2025  
**arXiv**: [2506.08524](https://arxiv.org/abs/2506.08524)  
**Code**: None  
**Area**: Audio & Speech  
**Keywords**: Physical awareness, Acoustic channel, LLM audio understanding, Doppler effect, Acoustic simulation

## TL;DR

Proposes the ACORN framework, which teaches LLMs to understand physical world phenomena from sound. It generates large-scale training data through a physics-based acoustic channel simulator, coupled with an audio encoder that captures both magnitude and phase information.

## Background & Motivation

Large language models (LLMs) have made remarkable progress in text and multimodal understanding, but they fundamentally lack physical awareness—the ability to understand physical world phenomena. Humans intuitively perceive their environment through sound: the Doppler effect indicates whether a vehicle is approaching or moving away, multipath reflections reveal whether we are indoors or outdoors, and binaural hearing allows us to localize the source of a sound. However, existing audio LLMs primarily focus on speech recognition and audio content understanding, failing to extract physical attributes (such as motion states and spatial relationships) from sound.

This poses practical safety hazards: for instance, a voice-controlled car might execute a "roll down the window" command from someone outside the vehicle because it cannot determine the physical origin of the sound. The key challenge lies in the data: collecting and annotating large-scale physical acoustic data is costly and nearly unfeasible, as physical phenomena like the Doppler effect and multipath reflections cannot be easily labeled by humans.

The key insight of this paper is that received sound can be decomposed into two independent components: the sound source and the physical channel ($y = h \circledast s$). Consequently, training data can be synthesized by convolving real sound sources with simulated physical channels, thereby bypassing the bottleneck of data collection.

## Method

### Overall Architecture

The ACORN framework comprises three core components: (1) a physics-based acoustic channel simulator that generates diverse channel impulse responses (CIR); (2) an audio encoder that jointly captures both magnitude and phase information; and (3) an end-to-end architecture connected to an LLM. The system generates 1 million ⟨Audio, Question, Answer⟩ tuples (AQA-PHY dataset) using the simulator to conduct supervised fine-tuning of the LLM.

### Key Designs

1. **Acoustic Channel Simulator**: Models five independent components based on signal processing theory: line-of-sight (LOS) direct path, early reflections, reverberation, the Doppler effect, and microphone array reception. Each component is independently controllable and randomized. The CIR is modeled as $h(\tau) = \sum_{i=0}^{N} \alpha_i \delta(\tau - \tau_i) + R(\tau)$, where $R(\tau)$ represents the reverberation tail. The Doppler effect is modeled via time-varying delay as $h(t, \tau) = \delta(\tau - \frac{d_0 + v \cdot t}{c})$. Different tasks use different configurations: target parameters are precisely controlled, while non-critical parameters are randomized to maximize channel diversity. **Design Motivation**: Component-level modeling is more flexible and scalable than environment-level reconstruction.

2. **Magnitude-Phase Audio Encoder**: Unlike traditional encoders that focus only on magnitude (e.g., Whisper's Mel spectrogram), the ACORN encoder simultaneously extracts three components of STFT: magnitude $M(f,t) = |X(f,t)|$, phase sine $\sin(\angle X(f,t))$, and phase cosine $\cos(\angle X(f,t))$. Using sin/cos instead of raw phase angles avoids the phase wrapping issue from $\pi$ to $-\pi$. Each of the three components goes through a 3x3 1D convolution (128 -> 1280 channels) + GELU activation. After concatenation (3840 channels), they are fused and down-projected to 1280 channels via two layers of 3x3 convolutions. Sinusoidal positional encodings are added to preserve temporal context, and finally, a 32-layer Transformer outputs audio tokens. The total parameters are approximately 0.65B. The magnitude part is initialized from Whisper-large-v2 to leverage pre-trained magnitude representations, while the phase subnetwork is trained from scratch.

    - **Audio Preprocessing**: 16kHz sampling, STFT window length 254 (yielding 128 frequency bins), with a hop size of 10ms. It preserves the complete spectral resolution instead of converting to Mel spectrograms to keep the fine-grained characteristics of physical signals.
    - **Design Motivation**: Physical effects (e.g., Doppler shift, multipath delay) are primarily manifested in subtle phase relationships, which are uncapturable by magnitude alone. Experiments demonstrate that introducing phase information reduces distance estimation errors by 7 times.

3. **Modular Task Configurations**: Five acoustic perception tasks: LOS detection (determining the presence of a line-of-sight path), Doppler estimation (estimating frequency shift), Direction of Arrival (DoA) estimation (utilizing TDoA $\tau_\theta = d\cos(\theta)/c$), multipath analysis (assessing the level of reverberation), and distance estimation (based on echo-return analysis). Each task controls physical parameters by selectively enabling/disabling simulation components.

### Loss & Training

- Employs the standard next-token prediction loss, targeting the answer text as labels.
- The audio encoder is initialized from Whisper-large-v2 (to leverage pre-trained magnitude representations), while the phase subnetwork is trained from scratch.
- The LLM is fine-tuned using LoRA to reduce training resource overhead and leverage its pre-trained language capabilities.
- Audio tokens are mapped to the LLM's word embedding dimension via a linear projection layer, wrapped with `<soa>` and `<eoa>` tokens.
- 4 × A100 GPUs, batch size of 32, for 7 epochs, with a total training time of approximately 61 hours.
- Generates 200k closed-ended QA pairs + 10k open-ended QA pairs for each task.

## Key Experimental Results

### Main Results

| Task | Metric | ACORN + Qwen2 | Whisper + Qwen2 | Gain |
| :--- | :--- | :--- | :--- | :--- |
| LOS Detection | BCA↑ | 0.924 | 0.881 | +4.3pp |
| Doppler Estimation | MAE_f↓ | 0.181 | 1.042 | 82.6% reduction |
| DoA Estimation | MAE_t↓ | 0.907 | 2.716 | 66.6% reduction |
| Multipath Analysis | TCA↑ | 0.903 | 0.848 | +5.5pp |
| Distance Estimation | REP↓ | 1.599 | 10.609 | 84.9% reduction |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| With LOS vs Without LOS (Multipath TCA) | 0.895 vs 0.912 | Direct LOS signals mask multipath features. |
| With Doppler vs Without Doppler (LOS Detection BCA) | 0.912 vs 0.936 | Doppler introduces signal distortion. |
| SNR <10dB vs >40dB (Distance MAE) | 5.33 vs 0.80 | High SNR significantly improves accuracy. |
| Merged vs Sole Training | See table for detail | Joint training is close to independent training. |

### Key Findings

- Phase information is crucial for physical awareness: The ACORN encoder consistently outperforms the magnitude-only Whisper across all tasks, especially in distance estimation, where performance is improved by 7x.
- The method is model-agnostic: Both Llama3.1-8B and Qwen2-7B obtain consistent improvements when paired with the ACORN encoder.
- Open-ended QA performance is reasonable, with the model demonstrating the ability to explain physical phenomena in natural language and perform multi-step computations.
- Zero-shot transfer to real-world environments is feasible: reaching 0.870 on LOS detection and 0.925 on DoA estimation in in-vehicle environments.
- The model possesses good robustness to various acoustic interferences, experiencing limited performance degradation.

## Highlights & Insights

- The decomposition concept of sound = source × physical channel is elegant, turning an otherwise impossible data collection problem into a manageable simulation problem.
- The incorporation of phase information is a key innovation, revealing crucial physical information that is discarded by traditional audio encoders (based on Mel spectrograms).
- Opens up a completely new research direction of "LLM physical awareness," which differs from visual or textual physical reasoning.
- The componentized simulator design allows each physical phenomenon to be independently controlled and studied.
- Validation in real vehicle scenarios reinforces the practical application value.

## Limitations & Future Work

- Only supports single-turn conversation, lacking the capability for multi-turn interactive reasoning.
- The scale of real-world experiments is limited, restricted only to a few scenarios in a single vehicle.
- Although the simulator is highly diverse, a discrepancy still exists compared to real acoustic environments (domain shift problem).
- Tasks are modeled independently, without exploring the relationships and joint reasoning among physical phenomena.
- Reasoning enhancement techniques, such as Chain-of-Thought (CoT), have not yet been introduced.
- Only LLM scales of 7-8B have been tested; it remains unknown whether larger models can learn physical reasoning better.

## Related Work & Insights

- Difference from BAT (spatial audio understanding): ACORN directly extracts phase from a single channel, avoiding the quadratic growth associated with pairwise calculations.
- Complementary to physical reasoning study such as NEWTON: those reason about physics through text/vision whereas ACORN does so through acoustics.
- Traditional methods in the acoustic sensing field (ToA, TDoA, FMCW) rely on hand-crafted features; ACORN replaces them with end-to-end learning.
- Traditional Audio LLMs like AudioGPT, Pengi, and Qwen-Audio focus on semantic understanding, whereas ACORN is the first to focus on physical attribute understanding.
- Insight: Similar "physical channel separation + simulation" methodologies can be generalized to areas like wireless communication and radar signals.
- The vehicle safety scenario (determining if voice commands originate from inside or outside the vehicle) is an extremely compelling application case.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Opens a completely new direction; audio-to-physical perception is highly unique.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across five tasks, including real vehicle experiments, though the scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, thorough explanation of physical principles, and well-designed figures/tables.
- **Value**: ⭐⭐⭐⭐ Demonstrates the feasibility of LLM physical awareness, offering a new path for embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning](../../AAAI2026/audio_speech/do_llms_feel_teaching_emotion_recognition_with_prompts_retrieval_and_curriculum_.md)
- [\[ACL 2025\] Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking](../../ACL2025/audio_speech/mitigating_confounding_in_speech-based_dementia_detection_through_weight_masking.md)
- [\[ICCV 2025\] Everything is a Video: Unifying Modalities through Next-Frame Prediction](../../ICCV2025/audio_speech/everything_is_a_video_unifying_modalities_through_next-frame_prediction.md)
- [\[ECCV 2024\] Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos](../../ECCV2024/audio_speech/action2sound_ambientaware_generation_of_action_sounds_from_e.md)
- [\[NeurIPS 2025\] AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound](../../NeurIPS2025/audio_speech/audsemthinker_enhancing_audio-language_models_through_reasoning_over_semantics_o.md)

</div>

<!-- RELATED:END -->
