---
title: >-
  [Paper Note] E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis
description: >-
  [NEURIPS2025][Audio & Speech][voice cloning defense] E2E-VGuard is a proactive defense framework against voice cloning threats in LLM-based end-to-end speech synthesis. It disrupts timbre via encoder ensemble perturbatio…
tags:
  - "NEURIPS2025"
  - "Audio & Speech"
  - "voice cloning defense"
  - "adversarial examples"
  - "speech synthesis"
  - "psychoacoustic model"
  - "LLM-based TTS"
date: 2026-05-08
content_hash: 3b375352ffed4a80
---

# E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis

**Conference**: NEURIPS2025  
**arXiv**: [2511.07099](https://arxiv.org/abs/2511.07099)  
**Code**: [wxzyd123/e2e-vguard](https://wxzyd123.github.io/e2e-vguard/)  
**Area**: Audio & Speech  
**Keywords**: voice cloning defense, adversarial examples, speech synthesis, psychoacoustic model, LLM-based TTS

## TL;DR

E2E-VGuard is a proactive defense framework against voice cloning threats in LLM-based end-to-end speech synthesis. It disrupts timbre via encoder ensemble perturbation, interferes with pronunciation recognition via adversarial attacks on ASR systems, and ensures imperceptibility through a psychoacoustic model. Effectiveness is validated across 19 TTS models and 7 ASR systems.

## Background & Motivation

- LLM-based speech synthesis systems (e.g., CosyVoice, GPT-SoVITS) have achieved human-level synthesis quality, but also introduce serious voice cloning fraud risks (e.g., telecom scams).
- Existing defenses (AntiFake, AttackVC, POP, SafeSpeech) primarily target traditional DNN-based TTS or assume manually annotated transcripts, and cannot handle two emerging scenarios:
    1. **Production-grade LLM-based TTS**: audio is encoded into discrete tokens for LLM input, differing fundamentally from traditional continuous feature extraction.
    2. **End-to-end (E2E) scenario**: commercial APIs (e.g., ByteDance, Alibaba) accept only audio input and internally use ASR to transcribe text automatically, requiring no manual annotation from attackers.
- In practice, attackers collect audio from platforms such as YouTube/Bilibili without accompanying transcripts and must rely on ASR for automatic recognition—making the ASR module itself a novel exploitable defense point.

## Core Problem

How to simultaneously defend against voice cloning along both **timbre** and **pronunciation** dimensions within an end-to-end speech synthesis pipeline, such that TTS models produce unrecognizable cloned speech while ensuring that the perturbations added to protected audio remain imperceptible to human listeners.

## Method

The overall optimization objective of E2E-VGuard is:

$$\mathcal{L}(x') = \mathcal{L}_{asr}(x') + \alpha \cdot \mathcal{L}_{fea}(x') + \beta \cdot \mathcal{L}_{psy}(x')$$

where $x'$ denotes the protected audio, $\alpha=500$, $\beta=5 \times 10^{-3}$, perturbation budget $\epsilon = 8/255$, and optimization runs for 500 iterations.

### 1. Timbre Prevention

An **encoder ensemble** strategy is adopted, employing 6 heterogeneous encoders (VITS/GSV posterior encoder, MFCC, WavLM, CAM++, StyleTTS2 style encoder) to extract audio features and improve cross-model generalizability.

- **Untargeted protection**: maximizes the feature distance between the original audio $x$ and the protected audio $x'$:

$$\mathcal{L}_{fea}(x') = \sum_{i=1}^{k} \text{CS}(E_i(x), E_i(x')) + \text{CS}(M(x), M(x'))$$

- **Targeted protection**: steers audio features toward a pre-selected least-similar speaker $x_t$:

$$\mathcal{L}_{fea}(x') = -\sum_{i=1}^{k} [\text{CS}(E_i(x_t), E_i(x')) + \text{CS}(M(x_t), M(x'))]$$

An MFCC feature extractor is specifically introduced to counter the discrete tokenization encoding characteristic of LLM-based TTS, disrupting the prosody and intonation information acquired by the LLM component.

### 2. Pronunciation Prevention

Adversarial examples are used to attack ASR systems, causing them to misrecognize the protected audio as a designated target transcript:

$$\mathcal{L}_{asr}(x') = \mathcal{F}(\text{ASR}(x'), y_t)$$

- **Targeted attacks** (rather than gibberish) are employed to generate readable but erroneous transcripts, reducing attacker suspicion.
- Target text selection strategy: the target speaker's transcript is used in targeted timbre protection; a different text of equal length is selected in untargeted timbre protection.
- Incorrect text–audio pairs disrupt text-to-pronunciation alignment learning in TTS models (e.g., monotonic alignment search in VITS).

### 3. Psychoacoustic Model

Frequency masking effects are leveraged to ensure perturbation imperceptibility:

$$\mathcal{L}_{psy}(x') = \frac{1}{F} \sum_{f=1}^{F} \max(0, p_{x'-x}(f) - \theta_x(f))$$

An $\ell_2$ norm constraint is additionally applied to further reduce human perception of the embedded perturbation. The final audio features are mapped back to the range $[-1, 1]$ to ensure waveform validity.

## Key Experimental Results

**Experimental scale**: 16 open-source TTS + 3 commercial APIs (ByteDance / Alibaba / MiniMax), 7 ASR systems, Chinese and English datasets (LibriTTS / CMU ARCTIC / THCHS30), conducted on a single NVIDIA 4090.

### End-to-End Fine-tuning Scenario (Core Results, Table 1)

| Method | GSV WER↑ | GSV SIM↓ | CosyVoice WER↑ | CosyVoice SIM↓ | VITS WER↑ | VITS SIM↓ | SNR↑ |
|------|----------|----------|-----------------|-----------------|-----------|-----------|------|
| Clean | 3.4 | 0.685 | 4.3 | 0.700 | 7.8 | 0.710 | - |
| AntiFake | 28.8 | 0.149 | 7.8 | 0.232 | 41.5 | 0.257 | 12.8 |
| SafeSpeech | 44.8 | 0.339 | 8.6 | 0.459 | 105.5 | 0.180 | 7.6 |
| **E2E-VGuard (UT)** | **66.5** | **0.123** | **21.6** | **0.091** | **95.7** | **0.106** | **18.5** |
| **E2E-VGuard (T)** | **94.8** | 0.284 | **72.1** | 0.375 | **125.3** | 0.245 | **20.5** |

- WER improves by an average of 19.8% (T mode) over the best baseline; SIM decreases by an average of 0.043 (UT mode).
- Improvements are more pronounced in the zero-shot scenario: average WER gain of 32.8% (UT) / 50.1% (T), SIM reduction of 0.119 (UT).

### Zero-Shot Scenario (Table 2, Industrial-Grade LLM-based Models)

Evaluated on 7 recent models including Index-TTS, FireRedTTS-1S, Step-Audio-TTS, and Spark-TTS. E2E-VGuard (UT) achieves state-of-the-art SIM across all models, with mean WER of 21.6% (UT) / 23.6% (T), substantially outperforming AntiFake (4.9%) and SafeSpeech (19.3%).

### Perceptual Quality

E2E-VGuard achieves higher SNR than all baselines (18.5–20.5 dB) and PESQ scores of 1.9–2.3, indicating the lowest perturbation noise ratio and minimal audio quality degradation.

## Highlights & Insights

1. **First systematic formalization of the E2E speech synthesis defense scenario**: explicitly establishes the ASR module in the ASR→TTS pipeline as a novel defense point, closely aligned with practical commercial API deployment.
2. **Joint timbre and pronunciation defense**: simultaneously disrupts timbre features and text-to-pronunciation alignment, forming a two-dimensional protection mechanism.
3. **Encoder ensemble + MFCC**: heterogeneous encoder ensembles improve cross-model transferability; MFCC specifically addresses discrete token encoding in LLM-based TTS.
4. **Psychoacoustic model for imperceptibility**: frequency masking combined with $\ell_2$ constraints renders perturbations nearly imperceptible to human listeners.
5. **Comprehensive evaluation**: 19 TTS models (including 3 commercial APIs) × 7 ASR systems × Chinese and English datasets, with real-world deployment validation.

## Limitations & Future Work

1. **Adversarial robustness**: although robustness against data augmentation and perturbation removal is tested, advanced adaptive attacks (targeted circumvention by adversaries aware of the defense) are not sufficiently discussed.
2. **Computational efficiency**: 500-step iterative optimization entails substantial computational overhead, making real-time audio protection difficult.
3. **ASR system dependency**: adversarial examples are generated targeting specific ASR systems; effectiveness may degrade if attackers employ entirely unseen ASR systems.
4. **Perceptual quality trade-off**: PESQ scores (1.9–2.3) leave room for improvement, particularly in high-quality audio sharing scenarios.
5. **Target text selection strategy**: the current target text selection is relatively heuristic; more automated search for optimal target texts warrants exploration.

## Related Work & Insights

| Method | Defense Type | Timbre Protection | Pronunciation Protection | LLM-based TTS | E2E Scenario | Psychoacoustic |
|------|---------|---------|---------|---------------|---------|---------|
| AttackVC | Adversarial examples | ✓ | ✗ | ✗ | ✗ | ✗ |
| AntiFake | Adversarial examples + encoder ensemble | ✓ | ✗ | ✗ | ✗ | ✗ |
| POP / SafeSpeech | Unlearnable examples | ✓ | ✗ | ✗ | ✗ | ✗ |
| **E2E-VGuard** | **Adversarial examples + encoder ensemble + ASR attack** | **✓** | **✓** | **✓** | **✓** | **✓** |

E2E-VGuard is the first method to simultaneously cover LLM-based TTS defense and the E2E scenario, achieving pronunciation-level defense by introducing ASR attacks—a dimension entirely absent in prior work.

**Broader implications:**
- **Attack surface extension**: targeting the ASR system as an indirect attack objective is an elegant insight—in multi-stage pipelines, any intermediate module may serve as a defense point.
- **Connection to multimodal security**: similar ideas can generalize to proactive defense in image→text→generation pipelines (e.g., OCR→LLM scenarios).
- **Encoder ensemble strategy**: the approach of leveraging heterogeneous encoder ensembles to improve adversarial transferability is applicable to other adversarial attack and defense domains (image, video).
- **Industrial deployment reference**: validation against commercial APIs (ByteDance / Alibaba / MiniMax) provides a viable pathway from academic research to production deployment.

## Rating
- Novelty: 8/10 — The formalization of the E2E scenario and the idea of defending pronunciation via ASR attacks are genuinely novel.
- Experimental Thoroughness: 9/10 — 19 TTS + 7 ASR + commercial APIs + real-world deployment; extremely comprehensive.
- Writing Quality: 7/10 — Content is rich, but notation in certain equations could be explained more clearly.
- Value: 8/10 — Directly addresses current pain points in voice cloning security with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] VoxMind: An End-to-End Agentic Spoken Dialogue System](../../ACL2026/audio_speech/voxmind_an_end-to-end_agentic_spoken_dialogue_system.md)
- [\[AAAI 2026\] End-to-end Contrastive Language-Speech Pretraining Model For Long-form Spoken Question Answering](../../AAAI2026/audio_speech/end-to-end_contrastive_language-speech_pretraining_model_for_long-form_spoken_qu.md)
- [\[ACL 2026\] VAPO: End-to-end Slide-Enhanced Speech Recognition with Omni-modal Large Language Models](../../ACL2026/audio_speech/vapo_end-to-end_slide-enhanced_speech_recognition_with_omni-modal_large_language.md)
- [\[ACL 2026\] Speculative End-Turn Detector for Efficient Speech Chatbot Assistant](../../ACL2026/audio_speech/speculative_end-turn_detector_for_efficient_speech_chatbot_assistant.md)
- [\[NeurIPS 2025\] Adapting Speech Language Model to Singing Voice Synthesis](adapting_speech_language_model_to_singing_voice_synthesis.md)

</div>

<!-- RELATED:END -->
