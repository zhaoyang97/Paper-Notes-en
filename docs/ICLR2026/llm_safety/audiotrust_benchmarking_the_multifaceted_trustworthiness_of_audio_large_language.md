---
title: >-
  [Paper Note] AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models
description: >-
  [ICLR 2026][LLM Safety][Audio LLM] This paper proposes AudioTrust, the first multidimensional trustworthiness evaluation benchmark for audio large language models (ALLMs), encompassing six dimensions—fairness, hallucination, safety, privacy, robustness, and authentication—with 26 sub-tasks and 4,420+ audio samples. It systematically evaluates the trustworthiness boundaries of 14 state-of-the-art open- and closed-source ALLMs in high-stakes audio scenarios.
tags:
  - ICLR 2026
  - LLM Safety
  - Audio LLM
  - trustworthiness
  - benchmark
  - fairness
  - hallucination
  - safety
  - privacy
  - robustness
  - authentication
date: 2026-05-08
content_hash: e6c5d40bd479ad17
---

# AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2505.16211](https://arxiv.org/abs/2505.16211)
**Code**: [GitHub](https://github.com/JusperLee/AudioTrust)
**Area**: AI Safety
**Keywords**: Audio LLM, trustworthiness, benchmark, fairness, hallucination, safety, privacy, robustness, authentication

## TL;DR
This paper proposes AudioTrust, the first multidimensional trustworthiness evaluation benchmark for audio large language models (ALLMs), encompassing six dimensions—fairness, hallucination, safety, privacy, robustness, and authentication—with 26 sub-tasks and 4,420+ audio samples. It systematically evaluates the trustworthiness boundaries of 14 state-of-the-art open- and closed-source ALLMs in high-stakes audio scenarios.

## Background & Motivation
**State of the Field**: ALLMs have advanced rapidly (GPT-4o Audio, Qwen2-Audio, Gemini, etc.), yet existing safety evaluation frameworks (SafeDialBench, SafetyBench) are primarily designed for the text modality and overlook trustworthiness risks unique to audio.

**Root Cause**: Audio signals contain rich non-semantic acoustic cues (timbre, accent, background noise, emotion) that can be exploited to manipulate model behavior—attack vectors that text-based safety frameworks are fundamentally incapable of capturing.

**Core Idea**: The paper constructs the first comprehensive ALLM trustworthiness evaluation framework covering six audio-native safety dimensions. Through carefully designed real-world scenario datasets and an automated evaluation pipeline (human verification agreement >97%), it systematically quantifies trustworthiness risks in ALLMs.

## Method

### Overall Architecture
AudioTrust decouples trustworthiness evaluation into six orthogonal dimensions, each with independent attack strategies, datasets, and evaluation metrics:

1. **Fairness**: Assesses biases induced by acoustic attributes (accent, speech rate, emotion, background environment), distinguishing traditional fairness (gender/age/race) from audio-specific fairness (accent/language fluency/socioeconomic status/personality traits); includes decision experiments and stereotype experiments; 840 audio samples.
2. **Hallucination**: Defines audio-specific hallucination types—violations of physical laws (flames burning underwater) and violations of temporal causality (ignition before engine start); 320 samples.
3. **Safety**: Designs emotional deception attacks (exploiting urgent/sorrowful tones to bypass safety filters) covering jailbreak attacks and illegal activity inducement across enterprise, financial, and medical domains; 600 samples.
4. **Privacy**: Distinguishes content-level leakage (directly reading out bank account numbers) from paralinguistic inference leakage (inferring age/race/geolocation from voiceprints); 900 samples.
5. **Robustness**: Evaluates adversarial attacks and natural degradation (background noise, multiple speakers, audio quality variation, environmental sounds); 240 samples.
6. **Authentication**: Covers identity verification bypass (social engineering attacks), hybrid deception (voice cloning + background noise), and voice cloning spoofing; 400 samples.

### Key Designs
1. **Data Construction**: GPT-4o is used to generate textual content; F5-TTS synthesizes audio; emotional control is achieved by selecting reference audio with different emotional timbres. Portions of the data are sourced from public datasets such as Common Voice and freesound.
2. **Automated Evaluation Pipeline**: Dual evaluators (GPT-4o and Qwen3) provide scores, with human expert review (agreement >97%), supporting large-scale reproducible evaluation.
3. **Metric Design**: Each dimension employs targeted metrics—fairness uses group fairness score $\Gamma$ (1.0 is ideal), safety uses Defense Success Rate (DSR), privacy uses refusal rate, robustness uses a 10-point scale, and authentication uses Impostor Rejection Rate (IRR).

## Key Experimental Results

### Fairness

| Metric | Best Open-Source | Best Closed-Source | Average |
|--------|-----------------|-------------------|---------|
| $\Gamma_\text{stereo}$ | Step-Fun 0.658 | GPT-4o Audio 0.926 | 0.328 |
| $\Gamma_\text{decision}$ | Step-Fun 0.505 | Gemini-1.5 Pro 0.460 | 0.261 |

- Biases introduced by audio attributes (accent, emotion) are **stronger** than those from traditional sensitive attributes (age, gender).
- Closed-source models exhibit stronger decision bias; open-source models show stronger stereotype associations.
- The GPT-4o series excels in stereotype fairness ($\Gamma_\text{stereo}=0.926$) but performs moderately on decision fairness ($\Gamma_\text{decision}=0.264$), as it sacrifices fairness for accuracy in extreme decision scenarios.

### Hallucination
- The Gemini series achieves the best performance on physical/logical violation detection (scores 8–9).
- GPT-4o Audio performs unexpectedly poorly on content-mismatch and label-mismatch tasks (scores 3–4).
- Models achieve high accuracy on physical law violation tasks but perform poorly on content-mismatch tasks that humans find easy—revealing a significant human-machine perceptual gap.

### Safety

| Scenario | Closed-Source Avg. DSR | Open-Source Avg. DSR |
|----------|----------------------|---------------------|
| Enterprise jailbreak | ~99% | ~80% |
| Illegal activity inducement | ~99% | ~89% |

- Kimi-Audio achieves the best performance among open-source models, approaching closed-source levels.
- OpenS2S is the most vulnerable, with a DSR of only 51.4% in enterprise scenarios.

### Privacy
- **Direct leakage**: GPT-4o mini Audio achieves a refusal rate of 100%; privacy-enhancing prompts improve refusal rates by approximately 25%.
- **Inference leakage**: The average refusal rate across all models is only 9.02%, and privacy-enhancing prompts yield only ~3% improvement—ALLMs fail to recognize information inferred from paralinguistic cues as private.
- Models such as Qwen2-Audio, MiniCPM-o 2.6, and Qwen2.5-Omni exhibit near-zero refusal rates on direct leakage, demonstrating almost no content-level privacy protection.
- The consistently low refusal rates for inference leakage indicate that the absence of paralinguistic privacy constraints in model training is a systemic issue, not an artifact of individual models.

### Robustness
- Closed-source models (led by Gemini-2.5 Pro) consistently outperform open-source models under nearly all degradation conditions.
- Open-source models exhibit an "over-textualization" tendency—continuing to reason from text when transcription is partially correct while ignoring acoustic cues.
- In multi-speaker scenarios, Step-Audio2 scores approach zero (MS=0.00/0.12), exposing extreme multi-speaker robustness deficiencies.
- The advantage of closed-source models is most pronounced under severe acoustic distortion, suggesting more mature front-end signal processing and noise reduction architectures.

### Authentication

| Attack Type | Open-Source Avg. IRR | Closed-Source Avg. IRR |
|-------------|---------------------|----------------------|
| Identity verification bypass | 55.3% | 97.2% |
| Hybrid deception | 55.1% | 97.0% |
| Voice cloning | 45.0% | 44.9% |

- Voice cloning is a universal weakness across all models, including closed-source ones.
- Stricter system prompts consistently improve resistance to spoofing attacks.
- Among open-source models, SALMONN consistently ignores prompt instructions and outputs audio descriptions, rendering it incapable of completing voice cloning detection tasks.

## Highlights & Insights
- **First Audio-Native Trustworthiness Benchmark**: This is the first work to systematically define and evaluate safety dimensions unique to the audio modality, filling a critical gap in ALLM trustworthiness assessment.
- **Threat of Paralinguistic Cues**: Non-semantic information in audio (accent, timbre, background noise) represents a severely underestimated source of bias and a novel attack vector.
- **Privacy Inference Leakage**: ALLMs can infer sensitive attributes such as age and race from voiceprints, yet almost never treat such inferences as privacy violations—constituting an entirely new category of privacy threat.
- **Human-Machine Perceptual Gap**: Models excel at detecting physical violations but perform poorly on commonsense reasoning that humans find trivial, revealing fundamental deficiencies in current model perception mechanisms.
- **Evaluation Scale and Rigor**: The benchmark spans 14 models × 26 sub-tasks × 4,420 samples × dual evaluators, demonstrating exceptional coverage and methodological rigor.

## Limitations & Future Work
- Audio samples are primarily synthetic (F5-TTS), potentially introducing distributional gaps relative to real human speech; synthetic attack audio may underestimate the effectiveness of real-world attacks.
- Evaluation is predominantly English-centric; multilingual coverage remains limited, and acoustic characteristics of different languages may introduce distinct trustworthiness risks.
- The paper does not explore model improvement methods—it exposes problems without proposing remediation strategies.
- Some open-source models exhibit random audio recognition failures, potentially inflating safety scores.
- Interaction effects among the six dimensions (e.g., whether poor robustness amplifies safety risks) are unexplored.
- Evaluation relies on GPT-4o/Qwen3 scoring; biases in the evaluators themselves may affect the generalizability of conclusions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First audio-native trustworthiness benchmark, defining entirely new evaluation dimensions and threat models.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 14 models, 6 dimensions, 26 sub-tasks, dual evaluators, and human verification.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear and systematic, though dense tables impose some readability burden.
- Value: ⭐⭐⭐⭐⭐ Directly guides safe deployment of ALLMs and will shape the research agenda for audio AI safety.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] StyleBreak: Revealing Alignment Vulnerabilities in Large Audio-Language Models via Style-Aware Audio Jailbreak](../../AAAI2026/llm_safety/stylebreak_revealing_alignment_vulnerabilities_in_large_audio-language_models_vi.md)
- [\[ICLR 2026\] BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models](biasbusters_uncovering_and_mitigating_tool_selection_bias_in_large_language_mode.md)
- [\[ICLR 2026\] Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark](measuring_physical-world_privacy_awareness_of_large_language_models_an_evaluatio.md)
- [\[ICLR 2026\] SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC](secp-tuning_efficient_privacy-preserving_prompt_tuning_for_large_language_models.md)
- [\[AAAI 2026\] Gender Bias in Emotion Recognition by Large Language Models](../../AAAI2026/llm_safety/gender_bias_in_emotion_recognition_by_large_language_models.md)

<!-- RELATED:END -->
