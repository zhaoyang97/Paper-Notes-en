---
title: >-
  [Paper Note] Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Audio-Language Models
description: >-
  [ACL2025][Audio & Speech][adversarial audio attacks] This work proposes the Chat-Audio Attacks (CAA) benchmark, comprising four categories of universal adversarial audio attacks (content, emotional, explicit noise, and implicit noise attacks). Through three evaluation methodologies, it systematically assesses the robustness of six SOTA Large Audio-Language Models (LALMs), finding that GPT-4o performs the best but all models exhibit significant vulnerabilities.
tags:
  - "ACL2025"
  - "Audio & Speech"
  - "adversarial audio attacks"
  - "large audio-language models"
  - "robustness evaluation"
  - "benchmark"
date: 2026-05-08
content_hash: 30dbe11ed725a6f8
---

# Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Audio-Language Models

**Conference**: ACL2025  
**arXiv**: [2411.14842](https://arxiv.org/abs/2411.14842)  
**Authors**: Wanqi Yang, Yanda Li, Meng Fang, Yunchao Wei, Ling Chen (UTS, Liverpool, Beijing Jiaotong University)
**Code**: [CAA Benchmark](https://github.com/YanqiYang/CAA)  
**Area**: Audio & Speech  
**Keywords**: adversarial audio attacks, large audio-language models, robustness evaluation, benchmark

## TL;DR

This work proposes the Chat-Audio Attacks (CAA) benchmark, comprising four categories of universal adversarial audio attacks (content, emotional, explicit noise, and implicit noise attacks). Through three evaluation methodologies, it systematically assesses the robustness of six SOTA Large Audio-Language Models (LALMs), finding that GPT-4o performs the best but all models exhibit significant vulnerabilities.

## Background & Motivation

- **Problem**: Large Audio-Language Models (LALMs) are increasingly popular in speech-based human-computer interaction, but they are threatened by adversarial audio attacks. Prior research has mostly focused on model-specific attack methods, lacking a universal attack evaluation framework tailored to real-world scenarios.
- **Limitations of Prior Work**: (1) Previous adversarial audio attacks (e.g., Carlini & Wagner 2018) rely on gradient signals and target specific models, lacking generalizability; (2) The impact of universal adversarial audio attacks (e.g., speaker slips of the tongue, noisy environments) on LALMs remains under-explored; (3) There is a lack of systematic evaluation benchmarks that target conversational scenarios and cover diverse attack types.
- **Key Motivation**: To construct a universal adversarial audio attack benchmark based on conversational scenarios to comprehensively evaluate the robustness of LALMs under various attack types, thereby driving the development of defense mechanisms.

## Method

### CAA Benchmark Construction

Each data entry is a quadruple consisting of the original audio, text transcript, attack-free audio, and a set of attack variants.

**Data Collection**: 360 English speech samples were manually selected from three public datasets:
- **MELD** (120 samples): An emotional conversation dataset (Friends) containing emotion labels.
- **TVQA** (120 samples): A TV show conversation dataset.
- **Common Voice** (120 samples): A multilingual speech recognition dataset.

**Generation of Four Attack Categories**:

1. **Content Attack**: Modifies a small number of tokens while preserving semantic consistency (synonym replacement, token rearrangement, minor token variations). Audio is re-generated using AzureSpeechSDK. Goal: To examine the sensitivity of models to minor token variations.
2. **Emotional Attack**: Alters emotional intonation without changing the content via (a) opposing emotional intonation (e.g., changing original "angry" to stressed "happy"); (b) overlaying background music with opposing emotions. This attack is only supported on MELD data. Goal: To detect the model's sensitivity to speech emotion and content-emotion mismatches.
3. **Explicit Noise Attack**: Overlays three categories of audible noise: natural noise (bird chirping, wind, thunder), industrial noise (car horns, machinery), and human noise (crowd murmur, shouting, laughter). Goal: To evaluate the model's ability to distinguish speech from background noise.
4. **Implicit Noise Attack**: Overlays inaudible noise to the human ear, specifically infrasound (15 Hz) and ultrasound (22000 Hz). While inaudible to humans, these may affect models. Goal: To test whether models are as unaffected by inaudible noise as humans are.

**Quality Control**: 7 manual screening criteria plus GPT-4-assisted filtering were employed to ensure sample answerability. Attack-free audios were re-synthesized using AzureSpeechSDK to guarantee clarity.

### Three Evaluation Methodologies

1. **Standard Evaluation**: Compares model outputs between attack-free and attacked audio using three metrics: WER, ROUGE-L, and cosine similarity (COS).
2. **GPT-4o-Based Evaluation**: Simulates real interaction scenarios, scoring from 1 to 5 based on four indicators: NC (No-attack Coherence), ACoh (Attack Coherence), ACor (Attack Relevance), and LR (Language Robustness).
3. **Human Evaluation**: Five native English-speaking university students independently score NC and ACoh, reflecting real user experience and trust.

## Key Experimental Results

Evaluating six SOTA LALMs: SpeechGPT, SALMONN, Qwen2-Audio, LLama-Omni, Gemini-1.5-Pro, and GPT-4o.

### Table 2: Standard Evaluation Results (Selected Key Data)

| Model | Metric | Content Attack | Explicit-Natural | Implicit-Infrasound | Implicit-Ultrasound |
|------|------|------|------|------|------|
| GPT-4o | WER | 1.12 | 1.18 | 1.25 | 1.13 |
| GPT-4o | ROUGE-L | 0.25 | 0.20 | 0.22 | 0.17 |
| LLama-Omni | WER | 1.04 | 0.64 | 0.67 | 0.37 |
| LLama-Omni | ROUGE-L | 0.36 | 0.58 | 0.56 | 0.75 |
| SpeechGPT | WER | 1.79 | 2.25 | 2.21 | 1.28 |
| SpeechGPT | ROUGE-L | 0.17 | 0.12 | 0.14 | 0.20 |

**Findings**: LLama-Omni performs most consistently on standard metrics, while SpeechGPT experiences the most severe performance degradation under all attacks.

### Table 3: GPT-4o Evaluation Results (Scores 1-5)

| Model | NC | Content ACoh | Emo-Tone ACoh | Natural ACoh | Infrasound ACoh |
|------|------|------|------|------|------|
| GPT-4o | 4.45 | 3.94 | 4.35 | 2.57 | 3.02 |
| LLama-Omni | 3.50 | 3.05 | 3.08 | 2.72 | 2.79 |
| Gemini-1.5-Pro | 3.58 | 3.15 | 3.30 | 2.42 | 2.78 |
| SpeechGPT | 2.39 | 1.76 | 1.49 | 1.32 | 1.23 |

**Findings**: GPT-4o performs the best in no-attack coherence (NC=4.45) and under most attacks; however, all models suffer a significant decrease in ACoh under natural noise.

### Table 4: Human Evaluation Results

| Model | NC | Content ACoh | Emotion ACoh | Explicit ACoh | Implicit ACoh |
|------|------|------|------|------|------|
| GPT-4o | 4.33 | 3.88 | 4.12 | 3.27 | 3.08 |
| Gemini-1.5-Pro | 3.92 | 3.20 | 3.41 | 3.24 | 2.87 |
| LLama-Omni | 3.75 | 3.40 | 3.22 | 2.88 | 3.15 |

**Findings**: The trends in human evaluation align with automatic evaluations; GPT-4o remains the most robust at the user-perception level.

## Highlights & Insights

- **Systematic Benchmark**: The first multi-type audio attack benchmark (1,680 samples, 4 attack types) designed for conversational LALM scenarios, accounting for both audible and inaudible noises.
- **Three-Dimensional Evaluation System**: Combines standard metrics, GPT-4o simulation, and human evaluation to provide a complementary, comprehensive, and credible assessment.
- **In-Depth Analysis**: Explains robustness discrepancies from the perspective of model architecture—transcript modules (e.g., SpeechGPT) are a weak link, while noise-inclusive training data (e.g., Qwen2-Audio, LLama-Omni) enhances robustness.
- **Finding of Implicit Noise Impact**: Infrasound and ultrasound (inaudible to humans) surprisingly affect the outputs of certain models, revealing a discrepancy between machine and human auditory perception.

## Limitations & Future Work

- The benchmark is based on controlled conversational settings, which may not cover highly dynamic real-world environments.
- Lack of investigation into audio jailbreak attacks, with very few open-source works available in this domain.
- Only English is evaluated, leaving multilingual scenarios unaddressed.
- Emotional labels are exclusive to the MELD dataset, restricting emotional attacks to only the MELD subset (120 samples).
- Deep analysis of robustness causes for proprietary models (GPT-4o, Gemini) is infeasible due to the lack of architectural access.
- The attack types are relatively fixed and do not account for adaptive attacks or post-adversarial training responses.

## Related Work & Insights

| Dimension | Ours (CAA) | Traditional Audio Adversarial Attacks | Image Adversarial Attacks |
|------|------|------|------|
| Attack Generalizability | Universal attacks, model-independent | Model-specific (gradient-dependent) | Model-specific |
| Target Model | 6 SOTA LALMs | Single ASR/ASV system | CNN classifiers |
| Attack Dimension | Content + Emotion + Noise + Implicit | Content perturbation only | Pixel perturbation |
| Evaluation Methodology | 3 complementary methods | Single metric | Classification accuracy |
| Scenario | Conversational interaction | Speech recognition/verification | Image classification |

## Rating

- Novelty: ⭐⭐⭐⭐ — First multi-type audio attack benchmark for conversational LALM scenarios; the perspective of implicit noise attacks is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage with 6 models x 4 attacks x 3 evaluation methods, though experiments on defensive methods are lacking.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and in-depth discussions, though some tables are quite dense.
- Value: ⭐⭐⭐⭐ — Provides a standardized evaluation platform for audio adversarial attack and defense research, offering high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models](audio_dialogue_benchmark.md)
- [\[ACL 2026\] SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?](../../ACL2026/audio_speech/speakersleuth_can_large_audio-language_models_judge_speaker_consistency_across_m.md)
- [\[AAAI 2026\] DiffA: Large Language Diffusion Models Can Listen and Understand](../../AAAI2026/audio_speech/diffa_large_language_diffusion_models_can_listen_and_understand.md)
- [\[ACL 2025\] Mind the Gap! Static and Interactive Evaluations of Large Audio Models](mind_the_gap_static_and_interactive_evaluations_of_large_audio_models.md)
- [\[ACL 2025\] Towards Reliable Large Audio Language Model](towards_reliable_large_audio_language_model.md)

</div>

<!-- RELATED:END -->
