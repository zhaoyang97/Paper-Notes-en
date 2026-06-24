---
title: >-
  [Paper Note] Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models
description: >-
  [ACL 2025][Audio & Speech][Large Audio-Language Models] This paper proposes ADU-Bench, a comprehensive benchmark comprising 4 sub-datasets (general dialogue, professional skills, multilingualism, and ambiguity handling) totaling over 20,000 open-ended audio dialogues. It systematically evaluates 16 Large Audio-Language Models (LALMs) on their audio dialogue understanding capabilities, revealing significant deficiencies in current models regarding mathematical formula understa…
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Large Audio-Language Models"
  - "audio dialogue"
  - "LALM"
  - "benchmark"
  - "ambiguity handling"
date: 2026-05-08
content_hash: 1f521b623a98a49e
---

# Benchmarking Open-ended Audio Dialogue Understanding for Large Audio-Language Models

**Conference**: ACL 2025  
**arXiv**: [2412.05167](https://arxiv.org/abs/2412.05167)  
**Code**: [Project Page](https://adu-bench.github.io/)  
**Area**: Audio & Speech / Multimodal Evaluation  
**Keywords**: Large Audio-Language Models, audio dialogue, LALM, benchmark, ambiguity handling

## TL;DR

This paper proposes ADU-Bench, a comprehensive benchmark comprising 4 sub-datasets (general dialogue, professional skills, multilingualism, and ambiguity handling) totaling over 20,000 open-ended audio dialogues. It systematically evaluates 16 Large Audio-Language Models (LALMs) on their audio dialogue understanding capabilities, revealing significant deficiencies in current models regarding mathematical formula understanding, role-playing, multilingual processing, and speech ambiguity resolution.

## Background & Motivation

**Background**: Large Audio-Language Models (LALMs) such as GPT-4o have recently unlocked speech dialogue capabilities, enabling direct speech interaction with humans. These models show promising prospects in various real-world scenarios.

**Limitations of Prior Work**: Existing LALM benchmarks either focus on fundamental audio tasks (e.g., speech recognition, emotion detection), adopt a "text instruction + audio input" QA format, or cover only general conversation scenarios. A systematic, comprehensive benchmark is lacking to assess LALMs' overall capabilities in open-ended audio dialogue understanding—particularly in domain-specific skills, multilingual settings, and speech-specific ambiguity.

**Key Challenge**: While LALMs' capabilities are advancing rapidly, evaluation methods lag behind. Comparisons between different models lack standard benchmarks, leaving researchers unable to systematically identify which capabilities are mature and which remain bottlenecks. In particular, unique speech ambiguity phenomena (such as inflections changing meaning) cannot be evaluated by text-based benchmarks at all.

**Goal**: (1) Construct a multi-dimensional audio dialogue understanding benchmark; (2) Evaluate LALMs' speech ambiguity processing capabilities for the first time; (3) Conduct a systematic comparative analysis across 16 LALMs.

**Key Insight**: Taking real human-to-human speech interaction as a starting point, this work broadens the evaluation scope to four dimensions: general understanding, domain-specific skills, multilingual performance, and ambiguity resolution. The design of ambiguity resolution is most innovative—where identical text transcriptions correspond to different intonations/pauses, forcing models to distinguish meaning from the audio rather than the text.

**Core Idea**: Build an audio dialogue benchmark of over 20K samples covering four dimensions (general/skills/multilingual/ambiguity) along with a GPT-4-based evaluation framework to systematically reveal the complete landscape and deficiencies of LALMs in audio dialogue understanding.

## Method

### Overall Architecture

ADU-Bench comprises 4 datasets: (1) ADU-General (12,000 instances) covering 3 general scenarios; (2) ADU-Skill (3,725 instances) covering 12 domain skills; (3) ADU-Multilingual (3,600 instances) covering 9 languages; (4) ADU-Ambiguity (1,390 instances) covering 4 speech ambiguity categories. Each data item is a tuple of (audio query, reference text answer). After feeding the audio input into the LALM to obtain a text response, a GPT-4 evaluator scores the response quality (0-10 scale).

### Key Designs

1. **GPT-4 Bidirectional Evaluation to Eliminate Position Bias**:

    - **Function**: Fairly evaluate the quality of LALM-generated responses.
    - **Mechanism**: Feed the audio transcription, reference answer, and model response to the GPT-4 evaluator, scoring across four dimensions: helpfulness, relevance, accuracy, and comprehensiveness. To eliminate position bias (where GPT-4 favors the text appearing first), the reference answer and model response positions are swapped for a second scoring phase, and the average is computed. Concurrently, LLaMA-3-70B and Qwen-2-72B are used as auxiliary evaluators to validate consistency.
    - **Design Motivation**: Experiments show noticeable differences between the two scores when positions are not swapped. The bidirectional evaluation aligns with human preference judgments by over 85%.

2. **ADU-Ambiguity Dataset (Pioneering Speech Ambiguity Evaluation)**:

    - **Function**: Assess LALMs' capability to comprehend information in speech that transcends textual semantics.
    - **Mechanism**: Design 4 types of ambiguity: (a) **Intonation Ambiguity**—identically transcribed sentences expressing different meanings through varied intonations (e.g., "What a perfect day!" in a rising tone vs. a disappointed tone); (b) **Pause Ambiguity**—pause placement altering syntactic modification relations (e.g., "professional | reviewers and authors" vs. "professional reviewers | and authors"); (c) **Homophone Ambiguity**—words pronounced nearly identically but with distinct definitions (e.g., "weight" vs. "wait"); (d) **Repetition Ambiguity**—repetitive identical words causing phonetic confusion (e.g., "I saw a man saw a saw with a saw"). SSML tags are used to meticulously control intonation and pauses in synthetic speech.
    - **Design Motivation**: Unique audio features compared to text (such as prosody, intonation, and pauses) are the core differentiators of audio dialogue capabilities, yet they have never been systematically evaluated.

3. **Hybrid Data Sources (Real Recording + Synthetic Audio)**:

    - **Function**: Balance data diversity and scalability.
    - **Mechanism**: Among the 20,715 total audio items, over 8,000 are real recordings (from Common Voice, Slue, etc.), while the rest are synthesized via Microsoft Azure's SSML services. These cover random combinations of 2 genders, 4 speakers, 4 emotions, and 3 distinct speaking rates/pitches/volumes. Ablation experiments demonstrate no significant difference between synthetic and real audio when evaluating models.
    - **Design Motivation**: Meet the complex data generation requirements for diverse scenarios (skill tests, multilingualism, precise ambiguity control) while validating the methodology's effectiveness via ablation.

### Loss & Training

This work is benchmarking research and does not involve model training. The evaluation process is: LALM receives audio query $\rightarrow$ generates text/audio response (audio transcribed to text) $\rightarrow$ GPT-4 evaluator assigns score $\rightarrow$ bidirectional scores averaged.

## Key Experimental Results

### Main Results

| Model | Scale | General | Skill | Multilingual | Ambiguity | Avg |
|------|------|---------|-------|-------------|-----------|-----|
| PandaGPT | 7B | 1.02 | 0.98 | 0.98 | 0.50 | 0.87 |
| BLSP | 7B | 4.66 | 4.49 | 2.89 | 3.37 | 3.85 |
| Step-Audio-Chat | 130B | 6.37 | 7.31 | 2.45 | 4.72 | 5.21 |
| Whisper+LLaMA-3 | 70B | 7.26 | 8.03 | 6.12 | 5.13 | 6.64 |
| Whisper+GPT-4 | - | 8.42 | 8.62 | 8.07 | 5.54 | 7.66 |
| **GPT-4o** | - | **8.64** | **8.97** | **8.16** | **6.87** | **8.16** |

### Ambiguity Handling Performance (ADU-Ambiguity Subset)

| Ambiguity Type | GPT-4o | Whisper+GPT-4 | BLSP | Explanation |
|----------|--------|---------------|------|------|
| Intonation Ambiguity | 7.32 | 4.78 | 3.05 | Even the strongest model achieves only a moderate score |
| Pause Ambiguity | 5.22 | 4.72 | 2.82 | Generally difficult for LALMs |
| Homophone Ambiguity | 6.05 | 5.55 | 3.05 | Hard to distinguish homophones |
| Repetition Ambiguity | 7.90 | 7.12 | 4.55 | Relatively the best performing type |

### Key Findings

- **Large gap between open-source LALMs and GPT-4o**: The best open-source model, BLSP, averages 3.85, whereas GPT-4o achieves 8.16, exhibiting a gap of over 4 points (out of 10). Even the cascaded scheme Whisper+LLaMA-3-70B reaches only 6.64.
- **Math and coding remain specialized bottlenecks**: LALMs show the worst performance in STEM fields with formulas and code (e.g., mathematics, physics, and coding), as mathematical symbols and programming languages are difficult to convey effectively via audio. Conversely, purely verbal comprehension tasks like biology, law, and medicine yield better performance.
- **Highly imbalanced multilingual capabilities**: Performance is acceptable for English and Indo-European languages (such as German and Spanish) but severely drops for East Asian and Arabic languages, reflecting a strong linguistic bias in the training datasets.
- **Ambiguity processing is a universal weakness**: Even GPT-4o scores only around 5–7 on intonation and pause ambiguities, much lower than the 7.9 achieved on repetition ambiguity. This highlights a fundamental deficiency in current models' grasp of how prosody and pauses map to semantics. GPT-4o often generates responses encompassing "both possible interpretations," revealing its inability to resolve the specific meaning from the audio alone.
- **Model scale is generally helpful but not a silver bullet**: While SALMONN exhibits a noticeable improvement when scaling from 7B to 13B, scaling LLaMA-3 from 8B to 70B actually degrades performance on common sense and non-Indo-European languages.

## Highlights & Insights

- **Pioneering the speech ambiguity evaluation dimension** is the most significant contribution of this work. Challenges like intonation and pause ambiguity are completely native to speech modality and cannot be captured by text benchmarks. This highlights a neglected yet highly crucial direction for current LALM research.
- **Rigorous design of the GPT-4 bidirectional evaluation + human validation methodology**: Swapping positions rules out bias, cross-validation using multiple LLM evaluators and high correlation (85%+) with pairwise human preference ensures the reliability of the benchmark conclusions.
- **Ablation comparing real vs. synthetic audio** demonstrates that synthetic speech can effectively substitute for real recordings for assessment purposes, offering methodological support for low-cost scaling of future benchmarks.

## Limitations & Future Work

- **Limited number of evaluated models** (16), leaving out many recently released state-of-the-art LALMs (such as GPT-4o-mini and Gemini 2.0). Consistent updates are required given the rapid advancement of this field.
- **While synthetic audio was validated, SSML prosody control remains relatively rigid** and lacks the natural variability of real human speech, potentially underestimating the difficulty of certain ambiguity resolutions.
- **Evaluation relies on ASR transcriptions + LLM scoring** rather than evaluating the raw audio response directly. This evaluates a joint pipeline of "auditory comprehension $\rightarrow$ text generation," making it difficult to isolate pure audio-to-audio understanding.
- **The small scale of the ambiguity dataset** (1,390 samples) limits the statistical power of fine-grained analyses.

## Related Work & Insights

- **vs AIR-Bench (Yang et al., 2024)**: AIR-Bench targets audio QA (text instruction + audio input), whereas ADU-Bench prioritizes end-to-end spoken dialogue (direct audio query input). They complement each other by evaluating different interaction paradigms.
- **vs SD-Eval (Ao et al., 2024) / VoiceBench (Chen et al., 2024)**: These benchmarks also target spoken dialogue but mainly focus on general scenarios. ADU-Bench enriches evaluation with three new dimensions: professional skills, multilingualism, and ambiguity resolution, offering broader coverage.
- **vs Dynamic-SUPERB (Huang et al., 2024)**: Covers 180 foundational speech tasks but does not evaluate open-ended dialogue capability. ADU-Bench fills this gap in open-ended evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐ The speech ambiguity evaluation dimension is a highly novel contribution, though the overall benchmark construction methodology (synthesized audio + GPT-4 evaluation) is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Rigorous systematic comparison across 16 models and 4 dimensions, thoroughly validated by multiple evaluation approaches (multi-evaluators, human evaluation, ablations).
- Writing Quality: ⭐⭐⭐⭐ Highly structured and comprehensive in analysis, though some descriptions could be slightly more concise.
- Value: ⭐⭐⭐⭐ Delivers a highly needed comprehensive evaluation benchmark for the LALM domain, pointing to crucial research pathways via its speech ambiguity dimension.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Audio-Language Models](who_can_withstand_chat-audio_attacks_an_evaluation_benchmark_for_large_audio-lan.md)
- [\[ACL 2025\] Towards Reliable Large Audio Language Model](towards_reliable_large_audio_language_model.md)
- [\[ACL 2025\] Investigating and Enhancing Vision-Audio Capability in Omnimodal Large Language Models](investigating_and_enhancing_vision-audio_capability_in_omnimodal_large_language_.md)
- [\[ICLR 2026\] JALMBench: Benchmarking Jailbreak Vulnerabilities in Audio Language Models](../../ICLR2026/audio_speech/jalmbench_benchmarking_jailbreak_vulnerabilities_in_audio_language_models.md)
- [\[ACL 2025\] WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models](wavrag_audio-integrated_retrieval_augmented_generation_for_spoken_dialogue_model.md)

</div>

<!-- RELATED:END -->
