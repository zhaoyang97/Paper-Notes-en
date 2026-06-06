---
title: >-
  [Paper Note] Protecting Bystander Privacy via Selective Hearing in Audio LLMs
description: >-
  [ACL 2026][Audio & Speech][Bystander Privacy] The authors propose the first bystander privacy benchmark, SH-Bench, and a Bystander Privacy Fine-Tuning (BPFT) method to evaluate and enhance the ability of audio LLMs to fo…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Bystander Privacy"
  - "Selective Hearing"
  - "Audio LLMs"
  - "Multi-speaker"
  - "Privacy-preserving Fine-tuning"
date: 2026-05-08
content_hash: 54c80e08fc427093
---

# Protecting Bystander Privacy via Selective Hearing in Audio LLMs

**Conference**: ACL 2026  
**arXiv**: [2512.06380](https://arxiv.org/abs/2512.06380)  
**Code**: [GitHub](https://github.com/Elocinacademia/SelectiveHearing-Bench)  
**Area**: AI Safety / Audio Privacy  
**Keywords**: Bystander Privacy, Selective Hearing, Audio LLMs, Multi-speaker, Privacy-preserving Fine-tuning

## TL;DR
The authors propose the first bystander privacy benchmark, SH-Bench, and a Bystander Privacy Fine-Tuning (BPFT) method to evaluate and enhance the ability of audio LLMs to focus solely on the primary speaker and refuse to leak bystander information in multi-speaker environments. After BPFT, the SE metric improves by 16% over Gemini 2.5 Pro.

## Background & Motivation

**Background**: Audio LLMs (e.g., SALMONN, Qwen-Audio) are being widely deployed in voice assistants and wearable devices, where they passively capture speech in open environments. Existing privacy research primarily focuses on active users interacting with the model.

**Limitations of Prior Work**: In real-world scenarios (coffee shops, public transport, etc.), audio LLMs inevitably capture the speech of surrounding bystanders. These bystanders do not actively interact with the system or realize their speech is being processed, facing severe privacy leakage risks. Existing benchmarks and defenses completely ignore bystander privacy.

**Key Challenge**: Audio LLMs require strong multi-speaker understanding capabilities to serve the primary user, but this same capability enables them to extract sensitive information from bystanders. There is a fundamental tension between understanding performance and privacy protection.

**Goal**: (1) Establish the first benchmark for evaluating bystander privacy, SH-Bench; (2) Propose a unified metric, SE, to measure the balance between understanding and privacy protection; (3) Design the BPFT method to enhance bystander privacy protection.

**Key Insight**: The authors propose the concept of "Selective Hearing"—the model should focus only on the target speaker and choose "I don't know" for queries related to bystander speech.

**Core Idea**: By constructing multi-speaker audio samples containing both primary speakers and bystanders, the model is trained to refuse answering bystander-related questions when instructed to protect privacy, without compromising understanding of the primary speaker.

## Method

### Overall Architecture
SH-Bench contains 3,968 multi-speaker audio mixed samples (approx. 157.5 hours) paired with 77k multiple-choice questions. Evaluation includes two modes: General mode (answering all questions) and Selective mode (answering only primary speaker questions, selecting "I don't know" for bystanders). BPFT is a fine-tuning pipeline based on synthetic data.

### Key Designs

1.  **SH-Bench Data Construction**:

    - **Function**: Provides multi-speaker bystander privacy evaluation data for both real and synthetic scenarios.
    - **Mechanism**: For real scenarios, participants were recruited via Prolific to record in five daily environments (coffee shops, gyms, public transport, etc.); primary speakers recorded structured content, while bystanders recorded informal sensitive conversations. Synthetic scenarios are based on the AMI meeting corpus, mixing bystander audio at -10dB into the primary speaker's audio. Each audio is paired with 10 five-choice MCQs, where one option is always a variation of "I don't know."
    - **Design Motivation**: Real scenarios capture natural acoustic variations, while synthetic scenarios provide controllable large-scale data; the IDK option is critical for testing privacy protection.

2.  **Selective Efficacy (SE) Metric**:

    - **Function**: Unifies the measurement of multi-speaker understanding and bystander privacy protection.
    - **Mechanism**: SE is the harmonic mean of four accuracies: General/Selective mode accuracies for both the primary speaker and the bystander. $$SE = \frac{4}{\sum_{m,n} Acc_{m,n}^{-1}}$$. SE is high only when all four indicators are high; a low value in any category pulls down the overall score.
    - **Design Motivation**: Prevents models from gaming a single metric by always choosing IDK (high bystander Selective but low primary speaker) or always answering (high General but low privacy protection).

3.  **Bystander Privacy Fine-Tuning (BPFT)**:

    - **Function**: Teaches the model to refuse answering bystander-related questions when instructed.
    - **Mechanism**: The authors construct 3,768 synthetic audio mixed samples paired with 75k questions (half primary speaker, half bystander). Each question has two instruction sets (General and Selective). SFT is performed on the LLM backbone using LoRA (rank 32), while other modules are frozen.
    - **Design Motivation**: Training exclusively on synthetic data enables generalization to real scenarios without compromising the understanding of the primary speaker.

### Loss & Training
BPFT uses standard SFT loss, fine-tuning only the LLM backbone (LoRA rank 32) and freezing other modules like the audio encoder. The method is validated on Qwen-2.5-Omni 7B and Step-Audio-2-mini.

## Key Experimental Results

### Main Results

| Model | Main-Gen↑ | Main-Sel↑ | By-Gen↑ | By-Sel↑ | SE↑ |
|-------|-----------|-----------|---------|---------|-----|
| Gemini 2.5 Pro | 97.3 | 97.0 | 65.5 | 59.2 | 75.8 |
| Kimi-Audio 7B | 96.9 | 96.3 | 67.4 | 31.4 | 59.4 |
| Qwen-2.5-Omni 7B | 96.0 | 95.5 | 48.2 | 47.6 | 63.9 |
| Step-Audio-2-mini + BPFT | **97.4** | 94.3 | **81.0** | **96.1** | **91.7** |
| Qwen-2.5-Omni 7B + BPFT | 93.3 | 92.7 | 82.0 | 93.8 | 90.2 |

### Ablation Study

| Configuration | Main-Sel↑ | By-Sel↑ | SE↑ | Description |
|---------------|-----------|---------|-----|-------------|
| Step-Audio + BPFT w/ desc | 94.3 | 96.1 | 91.7 | Full model |
| Step-Audio + BPFT w/o desc | 93.9 | 94.1 | 91.1 | Removed speaker description, still maintains high performance |
| Step-Audio w/ desc | 93.7 | 31.5 | 56.1 | No BPFT; extremely poor bystander protection |
| Gemini 2.5 Pro w/ desc | 97.0 | 59.2 | 75.8 | Strongest commercial model achieves only 75.8% SE |

### Key Findings
- All models without BPFT perform extremely poorly in the bystander Selective mode (31-59%), demonstrating that strong audio understanding does not equate to privacy protection capability.
- BPFT brings a massive 50-60 percentage point improvement in bystander Selective accuracy and generalizes to real scenarios using only synthetic data.
- Speaker descriptions are important for models without BPFT (Kimi-Audio: 31.4% vs 22.0%), but have negligible impact on BPFT models (94.1% vs 96.1%).
- Llama-Omni 2 exhibits over-conservatism—always selecting IDK, resulting in an SE of only 34%.

## Highlights & Insights
- Systematically defines the bystander privacy problem in audio LLMs for the first time and constructs a comprehensive evaluation framework. This issue has significant real-world implications as voice assistants become ubiquitous.
- The SE metric is ingeniously designed; the harmonic mean ensures the model must excel in both understanding and privacy protection, preventing it from cheating via extreme strategies.
- BPFT achieves substantial privacy gains using only synthetic data, indicating that the primary bottleneck for models lies in behavioral alignment rather than raw capability.

## Limitations & Future Work
- BPFT causes a slight degradation in primary speaker accuracy on Qwen-2.5-Omni (96.0→93.3), indicating a minor trade-off.
- The evaluation is limited to English; multi-lingual scenarios remain to be validated.
- Five scenarios may not be sufficient to cover all real-world deployment environments.
- Bystanders are limited to single individuals; multi-bystander scenarios present a greater challenge.
- Future research could explore zero-shot privacy protection that does not rely on speaker descriptions.

## Related Work & Insights
- **vs SACRED-Bench**: While SACRED-Bench focuses on multi-speaker jailbreak attacks, this paper addresses bystander privacy, representing a complementary security dimension.
- **vs Representation Layer Anonymization**: Front-end defenses modify audio signals; this paper teaches the model to refuse to answer at the behavioral level, which is more flexible.
- **vs Pipeline Systems**: A pipeline system (speech separation + ASR + LLM) achieves an SE of only 65.9%, far underperforming BPFT's 91.7%.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to define and systematically study the bystander privacy problem in audio LLMs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across multiple models, though scenario and language coverage are limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition and sophisticated evaluation framework design.
- Value: ⭐⭐⭐⭐⭐ Addresses a highly practical privacy security issue; the framework can be directly applied to product deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval](omni-embed-audio_leveraging_multimodal_llms_for_robust_audio-text_retrieval.md)
- [\[ACL 2026\] Privacy-preserving Prosody Representation Learning](privacy-preserving_prosody_representation_learning.md)
- [\[ICML 2026\] Probing Cross-modal Information Hubs in Audio-Visual LLMs](../../ICML2026/audio_speech/probing_cross-modal_information_hubs_in_audio-visual_llms.md)
- [\[ACL 2026\] Mind the Pause: Disfluency-Aware Objective Tuning for Multilingual Speech Correction with LLMs](mind_the_pause_disfluency-aware_objective_tuning_for_multilingual_speech_correct.md)
- [\[ICML 2026\] Do Audio LLMs Listen or Read? Analyzing and Mitigating Paralinguistic Failures with VoxParadox](../../ICML2026/audio_speech/do_audio_llms_listen_or_read_analyzing_and_mitigating_paralinguistic_failures_wi.md)

</div>

<!-- RELATED:END -->
