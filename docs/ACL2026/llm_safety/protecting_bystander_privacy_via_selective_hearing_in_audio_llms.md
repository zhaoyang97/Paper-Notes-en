---
title: >-
  [Paper Note] Protecting Bystander Privacy via Selective Hearing in Audio LLMs
description: >-
  [ACL 2026][LLM Safety][bystander privacy] This work introduces SH-Bench, the first benchmark for bystander privacy evaluation, and proposes Bystander Privacy Fine-Tuning (BPFT)…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "bystander privacy"
  - "selective hearing"
  - "audio LLM"
  - "multi-speaker"
  - "privacy-preserving fine-tuning"
date: 2026-05-08
content_hash: 2e058ff752d8a043
---

# Protecting Bystander Privacy via Selective Hearing in Audio LLMs

**Conference**: ACL 2026
**arXiv**: [2512.06380](https://arxiv.org/abs/2512.06380)
**Code**: [GitHub](https://github.com/Elocinacademia/SelectiveHearing-Bench)
**Area**: AI Safety / Speech Privacy
**Keywords**: bystander privacy, selective hearing, audio LLM, multi-speaker, privacy-preserving fine-tuning

## TL;DR
This work introduces SH-Bench, the first benchmark for bystander privacy evaluation, and proposes Bystander Privacy Fine-Tuning (BPFT), a method that improves the ability of audio LLMs to focus exclusively on the target speaker and refuse to disclose bystander information in multi-speaker environments. After BPFT, the SE metric surpasses Gemini 2.5 Pro by 16%.

## Background & Motivation

**Background**: Audio LLMs (e.g., SALMONN, Qwen-Audio) are increasingly deployed in voice assistants and wearable devices, passively capturing speech in open environments. Existing privacy research has primarily focused on users who actively interact with the model.

**Limitations of Prior Work**: In real-world scenarios (cafés, public transit, etc.), audio LLMs inevitably capture the speech of surrounding bystanders. These bystanders do not actively interact with the system and are unaware that their speech is being processed, exposing them to serious privacy leakage risks. Existing benchmarks and defenses entirely overlook bystander privacy.

**Key Challenge**: Audio LLMs require strong multi-speaker understanding to serve the primary user, yet this same capability enables extraction of sensitive bystander information. There exists a fundamental tension between comprehension capability and privacy protection.

**Goal**: (1) Establish SH-Bench, the first benchmark for bystander privacy evaluation; (2) propose the unified SE metric to measure the balance between understanding and privacy protection; (3) design BPFT to enhance bystander privacy protection.

**Key Insight**: The paper introduces the concept of "selective hearing"—a model should attend only to the target speaker and selectively respond with "I don't know" to queries related to bystander speech.

**Core Idea**: By constructing multi-speaker audio samples containing both a primary speaker and bystanders, the model is trained to refuse answering bystander-related questions when instructed to protect privacy, while preserving its understanding of the primary speaker.

## Method

### Overall Architecture
SH-Bench comprises 3,968 multi-speaker audio mixture samples (approximately 157.5 hours), paired with 77k multiple-choice questions. Evaluation is conducted in two modes: General mode (answer all questions) and Selective mode (answer only primary-speaker-related questions; select "I don't know" for bystander-related ones). BPFT is a fine-tuning pipeline based on synthetic data.

### Key Designs

1. **SH-Bench Data Construction**:

    - Function: Provides multi-speaker bystander privacy evaluation data in both real and synthetic scenarios.
    - Mechanism: In real scenarios, participants recruited via Prolific record audio across five everyday settings (café, gym, public transit, etc.); the primary speaker records structured content while bystanders record informal, sensitive conversations. In synthetic scenarios, bystander audio from the AMI Meeting Corpus is mixed into primary-speaker audio at −10 dB. Each audio sample is paired with 10 five-option MCQs, one of which is always a variant of "I don't know."
    - Design Motivation: Real scenarios capture natural acoustic variation, while synthetic scenarios provide controlled, large-scale data. The IDK option is critical for testing privacy protection.

2. **Selective Efficacy (SE) Metric**:

    - Function: Jointly measures multi-speaker comprehension and bystander privacy protection.
    - Mechanism: SE is the harmonic mean of four accuracy scores—primary-speaker and bystander accuracy under both General and Selective modes: $SE = \frac{4}{\sum_{m,n} Acc_{m,n}^{-1}}$. SE is high only when all four component metrics are high; any single low value depresses the overall score.
    - Design Motivation: Prevents models from gaming a single metric by always selecting IDK (high bystander Selective but low primary-speaker accuracy) or always answering (high General but poor privacy protection).

3. **Bystander Privacy Fine-Tuning (BPFT)**:

    - Function: Trains the model to refuse answering bystander-related questions when instructed to do so.
    - Mechanism: A synthetic dataset of 3,768 audio mixture samples paired with 75k questions (evenly split between primary speaker and bystander) is constructed. Each question has two instruction variants (General and Selective). The LLM backbone is fine-tuned via LoRA (rank 32) with SFT, while other modules (e.g., the audio encoder) are frozen.
    - Design Motivation: Training on synthetic data alone generalizes to real-world scenarios without degrading primary-speaker comprehension.

### Loss & Training
BPFT applies standard SFT loss, fine-tuning only the LLM backbone via LoRA (rank 32) while freezing the audio encoder and other modules. Validation is performed on Qwen-2.5-Omni 7B and Step-Audio-2-mini.

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

| Configuration | Main-Sel↑ | By-Sel↑ | SE↑ | Note |
|---------------|-----------|---------|-----|------|
| Step-Audio + BPFT w/ desc | 94.3 | 96.1 | 91.7 | Full model |
| Step-Audio + BPFT w/o desc | 93.9 | 94.1 | 91.1 | Without speaker description; still high performance |
| Step-Audio w/ desc | 93.7 | 31.5 | 56.1 | Without BPFT; bystander protection is severely degraded |
| Gemini 2.5 Pro w/ desc | 97.0 | 59.2 | 75.8 | Strongest commercial model achieves only 75.8% SE |

### Key Findings
- All models without BPFT perform poorly in bystander Selective mode (31–59%), demonstrating that strong audio comprehension does not imply privacy protection.
- BPFT yields a 50–60 percentage point improvement in bystander Selective accuracy, and generalizes to real-world scenarios using only synthetic training data.
- Speaker descriptions are important for models without BPFT (Kimi-Audio: 31.4% vs. 22.0%) but have minimal impact after BPFT (94.1% vs. 96.1%).
- Llama-Omni 2 exhibits over-conservative behavior—always selecting IDK—resulting in an SE of only 34%.

## Highlights & Insights
- This work is the first to systematically define and study bystander privacy in audio LLMs, establishing a comprehensive evaluation framework. The problem is highly relevant given the widespread deployment of voice assistants.
- The SE metric is elegantly designed: the harmonic mean ensures that a model must perform well on both comprehension and privacy protection simultaneously, precluding exploitation through extreme strategies.
- BPFT achieves substantial privacy protection improvements using only synthetic data, indicating that the key bottleneck is behavioral alignment rather than model capability.

## Limitations & Future Work
- BPFT causes a slight decrease in primary-speaker accuracy on Qwen-2.5-Omni (96.0→93.3), indicating a non-trivial trade-off.
- Evaluation is limited to English; multilingual scenarios remain to be validated.
- The five scenarios may be insufficient to cover all real-world deployment environments.
- Only single-bystander settings are considered; multi-bystander scenarios pose additional challenges.
- Future work may explore zero-shot privacy protection that does not rely on speaker descriptions.

## Related Work & Insights
- **vs. SACRED-Bench**: Focuses on multi-speaker jailbreak attacks, whereas this work targets bystander privacy—complementary security dimensions.
- **vs. Representation-Level Anonymization**: Front-end defenses modify the audio signal; this work trains models to refuse responses at the behavioral level, offering greater flexibility.
- **vs. Pipeline Systems**: A speech separation + ASR + LLM pipeline achieves an SE of only 65.9%, far below BPFT's 91.7%.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to define and systematically study bystander privacy in audio LLMs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-model evaluation, though scenario and language coverage is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is clear and the evaluation framework is elegantly designed.
- Value: ⭐⭐⭐⭐⭐ Addresses a practically significant privacy and safety concern; the framework is directly applicable to real-world product deployment.

## Highlights & Insights
To be supplemented after a thorough reading of the paper.

## Limitations & Future Work
To be supplemented after a thorough reading of the paper.

## Related Work & Insights
To be supplemented after a thorough reading of the paper.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens](forget_what_matters_keep_the_rest_selective_unlearning_of_informative_tokens.md)
- [\[AAAI 2026\] StyleBreak: Revealing Alignment Vulnerabilities in Large Audio-Language Models via Style-Aware Audio Jailbreak](../../AAAI2026/llm_safety/stylebreak_revealing_alignment_vulnerabilities_in_large_audio-language_models_vi.md)
- [\[ACL 2026\] Indic-CodecFake meets SATYAM: Towards Detecting Neural Audio Codec Synthesized Speech Deepfakes in Indic Languages](indic-codecfake_meets_satyam_towards_detecting_neural_audio_codec_synthesized_sp.md)
- [\[ACL 2026\] MeasHalu: Mitigation of Scientific Measurement Hallucinations for LLMs](meashalu_mitigation_of_scientific_measurement_hallucinations_for_large_language_.md)
- [\[ICLR 2026\] AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models](../../ICLR2026/llm_safety/audiotrust_benchmarking_the_multifaceted_trustworthiness_of_audio_large_language.md)

</div>

<!-- RELATED:END -->
