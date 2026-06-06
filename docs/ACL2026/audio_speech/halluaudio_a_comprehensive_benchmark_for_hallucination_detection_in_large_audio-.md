---
title: >-
  [Paper Note] HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models
description: >-
  [ACL 2026][Audio & Speech][Audio Hallucination] This paper proposes HalluAudio, the first large-scale cross-domain (speech/environmental sound/music) benchmark for audio hallucination detection. It comprises over 5…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Audio Hallucination"
  - "Large Audio-Language Models"
  - "Benchmark Evaluation"
  - "Adversarial Prompting"
  - "Multidimensional Analysis"
date: 2026-05-08
content_hash: cec3fbacbe7ad040
---

# HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.19300](https://arxiv.org/abs/2604.19300)  
**Code**: [https://github.com/Feiyuzhao25/halluaudio](https://github.com/Feiyuzhao25/halluaudio)  
**Area**: Audio & Speech  
**Keywords**: Audio Hallucination, Large Audio-Language Models, Benchmark Evaluation, Adversarial Prompting, Multidimensional Analysis

## TL;DR

This paper proposes HalluAudio, the first large-scale cross-domain (speech/environmental sound/music) benchmark for audio hallucination detection. It comprises over 5,000 human-verified QA pairs with systematic adversarial prompt designs. By evaluating mainstream LALMs across multidimensional metrics (Accuracy, Hallucination Rate, Yes-No Bias, Rejection Rate, and Error Types), the study reveals significant deficiencies in current models regarding acoustic anchoring, temporal reasoning, and music attribute understanding.

## Background & Motivation

**Background**: Large Audio-Language Models (LALMs) have demonstrated powerful capabilities in speech recognition, sound-based QA, and music understanding. While hallucination issues have been extensively studied in text and vision domains, research in the audio domain remains severely insufficient.

**Limitations of Prior Work**: (1) Existing audio benchmarks primarily focus on capability evaluation rather than reliability; (2) The few studies on audio hallucinations (e.g., AHa-Bench) are small-scale, limited to binary classification, and lack diagnostic depth; (3) There is a lack of systematic adversarial prompting and mixed-audio conditions to induce hallucinations.

**Key Challenge**: Models performing strongly on standard benchmarks do not necessarily resist hallucinations—a gap exists between capability evaluation and reliability assessment.

**Goal**: To build the first large-scale, cross-domain, and multi-dimensional audio hallucination detection benchmark to systematically analyze the failure modes of LALMs.

**Key Insight**: Utilize three domains (speech/environmental sound/music) $\times$ multiple task types (binary judgment/multiple-choice reasoning/attribute verification/open-ended QA) $\times$ adversarial designs (adversarial prompts/mixed audio), complemented by multidimensional evaluation metrics.

**Core Idea**: Audio hallucination is defined as the model generating claims unsupported by input acoustic evidence, including fabrication (claiming non-existent events), evidence contradiction, and unfounded affirmative bias.

## Method

### Overall Architecture

HalluAudio is constructed via a five-step pipeline: (1) Audio Selection—chosen from high-quality labeled corpora like Common Voice, FSD50K, and GTZAN; (2) Templated Prompt Generation—parameterized prompt templates with positive/negative instantiations; (3) Adversarial Construction—creating controlled positive-negative contrasts through minimal modifications to prompts or audio attributes; (4) Verification and Quality Control—three rounds of human verification (two independent annotators + one senior auditor); (5) Packaging and Balancing—balancing across domains, task types, and hallucination categories.

### Key Designs

1.  **Three-Domain Multi-Task Evaluation System**:
    - **Function**: Covers hallucination behaviors across three major audio domains: speech, environmental sounds, and music.
    - **Mechanism**: Speech tasks include overlap detection, word order judgment, counting, gender verification, noise verification, transcription matching, and speed/loudness comparison. Environmental sound tasks include overlap/sequence/existence/co-occurrence detection, mismatch queries, multi-label checks, and loudness comparison. Music tasks include genre matching, instrument presence, rhythm/tempo comparison, and tonality discrimination. Each task incorporates explicit hallucination induction mechanisms.
    - **Design Motivation**: Different audio domains exhibit distinct hallucination patterns—temporal hallucinations in speech, event fabrication in environmental sounds, and attribute misjudgment in music. Covering all three ensures comprehensive diagnosis.

2.  **Adversarial Prompting and Mixed Audio Design**:
    - **Function**: Systematically induces and measures hallucinations.
    - **Mechanism**: Adversarial prompts use deliberately misleading descriptions to test if the model blindly agrees (e.g., asking "What did the female voice say?" for a male-only recording). Mixed audio concatenates two clips to test if the model correctly distinguishes temporal order and event attribution. Positive/negative contrast groups isolate hallucination triggers via minimal modifications (changing only one attribute).
    - **Design Motivation**: Standard tests fail to expose hallucinations—models may perform well on standard inputs but collapse on adversarial ones. Systematic issues like Yes/No bias require specifically designed tests to be uncovered.

3.  **Multidimensional Evaluation Metric System**:
    - **Function**: Comprehensive failure mode analysis beyond simple accuracy.
    - **Mechanism**: (1) Accuracy—baseline correctness; (2) Hallucination Rate—the proportion of model-fabricated non-existent facts; (3) Yes/No Bias—whether the model systematically favors affirmative or negative responses; (4) Error Type Analysis—distinguishing between fabrication, contradiction, and affirmative bias; (5) Rejection Rate—the proportion of cases where the model refuses to answer.
    - **Design Motivation**: Relying solely on accuracy masks systematic biases. Yes/No bias and refusal behaviors are unique failure modes of LALMs.

### Loss & Training

HalluAudio is an evaluation benchmark and does not involve model training. It employs a unified zero-shot evaluation protocol, with outputs standardized and verified by an automated evaluation engine.

## Key Experimental Results

### Main Results

**Average Accuracy of Mainstream LALMs Across Three Domains**

| Model | Speech Acc | Env. Sound Acc | Music Acc | Overall Acc |
| :--- | :--- | :--- | :--- | :--- |
| Gemini-2.5-Pro | Top Tier | Top Tier | Top Tier | ~70-80% |
| Qwen2-Audio | Medium | Medium | Low | ~50-60% |
| SALMONN | Low | Medium | Low | ~40-50% |

### Ablation Study

| Dimension | Key Findings | Description |
| :--- | :--- | :--- |
| Yes/No Bias | Most models tend towards "Yes" | Unfounded affirmative bias is widespread. |
| Rejection Behavior | Some models reject frequently | Indicates over-alignment for safety. |
| Domain Variance | Music is the most difficult | Understanding of musical attributes is weakest. |
| Adversarial vs. Standard | Significant drop | Confirms that hallucination issues are not evident in standard evaluations. |

### Key Findings

*   The music domain is the single greatest weakness for all models—understanding of musical attributes (tonality, rhythm, instrumental details) is severely lacking.
*   Systematic Yes/No bias is prevalent—models tend to provide unconditional affirmations, even when the queried elements are absent from the audio.
*   High scores on standard benchmarks $\neq$ hallucination robustness—the gap between capability and reliability is as significant in the audio domain as it is in others.
*   Closed-source LLMs generally outperform open-source models in hallucination resistance, though the gap is less pronounced than in text or vision domains.

## Highlights & Insights

*   The first systematic audio hallucination benchmark—filling the void in audio research compared to the mature hallucination studies in text and vision.
*   The design of three domains $\times$ multiple tasks $\times$ multidimensional metrics provides unprecedented diagnostic granularity.
*   Analysis of Yes/No bias and rejection rates reveals systematic issues unique to LALMs.

## Limitations & Future Work

*   The dataset size (5K+) is still relatively small compared to visual hallucination benchmarks.
*   Audio sources are derived from a limited set of datasets, potentially failing to cover all real-world scenarios.
*   Multilingual speech hallucinations are not yet addressed.
*   Future work could extend to audio-visual joint scenarios and conversational audio understanding.

## Related Work & Insights

*   **vs. AHa-Bench**: AHa-Bench uses small-scale binary QA; HalluAudio provides comprehensive multi-task, multidimensional evaluation.
*   **vs. CHAIR (Vision)**: While CHAIR detects object-level hallucinations, HalluAudio adapts similar logic to the audio domain.
*   **vs. Frieske & Shi (2024)**: Their work focuses solely on ASR hallucinations, whereas HalluAudio covers speech, environmental sound, and music.

## Rating

*   **Novelty**: ⭐⭐⭐⭐⭐ First large-scale cross-domain audio hallucination benchmark; fills a critical gap.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated multiple models across several dimensions, though deeper analysis of specific model behaviors could be expanded.
*   **Writing Quality**: ⭐⭐⭐⭐ Clear benchmark design and systematic taxonomy.
*   **Value**: ⭐⭐⭐⭐⭐ Provides a much-needed evaluation tool for audio AI safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HCFD: A Benchmark for Audio Deepfake Detection in Healthcare](hcfd_a_benchmark_for_audio_deepfake_detection_in_healthcare.md)
- [\[ACL 2026\] SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?](speakersleuth_can_large_audio-language_models_judge_speaker_consistency_across_m.md)
- [\[ACL 2026\] Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models](temporal_contrastive_decoding_a_training-free_method_for_large_audio-language_mo.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[ACL 2026\] MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models](mtr-duplexbench_towards_a_comprehensive_evaluation_of_multi-round_conversations_.md)

</div>

<!-- RELATED:END -->
