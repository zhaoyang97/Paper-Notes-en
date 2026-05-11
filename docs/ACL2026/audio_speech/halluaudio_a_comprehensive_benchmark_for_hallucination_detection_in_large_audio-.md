---
title: >-
  [Paper Note] HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models
description: >-
  [ACL 2026][Audio & Speech][audio hallucination] This paper presents HalluAudio, the first large-scale cross-domain (speech/environmental sound/music) benchmark for hallucination detection in large audio-language models (…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "audio hallucination"
  - "large audio-language models"
  - "benchmark"
  - "adversarial prompting"
  - "multi-dimensional analysis"
date: 2026-05-08
content_hash: 9473e269d2561150
---

# HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models

**Conference**: ACL 2026
**arXiv**: [2604.19300](https://arxiv.org/abs/2604.19300)
**Code**: [https://github.com/Feiyuzhao25/halluaudio](https://github.com/Feiyuzhao25/halluaudio)
**Area**: Audio & Speech
**Keywords**: audio hallucination, large audio-language models, benchmark, adversarial prompting, multi-dimensional analysis

## TL;DR

This paper presents HalluAudio, the first large-scale cross-domain (speech/environmental sound/music) benchmark for hallucination detection in large audio-language models (LALMs), comprising 5,000+ human-verified QA pairs and a systematic adversarial prompt design. It evaluates mainstream LALMs across multiple dimensions (accuracy, hallucination rate, Yes-No bias, rejection rate, and error type), revealing significant deficiencies in acoustic grounding, temporal reasoning, and music attribute understanding.

## Background & Motivation

**Background**: LALMs have demonstrated strong capabilities in speech recognition, audio question answering, and music understanding. Hallucination has been extensively studied in the text and vision domains, but remains critically underexplored in the audio domain.

**Limitations of Prior Work**: (1) Existing audio benchmarks primarily target capability evaluation rather than reliability assessment; (2) the few audio hallucination studies (e.g., AHa-Bench) are small-scale, limited to binary classification, and lack diagnostic depth; (3) there is no systematic adversarial prompting or mixed-audio condition to elicit hallucinations.

**Key Challenge**: Models that perform well on standard benchmarks are not necessarily robust to hallucinations — a fundamental gap exists between capability evaluation and reliability evaluation.

**Goal**: To construct the first large-scale, cross-domain, multi-dimensional benchmark for audio hallucination detection, enabling systematic analysis of LALM failure modes.

**Key Insight**: Three domains (speech / environmental sound / music) × multiple task types (binary judgment / multi-choice reasoning / attribute verification / open-ended QA) × adversarial designs (adversarial prompts / mixed audio), paired with a multi-dimensional evaluation protocol.

**Core Idea**: Audio hallucination is defined as model-generated claims that are not supported by the input acoustic evidence, encompassing three subtypes: fabrication (asserting non-existent events), evidence contradiction, and ungrounded affirmation bias.

## Method

### Overall Architecture

HalluAudio is constructed via a five-stage pipeline: (1) *audio selection* — curated from high-quality annotated corpora including Common Voice, FSD50K, and GTZAN; (2) *templatized prompt generation* — parameterized prompt templates instantiated with positive and negative examples; (3) *adversarial construction* — minimally modified prompts or audio attributes to create controlled positive/negative contrast pairs; (4) *validation and quality control* — three rounds of human verification (two independent annotators + one senior reviewer); (5) *packaging and balancing* — balanced across domains, task types, and hallucination categories.

### Key Designs

1. **Three-Domain Multi-Task Evaluation Framework**:

    - *Function*: Covers hallucination behavior across three major audio domains — speech, environmental sound, and music.
    - *Mechanism*: Speech tasks include overlap detection, word-order judgment, counting, gender verification, noise verification, transcription matching, and speed/loudness comparison. Environmental sound tasks include overlap, sequence, existence, and co-occurrence detection, mismatch queries, multi-label checks, and loudness comparison. Music tasks include genre matching, instrument presence, tempo/speed comparison, and key identification. Each task category has an explicit hallucination-elicitation mechanism.
    - *Design Motivation*: Different audio domains exhibit distinct hallucination patterns — temporal hallucination in speech, event fabrication in environmental sound, and attribute misidentification in music. Coverage across all three domains ensures comprehensive diagnosis.

2. **Adversarial Prompt and Mixed-Audio Design**:

    - *Function*: Systematically elicit and measure hallucinations.
    - *Mechanism*: Adversarial prompts use deliberately misleading descriptions to test whether models blindly agree (e.g., asking "What did the female speaker say?" for a recording of a male speaker). Mixed audio concatenates two audio clips to test whether models correctly distinguish temporal order and event attribution. Positive/negative contrast pairs isolate hallucination triggers through minimal modification (altering only a single attribute).
    - *Design Motivation*: Standard tests fail to expose hallucinations — models may perform well on normal inputs but fail catastrophically on adversarial ones. Systematic biases such as Yes/No imbalance require purpose-built test designs to surface.

3. **Multi-Dimensional Evaluation Metric System**:

    - *Function*: Enables comprehensive failure-mode analysis beyond accuracy.
    - *Mechanism*: (1) *Accuracy* — basic correctness; (2) *Hallucination rate* — proportion of responses in which the model fabricates non-existent facts; (3) *Yes/No bias* — systematic tendency toward affirmative or negative responses; (4) *Error type analysis* — distinguishing fabrication, contradiction, and affirmation bias; (5) *Rejection rate* — proportion of responses in which the model refuses to answer.
    - *Design Motivation*: Accuracy alone conceals systematic biases. Yes/No bias and rejection behavior represent failure modes specific to LALMs.

### Loss & Training

HalluAudio is an evaluation benchmark and does not involve model training. A unified zero-shot evaluation protocol is adopted, with outputs standardized and validated through an automated evaluation engine.

## Key Experimental Results

### Main Results

**Average accuracy of mainstream LALMs across three domains**

| Model | Speech Acc | Env. Sound Acc | Music Acc | Overall Acc |
|-------|-----------|----------------|-----------|-------------|
| Gemini-2.5-Pro | Top tier | Top tier | Top tier | ~70–80% |
| Qwen2-Audio | Mid tier | Mid tier | Low | ~50–60% |
| SALMONN | Low | Mid tier | Low | ~40–50% |

### Ablation Study

| Dimension | Finding | Explanation |
|-----------|---------|-------------|
| Yes/No bias | Most models lean toward "Yes" | Ungrounded affirmation bias is pervasive |
| Rejection behavior | Some models refuse frequently | Over-alignment with safety constraints |
| Domain difference | Music is hardest | Music attribute understanding is weakest |
| Adversarial vs. standard | Significant accuracy drop | Confirms that hallucinations are not exposed by standard evaluation |

### Key Findings

- The music domain is the greatest weakness across all models — understanding of music attributes (key, tempo, instrument details) is severely lacking.
- Systematic Yes/No bias is pervasive — models tend to unconditionally affirm, even when the queried element is absent from the audio.
- High scores on standard benchmarks do not imply hallucination robustness — the gap between capability evaluation and reliability evaluation is equally pronounced in the audio domain.
- Closed-source large models generally outperform open-source models in hallucination resistance, though the margin is smaller than in the text and vision domains.

## Highlights & Insights

- HalluAudio is the first systematic audio hallucination benchmark, filling a critical gap given that hallucination research is abundant in text and vision yet nearly absent in audio.
- The three-domain × multi-task × multi-dimensional metric design provides unprecedented diagnostic granularity.
- Analysis of Yes/No bias and rejection rate reveals systemic failure modes specific to LALMs.

## Limitations & Future Work

- The dataset scale (5K+) remains relatively small compared to vision hallucination benchmarks.
- Audio sources are drawn from a limited set of datasets, which may not cover all real-world scenarios.
- Multilingual speech hallucination is not addressed.
- Future work could extend to audio-video joint scenarios and conversational audio understanding.

## Related Work & Insights

- **vs. AHa-Bench**: AHa-Bench is a small-scale binary QA dataset; HalluAudio provides multi-task, multi-dimensional comprehensive evaluation.
- **vs. CHAIR (vision)**: CHAIR detects object-level hallucinations; HalluAudio transfers a similar paradigm to the audio domain.
- **vs. Frieske & Shi (2024)**: Their work analyzes only ASR hallucinations; HalluAudio covers all three domains of speech, environmental sound, and music.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First large-scale cross-domain audio hallucination benchmark, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-dimensional evaluation; in-depth analysis of model behavior could be further elaborated.
- Writing Quality: ⭐⭐⭐⭐ Benchmark design is clearly presented with a systematic taxonomy.
- Value: ⭐⭐⭐⭐⭐ Provides a much-needed evaluation tool for audio AI safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?](speakersleuth_can_large_audio-language_models_judge_speaker_consistency_across_m.md)
- [\[ACL 2026\] Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models](temporal_contrastive_decoding_a_training-free_method_for_large_audio-language_mo.md)
- [\[ACL 2026\] Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models](breaking_block_boundaries_anchor-based_history-stable_decoding_for_diffusion_lar.md)
- [\[ACL 2026\] SEPT: Semantically Expanded Prompt Tuning for Audio-Language Models](generalizable_prompt_tuning_for_audio-language_models_via_semantic_expansion.md)
- [\[AAAI 2026\] AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions](../../AAAI2026/audio_speech/ahamask_reliable_task_specification_for_large_audio_language.md)

</div>

<!-- RELATED:END -->
