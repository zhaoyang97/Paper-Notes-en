---
title: >-
  [Paper Note] MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio
description: >-
  [ICML 2026][Audio & Speech][Medical Audio QA] MedMosaic constructs a medical audio QA benchmark (46,701 QA pairs, 10 question types) covering physiological sounds and real/synthetic clinical dialogues via a synthetic pip…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "Medical Audio QA"
  - "Synthetic Clinical Speech"
  - "Multi-turn Reasoning"
  - "Open-ended Answering"
  - "Embedded Voice QA"
date: 2026-05-08
content_hash: 4bc941fbe4a2d38c
---

# MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio

**Conference**: ICML 2026  
**arXiv**: [2605.00969](https://arxiv.org/abs/2605.00969)  
**Code**: Sample data https://shorturl.at/Lyp33  
**Area**: Medical Audio / Multimodal Evaluation  
**Keywords**: Medical Audio QA, Synthetic Clinical Speech, Multi-turn Reasoning, Open-ended Answering, Embedded Voice QA

## TL;DR
MedMosaic constructs a medical audio QA benchmark (46,701 QA pairs, 10 question types) covering physiological sounds and real/synthetic clinical dialogues via a synthetic pipeline. It systematically evaluates 13 audio/multimodal models, revealing that even Gemini-2.5-Pro achieves only about 68.1% weighted accuracy, exposing fundamental limitations of contemporary LALMs in medical audio reasoning.

## Background & Motivation

**Background**: With the rise of LLMs/MLLMs/LALMs, evaluation focus has shifted from "unimodal recognition" to "cross-modal multi-step reasoning." General audio QA already has mature benchmarks such as ClothoAQA, MMAU, MMAU-Pro, MDAR, MMAR, AudioBench, and AudioPedia; on the model side, Qwen-Audio, Audio Flamingo, SALMONN, LTU-AS, AudioPaLM, etc., are rapidly advancing.

**Limitations of Prior Work**: (1) Existing audio QA is almost entirely focused on general environmental sounds, music, or short speech segments; medical audio is extremely scarce, with CaReAQA being one of the few attempts, but it is small-scale and only tests short, isolated clips. (2) Text-based medical QA (MedQA, MeDiaQA) removes all acoustic information, making it impossible to assess clinical cues such as "cough characteristics, breathing rhythm, vocal stress, hesitation in dialogue" that only audio can convey. (3) Evaluation protocols overly rely on closed multiple-choice, failing to assess generative reasoning; there is a lack of scenarios involving long dialogues, multi-turn interactions, and embedded QA, which are common in real clinical settings.

**Key Challenge**: Medical decision-making heavily depends on the ability to "align semantics with acoustic markers," yet existing benchmarks lack both long-duration, multi-source audio data and multi-hop clinical reasoning tasks; meanwhile, medical data is difficult to scale due to privacy and annotation costs.

**Goal**: (i) Build a large-scale medical audio QA benchmark spanning multiple audio types (physiological sounds + short/long clinical dialogues) and covering various reasoning modes (multiple-choice, multi-turn, open-ended, embedded voice QA); (ii) Propose a controllable synthetic audio generation pipeline to enable on-demand benchmark expansion; (iii) Systematically evaluate mainstream LALMs to quantify current performance ceilings.

**Key Insight**: The authors find that "synthesis + expert prompting" enables precise control over clinical scenario complexity (e.g., cough embedding, emotional markers, timeline information distribution) without touching real patient data. Using Gemini-3-flash as a QA generator, combined with carefully designed prompts (10 highly similar contrastive options per question + anti-hallucination constraints), allows for the creation of large-scale, challenging questions.

**Core Idea**: By leveraging a "synthetic pipeline + strict anti-hallucination prompts + 10 question types," a large and difficult medical audio QA benchmark is constructed, incorporating open-ended and embedded voice QA to expose LALM medical reasoning abilities to multidimensional testing.

## Method

### Overall Architecture
MedMosaic consists of two parts: (A) QA generation pipeline—using specialized Gemini-3-flash prompts for each audio type (sound-only physiological sounds, speech-only clinical dialogues, speech+sound mixtures), generating 10 contrastive options per question (except open-ended), and three difficulty levels (easy/medium/hard). Each question enforces "lexically similar but interpretively distinct" options to prevent keyword guessing; (B) Question types—10 in total: MCQ_Sound_(Cough/Heart/Lung), MCQ_Speech, MCQ_Speech_Sound, MCQ_Long_Form, Multi_Turn, OE_Speech, OE_Speech_Sound, Voice_QA, covering single/multi-source, long-duration, multi-turn, open-ended, and embedded scenarios. Ultimately, 46,701 QA pairs are generated, with weighted average accuracy calculated by type.

### Key Designs

1. **Fine-grained Temporal Construction for Physiological Sound QA**:

    - **Function**: Ensures sound-only QA requires "temporal reasoning" rather than mere "sound classification."
    - **Mechanism**: Physiological sounds are subdivided into clinically relevant subtypes—lung sounds (wheeze: sustained narrowband / crackle: brief broadband / stridor: high-pitched continuous single frequency), coughs (wet: moist / dry: brief and dry / pertussis: paroxysmal + high-energy inspiratory whoop / barky: low-pitched resonant), heart sounds (murmur: different at S1 → systole → S2 → diastole). Questions go beyond "what is this sound" to ask "at which respiratory phase did the cough occur," "heartbeat rhythm changes," "estimate breaths in 30 seconds," "sound/silence time ratio"—all requiring models to anchor acoustic events to physiological cycles.
    - **Design Motivation**: Surface-level sound classification can be guessed from a few features, but "temporal coupling + counting estimation" forces models to truly parse the internal time structure of audio, preventing shortcutting via pretrained general knowledge.

2. **Strong Contrastive MCQ + Anti-hallucination Prompt Engineering**:

    - **Function**: Makes multiple-choice difficulty depend on "understanding audio details" rather than "easily distinguishable options."
    - **Mechanism**: (i) Each question has 10 options, with distractors designed to be lexically similar yet semantically distinct from the correct answer, deliberately reusing keywords to increase surface similarity and invalidate keyword matching; (ii) Common distractor patterns—temporal misalignment (correct event, wrong phase), similar acoustic features but different clinical interpretations, over-reliance on common associations in training data; (iii) Anti-hallucination constraint—all correct answers must be derivable from the audio itself, prohibiting reliance on external medical knowledge bases; each option must lead to an independent clinical interpretation, preventing elimination by common sense; (iv) Three difficulty levels (Easy/Medium/Hard) systematically increase perceptual precision requirements.
    - **Design Motivation**: Existing medical QA can be answered by LLMs memorizing medical knowledge, ignoring audio; strong constraints ensure questions "cannot be answered without listening," truly testing audio reasoning.

3. **Embedded Voice QA (Voice_QA) + Multi-turn + Open-ended: Three Novel Question Types**:

    - **Function**: Incorporates three real clinical interaction scenarios—"interleaved questions and dialogue," "multi-step follow-up," and "generation rather than selection"—into evaluation.
    - **Mechanism**: Voice_QA embeds questions and answers directly into the audio waveform—models must switch context after listening to clinical dialogue to answer the embedded voice question, testing context switching and resistance to attention drift; Multi_Turn involves multi-turn follow-up on long dialogues, requiring models to maintain state across turns; Open-Ended (OE_Speech / OE_Speech_Sound) requires unconstrained generation on long audio, with concise but correct answers, representing the strictest generative reasoning test.
    - **Design Motivation**: MCQ tests "discrimination," but real clinical scenarios are almost always "doctor speaks after listening" generation; embedded QA further simulates real clinical interactions where devices/colleagues insert questions during patient dialogue.

### Loss & Training
This is not a training paper, so no loss is involved. All QA pairs are generated by Gemini-3-flash, and 13 candidate models (Audio Flamingo 3, Audio Reasoner, Baichuan-Omni, Desta25-Audio, Gama, Gemini-2.5-flash/pro, Qwen-2.5-Omni, etc.) are evaluated for inference.

## Key Experimental Results

### Main Results (Table 1 excerpt, accuracy %)

| Model | Weighted Avg | MCQ_Speech | MCQ_Sound_Heart | OE_Speech | Voice_QA |
|---|---|---|---|---|---|
| Audio-flamingo-3 | 24.1 | 10.7 | 37.8 | 55.2 | 0.1 |
| Audio-reasoner | 32.8 | 23.7 | 35.6 | 51.2 | 9.9 |
| Baichuan-omni | 38.6 | 43.5 | 26.6 | 57.6 | 31.5 |
| Desta25-audio | 41.0 | 49.4 | 37.1 | 56.0 | 9.1 |
| Gama | 23.2 | 12.7 | 36.6 | 38.1 | 8.9 |
| Gemini-2.5-flash | 60.5 | 73.6 | 52.8 | ... | ... |
| **Gemini-2.5-Pro** | **~68.1** | (Best per column in paper) | | | |
| Qwen-2.5-Omni-7B | 42.8 | ... | ... | ... | ... |

Even the strongest commercial model, Gemini-2.5-Pro, achieves only 68.1% weighted average, demonstrating the benchmark's difficulty.

### Ablation Study / Question Type Comparison

| Phenomenon | Explanation |
|---|---|
| Most models score <32% on Voice_QA, some even <1% | Embedded voice QA is the biggest weakness—context switching ability is very poor |
| OE_Speech generally outperforms MCQ_Speech | Higher open-ended scores are due to lenient grading (credit given for correct facts), not true understanding |
| MCQ_Sound_Heart > MCQ_Sound_Cough / Lung generally | Heart sounds have more regular temporal structure (S1/S2), making them easier to recognize than the randomness of cough/lung sounds |
| MCQ_Long_Form generally low | Long dialogue reasoning is a common weakness, consistent with literature noting LALMs' poor long-context handling |

### Key Findings
- Even the strongest general models are far below human clinical performance (>90%), proving that medical audio reasoning is far from being covered by current LALMs; specialized pretraining/adaptation is necessary.
- Audio-flamingo-3 scores almost zero (0.1%) on Voice_QA, indicating a complete lack of context switching ability—an entirely new evaluation dimension revealed by embedded QA.
- The synthetic QA pipeline finds a working point between "minimal human supervision" and "benchmark remains difficult," validating "synthetic evaluation data" as a scalable paradigm for sensitive domains like medicine/privacy.

## Highlights & Insights
- Decomposing medical audio into a matrix of "sound-only / speech-only / speech+sound / voice-embedded" orthogonal question types enables precise diagnosis of model weaknesses by dimension—a replicable methodology for medical/clinical evaluation.
- The anti-hallucination constraint—"correct answers must be derivable from audio, distractors must have independent clinical interpretations"—is a robust prompt engineering paradigm, transferable to other professional QA datasets to prevent LLMs from shortcutting via memorized knowledge.
- The design of Voice_QA, embedding questions into waveforms, is truly innovative—clinically, doctors must answer colleagues' questions while listening to patients, and this "continuous monitoring + interruption response" ability is entirely missing from existing benchmarks.

## Limitations & Future Work
- Data is synthetic rather than real clinical recordings, so there remains a domain gap with actual patient/doctor dialogues; the authors attempt to mitigate this by "retaining clinical artistry + embedding physical artifacts," but cannot fully eliminate it.
- Annotation relies on Gemini-3-flash generation, risking the introduction of generator biases into the benchmark; sample validation is limited in scale.
- The 13 evaluated models are still mainly general-purpose LALMs, lacking comparison with models specifically fine-tuned for medical audio (e.g., future MedAudio-LLM).
- The open-ended scoring protocol is described only briefly in the paper, leaving room for improved reproducibility.

## Related Work & Insights
- **vs CaReAQA**: CaReAQA also targets medical audio but is small-scale and only tests short clips; MedMosaic uses a synthetic pipeline to scale up by two orders of magnitude and adds long dialogue/multi-turn/embedded QA.
- **vs MMAU / MMAU-Pro / MMAR**: General audio QA, broad coverage but not specialized; MedMosaic provides depth in the medical subdomain, complementing MMAU-Pro.
- **vs CORAAL-QA**: CORAAL-QA focuses on long-form multi-turn interaction; MedMosaic introduces domain expertise and physiological sound specificity.
- **vs MedQA (text)**: MedQA is entirely text-based clinical knowledge; this work is the first to systematically supplement "audio-dimension clinical reasoning evaluation."

## Rating
- Novelty: ⭐⭐⭐⭐ Embedded Voice_QA, multi-turn dialogue, and temporal reasoning for physiological sounds all appear for the first time in a medical audio benchmark.
- Experimental Thoroughness: ⭐⭐⭐⭐ 13 models × 10 question types evaluated, with per-model, per-type scores; missing: human baseline, medical expert fine-tuned models.
- Writing Quality: ⭐⭐⭐ Clear flowcharts and detailed prompt templates; some experimental details (e.g., open-ended scoring metrics) are only briefly mentioned.
- Value: ⭐⭐⭐⭐ Provides the first large-scale, extensible evaluation for medical audio LALMs, highly practical for future multimodal medical model development; the synthetic data paradigm is also instructive for other privacy-sensitive domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety](multibreak_a_scalable_and_diverse_multi-turn_jailbreak_benchmark_for_evaluating_.md)
- [\[ACL 2026\] DuIVRS-2: An LLM-based Interactive Voice Response System for Large-scale POI Attribute Acquisition](../../ACL2026/audio_speech/duivrs-2_an_llm-based_interactive_voice_response_system_for_large-scale_poi_attr.md)
- [\[ICML 2026\] MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks](mecat_a_multi-experts_constructed_benchmark_for_fine-grained_audio_understanding.md)
- [\[ICML 2026\] VocSim: A Training-Free Benchmark for Zero-Shot Content Identity Recognition of Single-Source Audio](vocsim_a_training-free_benchmark_for_zero-shot_content_identity_in_single-source.md)
- [\[ACL 2026\] HCFD: A Benchmark for Audio Deepfake Detection in Healthcare](../../ACL2026/audio_speech/hcfd_a_benchmark_for_audio_deepfake_detection_in_healthcare.md)

</div>

<!-- RELATED:END -->
