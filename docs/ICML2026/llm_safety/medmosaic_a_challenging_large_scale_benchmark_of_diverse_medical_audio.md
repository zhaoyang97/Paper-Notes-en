---
title: >-
  [Paper Note] MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio
description: >-
  [ICML 2026][LLM Safety][Medical Audio QA] MedMosaic constructs a medical audio QA benchmark (46,701 QA pairs, 10 question types) covering physiological sounds and real/synthetic clinical dialogues using a synthetic pipel…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Medical Audio QA"
  - "Synthetic Clinical Speech"
  - "Multi-turn Reasoning"
  - "Open-ended QA"
  - "Embedded Voice QA"
date: 2026-05-08
content_hash: 87db67a3bd26997e
---

# MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio

**Conference**: ICML 2026  
**arXiv**: [2605.00969](https://arxiv.org/abs/2605.00969)  
**Code**: Sample data https://shorturl.at/Lyp33  
**Area**: Medical Audio / Multimodal Evaluation  
**Keywords**: Medical Audio QA, Synthetic Clinical Speech, Multi-turn Reasoning, Open-ended QA, Embedded Voice QA

## TL;DR
MedMosaic constructs a medical audio QA benchmark (46,701 QA pairs, 10 question types) covering physiological sounds and real/synthetic clinical dialogues using a synthetic pipeline. Systematic evaluation of 13 audio/multimodal models reveals that even Gemini-2.5-Pro achieves only approximately 68.1% weighted accuracy, highlighting fundamental deficiencies in contemporary LALMs regarding medical audio reasoning.

## Background & Motivation

**Background**: With the rise of LLMs/MLLMs/LALMs, evaluation focus has shifted from "single-modality recognition" to "cross-modal multi-step reasoning." General audio QA benchmarks such as ClothoAQA, MMAU, MMAU-Pro, MDAR, MMAR, AudioBench, and AudioPedia are mature, and model-side progress is rapid with Qwen-Audio, Audio Flamingo, SALMONN, LTU-AS, and AudioPaLM.

**Limitations of Prior Work**: (1) Existing audio QA focuses almost exclusively on general ambient sounds, music, and short speech segments; medical audio is extremely scarce, with CaReAQA being a rare but small-scale attempt limited to short independent fragments. (2) Text-based medical QA (MedQA, MeDiaQA) removes all acoustic information, failing to evaluate clinical clues conveyed only by sound, such as "cough nature, respiratory rhythm, vocal stress, and dialogue hesitation." (3) Evaluation protocols rely excessively on closed-ended multiple-choice questions, failing to examine generative reasoning; there is a lack of scenarios involving long-form dialogues, multi-turn interactions, and embedded voice QA typical of real clinical interactions.

**Key Challenge**: Medical decision-making relies heavily on the ability to "align semantics with acoustic markers," yet existing benchmarks lack such long-duration, multi-source audio data and tasks involving multi-hop clinical reasoning. Meanwhile, medical data is difficult to collect at scale due to privacy and annotation costs.

**Goal**: (i) Construct a large-scale medical audio QA benchmark across multiple audio types (physiological sounds + short/long clinical dialogues) and reasoning modes (multiple-choice, multi-turn, open-ended, embedded voice); (ii) Propose a controllable synthetic audio generation pipeline for benchmark scalability; (iii) Systematically evaluate mainstream LALMs to quantify the current performance ceiling.

**Key Insight**: The authors found that "synthesis + expert prompting" can precisely control the complexity of clinical scenarios (cough embedding, emotional markers, timeline information distribution) without accessing real patient data. Using Gemini-3-flash as a QA generator with carefully designed prompts (10 highly similar distractors per question + anti-hallucination constraints) produced a large-scale and difficult dataset.

**Core Idea**: Create a large and challenging medical audio QA benchmark using a "synthetic pipeline + rigorous anti-hallucination prompting + 10 question types," incorporating open-ended and embedded voice QA to expose the medical reasoning capabilities of LALMs under multi-dimensional testing.

## Method

### Overall Architecture
MedMosaic consists of two parts: (A) A QA generation pipeline—utilizing specialized Gemini-3-flash prompts based on audio types (sound-only physiological sounds, speech-only clinical dialogues, and speech+sound mixtures), generating 10 distractors per question (excluding open-ended) across three difficulty levels (easy/medium/hard). Distractors are forced to be "lexically similar but interpretively distinct." (B) 10 Question types: MCQ_Sound_(Cough/Heart/Lung), MCQ_Speech, MCQ_Speech_Sound, MCQ_Long_Form, Multi_Turn, OE_Speech, OE_Speech_Sound, and Voice_QA, covering single/multi-source, long-duration, multi-turn, open-ended, and embedded formats. A total of 46,701 QA pairs were generated, with performance measured via weighted average accuracy across types.

### Key Designs

1.  **Fine-grained Temporal Construction of Physiological Sound QA**:
    *   **Function**: Ensure sound-only QA requires "temporal reasoning" rather than simple "sound classification."
    *   **Mechanism**: Subdivide physiological sounds into clinically relevant subcategories—lung sounds (wheeze: continuous narrowband / crackle: short explosive broadband / stridor: high-pitched continuous monophonic), cough (wet: aqueous / dry: short and dry / pertussis: clusters + high-energy inspiratory whoop / barky: deep resonance), and heart sounds (murmurs at different stages: $S_1 \rightarrow \text{systole} \rightarrow S_2 \rightarrow \text{diastole}$). Questions go beyond "what is this" to ask about "respiratory phase of cough," "heartbeat rhythm changes," "estimated breath count in 30s," or "sound/silence duration ratios."
    *   **Design Motivation**: Surface sound classification can be guessed via signature features, but "temporal coupling + estimation" forces models to parse the internal temporal structure, preventing reliance on pre-trained general knowledge.

2.  **Prompt Engineering with Strong Contrastive MCQ + Anti-hallucination Constraints**:
    *   **Function**: Ensure difficulty depends on "accurately hearing audio details" rather than "easily distinguishing options."
    *   **Mechanism**: (i) 10 options per question, where distractors are lexically similar to the correct answer yet semantically distinct; (ii) Distractor patterns include temporal misalignment (correct event, wrong phase), similar acoustic features with different clinical interpretations, and over-reliance on training data priors; (iii) Anti-hallucination constraints—all answers must be derivable from the audio itself, prohibiting reliance on external medical knowledge; (iv) Three difficulty levels systematically increase demands on perceptual precision.
    *   **Design Motivation**: Existing medical QA is often solved by LLM memorization of medical facts; strong constraints make the task impossible without the audio, truly testing audio reasoning.

3.  **Embedded Voice QA (Voice_QA) + Multi-turn + Open-ended Question Types**:
    *   **Function**: Incorporate three scenarios from real clinical interactions: "questions interleaved with dialogue," "multi-step follow-ups," and "generation over selection."
    *   **Mechanism**: Voice_QA embeds questions and answers directly into the audio waveform—models must switch context to answer an embedded voice question after hearing clinical dialogue, testing context switching and attention drift. Multi_Turn involves follow-up questions on long dialogues, requiring state maintenance across turns. Open-Ended (OE_Speech / OE_Speech_Sound) requires unconstrained generation on long audio; answers must be concise but accurate, representing a rigorous test of generative reasoning.
    *   **Design Motivation**: MCQs test "differentiation," but real clinical scenarios mostly involve "doctor speaking after listening" generation. Embedded QA further simulates real-world clinical interaction where devices or colleagues insert questions during patient dialogues.

### Loss & Training
This is not a training-focused paper; no loss function is used. All QA pairs were generated by Gemini-3-flash, and 13 candidate models (including Audio Flamingo 3, Audio Reasoner, Baichuan-Omni, Desta25-Audio, Gama, Gemini-2.5-flash/pro, and Qwen-2.5-Omni) were evaluated for inference.

## Key Experimental Results

### Main Results (Table 1 Excerpt, Accuracy %)

| Model | Weighted Avg | MCQ_Speech | MCQ_Sound_Heart | OE_Speech | Voice_QA |
|---|---|---|---|---|---|
| Audio-flamingo-3 | 24.1 | 10.7 | 37.8 | 55.2 | 0.1 |
| Audio-reasoner | 32.8 | 23.7 | 35.6 | 51.2 | 9.9 |
| Baichuan-omni | 38.6 | 43.5 | 26.6 | 57.6 | 31.5 |
| Desta25-audio | 41.0 | 49.4 | 37.1 | 56.0 | 9.1 |
| Gama | 23.2 | 12.7 | 36.6 | 38.1 | 8.9 |
| Gemini-2.5-flash | 60.5 | 73.6 | 52.8 | ... | ... |
| **Gemini-2.5-Pro** | **~68.1** | (Best per column) | | | |
| Qwen-2.5-Omni-7B | 42.8 | ... | ... | ... | ... |

The strongest commercial model, Gemini-2.5-Pro, reached only 68.1% weighted average, validating the benchmark's difficulty.

### Ablation Study

| Phenomenon | Description |
|---|---|
| Voice_QA < 32% for most, < 1% for some | Embedded voice QA is the primary weakness—extremely poor context-switching capability. |
| OE_Speech generally > MCQ_Speech | Open-ended scores are higher due to loose evaluation (awarding points if facts are included), not necessarily superior understanding. |
| MCQ_Sound_Heart > Cough / Lung | Heart sound temporal structures ($S_1/S_2$) are relatively regular and easier to recognize than the stochasticity of coughs/lung sounds. |
| MCQ_Long_Form generally low | Long-form dialogue reasoning is a universal weakness, consistent with literature stating LALMs struggle with long contexts. |

### Key Findings
- Even the strongest general-purpose models fall significantly below human clinical levels (>90%), proving medical audio reasoning is not covered by existing LALMs; specialized pre-training data and adaptation are necessary.
- Audio-flamingo-3 scored nearly zero (0.1%) on Voice_QA, indicating a total lack of "context switching" ability—a new evaluation dimension revealed by embedded QA.
- The synthetic QA pipeline finds an effective point between "minimized human supervision" and "remaining a difficult benchmark," validating synthetic data as a scalable evaluation paradigm for privacy-sensitive medical fields.

## Highlights & Insights
- Decomposing medical audio into an orthogonal question matrix (sound-only / speech-only / speech+sound / voice-embedded) allows for precise diagnosis of model weaknesses—a reproducible methodology for clinical scenario evaluation.
- The anti-hallucination constraint "correct answers must be derivable from audio, distractors need independent clinical interpretations" is a rigorous prompt engineering paradigm transferable to other specialized domain QA construction to prevent LLM judges from using general knowledge.
- The "Voice_QA" design (embedding questions in waveforms) is truly innovative—clinicians must answer questions while listening to patients; this "continuous monitoring + interrupt response" capability was completely missing from existing benchmarks.

## Limitations & Future Work
- Data is synthetic rather than real clinical recordings, leaving a domain gap; the authors mitigate this by preserving clinical nuance and embedding physical artifacts, but it is not eliminated.
- Ground truth is generated by Gemini-3-flash, carrying the risk of generator bias; human verification scale was limited.
- Evaluated models are primarily generic LALMs; there is a lack of comparison with models specifically fine-tuned for medical audio (e.g., a future MedAudio-LLM).
- The scoring protocol for open-ended questions is simplified in the paper, with room for improvement in reproducibility.

## Related Work & Insights
- **vs CaReAQA**: CaReAQA targets medical audio but is small and limited to short fragments; MedMosaic scales up by two orders of magnitude and adds long-form/multi-turn/embedded QA.
- **vs MMAU / MMAU-Pro / MMAR**: These cover general audio broadly but lack specialization; MedMosaic provides depth in the medical sub-domain, complementing MMAU-Pro.
- **vs CORAAL-QA**: While CORAAL-QA focuses on long-form multi-turn interactions, MedMosaic introduces domain specificity and physiological sound characteristics.
- **vs MedQA (Text)**: MedQA is purely clinical knowledge text; this work systematically completes the "clinical reasoning evaluation in the audio dimension."

## Rating
- Novelty: ⭐⭐⭐⭐ Embedded Voice_QA, multi-turn dialogue, and physiological temporal reasoning are firsts for medical audio benchmarks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated 13 models across 10 question types; lacks human benchmarks and med-specific fine-tuned model comparisons.
- Writing Quality: ⭐⭐⭐ Clear flowcharts and prompt templates; however, some experimental details (e.g., OE metrics) are brief.
- Value: ⭐⭐⭐⭐ Provides the first large-scale scalable evaluation for medical audio LALMs; the synthetic data paradigm is applicable to other privacy-sensitive domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety](multibreak_a_scalable_and_diverse_multi-turn_jailbreak_benchmark_for_evaluating_.md)
- [\[AAAI 2026\] StyleBreak: Revealing Alignment Vulnerabilities in Large Audio-Language Models via Style-Aware Audio Jailbreak](../../AAAI2026/llm_safety/stylebreak_revealing_alignment_vulnerabilities_in_large_audio-language_models_vi.md)
- [\[ICML 2026\] Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models](less_diverse_less_safe_the_indirect_but_pervasive_risk_of_test-time_scaling_in_l.md)
- [\[ICLR 2026\] AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models](../../ICLR2026/llm_safety/audiotrust_benchmarking_the_multifaceted_trustworthiness_of_audio_large_language.md)
- [\[ICML 2026\] FoeGlass: Simple In-Context Learning Is Enough for Red Teaming Audio Deepfake Detectors](foeglass_simple_in-context_learning_is_enough_for_red_teaming_audio_deepfake_det.md)

</div>

<!-- RELATED:END -->
