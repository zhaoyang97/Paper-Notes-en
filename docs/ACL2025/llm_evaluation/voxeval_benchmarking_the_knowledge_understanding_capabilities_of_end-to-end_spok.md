---
title: >-
  [Paper Note] VoxEval: Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models
description: >-
  [ACL2025][LLM Evaluation][spoken language model] Proposes VoxEval, the first SpeechQA benchmark supporting end-to-end speech-only input-output evaluation, covering 56 subjects and 26 input audio conditions, systematically revealing the severe deficiencies of current end-to-end spoken language models in knowledge understanding and mathematical reasoning.
tags:
  - "ACL2025"
  - "LLM Evaluation"
  - "spoken language model"
  - "speech QA"
  - "benchmark"
  - "end-to-end evaluation"
  - "robustness"
date: 2026-05-08
content_hash: 077c6f1de45c4e9f
---

# VoxEval: Benchmarking the Knowledge Understanding Capabilities of End-to-End Spoken Language Models

**Conference**: ACL2025  
**arXiv**: [2501.04962](https://arxiv.org/abs/2501.04962)  
**Code**: [dreamtheater123/VoxEval](https://github.com/dreamtheater123/VoxEval)  
**Area**: LLM Evaluation  
**Keywords**: spoken language model, speech QA, benchmark, end-to-end evaluation, robustness

## TL;DR

Proposes VoxEval, the first SpeechQA benchmark supporting end-to-end speech-only input-output evaluation, covering 56 subjects and 26 input audio conditions, systematically revealing the severe deficiencies of current end-to-end spoken language models in knowledge understanding and mathematical reasoning.

## Background & Motivation

1. **Growing demand for speech interaction**: Natural human-computer interaction typically occurs in spoken form. End-to-end spoken language models (SLMs) have become a key direction in this field, but their knowledge understanding capabilities lack systematic evaluation.
2. **Existing benchmarks do not support end-to-end evaluation**: Prior SpeechLLM benchmarks (such as AudioBench and VoiceBench) either pair audio inputs with text questions and answers or only evaluate text outputs, failing to reflect the performance of real speech-to-speech interaction pipelines.
3. **Neglecting the diversity of input audio conditions**: In real-world scenarios, speakers vary in timbre, speech rate, accent, and ambient noise, but existing benchmarks do not systematically evaluate the robustness of SLMs to these changes.
4. **Performance cliff from S2T to S2S**: Multiple studies indicate that the performance of SLMs drops significantly when switching from speech-to-text to speech-to-speech evaluation, highlighting the necessity of end-to-end evaluation.
5. **Mathematical reasoning is difficult to evaluate in spoken format**: While mathematical expressions are concise in written form, evaluating them after transcription into spoken words (e.g., "two thousand three hundred and fifty-one" instead of "2351") is highly challenging, and no relevant benchmarks existed prior to this work.
6. **SLMs urgently need comprehensive evaluation for knowledge understanding**: Existing speech model evaluations mostly focus on language modeling (lexicon/syntax/semantics) or paralinguistic features, lacking a comprehensive benchmark for world knowledge understanding.

## Method

### Overall Architecture: VoxEval Benchmark Construction

- **Function**: Builds a speech-only knowledge-understanding QA benchmark based on the MMLU test set, containing 13,938 unique speech Q&A pairs covering 56 subjects (excluding high-school computer science containing code snippets). Each question is expanded into 153,318 variants across 26 input audio conditions.
- **Design Motivation**: MMLU has a comprehensive subject structure (STEM, Social Sciences, Humanities) and is widely used for evaluating text LLMs. However, prior studies only evaluated SLMs' added text processing capabilities using text formats, which does not reflect speech processing performance; the spoken version of the same data is significantly more challenging than the written version.
- **Mechanism**: Concatenates MMLU multiple-choice questions into natural language sequences ("Question... Please choose from ABCD... Option A... Option B..."), and synthesizes them into speech along with the answers via the OpenAI TTS API. During evaluation, Whisper-large-v3 is used to transcribe the SLM's spoken response into text, followed by string matching to calculate accuracy.

### Key Designs 1: Diverse Input Audio Conditions

- **Function**: Systematically constructs 26 input audio conditions, including 6 speakers, 5 linguistic variants (filler words, mispronunciation, disfluency, self-correction, non-native), 2 paralinguistic variants (pitch shifting, speech rate variation), and various audio quality variants (Gaussian noise, colored noise, background music, reverberation, diverse filters, etc.).
- **Design Motivation**: SLMs should maintain consistent response accuracy when semantic content remains unchanged while audio features vary, which is particularly crucial for factual questions. However, no benchmark has systematically tested this robustness prior to this work.
- **Mechanism**: Speakers use 6 voices from OpenAI TTS (alloy, echo, fable, etc.); linguistic variants are generated by modifying the original question text via GPT-4o before TTS synthesis; paralinguistic variants are realized through audio augmentation ($\pm 5$ semitones pitch shift, $0.5\times$-$2\times$ speech rate variation); audio quality variants are generated using the `audiomentations` library to add noise, reverberation, and filtering. All variants are based on the "alloy" speaker.

### Key Designs 2: Spoken Mathematical Reasoning Evaluation

- **Function**: For the first time, introduces mathematical reasoning into SLM evaluation by converting written mathematical expressions into spoken form and designing two evaluation modes: Direct Answer and Chain-of-Thought (CoT).
- **Design Motivation**: Mathematical reasoning is a core capability for applications like AI tutoring, and humans frequently express mathematical problems in spoken language. However, existing TTS systems cannot correctly pronounce mathematical expressions (arabic numerals, operators, parentheses, etc.).
- **Mechanism**: A two-step method: first, use GPT-4o with few-shot prompting to convert written mathematics to spoken language (e.g., "$4 \div (2+8)$" $\rightarrow$ "four divided by the sum of two and eight"), then synthesize it via TTS. The CoT mode adds the voice prompt "Please explain your reasoning step by step" before the question. The evaluation is categorized into "truncated" (only extracting the final answer segment) and "non-truncated" (including the full reasoning chain) formats.

## Key Experimental Results

### Experiment 1: Overall Knowledge Understanding Performance

| Model | Alloy Accuracy | Best Accuracy | Random Baseline |
|------|-------------|-----------|---------|
| SpeechGPT | 0.01% | 0.02% | 25% |
| TWIST | 4.80% | 5.58% | 25% |
| SPIRIT-LM | 20.84% | 20.96% | 25% |
| Moshi | 12.16% | 12.98% | 25% |
| GLM-4-Voice | **37.63%** | **38.15%** | 25% |
| Whisper+Llama-3.1-8B | 55.25% | 55.73% | 25% |

**Key Findings**: (1) Most SLMs do not exceed the 25% random guessing baseline; only GLM-4-Voice surpasses it, indicating that current SLMs are severely deficient in instruction following and knowledge understanding; (2) The cascaded system (Whisper+Llama) significantly outperforms all end-to-end SLMs, with a gap of over 17 percentage points.

### Experiment 2: Audio Condition Robustness

| Condition | GLM-4-Voice | Moshi | SPIRIT-LM |
|------|------------|-------|-----------|
| Standard (Alloy) | 37.63% | 12.16% | 20.84% |
| Pitch Shift (Pitch) | 33.45%（↓4.2） | 6.09%（↓6.1） | 17.88%（↓3.0） |
| Speed Variation (Speed) | 34.69%（↓2.9） | 10.13%（↓2.0） | 19.11%（↓1.7） |
| Background Noise (Noise) | 36.95%（↓0.7） | 10.18%（↓2.0） | 19.50%（↓1.3） |

**Key Findings**: (1) Pitch shift is the condition that affects SLMs the most, with Moshi's accuracy being almost halved; (2) Performance differences between different speakers can reach 1–4 percentage points, with the "fable" voice typically being the most challenging; (3) In terms of mathematical reasoning, all models exhibit extremely low accuracy (the highest being around 27%). CoT prompting failed to improve and even degraded the performance of some models, indicating that SLMs lack step-by-step reasoning capabilities in spoken formats.

## Highlights & Insights

- **First end-to-end speech QA benchmark**: Both input and output are speech, filling the gap in evaluating SLMs in end-to-end settings.
- **Systematic robustness evaluation**: The design of 26 audio conditions covers a wide spectrum, directly revealing the vulnerability of SLMs to pitch, speed, and noise.
- **First spoken mathematical reasoning evaluation**: Innovatively converts written math to spoken format and introduces CoT evaluation, paving a new evaluation dimension.
- **Sound benchmark design**: Built upon MMLU, ensuring comprehensive subject coverage and comparability with text baselines.
- **Large scale data**: 13,938 basic QA pairs $\times$ 26 conditions = 153,318 evaluation samples, offering statistically sufficient assessment.

## Limitations & Future Work

- **Evaluation metrics rely on ASR**: The final answer requires Whisper transcription followed by string matching. ASR errors can introduce noise, meaning this is not a "true" end-to-end evaluation.
- **Limited to MCQ format**: Real-world voice interaction is far more complex than multiple-choice questions, and the benchmark lacks evaluation for open-ended QA.
- **Single data source**: Entirely based on MMLU, inheriting MMLU's inherent biases and limitations (such as uneven subject coverage and controversial questions).
- **Gap between TTS-synthesized speech and natural human speech**: All audio is generated by TTS, which may not fully represent the natural variations of real human speech.
- **No evaluation of commercial closed-source SLMs**: Such as GPT-4o's voice mode.

## Related Work & Insights

### vs VoiceBench (Chen et al., 2024)
VoiceBench evaluates SpeechLLMs under various audio conditions but only bases its evaluation on text outputs (non-end-to-end) and lacks mathematical reasoning. VoxEval achieves end-to-end evaluation where both input and output are speech for the first time, featuring richer audio conditions (26 vs. VoiceBench's limited variants) and pioneering the incorporation of mathematical reasoning.

### vs AudioBench (Wang et al., 2024)
AudioBench contains 5,196 knowledge understanding questions, but the QA pairs are in text format, making it suitable only for evaluating S2T-LLMs. VoxEval's 13,938 speech-only QA pairs are specifically designed for end-to-end SLMs and systematically evaluate robustness through 153,318 variants.

### vs MMLU (Hendrycks et al., 2021)
VoxEval is built on top of MMLU, extending it from text to the speech modality. Experiments show that the speech version poses a significantly greater challenge to models than the text version (GLM-4-Voice achieves ~37% on VoxEval vs. its corresponding text LLM's 60%+ on MMLU), verifying the necessity of cross-modal evaluation. Additionally, VoxEval excludes subjects containing code snippets, reflecting careful adaptation to the spoken format.

## Rating

- Novelty: ⭐⭐⭐⭐ — First end-to-end speech QA benchmark + spoken mathematical reasoning evaluation
- Experimental Thoroughness: ⭐⭐⭐⭐ — 5 SLMs + 1 cascaded baseline, 26 audio conditions, detailed analysis
- Writing Quality: ⭐⭐⭐⭐ — Clear problem definition, rich figures/tables, sound structure
- Value: ⭐⭐⭐⭐ — Provides much-needed evaluation tools for the SLM community, effectively exposing key shortcomings

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](../../ACL2026/llm_evaluation/e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)
- [\[ACL 2025\] SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)
- [\[ACL 2025\] CodeMEnv: Benchmarking Large Language Models on Code Migration](codemenv_benchmarking_large_language_models_on_code_migration.md)
- [\[ACL 2025\] Exposing Numeracy Gaps: A Benchmark to Evaluate Fundamental Numerical Abilities in Large Language Models](exposing_numeracy_gaps_a_benchmark_to_evaluate_fundamental_numerical_abilities_i.md)

</div>

<!-- RELATED:END -->
