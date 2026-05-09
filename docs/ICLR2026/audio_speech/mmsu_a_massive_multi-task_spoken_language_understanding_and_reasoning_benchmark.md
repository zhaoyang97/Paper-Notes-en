---
title: >-
  [Paper Note] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark
description: >-
  [ICLR 2026][Audio & Speech][Speech Understanding] This paper introduces MMSU (5,000 audio QA items across 47 tasks), the first benchmark to systematically incorporate linguistic theory into spoken language understanding and reasoning evaluation. Evaluating 22 SpeechLLMs, it reveals significant gaps in phonological perception and complex reasoning among existing models.
tags:
  - ICLR 2026
  - "Audio & Speech"
  - Speech Understanding
  - SpeechLLM
  - Linguistics Benchmark
  - Multi-task Evaluation
  - Perception and Reasoning
date: 2026-05-08
content_hash: 38f111535f996e3d
---

# MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark

**Conference**: ICLR 2026
**arXiv**: [2506.04779](https://arxiv.org/abs/2506.04779)
**Code**: [https://huggingface.co/datasets/ddwang2000/MMSU](https://huggingface.co/datasets/ddwang2000/MMSU)
**Area**: Audio & Speech
**Keywords**: Speech Understanding, SpeechLLM, Linguistics Benchmark, Multi-task Evaluation, Perception and Reasoning

## TL;DR
This paper introduces MMSU (5,000 audio QA items across 47 tasks), the first benchmark to systematically incorporate linguistic theory into spoken language understanding and reasoning evaluation. Evaluating 22 SpeechLLMs, it reveals significant gaps in phonological perception and complex reasoning among existing models.

## Background & Motivation

**Background**: SpeechLLMs (e.g., Qwen-Audio, Kimi-Audio, Gemini) have demonstrated strong capabilities in processing audio inputs, achieving impressive performance on ASR and audio understanding tasks. However, their abilities in fine-grained speech perception and complex reasoning remain systematically unevaluated.

**Limitations of Prior Work**: Existing speech benchmarks suffer from three major shortcomings:
   - **Narrow coverage**: Primarily focused on semantic-level tasks, neglecting non-linguistic phenomena common in everyday speech (hesitations, sarcasm, self-corrections, prosodic variations, etc.)
   - **Insufficient data authenticity**: Heavy reliance on TTS-synthesized speech, lacking the acoustic diversity of real human speech
   - **Absence of linguistic theory**: Evaluation designs do not consider foundational principles from phonetics, prosody, rhetoric, and related fields, resulting in systematic blind spots

**Key Challenge**: Genuine spoken language understanding requires not only comprehending *what is said* (semantics), but also *how it is said* (prosody, emotion) and *what is truly meant* (pragmatics)—dimensions that existing benchmarks fail to assess.

**Goal**: To construct a comprehensive, linguistically grounded evaluation framework that systematically assesses SpeechLLM capabilities along both perception and reasoning dimensions.

**Key Insight**: A top-down task taxonomy is designed based on a structured linguistic theoretical framework spanning phonetics, prosody, rhetoric, syntax, semantics, and paralinguistics.

**Core Idea**: Systematically integrate linguistic theory into speech benchmark design, creating a comprehensive evaluation framework across 47 tasks that exposes critical weaknesses of SpeechLLMs in phonological perception and reasoning.

## Method

### Overall Architecture
MMSU comprises 5,000 expert-annotated multiple-choice questions (MCQs) covering 47 tasks, organized in a three-level hierarchy:
- **Level 1**: Perception (24 tasks) vs. Reasoning (23 tasks)
- **Level 2**: Linguistics vs. Paralinguistics
- **Level 3**: Semantics / Phonology / Speaker Traits / Speaking Style

### Key Designs

1. **Fine-grained Acoustic Feature Coverage**:

    - **Function**: Covers non-linguistic sounds (crying, coughing), accents (Indian, British), emotional states, prosodic features (stress, lengthening, pauses), and intonation variation
    - **Mechanism**: Dedicated tasks are designed for each dimension based on sub-field theories within phonetics
    - **Design Motivation**: To fill the gap left by existing benchmarks in acoustic feature coverage

2. **High-quality Data Assurance**:

    - **Function**: Prioritizes authentic speech data, supplemented by professional voice actor recordings and a small number of multi-speaker additions
    - **Mechanism**: A four-stage pipeline—linguistic framework design → question collection and option augmentation → audio acquisition → human review (10 annotators, multiple review rounds)
    - **Design Motivation**: TTS-synthesized speech cannot capture the subtle acoustic characteristics of human speech

3. **Systematic Integration of Linguistic Theory**:

    - **Function**: First benchmark to include tasks such as tongue-twister comprehension, sarcasm detection, homophone reasoning, intonation inference, and couplet matching
    - **Mechanism**: Tasks are derived from six sub-disciplines: phonetics, prosody, rhetoric, syntax, semantics, and paralinguistics
    - **Design Motivation**: To move evaluation beyond surface-level semantics toward a deeper, multi-layered linguistic understanding

### Loss & Training
Not applicable (this is a benchmark paper). Evaluation uses unified instruction prompts with randomized option ordering to mitigate position bias.

## Key Experimental Results

### Main Results

| Model | Size | Perception Avg | Reasoning Avg | Overall Avg |
|-------|------|----------------|---------------|-------------|
| Human | - | 91.24 | 86.77 | 89.72 |
| Gemini-2.0-Flash | - | 57.51 | 68.15 | 62.63 |
| GPT-4o-Audio | - | 57.30 | 66.62 | 61.67 |
| Qwen2.5-Omni-7B | 7B | 53.26 | 69.99 | 61.25 |
| Kimi-Audio | 7B | 43.52 | 76.03 | 59.28 |
| Qwen2.5-Omni-3B | 3B | 42.37 | 72.76 | 56.83 |
| MiniCPM-O | 8.6B | 40.54 | 73.57 | 56.53 |
| MERaLiON | 10B | 35.74 | 73.68 | 54.10 |
| SALMONN | 7B | 29.83 | 30.04 | 30.01 |
| Random Guess | - | 25.02 | 25.37 | 25.37 |

### Ablation Study

| Dimension | Best Model | Accuracy | Human Performance | Gap |
|-----------|------------|----------|-------------------|-----|
| Perception–Semantics | Kimi-Audio | 57.64% | 87.10% | −29.5 |
| Perception–Phonology | Qwen2-Audio | 44.93% | 94.32% | −49.4 |
| Perception–Paralinguistics | Qwen2.5-Omni-3B | 39.19% | 92.88% | −53.7 |
| Reasoning–Semantics | Qwen2.5-Omni-7B | 81.52% | 82.16% | −0.6 |
| Reasoning–Phonology | Qwen2.5-Omni-7B | 82.39% | 87.60% | −5.2 |

### Key Findings
- **Large human–machine gap**: The best model achieves an overall accuracy of 62.63%, compared to 89.72% for humans—a gap of 27 percentage points
- **Phonological perception is the largest bottleneck**: The best model reaches only 44.93% on the Perception–Phonology dimension, nearly 50 points below human performance
- **Reasoning outperforms perception**: Models approach human-level performance on semantic reasoning but fall significantly short on perception tasks requiring integration of acoustic cues
- **Closed-source models show no clear advantage**: Gemini/GPT-4o only marginally outperform Qwen2.5-Omni-7B, suggesting that perception capabilities do not scale proportionally with model size
- **End-to-end models outperform cascade models**: Models that directly process audio outperform those relying on ASR transcription followed by text-based understanding

## Highlights & Insights
- The first benchmark to systematically incorporate linguistic theory into spoken language understanding evaluation, yielding task designs with genuine disciplinary depth
- The 47-task coverage substantially exceeds prior benchmarks, most notably MMAU (27 tasks)
- A key insight is revealed: SpeechLLMs' reasoning capabilities already approach human-level performance, whereas their perceptual capabilities—particularly phonological perception—lag far behind
- High data quality is ensured through prioritization of authentic speech, expert review, and multi-round annotation

## Limitations & Future Work
- Currently limited to English; multilingual coverage remains to be extended
- The four-option MCQ format may not fully reflect open-ended spoken language understanding ability
- Some tasks have limited sample sizes (~100 items per task), warranting attention to statistical significance
- Multi-turn conversational speech understanding scenarios are not included
- Further analysis of error types and patterns could guide targeted model improvement

## Related Work & Insights
- MMSU is complementary to benchmarks such as VoiceBench, MMAU, and AIR-Bench, being the first to cover prosody, intonation, and rhetorical dimensions
- The finding that "perception ≠ reasoning" provides an important direction for SpeechLLM training strategies: acoustic perception capabilities should be a primary focus of improvement
- MMSU offers a new paradigm for multimodal evaluation: using disciplinary theory to guide benchmark design, avoiding the passive approach of "evaluating only what is readily available"

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First systematic application of linguistic theory to speech benchmark design
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 22 models, 47 tasks, with human baselines; evaluation is exceptionally comprehensive
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with a coherent task taxonomy
- **Value**: ⭐⭐⭐⭐⭐ — Reveals critical bottlenecks in SpeechLLMs and provides an important evaluation infrastructure for the community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Stitch: Simultaneous Thinking and Talking with Chunked Reasoning for Spoken Language Models](stitch_simultaneous_thinking_and_talking_with_chunked_reasoning_for_spoken_langu.md)
- [\[AAAI 2026\] HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding](../../AAAI2026/audio_speech/hpsu_a_benchmark_for_human-level_perception_in_real-world_spoken_speech_understa.md)
- [\[NeurIPS 2025\] A Multi-Task Benchmark for Abusive Language Detection in Low-Resource Settings](../../NeurIPS2025/audio_speech/a_multitask_benchmark_for_abusive_language_detection_in_lowr.md)
- [\[ICLR 2026\] EchoMind: An Interrelated Multi-level Benchmark for Evaluating Empathetic Speech Language Models](echomind_an_interrelated_multi-level_benchmark_for_evaluating_empathetic_speech_.md)
- [\[ICLR 2026\] SPARTA: Scalable and Principled Benchmark of Tree-Structured Multi-hop QA over Text and Tables](sparta_scalable_and_principled_benchmark_of_tree-structured_multi-hop_qa_over_te.md)

</div>

<!-- RELATED:END -->
