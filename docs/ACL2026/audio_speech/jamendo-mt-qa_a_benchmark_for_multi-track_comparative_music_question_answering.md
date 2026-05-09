---
title: >-
  [Paper Note] Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering
description: >-
  [ACL 2026][Audio & Speech][Music Question Answering] This paper introduces Jamendo-MT-QA, a multi-track comparative music question answering benchmark comprising 36,519 QA pairs across 12,173 track pairs. It is the first systematic evaluation of audio-language models on cross-track comparative reasoning, revealing significant deficiencies in sentence-level comparative generation among existing models.
tags:
  - ACL 2026
  - "Audio & Speech"
  - Music Question Answering
  - Multi-Track Comparative Reasoning
  - Audio-Language Models
  - Benchmark Dataset
  - LLM-as-a-Judge
date: 2026-05-08
content_hash: 50807961d989a088
---

# Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering

**Conference**: ACL 2026  
**arXiv**: [2604.09721](https://arxiv.org/abs/2604.09721)  
**Code**: None  
**Area**: Audio & Speech / Music Understanding  
**Keywords**: Music Question Answering, Multi-Track Comparative Reasoning, Audio-Language Models, Benchmark Dataset, LLM-as-a-Judge

## TL;DR

This paper introduces Jamendo-MT-QA, a multi-track comparative music question answering benchmark comprising 36,519 QA pairs across 12,173 track pairs. It is the first systematic evaluation of audio-language models on cross-track comparative reasoning, revealing significant deficiencies in sentence-level comparative generation among existing models.

## Background & Motivation

**Background**: Music question answering (Music-QA) research has primarily focused on single-track understanding tasks such as tag prediction, caption generation, and classification. However, listeners frequently describe music in comparative terms (e.g., "this track is darker than the previous one"), and no existing benchmark systematically evaluates cross-track comparative reasoning.

**Limitations of Prior Work**: (1) Single-track benchmarks may be driven by textual cues rather than genuine audio perception; (2) although audio-language models (e.g., CLAP, MU-LLaMA) perform well on single-track tasks, there is no evaluation of their multi-track comparative reasoning capabilities; (3) no dataset specifically targeting music comparative reasoning exists.

**Key Challenge**: Existing Music-QA benchmarks cannot distinguish whether a model truly understands audio content or relies on textual shortcuts, nor can they assess cross-track relational reasoning ability.

**Goal**: To construct a systematic multi-track comparative QA benchmark that evaluates and exposes the shortcomings of existing models.

**Key Insight**: Building upon the Jamendo-QA dataset, LLM-assisted generation is used to produce three types of comparative questions (yes/no, short-answer, sentence-level), with quality control performed via human evaluation and LLM-as-a-Judge.

**Core Idea**: A high-quality comparative QA benchmark is constructed through a four-stage LLM-assisted pipeline: music captioning → single-track QA expansion → multi-track comparative QA generation → quality filtering.

## Method

### Overall Architecture

The construction follows a four-stage pipeline: Stage 1 uses Music Flamingo to generate high-quality captions for each track; Stage 2 uses GPT-4.1 to expand captions into single-track QA pairs; Stage 3 uses GPT-4o mini to generate three types of comparative questions (yes/no, short-answer, sentence-level) for each track pair; Stage 4 applies human evaluation and LLM-as-a-Judge for quality filtering.

### Key Designs

1. **Multi-Type Comparative Question Design**: Three question types are generated per track pair — yes/no questions (e.g., "Is Track A rhythmically faster than Track B?"), short-answer questions (selecting the track matching a given description), and sentence-level questions (requiring a full comparative analysis). The three types span a difficulty gradient from simple judgment to complex reasoning; human evaluation confirms that sentence-level questions are substantially more difficult than the other two types.

2. **LLM-as-a-Judge Quality Control**: Human evaluations and GPT-4o mini scores are first aligned on 300 samples, verifying that LLM judgments are consistent with human trends across semantic quality criteria (Correctness: 4.87 vs. human 4.79; Comparative Validity: 4.61 vs. 4.83; Reasoning Quality: 4.37 vs. 4.78). The LLM reviewer is then scaled to the full dataset, retaining only QA groups receiving 5/5/5 on all three semantic criteria.

3. **Dual-Path Baseline Evaluation**: Two baseline categories are designed — multi-audio baselines (e.g., GPT-4o Audio, Qwen3-Omni processing dual audio inputs directly) and caption baselines (e.g., Music Flamingo generating captions followed by LLM-based comparison) — to disentangle the contributions of multi-audio perceptual capability and high-level semantic reasoning.

### Loss & Training

This paper presents a benchmark construction effort and involves no model training. Evaluation metrics include accuracy for yes/no and short-answer questions, and BLEU, ROUGE-1/2/L, BERTScore, and LLM-as-a-Judge scores (1–5) for sentence-level questions.

## Key Experimental Results

### Main Results (Full 12,173 Track Pairs)

| Model | Type | Yes/No Acc | Short Acc | BLEU | BERT-F1 | GPT Judge | Claude Judge |
|---|---|---|---|---|---|---|---|
| Music Flamingo | Cap | 77.4% | 89.7% | 4.00 | 0.879 | 3.24 | 3.87 |
| Qwen2-Audio | Cap | 37.4% | 39.1% | 1.88 | 0.849 | 1.49 | 1.53 |
| MU-LLaMA | Cap | 20.6% | 55.3% | 2.39 | 0.857 | 2.36 | 2.01 |
| Qwen2-Audio | Multi | 50.9% | 80.2% | 2.09 | 0.847 | 1.37 | 1.62 |
| Qwen3-Omni | Multi | 62.9% | 80.3% | 3.58 | 0.863 | 3.11 | 3.48 |

### Ablation Study

- Music Flamingo (caption baseline) achieves 77.4% on yes/no questions, outperforming most multi-audio baselines, indicating that high-quality captioning combined with text-based reasoning is a viable approach.
- Qwen2-Audio in multi-audio mode (50.9%) shows a notable improvement over its caption mode (37.4%) on yes/no accuracy.
- All models achieve LLM Judge scores ≤ 3.87/5 on sentence-level questions, exposing the significant challenge of comparative reasoning.

### Key Findings

- The caption baseline Music Flamingo achieves the best overall performance, suggesting that current multi-audio models do not yet fully exploit the advantages of direct audio input.
- Sentence-level comparative generation is the primary bottleneck, requiring multi-attribute integration across tracks and coherent natural language expression.
- 92.9% of cross-genre track pairs are retained after filtering, demonstrating that the filtering strategy does not compromise dataset diversity.
- Human difficulty ratings confirm the difficulty gradient: sentence-level > short-answer > yes/no.

## Highlights & Insights

- **Filling the Comparative Reasoning Gap**: The first Music-QA benchmark specifically designed to evaluate cross-track comparative reasoning.
- **Diagnostic Design**: The dual-path baseline (caption vs. multi-audio) effectively decouples perceptual capability from reasoning capability.
- **Quality Control Innovation**: A large-scale LLM review pipeline validated through human-LLM alignment, generalizable to other benchmark construction efforts.

## Limitations & Future Work

- Comparative questions are generated from captions and metadata rather than directly from audio, potentially introducing textual bias.
- Sentence-level question evaluation relies on LLM-as-a-Judge, whose reliability has room for further improvement.
- The comparative understanding capabilities of generative music models are not evaluated.
- Future work may extend to comparative reasoning across more than two tracks.

## Related Work & Insights

- Relational reasoning concepts from multi-hop QA (HotpotQA, DROP) transferred to the audio domain.
- The LLM-as-a-Judge evaluation paradigm is becoming increasingly prevalent in NLP benchmarks.
- This work may inspire the construction of comparative reasoning benchmarks in other modalities (e.g., video comparative QA).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The problem formulation of multi-track comparative reasoning is novel, with a well-developed dataset construction methodology.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model baseline evaluation is comprehensive, though some models are evaluated on subsets due to computational constraints.
- **Writing Quality**: ⭐⭐⭐⭐ The pipeline description is clear and the quality control procedure is presented in thorough detail.
- **Value**: ⭐⭐⭐⭐ Provides an important comparative reasoning benchmark and diagnostic tool for the music understanding community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Music Audio-Visual Question Answering Requires Specialized Multimodal Designs](music_audio-visual_question_answering_requires_specialized_multimodal_designs.md)
- [\[ICLR 2026\] SyncTrack: Rhythmic Stability and Synchronization in Multi-Track Music Generation](../../ICLR2026/audio_speech/synctrack_rhythmic_stability_and_synchronization_in_multi-track_music_generation.md)
- [\[ACL 2026\] Retrieving to Recover: Towards Incomplete Audio-Visual Question Answering via Semantic-consistent Purification](retrieving_to_recover_towards_incomplete_audio-visual_question_answering_via_sem.md)
- [\[ICLR 2026\] SPARTA: Scalable and Principled Benchmark of Tree-Structured Multi-hop QA over Text and Tables](../../ICLR2026/audio_speech/sparta_scalable_and_principled_benchmark_of_tree-structured_multi-hop_qa_over_te.md)
- [\[ICLR 2026\] Query-Guided Spatial-Temporal-Frequency Interaction for Music Audio-Visual Question Answering](../../ICLR2026/audio_speech/query-guided_spatial-temporal-frequency_interaction_for_music_audio-visual_quest.md)

</div>

<!-- RELATED:END -->
