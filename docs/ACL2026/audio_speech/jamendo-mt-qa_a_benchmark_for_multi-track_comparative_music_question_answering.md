---
title: >-
  [Paper Note] Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering
description: >-
  [ACL 2026][Audio & Speech][Music QA] Constructs Jamendo-MT-QA, a multi-track comparative music QA benchmark containing 36,519 comparative QA pairs (covering 12…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Music QA"
  - "multi-track comparative reasoning"
  - "audio-language models"
  - "benchmark dataset"
  - "LLM-as-a-Judge"
date: 2026-05-08
content_hash: 736d3d484ced29fd
---

# Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering

**Conference**: ACL 2026  
**arXiv**: [2604.09721](https://arxiv.org/abs/2604.09721)  
**Code**: None  
**Area**: Audio & Speech / Music Understanding  
**Keywords**: Music QA, multi-track comparative reasoning, audio-language models, benchmark dataset, LLM-as-a-Judge

## TL;DR

Constructs Jamendo-MT-QA, a multi-track comparative music QA benchmark containing 36,519 comparative QA pairs (covering 12,173 track pairs). It provides the first systematic evaluation of audio-language models' cross-track comparative reasoning capabilities, revealing significant deficiencies in existing models regarding sentence-level comparative generation.

## Background & Motivation

**Background**: Music Question Answering (Music-QA) research primarily focuses on single-track understanding, such as tag prediction, captioning, and classification. However, listeners often describe music in a comparative manner (e.g., "this song is darker than the last one"), but existing benchmarks do not systematically evaluate cross-track comparative reasoning.

**Limitations of Prior Work**: (1) Single-track benchmarks may produce high scores driven by text cues rather than true audio perception; (2) Audio-language models (e.g., CLAP, MU-LLaMA) perform strongly on single-track tasks but lack evaluation for multi-track comparative reasoning; (3) There is a lack of specialized datasets for music comparative reasoning.

**Key Challenge**: Existing Music-QA benchmarks cannot distinguish whether a model truly understands audio content or relies on text shortcuts, nor can they evaluate capability for cross-track relational reasoning.

**Goal**: Construct a systematic multi-track comparative QA benchmark to evaluate and expose the shortcomings of existing models.

**Key Insight**: Based on the Jamendo-QA dataset, LLMs are utilized to assist in generating three types of comparative questions (Yes/No, Short-answer, Sentence-level), with quality control performed through human evaluation + LLM-as-a-Judge.

**Core Idea**: Construct a high-quality comparative QA benchmark through an LLM-assisted four-stage pipeline (music captioning $\rightarrow$ single-track QA expansion $\rightarrow$ multi-track comparative QA generation $\rightarrow$ quality filtering).

## Method

### Overall Architecture

Four-stage construction process: Stage 1 uses Music Flamingo to generate high-quality captions for each track; Stage 2 involves GPT-5.1 expansion into single-track QA pairs; Stage 3 uses GPT-5 mini to generate three types of comparative questions (yes/no, short-answer, sentence-level) for each track pair; Stage 4 performs quality filtering via human evaluation and LLM-as-a-Judge.

### Key Designs

1.  **Multi-type Comparative Question Design**: Three types of questions are generated for each track pair: Yes/No questions (e.g., "Is Track A faster than Track B?"), Short-answer questions (choosing the track that matches a description), and Sentence-level questions (requiring a complete comparative analysis). These three types cover a difficulty gradient from simple judgment to complex reasoning, with human evaluation showing that sentence-level questions are significantly more difficult than the first two.
2.  **LLM-as-a-Judge Quality Control**: Human evaluation and GPT-5 mini scores were first aligned on 300 samples. After verifying that LLM evaluation matches human judgment across semantic quality standards (Correctness 4.87 vs. human 4.79, Comparative Validity 4.61 vs. 4.83, Reasoning Quality 4.37 vs. 4.78), it was scaled to the full dataset. Only QA groups scoring 5/5/5 across three semantic criteria were retained.
3.  **Dual-path Baseline Evaluation**: Two types of baselines were designed—multi-audio baselines (e.g., GPT-4o Audio, Qwen3-Omni directly processing dual audio inputs) and caption baselines (e.g., Music Flamingo generating captions followed by LLM comparison)—to separate the contributions of multi-audio perception capabilities from high-level semantic reasoning.

### Loss & Training

Ours is a benchmark construction effort and does not involve model training. Evaluation metrics include: Accuracy for Yes/No and Short-answer; BLEU, ROUGE-1/2/L, BERTScore, and LLM-as-a-Judge 1-5 scores for sentence-level responses.

## Key Experimental Results

### Main Results (Full 12,173 track pairs)

| Model | Type | Yes/No Acc | Short Acc | BLEU | BERT-F1 | GPT Judge | Claude Judge |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Music Flamingo | Cap | 77.4% | 89.7% | 4.00 | 0.879 | 3.24 | 3.87 |
| Qwen2-Audio | Cap | 37.4% | 39.1% | 1.88 | 0.849 | 1.49 | 1.53 |
| MU-LLaMA | Cap | 20.6% | 55.3% | 2.39 | 0.857 | 2.36 | 2.01 |
| Qwen2-Audio | Multi | 50.9% | 80.2% | 2.09 | 0.847 | 1.37 | 1.62 |
| Qwen3-Omni | Multi | 62.9% | 80.3% | 3.58 | 0.863 | 3.11 | 3.48 |

### Ablation Study

*   The caption baseline Music Flamingo reached 77.4% on Yes/No, outperforming most multi-audio baselines, indicating that high-quality captioning + text reasoning is a viable path.
*   Qwen2-Audio's multi-audio mode (50.9%) showed a significant improvement over its caption mode (37.4%) in Yes/No accuracy.
*   All models scored $\le 3.87/5$ on the LLM Judge for sentence-level questions, exposing the massive challenge of comparative reasoning.

### Key Findings

*   The caption baseline Music Flamingo achieved the best overall performance, suggesting current multi-audio models have not yet fully utilized the advantages of audio input.
*   Sentence-level comparative generation is the biggest bottleneck, requiring cross-track multi-attribute integration and coherent natural language expression.
*   92.9% of cross-genre track pairs were retained in the dataset, indicating that the filtering strategy does not compromise diversity.
*   Human difficulty ratings confirm the gradient: sentence-level > short-answer > yes/no.

## Highlights & Insights

*   **Filling the Gap in Comparative Reasoning**: The first music QA benchmark specifically for evaluating cross-track comparative reasoning.
*   **Diagnostic Design**: The dual-path baseline (captioning vs. multi-audio) effectively separates perception capabilities from reasoning capabilities.
*   **Quality Control Innovation**: A large-scale LLM review process, validated by human-LLM alignment, which can be generalized to the construction of other datasets.

## Limitations & Future Work

*   Comparative questions are generated based on captions and metadata rather than directly from audio, which may introduce text bias.
*   Evaluation of sentence-level questions relies on LLM Judge, where reliability still has room for improvement.
*   The comparative understanding capabilities of generative music models were not evaluated.
*   Future work could extend to comparative reasoning across more tracks ($>2$).

## Related Work & Insights

*   Transfer of relational reasoning ideas from multi-hop QA (HotpotQA, DROP) to the audio domain.
*   The LLM-as-a-Judge evaluation paradigm is becoming popular in NLP benchmarks.
*   May inspire the construction of comparative reasoning benchmarks for other modalities (e.g., video comparative QA).

## Rating

*   **Novelty**: ⭐⭐⭐⭐ The problem definition of multi-track comparative reasoning is novel, and the dataset construction methodology is sound.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model baseline evaluations are sufficient, though some models used only subsets due to computational costs.
*   **Writing Quality**: ⭐⭐⭐⭐ Pipeline descriptions are clear, and quality control processes are detailed.
*   **Value**: ⭐⭐⭐⭐ Provides an important comparative reasoning benchmark and diagnostic tool for the music understanding field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Music Audio-Visual Question Answering Requires Specialized Multimodal Designs](music_audio-visual_question_answering_requires_specialized_multimodal_designs.md)
- [\[ACL 2026\] Retrieving to Recover: Towards Incomplete Audio-Visual Question Answering via Semantic-consistent Purification](retrieving_to_recover_towards_incomplete_audio-visual_question_answering_via_sem.md)
- [\[ICLR 2026\] SyncTrack: Rhythmic Stability and Synchronization in Multi-Track Music Generation](../../ICLR2026/audio_speech/synctrack_rhythmic_stability_and_synchronization_in_multi-track_music_generation.md)
- [\[ICLR 2026\] Query-Guided Spatial-Temporal-Frequency Interaction for Music Audio-Visual Question Answering](../../ICLR2026/audio_speech/query-guided_spatial-temporal-frequency_interaction_for_music_audio-visual_quest.md)
- [\[CVPR 2026\] ViDscribe: Multimodal AI for Customizing Audio Description and Question Answering in Online Videos](../../CVPR2026/audio_speech/vidscribe_multimodal_ai_for_customizing_audio_description_and_question_answering.md)

</div>

<!-- RELATED:END -->
