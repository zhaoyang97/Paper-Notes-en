---
title: >-
  [Paper Note] SIFT-50M: A Large-Scale Multilingual Dataset for Speech Instruction Fine-Tuning
description: >-
  [ACL 2025][Multilingual & Machine Translation][speech instruction fine-tuning] This work constructs SIFT-50M, a speech instruction fine-tuning dataset containing 50 million samples across 5 languages. It automatically generates diverse speech understanding and controllable speech generation instructions from public speech corpora using LLMs and expert models. Training SIFT-LLM on this dataset yields performance that significantly outperforms existing speech-text LLMs on instr…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "speech instruction fine-tuning"
  - "multilingual dataset"
  - "speech-text LLM"
  - "instruction-following"
  - "data construction"
date: 2026-05-08
content_hash: b2b7d1fae1532836
---

# SIFT-50M: A Large-Scale Multilingual Dataset for Speech Instruction Fine-Tuning

**Conference**: ACL 2025  
**arXiv**: [2504.09081](https://arxiv.org/abs/2504.09081)  
**Code**: None  
**Area**: Multilingual Translation  
**Keywords**: speech instruction fine-tuning, multilingual dataset, speech-text LLM, instruction-following, data construction

## TL;DR

This work constructs SIFT-50M, a speech instruction fine-tuning dataset containing 50 million samples across 5 languages. It automatically generates diverse speech understanding and controllable speech generation instructions from public speech corpora using LLMs and expert models. Training SIFT-LLM on this dataset yields performance that significantly outperforms existing speech-text LLMs on instruction-following benchmarks.

## Background & Motivation

**Background**: Speech-text large language models (speech-text LLMs) represent an important current direction in multimodal AI. These models need to process speech inputs and text outputs (or vice versa) to achieve tasks such as speech understanding, speech translation, and speech generation. While models like GPT-4o and Gemini have demonstrated powerful speech interaction capabilities recently, the open-source community still lacks large-scale, high-quality training data in this area.

**Limitations of Prior Work**: Existing speech instruction fine-tuning datasets are relatively small in scale, limited in language coverage, and restricted in task types. Most datasets focus purely on fundamental tasks such as automatic speech recognition (ASR) or text-to-speech (TTS), lacking diversity at the instruction-following level. Furthermore, constructing high-quality speech-instruction pairs requires extensive human annotation, which is highly expensive.

**Key Challenge**: Training speech-text LLMs demands a massive and diverse set of instruction pairs, yet there is a huge gulf between the cost of annotating high-quality speech data and the scale required. Automatically constructing large-scale, multilingual, multi-task speech instruction datasets at a low cost remains a critical challenge to be resolved.

**Goal**: (1) Construct SIFT-50M, a large-scale speech instruction dataset covering 5 languages with 50 million samples; (2) Verify the effectiveness of this dataset in training speech-text LLMs; (3) Propose the EvalSIFT benchmark specifically designed to evaluate instruction-following capabilities.

**Key Insight**: The authors observe that public speech corpora (such as CommonVoice, LibriSpeech, etc.) contain approximately 14,000 hours of speech data. Although these data lack instruction formatting, they can be automatically extended and reorganized using LLMs and existing expert models (ASR, TTS, translation models, etc.) to generate diverse instruction-response pairs.

**Core Idea**: Utilize an LLM as an "instruction generator" combined with existing speech expert models as "annotators" to automatically synthesize large-scale, multi-task instruction data from public speech corpora, covering both speech understanding and controllable speech generation tasks.

## Method

### Overall Architecture

The construction pipeline of SIFT-50M consists of three phases: (1) collecting and compiling public speech corpora, totaling approximately 14,000 hours and covering five languages (English, Spanish, French, German, and Italian); (2) generating instruction templates via LLMs and leveraging expert models (ASR models, speech emotion recognition models, speech property extractors, etc.) to automatically generate rich instruction-response pairs for each speech segment; (3) categorizing and organizing the generated data by task to form a unified-format instruction fine-tuning dataset.

### Key Designs

1. **Multi-Source Speech Corpora Integration**:

    - **Function**: Provides large-scale, multilingual underlying speech data
    - **Mechanism**: Collects data from multiple public speech datasets including CommonVoice, LibriSpeech, VoxPopuli, and FLEURS, totaling approximately 14,000 hours. Each dataset contributes speech from different languages and domains to ensure diversity. Different audio formats are aligned to standard sample rates and encoding formats through uniform preprocessing.
    - **Design Motivation**: A single speech dataset typically only covers a few languages and limited scenarios. Multi-source integration maximizes data diversity and coverage, providing rich source material for downstream instruction generation.

2. **LLM-Driven Instruction Generation Pipeline**:

    - **Function**: Automatically generates diverse instruction-response pairs for each speech segment.
    - **Mechanism**: The authors design two main categories of task templates: speech understanding (e.g., ASR, speech translation, speech emotion recognition, speaker property identification) and controllable speech generation (e.g., TTS instructions specifying emotion, speech rate, or pitch). LLMs are employed to generate natural language instruction variants, while corresponding expert models produce standard ground truths. For instance, for a given speech segment, an ASR expert model provides the transcript, while an LLM generates varying formulations of ASR instructions (e.g., "Please transcribe this," "Write down this speech," etc.).
    - **Design Motivation**: Directly using fixed instruction templates can cause models to overfit to specific instruction formats. Instruction variants generated by LLMs are closer to real-user diverse formulations, thereby enhancing the model's instruction generalization capability.

3. **EvalSIFT Evaluation Benchmark**:

    - **Function**: Specifically evaluates the instruction-following capability of speech-text LLMs.
    - **Mechanism**: Representative samples are carefully selected from the test set of SIFT-50M, spanning all task categories and languages. Evaluation metrics include task completion accuracy and instruction compliance — checking not only whether the model can complete the task, but also whether its output conforms to the format and style requested by the instruction.
    - **Design Motivation**: Most existing speech evaluation benchmarks focus on single tasks (such as using WER to evaluate ASR), lacking a systematic evaluation of the comprehensive capability of instruction following.

### Loss & Training

Based on an open-source speech-text LLM architecture, SIFT-LLM employs the standard next-token prediction loss for instruction fine-tuning. The training consists of two stages: first, pre-training on SIFT-50M to learn speech-text alignment, followed by instruction fine-tuning to improve instruction-following capabilities.

## Key Experimental Results

### Main Results

| Benchmark | Metric | SIFT-LLM | Qwen-Audio | SALMONN | Gain |
|----------|------|----------|------------|---------|------|
| EvalSIFT (Instruction-following) | Acc | **Best** | Second Best | Poor | Significantly Outperforms |
| LibriSpeech (ASR) | WER↓ | Competitive | Competitive | Competitive | Comparable |
| CoVoST2 (Translation) | BLEU | Competitive | Competitive | — | Comparable |
| Emotion Recognition | Acc | **Best** | Second Best | Poor | Significant Gain |

### Ablation Study

| Configuration | EvalSIFT Score | Description |
|------|-------------|------|
| Full SIFT-50M | Best | Trained on full data |
| Speech understanding tasks only | Decreased | Generative tasks removed, decreasing instruction diversity |
| English data only | Significantly decreased | Multilingual coverage is critical for cross-lingual capability |
| 10M subset | Decreased | Data scale has a positive impact on performance |
| Fixed-template instructions | Decreased | Diverse instructions generated by LLMs outperform fixed templates |

### Key Findings

- SIFT-LLM significantly outperforms existing models on the EvalSIFT instruction-following benchmark, indicating that large-scale, diverse instruction data is key to enhancing the instruction-following capabilities of speech LLMs.
- SIFT-LLM remains competitive on traditional core speech tasks (ASR, translation, etc.), demonstrating that base performance is not sacrificed for instruction diversity.
- Data scale and task diversity are the two most critical factors, both being indispensable.
- The inclusion of multilingual data not only enhances performance in multilingual scenarios but also exerts a positive spillover effect on English-language performance.

## Highlights & Insights

- **Low-Cost Large-Scale Data Construction Paradigm**: Mining instruction pairs from existing corpora using a combination of LLMs and expert models. This paradigm can be transferred to other modalities like computer vision and robotics to address instruction data scarcity. The key insight is that high-quality data does not need to be annotated from scratch; it can be automatically derived from existing data.
- **Controllable Speech Generation Instructions**: Incorporating both speech understanding and speech generation enables the trained model to handle *bidirectional* tasks. This co-training strategy of "understanding + generation" is highly worth replicating in other multimodal scenarios.
- **EvalSIFT Evaluation Framework**: Fills the gap in testing the instruction-following performance of speech LLMs, providing a standardized platform for comparison in subsequent research.

## Limitations & Future Work

- The public speech corpora on which the dataset relies consist primarily of read speech, lacking complex scenarios such as spontaneous speech or noisy environments.
- While covering 5 languages is an improvement over prior work, it still falls short of true "multilingualism" (e.g., 100+ languages).
- The annotation quality of the expert models defines the upper bound of the data quality; errors in the ASR models themselves will introduce noise into the generated instruction pairs.
- Evaluating the quality of controllable speech generation instructions remains difficult. Current evaluations rely heavily on automated metrics and lack human evaluation.
- Future work can extend this to cover more languages and speech task types (such as spoken dialogue, speech editing, etc.).

## Related Work & Insights

- **vs Qwen-Audio**: Qwen-Audio utilizes multi-task pre-training but lacks large-scale instruction fine-tuning data; Ours achieves a distinct advantage in instruction following via SIFT-50M.
- **vs SALMONN**: SALMONN focuses on general capabilities for audio understanding but is weaker in controllable speech generation; Ours provides more comprehensive coverage through bidirectional tasks.
- **vs CommonVoice Series**: Public speech corpora provide foundational data but lack instruction formatting; the contribution of SIFT lies in this "instructionalization" conversion process.

## Rating

- Novelty: ⭐⭐⭐⭐ The data construction paradigm features prominent highlights, but the core idea (LLM-synthesized data) is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark evaluation with ablations is conducted, but some experimental details are not sufficiently comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, and the description of the dataset is highly detailed.
- Value: ⭐⭐⭐⭐ The large-scale open dataset contributes significantly to the community, filling the data gap in speech instruction fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Global AI Inclusivity: A Large-Scale Multilingual Terminology Dataset (GIST)](towards_global_ai_inclusivity_a_large-scale_multilingual_terminology_dataset_gis.md)
- [\[ACL 2026\] Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models](../../ACL2026/multilingual_mt/exploring_continual_fine-tuning_for_enhancing_language_ability_in_large_language.md)
- [\[ACL 2025\] CC-Tuning: A Cross-Lingual Connection Mechanism for Improving Joint Multilingual Supervised Fine-Tuning](cc-tuning_a_cross-lingual_connection_mechanism_for_improving_joint_multilingual_.md)
- [\[ACL 2025\] LangMark: A Multilingual Dataset for Automatic Post-Editing](langmark_a_multilingual_dataset_for_automatic_post-editing.md)
- [\[ACL 2025\] Marco-Bench-MIF: On Multilingual Instruction-Following Capability of Large Language Models](marco_bench_multilingual_if.md)

</div>

<!-- RELATED:END -->
