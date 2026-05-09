---
title: >-
  [Paper Note] An Exploration of Mamba for Speech Self-Supervised Models
description: >-
  [ACL 2026][Audio & Speech][Mamba] This work presents the first comprehensive exploration of the Mamba architecture as a backbone for speech self-supervised learning (SSL), demonstrating that Mamba-based HuBERT outperforms Transformers in long-context ASR, streaming ASR, and causal probing tasks while maintaining linear time complexity.
tags:
  - ACL 2026
  - "Audio & Speech"
  - Mamba
  - speech self-supervised learning
  - HuBERT
  - state space model
  - streaming ASR
date: 2026-05-08
content_hash: 6ae30010af3df406
---

# An Exploration of Mamba for Speech Self-Supervised Models

**Conference**: ACL 2026
**arXiv**: [2506.12606](https://arxiv.org/abs/2506.12606)
**Code**: [GitHub](https://github.com/hckuo145/Mamba-based-HuBERT)
**Area**: Speech / Self-Supervised Learning
**Keywords**: Mamba, speech self-supervised learning, HuBERT, state space model, streaming ASR

## TL;DR

This work presents the first comprehensive exploration of the Mamba architecture as a backbone for speech self-supervised learning (SSL), demonstrating that Mamba-based HuBERT outperforms Transformers in long-context ASR, streaming ASR, and causal probing tasks while maintaining linear time complexity.

## Background & Motivation

**Background**: Transformer-based speech SSL models (e.g., HuBERT, wav2vec 2.0) have achieved remarkable success, yet their quadratic complexity incurs high computational costs and memory bottlenecks when processing long sequences.

**Limitations of Prior Work**: (1) Although Mamba has demonstrated superior performance over Transformers in language modeling, its application in speech has been limited to isolated single-task studies. (2) Existing speech Mamba works typically report performance on par with or slightly below Transformers and often require hybrid designs. (3) A unified cross-task evaluation is lacking.

**Key Challenge**: While Mamba's linear time complexity is theoretically well-suited to the long-sequence nature of speech, its overall performance in speech SSL remains unclear.

**Goal**: To systematically train and evaluate Mamba-based HuBERT models, comprehensively exploring their potential as speech foundation models and feature extractors.

**Key Insight**: Mamba blocks replace Transformer blocks in HuBERT while retaining the same training pipeline (two-iteration k-means pseudo-label training), with evaluation conducted across multiple tasks including ASR and SUPERB.

**Core Idea**: Mamba's inherently causal architecture makes it particularly suitable for building causal speech SSL models, yielding distinctive advantages in streaming ASR and long-context scenarios.

## Method

### Overall Architecture

Mamba blocks replace the Transformer blocks in HuBERT, while the CNN feature encoder and positional encoder are retained. The training pipeline follows HuBERT's two-iteration scheme: the first iteration trains for 250k steps with MFCC targets, and the second iteration trains for 400k steps using the sixth-layer outputs of the first iteration as targets. Pre-training is performed on LibriSpeech 960h.

### Key Designs

1. **Systematic Comparison of Multiple Mamba Variants**:

    - Function: Comprehensively evaluate the speech representation capability of different Mamba configurations.
    - Mechanism: Test causal settings (Mamba, Mamba+MLP) and bidirectional settings (ExtBiMamba, InnBiMamba), with fair comparison against corresponding Transformer variants.
    - Design Motivation: Mamba's causal nature may be advantageous in certain tasks (streaming ASR) while disadvantageous in others (tasks requiring global context).

2. **Long-Context and Streaming ASR Evaluation**:

    - Function: Validate the practical value of Mamba's linear complexity in real-world scenarios.
    - Mechanism: Process entire speech utterances without sentence segmentation for long-context ASR; perform streaming ASR under the constraint of using only past information. MACs/second and RTF are quantified as a function of sequence length.
    - Design Motivation: This represents Mamba's greatest theoretical advantage over Transformers — Transformers run out of memory beyond 80 seconds, whereas Mamba can handle sequences exceeding 5 minutes.

3. **Representation Quality Analysis**:

    - Function: Gain deeper understanding of the characteristics of speech representations learned by Mamba.
    - Mechanism: Phone purity is used to assess the phonetic quality of discrete representations; CCA analysis examines how phoneme and speaker features are encoded.
    - Design Motivation: The goal is not only to determine whether representations are good, but also to understand why and in what respects.

### Loss & Training

The standard HuBERT training objective is followed: masked prediction loss. The Adam optimizer is used with linear warm-up over the first 8% of training steps followed by linear decay. Due to computational constraints, training is conducted on a single V100 GPU with a batch size one-quarter of the original.

## Key Experimental Results

### Main Results

| Setting | Model | Parameters | WER | Key Finding |
|---------|-------|------------|-----|-------------|
| Streaming ASR | Mamba HuBERT | 78M | 15.77% | Outperforms causal Transformer (94M, 16.66%) |
| Long-context ASR | ExtBiMamba | — | 11.08% | Transformer fails due to OOM |
| Standard ASR | ExtBiMamba (Small) | — | Comparable to Transformer | Effective at small scale |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|------------|------|
| Causal SUPERB | Mamba > Causal Transformer | Superior on phoneme and speaker tasks |
| Phone Purity | Higher for Mamba | Better phonetic quality of discrete representations |
| CCA Analysis | Speaker features more distinct | Mamba encodes speaker information more clearly |
| ExtBiMamba Base | Below Transformer | Large-scale bidirectional Mamba still needs improvement |

### Key Findings
- Mamba's causal nature is a natural advantage in streaming speech scenarios — 78M parameters outperform a 94M causal Transformer.
- Computational cost remains nearly constant with sequence length, while Transformers run out of memory beyond 80 seconds.
- Mamba produces discrete representations with higher phone purity, benefiting spoken language models that take SSL units as input.
- Large-scale bidirectional Mamba (Base) remains comprehensively below Transformer, suggesting that scalability still requires improvement.

## Highlights & Insights
- This is the first work to systematically evaluate Mamba as a speech foundation model across multiple tasks, rather than testing it in isolation on a single benchmark.
- The finding that "causal nature is an advantage rather than a limitation" reshapes the understanding of Mamba's applicability in speech.
- The findings on discrete representation quality have direct implications for the spoken language model community.

## Limitations & Future Work
- Large-scale training of bidirectional Mamba underperforms; scalability remains a key challenge.
- Pre-training and evaluation are conducted solely on LibriSpeech; multilingual and noisy settings are not tested.
- Training scale is far smaller than the original HuBERT due to the constraint of a single V100 GPU.
- Future work may explore improved architectures such as Mamba2 and larger-scale training.

## Related Work & Insights
- **vs. Hybrid Mamba-Transformer**: This work uses a pure Mamba architecture, more clearly revealing the strengths and weaknesses of Mamba.
- **vs. SSAM**: SSAM focuses on general audio rather than speech; this work is dedicated to speech SSL.
- **vs. Mamba Streaming ASR**: Prior works require additional mechanisms (e.g., lookahead), whereas this work demonstrates that pure Mamba already holds an advantage.

## Rating
- Novelty: ⭐⭐⭐⭐ First comprehensive exploration of Mamba as a speech SSL foundation model.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-dimensional evaluation covering ASR, SUPERB, representation analysis, long-context, and streaming settings.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with careful experimental design.
- Value: ⭐⭐⭐⭐ Provides important empirical evidence for efficient architecture selection in the speech domain.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models](do_we_need_distinct_representations_for_every_speech_token_unveiling_and_exploit.md)
- [\[ICCV 2025\] MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](../../ICCV2025/audio_speech/mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)
- [\[ICLR 2026\] EchoMind: An Interrelated Multi-level Benchmark for Evaluating Empathetic Speech Language Models](../../ICLR2026/audio_speech/echomind_an_interrelated_multi-level_benchmark_for_evaluating_empathetic_speech_.md)
- [\[NeurIPS 2025\] Sensorium Arc: AI Agent System for Oceanic Data Exploration and Interactive Eco-Art](../../NeurIPS2025/audio_speech/sensorium_arc_ai_agent_system_for_oceanic_data_exploration_and_interactive_eco-a.md)
- [\[ACL 2026\] SEPT: Semantically Expanded Prompt Tuning for Audio-Language Models](generalizable_prompt_tuning_for_audio-language_models_via_semantic_expansion.md)

<!-- RELATED:END -->
