---
title: >-
  [Paper Note] An Exploration of Mamba for Speech Self-Supervised Models
description: >-
  [ACL 2026][Audio & Speech][Mamba] This work presents the first comprehensive exploration of Mamba as a foundation model for speech self-supervised learning (SSL). It finds that Mamba-based HuBERT outperforms Transformers in long-context ASR, streaming ASR, and causal probing tasks while maintaining linear time complexity.
tags:
  - ACL 2026
  - Audio & Speech
  - Mamba
  - HuBERT
  - State Space Model
  - Streaming ASR
date: 2026-05-08
content_hash: 4e52554bbd2ee027
---
# An Exploration of Mamba for Speech Self-Supervised Models

**Conference**: ACL 2026  
**arXiv**: [2506.12606](https://arxiv.org/abs/2506.12606)  
**Code**: [GitHub](https://github.com/hckuo145/Mamba-based-HuBERT)  
**Area**: Speech / Self-Supervised Learning  
**Keywords**: Mamba, Speech SSL, HuBERT, State Space Models, Streaming ASR

## TL;DR

This work presents the first comprehensive exploration of Mamba as a foundation model for speech self-supervised learning (SSL). It finds that Mamba-based HuBERT outperforms Transformers in long-context ASR, streaming ASR, and causal probing tasks while maintaining linear time complexity.

## Background & Motivation

**Background**: Transformer-based speech SSL models (e.g., HuBERT, wav2vec 2.0) have achieved significant success. However, their quadratic complexity imposes high computational costs and memory bottlenecks when processing long sequences.

**Limitations of Prior Work**: (1) While Mamba has shown capabilities surpassing Transformers in language modeling, its application in speech has been limited to isolated single-task studies; (2) Existing speech Mamba works often report performance comparable to or slightly worse than Transformers, frequently requiring hybrid designs; (3) There is a lack of unified evaluation across diverse tasks.

**Key Challenge**: The linear time complexity of Mamba is theoretically ideal for the long-sequence nature of speech, but its comprehensive performance within speech SSL remains unclear.

**Goal**: To systematically train and evaluate Mamba-based HuBERT models, fully exploring their potential as speech foundation models and feature extractors.

**Key Insight**: By replacing Transformer blocks in HuBERT with Mamba blocks while holding the training pipeline constant (two-stage iterative k-means pseudo-label training), models are evaluated across multiple tasks including ASR and SUPERB.

**Core Idea**: Mamba's inherent causal architecture makes it particularly suitable for building causal speech SSL models, demonstrating unique advantages in streaming ASR and long-context scenarios.

## Method

### Overall Architecture

This study performs a controlled substitution experiment: Transformer blocks in HuBERT are replaced with Mamba blocks. The CNN feature encoder and positional encoder remain unchanged. The training process follows the HuBERT two-stage iteration (Stage 1: training for 250k steps using MFCC targets; Stage 2: training for 400k steps using pseudo-labels from the 6th layer of the first stage), pre-trained on LibriSpeech 960h. The backbone is the sole variable, ensuring that performance differences between the input speech, Mamba encoding, and SSL representations are attributable to the intrinsic differences between Mamba and Transformer.

### Key Designs

**1. Systematic Comparison of Mamba Variants: Determining if Causality is an Advantage or a Burden**

Mamba is naturally causal, a property that functions differently across tasks. In streaming ASR, causality is an advantage as only past information is available; however, in tasks requiring global context, unidirectionality might be a disadvantage. To define this boundary, the study evaluates both causal setups (Mamba, Mamba+MLP) and bidirectional setups (ExtBiMamba, InnBiMamba), providing fair comparisons against Transformer variants of similar parameter scales. This paired design allows for decoupled analysis of the "causal vs. bidirectional" and "Mamba vs. Transformer" dimensions.

**2. Long-Context and Streaming ASR Evaluation: Translating Linear Complexity into Measurable Scenarios**

The primary advantage of Mamba over Transformer is $O(n)$ versus $O(n^2)$ complexity, which becomes apparent only with long sequences. Two scenarios are designed: long-context ASR directly processing unsegmented speech, and streaming ASR where the model is constrained to decode frame-by-frame using only past information. Computational metrics like MACs/sec and Real-Time Factor (RTF) are quantified against sequence length. Results indicate that while Transformers encounter OOM (Out of Memory) errors beyond 80 seconds, Mamba maintains nearly constant computation and can process audio exceeding 5 minutes.

**3. Representation Quality Analysis: Investigating "Why" Performance Differs**

Because downstream WER alone cannot explain the internal characteristics of Mamba representations, the study analyzes the representation layer: phone purity is used to quantify phonetic quality, and CCA (Canonical Correlation Analysis) characterizes how phoneme and speaker features are encoded. Findings show that Mamba produces higher phone purity in quantized representations and clarifies speaker information encoding, which is valuable for spoken language models that utilize SSL units as input.

### Loss & Training

The model follows the standard HuBERT training objective: masked prediction loss. It utilizes the Adam optimizer with a learning rate that undergoes linear warm-up (first 8%) followed by linear decay. Due to hardware constraints, training was conducted on a single V100 with a batch size set to 1/4 of the original configuration.

## Key Experimental Results

### Main Results

| Setup | Model | Parameters | WER | Key Finding |
|-------|-------|------------|-----|-------------|
| Streaming ASR | Mamba HuBERT | 78M | 15.77% | Outperforms 94M Causal Transformer (16.66%) |
| Long-Context ASR | ExtBiMamba | - | 11.08% | Transformer failed due to OOM |
| Standard ASR | ExtBiMamba(Small) | - | Near Transformer | Effective at small scales |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| Causal SUPERB | Mamba > Causal Transformer | Superior in phoneme and speaker tasks |
| Phone Purity | Higher for Mamba | Better quality in quantized speech representations |
| CCA Analysis | More distinct speaker features | Mamba encodes speaker information more clearly |
| ExtBiMamba Base | Lower than Transformer | Large-scale bidirectional Mamba still requires optimization |

### Key Findings
- Mamba's causal nature provides a natural advantage for streaming speech scenarios—78M parameters outperformed the 94M Causal Transformer.
- Computational costs remain nearly constant relative to sequence length, whereas Transformers OOM beyond 80 seconds.
- Mamba generates quantized representations with higher phone purity, benefiting spoken language models using SSL units.
- Large-scale bidirectional Mamba (Base) still performs below Transformers, suggesting that scalability remains a challenge.

## Highlights & Insights
- Performed the first systematic evaluation of Mamba as a speech foundation model across multiple dimensions rather than isolated tasks.
- Found that "causality is an advantage rather than a limitation," shifting perceptions of Mamba's application in speech.
- Discovered representation quality benefits that offer direct implications for the field of spoken language modeling.

## Limitations & Future Work
- Poor performance in large-scale training of bidirectional Mamba indicates that scalability is a key challenge.
- Pre-training and evaluation were limited to LibriSpeech; multilingual and noisy scenarios remain untested.
- Training scale was significantly smaller than the original HuBERT due to single V100 constraints.
- Future work could explore improved architectures like Mamba-2 and larger-scale training.

## Related Work & Insights
- **vs. Hybrid Mamba-Transformer**: This work uses a pure Mamba architecture to more clearly reveal innate strengths and weaknesses.
- **vs. SSAM**: While SSAM focused on general audio, this work focuses specifically on speech SSL.
- **vs. Mamba Streaming ASR**: Unlike previous works requiring additional mechanisms (e.g., lookahead), this study demonstrates that pure Mamba inherently possesses advantages.

## Rating
- Novelty: ⭐⭐⭐⭐ First comprehensive exploration of Mamba as a speech SSL foundation model.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-dimensional evaluation including ASR, SUPERB, representation analysis, long-context, and streaming.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and detailed experimentation.
- Value: ⭐⭐⭐⭐ Provides important empirical evidence for selecting efficient architectures in the speech domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] \[b\] = \[d\] − \[t\] + \[p\]: Self-supervised Speech Models Discover Phonological Vector Arithmetic](bd-tp_self-supervised_speech_models_discover_phonological_vector_arithmetic.md)
- [\[ACL 2026\] XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection](xlsr-mambo_scaling_the_hybrid_mamba-attention_backbone_for_audio_deepfake_detect.md)
- [\[ACL 2026\] Exploration of Perceptual Speech Features for Clinical Decision-Support in Mental Health Care](exploration_of_perceptual_speech_features_for_clinical_decision-support_in_menta.md)
- [\[ACL 2026\] Semi-Supervised Diseased Detection from Speech Dialogues with Multi-Level Data Modeling](semi-supervised_diseased_detection_from_speech_dialogues_with_multi-level_data_m.md)
- [\[ACL 2026\] Speech-Hands: A Self-Reflection Voice Agentic Approach to Speech Recognition and Audio Reasoning with Omni Perception](speech-hands_a_self-reflection_voice_agentic_approach_to_speech_recognition_and_.md)

</div>

<!-- RELATED:END -->
