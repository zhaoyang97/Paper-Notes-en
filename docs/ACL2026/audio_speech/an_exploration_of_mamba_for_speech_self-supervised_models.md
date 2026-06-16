---
title: >-
  [Paper Note] An Exploration of Mamba for Speech Self-Supervised Models
description: >-
  [ACL 2026][Audio & Speech][Mamba] This paper presents the first comprehensive exploration of Mamba architecture as a potential foundation model for speech self-supervised learning (SSL), finding that Mamba-based HuBERT outperforms Transformers in long-context ASR, streaming ASR, and causal probing tasks while maintaining linear time complexity.
tags:
  - ACL 2026
  - Audio & Speech
  - Mamba
  - HuBERT
  - State Space Model
  - Streaming ASR
date: 2026-05-08
content_hash: d51ff02b5a6cfab6
---
# An Exploration of Mamba for Speech Self-Supervised Models

**Conference**: ACL 2026  
**arXiv**: [2506.12606](https://arxiv.org/abs/2506.12606)  
**Code**: [GitHub](https://github.com/hckuo145/Mamba-based-HuBERT)  
**Area**: Speech / Self-Supervised Learning  
**Keywords**: Mamba, Speech Self-Supervised Learning, HuBERT, State Space Models, Streaming ASR

## TL;DR

This paper presents the first comprehensive exploration of Mamba architecture as a potential foundation model for speech self-supervised learning (SSL), finding that Mamba-based HuBERT outperforms Transformers in long-context ASR, streaming ASR, and causal probing tasks while maintaining linear time complexity.

## Background & Motivation

**Background**: Transformer-based speech SSL models (such as HuBERT, wav2vec 2.0) have achieved massive success, but their quadratic complexity creates high computational costs and memory bottlenecks when processing long sequences.

**Limitations of Prior Work**: (1) Mamba has demonstrated capabilities exceeding Transformers in language modeling, but its application in the speech domain is limited to isolated studies on single tasks; (2) existing speech Mamba works often report performance comparable to or even slightly worse than Transformers, and frequently require hybrid designs; (3) there is a lack of unified cross-task evaluation.

**Key Challenge**: While Mamba's linear time complexity is theoretically ideal for the long-sequence characteristics of speech, its comprehensive performance in speech SSL remains unclear.

**Goal**: Systematically train and evaluate Mamba-based HuBERT models to fully explore their potential as speech foundation models and feature extractors.

**Key Insight**: Replace the Transformer blocks in HuBERT with Mamba blocks while maintaining the same training pipeline (two-round iterative k-means pseudo-label training) and evaluating on multi-tasks including ASR and SUPERB.

**Core Idea**: Mamba's natural causal architecture makes it particularly suitable for building causal speech SSL models, demonstrating unique advantages in streaming ASR and long-context scenarios.

## Method

### Overall Architecture

Ours does not invent a new architecture but conducts a controlled replacement experiment: replacing the Transformer blocks in HuBERT with Mamba blocks. The CNN feature encoder and position encoder remain unchanged, and the training process strictly follows HuBERT's two-round iteration (the first round targets MFCCs for 250k steps, and the second round uses the 6th layer output from the first round as pseudo-labels for 400k steps), pre-training on LibriSpeech 960h. In this way, the backbone is the only variable; any performance differences in the pipeline from input speech $\to$ Mamba encoding $\to$ SSL representation $\to$ downstream ASR/SUPERB probes can be cleanly attributed to the intrinsic differences between Mamba and Transformers.

### Key Designs

**1. Systematic comparison of multiple Mamba variants: Clarifying whether causality is an advantage or a burden**

Mamba is inherently causal, a property that functions differently across tasks—causality is an advantage in streaming ASR where only past information can be seen, while it might be a disadvantage for tasks requiring global context. To clearly define this boundary, Ours evaluates both causal settings (Mamba, Mamba+MLP) and bidirectional settings (ExtBiMamba, InnBiMamba), providing fair comparisons against Transformer variants of equivalent parameter sizes. This paired design allows the "causal vs. bidirectional" and "Mamba vs. Transformer" dimensions to be analyzed independently rather than providing a vague conclusion.

**2. Evaluation of long-context and streaming ASR: Realizing the theoretical advantages of Mamba's linear complexity in measurable scenarios**

The main selling point of Mamba over Transformers is $O(n)$ rather than $O(n^2)$, but this advantage only becomes apparent with long sequences. Two targeted scenarios were designed: long-context ASR treats entire unsegmented speech segments, while streaming ASR constrains the model to decode frame-by-frame using only past information. Simultaneously, MACs/second and RTF (Real-Time Factor) curves are quantified relative to sequence length. Results show that Transformers OOM and fail at lengths over 80 seconds, while Mamba's computational load remains nearly constant, allowing for processing of speech over 5 minutes.

**3. Representation quality analysis: Not just "how well," but "where and why" it performs**

Downstream WER alone cannot explain the intrinsic characteristics of Mamba representations. Ours further performs representation-level analysis: using phone purity to quantify phonetic purity and CCA (Canonical Correlation Analysis) to characterize how phoneme and speaker features are encoded. It was discovered that Mamba's quantized representations have higher phone purity and clearer encoding of speaker information. This provides direct value for spoken language models that use SSL units as input, upgrading performance metrics to interpretable representation characteristics.

### Loss & Training

Ours follows the standard HuBERT training objective: masked prediction loss. An Adam optimizer is used, with the learning rate following linear warm-up (first 8%) and then linear decay. Due to computational resource constraints, models were trained on a single V100 with a batch size 1/4 of the original configuration.

## Key Experimental Results

### Main Results

| Setting | Model | Parameters | WER | Key Findings |
|---------|-------|------------|-----|--------------|
| Streaming ASR | Mamba HuBERT | 78M | 15.77% | Outperforms 94M Causal Transformer (16.66%) |
| Long-context ASR | ExtBiMamba | - | 11.08% | Transformer fails due to OOM |
| Standard ASR | ExtBiMamba(Small) | - | Close to Transformer | Effective at small scales |

### Ablation Study

| Configuration | Key Metrics | Description |
|---------------|-------------|-------------|
| Causal SUPERB | Mamba > Causal Transformer | Superior in phoneme and speaker tasks |
| Phone Purity | Mamba higher | Better speech quality in quantized representations |
| CCA Analysis | Speaker features more distinct | Mamba encodes speaker information more clearly |
| ExtBiMamba Base | Lower than Transformer | Large-scale bidirectional Mamba still requires improvement |

### Key Findings
- Mamba's causal nature is a natural advantage for streaming speech scenarios—78M parameters outperform a 94M causal Transformer.
- Computational costs remain almost constant with sequence length, whereas Transformers OOM beyond 80 seconds.
- Mamba produces quantized representations with higher phone purity, which is beneficial for spoken language models using SSL units as input.
- Large-scale bidirectional Mamba (Base) still results in lower performance than Transformers, suggesting that scalability remains a key challenge.

## Highlights & Insights
- The first systematic evaluation of Mamba as a speech foundation model instead of testing on isolated tasks.
- The discovery that "causal nature is an advantage rather than a limitation" shifts the understanding of Mamba's application in speech.
- Findings regarding quantized representation quality provide direct insights for the field of spoken language models.

## Limitations & Future Work
- Large-scale training of bidirectional Mamba performs poorly; scalability is a critical challenge.
- Pre-training and evaluation were only conducted on LibriSpeech; multilingual and noisy scenarios have not been tested.
- Constrained by a single V100, the training scale is significantly smaller than the original HuBERT.
- Future work may explore improved architectures like Mamba2 and larger-scale training.

## Related Work & Insights
- **vs. Hybrid Mamba-Transformer**: Ours uses a pure Mamba architecture to more clearly reveal its strengths and weaknesses.
- **vs. SSAM**: SSAM focuses on general audio rather than speech; Ours focuses specifically on speech SSL.
- **vs. Mamba Streaming ASR**: Previous work required additional mechanisms (e.g., lookahead); Ours demonstrates that pure Mamba holds an inherent advantage.

## Rating
- Novelty: ⭐⭐⭐⭐ First comprehensive exploration of Mamba as a speech SSL foundation model.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-dimensional evaluation across ASR, SUPERB, representation analysis, long-context, and streaming.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed experiments.
- Value: ⭐⭐⭐⭐ Provides important empirical evidence for efficient architecture selection in the speech domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] \[b\] = \[d\] − \[t\] + \[p\]: Self-supervised Speech Models Discover Phonological Vector Arithmetic](bd-tp_self-supervised_speech_models_discover_phonological_vector_arithmetic.md)
- [\[ACL 2026\] XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection](xlsr-mambo_scaling_the_hybrid_mamba-attention_backbone_for_audio_deepfake_detect.md)
- [\[ACL 2026\] Exploration of Perceptual Speech Features for Clinical Decision-Support in Mental Health Care](exploration_of_perceptual_speech_features_for_clinical_decision-support_in_menta.md)
- [\[ACL 2026\] Speech-Hands: A Self-Reflection Voice Agentic Approach to Speech Recognition and Audio Reasoning with Omni Perception](speech-hands_a_self-reflection_voice_agentic_approach_to_speech_recognition_and_.md)
- [\[ACL 2026\] Semi-Supervised Diseased Detection from Speech Dialogues with Multi-Level Data Modeling](semi-supervised_diseased_detection_from_speech_dialogues_with_multi-level_data_m.md)

</div>

<!-- RELATED:END -->
