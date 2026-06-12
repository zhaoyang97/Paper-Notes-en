---
title: >-
  [Paper Note] An Exploration of Mamba for Speech Self-Supervised Models
description: >-
  [ACL 2026][Audio & Speech][Mamba] This paper presents the first comprehensive exploration of the Mamba architecture as a foundation model for Speech Self-Supervised Learning (SSL). The findings demonstrate that Mamba-bas…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Mamba"
  - "Speech SSL"
  - "HuBERT"
  - "State Space Models"
  - "Streaming ASR"
date: 2026-05-08
content_hash: 16f72256c1bc8a3a
---

# An Exploration of Mamba for Speech Self-Supervised Models

**Conference**: ACL 2026  
**arXiv**: [2506.12606](https://arxiv.org/abs/2506.12606)  
**Code**: [GitHub](https://github.com/hckuo145/Mamba-based-HuBERT)  
**Area**: Speech / Self-Supervised Learning  
**Keywords**: Mamba, Speech SSL, HuBERT, State Space Models, Streaming ASR

## TL;DR

This paper presents the first comprehensive exploration of the Mamba architecture as a foundation model for Speech Self-Supervised Learning (SSL). The findings demonstrate that Mamba-based HuBERT outperforms Transformers in long-context ASR, streaming ASR, and causal probing tasks while maintaining linear time complexity.

## Background & Motivation

**Background**: Transformer-based speech SSL models (e.g., HuBERT, wav2vec 2.0) have achieved significant success, but their quadratic complexity leads to high computational costs and memory bottlenecks when processing long sequences.

**Limitations of Prior Work**: (1) While Mamba has shown capabilities surpassing Transformers in language modeling, its application in speech has been limited to isolated studies on single tasks; (2) Existing speech Mamba works often report performance comparable to or slightly worse than Transformers and frequently require hybrid designs; (3) There is a lack of unified evaluation across multiple tasks.

**Key Challenge**: The linear time complexity of Mamba is theoretically ideal for the long-sequence nature of speech, but its comprehensive performance in speech SSL remains unclear.

**Goal**: Systematically train and evaluate Mamba-based HuBERT models to comprehensively explore their potential as speech foundation models and feature extractors.

**Key Insight**: Replace Transformer blocks in HuBERT with Mamba blocks while maintaining the same training pipeline (two-round iterative k-means pseudo-label training) and evaluate across ASR, SUPERB, and other tasks.

**Core Idea**: The inherent causal architecture of Mamba makes it particularly suitable for building causal speech SSL models, demonstrating unique advantages in streaming ASR and long-context scenarios.

## Method

### Overall Architecture

The Transformer blocks in HuBERT are replaced with Mamba blocks, while the CNN feature encoder and positional encoder are retained. The training process follows the two-round HuBERT iteration: the first round targets MFCCs for 250k steps, and the second round uses the 6th-layer output from the first round as targets for 400k steps. Pre-training is conducted on LibriSpeech 960h.

### Key Designs

1. **Systematic Comparison of Mamba Variants**:

    - Function: Comprehensively evaluate the speech representation capabilities of different Mamba configurations.
    - Mechanism: Test causal settings (Mamba, Mamba+MLP) and bidirectional settings (ExtBiMamba, InnBiMamba), providing fair comparisons with corresponding Transformer variants.
    - Design Motivation: The causal nature of Mamba may be an advantage in some tasks (streaming ASR) and a disadvantage in others (tasks requiring global information).

2. **Long-context and Streaming ASR Evaluation**:

    - Function: Verify the value of Mamba's linear complexity in practical scenarios.
    - Mechanism: Process entire speech segments without sentence segmentation for long-context ASR; perform streaming ASR under the constraint of using only past information. Quantify changes in MACs/sec and RTF relative to sequence length.
    - Design Motivation: This is Mamba's greatest theoretical advantage over Transformers—Transformers encounter OOM beyond 80 seconds, while Mamba can handle over 5 minutes.

3. **Representation Quality Analysis**:

    - Function: Deeply understand the characteristics of speech representations learned by Mamba.
    - Mechanism: Use phone purity assessments to quantify the phonetic quality of representations and CCA analysis to examine how phoneme and speaker features are encoded.
    - Design Motivation: To understand not just "if it is good," but "why it is good" and "where it excels."

### Loss & Training

Ours follows the standard HuBERT training: masked prediction loss. It uses the Adam optimizer with a linear warm-up (first 8%) followed by linear decay. Due to computational resource constraints, training was performed on a single V100 with a batch size $1/4$ of the original.

## Key Experimental Results

### Main Results

| Setting | Model | Params | WER | Key Findings |
|------|------|--------|-----|--------|
| Streaming ASR | Mamba HuBERT | 78M | 15.77% | Outperforms 94M Causal Transformer (16.66%) |
| Long-context ASR | ExtBiMamba | - | 11.08% | Transformer failed due to OOM |
| Standard ASR | ExtBiMamba (Small) | - | Close to Transformer | Effective at small scales |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Causal SUPERB | Mamba > Causal Transformer | Superior in phoneme and speaker tasks |
| Phone Purity | Mamba Higher | Quantized representations have better phonetic quality |
| CCA Analysis | More distinct speaker features | Mamba encodes speaker information more clearly |
| ExtBiMamba Base | Lower than Transformer | Large-scale bidirectional Mamba still requires improvement |

### Key Findings
- Mamba's causal nature is a natural advantage for streaming speech scenarios—78M parameters outperform a 94M Causal Transformer.
- Computational cost remains almost constant relative to sequence length, whereas Transformers hit OOM beyond 80 seconds.
- Mamba produces quantized representations with higher phone purity, which benefits spoken language models using SSL units as input.
- Large-scale bidirectional Mamba (Base) still performs worse than Transformers across the board, suggesting that scalability remains a key challenge.

## Highlights & Insights
- This is the first systematic evaluation of Mamba as a speech foundation model across multiple domains rather than a single task.
- The finding that "causality is an advantage rather than a limitation" shifts the perception of using Mamba in speech.
- Insights into the quality of quantized representations have direct implications for the field of spoken language modeling.

## Limitations & Future Work
- Large-scale training of bidirectional Mamba yields suboptimal results; scalability is a critical challenge.
- Pre-training and evaluation were limited to LibriSpeech; multilingual and noisy scenarios remain untested.
- Restricted by a single V100, the training scale is significantly smaller than the original HuBERT.
- Future work could explore improved architectures like Mamba2 and larger-scale training.

## Related Work & Insights
- **vs. Hybrid Mamba-Transformer**: Ours uses a pure Mamba architecture to more clearly reveal its strengths and weaknesses.
- **vs. SSAM**: While SSAM focuses on general audio, Ours focuses specifically on speech SSL.
- **vs. Mamba Streaming ASR**: Previous works required additional mechanisms (e.g., lookahead), whereas Ours demonstrates that pure Mamba inherently possesses advantages.

## Rating
- Novelty: ⭐⭐⭐⭐ First comprehensive exploration of Mamba as a speech SSL foundation model.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-dimensional evaluation including ASR, SUPERB, representation analysis, long-context, and streaming.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and detailed experiments.
- Value: ⭐⭐⭐⭐ Provides important empirical evidence for selecting efficient architectures in the speech domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] \[b\] = \[d\] − \[t\] + \[p\]: Self-supervised Speech Models Discover Phonological Vector Arithmetic](bd-tp_self-supervised_speech_models_discover_phonological_vector_arithmetic.md)
- [\[ACL 2026\] XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection](xlsr-mambo_scaling_the_hybrid_mamba-attention_backbone_for_audio_deepfake_detect.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[ACL 2026\] Exploration of Perceptual Speech Features for Clinical Decision-Support in Mental Health Care](exploration_of_perceptual_speech_features_for_clinical_decision-support_in_menta.md)
- [\[ACL 2026\] Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models](do_we_need_distinct_representations_for_every_speech_token_unveiling_and_exploit.md)

</div>

<!-- RELATED:END -->
